"""Memory across chats: short facts about the user that every chat (on any server) can recall.

Ported from our own memory code (read-only sources):
  - recall ranking: /usr/local/bin/pxa-mythos-factrecall.mjs (keywordScore + rankOwnerFacts keyword-only leg:
    0.75 * keyword overlap + 0.25 * prior, a relevance floor of one real word hit), here with idf weights so
    rare words count more (BM25-ish) and recency as the prior;
  - the block format + "possibly out of date" framing: factrecall formatFactLines / digestTruncationNote;
  - secret refusal: /usr/local/bin/pxa-mythos-guards.mjs SECRET_WHOLE / SECRET_PREFIXED (redactSecrets);
  - reply-style gate (PXA_MEMGATE_v1): "keep answers short"
    style prescriptions wedge models when injected into every prompt, so they are refused, not stored;
  - dedupe-on-write: pxa-web/lib/record-fact.mjs (update the existing row instead of adding a twin).

Store: <config>/chat/memory.json, 0600, atomic replace, one lock. Facts: {id, text, created, updated, source,
pinned}. Size cap with oldest-unpinned eviction."""
import json
import math
import os
import re
import threading
import time

MAX_FACTS = 200
MAX_LEN = 300
DEFAULT_K = 8
DEFAULT_BUDGET = 400          # tokens (~4 chars each) for the recalled block
_lock = threading.Lock()

# ---- secrets (pxa-mythos-guards.mjs SECRET_WHOLE / SECRET_PREFIXED, plus a few obvious shapes) ----------
SECRET_RES = [
    (re.compile(r"\btskey-[A-Za-z0-9_-]{10,}"), "a Tailscale key"),
    (re.compile(r"\bhf_[A-Za-z0-9]{10,}"), "a Hugging Face token"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"), "an API key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{10,}|\bgithub_pat_[A-Za-z0-9_]{20,}"), "a GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "an AWS key"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "a Slack token"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "a Google API key"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{10,}", re.I), "a bearer token"),
    (re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"), "a private key"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "a login token"),
    (re.compile(r"\b(?:password|passwd|passcode|pass ?phrase|pin|api[ _-]?key|secret|token|access[ _-]?key|"
                r"private[ _-]?key|2fa|otp|recovery[ _-]?code|seed[ _-]?phrase)s?\b\s*(?:is|are|was|=|:)\s*\S+", re.I),
     "a password or secret"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "a card or account number"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "an ID number"),
    (re.compile(r"\b(?=[A-Za-z0-9+/_-]{32,}\b)(?=\S*\d)(?=\S*[A-Za-z])[A-Za-z0-9+/_-]{32,}\b"), "a long key-like string"),
]

# ---- reply-style gate (memory-gate.cjs PATTERNS, verbatim shapes) ------------------------------------------
_SMALLNUM = r"(?:[1-9]|10|one|two|three|four|five|six|seven|eight|nine|ten|a few|few(?:er)?|a couple(?: of)?|single)"
_STYLE = (r"(?:concise|short(?:er)?|terse(?:r)?|brief(?:er)?|brevity|minimal(?:ist(?:ic)?)?|laconic|succinct|curt|"
          r"clipped|monosyllabic|tight(?:er)?|one[-\s]?liners?|" + _SMALLNUM + r"[-\s]?(?:word|sentence|line)s?)")
_RESP = r"(?:response|repl(?:y|ies)|answer|message|output|wording|communication)s?"
_PREF = (r"(?:prefer|like|want|value|need|expect|enjoy|appreciate|favor|request|demand)(?:s|ed|ing)?|asks?\s+for|"
         r"insists?\s+on")
STYLE_RES = [re.compile(p, re.I) for p in (
    r"\b(?:" + _PREF + r")\b[^.;:!?\n]{0,80}\b" + _STYLE + r"\b[^.;:!?\n]{0,80}\b" + _RESP + r"\b",
    r"\b" + _RESP + r"\b[^.;:!?\n]{0,50}\b(?:should|must|needs?\s+to|ought\s+to|have\s+to|are\s+to|to\s+be|kept|stay)\b"
    r"[^.;:!?\n]{0,50}\b" + _STYLE + r"\b",
    r"\bkeep\b[^.;:!?\n]{0,40}\b" + _RESP + r"\b[^.;:!?\n]{0,40}\b(?:" + _STYLE + r"|under|below|max(?:imum)?)\b",
    r"\b(?:respond|reply|answer|speak|write)\b[^.;:!?\n]{0,40}\b(?:in|with|using|under)\b[^.;:!?\n]{0,30}\b"
    + _SMALLNUM + r"[-\s]?(?:word|sentence|line)s?\b",
    r"\b(?:always|only)\b[^.;:!?\n]{0,40}\b(?:respond|reply|answer)s?\b[^.;:!?\n]{0,40}\b(?:concise(?:ly)?|brief(?:ly)?|"
    r"terse(?:ly)?|short|minimal(?:ly)?|" + _SMALLNUM + r"[-\s]?words?)\b",
    r"\b" + _SMALLNUM + r"[-\s]?word\s+" + _RESP + r"\b",
    r"\bno\s+more\s+than\s+" + _SMALLNUM + r"\s+(?:word|sentence|line)s?\b",
)]

STOP = set("""a an and are as at be been but by can could did do does for from had has have he her hers him his how i
i'm im if in into is it it's its just me my myself no not of on or our ours she so some than that the their them
then there these they this those to too us was we were what when where which who whom why will with would you your
yours user user's users also very really please remember forget note save that's""".split())


class MemoryError_(ValueError):
    """a fact that may not be stored (secret, style rule, empty, too long). str() is the plain-words reason."""


def words(text):
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def content_words(text):
    return [w for w in words(text) if w not in STOP and len(w) > 1]


_CTRL = re.compile(r"\[\[[^\]]*\]\]")


def clean(text):
    """one stored sentence. A [[control]] tag is not part of a fact. Ordinary [brackets] stay."""
    t = _CTRL.sub(" ", str(text or ""))
    t = " ".join(t.split()).strip()
    t = t.strip("-*• ").strip()
    return t


def screen(text):
    """-> None when the fact may be stored, else the reason (plain words) it is refused."""
    t = clean(text)
    if not t:
        return "the fact is empty"
    if len(t) > MAX_LEN:
        return f"a memory is one short sentence (at most {MAX_LEN} characters)"
    for rx, what in SECRET_RES:
        if rx.search(t):
            return f"it looks like {what}; secrets are never saved to memory"
    for rx in STYLE_RES:
        if rx.search(t):
            return ("that is a rule about how to reply, not a fact about the user; reply-style rules are not saved "
                    "(they make every later answer worse)")
    return None


def similarity(a, b):
    """Jaccard over content words (the dedupe test)."""
    sa, sb = set(content_words(a)), set(content_words(b))
    if not sa or not sb:
        return 1.0 if clean(a).lower() == clean(b).lower() else 0.0
    return len(sa & sb) / float(len(sa | sb))


class Store(object):
    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, "memory.json")

    # ---- file -------------------------------------------------------------------------------------------
    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        if not isinstance(d, dict):
            d = {}
        d.setdefault("version", 1)
        d.setdefault("enabled", True)
        d.setdefault("k", DEFAULT_K)
        d.setdefault("budget", DEFAULT_BUDGET)
        d.setdefault("next_id", 1)
        facts = d.get("facts") if isinstance(d.get("facts"), list) else []
        d["facts"] = [f for f in facts if isinstance(f, dict) and f.get("id") and f.get("text")]
        return d

    def _save(self, d):
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def _mutate(self, fn):
        with _lock:
            d = self.load()
            out = fn(d)
            self._save(d)
            return out

    # ---- settings ---------------------------------------------------------------------------------------
    def settings(self):
        d = self.load()
        return {"enabled": bool(d["enabled"]), "k": int(d["k"]), "budget": int(d["budget"])}

    def set_settings(self, enabled=None, k=None, budget=None):
        def fn(d):
            if enabled is not None:
                d["enabled"] = bool(enabled)
            if k is not None:
                d["k"] = max(0, min(int(k), 50))
            if budget is not None:
                d["budget"] = max(50, min(int(budget), 4000))
            return {"enabled": d["enabled"], "k": d["k"], "budget": d["budget"]}
        return self._mutate(fn)

    # ---- facts ------------------------------------------------------------------------------------------
    def facts(self):
        return sorted(self.load()["facts"], key=lambda f: (not f.get("pinned"), -(f.get("updated") or 0)))

    def add(self, text, source=None, pinned=False, created=None):
        """-> {"fact", "status": "added"|"updated"|"same", "evicted": [...]}. Raises MemoryError_ on refusal."""
        t = clean(text)
        why = screen(t)
        if why:
            raise MemoryError_(why)
        now = time.time()

        def fn(d):
            best, score = None, 0.0
            for f in d["facts"]:
                s = similarity(t, f["text"])
                if s > score:
                    best, score = f, s
            if best is not None and (score >= 0.8 or words(best["text"]) == words(t)):
                status = "same" if words(best["text"]) == words(t) else "updated"
                best.update(updated=now)
                if status == "updated":
                    best["text"] = t
                if pinned:
                    best["pinned"] = True
                return {"fact": dict(best), "status": status, "evicted": []}
            f = {"id": f"m{d['next_id']}", "text": t, "created": created or now, "updated": now,
                 "source": source, "pinned": bool(pinned)}
            d["next_id"] += 1
            d["facts"].append(f)
            evicted = []
            while len(d["facts"]) > MAX_FACTS:
                unp = [x for x in d["facts"] if not x.get("pinned")]
                if not unp:
                    break
                old = min(unp, key=lambda x: x.get("updated") or 0)
                d["facts"].remove(old)
                evicted.append(old)
            return {"fact": dict(f), "status": "added", "evicted": evicted}
        return self._mutate(fn)

    def find(self, fact_or_id, facts=None):
        """the fact an id ('m3') or a description points at, or None."""
        q = clean(fact_or_id)
        facts = facts if facts is not None else self.load()["facts"]
        if not q:
            return None
        for f in facts:
            if f["id"] == q or f["id"] == q.lstrip("[").rstrip("]"):
                return f
        best, score = None, 0.0
        qs = set(content_words(q))
        for f in facts:
            fs = set(content_words(f["text"]))
            s = similarity(q, f["text"])
            if qs and qs <= fs:                     # "vegetarian" -> "The user is vegetarian."
                s = max(s, 0.6 + 0.4 * len(qs) / max(1, len(fs)))
            if q.lower() in f["text"].lower():
                s = max(s, 0.7)
            if s > score:
                best, score = f, s
        return best if score >= 0.5 else None

    def forget(self, fact_or_id):
        def fn(d):
            f = self.find(fact_or_id, d["facts"])
            if f is None:
                return None
            d["facts"].remove(f)
            return dict(f)
        return self._mutate(fn)

    def update(self, fid, text=None, pinned=None):
        if text is not None:
            why = screen(text)
            if why:
                raise MemoryError_(why)

        def fn(d):
            for f in d["facts"]:
                if f["id"] == fid:
                    if text is not None:
                        f["text"] = clean(text)
                    if pinned is not None:
                        f["pinned"] = bool(pinned)
                    f["updated"] = time.time()
                    return dict(f)
            return None
        return self._mutate(fn)

    def restore(self, fact):
        """put back a fact exactly as it was (undo of a forget)."""
        if not isinstance(fact, dict) or not fact.get("text"):
            raise MemoryError_("nothing to restore")
        why = screen(fact["text"])
        if why:
            raise MemoryError_(why)

        def fn(d):
            if any(f["id"] == fact.get("id") for f in d["facts"]):
                return None
            f = {"id": str(fact.get("id") or f"m{d['next_id']}"), "text": clean(fact["text"]),
                 "created": fact.get("created") or time.time(), "updated": fact.get("updated") or time.time(),
                 "source": fact.get("source"), "pinned": bool(fact.get("pinned"))}
            if not re.match(r"^m\d+$", f["id"]):
                f["id"] = f"m{d['next_id']}"
            d["next_id"] = max(d["next_id"], int(f["id"][1:]) + 1)
            d["facts"].append(f)
            return dict(f)
        return self._mutate(fn)

    def clear(self):
        def fn(d):
            n = len(d["facts"])
            d["facts"] = []
            return n
        return self._mutate(fn)

    def import_facts(self, items, replace=False):
        """-> {"added", "skipped": [{"text", "why"}]}"""
        if replace:
            self.clear()
        added, skipped = 0, []
        for it in items or []:
            text = it.get("text") if isinstance(it, dict) else it
            try:
                r = self.add(text, source=(it.get("source") if isinstance(it, dict) else None) or "import",
                             pinned=bool(isinstance(it, dict) and it.get("pinned")))
                added += r["status"] == "added"
            except MemoryError_ as e:
                skipped.append({"text": str(text)[:80], "why": str(e)})
        return {"added": added, "skipped": skipped}

    # ---- recall -----------------------------------------------------------------------------------------
    def recall(self, query, k=None, budget=None):
        """-> the facts to show the model, within the token budget: pinned first, then the top-k by idf-weighted
        keyword overlap with the query (factrecall's relevance floor: at least one real word hit), then -- while
        slots remain -- the most recently saved facts (the owner-memory digest leg of our mythos agent), so a
        preference with no shared word ("vegetarian" for "what's for dinner?") still reaches the model."""
        d = self.load()
        k = d["k"] if k is None else k
        budget = d["budget"] if budget is None else budget
        facts = d["facts"]
        if not facts:
            return []
        docs = {f["id"]: set(content_words(f["text"])) for f in facts}
        n = len(facts)
        df = {}
        for ws in docs.values():
            for w in ws:
                df[w] = df.get(w, 0) + 1
        idf = lambda w: math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))  # noqa: E731  (BM25 idf)
        q = set(content_words(query))
        newest = max(f.get("updated") or 0 for f in facts) or 1
        oldest = min(f.get("updated") or 0 for f in facts)
        span = max(1.0, newest - oldest)
        ranked = []
        for f in facts:
            if f.get("pinned"):
                continue
            hit = q & docs[f["id"]]
            if not hit:
                continue
            kw = sum(idf(w) for w in hit) / max(1e-9, sum(idf(w) for w in q))
            rec = ((f.get("updated") or 0) - oldest) / span
            ranked.append((0.75 * kw + 0.25 * rec, f))
        ranked.sort(key=lambda x: -x[0])
        picked = [f for _s, f in ranked[:max(0, k)]]
        ids = {f["id"] for f in picked}
        recent = sorted((f for f in facts if not f.get("pinned") and f["id"] not in ids),
                        key=lambda f: -(f.get("updated") or 0))
        picked += recent[:max(0, k - len(picked))]
        out, used = [], 0
        for f in [f for f in facts if f.get("pinned")] + picked:
            cost = len(f["text"]) // 4 + 6
            if used + cost > budget and out:
                break
            out.append(f)
            used += cost
        return out

    def search(self, query, k=8):
        """facts that actually match the query. Unlike recall, this does not fill the rest with recent facts."""
        q = clean(query)
        qwords = set(content_words(q))
        if not q:
            return []
        ql = q.lower()
        ranked = []
        for f in self.load()["facts"]:
            text = f.get("text") or ""
            hit = qwords & set(content_words(text))
            sub = ql in text.lower()
            if not hit and not sub:
                continue
            score = len(hit) + (1 if sub else 0) + (0.5 if f.get("pinned") else 0)
            ranked.append((-score, -(f.get("updated") or 0), f))
        ranked.sort()
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = 8
        return [f for _s, _t, f in ranked[:max(1, min(k, 20))]]


def rank_summaries(query, chats, k=3):
    """the past-chat leg: same 0.75 keyword + 0.25 recency shape as fact recall, over compact summaries.
    A summary with no shared content word is left out (no floor-less dump of every old chat)."""
    docs = []
    for c in chats or []:
        if not isinstance(c, dict):
            continue
        ws = set(content_words(c.get("text") or ""))
        if ws:
            docs.append((c, ws))
    if not docs:
        return []
    n = len(docs)
    df = {}
    for _c, ws in docs:
        for w in ws:
            df[w] = df.get(w, 0) + 1

    def idf(w):
        return math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))

    q = set(content_words(query))
    if not q:
        return []
    newest = max((c.get("updated") or 0) for c, _ws in docs) or 1
    oldest = min((c.get("updated") or 0) for c, _ws in docs)
    span = max(1.0, newest - oldest)
    ranked = []
    q_idf = sum(idf(w) for w in q) or 1e-9
    for c, ws in docs:
        hit = q & ws
        if not hit:
            continue
        kw = sum(idf(w) for w in hit) / q_idf
        rec = ((c.get("updated") or 0) - oldest) / span
        ranked.append((0.75 * kw + 0.25 * rec, c))
    ranked.sort(key=lambda x: -x[0])
    return [c for _s, c in ranked[:max(0, k)]]


def pick_summaries(query, chats, budget, used=0, k=3):
    """summaries that still fit the memory token budget after the facts have taken their share."""
    out = []
    for c in rank_summaries(query, chats, k=k):
        snippet = " ".join(str(c.get("text") or "").split())
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        cost = len(snippet) // 4 + 8
        if used + cost > budget:
            break
        item = dict(c)
        item["text"] = snippet
        out.append(item)
        used += cost
    return out


def block(facts, chats=None):
    """the delimited system-prompt block (factrecall formatFactLines shape: id, text, date; 'may be stale').
    `chats` is the past-chat summary leg, under the same 'may be out of date' framing."""
    parts = []
    if facts:
        lines = [f"- [{f['id']}] {f['text']} (saved {time.strftime('%Y-%m-%d', time.localtime(f.get('updated') or 0))}"
                 + (", pinned" if f.get("pinned") else "") + ")" for f in facts]
        parts.append("What you remember about the user from earlier chats (it may be out of date; the user's "
                     "words in this chat win):\n" + "\n".join(lines))
    if chats:
        clines = []
        for c in chats:
            when = time.strftime("%Y-%m-%d", time.localtime(c.get("updated") or 0))
            title = " ".join(str(c.get("title") or "Chat").split())[:80]
            clines.append(f"- [{title}, {when}] {c.get('text') or ''}")
        parts.append("Summaries of earlier chats (they may be out of date; the user's words in this chat win):\n"
                     + "\n".join(clines))
    if not parts:
        return ""
    return "<memory>\n" + "\n".join(parts) + "\n</memory>"


GUIDANCE = ("You have a memory that carries across chats. remember and memory_save add to it; memory_search looks "
            "up what is already saved. Whenever the user shares a lasting fact about themselves or a lasting "
            "preference (their name, where they live, diet, units, tools, projects, people they mention often), "
            "call remember or memory_save in that same turn, without being asked, with one short sentence about the "
            "user, e.g. \"The user prefers metric units.\" (one call per fact). Never say you noted or will remember "
            "something unless you called one of those. Don't save small talk, one-off requests, or things only about "
            "this chat. Never save passwords, keys, tokens, card numbers or other secrets, not even when asked; say "
            "you can't keep secrets instead. When the user asks you to forget something, call forget with the fact "
            "or its [id] from the memory block. To look through an older chat, call chat_search, then chat_read with "
            "the id it gives you (range is summary, all, last:N, or A-B). Incognito chats are not in those results. "
            "When the user attaches an earlier chat, it is in a referenced-chats block; use that. When they mention "
            "a chat but do not attach it, call chat_search. To hand one self-contained task to a fresh sub-agent, "
            "call spawn_agent. It can use only tools you already have, and it cannot spawn further. "
            "Use remembered facts when they help, without announcing that you remember them.")


# ---- safety net: when the model shares-but-forgets ------------------------------------------------------------
# A model told about the user ("I live in Lyon and use metric units") sometimes answers "noted!" without calling
# remember. When the message clearly talks about the user and the turn saved nothing, the harness asks the same
# model once, in a tiny separate request, to list the lasting facts; each goes through the same screen + dedupe.
SELF_RE = re.compile(r"\b(?:i am|i'm|im|i've|my (?:name|wife|husband|partner|kids?|son|daughter|dog|cat|job|work|"
                     r"favou?rite|home|team|company|setup|rig)|i live|i work|i prefer|i use|i always|i never|"
                     r"i (?:don'?t|do not) (?:eat|drink|use|like)|i eat|i like|i love|i hate|i have|call me|i speak|"
                     r"i'm from|i come from)\b", re.I)
CAPTURE_SYS = "You extract lasting facts about the user for an assistant's memory that carries across chats."
CAPTURE_ASK = ("Message from the user:\n<<<\n{text}\n>>>\nList each lasting fact or preference the user states about "
               "themselves (name, home, diet, units, tools, projects, family, pets, likes). One per line, each a short "
               "sentence starting with \"The user\". Only what the user states outright: nothing guessed, and nothing taken "
               "from a question. Skip one-off requests, questions, small talk and anything secret "
               "(passwords, keys, account numbers). If there are none, reply exactly NONE.")


def wants_capture(text):
    """the message STATES something about the user (a question like "how warm is it where I live?" does not)."""
    t = str(text or "")
    if len(t) >= 4000 or re.search(r"\bforget\b", t, re.I):
        return False
    said = [x for x in re.split(r"(?<=[.!?\n])\s+", t) if x.strip() and not x.strip().endswith("?")]
    return any(SELF_RE.search(x) for x in said)


CAPTURE_ASK_SUMMARY = (
    "Summary of earlier turns of a chat:\n<<<\n{text}\n>>>\nList each lasting fact or preference about the user "
    "that this summary already states (name, home, diet, units, tools, projects, family, pets, likes). One per "
    "line, each a short sentence starting with \"The user\". Nothing guessed, and nothing that is only a task "
    "or a file name. Skip one-off requests and anything secret (logins, keys, account numbers). If there are "
    "none, reply exactly NONE.")


def parse_capture(reply):
    out = []
    for line in str(reply or "").splitlines():
        line = clean(re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line))
        if line.lower().startswith("the user") and 8 < len(line) <= MAX_LEN:
            out.append(line if line.endswith((".", "!", "?")) else line + ".")
    return out[:3]
