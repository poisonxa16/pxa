"""Chat sessions saved on this machine (Control's config folder, chat/sessions/<id>.json, 0600), with Markdown
and JSON export. A session keeps the OpenAI-format messages (what the model sees next turn) and the UI
transcript (what the page shows, tool cards included)."""
import json
import os
import re
import secrets
import threading
import time

ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,48}$")
MAX_SESSIONS = 300
_lock = threading.Lock()
# Held across a rewind, a version switch, and the save that commits a new version.
# Store.save takes _lock inside this, so this lock is always acquired first.
mutate_lock = threading.Lock()


class Store(object):
    def __init__(self, root):
        self.root = root
        self.dir = os.path.join(root, "sessions")

    def _path(self, sid):
        if not isinstance(sid, str) or not ID_RE.match(sid):
            raise KeyError("bad session id")
        return os.path.join(self.dir, sid + ".json")

    def new_id(self):
        return "c_" + secrets.token_urlsafe(9).replace("-", "x").replace("_", "y")

    def get(self, sid):
        try:
            with open(self._path(sid), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def save(self, sess):
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        sess["updated"] = time.time()
        p = self._path(sess["id"])
        tmp = p + ".tmp"
        with _lock:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(sess, f)
            os.replace(tmp, p)
        self._prune()
        return sess

    def create(self, server_key=None, preset="assistant"):
        now = time.time()
        return {"id": self.new_id(), "title": "New chat", "created": now, "updated": now, "server_key": server_key,
                "server_name": None, "preset": preset, "messages": [], "turns": []}

    def delete(self, sid):
        try:
            os.unlink(self._path(sid))
            return True
        except OSError:
            return False

    def list(self):
        out = []
        try:
            names = os.listdir(self.dir)
        except OSError:
            return out
        for n in names:
            if not n.endswith(".json"):
                continue
            s = self.get(n[:-5])
            if s:
                out.append({"id": s["id"], "title": s.get("title") or "Chat", "updated": s.get("updated"),
                            "server_name": s.get("server_name"), "preset": s.get("preset"),
                            "turns": len(s.get("turns") or [])})
        out.sort(key=lambda x: -(x["updated"] or 0))
        return out

    def _prune(self):
        items = self.list()
        for x in items[MAX_SESSIONS:]:
            self.delete(x["id"])


class Ephemeral(object):
    """a Store plus an in-process map for incognito chats.

    An incognito chat is answered normally and can be read back during this process, but it is never written
    to disk, never listed, and never pruned onto disk. Restarting Control drops it. Turning incognito off
    writes the chat on the next save."""

    def __init__(self, store):
        self.disk = store
        self.ram = {}
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.disk, name)

    def get(self, sid):
        with self._lock:
            hit = self.ram.get(sid)
        if hit is not None:
            return hit
        return self.disk.get(sid)

    def save(self, sess):
        if sess.get("incognito"):
            sess["updated"] = time.time()
            with self._lock:
                self.ram[sess["id"]] = sess
            try:
                self.disk.delete(sess["id"])
            except KeyError:
                pass
            return sess
        with self._lock:
            self.ram.pop(sess.get("id"), None)
        return self.disk.save(sess)

    def delete(self, sid):
        with self._lock:
            had = self.ram.pop(sid, None) is not None
        try:
            disk = self.disk.delete(sid)
        except KeyError:
            disk = False
        return had or disk

    def list(self):
        return self.disk.list()

    def create(self, *a, **k):
        return self.disk.create(*a, **k)


def summaries(store, skip_id=None):
    """past chats' compact summaries, for the memory recall leg. Incognito chats are not on disk, so they
    are not here. The chat being answered is skipped: its own summary is already in what gets sent."""
    out = []
    for meta in store.list():
        if skip_id and meta.get("id") == skip_id:
            continue
        s = store.get(meta["id"])
        if not s or s.get("incognito"):
            continue
        comp = s.get("compact")
        if not isinstance(comp, dict):
            continue
        text = str(comp.get("summary") or "").strip()
        if not text:
            continue
        out.append({"id": s["id"], "title": s.get("title") or "Chat",
                    "updated": s.get("updated") or 0, "text": text})
    return out


def message_text(msg):
    """the readable text of one stored message. A list of parts keeps the text parts only."""
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return ""


def _snippet(summary, texts, qwords):
    for blob in [summary] + list(texts or []):
        for line in str(blob or "").splitlines():
            low = line.lower()
            if any(w in low for w in qwords):
                s = " ".join(line.split())
                return s[:240]
    s = " ".join(str(summary or "").split())
    return s[:240]


def search_chats(store, query, skip_id=None, k=5):
    """past chats whose title, compact summary or messages share a content word with the query.

    Incognito chats are not on the list, and a chat flagged incognito is skipped if it is seen anyway.
    The chat being answered is skipped: the model already has that transcript."""
    from .memory import content_words
    qwords = set(content_words(query))
    if not qwords:
        return []
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(k, 20))
    hits = []
    for meta in store.list() or []:
        sid = meta.get("id")
        if not sid or sid == skip_id:
            continue
        s = store.get(sid)
        if not s or s.get("incognito"):
            continue
        title = str(s.get("title") or "Chat")
        comp = s.get("compact") if isinstance(s.get("compact"), dict) else {}
        summary = str(comp.get("summary") or "")
        texts = [message_text(m) for m in (s.get("messages") or [])]
        for turn in s.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            for note in turn.get("notes") or []:
                if isinstance(note, dict) and (note.get("key") or note.get("text")):
                    texts.append(str(note.get("key") or "") + ": " + str(note.get("text") or ""))
        texts = [t for t in texts if t]
        bag = set(content_words("\n".join([title, summary] + texts)))
        hit = qwords & bag
        if not hit:
            continue
        title_hit = len(qwords & set(content_words(title)))
        hits.append((-(len(hit) + 3 * title_hit), -(s.get("updated") or 0), {
            "id": sid, "title": title, "updated": s.get("updated") or 0,
            "snippet": _snippet(summary, texts, qwords)}))
    hits.sort()
    return [h[2] for h in hits[:k]]


_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def read_chat(store, sid, span="summary"):
    """(ok, text). span is summary, all, last:N, or A-B (1-based message indexes).

    An incognito chat is refused even when this process still holds it in memory."""
    if not isinstance(sid, str) or not ID_RE.match(sid):
        return False, "error: bad session id"
    try:
        s = store.get(sid)
    except KeyError:
        return False, "error: bad session id"
    if not s:
        return False, "error: that chat does not exist"
    if s.get("incognito"):
        return False, "error: that chat is incognito and cannot be read"
    title = str(s.get("title") or "Chat")
    span = str(span or "summary").strip().lower()
    comp = s.get("compact") if isinstance(s.get("compact"), dict) else {}
    summary = str(comp.get("summary") or "").strip()
    if span in ("summary", "compact"):
        if not summary:
            return True, f"{title} ({sid})\nThis chat has no summary yet."
        return True, f"{title} ({sid})\n{summary}"
    msgs = [m for m in (s.get("messages") or []) if isinstance(m, dict)]
    n = len(msgs)
    if span in ("all", "*"):
        lo, hi = 1, max(1, n)
    elif span.startswith("last:"):
        try:
            count = int(span.split(":", 1)[1])
        except ValueError:
            return False, "error: range must be summary, all, last:N, or A-B"
        count = max(1, min(count, 40))
        lo, hi = max(1, n - count + 1), n
    else:
        m = _RANGE_RE.match(span)
        if not m:
            return False, "error: range must be summary, all, last:N, or A-B"
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo < 1 or hi < lo:
            return False, "error: range must be summary, all, last:N, or A-B"
        hi = min(hi, lo + 39)
    lines = [f"{title} ({sid}), messages {lo}-{min(hi, n)} of {n}"]
    if not n:
        lines.append("(no messages)")
    for i in range(lo, min(hi, n) + 1):
        msg = msgs[i - 1]
        body = " ".join(message_text(msg).split())
        if len(body) > 500:
            body = body[:497] + "..."
        lines.append(f"[{i}] {msg.get('role') or '?'}: {body}")
    return True, "\n".join(lines)


MAX_REFS = 4
REF_SYS = (
    "You summarize one earlier chat so it can be attached to a new one. "
    "Keep the goals, decisions, facts, file names and numbers exactly as written. Do not invent. "
    "Write one short paragraph."
)


class StopAttach(Exception):
    """the user pressed Stop while a referenced chat was being summarized."""


def fresh_brief(sess):
    """the cached summary of this chat, when it still covers every saved message."""
    b = sess.get("brief") if isinstance(sess, dict) else None
    n = len((sess or {}).get("messages") or [])
    if isinstance(b, dict) and int(b.get("upto") or -1) == n and str(b.get("text") or "").strip():
        return str(b["text"]).strip()
    return ""


def transcript_for_brief(sess, limit=12000):
    """the chat as lines a summary call can read. A long transcript keeps its head and its tail."""
    lines = []
    for m in (sess or {}).get("messages") or []:
        if not isinstance(m, dict):
            continue
        body = " ".join(message_text(m).split())
        if not body:
            continue
        if len(body) > 500:
            body = body[:497] + "..."
        lines.append(f"{m.get('role') or '?'}: {body}")
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...\n" + text[-half:]


def clip_brief(sess):
    """a summary with no model: the title, then the first and last lines, so an early fact survives."""
    title = " ".join(str((sess or {}).get("title") or "Chat").split())[:80] or "Chat"
    lines = []
    for m in (sess or {}).get("messages") or []:
        if not isinstance(m, dict):
            continue
        body = " ".join(message_text(m).split())
        if body:
            lines.append(f"{m.get('role') or '?'}: {body[:400]}")
    if not lines:
        return title + "\n(no messages yet)"
    if len(lines) > 8:
        lines = lines[:4] + ["..."] + lines[-4:]
    return title + "\n" + "\n".join(lines)


def store_brief(store, sess, text, method):
    """remember a summary on the chat it describes. Incognito chats are not written."""
    if not sess or sess.get("incognito") or not str(text or "").strip():
        return
    sess["brief"] = {"text": str(text).strip()[:4000], "upto": len(sess.get("messages") or []),
                     "method": method if method in ("summary", "clipped") else "clipped"}
    store.save(sess)


def take_refs(store, ids, skip_id=None):
    """the past chats a message asked to attach. The current chat, incognito chats and bad ids are dropped."""
    out, seen = [], set()
    if not isinstance(ids, list):
        return out
    for raw in ids:
        if len(out) >= MAX_REFS:
            break
        sid = str(raw or "")
        if not ID_RE.match(sid) or sid in seen or sid == skip_id:
            continue
        seen.add(sid)
        try:
            s = store.get(sid)
        except KeyError:
            continue
        if not s or s.get("incognito"):
            continue
        out.append(s)
    return out


def ref_block(parts):
    """the labelled context block for chats the user attached. Empty when there are none."""
    if not parts:
        return ""
    lines = ["The user attached these earlier chats. They may be out of date; the user's words in this chat win."]
    for p in parts:
        title = " ".join(str(p.get("title") or "Chat").split())[:80] or "Chat"
        lines.append(f"[{title}] ({p.get('id')})")
        lines.append(str(p.get("text") or "").strip())
    return "<referenced-chats>\n" + "\n".join(lines) + "\n</referenced-chats>"


def attach_refs(store, ids, skip_id, summarize):
    """summarize(title, transcript) -> text, or raise. A failure attaches a clip instead. The summary is cached
    on that chat until its messages change. Returns the block to send (maybe empty)."""
    parts = []
    for s in take_refs(store, ids, skip_id):
        text = fresh_brief(s)
        if not text:
            source = transcript_for_brief(s)
            method = "summary"
            try:
                if not source.strip():
                    raise ValueError("nothing to summarize")
                text = str(summarize(s.get("title") or "Chat", source) or "").strip()
                if not text:
                    raise ValueError("empty summary")
            except StopAttach:
                raise
            except Exception:           # noqa: BLE001 — a missed summary still attaches the lines we have
                text = clip_brief(s)
                method = "clipped"
            if text:
                store_brief(store, s, text, method)
        if text:
            parts.append({"id": s.get("id"), "title": s.get("title") or "Chat", "text": text})
    return ref_block(parts)


def turn_starts(sess):
    """index into sess["messages"] where each turn begins. Turns saved since v3.1 carry msg_start; older ones are
    found by their user message (a text-protocol tool result is also a user message, but starts with '[')."""
    msgs, turns = sess.get("messages") or [], sess.get("turns") or []
    if turns and all(isinstance(t.get("msg_start"), int) for t in turns):
        return [t["msg_start"] for t in turns]
    found = [i for i, m in enumerate(msgs) if m.get("role") == "user"
             and not str(m.get("content") or "").startswith(("[result of", "[tool result]"))]
    found = found[-len(turns):] if turns else []
    return found + [len(msgs)] * (len(turns) - len(found))


class ForkError(Exception):
    def __init__(self, msg, code="bad_branch", status=400):
        Exception.__init__(self, msg)
        self.code, self.status = code, status


def _copy(obj):
    return json.loads(json.dumps(obj))


def _fork_start(sess, rewind):
    """message index where turn `rewind` begins. Equal to the length of the shared prefix once that turn is gone."""
    turns = sess.get("turns") or []
    messages = sess.get("messages") or []
    if rewind < len(turns):
        starts = turn_starts(sess)
        if rewind < len(starts):
            return starts[rewind]
    return len(messages)


def _take_nested(forks, rewind, key):
    """pull forks that belong to a later turn out of the live map. The caller stores the copy on one version."""
    nested = {}
    for k in list(forks.keys()):
        if k == key:
            continue
        try:
            ik = int(k)
        except (TypeError, ValueError):
            continue
        if ik > rewind:
            nested[str(ik)] = forks.pop(k)
    return nested


def _suffix(sess, rewind, forks):
    key = str(rewind)
    start = _fork_start(sess, rewind)
    nested = _take_nested(forks, rewind, key)
    turns = sess.get("turns") or []
    messages = sess.get("messages") or []
    return start, {"turns": _copy(turns[rewind:]), "messages": _copy(messages[start:]), "forks": _copy(nested)}


def stash_fork(sess, rewind):
    """keep the turns from `rewind` onward as a version, and mark a new version pending. Returns the prefix length.
    Call this before truncating. A second rewind at the same point updates the active version instead of copying it."""
    rewind = int(rewind)
    forks = sess.get("forks")
    if not isinstance(forks, dict):
        forks = {}
        sess["forks"] = forks
    key = str(rewind)
    slot = forks.get(key)
    turns = sess.get("turns") or []
    if isinstance(slot, dict) and slot.get("pending") and len(turns) <= rewind:
        return _fork_start(sess, rewind)   # the previous rewind is still waiting for its reply
    if isinstance(slot, dict) and slot.get("pending"):
        commit_forks(sess)
    start, snap = _suffix(sess, rewind, forks)
    slot = forks.get(key)
    if not isinstance(slot, dict) or not isinstance(slot.get("versions"), list):
        forks[key] = {"active": 0, "pending": True, "versions": [snap]}
        return start
    versions = slot["versions"]
    try:
        active = int(slot.get("active") or 0)
    except (TypeError, ValueError):
        active = 0
    if not 0 <= active < len(versions):
        versions.append(snap)
        slot["active"] = len(versions) - 1
    else:
        versions[active] = snap
    slot["pending"] = True
    return start


def commit_forks(sess):
    """store a pending rewind's live suffix as the next version. No-op until that suffix has a turn.
    Pending is cleared as the version is added, so a second call does not add it again."""
    forks = sess.get("forks")
    if not isinstance(forks, dict):
        return False
    changed = False
    turns = sess.get("turns") or []
    messages = sess.get("messages") or []
    for key, slot in list(forks.items()):
        if not isinstance(slot, dict) or not slot.get("pending"):
            continue
        try:
            rewind = int(key)
        except (TypeError, ValueError):
            continue
        if len(turns) <= rewind:
            continue
        start = _fork_start(sess, rewind)
        nested = {}
        for k, v in forks.items():
            if k == key:
                continue
            try:
                ik = int(k)
            except (TypeError, ValueError):
                continue
            if ik > rewind and isinstance(v, dict):
                nested[str(ik)] = _copy(v)
        versions = list(slot.get("versions") or [])
        versions.append({"turns": _copy(turns[rewind:]), "messages": _copy(messages[start:]), "forks": nested})
        slot["versions"] = versions
        slot["active"] = len(versions) - 1
        slot["pending"] = False
        changed = True
    return changed


def switch_fork(sess, turn, index):
    """show version `index` (0-based) of the fork at `turn`. The live suffix is written back first.
    A compaction that covered the forked messages is dropped, so the next send summarizes the version on screen."""
    if isinstance(turn, bool) or isinstance(index, bool):
        raise ForkError("say which version to show")
    try:
        turn, index = int(turn), int(index)
    except (TypeError, ValueError):
        raise ForkError("say which version to show")
    commit_forks(sess)
    forks = sess.get("forks")
    if not isinstance(forks, dict):
        raise ForkError("that message has no other versions")
    slot = forks.get(str(turn))
    if not isinstance(slot, dict) or not isinstance(slot.get("versions"), list) or len(slot["versions"]) < 2:
        raise ForkError("that message has no other versions")
    if slot.get("pending"):
        raise ForkError("That version is still being saved.", "busy", 409)
    versions = slot["versions"]
    if not 0 <= index < len(versions):
        raise ForkError("that version is not in this chat")
    try:
        active = int(slot.get("active") or 0)
    except (TypeError, ValueError):
        active = 0
    if index == active:
        return sess
    turns = list(sess.get("turns") or [])
    messages = list(sess.get("messages") or [])
    start = _fork_start(sess, turn)
    nested = _take_nested(forks, turn, str(turn))
    if 0 <= active < len(versions):
        versions[active] = {"turns": _copy(turns[turn:]), "messages": _copy(messages[start:]), "forks": _copy(nested)}
    ver = versions[index] if isinstance(versions[index], dict) else {}
    sess["turns"] = turns[:turn] + _copy(ver.get("turns") or [])
    sess["messages"] = messages[:start] + _copy(ver.get("messages") or [])
    for k, v in (ver.get("forks") or {}).items():
        if isinstance(v, dict):
            forks[str(k)] = _copy(v)
    slot["active"] = index
    cst = sess.get("compact")
    if isinstance(cst, dict):
        try:
            upto = int(cst.get("upto") or 0)
        except (TypeError, ValueError):
            upto = 0
        if upto > start:
            sess.pop("compact", None)
    return sess


def for_client(sess, running=None):
    """the session the page sees: no stored versions, and a 1-based branch count where there is a choice."""
    forks = sess.get("forks") if isinstance(sess.get("forks"), dict) else {}
    turns = []
    for i, t in enumerate(sess.get("turns") or []):
        if not isinstance(t, dict):
            turns.append(t)
            continue
        t = dict(t)
        slot = forks.get(str(i))
        if isinstance(slot, dict) and not slot.get("pending") and isinstance(slot.get("versions"), list):
            n = len(slot["versions"])
            if n > 1:
                try:
                    active = int(slot.get("active") or 0)
                except (TypeError, ValueError):
                    active = 0
                t["branch"] = {"i": active + 1, "n": n}
        turns.append(t)
    out = {k: v for k, v in sess.items() if k != "forks"}
    out["turns"] = turns
    out["running"] = running
    return out


def title_from(text):
    """the automatic name of a new chat. A control tag is not part of the title. A name the user typed is stored as they wrote it and does not come through here."""
    t = re.sub(r"\[\[[^\]]*\]\]", " ", str(text or ""))
    t = " ".join(t.split())
    return (t[:57] + "...") if len(t) > 60 else (t or "New chat")


def to_markdown(sess):
    lines = [f"# {sess.get('title') or 'Chat'}", "",
             f"_PXA Control chat, {time.strftime('%Y-%m-%d %H:%M', time.localtime(sess.get('created') or 0))}"
             f" - server: {sess.get('server_name') or sess.get('server_key') or '?'} - preset: {sess.get('preset')}_", ""]
    for t in sess.get("turns") or []:
        lines += ["## You", "", str(t.get("user") or ""), "", "## Assistant", ""]
        for st in t.get("steps") or []:
            mark = "ok" if st.get("ok") else ("skipped" if st.get("decision") in ("deny", "timeout") else "failed")
            lines.append(f"- **{st.get('say')}** ({mark})")
            lines.append("  ```json")
            lines.append("  " + json.dumps(st.get("args"), ensure_ascii=False)[:2000])
            lines.append("  ```")
        if t.get("steps"):
            lines.append("")
        lines += [str(t.get("assistant") or ""), ""]
        end = t.get("end") or {}
        if end.get("type") == "run.error":
            lines += [f"> Error: {end.get('data', {}).get('error')}", ""]
        elif end.get("type") == "run.cancelled":
            lines += ["> Stopped.", ""]
    return "\n".join(lines)
