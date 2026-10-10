"""HTTP routes for the chat agent, registered into PXA Control with one ROUTES.update()/STATIC.update().
Errors are {"error", "code"} with an HTTP status. Live events go out over SSE (/api/chat/events)."""
import json
import os
import re
import shutil
import threading
import time

from . import approvals as AP
from . import guards, loop, memory, oai, picker, search, sessions, tools

STYLE = ("Answer the question directly, in your own words. Match the length and tone of the user's message: "
         "a short question gets a short answer, a casual one a casual reply. Use Markdown (lists, tables, code "
         "blocks) only when it helps. Do not announce or narrate what you are about to do, and do not describe "
         "your tools or steps; when you used a tool, give the answer it led to, in context.")
PRESETS = [
    {"id": "chat", "name": "Chat", "blurb": "Just talk. No tools, fastest replies.",
     "system": "You are a friendly, helpful assistant. " + STYLE, "tools": [], "temperature": 0.7, "max_steps": 1, "thinking": "off"},
    {"id": "assistant", "name": "Assistant", "blurb": "Helps with everyday tasks: notes, files, math, web pages.",
     "system": "You are a capable personal assistant running on the user's own computer. Use your tools when they "
               "help: files in your sandbox folder, exact math with the calculator, reading web pages. " + STYLE,
     "tools": ["list_files", "read_file", "write_file", "search_files", "calculate", "web_fetch", "spawn_agent",
               "note_write", "note_read"],
     "temperature": 0.6, "max_steps": 8, "thinking": "off"},
    {"id": "researcher", "name": "Researcher", "blurb": "Looks things up, reads sources and cites them.",
     "system": "You are a careful researcher. Find and read sources before answering (search the web when you "
               "can, open the best pages with web_fetch), then answer with what they say. Cite the sources you "
               "used as Markdown links. Say plainly when you are not sure. On hard questions you may ask another "
               "server for a second opinion. " + STYLE,
     "tools": ["web_search", "web_fetch", "calculate", "ask_server", "write_file", "read_file", "spawn_agent",
               "note_write", "note_read"],
     "temperature": 0.4, "max_steps": 12},
    {"id": "coder", "name": "Coder", "blurb": "Writes and fixes code in its own sandbox folder.",
     "system": "You are an expert programmer working in your sandbox folder. Read files before you change them "
               "and write complete files, never placeholders. In Advanced mode you may run programs to test your "
               "work; the user approves every run. Keep explanations short. " + STYLE,
     "tools": ["list_files", "read_file", "write_file", "search_files", "calculate", "run_command", "spawn_agent",
               "note_write", "note_read"],
     "temperature": 0.2, "max_steps": 16},
]
PRESET_BY = {p["id"]: p for p in PRESETS}
MAX_RUNS = 6
MAX_HISTORY = 80
HOST_STAR = "*"                     # the switch set before a chat exists: applies to the next chat on this page


class ChatError(Exception):
    def __init__(self, msg, code, status=400):
        Exception.__init__(self, msg)
        self.code, self.status = code, status


class HostPolicy(object):
    """what PXA Control hands the chat hub about host access: whether it is permitted at all here (a
    Control exposed beyond localhost refuses it), where the allowlist file is, and where an audit record goes.
    Every default refuses, so a hub built without a policy cannot run a host command."""

    def __init__(self, gate=None, allow_path=None, audit=None):
        self._gate = gate or (lambda: (False, "host access is not available on this Control"))
        self.allow_path = allow_path
        self.audit = audit

    def gate(self):
        """-> (ok, why). Reads the --lan / expose decision each time, so the answer follows the live state."""
        try:
            ok, why = self._gate()
            return bool(ok), str(why or "")
        except Exception as e:                 # noqa: BLE001   a broken gate refuses, it does not open
            return False, f"host access could not be checked ({e.__class__.__name__})"


class Hub(object):
    """per-Control chat state: runs, approvals, sessions, the remembered server."""

    def __init__(self, cfg_root, load_config=None, host=None):
        self.root = cfg_root
        self.load_config = load_config or (lambda: {})
        self.store = sessions.Ephemeral(sessions.Store(cfg_root))
        self.memory = memory.Store(cfg_root)       # the user's, so shared by every chat on every server
        self.approvals = AP.Approvals(float(os.environ.get("PXA_CHAT_APPROVAL_TIMEOUT", "300")))
        self.runs = {}
        self.active = {}             # chat id -> run id
        self.lock = threading.Lock()
        self.host = host or HostPolicy()
        self.host_chats = set()      # chat ids with host access on, plus HOST_STAR; in memory only, never saved

    # ---- remembered state -------------------------------------------------------------------
    def _state_path(self):
        return os.path.join(self.root, "state.json")

    def state(self):
        try:
            with open(self._state_path(), encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def set_state(self, **kw):
        d = self.state()
        d.update(kw)
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        tmp = self._state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(tmp, self._state_path())

    def search_cfg(self):
        """the web-search settings (Settings > Web search), stored with the chat state."""
        return search.config(self.state().get("search"))

    def search_url(self):          # kept for callers that only need "is search on": the built-in provider is always there
        return "builtin"

    def sandbox(self, chat_id):
        return os.path.join(self.root, "sandbox", chat_id)

    # ---- host access (session only; nothing here is ever written to state.json) -----------------
    def host_allowlist(self):
        """the allowlist as of now. Re-read per run, so an edit takes effect on the next message."""
        return guards.host_allow_load(self.host.allow_path) if self.host.allow_path else {"commands": [], "read_paths": []}

    def host_status(self, chat_id=""):
        """-> {"available", "on", "why"}. `available` is the deployment gate (a Control exposed beyond
        localhost refuses until the owner sets the flag); `on` is this chat's switch. Both must hold."""
        ok, why = self.host.gate()
        if not ok:
            return {"available": False, "on": False, "why": why}
        with self.lock:
            on = HOST_STAR in self.host_chats or bool(chat_id and chat_id in self.host_chats)
        return {"available": True, "on": on, "why": ""}

    def host_set(self, chat_id, on):
        cid = chat_id or HOST_STAR
        with self.lock:
            self.host_chats.add(cid) if on else self.host_chats.discard(cid)
        self.approvals.forget_all_host()      # the switch moved: no session grant outlives it
        return self.host_status(chat_id)

    def host_ctx(self, chat_id):
        """-> (host_ok, why, allow) for a run. The refusal sentence is shown to the model and the user alike."""
        st = self.host_status(chat_id)
        allow = self.host_allowlist()
        if not st["available"]:
            return False, st["why"] or "host access is off", allow
        if not st["on"]:
            return False, "host access is off for this chat (turn it on in the chat panel)", allow
        return True, "", allow

    def gc(self):
        now = time.time()
        with self.lock:
            for k, r in list(self.runs.items()):
                if r.done and now - (r.ended or now) > 1800:
                    self.runs.pop(k, None)


def _err(reply, e):
    return reply(json.dumps({"error": str(e), "code": e.code}), "application/json; charset=utf-8", e.status)


def _q1(query, k, d=None):
    v = (query or {}).get(k)
    return v[0] if v else d


_IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
_IMAGE_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_IMAGE_MAX = 4
_IMAGE_CHARS = 1400000


def take_images(body):
    """up to four small images as {media_type, data} base64. Junk is dropped. The bytes are not logged."""
    raw = body.get("images") if isinstance(body, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for im in raw:
        if len(out) >= _IMAGE_MAX:
            break
        if not isinstance(im, dict):
            continue
        mt = str(im.get("media_type") or "").lower()
        data = im.get("data")
        if not isinstance(data, str):
            continue
        data = "".join(data.split())
        if data.startswith("data:"):
            head, _, b64 = data.partition(",")
            if ";base64" not in head.lower() or not b64:
                continue
            got = head[5:].split(";", 1)[0].lower()
            if got not in _IMAGE_TYPES:
                continue
            mt, data = got, b64
        if mt not in _IMAGE_TYPES or not data or len(data) > _IMAGE_CHARS:
            continue
        if not _IMAGE_RE.fullmatch(data):
            continue
        out.append({"media_type": mt, "data": data})
    return out


def _num(body, k, lo, hi, cast=float):
    v = body.get(k)
    if v is None or v == "":
        return None
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise ChatError(f"{k} must be a number", "bad_request")
    if not lo <= v <= hi:
        raise ChatError(f"{k} must be between {lo} and {hi}", "bad_request")
    return v


class SSE(object):
    """returned by the events route: Control's dispatcher hands it the request handler (stream_to)."""

    def __init__(self, run, since):
        self.run, self.since = run, since

    def stream_to(self, h):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-store")
        h.send_header("X-Accel-Buffering", "no")
        h.end_headers()
        seq = self.since
        last = time.time()
        try:
            while True:
                evs, done = self.run.since(seq, 10.0)
                for ev in evs:
                    seq = ev["seq"]
                    h.wfile.write(f"id: {seq}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())
                if evs:
                    h.wfile.flush()
                    last = time.time()
                if done and seq >= len(self.run.events):
                    return
                if time.time() - last > 12:
                    h.wfile.write(b": ping\n\n")
                    h.wfile.flush()
                    last = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def make_routes(hub_getter, reply):
    """-> {(method, path): fn(app, body, query)}. hub_getter(app) -> Hub."""

    def wrap(fn):
        def route(app, body, query):
            try:
                if body is not None and not isinstance(body, dict):
                    raise ChatError("the request body must be a JSON object", "bad_request")
                return fn(app, hub_getter(app), body or {}, query or {})
            except ChatError as e:
                return _err(reply, e)
        route.__name__ = "r_chat_" + fn.__name__
        return route

    def servers(app, hub, body, query):
        lst = picker.list_servers(app)
        last = hub.state().get("last_server")
        key, why = picker.choose(lst, last)
        return {"servers": lst, "selected": key, "reason": why, "last": last, "search": bool(hub.search_url()),
                "host": hub.host_status(_q1(query, "session") or "")}

    def select(app, hub, body, query):
        key = str(body.get("key") or "")
        if not key:
            raise ChatError("pick a server", "bad_request")
        hub.set_state(last_server=key)
        return {"ok": True, "last": key}

    def presets(app, hub, body, query):
        su = hub.search_url()
        return {"presets": PRESETS, "search": bool(su),
                "tools": [{"name": n, "label": tools.LABELS[n][0], "blurb": tools.LABELS[n][1],
                           "advanced_only": n in tools.ADVANCED_ONLY, "host_only": n in tools.HOST_ONLY,
                           "available": True}
                          for n in tools.ORDER]}

    def search_get(app, hub, body, query):
        c = hub.search_cfg()
        found = {k: v for k, v in search.detect().items() if c.get(k) != v}
        return {"settings": search.public(c), "found": found, "providers": list(search.PROVIDERS)}

    def search_post(app, hub, body, query):
        cur = hub.search_cfg()
        new = dict(cur)
        for k in search.DEFAULTS:
            if isinstance(body.get(k), str):
                v = body[k].strip()
                if k in search.SECRETS and not v and not body.get("clear_" + k):
                    continue           # a blank secret field keeps the saved one
                new[k] = v
        new = search.config(new) if not os.environ.get("PXA_CHAT_SEARCH_URL") else new
        hub.set_state(search={k: new.get(k, "") for k in search.DEFAULTS})
        return {"ok": True, "settings": search.public(search.config(hub.state().get("search")))}

    def search_test(app, hub, body, query):
        cfg = dict(hub.search_cfg())
        for k in search.DEFAULTS:
            if isinstance(body.get(k), str) and (body[k].strip() or k not in search.SECRETS):
                cfg[k] = body[k].strip()
        cfg = search.config(cfg)
        t0 = time.time()
        try:
            res, note = search.search(str(body.get("query") or "pxa llama.cpp pascal"), 5, cfg)
            return {"ok": True, "results": res, "note": note, "ms": int((time.time() - t0) * 1000)}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    def run(app, hub, body, query):
        text = str(body.get("message") or "").strip()
        if not text:
            raise ChatError("write a message first", "bad_request")
        if len(text) > 200000:
            raise ChatError("that message is too long", "too_large", 413)
        advanced = bool(body.get("advanced"))
        preset = PRESET_BY.get(body.get("preset") or "assistant")
        if not preset:
            raise ChatError("unknown preset", "bad_request")
        sid = body.get("session_id")
        prev = hub.runs.get(hub.active.get(sid) or "") if sid else None
        if prev is not None and prev.done:
            prev.saved.wait(10)          # the last turn is on disk before this one reads the history
        sess = hub.store.get(sid) if sid else None
        if sid and sess is None:
            raise ChatError("that chat no longer exists", "unknown_session", 404)
        rewind = body.get("rewind")
        if rewind is not None and (isinstance(rewind, bool) or not isinstance(rewind, int) or sess is None
                                   or not 0 <= rewind < len(sess.get("turns") or [])):
            raise ChatError("that message is not in this chat any more", "bad_rewind")
        key = str(body.get("server") or (sess or {}).get("server_key") or hub.state().get("last_server") or "")
        lst = picker.list_servers(app)
        if not key:
            key, _why = picker.choose(lst, None)
        if not key:
            raise ChatError("No server is running yet. Start one on the Launch tab, then come back.", "no_server", 409)
        srv = next((s for s in lst if s["key"] == key), None)
        if srv is None:
            raise ChatError("That server is not one PXA Control knows. Pick one from the list.", "unknown_server", 404)
        su = hub.search_url()
        host_ok, host_why, host_allow = hub.host_ctx(sid or "")
        allowed = set(tools.available(su, advanced, host_ok))
        if advanced and isinstance(body.get("tools"), list):
            names = [n for n in body["tools"] if n in allowed]
        else:
            names = [n for n in preset["tools"] if n in allowed]
        if body.get("web") is False:       # the chat's web switch is off: no search, no page fetching
            names = [n for n in names if n not in ("web_search", "web_fetch")]
        settings = {"tools": names, "max_steps": preset["max_steps"], "temperature": preset["temperature"],
                    "system": preset["system"]}
        if preset.get("thinking") in ("on", "off"):     # quick everyday modes skip a reasoning model's long think
            settings["thinking"] = preset["thinking"]
        if advanced:
            if isinstance(body.get("system"), str):
                settings["system"] = body["system"][:20000]
            for k, lo, hi, cast in (("temperature", 0, 2, float), ("top_p", 0, 1, float), ("top_k", 0, 500, int),
                                    ("min_p", 0, 1, float), ("max_tokens", 1, 262144, int),
                                    ("repeat_penalty", 0.5, 2, float), ("seed", -1, 2 ** 31, int),
                                    ("max_steps", 1, 40, int), ("step_timeout", 10, 1800, int)):
                v = _num(body, k, lo, hi, cast)
                if v is not None:
                    settings[k] = v
            if body.get("thinking") in ("on", "off", "auto"):
                settings["thinking"] = body["thinking"]
            if body.get("tool_mode") in ("auto", "native", "text"):
                settings["tool_mode"] = body["tool_mode"]
        if isinstance(body.get("use_memory"), bool) and sess is not None:
            sess["no_memory"] = not body["use_memory"]
        no_mem = (not body["use_memory"]) if isinstance(body.get("use_memory"), bool) else bool((sess or {}).get("no_memory"))
        if isinstance(body.get("incognito"), bool):
            incognito = body["incognito"]
            if sess is not None:
                sess["incognito"] = incognito
        else:
            incognito = bool((sess or {}).get("incognito"))
        if incognito:
            no_mem = True          # stronger than the per-chat memory switch: nothing is saved or recalled
        if isinstance(body.get("compact"), bool):
            compact_on = body["compact"]
        else:
            compact_on = True if sess is None else bool((sess or {}).get("compact_on", True))
        th = body.get("compact_threshold")
        if isinstance(th, (int, float)) and not isinstance(th, bool) and 0.02 <= float(th) <= 0.95:
            compact_at = float(th)
        else:
            compact_at = float((sess or {}).get("compact_at") or 0.75)
            if not 0.02 <= compact_at <= 0.95:
                compact_at = 0.75
        settings["compact"] = compact_on
        settings["compact_threshold"] = compact_at
        settings["force_compact"] = body.get("force_compact") is True
        ck = body.get("compact_keep")
        if isinstance(ck, int) and not isinstance(ck, bool) and 1 <= ck <= 12:
            settings["compact_keep"] = ck
        raw_refs = body.get("refs")
        if isinstance(raw_refs, list):
            settings["refs"] = [str(x) for x in raw_refs if isinstance(x, str)][:4]
        for sk, lo, hi in (("sub_max", 1, 4), ("sub_steps", 1, 8), ("sub_seconds", 15, 120)):
            v = body.get(sk)
            if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi:
                settings[sk] = v
        settings["images"] = take_images(body)
        for pk, lo, hi, cast in (("repeat_penalty", 0.5, 2, float), ("seed", -1, 2 ** 31, int)):
            if settings.get(pk) is not None:
                continue
            v = body.get(pk)
            if isinstance(v, bool) or v is None or v == "":
                continue
            settings[pk] = _num(body, pk, lo, hi, cast)
        if (not advanced) and body.get("system_snippet") is True and isinstance(body.get("system"), str):
            settings["system"] = body["system"][:20000]
        mset = hub.memory.settings()
        mem_on = mset["enabled"] and not no_mem
        recalled, chat_hits = [], []
        if mem_on:
            names = names + [n for n in tools.MEMORY_TOOLS if n not in names]
            settings["tools"] = names
            settings["max_steps"] = max(3, settings["max_steps"])   # room to save a fact and still answer
            prior = [sessions.message_text(m) for m in (sess or {}).get("messages") or [] if m.get("role") == "user"][-1:]
            query = " ".join(prior + [text])
            recalled = hub.memory.recall(query)
            used = sum(len(f["text"]) // 4 + 6 for f in recalled)
            chat_hits = memory.pick_summaries(query, sessions.summaries(hub.store, (sess or {}).get("id")),
                                              mset["budget"], used)
            block = memory.block(recalled, chat_hits)
            if block:
                settings["memory_block"] = block
        if "ask_server" in names:
            settings["other_servers"] = ", ".join(f"{s['name']} (key {s['key']})" for s in lst
                                                  if s["key"] != key and s["status"] in ("ready", "busy")) or "none"
        with hub.lock:
            if sess is None:
                sess = hub.store.create(key, preset["id"])
                if no_mem:
                    sess["no_memory"] = True
                if incognito:
                    sess["incognito"] = True
                    sess["no_memory"] = True
            prev = hub.runs.get(hub.active.get(sess["id"]) or "")
            if prev is not None and not prev.done:
                raise ChatError("This chat is still answering. Wait for it or press Stop.", "busy", 409)
            if sum(1 for r in hub.runs.values() if not r.done) >= MAX_RUNS:
                raise ChatError("Too many chats are answering at once. Try again in a moment.", "too_many", 429)
            r = loop.Run(sess["id"])
            hub.runs[r.id] = r
            hub.active[sess["id"]] = r.id
        with sessions.mutate_lock:
            if sid:
                fresh = hub.store.get(sid)
                if fresh is None:
                    raise ChatError("that chat no longer exists", "unknown_session", 404)
                sess = fresh
                if rewind is not None and not 0 <= rewind < len(sess.get("turns") or []):
                    raise ChatError("that message is not in this chat any more", "bad_rewind")
            if rewind is not None:               # edit or regenerate: keep the old turns as a version, then start the new one
                start = sessions.stash_fork(sess, rewind)
                sess["messages"] = (sess.get("messages") or [])[:start]
                sess["turns"] = (sess.get("turns") or [])[:rewind]
                cst = sess.get("compact")
                if isinstance(cst, dict) and int(cst.get("upto") or 0) > len(sess.get("messages") or []):
                    sess.pop("compact", None)
            sess["compact_on"], sess["compact_at"] = compact_on, compact_at
            sess.update(server_key=key, server_name=srv["name"], preset=preset["id"])
            if not sess.get("turns") and not sess.get("titled_by_user"):
                sess["title"] = sessions.title_from(text)
            hub.store.save(sess)
        hub.set_state(last_server=key)
        ctx = tools.Ctx(hub.sandbox(sess["id"]), chat_id=sess["id"], run_id=r.id, advanced=advanced, app=app,
                        current_key=key, search_url=su, search_cfg=hub.search_cfg(), cancel_ev=r.cancel_ev,
                        memory=hub.memory if mem_on else None, sessions=hub.store,
                        host_ok=host_ok, host_why=host_why, host_allow=host_allow, host_audit=hub.host.audit)
        all_msgs = list(sess.get("messages") or [])
        if compact_on or settings.get("force_compact"):
            history = all_msgs          # compaction, not a silent 80-message cut, decides what is sent
        else:
            history = all_msgs[-MAX_HISTORY:]
            while history and history[0].get("role") == "tool":
                history = history[1:]
        settings["compact_state"] = sess.get("compact") if isinstance(sess.get("compact"), dict) else None
        ag = loop.Agent(r, lambda: picker.resolve(app, key), hub.approvals, ctx, history, text, settings)

        def work():
            ag.go()
            with sessions.mutate_lock:
                s2 = hub.store.get(sess["id"]) or sess
                start = len(s2.get("messages") or [])
                if r.events and r.events[-1]["type"] == "run.done":
                    fresh = list(r.new_messages)
                else:
                    fresh = [m for m in r.new_messages if m["role"] == "user"] + (
                        [{"role": "assistant", "content": loop.transcript(r)["assistant"] or "(stopped)"}])
                s2.setdefault("messages", []).extend(tools.redact_note_messages(fresh))
                s2.setdefault("turns", []).append(dict(loop.transcript(r), msg_start=start))
                if isinstance(getattr(r, "compact_state", None), dict):
                    s2["compact"] = {k: r.compact_state.get(k) for k in
                                     ("summary", "upto", "turns", "method", "estimated", "note")}
                sessions.commit_forks(s2)
                try:
                    hub.store.save(s2)
                finally:
                    r.saved.set()
            hub.gc()
        threading.Thread(target=work, daemon=True, name="pxa-chat-" + r.id).start()
        mem_info = {"on": mem_on, "recalled": [f["id"] for f in recalled]}
        if mem_on:
            mem_info["chats"] = [c["id"] for c in chat_hits]
        return {"run_id": r.id, "session_id": sess["id"], "server": srv["name"], "tools": names,
                "memory": mem_info,
                "settings": {k: v for k, v in settings.items() if k not in ("other_servers", "memory_block", "ref_block", "images")}}

    def events(app, hub, body, query):
        rid = _q1(query, "run")
        r = hub.runs.get(rid or "")
        if r is None:
            raise ChatError("that run is not known (it may have finished a while ago)", "unknown_run", 404)
        try:
            since = max(0, int(_q1(query, "since", "0")))
        except ValueError:
            since = 0
        return SSE(r, since)

    def cancel(app, hub, body, query):
        r = hub.runs.get(str(body.get("run_id") or ""))
        if r is None:
            raise ChatError("that run is not known", "unknown_run", 404)
        r.cancel()
        return {"ok": True}

    def approve(app, hub, body, query):
        ok, code = hub.approvals.answer(str(body.get("approval_id") or ""), str(body.get("run_id") or ""),
                                        str(body.get("decision") or ""))
        if not ok:
            msg = {"unknown_approval": "that approval is not waiting any more (it may have timed out)",
                   "wrong_run": "that approval belongs to another chat", "bad_decision": "decision must be once, always or deny",
                   "not_allowed": "this kind of action asks every time; 'always' is not offered"}[code]
            raise ChatError(msg, code, 404 if code in ("unknown_approval", "wrong_run") else 400)
        return {"ok": True}

    def session_list(app, hub, body, query):
        return {"sessions": hub.store.list()}

    def session_get(app, hub, body, query):
        s = hub.store.get(_q1(query, "id") or "")
        if s is None:
            raise ChatError("that chat no longer exists", "unknown_session", 404)
        rid = hub.active.get(s["id"])
        run = hub.runs.get(rid) if rid else None
        return sessions.for_client(s, run.id if run and not run.done else None)

    def branch(app, hub, body, query):
        sid = str(body.get("session_id") or body.get("id") or "")
        turn, index = body.get("turn"), body.get("index")
        if isinstance(turn, bool) or isinstance(index, bool) or not isinstance(turn, int) or not isinstance(index, int):
            raise ChatError("say which version to show", "bad_branch")
        run = hub.runs.get(hub.active.get(sid) or "")
        if run is not None and not run.done:
            raise ChatError("This chat is still answering. Wait for it or press Stop.", "busy", 409)
        if run is not None and run.done:
            run.saved.wait(30)
        if run is not None and not run.saved.is_set():
            raise ChatError("That reply is still being saved. Try again in a moment.", "busy", 409)
        with sessions.mutate_lock:
            sess = hub.store.get(sid)
            if sess is None:
                raise ChatError("that chat no longer exists", "unknown_session", 404)
            try:
                sessions.switch_fork(sess, turn, index)
            except sessions.ForkError as e:
                raise ChatError(str(e), e.code, e.status)
            hub.store.save(sess)
        return {"ok": True, "turn": turn, "index": index}

    def tokenize(app, hub, body, query):
        # A count of the text the page is showing. It is not stored and not logged.
        text = body.get("text") if isinstance(body.get("text"), str) else ""
        n_est = (len(text) + 3) // 4
        if len(text) > 400000:
            return {"tokens": n_est, "estimated": True}
        key = str(body.get("server") or hub.state().get("last_server") or "")
        lst = picker.list_servers(app)
        if not key:
            key, _why = picker.choose(lst, None)
        srv = next((s for s in lst if s["key"] == key and s.get("status") in ("ready", "busy", "starting")), None)
        if srv is not None:
            try:
                res = oai.post_json(srv["base_url"], "/tokenize", {"content": text}, timeout=8)
                toks = res.get("tokens") if isinstance(res, dict) else None
                if isinstance(toks, list):
                    return {"tokens": len(toks), "estimated": False}
            except oai.OAIError:
                pass
        return {"tokens": n_est, "estimated": True}

    def session_delete(app, hub, body, query):
        sid = str(body.get("id") or "")
        try:
            ok = hub.store.delete(sid)
        except KeyError:
            ok = False
        if not ok:
            raise ChatError("that chat no longer exists", "unknown_session", 404)
        hub.approvals.forget_chat(sid)
        box = hub.sandbox(sid)                   # its sandbox folder goes with it (sid passed the id check above)
        if os.path.isdir(box) and not os.path.islink(box):
            shutil.rmtree(box, ignore_errors=True)
        return {"ok": True}

    def session_rename(app, hub, body, query):
        s = hub.store.get(str(body.get("id") or ""))
        if s is None:
            raise ChatError("that chat no longer exists", "unknown_session", 404)
        title = " ".join(str(body.get("title") or "").split())[:80]
        if not title:
            raise ChatError("give the chat a name", "bad_request")
        s["title"], s["titled_by_user"] = title, True
        hub.store.save(s)
        return {"ok": True, "title": title}

    def export(app, hub, body, query):
        s = hub.store.get(_q1(query, "id") or "")
        if s is None:
            raise ChatError("that chat no longer exists", "unknown_session", 404)
        fmt = _q1(query, "format", "md")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", s.get("title") or "chat").strip("-")[:40] or "chat"
        if fmt == "json":
            return reply(json.dumps(s, indent=1, ensure_ascii=False), "application/json; charset=utf-8", 200,
                         {"Content-Disposition": f'attachment; filename="{safe}.json"'})
        if fmt != "md":
            raise ChatError("format must be md or json", "bad_request")
        return reply(sessions.to_markdown(s), "text/markdown; charset=utf-8", 200,
                     {"Content-Disposition": f'attachment; filename="{safe}.md"'})

    def memory_get(app, hub, body, query):
        titles, facts = {}, []
        for raw in hub.memory.facts():
            f = dict(raw)
            sid = f.get("source")
            if isinstance(sid, str) and sid not in ("you", "import", ""):
                if sid not in titles:
                    s = hub.store.get(sid)
                    titles[sid] = (s.get("title") if isinstance(s, dict) else None) or "a deleted chat"
                f["source_title"] = titles[sid]
            facts.append(f)
        return dict(hub.memory.settings(), facts=facts, limit=memory.MAX_FACTS)

    def memory_post(app, hub, body, query):
        op, m = str(body.get("op") or ""), hub.memory
        fid = str(body.get("id") or "")
        try:
            if op == "add":
                r = m.add(body.get("text"), source=body.get("source") or "you", pinned=bool(body.get("pinned")))
                return {"ok": True, "fact": r["fact"], "status": r["status"]}
            if op in ("update", "pin"):
                f = m.update(fid, text=body.get("text") if op == "update" else None,
                             pinned=body.get("pinned") if isinstance(body.get("pinned"), bool) else None)
                if f is None:
                    raise ChatError("that memory is gone", "unknown_fact", 404)
                return {"ok": True, "fact": f}
            if op == "delete":
                f = m.forget(fid) if fid else None
                if f is None:
                    raise ChatError("that memory is gone", "unknown_fact", 404)
                return {"ok": True, "fact": f}
            if op == "restore":
                return {"ok": True, "fact": m.restore(body.get("fact"))}
            if op == "clear":
                if body.get("confirm") is not True:
                    raise ChatError("confirm to clear all memories", "confirm_needed")
                return {"ok": True, "cleared": m.clear()}
            if op == "settings":
                return dict(m.set_settings(enabled=body.get("enabled") if isinstance(body.get("enabled"), bool) else None,
                                           k=_num(body, "k", 0, 50, int), budget=_num(body, "budget", 50, 4000, int)),
                            ok=True)
            if op == "import":
                items = body.get("facts")
                if isinstance(items, dict):
                    items = items.get("facts")
                if not isinstance(items, list) or len(items) > 2000:
                    raise ChatError("import a list of facts (the file Export makes)", "bad_request")
                return dict(m.import_facts(items, replace=body.get("replace") is True), ok=True)
        except memory.MemoryError_ as e:
            raise ChatError(f"Not saved: {e}.", "refused", 422)
        raise ChatError("unknown memory action", "bad_request")

    def host_get(app, hub, body, query):
        cid = _q1(query, "session") or ""
        return dict(hub.host_status(cid), chat=cid, allow=hub.host_allowlist())

    def host_post(app, hub, body, query):
        if not isinstance(body.get("on"), bool):
            raise ChatError("send on: true or on: false", "bad_request")
        cid = str(body.get("session_id") or "")
        return dict(hub.host_set(cid, body["on"]), chat=cid, allow=hub.host_allowlist())

    def memory_export(app, hub, body, query):
        d = {"version": 1, "exported": time.strftime("%Y-%m-%dT%H:%M:%S"), "facts": hub.memory.facts()}
        return reply(json.dumps(d, indent=1, ensure_ascii=False), "application/json; charset=utf-8", 200,
                     {"Content-Disposition": 'attachment; filename="pxa-chat-memory.json"'})

    table = {("GET", "/api/chat/servers"): servers, ("POST", "/api/chat/select"): select,
             ("GET", "/api/chat/presets"): presets, ("GET", "/api/chat/search"): search_get,
             ("POST", "/api/chat/search"): search_post, ("POST", "/api/chat/search/test"): search_test, ("POST", "/api/chat/run"): run,
             ("GET", "/api/chat/events"): events, ("POST", "/api/chat/cancel"): cancel,
             ("POST", "/api/chat/approve"): approve, ("GET", "/api/chat/host"): host_get,
             ("POST", "/api/chat/host"): host_post, ("GET", "/api/chat/sessions"): session_list,
             ("GET", "/api/chat/session"): session_get, ("DELETE", "/api/chat/session"): session_delete,
             ("POST", "/api/chat/rename"): session_rename, ("POST", "/api/chat/branch"): branch,
             ("POST", "/api/chat/tokenize"): tokenize,
             ("GET", "/api/chat/export"): export, ("GET", "/api/chat/memory"): memory_get,
             ("POST", "/api/chat/memory"): memory_post, ("GET", "/api/chat/memory/export"): memory_export}
    return {k: wrap(v) for k, v in table.items()}
