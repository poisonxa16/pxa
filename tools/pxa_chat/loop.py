"""The chat agent loop: a stdlib OpenAI-compatible tool-calling loop. Native tool calls when the server's
template supports them, the hive text protocol (protocol.py) otherwise, and the text protocol as a fallback
when a native-mode reply carries a call as text. Max steps, cancel, per-step timeouts, approvals, and a
re-attach wait when the server restarts mid-chat. Every step is an event the UI receives over SSE.

Loop shape follows our in-house agent (its runAgent): call model -> run tools -> feed results
back -> repeat until a reply without a call or the step budget.

Agentic-harness rules (edge-case ids in brackets):
- a tool runs on a worker thread under a per-tool deadline; Stop interrupts it (process group killed, socket shut)
  and the turn ends at once [T14 X03 X05];
- every tool error the model sees is one line with a code: what was wrong and the shape that works [T07 T08 T10 T11];
- the identical call repeated with the identical result is refused, then tools are taken away so the model answers
  [T16]; at most max_calls_per_step calls run per reply [T17]; call ids are unique per turn [T05];
- tool output is clipped to a share of the real per-request window and, from untrusted tools, fenced: template
  control tokens neutralised, instruction-shaped text tagged, and the system prompt says tool text is data [T15 T19 T20];
- the model call gets a first-byte allowance with a progress note (prefill, queued slots) [S12 K04]; a lost or cut
  stream is retried with the dead attempt's text retracted [S06 S07]; 429/503 back off [K08]; a template that refuses
  native tools falls back to the text protocol once [E08]; a server that cannot see images gets the turn without them
  [C12]; a reply cut at the length limit says so [P12]; a message larger than the window is refused up front [C05]."""
import hashlib
import json
import os
import re
import secrets
import threading
import time

from . import compact, guards, oai, protocol, sessions, tools

EVENT_TYPES = ("run.started", "status", "step", "text.delta", "think.delta", "text.retract", "tool.call",
               "approval.request", "approval.answered", "tool.result", "run.done", "run.error", "run.cancelled")
TERMINAL = ("run.done", "run.error", "run.cancelled")
THINK_RE = re.compile(r"(?s)^\s*<think>.*?(</think>|$)\s*")


def strip_think(text):
    """a server without a reasoning parser streams <think>...</think> inside the content: not part of the reply."""
    return THINK_RE.sub("", text or "", count=1)


COALESCE_S = 0.04
COALESCE_MAX = 16384                 # characters: a merged delta never grows past this
COALESCE_TYPES = ("text.delta", "think.delta")


class Run(object):
    def __init__(self, chat_id):
        self.id = "run_" + secrets.token_urlsafe(10)
        self.chat_id = chat_id
        self.events = []
        self.cond = threading.Condition()
        self._pend, self._timer, self._gen, self._last_flush = None, None, 0, 0.0
        try:
            self._co_s = max(0.0, float(os.environ.get("PXA_CHAT_COALESCE_MS", "40"))) / 1000.0
        except ValueError:
            self._co_s = COALESCE_S
        self.cancel_ev = threading.Event()
        self.done = False
        self.created = time.time()
        self.ended = None
        self._stream = None
        self._extra_streams = []         # child model calls; cancel() stops these too
        self._conns = set()              # non-stream helper calls in flight (summary, memory capture); cancel() shuts them
        self._conns_lock = threading.Lock()
        self.saved = threading.Event()   # set once the session file has this turn
        self.new_messages = []        # OpenAI-format messages this run added (user, assistant, tool)
        self.transcript = None        # the UI view of this turn, built from the events

    def _append_locked(self, typ, data, t=None):
        ev = {"seq": len(self.events) + 1, "type": typ, "t": round(t or time.time(), 3), "data": data or {}}
        self.events.append(ev)
        if typ in TERMINAL:
            self.done = True
            self.ended = time.time()
        self.cond.notify_all()
        return ev

    def _flush_locked(self):
        p = self._pend
        if p is None:
            return
        self._pend = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._last_flush = time.time()
        self._append_locked(p["type"], {"step": p["step"], "text": p["text"]}, p["t"])

    def flush(self):
        with self.cond:
            self._flush_locked()

    def _tick(self, gen):
        with self.cond:
            if self._pend is not None and self._pend["gen"] == gen:
                self._flush_locked()

    def emit(self, typ, data=None):
        """append one event. text.delta / think.delta that arrive faster than one per COALESCE_S are merged into one
        event (same step, same kind, in order) and published at most ~40 ms late; every other event first publishes
        what is waiting, so the order never changes and a published event is never edited: a reconnect with
        ?since=N replays exactly what a live reader saw [ST10]. PXA_CHAT_COALESCE_MS=0 turns it off."""
        with self.cond:
            if (self._co_s > 0 and typ in COALESCE_TYPES and isinstance(data, dict) and isinstance(data.get("text"), str)
                    and set(data) <= {"step", "text"}):
                now = time.time()
                p = self._pend
                if p is not None and (p["type"] != typ or p["step"] != data.get("step", 0)):
                    self._flush_locked()
                    p = None
                if p is not None:
                    p["text"] += data["text"]
                    if len(p["text"]) >= COALESCE_MAX:
                        self._flush_locked()
                    return None
                if now - self._last_flush >= self._co_s:           # nothing published lately: no delay at all
                    self._last_flush = now
                    return self._append_locked(typ, data)
                self._gen += 1
                self._pend = {"type": typ, "step": data.get("step", 0), "text": data["text"], "t": now, "gen": self._gen}
                self._timer = threading.Timer(max(0.001, self._last_flush + self._co_s - now), self._tick, (self._gen,))
                self._timer.daemon = True
                self._timer.start()
                return None
            self._flush_locked()
            return self._append_locked(typ, data)

    def since(self, seq, timeout=15.0):
        """events after seq; waits up to timeout for one. -> (events, done)"""
        with self.cond:
            if len(self.events) <= seq and not self.done:
                self.cond.wait(timeout)
            return self.events[seq:], self.done

    def cancel(self):
        self.cancel_ev.set()
        s = self._stream
        if s is not None:
            s.cancel()
        for extra in list(getattr(self, "_extra_streams", None) or []):
            try:
                extra.cancel()
            except Exception:              # noqa: BLE001 — stopping one child must not skip the others
                pass
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            oai.shutdown_conn(c)

    def track(self, conn, alive):
        """on_conn hook for oai.post_json: a helper call in flight can be shut by cancel()."""
        with self._conns_lock:
            if alive:
                self._conns.add(conn)
            else:
                self._conns.discard(conn)
        if alive and self.cancel_ev.is_set():
            oai.shutdown_conn(conn)


def for_mode(msgs, native):
    """history for this server: a text-protocol server gets tool calls and results as plain text."""
    if native:
        return list(msgs)
    out = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = "\n".join(json.dumps(dict(protocol.parse_lenient(tc["function"].get("arguments") or "{}") or {},
                                               tool=tc["function"]["name"])) for tc in m["tool_calls"])
            out.append({"role": "assistant", "content": ((m.get("content") or "") + "\n" + calls).strip()})
        elif m.get("role") == "tool":
            out.append({"role": "user", "content": f"[tool result]\n{m.get('content') or ''}"})
        else:
            out.append(m)
    return out


def lift_system(sysp, msgs):
    """strict chat templates (several Qwen/Gemma ones) raise 'System message must be at the beginning' when a system
    message comes later in the conversation. Control's own turn is always [system, user/assistant/tool ...], but a
    stored or imported history may carry a system-role message: its text is merged into the front system message
    (in order), never left mid-conversation. -> (system text, messages without system-role entries)."""
    lifted, rest = [], []
    for m in msgs:
        if m.get("role") == "system":
            t = compact.message_text(m).strip()
            if t:
                lifted.append(t)
        else:
            rest.append(m)
    if not lifted:
        return sysp, list(msgs)
    return "\n\n".join(([sysp] if sysp else []) + lifted), rest


def _args(raw):
    if isinstance(raw, dict):
        return raw, None
    if not raw or not str(raw).strip():
        return {}, None
    o = protocol.parse_lenient(str(raw))
    if isinstance(o, str) and o.strip().startswith("{"):
        o = protocol.parse_lenient(o)          # arguments double-encoded as a JSON string [T31]
    if not isinstance(o, dict):
        why = "not a JSON object"
        try:
            json.loads(str(raw))
        except ValueError as e:
            why = str(e)
        return None, why
    return o, None


TEMPLATE_RE = re.compile(r"requires --jinja|jinja|chat template|template (?:error|render)|does not support tools|"
                         r"tools? (?:is|are) not supported|unsupported tool", re.I)
IMAGE_RE = re.compile(r"image input is not supported|does not support (?:image|vision|multimodal)|"
                      r"multimodal data provided, but model does not support|mmproj", re.I)


def _template_error(e):
    return getattr(e, "status", None) in (400, 500) and bool(TEMPLATE_RE.search(str(e)))


def _image_error(e):
    return bool(IMAGE_RE.search(str(e)))


def has_images(msgs):
    return any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "image_url"
                                                            for p in m["content"]) for m in msgs or [])


def strip_images(msgs, note):
    """the send list without image parts: each becomes a short text note, and a list of text parts becomes one
    string (a text-only template may not take content parts) [C12]. Stored messages are not touched."""
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            bits = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    bits.append(note)
                elif isinstance(p, dict) and p.get("type") == "text":
                    bits.append(str(p.get("text") or ""))
                elif isinstance(p, str):
                    bits.append(p)
            out.append(dict(m, content="\n".join(b for b in bits if b)))
        else:
            out.append(m)
    return out


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v != v:                                  # NaN
        return default
    return max(lo, min(v, hi))


class Agent(object):
    """one turn: settings in, events out. resolve() -> the current picker entry for the chat's server."""

    def __init__(self, run, resolve, approvals, ctx, history, user_text, settings):
        self.run, self.resolve, self.approvals, self.ctx = run, resolve, approvals, ctx
        self.history, self.user_text, self.s = list(history or []), user_text, settings
        self.tool_names = [n for n in settings.get("tools") or [] if n in tools.SPECS]
        self.max_steps = max(1, min(int(settings.get("max_steps") or 8), 40))
        self.step_timeout = max(10, min(float(settings.get("step_timeout") or 180), 1800))
        self.reattach_s = float(settings.get("reattach_s", 45))
        self.mode = settings.get("tool_mode") or "auto"          # auto | native | text
        # harness settings; every one has a safe default and an Advanced override
        self.first_byte_timeout = _clamp(settings.get("first_byte_timeout") or 900, 10, 7200, 900.0)
        self.tool_timeout = _clamp(settings.get("tool_timeout") or 120, 5, 1800, 120.0)
        self.busy_retries = int(_clamp(settings.get("busy_retries", 4), 0, 10, 4))
        self.loop_guard = int(_clamp(settings.get("loop_guard", 3), 0, 20, 3))          # 0 = off
        self.max_calls = int(_clamp(settings.get("max_calls_per_step", 8), 1, 64, 8))
        self.fence_on = settings.get("tool_fence", True) is not False
        self.lost_retries = 2               # a lost/cut stream is retried at most this many times per turn
        self._lost = self._busy = self._guard_hits = 0
        self._tpl_fallback = self._img_stripped = self._force_answer = False
        self._no_vision = None              # True once the server is known not to take images
        self._calls_seen = {}               # (name, args) -> {"n", "outs"} for the repeated-call guard
        self._ids = set()                   # tool call ids used this turn
        self._srv = None
        self._streamed = False              # a model call was attempted this turn (re-attach waits after that)
        self.last_tool = None        # {"name", "args", "ok", "out", "decision"}: only for an empty final turn
        self.saved_memory = False    # a remember/forget ran this turn (then the memory safety net stays out)
        self._nctx = None            # (n_ctx, estimated) once /props has been read this run
        self._overflow_used = False
        st = settings.get("compact_state")
        self.run.compact_state = dict(st) if isinstance(st, dict) and st.get("upto") else None

    def _server(self):
        """the chat's server, waiting (re-attach) while it restarts."""
        t0, said = time.time(), False
        while True:
            srv = self.resolve()
            if srv and srv["status"] in ("ready", "busy"):
                if said:
                    self.run.emit("status", {"text": f"{srv['name']} is back. Continuing.", "kind": "reattached"})
                return srv
            if srv and srv["status"] == "error" and not srv.get("model") and not self._streamed:
                # up but names no model at the start of the turn: refuse now, not after the re-attach wait [E11].
                # Once a call was made, the same reading is a server coming back (its probe can lag): wait [E02].
                return srv
            if self.run.cancel_ev.is_set():
                return None
            if time.time() - t0 >= self.reattach_s:
                return srv or False
            if not said:
                nm = srv["name"] if srv else "the server"
                self.run.emit("status", {"text": f"Waiting for {nm} to come back... your chat is kept.", "kind": "waiting"})
                said = True
            self.run.cancel_ev.wait(1.0)

    def _system(self, native):
        sysp = (self.s.get("system") or "").strip()
        if self.tool_names and not native:
            sysp = (sysp + "\n\n" if sysp else "") + tools.text_protocol_prompt(self.tool_names)
        elif self.tool_names:
            others = ""
            if "ask_server" in self.tool_names:
                others = "\nOther servers you can ask with ask_server: " + (self.s.get("other_servers") or "none")
            sysp = (sysp + "\n\n" if sysp else "") + ("Files you read or write live in your own sandbox folder; use "
                                                      "relative paths like notes.txt." + others)
        if "remember" in self.tool_names:
            from . import memory
            sysp = (sysp + "\n\n" if sysp else "") + memory.GUIDANCE
        if self.s.get("memory_block"):
            sysp = (sysp + "\n\n" if sysp else "") + self.s["memory_block"]
        if self.s.get("ref_block"):
            sysp = (sysp + "\n\n" if sysp else "") + self.s["ref_block"]
        if self.tool_names and self.fence_on:
            sysp = (sysp + "\n\n" if sysp else "") + guards.DATA_NOTICE
        if self._force_answer:
            sysp = (sysp + "\n\n" if sysp else "") + ("Tools are off for this reply: you repeated the same call "
                                                       "without progress. Answer the user now with what you have.")
        return sysp

    def go(self):
        r = self.run
        try:
            self._go()
        except Exception as e:         # noqa: BLE001
            if not r.done:
                r.emit("run.error", {"error": f"something went wrong inside PXA Control: {e.__class__.__name__}: {e}",
                                     "code": "internal"})
        finally:
            r.flush()                  # a coalesced delta still waiting is published before anyone reads the events

    def _wire(self, msgs):
        """the fields a server understands. pinned and other bookkeeping stay off the wire."""
        out = []
        for m in msgs:
            d = {"role": m.get("role"), "content": m.get("content") if m.get("content") is not None else ""}
            if m.get("tool_calls"):
                d["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                d["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                d["name"] = m["name"]
            out.append(d)
        return out

    def _ctx_limits(self, srv, harder=False):
        """(n_ctx, high, target, estimated). /props once per run; 8192 and estimated=True when it is missing."""
        if self._nctx is None:
            n_ctx, estimated = None, False
            try:
                n_ctx = compact.n_ctx_of(oai.get_json(srv["base_url"], "/props", timeout=3))
            except oai.OAIError:
                n_ctx = None
            if not n_ctx:
                n_ctx, estimated = 8192, True
            self._nctx = (n_ctx, estimated)
        n_ctx, estimated = self._nctx
        high, target = compact.limits(n_ctx, self.s.get("max_tokens"), self.s.get("compact_threshold"), harder=harder)
        return n_ctx, high, target, estimated

    def _tokenize(self, srv, msgs):
        """honest token count of these messages. -> (n, estimated)."""
        text = "\n".join((m.get("role") or "") + "\n" + compact.message_text(m) for m in msgs)
        try:
            res = oai.post_json(srv["base_url"], "/tokenize", {"content": text}, timeout=10)
            toks = res.get("tokens") if isinstance(res, dict) else None
            if isinstance(toks, list):
                return len(toks), False
        except oai.OAIError:
            pass
        return compact.estimate(msgs), True

    def _summarize(self, srv, model, prior_summary, dropped_text):
        user = "Previous summary:\n" + (prior_summary or "(none)") + "\n\nTurns to fold in:\n" + (dropped_text or "")
        body = {"model": model, "stream": False, "temperature": 0, "max_tokens": 700,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": compact.SUMMARY_SYS},
                             {"role": "user", "content": user[:24000]}]}
        res = oai.post_json(srv["base_url"], "/v1/chat/completions", body, 60.0, on_conn=self.run.track)
        reply = strip_think(((res.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        if not str(reply or "").strip():
            raise oai.OAIError("the summary was empty", "empty_summary")
        return reply.strip()

    def _ref_summarize(self, srv, model, title, source):
        """one non-streaming call: a short summary of a chat the user attached with @."""
        user = "Title: " + str(title or "Chat") + "\n\n" + (source or "")
        body = {"model": model, "stream": False, "temperature": 0, "max_tokens": 400,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": sessions.REF_SYS},
                             {"role": "user", "content": user[:14000]}]}
        res = oai.post_json(srv["base_url"], "/v1/chat/completions", body, 45.0, on_conn=self.run.track)
        reply = strip_think(((res.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        if not str(reply or "").strip():
            raise oai.OAIError("the summary was empty", "empty_summary")
        return reply.strip()[:4000]

    def _attach_refs(self, srv, model):
        """put attached chats' summaries into the system prompt. A failed summary still attaches a clip.
        Stop while this runs cancels the rest and does not kill the turn."""
        ids = self.s.get("refs") or []
        if not ids or self.ctx.sessions is None:
            return
        if self.run.cancel_ev.is_set():
            return

        def summarize(title, source):
            if self.run.cancel_ev.is_set():
                raise sessions.StopAttach()
            return self._ref_summarize(srv, model, title, source)

        try:
            block = sessions.attach_refs(self.ctx.sessions, ids, self.ctx.chat_id, summarize)
        except sessions.StopAttach:
            return
        except Exception:           # noqa: BLE001 — attaching context must not sink the turn
            return
        if block:
            self.s["ref_block"] = block

    def _send_view(self, srv, model, sysp, native, force=False, harder=False):
        """the message list to send. Compaction rewrites this list only; self.full stays the real transcript."""
        r = self.run
        on = self.s.get("compact", True)
        if on is False and not force and not harder:
            tail = self.full[-80:]
            while tail and tail[0].get("role") == "tool":
                tail = tail[1:]
            return self._wire(tail)
        _nctx, high, target, est_ctx = self._ctx_limits(srv, harder=harder)
        overhead = compact.estimate([{"role": "system", "content": sysp}]) if sysp else 0
        prior = r.compact_state
        projected = compact.project(self.full, prior)
        # chars/4 well under the line needs no /tokenize round trip (and must not disturb tests that read the
        # last chat request). Near the line, count for real.
        if not force and not harder and compact.estimate(projected) + overhead < high * 0.6:
            return self._wire(projected)
        counted = ([{"role": "system", "content": sysp}] if sysp else []) + self._wire(projected)
        n, est_tok = self._tokenize(srv, counted)
        estimated = bool(est_ctx or est_tok)
        if not force and not harder and n <= high:
            return self._wire(projected)

        def tokens(msgs):
            return compact.estimate(msgs)

        def summarize(prev, dropped):
            return self._summarize(srv, model, prev, dropped)

        keep = self.s.get("compact_keep") or compact.KEEP_TURNS
        # Reaching here means the honest count is over the line, or the user asked. force makes the fold
        # happen even when the chars/4 planner is a little under that honest count.
        send, state, event = compact.decide(
            self.full, prior, overhead, high, target, tokens, summarize,
            keep_turns=keep, force=True, harder=harder, estimated=estimated)
        if r.cancel_ev.is_set():             # Stop during the summary call: keep the stored state as it was [X05]
            return self._wire(projected)
        if state:
            r.compact_state = state
        if event and not harder:
            n2, est2 = self._tokenize(srv, ([{"role": "system", "content": sysp}] if sysp else []) + self._wire(send))
            if n2 > high:
                send2, state2, event2 = compact.decide(
                    self.full, state, overhead, high, target, tokens, summarize,
                    keep_turns=1, force=True, harder=True, estimated=estimated or est2)
                send = send2
                if state2:
                    r.compact_state = state2
                event = event2 or event
        if event:
            note = event.get("note") or ""
            if estimated and "estimated" not in note:
                note = (note + " " if note else "") + "token count estimated (chars/4); /tokenize or /props was unavailable"
            r.emit("compact", {"turns": event["turns"], "method": event["method"], "summary": (event.get("summary") or "")[:4000],
                               "estimated": bool(estimated or event.get("estimated")), "note": note.strip(),
                               "upto": event.get("upto")})
            if event.get("method") == "summary":
                self._capture_summary(srv, model, event.get("summary") or "")
        return self._wire(send)

    def _arm_children(self):
        """fields a sub-agent reads off this turn. A child builds its own context at depth 1."""
        from . import subagent
        c = self.ctx
        c.emit = self.run.emit
        c.run = self.run
        c.approvals = self.approvals
        c.parent_tools = list(self.tool_names)
        c.depth = 0
        c.spawned = 0
        c.sub_max = subagent.clamp(self.s.get("sub_max"), 1, 4, 2)
        c.sub_steps = subagent.clamp(self.s.get("sub_steps"), 1, 8, 4)
        c.sub_seconds = subagent.clamp(self.s.get("sub_seconds"), 15, 120, 60)
        c.resolve = self._child_server
        c.pad = tools.Pad()          # this turn only. The previous turn's notes stay on the saved session.

    def _child_server(self, key=None):
        if not key:
            return self.resolve()
        from . import picker
        return picker.resolve(self.ctx.app, str(key))

    def _user_message(self):
        """text, or text plus image parts when this turn attached images. The bytes are not logged."""
        imgs = [im for im in (self.s.get("images") or []) if isinstance(im, dict) and im.get("data") and im.get("media_type")]
        if not imgs:
            return {"role": "user", "content": self.user_text}
        parts = [{"type": "text", "text": self.user_text}]
        for im in imgs[:4]:
            parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (im["media_type"], im["data"])}})
        return {"role": "user", "content": parts}

    def _go(self):
        self._arm_children()
        r = self.run
        user_msg = self._user_message()
        self.full = list(self.history) + [user_msg]
        r.new_messages.append(user_msg)
        final_text, usage, timings = "", None, None
        for step in range(1, self.max_steps + 1):
            if r.cancel_ev.is_set():
                r.emit("run.cancelled", {"text": "Stopped."})
                return
            srv = self._server()
            if srv is None:
                r.emit("run.cancelled", {"text": "Stopped."})
                return
            if srv is False or not srv:
                r.emit("run.error", {"error": "The server for this chat is not running. Start it on the Servers tab, "
                                              "or pick another one; your chat is kept.", "code": "server_down"})
                return
            model = srv.get("model")
            if not model:      # the seat rule: the model is what the URL reports, never a made-up name
                r.emit("run.error", {"error": f"{srv['name']} did not report a model at /v1/models, so chat will not "
                                              "guess one.", "code": "no_model"})
                return
            native = bool(self.tool_names) and (self.mode == "native" or (self.mode == "auto" and srv["caps"].get("tools")))
            if step == 1:
                r.emit("run.started", {"server": srv["name"], "server_key": srv["key"], "model": model,
                                       "user": self.user_text[:20000],
                                       "tool_mode": ("native" if native else "text") if self.tool_names else "none",
                                       "tools": self.tool_names})
                self._attach_refs(srv, model)
                if r.cancel_ev.is_set():
                    r.emit("run.cancelled", {"text": "Stopped."})
                    return
            r.emit("step", {"n": step, "max": self.max_steps})
            self._srv = srv
            if step == 1 and (self._fit_system(srv, native) or self._too_long(srv)):
                return
            if self._no_vision is None and (srv.get("caps") or {}).get("vision") is False:
                self._no_vision = True             # /props says no vision: leave images out before the first call
            sysp = self._system(native)
            force = bool(self.s.get("force_compact")) and step == 1
            send = self._send_view(srv, model, sysp, native, force=force)
            res = None
            while res is None:
                if self._no_vision and has_images(send):
                    send = strip_images(send, "[image left out: %s cannot see images]" % srv["name"])
                    if not self._img_stripped:
                        self._img_stripped = True
                        r.emit("status", {"text": f"{srv['name']} cannot see images, so the image was left out. "
                                                  "Start a vision model (one with its mmproj file) to use images.",
                                          "kind": "no_vision"})
                sysp_w, send_w = lift_system(sysp, send)
                body = {"model": model, "messages": ([{"role": "system", "content": sysp_w}] if sysp_w else []) + for_mode(send_w, native),
                        "stream_options": {"include_usage": True}}
                for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "repeat_penalty", "seed"):
                    if self.s.get(k) is not None:
                        body[k] = self.s[k]
                for k in ("stop", "json_schema", "response_format", "grammar"):       # P08: validated in routes
                    if self.s.get(k):
                        body[k] = self.s[k]
                if self.s.get("thinking") in ("on", "off"):
                    body["chat_template_kwargs"] = {"enable_thinking": self.s["thinking"] == "on"}
                if native and not self._force_answer:
                    body["tools"] = [tools.SPECS[n] for n in self.tool_names]
                    body["tool_choice"] = "auto"
                st = oai.Stream(srv["base_url"], body, step_timeout=self.step_timeout,
                                idle_timeout=min(self.step_timeout, 120), first_byte_timeout=self.first_byte_timeout,
                                on_wait=lambda waited, srv=srv, step=step: self._waiting(srv, waited, step))
                r._stream = st
                self._streamed = True
                if r.cancel_ev.is_set():
                    st.cancel()
                try:
                    res = st.run(on_text=lambda d: r.emit("text.delta", {"step": step, "text": d}),
                                 on_think=lambda d: r.emit("think.delta", {"step": step, "text": d}))
                except oai.OAIError as e:
                    if e.code == "cancelled" or r.cancel_ev.is_set():
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    if e.code in ("unreachable", "incomplete", "bad_stream") and self._lost < self.lost_retries \
                            and step < self.max_steps:
                        self._lost += 1
                        r.emit("text.retract", {"step": step, "text": ""})      # the dead attempt's words go [S07]
                        why = {"unreachable": "Lost the server mid-reply", "incomplete": "The reply was cut off "
                               "(the server closed the stream early)", "bad_stream": "The server sent an unreadable "
                               "stream"}[e.code]
                        r.emit("status", {"text": why + "; trying again when it is back.", "kind": "retry",
                                          "code": e.code})
                        r.cancel_ev.wait(1.0)
                        res = "retry"
                        break
                    if e.code in ("http_429", "http_503") and self._busy < self.busy_retries:
                        self._busy += 1
                        wait = e.retry_after if e.retry_after is not None else 2.0 * 2 ** (self._busy - 1)
                        wait = max(0.5, min(30.0, float(wait)))
                        r.emit("status", {"text": f"{srv['name']} is busy (HTTP {e.status}); trying again in "
                                                  f"{wait:g} s ({self._busy} of {self.busy_retries}).", "kind": "busy"})
                        if r.cancel_ev.wait(wait):
                            r.emit("run.cancelled", {"text": "Stopped."})
                            return
                        continue
                    if native and not self._tpl_fallback and _template_error(e):
                        self._tpl_fallback = True       # the template refused native tools: text protocol, once [E08]
                        self.mode, native = "text", False
                        r.emit("status", {"text": f"{srv['name']}'s chat template refused tool definitions; using "
                                                  "text tool calls instead.", "kind": "fallback_text"})
                        sysp = self._system(native)
                        continue
                    if not self._img_stripped and has_images(send) and _image_error(e):
                        self._no_vision = True          # the image goes; this and later turns are not poisoned [C12]
                        continue
                    if not self._overflow_used and compact.is_overflow(e):
                        self._overflow_used = True
                        r.emit("status", {"text": "That reply did not fit the context window. Compacting and trying once.",
                                          "kind": "compact"})
                        send = self._send_view(srv, model, sysp, native, force=True, harder=True)
                        continue
                    msg, code = str(e), e.code
                    if compact.is_overflow(e):
                        msg, code = self._overflow_words(srv, e), "context_overflow"
                    r.emit("run.error", {"error": msg, "code": code})
                    return
                finally:
                    r._stream = None
            if res == "retry":
                continue
            usage, timings = res.get("usage") or usage, res.get("timings") or timings
            content = res.get("content") or ""
            cut = res.get("finish_reason") == "length"
            calls, textual = [], False
            for tc in res.get("tool_calls") or []:
                calls.append({"id": tc["id"], "name": tc["name"], "raw": tc["arguments"]})
            if self._force_answer:
                calls = []                         # tools were taken away: whatever came back is the answer
            if not calls and self.tool_names and not self._force_answer:
                prose, call = protocol.split_call(content)
                if call:
                    name, a = protocol.call_to_openai(call)
                    calls = [{"id": "txt_" + secrets.token_hex(4), "name": name, "raw": a,
                              "multi": bool(call.get("_multi"))}]
                    textual = True
                    r.emit("text.retract", {"step": step, "text": prose})
            if not calls:
                # the reply is the model's own words, never wrapped in a fixed prefix/suffix. Only when the model
                # sends an EMPTY final turn after using tools does PXA write one short line from the last result.
                content = strip_think(content).strip()
                if self._force_answer:
                    content = protocol.split_call(content)[0].strip() if protocol.parse_action(content) else content
                fallback = False
                if not content and self.last_tool:
                    lt = self.last_tool
                    content = tools.fallback_line(lt["name"], lt["args"], lt["ok"], lt["out"], lt.get("decision"))
                    fallback = True
                    r.emit("text.delta", {"step": step, "text": content, "fallback": True})
                final_text = content
                am = {"role": "assistant", "content": content}
                self.full.append(am)
                r.new_messages.append(am)
                self._capture(srv, model)
                done = {"text": content, "steps": step, "usage": usage, "timings": timings,
                        "fallback": fallback, "empty": not content}
                if cut:
                    done["truncated"] = True         # [P12] the page says it was cut and offers to continue
                r.emit("run.done", done)
                return
            for c in calls:                          # unique ids per turn [T05]
                base, n = str(c["id"] or "call"), 1
                while c["id"] in self._ids or not c["id"]:
                    n += 1
                    c["id"] = f"{base}_{n}"
                self._ids.add(c["id"])
            if textual:
                am = {"role": "assistant", "content": content}
            else:
                am = {"role": "assistant", "content": content or "",
                      "tool_calls": [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["raw"] if isinstance(c["raw"], str)
                                                   else json.dumps(c["raw"])}} for c in calls]}
            self.full.append(am)
            r.new_messages.append(am)
            batched = {}
            runnable = calls[:self.max_calls]
            spawn_calls = [c for c in runnable if c["name"] == "spawn_agent" and "spawn_agent" in self.tool_names]
            if len(spawn_calls) >= 2:
                from . import subagent
                for c, item in zip(spawn_calls, subagent.run_batch(spawn_calls, self, step)):
                    batched[c["id"]] = item
            for i, c in enumerate(calls):
                if i >= self.max_calls:              # a call storm: the rest are refused with a reason [T17]
                    ok, out = self._refuse(c, step, "too_many_calls",
                                           f"only the first {self.max_calls} tool calls of one reply run; this one "
                                           "was skipped. Send the rest in your next reply, fewer at a time.",
                                           "Skipped: too many calls at once")
                elif c["id"] in batched:
                    item = batched[c["id"]]
                    if item is None:
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    ok, out = item
                    shown, _err = _args(c["raw"])
                    self.last_tool = {"name": "spawn_agent", "args": shown if isinstance(shown, dict) else {},
                                      "ok": ok, "out": out, "decision": None}
                    out = self._for_model("spawn_agent", out)
                else:
                    if r.cancel_ev.is_set():
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    out_msg = self._one_call(c, step, cut=cut)
                    if out_msg is None:          # cancelled while waiting for an approval or while the tool ran
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    ok, out = out_msg
                    if c.get("multi"):
                        out += ("\n[note: your reply had more than one tool call; only the first ran. Send one call "
                                "per reply.]")
                tm = ({"role": "user", "content": f"[result of {c['name']}]\n{out}"} if textual
                      else {"role": "tool", "tool_call_id": c["id"], "content": out})
                self.full.append(tm)
                r.new_messages.append(tm)
        r.emit("run.error", {"error": f"The assistant used all {self.max_steps} steps without finishing. Say "
                                      "'continue' to let it go on, or raise the step limit in Advanced.",
                             "code": "max_steps", "text": final_text})

    def _extract(self, srv, model, ask):
        """one small non-stream extraction. Each fact goes through remember's own path (screen, dedupe, cap).
        A refusal or a failed request never fails the turn."""
        from . import memory
        r = self.run
        if "remember" not in self.tool_names or self.ctx.memory is None or r.cancel_ev.is_set():
            return
        if not str(ask or "").strip():
            return
        body = {"model": model, "stream": False, "temperature": 0, "max_tokens": 220,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": memory.CAPTURE_SYS},
                             {"role": "user", "content": ask[:8000]}]}
        try:
            res = oai.post_json(srv["base_url"], "/v1/chat/completions", body, 60.0, on_conn=self.run.track)
            reply = strip_think(((res.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        except (oai.OAIError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            return
        for fact in memory.parse_capture(reply):
            if memory.screen(fact):                 # a secret / reply-style rule: dropped quietly, never shown
                continue
            cid = "mem_" + secrets.token_hex(4)
            a = {"fact": fact}
            r.emit("tool.call", {"call_id": cid, "name": "remember", "args": a, "say": tools.say("remember", a),
                                 "auto": True})
            self.ctx.memory_event = None
            ok, out = tools.execute("remember", a, self.ctx)
            ev = {"call_id": cid, "ok": ok, "output": out, "chars": len(out), "secs": 0,
                  "say": "Done" if ok else "That didn't work", "done": tools.said("remember", a), "auto": True}
            if self.ctx.memory_event:
                ev["memory"] = self.ctx.memory_event
            r.emit("tool.result", ev)

    def _capture(self, srv, model):
        """the memory safety net (memory.wants_capture): the user said something lasting about themselves, the
        model saved nothing this turn -> one extraction. Never fails the turn."""
        from . import memory
        if self.saved_memory or not memory.wants_capture(self.user_text):
            return
        self._extract(srv, model, memory.CAPTURE_ASK.format(text=self.user_text[:3000]))

    def _capture_summary(self, srv, model, summary):
        """the same extraction, run on a compaction summary. Incognito chats have no memory store, so they
        never reach here. A secret stated in the summary is refused by the same screen."""
        from . import memory
        text = str(summary or "").strip()
        if not text:
            return
        self._extract(srv, model, memory.CAPTURE_ASK_SUMMARY.format(text=text[:6000]))

    # ---- harness helpers --------------------------------------------------------------------------------
    def _waiting(self, srv, waited, step):
        """a progress note while the server has not sent its first byte (prefill, or a queue behind busy slots)."""
        busy = None
        try:
            slots = oai.get_json(srv["base_url"], "/slots", 2.0)
            if isinstance(slots, list) and slots:
                busy = (sum(1 for x in slots if isinstance(x, dict) and x.get("is_processing")), len(slots))
        except oai.OAIError:
            busy = None                      # /slots can be off (--no-slots): say less, never guess
        text = f"{srv['name']} has not started answering yet ({int(waited)} s): it is reading the prompt"
        if busy and busy[1] > 1 and busy[0] >= busy[1]:
            text += f" or waiting for a free slot (all {busy[1]} are busy)"
        text += ". Long prompts take a while on these cards; Stop works."
        data = {"text": text, "kind": "first_byte", "waited": int(waited), "step": step}
        if busy:
            data["slots_busy"], data["slots"] = busy
        self.run.emit("status", data)

    def _usable(self, srv):
        n_ctx, _high, _target, est = self._ctx_limits(srv)
        return compact.usable(n_ctx, self.s.get("max_tokens")), n_ctx, est

    SYS_SHARE = 0.5                   # the system part (prompt + tool text + memory + refs) may take this share of one request

    def _fit_system(self, srv, native):
        """[C06] system prompt + memory + attached chats bigger than the window used to end in a context error that
        compaction could not fix. Memory and attached-chat blocks are trimmed first (attached chats, then memory);
        a system prompt that alone is over about half of the usable window is refused, with the numbers.
        Costs nothing (one chars/4 estimate) while the whole system part is under half the window. -> True = refused."""
        try:
            usable, n_ctx, est = self._usable(srv)
        except Exception:                    # noqa: BLE001 - a courtesy check; the overflow path still exists
            return False
        cap = int(usable * self.SYS_SHARE)

        def toks(text):
            return compact.estimate([{"role": "system", "content": text}]) if text else 0
        full = toks(self._system(native))
        if full <= cap:
            return False
        blocks = {k: self.s.get(k) or "" for k in ("memory_block", "ref_block")}
        for k in blocks:
            self.s[k] = ""
        fixed_text = self._system(native)
        fixed = toks(fixed_text)
        if fixed > cap * 0.8:                # near the line: count for real before refusing anything
            n, est_tok = self._tokenize(srv, [{"role": "system", "content": fixed_text}])
            fixed, est = n, (est or est_tok)
        if fixed > cap:
            for k, v in blocks.items():
                self.s[k] = v
            self.run.emit("run.error", {
                "error": f"The system prompt is about {fixed:,} tokens, but {srv['name']} has room for {usable:,} in one "
                         f"request ({n_ctx:,}-token window, minus room for the reply), and the system part may use at most "
                         f"{cap:,} (half) so your chat still fits. Shorten the system prompt, or use a server with a larger "
                         "context." + (" (counts estimated)" if est else ""),
                "code": "system_too_big", "tokens": fixed, "usable": usable, "n_ctx": n_ctx, "limit": cap})
            return True
        room = max(0, cap - fixed)
        cut = []
        for k, label in (("ref_block", "attached chats"), ("memory_block", "memory")):   # attached chats go first
            v = blocks[k]
            if not v:
                continue
            had = toks(v)
            if had <= room:
                self.s[k] = v
                room -= had
                continue
            keep_chars = max(0, room * 4 - 120)
            if keep_chars < 200:
                self.s[k] = ""
                cut.append(f"{label} left out ({had:,} tokens)")
                room = 0
                continue
            clip = v[:keep_chars]
            clip = clip[:clip.rfind("\n")] if "\n" in clip[200:] else clip
            self.s[k] = clip + f"\n[{label} shortened to fit the window]"
            cut.append(f"{label} cut from {had:,} to about {toks(self.s[k]):,} tokens")
            room = 0
        self.run.emit("status", {"kind": "system_trimmed", "tokens": fixed + (cap - fixed - room), "limit": cap,
                                 "text": f"Context was tight ({usable:,} tokens per request): " + "; ".join(cut) + "."})
        return False

    def _too_long(self, srv):
        """[C05] a message that cannot fit the window even alone is refused before the call, with numbers."""
        um = self.full[-1] if self.full else None
        if not um:
            return False
        need = compact.estimate([um])
        try:
            usable, n_ctx, est = self._usable(srv)
        except Exception:                    # noqa: BLE001 - the check is a courtesy; the call still has the overflow path
            return False
        if need <= usable:
            return False
        n, est_tok = self._tokenize(srv, [um])          # near or over the line: count for real
        if n <= usable:
            return False
        self.run.emit("run.error", {"error": f"Your message is about {n:,} tokens, but {srv['name']} has room for "
                                             f"{usable:,} in one request ({n_ctx:,}-token window, minus room for the "
                                             "reply). Shorten it or send it in parts, or use a server with a larger "
                                             "context." + (" (counts estimated)" if (est or est_tok) else ""),
                                    "code": "too_long", "tokens": n, "usable": usable, "n_ctx": n_ctx})
        return True

    def _overflow_words(self, srv, e):
        try:
            usable, n_ctx, _est = self._usable(srv)
            room = f" ({n_ctx:,}-token window per request, {usable:,} for the prompt)"
        except Exception:                    # noqa: BLE001
            room = ""
        return (f"This chat no longer fits {srv['name']}'s context window{room}, even after compacting. Start a new "
                f"chat (Memory keeps what matters), turn compaction on, or use a server with more context. "
                f"The server said: {str(e)[:200]}")

    def _refuse(self, c, step, code, text, say):
        """a call that is not run: the card shows why, the model gets one line with the code."""
        r, name = self.run, c["name"]
        a, _err = _args(c["raw"])
        shown = a if isinstance(a, dict) else {"raw": c["raw"]}
        r.emit("tool.call", {"call_id": c["id"], "step": step, "name": name, "args": shown,
                             "say": tools.say(name, shown), "label": tools.LABELS.get(name, (name, ""))[0]})
        out = "error: " + text
        r.emit("tool.result", {"call_id": c["id"], "ok": False, "output": out, "say": say,
                               "done": tools.said(name, shown), "code": code})
        return False, out

    def _call_key(self, name, a):
        try:
            return name + "\x00" + json.dumps(a, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return name + "\x00" + repr(a)

    def _guard(self, name, a):
        """[T16] the identical call whose identical result the model already has: refused with a reason. The second
        refusal in a turn takes the tools away for the next reply, so the model answers."""
        if not self.loop_guard:
            return None
        h = self._calls_seen.get(self._call_key(name, a))
        if not h or h["n"] < self.loop_guard - 1 or len(set(h["outs"])) != 1:
            return None
        self._guard_hits += 1
        if self._guard_hits >= 2:
            self._force_answer = True
        return (f"you already called {name} with exactly these arguments {h['n']} times in this turn and got the same "
                "result each time. Do not repeat it: use that result, change the arguments, or answer the user.")

    def _remember_call(self, name, a, out):
        h = self._calls_seen.setdefault(self._call_key(name, a), {"n": 0, "outs": []})
        h["n"] += 1
        h["outs"].append(hashlib.sha1(str(out).encode("utf-8", "replace")).hexdigest())

    def _budget(self, out):
        """[T15] one tool result may take at most a quarter of the per-request window (chars ~ 4 x tokens)."""
        n_ctx = (self._nctx or (None, None))[0]
        if not n_ctx and self._srv:
            try:
                n_ctx = self._ctx_limits(self._srv)[0]
            except Exception:                # noqa: BLE001
                n_ctx = None
        limit = int(max(2000, min(tools.MAX_OUT_HARD, (n_ctx or 8192) * 4 * tools.OUT_SHARE)))
        out = str(out)
        if len(out) <= limit:
            return out
        note = (f"\n[tool output trimmed to about {limit:,} of {len(out):,} characters to fit the context window; "
                "ask for a smaller part (offset/limit, a narrower search) if you need the middle]")
        return compact.head_tail(out, limit) + note

    def _for_model(self, name, out):
        out = self._budget(out)
        if self.fence_on and name in tools.UNTRUSTED:
            out = guards.fence_tool_output(out)
        return out

    def _execute(self, name, a):
        """tools.execute on a worker thread -> (ok, out, code), or None when Stop was pressed. The per-tool deadline
        and Stop both interrupt the tool: a process group is killed, a socket shut down [T14 X03]."""
        limit = tools.deadline_for(name, a, self.tool_timeout, self.ctx)
        box, done = {}, threading.Event()
        self.ctx.interrupts = []
        self.ctx.interrupted = False

        def work():
            try:
                box["res"] = tools.execute(name, a, self.ctx)
            except BaseException as e:      # noqa: BLE001 - execute never raises; a dead thread must still answer
                box["res"] = (False, f"error: {e.__class__.__name__}: {e}")
            finally:
                done.set()
        th = threading.Thread(target=work, daemon=True, name=f"pxa-tool-{name}")
        th.start()
        t0 = time.time()
        while not done.wait(0.1):
            if self.run.cancel_ev.is_set():
                tools.interrupt(self.ctx)
                done.wait(3.0)
                return None
            if limit and time.time() - t0 > limit:
                tools.interrupt(self.ctx)
                done.wait(3.0)
                return (False, f"error: {name} did not finish within {int(limit)} s and was stopped. Try a smaller "
                               "task, or answer with what you have.", "timeout")
        ok, out = box.get("res") or (False, "error: the tool ended without a result")
        return ok, str(out), (None if ok else "tool_error")

    def _one_call(self, c, step, cut=False):
        r, name = self.run, c["name"]
        a, err = _args(c["raw"])
        shown = a if isinstance(a, dict) else {"raw": c["raw"]}
        r.emit("tool.call", {"call_id": c["id"], "step": step, "name": name, "args": shown,
                             "say": tools.say(name, shown), "label": tools.LABELS.get(name, (name, ""))[0]})
        done = tools.said(name, shown)
        self.last_tool = {"name": name, "args": shown, "ok": False, "out": "", "decision": None}

        def fail(code, out, say):
            out = out if out.startswith(("error:", "refused:")) else "error: " + out
            self.last_tool["out"] = out
            r.emit("tool.result", {"call_id": c["id"], "ok": False, "output": out, "say": say, "done": done,
                                   "code": code})
            return False, out
        avail = ", ".join(self.tool_names) or "none"
        if err:
            if cut:
                return fail("cut_off", tools.cut_message(name, self.s.get("max_tokens")), "Cut off")
            return fail("invalid_arguments", tools.bad_args_message(name, err), "That didn't work")
        if name not in tools.SPECS:
            return fail("unknown_tool", f"there is no tool called {name!r}. The tools you can use are: {avail}.",
                        "No such tool")
        if name not in self.tool_names:
            return fail("tool_off", f"the tool {name!r} is not turned on for this chat. The tools you can use are: "
                                    f"{avail}.", "That tool is turned off")
        missing = tools.missing_args(name, a)
        if missing:
            return fail("missing_argument", tools.missing_message(name, missing), "That didn't work")
        guard = self._guard(name, a)
        if guard:
            return fail("loop_guard", guard, "Repeated call skipped")
        self.ctx.approved_private = False
        need = tools.approval_needed(name, a, self.ctx)
        if need and not (need["always_ok"] and self.approvals.allowed(r.chat_id, need["scope"])):
            aid = self.approvals.issue(r.id, r.chat_id, name, need["scope"])
            r.emit("approval.request", {"approval_id": aid, "call_id": c["id"], "name": name, "args": shown,
                                        "say": tools.say(name, shown), "question": need["question"],
                                        "detail": need["detail"], "always_ok": need["always_ok"],
                                        "timeout_s": self.approvals.timeout_s})
            d = self.approvals.wait(aid, r.cancel_ev)
            r.emit("approval.answered", {"approval_id": aid, "call_id": c["id"], "decision": d})
            if d == "cancelled":
                return None
            if d not in ("once", "always"):
                out = ("error: the user said no to this. Do not try it again; carry on without it or ask the user."
                       if d == "deny" else "error: nobody answered the approval in time, so it was not done.")
                self.last_tool.update(out=out, decision=d)
                r.emit("tool.result", {"call_id": c["id"], "ok": False, "output": out, "done": done,
                                       "say": "You said no" if d == "deny" else "No answer in time, skipped",
                                       "code": "denied" if d == "deny" else "approval_timeout"})
                return False, out
            if name == "web_fetch":
                self.ctx.approved_private = True
        elif need and name == "web_fetch":
            self.ctx.approved_private = True                       # "always for this chat"
        t0 = time.time()
        self.ctx.memory_event = None
        self.ctx.call_id = c["id"]
        got = self._execute(name, a)
        self.ctx.approved_private = False
        if got is None:                                           # Stop while the tool ran
            r.emit("tool.result", {"call_id": c["id"], "ok": False, "output": "error: stopped by the user",
                                   "say": "Stopped", "done": done, "code": "cancelled",
                                   "secs": round(time.time() - t0, 2)})
            return None
        ok, out, code = got
        if name in tools.MEMORY_WRITES:
            self.saved_memory = True
        self._remember_call(name, a, out)
        self.last_tool.update(ok=ok, out=out)
        ev = {"call_id": c["id"], "ok": ok, "output": out[:6000], "chars": len(out),
              "secs": round(time.time() - t0, 2), "say": "Done" if ok else "That didn't work", "done": done}
        if code:
            ev["code"] = code
        if self.ctx.memory_event:
            ev["memory"] = self.ctx.memory_event
        if name == "spawn_agent":
            from . import subagent
            subagent._attach(self.ctx, c["id"], ev)
        r.emit("tool.result", ev)
        return ok, self._for_model(name, out)


def transcript(run):
    """the UI view of one turn (saved with the session): the user text, then the parts in order (thinking, the
    model's own text per step, tool steps) and the final reply. `assistant` is the model's final reply only:
    text a tool-calling step wrote before its call stays in `parts`, not in the answer."""
    msg = next((m for m in run.new_messages if m.get("role") == "user"), None)
    user = sessions.message_text(msg) if msg else ""
    content = msg.get("content") if msg else None
    if isinstance(content, list):
        nimg = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
        if nimg:
            user = (user + " " if user else "") + ("[image]" if nimg == 1 else "[%d images]" % nimg)
    steps, texts, think, cur, end, notes = [], {}, "", {}, None, None
    parts, think_t, step_t = [], [None, None], None

    def part_text(step):
        for p in reversed(parts):
            if p["k"] == "text" and p["step"] == step:
                return p
        p = {"k": "text", "step": step, "text": ""}
        parts.append(p)
        return p
    for ev in run.events:
        t, d = ev["type"], ev["data"]
        if t == "step":
            step_t = ev["t"]
        if think_t[0] and not think_t[1] and t in ("text.delta", "tool.call") + TERMINAL:
            think_t[1] = ev["t"]                 # thought from the step start until the first visible output
        if t == "text.delta":
            texts[d.get("step", 0)] = texts.get(d.get("step", 0), "") + d.get("text", "")
            part_text(d.get("step", 0))["text"] = texts[d.get("step", 0)]
        elif t == "think.delta":
            think += d.get("text", "")
            think_t[0] = think_t[0] or step_t or ev["t"]
        elif t == "text.retract":
            texts[d.get("step", 0)] = d.get("text", "")
            part_text(d.get("step", 0))["text"] = texts[d.get("step", 0)]
        elif t == "tool.call":
            cur[d["call_id"]] = dict(d, ok=None, output=None)
            steps.append(cur[d["call_id"]])
            parts.append({"k": "tool", "call_id": d["call_id"]})
        elif t == "approval.answered" and d.get("call_id") in cur:
            cur[d["call_id"]]["decision"] = d.get("decision")
        elif t == "tool.result" and d.get("call_id") in cur:
            patch = dict(ok=d.get("ok"), output=d.get("output"), result_say=d.get("say"), done=d.get("done"))
            for k in ("status", "steps", "tokens", "trace", "task", "secs"):
                if d.get(k) is not None:
                    patch[k] = d[k]
            if d.get("memory"):
                patch["memory"] = d["memory"]
            cur[d["call_id"]].update(patch)
        elif t == "notes" and isinstance(d.get("notes"), list):
            notes = d.get("notes")
        elif t in TERMINAL:
            end = {"type": t, "data": d}
    for p in parts:
        if p["k"] == "text":
            p["text"] = strip_think(p["text"]).strip()
    parts = [p for p in parts if p["k"] != "text" or p["text"]]
    if end and end["type"] == "run.done":
        final = (end["data"].get("text") or "").strip()
    else:
        final = "\n\n".join(strip_think(v).strip() for _k, v in sorted(texts.items()) if strip_think(v).strip())
    think_s = round(max(0.0, (think_t[1] or think_t[0]) - think_t[0]), 1) if think_t[0] else None
    out = {"user": user, "assistant": final, "think": think, "think_s": think_s, "steps": steps, "parts": parts,
           "end": end, "ts": run.created}
    if notes:
        out["notes"] = notes
    return out
