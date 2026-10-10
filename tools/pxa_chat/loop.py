"""The chat agent loop: a stdlib OpenAI-compatible tool-calling loop. Native tool calls when the server's
template supports them, the hive text protocol (protocol.py) otherwise, and the text protocol as a fallback
when a native-mode reply carries a call as text. Max steps, cancel, per-step timeouts, approvals, and a
re-attach wait when the server restarts mid-chat. Every step is an event the UI receives over SSE.

Loop shape follows the Mythos agent (pxa-mythos-agent.mjs runAgent): call model -> run tools -> feed results
back -> repeat until a reply without a call or the step budget."""
import json
import re
import secrets
import threading
import time

from . import compact, oai, protocol, sessions, tools

EVENT_TYPES = ("run.started", "status", "step", "text.delta", "think.delta", "text.retract", "tool.call",
               "approval.request", "approval.answered", "tool.result", "run.done", "run.error", "run.cancelled")
TERMINAL = ("run.done", "run.error", "run.cancelled")
THINK_RE = re.compile(r"(?s)^\s*<think>.*?(</think>|$)\s*")


def strip_think(text):
    """a server without a reasoning parser streams <think>...</think> inside the content: not part of the reply."""
    return THINK_RE.sub("", text or "", count=1)


class Run(object):
    def __init__(self, chat_id):
        self.id = "run_" + secrets.token_urlsafe(10)
        self.chat_id = chat_id
        self.events = []
        self.cond = threading.Condition()
        self.cancel_ev = threading.Event()
        self.done = False
        self.created = time.time()
        self.ended = None
        self._stream = None
        self._extra_streams = []         # child model calls; cancel() stops these too
        self.saved = threading.Event()   # set once the session file has this turn
        self.new_messages = []        # OpenAI-format messages this run added (user, assistant, tool)
        self.transcript = None        # the UI view of this turn, built from the events

    def emit(self, typ, data=None):
        with self.cond:
            ev = {"seq": len(self.events) + 1, "type": typ, "t": round(time.time(), 3), "data": data or {}}
            self.events.append(ev)
            if typ in TERMINAL:
                self.done = True
                self.ended = time.time()
            self.cond.notify_all()
        return ev

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


def _args(raw):
    if isinstance(raw, dict):
        return raw, None
    if not raw or not str(raw).strip():
        return {}, None
    o = protocol.parse_lenient(str(raw))
    if not isinstance(o, dict):
        return None, "the tool arguments were not valid JSON"
    return o, None


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
        return sysp

    def go(self):
        r = self.run
        try:
            self._go()
        except Exception as e:         # noqa: BLE001
            if not r.done:
                r.emit("run.error", {"error": f"something went wrong inside PXA Control: {e.__class__.__name__}: {e}",
                                     "code": "internal"})

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
        res = oai.post_json(srv["base_url"], "/v1/chat/completions", body, 60.0)
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
        res = oai.post_json(srv["base_url"], "/v1/chat/completions", body, 45.0)
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
            sysp = self._system(native)
            force = bool(self.s.get("force_compact")) and step == 1
            send = self._send_view(srv, model, sysp, native, force=force)
            res = None
            while res is None:
                body = {"model": model, "messages": ([{"role": "system", "content": sysp}] if sysp else []) + for_mode(send, native),
                        "stream_options": {"include_usage": True}}
                for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "repeat_penalty", "seed"):
                    if self.s.get(k) is not None:
                        body[k] = self.s[k]
                if self.s.get("thinking") in ("on", "off"):
                    body["chat_template_kwargs"] = {"enable_thinking": self.s["thinking"] == "on"}
                if native:
                    body["tools"] = [tools.SPECS[n] for n in self.tool_names]
                    body["tool_choice"] = "auto"
                st = oai.Stream(srv["base_url"], body, step_timeout=self.step_timeout,
                                idle_timeout=min(self.step_timeout, 120))
                r._stream = st
                if r.cancel_ev.is_set():
                    st.cancel()
                try:
                    res = st.run(on_text=lambda d: r.emit("text.delta", {"step": step, "text": d}),
                                 on_think=lambda d: r.emit("think.delta", {"step": step, "text": d}))
                except oai.OAIError as e:
                    if e.code == "cancelled" or r.cancel_ev.is_set():
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    if e.code == "unreachable" and step < self.max_steps:
                        r.emit("status", {"text": "Lost the server mid-reply; trying again when it is back.", "kind": "retry"})
                        time.sleep(1.0)
                        res = "retry"
                        break
                    if not self._overflow_used and compact.is_overflow(e):
                        self._overflow_used = True
                        r.emit("status", {"text": "That reply did not fit the context window. Compacting and trying once.",
                                          "kind": "compact"})
                        send = self._send_view(srv, model, sysp, native, force=True, harder=True)
                        continue
                    r.emit("run.error", {"error": str(e), "code": e.code})
                    return
                finally:
                    r._stream = None
            if res == "retry":
                continue
            usage, timings = res.get("usage") or usage, res.get("timings") or timings
            content = res.get("content") or ""
            calls, textual = [], False
            for tc in res.get("tool_calls") or []:
                calls.append({"id": tc["id"], "name": tc["name"], "raw": tc["arguments"]})
            if not calls and self.tool_names:
                prose, call = protocol.split_call(content)
                if call:
                    name, a = protocol.call_to_openai(call)
                    calls = [{"id": "txt_" + secrets.token_hex(4), "name": name, "raw": a}]
                    textual = True
                    r.emit("text.retract", {"step": step, "text": prose})
            if not calls:
                # the reply is the model's own words, never wrapped in a fixed prefix/suffix. Only when the model
                # sends an EMPTY final turn after using tools does PXA write one short line from the last result.
                content = strip_think(content).strip()
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
                r.emit("run.done", {"text": content, "steps": step, "usage": usage, "timings": timings,
                                    "fallback": fallback, "empty": not content})
                return
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
            spawn_calls = [c for c in calls if c["name"] == "spawn_agent" and "spawn_agent" in self.tool_names]
            if len(spawn_calls) >= 2:
                from . import subagent
                for c, item in zip(spawn_calls, subagent.run_batch(spawn_calls, self, step)):
                    batched[c["id"]] = item
            for c in calls:
                if c["id"] in batched:
                    item = batched[c["id"]]
                    if item is None:
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    ok, out = item
                    shown, _err = _args(c["raw"])
                    self.last_tool = {"name": "spawn_agent", "args": shown if isinstance(shown, dict) else {},
                                      "ok": ok, "out": out, "decision": None}
                else:
                    if r.cancel_ev.is_set():
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    out_msg = self._one_call(c, step)
                    if out_msg is None:          # cancelled while waiting for an approval
                        r.emit("run.cancelled", {"text": "Stopped."})
                        return
                    ok, out = out_msg
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
            res = oai.post_json(srv["base_url"], "/v1/chat/completions", body, 60.0)
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

    def _one_call(self, c, step):
        r, name = self.run, c["name"]
        a, err = _args(c["raw"])
        shown = a if isinstance(a, dict) else {"raw": c["raw"]}
        r.emit("tool.call", {"call_id": c["id"], "step": step, "name": name, "args": shown,
                             "say": tools.say(name, shown), "label": tools.LABELS.get(name, (name, ""))[0]})
        done = tools.said(name, shown)
        self.last_tool = {"name": name, "args": shown, "ok": False, "out": "", "decision": None}
        if err:
            out = f"error: {err}"
            self.last_tool["out"] = out
            r.emit("tool.result", {"call_id": c["id"], "ok": False, "output": out, "say": "That didn't work", "done": done})
            return False, out
        if name not in self.tool_names:
            out = f"error: the tool {name!r} is not turned on for this chat"
            self.last_tool["out"] = out
            r.emit("tool.result", {"call_id": c["id"], "ok": False, "output": out, "say": "That tool is turned off", "done": done})
            return False, out
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
                                       "say": "You said no" if d == "deny" else "No answer in time, skipped"})
                return False, out
            if name == "web_fetch":
                self.ctx.approved_private = True
        elif need and name == "web_fetch":
            self.ctx.approved_private = True                       # "always for this chat"
        t0 = time.time()
        self.ctx.memory_event = None
        self.ctx.call_id = c["id"]
        ok, out = tools.execute(name, a, self.ctx)
        self.ctx.approved_private = False
        if name in tools.MEMORY_WRITES:
            self.saved_memory = True
        self.last_tool.update(ok=ok, out=out)
        ev = {"call_id": c["id"], "ok": ok, "output": out[:6000], "chars": len(out),
              "secs": round(time.time() - t0, 2), "say": "Done" if ok else "That didn't work", "done": done}
        if self.ctx.memory_event:
            ev["memory"] = self.ctx.memory_event
        if name == "spawn_agent":
            from . import subagent
            subagent._attach(self.ctx, c["id"], ev)
        r.emit("tool.result", ev)
        return ok, out


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
