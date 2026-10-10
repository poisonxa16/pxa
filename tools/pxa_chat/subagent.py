"""A sub-agent: one task, a fresh context, a subset of the parent's tools, and no further spawning.

The child borrows the parent's sandbox, approvals, guards and host-access switch. It cannot widen any of them.
A failure, a stop or a full step budget comes back as text. It does not raise, and it does not end the parent.
Several children run at once only while the server has a free slot; otherwise they wait.
"""
import threading
import time

CHILD_SYS = ("You are a sub-agent inside PXA Control. Another assistant gave you one task. "
             "Answer it directly. You cannot spawn further agents. "
             "This turn has a shared scratchpad: note_write and note_read, when you have those tools. "
             "Your siblings and the parent see the same notes. Notes do not carry into the next turn. "
             "Use a tool only when the task needs it, then give the final answer in plain text.")

_GATE = threading.Semaphore(3)     # max children running on this process at once
_COUNT = threading.Lock()
MAX_CONCURRENT = 3


def clamp(v, lo, hi, default):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return max(lo, min(int(v), hi))


def free_slots(base, get):
    """how many llama slots are idle, or None when /slots does not say."""
    try:
        slots = get(base, "/slots", 3.0)
    except Exception:                          # noqa: BLE001 — unknown means we do not pretend to know
        return None
    if not isinstance(slots, list) or not slots:
        return None
    return sum(1 for s in slots if isinstance(s, dict) and not s.get("is_processing"))


def _note(ctx, call_id, status=None, step=None, tokens=None, trace=None, task=None):
    """one structured progress event. Heartbeat words are fields, not lines in the trace."""
    emit = getattr(ctx, "emit", None)
    if not emit or not call_id:
        return
    data = {"call_id": call_id}
    if status is not None:
        data["status"] = status
    if step is not None:
        data["step"] = step
    if tokens is not None:
        data["tokens"] = int(tokens)
    if trace is not None:
        data["trace"] = list(trace)
    if task:
        data["task"] = task
    emit("subagent", data)


def _meta(status, trace, tokens, steps, task):
    return {"status": status, "steps": int(steps or 0), "tokens": int(tokens or 0),
            "trace": list(trace or []), "task": task or ""}


def _stash(ctx, call_id, meta):
    """per call id. A batch shares one parent context, so a single slot would race."""
    bag = getattr(ctx, "sub_reports", None)
    if not isinstance(bag, dict):
        bag = {}
        ctx.sub_reports = bag
    bag[call_id or ""] = meta


def _fail(ctx, call_id, text, task, status="failed"):
    meta = _meta(status, [], 0, 0, task)
    _stash(ctx, call_id, meta)
    _note(ctx, call_id, status=status, step=0, tokens=0, trace=[], task=task)
    return False, text


def _allowed(ctx, requested):
    """the child's tools: a subset of the parent's, never spawn_agent, never a tool the parent lacks."""
    parent = list(getattr(ctx, "parent_tools", None) or [])
    if not isinstance(requested, list):
        return []
    out = []
    for n in requested:
        n = str(n or "")
        if n and n != "spawn_agent" and n in parent and n not in out:
            out.append(n)
    return out


def _pick(ctx, want):
    """the server this child runs on. Empty means the parent's server. A name or key goes through the picker."""
    key = str(want or "").strip()
    resolve = getattr(ctx, "resolve", None)
    if not key:
        if resolve is None:
            return None
        try:
            srv = resolve()
        except Exception:                      # noqa: BLE001
            return None
        if not srv or srv.get("status") not in ("ready", "busy"):
            return None
        return srv
    servers = []
    try:
        from . import picker
        servers = picker.list_servers(getattr(ctx, "app", None)) or []
    except Exception:                          # noqa: BLE001
        servers = []
    low = key.lower()
    srv = next((s for s in servers if s.get("status") in ("ready", "busy") and (
        low == str(s.get("key") or "").lower() or low == str(s.get("name") or "").lower()
        or low in str(s.get("name") or "").lower() or key == str(s.get("port") or ""))), None)
    if srv:
        return srv
    if resolve is None:
        return None
    try:
        srv = resolve(key)
    except TypeError:
        return None
    except Exception:                          # noqa: BLE001
        return None
    if not srv or srv.get("status") not in ("ready", "busy"):
        return None
    return srv


def _wait_slot(ctx, base, get, deadline, poll, slots_fn):
    """block while every slot is busy. Unknown /slots (None) does not wait."""
    cancel = getattr(ctx, "cancel_ev", None)
    while True:
        if cancel is not None and cancel.is_set():
            return "stopped"
        if time.time() >= deadline:
            return "timeout"
        try:
            free = slots_fn(base) if slots_fn else free_slots(base, get)
        except Exception:                      # noqa: BLE001
            return "go"
        if free is None or free > 0:
            return "go"
        if cancel is not None:
            cancel.wait(poll)
        else:
            time.sleep(poll)


def run(a, ctx, complete=None, gate=None, poll=0.4, slots_fn=None, get=None, call_id=None):
    """-> (ok, text). complete(messages, native, tool_names) is the test stand-in for one model call.
    call_id is the parent's tool-call id. Pass it in when several children share one context."""
    from . import oai, tools
    a = a if isinstance(a, dict) else {}
    task = str(a.get("task") or "").strip()
    if not call_id:
        call_id = getattr(ctx, "call_id", "") or ""
    if not task:
        return False, "error: task is required"
    if int(getattr(ctx, "depth", 0) or 0) >= 1:
        return _fail(ctx, call_id, "error: a sub-agent cannot spawn another sub-agent", task)
    limit = clamp(getattr(ctx, "sub_max", None), 1, 4, 2)
    with _COUNT:
        ctx.spawned = int(getattr(ctx, "spawned", 0) or 0) + 1
        n_this_turn = ctx.spawned
    if n_this_turn > limit:
        return _fail(ctx, call_id, "error: this turn already has the maximum number of sub-agents (%d)" % limit, task)
    names = _allowed(ctx, a.get("tools"))
    steps = clamp(a.get("max_steps"), 1, 8, clamp(getattr(ctx, "sub_steps", None), 1, 8, 4))
    tokens = clamp(a.get("max_tokens"), 64, 2048, 512)
    seconds = clamp(getattr(ctx, "sub_seconds", None), 15, 120, 60)
    deadline = time.time() + seconds
    srv = _pick(ctx, a.get("server"))
    if srv is None:
        return _fail(ctx, call_id, "error: no running local server for that sub-agent", task)
    _note(ctx, call_id, status="running", step=0, tokens=0, trace=[], task=task)
    gate = gate or _GATE
    get = get or oai.get_json
    waited = _wait_slot(ctx, srv.get("base_url"), get, deadline, poll, slots_fn)
    if waited == "stopped":
        return _fail(ctx, call_id, "error: the sub-agent was stopped", task, "stopped")
    if waited == "timeout":
        return _fail(ctx, call_id, "error: the sub-agent waited for a free slot until its time ran out", task)
    acquired = False
    try:
        while True:
            cancel = getattr(ctx, "cancel_ev", None)
            if cancel is not None and cancel.is_set():
                return _fail(ctx, call_id, "error: the sub-agent was stopped", task, "stopped")
            left = deadline - time.time()
            if left <= 0:
                return _fail(ctx, call_id, "error: too many sub-agents are already running", task)
            if gate.acquire(timeout=min(poll, left)):
                acquired = True
                break
        return _drive(a, ctx, srv, names, steps, tokens, deadline, call_id, complete, task)
    except Exception as e:                     # noqa: BLE001 — a child must not kill the parent
        return _fail(ctx, call_id, "error: the sub-agent failed: %s" % e.__class__.__name__, task)
    finally:
        if acquired:
            gate.release()


def _drive(a, ctx, srv, names, steps, tokens, deadline, call_id, complete, task):
    from . import oai, protocol, tools
    native = bool(names) and bool((srv.get("caps") or {}).get("tools"))
    sysp = CHILD_SYS
    if names and not native:
        sysp = sysp + "\n\n" + tools.text_protocol_prompt(names)
    messages = [{"role": "system", "content": sysp}, {"role": "user", "content": task[:8000]}]
    trace, final, used = [], "", 0
    label = "Sub-agent"
    child = _child_ctx(ctx)
    child.call_id = call_id

    def finish(ok, answer, status, step_n):
        """the model sees the answer only. The trace stays on the card and on the saved step."""
        text = (answer or "").strip()
        if not text:
            text = "The sub-agent returned no text." if ok else "The sub-agent did not finish."
        meta = _meta(status, trace, used, step_n, task)
        _stash(ctx, call_id, meta)
        _note(ctx, call_id, status=status, step=step_n, tokens=used, trace=trace, task=task)
        return ok, text

    for step in range(1, steps + 1):
        if getattr(ctx, "cancel_ev", None) is not None and ctx.cancel_ev.is_set():
            trace.append("step %d: stopped" % step)
            return finish(False, "Stopped.", "stopped", step)
        if time.time() >= deadline:
            trace.append("step %d: out of time" % step)
            return finish(False, "The sub-agent ran out of time.", "failed", step)
        _note(ctx, call_id, status="running", step=step, tokens=used, trace=trace, task=task)
        try:
            if complete:
                res = complete(messages, native, names)
            else:
                res = _complete(ctx, srv, messages, native, names, tokens, deadline)
        except oai.OAIError as e:
            if getattr(e, "code", "") == "cancelled" or (getattr(ctx, "cancel_ev", None) is not None and ctx.cancel_ev.is_set()):
                trace.append("step %d: stopped" % step)
                return finish(False, "Stopped.", "stopped", step)
            trace.append("step %d: the model call failed" % step)
            return finish(False, "error: %s" % e, "failed", step)
        used += int((res.get("usage") or {}).get("completion_tokens") or 0)
        content = res.get("content") or ""
        calls = []
        for tc in res.get("tool_calls") or []:
            calls.append({"name": tc.get("name"), "raw": tc.get("arguments")})
        if not calls and names:
            _prose, call = protocol.split_call(content)
            if call:
                name, args = protocol.call_to_openai(call)
                calls = [{"name": name, "raw": args}]
                content = _prose
        if not calls:
            final = (content or "").strip()
            trace.append("step %d: replied (%d tokens)" % (step, used))
            return finish(True, final or "The sub-agent returned no text.", "finished", step)
        messages.append({"role": "assistant", "content": content or "", "tool_calls": [
            {"id": "c%d" % i, "type": "function", "function": {"name": c["name"], "arguments": c["raw"] if isinstance(c["raw"], str) else __import__("json").dumps(c["raw"])}}
            for i, c in enumerate(calls)]})
        for i, c in enumerate(calls):
            name = str(c.get("name") or "")
            args, err = _loads(c.get("raw"))
            if name not in names:
                out, ok = "error: that tool is not available to this sub-agent", False
            elif err:
                out, ok = "error: " + err, False
            else:
                out, ok = _child_tool(child, name, args, label + ": " + tools._short(task, 40))
            trace.append("step %d: %s -> %s" % (step, name or "?", tools._short(out, 80)))
            _note(ctx, call_id, status="running", step=step, tokens=used, trace=trace, task=task)
            messages.append({"role": "tool", "tool_call_id": "c%d" % i, "content": out})
    trace.append("used all %d steps" % steps)
    return finish(False, final or "The sub-agent used every step without a final answer.", "failed", steps)


def _loads(raw):
    from . import protocol
    if isinstance(raw, dict):
        return raw, None
    if not raw or not str(raw).strip():
        return {}, None
    o = protocol.parse_lenient(str(raw))
    if not isinstance(o, dict):
        return None, "the tool arguments were not valid JSON"
    return o, None


def _child_ctx(parent):
    """same sandbox, same host switch, same approvals. Depth 1 so this child cannot spawn."""
    from . import tools
    c = tools.Ctx(parent.sandbox, chat_id=parent.chat_id, run_id=parent.run_id, advanced=parent.advanced,
                  app=parent.app, current_key=parent.current_key, search_url=parent.search_url,
                  cancel_ev=parent.cancel_ev, memory=parent.memory, sessions=parent.sessions,
                  host_ok=parent.host_ok, host_why=parent.host_why, host_allow=parent.host_allow,
                  host_audit=parent.host_audit, search_cfg=parent.search_cfg)
    c.depth = 1
    c.parent_tools = list(getattr(parent, "parent_tools", None) or [])
    c.approvals = getattr(parent, "approvals", None)
    c.emit = getattr(parent, "emit", None)
    c.run = getattr(parent, "run", None)
    c.resolve = getattr(parent, "resolve", None)
    c.sub_max = getattr(parent, "sub_max", 2)
    c.sub_steps = getattr(parent, "sub_steps", 4)
    c.sub_seconds = getattr(parent, "sub_seconds", 60)
    c.pad = tools.pad_of(parent)       # the parent's pad, not a fresh one
    return c


def _child_tool(ctx, name, args, label):
    """run one tool. Approvals still apply, and the card says it came from the sub-agent."""
    from . import tools
    need = tools.approval_needed(name, args, ctx)
    if need and ctx.approvals is not None:
        question = label + " — " + need["question"]
        run = getattr(ctx, "run", None)
        if not (need["always_ok"] and ctx.approvals.allowed(ctx.chat_id, need["scope"])):
            aid = ctx.approvals.issue(ctx.run_id, ctx.chat_id, name, need["scope"])
            if getattr(ctx, "emit", None):
                ctx.emit("approval.request", {"approval_id": aid, "call_id": getattr(ctx, "call_id", ""),
                                               "name": name, "args": args, "say": tools.say(name, args),
                                               "question": question, "detail": need["detail"],
                                               "always_ok": need["always_ok"], "timeout_s": ctx.approvals.timeout_s,
                                               "subagent": label})
            d = ctx.approvals.wait(aid, ctx.cancel_ev)
            if getattr(ctx, "emit", None):
                ctx.emit("approval.answered", {"approval_id": aid, "call_id": getattr(ctx, "call_id", ""), "decision": d})
            if d == "cancelled":
                return "error: stopped", False
            if d not in ("once", "always"):
                return ("error: the user said no to this." if d == "deny"
                        else "error: nobody answered the approval in time."), False
            if name == "web_fetch":
                ctx.approved_private = True
        elif name == "web_fetch":
            ctx.approved_private = True
    ok, text = tools.execute(name, args, ctx)
    return text, ok


def _complete(ctx, srv, messages, native, names, tokens, deadline):
    from . import oai, tools
    model = srv.get("model")
    if not model:
        raise oai.OAIError("the server did not report a model", "no_model")
    body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": tokens,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if native:
        body["tools"] = [tools.SPECS[n] for n in names]
        body["tool_choice"] = "auto"
    left = max(5.0, deadline - time.time())
    st = oai.Stream(srv["base_url"], body, step_timeout=min(left, 90), idle_timeout=min(left, 60))
    run = getattr(ctx, "run", None)
    if run is not None:
        extra = getattr(run, "_extra_streams", None)
        if extra is None:
            run._extra_streams = extra = []
        extra.append(st)
        if getattr(ctx, "cancel_ev", None) is not None and ctx.cancel_ev.is_set():
            st.cancel()
    try:
        return st.run()
    finally:
        if run is not None and st in getattr(run, "_extra_streams", ()):
            run._extra_streams.remove(st)


def _attach(ctx, call_id, ev):
    """copy this child's report onto the tool result. Pop so a later call cannot reuse it."""
    reports = getattr(ctx, "sub_reports", None)
    extra = reports.pop(call_id, None) if isinstance(reports, dict) else None
    if not isinstance(extra, dict):
        return ev
    for k in ("status", "steps", "tokens", "trace", "task"):
        if extra.get(k) is not None:
            ev[k] = extra[k]
    return ev


def run_batch(calls, agent, step, slots_fn=None, complete=None, gate=None, poll=0.4):
    """run several spawn_agent calls from one step at once when a slot is free. Results stay in call order.
    Each item is (ok, text) or None when the parent was stopped before that child started."""
    from . import tools
    r = agent.run
    out = [None] * len(calls)

    def one(i, c):
        if r.cancel_ev.is_set():
            out[i] = None
            return
        args = c.get("args")
        if not isinstance(args, dict):
            out[i] = (False, "error: arguments must be an object")
            return
        ok, text = run(args, agent.ctx, complete=complete, gate=gate, poll=poll, slots_fn=slots_fn, call_id=c["id"])
        out[i] = (ok, text)

    # The loop parses args and emits the opening row before the threads, so the UI can show each one.
    prepared = []
    for c in calls:
        args, err = _loads(c.get("raw"))
        shown = args if isinstance(args, dict) else {"raw": c.get("raw")}
        r.emit("tool.call", {"call_id": c["id"], "step": step, "name": "spawn_agent", "args": shown,
                             "say": tools.say("spawn_agent", shown), "label": "Sub-agent"})
        if err:
            prepared.append((c, None, err))
        else:
            prepared.append((c, shown, None))
    live = []
    for i, (c, shown, err) in enumerate(prepared):
        if err:
            out[i] = (False, "error: " + err)
            continue
        c = dict(c, args=shown)
        t = threading.Thread(target=one, args=(i, c), name="pxa-sub-" + c["id"])
        live.append((i, c, t))
        t.start()
    for _i, _c, t in live:
        t.join()
    for i, (c, shown, err) in enumerate(prepared):
        if out[i] is None:
            continue
        ok, text = out[i]
        ev = {"call_id": c["id"], "ok": ok, "output": text[:6000], "chars": len(text),
              "say": "Done" if ok else "That didn't work", "done": tools.said("spawn_agent", shown or {})}
        r.emit("tool.result", _attach(agent.ctx, c["id"], ev))
    return out
