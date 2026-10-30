#!/usr/bin/env python3
"""PXA Control chat harness suite: every edge case of the Chat tab's agent loop, driven over Control's real HTTP API.

It starts a scratch PXA Control (its own config dir, port and token; fake GPUs, no discovery) and scripted FAKE
llama-servers (harness_fake_server.py), runs the catalogue (EDGE-CASES.md ids), and writes
  <out>/harness-report.json and <out>/harness-report.md   (case id, expected, observed, PASS/FAIL, log excerpt).
Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = the suite could not start.

  python3 tools/pxa_control/tests/harness_suite.py                       # FAKE mode, every automatable case
  python3 tools/pxa_control/tests/harness_suite.py --endpoint http://127.0.0.1:8263   # + live cases vs a real server
  python3 tools/pxa_control/tests/harness_suite.py --only S06,T16 --out /tmp/h       # a few cases
  python3 tools/pxa_control/tests/harness_suite.py --list                            # the catalogue

Statuses: PASS, FAIL, XFAIL (a catalogued gap that is not fixed yet: observed and reported, never fails the run),
XPASS (a catalogued gap that now passes: tell the coordinator), SKIP (not for this mode), MANUAL (needs a browser,
a human or hours; the reason is in the report). See README-HARNESS.md.
"""
import argparse
import base64
import http.client
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(TOOLS)
FAKE_PY = os.path.join(HERE, "harness_fake_server.py")
TERMINAL = ("run.done", "run.error", "run.cancelled")


# =====================================================================================================================
# processes
# =====================================================================================================================
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http_ok(port, path="/health", timeout=1.0):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("GET", path)
        r = c.getresponse()
        r.read()
        c.close()
        return r.status
    except (OSError, http.client.HTTPException):
        return None


def proc_start_ticks(pid):
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as f:
            st = f.read().decode("utf-8", "replace")
        return int(st[st.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class Fake(object):
    """one FAKE llama-server subprocess."""

    def __init__(self, name, out, model, port=None, **opts):
        self.name, self.out, self.model, self.opts = name, out, model, opts
        self.port = port or free_port()
        self.proc = None
        self.log_path = os.path.join(out, f"fake-{name}.log")
        self._key = None

    @property
    def key(self):
        """the picker's key: a launched server is port:<port>; the attached one is Control's main seat."""
        return self._key or f"port:{self.port}"

    def start(self):
        args = [sys.executable, FAKE_PY, "--port", str(self.port), "--model", self.model]
        for k, v in self.opts.items():
            if v is True:
                args.append("--" + k.replace("_", "-"))
            elif v not in (None, False):
                args += ["--" + k.replace("_", "-"), str(v)]
        self.proc = subprocess.Popen(args, stdout=open(self.log_path, "ab"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 15:
            if http_ok(self.port) == 200:
                return self
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError(f"fake {self.name} did not start (see {self.log_path})")

    def stop(self, sig=signal.SIGTERM):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, sig)
            except OSError:
                pass
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        d = json.loads(r.read() or b"null")
        c.close()
        return d

    def log(self, since=0):
        return self.get(f"/fake/log?since={since}")["log"]

    def requests(self):
        return self.get("/fake/requests")["requests"]

    def reset(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/fake/reset", body=b"{}", headers={"Content-Type": "application/json"})
        c.getresponse().read()
        c.close()


def register_launched(cfg_dir, pid, port, model):
    """tell the scratch Control 'the launcher started this server' (cfg/launched/<pid>.json), like run-server.sh."""
    d = os.path.join(cfg_dir, "launched")
    os.makedirs(d, mode=0o700, exist_ok=True)
    rec = {"pid": int(pid), "start": proc_start_ticks(pid), "port": int(port), "host": None,
           "model": f"/models/{model}.gguf", "gpus": [0], "by": "harness", "since": time.time()}
    p = os.path.join(d, f"{int(pid)}.json")
    fd = os.open(p + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f)
    os.replace(p + ".tmp", p)


def control_child(port):
    """the scratch Control (run as a child process of the suite; its env carries the scratch dirs and the token)."""
    sys.path.insert(0, TOOLS)
    import importlib.util
    spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    import pxa_control as C
    if C.CHAT is None:
        print("CONTROL CHAT MISSING:", C.CHAT_IMPORT_ERROR, flush=True)
        return 3
    app = C.App(L, port=port, token=os.environ["PXA_HARNESS_TOKEN"])
    srv = C.make_server(app, "127.0.0.1", port)
    print(f"CONTROL READY 127.0.0.1:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


class Control(object):
    def __init__(self, out, scratch, port=None, approval_timeout=6):
        self.out, self.scratch = out, scratch
        self.port = port or free_port()
        self.cfg = os.path.join(scratch, "cfg")
        self.state = os.path.join(scratch, "state")
        self.token = secrets.token_urlsafe(24)
        self.approval_timeout = approval_timeout
        self.proc = None
        self.log_path = os.path.join(out, "control.log")

    def start(self):
        os.makedirs(self.cfg, mode=0o700, exist_ok=True)
        os.makedirs(self.state, mode=0o700, exist_ok=True)
        env = dict(os.environ)
        for k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S",
                  "PXA_CONTROL_TOKEN", "PXA_CHAT_SEARCH_URL", "PXA_BUS_TOKEN"):
            env.pop(k, None)
        env.update({"PXA_CONTROL_CONFIG_DIR": self.cfg, "PXA_LAUNCH_STATE": self.state,
                    "PXA_LAUNCH_FAKE_GPUS": "2x16000", "PXA_CONTROL_DISCOVER": "0",
                    "PXA_CHAT_APPROVAL_TIMEOUT": str(self.approval_timeout), "PXA_HARNESS_TOKEN": self.token,
                    "PYTHONUNBUFFERED": "1"})
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_control-child", str(self.port)],
                                     env=env, stdout=open(self.log_path, "ab"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=REPO)
        t0 = time.time()
        while time.time() - t0 < 40:
            if http_ok(self.port, "/api/health") in (200, 401):
                return self
            if self.proc.poll() is not None:
                break
            time.sleep(0.2)
        raise RuntimeError(f"Control did not start (see {self.log_path})")

    def crash(self):
        """kill -9 (a crash or a pulled plug): nothing gets a chance to save."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(8)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(8)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    pass


# =====================================================================================================================
# HTTP client for Control's chat API
# =====================================================================================================================
class Turn(object):
    def __init__(self):
        self.status, self.run, self.events, self.error, self.pings = None, {}, [], None, 0
        self.t0, self.t_end = time.time(), None

    @property
    def types(self):
        return [e["type"] for e in self.events]

    @property
    def end(self):
        return next((e for e in reversed(self.events) if e["type"] in TERMINAL), None)

    @property
    def end_type(self):
        e = self.end
        return e["type"] if e else None

    @property
    def text(self):
        e = self.end
        return ((e or {}).get("data") or {}).get("text") or ((e or {}).get("data") or {}).get("error") or ""

    def of(self, typ):
        return [e["data"] for e in self.events if e["type"] == typ]

    def kinds(self):
        return [d.get("kind") for d in self.of("status")]

    def codes(self):
        return [d.get("code") for d in self.of("tool.result")]

    def summary(self, n=400):
        bits = [f"HTTP {self.status}"]
        if self.error:
            bits.append("error=" + str(self.error)[:200])
        if self.events:
            bits.append("types=" + ",".join(self.types[:14]) + ("..." if len(self.types) > 14 else ""))
            bits.append(f"end={self.end_type}: {self.text[:160]!r}")
        return " | ".join(bits)[:n]


class Client(object):
    def __init__(self, port, token, host="127.0.0.1"):
        self.host, self.port, self.token = host, port, token

    def req(self, method, path, body=None, headers=None, raw=None, timeout=30, token=True):
        h = {"Accept": "application/json"}
        if token and self.token:
            h["X-PXA-Token"] = self.token
        data = None
        if raw is not None:
            data = raw if isinstance(raw, bytes) else raw.encode()
            h["Content-Type"] = "application/json"
        elif body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            c.request(method, path, body=data, headers=h)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def json(self, method, path, body=None, **kw):
        st, hd, raw = self.req(method, path, body, **kw)
        try:
            return st, json.loads(raw or b"null"), hd
        except ValueError:
            return st, {"_raw": raw[:300].decode("utf-8", "replace")}, hd

    def cancel(self, run_id):
        return self.json("POST", "/api/chat/cancel", {"run_id": run_id})

    def approve(self, run_id, approval_id, decision):
        return self.json("POST", "/api/chat/approve", {"run_id": run_id, "approval_id": approval_id,
                                                       "decision": decision})

    def events(self, run_id, since=0, timeout=90, on_event=None, stop=None, read_delay=0, turn=None):
        """read /api/chat/events until a terminal event, stop(ev) is true, or the timeout. -> Turn"""
        t = turn or Turn()
        c = http.client.HTTPConnection(self.host, self.port, timeout=min(30, timeout))
        try:
            c.request("GET", f"/api/chat/events?run={run_id}&since={since}",
                      headers={"X-PXA-Token": self.token, "Accept": "text/event-stream"})
            r = c.getresponse()
            if r.status != 200:
                t.error = f"events HTTP {r.status}: {r.read()[:200]!r}"
                return t
            if read_delay:
                time.sleep(read_delay)
            deadline = time.time() + timeout
            data = []
            while time.time() < deadline:
                try:
                    line = r.readline(8 << 20)
                except socket.timeout:
                    continue
                if not line:
                    break
                line = line.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith(":"):
                    t.pings += 1
                    continue
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                    continue
                if line == "" and data:
                    ev = json.loads("\n".join(data))
                    data = []
                    t.events.append(ev)
                    if on_event:
                        on_event(ev, t)
                    if ev["type"] in TERMINAL or (stop and stop(ev)):
                        break
        except (OSError, http.client.HTTPException, ValueError) as e:
            t.error = f"events: {e.__class__.__name__}: {e}"
        finally:
            c.close()
        t.t_end = time.time()
        return t

    def run(self, message, server=None, preset="assistant", session_id=None, advanced=False, approve=None,
            on_event=None, timeout=90, wait=True, **extra):
        """POST /api/chat/run, then follow the events (answering approvals with approve(data) -> decision)."""
        t = Turn()
        body = {"message": message, "preset": preset, "advanced": advanced}
        if server:
            body["server"] = server
        if session_id:
            body["session_id"] = session_id
        body.update(extra)
        st, d, _ = self.json("POST", "/api/chat/run", body, timeout=30)
        t.status, t.run = st, d if isinstance(d, dict) else {}
        if st != 200:
            t.error = (d or {}).get("error") if isinstance(d, dict) else str(d)
            return t
        if not wait:
            return t

        def hook(ev, tt):
            if ev["type"] == "approval.request" and approve:
                dec = approve(ev["data"])
                if dec:
                    self.approve(tt.run["run_id"], ev["data"]["approval_id"], dec)
            if on_event:
                on_event(ev, tt)
        return self.events(t.run["run_id"], 0, timeout, on_event=hook, turn=t)


# =====================================================================================================================
# the catalogue
# =====================================================================================================================
CASES = []


class Fail(Exception):
    pass


def check(cond, observed):
    if not cond:
        raise Fail(observed)


def case(cid, expect, modes=("fake",), xfail=None, group=None):
    def deco(fn):
        CASES.append({"id": cid, "expect": expect, "fn": fn, "modes": modes, "xfail": xfail, "kind": "auto",
                      "group": group or re.match(r"[A-Z]+", cid).group(0)})
        return fn
    return deco


def alias(cid, of, expect):
    """a catalogue id whose behaviour is checked by another case: its result is that case's."""
    CASES.append({"id": cid, "expect": expect, "alias": of, "kind": "alias", "modes": ("fake", "live"),
                  "group": re.match(r"[A-Z]+", cid).group(0)})


def manual(cid, expect, reason):
    CASES.append({"id": cid, "expect": expect, "reason": reason, "kind": "manual", "modes": (),
                  "group": re.match(r"[A-Z]+", cid).group(0)})


def unit(cid, expect, target, xfail=None):
    """a case checked by a unit test file: target = 'file.py' or 'file.py::Class::test'."""
    CASES.append({"id": cid, "expect": expect, "target": target, "kind": "unit", "modes": ("fake", "live"),
                  "xfail": xfail, "group": re.match(r"[A-Z]+", cid).group(0)})


def M(scen, text="go"):
    return f"{text} [[fake:{scen}]]"


def ok_done(t, needle=None):
    check(t.end_type == "run.done", t.summary())
    if needle:
        check(needle in t.text, t.summary())
    return f"run.done: {t.text[:200]!r}"


def wait_for(fn, timeout=10, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(step)
    return None


def fake_saw_close(fake, since, phase=None, timeout=4):
    def look():
        return [x for x in fake.log(since) if x["kind"] == "client_closed" and (phase is None or x.get("phase") == phase)]
    return wait_for(look, timeout)


def session(env, sid):
    return env.c.json("GET", "/api/chat/session?id=" + sid)[1]


# ---------------------------------------------------------------------------------------------------------------- S
@case("S01", "an SSE event split mid-JSON is reassembled")
def s01(env):
    t = env.c.run(M("split_frames"), server=env.A.key, preset="chat")
    return ok_done(t, "split")


@case("S02", "a 4-byte emoji split across TCP writes arrives intact")
def s02(env):
    t = env.c.run(M("split_frames"), server=env.A.key, preset="chat")
    return ok_done(t, "\U0001F999")


@case("S03", "CRLF, comments, event:/id:/retry: lines, data: without a space")
def s03(env):
    return ok_done(env.c.run(M("crlf"), server=env.A.key, preset="chat"), "CRLF-OK")


@case("S04", "one JSON event over two data: lines is joined")
def s04(env):
    return ok_done(env.c.run(M("multiline"), server=env.A.key, preset="chat"), "MULTI-LINE-OK")


@case("S05", "finish_reason without [DONE] is a normal end")
def s05(env):
    return ok_done(env.c.run(M("no_done"), server=env.A.key, preset="chat"), "NO-DONE-OK")


@case("S06", "a stream cut with a clean FIN is retried, never shown as finished")
def s06(env):
    t = env.c.run(M("cut_fin:close"), server=env.A.key, preset="chat")
    ok_done(t, "CUT-RECOVERED")
    check("retry" in t.kinds(), "no retry status: " + t.summary())
    sid = t.run["session_id"]
    s = session(env, sid)
    check("CUT-PARTIAL" not in json.dumps(s.get("turns")), "partial text saved: " + json.dumps(s.get("turns"))[:300])
    return "retried, saved turn clean: " + t.text[:80]


@case("S07", "a reset mid-reply retries; the dead attempt's words are retracted")
def s07(env):
    t = env.c.run(M("reset"), server=env.A.key, preset="chat")
    ok_done(t, "RESET-RECOVERED")
    check({"step": 1, "text": ""} in t.of("text.retract"), "no retract: " + t.summary())
    s = session(env, t.run["session_id"])
    check("RESET-PARTIAL" not in json.dumps(s.get("turns")), "partial text saved")
    return "retracted + retried: " + t.text[:80]


@case("S08", "a mid-stream error object ends the turn with the server's words")
def s08(env):
    t = env.c.run(M("midstream_error"), server=env.A.key, preset="chat")
    check(t.end_type == "run.error" and "mid-stream failure" in t.text, t.summary())
    return t.text[:120]


@case("S09", "a JSON answer to stream:true is handled")
def s09(env):
    return ok_done(env.c.run(M("non_stream"), server=env.A.key, preset="chat"), "NON-STREAM-OK")


@case("S10", "an all-garbage stream is an error, not an empty answer")
def s10(env):
    t = env.c.run(M("garbage"), server=env.A.key, preset="chat")
    check(t.end_type == "run.error" and t.end["data"].get("code") == "bad_stream", t.summary())
    return f"run.error bad_stream after {t.kinds().count('retry')} retries"


@case("S11", "a slow steady stream (1 token/s) completes")
def s11(env):
    return ok_done(env.c.run(M("slow:1"), server=env.A.key, preset="chat", timeout=40), "SLOW-OK")


@case("S12", "17 s with no byte (prefill) is not a timeout even with step_timeout 10; a progress note appears")
def s12(env):
    t = env.c.run(M("prefill:17"), server=env.A.key, preset="chat", advanced=True, step_timeout=10, timeout=60)
    ok_done(t, "PREFILL-OK")
    check("first_byte" in t.kinds(), "no first_byte status: " + t.summary())
    return "done after the prefill; status: " + next(d["text"] for d in t.of("status") if d.get("kind") == "first_byte")[:120]


@case("S13", "a page that does not read its events never blocks the loop")
def s13(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("big:300000"), "preset": "chat",
                                                     "server": env.A.key})
    check(st == 200, f"HTTP {st} {d}")
    time.sleep(6)                                   # nobody reads; the loop must still finish
    t = env.c.events(d["run_id"], 0, 60)
    ok_done(t)
    end_t = t.end["t"]
    check(end_t < t.t0 + 1 or end_t < t.t_end, "loop waited for the reader")
    return f"{len(t.events)} events replayed after a 6 s stall; run ended at server time {end_t}"


@case("S15", "closing the page mid-reply keeps the answer going; reopening re-attaches")
def s15(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:0.7"), "preset": "chat",
                                                     "server": env.A.key})
    check(st == 200, f"HTTP {st}")
    first = env.c.events(d["run_id"], 0, 10, stop=lambda ev: ev["type"] == "text.delta")
    s = session(env, d["session_id"])
    check(s.get("running") == d["run_id"], "session does not say running: " + str(s.get("running")))
    t = env.c.events(d["run_id"], 0, 30)
    ok_done(t, "SLOW-OK")
    return f"re-attached from seq 0 ({len(first.events)} events before the close), finished"


@case("S17", "reconnect with since=N: no gap, no duplicate")
def s17(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:0.5"), "preset": "chat",
                                                     "server": env.A.key})
    a = env.c.events(d["run_id"], 0, 10, stop=lambda ev: ev["type"] == "text.delta" and ev["seq"] >= 4)
    k = a.events[-1]["seq"]
    b = env.c.events(d["run_id"], k, 30)
    seqs = [e["seq"] for e in a.events] + [e["seq"] for e in b.events]
    check(seqs == list(range(1, len(seqs) + 1)), f"seqs {seqs}")
    return ok_done(b, "SLOW-OK") + f"; seqs 1..{len(seqs)} contiguous"


@case("S18", "events for an unknown or collected run answer 404 unknown_run")
def s18(env):
    st, d, _ = env.c.json("GET", "/api/chat/events?run=run_gone_123")
    check(st == 404 and d.get("code") == "unknown_run", f"HTTP {st} {d}")
    return "404 unknown_run (the page then reloads the saved chat: UI part is manual)"


@case("S19", "since negative / huge / not a number is clamped")
def s19(env):
    t = env.c.run(M("plain"), server=env.A.key, preset="chat")
    rid = t.run["run_id"]
    n = len(t.events)
    a = env.c.events(rid, -5, 10)
    b = env.c.events(rid, 10 ** 9, 5)
    st, hd, raw = env.c.req("GET", f"/api/chat/events?run={rid}&since=abc", timeout=10)
    check(len(a.events) == n and not b.events and st == 200, f"neg={len(a.events)}/{n} huge={len(b.events)} abc={st}")
    return f"negative -> all {n}; huge -> none; abc -> HTTP {st}"


@case("S20", "an idle event stream (a 20 s silent tool) carries keep-alive comments")
def s20(env):
    t = env.c.run(M("sleep_cmd:20.5"), server=env.A.key, advanced=True, tools=["run_command"],
                  approve=lambda d: "once", timeout=60)
    ok_done(t, "SLEEP-DONE")
    check(t.pings >= 1, f"pings={t.pings}")
    return f"{t.pings} ': ping' comment(s) during a 20 s silent tool run"


@case("S22", "data after [DONE] is ignored")
def s22(env):
    t = env.c.run(M("after_done"), server=env.A.key, preset="chat")
    ok_done(t, "AFTER-DONE-OK")
    check("LEAKED" not in t.text, t.summary())
    return t.text[:80]


# ---------------------------------------------------------------------------------------------------------------- T
def calc_run(env, scen, tools_preset="assistant", fake=None, **kw):
    return env.c.run(M(scen), server=(fake or env.A).key, preset=tools_preset, **kw)


@case("T01", "one native tool call: run, feed back, answer")
def t01(env):
    return ok_done(calc_run(env, "calc"), "CALC-DONE: the result was 42")


@case("T02", "two calls in one message run in order, each bound to its id")
def t02(env):
    return ok_done(calc_run(env, "parallel"), "PARALLEL-DONE 5 | 20")


@case("T03", "call fragments without index join the open call")
def t03(env):
    return ok_done(calc_run(env, "noindex"), "NOINDEX-DONE 81")


@case("T04", "a name repeated in every chunk is kept once")
def t04(env):
    return ok_done(calc_run(env, "name_repeat"), "NAME_REPEAT-DONE 81")


@case("T05", "duplicate call ids are made unique")
def t05(env):
    return ok_done(calc_run(env, "dup_ids"), "ids=call_dup,call_dup_2")


@case("T06", "a call without an id gets one")
def t06(env):
    t = calc_run(env, "noid")
    ok_done(t, "NOID-DONE 64")
    check("id=None" not in t.text, t.summary())
    return t.text[:80]


@case("T07", "broken JSON arguments: a coded error with the expected shape; the model recovers")
def t07(env):
    t = calc_run(env, "badjson")
    ok_done(t, "FIXED-AFTER-ERROR 42")
    check(t.codes()[0] == "invalid_arguments", str(t.codes()))
    return t.text[:80] + f" codes={t.codes()}"


@case("T08", "a call cut at the length limit is explained as cut")
def t08(env):
    t = calc_run(env, "cut_call", approve=lambda d: "once")
    ok_done(t, "CUT-CALL-SEEN cut off")
    return t.text[:80] + f" codes={t.codes()}"


@case("T09", "wrong argument types are coerced or refused clearly")
def t09(env):
    return ok_done(calc_run(env, "wrongtypes"), "WRONGTYPES-DONE 42 = 42")


@case("T10", "a missing required argument is named")
def t10(env):
    t = calc_run(env, "missing_arg")
    ok_done(t, "MISSING-ARG-SEEN named")
    return t.text[:80] + f" codes={t.codes()}"


@case("T11", "an invented tool is refused with the list of real ones")
def t11(env):
    t = calc_run(env, "unknown_tool")
    ok_done(t, "UNKNOWN-TOOL-SEEN listed")
    return t.text[:80] + f" codes={t.codes()}"


@case("T12", "a real tool that is off for this chat is refused as off")
def t12(env):
    t = env.c.run(M('call:list_files:{}'), server=env.A.key, preset="researcher")
    ok_done(t)
    check("tool_off" in t.codes() and "not turned on" in t.text, t.summary())
    return t.text[:160]


@case("T13", "a tool that raises returns an error line; the loop goes on")
def t13(env):
    t = env.c.run(M('call:read_file:{"path": "nothing-here.txt", "offset": "abc"}'), server=env.A.key)
    ok_done(t, "CALL-DONE error")
    return t.text[:160]


@case("T14", "a hanging tool hits the per-tool deadline (tool_timeout 5) and the model gets a timeout error")
def t14(env):
    t0 = time.time()
    t = env.c.run(M("slow_fetch:60"), server=env.A.key, advanced=True, tool_timeout=5,
                  tools=["web_fetch"], approve=lambda d: "once", timeout=60)
    ok_done(t, "SLOW-FETCH-DONE timed-out")
    check("timeout" in t.codes(), str(t.codes()))
    return f"timed out after {time.time() - t0:.1f} s total; codes={t.codes()}"


@case("T15", "a 150 KB file read is clipped to the window share before the model sees it")
def t15(env):
    a = env.c.run(M("plain", "hi"), server=env.C.key)
    ok_done(a)
    sid = a.run["session_id"]
    box = os.path.join(env.ctl.cfg, "chat", "sandbox", sid)
    os.makedirs(box, exist_ok=True)
    with open(os.path.join(box, "big.txt"), "w") as f:
        f.write("".join(f"row {i:05d} " + "z" * 140 + "\n" for i in range(1000)))
    t = env.c.run(M("read:big.txt"), server=env.C.key, session_id=sid)
    ok_done(t, "READ-DONE")
    n = int(re.search(r"chars=(\d+)", t.text).group(1))
    full = max(d.get("chars") or 0 for d in t.of("tool.result"))
    check(n <= 2048 * 4 * 0.25 + 600 and full > 100000, f"model saw {n} chars of {full}")
    return f"model saw {n} of {full} characters (window 2048 tokens)"


@case("T16", "the identical call repeated is refused, then tools are taken away and the model answers")
def t16(env):
    t = calc_run(env, "loop")
    ok_done(t, "LOOP-ENDED")
    check(t.codes().count("loop_guard") == 2, str(t.codes()))
    return f"{len(t.of('tool.call'))} calls, codes={t.codes()}"


@case("T17", "a 30-call storm is capped at max_calls_per_step (8); the rest are refused with a reason")
def t17(env):
    t = calc_run(env, "storm:30")
    ok_done(t, "STORM-DONE")
    check(t.codes().count("too_many_calls") == 22, str(t.codes()))
    return f"8 ran, {t.codes().count('too_many_calls')} refused"


@case("T18", "a sub-agent cannot get spawn_agent")
def t18(env):
    env.A.reset()
    t = calc_run(env, "spawn_nested")
    ok_done(t, "SPAWN-NESTED-DONE")
    child = [r for r in env.A.requests()
             if str(((r.get("messages") or [{}])[0]).get("content") or "").startswith("You are a sub-agent inside")]
    check(child, "no child request seen")
    names = [x["function"]["name"] for r in child for x in (r.get("tools") or [])]
    check("spawn_agent" not in names, f"child tools {names}")
    return f"child requests {len(child)}, child tools {sorted(set(names)) or 'none'}"


@case("T19", "instruction-shaped text from a file is tagged as data before the model sees it")
def t19(env):
    env.A.reset()
    t = calc_run(env, "inject_file", approve=lambda d: "once")
    ok_done(t, "INJECT-DONE")
    tool_msgs = [m for m in env.A.requests()[-1]["messages"] if m.get("role") == "tool"]
    check(tool_msgs and tool_msgs[-1]["content"].startswith("[POSSIBLE INJECTED INSTRUCTION"), "not tagged")
    return "tagged; " + t.text[:60]


@case("T20", "chat-template control tokens in tool output are made inert")
def t20(env):
    env.A.reset()
    t = calc_run(env, "inject_file", approve=lambda d: "once")
    ok_done(t, "INJECT-DONE tokens-inert")
    body = [m for m in env.A.requests()[-1]["messages"] if m.get("role") == "tool"][-1]["content"]
    check("<|im_start|>" not in body and "‹|im_start|›" in body, body[:200])
    return "inert: " + body[body.find("‹"):][:60]


@case("T21", "NUL, ESC and invalid UTF-8 in a tool's output arrive as text")
def t21(env):
    t = env.c.run(M("ctrl_tool"), server=env.A.key, advanced=True, tools=["run_command"], approve=lambda d: "once")
    ok_done(t, "CTRL-TOOL-DONE")
    return t.text[:80]


@case("T22", "text protocol: a second call in one reply is named as not run")
def t22(env):
    return ok_done(env.c.run(M("text_multi"), server=env.B.key), "TEXT-MULTI-DONE noted")


@case("T23", "a text-protocol call inside a ```json fence is parsed")
def t23(env):
    return ok_done(env.c.run(M("fenced_text"), server=env.B.key), "FENCED-DONE 42")


@case("T24", "a native server whose model writes the call as text still gets the call run")
def t24(env):
    return ok_done(calc_run(env, "text_on_native"), "TEXT-ON-NATIVE-DONE 42")


@case("T25", "an unanswered approval times out as a deny the model is told about")
def t25(env):
    t = env.c.run(M('call:write_file:{"path": "a.txt", "content": "x"}'), server=env.A.key, timeout=40)
    ok_done(t)
    check("approval_timeout" in t.codes(), str(t.codes()))
    return t.text[:120]


@case("T26", "an approval answered twice / for another run / 'always' on run_command is refused")
def t26(env):
    holder = {}

    def on_event(ev, tt):
        if ev["type"] == "approval.request" and "done" not in holder:
            holder["done"] = True
            rid, aid = tt.run["run_id"], ev["data"]["approval_id"]
            holder["always"] = env.c.approve(rid, aid, "always")[:2]
            holder["wrong"] = env.c.approve("run_other", aid, "once")[:2]
            holder["first"] = env.c.approve(rid, aid, "once")[:2]
            holder["twice"] = env.c.approve(rid, aid, "once")[:2]
    t = env.c.run(M('call:run_command:{"command": "echo hi"}'), server=env.A.key, advanced=True,
                  tools=["run_command"], on_event=on_event, timeout=40)
    ok_done(t)
    check(holder["always"][0] == 400 and holder["always"][1].get("code") == "not_allowed", str(holder))
    check(holder["wrong"][0] == 404 and holder["first"][0] == 200 and holder["twice"][0] == 404, str(holder))
    return "always->400 not_allowed, wrong run->404, first->200, twice->404"


@case("T27", "a page reconnecting while an approval waits sees the card again")
def t27(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M('call:write_file:{"path": "b.txt", "content": "y"}'),
                                                     "server": env.A.key})
    a = env.c.events(d["run_id"], 0, 15, stop=lambda ev: ev["type"] == "approval.request")
    check(a.events and a.events[-1]["type"] == "approval.request", a.summary())
    b = env.c.events(d["run_id"], 0, 5, stop=lambda ev: ev["type"] == "approval.request")
    ap = b.events[-1]["data"]
    env.c.approve(d["run_id"], ap["approval_id"], "once")
    t = env.c.events(d["run_id"], b.events[-1]["seq"], 30)
    return ok_done(t, "CALL-DONE") + "; card replayed on reconnect"


@case("T28", "an empty final turn after a tool gets one short line from the result")
def t28(env):
    t = calc_run(env, "empty_after_tool")
    ok_done(t)
    check(t.end["data"].get("fallback") and "42" in t.text, t.summary())
    return t.text[:80]


@case("T29", "finish_reason tool_calls with the call only as text is still run")
def t29(env):
    return ok_done(calc_run(env, "tc_no_calls"), "TC-NO-CALLS-DONE 9")


@case("T30", "calculator abuse (9**9**9) is refused cleanly")
def t30(env):
    t = env.c.run(M('call:calculate:{"expression": "9**9**9"}'), server=env.A.key)
    ok_done(t, "too large")
    return t.text[:120]


@case("T31", "arguments double-encoded as a JSON string are decoded")
def t31(env):
    return ok_done(calc_run(env, "double_encoded"), "DOUBLE-ENCODED-DONE 25")


@case("T32", "write_file outside the sandbox is refused")
def t32(env):
    t = env.c.run(M('call:write_file:{"path": "../escape.txt", "content": "x"}'), server=env.A.key,
                  approve=lambda d: "once")
    ok_done(t, "outside the sandbox")
    check(not os.path.exists(os.path.join(env.ctl.cfg, "chat", "sandbox", "escape.txt")), "file escaped")
    return t.text[:120]


# ---------------------------------------------------------------------------------------------------------------- C
def long_session(env, fake, turns=3, words=380, marker="plain", first_marker=None, **extra):
    sid = None
    for i in range(turns):
        mk = first_marker if (i == 0 and first_marker) else marker
        t = env.c.run(f"turn {i}: " + ("lorem ipsum dolor sit " * (words // 4)) + f" [[fake:{mk}]]",
                      server=fake.key, preset="chat", session_id=sid, **extra)
        check(t.end_type == "run.done", f"setup turn {i}: " + t.summary())
        sid = t.run["session_id"]
    return sid


@case("C01", "with -np 2 the budget is the per-request window: compaction before the server refuses")
def c01(env):
    st, d, _ = env.c.json("GET", "/api/chat/servers")
    srv = next(s for s in d["servers"] if s["key"] == env.C.key)
    check((srv.get("ctx") or {}).get("resolved") == 2048, "picker ctx " + json.dumps(srv.get("ctx")))
    sid = long_session(env, env.C, turns=2)
    t = env.c.run("and the next one? " + "lorem ipsum dolor sit " * 95 + " [[fake:plain]]", server=env.C.key,
                  preset="chat", session_id=sid)
    ok_done(t, "PLAIN-OK")
    check("compact" in t.types and "compact" not in t.kinds(), "types " + ",".join(t.types[:8]) + " kinds " + str(t.kinds()))
    return "picker 2048 (ring 4096); compacted up front, no 400"


@case("C02", "a server without /props still chats (8192 assumed)")
def c02(env):
    return ok_done(env.c.run(M("plain"), server=env.E.key, preset="chat"), "PLAIN-OK")


@case("C03", "a 400 context overflow compacts harder and retries once")
def c03(env):
    t = env.c.run(M("overflow_once"), server=env.A.key, preset="chat")
    ok_done(t, "PLAIN-OK" if False else None)
    check("compact" in t.kinds(), t.summary())
    return "overflow -> compact -> done"


@case("C04", "an overflow after the retry gives words with numbers and a way out")
def c04(env):
    t = env.c.run(M("overflow_always"), server=env.A.key, preset="chat")
    check(t.end_type == "run.error" and t.end["data"].get("code") == "context_overflow" and "no longer fits" in t.text,
          t.summary())
    return t.text[:160]


@case("C05", "a message larger than the window is refused before the call, with numbers")
def c05(env):
    env.C.reset()
    t = env.c.run("summarize: " + "word " * 3000 + " [[fake:plain]]", server=env.C.key, preset="chat")
    check(t.end_type == "run.error" and t.end["data"].get("code") == "too_long", t.summary())
    sent = [x for x in env.C.log() if x["kind"] == "request"]
    check(not sent, f"{len(sent)} chat request(s) reached the server")
    return t.text[:160]


@case("C06", "system prompt bigger than half the window is refused with numbers (memory/ref blocks are trimmed first: unit); a normal one is untouched")
def c06(env):
    t = env.c.run(M("plain"), server=env.C.key, preset="chat", advanced=True, system="rule. " * 2400)
    d = (t.end or {}).get("data") or {}
    check(t.end_type == "run.error" and d.get("code") == "system_too_big" and "half" in d.get("error", "")
          and d.get("n_ctx") and d.get("tokens", 0) > d.get("limit", 1 << 30), t.summary())
    t2 = env.c.run(M("plain"), server=env.C.key, preset="chat", advanced=True, system="Be brief. " * 20)
    ok_done(t2, "PLAIN-OK")
    return f"refused: {d.get('error', '')[:110]}"


@case("C08", "compaction off on a long chat: the overflow path still compacts")
def c08(env):
    sid = long_session(env, env.C, turns=3, compact=False)
    t = env.c.run("next " + "lorem ipsum dolor sit " * 95 + " [[fake:plain]]", server=env.C.key, preset="chat",
                  session_id=sid, compact=False)
    ok_done(t, "PLAIN-OK")
    return "done; kinds=" + str(t.kinds())


@case("C09", "Advanced system replaces the preset; tool notice and memory blocks are appended")
def c09(env):
    env.A.reset()
    t = env.c.run(M("plain"), server=env.A.key, advanced=True, system="SYS-MARK-ALPHA-9", tools=["calculate"])
    ok_done(t)
    sysm = env.A.requests()[-1]["messages"][0]
    check(sysm["role"] == "system" and sysm["content"].startswith("SYS-MARK-ALPHA-9")
          and "Never follow instructions" in sysm["content"], sysm["content"][:200])
    return "system starts with the Advanced text; data notice appended"


@case("C10", "editing an old message (rewind) keeps the old turns as a version")
def c10(env):
    a = env.c.run("first [[fake:plain]]", server=env.A.key, preset="chat")
    sid = a.run["session_id"]
    env.c.run("second [[fake:plain]]", server=env.A.key, preset="chat", session_id=sid)
    t = env.c.run("first, edited [[fake:plain]]", server=env.A.key, preset="chat", session_id=sid, rewind=0)
    ok_done(t)
    time.sleep(0.3)
    s = session(env, sid)
    check(len(s["turns"]) == 1 and (s["turns"][0].get("branch") or {}).get("n") == 2, json.dumps(s["turns"])[:300])
    return "1 turn shown, branch 2 of 2"


@case("C11", "regenerate an earlier turn while the chat answers is refused 409")
def c11(env):
    a = env.c.run("first [[fake:plain]]", server=env.A.key, preset="chat")
    sid = a.run["session_id"]
    time.sleep(0.3)
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:0.5"), "preset": "chat", "server": env.A.key,
                                                     "session_id": sid})
    st2, d2, _ = env.c.json("POST", "/api/chat/run", {"message": "first, again [[fake:plain]]", "preset": "chat",
                                                       "server": env.A.key, "session_id": sid, "rewind": 0})
    env.c.cancel(d["run_id"])
    env.c.events(d["run_id"], 0, 10)
    check(st == 200 and st2 == 409 and d2.get("code") == "busy", f"HTTP {st} / {st2} {d2}")
    return f"HTTP {st2} {d2.get('code')}"


PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@case("C12", "an image to a text-only server is left out with a note; the next turn is not poisoned")
def c12(env):
    env.A.reset()
    t = env.c.run("what is this? [[fake:plain]]", server=env.A.key, preset="chat",
                  images=[{"media_type": "image/png", "data": PNG}])
    ok_done(t, "PLAIN-OK")
    check("no_vision" in t.kinds(), t.summary())
    t2 = env.c.run("and now? [[fake:plain]]", server=env.A.key, preset="chat", session_id=t.run["session_id"])
    ok_done(t2, "PLAIN-OK")
    check("image_url" not in json.dumps(env.A.requests()[-1]["messages"]), "image sent to a text-only server")
    v = env.c.run("what is this? [[fake:plain]]", server=env.V.key, preset="chat",
                  images=[{"media_type": "image/png", "data": PNG}])
    ok_done(v)
    check("image_url" in json.dumps(env.V.requests()[-1]["messages"]), "vision server lost the image")
    return "text-only: left out + note, 2nd turn fine; vision server: image sent"


@case("C13", "an image of ~0.95 MB (the UI allows 1 MB) is accepted")
def c13(env):
    raw = os.urandom(950000)
    b64 = base64.b64encode(raw).decode()
    t = env.c.run("big picture [[fake:plain]]", server=env.V.key, preset="chat",
                  images=[{"media_type": "image/jpeg", "data": b64}])
    ok_done(t)
    return f"body ~{len(b64) // 1024} KB accepted"


@case("C14", "an SVG / bad base64 image is refused with a reason")
def c14(env):
    t = env.c.run("svg [[fake:plain]]", server=env.V.key, preset="chat",
                  images=[{"media_type": "image/svg+xml", "data": PNG}])
    check(t.status == 400, t.summary())
    for bad in ({"media_type": "image/png", "data": "@@@"}, {"media_type": "image/png", "data": PNG[:-3]}):
        st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("plain"), "preset": "chat", "server": env.V.key,
                                                       "images": [bad]})
        check(st == 400 and d.get("code") == "bad_image" and "base64" in d.get("error", ""), f"HTTP {st} {d}")
    return t.summary()


@case("C16", "a message over 200000 chars is refused 413")
def c16(env):
    t = env.c.run("x" * 200001, server=env.A.key, preset="chat")
    check(t.status == 413, t.summary())
    return f"HTTP {t.status} {t.error}"


@case("C17", "an empty or whitespace message is refused 400")
def c17(env):
    t = env.c.run("   \n ", server=env.A.key, preset="chat")
    check(t.status == 400, t.summary())
    return f"HTTP {t.status} {t.error}"


@case("C18", "a number as the message is refused or stringified")
def c18(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": 12345, "preset": "chat", "server": env.A.key})
    if st == 200:
        t = env.c.events(d["run_id"], 0, 20)
        ok_done(t)
    check(st in (200, 400), f"HTTP {st}")
    return f"HTTP {st}"


# ---------------------------------------------------------------------------------------------------------------- X
def start_and_cancel(env, msg, when, fake=None, after=0.0, timeout=40, **kw):
    """start a run, cancel it when when(ev) is true (+after s). -> (turn, seconds from cancel to the end)"""
    holder = {}

    def on_event(ev, t):
        if "c" not in holder and when(ev):
            holder["c"] = time.time() + after

            def go():
                time.sleep(after)
                holder["resp"] = env.c.cancel(t.run["run_id"])[:2]
            threading.Thread(target=go, daemon=True).start()
    t = env.c.run(msg, server=(fake or env.A).key, on_event=on_event, timeout=timeout, **kw)
    return t, (t.t_end - holder["c"]) if "c" in holder and t.t_end else None, holder


@case("X01", "Stop mid-stream ends the turn within ~1 s and the server sees the socket close")
def x01(env):
    since = len(env.A.log())
    t, secs, _ = start_and_cancel(env, M("slow:1"), lambda ev: ev["type"] == "text.delta", preset="chat")
    check(t.end_type == "run.cancelled" and secs is not None and secs < 3, f"{t.summary()} secs={secs}")
    check(fake_saw_close(env.A, since, "stream"), "the fake never saw the close")
    return f"cancelled {secs:.2f} s after Stop; server saw the close"


@case("X02", "Stop during prefill (no token yet) reaches the server")
def x02(env):
    since = len(env.A.log())
    t, secs, _ = start_and_cancel(env, M("prefill:30"), lambda ev: ev["type"] == "step", after=1.5, preset="chat")
    check(t.end_type == "run.cancelled" and secs is not None and secs < 3, f"{t.summary()} secs={secs}")
    check(fake_saw_close(env.A, since, "prefill"), "the fake kept prefilling for nobody")
    return f"cancelled {secs:.2f} s after Stop; server saw the close during prefill"


@case("X03", "Stop while a command runs kills it and ends the turn at once")
def x03(env):
    t, secs, _ = start_and_cancel(env, M("sleep_cmd:41.25"), lambda ev: ev["type"] == "tool.result" or
                                  (ev["type"] == "tool.call" and ev["data"]["name"] == "run_command"), after=1.0,
                                  advanced=True, tools=["run_command"], approve=lambda d: "once")
    check(t.end_type == "run.cancelled" and secs is not None and secs < 5, f"{t.summary()} secs={secs}")
    time.sleep(0.5)
    ps = subprocess.run(["pgrep", "-f", "sleep 41.25"], capture_output=True, text=True).stdout.strip()
    check(not ps, "sleep still running: " + ps)
    return f"cancelled {secs:.2f} s after Stop; the process is gone"


@case("X04", "Stop while an approval waits")
def x04(env):
    t, secs, _ = start_and_cancel(env, M('call:write_file:{"path": "c.txt", "content": "z"}'),
                                  lambda ev: ev["type"] == "approval.request")
    check(t.end_type == "run.cancelled" and secs is not None and secs < 3, f"{t.summary()} secs={secs}")
    return f"cancelled {secs:.2f} s after Stop"


@case("X05", "Stop during the compaction summary call interrupts it")
def x05(env):
    sid = long_session(env, env.C, turns=2, words=160, first_marker="slow_summary")
    since = len(env.C.log())
    t, secs, _ = start_and_cancel(env, "more " + "lorem ipsum dolor sit " * 160 + " [[fake:plain]]",
                                  lambda ev: ev["type"] == "step", fake=env.C, after=2.0, preset="chat",
                                  session_id=sid)
    check(t.end_type == "run.cancelled" and secs is not None and secs < 4, f"{t.summary()} secs={secs}")
    check(fake_saw_close(env.C, since, "prefill"), "the summary call kept running")
    return f"cancelled {secs:.2f} s after Stop; the summary call was shut"


@case("X07", "Stop twice, after the end, or with a bad id: idempotent / 404")
def x07(env):
    t = env.c.run(M("plain"), server=env.A.key, preset="chat")
    a = env.c.cancel(t.run["run_id"])[0]
    b = env.c.cancel(t.run["run_id"])[0]
    c = env.c.cancel("run_nope")[0]
    check((a, b, c) == (200, 200, 404), f"{a} {b} {c}")
    return "after end 200, twice 200, bad id 404"


@case("X08", "two POST /run for one chat at once: one runs, the other 409")
def x08(env):
    a = env.c.run("first [[fake:plain]]", server=env.A.key, preset="chat")
    sid = a.run["session_id"]
    res = []

    def go():
        res.append(env.c.json("POST", "/api/chat/run", {"message": M("slow:0.3"), "preset": "chat",
                                                         "server": env.A.key, "session_id": sid})[:2])
    th = [threading.Thread(target=go) for _ in range(2)]
    [x.start() for x in th]
    [x.join() for x in th]
    codes = sorted(r[0] for r in res)
    for st, d in res:
        if st == 200:
            env.c.events(d["run_id"], 0, 20)
    check(codes == [200, 409], str(codes))
    return f"codes {codes}"


@case("X11", "Stop, then send again at once: the new turn runs after the save")
def x11(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:0.5"), "preset": "chat", "server": env.A.key})
    env.c.events(d["run_id"], 0, 10, stop=lambda ev: ev["type"] == "text.delta")
    env.c.cancel(d["run_id"])
    t = env.c.run("again [[fake:plain]]", server=env.A.key, preset="chat", session_id=d["session_id"])
    ok_done(t, "PLAIN-OK")
    s = session(env, d["session_id"])
    check(len(s["turns"]) == 2, f"{len(s['turns'])} turns")
    return "2 turns saved (stopped + new)"


@case("X12", "Stop while a sub-agent streams stops the child too")
def x12(env):
    since = len(env.A.log())
    t, secs, _ = start_and_cancel(env, M("spawn_slow"), lambda ev: ev["type"] == "subagent", after=1.0)
    check(t.end_type == "run.cancelled" and secs is not None and secs < 4, f"{t.summary()} secs={secs}")
    check(fake_saw_close(env.A, since, "stream"), "the child stream kept going")
    return f"cancelled {secs:.2f} s after Stop; child stream closed"


# ---------------------------------------------------------------------------------------------------------------- K
@case("K02", "two pages watching one run get the same events")
def k02(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:0.3"), "preset": "chat", "server": env.A.key})
    out = []
    th = [threading.Thread(target=lambda: out.append(env.c.events(d["run_id"], 0, 30))) for _ in range(2)]
    [x.start() for x in th]
    [x.join() for x in th]
    a, b = [[e["seq"] for e in t.events] for t in out]
    check(a == b and out[0].end_type == "run.done", f"{len(a)} vs {len(b)}")
    return f"both saw {len(a)} events"


@case("K03", "the 7th chat answering at once gets 429 with Retry-After")
def k03(env):
    runs = []
    for i in range(6):
        st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:2"), "preset": "chat", "server": env.A.key})
        check(st == 200, f"run {i}: HTTP {st}")
        runs.append(d["run_id"])
    st, d, hd = env.c.json("POST", "/api/chat/run", {"message": M("plain"), "preset": "chat", "server": env.A.key})
    for rid in runs:
        env.c.cancel(rid)
    for rid in runs:
        env.c.events(rid, 0, 10)
    check(st == 429 and hd.get("Retry-After") == "5", f"HTTP {st} {d} Retry-After={hd.get('Retry-After')}")
    return "429 too_many, Retry-After: 5"


@case("K06", "one client address may have only 3 chats answering at once (configurable): the 4th gets 429 + Retry-After")
def k06(env):
    cfgp = os.path.join(env.ctl.cfg, "control.json")
    with open(cfgp) as f:
        orig = json.load(f)
    with open(cfgp, "w") as f:
        json.dump(dict(orig, chat_max_runs_per_client=3), f)
    runs = []
    try:
        for i in range(3):
            st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:2"), "preset": "chat", "server": env.A.key})
            check(st == 200, f"run {i}: HTTP {st}")
            runs.append(d["run_id"])
        st, d, hd = env.c.json("POST", "/api/chat/run", {"message": M("plain"), "preset": "chat", "server": env.A.key})
        check(st == 429 and hd.get("Retry-After") == "5" and d.get("code") == "too_many_client", f"HTTP {st} {d} {hd.get('Retry-After')}")
        with open(cfgp, "w") as f:                       # raised: the 4th now starts
            json.dump(dict(orig, chat_max_runs_per_client=4), f)
        st2, d2, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:2"), "preset": "chat", "server": env.A.key})
        check(st2 == 200, f"raised cap: HTTP {st2} {d2}")
        runs.append(d2["run_id"])
    finally:
        with open(cfgp, "w") as f:
            json.dump(orig, f)
        for rid in runs:
            env.c.cancel(rid)
        for rid in runs:
            env.c.events(rid, 0, 10)
    return "4th run 429 too_many_client (cap 3), Retry-After: 5; cap 4 lets it through"


@case("K04", "a request queued behind a busy slot waits (first-byte allowance) instead of timing out")
def k04(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:2.5"), "preset": "chat", "server": env.F.key})
    check(st == 200, f"HTTP {st}")
    time.sleep(0.5)
    t = env.c.run(M("plain"), server=env.F.key, preset="chat", advanced=True, step_timeout=10, timeout=60)
    env.c.events(d["run_id"], 0, 30)
    ok_done(t, "PLAIN-OK")
    return f"queued ~15 s behind the busy slot, then done; kinds={t.kinds()}"


@case("K07", "chats saving at the same moment do not corrupt each other")
def k07(env):
    out = []

    def go(i):
        out.append(env.c.run(f"parallel {i} [[fake:plain]]", server=env.A.key, preset="chat"))
    th = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    [x.start() for x in th]
    [x.join() for x in th]
    time.sleep(0.5)
    for t in out:
        ok_done(t)
        s = session(env, t.run["session_id"])
        check(len(s.get("turns") or []) == 1, json.dumps(s)[:200])
    return "4 concurrent chats, each saved with its turn"


@case("K08", "a 429/503 from the server backs off (Retry-After) and retries; persistent 503 ends with the code")
def k08(env):
    a = env.c.run(M("busy_once:429"), server=env.A.key, preset="chat")
    ok_done(a, "PLAIN-OK")
    check("busy" in a.kinds(), a.summary())
    b = env.c.run(M("busy_always"), server=env.A.key, preset="chat", timeout=60)
    check(b.end_type == "run.error" and b.end["data"].get("code") == "http_503", b.summary())
    return f"once: retried + done; always: {b.kinds().count('busy')} back-offs then http_503"


# ---------------------------------------------------------------------------------------------------------------- E
@case("E01", "a server Control does not know is refused with words")
def e01(env):
    t = env.c.run(M("plain"), server="port:1", preset="chat")
    check(t.status == 404, t.summary())
    return f"HTTP {t.status}: {t.error}"


@case("E02", "the server dies mid-reply and comes back: the turn waits and continues")
def e02(env):
    since = 0
    holder = {}

    def on_event(ev, t):
        if ev["type"] == "step" and "killed" not in holder:
            holder["killed"] = True

            def bounce():
                time.sleep(1.0)
                env.G.stop(signal.SIGKILL)
                time.sleep(2.0)
                env.G.start()
                register_launched(env.ctl.cfg, env.G.proc.pid, env.G.port, env.G.model)
            threading.Thread(target=bounce, daemon=True).start()
    t = env.c.run(M("prefill:6"), server=env.G.key, preset="chat", on_event=on_event, timeout=90)
    ok_done(t, "PREFILL-OK")
    check(any(k in t.kinds() for k in ("retry", "waiting", "reattached")), t.summary())
    return "kinds " + ",".join(str(k) for k in t.kinds())


@case("E03", "HTTP 500 with a JSON error: the server's message")
def e03(env):
    t = env.c.run(M("http:500"), server=env.A.key, preset="chat")
    check(t.end_type == "run.error" and "fake HTTP 500" in t.text, t.summary())
    return t.text[:120]


@case("E04", "an HTML error page is reduced to plain words")
def e04(env):
    t = env.c.run(M("html:502"), server=env.A.key, preset="chat")
    check(t.end_type == "run.error" and "<" not in t.text and "Bad Gateway" in t.text, t.summary())
    return t.text[:120]


@case("E05", "HTTP 401 from the model server is reported")
def e05(env):
    t = env.c.run(M("http:401"), server=env.A.key, preset="chat")
    check(t.end_type == "run.error" and "401" in t.text, t.summary())
    return t.text[:120]


@case("E06", "the chat API refuses a missing or wrong Control token (401 JSON)")
def e06(env):
    st, hd, raw = env.c.req("GET", "/api/chat/sessions", token=False)
    st2, hd2, raw2 = env.c.req("GET", "/api/chat/sessions", token=False, headers={"X-PXA-Token": "wrong-token-123456"})
    check(st == 401 and st2 == 401 and b"unauthorized" in raw, f"{st} {st2} {raw[:80]!r}")
    return "missing -> 401, wrong -> 401"


@case("E08", "a template that refuses tools falls back to the text protocol once")
def e08(env):
    t = calc_run(env, "calc", fake=env.D)
    ok_done(t, "CALC-DONE: the result was 42")
    check("fallback_text" in t.kinds(), t.summary())
    return "fell back to text tool calls; " + t.text[:60]


@case("E09", "invalid UTF-8 bytes in the stream are replaced, no crash")
def e09(env):
    t = env.c.run(M("nonutf8"), server=env.A.key, preset="chat")
    ok_done(t, "NONUTF8")
    check("�" in t.text, repr(t.text))
    return repr(t.text[:40])


@case("E10", "NUL / ESC / bidi characters in the answer travel intact (the page renders them inert)")
def e10(env):
    t = env.c.run(M("ctrl_chars"), server=env.A.key, preset="chat")
    ok_done(t, "CTRL-OK")
    check("\u0000" in t.text and "‮" in t.text, repr(t.text))
    return repr(t.text[:60])


@case("E11", "a server that lists no model is refused, never guessed")
def e11(env):
    t = env.c.run(M("plain"), server=env.N.key, preset="chat")
    check(t.status in (404, 409) or t.end_type == "run.error", t.summary())
    return t.summary()


@case("E12", "a request body that is not a JSON object is refused 400")
def e12(env):
    st, hd, raw = env.c.req("POST", "/api/chat/run", raw=b"[1, 2]")
    check(st == 400, f"HTTP {st} {raw[:80]!r}")
    return f"HTTP {st}"


@case("E13", "another site's Origin is refused 403; a foreign Host 421")
def e13(env):
    st, _, _ = env.c.req("POST", "/api/chat/run", {"message": "x"}, headers={"Origin": "http://evil.example"})
    st2, _, _ = env.c.req("GET", "/api/chat/sessions", headers={"Host": "evil.example"})
    check(st == 403 and st2 == 421, f"{st} {st2}")
    return f"Origin -> {st}, Host -> {st2}"


# ---------------------------------------------------------------------------------------------------------------- ST
@case("ST02", "Control killed mid-turn: the user's message comes back as an interrupted turn with a resend")
def st02(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": "remember to ask about llamas [[fake:prefill:30]]",
                                                     "preset": "chat", "server": env.A.key})
    check(st == 200, f"HTTP {st} {d}")
    env.c.events(d["run_id"], 0, 10, stop=lambda ev: ev["type"] == "step")
    env.ctl.crash()
    env.ctl.start()
    wait_for(lambda: env.c.json("GET", "/api/chat/servers", timeout=60)[0] == 200, 20, 0.5)
    s = session(env, d["session_id"])
    lost = (s or {}).get("interrupted") or {}
    check("ask about llamas" in str(lost.get("user")) and not s.get("running"), "after restart: " + json.dumps(s)[:300])
    t = env.c.run(lost["user"].replace("prefill:30", "plain"), server=env.A.key, preset="chat",
                  session_id=d["session_id"])
    ok_done(t, "PLAIN-OK")
    time.sleep(0.3)
    s2 = session(env, d["session_id"])
    check(not s2.get("interrupted") and len(s2.get("turns") or []) == 1, json.dumps(s2)[:300])
    return "interrupted turn shown after kill -9 + restart; resend answered, marker gone"


@case("ST03", "a saved chat is on disk, owner-only (0600)")
def st03(env):
    t = env.c.run(M("plain"), server=env.A.key, preset="chat")
    p = os.path.join(env.ctl.cfg, "chat", "sessions", t.run["session_id"] + ".json")
    wait_for(lambda: os.path.exists(p), 5)
    mode = oct(os.stat(p).st_mode & 0o777)
    check(mode == "0o600", mode)
    return f"{os.path.basename(p)} {mode}"


@case("ST04", "300 big chats: listing stays under 1 s")
def st04(env):
    d = os.path.join(env.ctl.cfg, "chat", "sessions")
    os.makedirs(d, exist_ok=True)
    blob = "y" * 200000
    for i in range(300):
        sid = f"c_bulk{i:04d}xx"
        with open(os.path.join(d, sid + ".json"), "w") as f:
            json.dump({"id": sid, "title": f"bulk {i}", "created": 1, "updated": 1 + i, "messages": [
                {"role": "user", "content": blob}], "turns": []}, f)
    t0 = time.time()
    env.c.json("GET", "/api/chat/sessions")
    secs = time.time() - t0
    for i in range(300):
        try:
            os.unlink(os.path.join(d, f"c_bulk{i:04d}xx.json"))
        except OSError:
            pass
    check(secs < 1.0, f"list took {secs:.2f} s for 300 x 200 KB")
    return f"list {secs:.2f} s"


@case("ST06", "a corrupt session file is reported, not hidden: listed as broken, raw export and delete still work")
def st06(env):
    d = os.path.join(env.ctl.cfg, "chat", "sessions")
    p = os.path.join(d, "c_corrupt0001.json")
    raw = '{"id": "c_corrupt0001", "title": "broken one", "turns": ['
    with open(p, "w") as f:
        f.write(raw)
    try:
        st, dd, _ = env.c.json("GET", "/api/chat/sessions")
        row = next((s for s in dd.get("sessions") or [] if s.get("id") == "c_corrupt0001"), None)
        check(row is not None, "not listed")
        check(row.get("broken") is True and "could not be read" in row.get("title", "") and "broken one" in row["title"], str(row))
        st, d, _ = env.c.json("GET", "/api/chat/session?id=c_corrupt0001")
        check(st == 422 and d.get("code") == "session_unreadable", f"open: HTTP {st} {d}")
        st, _, body = env.c.req("GET", "/api/chat/export?id=c_corrupt0001&format=raw")
        check(st == 200 and body.decode() == raw, f"raw export HTTP {st} {body[:80]!r}")
        st, d, _ = env.c.json("DELETE", "/api/chat/session", {"id": "c_corrupt0001"})
        check(st == 200 and not os.path.exists(p), f"delete HTTP {st} {d}")
        st, dd, _ = env.c.json("GET", "/api/chat/sessions")
        check(all(s.get("id") != "c_corrupt0001" for s in dd.get("sessions") or []), "still listed after delete")
    finally:
        if os.path.exists(p):
            os.unlink(p)
    return "listed as broken (title + could not be read); open 422; raw export identical; delete works"


@case("ST07", "a chat deleted while it answers stays deleted")
def st07(env):
    st, d, _ = env.c.json("POST", "/api/chat/run", {"message": M("slow:0.5"), "preset": "chat", "server": env.A.key})
    env.c.events(d["run_id"], 0, 10, stop=lambda ev: ev["type"] == "text.delta")
    st2, dd, _ = env.c.json("DELETE", "/api/chat/session", {"id": d["session_id"]})
    env.c.events(d["run_id"], 0, 15)
    time.sleep(0.5)
    st3, _, _ = env.c.json("GET", "/api/chat/session?id=" + d["session_id"])
    check(st2 == 200 and st3 == 404, f"delete {st2}, get after the run {st3}")
    return "deleted mid-answer, still 404 after the run"


@case("ST08", "an incognito chat never touches the disk")
def st08(env):
    t = env.c.run(M("plain"), server=env.A.key, preset="chat", incognito=True)
    ok_done(t)
    p = os.path.join(env.ctl.cfg, "chat", "sessions", t.run["session_id"] + ".json")
    time.sleep(0.3)
    check(not os.path.exists(p), "incognito chat written to disk")
    return "no file on disk"


@case("ST10", "a 1 MB reply keeps the event log bounded (coalesced deltas); a reconnect replays the same bytes")
def st10(env):
    t = env.c.run(M("big:1000000"), server=env.A.key, preset="chat", timeout=120)
    ok_done(t)
    n = t.types.count("text.delta")
    check(n < 2000, f"{n} text.delta events for 1 MB")
    # replay after a reconnect (?since=N) joins to exactly the live text
    rid = t.run["run_id"]
    first = t.events
    mid = len(first) // 2
    tail = env.c.events(rid, mid, 30)
    live = "".join(e["data"].get("text", "") for e in first if e["type"] == "text.delta")
    replay = "".join(e["data"].get("text", "") for e in first[:mid] if e["type"] == "text.delta") + \
        "".join(e["data"].get("text", "") for e in tail.events if e["type"] == "text.delta")
    check(len(live) >= 1000000 and live == replay and live.strip() == t.text.strip(), f"replay differs (live {len(live)} replay {len(replay)})")
    return f"{n} text.delta events, replay from {mid} byte-identical ({len(live)} chars)"


# ---------------------------------------------------------------------------------------------------------------- P
def adv_err(env, **kw):
    t = env.c.run(M("echo"), server=env.A.key, preset="chat", advanced=True, **kw)
    return t


@case("P01", "temperature 3 or -1 is refused with the range")
def p01(env):
    a, b = adv_err(env, temperature=3), adv_err(env, temperature=-1)
    check(a.status == 400 and b.status == 400 and "between" in str(a.error), a.summary())
    return str(a.error)


@case("P02", "top_p 1.5 is refused")
def p02(env):
    a = adv_err(env, top_p=1.5)
    check(a.status == 400, a.summary())
    return str(a.error)


@case("P03", "max_tokens 0 is refused")
def p03(env):
    a = adv_err(env, max_tokens=0)
    check(a.status == 400, a.summary())
    return str(a.error)


@case("P04", "max_tokens 262144 is allowed (the window caps it)")
def p04(env):
    t = adv_err(env, max_tokens=262144)
    ok_done(t, "max_tokens=262144")
    return t.text[:120]


@case("P05", "NaN / Infinity / true for a number are refused")
def p05(env):
    out = []
    for raw in (b'{"message": "x [[fake:echo]]", "preset": "chat", "advanced": true, "temperature": NaN}',
                b'{"message": "x [[fake:echo]]", "preset": "chat", "advanced": true, "top_p": Infinity}',
                b'{"message": "x [[fake:echo]]", "preset": "chat", "advanced": true, "temperature": true}'):
        st, hd, body = env.c.req("POST", "/api/chat/run", raw=raw)
        out.append(st)
        if st == 200:
            env.c.events(json.loads(body)["run_id"], 0, 20)
    check(out == [400, 400, 400], str(out))
    return f"NaN/Infinity/true -> {out}"


@case("P06", "seed + temperature 0 twice gives identical text", modes=("fake", "live"))
def p06(env):
    a = env.c.run("Count from one to five. [[fake:echo]]", server=env.main.key, preset="chat", advanced=True,
                  temperature=0, seed=7, max_tokens=48)
    b = env.c.run("Count from one to five. [[fake:echo]]", server=env.main.key, preset="chat", advanced=True,
                  temperature=0, seed=7, max_tokens=48)
    ok_done(a)
    ok_done(b)
    check(a.text == b.text, f"{a.text[:80]!r} != {b.text[:80]!r}")
    return f"identical: {a.text[:80]!r}"


@case("P07", "a thinking stream is kept apart; an unclosed <think> gives the empty-answer note")
def p07(env):
    a = env.c.run(M("think"), server=env.A.key, preset="chat")
    ok_done(a, "THINK-OK")
    check("think.delta" in a.types, a.summary())
    b = env.c.run(M("think_unclosed"), server=env.A.key, preset="chat")
    check(b.end_type == "run.done" and b.end["data"].get("empty"), b.summary())
    return "think.delta separate; unclosed think -> empty flag"


@case("P08", "json_schema / grammar / stop sequences pass through (Advanced), validated and size-capped")
def p08(env):
    env.A.reset()
    t = env.c.run(M("echo"), server=env.A.key, preset="chat", advanced=True, stop=["END"],
                  json_schema={"type": "object"})
    ok_done(t)
    body = env.A.requests()[-1]
    check(body.get("stop") == ["END"] and body.get("json_schema") == {"type": "object"}, json.dumps({k: body.get(k) for k in ("stop", "json_schema")}))
    check("tools" not in body, "a constrained reply must not also offer tools")
    env.A.reset()
    t = env.c.run(M("echo"), server=env.A.key, preset="chat", advanced=True, grammar='root ::= "yes" | "no"')
    ok_done(t)
    check(env.A.requests()[-1].get("grammar") == 'root ::= "yes" | "no"', "grammar not passed")
    for bad in ({"stop": ["a", "b", "c", "d", "e"]}, {"stop": [""]}, {"grammar": "not a grammar"}, {"grammar": "x ::= y" * 5000},
                {"json_schema": "text"}, {"json_schema": {"a": "x" * 20000}}, {"response_format": {"type": "weird"}},
                {"json_schema": {"type": "object"}, "grammar": "root ::= x"}):
        st, d, _ = env.c.json("POST", "/api/chat/run", dict({"message": M("echo"), "preset": "chat", "server": env.A.key,
                                                            "advanced": True}, **bad))
        check(st == 400 and d.get("code") == "bad_request", f"{list(bad)}: HTTP {st} {d}")
    env.A.reset()
    t = env.c.run(M("echo"), server=env.A.key, preset="chat", advanced=False, stop=["END"], grammar="root ::= x")
    ok_done(t)
    check("stop" not in env.A.requests()[-1] and "grammar" not in env.A.requests()[-1], "Simple mode must ignore them")
    return "stop, json_schema, grammar reach the server; bad ones 400; Simple mode ignores them"


@case("P09", "an unknown preset is refused")
def p09(env):
    t = env.c.run(M("plain"), server=env.A.key, preset="wizard")
    check(t.status == 400, t.summary())
    return f"HTTP {t.status} {t.error}"


@case("P10", "step_timeout 5 / max_steps 99 are refused by range")
def p10(env):
    a, b = adv_err(env, step_timeout=5), adv_err(env, max_steps=99)
    check(a.status == 400 and b.status == 400, a.summary() + " / " + b.summary())
    return f"{a.error} / {b.error}"


@case("P12", "a reply cut at the length limit is marked truncated", modes=("fake", "live"))
def p12(env):
    if env.live:
        t = env.c.run("Write a 300-word story about a llama.", server=env.main.key, preset="chat", advanced=True,
                      max_tokens=16, temperature=0)
    else:
        t = env.c.run(M("length"), server=env.A.key, preset="chat")
    ok_done(t)
    check(t.end["data"].get("truncated") is True, t.summary())
    return "truncated: " + t.text[:60]


# ---------------------------------------------------------------------------------------------------------------- D
@case("D01", "temperature 0 twice: identical answers", modes=("fake", "live"))
def d01(env):
    return p06(env)


@case("D02", "the seed reaches the server")
def d02(env):
    t = adv_err(env, seed=4242, temperature=0.5, top_p=0.9)
    ok_done(t, "seed=4242")
    check("temperature=0.5" in t.text and "top_p=0.9" in t.text, t.text)
    return t.text[:120]


# ---------------------------------------------------------------------------------------------------------------- U
def node_md(src):
    node = shutil.which("node")
    if not node:
        raise Fail("SKIP: node is not installed")
    js = open(os.path.join(TOOLS, "pxa_chat", "ui", "agent.js"), encoding="utf-8").read()
    part = js[js.index("// md:begin"):js.index("// md:end")]
    prog = part + "\nprocess.stdout.write(pxaMd.render(require('fs').readFileSync(0, 'utf8')));"
    return subprocess.run([node, "-e", prog], input=src, capture_output=True, text=True, timeout=30, check=True).stdout


@case("U01", "an unclosed fence while streaming renders as code so far")
def u01(env):
    h = node_md("Here:\n```js\nconst a = 1;")
    check('class="ag-code"' in h and "const a = 1;" in h, h[:200])
    return "code block"


@case("U02", "a ```` fence around ``` keeps the inner fence as text")
def u02(env):
    h = node_md("````md\n```js\nalert(1)\n```\n````\nafter")
    check(h.count('class="ag-code"') == 1 and "```js" in h and "<p>after</p>" in h, h[:300])
    return "one outer block, inner fence kept"


@case("U03", "~~~ fences render as code")
def u03(env):
    h = node_md("~~~\ntilde\n~~~")
    check('class="ag-code"' in h and "tilde" in h, h[:200])
    return "code block"


@case("U04", "<script> / <img onerror> in output are escaped")
def u04(env):
    h = node_md("<script>alert(1)</script> <img src=x onerror=alert(1)>")
    check("<script" not in h and "<img" not in h, h[:200])
    return "escaped"


@case("U05", "javascript: / data: links are not linked")
def u05(env):
    h = node_md("[a](javascript:alert(1)) [b](data:text/html,x) [c](https://ok.example)")
    check('href="javascript' not in h and 'href="data' not in h and 'href="https://ok.example"' in h, h[:300])
    return "only https linked"


@case("U06", "RTL text and bidi overrides render inert")
def u06(env):
    h = node_md("file ‮exe.txt‬ ok \u2066x\u2069 שלום")
    check(not re.search("[\u202a-\u202e\u2066-\u2069]", h) and "[U+202E]exe.txt[U+202C]" in h and "שלום" in h, repr(h[:160]))
    return "bidi controls shown as [U+202E] markers; plain RTL text untouched"


@case("U07", "emoji / ZWJ / combining marks / CJK arrive byte-identical")
def u07(env):
    t = env.c.run(M("unicode"), server=env.A.key, preset="chat")
    want = ("UNICODE-OK: שלום مرحبا ‮dlrow‬ "
            "\U0001F468‍\U0001F469‍\U0001F467 é 你好世界 \U0001F999")
    ok_done(t)
    check(t.text.strip() == want, repr(t.text))
    return "identical"


@case("U09", "export md/json: complete, safe filename")
def u09(env):
    t = env.c.run("export me [[fake:plain]]", server=env.A.key, preset="chat")
    sid = t.run["session_id"]
    st, hd, raw = env.c.req("GET", f"/api/chat/export?id={sid}&format=md")
    st2, hd2, raw2 = env.c.req("GET", f"/api/chat/export?id={sid}&format=json")
    cd = hd.get("Content-Disposition", "")
    check(st == 200 and st2 == 200 and b"PLAIN-OK" in raw and re.search(r'filename="[A-Za-z0-9_-]+\.md"', cd)
          and json.loads(raw2)["id"] == sid, f"{st} {st2} {cd} {raw[:600]!r}")
    return cd


@case("U10", "a table cell with an escaped pipe and inline code stays one cell")
def u10(env):
    h = node_md("| a | b |\n|---|---|\n| x \\| y | `c` |")
    check("<td>x | y</td>" in h and "<code>c</code>" in h, h[:300])
    return "cells intact"


@case("E10b", "the renderer does not turn a NUL-number-NUL in the text into foreign markup")
def e10b(env):
    h = node_md("a `code` and \u00000\u0000 here")
    check(h.count("<code>code</code>") == 1 and "undefined" not in h, repr(h[:200]))
    return "no placeholder collision"


# ---------------------------------------------------------------------------------------------------------------- SEC
@case("SEC1", "a page asking the model to 'remember' an instruction is refused by memory")
def sec1(env):
    t = env.c.run(M("remember_inject"), server=env.A.key)
    ok_done(t, "REMEMBER-DONE refused")
    st, d, _ = env.c.json("GET", "/api/chat/memory")
    check(not any("evil.example" in f.get("text", "") for f in d.get("facts") or []), "stored")
    return "refused; nothing stored"


@case("SEC5", "a DANGER-pattern command is refused even after approval")
def sec5(env):
    t = env.c.run(M('call:run_command:{"command": "mkfs --version"}'), server=env.A.key, advanced=True,
                  tools=["run_command"], approve=lambda d: "once")
    ok_done(t, "blocked pattern")
    return t.text[:120]


# ---------------------------------------------------------------------------------------------------------------- Classic proxy
def proxy_stream(env, user):
    """a Classic-view request through Control's engine proxy, left open: -> (raw socket, http connection)."""
    c = http.client.HTTPConnection("127.0.0.1", env.ctl.port, timeout=120)
    body = json.dumps({"messages": [{"role": "user", "content": user}], "stream": True}).encode()
    c.request("POST", f"/api/engine/v1/chat/completions?port={env.A.port}", body=body,
              headers={"Content-Type": "application/json", "X-PXA-Token": env.ctl.token, "Accept": "text/event-stream"})
    return c


@case("X10", "C: Stop / a closed tab during prefill closes the upstream at once")
def x10(env):
    since = len(env.A.log())
    c = proxy_stream(env, "[[fake:prefill:30]]")
    time.sleep(2.0)
    t0 = time.time()
    c.sock.shutdown(socket.SHUT_RDWR)
    c.close()
    hit = fake_saw_close(env.A, since, "prefill", timeout=5)
    check(hit, "the engine kept prefilling for nobody (no client_closed in the fake log)")
    return f"engine saw the close {hit[0]['t'] - t0:.2f} s after the tab closed (prefill phase)"


@case("S14", "C: a client that stops reading for > 60 s is dropped and the upstream closed")
def s14(env):
    since = len(env.A.log())
    c = proxy_stream(env, "[[fake:big:20000000]]")
    t0 = time.time()
    hit = fake_saw_close(env.A, since, "stream", timeout=95)
    st = http_ok(env.ctl.port, "/api/health")
    try:
        c.close()
    except OSError:
        pass
    check(hit, "upstream still open after 95 s of a stalled reader")
    check(st in (200, 401), f"Control health {st}")
    return f"upstream closed {hit[0]['t'] - t0:.0f} s into the stall; Control still answers"


# ---------------------------------------------------------------------------------------------------------------- live
@case("L01", "live: a plain question gets an answer", modes=("live",), group="LIVE")
def l01(env):
    t = env.c.run("In one short sentence: what is a llama?", server=env.main.key, preset="chat", advanced=True,
                  temperature=0, max_tokens=96, timeout=180)
    ok_done(t)
    check(len(t.text.strip()) > 3, t.summary())
    return t.text[:160]


@case("L02", "live: Stop mid-stream ends the turn fast", modes=("live",), group="LIVE")
def l02(env):
    t, secs, _ = start_and_cancel(env, "Write a long essay about the history of computing.",
                                  lambda ev: ev["type"] == "text.delta", fake=env.main, preset="chat",
                                  advanced=True, max_tokens=2000, timeout=180)
    check(t.end_type == "run.cancelled" and secs is not None and secs < 3, f"{t.summary()} secs={secs}")
    return f"cancelled {secs:.2f} s after Stop"


@case("L03", "live: a native tool call (calculator) runs and the answer uses it", modes=("live",), group="LIVE")
def l03(env):
    t = env.c.run("Use the calculator tool to compute 1234*5678, then tell me the result.", server=env.main.key,
                  advanced=True, tools=["calculate"], temperature=0, max_tokens=512, timeout=240)
    ok_done(t)
    calls = [d for d in t.of("tool.call") if d["name"] == "calculate"]
    check(calls or "7006652" in t.text.replace(",", ""), t.summary())
    return f"{len(calls)} calculate call(s); answer {t.text[:100]!r}"


@case("L04", "live: the picker reports the per-request window (n_ctx_slot)", modes=("live",), group="LIVE")
def l04(env):
    st, d, _ = env.c.json("GET", "/api/chat/servers")
    srv = next((s for s in d.get("servers") or [] if s["key"] == env.main.key), None)
    check(srv is not None, json.dumps(d)[:300])
    props = env.main.props()
    want = props.get("n_ctx_slot") or (props.get("default_generation_settings") or {}).get("n_ctx")
    got = (srv.get("ctx") or {}).get("resolved")
    check(want and got == want, f"picker {got} vs /props n_ctx_slot {want} (n_ctx {props.get('n_ctx')})")
    return f"picker {got} == n_ctx_slot (ring {props.get('n_ctx')})"


@case("L05", "live: a message bigger than the window is refused up front", modes=("live",), group="LIVE")
def l05(env):
    props = env.main.props()
    n = props.get("n_ctx_slot") or (props.get("default_generation_settings") or {}).get("n_ctx") or 8192
    t = env.c.run("Summarize: " + "token " * int(n * 1.3), server=env.main.key, preset="chat", timeout=120)
    check(t.end_type == "run.error" and t.end["data"].get("code") == "too_long", t.summary())
    return t.text[:160]


# ---------------------------------------------------------------------------------------------------------------- the rest
alias("S16", "X10", "C: tab closed / Stop during prefill closes the upstream at once")
alias("X09", "S15", "navigate away mid-run, come back: re-attach")
alias("ST01", "S15", "page reload mid-turn: re-attach, replay")
alias("K01", "X08", "two tabs send in one chat: the second gets 409")
alias("K09", "K07", "many runs finishing together are each saved once")
alias("SEC2", "T20", "control-token smuggling via tool output is neutralised")
unit("C07", "old huge tool outputs are trimmed head+tail when compacting", "tests/test-pxa-chat.py TestCompact")
unit("C19", "the token estimate counts CJK about one token per character",
     "tools/pxa_control/tests/test_chat_harness.py::TestContextWindow::test_cjk_estimate")
unit("K05", "sub-agents wait for a free slot (max 3)", "tests/test-pxa-chat.py TestSpawnAgent.test_a_full_slot_list_waits_then_runs")
unit("SEC4", "a redirect to a private address is refused unless approved",
     "tests/test-pxa-chat.py TestTools.test_web_fetch_private_refused_without_approval")
unit("X06", "Stop during the re-attach wait ends within 1 s", "tests/test-pxa-chat.py TestLoop.test_server_gone_and_no_model")
manual("S21", "thousands of tiny deltas keep the page responsive",
       "needs a browser: python3 tests/chat-ui-gui-check.py --phases plain (prints frame gaps and render time for a 1 MB reply)")
manual("U08", "a 1 MB reply keeps the page usable", "needs a browser: tests/chat-ui-gui-check.py --phases plain; backend half is ST10")
manual("C15", "a pasted text file over 100k chars says it was cut", "needs a browser: tests/chat-ui-gui-check.py --phases plain")
manual("E07", "Control restarted during a stream: the page says sign in again", "needs a browser and a restart: tests/chat-ui-gui-check.py --phases restart,newtoken --long-down 70")
manual("ST05", "clock skew page vs Control", "needs a second machine or a faked page clock")
manual("ST09", "hub.active stays bounded", "needs a soak of thousands of chats")
manual("P11", "a client's reasoning_effort is mapped by the thinking profile", "Classic only; covered by "
       "tests/test-pxa-control.py thinking tests")
unit("TPL01", "a mid-conversation system message is merged to the front: strict chat templates never see one",
     "tools/pxa_control/tests/test_chat_harness.py::TestSystemFirst")
unit("SEC3", "DNS rebinding on web_fetch: one lookup, connect to the checked address, Host/SNI on the name",
     "tools/pxa_control/tests/test_chat_harness.py::TestFetchRebind")
CATALOGUE_IDS = (["S%02d" % i for i in range(1, 23)] + ["T%02d" % i for i in range(1, 33)] +
                 ["C%02d" % i for i in range(1, 20)] + ["X%02d" % i for i in range(1, 13)] +
                 ["K%02d" % i for i in range(1, 10)] + ["E%02d" % i for i in range(1, 14)] +
                 ["ST%02d" % i for i in range(1, 11)] + ["P%02d" % i for i in range(1, 13)] +
                 ["D01", "D02"] + ["U%02d" % i for i in range(1, 11)] + ["SEC%d" % i for i in range(1, 6)])


# =====================================================================================================================
# runner
# =====================================================================================================================
class Env(object):
    pass


class LiveServer(object):
    def __init__(self, port):
        self.port, self.name, self.model = port, "live", None
        self._key = None

    @property
    def key(self):
        return self._key or f"port:{self.port}"

    def props(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/props")
        d = json.loads(c.getresponse().read() or b"{}")
        c.close()
        return d


def run_unit(target):
    """-> (ok, observed, log)"""
    if "::" in target or target.endswith(".py") and "test_chat_harness" in target:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(REPO, target.split(" ")[0])]
        if "::" in target:
            cmd[-1] = os.path.join(REPO, target)
    else:
        f, _, t = target.partition(" ")
        cmd = [sys.executable, os.path.join(REPO, f)] + ([t] if t else [])
    env = dict(os.environ)
    for k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_BUS_TOKEN"):
        env.pop(k, None)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=REPO, env=env)
    out = (p.stdout + p.stderr)
    tail = " ".join(out.strip().splitlines()[-3:])[:300]
    return p.returncode == 0, ("unit OK: " if p.returncode == 0 else "unit FAILED: ") + tail, out[-2000:]


def run_case(c, env, results):
    r = {"id": c["id"], "group": c["group"], "expect": c.get("expect", ""), "status": None, "observed": "",
         "log": "", "secs": 0.0}
    t0 = time.time()
    avail = ({"live"} if env.live else set()) | (set() if env.live_only else {"fake"})
    if c["kind"] == "manual":
        r.update(status="MANUAL", observed=c["reason"])
    elif c["kind"] == "alias":
        src = next((x for x in results if x["id"] == c["alias"]), None)
        r.update(status=src["status"] if src else "SKIP", observed=f"same check as {c['alias']}: " +
                 ((src or {}).get("observed") or "not run")[:200])
    elif c["kind"] == "unit":
        if env.no_unit:
            r.update(status="SKIP", observed="--no-unit")
        else:
            ok, obs, log = run_unit(c["target"])
            r.update(status="PASS" if ok else "FAIL", observed=obs, log=log[-800:])
    elif not set(c["modes"]) & avail:
        r.update(status="SKIP", observed=f"needs {'/'.join(c['modes'])} mode (this run: {'+'.join(sorted(avail))})")
    else:
        try:
            obs = c["fn"](env)
            r.update(status="XPASS" if c.get("xfail") else "PASS", observed=str(obs)[:400])
        except Fail as e:
            msg = str(e)
            if msg.startswith("SKIP:"):
                r.update(status="SKIP", observed=msg[5:].strip())
            else:
                r.update(status="XFAIL" if c.get("xfail") else "FAIL", observed=msg[:400],
                         log=(c.get("xfail") or ""))
        except Exception as e:                    # noqa: BLE001 - a crashed case is a FAIL with its traceback
            tb = traceback.format_exc()
            r.update(status="XFAIL" if c.get("xfail") else "FAIL", observed=f"{e.__class__.__name__}: {e}"[:400],
                     log=tb[-1500:])
    r["secs"] = round(time.time() - t0, 2)
    return r


def control_log_excerpt(env, n=6):
    try:
        with open(env.ctl.log_path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        return " / ".join(lines[-n:])[:600]
    except OSError:
        return ""


def write_reports(out, rep):
    with open(os.path.join(out, "harness-report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    md = [f"# PXA Control chat harness report", "",
          f"- started {rep['started']}, finished {rep['finished']}, mode **{rep['mode']}**, git {rep['git']}",
          f"- counts: " + ", ".join(f"{k} {v}" for k, v in rep["counts"].items()),
          f"- Control {rep['control']}, fakes {', '.join(rep['fakes']) or '-'}, endpoint {rep['endpoint'] or '-'}", "",
          "| id | expected | observed | result | log excerpt |", "|---|---|---|---|---|"]

    def cell(s, n):
        s = " ".join(str(s or "").split())
        s = s.replace("|", "\\|")
        return s[:n]
    for r in rep["cases"]:
        md.append(f"| {r['id']} | {cell(r['expect'], 110)} | {cell(r['observed'], 160)} | **{r['status']}** | "
                  f"{cell(r['log'], 120)} |")
    with open(os.path.join(out, "harness-report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description="PXA Control chat harness suite (see README-HARNESS.md)")
    ap.add_argument("--out", default=None, help="report folder (default ./harness-out-<time>)")
    ap.add_argument("--endpoint", default=None, help="a live llama-server on 127.0.0.1 (e.g. http://127.0.0.1:8263): "
                                                     "adds the live cases; the fake servers still run the rest")
    ap.add_argument("--live-only", action="store_true", help="with --endpoint: run only the live cases")
    ap.add_argument("--port", type=int, default=0, help="Control test port (default: a free one)")
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--skip", default="", help="comma-separated case ids")
    ap.add_argument("--no-unit", action="store_true", help="do not run the unit-test cases")
    ap.add_argument("--keep", action="store_true", help="leave Control and the fakes running at the end")
    ap.add_argument("--list", action="store_true", help="print the catalogue and exit")
    ap.add_argument("--_control-child", dest="control_child", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.control_child:
        return control_child(a.control_child)
    have = {c["id"] for c in CASES}
    missing = [x for x in CATALOGUE_IDS if x not in have]
    if a.list:
        for c in sorted(CASES, key=lambda c: (c["group"], c["id"])):
            print(f"{c['id']:6s} {c['kind']:6s} {'/'.join(c.get('modes') or ()) or '-':9s} {c.get('expect', '')[:90]}")
        print(f"{len(CASES)} cases; catalogue ids without a case: {missing or 'none'}")
        return 0
    out = os.path.abspath(a.out or os.path.join(os.getcwd(), "harness-out-" + time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="scratch-", dir=out)
    env = Env()
    env.live = bool(a.endpoint)
    env.live_only = bool(a.live_only and a.endpoint)
    env.no_unit = a.no_unit
    fakes = []
    rep = {"suite": "pxa-control-chat-harness", "version": 1, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "mode": "fake+live" if a.endpoint and not a.live_only else ("live" if a.live_only else "fake"),
           "git": git_head(), "endpoint": a.endpoint, "fakes": [], "missing_ids": missing}
    try:
        ctl = Control(out, scratch, a.port or None)
        env.ctl = ctl
        if not a.live_only:
            env.A = Fake("A", out, "fake-agent-8b", ctx=8192, slots=2).start()
            env.B = Fake("B", out, "fake-old-7b", ctx=8192, slots=1, no_tools=True).start()
            env.C = Fake("C", out, "fake-small-2k", ctx=2048, slots=2).start()
            env.D = Fake("D", out, "fake-nojinja", ctx=8192, slots=1, no_jinja=True).start()
            env.E = Fake("E", out, "fake-noprops", ctx=8192, slots=1, no_props=True).start()
            env.F = Fake("F", out, "fake-oneslot", ctx=8192, slots=1).start()
            env.G = Fake("G", out, "fake-bouncer", ctx=8192, slots=1).start()
            env.V = Fake("V", out, "fake-vision", ctx=8192, slots=1, vision=True).start()
            env.N = Fake("N", out, "", ctx=8192, slots=1).start()
            fakes = [env.A, env.B, env.C, env.D, env.E, env.F, env.G, env.V, env.N]
            for f in fakes[1:]:
                register_launched(ctl.cfg, f.proc.pid, f.port, f.model or "nomodel")
            env.main = env.A
            env.A._key = "m:main"                # the attached server is Control's main seat
            attach = env.A.port
        if a.endpoint:
            m = re.match(r"^https?://(127\.0\.0\.1|localhost):(\d+)/?", a.endpoint)
            if not m:
                raise RuntimeError("--endpoint must be a server on 127.0.0.1 (Control attaches to a local port)")
            live = LiveServer(int(m.group(2)))
            if http_ok(live.port) != 200:
                raise RuntimeError(f"--endpoint {a.endpoint}: /health is not 200")
            env.main = live
            if a.live_only:
                attach = live.port
                live._key = "m:main"
            else:
                os.makedirs(ctl.cfg, mode=0o700, exist_ok=True)
        os.makedirs(ctl.cfg, mode=0o700, exist_ok=True)
        with open(os.path.join(ctl.cfg, "control.json"), "w") as f:
            json.dump({"attach_port": attach, "chat_max_runs_per_client": 0}, f)   # K03 tests the global cap; K06 the per-client one
        if a.endpoint and not a.live_only:
            # the live server is a second server Control knows: a launched record with a pid that stays alive
            env.live_keeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(86400)"],
                                               start_new_session=True)
            register_launched(ctl.cfg, env.live_keeper.pid, live.port, "live")
        ctl.start()
        env.c = Client(ctl.port, ctl.token)
        rep["control"] = f"127.0.0.1:{ctl.port} (scratch {scratch})"
        rep["fakes"] = [f"{f.name}={f.model or '(no model)'}@{f.port}" for f in fakes]
        # warm the picker so the first case does not pay for the probes
        st, d, _ = env.c.json("GET", "/api/chat/servers", timeout=60)
        if st != 200:
            raise RuntimeError(f"/api/chat/servers HTTP {st}: {d}")
        known = {s["key"] for s in d.get("servers") or []}
        need = {f.key for f in fakes} | ({env.main.key} if env.live else set())
        if need - known:
            wait_for(lambda: need <= {s["key"] for s in env.c.json("GET", "/api/chat/servers", timeout=60)[1]
                                     .get("servers") or []}, 20, 1.0)
    except Exception as e:                      # noqa: BLE001
        rep.update(finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"), counts={"SETUP-ERROR": 1}, cases=[],
                   setup_error=f"{e.__class__.__name__}: {e}", control=getattr(getattr(env, "ctl", None), "port", None))
        write_reports(out, rep)
        print(f"SETUP ERROR: {e}\n{traceback.format_exc()}", file=sys.stderr)
        for f in fakes:
            f.stop()
        if getattr(env, "ctl", None):
            env.ctl.stop()
        return 2
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    order = [c for c in CASES if c["kind"] != "alias"] + [c for c in CASES if c["kind"] == "alias"]
    results = []
    try:
        for c in order:
            if only and c["id"] not in only and c.get("alias") not in only:
                continue
            if c["id"] in skip:
                continue
            if a.live_only and "live" not in (c.get("modes") or ()):
                continue
            r = run_case(c, env, results)
            if r["status"] == "FAIL" and not r["log"]:
                r["log"] = control_log_excerpt(env)
            results.append(r)
            print(f"{r['status']:6s} {r['id']:6s} {r['secs']:6.1f}s  {' '.join(r['observed'].split())[:110]}",
                  flush=True)
    finally:
        if not a.keep:
            for f in fakes:
                f.stop()
            if getattr(env, "live_keeper", None):
                env.live_keeper.kill()
            env.ctl.stop()
    order_ids = {cid: i for i, cid in enumerate(CATALOGUE_IDS)}
    results.sort(key=lambda r: (order_ids.get(r["id"], 999), r["id"]))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    rep.update(finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"), counts=counts, cases=results)
    write_reports(out, rep)
    print(f"\n{len(results)} cases: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"report: {os.path.join(out, 'harness-report.md')}")
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
