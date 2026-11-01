#!/usr/bin/env python3
"""PXA Control's chat in a tree WITHOUT the in-house agent (tools/anibal is a closed path, so the public source tree
and any package built from it lack it): tools/ is copied the way the export does it (every tools/ path in
scripts/pxa-closed-files.txt left out), then, in a child Python that sees only that copy, pxa_chat imports, Control
registers the chat routes, one chat turn runs to run.done against the stdlib mock OpenAI server, and the agent's
routes answer 503 agent_unavailable only when called. No GPU, no model, no network beyond 127.0.0.1."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_BIG = ("vllm-pxq4", "mtp-train")          # not closed, just large and unrelated to Control's chat

CHILD = r'''
import importlib.util, json, os, sys, threading, time, urllib.error, urllib.request
TOOLS, TESTS = sys.argv[1], sys.argv[2]
sys.path[:0] = [TOOLS, TESTS]
assert importlib.util.find_spec("anibal") is None, "the exported tree still carries the agent package"
import pxa_chat                                   # must import without the agent
from pxa_chat import routes
assert "pxa_chat.vigil_bind" not in sys.modules and "anibal" not in sys.modules, "pxa_chat pulled the agent in at import"
import pxa_chat_mock as MOCK
M = MOCK.serve(0, "mock-qwen-27b", tools=1, ctx=65536, think=1)
cfg = os.environ["PXA_CONTROL_CONFIG_DIR"]
os.makedirs(cfg, exist_ok=True)
with open(os.path.join(cfg, "control.json"), "w") as f:
    json.dump({"attach_port": M.server_address[1]}, f)
spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
import pxa_control as C
assert C.CHAT is not None, C.CHAT_IMPORT_ERROR
app = C.App(L, port=7777)
srv = C.make_server(app, "127.0.0.1", 0)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port, tok = srv.server_address[1], app.token

def http(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"X-PXA-Token": tok}
    if data:
        h["Content-Type"] = "application/json"
    rq = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(rq, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

st, body = http("/api/chat/run", "POST", {"message": "hello there", "preset": "chat", "advanced": False})
assert st == 200, (st, body[:300])
rid = json.loads(body)["run_id"]
rq = urllib.request.Request("http://127.0.0.1:%d/api/chat/events?run=%s&since=0" % (port, rid), headers={"X-PXA-Token": tok})
evs = []
with urllib.request.urlopen(rq, timeout=30) as r:
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:"):
            evs.append(json.loads(line[5:]))
            if evs[-1]["type"] in ("run.done", "run.error"):
                break
types = [e["type"] for e in evs]
assert types and types[-1] == "run.done", types
assert M.last_body and any(m.get("role") == "user" for m in M.last_body["messages"]), "the mock never got the turn"
reply = "".join(str((e.get("data") or {}).get("text") or "") for e in evs if e["type"] == "text.delta")
assert "text.delta" in types and reply.strip(), (types, reply)
st, body = http("/api/chat/vigil/turn", "POST", {"chat_id": "c1", "message": "hi"})
assert st == 503 and json.loads(body).get("code") == "agent_unavailable", (st, body[:300])
st, body = http("/api/chat/vigil/pane/state")
assert st == 503, (st, body[:300])
assert "anibal" not in sys.modules
srv.shutdown()
print("NO-AGENT CHAT OK", len(evs), "events", sorted(set(types)))
'''


def export_tools(dst):
    """tools/ minus every closed tools/ path (the export's own list) and the big unrelated trees"""
    closed = []
    lst = os.path.join(ROOT, "scripts", "pxa-closed-files.txt")
    if os.path.exists(lst):              # absent = the list is closed itself: this IS an exported tree already
        with open(lst) as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#") and p.startswith("tools/"):
                    closed.append(p[len("tools/"):])
    src = os.path.join(ROOT, "tools")

    def ignore(d, names):
        rel = os.path.relpath(d, src).replace(os.sep, "/")
        rel = "" if rel == "." else rel + "/"
        return [n for n in names if n == "__pycache__" or (rel + n) in closed or (not rel and n in SKIP_BIG)]
    shutil.copytree(src, dst, ignore=ignore, symlinks=True)
    return closed if os.path.exists(lst) else None


class TestChatWithoutAgent(unittest.TestCase):
    def test_export_tree_chat_turn(self):
        tmp = tempfile.mkdtemp(prefix="pxa-chat-noagent-")
        try:
            tools = os.path.join(tmp, "tools")
            closed = export_tools(tools)
            if closed is not None:
                self.assertIn("anibal", closed, "tools/anibal must be a closed path")
            self.assertFalse(os.path.exists(os.path.join(tools, "anibal")))
            self.assertTrue(os.path.isfile(os.path.join(tools, "pxa_chat", "routes.py")))
            env = {k: v for k, v in os.environ.items()
                   if k not in ("PYTHONPATH", "DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED",
                                "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN", "PXA_CHAT_SEARCH_URL")}
            env.update({"PXA_LAUNCH_FAKE_GPUS": "2x600", "PXA_CONTROL_CONFIG_DIR": os.path.join(tmp, "cfg"),
                        "PXA_LAUNCH_STATE": os.path.join(tmp, "state"), "PXA_CONTROL_DISCOVER": "0",
                        "PYTHONDONTWRITEBYTECODE": "1"})
            r = subprocess.run([sys.executable, "-I", "-c", CHILD, tools, HERE], env=env, cwd=tmp,
                               capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 0, (r.stdout[-2000:], r.stderr[-4000:]))
            self.assertIn("NO-AGENT CHAT OK", r.stdout)
            print(r.stdout.strip().splitlines()[-1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
