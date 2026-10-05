"""A complete, hermetic PXA Control + Encode environment for the headless-browser check (tests/encode-gui-check.py) and for ad-hoc
poking: fake GPUs, a fake engine folder, a fake Hugging Face server + CDN, a fake licence server, empty encoder / config / data
folders, and PXA Control itself running in this process. A tiny side channel lets the browser script act on the environment
(restart Control the way a crash or a reboot would, change what the fake encoder does) while the page stays open."""
import http.server
import importlib.util
import json
import os
import shutil
import signal
import socketserver
import sys
import tempfile
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import encode_fakes as F  # noqa: E402


class Env(object):
    def __init__(self, gpus="2x700", port=0, keep=False):
        self.tmp = tempfile.mkdtemp(prefix="pxa-encode-env-")
        self.keep = keep
        t = self.tmp
        self.paths = {k: os.path.join(t, k) for k in ("cfg", "data", "enchome", "models", "work", "engine", "pkg", "state", "logs")}
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)
        env = os.environ
        self.orig_path = env["PATH"]
        env["PXA_LAUNCH_FAKE_GPUS"] = gpus
        env["PXA_CONTROL_CONFIG_DIR"] = self.paths["cfg"]
        env["PXA_LAUNCH_STATE"] = self.paths["state"]
        env["PXA_ENCODE_HOME"] = self.paths["data"]
        env["PXA_ENCODER_HOME"] = self.paths["enchome"]
        env["PXA_CONTROL_DISCOVER"] = "0"
        env["PXA_CONVERT_MODULES"] = ""
        env["PXA_ENGINE_DIR"] = self.paths["engine"]
        env["PXA_PACKAGE_PUBKEY"] = F.PUB_HEX
        env["PATH"] = os.path.join(t, "emptybin") + os.pathsep + "/usr/bin:/bin"       # no pxqe on PATH: the tab starts empty
        os.makedirs(os.path.join(t, "emptybin"), exist_ok=True)
        env["PXA_MODELS_DIR"] = self.paths["models"]
        env.pop("HF_TOKEN", None)
        env.pop("PXA_ENCODER", None)
        env["HOME"] = os.path.join(t, "home")
        os.makedirs(env["HOME"], exist_ok=True)
        F.make_engine_dir(self.paths["engine"])
        self.hf = None
        self.cdn = None
        self.lic = None
        self.board = None
        self.srv = None
        self.app = None
        self.port = port
        self.ctl = None
        self.L = None

    # ---- servers
    def start_fakes(self, repos=None):
        self.hf = F.FakeHF(repos or {"tiny/Tiny-Qwen3": F.tiny_repo(), "tiny/NoDerivs": F.tiny_repo(license="cc-by-nd-4.0"),
                                      "tiny/Gated": dict(F.tiny_repo(), need_token=True, gated=True)})
        self.cdn = F.FakeCDN(self.hf)
        self.hf.cdn = self.cdn
        self.hf.start()
        self.cdn.start()
        self.lic = F.FakeLicence(self.paths["pkg"]).start()
        self.board = F.FakeBoard().start()
        os.environ["PXA_BUG_URL"] = self.board.url + "/v1/report"
        os.environ["HF_ENDPOINT"] = self.hf.url
        os.environ["PXA_LICENCE_URL"] = self.lic.url
        return self

    def start_control(self):
        spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
        self.L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.L)
        import pxa_control as C
        self.C = C
        self.app = C.App(self.L)
        self.srv = C.make_server(self.app, "127.0.0.1", self.port)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def stop_control(self, kill_children=False):
        svc = getattr(self.app, "_enc", None) if self.app is not None else None
        if svc is not None:
            # a crash or a reboot writes nothing on the way out: the old threads must not touch job.json again
            svc._closed = True
            if kill_children:
                for rt in list(svc.rt.values()):
                    p = rt.proc
                    if p is not None and p.poll() is None:
                        try:
                            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                        except OSError:
                            pass
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()
            self.srv = None
        self.app = None

    def restart_control(self, kill_children=True):
        """What a crash (kill_children=False: the child process survives, orphaned) or a reboot (True) does to Control."""
        self.stop_control(kill_children)
        time.sleep(0.3)
        return self.start_control()

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port

    # ---- side channel for the browser script
    def start_ctl(self):
        env = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
                try:
                    out = env.command(u.path.strip("/"), q)
                    code = 200
                except Exception as e:      # noqa: BLE001
                    out, code = {"error": "%s: %s" % (type(e).__name__, e)}, 500
                data = json.dumps(out).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        self.ctl = S(("127.0.0.1", 0), H)
        threading.Thread(target=self.ctl.serve_forever, daemon=True).start()
        return self.ctl.server_address[1]

    def command(self, name, q):
        if name == "restart":
            self.restart_control(kill_children=q.get("kill", "1") == "1")
            return {"ok": True}
        if name == "fake_pro":      # make a fake encoder folder available that Rescan will find (PXA_ENCODER)
            F.make_fake_encoder(os.path.join(self.tmp, "proenc"), "pro", build_id="b-pro-local")
            return {"path": os.path.join(self.tmp, "proenc")}
        if name == "set_encoder":   # change fake.json of an installed encoder: edition dir name + key=value
            d = q.pop("dir")
            kv = {k: (json.loads(v) if v[:1] in "[{\"0123456789tfn" else v) for k, v in q.items()}
            F.set_fake(d, **kv)
            return {"ok": True}
        if name == "engine_delay":
            F.set_engine_delay(self.paths["engine"], float(q["s"]))
            return {"ok": True}
        if name == "licence_mode":
            self.lic.mode = q["m"]
            self.lic.rebuild()
            return {"ok": True}
        if name == "licence_new_build":
            self.lic.free_build = q.get("free", self.lic.free_build)
            self.lic.pro_build = q.get("pro", self.lic.pro_build)
            self.lic.rebuild()
            return {"ok": True}
        if name == "licence_runtime":    # the GPU runtime pack the fake licence server offers: m = ok | none | tamper | bad_signature ..., bulk = bytes of incompressible data, delay = s per 8 KiB
            self.lic.rt_mode = q.get("m", "ok")
            self.lic.rt_bulk = int(q.get("bulk", 0))
            self.lic.rt_delay = float(q.get("delay", 0))
            self.lic.rebuild()
            return {"ok": True}
        if name == "runtime_installed":
            d = os.path.join(self.paths["enchome"], "runtime")
            return {"installed": sorted(os.listdir(d)) if os.path.isdir(d) else [], "root": d}
        if name == "licence_stop":
            self.lic.stop()
            return {"ok": True}
        if name == "requests":
            return {"hf": self.hf.requests, "cdn": self.cdn.requests, "lic": self.lic.requests, "board": self.board.requests}
        if name == "state":
            out = {}
            for root, _d, files in os.walk(self.tmp):
                for f in files:
                    if f in ("fake-state.json",):
                        with open(os.path.join(root, f)) as fh:
                            out[os.path.relpath(os.path.join(root, f), self.tmp)] = json.load(fh)
            return out
        if name == "installed":
            out = []
            for root, _d, files in os.walk(self.paths["enchome"]):
                for f in files:
                    if f == "pxqe":
                        out.append(os.path.relpath(os.path.join(root, f), self.paths["enchome"]))
            return {"installed": sorted(out), "root": self.paths["enchome"]}
        if name == "models":
            return {"files": sorted(os.listdir(self.paths["models"]))}
        if name == "config":
            with open(os.path.join(self.paths["cfg"], "control.json")) as f:
                return json.load(f)
        raise ValueError("unknown command " + name)

    def close(self):
        self.stop_control(True)
        for s in (self.hf, self.cdn, self.lic, self.board):
            if s:
                s.stop()
        if self.ctl:
            self.ctl.shutdown()
        if not self.keep:
            shutil.rmtree(self.tmp, ignore_errors=True)


if __name__ == "__main__":
    e = Env(keep=True).start_fakes().start_control()
    cp = e.start_ctl()
    print("control", e.url, "ctl", cp, "tmp", e.tmp, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        e.close()
