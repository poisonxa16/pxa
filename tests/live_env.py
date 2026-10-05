"""A hermetic PXA Control + live-workload environment for the Live tab (tests/live-gui-check.py, and ad-hoc poking):
a fake `nvidia-smi` on PATH whose numbers follow the load of the fake servers, N fake llama-servers that answer /health, /props,
/slots, /metrics and /pxa/stats the way the PXA engine does while they run a scripted mix of prompts, and PXA Control itself
(in this process) with those servers as its own. No GPU, no model, no network beyond 127.0.0.1.

    python3 tests/live_env.py [--port 7791] [--cards 4]      starts it and prints the address (Ctrl-C stops it)
"""
import http.server
import json
import math
import os
import random
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")

SMI = r'''#!%(py)s
import json, math, os, sys, time
st = {}
try:
    st = json.load(open(%(state)r))
except Exception:
    pass
cards = st.get("cards") or []
args = sys.argv[1:]
q = next((a.split("=", 1)[1] for a in args if a.startswith("--query-gpu=")), None)
qa = next((a.split("=", 1)[1] for a in args if a.startswith("--query-compute-apps=")), None)
t = time.time()
def val(f, i, c):
    load, heat = c.get("load", 0.0), c.get("heat", 0.0)
    wob = math.sin(t / 3.1 + i) * .5 + math.sin(t / 1.3 + 2 * i) * .3
    power = c["idle_w"] + (c["limit_w"] - c["idle_w"]) * min(1.0, load * .9 + wob * .05 * load)
    return {"index": str(i), "name": c["name"], "compute_cap": c["cc"], "memory.total": str(c["mem_total"]),
            "memory.used": str(int(c["mem_base"] + c["mem_total"] * 0.08 * load)), "uuid": "GPU-fake-%%d" %% i,
            "temperature.gpu": "%%d" %% round(34 + 44 * heat + wob),
            "power.draw": "%%.2f" %% power, "power.limit": "%%.2f" %% c["limit_w"], "power.default_limit": "%%.2f" %% c["limit_w"],
            "utilization.gpu": "%%d" %% round(max(0, min(100, 100 * load + wob * 4 * load))),
            "clocks.sm": "%%d" %% (1380 if power < c["limit_w"] * .96 else 1290), "clocks.max.sm": "1530", "clocks.mem": "877", "clocks.max.mem": "877",
            "pcie.link.gen.current": "3", "pcie.link.width.current": str(c.get("lanes", 16)), "pcie.link.gen.max": "3",
            "pcie.link.width.max": "16", "fan.speed": "[N/A]",
            "clocks_throttle_reasons.active": "0x4" if power > c["limit_w"] * .97 else "0x0",
            "clocks_event_reasons.active": "0x4" if power > c["limit_w"] * .97 else "0x0"}.get(f, "[N/A]")
if q:
    fields = [x.strip() for x in q.split(",")]
    for i, c in enumerate(cards):
        print(", ".join(val(f, i, c) for f in fields))
elif qa:
    for a in st.get("apps") or []:
        print("%%d, %%d, GPU-fake-%%d" %% (a["pid"], a["mib"], a["gpu"]))
sys.exit(0)
'''


class Engine(object):
    """One fake llama-server: a scripted workload behind the endpoints the Live tab reads."""

    def __init__(self, name, gpus, port=0, slots=2, dec_tps=34.0, pre_tps=820.0, spec=True, xcache=False, metrics=False, seed=1, idle_s=(2.0, 7.0),
                 stats=True, n_ctx=32768):
        self.name, self.port, self.gpus, self.np = name, port, gpus, slots
        self.dec_tps, self.pre_tps, self.spec, self.xcache, self.metrics, self.stats = dec_tps, pre_tps, spec, xcache, metrics, stats
        self.n_ctx = n_ctx
        self.rnd = random.Random(seed)
        self.idle_s = idle_s
        self.lock = threading.Lock()
        self.slots = [{"id": i, "task": -1, "busy": False, "next": time.time() + self.rnd.uniform(0.5, 3.0), "n_decoded": 0, "n_remain": -1,
                       "dtot": 0, "dacc": 0, "ema": 0.7, "ctx": 0} for i in range(slots)]
        self.records = []
        self.task = 100
        self.x = {"hits": 0, "misses": 0, "swaps_started": 0, "swaps_admitted": 0, "swaps_evicted": 0}
        self.tot = {"pred": 0, "pred_s": 0.0, "pp": 0, "pp_s": 0.0}
        self.stop_ev = threading.Event()
        self.srv = None

    # ---- workload
    def busy_fraction(self):
        with self.lock:
            return sum(1 for s in self.slots if s["busy"]) / float(self.np)

    def _tick(self, now, dt):
        with self.lock:
            for s in self.slots:
                if not s["busy"]:
                    if now >= s["next"]:
                        n_prompt = int(self.rnd.choice([300, 800, 1500, 3200, 6000, 12000]) * self.rnd.uniform(.7, 1.3))
                        cached = int(n_prompt * self.rnd.choice([0, 0, .5, .9]))
                        self.task += 1
                        s.update(busy=True, task=self.task, n_prompt=n_prompt, cached=cached, prompt_n=n_prompt - cached, t_start=now,
                                 n_gen=int(self.rnd.uniform(60, 700)), n_decoded=0, dtot=0, dacc=0, t_dec=None,
                                 pre_tps=self.pre_tps * self.rnd.uniform(.85, 1.1), dec_tps=self.dec_tps * self.rnd.uniform(.85, 1.1))
                        s["t_pre_end"] = now + s["prompt_n"] / s["pre_tps"]
                        s["ctx"] = n_prompt
                    continue
                if now < s["t_pre_end"]:
                    self.tot["pp"] += int(s["pre_tps"] * dt); self.tot["pp_s"] += dt
                    continue
                if s["t_dec"] is None:
                    s["t_dec"] = s["t_pre_end"]
                want = min(s["n_gen"], int((now - s["t_dec"]) * s["dec_tps"] * (1 + .06 * math.sin(now / 2.0))))
                step = max(0, want - s["n_decoded"])
                s["n_decoded"] = want
                s["n_remain"] = s["n_gen"] - want
                s["ctx"] = s["n_prompt"] + want
                self.tot["pred"] += step; self.tot["pred_s"] += dt if step else 0.0
                if self.spec and step:
                    acc = min(.97, max(.3, .74 + .12 * math.sin(now / 9.0) + self.rnd.uniform(-.05, .05)))
                    drafted = step * 2
                    s["dtot"] += drafted; s["dacc"] += int(drafted * acc)
                    s["ema"] = .9 * s["ema"] + .1 * acc
                if self.xcache and step:
                    hit = min(.99, max(.5, .88 + .06 * math.sin(now / 17.0)))
                    n = step * 40
                    self.x["hits"] += int(n * hit); self.x["misses"] += n - int(n * hit)
                    if self.rnd.random() < .15:
                        self.x["swaps_started"] += 1; self.x["swaps_admitted"] += 1
                if s["n_decoded"] >= s["n_gen"]:
                    t0 = s["t_start"]
                    rec = {"ts": round(now, 3), "model": self.name + ".gguf", "slot": s["id"], "n_prompt": s["n_prompt"], "n_cached": s["cached"],
                           "prompt_n": s["prompt_n"], "prompt_ms": round((s["t_pre_end"] - t0) * 1000, 1), "prefill_tps": round(s["pre_tps"], 2),
                           "n_gen": s["n_gen"], "gen_ms": round((now - s["t_dec"]) * 1000, 1),
                           "decode_tps": round(s["n_gen"] / max(1e-3, now - s["t_dec"]), 2), "split": "tensor", "n_gpu": len(self.gpus)}
                    if self.spec:
                        rec["draft_n"], rec["draft_acc"] = s["dtot"], s["dacc"]
                    self.records.append(rec)
                    s.update(busy=False, next=now + self.rnd.uniform(*self.idle_s), task=-1)

    def start(self):
        eng = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, ctype="application/json"):
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(u.query)
                p = u.path
                with eng.lock:
                    busy = sum(1 for s in eng.slots if s["busy"])
                    if p == "/health":
                        return self._send(200, {"status": "ok", "slots_idle": eng.np - busy, "slots_processing": busy})
                    if p == "/props":
                        used = sum(s["ctx"] for s in eng.slots)
                        d = {"total_slots": eng.np, "n_ctx": eng.n_ctx, "model_alias": eng.name, "kv_cache_used_cells": used,
                             "chat_template": "{{ x }}" * 200}
                        if eng.xcache:
                            d["pxa_xcache"] = dict(eng.x)
                        return self._send(200, d)
                    if p == "/slots":
                        out = []
                        for s in eng.slots:
                            out.append({"id": s["id"], "id_task": s["task"], "state": 1 if s["busy"] else 0, "n_ctx": eng.n_ctx // eng.np,
                                        "prompt": "SECRET-PROMPT-TEXT " * 400, "next_token": {"has_next_token": s["busy"], "n_remain": s["n_remain"],
                                                                                              "n_decoded": s["n_decoded"]},
                                        "n_draft_total": s["dtot"], "n_draft_accepted": s["dacc"], "spec_accept_ema": s["ema"]})
                        return self._send(200, out)
                    if p == "/metrics":
                        if not eng.metrics:
                            return self._send(501, {"error": {"message": "This server does not support metrics endpoint."}})
                        t = eng.tot
                        txt = "".join("llamacpp:%s %s\n" % (k, v) for k, v in (
                            ("prompt_tokens_total", t["pp"]), ("prompt_seconds_total", t["pp_s"]), ("tokens_predicted_total", t["pred"]),
                            ("tokens_predicted_seconds_total", t["pred_s"]), ("requests_processing", busy)))
                        return self._send(200, txt.encode(), "text/plain")
                    if p == "/pxa/stats":
                        if not eng.stats:
                            return self._send(404, {"error": "not found"})
                        since = float((q.get("since") or ["0"])[0])
                        lim = int((q.get("limit") or ["0"])[0])
                        recs = [r for r in eng.records if r["ts"] >= since]
                        if lim:
                            recs = recs[-lim:]
                        return self._send(200, {"now": time.time(), "count": len(recs), "stored": len(eng.records), "records": recs})
                self._send(404, {"error": "not found"})

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True
        self.srv = S(("127.0.0.1", self.port), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

        def run():
            last = time.time()
            while not self.stop_ev.is_set():
                now = time.time()
                self._tick(now, now - last)
                last = now
                self.stop_ev.wait(0.1)
        threading.Thread(target=run, daemon=True).start()
        return self

    def stop(self):
        self.stop_ev.set()
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()


CARD_KINDS = [("Tesla V100-PCIE-16GB", "7.0", 16384, 250), ("Tesla P100-PCIE-16GB", "6.0", 16384, 250), ("NVIDIA GeForce GTX 1080 Ti", "6.1", 11264, 250)]


class Env(object):
    def __init__(self, cards=4, servers=None, port=0, real=None):
        self.tmp = tempfile.mkdtemp(prefix="pxa-live-env-")
        self.cards_n = cards
        self.real = real or []          # [{name, port, gpus}]: servers that already run (a real llama-server): Control just watches them
        self.port = port
        self.engines = []
        self.state_path = os.path.join(self.tmp, "smi-state.json")
        self.spec = servers if servers is not None else [
            dict(name="Qwen3.8-27B", gpus=[0, 1], slots=2, dec_tps=36, pre_tps=900, spec=True, seed=3),
            dict(name="Flash-Next", gpus=[2, 3], slots=1, dec_tps=22, pre_tps=500, spec=False, xcache=True, metrics=True, seed=5)]
        os.makedirs(os.path.join(self.tmp, "bin"), exist_ok=True)
        smi = os.path.join(self.tmp, "bin", "nvidia-smi")
        with open(smi, "w") as f:
            f.write(SMI % {"py": sys.executable, "state": self.state_path})
        os.chmod(smi, 0o755)
        env = os.environ
        self.orig_path = env["PATH"]
        env["PATH"] = os.path.join(self.tmp, "bin") + os.pathsep + env["PATH"]
        env.pop("PXA_LAUNCH_FAKE_GPUS", None)
        env["PXA_CONTROL_CONFIG_DIR"] = os.path.join(self.tmp, "cfg")
        env["PXA_LAUNCH_STATE"] = os.path.join(self.tmp, "state")
        env["PXA_CONTROL_DISCOVER"] = "0"
        env["HOME"] = os.path.join(self.tmp, "home")
        os.makedirs(env["HOME"], exist_ok=True)
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(self.models, exist_ok=True)
        self.card_defs = []
        for i in range(cards):
            n, cc, mem, lim = CARD_KINDS[i % len(CARD_KINDS)]
            self.card_defs.append({"name": n, "cc": cc, "mem_total": mem, "limit_w": lim, "idle_w": 28 + i, "mem_base": 300 + 40 * i, "load": 0.0, "heat": 0.0,
                                   "lanes": 16 if i != 3 else 4})
        self.apps = []
        self.seats_procs = []
        self.srv = None
        self.app = None
        self.stop_ev = threading.Event()

    def _write_state(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"cards": self.card_defs, "apps": self.apps}, f)
        os.replace(tmp, self.state_path)

    def _heat_loop(self):
        while not self.stop_ev.is_set():
            load = [0.0] * self.cards_n
            for e in self.engines:
                b = e.busy_fraction()
                for g in e.gpus:
                    if g < self.cards_n:
                        load[g] = max(load[g], b)
            for i, c in enumerate(self.card_defs):
                c["load"] = load[i] * (0.93 if i % 2 else 1.0)
                c["heat"] += (c["load"] - c["heat"]) * 0.03
            self._write_state()
            self.stop_ev.wait(0.5)

    def start(self):
        sys.path.insert(0, TOOLS)
        import importlib.util
        spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
        self.L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.L)
        import pxa_control as C
        self.C = C
        model = os.path.join(self.models, "tiny.gguf")
        with open(model, "wb") as f:
            f.write(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)
        self._write_state()
        for sp in self.spec:
            self.engines.append(Engine(**sp).start())
        threading.Thread(target=self._heat_loop, daemon=True).start()
        self.app = C.App(self.L, models_dirs=[self.models])
        for i, e in enumerate(self.engines):
            sid, _ent = self.app.save_profile(None if i else "main", e.name, {"gpus": e.gpus, "model": model, "port": e.port})
            seat = self.app.get_seat(sid, create=True)
            seat.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(86400)"])
            seat.req = {"port": e.port, "gpus": list(e.gpus), "model": model}
            self.seats_procs.append(seat.proc)
            for g in e.gpus:
                self.apps.append({"pid": seat.proc.pid, "mib": 6000 + 900 * i, "gpu": g})
        for j, r in enumerate(self.real):
            sid, _ent = self.app.save_profile(None, r["name"], {"gpus": r.get("gpus", []), "model": model, "port": r["port"]})
            seat = self.app.get_seat(sid, create=True)
            seat.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(86400)"])
            seat.req = {"port": r["port"], "gpus": list(r.get("gpus", [])), "model": model}
            self.seats_procs.append(seat.proc)
        self._write_state()
        self.srv = C.make_server(self.app, "127.0.0.1", self.port)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port

    def close(self):
        self.stop_ev.set()
        try:
            self.app.live.close()
        except Exception:
            pass
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()
        for p in self.seats_procs:
            p.kill()
        for e in self.engines:
            e.stop()
        os.environ["PATH"] = self.orig_path
        shutil.rmtree(self.tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--cards", type=int, default=4)
    ap.add_argument("--real-port", type=int, default=0, help="also watch a real llama-server already running on this port, and keep it busy")
    ap.add_argument("--no-fake", action="store_true", help="no fake servers (only --real-port)")
    a = ap.parse_args()
    real = [{"name": "tiny model (real engine)", "port": a.real_port, "gpus": [a.cards - 1]}] if a.real_port else []    # a profile needs a card
    env = Env(cards=a.cards, port=a.port, real=real, servers=[] if a.no_fake else None).start()
    if a.real_port:
        def keep_busy():
            import urllib.request
            rnd = random.Random(7)
            while True:
                body = json.dumps({"prompt": "Write a long story about a fox. " * rnd.choice([1, 8, 40]), "n_predict": rnd.choice([300, 900, 2000]),
                                   "temperature": 0.8, "ignore_eos": True}).encode()
                try:
                    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%d/completion" % a.real_port, data=body,
                                                                   headers={"Content-Type": "application/json"}), timeout=300).read()
                except Exception:
                    time.sleep(5)
                time.sleep(rnd.uniform(1, 6))
        threading.Thread(target=keep_busy, daemon=True).start()
    print("live env at", env.url, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        env.close()
