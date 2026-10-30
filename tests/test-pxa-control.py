#!/usr/bin/env python3
"""PXA Control (pxa-launch --gui, tools/pxa_control.py): HTTP handlers, token auth, lever validation,
preset persistence, the engine proxy and the seat's process handling. No GPU, no model, no network
beyond 127.0.0.1: cards come from PXA_LAUNCH_FAKE_GPUS and the "engine" is a stub HTTP server.

    python3 tests/test-pxa-control.py          (wired into CTest as test-pxa-control)
"""
import contextlib
import hashlib
import http.server
import io
import importlib.util
import json
import os
import re
import shutil
import socket
import socketserver
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-control-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""          # hermetic: the fixture store, never the live licd (the release default)
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ.pop("PXA_MODELS_DIR", None)
os.environ["PXA_CONTROL_DISCOVER"] = "0"      # hermetic: never list the host's real servers
for _k in ("PXQE_CUDA_LIBS", "PXQE_RUNTIME_DIR", "PXQE_ENGINE"):   # hermetic: test_08 models a computer without the CUDA 12 libraries
    os.environ.pop(_k, None)
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN"):
    os.environ.pop(_k, None)                  # no browser opens, and the auto-start rule starts from its defaults
sys.path.insert(0, TOOLS)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402

_bspec = importlib.util.spec_from_file_location("pxa_bench_real", os.path.join(TOOLS, "pxa-bench.py"))
BENCH = importlib.util.module_from_spec(_bspec)        # the real summarize_runs / run_tag (FR#12 honest repeats)
_bspec.loader.exec_module(BENCH)

MODELS = os.path.join(TMP, "models")
os.makedirs(MODELS, exist_ok=True)
FAKE_MODEL = os.path.join(MODELS, "tiny.gguf")
with open(FAKE_MODEL, "wb") as f:
    f.write(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)       # a header with 0 tensors, 0 KVs
CATALOG = {"PXA_ENHANCE", "PXA_CUDA_GRAPH_LRU", "PXQ_TEST"}


TOKENS = {}          # port -> the session token of an in-process Control on loopback (H1: every API call needs it)


def serve_app(app):
    srv = C.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if app.token and not app.lan:
        TOKENS[srv.server_address[1]] = app.token
    return srv, srv.server_address[1]


def token_for(port):
    """the token a loopback Control on port wants: a live run record's first (TOKENS keeps closed test servers'
    tokens, and a spawned Control can get their port back), else the one this test serves."""
    for rec in C.running_controls():
        if rec.get("port") == port and rec.get("token") and not rec.get("lan"):
            return rec["token"]
    return TOKENS.get(port)


def req(port, path, method="GET", body=None, headers=None, host=None, auth=True):
    """auth=True signs a request to a loopback Control in like its own page would (the X-PXA-Token header),
    unless the caller already carries a cookie, a token header or a bearer; auth=False sends none."""
    h = dict(headers or {})
    if auth and not any(k.lower() in ("cookie", "x-pxa-token", "authorization") for k in h):
        t = token_for(port)
        if t:
            h["X-PXA-Token"] = t
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h, method=method)
    if host:
        r.add_unredirected_header("Host", host)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    op = urllib.request.build_opener(NoRedirect)
    try:
        with op.open(r, timeout=20) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class LeverValidation(unittest.TestCase):
    def test_known_names_and_values(self):
        ok, errs = C.validate_levers({"PXA_ENHANCE": "1", "PXA_CUDA_GRAPH_LRU": 4, "PXQ_TEST": True}, CATALOG)
        self.assertEqual(errs, [])
        self.assertEqual(ok, {"PXA_ENHANCE": "1", "PXA_CUDA_GRAPH_LRU": "4", "PXQ_TEST": "1"})

    def test_unknown_and_foreign_names_refused(self):
        for name in ("LD_PRELOAD", "PATH", "PXA_NOT_IN_CATALOG", "pxa_enhance", "PXA_ENHANCE;rm"):
            _ok, errs = C.validate_levers({name: "1"}, CATALOG)
            self.assertTrue(errs, name)

    def test_bad_values_refused(self):
        for v in ("1; rm -rf /", "$(id)", "a b", "x\ny", "`id`", "a" * 201, "'q'", {"x": 1}):
            _ok, errs = C.validate_levers({"PXA_ENHANCE": v}, CATALOG)
            self.assertTrue(errs, repr(v))

    def test_empty_value_means_default(self):
        ok, errs = C.validate_levers({"PXA_ENHANCE": ""}, CATALOG)
        self.assertEqual((ok, errs), ({}, []))

    def test_shape(self):
        self.assertTrue(C.validate_levers(["PXA_ENHANCE"], CATALOG)[1])
        self.assertTrue(C.validate_levers({f"PXA_X{i}": "1" for i in range(65)}, CATALOG)[1])

    def test_catalog_parser(self):
        p = os.path.join(TMP, "cat.inc")
        with open(p, "w") as f:
            f.write('// comment\n{ "PXA_A", "on", "any", "default-on", "", "house lever" },\n'
                    '{ "PXA_B", "off (=1 on)", "any", "lever-off", "ev", "a \\"quoted\\" rule" },\n'
                    '{ "not_a_lever", "x", "y", "z", "", "" },\n')
        rows, src = C.load_catalog(p)
        self.assertEqual([r["name"] for r in rows], ["PXA_A", "PXA_B"])
        self.assertEqual(rows[1]["rule"], 'a "quoted" rule')
        self.assertEqual(src, os.path.abspath(p))

    def test_real_catalog_loads(self):
        rows, src = C.load_catalog()
        if src is None:
            self.skipTest("no catalog in this tree")
        self.assertGreater(len(rows), 50)
        self.assertIn("PXA_ENHANCE", {r["name"] for r in rows})


class BuiltinLevers(unittest.TestCase):
    """A release catalog drops the levers the closed library reads and keeps only their hashes: those names are
    accepted (no refusal, no lint warning), labelled built-in, and a typo is still refused."""
    def setUp(self):
        self.p = os.path.join(TMP, "cat-release.inc")
        h = C.lever_hash
        with open(self.p, "w") as f:
            f.write('{ "PXA_ENHANCE", "level 2", "any", "default-on", "", "x" },\n')
            f.write("// built-in: %s %s %s\n" % (h("PXA_HIDDEN_A"), h("PXA_LOCKY"), h("PXA_FAM*")))
            f.write("// built-in-preset: %s %s\n" % (h("PXA_HIDDEN_A"), h("PXA_FAM*")))
        rows, src = C.load_catalog(self.p)
        self.rows = rows
        self.names = C.LeverNames({r["name"] for r in rows}, *C.load_builtin(src))

    def test_accepted_and_labelled(self):
        ok, errs = C.validate_levers({"PXA_HIDDEN_A": "1", "PXA_FAM_X": "2", "PXA_LOCKY": "1", "PXA_ENHANCE": "1"}, self.names)
        self.assertEqual(errs, [])
        self.assertEqual(ok, {"PXA_HIDDEN_A": "1", "PXA_FAM_X": "2", "PXA_LOCKY": "1", "PXA_ENHANCE": "1"})
        self.assertEqual(self.names.builtin("PXA_HIDDEN_A"), "preset")
        self.assertEqual(self.names.builtin("PXA_FAM_X"), "preset")
        self.assertEqual(self.names.builtin("PXA_LOCKY"), "builtin")
        self.assertIsNone(self.names.builtin("PXA_ENHANCE"))
        self.assertEqual(C.lint_levers({"PXA_HIDDEN_A": "1"}, self.rows), [])

    def test_typo_still_refused(self):
        _ok, errs = C.validate_levers({"PXA_HIDDEN_B": "1"}, self.names)
        self.assertTrue(errs)
        _ok, errs = C.validate_levers({"LD_PRELOAD": "x"}, self.names)
        self.assertTrue(errs)

    def test_full_catalog_has_no_builtin(self):
        self.assertEqual(C.load_builtin(os.path.join(TMP, "nope.inc")), (set(), set()))

    def test_hash_matches_catalog_script(self):
        self.assertEqual(C.lever_hash("PXA_ENHANCE"), "%016x" % self._fnv(b"PXA_ENHANCE"))

    @staticmethod
    def _fnv(b):
        h = 0xcbf29ce484222325
        for x in b:
            h = ((h ^ x) * 0x100000001b3) % (1 << 64)
        return h


class LaunchValidation(unittest.TestCase):
    def v(self, **kw):
        body = {"gpus": [0], "model": FAKE_MODEL}
        body.update(kw)
        return C.validate_launch(body, {0, 1}, CATALOG, [MODELS], gui_port=7777)

    def test_defaults(self):
        r = self.v()
        self.assertEqual((r["gpus"], r["kv"], r["sm"], r["fa"], r["ctx"], r["np"]),
                         ([0], "auto", "auto", "auto", 0, 0))
        # port auto = the first FREE port from 8080 up (8080 itself is often taken)
        self.assertGreaterEqual(r["port"], 8080)
        self.assertFalse(C.port_in_use(r["port"]))

    def test_port_busy_and_content_length(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        try:
            self.assertTrue(C.port_in_use(busy))
            self.assertNotEqual(C.free_port(busy), busy)
        finally:
            s.close()
        self.assertEqual(C.content_length({}, 10), 0)
        self.assertEqual(C.content_length({"Content-Length": "7"}, 10), 7)
        for bad in ("-1", "abc", "11", " 1e3"):
            with self.assertRaises(C.Invalid):
                C.content_length({"Content-Length": bad}, 10)

    def test_refusals(self):
        bad = [dict(gpus=[]), dict(gpus=[7]), dict(gpus=["x"]), dict(model=""), dict(model="/etc/passwd"),
               dict(model=os.path.join(TMP, "nope.gguf")), dict(kv="q1_0"), dict(sm="graph"), dict(fa="maybe"),
               dict(port=80), dict(port=7777), dict(port="8080; id"), dict(ctx=-1), dict(ctx=1 << 22),
               dict(np=65), dict(np=True), dict(levers={"LD_PRELOAD": "x"})]
        for b in bad:
            with self.assertRaises(C.Invalid, msg=repr(b)):
                self.v(**b)

    def test_model_must_be_under_a_folder(self):
        other = os.path.join(TMP, "elsewhere.gguf")
        with open(other, "wb") as f:
            f.write(b"GGUF")
        with self.assertRaises(C.Invalid):
            self.v(model=other)
        # a link in the folder to a real GGUF elsewhere (an offloaded model) is allowed (2026-10-06);
        # a link out of the folder to anything that is not a GGUF is still refused (SymlinkedModels)
        link = os.path.join(MODELS, "link.gguf")
        if not os.path.lexists(link):
            os.symlink(other, link)
        self.assertEqual(self.v(model=link)["model"], link)
        notg = os.path.join(TMP, "elsewhere-text.gguf")
        with open(notg, "wb") as f:
            f.write(b"text")
        badlink = os.path.join(MODELS, "badlink.gguf")
        if not os.path.lexists(badlink):
            os.symlink(notg, badlink)
        with self.assertRaises(C.Invalid):
            self.v(model=badlink)

    def test_argv(self):
        r = self.v(gpus="1,0", ctx=16384, np=2, kv="q8_0", sm="tensor", fa="on", mtp=True, port=8123,
                   expose=True, accept_unmeasured=True)
        a = C.launcher_argv(r)
        self.assertEqual(a[:4], ["--gpus", "1,0", "--model", FAKE_MODEL])
        for want in (["--ctx", "16384"], ["--np", "2"], ["--ctk", "q8_0"], ["--ctv", "q8_0"], ["--sm", "tensor"],
                     ["--workload", "serve"], ["--spec", "mtp"], ["--host", "0.0.0.0"], ["--port", "8123"]):
            i = a.index(want[0])
            self.assertEqual(a[i:i + 2], want)
        self.assertIn("--accept-unmeasured", a)
        self.assertNotIn("--allow-busy", a)
        # kv auto (the default) passes no --ctk/--ctv: the engine registry picks the KV type
        a2 = C.launcher_argv(self.v())
        self.assertNotIn("--ctk", a2)
        self.assertNotIn("--ctv", a2)
        ns2 = L.build_parser().parse_args(a2)
        self.assertEqual((ns2.ctk, ns2.ctv), ("auto", "auto"))
        # the launcher's own parser accepts every argv the GUI builds
        ns = L.build_parser().parse_args(a)
        self.assertEqual((ns.gpus, ns.np, ns.workload, ns.spec), ("1,0", 2, "serve", "mtp"))
        self.assertEqual(C.launcher_argv(self.v(fa="off"))[C.launcher_argv(self.v(fa="off")).index("--workload") + 1],
                         "longdoc")
        self.assertNotIn("--workload", C.launcher_argv(self.v()))

    def test_ngram_toggle(self):
        # auto (the default) sets nothing: the engine decides; on/off pin the engine switch; anything else is refused
        r = self.v(mtp=True)
        self.assertEqual(r["ngram"], "auto")
        self.assertNotIn("PXA_SPEC_MTP_NGRAM", r["levers"])
        self.assertEqual(self.v(mtp=True, ngram="on")["levers"]["PXA_SPEC_MTP_NGRAM"], "1")
        off = self.v(mtp=True, ngram="off", levers={"PXA_ENHANCE": "1"})
        self.assertEqual(off["levers"], {"PXA_ENHANCE": "1", "PXA_SPEC_MTP_NGRAM": "0"})
        self.assertTrue(C.cli_line(off).startswith("PXA_ENHANCE=1 PXA_SPEC_MTP_NGRAM=0 pxa-launch "))
        self.assertIn(["--spec", "mtp"], [C.launcher_argv(off)[i:i + 2] for i in range(len(C.launcher_argv(off)))])
        for bad in ("yes", 1, "ON", "auto; id", ["on"], {"a": 1}):
            with self.assertRaises(C.Invalid, msg=repr(bad)):
                self.v(ngram=bad)
        ui = open(os.path.join(TOOLS, "pxa_control_ui", "index.html")).read()
        self.assertIn('id="l-ngram"', ui)
        self.assertIn('ngram: $("#l-ngram").dataset.v', ui)

    def test_cli_line_quotes(self):
        r = self.v(levers={"PXA_ENHANCE": "0"})
        line = C.cli_line(r)
        self.assertTrue(line.startswith("PXA_ENHANCE=0 pxa-launch --gpus 0"))


class OverdriveLine(unittest.TestCase):
    """od_line lifts the launcher's one-line Overdrive verdict onto the plan heading.

    The heading is the only place an operator sees WHICH setup the plan picked without
    expanding the report, so the extraction has to survive the report's real shape (the
    verdict line is indented, and the block below it is indented further) and must return
    None - never a half-line - for a model of another class."""

    def test_lifts_the_verdict_and_collapses_whitespace(self):
        text = ("  class: qwen4exp\n"
                "  OVERDRIVE: this file is the measured Flash-Next class - 24 switches will be\n"
                "             exported (incl. PXA_XCACHE_MTP=1).\n"
                "  Overdrive is on because this class is what it was measured on.\n")
        self.assertEqual(C.od_line(text),
                         "OVERDRIVE: this file is the measured Flash-Next class - 24 switches will be")

    def test_off_and_absent_are_both_carried(self):
        self.assertTrue(C.od_line("  OVERDRIVE: OFF (PXA_NO_OVERDRIVE=1).").startswith("OVERDRIVE: OFF"))
        self.assertTrue(C.od_line("  Overdrive    APPLIES: arch qwen4exp").startswith("Overdrive APPLIES"))
        self.assertIsNone(C.od_line("  class: qwen35moe\n  recipe: something\n"))
        self.assertIsNone(C.od_line(""))
        self.assertIsNone(C.od_line(None))


# ---- box topology fixtures (2026-10-12) ------------------------------
# CriticalTechGuy's host: dual Xeon E5-2640 v2 (8 cores / 16 threads per socket, NO AVX2), NUMA node0 =
# 0-30:2 and node1 = 1-31:2 (INTERLEAVED numbering), GPU0 on node 0, GPU1 on node 1, GPU0<->GPU1 = SYS.
CTG_TOPO = ("\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
            "GPU0\t X \tSYS\t0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30\t0\t\tN/A\n"
            "GPU1\tSYS\t X \t1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31\t1\t\tN/A\n\n"
            "Legend:\n\n  X    = Self\n  SYS  = Connection traversing PCIe as well as the SMP interconnect\n")
CTG_LSCPU = ("Architecture:            x86_64\nCPU(s):                  32\n"
             "Model name:              Intel(R) Xeon(R) CPU E5-2640 v2 @ 2.00GHz\n"
             "Thread(s) per core:      2\nCore(s) per socket:      8\nSocket(s):               2\n"
             "Flags:                   fpu vme de pse tsc msr sse sse2 ssse3 sse4_1 sse4_2 popcnt aes avx f16c rdrand\n"
             "NUMA node(s):            2\nNUMA node0 CPU(s):       0-30:2\nNUMA node1 CPU(s):       1-31:2\n")
# this box: 2x E5-2699 v3, node0 = 0-17,36-53, all seven cards on node 0 behind host bridges (PHB)
BOX_TOPO = ("\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
            + "".join("GPU%d\t" % i + "\t".join(" X " if j == i else "PHB" for j in range(7))
                      + "\t0-17,36-53\t0\t\tN/A\n" for i in range(7)))
BOX_LSCPU = ("Model name: Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz\nSocket(s): 2\nThread(s) per core: 2\n"
             "Flags: fpu sse sse2 avx avx2 fma f16c\nNUMA node0 CPU(s): 0-17,36-53\nNUMA node1 CPU(s): 18-35,54-71\n")
BOX_SIB = {c: frozenset({c, c + 36}) for c in range(36)}
BOX_SIB.update({c + 36: frozenset({c, c + 36}) for c in range(36)})
# a 4-GPU single-socket workstation: one node, cards behind a PLX switch (PXB) and a host bridge
WS4_TOPO = ("\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\n"
            "GPU0\t X \tPXB\tPHB\tPHB\t0-23\t0\nGPU1\tPXB\t X \tPHB\tPHB\t0-23\t0\n"
            "GPU2\tPHB\tPHB\t X \tPXB\t0-23\t0\nGPU3\tPHB\tPHB\tPXB\t X \t0-23\t0\n")
WS4_LSCPU = "Model name: AMD Ryzen Threadripper 2920X\nSocket(s): 1\nFlags: avx avx2 fma\nNUMA node0 CPU(s): 0-23\n"


def _topo(topo_txt, lscpu_txt, allowed=None, sib=None, container=False, nice=None, cache_mount=None):
    tt, lt = L.parse_nvidia_topo(topo_txt), L.parse_lscpu(lscpu_txt)
    return {"gpus": tt["gpus"], "links": tt["links"], "nodes": lt["nodes"], "sockets": lt["sockets"],
            "flags": lt["flags"], "model": lt["model"], "siblings": sib or {}, "allowed": allowed,
            "container": container, "cap_sys_nice": nice, "cache_dir": "/work/.cache/pxa",
            "cache_mount": cache_mount, "source": "test"}


class BoxTopology(unittest.TestCase):
    """NUMA / socket detection and the CPU pool, against real `nvidia-smi topo -m` and `lscpu` text."""

    def test_cpulist_strides_and_interleaving(self):
        self.assertEqual(L._parse_cpulist("0-30:2"), set(range(0, 31, 2)))
        self.assertEqual(L._parse_cpulist("1-31/2"), set(range(1, 32, 2)))
        self.assertEqual(L._parse_cpulist("0-17,36-53"), set(range(18)) | set(range(36, 54)))
        self.assertEqual(L._cpuset_to_list(set(range(0, 31, 2))), ",".join(str(c) for c in range(0, 31, 2)))
        self.assertEqual(L._sysfs_bus_id("00000000:02:00.0"), "0000:02:00.0")

    def test_his_box_parses(self):
        t = _topo(CTG_TOPO, CTG_LSCPU)
        self.assertEqual(t["gpus"][0]["numa"], 0)
        self.assertEqual(t["gpus"][1]["numa"], 1)
        self.assertEqual(t["links"][(0, 1)], "SYS")
        self.assertEqual(t["nodes"][0], set(range(0, 31, 2)))
        self.assertEqual(t["sockets"], 2)
        self.assertNotIn("avx2", t["flags"])

    def test_his_box_one_card_pins_to_its_own_node(self):
        for card, node in ((0, 0), (1, 1)):
            d = L.numa_decide(_topo(CTG_TOPO, CTG_LSCPU, allowed=set(range(32))), [card])
            self.assertEqual(d["node"], node)
            self.assertFalse(d["spans"])
            self.assertEqual(L._parse_cpulist(d["pool"]), set(range(node, 32, 2)))   # never the other socket
            self.assertEqual(d["warnings"], [])
        # the preset's 0-15 would have named 8 CPUs of EACH socket
        self.assertNotEqual(L._parse_cpulist("0-15"), set(range(0, 31, 2)))

    def test_his_box_two_cards_warn_span(self):
        d = L.numa_decide(_topo(CTG_TOPO, CTG_LSCPU, allowed=set(range(32))), [0, 1])
        self.assertTrue(d["spans"])
        self.assertIsNone(d["node"])
        self.assertEqual(d["link"], "SYS")
        self.assertTrue(any(w.startswith("spans: ") and "--gpus 0" in w for w in d["warnings"]))

    def test_his_box_in_docker_without_sys_nice(self):
        d = L.numa_decide(_topo(CTG_TOPO, CTG_LSCPU, allowed=set(range(32)), container=True, nice=False), [0])
        self.assertTrue(any(w.startswith("sysnice: ") and "--cap-add SYS_NICE" in w for w in d["warnings"]))
        d = L.numa_decide(_topo(CTG_TOPO, CTG_LSCPU, allowed=set(range(32)), container=True, nice=True), [0])
        self.assertFalse(any(w.startswith("sysnice: ") for w in d["warnings"]))

    def test_our_box_pool_is_the_measured_0_15(self):
        t = _topo(BOX_TOPO, BOX_LSCPU, allowed=set(range(72)), sib=BOX_SIB)
        for sel in ([0], [5, 6], [0, 1, 2, 4]):
            d = L.numa_decide(t, sel)
            self.assertEqual(d["node"], 0)
            self.assertFalse(d["spans"])
            self.assertEqual(d["pool"], "0-15")        # 16 physical cores of node 0 = the preset's measured pool
            self.assertEqual(d["warnings"], [])
        # a launcher held to the other socket (taskset 54-71) says so instead of silently using it
        d = L.numa_decide(_topo(BOX_TOPO, BOX_LSCPU, allowed=set(range(54, 72)), sib=BOX_SIB), [5])
        self.assertEqual(L._parse_cpulist(d["pool"]) - set(range(54, 72)), set())
        self.assertTrue(any(w.startswith("cpuset: ") and "--cpuset-cpus=0-17,36-53" in w for w in d["warnings"]))

    def test_four_gpu_single_node(self):
        t = _topo(WS4_TOPO, WS4_LSCPU, allowed=set(range(24)))
        self.assertEqual(t["links"][(0, 1)], "PXB")
        d = L.numa_decide(t, [0, 1, 2, 3])
        self.assertEqual(d["node"], 0)
        self.assertFalse(d["spans"])
        self.assertFalse(d["multi_node"])
        self.assertEqual(d["warnings"], [])
        self.assertEqual(len(L._parse_cpulist(d["pool"])), 16)

    def test_no_numa_info_falls_back_to_allowed(self):
        t = _topo("", "", allowed={2, 3, 4})
        d = L.numa_decide(t, [0])
        self.assertEqual(d["pool"], "2-4")
        self.assertIn("allowed CPU set", d["pool_why"])


class BoxDecisions(unittest.TestCase):
    """KV / -b / -c for the Flash-Next class, the counts env, the cache volume and heat."""
    FN = {"arch": "qwen4exp", "is_moe": True, "deltanet": True, "n_expert": 512, "tier": "PXQ_UNIVERSAL",
          "mtp_tensors": 6, "kv_bytes_tok": 100352, "ple_bytes": 0}

    def test_kv_pick_flashnext_only_on_p100_v100(self):
        p100 = [(0, "P100", 60, 16384, 0, "u")]
        self.assertEqual(L.flashnext_kv_pick(self.FN, p100), "q8_0")
        self.assertEqual(L.flashnext_kv_pick(self.FN, [(0, "V100", 70, 16384, 0, "u")]), "q8_0")
        self.assertIsNone(L.flashnext_kv_pick(self.FN, [(0, "3090", 86, 24576, 0, "u")]))
        self.assertIsNone(L.flashnext_kv_pick({"arch": "qwen35"}, p100))

    def test_never_b8192_on_16gb(self):
        sel = [(0, "P100", 60, 16384, 0, "u"), (1, "P100", 60, 16384, 0, "u")]
        cmd = ["x", "-b", "8192", "-ub", "256"]
        line = L.flashnext_batch_cap(cmd, self.FN, sel, False)
        self.assertEqual(cmd[2], "2048")
        self.assertIn("DECISION batch: -b 2048 (not 8192)", line)
        cmd = ["x", "-b", "8192", "-ub", "256"]
        self.assertIsNone(L.flashnext_batch_cap(cmd, self.FN, sel, True))           # forced by the user
        self.assertIsNone(L.flashnext_batch_cap(cmd, {"arch": "qwen35"}, sel, False))  # the dense cell keeps 8192
        self.assertIsNone(L.flashnext_batch_cap(cmd, self.FN, [(0, "V100", 70, 32768, 0, "u")], False))
        self.assertEqual(cmd[2], "8192")

    def test_mtp_ctx_lowered_so_the_draft_floor_fits(self):
        with tempfile.TemporaryDirectory() as d:
            m = os.path.join(d, "fn.gguf")
            with open(m, "wb") as f:
                f.truncate(int(27464 * 1048576))          # the 26.8 GiB on-card share of the 32 GB file
            sel = [(0, "P100", 60, 16384, 0, "u"), (1, "P100", 60, 16384, 0, "u")]
            c, line = L.mtp_ctx_fit(m, self.FN, sel, 32768, "f16", False, True)
            self.assertEqual(c, 16384)
            self.assertIn("lowered from 32768", line)
            c, line = L.mtp_ctx_fit(m, self.FN, sel, 32768, "q8_0", False, True)   # q8_0 KV fits at 32k
            self.assertEqual((c, line), (32768, None))
            c, line = L.mtp_ctx_fit(m, self.FN, sel, 32768, "f16", True, True)     # the user's -c is kept, warned
            self.assertEqual(c, 32768)
            self.assertIn("DECISION !! ctx", line)
            c, line = L.mtp_ctx_fit(m, self.FN, sel[:1], 32768, "f16", False, True)  # one card: the cache's job
            self.assertEqual(c, 32768)
            self.assertIn("expert cache plans", line)

    def test_counts_env_states(self):
        with tempfile.TemporaryDirectory() as d:
            ok = os.path.join(d, "m.csv")
            open(ok, "w").close()
            self.assertEqual(L.counts_env_state(ok)[0], "ok")
            self.assertEqual(L.counts_env_state(os.path.join(d, "new.csv"))[0], "write")
            dang = os.path.join(d, "dang.csv")
            os.symlink(os.path.join(d, "nowhere", "x.csv"), dang)
            self.assertEqual(L.counts_env_state(dang)[0], "bad")
            self.assertEqual(L.counts_env_state("")[0], "bad")
            self.assertEqual(L.counts_env_state("/proc/no-such/x.csv")[0], "bad")
            e = {"PXA_XCACHE_COUNTS": dang, "OTHER": "1"}
            self.assertIn("DROPPED", L.scrub_counts_env(e))
            self.assertNotIn("PXA_XCACHE_COUNTS", e)
            e = {"PXA_XCACHE_COUNTS": ok}
            self.assertIsNone(L.scrub_counts_env(e))
            self.assertEqual(e["PXA_XCACHE_COUNTS"], ok)

    def test_expert_map_under_either_name(self):
        with tempfile.TemporaryDirectory() as d:
            store, rel = os.path.join(d, "store"), os.path.join(d, "release")
            os.mkdir(store)
            os.mkdir(rel)
            real = os.path.join(store, "FN-32GB.relaid.gguf")
            open(real, "w").close()
            link = os.path.join(rel, "FN-32GB.gguf")
            os.symlink(real, link)
            csv = os.path.join(store, "FN-32GB.relaid.gguf.expert-counts.csv")
            open(csv, "w").close()
            self.assertEqual(L.overdrive_expert_map(link)[0], csv)          # map named after the real file
            os.unlink(csv)
            csv2 = link + ".expert-counts.csv"
            open(csv2, "w").close()
            self.assertEqual(L.overdrive_expert_map(link)[0], csv2)
            os.symlink(real, os.path.join(store, "FN-32GB.gguf"))             # link beside the real file
            os.unlink(csv2)
            csv3 = os.path.join(store, "FN-32GB.gguf.expert-counts.csv")
            open(csv3, "w").close()
            self.assertEqual(L.overdrive_expert_map(real)[0], csv3)
            # another model's map in the cache dir is never claimed
            cache = os.path.join(d, "cache")
            os.mkdir(cache)
            open(os.path.join(cache, "other.gguf.expert-counts.csv"), "w").close()
            os.unlink(csv3)
            with mock_env(PXA_CACHE_DIR=cache):
                self.assertIsNone(L.overdrive_expert_map(real)[0])

    def test_box_decisions_his_box_flashnext(self):
        with tempfile.TemporaryDirectory() as d:
            for n, txt in (("topo.txt", CTG_TOPO), ("lscpu.txt", CTG_LSCPU), ("allowed.txt", "0-31"),
                           ("env.txt", "container=1\ncap_sys_nice=0\ncache_dir=/work/.cache/pxa\ncache_mount=/\n")):
                with open(os.path.join(d, n), "w") as f:
                    f.write(txt)
            m = os.path.join(d, "fn.gguf")
            open(m, "w").close()
            with mock_env(PXA_LAUNCH_FAKE_TOPO=d, PXA_LAUNCH_FAKE_TEMPS="0:83,1:45"):
                one = "\n".join(L.box_decisions(m, [0], self.FN, True))
                two = "\n".join(L.box_decisions(m, [0, 1], self.FN, True))
                dense = L.box_decisions(m, [0], {"arch": "qwen35"}, False)
        self.assertIn("DECISION numa: card(s) 0 on node 0 of 2", one)
        self.assertIn("PXA_XCACHE_CPUPOOL_CPUS=" + L._cpuset_to_list(set(range(0, 31, 2))), one)
        self.assertIn("DECISION cpu-lib: compat (this CPU has no AVX2)", one)
        self.assertIn("DECISION !! cache-volume", one)
        self.assertIn("-v pxa-cache:/work/.cache/pxa", one)
        self.assertIn("DECISION !! sys-nice", one)
        self.assertIn("DECISION !! heat: card 0 83 C", one)
        self.assertIn("DECISION !! cards-span-sockets", two)
        # another model: only what is wrong (the hot card), nothing about pools or maps
        self.assertEqual([x.split(":")[0] for x in dense], ["DECISION !! heat"])

    def test_od_and_decision_lines_reach_the_plan_dict(self):
        text = ("  OVERDRIVE: this file is the measured Flash-Next class\n"
                "  DECISION kv: -ctk q8_0 -ctv q8_0 -- why   [override: --ctk]\n"
                "  DECISION !! heat: card 0 83 C -- hot\n  not a decision\n")
        self.assertEqual(C.decision_lines(text), ["kv: -ctk q8_0 -ctv q8_0 -- why [override: --ctk]",
                                                  "!! heat: card 0 83 C -- hot"])
        self.assertEqual(C.decision_lines(None), [])


# ---- storage guard: fake sysfs trees (2026-10-13) -------------------------------------------
class FakeSys:
    """A sysfs tree under a temp dir, laid out the way a kernel does: devices under devices/<bus>/block/<name>,
    /dev/block/<maj:min> and /class/block/<name> as RELATIVE symlinks to them, partitions inside their disk's
    directory (no queue/ of their own), stacks (md, dm) under devices/virtual/block with slaves/<member> symlinks."""

    def __init__(self, root):
        self.root = root
        self.dev = {}                       # name -> directory

    def _link(self, name, maj, mnr, d):
        for sub, leaf in (("dev/block", "%d:%d" % (maj, mnr)), ("class/block", name)):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)
            lp = os.path.join(self.root, sub, leaf)
            os.symlink(os.path.relpath(d, os.path.dirname(lp)), lp)
        self.dev[name] = d

    def _flag(self, d, rot):
        if rot is not None:
            os.makedirs(os.path.join(d, "queue"), exist_ok=True)
            with open(os.path.join(d, "queue", "rotational"), "w") as f:
                f.write("%d\n" % rot)

    def disk(self, name, rot, maj, mnr, bus="pci0000:00/0000:00:1f.2/ata1/host0/target0:0:0/0:0:0:0"):
        d = os.path.join(self.root, "devices", bus, "block", name)
        os.makedirs(d)
        self._flag(d, rot)
        self._link(name, maj, mnr, d)
        return d

    def part(self, disk, name, maj, mnr, n=1):
        d = os.path.join(self.dev[disk], name)
        os.makedirs(d)
        with open(os.path.join(d, "partition"), "w") as f:
            f.write("%d\n" % n)
        self._link(name, maj, mnr, d)
        return d

    def stack(self, name, maj, mnr, members, own_flag=None, dm_name=None):
        d = os.path.join(self.root, "devices", "virtual", "block", name)
        os.makedirs(os.path.join(d, "slaves"))
        for m in members:
            os.symlink(os.path.relpath(self.dev[m], os.path.join(d, "slaves")), os.path.join(d, "slaves", m))
        self._flag(d, own_flag)
        if dm_name:
            os.makedirs(os.path.join(d, "dm"))
            with open(os.path.join(d, "dm", "name"), "w") as f:
                f.write(dm_name + "\n")
        self._link(name, maj, mnr, d)
        return d

    def loop(self, name, maj, mnr, backing):
        d = os.path.join(self.root, "devices", "virtual", "block", name)
        os.makedirs(os.path.join(d, "loop"))
        with open(os.path.join(d, "loop", "backing_file"), "w") as f:
            f.write(backing + "\n")
        self._flag(d, 0)                    # a loop device says "not rotational" whatever is under it
        self._link(name, maj, mnr, d)


def mi(mount, majmin, fstype, source, root="/"):
    """one /proc/self/mountinfo line"""
    return "%d 1 %s %s %s rw,noatime shared:1 - %s %s rw" % (hash(mount) % 900 + 30, majmin, root, mount, fstype, source)


class StorageGuard(unittest.TestCase):
    """The model (and the per-layer table in it) on a rotational disk: sysfs walk, md / LVM / LUKS / loop,
    docker bind mounts and overlay roots, an Unraid user share, and the DECISION / doctor / bug-report lines."""
    FN = {"arch": "qwen4exp", "is_moe": True, "deltanet": True, "n_expert": 512, "tier": "PXQ_UNIVERSAL",
          "mtp_tensors": 6, "ple_tensor": "per_layer_token_embd.weight", "ple_bytes": int(51.2e9 * 34 / 32),
          "ple_elems": 51200245760}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxa-storage-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sys = FakeSys(os.path.join(self.tmp, "sys"))
        for k in (L.STORAGE_KIND_ENV, L.STORAGE_FAKE_ENV, L.PLE_DIRECT_ENV, L.PLE_DIRECT_CACHE_ENV):
            if k in os.environ:
                old = os.environ.pop(k)
                self.addCleanup(os.environ.__setitem__, k, old)

    def st(self, path, *lines, fsroot=""):
        return L.storage_of_path(path, sysroot=self.sys.root, mountinfo="\n".join(lines), fsroot=fsroot)

    def test_sata_hdd_partition_is_judged_by_its_disk(self):
        self.sys.disk("sda", 1, 8, 0)
        self.sys.part("sda", "sda1", 8, 1)
        r = self.st("/data/m/x.gguf", mi("/", "8:1", "ext4", "/dev/root"), mi("/data", "8:1", "xfs", "/dev/sda1"))
        self.assertEqual((r["kind"], r["rotational"], r["devices"]), ("hdd", True, ["sda"]))
        self.assertEqual(r["fs"], "xfs")

    def test_nvme_through_a_btrfs_mount_with_an_anonymous_device_number(self):
        self.sys.disk("nvme1n1", 0, 259, 0, bus="pci0000:80/0000:80:01.1/0000:82:00.0/nvme/nvme1")
        self.sys.part("nvme1n1", "nvme1n1p1", 259, 1)
        # mountinfo says 0:45 (btrfs' anonymous number): the source names the device
        r = self.st("/fastdisk/m/x.gguf", mi("/", "8:1", "xfs", "/dev/sda1"), mi("/fastdisk", "0:45", "btrfs", "/dev/nvme1n1p1"))
        self.assertEqual((r["kind"], r["rotational"], r["devices"]), ("nvme", False, ["nvme1n1"]))

    def test_sata_ssd_is_ssd_not_nvme(self):
        self.sys.disk("sdb", 0, 8, 16)
        self.sys.part("sdb", "sdb1", 8, 17)
        r = self.st("/s/x", mi("/s", "8:17", "ext4", "/dev/sdb1"))
        self.assertEqual((r["kind"], r["rotational"]), ("ssd", False))

    def test_md_over_hdds_is_rotational_whatever_the_stack_flag_says(self):
        self.sys.disk("sdc", 1, 8, 32)
        self.sys.disk("sdd", 1, 8, 48)
        self.sys.stack("md0", 9, 0, ["sdc", "sdd"], own_flag=0)       # the md device's own flag is wrong; the members are the truth
        r = self.st("/raid/x", mi("/raid", "9:0", "xfs", "/dev/md0"))
        self.assertEqual((r["kind"], r["rotational"], r["devices"]), ("hdd", True, ["sdc", "sdd"]))
        self.assertEqual(r["chain"], "md0 -> sdc, sdd")

    def test_md_over_ssds_and_mixed_stacks(self):
        self.sys.disk("sde", 0, 8, 64)
        self.sys.disk("sdf", 0, 8, 80)
        self.sys.disk("sdg", 1, 8, 96)
        self.sys.stack("md1", 9, 1, ["sde", "sdf"], own_flag=1)       # flag says rotational, members are flash
        self.sys.stack("md2", 9, 2, ["sde", "sdg"])
        self.assertEqual(self.st("/a/x", mi("/a", "9:1", "xfs", "/dev/md1"))["kind"], "ssd")
        m = self.st("/b/x", mi("/b", "9:2", "xfs", "/dev/md2"))
        self.assertEqual((m["kind"], m["rotational"]), ("mixed", True))

    def test_lvm_by_mapper_source_and_luks_over_nvme(self):
        self.sys.disk("sdh", 1, 8, 112)
        self.sys.part("sdh", "sdh2", 8, 114, n=2)
        self.sys.stack("dm-0", 253, 0, ["sdh2"], dm_name="vg-models")       # LVM on an HDD partition
        self.sys.disk("nvme0n1", 0, 259, 5, bus="pci0000:00/0000:00:03.1/0000:05:00.0/nvme/nvme0")
        self.sys.part("nvme0n1", "nvme0n1p3", 259, 8, n=3)
        self.sys.stack("dm-1", 253, 1, ["nvme0n1p3"], dm_name="luks-abc")   # LUKS on NVMe
        # the mount carries an anonymous number: only the /dev/mapper name identifies the device
        r = self.st("/lvm/x", mi("/lvm", "0:60", "btrfs", "/dev/mapper/vg-models"))
        self.assertEqual((r["kind"], r["devices"]), ("hdd", ["sdh"]))
        r = self.st("/enc/x", mi("/enc", "253:1", "ext4", "/dev/mapper/luks-abc"))
        self.assertEqual((r["kind"], r["devices"]), ("nvme", ["nvme0n1"]))

    def test_docker_bind_mount_gives_the_host_disk_and_an_overlay_root_does_not(self):
        self.sys.disk("sdi", 1, 8, 128)
        self.sys.part("sdi", "sdi1", 8, 129)
        lines = (mi("/", "0:98", "overlay", "overlay"), mi("/models", "8:129", "xfs", "/dev/sdi1", root="/srv/models"))
        r = self.st("/models/x.gguf", *lines)
        self.assertEqual((r["kind"], r["devices"]), ("hdd", ["sdi"]))
        o = self.st("/work/x.gguf", *lines)                   # inside the image / container layer
        self.assertEqual((o["kind"], o["rotational"], o["chain"]), ("unknown", None, "overlay"))
        self.assertIn("bind-mount", o["why"])

    def test_unraid_user_share_resolves_to_the_disk_that_holds_the_file(self):
        self.sys.disk("sdq", 1, 65, 0)
        self.sys.stack("md3p1", 9, 3, [], own_flag=1)         # Unraid's md driver: no member list, the array flag only
        self.sys.disk("nvme2n1", 0, 259, 9)
        self.sys.part("nvme2n1", "nvme2n1p1", 259, 10)
        U = L._UNRAID
        fsroot = os.path.join(self.tmp, "fs")
        for d in ("mnt/user/models", "mnt/disk3/models", "mnt/cache/models", "mnt/disk4/models"):
            os.makedirs(os.path.join(fsroot, d))
        for f in ("mnt/disk3/models/hdd.gguf", "mnt/cache/models/fast.gguf"):
            open(os.path.join(fsroot, f), "w").close()
        lines = (mi(U + "/user", "0:42", "fuse.shfs", "shfs"), mi(U + "/disk3", "9:3", "xfs", "/dev/md3p1"),
                 mi(U + "/cache", "259:10", "btrfs", "/dev/nvme2n1p1"), mi(U + "/disk4", "9:3", "xfs", "/dev/md3p1"))
        h = self.st(U + "/user/models/hdd.gguf", *lines, fsroot=fsroot)
        self.assertEqual((h["kind"], h["chain"]), ("hdd", "shfs -> md3p1"))
        f = self.st(U + "/user/models/fast.gguf", *lines, fsroot=fsroot)
        self.assertEqual(f["kind"], "nvme")
        n = self.st(U + "/user/models/nowhere.gguf", *lines, fsroot=fsroot)
        self.assertEqual(n["kind"], "unknown")
        self.assertIn("pool path", n["why"])

    def test_loop_device_follows_its_backing_file(self):
        self.sys.disk("sdj", 1, 8, 144)
        self.sys.part("sdj", "sdj1", 8, 145)
        self.sys.loop("loop3", 7, 3, "/images/m.img")
        lines = (mi("/images", "8:145", "ext4", "/dev/sdj1"), mi("/loopimg", "7:3", "ext4", "/dev/loop3"))
        r = self.st("/loopimg/x.gguf", *lines)
        self.assertEqual((r["kind"], r["devices"]), ("hdd", ["sdj"]))
        self.assertIn("loop device", r["why"])

    def test_ram_and_network_and_unreadable_never_raise(self):
        self.assertEqual(self.st("/t/x", mi("/t", "0:30", "tmpfs", "tmpfs"))["kind"], "ram")
        n = self.st("/n/x", mi("/n", "0:31", "nfs4", "srv:/export"))
        self.assertEqual((n["kind"], n["rotational"]), ("unknown", None))
        self.assertIn("network", n["why"])
        self.assertEqual(self.st("/q/x", mi("/q", "8:99", "ext4", "/dev/sdz9"))["kind"], "unknown")   # no such device in sysfs
        self.assertEqual(self.st("/q/x", "garbage line\n\n")["kind"], "unknown")
        self.assertEqual(self.st("/q/x")["kind"], "unknown")                                          # empty mountinfo

    def test_the_real_box_answers_without_raising(self):
        for p in (FAKE_MODEL, "/", "/proc", "/no/such/file.gguf"):
            r = L.storage_of_path(p)
            self.assertIn(r["kind"], ("nvme", "ssd", "hdd", "mixed", "ram", "network", "unknown"))

    # ---- the decision lines ----------------------------------------------------------------------
    def probe_for(self, kind):
        return lambda p: {"kind": kind, "rotational": {"hdd": True, "mixed": True, "ssd": False, "nvme": False}.get(kind),
                          "fs": "xfs", "source": "/dev/x", "devices": ["x"], "chain": "md15p1" if kind == "hdd" else "nvme1n1",
                          "why": "queue/rotational=1" if kind == "hdd" else "queue/rotational=0"}

    def decide(self, kind, prof=None, overdrive=True, model=None):
        with mock_env(PXA_LAUNCH_FAKE_GPUS="1x600"):
            return L.storage_decisions(model or FAKE_MODEL, prof or self.FN, overdrive, probe=self.probe_for(kind))

    def test_hdd_warns_with_what_to_do(self):
        out = self.decide("hdd")
        self.assertEqual(len(out), 1)
        line = out[0]
        self.assertTrue(line.startswith("DECISION !! ple-storage: "))
        for want in ("54.4 GiB" if False else "50.7 GiB", "md15p1", "rotational", "seek", "5x", "Move the model file", "SSD / NVMe",
                     "PXA_PLE_MMAP=2", "O_DIRECT", "Q5_0", "[override: PXA_STORAGE_KIND=ssd"):
            self.assertIn(want, line)
        # the same line reaches --doctor (as a problem) and Control's plan (decision_lines)
        self.assertEqual(C.decision_lines("  " + line)[0].split(":")[0], "!! ple-storage")

    def test_mixed_stack_warns_too_and_other_models_stay_silent(self):
        self.assertIn("some members rotational", self.decide("mixed")[0])
        self.assertEqual(self.decide("hdd", prof={"arch": "qwen35", "ple_tensor": None}), [])         # no per-layer table
        small = dict(self.FN, ple_bytes=int(1.5 * 1024 ** 3), ple_elems=1000)
        self.assertEqual(self.decide("hdd", prof=small), [])                                          # under 4 GiB: the page cache holds it
        self.assertEqual(L.storage_decisions("", self.FN, True, probe=self.probe_for("hdd")), [])

    def test_ssd_is_named_for_overdrive_and_silent_otherwise(self):
        line = self.decide("nvme")[0]
        self.assertTrue(line.startswith("DECISION ple-storage: "))
        self.assertNotIn("!!", line)
        self.assertIn("NVMe", line)
        self.assertEqual(self.decide("ssd", overdrive=False), [])

    def test_unknown_disk_is_said_not_guessed(self):
        out = self.decide("unknown")
        self.assertEqual(len(out), 1)
        self.assertNotIn("!!", out[0])
        self.assertIn("cannot tell", out[0])
        self.assertIn("PXA_STORAGE_KIND=ssd|hdd", out[0])

    def test_the_users_word_beats_a_lying_virtual_disk(self):
        with mock_env(PXA_STORAGE_KIND="ssd", PXA_LAUNCH_FAKE_GPUS="1x600"):
            st = L.storage_of_model(FAKE_MODEL, self.probe_for("hdd"))
            out = L.storage_decisions(FAKE_MODEL, self.FN, True, probe=self.probe_for("hdd"))
        self.assertEqual((st["kind"], st["rotational"]), ("ssd", False))
        self.assertIn("forced by PXA_STORAGE_KIND=ssd", st["why"])
        self.assertTrue(out and "!!" not in out[0] and "forced by PXA_STORAGE_KIND" in out[0])
        with mock_env(PXA_STORAGE_KIND="hdd", PXA_LAUNCH_FAKE_GPUS="1x600"):
            self.assertIn("!! ple-storage", L.storage_decisions(FAKE_MODEL, self.FN, True, probe=self.probe_for("nvme"))[0])

    # ---- PXA_PLE_DIRECT: the table stops counting as resident RAM (the PLE RAM pass) -----------------------------------
    def test_ple_direct_mode_reads_the_env_like_the_engine(self):
        for v, want in ((None, 0), ("", 0), ("0", 0), ("1", 1), ("2", 2), ("on", 0), (" 1", 1), ("-1", 0)):
            self.assertEqual(L.ple_direct_mode({} if v is None else {L.PLE_DIRECT_ENV: v}), want, v)

    def test_ram_accounting_lever_off_counts_the_whole_table(self):
        full = self.FN["ple_bytes"]
        for kind in (None, "nvme", "hdd"):
            self.assertEqual(L.ple_host_ram_bytes(self.FN, kind, {}), full)
        self.assertEqual(L.ple_host_ram_bytes({"arch": "qwen35", "ple_tensor": None}, "nvme", {L.PLE_DIRECT_ENV: "1"}), 0)
        self.assertEqual(L.ple_host_ram_bytes(dict(self.FN, ple_bytes=0, ple_elems=1000), None, {}), int(1000 * 0.5625))

    def test_ram_accounting_lever_on_counts_the_row_cache_only(self):
        on = {L.PLE_DIRECT_ENV: "1"}
        for kind in ("nvme", "ssd", "ram"):
            self.assertEqual(L.ple_host_ram_bytes(self.FN, kind, on), 256 * 1048576, kind)
        self.assertEqual(L.ple_host_ram_bytes(self.FN, "nvme", dict(on, **{L.PLE_DIRECT_CACHE_ENV: "512"})), 512 * 1048576)
        self.assertEqual(L.ple_host_ram_bytes(self.FN, "nvme", dict(on, **{L.PLE_DIRECT_CACHE_ENV: "junk"})), 256 * 1048576)
        # the engine refuses on these: the RAM path stays and the whole table is counted (the lab mode skips the refusals)
        for kind in ("hdd", "mixed", "network", "unknown", None):
            self.assertEqual(L.ple_host_ram_bytes(self.FN, kind, on), self.FN["ple_bytes"], kind)
        self.assertEqual(L.ple_host_ram_bytes(self.FN, "hdd", {L.PLE_DIRECT_ENV: "2"}), 256 * 1048576)
        tiny = dict(self.FN, ple_bytes=100 * 1048576)
        self.assertEqual(L.ple_host_ram_bytes(tiny, "nvme", on), 100 * 1048576)          # never more than the table

    def test_decision_line_says_how_the_table_is_counted(self):
        with mock_env(PXA_PLE_DIRECT="1"):
            out = self.decide("nvme")
            self.assertEqual(len(out), 2)
            self.assertTrue(out[1].startswith("DECISION ple-ram: ") and "NOT counted as resident RAM" in out[1] and "0.2 GiB" in out[1])
            self.assertEqual(C.decision_lines("  " + out[1])[0].split(":")[0], "ple-ram")
            bad = self.decide("hdd")
            self.assertEqual(len(bad), 2)
            self.assertTrue(bad[1].startswith("DECISION !! ple-ram: ") and "still counted as resident RAM" in bad[1])
            self.assertEqual(len(self.decide("nvme", overdrive=False)), 1)                 # ple-ram only; the ple-storage line stays Overdrive-only
            self.assertEqual(self.decide("hdd", prof={"arch": "qwen35", "ple_tensor": None}), [])
        with mock_env(PXA_PLE_DIRECT="0"):
            self.assertEqual(len(self.decide("nvme")), 1)                                   # lever off: no new line
            self.assertEqual(len(self.decide("hdd")), 1)

    def test_every_shard_is_checked_and_the_worst_wins(self):
        d = os.path.join(self.tmp, "m")
        os.makedirs(d)
        shards = [os.path.join(d, "fn-%05d-of-00002.gguf" % i) for i in (1, 2)]
        for sp in shards:
            open(sp, "w").close()
        seen = []

        def probe(p):
            seen.append(p)
            return self.probe_for("hdd" if "-00002-of-" in p else "nvme")(p)
        with mock_env(PXA_LAUNCH_FAKE_GPUS="1x600"):
            st = L.storage_of_model(shards[0], probe)
        self.assertEqual(seen, shards)
        self.assertEqual((st["kind"], st["shards"], st["worst"]), ("hdd", 2, shards[1]))

    def test_dry_run_fixtures_read_nothing_and_the_fake_env_forces_a_kind(self):
        with mock_env(PXA_LAUNCH_FAKE_GPUS="2x600"):
            self.assertIsNone(L.storage_of_model(FAKE_MODEL))
            self.assertEqual(L.storage_decisions(FAKE_MODEL, self.FN, True), [])
            self.assertIsNone(L.storage_row(FAKE_MODEL))
            self.assertIsNone(L.storage_summary(FAKE_MODEL))
        with mock_env(PXA_LAUNCH_FAKE_STORAGE="hdd", PXA_LAUNCH_FAKE_GPUS="2x600"):
            self.assertIn("!! ple-storage", "\n".join(L.box_decisions(FAKE_MODEL, [0], self.FN, True)))

    def test_doctor_row_and_bug_report_field(self):
        row = L.storage_row(FAKE_MODEL, self.probe_for("hdd"))
        self.assertIn("model on md15p1, ROTATIONAL", row)
        self.assertIn("queue/rotational=1", row)
        self.assertIn("NVMe", L.storage_row(FAKE_MODEL, self.probe_for("nvme")))
        summ = L.storage_summary(FAKE_MODEL, self.probe_for("hdd"))
        self.assertEqual((summ["kind"], summ["rotational"], summ["chain"]), ("hdd", True, "md15p1"))
        self.assertNotIn(MODELS, json.dumps(summ))                  # a device chain, never a path
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.req = {"model": FAKE_MODEL}
        with mock_env(PXA_LAUNCH_FAKE_STORAGE="hdd"):                 # this suite runs on fake cards: the fixture env names the disk
            b = app.report_bundle()["bundle"]
        self.assertIn("storage", b)
        self.assertEqual((b["storage"]["kind"], b["storage"]["rotational"]), ("hdd", True))
        self.assertNotIn(MODELS, json.dumps(b["storage"]))
        app.seat.req = {}
        self.assertIsNone(app.report_bundle()["bundle"]["storage"])

    def test_q5_0_and_q4_0_tables_are_sized_by_the_profile(self):
        """a revision-2 Flash-Next file (PLE at Q5_0 / Q4_0) must still subtract its table from the on-card share"""
        def gguf(path, ty, elems):
            def st(x):
                b = x.encode()
                return struct.pack("<Q", len(b)) + b
            h = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
            h += st("general.architecture") + struct.pack("<I", 8) + st("qwen4exp")
            h += st("per_layer_token_embd.weight") + struct.pack("<IQQIQ", 2, 160, elems // 160, ty, 0)
            with open(path, "wb") as f:
                f.write(h)
        el = 160 * 1000000
        for ty, bpe in ((8, 34 / 32), (6, 22 / 32), (2, 18 / 32)):
            p = os.path.join(self.tmp, "t%d.gguf" % ty)
            gguf(p, ty, el)
            prof = L.model_profile(p, "gguf")
            self.assertEqual((prof.get("ple_type"), prof.get("ple_bytes")), (ty, int(el * bpe)), ty)


@contextlib.contextmanager
def mock_env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update(kw)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Presets(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.app = C.App(L, port=7777, models_dirs=[MODELS])
        self.app.catalog_names = CATALOG

    def test_roundtrip_and_persistence(self):
        self.app.save_preset("1080ti chat", {"gpus": [1], "model": FAKE_MODEL, "kv": "q8_0",
                                             "levers": {"PXA_ENHANCE": "1"}})
        again = C.App(L, port=7777)
        p = again.presets()["1080ti chat"]
        self.assertEqual((p["gpus"], p["kv"], p["levers"]), ([1], "q8_0", {"PXA_ENHANCE": "1"}))
        mode = os.stat(C.config_path()).st_mode & 0o777
        self.assertEqual(mode, 0o600)
        self.assertTrue(again.delete_preset("1080ti chat"))
        self.assertNotIn("1080ti chat", C.App(L, port=7777).presets())

    def test_bad_names_and_bodies(self):
        for n in ("", "../x", "a" * 49, "semi;colon", None):
            with self.assertRaises(C.Invalid):
                self.app.save_preset(n, {"gpus": [0], "model": FAKE_MODEL})
        with self.assertRaises(C.Invalid):
            self.app.save_preset("ok", {"gpus": [0], "model": FAKE_MODEL, "levers": {"PATH": "/tmp"}})

    def test_model_dirs(self):
        self.assertEqual(self.app.set_model_dirs([MODELS]), [MODELS])
        self.assertEqual(C.load_config()["model_dirs"], [MODELS])
        with self.assertRaises(C.Invalid):
            self.app.set_model_dirs(["/definitely/not/here"])
        with self.assertRaises(C.Invalid):
            self.app.set_model_dirs("notalist")


class Handlers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        cls.app = C.App(L, port=7777, models_dirs=[MODELS])
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_static_and_info(self):
        code, h, body = req(self.port, "/")
        self.assertEqual(code, 200)
        self.assertIn(b"PXA Control", body)
        self.assertIn("frame-ancestors 'none'", h.get("Content-Security-Policy", ""))
        code, _, body = req(self.port, "/mark.png")
        self.assertEqual((code, body[:4]), (200, b"\x89PNG"))
        code, _, body = req(self.port, "/api/info")
        d = json.loads(body)
        self.assertEqual((code, d["lan"], d["kv_types"][0]), (200, False, "auto"))

    def test_status_rig_presets(self):
        self.assertEqual(json.loads(req(self.port, "/api/status")[2])["running"], False)
        live = json.loads(req(self.port, "/api/rig/live")[2])
        self.assertEqual([c["index"] for c in live["cards"]], [0, 1])
        self.assertEqual(live["cards"][0]["sm"], 60)
        code, _, body = req(self.port, "/api/presets", "POST",
                            {"name": "p1", "settings": {"gpus": [0, 1], "model": FAKE_MODEL, "sm": "tensor"}})
        self.assertEqual(code, 200, body)
        self.assertEqual(json.loads(req(self.port, "/api/presets")[2])["p1"]["sm"], "tensor")
        code, _, body = req(self.port, "/api/presets", "DELETE", {"name": "p1"})
        self.assertEqual((code, json.loads(body)["deleted"]), (200, True))

    def test_validation_is_400(self):
        code, _, body = req(self.port, "/api/plan", "POST", {"gpus": [0], "model": FAKE_MODEL,
                                                             "levers": {"LD_PRELOAD": "/x.so"}})
        self.assertEqual(code, 400)
        self.assertIn("LD_PRELOAD", json.loads(body)["error"])
        code, _, _ = req(self.port, "/api/start", "POST", {"gpus": [9], "model": FAKE_MODEL})
        self.assertEqual(code, 400)
        code, _, _ = req(self.port, "/api/plan", "POST", None, headers={"Content-Length": "0"})
        self.assertEqual(code, 400)

    def test_guards(self):
        self.assertEqual(req(self.port, "/api/info", host="evil.example:7777")[0], 421)
        self.assertEqual(req(self.port, "/api/info", host=f"localhost:{self.port}")[0], 200)
        code, _, _ = req(self.port, "/api/stop", "POST", {}, headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, _, _ = req(self.port, "/api/stop", "POST", {}, headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(code, 200)
        self.assertEqual(req(self.port, "/api/nope")[0], 404)
        self.assertEqual(req(self.port, "/api/engine/slots/save", "POST", {})[0], 404)   # not proxied
        self.assertEqual(req(self.port, "/api/engine/../../etc/passwd")[0], 404)

    def test_engine_proxy_without_server(self):
        C.save_config(dict(C.load_config(), attach_port=0))
        self.assertEqual(req(self.port, "/api/engine/health")[0], 503)
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": 7777})[0], 400)   # its own port
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": 70000})[0], 400)
        self.assertEqual(req(self.port, "/api/bench", "POST", {})[0], 400)                 # no server


class StubEngine(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            b = b'{"status":"ok"}'
            self.send_response(200)
        elif self.path.startswith("/pxa/stats"):
            b = json.dumps({"count": 1, "records": [{"ts": time.time(), "model": "m", "decode_tps": 50.0,
                                                     "prefill_tps": 900.0, "n_prompt": 100}]}).encode()
            self.send_response(200)
        else:
            b = b'{"error":"nf"}'
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in ("hel", "lo"):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": w}}]}).encode() + b"\n\n")
            self.wfile.flush()
        last = {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "stub-model",
                "timings": {"prompt_n": 7, "predicted_n": 2, "prompt_per_second": 321.0,
                            "predicted_per_second": 42.5}, "echo_stream": body.get("stream")}
        self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\ndata: [DONE]\n\n")


class EngineProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        cls.stub = socketserver.ThreadingTCPServer(("127.0.0.1", 0), StubEngine)
        cls.stub.daemon_threads = True
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.app = C.App(L, port=7777)
        cls.srv, cls.port = serve_app(cls.app)
        code, _, body = req(cls.port, "/api/attach", "POST", {"port": cls.stub.server_address[1]})
        assert code == 200, body

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.stub.shutdown()

    def test_health_and_stats(self):
        st = json.loads(req(self.port, "/api/status")[2])
        self.assertEqual((st["health"], st["attached"]), ("ok", True))
        code, _, body = req(self.port, "/api/engine/pxa/stats?since=0")
        self.assertEqual((code, json.loads(body)["records"][0]["decode_tps"]), (200, 50.0))
        self.assertEqual(req(self.port, "/api/engine/props")[0], 404)     # passed through from the stub

    def test_stream_and_history(self):
        code, h, body = req(self.port, "/api/engine/v1/chat/completions", "POST",
                            {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("event-stream", h.get("Content-Type", ""))
        text = body.decode()
        self.assertIn('"hel"', text)
        self.assertIn("[DONE]", text)
        hist = json.loads(req(self.port, "/api/history?since=0")[2])["records"]
        self.assertTrue(any(r.get("decode_tps") == 42.5 and r.get("model") == "stub-model" for r in hist), hist)


class AttachedSeatInFleet(unittest.TestCase):
    """d7121574b5 (TucsonJohn, #general 2026-10-05): a seat ATTACHED to a server the launcher started (attach_port) serves, so the fleet must
    carry attached + its port and the Speed/Chat/Report pickers must not call it '(stopped)'. The fix shipped without a test."""

    @classmethod
    def setUpClass(cls):
        cls.stub = socketserver.ThreadingTCPServer(("127.0.0.1", 0), StubEngine)
        cls.stub.daemon_threads = True
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.app = C.App(L, port=7777)
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        req(cls.port, "/api/attach", "POST", {"port": 0})        # leave no attach_port behind for later classes
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.stub.shutdown()

    def main_row(self):
        code, _, body = req(self.port, "/api/fleet?force=1")
        self.assertEqual(code, 200, body)
        rows = [x for x in json.loads(body)["instances"] if x.get("key") == "m:main"]
        self.assertEqual(len(rows), 1, body)
        return rows[0]

    def test_attached_seat_is_attached_with_the_launcher_port(self):
        sport = self.stub.server_address[1]
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": sport})[0], 200)
        row = self.main_row()
        self.assertFalse(row["running"])                  # Control did not start it ...
        self.assertTrue(row["attached"])                  # ... but it is attached, so not "stopped"
        self.assertEqual(row["port"], sport)
        st = json.loads(req(self.port, "/api/status")[2])
        self.assertEqual((st["attached"], st["engine_port"]), (True, sport))   # the header and the pickers agree

    def test_detached_seat_is_not_attached(self):
        self.assertEqual(req(self.port, "/api/attach", "POST", {"port": 0})[0], 200)
        row = self.main_row()
        self.assertFalse(row["running"])
        self.assertFalse(row["attached"])

    def test_the_pickers_label_an_attached_seat_attached_not_stopped(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            html = f.read()
        self.assertIn('x.attached ? " (attached)" : " (stopped)"', html)               # Speed / Report picker
        self.assertIn('x.attached ? " :" + x.port + " (attached)" : " (stopped)"', html)  # Chat targets
        self.assertIn("x.running || x.attached", html)                                    # an attached seat is picked as live


class HeaderShowsEveryServer(unittest.TestCase):
    """cizettabg, #feature-request 2026-10-11: "All running servers displayed in the header. Currently only the
    last one started is displayed." The header carried one pill for the server this page is pointed at, refreshed
    only while the Servers tab was open, with a bare "(+N serving)" for the rest. It now carries a chip per
    running server on every tab, and collapses to one chip when they do not fit or on a phone.

    The browser behaviour is checked against the fake workload by tests/header-gui-check.py; what is pinned here
    is the contract in the source: the header reads the fleet on every tab, a chip writes name + port + state,
    and the chip, the card and the Live tile read one state function."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            cls.html = f.read()

    def test_the_header_carries_a_chip_row(self):
        self.assertIn('<span class="fchips" id="fchips"', self.html)
        self.assertLess(self.html.index('id="fchips"'), self.html.index('id="health"'), "the chips sit beside the pill")

    def test_a_chip_writes_name_port_and_state(self):
        for frag in ('class: "fname"', 'class: "fport mono"', 'class: "fst"'):
            self.assertIn(frag, self.html, frag)                      # a chip is never a colour with no words on it

    def test_the_fleet_is_read_on_every_tab_not_only_the_servers_tab(self):
        self.assertIn("setInterval(() => { if (!document.hidden) loadFleet(); }, 5000)", self.html)
        self.assertNotIn('S.tab === "servers") loadFleet()', self.html, "the old Servers-tab-only refresh is gone")

    def test_the_chip_the_card_and_the_tile_read_one_state_function(self):
        self.assertIn("function srvPhase(x)", self.html)
        self.assertIn("const phase = srvPhase(x);", self.html)         # the card
        self.assertIn("const st = srvPhase(x)", self.html)             # the chip

    def test_a_chip_opens_that_server_on_the_servers_tab(self):
        self.assertIn("function openFleetCard(x)", self.html)
        self.assertIn('"data-key": x.key', self.html, "the card the chip jumps to can be found by its key")

    def test_the_chips_collapse_when_they_do_not_fit_and_on_a_phone(self):
        self.assertIn('matchMedia("(max-width:820px)")', self.html)
        self.assertIn("box.scrollWidth <= box.clientWidth + 1", self.html, "fitting is measured, not guessed")
        self.assertIn('class: "fchip fmore"', self.html, "the collapsed form is one chip that says how many there are")
        self.assertIn('class: "fmenu"', self.html, "and it opens the list")

    def test_a_header_with_no_room_steps_the_row_aside_rather_than_paint_over_the_pill(self):
        self.assertIn('hd.classList.add("chipspill")', self.html, "the pill stands down for the row first")
        self.assertIn(".hd.chipspill .pill{display:none}", self.html)
        self.assertIn('box.style.display = "none"', self.html, "and the row stands down if even that is not enough")
        self.assertIn('const menu = box.querySelector(".fmenu");', self.html)
        self.assertIn('if (menu) menu.style.display = "none";', self.html, "an open list is not part of what a row costs")


class FooterOnEveryPage(unittest.TestCase):
    """#feature-request FR#15 (2026-10-26): the footer with the Discord link, the Ko-fi link, the PXA version and the Update
    button showed only on the Servers page. It is now one footer of the page shell: the last child of <main>, after every
    page's section, so every page (Home first) shows it under its own content and a page that scrolls still reaches its last
    element above it. Plugin tabs add their sections before it. The browser side (every tab, both themes, a phone width) is
    tests/footer-gui-check.py."""

    @classmethod
    def setUpClass(cls):
        import html.parser
        ui = os.path.join(TOOLS, "pxa_control_ui")
        with open(os.path.join(ui, "index.html"), encoding="utf-8") as f:
            cls.html = f.read()
        with open(os.path.join(ui, "plugin-host.js"), encoding="utf-8") as f:
            cls.host = f.read()
        void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
        found = {"foots": [], "main_children": []}

        class P(html.parser.HTMLParser):
            def __init__(self):
                super().__init__()
                self.stack = []          # (tag, attrs) of the open elements
                self.cur = None          # the footer being read: its ids and hrefs

            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if self.stack and self.stack[-1][0] == "main":
                    found["main_children"].append((tag, a))
                if "foot" in (a.get("class") or "").split():
                    self.cur = {"parents": [t for t, _ in self.stack], "parent": self.stack[-1] if self.stack else None,
                                "tag": tag, "ids": [], "hrefs": [], "depth": len(self.stack)}
                    found["foots"].append(self.cur)
                elif self.cur is not None:
                    if a.get("id"):
                        self.cur["ids"].append(a["id"])
                    if a.get("href"):
                        self.cur["hrefs"].append(a["href"])
                if tag not in void:
                    self.stack.append((tag, a))

            def handle_endtag(self, tag):
                if tag in void:
                    return
                while self.stack:
                    t, _ = self.stack.pop()
                    if self.cur is not None and len(self.stack) == self.cur["depth"]:
                        self.cur = None
                    if t == tag:
                        break
        P().feed(cls.html)
        cls.foots, cls.main_children = found["foots"], found["main_children"]

    def test_there_is_one_footer_not_a_copy_per_page(self):
        self.assertEqual(len(self.foots), 1, "one footer for the whole app (no second one on Servers)")

    def test_the_footer_belongs_to_the_shell_not_to_the_servers_page(self):
        f = self.foots[0]
        self.assertNotIn("section", f["parents"], "the footer sat inside the Servers section: other pages never showed it")
        self.assertEqual(f["parent"][0], "main", "the footer is a child of <main>, shown with every page")

    def test_the_footer_comes_after_every_page(self):
        tags = [t for t, a in self.main_children]
        self.assertIn("section", tags)
        last_section = max(i for i, t in enumerate(tags) if t == "section")
        foot_at = [i for i, (t, a) in enumerate(self.main_children) if "foot" in (a.get("class") or "").split()]
        self.assertEqual(foot_at, [len(tags) - 1], "the footer is the last thing in <main>")
        self.assertGreater(foot_at[0], last_section)

    def test_the_footer_keeps_its_links_version_and_update_button(self):
        f = self.foots[0]
        self.assertIn("https://discord.pxanetwork.com", f["hrefs"])
        self.assertIn("https://ko-fi.com/shatteredrealms1", f["hrefs"])
        for i in ("sv-ver", "sv-up"):
            self.assertIn(i, f["ids"], i)
        self.assertEqual(self.html.count('id="sv-ver"'), 1)
        self.assertEqual(self.html.count('id="sv-up"'), 1)

    def test_a_plugin_tab_adds_its_page_before_the_footer(self):
        self.assertNotIn("main.append(sec)", self.host, "a plugin page appended to <main> would land under the footer")
        self.assertIn('main.insertBefore(sec, main.querySelector(":scope > .foot"))', self.host)


class TokenAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = C.App(L, port=7777, lan=True, token="s3cret-token")
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_random_token_when_lan(self):
        a = C.App(L, port=7777, lan=True)
        self.assertGreaterEqual(len(a.token), 20)
        # H1 (the Control hardening pass): this machine only gets a fresh token per session too, never the saved LAN one
        b, c = C.App(L, port=7777), C.App(L, port=7777)
        self.assertGreaterEqual(len(b.token), 32)
        self.assertNotEqual(b.token, c.token)
        self.assertNotEqual(b.token, a.token)
        self.assertIsNone(C.App(L, port=7777, token="").token)                 # the explicit test-only no-token mode

    def test_refused_without_token(self):
        self.assertEqual(req(self.port, "/api/info")[0], 401)
        code, _, body = req(self.port, "/")
        self.assertEqual(code, 401)
        self.assertIn(b"token", body)
        self.assertEqual(req(self.port, "/api/info", headers={"X-PXA-Token": "wrong"})[0], 401)
        self.assertEqual(req(self.port, "/api/info", headers={"Cookie": "pxa_control_token=nope"})[0], 401)

    def test_header_query_cookie(self):
        self.assertEqual(req(self.port, "/api/info", headers={"X-PXA-Token": "s3cret-token"})[0], 200)
        code, h, _ = req(self.port, "/?token=s3cret-token&x=1")
        self.assertEqual(code, 303)
        self.assertEqual(h.get("Location"), "/?x=1")
        self.assertIn("pxa_control_token_7777=s3cret-token", h.get("Set-Cookie", ""))     # H3: one cookie per port
        self.assertIn("HttpOnly", h.get("Set-Cookie", ""))
        self.assertEqual(req(self.port, "/api/info", headers={"Cookie": "a=b; pxa_control_token=s3cret-token"})[0], 200)
        # any Host is fine on a LAN bind (the token is the guard), cross-origin POST still refused
        self.assertEqual(req(self.port, "/api/info", host="192.168.1.5:7777",
                             headers={"X-PXA-Token": "s3cret-token"})[0], 200)
        self.assertEqual(req(self.port, "/api/stop", "POST", {}, headers={"X-PXA-Token": "s3cret-token",
                                                                         "Origin": "http://evil.example"})[0], 403)


class SeatProcess(unittest.TestCase):
    def built(self, code):
        plan = L.Plan()
        return (plan, [sys.executable, "-u", "-c", code], {"PXA_X": "1"}, "0", {}, 4096)

    def test_log_phase_exit(self):
        s = C.Seat(L)
        s.start(self.built("import os;print('llama_model_loader: x');print('server is listening');"
                           "print('env', os.environ.get('PXA_X'), os.environ.get('CUDA_VISIBLE_DEVICES'), "
                           "os.environ.get('PXA_ENHANCE'))"),
                {"port": 8999, "model": "m", "gpus": [0]}, {"PXA_ENHANCE": "0"})
        s.proc.wait(timeout=20)
        time.sleep(0.3)
        lines = [x[1] for x in s.lines_since(0)]
        self.assertTrue(lines[0].startswith("$ PXA_ENHANCE=0 "), lines[0])
        self.assertIn("env 1 0 0", lines)
        self.assertTrue(any("exited with code 0" in x for x in lines))
        self.assertEqual(s.status()["running"], False)
        self.assertEqual(s.lines_since(len(lines)), [])

    def test_stop_and_single_seat(self):
        s = C.Seat(L)
        s.start(self.built("import time\nprint('up', flush=True)\ntime.sleep(60)"),
                {"port": 8999, "model": "m", "gpus": [0]}, {})
        self.assertTrue(s.running())
        with self.assertRaises(C.Invalid):
            s.start(self.built("pass"), {"port": 8999, "model": "m", "gpus": [0]}, {})
        pid = s.proc.pid
        self.assertTrue(s.stop())
        self.assertFalse(s.running())
        self.assertFalse(os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z")
        self.assertFalse(s.stop())


class Redaction(unittest.TestCase):
    """the bug-report bundle leaves the machine only after redact_obj()."""
    H, U = {"rigbox"}, {"alice"}

    def r(self, s):
        return C.redact_text(s, self.H, self.U)

    def test_home_paths(self):
        home = os.path.expanduser("~")
        self.assertEqual(self.r(home + "/models/x.gguf"), "~/models/x.gguf")
        self.assertEqual(self.r("/home/bob/pxa/x"), "~/pxa/x")
        self.assertEqual(self.r("/Users/carol/Library/x"), "~/Library/x")
        self.assertEqual(self.r("C:\\Users\\dave\\m.gguf"), "~\\m.gguf")
        self.assertEqual(self.r("/" + "root" + "/m.gguf"), "~/m.gguf")

    def test_ips_hosts_users_mail(self):
        out = self.r("server 192.168.1.50:8080 from 10.0.0.7 and 127.0.0.1, host rigbox user alice mail a.b@example.com")
        for bad in ("192.168.1.50", "10.0.0.7", "rigbox", "alice", "example.com"):
            self.assertNotIn(bad, out)
        self.assertIn("127.0.0.1", out)
        self.assertIn("<ip>:8080", out)
        self.assertNotIn("fe80", self.r("addr fe80::1c2b:3aff:fe4d:5e6f end"))
        self.assertNotIn("aa:bb:cc:dd:ee:ff", self.r("mac aa:bb:cc:dd:ee:ff"))

    def test_secrets(self):
        cases = ["--api-key sk-abcdefghijklmnop1234", "--api-key=hunter2hunter2", "Authorization: Bearer abcdef0123456789xyz",
                 "HF_TOKEN=hf_" + "a" * 30, "token: abcdEFGH12345678", "password = 'p4ssw0rd!'",
                 "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc123def456",
                 "ghp_" + "Z" * 36, "MTIzNDU2Nzg5MDEyMzQ1Njc4" + "." + "GabcdE" + "." + "abcdefghijklmnopqrstuvwxyz0123",
                 "key AbCdEf0123456789AbCdEf0123456789xyz"]
        for c in cases:
            out = self.r(c)
            self.assertIn("<redacted", out, c)
            for frag in ("hunter2", "sk-abc", "abcdef0123456789xyz", "hf_aaaa", "abcdEFGH", "p4ssw0rd", "eyJhbGci", "ghp_ZZ",
                         "GabcdE", "AbCdEf0123456789AbCdEf"):
                self.assertNotIn(frag, out, c)

    def test_keeps_useful_diagnostics(self):
        for keep in ("PXA_TSPLIT: mode=tensor devs=2", "prompt eval time = 812.4 ms / 2048 tokens", "n_tokens = 512",
                     "llama_model_loader: loaded meta data with 41 key-value pairs", "CUDA error: out of memory",
                     "greedy512 sha 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "Tesla V100-SXM2-16GB cc 70",
                     "driver 535.104.05 CUDA 12.2"):
            self.assertEqual(self.r(keep), keep)

    def test_product_words_are_never_identity_words(self):
        # a rig named "pxa" (the docker service name, a Windows box called PXA) turned every PXA_REGISTRY line into
        # "<host>_REGISTRY": the bug intake then refused the report (VS beta 2, 10-19) and the score app read "<host>-control"
        for h in ({"pxa"}, {"PXA"}, {"pxa", "rigbox"}):
            for keep in ("PXA_REGISTRY: card 0 sm 60 class=p100", "PXA_AUTO: CTX=163840", "PXA_TSPLIT: mode=tensor devs=5",
                         "pxa-control", "(no PXA_REGISTRY / PXA_TSPLIT / PXA_AUTO lines in this seat's log yet)"):
                self.assertEqual(C.redact_text(keep, h, self.U), keep, (h, keep))
        self.assertNotIn("rigbox", C.redact_text("host rigbox PXA_AUTO: on", {"pxa", "rigbox"}, self.U))
        # a host or user word that is also a tag word is still redacted outside the PXA_* env names
        out = C.redact_text("PXA_AUTO: UB=512 on auto host", {"auto"}, {"registry"})
        self.assertTrue(out.startswith("PXA_AUTO: UB=512"), out)
        self.assertNotIn(" auto host", out)
        self.assertEqual(C.redact_text("PXA_REGISTRY: by registry", set(), {"registry"}), "PXA_REGISTRY: by <user>")
        old = os.environ.get("HOSTNAME")
        os.environ["HOSTNAME"] = "pxa"
        try:
            hosts, _u = C._identity_words()
        finally:
            os.environ.pop("HOSTNAME") if old is None else os.environ.__setitem__("HOSTNAME", old)
        self.assertNotIn("pxa", {x.lower() for x in hosts})

    def test_idempotent_and_recursive(self):
        o = {"a": ["/home/bob/x", {"k": "10.1.2.3"}], "n": 5, "log": ["--api-key abcdefgh12345"]}
        once = C.redact_obj(o, self.H, self.U)
        self.assertEqual(once, C.redact_obj(once, self.H, self.U))
        self.assertNotIn("bob", json.dumps(once))
        self.assertNotIn("10.1.2.3", json.dumps(once))
        self.assertEqual(once["n"], 5)


class ReportBundle(unittest.TestCase):
    def setUp(self):
        self.app = C.App(L, port=0, models_dirs=[MODELS])

    def test_gpu_telemetry_in_bundle(self):
        line = ("0, 83, 544, 1328, 715, 715, 121.50, 250.00, 0x0000000000000068\n"
                "1, 41, 1328, 1328, 715, 715, 30.00, 250.00, 0x1\n")
        real = self.app.L._run
        self.app.L._run = lambda argv, timeout=10: line if argv and argv[0] == "nvidia-smi" else real(argv, timeout=timeout)
        fake = os.environ.pop("PXA_LAUNCH_FAKE_GPUS")
        try:
            t = self.app.gpu_telemetry()
        finally:
            os.environ["PXA_LAUNCH_FAKE_GPUS"] = fake
            self.app.L._run = real
        self.assertEqual([r["index"] for r in t], [0, 1])
        self.assertEqual((t[0]["temp_c"], t[0]["sm_clock_mhz"], t[0]["power_limit_w"]), (83.0, 544.0, 250.0))
        self.assertEqual(sorted(t[0]["throttle_reasons"]), ["hw_slowdown", "hw_thermal_slowdown", "sw_thermal_slowdown"])
        self.assertEqual(t[1]["throttle_reasons"], [])   # idle only is not a throttle
        # the fake rig asks no nvidia-smi
        self.assertEqual(self.app.gpu_telemetry(), [])

    def test_heat_warning(self):
        w = C.App.heat_warning(2, 84.0, ["hw_thermal_slowdown"])
        self.assertEqual(w, "Card 2 is running hot (84 \u00b0C, thermal throttling): expect much lower speed. "
                            "Passive Tesla cards need forced airflow.")
        self.assertIn("80 \u00b0C", C.App.heat_warning(0, 80.0, []))
        self.assertIn("power throttling", C.App.heat_warning(1, None, ["sw_power_cap"]))
        self.assertIsNone(C.App.heat_warning(0, 79.0, []))
        self.assertIsNone(C.App.heat_warning(0, 40.0, ["gpu_idle"]))
        self.assertIn("heat_warning", self.app.rig_live()["cards"][0])

    def test_banner_is_kept_after_the_log_ring_drops_it(self):
        # an 18 min load + hours of traffic pushes the boot banner out of the 6000-line ring; the report still carries it
        self.app.seat._append("PXA_REGISTRY: card 0 sm 60 class=p100")
        self.app.seat._append("PXA_AUTO: CTX=163840 (no -c given)")
        for i in range(self.app.seat.log.maxlen + 50):
            self.app.seat._append("slot update_slots: id 0 | task %d | prompt done" % i)
        self.assertFalse(any("PXA_REGISTRY" in x[1] for x in self.app.seat.log))
        bu = self.app.report_bundle()["bundle"]
        self.assertEqual(bu["banner"][:2], ["PXA_REGISTRY: card 0 sm 60 class=p100", "PXA_AUTO: CTX=163840 (no -c given)"])
        for i in range(60):
            self.app.seat._append("PXA_AUTO: UB=%d fits" % i)
        self.assertEqual(len(self.app.seat.banner_lines), C.BANNER_KEEP)

    def test_bundle_shape_and_send_failure_is_reported(self):
        self.app.seat._append("PXA_REGISTRY: card 0 sm 60 class=p100")
        self.app.seat._append("build: 4711 (abc1234)")
        self.app.seat._append("using key sk-abcdefghijklmnop1234 at /home/bob/x")
        b = self.app.report_bundle()
        self.assertEqual((b["app"], b["version"]), ("pxa-control", C.CONTROL_VERSION))
        bu = b["bundle"]
        self.assertTrue(any("PXA_REGISTRY" in x for x in bu["banner"]))
        self.assertEqual(bu["engine"]["version"], "4711 (abc1234)")
        self.assertEqual(len(bu["gpus"]), 2)
        self.assertIn("gpu_telemetry", bu)
        self.assertNotIn("sk-abc", json.dumps(b))
        self.assertNotIn("/home/bob", json.dumps(b))
        old = os.environ.get("PXA_BUG_URL")
        os.environ["PXA_BUG_URL"] = "http://127.0.0.1:1/v1/report"     # nothing listens
        try:
            r = self.app.report_send({"payload": b})
        finally:
            os.environ.pop("PXA_BUG_URL") if old is None else os.environ.__setitem__("PXA_BUG_URL", old)
        self.assertFalse(r["ok"])
        self.assertIn("could not reach", r["error"])
        with self.assertRaises(C.Invalid):
            self.app.report_send({"payload": "{not json"})
        with self.assertRaises(C.Invalid):
            self.app.report_send({"payload": {"x": "y" * (C.REPORT_MAX + 10)}})


    def test_engine_version_survives_the_log_ring(self):
        seat = self.app.seat
        seat._append("build: 4711 (abc1234)")
        for i in range(seat.log.maxlen + 50):          # a chatty server pushes the build line out of the ring
            seat._append(f"slot update_slots: id 0 | task {i} | n_past = {i}")
        self.assertFalse(any("build:" in x[1] for x in seat.log))
        self.assertEqual(self.app.report_bundle()["bundle"]["engine"]["version"], "4711 (abc1234)")
        seat.log.clear()
        seat.build_lines = []

    def test_engine_version_from_the_binary_when_the_log_has_none(self):
        d = tempfile.mkdtemp(prefix="pxa-engver-")
        exe = os.path.join(d, "bin", "llama-server")
        os.makedirs(os.path.dirname(exe))
        with open(exe, "w") as f:
            f.write("#!/bin/sh\necho 'version: 9999 (deadbee)' >&2\necho 'built with cc for x86_64' >&2\n")
        os.chmod(exe, 0o755)
        seat = self.app.seat
        old = (seat.cmd, seat.req, list(seat.log), list(seat.build_lines))
        try:
            seat.cmd, seat.req = [exe, "-m", "x.gguf"], None
            seat.log.clear()
            seat.build_lines = []
            v = self.app.report_bundle()["bundle"]["engine"]["version"]
            self.assertEqual(v, "9999 (deadbee) (from --version)")
            self.assertIn("9999 (deadbee)", self.app._engine_ver_cache.values())     # cached per binary + mtime
        finally:
            seat.cmd, seat.req = old[0], old[1]
            seat.log.clear()
            seat.log.extend(old[2])
            seat.build_lines = old[3]
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(C.App.parse_engine_version("main: x\nbuild: 7046 (1a2b3c4) with gcc"), "7046 (1a2b3c4) with gcc")
        self.assertIsNone(C.App.parse_engine_version("no version here"))


class ScoreBoardClient(unittest.TestCase):
    def test_name_filter_matches_board(self):
        for bad in ("http://x.example", "see example.com", "@everyone", "f u c k", "sh1t", "<b>", ""):
            self.assertEqual(C.clean_board_name(bad), "", bad)
        self.assertEqual(C.clean_board_name("  Pat O'Neil "), "Pat O'Neil")
        self.assertEqual(len(C.clean_board_name("y" * 90)), 32)

    def test_payload_shape_and_redaction(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "-sm", "tensor", "--api-key", "sk-abcdefghijklmnop1234"]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [0, 1], "port": 8999}
        res = {"ts": 1.0, "model": FAKE_MODEL, "cards": [0, 1], "reps": 3, "greedy512_sha": "ab" * 32,
               "classes": [{"class": "prose", "decode_tps": 41.5, "prefill_tps": 0, "n_prompt": 20, "decode_all": [41, 42], "prefill_all": []},
                           {"class": "long", "decode_tps": 30.0, "prefill_tps": 900.0, "n_prompt": 8000, "decode_all": [], "prefill_all": [900]}]}
        p = app.score_payload(res, "Zed")
        self.assertEqual((p["bracket"]["mode"], p["bracket"]["gpu_count"], p["name"]), ("tensor", 2, "Zed"))
        self.assertEqual((p["metrics"]["decode_tps"], p["metrics"]["prefill_tps"]), (41.5, 900.0))
        self.assertNotIn("sk-abc", json.dumps(p))
        self.assertRegex(p["bracket"]["model_sha8"], r"^[0-9a-f]{8}$")
        self.assertEqual(app.score_check()["available"], False)          # no benchmark run on this App yet

    def test_params_block_has_the_facts_and_no_secrets(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "--host", "192.168.7.9", "-sm", "tensor", "-c", "16384", "-ctk", "q4_0", "-ctv", "q4_0",
                        "-b", "2048", "-ub", "512", "-fa", "on", "--spec-type", "mtp:n_max=3", "--api-key", "sk-abcdefghijklmnop1234"]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [0, 1], "port": 8999}
        res = {"ts": 1_780_000_000.0, "model": FAKE_MODEL, "cards": [0, 1], "reps": 3, "greedy512_sha": "ab" * 32,
               "classes": [{"class": "prose", "decode_tps": 41.5, "prefill_tps": 0, "n_prompt": 20, "decode_all": [41], "prefill_all": []}]}
        p = app.score_payload(res, "Zed")
        q = p["params"]
        self.assertEqual((q["card_count"], q["split_mode"], q["ctx"], q["kv_k"], q["batch"], q["ubatch"], q["flash_attn"], q["n_max"], q["mtp"]),
                         (2, "tensor", 16384, "q4_0", 2048, 512, "on", 3, True))
        self.assertEqual((q["reps"], q["greedy512_sha"], q["prompt_class"], q["model_file"]), (3, "ab" * 32, "prose", os.path.basename(FAKE_MODEL)))
        self.assertTrue(q["cards"] and q["cards"][0]["vram_mb"] > 0)
        blob = json.dumps(p)
        for bad in ("sk-abc", "192.168", os.path.dirname(FAKE_MODEL)):
            self.assertNotIn(bad, blob)
        self.assertNotIn("--host", p["cmd"])

    def test_name_is_saved_and_reused(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        app = C.App(L, port=0, models_dirs=[MODELS])
        self.assertEqual(app.user_name()["name"], "")
        self.assertEqual(app.set_user_name({"name": "  Pat  "})["name"], "Pat")
        self.assertEqual(C.App(L, port=0).user_name()["name"], "Pat")         # survives a restart (control.json)
        with self.assertRaises(C.Invalid):
            app.set_user_name({"name": "http://evil.example"})
        self.assertEqual(app.user_name()["name"], "Pat")                      # a refused name changes nothing
        self.assertEqual(app.set_user_name({"name": ""})["name"], "")         # forget
        # a submit saves the name too (first entry)
        real = C.urllib.request.urlopen
        C.urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
        try:
            app.score_send({"payload": {"name": "Zed", "bracket": {}}})
        finally:
            C.urllib.request.urlopen = real
        self.assertEqual(app.user_name()["name"], "Zed")
        self.assertEqual(json.load(open(C.config_path()))["user_name"], "Zed")

    def test_detail_validates_id(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        for bad in ("x", "0", "-4", "99999999999", None):
            with self.assertRaises(C.Invalid):
                app.score_detail(bad)

    def _fake_board(self, answers):
        """urlopen stand-in for the board: POST /v1/score answers from `answers` (a list, one per call); records the calls."""
        calls = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.close()

        def urlopen(req, timeout=None):
            calls.append((req.get_method(), req.full_url))
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return Resp(json.dumps(a).encode())
        return urlopen, calls

    def test_a_submitted_result_is_remembered_and_never_offered_again(self):
        # FR#13: after Submit the frame closed only in the page; the same result came back on the next poll, after a reload, and
        # the Check button stayed. Control now keeps the submitted result (its ts) in control.json.
        R = sys.modules["pxa_control.routes"]
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [0], "port": 8999}
        res = {"ts": 1_780_000_123.456789, "model": FAKE_MODEL, "cards": [0], "port": 8999, "reps": 3, "greedy512_sha": "ab" * 32,
               "classes": [{"class": "prose", "decode_tps": 41.5, "prefill_tps": 0, "n_prompt": 20, "decode_all": [41], "prefill_all": []}]}
        app.bench = {"running": False, "lines": [], "result": res, "error": None, "started": 1.0, "port": 8999, "cards": [0]}
        self.assertFalse(R.r_bench(app, None, {})["job"]["result"].get("submitted"))
        real = C.urllib.request.urlopen
        # a failed send records nothing: the frame may offer the result again
        C.urllib.request.urlopen, calls = self._fake_board([OSError("offline")])
        try:
            r = app.score_send({"payload": app.score_payload(res, "Zed"), "ts": res["ts"]})
        finally:
            C.urllib.request.urlopen = real
        self.assertFalse(r["ok"])
        self.assertFalse(R.r_bench(app, None, {})["job"]["result"].get("submitted"))
        # the board takes it (held for review counts too)
        C.urllib.request.urlopen, calls = self._fake_board([{"ok": True, "id": 7, "rank": None, "status": "flagged"}])
        try:
            r = app.score_send({"payload": app.score_payload(res, "Zed"), "ts": res["ts"]})
        finally:
            C.urllib.request.urlopen = real
        self.assertTrue(r["ok"], r)
        self.assertEqual(calls, [("POST", C.score_base_url() + "/v1/score")])
        self.assertTrue(R.r_bench(app, None, {})["job"]["result"]["submitted"])
        self.assertIsNot(app.bench["result"].get("submitted"), True)      # the route marks a copy; the job itself is untouched
        # the check does not offer it again, and asks the board nothing
        C.urllib.request.urlopen, calls = self._fake_board([])
        try:
            c = app.score_check()
        finally:
            C.urllib.request.urlopen = real
        self.assertEqual((c["available"], calls), (False, []), c)
        self.assertIn("already", c["reason"])
        # a reload of the page, and a restart of Control, read the same record
        app2 = C.App(L, port=0, models_dirs=[MODELS])
        app2.bench = dict(app.bench)
        self.assertTrue(R.r_bench(app2, None, {})["job"]["result"]["submitted"])
        # a new run of the same model is a new result: offered again
        app2.bench = dict(app.bench, result=dict(res, ts=res["ts"] + 60))
        self.assertFalse(R.r_bench(app2, None, {})["job"]["result"].get("submitted"))
        # a bad ts in the body falls back to the current result; the record stays bounded
        for i in range(60):
            C._score_mark_sent(1_000.0 + i)
        self.assertEqual(len(C.load_config()["score_sent"]), 50)
        self.assertTrue(C._score_was_sent(1_059.0) and not C._score_was_sent(1_000.0))


# ---------------------------------------------------------------------------------------------
# 2026.10.2: several servers per GUI, the fleet, extra args, lever hints, telemetry, bounded report
# ---------------------------------------------------------------------------------------------
def fake_running_seat(app, sid, port, gpus):
    """a seat whose process is a real sleeping child (running() is true) holding port/gpus."""
    import subprocess
    seat = app.get_seat(sid, create=True)
    seat.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    seat.req = {"port": port, "gpus": list(gpus), "model": FAKE_MODEL}
    return seat


class Servers(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.app = C.App(L, port=7777, models_dirs=[MODELS])
        self.app.catalog_names = CATALOG
        self.kids = []

    def tearDown(self):
        for s in self.app.seats.values():
            if s.proc and s.proc.poll() is None:
                s.proc.kill()
                s.proc.wait()

    def test_profiles_crud_and_persistence(self):
        self.assertEqual(list(self.app.profiles()), ["main"])
        sid, ent = self.app.save_profile(None, "V100 box", {"gpus": [1], "model": FAKE_MODEL, "port": 0})
        self.assertEqual(sid, "v100-box")
        sid2, _ = self.app.save_profile(None, "V100 box", {})
        self.assertEqual(sid2, "v100-box-2")                       # names may repeat, ids may not
        again = C.App(L, port=7777)
        self.assertEqual(again.profiles()["v100-box"]["settings"]["gpus"], [1])
        self.assertEqual(again.profiles()["v100-box"]["settings"]["port"], 0)   # auto stays auto
        with self.assertRaises(C.Invalid):
            self.app.delete_profile("main")
        for bad in ("", "x" * 49, "semi;colon", None):
            with self.assertRaises(C.Invalid):
                self.app.save_profile(None, bad, {})
        with self.assertRaises(C.Invalid):
            self.app.get_seat("../etc")
        with self.assertRaises(C.Invalid):
            self.app.get_seat("nope")                                 # unknown and not created
        self.assertTrue(self.app.delete_profile("v100-box-2"))
        self.assertNotIn("v100-box-2", C.App(L, port=7777).profiles())

    def test_delete_running_refused(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        with self.assertRaises(C.Invalid):
            self.app.delete_profile("b")

    def test_port_and_card_conflicts(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        req = self.app.validate({"gpus": [1], "model": FAKE_MODEL, "port": 18555}, resolve_port=False)
        c = self.app.conflicts(req, "main")
        self.assertTrue(any("port 18555" in e and "'B'" in e for e in c["errors"]), c)
        self.assertTrue(any(e.startswith("card 1 is in use by server 'B'") for e in c["errors"]), c)
        req = self.app.validate({"gpus": [1], "model": FAKE_MODEL, "port": 18556, "allow_busy": True}, resolve_port=False)
        c = self.app.conflicts(req, "main")
        self.assertEqual(c["errors"], [])
        self.assertTrue(any("allowed" in w for w in c["warnings"]))
        req = self.app.validate({"gpus": [0], "model": FAKE_MODEL, "port": 18556}, resolve_port=False)
        self.assertEqual(self.app.conflicts(req, "main"), {"errors": [], "warnings": []})
        # a server's own port/cards are not a conflict with itself
        req = self.app.validate({"gpus": [1], "model": FAKE_MODEL, "port": 18555}, resolve_port=False)
        self.assertEqual(self.app.conflicts(req, "b")["errors"], [])
        # auto port skips the ports other servers hold
        self.assertNotIn(self.app.validate({"gpus": [0], "model": FAKE_MODEL})["port"], self.app.held_ports())

    def test_stop_then_start_on_the_same_port(self):
        # regression (TucsonJohn, 2026-10-07): the engine listens with SO_REUSEPORT only (cpp-httplib), so
        # the TIME_WAIT sockets its health polls leave on its port refused Control's SO_REUSEADDR probe for
        # up to a minute after Stop: "port N is already in use by another program" until the port changed
        engine = ("import socket, sys\n"
                  "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)\n"
                  "s.bind(('127.0.0.1', 0)); s.listen(8); print(s.getsockname()[1], flush=True)\n"
                  "while True:\n"
                  "    c, _ = s.accept(); c.recv(1024); c.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Length: 0\\r\\n\\r\\n'); c.close()\n")
        seat = self.app.get_seat("main")
        seat.start((L.Plan(), [sys.executable, "-u", "-c", engine], {}, "0", {}, 4096),
                   {"port": 0, "model": FAKE_MODEL, "gpus": [0]}, {})
        t0 = time.time()
        while not any(x[1].strip().isdigit() for x in seat.lines_since(0)) and time.time() - t0 < 20:
            time.sleep(0.05)
        port = int(next(x[1] for x in seat.lines_since(0) if x[1].strip().isdigit()))
        seat.req["port"] = port
        for _ in range(3):                      # health polls: the server closes first -> TIME_WAIT on its port
            with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
                c.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
                c.recv(1024)
                time.sleep(0.05)
        self.assertTrue(C.port_in_use(port))    # a live listener is still refused
        self.assertTrue(self.app.stop("main"))
        self.assertFalse(C.port_in_use(port))
        req = self.app.validate({"gpus": [0], "model": FAKE_MODEL, "port": port}, resolve_port=False)
        self.assertEqual(self.app.conflicts(req, "main")["errors"], [])
        s2 = socket.socket()                    # and the next engine really binds there
        s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        try:
            s2.bind(("127.0.0.1", port))
            s2.listen(1)
        finally:
            s2.close()

    def test_first_plan_sees_discovered_servers(self):
        # regression: right after start the fleet cache was empty and the first plan saw no clash
        snap = {"instances": [{"key": "d:seat", "kind": "docker", "name": "seat", "container": "seat",
                               "running": True, "port": 18700, "gpus": [0], "vram": {}}], "cards": []}
        calls = []

        def fake_fleet(max_age=2.5):
            calls.append(max_age)
            self.app._fleet_cache = (time.time(), snap)
            return snap
        self.app.fleet = fake_fleet
        d = self.app.plan({"gpus": [0], "model": FAKE_MODEL, "port": 18700})
        self.assertTrue(calls)
        errs = " ".join(d["conflicts"]["errors"])
        self.assertIn("card 0 is in use by docker seat", errs)
        self.assertIn("port 18700", errs)

    def test_target_port_only_known_servers(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        self.assertEqual(self.app.target_port({"sid": ["b"]}), 18555)
        self.assertEqual(self.app.target_port({"port": ["18555"]}), 18555)
        for bad in ("22", "631", "abc"):
            with self.assertRaises(C.Invalid):
                self.app.target_port({"port": [bad]})

    def test_fleet_lists_managed_and_holders(self):
        self.app.save_profile("b", "B", {})
        fake_running_seat(self.app, "b", 18555, [1])
        f = self.app.fleet(max_age=0)
        keys = {x["key"]: x for x in f["instances"]}
        self.assertIn("m:main", keys)
        self.assertNotIn("delete", keys["m:main"]["actions"])     # the first server cannot be deleted
        self.assertEqual((keys["m:b"]["running"], keys["m:b"]["port"], keys["m:b"]["gpus"]), (True, 18555, [1]))
        self.assertEqual(keys["m:b"]["health"], "down")            # nothing answers on that port
        card1 = [c for c in f["cards"] if c["index"] == 1][0]
        self.assertEqual([u["sid"] for u in card1["users"]], ["b"])

    def test_external_control_needs_adopt_and_confirm(self):
        snap = {"instances": [{"key": "d:pxa-seat", "kind": "docker", "name": "pxa-seat", "container": "pxa-seat",
                               "running": True, "port": 18600, "gpus": [0], "vram": {}, "adopted": False, "control": False}],
                "cards": []}
        self.app._fleet_cache = (time.time(), snap)
        self.app.fleet = lambda max_age=2.5: snap
        with self.assertRaises(C.Invalid) as e:
            self.app.external_control("d:pxa-seat", "stop", "pxa-seat")
        self.assertIn("read-only", str(e.exception))
        snap["instances"][0].update(adopted=True, control=True)
        with self.assertRaises(C.Invalid) as e:
            self.app.external_control("d:pxa-seat", "stop", "pxa-sea")
        self.assertIn("confirm", str(e.exception))
        with self.assertRaises(C.Invalid):
            self.app.external_control("d:pxa-seat", "rm", "pxa-seat")
        with self.assertRaises(C.Invalid):
            self.app.adopt("p:1234")                                   # a bare process cannot be adopted
        # a discovered server's card is a conflict for a new launch, its port is known to the proxy
        req = self.app.validate({"gpus": [0], "model": FAKE_MODEL, "port": 18601}, resolve_port=False)
        self.assertTrue(any("docker pxa-seat" in e for e in self.app.conflicts(req, "main")["errors"]))
        self.assertEqual(self.app.target_port({"port": ["18600"]}), 18600)

    def test_handlers(self):
        srv, port = serve_app(self.app)
        try:
            code, _, body = req(port, "/api/servers", "POST", {"name": "Second", "settings": {}})
            self.assertEqual(code, 200, body)
            sid = json.loads(body)["sid"]
            d = json.loads(req(port, "/api/servers")[2])
            self.assertEqual([x["sid"] for x in d["servers"]], ["main", sid])
            self.assertEqual(json.loads(req(port, f"/api/status?sid={sid}")[2])["running"], False)
            self.assertEqual(req(port, "/api/status?sid=nope")[0], 400)
            f = json.loads(req(port, "/api/fleet?force=1")[2])
            self.assertEqual(len(f["instances"]), 2)
            code, _, body = req(port, "/api/plan", "POST", {"sid": sid, "gpus": [0], "model": FAKE_MODEL,
                                                             "extra_args": "--path /etc"})
            self.assertEqual(code, 400)
            self.assertIn("--path", json.loads(body)["error"])
            code, _, body = req(port, "/api/levers/lint", "POST", {"levers": {"PXA_ENHANCE": "1"}})
            self.assertEqual(code, 200, body)
            self.assertIn("warnings", json.loads(body))
            self.assertEqual(req(port, "/api/servers", "DELETE", {"sid": sid})[0], 200)
            self.assertEqual(req(port, "/api/servers", "DELETE", {"sid": "main"})[0], 400)
            self.assertEqual(req(port, "/api/engine/health?port=22")[0], 400)
        finally:
            srv.shutdown()
            srv.server_close()


class ExtraArgsAndLevers(unittest.TestCase):
    def test_allow_list(self):
        self.assertEqual(C.validate_extra_args("--alias my-model -t 8 --metrics --temp=0.6"),
                         ["--alias", "my-model", "-t", "8", "--metrics", "--temp", "0.6"])
        self.assertEqual(C.validate_extra_args(["-ot", "blk\\.[0-9]+\\.ffn_.*=CPU"]), ["-ot", "blk\\.[0-9]+\\.ffn_.*=CPU"])
        self.assertEqual(C.validate_extra_args(""), [])
        for bad in ("--path /etc", "--log-file /tmp/x", "--host 0.0.0.0", "--api-key k", "-m /x.gguf",
                    "--slot-save-path /tmp", "-t", "-t -c", "--alias 'a b'", "--alias $(id)", "--alias a;b",
                    "--metrics=1", 42):
            with self.assertRaises(C.Invalid, msg=repr(bad)):
                C.validate_extra_args(bad)

    def test_extra_args_reach_the_command(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        r = app.validate({"gpus": [0], "model": FAKE_MODEL, "extra_args": "--alias zed"}, resolve_port=False)
        self.assertEqual(r["extra_args"], ["--alias", "zed"])

    def test_lever_kind_default_and_lint(self):
        rows = [{"name": "PXA_A", "default": "off (=1 on)", "rule": "", "status": "measured"},
                {"name": "PXA_B", "default": "16", "rule": "", "status": "measured"},
                {"name": "PXA_C", "default": "see site rule", "rule": "", "status": "diagnostic"}]
        self.assertEqual([C.lever_kind(r) for r in rows], ["bool", "int", "text"])
        self.assertEqual([C.lever_default_state(r) for r in rows], ["off", "value", "site"])
        w = C.lint_levers({"PXA_A": "yes please", "PXA_B": "x", "PXA_C": "1", "PXA_UNKNOWN": "1"}, rows)
        self.assertTrue(any("on/off" in x for x in w), w)
        self.assertTrue(any("whole number" in x for x in w), w)
        self.assertTrue(any("diagnostic" in x for x in w), w)
        self.assertTrue(any("already the default" in x for x in C.lint_levers({"PXA_A": "0"}, rows)))
        self.assertEqual(C.lint_levers({"PXA_A": "1", "PXA_B": "32"}, rows), [])

    def test_shell_line(self):
        self.assertEqual(C.shell_line({"B": "x y", "A": "1"}, ["/opt/llama-server", "-m", "/m/a b.gguf"]),
                         "A=1 B='x y' /opt/llama-server -m '/m/a b.gguf'")

    def test_preset_keeps_auto_port(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        app = C.App(L, port=7777, models_dirs=[MODELS])
        app.save_preset("auto", {"gpus": [0], "model": FAKE_MODEL})
        self.assertEqual(app.presets()["auto"]["port"], 0)


class TokenPersistence(unittest.TestCase):
    def test_token_survives_restart(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        old = os.environ.pop("PXA_CONTROL_TOKEN", None)
        try:
            t1 = C.load_or_make_token()
            self.assertGreaterEqual(len(t1), 16)
            self.assertEqual(C.load_or_make_token(), t1)
            self.assertEqual(os.stat(os.path.join(C.config_dir(), "token")).st_mode & 0o777, 0o600)
            os.environ["PXA_CONTROL_TOKEN"] = "env-token-0123456789"
            self.assertEqual(C.load_or_make_token(), "env-token-0123456789")
            os.environ["PXA_CONTROL_TOKEN"] = "short"
            with self.assertRaises(SystemExit):
                C.load_or_make_token()
        finally:
            os.environ.pop("PXA_CONTROL_TOKEN", None)
            if old is not None:
                os.environ["PXA_CONTROL_TOKEN"] = old


class BenchTelemetry(unittest.TestCase):
    def row(self, i, temp, clk, reasons=(), limit=250.0):
        return {"index": i, "temp_c": temp, "sm_clock_mhz": clk, "sm_clock_max_mhz": 1480.0, "power_w": 200.0,
                "power_limit_w": limit, "throttle_reasons": list(reasons)}

    def test_summary_and_warnings(self):
        samples = [[self.row(0, 60, 1400), self.row(1, 70, 1300)],
                   [self.row(0, 84, 800, ["hw_thermal_slowdown", "gpu_idle"]), self.row(1, 72, 1310)]]
        t = C.App.summarize_telemetry(samples, cards=[0], stock={0: 250.0})
        self.assertEqual([c["index"] for c in t["cards"]], [0])        # only the benchmarked card
        c = t["cards"][0]
        self.assertEqual((c["temp_max"], c["sm_clock_min"], c["samples"]), (84, 800, 2))
        self.assertIn("hw_thermal_slowdown", c["throttle"])
        self.assertTrue(any("84" in w for w in t["warnings"]))
        self.assertTrue(any("throttled" in w for w in t["warnings"]))
        low = C.App.summarize_telemetry([[self.row(0, 50, 1400, limit=150.0)]], stock={0: 250.0})
        self.assertTrue(any("below its stock 250 W" in w for w in low["warnings"]), low)
        slow = C.App.summarize_telemetry([[self.row(0, 50, 405)]], stock={})
        self.assertTrue(any("never clocked above 405" in w for w in slow["warnings"]), slow)
        self.assertEqual(C.App.summarize_telemetry([], cards=[0])["warnings"], [])


class StuckRigReport(unittest.TestCase):
    def test_report_comes_back_when_everything_hangs(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        hang = lambda *a, **k: time.sleep(30)      # noqa: E731
        app.gpus, app.gpu_telemetry, app.rig_static = hang, hang, hang
        app._rig_cache = (0.0, None)
        app.REPORT_BUDGET_S = {"gpus": 0.5, "rig": 0.5, "telemetry": 0.5}
        app.seat._append("llama_model_loader: loading model")
        t0 = time.time()
        b = app.report_bundle("main")
        self.assertLess(time.time() - t0, 5)
        notes = " ".join(b["bundle"]["collection_notes"])
        self.assertIn("nvidia-smi did not answer", notes)
        self.assertIn("telemetry timed out", notes)

    def test_bounded(self):
        self.assertEqual(C.bounded(lambda: 7, 1), 7)
        self.assertEqual(C.bounded(lambda: time.sleep(5), 0.2, "late"), "late")
        self.assertEqual(C.bounded(lambda: 1 / 0, 1, "err"), "err")


class GpuFallback(unittest.TestCase):
    def test_probe_never_raises(self):
        rows, src, err = C.probe_gpus_fallback(timeout=20)
        self.assertIsInstance(rows, list)
        self.assertTrue(rows or err)

    def test_used_when_nvidia_smi_is_missing(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        fake = os.environ.pop("PXA_LAUNCH_FAKE_GPUS")
        real_table, real_probe = app.L.gpu_table, C.probe_gpus_fallback
        app.L.gpu_table = lambda: ([], "nvidia-smi not found")
        C.probe_gpus_fallback = lambda timeout=15: ([(0, "Tesla V100-SXM2-16GB", 70, 16384, 0, "GPU-x")], "nvml", None)
        try:
            rows, err = app.gpus(max_age=0)
        finally:
            os.environ["PXA_LAUNCH_FAKE_GPUS"] = fake
            app.L.gpu_table, C.probe_gpus_fallback = real_table, real_probe
        self.assertEqual(rows[0][:3], (0, "Tesla V100-SXM2-16GB", 70))
        self.assertIn("cards read from nvml", err)


class EngineSearch(unittest.TestCase):
    def test_normalize_and_find(self):
        root = os.path.join(TMP, "src")
        for b in ("build", "build-cuda12"):
            os.makedirs(os.path.join(root, b, "bin"), exist_ok=True)
            with open(os.path.join(root, b, "bin", "llama-server"), "w") as f:
                f.write("#!/bin/sh\n")
        os.utime(os.path.join(root, "build", "bin", "llama-server"), (1, 1))
        b = os.path.join(root, "build")
        for given in (b, b + "/bin", b + "/bin/llama-server", b + "/"):
            self.assertEqual(L.normalize_engine_dir(given), b, given)
        self.assertEqual(L.find_source_builds([root]), [os.path.join(root, "build-cuda12"), b])
        old = os.environ.get("PXA_ENGINE_DIR")
        os.environ["PXA_ENGINE_DIR"] = b + "/bin"
        try:
            got = L.resolve_engine_dir()
        finally:
            os.environ.pop("PXA_ENGINE_DIR") if old is None else os.environ.__setitem__("PXA_ENGINE_DIR", old)
        self.assertEqual((got[0] if isinstance(got, tuple) else got), b)


# ---------------------------------------------------------------------------------------------
# the Encode tab: routes, auth, plain errors, the licence key never leaves its place
# ---------------------------------------------------------------------------------------------
sys.path.insert(0, os.path.join(ROOT, "tests"))
import encode_fakes as EF  # noqa: E402

ENC_SECRET = "A1b2C3d4A1b2C3d4"


class EncodeRedaction(unittest.TestCase):
    def test_licence_key_forms_are_redacted_in_every_text_that_leaves(self):
        key = EF.GOOD_KEY
        for text in ("key %s here" % key, "PXQE_KEY=%s" % key, "PXQE_KEY: %s" % key, "licence_key=%s" % key, "export PXQE_KEY='%s'" % key):
            out = C.redact_text(text, {"rigbox"}, {"alice"})
            self.assertNotIn(ENC_SECRET, out, text)
            self.assertIn("<redacted", out, text)
        o = C.redact_obj({"cmd": "pxqe run PXQE_KEY=%s" % key, "log": ["token %s" % key]}, {"rigbox"}, {"alice"})
        self.assertNotIn(ENC_SECRET, json.dumps(o))

    def test_a_key_in_the_server_log_never_reaches_the_report_bundle(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat._append("env PXQE_KEY=%s" % EF.GOOD_KEY)
        app.seat._append("plain %s line" % EF.GOOD_KEY)
        b = app.report_bundle()
        self.assertNotIn(ENC_SECRET, json.dumps(b))

    def test_score_payload_redacts_a_key_in_the_command(self):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "--alias", EF.GOOD_KEY, "-c", "4096"]
        app.seat.req = {"port": 8999, "model": FAKE_MODEL, "gpus": [0]}
        res = {"model": FAKE_MODEL, "port": 8999, "cards": [0], "classes": [{"class": "prose", "decode_tps": 40.0}, {"class": "long", "prefill_tps": 900.0}], "reps": 3, "greedy512_sha": "x"}
        p = app.score_payload(res)
        self.assertNotIn(ENC_SECRET, json.dumps(p))


class BenchHistoryFields(unittest.TestCase):
    """the bench history row carries MTP and the split mode (FR#1: the history table shows them); same values as the high-score payload"""

    class FakeBench(object):
        CLASS_PROMPTS = {c: {"prompt": c + " prompt", "n_predict": 8} for c in ("prose", "edit", "long")}

        @staticmethod
        def _get_json(url, path, timeout=10):
            return {"model_path": FAKE_MODEL}

        @staticmethod
        def measure_once(url, prompt, n, timeout=0):
            return {}

        @staticmethod
        def _post_json(url, path, body, timeout=0):
            return {"timings": {"predicted_per_second": 40.0, "prompt_per_second": 900.0, "prompt_n": 20}}

        @staticmethod
        def greedy512(url):
            return {"sha256": "cd" * 32, "empty": False}

    def _run(self, cmd, gpus):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = cmd
        app.seat.req = {"model": FAKE_MODEL, "gpus": gpus, "port": 8999}
        app.bench = {"running": True, "lines": [], "result": None, "error": None}
        app._bench_run(self.FakeBench, 8999, 1, 0, gpus)
        self.assertIsNone(app.bench["error"], app.bench["error"])
        return app, app.bench["result"]

    def test_row_stores_split_mode_and_mtp(self):
        app, res = self._run(["llama-server", "-m", FAKE_MODEL, "-sm", "tensor", "--spec-type", "mtp:n_max=3"], [0, 1])
        self.assertEqual((res["split_mode"], res["mtp"]), ("tensor", True))
        stored = C.read_jsonl("bench.jsonl", limit=1)[-1]
        self.assertEqual((stored["split_mode"], stored["mtp"], stored["ts"]), ("tensor", True, res["ts"]))
        p = app.score_payload(res)                                   # the board payload says the same
        self.assertEqual((p["params"]["split_mode"], p["params"]["mtp"]), (res["split_mode"], res["mtp"]))

    def test_layer_plain_and_mtp_off(self):
        _a, res = self._run(["llama-server", "-m", FAKE_MODEL], [0, 1])
        self.assertEqual((res["split_mode"], res["mtp"]), ("layer", False))
        _a, res = self._run(["llama-server", "-m", FAKE_MODEL], [0])
        self.assertEqual((res["split_mode"], res["mtp"]), ("plain", False))
        _a, res = self._run(["llama-server", "-m", FAKE_MODEL, "--spec-type", "mtp"], [0])
        self.assertEqual((res["split_mode"], res["mtp"]), ("plain", True))

    def test_route_pages_through_more_than_fifty_rows_without_telemetry(self):
        import pxa_control.routes as R
        app = C.App(L, port=0, models_dirs=[MODELS])
        for i in range(60):
            C.append_jsonl("bench.jsonl", {"ts": 1000.0 + i, "model": "m%d.gguf" % i, "classes": [], "telemetry": {"cards": [{"index": 0}]}})
        h = R.r_bench(app, {}, {})["history"]
        self.assertGreaterEqual(len(h), 60)
        self.assertTrue(all("telemetry" not in r for r in h))
        old = [r for r in h if r.get("model") == "m3.gguf"][0]       # a row from before the fields existed has neither key
        self.assertNotIn("mtp", old)
        self.assertNotIn("split_mode", old)


class BenchFirstPass(unittest.TestCase):
    """The bench reports FIRST-PASS speed (2026-10-19): a server with n-gram speculation replays an answer it already wrote,
    so the same prompt asked again without a slot erase reads ~3x. The fake server models exactly that; the old loop
    (warm-up on the prose prompt, then 3 reps of the same prompt, no erase) read 120 here instead of 40."""

    class ReplayServer(BenchHistoryFields.FakeBench):
        seen, erases, slots = set(), [], 2

        @classmethod
        def reset(cls, slots=2):
            cls.seen, cls.erases, cls.slots = set(), [], slots

        @classmethod
        def _get_json(cls, url, path, timeout=10):
            return {"model_path": FAKE_MODEL, "total_slots": cls.slots}

        @classmethod
        def measure_once(cls, url, prompt, n, timeout=0):
            cls.seen.add(prompt)                    # the warm-up writes the start of the prose answer into the table
            return {}

        @classmethod
        def _post_json(cls, url, path, body, timeout=0):
            if path.startswith("/slots/"):
                cls.erases.append(path)
                cls.seen.clear()
                return {"id_slot": int(path.split("/")[2].split("?")[0]), "n_erased": 0}
            replay = body["prompt"] in cls.seen
            cls.seen.add(body["prompt"])
            return {"timings": {"predicted_per_second": 120.0 if replay else 40.0, "prompt_per_second": 900.0, "prompt_n": 20}}

    class NoEraseServer(ReplayServer):
        @classmethod
        def _post_json(cls, url, path, body, timeout=0):
            if path.startswith("/slots/"):
                raise OSError("HTTP Error 501: slot erase not supported")
            return BenchFirstPass.ReplayServer._post_json.__func__(cls, url, path, body, timeout)

    def _run(self, mod, reps=3, warm=1):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "-sm", "tensor", "--spec-type", "mtp"]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [0, 1], "port": 8999}
        app.bench = {"running": True, "lines": [], "result": None, "error": None}
        app._bench_run(mod, 8999, reps, warm, [0, 1])
        self.assertIsNone(app.bench["error"], app.bench["error"])
        return app, app.bench["result"]

    def test_every_rep_is_first_pass(self):
        self.ReplayServer.reset(slots=2)
        _app, res = self._run(self.ReplayServer)
        for c in res["classes"]:
            self.assertEqual(c["decode_all"], [40.0, 40.0, 40.0], c)
            self.assertEqual(c["decode_tps"], 40.0)
        self.assertTrue(res["first_pass"])
        # every slot erased before every rep (3 classes x 3 reps) and before greedy512: 10 rounds x 2 slots
        self.assertEqual(len(self.ReplayServer.erases), 20)
        self.assertEqual(set(self.ReplayServer.erases), {"/slots/0?action=erase", "/slots/1?action=erase"})
        stored = C.read_jsonl("bench.jsonl", limit=1)[-1]
        self.assertTrue(stored["first_pass"])

    def test_a_server_without_erase_is_flagged(self):
        self.NoEraseServer.reset(slots=1)
        app, res = self._run(self.NoEraseServer)
        self.assertFalse(res["first_pass"])
        self.assertTrue(any("would not forget" in ln for ln in app.bench["lines"]), app.bench["lines"])
        self.assertEqual(sum("would not forget" in ln for ln in app.bench["lines"]), 1)   # said once, not per rep

    def test_warmup_uses_its_own_prompt(self):
        """FR#12: the warm-up never asks a measured prompt (it would seed the draft table of a server that refuses the erase)"""
        warm = []

        class W(self.ReplayServer):
            WARMUP_PROMPT = BENCH.WARMUP_PROMPT

            @classmethod
            def measure_once(cls, url, prompt, n, timeout=0):
                warm.append(prompt)
                return {}

        W.reset(slots=1)
        self._run(W)
        self.assertTrue(warm and set(warm) == {BENCH.WARMUP_PROMPT}, warm)
        self.assertNotIn(BENCH.WARMUP_PROMPT, [BENCH.CLASS_PROMPTS[c]["prompt"] for c in BENCH.CLASS_PROMPTS])

    def test_refused_erase_gets_a_fresh_prompt_per_run(self):
        """FR#12: a server without the erase replays a repeated prompt (120 vs 40 here); each run's prompt is distinct
        (a random per-bench tag at the START), so the reps read first pass and the result still says first_pass false"""
        prompts = []

        class N(self.NoEraseServer):
            @classmethod
            def _post_json(cls, url, path, body, timeout=0):
                if not path.startswith("/slots/"):
                    prompts.append(body["prompt"])
                return BenchFirstPass.NoEraseServer._post_json.__func__(cls, url, path, body, timeout)

        N.reset(slots=1)
        app, res = self._run(N)
        self.assertFalse(res["first_pass"])
        self.assertEqual(len(prompts), 9)
        self.assertEqual(len(set(prompts)), 9)
        self.assertTrue(all(p.startswith("[bench ") for p in prompts))
        for c in res["classes"]:
            self.assertEqual(c["decode_all"], [40.0, 40.0, 40.0], c)
        prompts.clear()
        N.reset(slots=1)
        self._run(N)                                                   # a second bench on the same server: new prompts again
        self.assertEqual(len(set(prompts)), 9)

    def test_first_pass_vs_later_runs_and_replay_flag(self):
        """FR#12: an erase that does not clear the drafter (the 'forget' answered, the table stayed): later runs accept
        more drafts and read 2x+; the class is flagged and keeps its first pass. Honest runs: no flag, median kept."""
        self.ReplayServer.reset(slots=1)
        app, res = self._run(self.ReplayServer)
        for c in res["classes"]:
            self.assertEqual((c["first_pass_tps"], c["later_mean_tps"], c["replay_suspect"], c["decode_tps"]), (40.0, 40.0, False, 40.0))
        self.assertTrue(any("prose: first pass 40.0 t/s, later runs mean 40.0 t/s" in ln for ln in app.bench["lines"]), app.bench["lines"])

        class Sticky(self.ReplayServer):
            WARMUP_PROMPT = BENCH.WARMUP_PROMPT                    # (the fallback warm-up on the prose prompt would make rep 1 replay too)

            @classmethod
            def _post_json(cls, url, path, body, timeout=0):
                if path.startswith("/slots/"):
                    return {"id_slot": 0, "n_erased": 0}           # erase answered, table kept
                replay = body["prompt"] in cls.seen
                cls.seen.add(body["prompt"])
                return {"timings": {"predicted_per_second": 95.0 if replay else 40.0, "prompt_per_second": 900.0, "prompt_n": 20,
                                    "draft_n": 60, "draft_n_accepted": 55 if replay else 6}}

        Sticky.reset(slots=1)
        app, res = self._run(Sticky)
        for c in res["classes"]:
            self.assertTrue(c["replay_suspect"], c)
            self.assertEqual((c["first_pass_tps"], c["decode_tps"]), (40.0, 40.0))
            self.assertEqual(c["drafts_accepted"], [6, 55, 55])
        self.assertTrue(any("replayed" in ln for ln in app.bench["lines"]))


    def test_summarize_runs(self):
        s = BENCH.summarize_runs([40.0, 41.0, 39.0])
        self.assertEqual((s["first_pass"], s["later_mean"], s["replay_suspect"], s["headline"]), (40.0, 40.0, False, 40.0))
        s = BENCH.summarize_runs([40.0, 88.0, 90.0])                    # FR#12: later runs >2x = replay
        self.assertEqual((s["replay_suspect"], s["headline"]), (True, 40.0))
        self.assertAlmostEqual(s["later_vs_first"], 1.225)
        s = BENCH.summarize_runs([40.0])
        self.assertEqual((s["later_mean"], s["replay_suspect"], s["headline"]), (None, False, 40.0))
        s = BENCH.summarize_runs([40.0, 88.0, 90.0], [10, 150, 160])  # with draft evidence: still flagged
        self.assertTrue(s["replay_suspect"])
        s = BENCH.summarize_runs([4.3, 33.4, 86.6], [0, 0, 0])        # a stalled first pass, nothing accepted anywhere: noise
        self.assertEqual((s["replay_suspect"], s["headline"]), (False, 33.4))
        s = BENCH.summarize_runs([4.3, 33.4, 86.6], [0, None, 0])     # incomplete draft counts: speed alone decides
        self.assertTrue(s["replay_suspect"])
        self.assertNotEqual(BENCH.run_tag("ab", 1), BENCH.run_tag("ab", 2))

    def test_forget_helper(self):
        self.ReplayServer.reset(slots=3)
        self.assertTrue(C.bench_forget(self.ReplayServer, "http://x"))
        self.assertEqual(self.ReplayServer.erases, ["/slots/%d?action=erase" % i for i in range(3)])
        self.NoEraseServer.reset(slots=1)
        self.assertFalse(C.bench_forget(self.NoEraseServer, "http://x"))

class BenchEarlyEnd(unittest.TestCase):
    """#361/#362 (2026-10-26): Swift-1.5 Flash-Next ended the raw-text "edit" and "long" bench prompts at once (first token
    <|endoftext|> / <|im_end|>); the server times the first token with the prompt, so one token read 1,000,000 t/s and the
    bench card showed it. A run that ends early is asked again with ignore_eos (the class times n_predict tokens); a run
    that does not end early sends the same request as before; a reply of one token is never a speed."""

    class EosServer(BenchHistoryFields.FakeBench):
        """prose decodes n_predict tokens; edit/long end after 1 token unless ignore_eos (then n_predict tokens)"""
        CLASS_PROMPTS = BENCH.CLASS_PROMPTS
        WARMUP_PROMPT = BENCH.WARMUP_PROMPT
        bodies = []
        stubborn = False                     # True: even ignore_eos gives one token (a server that ignores the field)

        @classmethod
        def reset(cls, stubborn=False):
            cls.bodies, cls.stubborn = [], stubborn

        @classmethod
        def _get_json(cls, url, path, timeout=10):
            return {"model_path": FAKE_MODEL, "total_slots": 1}

        @classmethod
        def _post_json(cls, url, path, body, timeout=0):
            if path.startswith("/slots/"):
                return {"id_slot": 0, "n_erased": 0}
            cls.bodies.append(dict(body))
            n = body["n_predict"]
            early = body["prompt"] != BENCH.CLASS_PROMPTS["prose"]["prompt"] and (cls.stubborn or not body.get("ignore_eos"))
            if early:
                return {"content": "", "tokens_predicted": 1, "stop_type": "eos",
                        "timings": {"predicted_per_second": 1000000.0, "predicted_n": 1, "prompt_per_second": 360.0, "prompt_n": 87}}
            return {"content": "x" * n, "tokens_predicted": n, "stop_type": "limit",
                    "timings": {"predicted_per_second": 80.0, "predicted_n": n, "prompt_per_second": 360.0, "prompt_n": 87}}

    def _run(self, mod, reps=3):
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL, "--spec-type", "mtp"]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [1], "port": 8999}
        app.bench = {"running": True, "lines": [], "result": None, "error": None}
        app._bench_run(mod, 8999, reps, 0, [1])
        self.assertIsNone(app.bench["error"], app.bench["error"])
        return app, app.bench["result"]

    def test_control_bench_retries_an_early_end_with_ignore_eos(self):
        self.EosServer.reset()
        app, res = self._run(self.EosServer)
        by = {c["class"]: c for c in res["classes"]}
        for c in ("prose", "edit", "long"):
            self.assertEqual(by[c]["decode_all"], [80.0, 80.0, 80.0], by[c])
            self.assertEqual(by[c]["decode_tps"], 80.0)
            self.assertLess(by[c]["decode_tps"], 5000)
        self.assertEqual((by["prose"]["ignore_eos_runs"], by["edit"]["ignore_eos_runs"], by["long"]["ignore_eos_runs"]), (0, 3, 3))
        prose = [b for b in self.EosServer.bodies if b["prompt"] == BENCH.CLASS_PROMPTS["prose"]["prompt"]]
        self.assertEqual(len(prose), 3)
        self.assertTrue(all("ignore_eos" not in b for b in prose), prose)       # an honest run sends the old request
        self.assertEqual(sum(1 for b in self.EosServer.bodies if b.get("ignore_eos")), 6)
        self.assertEqual(sum("ended after 1 token(s) (eos)" in ln for ln in app.bench["lines"]), 6, app.bench["lines"])
        hist = [r for r in C.read_jsonl("history.jsonl", limit=50) if r.get("source") == "bench"][-9:]
        self.assertTrue(all(r["decode_tps"] < 5000 for r in hist), hist)

    def test_control_bench_never_reports_one_token_as_a_speed(self):
        self.EosServer.reset(stubborn=True)
        _app, res = self._run(self.EosServer)
        by = {c["class"]: c for c in res["classes"]}
        for c in ("edit", "long"):
            self.assertEqual(by[c]["decode_tps"], 0.0, by[c])               # no number, not 1,000,000
            self.assertEqual(by[c]["decode_all"], [0.0, 0.0, 0.0])
        self.assertEqual(by["prose"]["decode_tps"], 80.0)

    def test_pxa_bench_run_class_and_helpers(self):
        self.EosServer.reset()
        saved = (BENCH._post_json, BENCH.forget)
        try:
            BENCH._post_json = self.EosServer._post_json
            BENCH.forget = lambda url: True
            for cls, want_eos in (("prose", 0), ("edit", 3), ("long", 3)):
                spec = BENCH.CLASS_PROMPTS[cls]
                row = BENCH.run_class("http://x", cls, spec["prompt"], spec["n_predict"], 3)
                self.assertEqual(row["decode_all"], [80.0, 80.0, 80.0], row)
                self.assertEqual(row["ignore_eos_runs"], want_eos)
            self.EosServer.reset(stubborn=True)
            m = BENCH.measure_once("http://x", BENCH.CLASS_PROMPTS["edit"]["prompt"], 96)
            self.assertEqual((m["decode_tok_s"], m["tokens_predicted"], m["ended_early"]), (0.0, 1, True))
        finally:
            BENCH._post_json, BENCH.forget = saved
        self.assertEqual(BENCH.completion_body("p", 64), {"prompt": "p", "n_predict": 64, "temperature": 0.0, "seed": 0,
                                                          "cache_prompt": False})
        self.assertTrue(BENCH.completion_body("p", 64, ignore_eos=True)["ignore_eos"])
        one = {"tokens_predicted": 1, "timings": {"predicted_per_second": 1e6}}
        self.assertEqual((BENCH.ended_early(one, 96), BENCH.decode_rate(one)), (True, 0.0))
        self.assertEqual((C.bench_ended_early(one, 96), C.bench_decode_rate(one)), (True, 0.0))
        full = {"tokens_predicted": 96, "timings": {"predicted_per_second": 80.0}}
        self.assertEqual((BENCH.ended_early(full, 96), BENCH.decode_rate(full)), (False, 80.0))
        late = {"tokens_predicted": 40, "stop_type": "eos", "timings": {"predicted_per_second": 70.0}}
        self.assertFalse(BENCH.ended_early(late, 96))                         # a real answer that ended: kept
        self.assertFalse(C.bench_ended_early({"timings": {"predicted_per_second": 40.0}}, 96))   # no count: unchanged
        md = BENCH.render_markdown({"generated_at": "t", "cards_line": "1x V100", "model_alias": "m", "model_path": "/m.gguf",
                                    "model_sha": "0" * 64, "model_tier": "pxqn", "engine_build": 1, "engine_commit": "x",
                                    "reps": 3, "board_note": "",
                                    "classes": [{"class": "edit", "decode_tok_s": 80.0, "prefill_tok_s": 360.0, "vs_board": "-",
                                                 "ignore_eos_runs": 3}],
                                    "greedy": {"sha256": "a" * 64, "chars": 10, "tokens_predicted": 512, "empty": False}})
        self.assertIn("ignore_eos", md)


class BenchAllThinking(unittest.TestCase):
    """#366 (beta 6): a bench class whose every token is thinking (no answer) showed a speed off zero answer tokens
    (1,000,000 t/s on the card). The card now says "no answer: all thinking", the rate counts the thinking tokens
    (every token the server generated), and the high-score board takes no such run."""

    class ThinkServer(BenchHistoryFields.FakeBench):
        """prose answers; edit opens a thought it never closes (raw text); long returns only reasoning_content
        (a server that splits the thought out); with stubborn, edit/long end after one token even with ignore_eos"""
        CLASS_PROMPTS = BENCH.CLASS_PROMPTS
        WARMUP_PROMPT = BENCH.WARMUP_PROMPT
        stubborn = False

        @classmethod
        def _get_json(cls, url, path, timeout=10):
            return {"model_path": FAKE_MODEL, "total_slots": 1}

        @classmethod
        def _post_json(cls, url, path, body, timeout=0):
            if path.startswith("/slots/"):
                return {"id_slot": 0, "n_erased": 0}
            n, pr = body["n_predict"], body["prompt"]
            t = {"prompt_per_second": 360.0, "prompt_n": 87}
            if pr == BENCH.CLASS_PROMPTS["prose"]["prompt"]:
                return {"content": "x" * n, "tokens_predicted": n, "stop_type": "limit",
                        "timings": dict(t, predicted_per_second=80.0, predicted_n=n)}
            if cls.stubborn:
                return {"content": "", "tokens_predicted": 1, "stop_type": "eos",
                        "timings": dict(t, predicted_per_second=1000000.0, predicted_n=1)}
            if pr == BENCH.CLASS_PROMPTS["edit"]["prompt"]:
                return {"content": "<think>\nLet me think about the edit" + " hmm" * n, "tokens_predicted": n,
                        "stop_type": "limit", "timings": dict(t, predicted_per_second=78.0, predicted_n=n)}
            return {"content": "", "reasoning_content": "thinking " * n, "tokens_predicted": n, "stop_type": "limit",
                    "timings": dict(t, predicted_per_second=75.0, predicted_n=n)}

    def _run(self, stubborn=False, thinking=None):
        self.ThinkServer.stubborn = stubborn
        app = C.App(L, port=0, models_dirs=[MODELS])
        app.seat.cmd = ["llama-server", "-m", FAKE_MODEL]
        app.seat.req = {"model": FAKE_MODEL, "gpus": [1], "port": 8999}
        if thinking:
            app.seat.req["thinking"] = {"mode": thinking, "budget": None, "level": None}
        app.bench = {"running": True, "lines": [], "result": None, "error": None}
        app._bench_run(self.ThinkServer, 8999, 2, 0, [1])
        self.assertIsNone(app.bench["error"], app.bench["error"])
        return app, app.bench["result"]

    def test_all_thinking_is_named_and_counted(self):
        app, res = self._run()
        by = {c["class"]: c for c in res["classes"]}
        self.assertIsNone(by["prose"].get("no_answer"))
        self.assertEqual((by["prose"]["decode_tps"], by["prose"]["all_thinking_runs"]), (80.0, 0))
        for cls, rate in (("edit", 78.0), ("long", 75.0)):
            self.assertEqual(by[cls]["no_answer"], "all thinking", by[cls])
            self.assertEqual(by[cls]["all_thinking_runs"], 2)
            self.assertEqual(by[cls]["decode_tps"], rate)            # the thinking tokens are the generated tokens
        self.assertGreaterEqual(sum("no answer: all thinking" in ln for ln in app.bench["lines"]), 4, app.bench["lines"])
        chk = app.score_check()
        self.assertFalse(chk["available"])
        self.assertIn("no answer", chk["reason"])

    def test_no_tokens_is_no_answer_not_a_speed(self):
        app, res = self._run(stubborn=True)
        by = {c["class"]: c for c in res["classes"]}
        for cls in ("edit", "long"):
            self.assertEqual(by[cls]["decode_tps"], 0.0)
            self.assertEqual(by[cls]["no_answer"], "ended at once", by[cls])
        self.assertIsNone(by["prose"].get("no_answer"))
        self.assertFalse(app.score_check()["available"])

    def test_thinking_off_launch_is_never_all_thinking(self):
        """bug #372 (beta 6 VS report): the bench asks raw /completion, so a launch with thinking OFF never reaches it;
        a Qwen3.8 model opened a thought at the long prompt's "Answer:" and the card said "no answer: all thinking"
        although the user had turned thinking off. With thinking off at launch the class keeps its speed and the board
        takes it; the log says the raw prompt opened a thought."""
        app, res = self._run(thinking="off")
        by = {c["class"]: c for c in res["classes"]}
        for cls, rate in (("edit", 78.0), ("long", 75.0)):
            self.assertIsNone(by[cls]["no_answer"], by[cls])
            self.assertEqual(by[cls]["all_thinking_runs"], 0)
            self.assertEqual(by[cls]["decode_tps"], rate)
        self.assertFalse(any("no answer" in ln for ln in app.bench["lines"]), app.bench["lines"])
        self.assertTrue(any("thinking is off" in ln for ln in app.bench["lines"]), app.bench["lines"])
        self.assertTrue(app.score_check()["available"], app.score_check())

    def test_thinking_on_launch_keeps_366(self):
        app, res = self._run(thinking="on")
        by = {c["class"]: c for c in res["classes"]}
        self.assertEqual(by["long"]["no_answer"], "all thinking", by["long"])
        self.assertFalse(app.score_check()["available"])

    def test_reply_kinds(self):
        self.assertEqual(C.bench_answer_kind({"content": "Paris is the capital.", "tokens_predicted": 40}), None)
        self.assertEqual(C.bench_answer_kind({"content": "<think>a b c", "tokens_predicted": 40}), "all thinking")
        self.assertEqual(C.bench_answer_kind({"content": "<think>a</think>\n\nParis.", "tokens_predicted": 40}), None)
        self.assertEqual(C.bench_answer_kind({"content": "<think>a</think>\n\n", "tokens_predicted": 40}), "all thinking")
        self.assertEqual(C.bench_answer_kind({"content": "", "reasoning_content": "hm", "tokens_predicted": 40}), "all thinking")
        self.assertEqual(C.bench_answer_kind({"content": "<|channel|>analysis<|message|>hmm", "tokens_predicted": 9}),
                         "all thinking")
        self.assertEqual(C.bench_answer_kind({"content": "", "tokens_predicted": 1}), "ended at once")
        self.assertEqual(C.bench_answer_kind({"timings": {"predicted_per_second": 40.0}}), None)     # a test double / old engine

    def test_card_says_no_answer(self):
        with open(os.path.join(ROOT, "tools", "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            html = f.read()
        self.assertTrue("\"no answer: \"" in html and "c.no_answer" in html, "the bench card has no no-answer label")
        self.assertNotIn("\u2014 no answer", html)



class WattBench(unittest.TestCase):
    """Speed per watt on a benchmark row. nvidia-smi is a fake on PATH. No driver, no power-cap write."""

    SMI = r"""
import os, sys
q = ""
for a in sys.argv[1:]:
    if a.startswith("--query-gpu="):
        q = a.split("=", 1)[1]
        break
if q == "index,name,compute_cap,memory.total,memory.used,uuid":
    sys.stdout.write("0, Tesla P100-PCIE-16GB, 6.0, 16384, 100, GPU-fake-0\n")
    sys.exit(0)
if q == "index,power.default_limit":
    sys.stdout.write("0, 250.0\n")
    sys.exit(0)
if q == "index,power.draw,power.limit":
    path = os.environ.get("PXA_WATT_FAKE_COUNT", "")
    n = 0
    if path:
        try:
            with open(path) as f:
                n = int((f.read() or "0").strip() or "0")
        except OSError:
            n = 0
        with open(path, "w") as f:
            f.write(str(n + 1))
    w = 100 + 50 * n
    sys.stdout.write("0, %s, 250.0\n" % w)
    if os.environ.get("PXA_WATT_FAKE_TWO") == "1":
        sys.stdout.write("1, %s, 300.0\n" % (w / 2))
    sys.exit(0)
if os.environ.get("PXA_WATT_FAKE_BOMB") == "1":
    open(os.environ["PXA_WATT_FAKE_BOMB_PATH"], "w").write("hit\n")
sys.exit(1)
"""

    def _install_smi(self, directory):
        path = os.path.join(directory, "nvidia-smi")
        with open(path, "w") as f:
            f.write("#!" + sys.executable + "\n" + self.SMI)
        os.chmod(path, 0o755)
        return path

    def test_parse_skips_missing_readings(self):
        rows = C.App.parse_power_csv("0, 100.5, 250\n1, [N/A], 250\n2, Not Supported, 300\nbad line\nindex, power.draw, power.limit\n")
        self.assertEqual([(r["index"], r["power_w"], r["power_limit_w"]) for r in rows],
                         [(0, 100.5, 250.0), (1, None, 250.0), (2, None, 300.0)])
        self.assertEqual(C.App.parse_power_csv(""), [])
        self.assertEqual(C.App.parse_power_csv(None), [])

    def test_summary_avg_peak_and_tokens_per_watt(self):
        samples = [
            [{"index": 0, "power_w": 100.0, "power_limit_w": 250.0},
             {"index": 1, "power_w": 80.0, "power_limit_w": 250.0},
             {"index": 2, "power_w": None, "power_limit_w": 250.0}],
            [{"index": 0, "power_w": 200.0, "power_limit_w": 250.0},
             {"index": 1, "power_w": 100.0, "power_limit_w": 250.0}],
        ]
        one = C.App.summarize_power(samples, cards=[0], decode_tps=30)
        self.assertEqual((one["avg_w"], one["peak_w"], one["samples"], one["tps_per_w"]), (150.0, 200.0, 2, 0.2))
        self.assertEqual(one["cards"], [{"index": 0, "avg_w": 150.0, "peak_w": 200.0, "limit_w": 250.0, "samples": 2}])
        both = C.App.summarize_power(samples, cards=[0, 1], decode_tps=48)
        self.assertEqual((both["avg_w"], both["peak_w"], both["tps_per_w"]), (240.0, 300.0, 0.2))
        self.assertEqual([c["index"] for c in both["cards"]], [0, 1])
        empty = C.App.summarize_power([[{"index": 0, "power_w": 0.0, "power_limit_w": 250.0}]], cards=[0], decode_tps=40)
        self.assertEqual((empty["avg_w"], empty["peak_w"], empty["tps_per_w"], empty["samples"]), (None, None, None, 0))
        self.assertEqual(C.App.bench_decode_tps([{"class": "edit", "decode_tps": 99}, {"class": "prose", "decode_tps": 40}]), 40.0)
        self.assertEqual(C.App.bench_decode_tps([{"class": "long", "decode_tps": 12}]), 12.0)
        self.assertIsNone(C.App.bench_decode_tps([]))

    def test_infinite_and_nan_readings_never_reach_a_row(self):
        """A host that prints inf / nan, or a decode speed that is not a number, gives a dash (None), never NaN / Infinity in the stored JSON."""
        rows = C.App.parse_power_csv("0, inf, 250\n1, nan, inf\n2, 120, 250\n")
        self.assertEqual([r["power_w"] for r in rows], [None, None, 120.0])
        self.assertIsNone(rows[1]["power_limit_w"])
        p = C.App.summarize_power([[{"index": 0, "power_w": float("inf")}], [{"index": 0, "power_w": float("nan")}]], None, 40)
        self.assertEqual((p["samples"], p["avg_w"], p["peak_w"], p["tps_per_w"]), (0, None, None, None))
        p = C.App.summarize_power([[{"index": 0, "power_w": 100, "power_limit_w": float("inf")}]], None, float("inf"))
        self.assertEqual((p["avg_w"], p["decode_tps"], p["tps_per_w"]), (100.0, None, None))
        self.assertIsNone(p["cards"][0]["limit_w"])
        json.loads(json.dumps(p, allow_nan=False))       # valid JSON for the browser

    def test_fake_gpu_table_never_calls_nvidia_smi(self):
        d = tempfile.mkdtemp(prefix="pxa-watt-bomb-")
        hit = os.path.join(d, "hit")
        self._install_smi(d)
        saved = os.environ["PATH"]
        os.environ["PATH"] = d + os.pathsep + saved
        os.environ["PXA_WATT_FAKE_BOMB"] = "1"
        os.environ["PXA_WATT_FAKE_BOMB_PATH"] = hit
        try:
            self.assertTrue(os.environ.get("PXA_LAUNCH_FAKE_GPUS"))
            app = C.App(L, port=0, models_dirs=[MODELS])
            self.assertEqual(app.sample_gpu_power(), [])
            self.assertFalse(os.path.exists(hit))
        finally:
            os.environ["PATH"] = saved
            os.environ.pop("PXA_WATT_FAKE_BOMB", None)
            os.environ.pop("PXA_WATT_FAKE_BOMB_PATH", None)
            shutil.rmtree(d, ignore_errors=True)

    def test_sample_reads_the_fake_nvidia_smi(self):
        d = tempfile.mkdtemp(prefix="pxa-watt-smi-")
        self._install_smi(d)
        saved_path = os.environ["PATH"]
        saved_fake = os.environ.pop("PXA_LAUNCH_FAKE_GPUS", None)
        os.environ["PATH"] = d + os.pathsep + saved_path
        os.environ["PXA_WATT_FAKE_TWO"] = "1"
        os.environ.pop("PXA_WATT_FAKE_COUNT", None)
        try:
            app = C.App(L, port=0, models_dirs=[MODELS])
            rows = app.sample_gpu_power()
        finally:
            os.environ["PATH"] = saved_path
            if saved_fake is None:
                os.environ.pop("PXA_LAUNCH_FAKE_GPUS", None)
            else:
                os.environ["PXA_LAUNCH_FAKE_GPUS"] = saved_fake
            os.environ.pop("PXA_WATT_FAKE_TWO", None)
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual([(r["index"], r["power_w"], r["power_limit_w"]) for r in rows],
                         [(0, 100.0, 250.0), (1, 50.0, 300.0)])

    def test_bench_row_stores_avg_peak_and_history_keeps_it(self):
        d = tempfile.mkdtemp(prefix="pxa-watt-run-")
        count = os.path.join(d, "n")
        self._install_smi(d)
        keys = ("PATH", "PXA_LAUNCH_FAKE_GPUS", "PXA_LAUNCH_NO_NVML", "PXA_LAUNCH_NO_PROC",
                "PXA_WATT_FAKE_COUNT", "PXA_WATT_FAKE_TWO")
        saved = {k: os.environ.get(k) for k in keys}
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        os.environ.pop("PXA_LAUNCH_FAKE_GPUS", None)
        os.environ["PXA_LAUNCH_NO_NVML"] = "1"
        os.environ["PXA_LAUNCH_NO_PROC"] = "1"
        os.environ["PXA_WATT_FAKE_COUNT"] = count
        os.environ["PXA_WATT_FAKE_TWO"] = "1"
        try:
            app = C.App(L, port=0, models_dirs=[MODELS])
            app.power_sample_s = 0.05
            app.seat.cmd = ["llama-server", "-m", FAKE_MODEL]
            app.seat.req = {"model": FAKE_MODEL, "gpus": [0], "port": 8999}
            app.bench = {"running": True, "lines": [], "result": None, "error": None}

            class Slow(BenchHistoryFields.FakeBench):
                @staticmethod
                def _post_json(url, path, body, timeout=0):
                    time.sleep(0.6)     # >= 2 samples at 0.05 s even when each fake nvidia-smi spawn is slow on a loaded box
                    return BenchHistoryFields.FakeBench._post_json(url, path, body, timeout)

            app._bench_run(Slow, 8999, 1, 0, [0])
            self.assertIsNone(app.bench["error"], app.bench["error"])
            res = app.bench["result"]
            p = res["power"]
            self.assertGreaterEqual(p["samples"], 2, p)
            self.assertEqual(p["interval_s"], 0.05)
            self.assertEqual([c["index"] for c in p["cards"]], [0])          # card 1 was drawing too, and is not in this run
            self.assertEqual(p["cards"][0]["limit_w"], 250.0)
            self.assertGreaterEqual(p["peak_w"], p["avg_w"])
            self.assertGreaterEqual(p["avg_w"], 100.0)
            self.assertEqual(p["decode_tps"], 40.0)
            self.assertEqual(p["tps_per_w"], round(40.0 / p["avg_w"], 3))
            self.assertTrue(any("tokens/s per watt" in ln for ln in app.bench["lines"]), app.bench["lines"])
            stored = C.read_jsonl("bench.jsonl", limit=1)[-1]
            self.assertEqual(stored["power"]["avg_w"], p["avg_w"])
            self.assertIn("telemetry", stored)          # the file keeps the temperature trace; the history route drops it
            import pxa_control.routes as R
            row = [r for r in R.r_bench(app, {}, {})["history"] if r.get("ts") == res["ts"]][0]
            self.assertNotIn("telemetry", row)
            self.assertEqual(row["power"]["peak_w"], p["peak_w"])
            self.assertEqual(row["power"]["tps_per_w"], p["tps_per_w"])
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            shutil.rmtree(d, ignore_errors=True)


class EncodeRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pxa-encode-routes-")
        cls.saved = {k: os.environ.get(k) for k in ("PXA_ENCODE_HOME", "PXA_ENCODER_HOME", "PXA_CONVERT_MODULES", "PXA_PACKAGE_PUBKEY", "HF_ENDPOINT",
                                                     "PXA_LICENCE_URL", "PXA_ENGINE_DIR", "PXA_MODELS_DIR", "PXA_CONTROL_CONFIG_DIR")}
        os.environ["PXA_ENCODE_HOME"] = os.path.join(cls.tmp, "data")
        os.environ["PXA_ENCODER_HOME"] = os.path.join(cls.tmp, "enchome")
        os.environ["PXA_CONVERT_MODULES"] = ""
        os.environ["PXA_PACKAGE_PUBKEY"] = EF.PUB_HEX
        os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(cls.tmp, "cfg")
        cls.models = os.path.join(cls.tmp, "models")
        os.makedirs(cls.models)
        os.environ["PXA_MODELS_DIR"] = cls.models
        cls.engine = EF.make_engine_dir(os.path.join(cls.tmp, "engine"))
        os.environ["PXA_ENGINE_DIR"] = cls.engine
        cls.hf = EF.FakeHF({"tiny/Tiny-Qwen3": EF.tiny_repo()}).start()
        os.environ["HF_ENDPOINT"] = cls.hf.url
        cls.lic = EF.FakeLicence(os.path.join(cls.tmp, "pkg")).start()
        os.environ["PXA_LICENCE_URL"] = cls.lic.url
        cls.app = C.App(L, port=0)
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.app.encode_shutdown()
        cls.hf.stop()
        cls.lic.stop()
        for k, v in cls.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def get(self, path, headers=None):
        code, _, body = req(self.port, path, headers=headers)
        return code, json.loads(body)

    def post(self, path, body=None):
        code, _, raw = req(self.port, path, "POST", body if body is not None else {})
        return code, json.loads(raw)

    def wait_pkg(self):
        t0 = time.time()
        while time.time() - t0 < 60:
            st = self.get("/api/encode/state")[1]
            if not st["pkg"]["running"]:
                return st
            time.sleep(0.1)
        self.fail("package install still running")

    def test_01_static_files_and_the_tab_are_served(self):
        for path, ctype in (("/encode.js", "application/javascript"), ("/encode.css", "text/css")):
            code, h, body = req(self.port, path)
            self.assertEqual(code, 200)
            self.assertTrue(h["Content-Type"].startswith(ctype))
            self.assertGreater(len(body), 1000)
        page = req(self.port, "/")[2].decode()
        self.assertIn('data-tab="encode"', page)
        self.assertIn('id="p-encode"', page)
        self.assertIn('src="/encode.js"', page)
        self.assertIn('href="/encode.css"', page)

    def test_02_empty_start_and_plain_errors(self):
        st = self.get("/api/encode/state")[1]
        self.assertEqual((st["encoders"], st["edition"], st["key_set"]), ([], None, False))
        self.assertEqual(len(st["catalog"]), len(__import__("pxa_encode_plan").TIERS))
        self.assertIn("ko-fi.com", st["how_to_get_pro"]["kofi"])
        for path, body, frag in (("/api/encode/inspect", {}, "Paste a Hugging Face model"), ("/api/encode/inspect", {"source": "justaword"}, "neither a Hugging Face"),
                                 ("/api/encode/plan", {"source": "tiny/Tiny-Qwen3"}, "Pick the cards"), ("/api/encode/start", {}, "Paste a Hugging Face model"),
                                 ("/api/encode/key", {"key": "nope"}, "starts with pxk1."), ("/api/encode/get", {"edition": "gold"}, "Free or Pro"),
                                 ("/api/encode/get", {"edition": "pro"}, "Paste your key first"), ("/api/encode/add-encoder", {"path": "/nope"}, "does not exist"),
                                 ("/api/encode/pause", {"id": "x"}, "no such encode job"), ("/api/encode/test", {"id": "x"}, "no such encode job")):
            code, j = self.post(path, body)
            self.assertEqual(code, 400, (path, j))
            self.assertIn(frag, j["error"])
            self.assertNotIn("Traceback", j["error"])
        code, j = self.get("/api/encode/job?id=zzz")
        self.assertEqual(code, 400)
        code, j = self.get("/api/encode/browse?path=" + urllib_quote(self.tmp))
        self.assertEqual(code, 200)
        self.assertIn("models", [e["name"] for e in j["entries"]])
        # a body that is not an object is ignored, not a 500
        code, _, raw = req(self.port, "/api/encode/inspect", "POST", [1, 2])
        self.assertEqual(code, 400)

    def test_03_free_download_then_a_whole_encode_through_the_routes(self):
        self.assertEqual(self.post("/api/encode/get", {"edition": "free"})[0], 200)
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        self.assertEqual((st["edition"], len(st["encoders"])), ("free", 1))
        code, p = self.post("/api/encode/inspect", {"source": "tiny/Tiny-Qwen3"})
        self.assertEqual((code, p["arch"]), (200, "Qwen3ForCausalLM"))
        code, pl = self.post("/api/encode/plan", {"source": "tiny/Tiny-Qwen3", "cards": [0, 1]})
        self.assertEqual(code, 200)
        self.assertTrue(pl["recommended"])
        self.assertEqual(len(pl["locked"]), 7)
        body = {"source": "tiny/Tiny-Qwen3", "cards": [0, 1], "tier": pl["recommended"][0], "work_dir": os.path.join(self.tmp, "work"), "out_dir": self.models}
        code, ck = self.post("/api/encode/checks", body)
        self.assertEqual((code, ck["can_start"]), (200, True), ck.get("refusal"))
        code, j = self.post("/api/encode/start", body)
        self.assertEqual(code, 200)
        t0 = time.time()
        while time.time() - t0 < 90:
            j = self.get("/api/encode/job?id=" + j["id"])[1]
            if j["status"] not in ("running", "queued"):
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "done", j.get("error"))
        self.assertTrue(os.path.isfile(j["result"]["path"]))
        lg = self.get("/api/encode/log?id=" + j["id"])[1]
        self.assertTrue(any(ln.startswith("pxqe make:") for ln in lg["lines"]))      # the one-command flow: pxqe make (its human lines, not its @pxqe progress lines)
        self.assertFalse(any(ln.startswith("@pxqe") for ln in lg["lines"]))
        self.encoded = j
        type(self).job = j

    def test_04_test_it_uses_the_existing_server_flow(self):
        j = getattr(type(self), "job", None)
        if not j:
            self.skipTest("needs the encode of the previous step")
        calls = []
        real = self.app.start
        self.app.start = lambda b: (calls.append(b), {"ok": True, "text": "stub", "sid": b["sid"]})[1]
        try:
            code, r = self.post("/api/encode/test", {"id": j["id"]})
        finally:
            self.app.start = real
        self.assertEqual(code, 200, r)
        self.assertEqual(r["sid"], "encode-test")
        self.assertEqual(calls[0]["model"], j["result"]["path"])
        self.assertEqual(calls[0]["gpus"], [0, 1])         # the cards it was encoded for
        self.assertIsNone(r["note"])
        self.app.start = lambda b: (calls.append(b), {"ok": True, "text": "stub", "sid": b["sid"]})[1]
        try:
            self.post("/api/encode/test", {"id": j["id"], "cards": [1]})
        finally:
            self.app.start = real
        self.assertEqual(calls[1]["gpus"], [1])
        self.assertIn(self.models, load_cfg_model_dirs())
        self.assertEqual(self.app.profiles()["encode-test"]["name"], "Encode test")
        code, r = self.post("/api/encode/test", {"id": "nope"})
        self.assertEqual(code, 400)

    def test_05_key_is_stored_0600_masked_and_never_returned(self):
        code, st = self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        self.assertEqual(code, 200)
        self.assertTrue(st["key_set"])
        self.assertNotIn(ENC_SECRET, json.dumps(st))
        self.assertTrue(st["key_masked"].startswith("pxk1.K-TEST0001."))
        mode = stat_mode(C.config_path())
        self.assertEqual(mode, 0o600)
        with open(C.config_path()) as f:
            self.assertEqual(json.load(f)["encode"]["licence_key"], EF.GOOD_KEY)
        for path in ("/api/encode/state", "/api/info", "/api/rig", "/api/presets", "/api/user", "/api/servers"):
            code, _, raw = req(self.port, path)
            self.assertNotIn(ENC_SECRET.encode(), raw, path)
        self.assertNotIn(ENC_SECRET, json.dumps(self.app.report_bundle()))
        code, st = self.post("/api/encode/key", {"key": ""})
        self.assertFalse(st["key_set"])
        self.assertNotIn(ENC_SECRET, open(C.config_path()).read())

    def test_06_the_key_only_reaches_the_licence_server_and_only_in_the_body(self):
        self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        self.lic.requests.clear()
        self.hf.requests.clear()
        self.post("/api/encode/get", {"edition": "pro"})
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        self.assertEqual(st["edition"], "pro")
        posts = [r for r in self.lic.requests if r["method"] == "POST"]
        self.assertTrue(posts)
        for r in self.lic.requests:
            self.assertNotIn(ENC_SECRET, r["path"])
            self.assertNotIn(ENC_SECRET, json.dumps(r["headers"]))
            if r["method"] != "POST":
                self.assertNotIn(ENC_SECRET, r["body"])
        self.assertFalse(any(ENC_SECRET in json.dumps(r) for r in self.hf.requests))
        code, lic = self.post("/api/encode/licence", {})
        self.assertEqual(code, 200)
        self.assertEqual(lic["encoders"][0]["licence"]["state"], "valid")

    def test_06b_a_slow_stale_rescan_never_overwrites_a_newer_encoder_pick(self):
        """The RC2.1 flake of test_06 ('free' != 'pro' right after the Pro install): the page polls /api/encode/state, and a poll whose 30 s cache had
        expired runs a full rescan. On a slow box that rescan began BEFORE the install switched the preferred encoder to the new Pro one, finished AFTER
        the install's own rescan and replaced the cache with its stale pick (free). A config change now voids every scan that began before it."""
        svc = self.app.encode()
        found = {e["edition"]: e["path"] for e in svc.encoders(True)}
        self.assertEqual(set(found), {"free", "pro"})
        svc.cfg_set(preferred=found["free"])
        self.assertEqual(svc.selected(True)["edition"], "free")
        gate, started, calls = threading.Event(), threading.Event(), []
        real = C.ENC.AD.discover

        def slow_first(*a, **kw):
            r = real(*a, **kw)              # computed with the OLD preference ...
            calls.append(1)
            if len(calls) == 1:
                started.set()
                gate.wait(30)               # ... and delivered late, like a rescan starved of CPU
            return r
        C.ENC.AD.discover = slow_first
        try:
            poll = threading.Thread(target=lambda: svc.encoders(True), daemon=True)
            poll.start()
            self.assertTrue(started.wait(30))
            svc.cfg_set(preferred=found["pro"])         # what the install does ...
            self.assertEqual(svc.selected(True)["edition"], "pro")      # ... and its own rescan
            gate.set()
            poll.join(30)
            self.assertFalse(poll.is_alive())
        finally:
            C.ENC.AD.discover = real
        self.assertEqual(svc.selected()["edition"], "pro", "the stale scan replaced the cache")
        self.assertEqual(self.get("/api/encode/state")[1]["edition"], "pro")

    def test_07_encode_tab_unavailable_is_a_plain_400_not_a_500(self):
        real = C.ENC
        C.ENC = None
        a = C.App(L, port=0)
        srv, port = serve_app(a)
        try:
            code, _, raw = req(port, "/api/encode/state")
            self.assertEqual(code, 400)
            self.assertIn("not available in this install", json.loads(raw)["error"])
        finally:
            C.ENC = real
            srv.shutdown()
            srv.server_close()

    def test_08_the_gpu_runtime_routes(self):
        """A Pro encoder on a computer without the CUDA libraries: state names what is missing, GET runtime fetches the offer, POST runtime/get downloads +
        verifies + installs it, and the encoder loads afterwards. Public data only: no key goes to /v1/runtime/*."""
        self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        self.lic.pro_fake = {"needs_runtime": True}
        self.lic.rebuild()
        self.post("/api/encode/get", {"edition": "pro", "update": True})
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        rt = st["runtime"]
        self.assertTrue(rt["applies"] and rt["need"], rt)
        self.assertEqual(rt["missing"], ["libcublas.so.12", "libcusolver.so.11"])
        self.assertIsNone(rt["offer"])
        code, v = self.get("/api/encode/runtime")
        self.assertEqual(code, 200)
        self.assertEqual((v["offer"]["id"], v["need"]), ("cuda12.8.1-r1", True))
        self.assertIn("B", v["offer"]["size_h"])
        self.lic.requests.clear()
        self.assertEqual(self.post("/api/encode/runtime/get", {})[0], 200)
        t0 = time.time()
        while time.time() - t0 < 60:
            st = self.get("/api/encode/state")[1]
            if not st["runtime"]["job"]["running"]:
                break
            time.sleep(0.1)
        self.assertEqual(st["runtime"]["job"]["phase"], "done", st["runtime"]["job"])
        self.assertTrue(st["runtime"]["ok"] and not st["runtime"]["need"])
        self.assertEqual(st["runtime"]["source"], "pack")
        self.assertEqual(st["encoders"][0]["runtime"]["lib"], "loadable")
        self.assertTrue(all(r["method"] == "GET" and r["body"] == "" for r in self.lic.requests if "/v1/runtime/" in r["path"]))
        self.assertEqual(self.post("/api/encode/runtime/cancel", {})[0], 200)
        self.assertEqual(self.post("/api/encode/runtime/get", {})[0], 200)                  # already installed: it is fetched again, harmlessly
        self.wait_pkg()
        t0 = time.time()
        while self.get("/api/encode/state")[1]["runtime"]["job"]["running"] and time.time() - t0 < 60:
            time.sleep(0.1)
        self.lic.pro_fake = {}
        self.lic.rebuild()

    def test_09_who_can_load_this_file_through_the_routes(self):
        """The lock choice end to end through the HTTP routes: the plan carries what the key's tier may choose, a choice the plan does not allow is refused in
        one sentence before anything runs, an allowed one reaches the encoder, and the finished job says who can load the file. The server's switch off:
        the choice is disabled and the file is written open."""
        self.post("/api/encode/key", {"key": EF.GOOD_KEY})
        on = {"enabled": True, "allowed_modes": ["personal", "supporters"], "default_mode": "personal", "epoch": "2026-10"}
        self.lic.pro_fake = {"lock_support": True, "lock_server": on}
        self.lic.rebuild()
        self.post("/api/encode/get", {"edition": "pro", "update": True})
        st = self.wait_pkg()
        self.assertEqual(st["pkg"]["phase"], "done", st["pkg"])
        self.assertEqual(self.post("/api/encode/rescan", {})[0], 200)
        base = {"source": "tiny/Tiny-Qwen3", "cards": [0, 1], "work_dir": os.path.join(self.tmp, "work-lock"), "out_dir": self.models}
        code, pl = self.post("/api/encode/plan", base)
        self.assertEqual(code, 200)
        self.assertEqual((pl["lock"]["state"], [c["label"] for c in pl["lock"]["choices"]], pl["lock"]["default"]),
                         ("on", ["Only me (recommended)", "Any PXA supporter"], "personal"))
        code, ck = self.post("/api/encode/checks", dict(base, tier="pxqn4", lock="open"))
        self.assertEqual(code, 200)
        self.assertFalse(ck["can_start"])
        self.assertIn("Your plan does not allow a file that anyone can load (no lock). It allows: only me, any PXA supporter.", ck["refusal"])
        code, err = self.post("/api/encode/start", dict(base, tier="pxqn4", lock="open"))
        self.assertEqual(code, 400)
        self.assertIn("It allows: only me, any PXA supporter.", err["error"])
        code, err = self.post("/api/encode/start", dict(base, tier="pxqn4", lock="everyone"))
        self.assertEqual(code, 400)
        self.assertIn("only you, any PXA supporter, or anyone", err["error"])
        code, j = self.post("/api/encode/start", dict(base, tier="pxqn4", lock="supporters"))
        self.assertEqual(code, 200)
        t0 = time.time()
        while time.time() - t0 < 90:
            j = self.get("/api/encode/job?id=" + j["id"])[1]
            if j["status"] not in ("running", "queued"):
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "done", j.get("error"))
        self.assertEqual(j["params"]["lock"], "supporters")
        lock = j["result"]["lock"]
        self.assertEqual((lock["mode"], lock["locked"], lock["title"]), ("supporters", True, "Locked to PXA supporters"))
        self.assertIn("loads in PXA v3 or newer", lock["needs"])
        self.assertEqual(L.gguf_header(j["result"]["path"])["kv"].get("pxa.lock.mode"), "supporters")
        # the licence server's switch goes off: the page reads it again on a Rescan; the choice is disabled and the next file is open
        self.lic.pro_fake = {"lock_support": True, "lock_server": {"enabled": False, "allowed_modes": ["open"], "default_mode": "open", "epoch": "2026-10"}}
        self.lic.rebuild()
        self.post("/api/encode/get", {"edition": "pro", "update": True})
        self.assertEqual(self.wait_pkg()["pkg"]["phase"], "done")
        self.post("/api/encode/rescan", {})
        pl = self.post("/api/encode/plan", base)[1]
        self.assertEqual((pl["lock"]["state"], pl["lock"]["choices"], pl["lock"]["text"]), ("off", [], "File locking turns on with the next encoder update"))
        code, j = self.post("/api/encode/start", dict(base, tier="pxqn3", lock="personal"))
        self.assertEqual(code, 200)
        t0 = time.time()
        while time.time() - t0 < 90:
            j = self.get("/api/encode/job?id=" + j["id"])[1]
            if j["status"] not in ("running", "queued"):
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "done", j.get("error"))
        self.assertEqual((j["params"]["lock"], j["result"]["lock"]["locked"], j["result"]["lock"]["title"]), ("open", False, "Not locked"))
        self.lic.pro_fake = {}
        self.lic.rebuild()


class EncodeAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pxa-encode-auth-")
        cls.saved = os.environ.get("PXA_ENCODE_HOME")
        os.environ["PXA_ENCODE_HOME"] = os.path.join(cls.tmp, "data")
        cls.app = C.App(L, port=7777, lan=True, token="s3cret-token")
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        if cls.saved is None:
            os.environ.pop("PXA_ENCODE_HOME", None)
        else:
            os.environ["PXA_ENCODE_HOME"] = cls.saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_encode_route_needs_the_token_and_a_same_origin_post(self):
        for path in ("/api/encode/state", "/api/encode/job?id=x", "/api/encode/log?id=x", "/api/encode/browse", "/api/encode/runtime"):
            self.assertEqual(req(self.port, path)[0], 401, path)
        for path in ("/api/encode/start", "/api/encode/key", "/api/encode/get", "/api/encode/add-encoder", "/api/encode/test", "/api/encode/cancel", "/api/encode/runtime/get", "/api/encode/runtime/cancel"):
            self.assertEqual(req(self.port, path, "POST", {})[0], 401, path)
            self.assertEqual(req(self.port, path, "POST", {}, headers={"X-PXA-Token": "s3cret-token", "Origin": "http://evil.example"})[0], 403, path)
        self.assertEqual(req(self.port, "/api/encode/state", headers={"X-PXA-Token": "s3cret-token"})[0], 200)


# ---------------------------------------------------------------------------------------------
# the Live tab: rolling history of the cards and the servers
# ---------------------------------------------------------------------------------------------
class ScriptedEngine(http.server.BaseHTTPRequestHandler):
    """what a PXA llama-server answers, from a dict the test edits between samples."""
    state = {}

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
        st, p = ScriptedEngine.state, self.path.split("?")[0]
        if p == "/health":
            return self._send(200, {"status": "ok"})
        if p == "/slots":
            if st.get("no_slots"):
                return self._send(501, {"error": "no slots"})
            return self._send(200, st["slots"])
        if p == "/props":
            return self._send(200, st["props"])
        if p == "/metrics":
            if not st.get("metrics"):
                return self._send(501, {"error": "no metrics"})
            return self._send(200, "".join("llamacpp:%s %s\n" % kv for kv in st["metrics"].items()).encode(), "text/plain")
        if p == "/pxa/stats":
            return self._send(200, {"records": st.get("records", [])})
        self._send(404, {"error": "nf"})


def slot(i, task, busy, n_decoded, remain=-1, dtot=0, dacc=0, ema=0.7):
    return {"id": i, "id_task": task, "state": 1 if busy else 0, "prompt": "SECRET PROMPT TEXT " * 50,
            "next_token": {"n_decoded": n_decoded, "n_remain": remain}, "n_draft_total": dtot, "n_draft_accepted": dacc,
            "spec_accept_ema": ema}


class LiveHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ScriptedEngine)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.eport = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.app = C.App(L, port=7777, models_dirs=[MODELS])
        snap = {"instances": [{"key": "m:main", "kind": "managed", "name": "Server 1", "label": "Server 1", "running": True,
                               "port": self.eport, "gpus": [0, 1], "health": "ok", "vram": {}, "model_file": "tiny.gguf"}], "cards": []}
        self.app.fleet = lambda max_age=2.5: snap
        self.snap = snap
        ScriptedEngine.state = {"slots": [slot(0, -1, False, 0)], "props": {"n_ctx": 1000, "kv_cache_used_cells": 100, "total_slots": 1}}

    def tearDown(self):
        self.app.live.close()

    def rows(self):
        return self.app.live.snapshot(since=0)["servers"][0]["rows"]

    def col(self, name):
        return C.SRV_COLS.index(name)

    def test_decode_rate_from_slot_progress(self):
        lv = self.app.live
        t = 10000.0
        lv.tick(now=t)                                                   # the first sample has nothing to difference
        self.assertIsNone(self.rows()[-1][self.col("dec")])
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 0, 100)]
        lv.tick(now=t + 2)
        r = self.rows()[-1]
        self.assertEqual((r[self.col("busy")], r[self.col("dec")]), (1, None))       # reading the prompt: nothing written yet, no speed
        self.assertEqual(lv.snapshot(0)["servers"][0]["phase"], "prefill")
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 70, 30)]
        lv.tick(now=t + 4)
        self.assertAlmostEqual(self.rows()[-1][self.col("dec")], 35.0, places=1)
        self.assertEqual(lv.snapshot(0)["servers"][0]["phase"], "decode")
        self.assertEqual(lv.snapshot(0)["servers"][0]["source"], "slot progress")
        ScriptedEngine.state["slots"] = [slot(0, -1, False, 100)]
        lv.tick(now=t + 6)
        self.assertIsNone(self.rows()[-1][self.col("dec")])                          # idle is a gap in the line, not a 0 t/s

    def test_a_slot_that_stalls_while_writing_is_zero(self):
        lv = self.app.live
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 50, 50)]
        lv.tick(now=25000.0)
        lv.tick(now=25002.0)                                                         # same counter two seconds later: it is stuck
        self.assertEqual(self.rows()[-1][self.col("dec")], 0.0)

    def test_a_new_task_does_not_count_the_old_counter(self):
        lv = self.app.live
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 500, 10)]
        lv.tick(now=20000.0)
        ScriptedEngine.state["slots"] = [slot(0, 8, True, 20, 100)]                  # another task on the same slot: counter restarted
        lv.tick(now=20002.0)
        self.assertIsNone(self.rows()[-1][self.col("dec")])

    def test_speculation_and_expert_cache_rates(self):
        lv = self.app.live
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 10, 90, dtot=20, dacc=10, ema=0.55)]
        ScriptedEngine.state["props"].update(pxa_xcache={"hits": 1000, "misses": 100, "swaps_started": 5})
        lv.tick(now=30000.0)
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 60, 40, dtot=60, dacc=40, ema=0.6)]
        ScriptedEngine.state["props"].update(pxa_xcache={"hits": 1090, "misses": 110, "swaps_started": 9})
        lv.tick(now=30002.0)
        r = self.rows()[-1]
        self.assertAlmostEqual(r[self.col("acc")], 30 / 40)
        self.assertAlmostEqual(r[self.col("ema")], 0.6)
        self.assertAlmostEqual(r[self.col("xhit")], 0.9)
        self.assertAlmostEqual(r[self.col("xswap")], 2.0)
        self.assertAlmostEqual(r[self.col("ctx")], 0.1)
        meta = lv.snapshot(0)["servers"][0]
        self.assertTrue(meta["has_spec"] and meta["has_xcache"])
        # counters going backwards (a restarted server) are a gap, never a negative rate
        ScriptedEngine.state["props"].update(pxa_xcache={"hits": 5, "misses": 1, "swaps_started": 0})
        lv.tick(now=30004.0)
        self.assertIsNone(self.rows()[-1][self.col("xhit")])

    def test_exact_counters_when_the_server_has_metrics(self):
        lv = self.app.live
        m = {"tokens_predicted_total": 1000, "tokens_predicted_seconds_total": 30.0, "prompt_tokens_total": 5000, "prompt_seconds_total": 6.0}
        ScriptedEngine.state.update(metrics=dict(m), slots=[slot(0, 7, True, 5, 90)])
        lv.tick(now=40000.0)
        ScriptedEngine.state["metrics"] = {"tokens_predicted_total": 1100, "tokens_predicted_seconds_total": 32.0,
                                           "prompt_tokens_total": 6000, "prompt_seconds_total": 7.0}
        lv.tick(now=40002.0)
        r = self.rows()[-1]
        self.assertAlmostEqual(r[self.col("dec")], 50.0)             # 100 tokens in 2 busy seconds
        self.assertAlmostEqual(r[self.col("pre")], 1000.0)           # 1000 prompt tokens in 1 busy second
        self.assertEqual(lv.snapshot(0)["servers"][0]["source"], "engine counters")

    def test_no_slots_endpoint_is_a_gap_not_a_crash(self):
        ScriptedEngine.state["no_slots"] = True
        self.app.live.tick(now=50000.0)
        self.app.live.tick(now=50002.0)
        r = self.rows()[-1]
        self.assertEqual((r[self.col("dec")], r[self.col("busy")]), (None, None))
        self.assertAlmostEqual(r[self.col("ctx")], 0.1)               # /props still answers

    def test_prompt_text_never_leaves_the_server_object(self):
        ScriptedEngine.state["props"]["model_alias"] = "/home/someone/models/secret-dir/tiny.gguf"
        ScriptedEngine.state["slots"] = [slot(0, 7, True, 5, 90)]
        self.app.live.tick(now=60000.0)
        self.app.live.tick(now=60002.0)
        snap = json.dumps(self.app.live.snapshot(0))
        self.assertNotIn("SECRET", snap)
        self.assertNotIn("secret-dir", snap)                          # a model is named by its file, never by its path
        self.assertEqual(self.app.live.snapshot(0)["servers"][0]["model"], "tiny.gguf")

    def test_finished_requests_are_kept_once(self):
        lv = self.app.live
        rec = {"ts": 70010.0, "slot": 0, "n_prompt": 900, "n_cached": 100, "prompt_n": 800, "prompt_ms": 1000.0, "prefill_tps": 800.0,
               "n_gen": 100, "gen_ms": 3000.0, "decode_tps": 33.3, "draft_n": 40, "draft_acc": 30}
        ScriptedEngine.state["records"] = [rec]
        lv.tick(now=70012.0)
        lv.tick(now=70020.0)                                         # same record offered again: not added twice
        reqs = lv.snapshot(0)["servers"][0]["reqs"]
        self.assertEqual(len(reqs), 1)
        self.assertEqual(dict(zip(C.REQ_COLS, reqs[0]))["decode_tps"], 33.3)
        ScriptedEngine.state["records"] = [rec, dict(rec, ts=70030.0, slot=1)]
        lv.tick(now=70032.0)
        lv.tick(now=70040.0)
        self.assertEqual(len(lv.snapshot(0)["servers"][0]["reqs"]), 2)

    def test_since_step_and_pruning(self):
        lv = self.app.live
        now = time.time()
        for i in range(10):
            lv.tick(now=now - 100 + i * 10)
        s = lv.snapshot(since=now - 45)
        self.assertEqual(len(s["servers"][0]["rows"]), 4)            # rows newer than `since`, nothing older
        b = lv.snapshot(since=0, step=30)
        self.assertLess(len(b["servers"][0]["rows"]), 10)
        lv.tick(now=now + C.LIVE_KEEP_S + 100)                       # everything older than the keep time goes
        s = lv.snapshot(since=0)
        self.assertEqual(len(s["servers"][0]["rows"]), 1)
        self.assertEqual(len(s["cards"][0]["rows"]), 1)

    def test_colour_slot_is_first_seen_and_never_repainted(self):
        lv = self.app.live
        a, b, c = lv.slot_for("m:main"), lv.slot_for("d:seat"), lv.slot_for("p:1")
        self.assertEqual((a, b, c), (0, 1, 2))
        self.assertEqual(lv.slot_for("d:seat"), 1)
        self.assertEqual(lv.slot_for("m:main"), 0)
        self.assertEqual([lv.slot_for("x%d" % i) for i in range(6)], [3, 4, 5, 6, 7, 0])     # eight colours, then they repeat

    def test_stopped_server_keeps_its_history_until_it_ages_out(self):
        lv = self.app.live
        lv.tick(now=time.time())
        self.snap["instances"][0]["running"] = False
        lv.tick(now=time.time() + 2)
        s = lv.snapshot(since=0)["servers"][0]
        self.assertFalse(s["up"])
        self.assertTrue(s["rows"])

    def test_route_and_the_first_look(self):
        srv, port = serve_app(self.app)
        try:
            code, _, body = req(port, "/api/live")
            self.assertEqual(code, 200, body)
            d = json.loads(body)
            self.assertEqual(d["cols"]["card"], C.CARD_COLS)
            self.assertGreaterEqual(d["ticks"], 1)                   # the first look takes a sample right away
            self.assertEqual(len(d["cards"]), 2)                     # the fake rig has two cards
            self.assertEqual([s["key"] for s in d["servers"]], ["m:main"])
            self.assertEqual(json.loads(req(port, "/api/live?since=abc&step=zzz")[2])["since"] > 0, True)
            self.assertEqual(req(port, "/live.js")[0], 200)
            self.assertEqual(req(port, "/live.css")[0], 200)
            page = req(port, "/")[2].decode()
            self.assertIn('data-tab="live"', page)
            self.assertIn('src="/live.js"', page)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_gpu_live_one_call_and_throttle_field_discovery(self):
        calls = []
        old_which, old_fake = C.shutil_which, os.environ.pop("PXA_LAUNCH_FAKE_GPUS", None)
        try:
            C.shutil_which = lambda x: "/usr/bin/" + x
            self.app.gpus = lambda max_age=1.5: ([(0, "NVIDIA Tesla V100-PCIE-16GB", 70, 16384, 0, "u0"), (1, "NVIDIA GeForce GTX 1080 Ti", 61, 11264, 0, "u1")], None)

            def run(cmd, timeout=20):
                calls.append(cmd)
                q = [a for a in cmd if a.startswith("--query-gpu=")][0]
                if "clocks_throttle_reasons.active" in q:
                    return None                                       # a newer driver renamed it
                if "clocks_event_reasons.active" in q:
                    return "0, 9000, 71, 64, 172.5, 250, 1380, 0x4\n1, [N/A], 0, 40, [N/A], 250, 1500, 0x0"
                return None
            self.app.L = type("LL", (), {"_run": staticmethod(run), "CARD_CLASS": L.CARD_CLASS})
            rows = self.app.gpu_live()
            self.assertEqual([r["index"] for r in rows], [0, 1])
            self.assertEqual((rows[0]["mem"], rows[0]["util"], rows[0]["temp"], rows[0]["power"], rows[0]["throttle"]), (9000.0, 71.0, 64.0, 172.5, ["sw_power_cap"]))
            self.assertEqual((rows[1]["mem"], rows[1]["power"], rows[1]["throttle"]), (0, None, []))     # [N/A] is None, memory falls back to the card table
            self.assertEqual(self.app._thr_field if hasattr(self.app, "_thr_field") else self.app.live._thr_field, "clocks_event_reasons.active")
            n = len(calls)
            self.app.gpu_live()
            self.assertEqual(len(calls) - n, 1)                       # the working field is remembered: one call per reading
        finally:
            C.shutil_which = old_which
            if old_fake is not None:
                os.environ["PXA_LAUNCH_FAKE_GPUS"] = old_fake

    @unittest.skipUnless(os.path.exists("/proc/meminfo") and os.path.exists("/proc/stat"), "needs /proc")
    def test_host_memory_and_cpu(self):
        lv = self.app.live
        now = time.time()
        lv.tick(now=now)
        lv.tick(now=now + 2)
        h = lv.snapshot(0)["host"]
        self.assertEqual(len(h["rows"]), 2)
        t, ram, cpu, swap = h["rows"][-1]
        self.assertTrue(0 < ram <= h["ram_total_mib"])
        self.assertTrue(cpu is not None and 0 <= cpu <= 100)               # the first sample has nothing to difference, the second does
        self.assertIsNone(h["rows"][0][2])

    def test_bucket_rows(self):
        rows = [(0.0, 1.0, None), (1.0, 3.0, None), (2.0, None, 4.0), (10.0, 5.0, 6.0), (11.0, 7.0, None), (12.0, 9.0, None)]
        out = C.bucket_rows(rows, 5)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1:], [2.0, 4.0])
        self.assertEqual(out[1][1:], [7.0, 6.0])
        self.assertEqual(C.bucket_rows(rows, 0), [list(r) for r in rows])


# ---------------------------------------------------------------------------------------------
# v3.1 telemetry history: the Live sampler keeps running in the background and fills telemetry.db
# ---------------------------------------------------------------------------------------------
class TelemetryHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ScriptedEngine)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.eport = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        for k in ("PXA_CONTROL_TELEMETRY", "PXA_CONTROL_METRICS", "PXA_CONTROL_TELEMETRY_DB"):
            os.environ.pop(k, None)
        self.apps = []
        ScriptedEngine.state = {"slots": [slot(0, -1, False, 0)], "props": {"n_ctx": 1000, "kv_cache_used_cells": 250, "total_slots": 1}}

    def tearDown(self):
        for app, srv in self.apps:
            if srv is not None:
                srv.shutdown()
                srv.server_close()
            app.live.close()
            app.stop_telemetry()
        for k in ("PXA_CONTROL_TELEMETRY", "PXA_CONTROL_METRICS", "PXA_CONTROL_TELEMETRY_DB"):
            os.environ.pop(k, None)

    def make(self, key="m:main", lan=False, token=None, http=True):
        app = C.App(L, port=7777, models_dirs=[MODELS], lan=lan, token=token)
        snap = {"instances": [{"key": key, "kind": "managed" if key.startswith("m:") else "process", "name": "Server 1",
                               "label": "Server 1", "running": True, "port": self.eport, "gpus": [0, 1], "health": "ok",
                               "vram": {}, "model_file": "tiny.gguf"}], "cards": []}
        app.fleet = lambda max_age=2.5: snap
        srv = port = None
        if http:
            srv, port = serve_app(app)
        self.apps.append((app, srv))
        return app, port

    def drive(self, app, n=4, every=10.0):
        """n samples, `every` seconds apart, ending now; decoding at 50 t/s with two finished requests."""
        t0 = time.time() - n * every
        for i in range(n):
            ScriptedEngine.state["slots"] = [slot(0, 7, True, 1 + int(50 * every * i), 100000)]
            ScriptedEngine.state["records"] = [
                {"ts": t0 + 1, "slot": 0, "n_prompt": 900, "n_cached": 100, "prompt_n": 800, "prompt_ms": 1000.0, "prefill_tps": 800.0,
                 "n_gen": 100, "gen_ms": 2000.0, "decode_tps": 50.0, "draft_n": 0, "draft_acc": 0},
                {"ts": t0 + every * i + 2, "slot": 0, "n_prompt": 50, "n_cached": 0, "prompt_n": 50, "prompt_ms": 100.0, "prefill_tps": 500.0,
                 "n_gen": 10, "gen_ms": 200.0, "decode_tps": 50.0, "draft_n": 0, "draft_acc": 0}]
            app.live.tick(now=t0 + every * i, wait=True)
        app.telemetry.flush(force=True)
        self.assertTrue(app.telemetry.sync())

    def test_history_is_recorded_and_served(self):
        app, port = self.make()
        self.assertIsNotNone(app.start_telemetry())
        self.assertTrue(app.telemetry.is_writer)
        self.drive(app)
        code, d = jreq(port, "/api/telemetry/series?kind=server&since=%d&step=10" % (time.time() - 3600))
        self.assertEqual(code, 200)
        self.assertEqual(d["source"], "raw")
        s = d["series"][0]
        self.assertEqual((s["key"], s["label"], s["info"]["model_file"]), ("m:main", "Server 1", "tiny.gguf"))
        col = d["cols"].index
        decs = [r[col("dec")] for r in s["rows"] if r[col("dec")] is not None]
        self.assertTrue(decs and all(abs(v - 50.0) < 0.5 for v in decs), decs)
        self.assertTrue(all(abs(r[col("ctx")] - 0.25) < 1e-6 for r in s["rows"]))
        # the old record back-filled by the first poll is kept as a request but not counted into an interval
        self.assertLessEqual(sum(r[col("req")] or 0 for r in s["rows"]), 4)
        code, cards = jreq(port, "/api/telemetry/series?kind=card&since=%d" % (time.time() - 3600))
        self.assertEqual(sorted(x["key"] for x in cards["series"]), ["0", "1"])
        code, r = jreq(port, "/api/telemetry/requests?since=%d" % (time.time() - 3600))
        self.assertGreaterEqual(len(r["rows"]), 2)
        code, h = jreq(port, "/api/telemetry/history?since=%d" % (time.time() - 3600))
        self.assertEqual(code, 200)
        self.assertEqual(set(h["cols"]), {"card", "server", "host"})
        self.assertEqual(h["servers"][0]["key"], "m:main")
        code, st = jreq(port, "/api/telemetry")
        self.assertTrue(st["recording"])
        self.assertGreater(st["raw_rows"], 0)
        self.assertIn("server", {x["kind"] for x in st["series_list"]})

    def test_prompt_text_and_paths_never_reach_the_file(self):
        app, port = self.make()
        app.start_telemetry()
        ScriptedEngine.state["props"]["model_alias"] = "/home/someone/secret-dir/tiny.gguf"
        self.drive(app)
        app.stop_telemetry()
        blob = b""
        for p in glob_db(C.config_dir()):
            with open(p, "rb") as f:
                blob += f.read()
        self.assertNotIn(b"SECRET", blob)
        self.assertNotIn(b"secret-dir", blob)

    def test_csv_download(self):
        app, port = self.make()
        app.start_telemetry()
        self.drive(app)
        u = "http://127.0.0.1:%d/api/telemetry/csv?kind=card&since=%d&step=10" % (port, time.time() - 3600)
        with urllib.request.urlopen(urllib.request.Request(u, headers={"X-PXA-Token": token_for(port)}), timeout=10) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/csv"))
            self.assertIn("attachment;", r.headers["Content-Disposition"])
            lines = r.read().decode().splitlines()
        self.assertTrue(lines[0].startswith("time_utc,unix_s,card,label,mem"))
        self.assertGreaterEqual(len(lines), 1 + 2)
        self.assertEqual(req(port, "/api/telemetry/csv?kind=disk")[0], 400)

    def test_off_by_env_and_by_setting(self):
        os.environ["PXA_CONTROL_TELEMETRY"] = "0"
        app, port = self.make()
        self.assertIsNone(app.start_telemetry())
        code, d = jreq(port, "/api/telemetry/series?kind=card")
        self.assertEqual(code, 400)
        self.assertIn("off", d["error"])
        self.assertFalse(app.live.background)
        os.environ.pop("PXA_CONTROL_TELEMETRY")
        code, st = jreq(port, "/api/telemetry/settings", "POST", {"enabled": True, "raw_days": 3})
        self.assertEqual(code, 200)
        self.assertTrue(st["recording"])
        self.assertEqual(st["settings"]["raw_days"], 3.0)
        self.assertEqual(C.load_config()["telemetry"]["raw_days"], 3.0)
        code, st = jreq(port, "/api/telemetry/settings", "POST", {"enabled": False})
        self.assertFalse(st["recording"])
        self.assertIsNone(app.telemetry)
        self.assertEqual(req(port, "/api/telemetry/settings", "POST", {"colour": "red"})[0], 400)

    def test_background_sampler_runs_with_nobody_looking(self):
        app, port = self.make(http=False)
        app.start_telemetry()
        self.assertTrue(app.live.background)
        app.live.bg_every = 0.25                     # test pace; the shipping floor is LIVE_FAST_S
        app.live.last_any = time.time() - C.LIVE_IDLE_S - 60     # the last viewer left long ago
        t = time.time()
        while app.live.ticks < 3 and time.time() - t < 15:
            time.sleep(0.1)
        self.assertGreaterEqual(app.live.ticks, 3)
        self.assertTrue(app.live.thread.is_alive())

    def test_without_history_the_sampler_still_stops_when_idle(self):
        app, port = self.make(http=False)
        app.live.touch()
        app.live.last_any = time.time() - C.LIVE_IDLE_S - 60
        app.live.last_fast = 0
        t = time.time()
        while app.live.thread is not None and app.live.thread.is_alive() and time.time() - t < 15:
            time.sleep(0.2)
        self.assertFalse(app.live.thread is not None and app.live.thread.is_alive())

    def test_second_control_reads_but_does_not_record(self):
        a, _p = self.make(http=False)
        a.start_telemetry()
        b, port_b = self.make()
        st = b.start_telemetry()
        self.assertFalse(st.is_writer)
        self.assertFalse(b.live.background)
        self.drive(a)
        code, d = jreq(port_b, "/api/telemetry/series?kind=card&since=%d" % (time.time() - 3600))
        self.assertEqual(code, 200)
        self.assertEqual(len(d["series"]), 2)
        self.assertFalse(jreq(port_b, "/api/telemetry")[1]["recording"])

    def test_bare_process_is_filed_under_its_port(self):
        app, port = self.make(key="p:4242")
        app.start_telemetry()
        self.drive(app, n=2)
        keys = [x["key"] for x in app.telemetry.list_series("server")]
        self.assertEqual(keys, ["p@%d" % self.eport])

    def test_metrics_is_off_until_asked_then_relabels_engine_counters(self):
        app, port = self.make()
        app.start_telemetry()
        ScriptedEngine.state["metrics"] = {"tokens_predicted_total": 1000, "tokens_predicted_seconds_total": 30.0,
                                           "prompt_tokens_total": 5000, "prompt_seconds_total": 6.0}
        self.drive(app, n=2)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen("http://127.0.0.1:%d/metrics" % port, timeout=10)
        self.assertEqual(cm.exception.code, 404)
        jreq(port, "/api/telemetry/settings", "POST", {"prometheus": True})
        with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % port, timeout=10) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/plain"))
            txt = r.read().decode()
        self.assertIn('pxa_card_memory_used_mib{card="0"', txt)
        self.assertIn('pxa_server_kv_used_ratio{server="m:main",name="Server 1"} 0.25', txt)
        self.assertIn('llamacpp:tokens_predicted_total{server="m:main"} 1000', txt)
        self.assertIn("pxa_server_requests_total", txt)

    def test_metrics_on_a_lan_bind_needs_the_token_as_bearer(self):
        os.environ["PXA_CONTROL_METRICS"] = "1"
        app, port = self.make(lan=True, token="s3cret-token-0123456789")
        app.start_telemetry()
        self.drive(app, n=2)
        code, body = raw_get(port, "/metrics")
        self.assertEqual(code, 401)
        self.assertIn(b"Bearer", body)
        code, body = raw_get(port, "/metrics", {"Authorization": "Bearer s3cret-token-0123456789"})
        self.assertEqual(code, 200)
        self.assertIn(b"pxa_card_", body)
        self.assertEqual(raw_get(port, "/metrics", {"Authorization": "Bearer wrong-token-0123456789"})[0], 401)


def jreq(port, path, method="GET", body=None, headers=None):
    code, _h, raw = req(port, path, method, body, headers)
    try:
        return code, json.loads(raw.decode() or "null")
    except ValueError:
        return code, raw


def glob_db(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.startswith("telemetry.db")] if os.path.isdir(d) else []


def raw_get(port, path, headers=None):
    rq = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), headers=dict(headers or {}, Host="127.0.0.1:%d" % port))
    try:
        with urllib.request.urlopen(rq, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ---------------------------------------------------------------------------------------------
# PXA Control opens by itself (2026-10-05): the on/off rule, reuse of a running Control, the port
# fallback, adoption of a server the launcher started (and Stop), and the idle exit of a Control
# that started in the background. Real processes (sleeping children) and loopback sockets only.
# ---------------------------------------------------------------------------------------------
def sleeper(*args, secs=60):
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({secs})"] + list(args))
    # under load /proc/<pid>/cmdline can still show the parent (pre-exec) for a moment: wait for our argv
    want = (list(args) or [f"time.sleep({secs})"])[-1].encode()
    end = time.time() + 10
    while time.time() < end:
        try:
            with open(f"/proc/{p.pid}/cmdline", "rb") as f:
                if want in f.read():
                    break
        except OSError:
            break
        time.sleep(0.02)
    return p


def os_free_port():
    """a port the kernel just handed out (bind to 0): a fixed scan start (free_port(17950)) gave two test processes running
    side by side (ctest -j, or the file twice at once) the same port, so one test talked to the other's Control"""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_until(fn, timeout=15.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(step)
    return fn()


class ControlFirst(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        self.kids, self.srvs = [], []

    def tearDown(self):
        for k in self.kids:
            if k.poll() is None:
                k.kill()
                k.wait()
        for srv in self.srvs:
            srv.shutdown()
            srv.server_close()
        for rec in C.running_controls():            # a background Control a failed test left behind
            if rec["pid"] != os.getpid():
                try:
                    os.kill(rec["pid"], 15)
                except OSError:
                    pass

    def kid(self, *args, secs=60):
        k = sleeper(*args, secs=secs)
        self.kids.append(k)
        return k

    def up(self, lan=False):
        """an in-process Control with its run record, as a running `pxa` would leave it."""
        app = C.App(L, port=0, lan=lan, token="t" * 20 if lan else None, models_dirs=[MODELS])
        srv, port = serve_app(app)
        self.srvs.append(srv)
        app.port = port
        C.write_run_record(port, lan, companion=False, token=None if lan else app.token)   # as serve() writes it (H1)
        return app, port

    def ns(self, *argv):
        return L.build_parser().parse_args(list(argv))

    # ---- the rule --------------------------------------------------------------------------
    def test_autostart_rule(self):
        d = C.autostart_decision
        self.assertEqual(d(False, {}, tty=True, container=None), (True, "a terminal"))
        self.assertFalse(d(False, {}, tty=False, container=None)[0])            # service, script, pipe
        self.assertIn("PXA_CONTROL=1", d(False, {}, tty=False, container=None)[1])
        self.assertFalse(d(False, {}, tty=True, container="docker")[0])         # container: opt-in only
        self.assertIn("-p 7777:7777", d(False, {}, tty=True, container="docker")[1])
        for v in ("1", "on", "yes", "true", "ON"):
            self.assertTrue(d(False, {"PXA_CONTROL": v}, tty=False, container="docker")[0], v)
        for v in ("0", "off", "no", "false"):
            self.assertEqual(d(False, {"PXA_CONTROL": v}, tty=True, container=None), (False, None), v)
        self.assertEqual(d(True, {"PXA_CONTROL": "1"}, tty=True, container=None), (False, None))   # --no-control wins
        self.assertFalse(d(False, {"PXA_CONTROL": "1", "PXA_CONTROL_SPAWNED": "1"}, tty=True)[0])  # never a Control in a Control
        self.assertEqual(C.control_env_mode({"PXA_CONTROL": "auto"}), "auto")

    def test_bare_invocation(self):
        ap = L.build_parser()
        bare = lambda *argv: L.bare_invocation(ap, ap.parse_args(list(argv)))      # noqa: E731
        self.assertTrue(bare())
        self.assertTrue(bare("--models-dir", MODELS, "--no-browser", "--lan", "--control-port", "7800"))
        for argv in (["-m", FAKE_MODEL], ["--gpus", "0"], ["--workload", "chat"], ["--doctor"], ["--explain"],
                     ["--no-control"], ["--no-tui"], ["--port", "8081"], ["--yes"], ["--selftest"]):
            self.assertFalse(bare(*argv), argv)

    def test_launcher_decision_and_off_paths(self):
        a = self.ns("--no-control")
        self.assertEqual(L.control_decision(a, tty=True), (False, None))
        a = self.ns("-m", FAKE_MODEL, "--gpus", "0")
        # the host's own container state must not decide this case (ctest runs inside the build container)
        _ck = L.container_kind
        L.container_kind = lambda: None
        try:
            on, why = L.control_decision(a, tty=False)
            self.assertFalse(on)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertIsNone(L.control_alongside(a, "0", tty=False))
        finally:
            L.container_kind = _ck
        self.assertIn("PXA Control: off (not started from a terminal", buf.getvalue())
        self.assertEqual(C.launched_servers(), {})                              # nothing recorded when off
        os.environ["PXA_CONTROL"] = "0"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertIsNone(L.control_alongside(a, "0", tty=True))
            self.assertEqual(buf.getvalue(), "")                                # your own off switch: silent
        finally:
            os.environ.pop("PXA_CONTROL", None)

    # ---- records ---------------------------------------------------------------------------
    def test_run_records_and_reuse(self):
        self.assertIsNone(C.find_running())
        app, port = self.up()
        r = C.find_running()
        self.assertEqual((r["port"], r["pid"], r["url"]), (port, os.getpid(), f"http://127.0.0.1:{port}/?token={app.token}"))
        self.assertIsNone(C.find_running(port=port + 1))                        # asked for another port
        self.assertIsNone(C.find_running(lan=True))                             # a LAN one was asked for
        # a record whose process is gone is dropped; one whose port is not a Control is ignored
        dead = self.kid(secs=0)
        dead.wait()
        rd = os.path.join(C.config_dir(), "run")
        with open(os.path.join(rd, f"control-{dead.pid}.json"), "w") as f:
            json.dump({"pid": dead.pid, "start": 1, "port": port, "lan": False}, f)
        stub = http.server.HTTPServer(("127.0.0.1", 0), StubEngine)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        try:
            other = self.kid()
            with open(os.path.join(rd, f"control-{other.pid}.json"), "w") as f:
                json.dump({"pid": other.pid, "start": C.proc_start_ticks(other.pid), "port": stub.server_address[1],
                           "lan": False, "companion": False, "since": 0}, f)
            self.assertEqual(C.find_running()["port"], port)
        finally:
            stub.shutdown()
            stub.server_close()
        self.assertFalse(os.path.exists(os.path.join(rd, f"control-{dead.pid}.json")))

    def test_lan_control_is_found_with_its_token(self):
        app, port = self.up(lan=True)
        os.environ["PXA_CONTROL_TOKEN"] = "t" * 20
        try:
            r = C.find_running(lan=True)
            self.assertEqual(r["url"], f"http://127.0.0.1:{port}/?token=" + "t" * 20)
        finally:
            os.environ.pop("PXA_CONTROL_TOKEN", None)

    def test_launched_records_need_the_same_process(self):
        k = self.kid()
        C.record_launched(k.pid, port=18123, model=FAKE_MODEL, gpus="0,1", by="pxa-launch")
        rec = C.launched_servers()[k.pid]
        self.assertEqual((rec["port"], rec["gpus"], rec["by"]), (18123, [0, 1], "pxa-launch"))
        self.assertEqual(stat_mode(os.path.join(C.config_dir(), "launched")), 0o700)
        self.assertEqual(stat_mode(os.path.join(C.config_dir(), "launched", f"{k.pid}.json")), 0o600)
        # the same pid with another start time is another process (a recycled pid): never matched
        p = os.path.join(C.config_dir(), "launched", f"{k.pid}.json")
        with open(p) as f:
            d = json.load(f)
        d["start"] = (d["start"] or 0) + 12345
        with open(p, "w") as f:
            json.dump(d, f)
        self.assertNotIn(k.pid, C.launched_servers())
        self.assertFalse(os.path.exists(p))
        C.record_launched(k.pid, port=18123)
        k.kill()
        k.wait()
        self.assertNotIn(k.pid, C.launched_servers())                           # gone: dropped

    # ---- adoption + Stop --------------------------------------------------------------------
    def test_fleet_adopts_the_launched_server_and_stops_it(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        k = self.kid("--port", "18124", "-m", FAKE_MODEL)
        C.record_launched(k.pid, port=18124, model=FAKE_MODEL, gpus="1", by="run-server.sh")
        x = [i for i in app.fleet(max_age=0)["instances"] if i["key"] == f"p:{k.pid}"]
        self.assertEqual(len(x), 1, "listed even with discovery off")
        x = x[0]
        self.assertEqual((x["kind"], x["origin"], x["adopted"], x["control"], x["port"], x["gpus"], x["model_file"]),
                         ("process", "launcher", True, True, 18124, [1], "tiny.gguf"))
        self.assertEqual(x["actions"], ["log", "stop"])
        self.assertIn("run-server.sh", x["label"])
        with self.assertRaises(C.Invalid):
            app.external_control(x["key"], "restart", "stop")                  # Stop only
        with self.assertRaises(C.Invalid):
            app.external_control(x["key"], "stop", "yes")                      # the typed word
        # a Control the launcher started ALSO watches it on the Live tab (same fleet)
        self.assertIn(x["key"], {i["key"] for i in app.fleet(max_age=0)["instances"] if i.get("running") and i.get("port")})
        r = app.external_control(x["key"], "stop", "stop")
        self.assertTrue(r["ok"])
        self.assertTrue(wait_until(lambda: k.poll() is not None, 10))
        self.assertNotIn(k.pid, C.launched_servers())
        self.assertNotIn(f"p:{k.pid}", {i["key"] for i in app.fleet(max_age=0)["instances"]})

    def test_a_new_launched_server_skips_the_fleet_cache(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        self.assertEqual([i for i in app.fleet(max_age=60)["instances"] if i["kind"] == "process"], [])
        k = self.kid()
        C.record_launched(k.pid, port=18130)
        self.assertIn(f"p:{k.pid}", [i["key"] for i in app.fleet(max_age=60)["instances"]])   # not the 60 s cache

    def test_stop_route_through_http(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        srv, port = serve_app(app)
        self.srvs.append(srv)
        k = self.kid()
        C.record_launched(k.pid, port=18125, model=FAKE_MODEL)
        st, _h, b = req(port, "/api/fleet")
        self.assertEqual(st, 200)
        self.assertIn(f"p:{k.pid}", [i["key"] for i in json.loads(b)["instances"]])
        st, _h, b = req(port, "/api/fleet/control", "POST", {"key": f"p:{k.pid}", "action": "stop", "confirm": "stop"})
        self.assertEqual(st, 200, b)
        self.assertTrue(wait_until(lambda: k.poll() is not None, 10))
        st, _h, b = req(port, "/api/info")
        self.assertFalse(json.loads(b)["companion"])

    # ---- the idle rule ----------------------------------------------------------------------
    def test_idle_clock(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        now = time.time()
        app.last_request = app.last_busy = now - 1000
        self.assertGreaterEqual(app.idle_seconds(now), 999)
        k = self.kid()
        C.record_launched(k.pid, port=18126)
        self.assertEqual(app.idle_seconds(now), 0.0)                            # a launched server runs
        self.assertEqual(app.last_busy, now)
        k.kill()
        k.wait()
        self.assertAlmostEqual(app.idle_seconds(now + 30), 30, delta=1)
        app.last_request = now + 20                                              # a page asked something
        self.assertAlmostEqual(app.idle_seconds(now + 30), 10, delta=1)
        seat = fake_running_seat(app, "main", 18127, [0])
        try:
            self.assertEqual(app.idle_seconds(now + 30), 0.0)                   # a server it started runs
        finally:
            seat.proc.kill()
            seat.proc.wait()

    def test_requests_reset_the_clock(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        srv, port = serve_app(app)
        self.srvs.append(srv)
        app.last_request = 0.0
        req(port, "/api/info")
        self.assertGreater(app.last_request, time.time() - 5)

    def test_companion_closes_itself_when_idle(self):
        out = {}
        port = os_free_port()

        def run():
            # idle_s 3.0, was 0.8 (pass 8e RC): under a parallel ctest the 0.8 s timer could expire before the first poll
            # below reached the server, which then failed "answers" although it closed exactly as designed
            out["rc"] = C.serve(L, port=port, open_browser=False, models_dirs=[MODELS], companion=True, idle_s=3.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            th = threading.Thread(target=run, daemon=True)
            th.start()
            self.assertTrue(wait_until(lambda: C.control_answers(port), 10))
            st, _h, b = req(port, "/api/info")
            info = json.loads(b)
            self.assertEqual((info["companion"], info["idle_exit_s"]), (True, 3.0))
            th.join(20)
        self.assertFalse(th.is_alive(), "a background Control with nothing to do must exit")
        self.assertEqual(out.get("rc"), 0)
        self.assertIn("closing", buf.getvalue())
        self.assertEqual(C.running_controls(), [])                              # its run record is gone

    def test_user_opened_control_does_not_idle_out(self):
        port = os_free_port()
        holder = {}

        def run():
            holder["rc"] = C.serve(L, port=port, open_browser=False, models_dirs=[MODELS], idle_s=0.3)
        with contextlib.redirect_stdout(io.StringIO()):
            th = threading.Thread(target=run, daemon=True)
            th.start()
            self.assertTrue(wait_until(lambda: C.control_answers(port), 10))
            time.sleep(1.5)
            self.assertTrue(th.is_alive())                                       # `pxa` / --gui: until Ctrl-C
            rec = C.find_running(port=port)
            self.assertFalse(rec["companion"])
            C.SERVING[port].shutdown()                                           # what Ctrl-C ends in
            th.join(10)
        self.assertEqual(holder.get("rc"), 0)
        self.assertEqual(C.running_controls(), [])

    # ---- start or reuse ---------------------------------------------------------------------
    def test_token_for_trusts_a_live_run_record_over_a_closed_test_server(self):
        # a test server that served this port earlier left its token in TOKENS; the port came back to a spawned
        # Control (ephemeral reuse, likelier under ctest -j 8), whose run record holds the token that counts
        port = 40000 + os.getpid() % 20000
        TOKENS[port] = "stale-token-of-a-closed-test-server"
        try:
            C.write_run_record(port, False, True, token="live-token-of-the-spawned-control")
            self.assertEqual(token_for(port), "live-token-of-the-spawned-control")
        finally:
            TOKENS.pop(port, None)

    def test_ensure_control_reuses_the_running_one(self):
        app, port = self.up()
        r = C.ensure_control("/nonexistent/pxa-launch.py", port=port + 1)
        self.assertEqual((r.get("reused"), r.get("port")), (True, port))

    def test_ensure_control_starts_one_in_the_background_on_the_next_free_port(self):
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        want = busy.getsockname()[1]
        k = self.kid()
        C.record_launched(k.pid, port=18128, model=FAKE_MODEL)
        os.environ["PXA_CONTROL_IDLE_S"] = "1"
        try:
            r = C.ensure_control(os.path.join(TOOLS, "pxa-launch.py"), port=want, avoid={want + 1},
                                 wait_s=30.0)   # the default 8 s missed under ctest -j 4 at load ~48 (RC2.1 r21a)
        finally:
            os.environ.pop("PXA_CONTROL_IDLE_S", None)
            busy.close()
        self.assertFalse(r.get("error"), r)
        self.assertFalse(r["reused"])
        self.assertNotIn(r["port"], (want, want + 1))                           # busy, and the server's own
        self.assertEqual(r["url"], f"http://127.0.0.1:{r['port']}/?token={token_for(r['port'])}")   # H1: opens signed in
        self.assertGreaterEqual(len(token_for(r["port"]) or ""), 32)
        rec = C.find_running()
        self.assertTrue(rec["companion"])
        self.assertNotEqual(rec["pid"], os.getpid())
        st, _h, b = req(r["port"], "/api/fleet")
        keys = [i["key"] for i in json.loads(b)["instances"]]
        self.assertIn(f"p:{k.pid}", keys)                                       # adopted at once
        # a second launch reuses it instead of starting another
        r2 = C.ensure_control(os.path.join(TOOLS, "pxa-launch.py"), port=want)
        self.assertEqual((r2["reused"], r2["port"]), (True, r["port"]))
        # the server goes; nobody is looking; the background Control closes itself
        k.kill()
        k.wait()
        self.assertTrue(wait_until(lambda: C.running_controls() == [], 30), "background Control did not exit")
        with open(C.control_log_path()) as f:
            self.assertIn("closing", f.read())

    def test_run_control_reuse_and_port_fallback(self):
        app, port = self.up()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = L.run_control(self.ns("--no-browser"), [], L=L)
        self.assertEqual(rc, 0)
        self.assertIn(f"PXA Control: http://127.0.0.1:{port}/", buf.getvalue())
        self.assertIn("already running", buf.getvalue())
        # --gui --port N names another port: that one is not reused (a second Control is asked for)
        self.assertIsNone(C.find_running(port=port + 1))
        # nothing running: the default port when free, else the next free one; --control-port is taken as given
        shutil.rmtree(os.path.join(C.config_dir(), "run"), ignore_errors=True)
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        busy_port = busy.getsockname()[1]
        got, saved = [], (C.serve, C.DEFAULT_PORT)
        C.serve = lambda L_, **kw: got.append(kw) or 0
        C.DEFAULT_PORT = busy_port
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(L.run_control(self.ns("--no-browser"), [], L=L), 0)
                self.assertEqual(L.run_control(self.ns("--gui", "--control-port", "17971"), [], L=L), 0)
                self.assertEqual(L.run_control(self.ns("--gui", "--port", "17972"), ["--gui", "--port", "17972"], L=L), 0)
        finally:
            C.serve, C.DEFAULT_PORT = saved
            busy.close()
        self.assertGreater(got[0]["port"], busy_port)                           # 7777 taken -> the next free one
        self.assertEqual([g["port"] for g in got[1:]], [17971, 17972])
        self.assertFalse(got[0]["lan"])
        self.assertFalse(got[0].get("companion", False))

    def test_control_for_pid_reads_the_server_line(self):
        app, port = self.up()                                                   # reused: nothing is spawned
        k = self.kid("--port", "18129", "-m", FAKE_MODEL)
        os.environ["PXA_CONTROL"] = "1"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(L.control_for_pid(self.ns("--control-for-pid", str(k.pid))), 0)
                rec = C.launched_servers()[k.pid]
                self.assertEqual((rec["port"], rec["model"], rec["by"]), (18129, FAKE_MODEL, "run-server.sh"))
                self.assertIn(f"PXA Control: http://127.0.0.1:{port}/", buf.getvalue())
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):                          # the entrypoint did it already
                    self.assertEqual(L.control_for_pid(self.ns("--control-for-pid", str(k.pid))), 0)
                self.assertEqual(buf2.getvalue(), "")
        finally:
            os.environ.pop("PXA_CONTROL", None)

    def test_control_model_dirs_seed_only_an_empty_library(self):
        a = self.ns("--models-dir", MODELS)
        self.assertEqual(L.control_model_dirs(a, C, FAKE_MODEL), [MODELS])
        other = os.path.join(TMP, "other-models")
        os.makedirs(other, exist_ok=True)
        m2 = os.path.join(other, "x.gguf")
        open(m2, "wb").close()
        self.assertEqual(L.control_model_dirs(self.ns(), C, m2), [other])       # first time: where the model is
        C.save_config(dict(C.load_config(), model_dirs=[MODELS]))
        self.assertEqual(L.control_model_dirs(self.ns(), C, m2), [])            # a library exists: left alone

    def test_bare_launcher_without_a_terminal_opens_nothing(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(TOOLS, "pxa-launch.py")], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)                   # the old answer: no model
        self.assertIn("no model", r.stderr)
        self.assertNotIn("PXA Control:", r.stdout)
        self.assertEqual(C.running_controls(), [])

    def test_page_offers_stop_for_command_line_servers(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html")) as f:
            page = f.read()
        self.assertIn('x.origin === "launcher"', page)
        self.assertIn('confirm: "stop"', page)
        self.assertIn("S.info.companion", page)


def load_cfg_model_dirs():
    with open(C.config_path()) as f:
        return json.load(f).get("model_dirs", []) + [os.environ.get("PXA_MODELS_DIR", "")]


# ---------------------------------------------------------------------------------------------
# PXA v3 launcher fixes (2026-10-05): the expert-cache route, measured PXQN files,
# no spurious --accept-unmeasured on GGUF plans, the Control version text.
# ---------------------------------------------------------------------------------------------
GIB = 1 << 30


def fake_gguf(path, arch="qwen35moe", n_expert=128, size_gib=20.0, ple_gib=0.0, mtp=False,
              types=(1,), kvs=()):
    """A SPARSE synthetic GGUF: a real header and tensor directory, whose last tensor (the routed
    experts, or a dense FFN when n_expert=0) runs to a sparse end of file, so it can be 20+ GiB on
    disk without using the space. Tensor bytes come from the directory's offsets, as in a real file."""
    import struct

    def s_(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, arch)] + list(kvs)
    if n_expert:
        kv.append((f"{arch}.expert_count", 4, n_expert))
    tensors = []                                          # (name, type, offset)
    off = 0
    small = 64 << 20                                      # 64 MiB of non-expert weights
    for i, t in enumerate(types):
        tensors.append((f"blk.{i}.attn_q.weight", t, off)); off += small
    if ple_gib:
        tensors.append(("per_layer_token_embd.weight", 8, off)); off += int(ple_gib * GIB)
    if mtp:
        tensors.append(("blk.1.nextn.eh_proj.weight", 8, off)); off += small
    tensors.append(("blk.0.ffn_gate_exps.weight" if n_expert else "blk.0.ffn_down.weight", types[0], off))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
        for k, t, v in kv:
            f.write(s_(k) + struct.pack("<I", t) + (s_(v) if t == 8 else struct.pack("<I", v)))
        for nm, t, o in tensors:
            ne = 1 << 20 if not nm.startswith("per_layer") else int(ple_gib * GIB / 1.0625)
            f.write(s_(nm) + struct.pack("<I", 1) + struct.pack("<Q", ne) + struct.pack("<I", t) + struct.pack("<Q", o))
        f.write(b"\0" * (32 - f.tell() % 32))
        f.truncate(f.tell() + off + int(size_gib * GIB))     # every tensor above, then the last one
    return path


def meminfo(path, total_gib, avail_gib):
    with open(path, "w") as f:
        f.write(f"MemTotal: {int(total_gib * 1048576)} kB\nMemAvailable: {int(avail_gib * 1048576)} kB\n")
    return path


class V3Plans(unittest.TestCase):
    P100 = (0, "Tesla P100-PCIE-16GB", 60, 16384, 0, "GPU-t-0")
    V100 = (0, "Tesla V100-PCIE-16GB", 70, 16384, 0, "GPU-t-0")

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="v3plans-", dir=TMP)
        cls.moe = fake_gguf(os.path.join(cls.td, "moe.gguf"), size_gib=24.0, ple_gib=0.5, mtp=True)
        cls.dense = fake_gguf(os.path.join(cls.td, "dense.gguf"), arch="llama", n_expert=0, size_gib=24.0)
        # a MoE whose NON-expert part alone is bigger than one card: experts are a small slice
        cls.fat = fake_gguf(os.path.join(cls.td, "fat.gguf"), size_gib=2.0, types=(1,) * 300)
        cls.ram_big = meminfo(os.path.join(cls.td, "mem128"), 128, 120)
        cls.ram_small = meminfo(os.path.join(cls.td, "mem24"), 24, 22)
        cls.ram_busy = meminfo(os.path.join(cls.td, "mem64busy"), 64, 12)

    def plan(self, model, gpus, mem, extra=()):
        env = {"PXA_MEMINFO": mem, "PXA_LAUNCH_TABLES_ONLY": "1"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            a = L.build_parser().parse_args(["--gpus", ",".join(str(g[0]) for g in gpus), "--model", model,
                                             "--yes"] + list(extra))
            a.explain = True
            return L._Capture().run(L.plan_and_build, a, list(gpus))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_profile_reads_expert_bytes_from_offsets(self):
        p = L.model_profile(self.moe, "gguf")
        self.assertTrue(p["is_moe"])
        self.assertGreater(p["expert_bytes"], 23 * GIB)
        self.assertTrue(p["typemap_fp"] and len(p["typemap_fp"]) == 16)

    def test_moe_bigger_than_card_with_ram_goes_expert_cache(self):
        cap = self.plan(self.moe, [self.P100], self.ram_big)
        self.assertEqual(cap.code, 0, cap.text[-3000:])
        plan, cmd, env, cv, prof, ctx = cap.value
        self.assertEqual(plan.xcache["verdict"], "ok")
        self.assertNotIn("R-17B", cap.text)
        self.assertIn("EXPERT CACHE", cap.text)
        self.assertIn("bigger than your card", plan.notes[0])
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "999")              # every layer asked for: the engine plans the cache
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "none")      # MTP head in the file: kept unloaded
        self.assertEqual(cmd[cmd.index("--cache-ram") + 1], "0")
        self.assertEqual(cmd[cmd.index("-sm") + 1], "layer")
        self.assertIn(r"per_layer_token_embd\.weight=CPU", cmd)          # the PLE table on the CPU

    def test_expert_cache_keeps_a_users_spec_and_partial_ngl_is_not_the_route(self):
        cap = self.plan(self.moe, [self.P100], self.ram_big, ["--spec", "ngram-mod:n_max=4,n_min=2"])
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertEqual(cap.value[1][cap.value[1].index("--spec-type") + 1], "ngram-mod:n_max=4,n_min=2")
        cap = self.plan(self.moe, [self.P100], self.ram_big, ["--ngl", "20"])
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertIsNone(cap.value[0].xcache)
        self.assertNotIn("--cache-ram", cap.value[1])

    def test_expert_cache_on_two_cards_is_a_layer_split(self):
        p2 = (1, "Tesla P100-PCIE-16GB", 60, 16384, 0, "GPU-t-1")
        big = fake_gguf(os.path.join(self.td, "moe40.gguf"), size_gib=40.0)
        cap = self.plan(big, [self.P100, p2], self.ram_big)
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertEqual(cap.value[1][cap.value[1].index("-sm") + 1], "layer")
        self.assertIn("bigger than your cards", cap.text)

    def test_moe_without_enough_ram_is_refused_in_plain_words(self):
        cap = self.plan(self.moe, [self.P100], self.ram_small)
        self.assertEqual(cap.code, 2)
        self.assertIn("[R-17C]", cap.text)
        self.assertRegex(cap.text, r"needs about \d+ GB of system RAM and this machine has 24 GB")

    def test_busy_ram_warns_without_refusing(self):
        cap = self.plan(self.moe, [self.P100], self.ram_busy)
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertIn("RIGHT NOW only 12 GB of RAM is free", cap.text)

    def test_dense_bigger_than_card_is_still_refused(self):
        cap = self.plan(self.dense, [self.P100], self.ram_big)
        self.assertEqual(cap.code, 2)
        self.assertIn("[R-17B]", cap.text)
        self.assertNotIn("EXPERT CACHE", cap.text)

    def test_moe_whose_non_expert_part_does_not_fit_is_refused(self):
        # 300 x 64 MiB = 18.75 GiB of non-expert weights on a 16 GiB card
        cap = self.plan(self.fat, [self.P100], self.ram_big)
        self.assertEqual(cap.code, 2)
        self.assertIn("[R-17D]", cap.text)

    def test_xcache_plan_measured_cell_quotes_twenty_tps_only_there(self):
        prof = {"is_moe": True, "arch": "qwen4exp", "ple_bytes": 50 * GIB, "expert_bytes": 24 * GIB,
                "kv_bytes_tok": 0}
        mem = (128 * GIB, 120 * GIB)
        x = L.xcache_plan([self.P100], 77 * GIB, prof, 8192, True, mem=mem)
        self.assertEqual(x["verdict"], "ok")
        self.assertIn("expect about 20 t/s on one P100", x["lines"][0])
        x = L.xcache_plan([self.V100], 77 * GIB, prof, 8192, True, mem=mem)
        self.assertIn("not measured by us", x["lines"][0])
        self.assertIsNone(L.xcache_plan([self.P100], 77 * GIB, dict(prof, is_moe=False), 8192, True, mem=mem))
        self.assertIsNone(L.xcache_plan([self.P100], 60 * GIB, prof, 8192, True, mem=mem))   # GPU part fits

    def test_flash_next_one_p100_row_is_the_gate_line(self):
        sel = [self.P100]
        prof = {"arch": "qwen4exp", "is_moe": True, "n_expert": 512, "deltanet": True, "tier": "PXQ_UNIVERSAL",
                "xcache_route": True}
        r, st, _e, _n = L.recipe_for(sel, prof, "chat")
        self.assertEqual((r.key, st, r.b, r.ub, r.ctx, r.threads), ("1xp100-flashnext-xcache", "MEASURED", 2048, 2048, 8192, 16))
        # the 35B PXQU row was measured on a file that FITS one card: never handed to an expert-cache file
        prof35 = {"arch": "qwen35moe", "is_moe": True, "n_expert": 256, "tier": "PXQ_UNIVERSAL"}
        self.assertEqual(L.recipe_for([self.V100], prof35, "chat")[0].key, "1xv100-pxqu16")
        self.assertNotEqual(L.recipe_for([self.V100], dict(prof35, xcache_route=True), "chat")[1], "MEASURED")
        r4 = [r for r in L.RECIPES if r.key == "4xp100-flashnext"][0]
        self.assertEqual((r4.ts, r4.ctx, r4.np), ("0.95,1.01,1.01,1.03", 32768, 1))

    def test_pxqn_mixed_file_needs_no_ack_classic_universal_still_does(self):
        uni = (257, 259)                                   # PXQN3 + PXQN4 tensors: a mixed (UNIVERSAL) tier map
        pxqn = fake_gguf(os.path.join(self.td, "bal.gguf"), arch="qwen35", n_expert=0, size_gib=1.0, types=uni,
                         kvs=[("pxa.pxqn.encoder", 8, "pxqn-ladder-test"), ("pxa.pxqn.alloc", 8, "0123456789abcdef")])
        classic = fake_gguf(os.path.join(self.td, "uni.gguf"), arch="qwen35", n_expert=0, size_gib=1.0, types=(252, 255))
        for path, want_ack in ((pxqn, False), (classic, True)):
            prof = L.model_profile(path, "gguf")
            self.assertEqual(prof["tier"], "PXQ_UNIVERSAL")
            p = L.decide([self.V100], "gguf", path, None, prof, 1, "chat", set(), None, [])
            self.assertEqual(bool([x for x in p.needs_ack if x.startswith("PXQ_UNIVERSAL")]), want_ack, p.needs_ack)
        cap = self.plan(pxqn, [self.V100], self.ram_big)
        self.assertEqual(cap.code, 0, cap.text[-2000:])
        self.assertIn("no release-gate result on record", cap.text)

    def test_onecard_cells_are_measured_by_identity_not_by_name(self):
        oc = {"typemap_fp": "050b79b16e836341", "pxqn_alloc": "ea306a5d31393142", "tier": "PXQ_UNIVERSAL",
              "pxqn_encoder": "pxqn-ladder-cc44672e5a"}
        for card, word in ((self.V100, "42.5"), (self.P100, "27.8")):
            st, text = L.measured_file_cell(oc, [card])
            self.assertEqual(st, "MEASURED")
            self.assertIn("12/12", text)
            self.assertIn("125,000", text)
            self.assertIn(word, text)
        # same tier map, another encode (a fine-tune): not claimed as gated
        st, text = L.measured_file_cell(dict(oc, pxqn_alloc="353c1391120bb694"), [self.V100])
        self.assertEqual(st, "NOTE")
        self.assertIn("not checked by us", text)
        # a card set the gate did not run: said, not refused
        st, text = L.measured_file_cell(oc, [(0, "x", 61, 11264, 0, "u")])
        self.assertEqual(st, "NOTE")
        # PXQN3bal is in the table too
        self.assertEqual(L.measured_file_cell({"typemap_fp": "579c89d9674e8ff8", "pxqn_alloc": "aabe9387a002b210"},
                                              [self.P100])[0], "MEASURED")

    @unittest.skipUnless(os.path.isfile(os.environ.get("PXA_TEST_ONECARD_GGUF", "")),
                         "the real one-card file is not on this machine")
    def test_real_onecard_file_plans_without_accept_unmeasured(self):
        path = os.environ.get("PXA_TEST_ONECARD_GGUF", "")
        self.assertEqual(L.model_profile(path, "gguf")["typemap_fp"], "050b79b16e836341")
        for card in (self.V100, self.P100):
            cap = self.plan(path, [card], self.ram_big)
            self.assertEqual(cap.code, 0, cap.text[-2000:])
            self.assertIn("MEASURED file: this is our Qwen3.8-27B one-card mix", cap.text)
            self.assertNotIn("--accept-unmeasured to proceed", cap.text)

    def test_gguf_plans_need_no_ack_for_card_count_or_unmeasured_sets(self):
        prof = {"tier": "PXQN4", "arch": "qwen35", "is_moe": False, "deltanet": True, "n_expert": 0}
        quad = [(i, "Tesla P100-PCIE-16GB", 60, 16384, 0, f"u{i}") for i in range(4)]
        mixed = [(0, "Tesla P100-PCIE-16GB", 60, 16384, 0, "u0"), (1, "Tesla V100-PCIE-16GB", 70, 16384, 0, "u1")]
        three = quad[:3]
        for sel in (quad, mixed, three):
            p = L.decide(sel, "gguf", "/x/m.gguf", None, prof, 1, "chat", set(), None, [])
            self.assertEqual((p.engine, p.needs_ack), ("llama", []), (len(sel), p.needs_ack))
        # the engine-choice table still asks when it CHOSE the engine (a converted vLLM dir, both eligible)
        pv = L.decide(quad, "vllm_dir", "/x/coder35", None, dict(prof, tier="PXQ4", is_moe=True, n_expert=256), 8,
                      "serve", {60}, "pxa-sm60-dev", [])
        self.assertTrue(pv.engine_from_table)
        self.assertTrue(any("table is 2-card only" in x for x in pv.needs_ack))

    def test_control_fit_badge_says_ram_for_the_expert_cache(self):
        e = {"size": 24 * GIB, "n_expert": 128, "expert_bytes": 23 * GIB, "ple_bytes": 0, "kv_bytes_tok": 0}
        old = os.environ.get("PXA_MEMINFO")
        os.environ["PXA_MEMINFO"] = self.ram_big
        try:
            app = C.App(L, port=7777, models_dirs=[MODELS])
            self.assertEqual(app.fits(e, [self.P100])["verdict"], "ram")
            self.assertEqual(app.fits(dict(e, n_expert=0, expert_bytes=0), [self.P100])["verdict"], "no")
            os.environ["PXA_MEMINFO"] = self.ram_small
            self.assertEqual(app.fits(e, [self.P100])["verdict"], "no")
        finally:
            os.environ.pop("PXA_MEMINFO", None) if old is None else os.environ.__setitem__("PXA_MEMINFO", old)

    def test_control_version_comes_from_the_package(self):
        src = "".join(open(os.path.join(TOOLS, "pxa_control", _f), encoding="utf-8").read()
                          for _f in sorted(os.listdir(os.path.join(TOOLS, "pxa_control"))) if _f.endswith(".py"))     # the package (the Control hardening pass)
        self.assertIn("CONTROL_VERSION = package_version()", src)
        self.assertNotIn('CONTROL_VERSION = "v3"', src)
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            tok = C.version_token(f.read())
        self.assertEqual(tok, "v3.1")
        self.assertEqual(C.version_token("tag:              v3.1\ncommit: abc\n"), "v3.1")
        self.assertEqual(C.version_token("v3.1.4\n"), "v3.1.4")
        self.assertEqual(C.version_token(""), "")
        self.assertEqual(C.CONTROL_VERSION, tok)
        d = tempfile.mkdtemp(dir=TMP)
        self.assertEqual(C.version_from_dir(d), "")
        os.makedirs(os.path.join(d, "pxa-v3.1.4"))
        os.symlink("pxa-v3.1.4", os.path.join(d, "current"))
        self.assertEqual(C.version_from_dir(d), "v3.1.4")
        with open(os.path.join(d, "VERSION"), "w", encoding="utf-8") as f:
            f.write("tag: v9.9\n")
        self.assertEqual(C.version_from_dir(d), "v9.9")
        with open(os.path.join(d, "pxa-v3.1.4", "VERSION"), "w", encoding="utf-8") as f:
            f.write("tag: v8.8\n")
        self.assertEqual(C.version_from_dir(d), "v8.8")
        app = C.App(L, port=7777, models_dirs=[MODELS])
        srv, port = serve_app(app)
        try:
            info = json.loads(req(port, "/api/info")[2])
            self.assertEqual(info["control_version"], tok)
        finally:
            srv.shutdown()
            srv.server_close()


class QuantizerTag(unittest.TestCase):
    """_extra_facts exposes the pxa.quantizer.* KVs (prefix stripped) or None."""

    def _facts(self, kv):
        d = tempfile.mkdtemp(dir=TMP)
        p = os.path.join(d, "m.gguf")
        open(p, "wb").write(b"x")
        fake = type("A", (), {})()
        fake._prof_cache = {}
        fake.L = type("LL", (), {"gguf_header": staticmethod(lambda path: {"kv": kv, "tensors": []}),
                                  "PXQ_GGML_TYPE": {}, "NON_PXQ_GGML_TYPE": {}})
        return C.App._extra_facts(fake, p)

    def test_tagged_and_untagged(self):
        q = self._facts({"general.name": "n", "pxa.quantizer.edition": "pro", "pxa.quantizer.licensee": "T",
                         "pxa.quantizer.license_expires": 1791244800})["quantizer"]
        self.assertEqual(q, {"edition": "pro", "licensee": "T", "license_expires": 1791244800})
        self.assertIsNone(self._facts({"general.name": "n"})["quantizer"])


class MtpHeadFacts(unittest.TestCase):
    """_extra_facts reports the draft head (public pxa #7): nextn/mtp tensor count and the KV, None when the header is unreadable."""

    def _facts(self, hdr):
        d = tempfile.mkdtemp(dir=TMP)
        p = os.path.join(d, "m.gguf")
        open(p, "wb").write(b"x")
        fake = type("A", (), {})()
        fake._prof_cache = {}
        fake.L = type("LL", (), {"gguf_header": staticmethod(lambda path: hdr),
                                  "PXQ_GGML_TYPE": {}, "NON_PXQ_GGML_TYPE": {}})
        return C.App._extra_facts(fake, p)

    def test_head_counts(self):
        kv = {"general.architecture": "qwen35", "qwen35.nextn_predict_layers": 1}
        with_head = {"ok": True, "kv": kv, "tensors": [("token_embd.weight", 0, 4), ("blk.64.nextn.eh_proj.weight", 0, 4)]}
        f = self._facts(with_head)
        self.assertEqual((f["mtp_tensors"], f["mtp_kv"]), (1, 1))
        liar = {"ok": True, "kv": kv, "tensors": [("token_embd.weight", 0, 4)]}
        f = self._facts(liar)
        self.assertEqual((f["mtp_tensors"], f["mtp_kv"]), (0, 1))
        nohead = {"ok": True, "kv": {"general.architecture": "qwen35"}, "tensors": [("token_embd.weight", 0, 4)]}
        f = self._facts(nohead)
        self.assertEqual((f["mtp_tensors"], f["mtp_kv"]), (0, None))
        self.assertIsNone(self._facts({"ok": False, "kv": {}, "tensors": []})["mtp_tensors"])   # unreadable: unknown, not "no head"

    def test_ui_switches_mtp_off_for_a_headless_file(self):
        html = open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8").read()
        self.assertIn('m.mtp_head === false', html)
        self.assertIn('mtp: $("#l-mtp").checked && !$("#l-mtp").disabled', html)


def urllib_quote(p):
    import urllib.parse
    return urllib.parse.quote(p, safe="")


def stat_mode(p):
    import stat as _st
    return _st.S_IMODE(os.stat(p).st_mode)


class SymlinkedModels(unittest.TestCase):
    """A model offloaded to another disk and linked back into a model folder can be planned and
    started; a link to anything that is not a regular GGUF, or a path that walks out, cannot."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pxa-models-root-")
        self.outside = tempfile.mkdtemp(prefix="pxa-models-outside-")
        def w(name, data):
            p = os.path.join(self.outside, name)
            with open(p, "wb") as f:
                f.write(data)
            return p
        self.real = w("real.gguf", b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)
        notg = w("secret.gguf", b"not a gguf at all")
        plain = w("passwd", b"root:x:0:0::/root:/bin/sh\n")
        os.symlink(self.real, os.path.join(self.root, "linked.gguf"))
        os.symlink(notg, os.path.join(self.root, "fake.gguf"))
        os.symlink(plain, os.path.join(self.root, "plain.gguf"))
        os.symlink(os.path.join(self.outside, "gone.gguf"), os.path.join(self.root, "dangling.gguf"))
        os.symlink(self.outside, os.path.join(self.root, "offload"))        # a linked folder
        os.symlink(self.outside, os.path.join(self.root, "offload.gguf"))   # a directory named .gguf

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.outside, ignore_errors=True)

    def check(self, model):
        return C.validate_launch({"gpus": [0], "model": model}, {0, 1}, CATALOG, [self.root], resolve_port=False)

    def test_link_to_a_gguf_outside_the_folders_is_allowed(self):
        for m in (os.path.join(self.root, "linked.gguf"), os.path.join(self.root, "offload", "real.gguf")):
            self.assertEqual(self.check(m)["model"], m)
            self.assertTrue(C.model_path_ok(m, [self.root]))

    def test_everything_else_outside_is_refused(self):
        bad = [os.path.join(self.root, "fake.gguf"),             # a link to a file without the GGUF magic
               os.path.join(self.root, "plain.gguf"),            # a link to a text file
               os.path.join(self.root, "dangling.gguf"),         # a link to nothing
               os.path.join(self.root, "offload.gguf"),          # a link to a directory
               self.real,                                        # the target itself, named directly
               os.path.join(self.root, "..", os.path.basename(self.outside), "real.gguf")]   # walking out
        for m in bad:
            with self.assertRaises(C.Invalid, msg=m):
                self.check(m)
        self.assertFalse(C.model_path_ok(self.real, [self.root]))

    def test_api_thinking_follows_the_same_rule(self):
        app = C.App(L, port=7777)
        app.model_roots = lambda: [self.root]
        prof, _st, src, model = app.thinking_target(urllib.parse.parse_qs("model=" + urllib.parse.quote(os.path.join(self.root, "linked.gguf"))))
        self.assertEqual(model, os.path.join(self.root, "linked.gguf"))
        with self.assertRaises(C.Invalid):
            app.thinking_target(urllib.parse.parse_qs("model=" + urllib.parse.quote(os.path.join(self.root, "fake.gguf"))))


class LaunchCtxAndLog(unittest.TestCase):
    """The plan header's context is the -c on the command, and the log pane loads the backlog."""

    def _args(self, ctx):
        return type("A", (), {
            "ctx": ctx, "sm": "layer", "model": FAKE_MODEL, "host": "127.0.0.1", "port": 9,
            "ngl": 99, "np": 1, "ctk": "f16", "ctv": "f16", "threads": 2, "emit_threads": False,
            "workload": "chat", "ub": 0, "b": 0, "ts": "", "spec": None, "draft_model": None,
            "hot_model": [], "no_mmap": False,
        })()

    def _plan(self, picked):
        plan = L.Plan()
        plan.fa = "on"
        plan.engine_ac = {"picks": {"c": {"value": str(picked), "status": "measured", "why": "test cell"}}}
        return plan

    def test_auto_ctx_returned_is_the_command_c(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd, env, ctx = L.build_llama_cmd(self._plan(32768), self._args(0),
                                              [(0, "Tesla P100", 60, 16384, 0, "u")], {}, 4096, ["n/a"], None,
                                              explain=True)
        self.assertEqual(cmd[cmd.index("-c") + 1], "32768")
        self.assertEqual(ctx, 32768)
        self.assertIn("was -c 4096", buf.getvalue())
        with open(os.path.join(TOOLS, "pxa-launch.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("cmd, env, ctx = build_llama_cmd", src)
        self.assertGreater(src.find('print(f"  serve:'), src.find("cmd, env, ctx = build_llama_cmd"))

    def test_a_hand_set_ctx_is_not_replaced(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd, env, ctx = L.build_llama_cmd(self._plan(32768), self._args(16384),
                                              [(0, "Tesla P100", 60, 16384, 0, "u")], {}, 16384, ["n/a"], None,
                                              explain=True)
        self.assertEqual((cmd[cmd.index("-c") + 1], ctx), ("16384", 16384))
        self.assertNotIn("from the engine's registry", buf.getvalue())

    def test_plan_dict_says_auto_only_when_the_request_left_it(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        plan = L.Plan()
        plan.engine = "llama"

        class Cap(object):
            code = 0
            text = "plan"
            thinking = None
        cap = Cap()
        cap.args = type("A", (), {"sm": "layer", "np": 1, "workload": "chat"})()
        cap.value = (plan, ["llama-server", "-m", FAKE_MODEL, "-c", "32768"], {}, "0", {}, 32768)
        auto = app.validate({"gpus": [0], "model": FAKE_MODEL, "ctx": 0}, resolve_port=False)
        d = app._plan_dict(auto, cap, "main", {"errors": [], "warnings": []})
        self.assertEqual(d["ctx"], 32768)
        self.assertTrue(d["ctx_auto"])
        self.assertIn("-c 32768", d["command"])
        cap.value = (plan, ["llama-server", "-m", FAKE_MODEL, "-c", "16384"], {}, "0", {}, 16384)
        hand = app.validate({"gpus": [0], "model": FAKE_MODEL, "ctx": 16384}, resolve_port=False)
        d2 = app._plan_dict(hand, cap, "main", {"errors": [], "warnings": []})
        self.assertEqual((d2["ctx"], d2["ctx_auto"]), (16384, False))

    def test_page_loads_the_log_before_the_serving_line(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            page = f.read()
        self.assertIn('ctx ${d.ctx_auto ? "auto " : ""}${d.ctx}', page)
        poll = page[page.find("async function pollStatus"):page.find("function logRender")]
        self.assertLess(poll.find("await loadLogBacklog()"), poll.find('badge("ok", "serving")'))
        opened = page[page.find("function openLog"):page.find("async function pollLog")]
        self.assertLess(opened.find("loadLogBacklog()"), opened.find("new EventSource"))
        reset = page[page.find("function resetLog"):page.find("function loadLogBacklog")]
        self.assertIn("clearTimeout(logAppend._t)", reset)
        self.assertIn("pre.dataset.live", page)

    def test_log_lines_written_before_attach_are_readable(self):
        seat = C.Seat(L)
        seat._append("$ llama-server -c 32768")
        seat._append("llama_model_loader: mock")
        items = seat.lines_since(0)
        self.assertEqual([x[1] for x in items][0], "$ llama-server -c 32768")
        self.assertEqual(items[-1][0], seat.seq)
        self.assertEqual(seat.lines_since(items[0][0])[0][1], "llama_model_loader: mock")


# ---------------------------------------------------------------------------------------------
# The licensed-library routes (/api/lib/update) on the SAME stub licence server the updater's own
# test uses (tests/test-pxa-lib-update.py).  That test covers tools/pxa_lib_update.py as a module;
# this covers the four HTTP handlers and the settings validation that exists only in Control, driven
# end to end: route -> LibUpdate -> tools/pxa_lib_update.py -> the fake licence server on loopback.
# ---------------------------------------------------------------------------------------------
_LUT_PATH = os.path.join(ROOT, "tests", "test-pxa-lib-update.py")
_LUT = None
if os.path.isfile(_LUT_PATH):
    _lut_spec = importlib.util.spec_from_file_location("pxa_lib_update_under_test", _LUT_PATH)
    _LUT = importlib.util.module_from_spec(_lut_spec)
    _lut_spec.loader.exec_module(_LUT)


@unittest.skipUnless(_LUT is not None, "tests/test-pxa-lib-update.py not present")
class LibUpdateRoutes(unittest.TestCase):
    """The library-update page's seams: status, a forced check, apply, rollback, and settings."""

    def setUp(self):
        self.ltmp = tempfile.mkdtemp(prefix="pxa-lib-routes-")
        self.addCleanup(shutil.rmtree, self.ltmp, ignore_errors=True)
        self.saved = {k: os.environ.get(k) for k in
                      ("PXA_PACKAGE_PUBKEY", "PXA_CONTROL_CONFIG_DIR", "PXA_LICENCE_KEY",
                       "PXA_LICENCE_URL", "PXA_ENGINE_DIR", "HOME")}
        self.addCleanup(self._restore_env)
        for k in ("PXA_LICENCE_URL", "PXA_LICENCE_KEY", "PXA_ENGINE_DIR"):
            os.environ.pop(k, None)
        os.environ["PXA_PACKAGE_PUBKEY"] = _LUT.F.PUB_HEX
        os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(self.ltmp, "cfg")
        os.environ["HOME"] = os.path.join(self.ltmp, "home")
        os.makedirs(os.environ["PXA_CONTROL_CONFIG_DIR"], exist_ok=True)
        os.makedirs(os.environ["HOME"], exist_ok=True)
        self.ins, self.engine = self._make_install()
        self.lic = _LUT.FakeLibLicence({_LUT.REL1["lib_id"]: _LUT.REL1},
                                       {"stable": _LUT.REL1["lib_id"]}).start()
        self.addCleanup(self.lic.stop)
        os.environ["PXA_LICENCE_URL"] = self.lic.url
        os.environ["PXA_LICENCE_KEY"] = _LUT.KEY
        self.app = C.App(L, port=0, models_dirs=[MODELS])
        # Pin the install under test to THIS temp tree.  On a real box rig_static()'s engine_dir is a
        # live install, and a route test must never apply or roll back a library on someone's engine.
        self.app.rig_static = lambda force=False: {"engine_dir": self.ins}
        self.srv, self.port = serve_app(self.app)
        self.addCleanup(self._stop)

    def _restore_env(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _stop(self):
        try:
            self.srv.shutdown()
            self.srv.server_close()
        finally:
            self.app.lib.close()

    def _make_install(self):
        """The engine's install tree exactly as the tarball lays it out: `current` -> pxa-v3, and the two
        copies of libggml-pxqn.so hard-linked to each other."""
        ins = os.path.join(self.ltmp, "install")
        v = os.path.join(ins, "pxa-v3")
        for sub in ("lib", "lib-compat", "bin"):
            os.makedirs(os.path.join(v, sub))
        with open(os.path.join(v, "VERSION"), "w") as f:
            f.write("v3.1\n")
        engine = b"\x7fELF the library the engine release shipped\n" * 40
        with open(os.path.join(v, _LUT.COPIES[0]), "wb") as f:
            f.write(engine)
        os.link(os.path.join(v, _LUT.COPIES[0]), os.path.join(v, _LUT.COPIES[1]))
        os.symlink("pxa-v3", os.path.join(ins, "current"))
        return ins, engine

    def _content(self, name):
        with open(os.path.join(self.ins, "current", name), "rb") as f:
            return f.read()

    def get(self, path):
        code, _, body = req(self.port, path)
        return code, json.loads(body)

    def post(self, path, body):
        code, _, raw = req(self.port, path, "POST", body)
        return code, json.loads(raw)

    def test_status_is_pristine_then_a_forced_check_reports_the_release(self):
        # A GET touches the page, which starts the background poller; it may already have refreshed
        # the cache, so `update` on the first read is not pinned. What must hold: the install is
        # pristine and the install's settings are the defaults.
        code, d = self.get("/api/lib/update")
        self.assertEqual(code, 200, d)
        self.assertTrue(d["available"], d)
        self.assertEqual(d["installed"]["version"], "", d)
        self.assertEqual(d["installed"]["lib_id"], None, d)
        self.assertEqual(d["channel"], "stable", d)
        self.assertFalse(d["auto"], d)
        code, d = self.get("/api/lib/update?check=1")
        self.assertEqual(code, 200, d)
        self.assertEqual(d["latest"], _LUT.REL1["version"], d)
        self.assertEqual(d["lib_id"], _LUT.REL1["lib_id"], d)
        self.assertTrue(d["update"], d)
        self.assertIsNone(d["error"], d)

    def test_apply_installs_the_release_then_rollback_restores_the_shipped_library(self):
        code, d = self.get("/api/lib/update?check=1")
        self.assertTrue(d["update"], d)
        code, a = self.post("/api/lib/update/apply", {})
        self.assertEqual(code, 200, a)
        self.assertTrue(a["applied"], a)
        self.assertEqual(a["version"], _LUT.REL1["version"], a)
        code, d = self.get("/api/lib/update")
        self.assertEqual(d["installed"]["version"], _LUT.REL1["version"], d)
        self.assertNotEqual(self._content(_LUT.COPIES[0]), self.engine, "the shipped bytes were swapped")
        with open(os.path.join(self.ins, "lib", "lib-state.json")) as f:
            self.assertNotIn(_LUT.KEY, f.read(), "the licence key is never written to the state")
        code, r = self.post("/api/lib/update/rollback", {})
        self.assertEqual(code, 200, r)
        self.assertTrue(r["applied"], r)
        self.assertEqual(self._content(_LUT.COPIES[0]), self.engine, "rollback did not restore the shipped library")
        self.assertEqual(_LUT.LU.installed_version(self.ins), "", "the install is not back to shipped")

    def test_settings_reject_a_bad_channel_and_a_non_boolean_auto(self):
        code, d = self.post("/api/lib/update/settings", {"channel": "gamma"})
        self.assertEqual(code, 400, d)
        self.assertIn("channel must be one of", d["error"], d)
        code, d = self.post("/api/lib/update/settings", {"auto": "yes"})
        self.assertEqual(code, 400, d)
        self.assertIn("auto must be true or false", d["error"], d)
        code, d = self.post("/api/lib/update/settings", {"channel": "beta", "auto": True})
        self.assertEqual(code, 200, d)
        self.assertEqual((d["channel"], d["auto"]), ("beta", True), d)
        code, d = self.get("/api/lib/update")
        self.assertEqual((d["channel"], d["auto"]), ("beta", True), "settings did not persist")

    def test_apply_refuses_with_a_sentence_while_a_server_is_running(self):
        """The page is promised a 4xx and a plain sentence.  A refusal that reaches the page as a 500
        is the one shape a browser cannot render, so the status is asserted, not only the text."""
        saved = _LUT.LU.server_running
        _LUT.LU.server_running = lambda d=None: True
        self.addCleanup(lambda: setattr(_LUT.LU, "server_running", saved))
        code, d = self.post("/api/lib/update/apply", {})
        self.assertEqual(code, 400, d)
        self.assertIn("running", d["error"].lower(), d)

    def test_a_slow_background_check_never_overwrites_a_newer_forced_answer(self):
        """Flaky-under-load root cause (10-19): POST /settings touches the page, which starts the background
        poller BEFORE the new channel is written, so the poller asks about `stable`; a forced ?check=1 on `beta`
        then gets not_valued, and the slower poller used to store its older `stable` answer on top of it (the
        page and the test then saw error=None). The answer of the check that STARTED last must stand."""
        import threading
        real = _LUT.LU.check
        gate, stale_done = threading.Event(), threading.Event()

        def slow_check(d, channel=None, **kw):
            if channel == "beta":
                return real(d, channel=channel, **kw)
            gate.wait(5)                # the poller's stable ask finishes only after the forced beta ask
            try:
                return real(d, channel=channel, **kw)
            finally:
                stale_done.set()
        _LUT.LU.check = slow_check
        self.addCleanup(lambda: setattr(_LUT.LU, "check", real))
        self.lic.valued = False
        self.app.lib.touch()            # the poller starts while the channel is still stable
        time.sleep(0.2)
        self.app.lib.cfg_set(channel="beta")
        code, d = self.get("/api/lib/update?check=1")
        self.assertEqual((d["error"] or {}).get("code"), "not_valued", d)
        gate.set()
        self.assertTrue(stale_done.wait(10), "the poller never asked")
        time.sleep(0.3)                 # its answer is stored right after the ask returns
        code, d = self.get("/api/lib/update")
        self.assertEqual((d["error"] or {}).get("code"), "not_valued", d)
        self.assertFalse(d["beta"], d)

    def test_apply_waits_for_the_background_check_instead_of_refusing(self):
        """Second load-only flake (10-19, ctest -j 4): the GET that opens the page starts the poller; when its check
        was still in flight, Apply was refused with 'A library check is already running'. A check only reads the
        server, so the click now waits for it; an apply/rollback in flight is still refused."""
        import threading
        real = _LUT.LU.check
        gate, entered = threading.Event(), threading.Event()

        def slow_check(d, channel=None, **kw):
            if threading.current_thread().name == "pxa-lib":
                entered.set()
                gate.wait(5)
            return real(d, channel=channel, **kw)
        _LUT.LU.check = slow_check
        self.addCleanup(lambda: setattr(_LUT.LU, "check", real))
        self.app.lib.touch()
        self.assertTrue(entered.wait(5), "the poller never asked")
        threading.Timer(0.5, gate.set).start()
        code, a = self.post("/api/lib/update/apply", {})
        self.assertEqual(code, 200, a)
        self.assertTrue(a["applied"], a)
        # an apply in flight is still refused
        self.app.lib._busy = "apply"
        self.addCleanup(lambda: setattr(self.app.lib, "_busy", ""))
        code, r = self.post("/api/lib/update/rollback", {})
        self.assertEqual(code, 400, r)
        self.assertIn("already running", r["error"], r)

    def test_a_key_without_the_valued_role_is_refused_on_beta(self):
        self.lic.valued = False
        code, d = self.post("/api/lib/update/settings", {"channel": "beta"})
        self.assertEqual(code, 200, d)
        code, d = self.get("/api/lib/update?check=1")
        self.assertEqual(code, 200, d)
        self.assertEqual((d["error"] or {}).get("code"), "not_valued", d)
        self.assertFalse(d["beta"], d)


# ---------------------------------------------------------------------------------------------
# the models PXA publishes on Hugging Face, and the one-click download (v3.1, tools/pxa_hf.py)
# ---------------------------------------------------------------------------------------------
# Nothing here touches the network: a fake origin stands in for huggingface.co, so the listing, the
# ranged resume, the sha256 check and the 401 on a supporter file are all exercised for real.

HF_SMALL = b"GGUF" + b"\x11" * 508                            # one file, one part
HF_BIG = b"GGUF" + b"\x22" * 40000                            # a split model's part
HF_MID = b"GGUF" + b"\x33" * (1 << 20)                        # more than one read, so a cancel can land mid-file
HF_CSV = b"layer,expert\n0,3\n"
HF_SHA = {n: hashlib.sha256(d).hexdigest() for n, d in
          (("small.gguf", HF_SMALL), ("big-00001-of-00002.gguf", HF_BIG), ("big-00002-of-00002.gguf", HF_BIG),
           ("one-1mib.gguf", HF_MID))}
HF_REPOS = {
    "poisonxa/Pub-Model": {"small.gguf": HF_SMALL, "small.gguf.expert-counts.csv": HF_CSV},
    "poisonxa/Pub-Split": {"big-00001-of-00002.gguf": HF_BIG, "big-00002-of-00002.gguf": HF_BIG},
    "poisonxa/Supporter-Model": {"secret.gguf": HF_SMALL},
}
HF_KEY = "hf_a_key_that_can_read_the_supporter_repo"
HF_SUPPORTER = "poisonxa/Supporter-Model"


class FakeHF(http.server.BaseHTTPRequestHandler):
    """The three huggingface.co endpoints this feature uses, and a 401 on a gated file without the key."""
    delay = 0.0                    # seconds before each piece: what makes a cancel land mid-file
    seen_ranges = []               # every Range header the client sent, so a resume can be proven

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/models":
            return self._json([{"id": r, "private": False, "downloads": 3,
                                "lastModified": "2026-09-%02dT00:00:00Z" % (20 + i)}
                               for i, r in enumerate(HF_REPOS)])
        if u.path.startswith("/api/models/") and u.path.endswith("/tree/main"):
            files = HF_REPOS.get(u.path[len("/api/models/"):-len("/tree/main")])
            if files is None:
                return self._send(404, b"{}")
            return self._json([{"type": "file", "path": n, "size": len(d),
                                "lfs": {"oid": HF_SHA.get(n) or hashlib.sha256(d).hexdigest(),
                                        "size": len(d)}} for n, d in files.items()])
        m = re.match(r"^/([^/]+/[^/]+)/resolve/main/(.+)$", u.path)
        if m:
            repo, name = m.group(1), urllib.parse.unquote(m.group(2))
            if repo == HF_SUPPORTER and (self.headers.get("Authorization") or "") != "Bearer " + HF_KEY:
                return self._send(401, b'{"error":"unauthorized"}')
            data = (HF_REPOS.get(repo) or {}).get(name)
            if data is None:
                return self._send(404, b"{}")
            rng = self.headers.get("Range") or ""
            start = int(rng[6:].split("-")[0]) if rng.startswith("bytes=") else 0
            FakeHF.seen_ranges.append((name, rng))
            self.send_response(206 if start else 200)
            body = data[start:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for off in range(0, len(body), 65536):          # in pieces, so a cancel can catch it part-way
                if FakeHF.delay:
                    time.sleep(FakeHF.delay)
                self.wfile.write(body[off:off + 65536])
                self.wfile.flush()                          # wfile is buffered: without this nothing arrives
            return
        self._send(404, b"{}")

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode())

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CountsBesideTheModel(unittest.TestCase):
    """The download saves the expert map under the name the engine reads: <the -m file>.expert-counts.csv.
    A map named after the unsplit model, or the only map of a one-model repository named after another name
    of the same file (a release dir built from a symlink), used to land under a name the engine never looked
    for, and the expert cache fell back to whole-layer streaming (2026-10-12)."""

    @staticmethod
    def _f(name):
        if name.endswith(".expert-counts.csv"):
            return {"path": name, "name": name, "kind": "counts", "model": name[:-len(".expert-counts.csv")]}
        return {"path": name, "name": name, "kind": "model"}

    def _plan(self, names, want):
        return [(f["name"], f.get("save_as")) for f in C.HF.download_plan([self._f(n) for n in names], want)]

    def test_same_name_is_unchanged(self):
        self.assertEqual(self._plan(["a.gguf", "a.gguf.expert-counts.csv"], "a.gguf"),
                         [("a.gguf", None), ("a.gguf.expert-counts.csv", None)])

    def test_split_model_takes_the_unsplit_map_under_the_first_shard_name(self):
        got = self._plan(["b-00001-of-00002.gguf", "b-00002-of-00002.gguf", "b.gguf.expert-counts.csv"],
                         "b-00001-of-00002.gguf")
        self.assertEqual(got[-1], ("b.gguf.expert-counts.csv", "b-00001-of-00002.gguf.expert-counts.csv"))

    def test_a_one_model_repo_pairs_its_only_map_whatever_it_is_named(self):
        got = self._plan(["FN-32GB.gguf", "FN-32GB.relaid.gguf.expert-counts.csv"], "FN-32GB.gguf")
        self.assertEqual(got[-1], ("FN-32GB.relaid.gguf.expert-counts.csv", "FN-32GB.gguf.expert-counts.csv"))

    def test_another_models_map_is_never_taken(self):
        got = self._plan(["x.gguf", "y.gguf", "y.gguf.expert-counts.csv"], "x.gguf")
        self.assertEqual(got, [("x.gguf", None)])


class FakeHFSrv(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        pass                          # a client that closes mid-body (a cancel) is not an error here


class ModelDownloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin = FakeHFSrv(("127.0.0.1", 0), FakeHF)
        threading.Thread(target=cls.origin.serve_forever, daemon=True).start()
        cls.saved = C.HF.HF
        C.HF.HF = "http://127.0.0.1:%d" % cls.origin.server_address[1]
        cls.dl_dir = os.path.join(TMP, "hfd")
        os.makedirs(cls.dl_dir, exist_ok=True)
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        cls.app = C.App(L, port=7777, models_dirs=[cls.dl_dir])
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.origin.shutdown()
        cls.origin.server_close()
        C.HF.HF = cls.saved

    def setUp(self):
        os.environ.pop("PXA_CONTROL_HF_TOKEN", None)
        for f in os.listdir(self.dl_dir):
            os.unlink(os.path.join(self.dl_dir, f))
        c = dict(C.load_config(), model_dirs=[self.dl_dir], hf_org="poisonxa")
        c.pop("hf_token", None)                # each test starts with no key, whatever the last one left
        C.save_config(c)
        self.app.hf.token_seen = None
        self.app.hf_dl.job = None
        self.app._hf_last = None
        FakeHF.delay, FakeHF.seen_ranges = 0.0, []

    def get(self, path):
        code, _, body = req(self.port, path)
        return code, json.loads(body)

    def post(self, path, body):
        code, _, b = req(self.port, path, "POST", body)
        return code, json.loads(b)

    def list_now(self):
        return self.get("/api/models/available?force=1")[1]

    def slurp(self, name):
        with open(os.path.join(self.dl_dir, name), "rb") as f:
            return f.read()

    def wait_end(self, secs=30):
        end = time.time() + secs
        while time.time() < end:
            d = self.get("/api/models/download")[1]["download"]
            if d and d["state"] in ("done", "failed", "cancelled"):
                return d
            time.sleep(0.02)
        self.fail("the download did not settle in %ss" % secs)

    def test_the_route_set_is_registered_and_the_methods_are_enforced(self):
        for m, p in (("GET", "/api/models/available"), ("POST", "/api/models/download"),
                     ("GET", "/api/models/download"), ("POST", "/api/models/download/cancel"),
                     ("POST", "/api/models/hf-token")):
            self.assertIn((m, p), C.ROUTES)
        self.assertEqual(req(self.port, "/api/models/available", "POST", {})[0], 405)
        self.assertEqual(req(self.port, "/api/models/hf-token", "GET")[0], 405)

    # ---- the remembered list (FR#3): a visit reads it with no network, a page reload / restart keeps it ----
    def _count_network(self):
        calls = []
        real = C.HF.list_repos

        def counting(*a, **k):
            calls.append(a)
            return real(*a, **k)
        C.HF.list_repos = counting
        self.addCleanup(setattr, C.HF, "list_repos", real)
        return calls

    def _fresh_app(self):
        return C.App(L, port=7777, models_dirs=[self.dl_dir])

    def test_a_torn_remembered_list_reads_as_no_list_not_a_500(self):
        # review 10-18 night: _hf_disk_load checked that repos is a list, not that its rows are repo dicts
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        c = dict(C.load_config(), model_dirs=[self.dl_dir], hf_org="poisonxa")
        C.save_config(c)
        with open(os.path.join(C.config_dir(), "hf-catalog.json"), "w") as f:
            json.dump({"org": "poisonxa", "ts": time.time(), "repos": [1, "x", {"files": []}], "models": []}, f)
        calls = self._count_network()
        d = self._fresh_app().hf_available(cached_only=True)
        self.assertEqual((d["repos"], d["ts"], d["cached"], d["stale"]), ([], None, True, True), d)
        self.assertEqual(calls, [])

    def test_cached_only_with_nothing_remembered_is_empty_and_makes_no_network_call(self):
        shutil.rmtree(C.config_dir(), ignore_errors=True)
        c = dict(C.load_config(), model_dirs=[self.dl_dir], hf_org="poisonxa")
        C.save_config(c)
        calls = self._count_network()
        d = self._fresh_app().hf_available(cached_only=True)
        self.assertEqual((d["repos"], d["ts"], d["cached"], d["stale"]), ([], None, True, True), d)
        self.assertEqual(calls, [], "a cached-only read must not touch the network")

    def test_the_list_survives_a_restart_and_is_read_without_the_network(self):
        first = self.list_now()                                   # a real (fake-origin) reading, written to disk
        self.assertTrue(os.path.isfile(os.path.join(C.config_dir(), "hf-catalog.json")))
        self.assertEqual(oct(os.stat(os.path.join(C.config_dir(), "hf-catalog.json")).st_mode & 0o777), "0o600")
        calls = self._count_network()
        d = self._fresh_app().hf_available(cached_only=True)      # a new App = a restarted Control: memory is empty
        self.assertEqual(calls, [], "cached-only must not touch the network, memory or disk")
        self.assertEqual([r["repo"] for r in d["repos"]], [r["repo"] for r in first["repos"]])
        self.assertTrue(d["cached"] and d["ts"] and d["age_s"] < 60 and not d["stale"], d)
        self.assertEqual(d["ts"], first["ts"])

    def test_cached_read_marks_what_is_in_the_folders_now(self):
        self.list_now()
        with open(os.path.join(self.dl_dir, "small.gguf"), "wb") as f:
            f.write(HF_SMALL)
        d = self._fresh_app().hf_available(cached_only=True)
        by = {f["name"]: f for r in d["repos"] for f in r["files"]}
        self.assertTrue(by["small.gguf"]["have"], by)

    def test_staleness_rule_is_24_hours_and_a_torn_or_foreign_file_is_no_list(self):
        self.list_now()
        path = os.path.join(C.config_dir(), "hf-catalog.json")
        with open(path) as f:
            disk = json.load(f)
        for age, stale in ((3600, False), (23 * 3600, False), (25 * 3600, True), (9 * 86400, True)):
            disk["ts"] = time.time() - age
            with open(path, "w") as f:
                json.dump(disk, f)
            d = self._fresh_app().hf_available(cached_only=True)
            self.assertEqual(d["stale"], stale, (age, d["age_s"]))
            self.assertTrue(d["repos"], "an old list is still shown: it is only marked stale")
        with open(path, "w") as f:
            f.write("{not json")
        self.assertEqual(self._fresh_app().hf_available(cached_only=True)["ts"], None)
        disk["org"] = "someone-else"
        with open(path, "w") as f:
            json.dump(disk, f)
        self.assertEqual(self._fresh_app().hf_available(cached_only=True)["repos"], [], "another org's list is not ours")

    def test_a_plain_read_inside_the_ttl_and_the_route_flag(self):
        self.list_now()
        calls = self._count_network()
        self.assertEqual(self.get("/api/models/available")[1]["cached"], False)
        self.assertEqual(calls, [], "inside LIST_TTL the plain read is served from memory")
        code, d = self.get("/api/models/available?cached=1")
        self.assertEqual((code, d["cached"], bool(d["repos"])), (200, True, True))
        self.get("/api/models/available?force=1")
        self.assertEqual(len(calls), 1, "force asks the origin again")

    def test_the_listing_names_every_repository_and_flags_the_supporter_one(self):
        d = self.list_now()
        self.assertFalse(d["token_set"])
        self.assertEqual([r["repo"] for r in d["repos"]],
                         ["poisonxa/Supporter-Model", "poisonxa/Pub-Split", "poisonxa/Pub-Model"], d)
        by = {r["repo"]: r for r in d["repos"]}
        self.assertTrue(by[HF_SUPPORTER]["supporter"], "a repository only a key can read must be flagged")
        self.assertFalse(by["poisonxa/Pub-Model"]["supporter"], by["poisonxa/Pub-Model"])
        self.assertEqual([(f["name"], f["size_h"], f["counts"]) for f in by["poisonxa/Pub-Model"]["files"]],
                         [("small.gguf", "512 B", True)], by["poisonxa/Pub-Model"])
        self.assertFalse(any(f["have"] for r in d["repos"] for f in r["files"]), "nothing is here yet")

    def test_the_key_is_kept_privately_never_echoed_and_clears_the_padlock_only_when_it_works(self):
        code, d = self.post("/api/models/hf-token", {"token": HF_KEY})
        self.assertEqual((code, d), (200, {"token_set": True}), d)
        self.assertNotIn(HF_KEY, json.dumps(d), "the key must never come back to the page")
        self.assertEqual(os.stat(C.config_path()).st_mode & 0o777, 0o600, "control.json holds a secret now")
        with open(C.config_path()) as f:
            self.assertIn(HF_KEY, f.read(), "it is kept on this machine")
        d = self.list_now()
        self.assertTrue(d["token_set"])
        by = {r["repo"]: r for r in d["repos"]}
        self.assertFalse(by[HF_SUPPORTER]["supporter"], "a key that reads the repository clears the padlock")
        code, d = self.post("/api/models/download", {"repo": HF_SUPPORTER, "file": "secret.gguf"})
        self.assertEqual(code, 200, d)
        self.assertEqual(self.wait_end()["state"], "done", d)
        self.assertEqual(self.slurp("secret.gguf"), HF_SMALL)

    def test_a_key_without_access_leaves_the_padlock_on(self):
        self.post("/api/models/hf-token", {"token": "hf_a_key_that_cannot_read_it"})
        by = {r["repo"]: r for r in self.list_now()["repos"]}
        self.assertTrue(by[HF_SUPPORTER]["supporter"], "a useless key must not unlock anything")
        code, d = self.post("/api/models/download", {"repo": HF_SUPPORTER, "file": "secret.gguf"})
        self.assertEqual((code, d["code"]), (403, "needs_token"), d)
        self.assertFalse(os.path.exists(os.path.join(self.dl_dir, "secret.gguf")))

    def test_a_short_or_absurd_key_is_refused_and_forgetting_removes_it(self):
        for bad in ("abc", "hf_" + "x" * 600, "hf_key\nwith a newline"):
            code, d = self.post("/api/models/hf-token", {"token": bad})
            self.assertEqual((code, d["code"]), (400, "invalid"), d)
            self.assertNotIn("hf_token", C.load_config(), "a refused key must not be stored")
        self.post("/api/models/hf-token", {"token": HF_KEY})
        self.assertEqual(self.post("/api/models/hf-token", {"token": ""})[1], {"token_set": False})
        self.assertNotIn("hf_token", C.load_config(), "forgetting must remove it")

    def test_a_download_lands_the_file_and_the_counts_csv_beside_it(self):
        code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Model", "file": "small.gguf"})
        self.assertEqual(code, 200, d)
        self.assertEqual(d["download"]["files"], 2, "the gguf and the counts csv that belongs to it")
        d = self.wait_end()
        self.assertEqual((d["state"], d["error"]), ("done", None), d)
        self.assertEqual(sorted(d["written"]), ["small.gguf", "small.gguf.expert-counts.csv"], d)
        self.assertEqual(self.slurp("small.gguf"), HF_SMALL)
        self.assertEqual(self.slurp("small.gguf.expert-counts.csv"), HF_CSV)
        self.assertFalse([f for f in os.listdir(self.dl_dir) if f.endswith(".part")], "no .part may survive")
        have = {f["name"] for r in self.list_now()["repos"] for f in r["files"] if f["have"]}
        self.assertEqual(have, {"small.gguf"}, "a landed file must read as already here")

    def test_a_split_model_fetches_every_part(self):
        code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Split", "file": "big-00002-of-00002.gguf"})
        self.assertEqual(code, 200, d)
        self.assertEqual(d["download"]["files"], 2, "asking for one part must fetch the whole model")
        self.assertEqual(self.wait_end()["state"], "done")
        self.assertEqual(sorted(os.listdir(self.dl_dir)),
                         ["big-00001-of-00002.gguf", "big-00002-of-00002.gguf"])

    def test_a_repository_listing_one_part_of_a_split_model_is_refused_not_half_fetched(self):
        both = HF_REPOS["poisonxa/Pub-Split"]
        HF_REPOS["poisonxa/Pub-Split"] = {"big-00001-of-00002.gguf": HF_BIG}
        try:
            code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Split",
                                                         "file": "big-00001-of-00002.gguf"})
            self.assertEqual((code, d["code"]), (502, "incomplete"), d)
            self.assertEqual(os.listdir(self.dl_dir), [], "nothing may be written for a model that cannot work")
        finally:
            HF_REPOS["poisonxa/Pub-Split"] = both

    def test_a_file_that_does_not_match_the_published_sha_is_thrown_away(self):
        good = bytearray(HF_BIG[:len(HF_SMALL)])
        good[-1] ^= 0xFF                                   # same length, one wrong byte: only the sha can tell
        HF_REPOS["poisonxa/Pub-Model"]["small.gguf"] = bytes(good)
        try:
            code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Model", "file": "small.gguf"})
            self.assertEqual(code, 200, d)
            d = self.wait_end()
            self.assertEqual(d["state"], "failed", d)
            self.assertIn("sha256", d["error"], d)
            self.assertEqual(os.listdir(self.dl_dir), [], "a bad file must not be kept, not even as a .part")
        finally:
            HF_REPOS["poisonxa/Pub-Model"]["small.gguf"] = HF_SMALL

    def test_one_download_at_a_time_and_it_resumes_where_it_stopped(self):
        HF_REPOS["poisonxa/Pub-Big"] = {"one-1mib.gguf": HF_MID}    # 1 MiB: several reads, so a cancel lands mid-file
        try:
            FakeHF.delay = 0.05                     # 16 pieces of 64 KiB, so the whole file takes ~0.8 s
            code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Big", "file": "one-1mib.gguf"})
            self.assertEqual(code, 200, d)
            code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Big", "file": "one-1mib.gguf"})
            self.assertEqual((code, d["code"]), (409, "busy"), d)
            part = os.path.join(self.dl_dir, "one-1mib.gguf.part")
            for _ in range(400):                    # cancel as soon as a read has landed, not on a guessed sleep
                if os.path.exists(part) and os.path.getsize(part):
                    break
                time.sleep(0.01)
            self.assertEqual(self.post("/api/models/download/cancel", {})[1], {"cancelled": True})
            d = self.wait_end()
            self.assertEqual(d["state"], "cancelled", d)
            self.assertTrue(os.path.exists(part), "the half-fetched file must be kept for the next attempt")
            got = os.path.getsize(part)
            self.assertTrue(0 < got < len(HF_MID), "the cancel must catch it part-way, not at either end")
            # the second attempt must ask for the rest, not start over
            FakeHF.delay, first = 0.0, len(FakeHF.seen_ranges)
            code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Big", "file": "one-1mib.gguf"})
            self.assertEqual(code, 200, d)
            self.assertEqual(self.wait_end()["state"], "done", d)
            self.assertEqual(self.slurp("one-1mib.gguf"), HF_MID)
            self.assertEqual([r for n, r in FakeHF.seen_ranges[first:] if n == "one-1mib.gguf"],
                             ["bytes=%d-" % got], "the resume must ask for exactly the bytes it is missing")
        finally:
            HF_REPOS.pop("poisonxa/Pub-Big", None)
            self.app.hf.token_seen = None

    def test_bad_input_is_a_sentence_a_page_can_show(self):
        for body in ({"file": "small.gguf"}, {"repo": "poisonxa/Pub-Model"}, {"repo": "poisonxa/Pub-Model",
                                                                             "file": "   "}):
            code, d = self.post("/api/models/download", body)
            self.assertEqual((code, d["code"]), (400, "invalid"), d)
        code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Model", "file": "small.gguf",
                                                     "folder": "/etc"})
        self.assertEqual((code, d["code"]), (400, "invalid"), d)
        self.assertIn("model folder", d["error"], d)
        code, d = self.post("/api/models/download", {"repo": "poisonxa/Pub-Model",
                                                     "file": "small.gguf.expert-counts.csv"})
        self.assertEqual((code, d["code"]), (400, "invalid"), d)
        code, d = self.post("/api/models/download", {"repo": "poisonxa/No-Such-Repo", "file": "a.gguf"})
        self.assertEqual((code, d["code"]), (404, "not_found"), d)
        self.assertEqual(req(self.port, "/api/models/download/cancel", "POST", {})[0], 200)


# The engine beta channel in Control (Valued Supporters first): GET /api/update shows channel + beta_allowed, POST
# /api/update/channel records it, GET /api/update/notes returns the verified notes as text. A fake pxa-update stands in for
# the binary, so these test the routes and their error mapping, never a download.
FAKE_UPDATER = r"""#!/bin/sh
d="$PXA_FAKE_UPD_DIR"
echo "$*" >> "$d/calls"
case "$1" in
  check)    ch=$(cat "$d/channel" 2>/dev/null || echo stable)
            if [ "$ch" = beta ]; then echo "current=v3.1.0 latest=4.0-beta.1 update=yes channel=beta";
            else echo "current=v3.1.0 latest=v3.1.0 update=no channel=stable"; fi ;;
  channels) if [ -e "$d/valued" ]; then echo "channels=stable,beta beta=yes current=stable";
            else echo "channels=stable beta=no current=stable"; fi ;;
  channel)  if [ "$2" = beta ] && [ ! -e "$d/valued" ]; then
              echo "pxa-update: the beta channel is for Valued Supporters" >&2; exit 1; fi
            echo "$2" > "$d/channel"; echo "channel=$2" ;;
  notes)    printf '{"version":"4.0-beta.1","build_id":"e4.0-beta.1-0123456789","notes":"# 4.0 beta\\n<b>not html</b>\\n"}\n' ;;
  apply)    echo "installed pxa-4.0-beta.1 channel=beta version=4.0-beta.1" ;;
  *)        exit 2 ;;
esac
"""


class EngineChannelRoutes(unittest.TestCase):
    """The Engine channel row's seams: who is offered beta, the switch, back to stable, and the notes."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="pxa-engine-ch-")
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.bin = os.path.join(self.d, "pxa-update")
        with open(self.bin, "w") as f:
            f.write(FAKE_UPDATER)
        os.chmod(self.bin, 0o755)
        self.saved = {k: os.environ.get(k) for k in ("PXA_UPDATE_BIN", "PXA_FAKE_UPD_DIR")}
        self.addCleanup(self._restore_env)
        os.environ["PXA_UPDATE_BIN"] = self.bin
        os.environ["PXA_FAKE_UPD_DIR"] = self.d
        R = sys.modules["pxa_control.routes"]
        self.R = R
        R._ENGINE_CH.update(t=0.0, beta=False, ttl=0, last="stable")      # no cached answer from another test
        self.addCleanup(R._ENGINE_CH.update, t=0.0, beta=False, ttl=0, last="stable")
        self.app = C.App(L, port=0, models_dirs=[MODELS])
        self.srv, self.port = serve_app(self.app)
        self.addCleanup(self._stop)

    def _restore_env(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _stop(self):
        self.srv.shutdown()
        self.srv.server_close()

    def get(self, path):
        code, _, body = req(self.port, path)
        return code, json.loads(body)

    def post(self, path, body):
        code, _, raw = req(self.port, path, "POST", body)
        return code, json.loads(raw)

    def calls(self):
        with open(os.path.join(self.d, "calls")) as f:
            return [x.split()[0] for x in f.read().splitlines() if x.strip()]

    def test_a_supporter_or_free_key_is_not_offered_beta_and_cannot_switch(self):
        code, d = self.get("/api/update")
        self.assertEqual(code, 200, d)
        self.assertEqual((d["channel"], d["beta_allowed"], d["update"]), ("stable", False, False), d)
        code, d = self.post("/api/update/channel", {"channel": "beta"})
        self.assertEqual((code, d["code"]), (403, "not_valued"), d)
        self.assertIn("Valued Supporters", d["error"])
        self.assertFalse(os.path.exists(os.path.join(self.d, "channel")))

    def test_a_valued_key_joins_beta_then_goes_back_to_stable(self):
        open(os.path.join(self.d, "valued"), "w").close()
        code, d = self.get("/api/update")
        self.assertEqual((code, d["channel"], d["beta_allowed"]), (200, "stable", True), d)
        code, d = self.post("/api/update/channel", {"channel": "beta"})
        self.assertEqual((code, d), (200, {"ok": True, "channel": "beta"}))
        code, d = self.get("/api/update")
        self.assertEqual((d["channel"], d["latest"], d["update"]), ("beta", "4.0-beta.1", True), d)
        code, d = self.post("/api/update/apply", {})
        self.assertEqual(code, 200, d)
        self.assertIn("channel=beta", d["output"])
        self.assertEqual(self.R._ENGINE_CH["last"], "beta")          # the long (tunnel) timeout applies
        code, d = self.post("/api/update/channel", {"channel": "stable"})
        self.assertEqual((code, d["channel"]), (200, "stable"), d)
        code, d = self.get("/api/update")
        self.assertEqual(d["channel"], "stable", d)
        # beta_allowed is cached: one `channels` call across the two GETs
        self.assertEqual(self.calls().count("channels"), 1, self.calls())

    def test_bad_channel_names_are_refused_before_the_binary_runs(self):
        for bad in ("", "nightly", None, "beta; rm -rf /"):
            code, d = self.post("/api/update/channel", {"channel": bad})
            self.assertEqual((code, d["code"]), (400, "invalid"), (bad, d))
        self.assertFalse(os.path.exists(os.path.join(self.d, "calls")))

    def test_update_counts_only_live_servers(self):
        # FR#14: "Stop the running servers first" with nothing running. A seat that ran and exited (proc, req and phase kept, a
        # profile restored from control.json) is not running; a live one is, and the install is refused only then.
        import subprocess as sp
        done = self.app.get_seat("old", create=True)
        done.proc = sp.Popen([sys.executable, "-c", "pass"])
        done.proc.wait()
        done.req, done.phase = {"port": 8123, "model": FAKE_MODEL, "gpus": [0]}, "exited (0)"
        self.app.get_seat("restored", create=True)
        code, d = self.get("/api/update")
        self.assertEqual((code, d["servers_running"], d["update"]), (200, 0, False), d)
        live = self.app.get_seat("live", create=True)
        live.proc = sp.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(live.proc.wait)
        self.addCleanup(live.proc.kill)
        code, d = self.get("/api/update")
        self.assertEqual((code, d["servers_running"]), (200, 1), d)
        code, d = self.post("/api/update/apply", {})
        self.assertEqual((code, d["code"]), (409, "servers_running"), d)
        live.proc.kill()
        live.proc.wait()
        code, d = self.get("/api/update")
        self.assertEqual(d["servers_running"], 0, d)
        code, d = self.post("/api/update/apply", {})
        self.assertEqual(code, 200, d)

    def test_notes_come_back_as_plain_text_fields(self):
        code, d = self.get("/api/update/notes")
        self.assertEqual(code, 200, d)
        self.assertEqual((d["version"], d["build_id"]), ("4.0-beta.1", "e4.0-beta.1-0123456789"))
        self.assertIn("<b>not html</b>", d["notes"])       # the page renders it with textContent
        self.assertEqual(self.calls(), ["notes"])

    def test_no_binary_means_no_channel_routes(self):
        os.environ["PXA_UPDATE_BIN"] = os.path.join(self.d, "missing")
        code, d = self.post("/api/update/channel", {"channel": "stable"})
        if code == 404:
            self.assertEqual(d["code"], "no_updater", d)
        else:   # an install that ships its own pxa-update next to Control: then the route works
            self.assertEqual(code, 200, d)

    def test_the_page_carries_the_channel_row_and_renders_notes_as_text(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "index.html"), encoding="utf-8") as f:
            page = f.read()
        for needle in ('id="up-ch"', 'id="up-beta"', 'id="up-stable"', 'id="up-notes"', "/api/update/channel",
                       "/api/update/notes", "(Beta channel)", "box.textContent = d.notes"):
            self.assertIn(needle, page)
        self.assertNotIn("up-notes\").innerHTML", page)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
