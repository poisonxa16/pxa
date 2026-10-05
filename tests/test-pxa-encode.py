#!/usr/bin/env python3
"""The Encode tab's backend (tools/pxa_encode*.py): the adapter that is the only code that knows the encoder's command line, the
fit math and tier recommendation, the source inspection, the disk / RAM / VRAM / tools / licence checks, the resumable job runner,
the "Get the encoder" download + signature check, and the rule that the licence key never leaves the licence server.

No GPU, no real model, no real encoder, no network beyond 127.0.0.1: the encoder, the engine tools, Hugging Face and the licence
server are the fakes of tests/encode_fakes.py.

    python3 tests/test-pxa-encode.py          (wired into CTest as test-pxa-encode)
"""
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import stat
import struct
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="pxa-encode-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x700"
os.environ["PXA_ENCODE_HOME"] = os.path.join(TMP, "encode-home")
os.environ["PXA_ENCODER_HOME"] = os.path.join(TMP, "encoder-home")
os.environ["PXA_CONVERT_MODULES"] = ""
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ.pop("PXA_ENCODER", None)
os.environ.pop("HF_TOKEN", None)
os.environ.pop("PXA_CALIB", None)
os.environ["HOME"] = os.path.join(TMP, "home")
os.makedirs(os.environ["HOME"], exist_ok=True)
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)

_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import encode_fakes as F  # noqa: E402
import pxa_encode as E  # noqa: E402
import pxa_encode_adapter as AD  # noqa: E402
import pxa_encode_pkg as PK  # noqa: E402
import pxa_encode_plan as PL  # noqa: E402

GOOD_KEY, SECRET = F.GOOD_KEY, "A1b2C3d4A1b2C3d4"
os.environ["PXA_PACKAGE_PUBKEY"] = F.PUB_HEX

REPOS = {"tiny/Tiny-Qwen3": F.tiny_repo(), "tiny/NoDerivs": F.tiny_repo(license="cc-by-nd-4.0"), "tiny/Gated": dict(F.tiny_repo(), need_token=True, gated=True),
         "tiny/Redirect": dict(F.tiny_repo(), redirect=True, need_token=True), "tiny/Slow": dict(F.tiny_repo(), delay=0.02),
         "tiny/Odd": F.tiny_repo(arch="MadeUpForCausalLM")}
HF = F.FakeHF(REPOS)
CDN = F.FakeCDN(HF)
HF.cdn = CDN
HF.start()
CDN.start()
os.environ["HF_ENDPOINT"] = HF.url


def tearDownModule():
    HF.stop()
    CDN.stop()
    shutil.rmtree(TMP, ignore_errors=True)


class FakeHost(object):
    """What the Encode service needs from PXA Control, without PXA Control."""
    L = L

    def __init__(self, engine_dirs=(), roots=(), cfg=None, gpu_rows=None):
        self.cfg = cfg if cfg is not None else {}
        self._engines = list(engine_dirs)
        self._roots = list(roots)
        self._rows = gpu_rows
        self.cuda = "12.4"

    def gpus(self):
        if self._rows is not None:
            return list(self._rows), None
        return L.gpu_table()

    def cfg_load(self):
        return json.loads(json.dumps(self.cfg))

    def cfg_update(self, fn):
        fn(self.cfg)

    def engine_dirs(self):
        return list(self._engines)

    def model_roots(self):
        return list(self._roots)

    def cuda_version(self):
        return self.cuda


_n = [0]
# The CLI version the fake encoders report: "2" = the whole flow is one `pxqe make` call (the real encoder today); "1" = an older encoder, for which
# Control keeps the multi-command path. The legacy classes below run the same job tests against the old path.
DEFAULT_CLI = [None]


def legacy():
    return DEFAULT_CLI[0] == "1"


class Rig(object):
    """One isolated Encode service with its own engine, encoders, model folder and data folder."""

    def __init__(self, edition="pro", key=True, gpus=None, with_encoder=True, tiers=None, cli=None, **enc_kw):
        _n[0] += 1
        cli = cli or DEFAULT_CLI[0]
        if cli:
            enc_kw["cli"] = cli
        self.dir = os.path.join(TMP, "rig%d" % _n[0])
        self.engine = F.make_engine_dir(os.path.join(self.dir, "engine"))
        self.models = os.path.join(self.dir, "models")
        os.makedirs(self.models)
        self.encoder_dir = os.path.join(self.dir, "enc-" + edition)
        cfg = {}
        if with_encoder:
            self.cli = F.make_fake_encoder(self.encoder_dir, edition, build_id="b-%s-1" % edition, tiers=tiers, **enc_kw)
            cfg["encode"] = {"extra": [self.cli]}
        else:
            self.cli = None
        if key:
            cfg.setdefault("encode", {})["licence_key"] = GOOD_KEY
        self.host = FakeHost([self.engine], [self.models], cfg, gpus)
        self.svc = E.EncodeService(self.host, data=os.path.join(self.dir, "data"))

    def body(self, tier="pxqn4", **kw):
        b = {"source": "tiny/Tiny-Qwen3", "cards": [0, 1], "tier": tier, "work_dir": os.path.join(self.dir, "work"), "out_dir": self.models}
        b.update(kw)
        return b

    def set_fake(self, **kw):
        F.set_fake(self.encoder_dir, **kw)

    def pace(self, d):
        F.set_engine_delay(self.engine, d)
        if self.cli:
            self.set_fake(delay=d)

    def wait(self, jid, states=("done", "failed", "cancelled", "interrupted"), timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.svc.job_view(jid)
            if j["status"] in states:
                return j
            time.sleep(0.05)
        raise AssertionError("job %s still %s after %ds: %s" % (jid, self.svc.job_view(jid)["status"], timeout, self.svc.log_view(jid)["lines"][-6:]))

    def wait_stage(self, jid, stage, status="running", timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.svc.job_view(jid)
            if next(s for s in j["stages"] if s["id"] == stage)["status"] == status:
                return j
            if j["status"] in ("failed", "done", "cancelled"):
                break
            time.sleep(0.03)
        raise AssertionError("stage %s never became %s" % (stage, status))

    def wait_proc(self, jid, timeout=10):
        t0 = time.time()
        while self.svc.rt[jid].proc is None and time.time() - t0 < timeout:
            time.sleep(0.02)
        return self.svc.rt[jid].proc

    def state_file(self):
        try:
            with open(os.path.join(self.encoder_dir, "fake-state.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}


# =====================================================================================================================
class AdapterInfo(unittest.TestCase):
    def test_parse_info_free_and_pro(self):
        free = AD.parse_info(json.dumps({"edition": "free", "version": "1.2", "build_id": "b1", "tiers": ["PXQ2", "pxq-3"], "features": []}))
        self.assertEqual((free["edition"], free["tiers"], free["licence"]), ("free", ["pxq2", "pxq3"], None))
        pro = AD.parse_info(json.dumps({"edition": "pro", "version": "1.2", "build_id": "b2", "tiers": ["pxqn4"], "features": ["ldlq"],
                                        "licence": {"state": "active", "user": "u", "encodes_left": 3, "expires": "2026-12-01"}}))
        self.assertEqual(pro["licence"], {"state": "valid", "user": "u", "encodes_left": 3, "unlimited": False, "key_id": "", "checked": False, "reason": "", "expires": "2026-12-01"})
        full = AD.parse_info(json.dumps({"edition": "pro", "build_id": "b", "tiers": ["pxq_universal"], "runtime": {"lib": "unloadable", "detail": "no NVIDIA driver"},
                                         "licence": {"state": "refused", "reason": "HTTP 403", "unlimited": True, "key_id": "K-1", "checked": True}}))
        self.assertEqual(full["tiers"], ["pxquniversal"])
        self.assertEqual({k: full["runtime"][k] for k in ("lib", "detail", "missing", "fix", "resolver")}, {"lib": "unloadable", "detail": "no NVIDIA driver", "missing": [], "fix": "", "resolver": 0})
        self.assertEqual((full["licence"]["state"], full["licence"]["reason"], full["licence"]["unlimited"]), ("refused", "HTTP 403", True))

    def test_licence_states_map_to_the_few_the_page_knows(self):
        for word, want in (("expired", "expired"), ("Revoked", "revoked"), ("none", "no_key"), ("no-key", "no_key"), ("no-server", "no_server"), ("unreachable", "offline"),
                           ("suspended", "suspended"), ("refused", "refused"), ("disabled", "revoked"), ("exhausted", "no_quota"), ("???", "unknown")):
            o = AD.parse_info(json.dumps({"edition": "pro", "build_id": "b", "tiers": [], "licence": {"state": word}}))
            self.assertEqual(o["licence"]["state"], want, word)
        self.assertIsNone(AD.parse_info(json.dumps({"edition": "pro", "build_id": "b", "tiers": [], "licence": {"state": "valid", "encodes_left": -3}}))["licence"]["encodes_left"])
        self.assertIsNone(AD.parse_info(json.dumps({"edition": "pro", "build_id": "b", "tiers": [], "licence": {"state": "valid", "encodes_left": True}}))["licence"]["encodes_left"])

    def test_broken_info_is_a_sentence_not_a_trace(self):
        for text, frag in (("not json", "valid JSON"), ("[]", "JSON object"), ('{"edition":"gold","tiers":[],"build_id":"b"}', "expected free or pro"),
                           ('{"edition":"free","build_id":"b"}', "list its quantization tiers"), ('{"edition":"free","tiers":[]}', "build id"),
                           ('{"edition":"free","tiers":[],"build_id":"a b"}', "build id")):
            with self.assertRaises(AD.AdapterError) as cm:
                AD.parse_info(text)
            self.assertIn(frag, str(cm.exception))

    def test_probe_real_fake_clis(self):
        d = os.path.join(TMP, "probe")
        pro = F.make_fake_encoder(os.path.join(d, "pro"), "pro", build_id="bp")
        free = F.make_fake_encoder(os.path.join(d, "free"), "free", build_id="bf")
        a, b = AD.probe(pro), AD.probe(free)
        self.assertTrue(a["ok"] and a["edition"] == "pro" and a["licence"]["state"] == "no_key")          # no key configured: state none
        self.assertEqual(AD.probe(pro, env={"PXQE_KEY": GOOD_KEY})["licence"]["state"], "unchecked")      # a key, not asked of the server yet
        self.assertEqual(a["runtime"]["lib"], "loadable")
        self.assertTrue(b["ok"] and b["edition"] == "free" and "pxq3" in b["tiers"] and b["licence"] is None)
        self.assertEqual(b["runtime"]["lib"], "none")
        self.assertEqual((a["platform"], a["cuda_major"]), ("linux-x86_64", 12))
        F.set_fake(os.path.join(d, "pro"), info_broken="garbage")
        g = AD.probe(pro)
        self.assertFalse(g["ok"])
        self.assertIn("valid JSON", g["error"])
        F.set_fake(os.path.join(d, "pro"), info_broken="crash")
        c = AD.probe(pro)
        self.assertFalse(c["ok"])
        self.assertIn("Segmentation", c["error"])
        m = AD.probe(os.path.join(d, "nope", "pxqe"))
        self.assertEqual((m["ok"], m["error"]), (False, "that file does not exist"))

    def test_check_licence_flag_is_a_separate_call(self):
        d = os.path.join(TMP, "probe2")
        pro = F.make_fake_encoder(os.path.join(d, "pro"), "pro", licence_check={"state": "revoked", "encodes_left": 0})
        env = {"PXQE_KEY": GOOD_KEY}
        self.assertEqual(AD.probe(pro, env=env)["licence"]["state"], "unchecked")                       # offline view: no server asked
        self.assertEqual(AD.probe(pro, check_licence=True, env=env)["licence"]["state"], "revoked")


class RealOutputs(unittest.TestCase):
    """Strings copied from the real thing (the real Free package's `pxqe info --json`, PACKAGE.md's Pro example, a real classic quantize
    run), so the parsers are checked against what the tools actually print and not only against the fakes."""
    REAL_FREE = ('{"build_id": "fre20261004a", "cli": "1", "cuda_major": 12, "edition": "free", "features": ["classic-quantizer"], "platform": "linux-x86_64", '
                 '"runtime": {"lib": "none"}, "tiers": ["pxq1", "pxq2", "pxq3", "pxq4", "pxq4hq", "pxq6", "pxq_universal"], "version": "free-fre20261004a"}')
    SPEC_PRO = ('{"edition":"pro","version":"pro-pxqe20261004b","build_id":"pxqe20261004b","cli":"1","platform":"linux-x86_64","cuda_major":12,'
                '"tiers":["pxq1","pxq2","pxq3","pxq4","pxq4hq","pxq6","pxq_universal","pxqn1","pxqn2","pxqn3","pxqn3s8","pxqn4","pxqn4s8","pxqn5"],'
                '"features":["classic-quantizer","ldlq","hessian","watermark","licence"],"runtime":{"lib":"loadable"},'
                '"licence":{"state":"unchecked","user":"Ann","key_id":"K-ABCDEF01","encodes_left":null,"unlimited":null,"expires":null,"checked":false}}')

    def test_real_free_and_the_specs_pro_example_parse(self):
        f = AD.parse_info(self.REAL_FREE)
        self.assertEqual((f["edition"], f["build_id"], f["runtime"]["lib"], f["licence"]), ("free", "fre20261004a", "none", None))
        self.assertEqual(f["tiers"], ["pxq1", "pxq2", "pxq3", "pxq4", "pxq4hq", "pxq6", "pxquniversal"])
        p = AD.parse_info(self.SPEC_PRO)
        self.assertEqual((p["licence"]["state"], p["licence"]["user"], p["licence"]["key_id"], p["licence"]["encodes_left"], p["licence"]["checked"]), ("unchecked", "Ann", "K-ABCDEF01", None, False))
        self.assertEqual(len(p["tiers"]), 14)
        self.assertEqual({PL.tier_key(t) for t in p["tiers"]} - {t["key"] for t in PL.TIERS}, set())        # every tier the build lists is one this Control knows

    def test_real_quantize_progress_lines(self):
        import re as _re
        rx = _re.compile(r"^\[\s*(\d+)/\s*(\d+)\]")
        for line, n, m in (("[ 305/ 311]                      blk.9.attn_k.weight q8_0     -> pxqn3    rot           0.81 MiB", 305, 311), ("[   1/ 311]      token_embd.weight - [ 2048, 151936,     1], type =   q8_0", 1, 311)):
            mm = rx.match(line.strip())
            self.assertEqual((int(mm.group(1)), int(mm.group(2))), (n, m))

    def test_the_real_quantizers_composition_refusal_is_explained_not_quoted(self):
        text = ("llama_model_quantize_internal: removed mislabelled output /tmp/x.tmp\n"
                "llama_model_quantize: failed to quantize: PXQ composition assertion: target PXQ3 - 3.27 bpw, LM8 bit-plane x E16-row scales, slab layout produced 43.8% "
                "PXQ-family bytes (floor 50%; backbone=v2). The output would misrepresent its contents.\n"
                "main: failed to quantize model from '/tmp/q17-Q8_0.gguf'")
        r = AD.explain_failure(text, 1)
        self.assertEqual(r["code"], "composition")
        self.assertIn("43.8%", r["message"])
        self.assertIn("PXQ3", r["message"])
        self.assertIn("higher tier", r["hint"])

    def test_the_most_informative_line_is_shown_not_the_generic_last_one(self):
        r = AD.explain_failure("loading...\nllama_model_quantize: failed to quantize: unsupported tensor shape for blk.3\nmain: failed to quantize model from '/x'", 1)
        self.assertIn("unsupported tensor shape", r["message"])
        self.assertNotIn("main: failed to quantize model from", r["message"])

    def test_small_models_get_a_composition_warning_before_the_run(self):
        # the real 1.7B (2.03B params, 311M embedding): about 49% of a PXQ3 file would be PXQ3 (measured 43.8%: the quantizer refused it), 61% for PXQ6
        P, EMB = 2_031_739_904, 311_164_928
        self.assertLess(PL.tier_fraction(P, EMB, EMB, 3.25), 0.55)
        self.assertGreater(PL.tier_fraction(P, EMB, EMB, 5.25), 0.55)
        self.assertIsNone(PL.tier_fraction(None, 0, 0, 3.25))
        big = PL.tier_fraction(27_320_697_856, 248_320 * 5120, 248_320 * 5120, 3.25)
        self.assertGreater(big, 0.75)           # the 27B is well clear of the 50% floor


class AdapterDiscovery(unittest.TestCase):
    def test_candidate_places_and_swap_order(self):
        d = os.path.join(TMP, "disc")
        shutil.rmtree(d, ignore_errors=True)
        free = F.make_fake_encoder(os.path.join(d, "installed", "free", "b1"), "free", build_id="bf")
        pro = F.make_fake_encoder(os.path.join(d, "installed", "pro", "b2"), "pro", build_id="bp")
        eng = os.path.join(d, "engine")
        os.makedirs(os.path.join(eng, "bin"))
        eng_cli = F.make_fake_encoder(os.path.join(eng, "bin"), "free", build_id="be")
        onpath = F.make_fake_encoder(os.path.join(d, "pathdir"), "free", build_id="bpath")
        os.environ["PXA_ENCODER_HOME"] = os.path.join(d, "installed")
        try:
            found = AD.discover([], [eng], path_env=os.path.dirname(onpath))
            got = {f["path"] for f in found}
            self.assertEqual(got, {free, pro, eng_cli, onpath})
            self.assertEqual(found[0]["edition"], "pro")                  # Pro first, then Free
            self.assertEqual(AD.choose(found)["edition"], "pro")
            self.assertEqual(AD.choose(found, preferred=free)["path"], free)    # the user can switch
            self.assertEqual(AD.choose([]), None)
            # a path the user configured is found even when it is not in any usual place
            odd = F.make_fake_encoder(os.path.join(d, "odd"), "pro", build_id="bo")
            self.assertIn(odd, {f["path"] for f in AD.discover([os.path.dirname(odd)], [], path_env="")})
        finally:
            os.environ["PXA_ENCODER_HOME"] = os.path.join(TMP, "encoder-home")

    def test_broken_candidates_are_kept_with_their_error_and_never_chosen(self):
        d = os.path.join(TMP, "disc2")
        bad = F.make_fake_encoder(os.path.join(d, "bad"), "pro", info_broken="garbage")
        ok = F.make_fake_encoder(os.path.join(d, "ok"), "free")
        found = AD.discover([bad, ok], [], path_env="")
        self.assertEqual([f["ok"] for f in found], [True, False])
        self.assertIn("valid JSON", found[1]["error"])
        self.assertEqual(AD.choose(found)["path"], ok)

    def test_missing_everything_is_an_empty_list(self):
        self.assertEqual(AD.discover([], [], path_env=""), [])

    def test_same_file_through_a_symlink_is_one_candidate(self):
        d = os.path.join(TMP, "disc3")
        cli = F.make_fake_encoder(os.path.join(d, "a"), "free")
        os.makedirs(os.path.join(d, "b"))
        os.symlink(cli, os.path.join(d, "b", "pxqe"))
        self.assertEqual(len(AD.candidate_paths([cli, os.path.join(d, "b", "pxqe")], [], "")), 1)

    def test_python_wrapper_runs_under_this_interpreter(self):
        self.assertEqual(AD.cli_argv("/x/pxqe.py", ["info"])[:2], [sys.executable, "/x/pxqe.py"])
        self.assertEqual(AD.cli_argv("/x/pxqe", ["info"]), ["/x/pxqe", "info"])


class AdapterCommands(unittest.TestCase):
    def test_key_only_goes_to_pro_and_only_by_environment(self):
        pro, free = {"edition": "pro"}, {"edition": "free"}
        self.assertEqual(AD.run_env(pro, "k", "srv"), {"PXQE_KEY": "k", "PXQE_SERVER": "srv"})
        self.assertEqual(AD.run_env(free, "k", "srv"), {})
        self.assertEqual(AD.run_env(pro, None, None), {})
        argv = AD.argv_run("/e/pxqe", "q8.gguf", "/w/out.gguf", act="/a", device=0, resume="J7")
        self.assertNotIn("k", argv)
        self.assertEqual(argv, ["/e/pxqe", "run", "--src", "q8.gguf", "--dst", "/w/out.gguf", "--act", "/a", "--device", "0", "--resume", "J7"])
        self.assertEqual(AD.hess_dir_for("/w/out.gguf"), "/w/hess")                # where `run --act` puts the Hessians

    def test_run_argv_shapes(self):
        self.assertIn("--hdir", AD.argv_run("/e/pxqe", "s", "d", hdir="/h"))
        plain = AD.argv_run("/e/pxqe", "s", "d")
        self.assertNotIn("--hdir", plain)
        self.assertNotIn("--act", plain)
        self.assertEqual(AD.argv_prep("/e/pxqe", "s", "/a", "/o", device=1)[:3], ["/e/pxqe", "prep", "--src"])
        self.assertEqual(AD.argv_quantize("/e/pxqe", "in.gguf", "out.gguf", "PXQ4", 8), ["/e/pxqe", "quantize", "in.gguf", "out.gguf", "PXQ4", "8"])
        self.assertEqual(AD.argv_quantize("/e/pxqe", "in.gguf", "out.gguf", "PXQ4", 8, True)[2:4], ["--allow-requantize", "--i-know-this-is-double-lossy"])
        self.assertTrue(AD.is_classic("pxq4") and AD.is_classic("PXQ_UNIVERSAL") and not AD.is_classic("pxqn4"))
        self.assertEqual(AD.ftype_name("pxq_universal"), "PXQ_UNIVERSAL")

    def test_skeleton_recipe_is_uniform_tiers_only(self):
        text = AD.skeleton_tier_map("PXQN-4")
        self.assertIn("pxqn4", text)
        self.assertIn("attn_k|attn_v", text)
        self.assertIn("^token_embd\\.weight$ q6_K", text)
        self.assertIsNone(AD.skeleton_tier_map("pxqn3bal"))
        self.assertIsNone(AD.skeleton_tier_map("something"))
        self.assertEqual(AD.ftype_name("pxqn4s8"), "PXQN4S8")
        self.assertEqual(AD.ftype_name("pxq4hq"), "PXQ4-HQ")


class AdapterProgress(unittest.TestCase):
    def test_lines_the_encoder_prints(self):
        P = AD.parse_line
        self.assertEqual(P("pxqe: job 20261004-abc started (4 left this month)"), {"event": "job", "jid": "20261004-abc", "left": 4, "unlimited": False})
        self.assertEqual(P("pxqe: job J1 started (3 encodes left)"), {"event": "job", "jid": "J1", "left": 3, "unlimited": False})
        self.assertEqual(P("pxqe: job J1 started (unlimited encodes left)"), {"event": "job", "jid": "J1", "left": None, "unlimited": True})
        self.assertEqual(P("pxqe encode: 1 file(s) layers 0-99 tensors 196 damp 0.3 minratio 1.5 ldlq 1 ldlq_down 1 wm 0 dev Tesla V100-PCIE-16GB sm_70 decoder-check engine"),
                         {"event": "total", "tensors": 196})
        t = P("mine-a3.gguf blk.0.ffn_down.weight t261 R2048 K6144 ldlq rot 1 relmse 0.412129 pack decode-exact enc 0s")
        self.assertEqual((t["event"], t["name"], t["mode"], t["exact"], t["bad"]), ("tensor", "blk.0.ffn_down.weight", "ldlq", True, False))
        r = P("ref-a3.gguf blk.9.attn_v.weight t257 R1024 K2048 rtn rot 1 relmse 0.024 pack bit-exact enc 0s")
        self.assertEqual((r["mode"], r["exact"]), ("rtn", True))
        bad = P("x.gguf blk.1.ffn_up.weight t259 R64 K64 ldlq rot 1 relmse 0.1 pack MISMATCH enc 0s")
        self.assertTrue(bad["bad"] and not bad["exact"])
        h = P("/data/x/ref/H/blk.12.down_in.hess K=6144 ntok=10240 5.5s")
        self.assertEqual((h["event"], h["name"], h["k"]), ("hess", "blk.12.down_in", 6144))
        self.assertEqual(P("SUMMARY mine-a3.gguf ldlq 196 rtn 0 skip 0")["event"], "summary")
        self.assertEqual(P("ALL DONE total 34s"), {"event": "done", "seconds": 34})
        self.assertEqual(P("pxqe: job J9 ok"), {"event": "job_end", "jid": "J9", "ok": True, "refunded": False})
        self.assertEqual(P("pxqe: job J9 failed (quota refunded)")["refunded"], True)
        self.assertEqual(P("pxqe: job J9 failed (encode refunded)")["refunded"], True)
        self.assertIsNone(P("some other line"))
        self.assertIsNone(P(""))

    def test_failures_in_plain_words(self):
        X = AD.explain_failure
        cases = [("pxqe: refused by the licence server (HTTP 402): no encodes left: monthly quota used", "no_quota", "no encodes left"),
                 ("pxqe: refused by the licence server (HTTP 403): account disabled", "revoked", "revoked, expired or suspended"),
                 ("pxqe: refused by the licence server (HTTP 401): invalid key", "bad_key", "does not recognise this key"),
                 ("pxqe: refused by the licence server (HTTP 409): build not supported any more", "old_build", "no longer accepts"),
                 ("pxqe: cannot load the encoder library (libcusolver.so.12: no such file). It needs an NVIDIA driver and the CUDA 12 runtime (cuBLAS, cuSOLVER).", "lib_unloadable", "cannot load on this machine"),
                 ("pxqe: the licence server said 402: monthly quota used", "no_quota", "no encodes left"),
                 ("pxqe: the licence server said 403: key revoked", "revoked", "revoked, expired or suspended"),
                 ("pxqe: the licence server said 409: unknown build", "old_build", "no longer accepts"),
                 ("pxqe: the licence server said 429: too many running", "busy", "too many requests"),
                 ("pxqe: cannot reach the licence server (URLError)", "offline", "Cannot reach the licence server"),
                 ("pxqe: set PXQE_KEY (your quantizer key) and PXQE_SERVER", "no_key", "No licence key"),
                 ("pxqe: no licence key. Put your quantizer key in PXA Control (Encode > Enter key)", "no_key", "No licence key"),
                 ("pxqe: cannot find libpxqe.so (set PXQE_LIB)", "broken_install", "cannot start"),
                 ("quantize: invalid ftype PXQX", "bad_tier", "does not know this tier"),
                 ("CUDA error: out of memory", "oom", "ran out of memory"),
                 ("write failed: No space left on device", "disk_full", "disk is full")]
        for text, code, frag in cases:
            r = X(text, 1)
            self.assertEqual(r["code"], code, text)
            self.assertIn(frag, r["message"])
            self.assertTrue(r["hint"])
        g = X("Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom", 3)
        self.assertEqual(g["code"], "failed")
        self.assertNotIn("Traceback", g["message"])
        self.assertIn("boom", g["message"])


# =====================================================================================================================
class Tiers(unittest.TestCase):
    P27, EMB27 = 27_320_697_856, 248_320 * 5120

    def test_names_and_labels(self):
        self.assertEqual(PL.tier_key("PXQN-3bal"), "pxqn3bal")
        self.assertEqual(PL.tier_key("PXQ4U"), "pxq4")
        self.assertEqual(PL.tier_key("pxq_universal"), "pxquniversal")
        for k, cls in (("pxqn2", "3-bit class"), ("pxqn3", "3.5-bit class"), ("pxqn3bal", "4-bit class"), ("pxqn4", "6-bit class"), ("pxqn5", "Q6_K class"),
                       ("pxq3", "3-bit class"), ("pxq4", "4-bit class"), ("pxq6", "6-bit class"), ("pxquniversal", "4-bit class")):
            self.assertEqual(PL.TIER_BY_KEY[k]["cls"], cls, k)
        self.assertFalse(PL.TIER_BY_KEY["pxqn4s8"]["measured"])               # not measured: the page says "about"
        self.assertTrue(PL.TIER_BY_KEY["pxqn4"]["measured"])
        unknown = PL.tier_entry("pxqn9")
        self.assertEqual((unknown["family"], unknown["cls"], unknown["measured"]), ("pxqn", "unrated", False))

    def test_file_size_matches_the_measured_27b_skeleton(self):
        est = PL.file_bytes(self.P27, self.EMB27, self.EMB27, 4.25)
        measured = 15_720_261_824            # Swift 27B PXQN4 skeleton, 2026-09-30
        self.assertLess(abs(est - measured) / measured, 0.03)
        self.assertGreater(PL.file_bytes(self.P27, self.EMB27, self.EMB27, 5.25), est)
        self.assertIsNone(PL.file_bytes(None, 0, 0, 4.25))
        self.assertLess(PL.file_bytes(self.P27, self.EMB27, self.EMB27, 4.25, tied=True), est)

    def test_context_that_fits(self):
        kv = 64 * 1024                        # a 27B hybrid is far lighter; this is a heavy case
        one = PL.fit(12 * 2 ** 30, kv, 16 * 1024, 1, 262144)
        self.assertEqual(one["verdict"], "fits")
        self.assertGreater(one["ctx"], 8192)
        self.assertEqual(one["ctx"] % 1024, 0)
        self.assertEqual(PL.fit(16 * 2 ** 30, kv, 16 * 1024, 1)["verdict"], "no")          # weights fill the card
        tight = PL.fit(int(14.5 * 2 ** 30), kv, 16 * 1024, 1)
        self.assertEqual(tight["verdict"], "tight")
        capped = PL.fit(2 ** 30, 1024, 16 * 1024, 1, 4096)                                    # the model's own limit, not memory
        self.assertEqual((capped["ctx"], capped["verdict"]), (4096, "fits"))
        self.assertEqual(PL.fit(2 ** 30, kv, 0, 0)["verdict"], "unknown")
        self.assertIsNone(PL.fit(2 ** 30, None, 16 * 1024, 1)["ctx"])
        # more cards add headroom cost per card but more room overall
        self.assertGreater(PL.fit(12 * 2 ** 30, kv, 32 * 1024, 2)["ctx"], one["ctx"])

    def src(self, **kw):
        s = {"params": self.P27, "emb": self.EMB27, "head": self.EMB27, "tied": False, "kv_bytes_tok": 16 * 1024, "n_ctx_train": 262144, "moe": False}
        s.update(kw)
        return s

    def info(self, edition, tiers):
        return {"edition": edition, "tiers": tiers}

    def test_free_recommends_classic_only_and_locks_every_pxqn_tier(self):
        rows = PL.build_tiers(self.src(), [{"vram_mib": 32768, "class": None}], self.info("free", ["pxq2", "pxq3", "pxq4", "pxq6"]))
        recs = PL.recommend(rows, "free")
        self.assertTrue(recs and all(k.startswith("pxq") and not k.startswith("pxqn") for k in recs))
        locked = [r for r in rows if r["locked"]]
        listed = {t["key"] for t in PL.TIERS if t["family"] == "pxqn" and not t.get("only_if_listed")}
        self.assertEqual({r["key"] for r in locked if r["family"] == "pxqn"}, listed)       # every PXQN tier a build can list; pxqn3bal only when listed
        self.assertEqual(len(listed), 7)
        self.assertNotIn("pxqn3bal", [r["key"] for r in rows])
        for r in locked:
            if r["family"] == "pxqn":
                self.assertEqual(r["locked_reason"], "Supporter feature")
                self.assertTrue(r["size_bytes"] and r["cls"])
        self.assertTrue(all(not r["available"] for r in locked))

    def test_pro_recommends_pxqn_first_and_the_build_decides_the_list(self):
        rows = PL.build_tiers(self.src(), [{"vram_mib": 32768, "class": None}], self.info("pro", ["pxq4", "pxqn3", "pxqn4", "pxqn5"]))
        recs = PL.recommend(rows, "pro")
        self.assertTrue(recs[0].startswith("pxqn"))
        by = {r["key"]: r for r in rows}
        self.assertTrue(by["pxqn4"]["available"] and not by["pxqn2"]["available"])
        self.assertEqual(by["pxqn2"]["locked_reason"], "Not included in your plan")
        self.assertIn("best", [r["role"] for r in rows if r["role"]])

    def test_tier_list_comes_from_info_not_from_a_table(self):
        rows = PL.build_tiers(self.src(), [{"vram_mib": 16384, "class": None}], self.info("free", ["pxq3"]))
        self.assertEqual([r["key"] for r in rows if r["available"]], ["pxq3"])
        rows = PL.build_tiers(self.src(), [{"vram_mib": 16384, "class": None}], self.info("pro", ["pxqn4", "pxqn9"]))
        self.assertEqual({r["key"] for r in rows if r["available"]}, {"pxqn4", "pxqn9"})           # a tier newer than this Control still works
        self.assertIn("pxqn9", [r["key"] for r in rows])

    def test_a_plan_locks_the_pxqn_tiers_it_does_not_include(self):
        info = self.info("pro", ["pxq4", "pxqn2", "pxqn4", "pxqn5"])
        rows = PL.build_tiers(self.src(), [{"vram_mib": 32768, "class": None}], info, None, allowed={"pxqn4"})
        by = {r["key"]: r for r in rows}
        self.assertTrue(by["pxqn4"]["available"] and by["pxq4"]["available"])               # classic tiers are never restricted by the plan
        for k in ("pxqn2", "pxqn5"):
            self.assertEqual((by[k]["available"], by[k]["locked"], by[k]["locked_reason"]), (False, True, "Not included in your plan"))
        self.assertEqual(PL.recommend(rows, "pro")[0], "pxqn4")
        self.assertTrue(PL.build_tiers(self.src(), [{"vram_mib": 32768, "class": None}], info, None, allowed=None)[[r["key"] for r in rows].index("pxqn2")]["available"])
        listed = PL.build_tiers(self.src(), [{"vram_mib": 32768, "class": None}], self.info("pro", ["pxqn3bal"]))
        self.assertIn("pxqn3bal", [r["key"] for r in listed])

    def test_no_encoder_means_everything_locked(self):
        rows = PL.build_tiers(self.src(), [{"vram_mib": 16384, "class": None}], None)
        self.assertTrue(all(r["locked"] and r["locked_reason"] == "Install an encoder to make this tier." for r in rows))
        self.assertEqual(PL.recommend(rows, None), [])

    def test_what_fits_changes_with_the_cards(self):
        info = self.info("pro", ["pxqn2", "pxqn3", "pxqn4", "pxqn5"])
        one = {r["key"]: r for r in PL.build_tiers(self.src(), [{"vram_mib": 16384, "class": None}], info)}
        two = {r["key"]: r for r in PL.build_tiers(self.src(), [{"vram_mib": 16384, "class": None}] * 2, info)}
        self.assertEqual(one["pxqn5"]["verdict"], "no")            # ~20 GiB does not fit one 16 GiB card
        self.assertNotEqual(two["pxqn5"]["verdict"], "no")         # but fits two
        self.assertIn(one["pxqn2"]["verdict"], ("fits", "tight"))
        recs = PL.recommend(list(one.values()), "pro")
        self.assertNotIn("pxqn5", recs)
        small = [r for r in one.values() if r["role"] == "small"]
        self.assertTrue(not small or small[0]["bpw"] <= min(r["bpw"] for r in one.values() if r["role"] == "best"))

    def test_nothing_fits_is_an_empty_recommendation_not_a_crash(self):
        rows = PL.build_tiers(self.src(params=400_000_000_000, emb=1, head=1), [{"vram_mib": 8192, "class": None}], self.info("free", ["pxq2", "pxq3"]))
        self.assertEqual(PL.recommend(rows, "free"), [])

    def test_measured_speed_only_when_a_cell_exists(self):
        cells = [{"tier": "pxq4", "card_class": "P100-class sm_60", "n_cards": 1, "params": 35e9, "decode_tps": 62.0, "source": "README"}]
        rows = PL.build_tiers(self.src(params=35e9), [{"vram_mib": 16384, "class": "P100-class sm_60"}], self.info("free", ["pxq4", "pxq3"]), cells)
        by = {r["key"]: r for r in rows}
        self.assertEqual((by["pxq4"]["decode_tps"], by["pxq4"]["cell_source"]), (62.0, "README"))
        self.assertIsNone(by["pxq3"]["decode_tps"])                # no invented number
        rows = PL.build_tiers(self.src(params=9e9), [{"vram_mib": 16384, "class": "P100-class sm_60"}], self.info("free", ["pxq4"]), cells)
        self.assertIsNone(rows[[r["key"] for r in rows].index("pxq4")]["decode_tps"])      # a different model size: no cell
        self.assertTrue(PL.load_cells(os.path.join(TOOLS, "pxa_encode_cells.json")) is not None)


class Estimates(unittest.TestCase):
    def test_disk_peak_for_the_27b_matches_what_the_chain_used(self):
        src = {"kind": "hf", "params": 27_320_697_856, "download_bytes": 54_657_734_528}
        tier = 15_720_261_824
        stages = ["download", "convert", "reference", "skeleton", "dump", "hessians", "encode", "verify"]
        sz = PL.stage_sizes(src, tier, stages)
        gb = 1e9
        # measured on the box: dump 119-130 GB, Hessians 94 GB, Q8_0 29 GB, skeleton 15.7 GB
        self.assertGreater(sz["peak"], 230 * gb)
        self.assertLess(sz["peak"], 300 * gb)
        self.assertEqual(max(sz["live"], key=sz["live"].get), "hessians")
        fast = PL.stage_sizes(src, tier, ["download", "convert", "reference", "skeleton", "encode", "verify"])
        self.assertLess(fast["peak"], 130 * gb)                    # no LDLQ: no dump, no Hessians
        gguf = PL.stage_sizes({"kind": "gguf", "params": 27e9, "download_bytes": 0}, tier, ["skeleton", "encode", "verify"])
        self.assertNotIn("download", gguf["live"])

    def test_time_estimate_is_a_range_that_scales(self):
        small = PL.time_estimate({"params": 1.7e9, "download_bytes": 3.4e9}, ["download", "convert", "reference", "skeleton", "dump", "hessians", "encode", "verify"])
        big = PL.time_estimate({"params": 27.3e9, "download_bytes": 54e9}, ["download", "convert", "reference", "skeleton", "dump", "hessians", "encode", "verify"])
        self.assertLess(small["low"], small["high"])
        self.assertLess(small["high"], big["low"])
        two = PL.time_estimate({"params": 27.3e9}, ["dump"], n_gpu_cards=2)
        one = PL.time_estimate({"params": 27.3e9}, ["dump"], n_gpu_cards=1)
        self.assertLess(two["high"], one["high"])
        self.assertEqual(PL.human_dur(45), "45 s")
        self.assertEqual(PL.human_dur(3600 * 5), "5.0 h")
        self.assertEqual(PL.human_bytes(3 * 2 ** 30), "3.0 GiB")

    def test_vram_and_ram_for_the_encode(self):
        self.assertLess(PL.encode_vram_bytes(5120, 17408) / 2 ** 30, 8)       # a 27B encode ran on a 16 GB card
        self.assertGreater(PL.encode_vram_bytes(5120, 17408), PL.encode_vram_bytes(2048, 6144))
        self.assertGreater(PL.ram_need_bytes(5120, 17408), 2 * 2 ** 30)


class SourceFacts(unittest.TestCase):
    def test_parse_source_forms(self):
        self.assertEqual(PL.parse_source("Qwen/Qwen3-1.7B"), ("hf", "Qwen/Qwen3-1.7B"))
        self.assertEqual(PL.parse_source("https://huggingface.co/Qwen/Qwen3-1.7B/tree/main"), ("hf", "Qwen/Qwen3-1.7B"))
        self.assertEqual(PL.parse_source("hf.co/org/name"), ("hf", "org/name"))
        self.assertEqual(PL.parse_source(TMP), ("path", TMP))
        self.assertEqual(PL.parse_source("~/x")[0], "path")
        for bad in ("", "   ", "justaword", "a/b/c", "x" * 5000, "bad\x00"):
            with self.assertRaises(PL.SourceError):
                PL.parse_source(bad)

    def test_architecture_support(self):
        self.assertEqual(PL.arch_support("Qwen3ForCausalLM")["level"], "supported")
        self.assertEqual(PL.arch_support("qwen3moe")["level"], "supported")
        self.assertEqual(PL.arch_support("Gemma4ForConditionalGeneration")["level"], "supported")
        self.assertEqual(PL.arch_support("Glm4MoeForCausalLM")["level"], "beta")
        self.assertEqual(PL.arch_support("LlamaForCausalLM")["level"], "untested")
        self.assertEqual(PL.arch_support("MadeUpForCausalLM", {"Qwen3ForCausalLM"})["level"], "unsupported")
        self.assertEqual(PL.arch_support("MadeUpForCausalLM", None)["level"], "untested")
        self.assertEqual(PL.arch_support(None)["level"], "unknown")

    def test_licence_classes(self):
        C = PL.classify_licence
        for lic, level in (("apache-2.0", "ok"), ("mit", "ok"), ("cc-by-4.0", "ok"), ("cc-by-sa-4.0", "conditions"), ("llama3.1", "conditions"), ("gemma", "conditions"),
                           ("qwen", "conditions"), ("cc-by-nc-4.0", "noncommercial"), ("cc-by-nd-4.0", "noderivs"), ("cc-by-nc-nd-4.0", "noderivs"),
                           ("polyform-noncommercial-1.0.0", "noderivs"), ("other", "unknown"), ("", "unknown"), (None, "unknown"), ("weird-thing", "unknown")):
            self.assertEqual(C(lic)["level"], level, lic)
        self.assertTrue(C("cc-by-nd-4.0")["forbids"])
        self.assertFalse(C("apache-2.0")["forbids"])
        self.assertIsNone(C("other")["forbids"])
        self.assertEqual(PL.licence_from_card("---\nlicense: apache-2.0\ntags: [x]\n---\n# Hi"), "apache-2.0")
        self.assertIsNone(PL.licence_from_card("# no front matter"))

    def test_kv_bytes_from_config(self):
        dense = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}
        self.assertEqual(PL.kv_bytes_per_token_config(dense), 32 * 2 * 8 * 128 * 2)
        hybrid = dict(dense, full_attention_interval=4)
        self.assertEqual(PL.kv_bytes_per_token_config(hybrid), 8 * 2 * 8 * 128 * 2)          # only 1 layer in 4 keeps a KV cache
        layered = dict(dense, layer_types=["full_attention", "linear_attention", "linear_attention", "linear_attention"] * 8)
        self.assertEqual(PL.kv_bytes_per_token_config(layered), 8 * 2 * 8 * 128 * 2)
        mla = {"num_hidden_layers": 10, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
        self.assertEqual(PL.kv_bytes_per_token_config(mla), 10 * 576 * 2)
        nested = {"text_config": dense}
        self.assertEqual(PL.kv_bytes_per_token_config(nested), PL.kv_bytes_per_token_config(dense))
        self.assertIsNone(PL.kv_bytes_per_token_config({"num_hidden_layers": 2}))
        self.assertIsNone(PL.kv_bytes_per_token_config({}))

    def test_safetensors_header_and_params(self):
        raw, n = F.safetensors_bytes(2)
        p = os.path.join(TMP, "t.safetensors")
        with open(p, "wb") as f:
            f.write(raw)
        h = PL.read_safetensors_header(p)
        total, emb, head = PL.params_from_headers(h)
        self.assertEqual((total, emb, head), (n, 128 * 64, 128 * 64))
        with open(p, "wb") as f:
            f.write(b"\xff" * 16)
        with self.assertRaises(PL.SourceError):
            PL.read_safetensors_header(p)


# =====================================================================================================================
class Inspect(unittest.TestCase):
    def setUp(self):
        self.r = Rig()

    def test_hugging_face_repo(self):
        s = self.r.svc.inspect("tiny/Tiny-Qwen3")
        self.assertEqual((s["kind"], s["arch"], s["layers"], s["gated"]), ("hf", "Qwen3ForCausalLM", 4, False))
        self.assertEqual(s["params"], REPOS["tiny/Tiny-Qwen3"]["params"])
        self.assertEqual(s["support"]["level"], "supported")
        self.assertEqual(s["licence"]["level"], "ok")
        self.assertGreater(s["download_bytes"], 100_000)
        self.assertEqual(s["kv_bytes_tok"], 4 * 2 * 2 * 16 * 2)
        self.assertEqual(self.r.svc.inspect("https://huggingface.co/tiny/Tiny-Qwen3")["id"], "tiny/Tiny-Qwen3")

    def test_no_derivatives_licence_is_flagged(self):
        s = self.r.svc.inspect("tiny/NoDerivs")
        self.assertEqual((s["licence"]["level"], s["licence"]["forbids"]), ("noderivs", True))

    def test_unknown_architecture_is_not_blocked_but_says_untested(self):
        s = self.r.svc.inspect("tiny/Odd")
        self.assertEqual(s["support"]["level"], "untested")

    def test_errors_are_one_sentence(self):
        for src, frag in (("nope/Missing", "no model called"), ("tiny/Gated", "gated"), ("justaword", "neither a Hugging Face model id")):
            with self.assertRaises(E.EncodeError) as cm:
                self.r.svc.inspect(src)
            self.assertIn(frag, str(cm.exception))
            self.assertNotIn("Traceback", str(cm.exception))

    def test_gated_with_a_token_works_and_offline_is_plain(self):
        os.environ["HF_TOKEN"] = "hf_" + "a" * 30
        try:
            self.assertEqual(self.r.svc.inspect("tiny/Gated", force=True)["id"], "tiny/Gated")
        finally:
            os.environ.pop("HF_TOKEN")
        old = os.environ["HF_ENDPOINT"]
        os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"
        try:
            with self.assertRaises(E.EncodeError) as cm:
                self.r.svc.inspect("tiny/Other")
            self.assertIn("Cannot reach Hugging Face", str(cm.exception))
        finally:
            os.environ["HF_ENDPOINT"] = old

    def test_local_folder(self):
        d = os.path.join(TMP, "localmodel")
        os.makedirs(d, exist_ok=True)
        raw, n = F.safetensors_bytes(4)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(raw)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(F.tiny_config(), f)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("---\nlicense: cc-by-nc-4.0\n---\n")
        s = self.r.svc.inspect(d)
        self.assertEqual((s["kind"], s["params"], s["download_bytes"]), ("folder", n, 0))
        self.assertEqual(s["licence"]["level"], "noncommercial")
        empty = os.path.join(TMP, "emptydir")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaises(E.EncodeError) as cm:
            self.r.svc.inspect(empty)
        self.assertIn("config.json and .safetensors", str(cm.exception))

    def test_local_gguf_q8_ok_and_already_quantized_refused(self):
        import fake_encode_tools as T
        q8 = os.path.join(TMP, "m-Q8_0.gguf")
        T.write_gguf(q8, 4, 8)
        s = self.r.svc.inspect(q8)
        self.assertEqual((s["kind"], s["gguf_kind"], s["arch"], s["layers"]), ("gguf", "q8_0", "qwen3", 4))
        self.assertEqual(s["name"], "m")
        bf = os.path.join(TMP, "m-BF16.gguf")
        T.write_gguf(bf, 4, 30, base_type=30, embed_type=30)
        self.assertEqual(self.r.svc.inspect(bf)["gguf_kind"], "bf16")
        q4 = os.path.join(TMP, "m-PXQ4.gguf")
        T.write_gguf(q4, 4, 252)
        with self.assertRaises(E.EncodeError) as cm:
            self.r.svc.inspect(q4)
        self.assertIn("already quantized", str(cm.exception))
        junk = os.path.join(TMP, "junk.gguf")
        with open(junk, "wb") as f:
            f.write(b"NOTGGUF" * 10)
        with self.assertRaises(E.EncodeError) as cm:
            self.r.svc.inspect(junk)
        self.assertIn("not a readable GGUF", str(cm.exception))

    def test_stage_lists_per_source_kind(self):
        svc = self.r.svc
        hf = {"kind": "hf"}
        self.assertEqual(svc.stages_for(hf, None, True), ["download", "convert", "reference", "skeleton", "dump", "hessians", "encode", "verify"])
        self.assertEqual(svc.stages_for({"kind": "folder"}, None, False), ["convert", "reference", "skeleton", "encode", "verify"])
        self.assertEqual(svc.stages_for({"kind": "gguf", "gguf_kind": "q8_0"}, None, False), ["skeleton", "encode", "verify"])
        self.assertEqual(svc.stages_for({"kind": "gguf", "gguf_kind": "bf16"}, None, True)[:2], ["reference", "skeleton"])

    def test_browse_lists_folders_and_ggufs_only(self):
        d = os.path.join(TMP, "br")
        os.makedirs(os.path.join(d, "sub"), exist_ok=True)
        os.makedirs(os.path.join(d, "hf"), exist_ok=True)
        open(os.path.join(d, "hf", "config.json"), "w").close()
        open(os.path.join(d, "a.gguf"), "wb").write(b"x")
        open(os.path.join(d, "notes.txt"), "w").close()
        os.makedirs(os.path.join(d, ".hidden"), exist_ok=True)
        r = self.r.svc.browse(d)
        self.assertEqual({(e["name"], e["kind"]) for e in r["entries"]}, {("sub", "dir"), ("hf", "hfdir"), ("a.gguf", "gguf")})
        self.assertEqual(r["parent"], os.path.dirname(d))
        self.assertTrue(self.r.svc.browse(os.path.join(d, "does-not-exist"))["path"])


# =====================================================================================================================
class Plan(unittest.TestCase):
    def test_free_vs_pro_through_the_service(self):
        free = Rig("free")
        p = free.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0, 1]})
        self.assertTrue(p["recommended"] and all(not k.startswith("pxqn") for k in p["recommended"]))
        self.assertEqual(len(p["locked"]), 7)
        self.assertEqual(p["encoder"]["edition"], "free")
        pro = Rig("pro")
        p = pro.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})
        self.assertTrue(p["recommended"][0].startswith("pxqn"))
        self.assertEqual(p["target"]["vram_gib"], 16.0)
        none = Rig(with_encoder=False)
        p = none.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})
        self.assertEqual(p["recommended"], [])
        self.assertIsNone(p["encoder"])

    def test_another_machine_by_memory(self):
        r = Rig("pro")
        p = r.svc.plan({"source": "tiny/Tiny-Qwen3", "vram_gib": 24})
        self.assertEqual(p["target"]["vram_gib"], 24.0)
        for bad in ({"vram_gib": "x"}, {"vram_gib": 1}, {"vram_gib": 99999}, {"cards": []}, {"cards": ["a"]}, {"cards": [7]}):
            with self.assertRaises(E.EncodeError):
                r.svc.plan(dict({"source": "tiny/Tiny-Qwen3"}, **bad))

    def test_switching_the_encoder_changes_the_tiers_without_a_restart(self):
        r = Rig("free")
        before = r.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})
        pro_cli = F.make_fake_encoder(os.path.join(r.dir, "pro2"), "pro", build_id="b-pro-9")
        st = r.svc.add_encoder(pro_cli)
        self.assertEqual(st["edition"], "pro")
        after = r.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})
        self.assertNotEqual(before["recommended"], after["recommended"])
        self.assertTrue(after["recommended"][0].startswith("pxqn"))
        st = r.svc.use(r.cli)                                                  # and back to Free
        self.assertEqual(st["edition"], "free")
        self.assertEqual(len(r.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})["locked"]), 7)


# =====================================================================================================================
class Checks(unittest.TestCase):
    def kinds(self, res):
        return {c["id"]: c["status"] for c in res["checks"]}

    def test_all_green_on_a_healthy_rig(self):
        r = Rig("pro")
        c = r.svc.checks(r.body())
        self.assertTrue(c["can_start"], c["refusal"])
        self.assertEqual(set(self.kinds(c).values()), {"ok"})
        self.assertIn("encodes left", [x for x in c["checks"] if x["id"] == "licence"][0]["text"])
        self.assertEqual([s["id"] for s in c["stages"]], ["download", "convert", "reference", "skeleton", "dump", "hessians", "encode", "verify"])
        self.assertLess(c["estimate"]["low_s"], c["estimate"]["high_s"])
        self.assertNotIn("runtime", self.kinds(c))             # only reported when the Pro library cannot load

    def test_a_pro_library_that_cannot_load_blocks_pxqn_but_not_the_classic_tiers(self):
        # an older wrapper: only the loader's own sentence. Control reads which CUDA library it names and offers the GPU runtime
        r = Rig("pro", runtime={"lib": "unloadable", "detail": "libcusolver.so.12: cannot open shared object file"})
        r.host.cfg.setdefault("encode", {})["licence_url"] = "http://127.0.0.1:9"        # hermetic: the offer's size lookup finds no server
        c = r.svc.checks(r.body("pxqn4"))
        self.assertFalse(c["can_start"])
        rt = [x for x in c["checks"] if x["id"] == "runtime"][0]
        self.assertEqual((rt["status"], rt.get("action")), ("bad", "runtime"))
        self.assertIn("NVIDIA CUDA 12 libraries", rt["text"])
        self.assertIn("libcusolver.so.12", rt["text"])
        self.assertIn("Download the GPU runtime", rt["fix"])
        self.assertTrue(r.svc.checks(r.body("pxq4"))["can_start"])               # the classic quantizer does not use the library
        self.assertEqual(r.svc.state()["encoders"][0]["runtime"]["lib"], "unloadable")
        # a library that fails for another reason keeps the generic sentence (nothing to download)
        r2 = Rig("pro", runtime={"lib": "unloadable", "detail": "undefined symbol: cusolverDnXpotrf"})
        c2 = r2.svc.checks(r2.body("pxqn4"))
        rt2 = [x for x in c2["checks"] if x["id"] == "runtime"][0]
        self.assertIn("cannot load on this machine", rt2["text"])
        self.assertIsNone(rt2.get("action"))
        self.assertFalse(r2.svc.state()["runtime"]["need"])

    def test_classic_tiers_need_no_licence_even_with_a_dead_key(self):
        r = Rig("pro", licence_check={"state": "revoked"})
        c = r.svc.checks(r.body("pxq4"))
        self.assertTrue(c["can_start"], c["refusal"])
        self.assertIn("need no licence", [x for x in c["checks"] if x["id"] == "licence"][0]["text"])
        self.assertEqual([s["id"] for s in c["stages"]], ["download", "convert", "quantize", "verify"])

    def test_universal_needs_a_tier_map_so_it_is_listed_but_not_offered(self):
        r = Rig("free", tiers=["pxq3", "pxq_universal"])
        p = r.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})
        by = {x["key"]: x for x in p["tiers"]}
        self.assertFalse(by["pxquniversal"]["available"])
        self.assertIn("tier map", by["pxquniversal"]["locked_reason"])
        self.assertNotIn("pxquniversal", p["recommended"])
        c = r.svc.checks(r.body("pxq_universal"))
        self.assertFalse(c["can_start"])
        self.assertIn("tier map", c["refusal"])

    def test_a_q8_gguf_for_a_classic_tier_warns_about_the_second_lossy_pass(self):
        import fake_encode_tools as T
        r = Rig("free", key=False)
        q8 = os.path.join(r.dir, "m-Q8_0.gguf")
        T.write_gguf(q8, 4, 8)
        c = r.svc.checks(r.body("pxq3", source=q8))
        self.assertTrue(c["can_start"], c["refusal"])
        d = [x for x in c["checks"] if x["id"] == "double"][0]
        self.assertEqual(d["status"], "warn")
        self.assertIn("second lossy pass", d["text"])
        j = r.wait(r.svc.start(r.body("pxq3", source=q8))["id"])
        self.assertEqual(j["status"], "done", j["error"])                              # both flags were passed
        self.assertEqual([s["id"] for s in j["stages"]], ["quantize", "verify"])
        bf = os.path.join(r.dir, "m-BF16.gguf")
        T.write_gguf(bf, 4, 30, base_type=30, embed_type=30)
        self.assertNotIn("double", [x["id"] for x in r.svc.checks(r.body("pxq3", source=bf))["checks"]])

    def test_free_without_a_key_is_fine(self):
        r = Rig("free", key=False)
        c = r.svc.checks(r.body("pxq3"))
        self.assertTrue(c["can_start"], c["refusal"])
        self.assertEqual([s["id"] for s in c["stages"]], ["download", "convert", "quantize", "verify"])      # the classic path: one CPU quantize
        self.assertIn("Not needed", [x for x in c["checks"] if x["id"] == "vram"][0]["text"])           # no graphics card for a classic tier

    def test_pxqn_tier_on_free_is_refused_as_a_supporter_feature(self):
        r = Rig("free")
        c = r.svc.checks(r.body("pxqn4"))
        self.assertFalse(c["can_start"])
        self.assertIn("Supporter feature", c["refusal"])
        with self.assertRaises(E.EncodeError) as cm:
            r.svc.start(r.body("pxqn4"))
        self.assertIn("Supporter feature", str(cm.exception))

    def test_licence_refusals_are_plain_sentences(self):
        for lic, frag in (({"state": "expired"}, "expired"), ({"state": "revoked"}, "revoked"), ({"state": "none"}, "does not have your key"),
                          ({"state": "valid", "encodes_left": 0}, "no encodes left"), ({"state": "exhausted"}, "no encodes left")):
            r = Rig("pro", licence_check=lic)
            c = r.svc.checks(r.body())
            self.assertFalse(c["can_start"], lic)
            self.assertIn(frag, c["refusal"])
            self.assertNotIn("Traceback", c["refusal"])
        r = Rig("pro", key=False)
        c = r.svc.checks(r.body())
        self.assertFalse(c["can_start"])
        self.assertIn("No licence key", c["refusal"])
        r = Rig("pro", licence_check={"state": "offline"})
        c = r.svc.checks(r.body())
        self.assertTrue(c["can_start"])                                         # a licence server outage is a warning; the run asks again
        self.assertEqual(self.kinds(c)["licence"], "warn")

    def test_no_encoder_is_refused_with_the_way_out(self):
        r = Rig(with_encoder=False)
        c = r.svc.checks(r.body("pxq3"))
        self.assertFalse(c["can_start"])
        self.assertIn("No PXA Quantizer is installed", c["refusal"])
        self.assertIn("Get the encoder", [x for x in c["checks"] if x["id"] == "encoder"][0]["fix"])

    def test_disk_refusal(self):
        r = Rig("pro")
        real = E._free_bytes
        E._free_bytes = lambda p: 50_000
        try:
            c = r.svc.checks(r.body())
        finally:
            E._free_bytes = real
        self.assertFalse(c["can_start"])
        d = [x for x in c["checks"] if x["id"] == "disk"][0]
        self.assertEqual(d["status"], "bad")
        self.assertRegex(d["text"], r"Needs about .* free on")
        self.assertIn("work folder", d["fix"])

    def test_disk_on_two_filesystems_names_the_short_one(self):
        r = Rig("pro")
        real_free, real_dev = E._free_bytes, E._dev
        E._free_bytes = lambda p: 10 if p.endswith("models") else 10 ** 12
        E._dev = lambda p: hash(p) % 1000
        try:
            c = r.svc.checks(r.body())
        finally:
            E._free_bytes, E._dev = real_free, real_dev
        d = [x for x in c["checks"] if x["id"] == "disk"][0]
        self.assertEqual(d["status"], "bad")
        self.assertIn("output folder needs", d["text"])

    def test_ram_check(self):
        r = Rig("pro")
        real = E._meminfo_available
        E._meminfo_available = lambda: 1000
        try:
            c = r.svc.checks(r.body())
        finally:
            E._meminfo_available = real
        ram = [x for x in c["checks"] if x["id"] == "ram"][0]
        self.assertEqual(ram["status"], "bad")
        self.assertFalse(c["can_start"])
        E._meminfo_available = lambda: None
        try:
            self.assertNotIn("ram", self.kinds(r.svc.checks(r.body())))        # unreadable: not invented
        finally:
            E._meminfo_available = real

    def test_vram_check_uses_the_free_memory_of_the_encode_card(self):
        rows = [(0, "Tesla V100", 70, 16384, 16000, "u0"), (1, "Tesla V100", 70, 16384, 100, "u1")]
        r = Rig("pro", gpus=rows)
        c = r.svc.checks(r.body(cards=[0, 1]))                                  # auto: card 1 (the freer one)
        v = [x for x in c["checks"] if x["id"] == "vram"][0]
        self.assertEqual(c["encode_card"], 1)
        self.assertEqual(v["status"], "ok")
        busy = Rig("pro", gpus=[(0, "Tesla V100", 70, 16384, 16384, "u0")])
        c = busy.svc.checks(busy.body(cards=[0]))
        self.assertEqual([x for x in c["checks"] if x["id"] == "vram"][0]["status"], "bad")
        self.assertFalse(c["can_start"])
        nogpu = Rig("pro", gpus=[])
        c = nogpu.svc.checks(nogpu.body(cards=[0]))
        self.assertIn("No graphics card", [x for x in c["checks"] if x["id"] == "vram"][0]["text"])

    def test_missing_tools_are_named(self):
        if not legacy():
            return          # (the one-command flow: test_make_steps_that_cannot_run_are_named)
        r = Rig("pro")
        r.host._engines = []
        real_path, real_here = os.environ["PATH"], E.HERE
        os.environ["PATH"] = os.path.join(TMP, "nowhere") + os.pathsep + "/usr/bin" + os.pathsep + "/bin"
        E.HERE = os.path.join(TMP, "no-source-tree", "tools")             # a source checkout would find its own converter
        try:
            c = r.svc.checks(r.body())
        finally:
            os.environ["PATH"] = real_path
            E.HERE = real_here
        t = [x for x in c["checks"] if x["id"] == "tools"][0]
        self.assertEqual(t["status"], "bad")
        for word in ("converter", "llama-quantize", "llama-imatrix"):
            self.assertIn(word, t["text"])
        self.assertFalse(c["can_start"])

    def test_no_derivatives_needs_an_explicit_ack(self):
        r = Rig("pro")
        c = r.svc.checks(r.body(source="tiny/NoDerivs"))
        self.assertFalse(c["can_start"])
        self.assertIn("does not let you share", c["refusal"])
        self.assertTrue(r.svc.checks(r.body(source="tiny/NoDerivs", licence_ack=True))["can_start"])
        with self.assertRaises(E.EncodeError):
            r.svc.start(r.body(source="tiny/NoDerivs"))

    def test_unknown_tier_and_bad_inputs(self):
        r = Rig("pro")
        for kw in ({"tier": ""}, {"tier": "pxqn-nonsense"}, {"encode_card": "x"}, {"encode_card": 9}):
            with self.assertRaises(E.EncodeError):
                r.svc.checks(r.body(**kw))


# =====================================================================================================================
class Pipeline(unittest.TestCase):
    def test_a_whole_encode_with_ldlq(self):
        r = Rig("pro")
        v = r.svc.start(r.body())
        j = r.wait(v["id"])
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual([s["status"] for s in j["stages"]], ["done"] * 8)
        res = j["result"]
        self.assertTrue(res["path"].endswith("Tiny-Qwen3-PXQN4.gguf"))
        self.assertTrue(os.path.isfile(res["path"]))
        import hashlib
        with open(res["path"], "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), res["sha256"])
        self.assertEqual((res["cls"], res["tier"]), ("6-bit class", "PXQN4"))
        self.assertEqual(j["counts"]["encoded"], j["counts"]["exact"])
        if legacy():
            self.assertGreater(j["counts"]["encoded"], 0)
        else:
            self.assertGreater(res["tier_tensors"], 0)          # `make` verifies the tensors itself: Control reports what is in the file, not a count it never saw
            self.assertEqual(j["params"]["flow"] if "flow" in j["params"] else "make", "make")
        self.assertEqual(j["pct"], 1.0)
        # the output really is the tier asked for, and the work folder is cleaned up
        h = L.gguf_header(res["path"])
        self.assertIn(259, {t[1] for t in h["tensors"]})
        self.assertEqual(os.listdir(os.path.join(r.dir, "work")), [])
        # one licence job, charged once
        self.assertEqual(r.state_file()["jobs"], 1)
        self.assertEqual(j["licence_jid"], "J1")

    def test_free_encode_without_hessians(self):
        r = Rig("free", key=False)
        j = r.wait(r.svc.start(r.body("pxq3"))["id"])
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual([s["id"] for s in j["stages"]], ["download", "convert", "quantize", "verify"])
        self.assertIsNone(j["licence_jid"])
        self.assertIn(255, {t[1] for t in L.gguf_header(j["result"]["path"])["tensors"]})
        self.assertEqual((j["result"]["tier_tensors"], list(j["result"]["types"])), (20, ["PXQ3"]))     # 4 layers x 5 tensors in the tier asked for (attn_k and attn_v stay 8 bit)

    def test_from_a_local_q8_gguf_skips_download_convert_and_reference(self):
        import fake_encode_tools as T
        r = Rig("pro")
        q8 = os.path.join(r.dir, "src-Q8_0.gguf")
        T.write_gguf(q8, 4, 8)
        j = r.wait(r.svc.start(r.body(source=q8))["id"])
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual([s["id"] for s in j["stages"]], ["skeleton", "dump", "hessians", "encode", "verify"])
        self.assertTrue(os.path.isfile(q8))                          # the user's own file is never cleaned up

    def test_from_a_local_folder_never_deletes_the_folder(self):
        r = Rig("pro")
        d = os.path.join(r.dir, "srcdir")
        os.makedirs(d)
        raw, _ = F.safetensors_bytes(4)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(raw)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(F.tiny_config(), f)
        j = r.wait(r.svc.start(r.body(source=d))["id"])
        self.assertEqual(j["status"], "done", j["error"])
        self.assertTrue(os.path.isfile(os.path.join(d, "model.safetensors")))
        self.assertEqual(j["stages"][0]["id"], "convert")

    def test_keep_intermediates_leaves_the_work_folder(self):
        r = Rig("pro")
        j = r.wait(r.svc.start(r.body(keep=True))["id"])
        self.assertEqual(j["status"], "done")
        left = os.listdir(os.path.join(r.dir, "work", j["id"]))
        self.assertIn("q8.gguf" if legacy() else "model-q8_0.gguf", left)
        self.assertIn("hess", left)

    def test_second_job_cannot_start_while_one_runs(self):
        r = Rig("pro")
        r.pace(0.1)
        j1 = r.svc.start(r.body())
        with self.assertRaises(E.EncodeError) as cm:
            r.svc.start(r.body(tier="pxqn3"))
        self.assertIn("already running", str(cm.exception))
        r.svc.cancel(j1["id"])
        r.wait(j1["id"])

    def test_output_never_overwrites_an_existing_file(self):
        r = Rig("free", key=False)
        a = r.wait(r.svc.start(r.body("pxq4"))["id"])
        b = r.wait(r.svc.start(r.body("pxq4"))["id"])
        self.assertNotEqual(a["result"]["path"], b["result"]["path"])
        self.assertTrue(b["result"]["path"].endswith("-2.gguf"))

    def test_progress_is_reported_per_stage_with_detail(self):
        r = Rig("pro")
        r.pace(0.05)
        jid = r.svc.start(r.body())["id"]
        j = r.wait_stage(jid, "encode")
        enc = next(s for s in j["stages"] if s["id"] == "encode")
        self.assertEqual(next(s for s in j["stages"] if s["id"] == "hessians")["status"], "done")
        self.assertGreater(j["pct"], 0.3)
        self.assertLess(j["pct"], 1.0)
        r.svc.cancel(jid)
        r.wait(jid)

    def test_missing_converter_is_a_sentence(self):
        r = Rig("pro")
        if legacy():
            os.unlink(os.path.join(r.engine, "convert_hf_to_gguf.py"))
        else:           # the converter is inside the package: `info --json` says the convert stage cannot run, with the reason
            r.set_fake(not_ready={"convert": "this package carries no converter (convert/convert_hf_to_gguf.py)"})
            r.svc.rescan()
        real_here = E.HERE
        E.HERE = os.path.join(TMP, "no-source-tree", "tools")
        try:
            c = r.svc.checks(r.body())
        finally:
            E.HERE = real_here
        self.assertFalse(c["can_start"])
        self.assertIn("converter", c["refusal"])


class Control(unittest.TestCase):
    def test_pause_resume_cancel(self):
        r = Rig("pro")
        r.pace(0.12)
        jid = r.svc.start(r.body())["id"]
        r.wait_stage(jid, "reference")
        r.svc.pause(jid)
        self.assertEqual(r.svc.job_view(jid)["status"], "paused")
        a = r.svc.job_view(jid)["pct"]
        time.sleep(1.2)
        self.assertEqual(r.svc.job_view(jid)["pct"], a)                  # the process is stopped: no progress
        r.svc.resume(jid)
        time.sleep(0.8)
        self.assertGreater(r.svc.job_view(jid)["pct"], a)
        r.svc.cancel(jid)
        j = r.wait(jid, ("cancelled",))
        self.assertIn("Cancelled", j["error"]["message"])
        self.assertTrue(all(s["status"] != "running" for s in j["stages"]))
        for fn in (r.svc.pause, r.svc.cancel):
            with self.assertRaises(E.EncodeError):
                fn(jid)
        # a cancelled job resumes where it stopped and finishes
        r.pace(0.0)
        r.svc.resume(jid)
        self.assertEqual(r.wait(jid)["status"], "done")

    def test_cancel_while_paused(self):
        r = Rig("pro")
        r.pace(0.2)
        jid = r.svc.start(r.body())["id"]
        r.wait_stage(jid, "convert")
        r.svc.pause(jid)
        r.svc.cancel(jid)
        self.assertEqual(r.wait(jid, ("cancelled",), 30)["status"], "cancelled")

    def test_discard_removes_the_work_and_the_record(self):
        r = Rig("pro")
        r.pace(0.2)
        jid = r.svc.start(r.body())["id"]
        r.wait_stage(jid, "reference")
        with self.assertRaises(E.EncodeError):
            r.svc.discard(jid)                                              # running: cancel first
        r.svc.cancel(jid)
        r.wait(jid, ("cancelled",))
        r.svc.discard(jid)
        self.assertEqual(r.svc.job_list(), [])
        self.assertFalse(os.path.exists(os.path.join(r.dir, "work", jid)))
        self.assertFalse(os.path.exists(os.path.join(r.svc.jobs_dir(), jid)))
        with self.assertRaises(E.EncodeError):
            r.svc.job_view(jid)


class ResumeState(unittest.TestCase):
    def test_a_job_running_when_control_died_comes_back_interrupted_and_resumes_without_redoing_finished_stages(self):
        r = Rig("pro")
        r.pace(0.1)
        jid = r.svc.start(r.body())["id"]
        r.wait_stage(jid, "hessians")
        t0 = time.time()
        while r.svc.job_view(jid)["licence_jid"] is None and time.time() - t0 < 20:       # the licence job exists once the encoder said so
            time.sleep(0.02)
        self.assertEqual(r.svc.job_view(jid)["licence_jid"], "J1")
        # a reboot: nothing is written on the way out, every process dies
        r.svc._closed = True
        p = r.svc.rt[jid].proc
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        time.sleep(0.4)
        on_disk = json.load(open(os.path.join(r.svc.jobs_dir(), jid, "job.json")))
        self.assertIn(on_disk["status"], ("running", "paused"))             # what the dead process left behind
        r2 = E.EncodeService(r.host, data=os.path.join(r.dir, "data"))      # Control starts again
        j = r2.job_view(jid)
        self.assertEqual(j["status"], "interrupted")
        self.assertIn("closed", j["error"]["message"])
        self.assertEqual({s["id"]: s["status"] for s in j["stages"]}["convert"], "done")
        self.assertEqual({s["id"]: s["status"] for s in j["stages"]}["hessians"], "pending")      # the interrupted stage restarts
        before_log = len([ln for ln in r2.log_view(jid)["lines"] if ln.startswith("$ ") and "convert_hf_to_gguf" in ln])
        r2.host.cfg["encode"]["extra"] = [r.cli]
        F.set_engine_delay(r.engine, 0)
        r.set_fake(delay=0)
        r2.resume(jid)
        t0 = time.time()
        while r2.job_view(jid)["status"] not in ("done", "failed") and time.time() - t0 < 90:
            time.sleep(0.1)
        j = r2.job_view(jid)
        self.assertEqual(j["status"], "done", j["error"])
        log = r2.log_view(jid)["lines"]
        self.assertEqual(len([ln for ln in log if ln.startswith("$ ") and "convert_hf_to_gguf" in ln]), before_log)      # finished stages were not redone
        st = r.state_file()
        self.assertEqual(st["jobs"], 1)                                      # still ONE licence charge
        self.assertEqual(st.get("resumes"), ["J1"])                          # the second run resumed the same licence job
        if not legacy():
            starts = [ln for ln in log if ln.startswith("$ ")]
            self.assertEqual(len(starts), 2)
            self.assertEqual(starts[0], starts[1])                           # "resume" is the SAME command again

    def test_crash_in_the_middle_of_the_encode_then_resume_keeps_the_finished_tensors(self):
        r = Rig("pro", crash_after=5)
        jid = r.svc.start(r.body())["id"]
        j = r.wait(jid)
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["licence_jid"], "J1")
        if legacy():
            self.assertTrue(os.path.isfile(os.path.join(r.dir, "work", jid, "hess.done")))     # Hessians survived
        else:
            self.assertIn("hess", json.load(open(os.path.join(r.dir, "work", jid, "state.json")))["done"])
        r.svc.resume(jid)
        j = r.wait(jid)
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual(r.state_file()["jobs"], 1)
        self.assertEqual(r.state_file()["resumes"], ["J1"])
        if legacy():
            self.assertGreater(j["counts"]["encoded"], 0)

    def test_orphan_process_is_stopped_before_the_stage_restarts(self):
        r = Rig("pro")
        r.pace(0.15)
        jid = r.svc.start(r.body())["id"]
        r.wait_stage(jid, "reference")
        pid = r.svc.job_view(jid)["pid"]
        r.svc._closed = True                                                 # Control "crashes": the child lives on, orphaned
        r2 = E.EncodeService(r.host, data=os.path.join(r.dir, "data"))
        self.assertEqual(r2.job_view(jid)["status"], "interrupted")
        self.assertTrue(E._pid_alive(pid))
        r2.host.cfg["encode"]["extra"] = [r.cli]
        F.set_engine_delay(r.engine, 0)
        r2.resume(jid)
        t0 = time.time()
        while time.time() - t0 < 5 and E._pid_alive(pid):
            time.sleep(0.1)
        self.assertFalse(E._pid_alive(pid))
        while r2.job_view(jid)["status"] not in ("done", "failed") and time.time() - t0 < 90:
            time.sleep(0.1)
        self.assertEqual(r2.job_view(jid)["status"], "done")

    def test_shutdown_makes_live_jobs_resumable_and_stops_their_processes(self):
        r = Rig("pro")
        r.pace(0.2)
        jid = r.svc.start(r.body())["id"]
        r.wait_stage(jid, "reference")
        p = r.wait_proc(jid)
        r.svc.shutdown()
        time.sleep(0.3)
        self.assertIsNotNone(p.poll())
        r2 = E.EncodeService(r.host, data=os.path.join(r.dir, "data"))
        self.assertEqual(r2.job_view(jid)["status"], "interrupted")

    def test_job_ids_on_disk_are_validated(self):
        r = Rig("pro")
        bad = os.path.join(r.svc.jobs_dir(), "../../evil")
        os.makedirs(os.path.join(r.svc.jobs_dir(), "not-a-job-id"), exist_ok=True)
        json.dump({"id": "not-a-job-id", "status": "running", "stages": [], "params": {"work": "/"}}, open(os.path.join(r.svc.jobs_dir(), "not-a-job-id", "job.json"), "w"))
        r2 = E.EncodeService(r.host, data=os.path.join(r.dir, "data"))
        self.assertEqual(r2.job_list(), [])


class LicenceRefusalsAtRunTime(unittest.TestCase):
    def run_with(self, **fake):
        r = Rig("pro", **fake)
        j = r.wait(r.svc.start(r.body())["id"])
        return r, j

    def test_quota_revoked_offline_and_oom_are_plain_sentences(self):
        for mode, code, frag in (("402", "no_quota", "no encodes left"), ("403", "revoked", "revoked, expired or suspended"), ("offline", "offline", "Cannot reach the licence server"),
                                 ("oom", "oom", "ran out of memory")):
            r, j = self.run_with(fail=mode)
            self.assertEqual(j["status"], "failed", mode)
            self.assertEqual(j["error"]["code"], code)
            self.assertIn(frag, j["error"]["message"])
            self.assertTrue(j["error"]["hint"])
            self.assertNotIn("Traceback", json.dumps(j["error"]))
            if legacy():
                self.assertEqual(next(s for s in j["stages"] if s["id"] == "hessians")["status"], "failed")
            elif mode == "oom":
                self.assertEqual(next(s for s in j["stages"] if s["id"] == "encode")["status"], "failed")
            else:           # a refusal comes before anything starts: no stage ran, nothing was written
                self.assertEqual({s["status"] for s in j["stages"]}, {"pending"})
                self.assertFalse(os.path.exists(os.path.join(r.dir, "work", j["id"], "state.json")))
            if mode == "offline":
                r.set_fake(fail=None)
                r.svc.resume(j["id"])
                self.assertEqual(r.wait(j["id"])["status"], "done")

    def test_a_key_the_encoder_prints_is_scrubbed_from_the_log_and_the_page(self):
        r, j = self.run_with(leak_key=True)
        self.assertEqual(j["status"], "done")
        log = "\n".join(r.svc.log_view(j["id"])["lines"])
        self.assertNotIn(SECRET, log)
        self.assertIn("<key>", log)
        with open(os.path.join(r.svc.jobs_dir(), j["id"], "log.txt")) as f:
            self.assertNotIn(SECRET, f.read())
        self.assertNotIn(SECRET, json.dumps(r.svc.job_view(j["id"])))
        self.assertNotIn(SECRET, json.dumps(r.svc.state()))


class Download(unittest.TestCase):
    """Control's own downloader: the multi-command path (an encoder with CLI version 1). With `pxqe make` the encoder downloads."""
    def setUp(self):
        DEFAULT_CLI[0] = "1"

    def tearDown(self):
        DEFAULT_CLI[0] = None

    def test_resume_a_partial_download_with_range(self):
        r = Rig("free", key=False)
        r.pace(0.0)
        HF.requests.clear()
        jid = r.svc.start(r.body("pxq3", source="tiny/Slow"))["id"]
        r.wait_stage(jid, "download")
        time.sleep(0.25)
        r.svc.cancel(jid)
        r.wait(jid, ("cancelled",))
        parts = [f for f in os.listdir(os.path.join(r.dir, "work", jid, "src")) if f.endswith(".part")]
        self.assertTrue(parts, "a partial file is kept so a resume continues it")
        REPOS["tiny/Slow"]["delay"] = 0.0
        try:
            r.svc.resume(jid)
            j = r.wait(jid)
        finally:
            REPOS["tiny/Slow"]["delay"] = 0.02
        self.assertEqual(j["status"], "done", j["error"])
        self.assertTrue(any((q["headers"].get("range") or "").startswith("bytes=") for q in HF.requests if "/resolve/" in q["path"]))

    def test_the_hugging_face_token_stays_on_hugging_face(self):
        os.environ["HF_TOKEN"] = "hf_" + "b" * 30
        try:
            r = Rig("free", key=False)
            HF.requests.clear()
            CDN.requests.clear()
            j = r.wait(r.svc.start(r.body("pxq3", source="tiny/Redirect"))["id"])
            self.assertEqual(j["status"], "done", j["error"])
        finally:
            os.environ.pop("HF_TOKEN")
        self.assertTrue(any("authorization" in q["headers"] for q in HF.requests))
        self.assertTrue(CDN.requests)
        self.assertFalse(any("authorization" in q["headers"] for q in CDN.requests), "the token followed the redirect to the CDN")

    def test_offline_download_is_a_sentence_and_resumable(self):
        r = Rig("free", key=False)
        real = os.environ["HF_ENDPOINT"]
        v = r.svc.start(r.body("pxq3"))
        r.wait(v["id"])                                                       # the model was cached by inspect; the run succeeds
        os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"
        try:
            r2 = Rig("free", key=False)
            r2.svc._insp[("hf:tiny/Tiny-Qwen3")] = (time.time(), r.svc.inspect("tiny/Tiny-Qwen3"))
            j = r2.wait(r2.svc.start(r2.body("pxq3"))["id"])
        finally:
            os.environ["HF_ENDPOINT"] = real
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["error"]["code"], "offline")
        self.assertIn("cannot reach Hugging Face", j["error"]["message"])


class _Legacy(object):
    """Run the same job tests against an older encoder (CLI version 1): Control keeps the multi-command path for it."""
    def setUp(self):
        DEFAULT_CLI[0] = "1"

    def tearDown(self):
        DEFAULT_CLI[0] = None


class PipelineLegacy(_Legacy, Pipeline):
    pass


class ControlLegacy(_Legacy, Control):
    pass


class ResumeStateLegacy(_Legacy, ResumeState):
    pass


class LicenceRefusalsAtRunTimeLegacy(_Legacy, LicenceRefusalsAtRunTime):
    pass


# =====================================================================================================================
class AdapterMake(unittest.TestCase):
    """`pxqe make` (CLI version 2): what info --json says, the argv, the `@pxqe ` lines, the exit codes."""
    REAL_INFO = {"edition": "pro", "version": "pro-pxqe20261004c", "build_id": "pxqe20261004c", "cli": "2", "platform": "linux-x86_64", "cuda_major": 12,
                 "tiers": ["pxq4", "pxqn4"], "features": ["classic-quantizer", "make", "skeleton", "dump", "ldlq", "hessian", "watermark", "licence"],
                 "runtime": {"lib": "loadable"},
                 "licence": {"state": "unchecked", "user": "Ann", "key_id": "K-ABCDEF01", "encodes_left": None, "unlimited": None, "expires": None, "checked": False},
                 "stages": [{"name": "fetch", "does": "download the model", "edition": "both", "ready": True, "reason": ""},
                            {"name": "convert", "does": "convert it to GGUF", "edition": "both", "ready": False, "reason": "Python packages missing: torch (pip install torch)"},
                            {"name": "dump", "does": "run the calibration text", "edition": "pro", "ready": True, "reason": "", "tool": "the bundled CPU tool"},
                            {"name": "verify-file", "does": "check a finished file", "edition": "pro", "ready": True, "reason": ""}, "junk", {"name": 5}],
                 "make": {"progress_prefix": "@pxqe ", "sources": ["hf-repo", "hf-dir", "gguf"], "classic_tiers": ["pxq1", "pxq4"],
                          "exit_codes": {"0": "done", "1": "a stage failed, run again to resume", "130": "stopped"}}}

    def test_info_reports_the_cli_version_the_stages_and_make(self):
        i = AD.parse_info(json.dumps(self.REAL_INFO))
        self.assertEqual(i["cli"], 2)
        self.assertTrue(AD.supports_make(i))
        self.assertEqual([x["name"] for x in i["stages"]], ["fetch", "convert", "dump", "verify-file"])      # junk entries are dropped, hyphens are fine
        self.assertEqual(AD.stages_not_ready(i, ["fetch", "convert", "dump", "skeleton"]), [("convert", "Python packages missing: torch (pip install torch)")])
        self.assertEqual(i["make"]["prefix"], "@pxqe ")
        self.assertEqual(i["make"]["exit_codes"]["130"], "stopped")

    def test_an_encoder_below_cli_2_keeps_the_old_path(self):
        for cli in ("1", 1, None, "", "x", "0"):
            o = dict(self.REAL_INFO)
            if cli is None:
                o.pop("cli")
                o.pop("make")
                o.pop("stages")
            else:
                o["cli"] = cli
            self.assertFalse(AD.supports_make(AD.parse_info(json.dumps(o))), cli)
        o = dict(self.REAL_INFO, make={"error": "ImportError"})        # its `make` could not even load: do not use it
        self.assertFalse(AD.supports_make(AD.parse_info(json.dumps(o))))
        self.assertTrue(AD.supports_make(AD.parse_info(json.dumps(dict(self.REAL_INFO, cli=3)))))
        self.assertFalse(AD.supports_make(None))

    def test_argv_and_environment(self):
        pro = {"edition": "pro"}
        a = AD.argv_make("/e/pxqe", "Qwen/Qwen3-1.7B", "pxqn4", "/o/m.gguf", "/w/j1", device=0, python="/usr/bin/python3", engine="/eng", gpu_layers=12, threads=7,
                         keep_work=True, quantizer_args=AD.REQUANTIZE_ARGS)
        self.assertEqual(a[:1], ["/e/pxqe"])
        self.assertEqual(a[1:8], ["make", "Qwen/Qwen3-1.7B", "--tier", "pxqn4", "--out", "/o/m.gguf", "--work"])
        for frag in (["--device", "0"], ["--python", "/usr/bin/python3"], ["--engine", "/eng", "--gpu-layers", "12"], ["--threads", "7"]):
            i = a.index(frag[0])
            self.assertEqual(a[i:i + len(frag)], frag)
        self.assertIn("--keep-work", a)
        self.assertIn("--quantizer-arg=--allow-requantize", a)
        self.assertNotIn("--gpu-layers", AD.argv_make("/e/pxqe", "x/y", "pxq4", "/o", "/w", engine=None, gpu_layers=3))        # only with an engine tool
        # a .py wrapper runs under this interpreter
        self.assertEqual(AD.argv_make("/e/pxqe.py", "x/y", "pxq4", "/o", "/w")[:2], [sys.executable, "/e/pxqe.py"])
        # the key: environment only, only for a PXQN tier on Pro; never argv
        env = AD.make_env(pro, "pxqn4", SECRET + "xxxxxxxx", "https://lic.example", "/py")
        self.assertEqual(env, {"PXQE_KEY": SECRET + "xxxxxxxx", "PXQE_SERVER": "https://lic.example", "PXQE_PYTHON": "/py"})
        self.assertNotIn(SECRET, " ".join(a))
        self.assertNotIn("PXQE_KEY", AD.make_env(pro, "pxq4", "k" * 20, "https://x"))                # the classic tiers need no licence
        self.assertNotIn("PXQE_KEY", AD.make_env({"edition": "free"}, "pxqn4", "k" * 20, "https://x"))
        self.assertEqual(AD.make_tier("PXQ_UNIVERSAL"), "pxq_universal")
        self.assertEqual(AD.make_tier("pxqn3s8"), "pxqn3s8")

    def test_the_progress_lines_the_real_encoder_prints(self):
        P = AD.parse_make_line
        plan = P('@pxqe {"event":"plan","out":"/o/x.gguf","source":"Qwen/Qwen3-1.7B","source_kind":"hf-repo","stages":[{"does":"download the model","name":"fetch","weight":8},'
                 '{"does":"build the per-layer statistics","name":"hess","weight":10},{"does":"?","name":"splice","weight":1}],"tier":"pxqn4","work":"/w"}')
        self.assertEqual([x["name"] for x in plan["stages"]], ["fetch", "hess"])          # a stage Control has no bar for is ignored
        self.assertEqual((plan["tier"], plan["out"], plan["work"], plan["source_kind"]), ("pxqn4", "/o/x.gguf", "/w", "hf-repo"))
        e = P('@pxqe {"eta_s":61,"event":"stage","message":"encoding","overall_percent":67.9,"percent":32.1,"stage":"encode","state":"progress"}')
        self.assertEqual((e["stage"], e["state"], e["eta_s"], e["message"]), ("encode", "progress", 61, "encoding"))
        self.assertAlmostEqual(e["percent"], 0.321)
        self.assertAlmostEqual(e["overall"], 0.679)
        self.assertIsNone(P('@pxqe {"eta_s":null,"event":"stage","message":"x","overall_percent":0.0,"percent":0.0,"stage":"fetch","state":"start"}')["eta_s"])
        self.assertEqual(P('@pxqe {"eta_s":0,"event":"stage","message":"already done","overall_percent":8.2,"percent":100.0,"stage":"fetch","state":"skipped"}')["state"], "skipped")
        d = P('@pxqe {"event":"done","out":"/o/x.gguf","sha256":"%s","size":1404418688,"tier":"pxqn4"}' % ("1c4be31c" * 8))
        self.assertEqual((d["size"], d["already"], d["sha256"][:8]), (1404418688, False, "1c4be31c"))
        self.assertTrue(P('@pxqe {"event":"done","out":"/o","sha256":"%s","size":5,"tier":"x","already":true}' % ("a" * 64))["already"])
        er = P('@pxqe {"code":"encoder","event":"error","message":"CUDA runtime error 100","resumable":true,"stage":"hess"}')
        self.assertEqual((er["code"], er["stage"], er["resumable"]), ("encoder", "hess", True))
        # out-of-range and odd numbers are clamped, never trusted
        w = P('@pxqe {"event":"stage","stage":"dump","state":"progress","percent":250,"eta_s":-4,"overall_percent":"x","message":%s}' % json.dumps("m" * 900))
        self.assertEqual((w["percent"], w["eta_s"], w["overall"], len(w["message"])), (1.0, None, None, 200))

    def test_anything_else_on_the_line_is_not_a_progress_event(self):
        P = AD.parse_make_line
        for bad in ("", "fetch: download the model", "@pxqe", "@pxqe {", "@pxqe []", "@pxqe 5", '@pxqe {"event":"weather"}', '@pxqe {"event":"stage","stage":"nope","state":"start"}',
                    '@pxqe {"event":"stage","stage":"fetch","state":"sleeping"}', '@pxqe {"event":"done","out":"/o","sha256":"zz","size":5}',
                    '@pxqe {"event":"done","out":"/o","sha256":"%s","size":0}' % ("a" * 64), "x @pxqe {}", None, 7):
            self.assertIsNone(P(bad), repr(bad))
        self.assertIsNotNone(P('   @pxqe {"event":"error"}'))

    def test_exit_codes_in_plain_sentences(self):
        X = AD.explain_make_failure
        for rc in (1, 2, 3, 130):
            self.assertIn(rc, AD.MAKE_EXIT_TEXT)
        # a licence refusal reads the way the older flow reads it
        r = X(3, {"code": "refused", "message": "No encodes left: 3 of 3 used.", "resumable": False})
        self.assertEqual(r["code"], "no_quota")
        self.assertIn("no encodes left", r["message"])
        self.assertFalse(r["resumable"])
        self.assertEqual(X(3, {"code": "refused", "message": "The licence server refused this encode (HTTP 403): key revoked"})["code"], "revoked")
        nk = X(3, {"code": "no-key", "message": "There is no quantizer key. Enter your key in PXA Control (Encode > Enter key), or set PXQE_KEY and PXQE_SERVER."})
        self.assertEqual(nk["code"], "no_key")
        self.assertNotIn("Enter key", nk["message"] + nk["hint"])                 # Control's own words, not the encoder's older ones
        self.assertIn("I have a key", nk["hint"])
        self.assertEqual(X(3, None, "")["code"], "refused")
        self.assertIn("Check your key", X(3, {"code": "refused", "message": "Your quantizer key is disabled-ish."})["hint"])
        # stopped
        s = X(130, {"code": "stopped", "message": "Stopped. Run the same command again to continue.", "resumable": True})
        self.assertTrue(s["resumable"])
        self.assertIn("Resume", s["message"])
        # the encoder's own sentence plus Control's next step
        o = X(1, {"code": "offline", "message": "Cannot reach Hugging Face (x). Check the internet connection and run again.", "resumable": True})
        self.assertTrue(o["message"].startswith("Cannot reach Hugging Face"))
        self.assertIn("Resume", o["hint"])
        self.assertTrue(o["resumable"])
        self.assertIn("pip", X(2, {"code": "no-python-packages", "message": "Converting a model needs these Python packages: torch.", "resumable": False})["hint"])
        self.assertEqual(X(1, {"code": "encoder", "message": "CUDA error: out of memory", "resumable": True})["code"], "oom")
        # a small model: the quantizer's refusal is explained, and Control does not repeat a flag the person cannot type
        c = X(1, {"code": "quantize-failed", "message": "The quantizer refused: PXQ composition assertion: target PXQ4 produced 41.3% PXQ-family bytes (floor 50%). "
                  "Add --quantizer-arg=--pxq-composition-override to write it anyway.", "resumable": False})
        self.assertEqual(c["code"], "composition")
        self.assertNotIn("--quantizer-arg", c["message"] + c["hint"])
        # died without a word: the exit code is the sentence
        d = X(1, None, "Segmentation fault")
        self.assertEqual(d["code"], "failed")
        self.assertIn("step failed", d["message"])
        self.assertTrue(d["resumable"])
        k = X(-9, None, "")
        self.assertEqual((k["code"], k["resumable"]), ("killed", True))
        self.assertIn("exit 77", X(77, None, "")["message"])
        self.assertNotIn("Traceback", json.dumps([d, c, o, r]))

    def test_licence_job_lines_of_the_pro_encoder(self):
        self.assertEqual(AD.parse_line("licence: job J-99ab0c started (2 encodes left after this one)"), {"event": "job", "jid": "J-99ab0c", "left": 2, "unlimited": False})
        self.assertEqual(AD.parse_line("licence: job J-99ab0c ok")["ok"], True)
        self.assertEqual(AD.parse_line("licence: job J-99ab0c failed (encode refunded)")["refunded"], True)
        self.assertEqual(AD.parse_line("pxqe: job J-99ee5649a972 (key K-C3A1C4F0, owner), build pxqe20261004c")["jid"], "J-99ee5649a972")


class MakeFlow(unittest.TestCase):
    """The Encode tab's job as ONE `pxqe make` call (CLI version 2)."""
    def test_one_command_with_the_key_in_the_environment_only(self):
        r = Rig("pro")
        v = r.svc.start(r.body(cards=[1], encode_card=1))
        j = r.wait(v["id"])
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual(r.svc.jobs[j["id"]]["params"]["flow"], "make")
        mk = r.state_file()["makes"]
        self.assertEqual(len(mk), 1)                                               # the whole job is ONE call
        m = mk[0]
        self.assertTrue(m["key_in_env"] and m["server_in_env"])
        self.assertFalse(m["key_in_argv"])
        self.assertEqual(m["argv"][:5], ["make", "tiny/Tiny-Qwen3", "--tier", "pxqn4", "--out"])
        self.assertEqual(m["argv"][m["argv"].index("--device") + 1], "0")
        self.assertEqual(m["cuda_visible"], "1")                                   # pinned to the encode card: --device 0 is that card
        self.assertEqual(m["argv"][m["argv"].index("--work") + 1], os.path.join(r.dir, "work", j["id"]))
        self.assertEqual(m["argv"][m["argv"].index("--out") + 1], j["result"]["path"])
        self.assertEqual(m["argv"][m["argv"].index("--engine") + 1], r.engine)    # the engine's GPU calibration tool is offered; make checks it carries the hook
        self.assertIn(m["python_env"], (sys.executable,))
        self.assertNotIn(GOOD_KEY, json.dumps(r.svc.job_view(j["id"])) + "\n".join(r.svc.log_view(j["id"])["lines"]))

    def test_the_machine_lines_drive_the_bars_and_stay_out_of_the_log(self):
        r = Rig("pro")
        r.pace(0.03)
        jid = r.svc.start(r.body())["id"]
        j = r.wait_stage(jid, "encode")
        by = {s["id"]: s for s in j["stages"]}
        self.assertEqual([by[k]["status"] for k in ("download", "convert", "reference", "skeleton", "dump", "hessians")], ["done"] * 6)
        self.assertGreater(j["pct"], 0.3)
        self.assertLess(j["pct"], 1.0)
        self.assertTrue(by["encode"]["detail"])                                      # the encoder's message is the detail line
        log = r.svc.log_view(jid)["lines"]
        self.assertFalse([ln for ln in log if ln.startswith("@pxqe")], "progress lines are for the bars, not the log")
        self.assertTrue([ln for ln in log if ln.startswith("pxqe make:")])           # the human lines are
        r.svc.cancel(jid)
        r.wait(jid)

    def test_overall_percent_and_eta_come_from_the_encoder(self):
        r = Rig("pro")
        r.pace(0.0)
        jid = r.svc.start(r.body())["id"]
        j = r.wait(jid)
        self.assertEqual((j["status"], j["pct"], j["eta_s"]), ("done", 1.0, None))
        # the headline percentage is `make`'s own; the stage's own ETA replaces the extrapolation
        job = {"id": "x", "status": "running", "params": {"flow": "make"}, "make_overall": 0.42,
               "stages": [{"id": "dump", "status": "running", "pct": 0.5, "weight": 100.0, "started": time.time() - 50, "tool_eta": 7},
                          {"id": "encode", "status": "pending", "pct": 0.0, "weight": 30.0}]}
        pct, eta = r.svc._overall(job)
        self.assertAlmostEqual(pct, 0.42)
        self.assertEqual(eta, 37)                                                    # 7 s for the stage it is in + 30 s planned for the next
        job["stages"][0].pop("tool_eta")
        self.assertEqual(r.svc._overall(job)[1], 80)                                  # no estimate from the encoder: 50 s done at 50% = 50 more, + 30

    def test_plan_from_the_encoder_is_what_the_page_shows(self):
        import fake_encode_tools as T
        r = Rig("pro")
        q8 = os.path.join(r.dir, "src-Q8_0.gguf")
        T.write_gguf(q8, 4, 8)
        v = r.svc.start(r.body(source=q8))
        self.assertEqual([s["id"] for s in v["stages"]], ["skeleton", "dump", "hessians", "encode", "verify"])
        j = r.wait(v["id"])
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual([s["status"] for s in j["stages"]], ["done"] * 5)

    def test_classic_tier_on_pro_needs_no_key_and_no_card(self):
        r = Rig("pro")
        j = r.wait(r.svc.start(r.body("pxq3", cards=[0]))["id"])
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual([s["id"] for s in j["stages"]], ["download", "convert", "quantize", "verify"])
        m = r.state_file()["makes"][0]
        self.assertFalse(m["key_in_env"])                                            # no licence for the classic tiers, so no key is handed over
        self.assertNotIn("--device", m["argv"])
        self.assertNotIn("--engine", m["argv"])
        self.assertIsNone(j["licence_jid"])

    def test_a_q8_source_for_a_classic_tier_passes_both_requantize_flags(self):
        import fake_encode_tools as T
        r = Rig("free", key=False)
        q8 = os.path.join(r.dir, "m-Q8_0.gguf")
        T.write_gguf(q8, 4, 8)
        j = r.wait(r.svc.start(r.body("pxq3", source=q8))["id"])
        self.assertEqual(j["status"], "done", j["error"])
        argv = r.state_file()["makes"][0]["argv"]
        self.assertIn("--quantizer-arg=--allow-requantize", argv)
        self.assertIn("--quantizer-arg=--i-know-this-is-double-lossy", argv)

    def test_the_same_command_again_is_the_resume(self):
        r = Rig("pro", crash_after=5)
        jid = r.svc.start(r.body())["id"]
        self.assertEqual(r.wait(jid)["status"], "failed")
        r.svc.resume(jid)
        self.assertEqual(r.wait(jid)["status"], "done")
        mk = r.state_file()["makes"]
        self.assertEqual(len(mk), 2)
        self.assertEqual(mk[0]["argv"], mk[1]["argv"])                               # nothing about the command changes between the runs
        j = r.svc.job_view(jid)
        self.assertEqual(r.state_file()["jobs"], 1)                                  # ONE charge across the crash
        self.assertEqual([s["status"] for s in j["stages"]], ["done"] * 8)          # what an earlier run finished reads as done too

    def test_stop_comes_back_as_cancelled_and_resumes(self):
        r = Rig("pro")
        r.pace(0.1)
        jid = r.svc.start(r.body())["id"]
        r.wait_stage(jid, "convert")
        r.svc.cancel(jid)                                                            # SIGINT: make stops itself cleanly (exit 130) and keeps its state
        j = r.wait(jid, ("cancelled",))
        self.assertIn("Cancelled", j["error"]["message"])
        r.pace(0.0)
        r.svc.resume(jid)
        self.assertEqual(r.wait(jid)["status"], "done")

    def test_an_encoder_that_dies_without_a_word_is_a_sentence(self):
        r = Rig("pro")
        # SIGKILL in the middle of the encode (crash_after): no error line, exit code -9
        r.set_fake(crash_after=3)
        j = r.wait(r.svc.start(r.body())["id"])
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["error"]["code"], "killed")
        self.assertIn("stopped from outside", j["error"]["message"])
        self.assertIn("Resume", j["error"]["message"])
        self.assertEqual(next(s for s in j["stages"] if s["id"] == "encode")["status"], "failed")

    def test_error_lines_become_the_failure(self):
        r = Rig("free", key=False)
        r.set_fake(composition=True)
        j = r.wait(r.svc.start(r.body("pxq4"))["id"])
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["error"]["code"], "composition")
        self.assertIn("at least 50%", j["error"]["message"])
        self.assertEqual(next(s for s in j["stages"] if s["id"] == "quantize")["status"], "failed")
        # Hugging Face unreachable: the encoder's sentence, with Control's next step
        real = os.environ["HF_ENDPOINT"]
        os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"
        try:
            r2 = Rig("free", key=False)
            r2.svc._insp["hf:tiny/Tiny-Qwen3"] = (time.time(), r.svc.inspect("tiny/Tiny-Qwen3"))
            j2 = r2.wait(r2.svc.start(r2.body("pxq3"))["id"])
        finally:
            os.environ["HF_ENDPOINT"] = real
        self.assertEqual((j2["status"], j2["error"]["code"]), ("failed", "offline"))
        self.assertIn("Cannot reach Hugging Face", j2["error"]["message"])
        self.assertIn("Resume", j2["error"]["hint"])

    def test_make_steps_that_cannot_run_are_named(self):
        r = Rig("pro")
        r.host._engines = []                      # nothing but the package: the checks do not ask for an engine install any more
        c = r.svc.checks(r.body())
        self.assertTrue(c["can_start"], c["refusal"])
        t = [x for x in c["checks"] if x["id"] == "tools"][0]
        self.assertEqual(t["status"], "ok")
        r.set_fake(not_ready={"convert": "Python packages missing: torch (pip install torch)", "dump": "no calibration tool in this package", "hess": "lib/libpxqe.so is missing"})
        r.svc.rescan()
        c = r.svc.checks(r.body())
        t = [x for x in c["checks"] if x["id"] == "tools"][0]
        self.assertEqual(t["status"], "bad")
        for word in ("convert it to GGUF", "torch", "calibration tool", "libpxqe.so"):
            self.assertIn(word, t["text"])
        self.assertFalse(c["can_start"])
        # a classic tier does not need the PXQN stages
        c = r.svc.checks(r.body("pxq3"))
        self.assertIn("torch", [x for x in c["checks"] if x["id"] == "tools"][0]["text"])           # but still needs the converter
        r.set_fake(not_ready={"dump": "no calibration tool in this package"})
        r.svc.rescan()
        self.assertTrue(r.svc.checks(r.body("pxq3"))["can_start"])

    def test_the_engine_tool_is_offered_and_the_gpu_layers_follow_what_fits(self):
        r = Rig("pro")
        job = {"params": {"tier": "pxqn4", "dump_cards": [0], "source": {"params": 1.0e9, "layers": 28}}}
        self.assertEqual(r.svc._make_engine(job), (r.engine, None))                          # the Q8_0 copy fits: every layer on the card (make's default)
        job["params"]["source"].update({"params": 20e9, "layers": 40})                       # ~21 GB of Q8_0 on a 16 GB card: part of the layers
        eng, ngl = r.svc._make_engine(job)
        self.assertEqual(eng, r.engine)
        self.assertTrue(0 < ngl < 40, ngl)
        job["params"]["tier"] = "pxq4"
        self.assertEqual(r.svc._make_engine(job), (None, None))                              # a classic tier has no calibration run
        job["params"]["tier"] = "pxqn4"
        os.unlink(os.path.join(r.engine, "bin", "llama-imatrix"))
        self.assertEqual(r.svc._make_engine(job), (None, None))                              # no engine tool: make runs the bundled CPU one

    def test_pause_stops_the_workers_make_starts_in_sessions_of_their_own(self):
        import subprocess
        # a stand-in for `pxqe make`: it starts a worker in a NEW session (as make does) and waits for it
        p = subprocess.Popen(["sh", "-c", "setsid sh -c 'sleep 30 & wait' & wait"], start_new_session=True)

        def state(pid):
            try:
                with open("/proc/%d/stat" % pid) as f:
                    raw = f.read()
                return raw[raw.rindex(")") + 2]
            except OSError:
                return "?"

        def until(cond, secs=15):
            t0 = time.time()
            while time.time() - t0 < secs and not cond():
                time.sleep(0.05)
            return cond()
        until(lambda: len(E.EncodeService._descendants(p.pid)) >= 2)                 # a loaded box may take a while to start them
        kids = E.EncodeService._descendants(p.pid)
        self.assertGreaterEqual(len(kids), 2)                                        # the setsid shell and its sleep
        svc = E.EncodeService.__new__(E.EncodeService)
        try:
            svc._signal(p, signal.SIGSTOP)
            self.assertTrue(until(lambda: {state(k) for k in kids} == {"T"}), "every process below the command is stopped: %s" % [state(k) for k in kids])
            svc._signal(p, signal.SIGCONT)
            self.assertTrue(until(lambda: state(kids[-1]) != "T"), "and continued")
        finally:
            for k in [p.pid] + kids:
                try:
                    os.kill(k, signal.SIGKILL)
                except OSError:
                    pass
            p.wait()

    def test_an_older_encoder_keeps_the_old_path_and_its_jobs_stay_on_it(self):
        r = Rig("pro", cli="1")
        c = r.svc.checks(r.body())
        self.assertTrue(c["can_start"], c["refusal"])
        jid = r.svc.start(r.body())["id"]
        j = r.wait(jid)
        self.assertEqual(j["status"], "done", j["error"])
        self.assertEqual(r.svc.jobs[jid]["params"]["flow"], "legacy")
        self.assertNotIn("makes", r.state_file())
        r2 = Rig("pro")
        j2 = r2.wait(r2.svc.start(r2.body())["id"])
        self.assertEqual(r2.svc.jobs[j2["id"]]["params"]["flow"], "make")
        self.assertNotIn("flow", {k for k in E.EncodeService.job_view(r2.svc, j2["id"])["params"]} - {"flow"})


# =====================================================================================================================
LOCK_ON_ME = {"enabled": True, "allowed_modes": ["personal"], "default_mode": "personal", "epoch": "2026-10"}                               # the default: every tier may write a personal file
LOCK_ON_SUP = {"enabled": True, "allowed_modes": ["personal", "supporters"], "default_mode": "personal", "epoch": "2026-10"}               # a tier the owner added `supporters` to
LOCK_ON_ALL = {"enabled": True, "allowed_modes": ["open", "personal", "supporters"], "default_mode": "personal", "epoch": "2026-10"}       # test / owner tiers
LOCK_OFF = {"enabled": False, "allowed_modes": ["open"], "default_mode": "open", "epoch": "2026-10"}                                      # the server's switch is off (today)
LOCK_FILE_SENTENCES = ("--lock", "--restart", "HTTP 4", "Traceback")


class LockedFiles(unittest.TestCase):
    """"Who can load this file" (Pro, PXQN tiers): the adapter is the only place that knows `pxqe make --lock`."""

    # ---- the adapter ----------------------------------------------------------------------------------------------------------------
    def test_info_says_whether_the_encoder_can_lock_and_what_the_server_allows(self):
        base = {"edition": "pro", "build_id": "b", "tiers": ["pxqn4"], "cli": "2", "make": {"progress_prefix": "@pxqe ", "sources": ["hf-repo"], "classic_tiers": ["pxq3"]}}
        old = AD.parse_info(json.dumps(base))
        self.assertEqual(old["make"]["lock_modes"], [])
        self.assertFalse(AD.supports_lock(old))
        new = AD.parse_info(json.dumps(dict(base, make=dict(base["make"], lock_modes=["auto", "open", "supporters", "personal", "weird"]))))
        self.assertEqual(new["make"]["lock_modes"], ["open", "supporters", "personal"])        # only modes this Control knows ("auto" is the encoder's own word)
        self.assertTrue(AD.supports_lock(new))
        lic = AD.parse_info(json.dumps(dict(base, licence={"state": "valid", "checked": True,
                                                           "lock": {"enabled": True, "allowed_modes": ["supporters", "personal", "x"], "default_mode": "personal", "epoch": "2026-10"}})))["licence"]
        self.assertEqual(lic["lock"], {"enabled": True, "allowed": ["personal", "supporters"], "default": "personal", "epoch": "2026-10"})   # in the page's order
        self.assertNotIn("lock", AD.parse_info(json.dumps(dict(base, licence={"state": "valid"})))["licence"])                              # no block, no key
        self.assertIsNone(AD.lock_policy("nonsense"))
        self.assertIsNone(AD.lock_policy(None))
        self.assertFalse(AD.lock_policy({"enabled": "yes"})["enabled"])                          # only a real true switches it on

    def test_argv_carries_the_lock_only_when_there_is_one(self):
        for mode in ("personal", "supporters", "open"):
            a = AD.argv_make("/x/pxqe", "m", "pxqn4", "/o.gguf", "/w", lock=mode)
            self.assertEqual(a[a.index("--lock") + 1], mode)
        for none in (None, "", "auto", "bogus", "PERSONAL"):
            self.assertNotIn("--lock", AD.argv_make("/x/pxqe", "m", "pxqn4", "/o.gguf", "/w", lock=none))        # never a value the encoder does not take
        self.assertNotIn("--lock", AD.argv_make("/x/pxqe", "m", "pxqn4", "/o.gguf", "/w"))

    def test_the_choices_for_each_state(self):
        cap = {"edition": "pro", "make": {"lock_modes": ["auto", "open", "supporters", "personal"]}}
        nocap = {"edition": "pro", "make": {}}
        self.assertIsNone(AD.lock_view({"edition": "free", "make": {}}, AD.lock_policy(LOCK_ON_ALL)))
        self.assertIsNone(AD.lock_view(None, None))
        v = AD.lock_view(nocap, AD.lock_policy(LOCK_ON_ALL))                                       # an encoder from before locks, even with the switch on
        self.assertEqual((v["state"], v["reason"], v["text"], v["choices"]), ("off", "encoder", "File locking turns on with the next encoder update", []))
        v = AD.lock_view(cap, AD.lock_policy(LOCK_OFF))
        self.assertEqual((v["state"], v["reason"], v["text"], v["choices"]), ("off", "server", "File locking turns on with the next encoder update", []))
        v = AD.lock_view(cap, AD.LOCK_NONE)                                                        # an older server that sends no lock block
        self.assertEqual((v["state"], v["reason"]), ("off", "server"))
        v = AD.lock_view(cap, None)
        self.assertEqual(v["state"], "unknown")
        self.assertIn("Press Rescan", v["text"])
        v = AD.lock_view(cap, AD.lock_policy(LOCK_ON_ME))
        self.assertEqual((v["state"], [c["id"] for c in v["choices"]], v["default"]), ("on", ["personal"], "personal"))
        self.assertEqual(v["choices"][0]["label"], "Only me (recommended)")
        v = AD.lock_view(cap, AD.lock_policy(LOCK_ON_SUP))
        self.assertEqual([c["label"] for c in v["choices"]], ["Only me (recommended)", "Any PXA supporter"])
        v = AD.lock_view(cap, AD.lock_policy(LOCK_ON_ALL))
        self.assertEqual([c["label"] for c in v["choices"]], ["Only me (recommended)", "Any PXA supporter", "Anyone (no lock)"])      # fixed order, whatever the server sent
        self.assertEqual(v["default"], "personal")
        self.assertEqual(v["needs"], "v3")
        v = AD.lock_view(cap, AD.lock_policy({"enabled": True, "allowed_modes": ["supporters", "open"], "default_mode": "open"}))   # a plan without `personal`
        self.assertEqual((v["default"], [c["id"] for c in v["choices"]]), ("open", ["supporters", "open"]))

    def test_the_mode_a_run_asks_for(self):
        cap = {"edition": "pro", "make": {"lock_modes": ["auto", "open", "supporters", "personal"]}}
        on = AD.lock_view(cap, AD.lock_policy(LOCK_ON_SUP))
        self.assertEqual(AD.lock_mode_for_run(on, "pxqn4"), ("personal", None))                    # no choice made: the default
        self.assertEqual(AD.lock_mode_for_run(on, "pxqn4", "supporters"), ("supporters", None))
        mode, err = AD.lock_mode_for_run(on, "pxqn4", "open")
        self.assertIsNone(mode)
        self.assertIn("does not allow a file that anyone can load (no lock)", err)
        self.assertIn("It allows: only me, any PXA supporter.", err)
        self.assertEqual(AD.lock_mode_for_run(AD.lock_view(cap, AD.lock_policy(LOCK_OFF)), "pxqn4", "personal"), ("open", None))   # the switch is off: written open, whatever the page held
        self.assertEqual(AD.lock_mode_for_run(AD.lock_view({"edition": "pro", "make": {}}, AD.lock_policy(LOCK_ON_ALL)), "pxqn4"), (None, None))   # an encoder with no --lock
        self.assertEqual(AD.lock_mode_for_run(AD.lock_view(cap, None), "pxqn4", "personal"), (None, None))                         # could not ask: the encoder asks itself
        self.assertEqual(AD.lock_mode_for_run(on, "pxq3", "personal"), (None, None))                                              # classic tiers have no lock
        self.assertEqual(AD.lock_mode_for_run(None, "pxqn4"), (None, None))

    def test_every_lock_refusal_of_the_encoder_is_a_plain_sentence(self):
        cases = [
            ("This licence server does not offer locked files yet, so the file would be written open. Run with --lock open, or try again later.", "lock_off"),
            ("Locked files are not switched on yet, so this encode could only be written open (unlocked). Run it with --lock open, or wait until PXA turns locking on.", "lock_off"),
            ("The licence server refused this encode (HTTP 403): locking models is not switched on yet, so this encode can only be written open (unlocked). Run it with --lock open, "
             "or wait until PXA turns locking on", "lock_off"),
            ("Your tier may not write an open file. The modes you may use: personal, supporters.", "lock_not_allowed"),
            ("The licence server refused this encode (HTTP 403): your tier may not write a supporters lock; the modes it may use: personal", "lock_not_allowed"),
            ("This run's licence job was started with lock 'open', not 'personal'. Run the same command with --lock open, or add --restart to start over.", "lock_changed"),
            ("The licence server did not grant a personal lock for this job (it granted open). Nothing was charged.", "lock_not_granted"),
            ("The licence server refused this encode (HTTP 503): the server has no lock keys configured", "lock_server"),
            ("usage: pxqe make [-h] --tier TIER\npxqe make: error: unrecognized arguments: --lock open", "lock_old_encoder"),
            ("the job ticket asks for a lock mode (weird) this encoder does not know: update the encoder", "lock_old_encoder"),
            ("the job ticket asks for a personal lock but carries no lock key", "lock_ticket"),
            ("the lock key in the job ticket is malformed", "lock_ticket"),
            ("this licence key does not open the lock key in the job ticket (wrong key, or the ticket is for another key)", "lock_ticket"),
            ("/w/encoded.gguf is a locked file (personal) but this job carries no lock key: it was started before locking, or by another encoder. Start the make again", "lock_resume"),
            ("this job writes a personal lock, but /w/encoded.gguf is an open skeleton: write the skeleton again with this job (the skeleton stage), then encode", "lock_resume"),
            ("/w/encoded.gguf: this job's key does not open the file's lock (the file was changed, or it belongs to another user)", "lock_resume"),
            ("/o.gguf is a locked file: its tensors are encrypted per file, so it cannot be spliced. Make the mix with one `pxqe make --tier-map` run instead", "lock_splice"),
        ]
        for text, code in cases:
            ex = AD.explain_failure(text, 3)
            self.assertEqual(ex["code"], code, text)
            for bad in LOCK_FILE_SENTENCES + ("lock %", "lock_"):
                self.assertNotIn(bad, ex["message"] + " " + ex["hint"], (code, bad))
            self.assertTrue(ex["message"].endswith(".") and ex["hint"].endswith("."), (code, ex))
            # and the same through the path a failed `make` takes (the encoder's last `error` line, its exit code)
            err = {"event": "error", "stage": "start", "code": "refused" if code in ("lock_off", "lock_not_allowed", "lock_not_granted", "lock_server") else "usage",
                   "message": text, "resumable": False}
            mk = AD.explain_make_failure(3 if err["code"] == "refused" else 2, err, "")
            self.assertEqual(mk["code"], code, text)
            self.assertFalse(mk["resumable"] and err["code"] == "refused")
        # a lock refusal that arrives as an HTTP 403 is NOT a revoked key
        self.assertNotEqual(AD.explain_failure("The licence server refused this encode (HTTP 403): locking models is not switched on yet", 3)["code"], "revoked")
        self.assertEqual(AD.explain_failure("The licence server refused this encode (HTTP 403): key revoked", 3)["code"], "revoked")      # and the real thing still is
        ex = AD.explain_failure("Your tier may not write an open file. The modes you may use: personal, supporters.", 3)
        self.assertEqual(ex["message"], "Your plan does not allow a file that anyone can load (no lock). It allows: only me, any PXA supporter.")

    def test_the_done_screens_words(self):
        me, sup, op = AD.lock_done_view("personal", "L-1"), AD.lock_done_view("supporters", "L-2"), AD.lock_done_view("open")
        self.assertEqual((me["locked"], me["title"]), (True, "Locked to you"))
        self.assertIn("Only you can load this file", me["text"])
        self.assertIn("loads in PXA v3 or newer", me["needs"])
        self.assertEqual((sup["locked"], sup["title"]), (True, "Locked to PXA supporters"))
        self.assertIn("Any active PXA supporter can load this file", sup["text"])
        self.assertIn("loads in PXA v3 or newer", sup["needs"])
        self.assertEqual((op["locked"], op["title"], op["needs"], op["file_id"]), (False, "Not locked", "", ""))
        self.assertIn("Anyone with a PXA engine can load this file", op["text"])
        self.assertTrue(AD.lock_done_view("locked")["locked"])

    def test_the_done_line_of_make_carries_the_lock(self):
        sha = "ab" * 32
        ev = AD.parse_make_line('@pxqe {"event":"done","out":"/o.gguf","sha256":"%s","size":9,"tier":"pxqn4","lock":"supporters","lock_file_id":"L-0123456789abcdef","lock_epoch":"2026-10"}' % sha)
        self.assertEqual((ev["lock"], ev["lock_file_id"], ev["lock_epoch"]), ("supporters", "L-0123456789abcdef", "2026-10"))
        ev = AD.parse_make_line('@pxqe {"event":"done","out":"/o.gguf","sha256":"%s","size":9,"tier":"pxqn4"}' % sha)
        self.assertEqual((ev["lock"], ev["lock_file_id"]), ("", ""))
        self.assertEqual(AD.parse_make_line('@pxqe {"event":"done","out":"/o.gguf","sha256":"%s","size":9,"tier":"x","lock":"strange"}' % sha)["lock"], "")

    def test_probe_of_a_lock_capable_encoder(self):
        d = os.path.join(TMP, "probe-lock")
        cap = F.make_fake_encoder(os.path.join(d, "cap"), "pro", build_id="bl", lock_support=True, lock_server=LOCK_ON_SUP)
        old = F.make_fake_encoder(os.path.join(d, "old"), "pro", build_id="bo", lock_server=LOCK_ON_SUP)
        env = {"PXQE_KEY": GOOD_KEY}
        a, b = AD.probe(cap, env=env), AD.probe(old, env=env)
        self.assertEqual(a["make"]["lock_modes"], ["open", "supporters", "personal"])
        self.assertTrue(AD.supports_lock(a) and not AD.supports_lock(b))
        self.assertNotIn("lock", a["licence"])                                                  # the offline view asks no server
        self.assertEqual(AD.probe(cap, check_licence=True, env=env)["licence"]["lock"]["allowed"], ["personal", "supporters"])
        self.assertNotIn("lock", AD.probe(old, check_licence=True, env=env)["licence"])        # an encoder from before locks does not copy the block

    # ---- the service: plan, checks, start ---------------------------------------------------------------------------------------------
    def rig(self, server=LOCK_ON_ME, support=True, **kw):
        extra = dict(kw)
        if support:
            extra["lock_support"] = True
        if server is not None:
            extra["lock_server"] = server
        return Rig("pro", **extra)

    def run_job(self, r, **body):
        j = r.wait(r.svc.start(r.body(**body))["id"])
        return j

    def make_argv(self, r, n=-1):
        return r.state_file()["makes"][n]["argv"]

    def test_default_is_personal_and_the_plan_offers_only_what_the_tier_allows(self):
        r = self.rig(LOCK_ON_ME)
        v = r.svc.plan(r.body())["lock"]
        self.assertEqual((v["state"], [c["id"] for c in v["choices"]], v["default"]), ("on", ["personal"], "personal"))
        c = r.svc.checks(r.body())
        lk = [x for x in c["checks"] if x["id"] == "lock"][0]
        self.assertEqual(lk["status"], "ok")
        self.assertIn("Only you will be able to load this file", lk["text"])
        self.assertIn("v3 or newer", lk["fix"])
        self.assertEqual(c["lock"], "personal")
        self.assertTrue(c["can_start"], c["refusal"])
        j = self.run_job(r)                                                                          # no choice made by the page: the default, passed explicitly
        self.assertEqual(j["status"], "done", j["error"])
        argv = self.make_argv(r)
        self.assertEqual(argv[argv.index("--lock") + 1], "personal")
        self.assertEqual(j["params"]["lock"], "personal")
        lock = j["result"]["lock"]
        self.assertEqual((lock["mode"], lock["locked"], lock["title"]), ("personal", True, "Locked to you"))
        self.assertIn("loads in PXA v3 or newer", lock["needs"])
        self.assertEqual(lock["file_id"], "L-0123456789abcdef")
        self.assertGreater(j["result"]["tier_tensors"], 0)                                           # a locked file lists its PXQN tensors under type id 4096 + type: still counted
        h = L.gguf_header(j["result"]["path"])
        self.assertEqual((h["kv"].get("pxa.lock.mode"), h.get("locked")), ("personal", True))
        self.assertNotIn(GOOD_KEY, json.dumps(r.svc.job_view(j["id"])) + "\n".join(r.svc.log_view(j["id"])["lines"]))

    def test_supporters_when_the_tier_allows_it(self):
        r = self.rig(LOCK_ON_SUP)
        self.assertEqual([c["label"] for c in r.svc.plan(r.body())["lock"]["choices"]], ["Only me (recommended)", "Any PXA supporter"])
        j = self.run_job(r, lock="supporters")
        self.assertEqual(j["status"], "done", j["error"])
        argv = self.make_argv(r)
        self.assertEqual(argv[argv.index("--lock") + 1], "supporters")
        self.assertEqual((j["result"]["lock"]["mode"], j["result"]["lock"]["title"]), ("supporters", "Locked to PXA supporters"))
        self.assertIn("Any active PXA supporter can load this file", j["result"]["lock"]["text"])
        self.assertEqual(L.gguf_header(j["result"]["path"])["kv"].get("pxa.lock.mode"), "supporters")

    def test_anyone_when_the_tier_allows_it(self):
        r = self.rig(LOCK_ON_ALL)
        self.assertEqual([c["id"] for c in r.svc.plan(r.body())["lock"]["choices"]], ["personal", "supporters", "open"])
        j = self.run_job(r, lock="open")
        self.assertEqual(j["status"], "done", j["error"])
        argv = self.make_argv(r)
        self.assertEqual(argv[argv.index("--lock") + 1], "open")
        lock = j["result"]["lock"]
        self.assertEqual((lock["mode"], lock["locked"], lock["title"], lock["needs"]), ("open", False, "Not locked", ""))
        self.assertNotIn("pxa.lock.mode", L.gguf_header(j["result"]["path"])["kv"])

    def test_a_choice_the_tier_does_not_allow_is_stopped_before_anything_runs(self):
        r = self.rig(LOCK_ON_ME)
        for want in ("supporters", "open"):
            c = r.svc.checks(r.body(lock=want))
            lk = [x for x in c["checks"] if x["id"] == "lock"][0]
            self.assertEqual(lk["status"], "bad")
            self.assertIn("Your plan does not allow a file that", lk["text"])
            self.assertIn("It allows: only me.", lk["text"])
            self.assertFalse(c["can_start"])
            self.assertEqual(c["refusal"], lk["text"])
            with self.assertRaises(E.EncodeError) as cm:
                r.svc.start(r.body(lock=want))
            self.assertIn("It allows: only me.", str(cm.exception))
        self.assertEqual(r.state_file().get("makes", []), [])                                      # the encoder was never called, nothing charged
        with self.assertRaises(E.EncodeError) as cm:                                               # a value that is no choice at all
            r.svc.start(r.body(lock="everyone"))
        self.assertIn("only you, any PXA supporter, or anyone", str(cm.exception))

    def test_while_the_server_switch_is_off_the_choice_is_disabled_and_the_file_is_open(self):
        r = self.rig(LOCK_OFF)
        v = r.svc.plan(r.body())["lock"]
        self.assertEqual((v["state"], v["choices"], v["text"]), ("off", [], "File locking turns on with the next encoder update"))
        c = r.svc.checks(r.body())
        lk = [x for x in c["checks"] if x["id"] == "lock"][0]
        self.assertEqual(lk["status"], "ok")
        self.assertIn("without a lock", lk["text"])
        self.assertIn("File locking turns on with the next encoder update", lk["text"])
        j = self.run_job(r, lock="personal")                                                         # a page that still held a choice: ignored while it is off
        self.assertEqual(j["status"], "done", j["error"])
        argv = self.make_argv(r)
        self.assertEqual(argv[argv.index("--lock") + 1], "open")
        self.assertEqual((j["result"]["lock"]["locked"], j["result"]["lock"]["title"]), (False, "Not locked"))
        self.assertIn("Anyone with a PXA engine can load this file", j["result"]["lock"]["text"])

    def test_a_server_that_sends_no_lock_block_is_the_same_as_off(self):
        r = self.rig(None)
        v = r.svc.plan(r.body())["lock"]
        self.assertEqual((v["state"], v["text"]), ("off", "File locking turns on with the next encoder update"))
        j = self.run_job(r)
        self.assertEqual(j["status"], "done", j["error"])
        argv = self.make_argv(r)
        self.assertEqual(argv[argv.index("--lock") + 1], "open")
        self.assertFalse(j["result"]["lock"]["locked"])

    def test_an_encoder_without_lock_support_is_never_given_the_flag(self):
        r = self.rig(LOCK_ON_ALL, support=False)                                                   # the server's switch is on, the encoder is from before locks
        v = r.svc.plan(r.body())["lock"]
        self.assertEqual((v["state"], v["reason"], v["text"]), ("off", "encoder", "File locking turns on with the next encoder update"))
        j = self.run_job(r, lock="personal")
        self.assertEqual(j["status"], "done", j["error"])
        self.assertNotIn("--lock", self.make_argv(r))                                              # (this fake exits 2 on an unknown flag, as argparse does)
        self.assertFalse(j["result"]["lock"]["locked"])
        self.assertTrue(r.state_file()["makes"][0]["argv"])

    def test_when_the_licence_server_cannot_be_asked_no_flag_is_passed(self):
        r = self.rig(LOCK_ON_ALL, status_fail=True)
        v = r.svc.plan(r.body())["lock"]
        self.assertEqual(v["state"], "unknown")
        self.assertIn("Press Rescan", v["text"])
        c = r.svc.checks(r.body(lock="personal"))
        lk = [x for x in c["checks"] if x["id"] == "lock"][0]
        self.assertEqual(lk["status"], "warn")
        self.assertIn("Done page says what lock the file got", lk["text"])
        self.assertTrue(c["can_start"], c["refusal"])
        j = self.run_job(r, lock="personal")
        self.assertEqual(j["status"], "done", j["error"])
        self.assertNotIn("--lock", self.make_argv(r))                                              # the encoder asks the server itself and uses the plan's default
        self.assertIsNone(j["params"]["lock"])

    def test_the_switch_can_flip_between_two_looks(self):
        r = self.rig(LOCK_OFF)
        self.assertEqual(r.svc.plan(r.body())["lock"]["state"], "off")
        r.set_fake(lock_server=LOCK_ON_SUP)
        self.assertEqual(r.svc.plan(r.body())["lock"]["state"], "off")                              # the answer is kept a few minutes ...
        r.svc.rescan()
        self.assertEqual(r.svc.plan(r.body())["lock"]["state"], "on")                               # ... Rescan asks again
        r.set_fake(lock_server=LOCK_OFF)
        j = self.run_job(r, lock="supporters")                                                      # and a start always asks: the file is written open, as the page said last
        self.assertEqual(j["status"], "done", j["error"])
        argv = self.make_argv(r)
        self.assertEqual(argv[argv.index("--lock") + 1], "open")

    def test_no_lock_for_the_free_encoder_a_classic_tier_or_no_key(self):
        free = Rig("free", key=False)
        self.assertIsNone(free.svc.plan(free.body("pxq3"))["lock"])
        pro = self.rig(LOCK_ON_ALL)
        c = pro.svc.checks(pro.body("pxq3", cards=[0]))
        self.assertNotIn("lock", [x["id"] for x in c["checks"]])
        self.assertIsNone(c["lock"])
        j = pro.wait(pro.svc.start(pro.body("pxq3", cards=[0], lock="personal"))["id"])             # a classic tier has no lock: the choice is not passed
        self.assertEqual(j["status"], "done", j["error"])
        self.assertNotIn("--lock", pro.state_file()["makes"][-1]["argv"])
        self.assertIsNone(j["result"]["lock"])
        nokey = Rig("pro", key=False, lock_support=True, lock_server=LOCK_ON_ALL)
        self.assertIsNone(nokey.svc.plan(nokey.body())["lock"])

    def test_the_resume_keeps_the_lock_it_started_with(self):
        r = self.rig(LOCK_ON_SUP, crash_after=5)
        jid = r.svc.start(r.body(lock="supporters"))["id"]
        self.assertEqual(r.wait(jid)["status"], "failed")
        r.svc.resume(jid)
        j = r.wait(jid)
        self.assertEqual(j["status"], "done", j["error"])
        a0, a1 = [m["argv"] for m in r.state_file()["makes"]]
        self.assertEqual(a0, a1)                                                                    # the same command again: nothing about the lock changes between runs
        self.assertEqual(a1[a1.index("--lock") + 1], "supporters")
        self.assertEqual(j["result"]["lock"]["mode"], "supporters")
        self.assertEqual(r.state_file()["jobs"], 1)                                                 # ONE charge

    def test_a_resume_under_another_lock_is_refused_in_plain_words(self):
        r = self.rig(LOCK_ON_SUP, crash_after=5)
        jid = r.svc.start(r.body(lock="supporters"))["id"]
        self.assertEqual(r.wait(jid)["status"], "failed")
        r.svc.jobs[jid]["params"]["lock"] = "personal"                                              # (what a hand-edited or older job record could hold)
        r.svc.resume(jid)
        j = r.wait(jid)
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["error"]["code"], "lock_changed")
        self.assertIn("started with the choice \"any PXA supporter\" and cannot continue with \"only me\"", j["error"]["message"])
        for bad in LOCK_FILE_SENTENCES:
            self.assertNotIn(bad, j["error"]["message"] + j["error"]["hint"])
        self.assertIn("discard", j["error"]["hint"])

    def test_refusals_from_the_licence_server_during_the_run(self):
        for fake, code, frag in (
                ({"lock_refuse": "locking models is not switched on yet, so this encode can only be written open (unlocked). Run it with --lock open, or wait until PXA turns locking on"},
                 "lock_off", "Locking files is not switched on at the PXA licence server right now"),
                ({"lock_refuse": "your tier may not write a supporters lock; the modes it may use: personal"}, "lock_not_allowed", "It allows: only me."),
                ({"lock_refuse": "the server has no lock keys configured"}, "lock_server", "cannot lock files at the moment"),
                ({"lock_grant": "open"}, "lock_not_granted", "did not give this file the lock you chose")):
            r = self.rig(LOCK_ON_ME, **fake)
            j = self.run_job(r)
            self.assertEqual(j["status"], "failed", fake)
            self.assertEqual(j["error"]["code"], code, (fake, j["error"]))
            self.assertIn(frag, j["error"]["message"])
            for bad in LOCK_FILE_SENTENCES + ("revoked",):
                self.assertNotIn(bad, j["error"]["message"] + " " + j["error"]["hint"], (code, bad))
            self.assertEqual(r.state_file().get("jobs", 0), 0)                                      # refused before the job was charged

    def test_the_done_screen_trusts_the_file_over_the_encoders_last_line(self):
        r = self.rig(LOCK_ON_ME)
        j = self.run_job(r)
        self.assertEqual(j["result"]["lock"]["mode"], "personal")
        # the encoder claims a lock the file does not have: the file's own header is what is shown, and the log says so
        open_gguf = os.path.join(r.dir, "open.gguf")
        import fake_encode_tools as T
        T.write_gguf(open_gguf, 4, 259)
        job = r.svc.jobs[j["id"]]
        res = r.svc._lock_result(job, L.gguf_header(open_gguf), {"lock": "personal", "lock_file_id": "L-1"})
        self.assertEqual((res["mode"], res["locked"]), ("open", False))
        self.assertTrue([ln for ln in r.svc.log_view(j["id"])["lines"] if "the file is what counts" in ln])
        # a locked file whose header names no mode is still said to be locked
        locked_gguf = os.path.join(r.dir, "locked.gguf")
        T.write_gguf(locked_gguf, 4, 259, lock_mode="mystery")
        res = r.svc._lock_result(job, L.gguf_header(locked_gguf), {"lock": "", "lock_file_id": ""})
        self.assertEqual((res["mode"], res["locked"]), ("locked", True))

    def test_a_finished_job_from_before_locks_has_no_lock_line(self):
        r = Rig("pro")                                                                               # the default fake: no lock support, no lock block
        j = self.run_job(r)
        self.assertEqual(j["status"], "done", j["error"])
        self.assertNotIn("--lock", self.make_argv(r))
        self.assertFalse(j["result"]["lock"]["locked"])
        self.assertEqual(j["result"]["lock"]["title"], "Not locked")


# =====================================================================================================================
RFC8032_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
RFC8032_PUB = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
RFC8032_SIG = bytes.fromhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")


class Signatures(unittest.TestCase):
    def test_rfc8032_vector(self):
        self.assertTrue(PK.ed25519_verify(RFC8032_PUB, b"", RFC8032_SIG))
        self.assertFalse(PK.ed25519_verify(RFC8032_PUB, b"x", RFC8032_SIG))
        bad = bytearray(RFC8032_SIG)
        bad[40] ^= 1
        self.assertFalse(PK.ed25519_verify(RFC8032_PUB, b"", bytes(bad)))
        self.assertFalse(PK.ed25519_verify(RFC8032_PUB[:31], b"", RFC8032_SIG))
        self.assertFalse(PK.ed25519_verify(RFC8032_PUB, b"", RFC8032_SIG[:63]))
        self.assertEqual(F.ed_pub(RFC8032_SEED), RFC8032_PUB)                    # the test signer is the same algorithm
        self.assertEqual(F.ed_sign(RFC8032_SEED, b""), RFC8032_SIG)

    def test_signature_forms(self):
        sig = F.ed_sign(F.SEED, b"m" * 32)
        import base64
        self.assertEqual(PK._decode_sig(sig.hex()), sig)
        self.assertEqual(PK._decode_sig(F._b64u(sig)), sig)                            # base64url without padding (what the server sends)
        self.assertEqual(PK._decode_sig(base64.b64encode(sig).decode()), sig)
        self.assertEqual(PK._decode_sig("!!!"), b"")

    def test_the_shipped_key_is_the_one_in_package_md(self):
        self.assertEqual(PK.PACKAGE_PUBKEYS["pkg-2026-10"], "de9a21ae8508a373452f422fe110ea08e68ac72569019cb2214442facfdf7af4")
        old = os.environ.pop("PXA_PACKAGE_PUBKEY")
        try:
            self.assertEqual(PK.pubkey_bytes("pkg-2026-10").hex(), PK.PACKAGE_PUBKEYS["pkg-2026-10"])
            self.assertEqual(PK.pubkey_bytes("other-kid"), b"")
            self.assertEqual(PK.pubkey_bytes(None), b"")
        finally:
            os.environ["PXA_PACKAGE_PUBKEY"] = old


class PackageStatement(unittest.TestCase):
    """The signed statement a package answer carries (PACKAGE.md): signature, what it says, the file it describes."""

    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.ans, self.archive = F.build_package(self.d, "pro", "b-t1")

    def ok(self, ans=None, **kw):
        return PK.check_statement(ans or self.ans, kw.get("edition", "pro"), kw.get("platform", "linux-x86_64"), kw.get("cuda", 12), kw.get("build"))

    def fails(self, code, frag, ans=None, **kw):
        with self.assertRaises(PK.PackageError) as cm:
            self.ok(ans, **kw)
        self.assertEqual(cm.exception.code, code)
        self.assertIn(frag, str(cm.exception))

    def test_good_statement_and_package_pass(self):
        m = self.ok()
        self.assertEqual((m["edition"], m["build_id"], m["kid"]), ("pro", "b-t1", "pkg-2026-10"))
        PK.verify_package(self.archive, m)
        self.assertEqual(json.loads(self.ans["statement"])["sha256"], m["sha256"])

    def test_every_way_to_be_wrong_is_refused_with_a_plain_message(self):
        self.fails("bad_signature", "NOT installed", dict(self.ans, statement=self.ans["statement"].replace("b-t1", "b-t9")))            # a changed statement
        other, _ = F.build_package(self.d, "pro", "b-t2", seed=F.OTHER_SEED)
        self.fails("bad_signature", "NOT installed", other)                                                                                  # signed by someone else
        self.fails("bad_signature", "NOT installed", dict(self.ans, signature="AAAA"))
        self.fails("bad_statement", "does not match", platform="linux-arm64")                                                                # another platform asked
        self.fails("bad_statement", "does not match", cuda=11)
        self.fails("bad_statement", "does not match", edition="free")
        self.fails("bad_statement", "does not match", build="b-other")
        wrong, _ = F.build_package(self.d, "pro", "b-t3", wrong_platform=True)
        self.fails("bad_statement", "does not match", wrong)                                                                                 # a valid signature for the wrong machine
        for k in ("statement", "signature", "url", "build_id"):
            self.fails("bad_manifest", "incomplete", {x: v for x, v in self.ans.items() if x != k})
        self.fails("no_pubkey", "does not know the key", dict(self.ans, kid="unknown-kid"), ) if "PXA_PACKAGE_PUBKEY" not in os.environ else None

    def test_unknown_key_id_with_no_override_is_refused(self):
        old = os.environ.pop("PXA_PACKAGE_PUBKEY")
        try:
            self.fails("no_pubkey", "does not know the key", dict(self.ans, kid="unknown-kid"))
        finally:
            os.environ["PXA_PACKAGE_PUBKEY"] = old

    def test_the_downloaded_file_must_match_the_statement(self):
        m = self.ok()

        def fails(mm, code, frag):
            with self.assertRaises(PK.PackageError) as cm:
                PK.verify_package(self.archive, mm)
            self.assertEqual(cm.exception.code, code)
            self.assertIn(frag, str(cm.exception))
        fails(dict(m, size=m["size"] + 1), "bad_checksum", "wrong size")
        bad, _ = F.build_package(self.d, "pro", "b-t4")
        flipped = os.path.join(self.d, "flipped.tar.gz")
        with open(self.archive, "rb") as f:
            raw = bytearray(f.read())
        raw[len(raw) // 2] ^= 0xFF                                               # one byte of the same-size file is different
        with open(flipped, "wb") as f:
            f.write(bytes(raw))
        with self.assertRaises(PK.PackageError) as cm:
            PK.verify_package(flipped, m)
        self.assertEqual((cm.exception.code, "damaged" in str(cm.exception)), ("bad_checksum", True))
        fails(dict(m, signature=F._b64u(b"x" * 64)), "bad_signature", "NOT installed")

    def test_install_extracts_runs_info_and_lands_in_edition_build_folder(self):
        root = os.path.join(self.d, "root")
        info = PK.install(self.archive, self.ok(), root)
        self.assertTrue(info["ok"])
        self.assertEqual((info["edition"], info["build_id"]), ("pro", "b-t1"))
        self.assertEqual(info["path"], os.path.join(root, "pro", "b-t1", "pxqe"))          # the top folder pxqe/ is stripped
        self.assertTrue(os.access(info["path"], os.X_OK))
        self.assertTrue(os.path.isfile(os.path.join(root, "pro", "b-t1", "MANIFEST.json")))
        self.assertFalse(os.path.exists(os.path.join(root, "pro", "b-t1.installing")))

    def test_the_packages_own_file_list_is_verified_too(self):
        bad, a = F.build_package(self.d, "pro", "b-t5", tamper_file=True)
        with self.assertRaises(PK.PackageError) as cm:
            PK.install(a, self.ok(bad), os.path.join(self.d, "r5"))
        self.assertEqual(cm.exception.code, "bad_package")
        self.assertIn("does not match its checksum", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.d, "r5", "pro", "b-t5")))
        evil, a2 = F.build_package(self.d, "pro", "b-t6", seed=F.SEED)
        # a manifest signature from the wrong key
        import tarfile as _t
        with _t.open(a2) as src:
            names = src.getnames()
        self.assertIn("pxqe/MANIFEST.sig", names)

    def test_an_unsigned_manifest_is_accepted_but_its_hashes_are_still_checked(self):
        # the real Free package carries a MANIFEST.json the server does not sign: hashes still have to match
        import tarfile as _t
        ok = os.path.join(self.d, "unsigned")
        os.makedirs(ok)
        open(os.path.join(ok, "a.txt"), "w").write("hello")
        man = {"edition": "free", "build_id": "x", "files": {"a.txt": hashlib.sha256(b"hello").hexdigest()}}
        json.dump(man, open(os.path.join(ok, "MANIFEST.json"), "w"))
        PK.verify_manifest_files(ok)
        open(os.path.join(ok, "a.txt"), "w").write("changed")
        with self.assertRaises(PK.PackageError) as cm:
            PK.verify_manifest_files(ok)
        self.assertIn("does not match its checksum", str(cm.exception))
        # a signature that is present but wrong is refused
        open(os.path.join(ok, "a.txt"), "w").write("hello")
        open(os.path.join(ok, "MANIFEST.sig"), "w").write(F._b64u(b"x" * 64) + "\n")
        with self.assertRaises(PK.PackageError) as cm:
            PK.verify_manifest_files(ok)
        self.assertEqual(cm.exception.code, "bad_signature")
        PK.verify_manifest_files(self.d + "/nonexistent-folder-without-manifest") if False else None

    def test_hostile_archives_with_a_valid_signature_are_still_refused(self):
        for kind in ("traversal", "symlink"):
            ans, a = F.build_evil_package(self.d, kind)
            root = os.path.join(self.d, "root-" + kind)
            with self.assertRaises(PK.PackageError) as cm:
                PK.install(a, self.ok(ans), root)
            self.assertEqual(cm.exception.code, "bad_package", kind)
            self.assertFalse(os.path.exists(os.path.join(self.d, "escaped.txt")))
            self.assertFalse(os.path.exists(os.path.join(root, "pro", ans["build_id"])))

    def test_package_that_is_not_what_was_promised_is_refused(self):
        ans, a = F.build_package(self.d, "pro", "b-real")
        m = self.ok(ans)
        with self.assertRaises(PK.PackageError) as cm:
            PK.install(a, dict(m, build_id="b-claimed"), os.path.join(self.d, "r1"))
        self.assertIn(cm.exception.code, ("bad_package", "bad_signature"))             # the claim no longer matches the signed statement either

    def test_package_whose_cli_does_not_start_is_refused_and_cleaned_up(self):
        ans, a = F.build_package(self.d, "pro", "b-broken", info_broken="garbage")
        root = os.path.join(self.d, "r2")
        with self.assertRaises(PK.PackageError) as cm:
            PK.install(a, self.ok(ans), root)
        self.assertEqual(cm.exception.code, "broken")
        self.assertEqual(os.listdir(os.path.join(root, "pro")) if os.path.isdir(os.path.join(root, "pro")) else [], [])


class LicenceServer(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.lic = F.FakeLicence(self.d).start()

    def tearDown(self):
        self.lic.stop()

    def test_free_package_end_to_end(self):
        m = PK.free_latest(self.lic.url)
        self.assertEqual((m["edition"], m["build_id"]), ("free", "b-free-2"))
        self.assertTrue(m["url"].startswith(self.lic.url + "/v1/free/download/"))        # a path is resolved against the server
        self.assertEqual(self.lic.requests[0]["path"], "/v1/free/latest?platform=linux-x86_64&cuda_major=12")
        info = PK.install_from_manifest(m, root=os.path.join(self.d, "root"), work=os.path.join(self.d, "work"))
        self.assertEqual((info["edition"], info["ok"]), ("free", True))
        self.assertEqual(os.listdir(os.path.join(self.d, "work")), [])          # the archive is deleted after install
        self.assertTrue(all("pxk1" not in json.dumps(r) for r in self.lic.requests))     # no key was involved
        self.assertEqual(info["runtime"]["lib"], "none")

    def test_pro_package_sends_the_key_only_in_the_post_body(self):
        m = PK.request_package(self.lic.url, F.GOOD_KEY, "linux-x86_64", 12)
        self.assertEqual(m["edition"], "pro")
        posts = [r for r in self.lic.requests if r["method"] == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertEqual(json.loads(posts[0]["body"]), {"key": F.GOOD_KEY, "platform": "linux-x86_64", "cuda_major": 12})
        self.assertNotIn(SECRET, posts[0]["path"])
        self.assertNotIn(SECRET, json.dumps(posts[0]["headers"]))
        PK.install_from_manifest(m, root=os.path.join(self.d, "root"), work=os.path.join(self.d, "work"))
        for r in self.lic.requests:
            if r["method"] == "GET":
                self.assertNotIn(SECRET, json.dumps(r))                         # the download carries no key
        self.assertEqual(json.loads(PK.request_package(self.lic.url, F.GOOD_KEY, "linux-x86_64", 12, build_id="b-pro-2") and
                                    self.lic.requests[-1]["body"])["build_id"], "b-pro-2")

    def test_update_question(self):
        r = PK.latest_pro(self.lic.url, F.GOOD_KEY, "b-pro-1")
        self.assertEqual(r, {"latest": "b-pro-2", "update_available": True, "edition": "pro"})
        self.assertFalse(PK.latest_pro(self.lic.url, F.GOOD_KEY, "b-pro-2")["update_available"])
        self.assertEqual(json.loads(self.lic.requests[0]["body"]), {"key": F.GOOD_KEY, "build_id": "b-pro-1"})

    def test_server_answers_in_plain_words(self):
        cases = (("revoked", "revoked, expired or suspended", "revoked"), ("noquota", "no encodes left", "no_quota"), ("server_error", "having trouble", "server"),
                 ("invalid_key", "does not recognise this key", "bad_key"))
        for mode, frag, code in cases:
            self.lic.mode = mode
            with self.assertRaises(PK.PackageError) as cm:
                PK.request_package(self.lic.url, F.GOOD_KEY, "linux-x86_64", 12)
            self.assertEqual(cm.exception.code, code)
            self.assertIn(frag, str(cm.exception))
            self.assertNotIn(SECRET, str(cm.exception))
        self.lic.mode = "ok"
        with self.assertRaises(PK.PackageError) as cm:
            PK.request_package(self.lic.url, F.OTHER_KEY, "linux-x86_64", 12)
        self.assertEqual(cm.exception.code, "bad_key")                          # 401: the server does not know it
        with self.assertRaises(PK.PackageError) as cm:
            PK.request_package(self.lic.url, "garbage", "linux-x86_64", 12)
        self.assertEqual(cm.exception.code, "bad_key")
        self.lic.mode = "wrong_edition"
        with self.assertRaises(PK.PackageError) as cm:
            PK.request_package(self.lic.url, F.GOOD_KEY, "linux-x86_64", 12)
        self.assertIn(cm.exception.code, ("bad_statement", "bad_manifest"))

    def test_expired_download_link_says_so(self):
        m = PK.request_package(self.lic.url, F.GOOD_KEY, "linux-x86_64", 12)
        self.lic.mode = "link_expired"
        with self.assertRaises(PK.PackageError) as cm:
            PK.install_from_manifest(m, root=os.path.join(self.d, "root"), work=os.path.join(self.d, "work"))
        self.assertEqual(cm.exception.code, "expired_link")
        self.assertIn("15 minutes", str(cm.exception))

    def test_offline_is_offer_retry_and_free(self):
        url = self.lic.url
        self.lic.stop()
        with self.assertRaises(PK.PackageError) as cm:
            PK.free_latest(url)
        self.assertEqual(cm.exception.code, "offline")
        self.assertIn("Free encoder", str(cm.exception))

    def test_bad_signature_checksum_platform_and_file_list_never_install(self):
        for mode, code in (("bad_signature", "bad_signature"), ("bad_checksum", "bad_checksum"), ("wrong_platform", "bad_statement"), ("bad_manifest_file", "bad_package")):
            self.lic.mode = mode
            self.lic.rebuild()
            root = os.path.join(self.d, "root-" + mode)
            with self.assertRaises(PK.PackageError) as cm:
                m = PK.request_package(self.lic.url, F.GOOD_KEY, "linux-x86_64", 12)
                PK.install_from_manifest(m, root=root, work=os.path.join(self.d, "work-" + mode))
            self.assertEqual(cm.exception.code, code, mode)
            pro = os.path.join(root, "pro")
            self.assertEqual(os.listdir(pro) if os.path.isdir(pro) else [], [], mode)            # no build folder and no half-installed leftover
            work = os.path.join(self.d, "work-" + mode)
            self.assertEqual(os.listdir(work) if os.path.isdir(work) else [], [], mode)            # the bad download was deleted

    def test_redirects_from_the_licence_server_are_not_followed(self):
        class R(F._Base):
            def handle(s, h, method, body):
                h.send_response(302)
                h.send_header("Location", "http://127.0.0.1:9/steal")
                h.send_header("Content-Length", "0")
                h.end_headers()
        red = R().start()
        try:
            with self.assertRaises(PK.PackageError):
                PK.request_package(red.url, F.GOOD_KEY, "linux-x86_64", 12)
            self.assertEqual(len(red.requests), 1)
        finally:
            red.stop()

    def test_plain_http_is_refused_off_loopback(self):
        with self.assertRaises(PK.PackageError) as cm:
            PK.free_latest("http://example.com")
        self.assertEqual(cm.exception.code, "bad_url")

    def test_download_resumes_with_range(self):
        m = PK.free_latest(self.lic.url)
        dest = os.path.join(self.d, "dl.tar.gz")
        with open(self.lic.archives["free"][1], "rb") as f:
            data = f.read()
        with open(dest + ".part", "wb") as f:
            f.write(data[:100])
        PK.download(m["url"], dest, m["size"])
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), data)
        self.assertTrue(any((r["headers"].get("range") or "") == "bytes=100-" for r in self.lic.requests))

    def test_platform_and_key_helpers(self):
        self.assertEqual(PK.detect_platform("12.4")[1], 12)
        self.assertEqual(PK.detect_platform(None)[1], 12)                       # unreadable CUDA version: the CUDA 12 package is the only one
        self.assertEqual(PK.detect_platform("13.0")[1], 12)                      # a newer driver runs the CUDA 12 build
        self.assertEqual(PK.detect_platform("11.8")[1], 11)
        self.assertTrue(PK.detect_platform("12.4")[0].startswith("linux"))
        self.assertTrue(PK.valid_key(F.GOOD_KEY))
        for bad in ("", "pxk1", "pxk2.K-X.abc", None, 5, "pxk1.K-TEST0001.short"):
            self.assertFalse(PK.valid_key(bad))
        m = PK.mask_key(F.GOOD_KEY)
        self.assertTrue(m.startswith("pxk1.K-TEST0001."))
        self.assertNotIn(SECRET, m)
        self.assertEqual(PK.mask_key(""), "")
        self.assertEqual(PK._scrub("x %s y" % F.GOOD_KEY, F.GOOD_KEY), "x <key> y")
        self.assertEqual(PK.licence_url({"licence_url": "http://127.0.0.1:1/"}), "http://127.0.0.1:1")
        self.assertEqual(PK.LICENCE_URL_DEFAULT, "https://lic.pxanetwork.com")
        os.environ["PXA_LICENCE_URL"] = "http://127.0.0.1:2"
        try:
            self.assertEqual(PK.licence_url({"licence_url": "http://127.0.0.1:1"}), "http://127.0.0.1:2")
        finally:
            os.environ.pop("PXA_LICENCE_URL")


class GetTheEncoder(unittest.TestCase):
    """The service flows behind the 'Get the encoder' panel, against the fake licence server."""

    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.lic = F.FakeLicence(os.path.join(self.d, "pkg")).start()
        os.environ["PXA_LICENCE_URL"] = self.lic.url
        os.environ["PXA_ENCODER_HOME"] = os.path.join(self.d, "enchome")
        self.r = Rig(with_encoder=False, key=False)

    def tearDown(self):
        self.lic.stop()
        os.environ.pop("PXA_LICENCE_URL", None)
        os.environ["PXA_ENCODER_HOME"] = os.path.join(TMP, "encoder-home")

    def wait_pkg(self, timeout=60):
        t0 = time.time()
        while self.r.svc.pkg.get("running") and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.r.svc.state()["pkg"]

    def test_nothing_installed_then_free_download_installs_and_selects(self):
        self.assertEqual(self.r.svc.state()["encoders"], [])
        self.r.svc.get_encoder("free")
        pkg = self.wait_pkg()
        self.assertEqual(pkg["phase"], "done", pkg)
        st = self.r.svc.state()
        self.assertEqual((st["edition"], len(st["encoders"])), ("free", 1))
        self.assertIn(os.path.join("free", "b-free-2"), st["selected"])

    def test_pro_needs_a_key_then_installs_and_beats_free(self):
        with self.assertRaises(E.EncodeError) as cm:
            self.r.svc.get_encoder("pro")
        self.assertIn("Paste your key first", str(cm.exception))
        with self.assertRaises(E.EncodeError):
            self.r.svc.get_encoder("pro", key="nope")
        self.r.svc.get_encoder("free")
        self.wait_pkg()
        self.r.svc.get_encoder("pro", key=F.GOOD_KEY)
        self.assertEqual(self.wait_pkg()["phase"], "done")
        st = self.r.svc.state()
        self.assertEqual(st["edition"], "pro")                                  # Pro preferred when both are installed
        self.assertEqual({e["edition"] for e in st["encoders"]}, {"free", "pro"})
        self.assertTrue(st["key_set"])
        self.assertNotIn(SECRET, json.dumps(st))
        self.assertTrue(st["key_masked"].endswith("\u2022" * 8))
        self.r.svc.use([e["path"] for e in st["encoders"] if e["edition"] == "free"][0])
        self.assertEqual(self.r.svc.state()["edition"], "free")                  # and the user can switch back

    def test_failures_are_stored_as_plain_messages_never_the_key(self):
        for mode, frag in (("revoked", "revoked"), ("noquota", "no encodes left"), ("bad_signature", "NOT installed"), ("bad_checksum", "damaged"), ("server_error", "having trouble"),
                           ("invalid_key", "does not recognise"), ("wrong_platform", "does not match"), ("bad_manifest_file", "does not match its checksum"), ("link_expired", "expired")):
            self.lic.mode = mode
            self.lic.rebuild()
            self.r.svc.get_encoder("pro", key=F.GOOD_KEY)
            pkg = self.wait_pkg()
            self.assertEqual(pkg["phase"], "failed", mode)
            self.assertIn(frag, pkg["message"], mode)
            self.assertNotIn(SECRET, json.dumps(pkg))
            self.assertEqual(self.r.svc.state()["encoders"], [], mode)
        self.lic.mode = "ok"
        self.lic.rebuild()
        self.r.svc.get_encoder("pro")                                           # the key was stored by the first call: retry needs no typing
        self.assertEqual(self.wait_pkg()["phase"], "done")

    def test_offline_failure_offers_retry(self):
        self.lic.stop()
        self.r.svc.get_encoder("free")
        pkg = self.wait_pkg()
        self.assertEqual((pkg["phase"], pkg["error"]["code"]), ("failed", "offline"))
        self.lic = F.FakeLicence(os.path.join(self.d, "pkg2")).start()
        os.environ["PXA_LICENCE_URL"] = self.lic.url
        self.r.svc.get_encoder("free")
        self.assertEqual(self.wait_pkg()["phase"], "done")

    def test_only_one_download_at_a_time(self):
        self.r.svc.pkg = {"running": True, "phase": "downloading"}
        with self.assertRaises(E.EncodeError):
            self.r.svc.get_encoder("free")
        self.r.svc.pkg = {"running": False, "phase": "idle"}

    def test_a_supporter_plan_locks_the_tiers_it_does_not_include(self):
        self.r.svc.get_encoder("pro", key=F.GOOD_KEY)
        self.wait_pkg()
        p = self.r.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})
        by = {r["key"]: r for r in p["tiers"]}
        self.assertTrue(by["pxqn4"]["available"])
        self.assertEqual(by["pxqn1"]["locked_reason"], "Not included in your plan")
        self.assertIn("PXQN4", p["plan_note"])
        c = self.r.svc.checks({"source": "tiny/Tiny-Qwen3", "cards": [0], "tier": "pxqn1", "work_dir": os.path.join(self.d, "w"), "out_dir": self.r.models})
        self.assertFalse(c["can_start"])
        self.assertIn("not included in your plan", c["refusal"].lower())

    def test_update_prompt_once_a_day_and_one_click(self):
        self.r.svc.get_encoder("free")
        self.wait_pkg()
        self.r.svc.get_encoder("pro", key=F.GOOD_KEY)
        self.wait_pkg()
        u = self.r.svc.update_check(force=True)
        self.assertEqual(u["available"], [])
        self.lic.free_build, self.lic.pro_build = "b-free-3", "b-pro-3"
        self.lic.rebuild()
        n = len(self.lic.requests)
        self.assertEqual(self.r.svc.update_check()["available"], [])           # asked less than a day ago: no network call
        self.assertEqual(len(self.lic.requests), n)
        u = self.r.svc.update_check(force=True)
        self.assertEqual({(x["edition"], x["build_id"]) for x in u["available"]}, {("pro", "b-pro-3"), ("free", "b-free-3")})
        self.assertEqual(self.r.svc.state()["update"]["available"][0]["build_id"], "b-pro-3")
        self.r.svc.get_encoder("pro", update=True)
        self.wait_pkg()
        u = self.r.svc.update_check(force=True)
        self.assertEqual([x["edition"] for x in u["available"]], ["free"])     # the Pro update is installed; Free still offered
        # an old throttle stamp makes the next call ask again
        self.r.host.cfg["encode"]["update"]["checked"] = time.time() - 90000
        self.lic.pro_build = "b-pro-4"
        self.lic.rebuild()
        self.assertTrue(any(x["build_id"] == "b-pro-4" for x in self.r.svc.update_check()["available"]))

    def test_update_check_offline_is_quiet(self):
        self.r.svc.get_encoder("free")
        self.wait_pkg()
        self.lic.stop()
        u = self.r.svc.update_check(force=True)
        self.assertEqual(u["available"], [])
        self.assertIn("Cannot reach the licence server", u["error"])

    def test_update_check_without_encoders_asks_nobody(self):
        n = len(self.lic.requests)
        self.assertEqual(self.r.svc.update_check(force=True)["available"], [])
        self.assertEqual(len(self.lic.requests), n)


class KeyHandling(unittest.TestCase):
    def test_set_validate_mask_forget(self):
        r = Rig(with_encoder=False, key=False)
        with self.assertRaises(E.EncodeError) as cm:
            r.svc.set_key("not a key")
        self.assertIn("starts with pxk1.", str(cm.exception))
        st = r.svc.set_key("  " + GOOD_KEY + " ")
        self.assertTrue(st["key_set"])
        self.assertEqual(r.host.cfg["encode"]["licence_key"], GOOD_KEY)
        self.assertNotIn(SECRET, json.dumps(st))
        self.assertFalse(r.svc.set_key("")["key_set"])
        self.assertNotIn("licence_key", r.host.cfg["encode"])

    def test_key_reaches_the_encoder_only_through_the_environment(self):
        r = Rig("pro")
        j = r.wait(r.svc.start(r.body())["id"])
        self.assertEqual(j["status"], "done")
        for line in r.svc.log_view(j["id"])["lines"]:
            if line.startswith("$ "):
                self.assertNotIn("pxk1", line)

    def test_free_encoder_never_gets_the_key(self):
        env = AD.run_env({"edition": "free"}, GOOD_KEY, "https://x")
        self.assertEqual(env, {})


class AddEncoder(unittest.TestCase):
    def test_picker_validates_with_info_json_and_unlocks_without_restart(self):
        r = Rig("free")
        pro = F.make_fake_encoder(os.path.join(r.dir, "dl", "pro"), "pro", build_id="b-new")
        st = r.svc.add_encoder(os.path.dirname(pro))                           # a folder works too
        self.assertEqual(st["edition"], "pro")
        self.assertIn(pro, r.host.cfg["encode"]["extra"])
        p = r.svc.plan({"source": "tiny/Tiny-Qwen3", "cards": [0]})
        self.assertTrue(p["recommended"][0].startswith("pxqn"))

    def test_refusals(self):
        r = Rig("free")
        d = os.path.join(r.dir, "dl")
        os.makedirs(d, exist_ok=True)
        cases = []
        junk = os.path.join(d, "pxqe-broken")
        F.make_fake_encoder(os.path.join(d, "broken"), "pro", info_broken="garbage")
        cases.append((os.path.join(d, "broken", "pxqe"), "does not look like a working PXA Quantizer"))
        cases.append((os.path.join(d, "missing"), "does not exist"))
        cases.append((d, "no pxqe command line"))
        open(os.path.join(d, "x.tar.gz"), "wb").write(b"x")
        cases.append((os.path.join(d, "x.tar.gz"), "packed download"))
        open(os.path.join(d, "notpxqe.sh"), "w").write("#!/bin/sh\necho hi\n")
        os.chmod(os.path.join(d, "notpxqe.sh"), 0o755)
        cases.append((os.path.join(d, "notpxqe.sh"), "not called pxqe"))
        cases.append(("", "Give the path"))
        for path, frag in cases:
            with self.assertRaises(E.EncodeError) as cm:
                r.svc.add_encoder(path)
            self.assertIn(frag, str(cm.exception), path)
        self.assertEqual(len(r.svc.state()["encoders"]), 1)                   # nothing was added

    def test_an_arbitrary_program_is_never_run_by_the_picker(self):
        r = Rig("free")
        marker = os.path.join(r.dir, "ran.flag")
        evil = os.path.join(r.dir, "evil")
        with open(evil, "w") as f:
            f.write("#!/bin/sh\ntouch %s\n" % marker)
        os.chmod(evil, 0o755)
        with self.assertRaises(E.EncodeError):
            r.svc.add_encoder(evil)
        self.assertFalse(os.path.exists(marker))

    def test_expired_pro_state_is_shown_and_free_still_works(self):
        r = Rig("pro", licence={"state": "expired", "user": "u", "encodes_left": 0})
        free = F.make_fake_encoder(os.path.join(r.dir, "freeone"), "free", build_id="bf")
        r.svc.add_encoder(free)
        st = r.svc.state()
        pro = [e for e in st["encoders"] if e["edition"] == "pro"][0]
        self.assertEqual(pro["licence"]["state"], "expired")
        r.svc.use(free)
        j = r.wait(r.svc.start(r.body("pxq3"))["id"])
        self.assertEqual(j["status"], "done", j["error"])


class DetectionEdgeCases(unittest.TestCase):
    def test_free_pro_broken_json_missing_in_one_state(self):
        r = Rig("free")
        d = os.path.join(r.dir, "more")
        pro = F.make_fake_encoder(os.path.join(d, "pro"), "pro", build_id="bp")
        bad = F.make_fake_encoder(os.path.join(d, "bad"), "free", info_broken="garbage")
        ghost = os.path.join(d, "ghost", "pxqe")
        r.host.cfg["encode"]["extra"] = [r.cli, pro, bad, ghost]
        st = r.svc.rescan()
        by = {os.path.dirname(e["path"]): e for e in st["encoders"]}
        self.assertTrue(by[os.path.dirname(pro)]["ok"])
        self.assertFalse(by[os.path.dirname(bad)]["ok"])
        self.assertIn("valid JSON", by[os.path.dirname(bad)]["error"])
        self.assertNotIn(os.path.dirname(ghost), by)                            # a path that does not exist is simply not listed
        self.assertEqual(st["edition"], "pro")
        self.assertEqual(len([e for e in st["encoders"] if e["ok"]]), 2)

    def test_preferred_choice_survives_a_rescan_and_a_vanished_one_falls_back(self):
        r = Rig("free")
        pro = F.make_fake_encoder(os.path.join(r.dir, "p2"), "pro", build_id="bp")
        r.svc.add_encoder(pro)
        self.assertEqual(r.svc.rescan()["edition"], "pro")
        r.svc.use(r.cli)
        self.assertEqual(r.svc.rescan()["edition"], "free")
        shutil.rmtree(os.path.dirname(r.cli))
        self.assertEqual(r.svc.rescan()["edition"], "pro")

    def test_a_checked_licence_survives_a_rescan_for_a_while(self):
        r = Rig("pro")
        self.assertEqual(r.svc.state()["encoders"][0]["licence"]["state"], "unchecked")
        self.assertEqual(r.svc.licence_check()["encoders"][0]["licence"]["state"], "valid")
        self.assertEqual(r.svc.rescan()["encoders"][0]["licence"]["state"], "valid")           # the offline probe says unchecked; the page keeps what the server said
        self.assertEqual(r.svc.rescan()["encoders"][0]["licence"]["encodes_left"], 4)
        r.svc._lic_seen[r.cli] = (time.time() - 2000, r.svc._lic_seen[r.cli][1])               # ... but not forever
        self.assertEqual(r.svc.rescan()["encoders"][0]["licence"]["state"], "unchecked")

    def test_licence_check_button(self):
        r = Rig("pro", licence={"state": "valid", "encodes_left": 4}, licence_check={"state": "revoked"})
        self.assertEqual(r.svc.state()["encoders"][0]["licence"]["state"], "valid")
        self.assertEqual(r.svc.licence_check()["encoders"][0]["licence"]["state"], "revoked")
        with self.assertRaises(E.EncodeError):
            Rig(with_encoder=False).svc.licence_check()


# =====================================================================================================================
class GpuRuntime(unittest.TestCase):
    """The shared GPU runtime pack (cuBLAS / cuSOLVER and what they load): the signed download and its safety, the wrapper contract the fake encoder
    emulates (`info --json` runtime{}: what is missing and the fix, PXQE_RUNTIME_DIR / PXQE_ENGINE, an older wrapper through LD_LIBRARY_PATH), the
    service flow behind the button 'Download the GPU runtime', and that a job runs once it is there."""

    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.lic = F.FakeLicence(os.path.join(self.d, "pkg")).start()
        os.environ["PXA_LICENCE_URL"] = self.lic.url
        self.enchome = os.path.join(self.d, "enchome")
        os.environ["PXA_ENCODER_HOME"] = self.enchome
        self.root = self.enchome

    def tearDown(self):
        self.lic.stop()
        os.environ.pop("PXA_LICENCE_URL", None)
        os.environ["PXA_ENCODER_HOME"] = os.path.join(TMP, "encoder-home")

    def rig(self, **kw):
        kw.setdefault("needs_runtime", True)
        return Rig("pro", **kw)

    def wait_rt(self, r, timeout=60):
        t0 = time.time()
        while r.svc.rtpkg.get("running") and time.time() - t0 < timeout:
            time.sleep(0.03)
        return r.svc.state()["runtime"]["job"]

    # ---- the download itself
    def test_signed_statement_download_install(self):
        m = PK.runtime_latest(self.lic.url)
        self.assertEqual((m["edition"], m["build_id"], m["cuda"]), ("runtime", "cuda12.8.1-r1", "12.8.1"))
        self.assertEqual(m["libs"][:1], ["libcublas.so.12"])
        self.assertTrue(m["url"].startswith(self.lic.url + "/v1/runtime/download/"))
        self.assertIsNone(PK.installed_runtime(self.root))
        info = PK.install_runtime_from_manifest(m, root=self.root, work=os.path.join(self.d, "work"))
        self.assertEqual((info["id"], info["cuda"]), ("cuda12.8.1-r1", "12.8.1"))
        self.assertEqual(info["lib_dir"], os.path.join(self.root, "runtime", "cuda12.8.1-r1", "lib"))
        for n in PK.RUNTIME_NEEDS + ("libcublasLt.so.12", "libcusparse.so.12", "libnvJitLink.so.12"):
            self.assertTrue(os.path.isfile(os.path.join(info["lib_dir"], n)), n)
        self.assertTrue(os.path.isfile(os.path.join(info["dir"], "LICENSE-NVIDIA-CUDA-EULA.txt")))          # the licence notice stays in the pack
        self.assertEqual(os.listdir(os.path.join(self.d, "work")), [])                                         # the archive is deleted after install
        self.assertEqual(os.listdir(os.path.join(self.root, "runtime")), ["cuda12.8.1-r1"])                    # no half-installed leftover
        self.assertTrue(all(r["body"] == "" for r in self.lic.requests))                                       # public: no key, no body anywhere
        self.assertEqual(self.lic.rt_downloads, 1)
        # the installed pack is found again cheaply, and a damaged one is not
        self.assertEqual(PK.installed_runtime(self.root)["id"], "cuda12.8.1-r1")
        os.truncate(os.path.join(info["lib_dir"], "libcusolver.so.11"), 3)
        self.assertIsNone(PK.installed_runtime(self.root))

    def test_a_newer_pack_replaces_the_older_one(self):
        PK.install_runtime_from_manifest(PK.runtime_latest(self.lic.url), root=self.root, work=os.path.join(self.d, "w"))
        ans, _ = F.build_runtime_pack(os.path.join(self.d, "pkg"), rid="cuda12.9.0-r1")
        self.lic.runtime = (ans, os.path.join(self.d, "pkg", "pxqe-runtime-cuda12.9.0-r1-linux-x86_64-cu12.tar.xz"))
        info = PK.install_runtime_from_manifest(PK.runtime_latest(self.lic.url), root=self.root, work=os.path.join(self.d, "w"))
        self.assertEqual(info["id"], "cuda12.9.0-r1")
        self.assertEqual(os.listdir(os.path.join(self.root, "runtime")), ["cuda12.9.0-r1"])

    def test_nothing_hostile_or_wrong_installs(self):
        cases = (("tamper", "bad_package", "does not match its checksum"), ("missing_lib", "bad_package", "does not contain libcusolver.so.11"),
                 ("wrong_id", "bad_package", "id differs"), ("bad_checksum", "bad_checksum", "damaged"), ("bad_signature", "bad_signature", "NOT installed"),
                 ("none", "no_package", "no GPU runtime pack"))
        for mode, code, frag in cases:
            self.lic.rt_mode = mode
            self.lic.rebuild()
            work = os.path.join(self.d, "work-" + mode)
            with self.assertRaises(PK.PackageError) as cm:
                PK.install_runtime_from_manifest(PK.runtime_latest(self.lic.url), root=self.root, work=work)
            self.assertEqual(cm.exception.code, code, mode)
            self.assertIn(frag, str(cm.exception), mode)
            self.assertIsNone(PK.installed_runtime(self.root), mode)
            rr = os.path.join(self.root, "runtime")
            self.assertEqual(os.listdir(rr) if os.path.isdir(rr) else [], [], mode)                           # not even a hidden .installing folder
            self.assertEqual(os.listdir(work) if os.path.isdir(work) else [], [], mode)                        # the bad download was deleted
        # a statement for another kind of package cannot pass for the runtime, and the other way round
        self.lic.rt_mode = "ok"
        self.lic.rebuild()
        with self.assertRaises(PK.PackageError) as cm:
            PK.check_statement(self.lic.runtime[0], "free", "linux-x86_64", 12)
        self.assertEqual(cm.exception.code, "bad_statement")
        with self.assertRaises(PK.PackageError) as cm:
            PK.check_statement(self.lic.archives["free"][0], "runtime", "linux-x86_64", 12)
        self.assertEqual(cm.exception.code, "bad_statement")
        m = PK.runtime_latest(self.lic.url)
        with self.assertRaises(PK.PackageError):
            PK.install_runtime(self.lic.runtime[1], dict(m, build_id=".."), self.root)                          # an id that is not a folder name

    def test_hostile_runtime_archives_with_a_valid_signature_are_still_refused(self):
        import io
        import tarfile

        def pack(name, members):
            path = os.path.join(self.d, name)
            with tarfile.open(path, "w:xz") as tf:
                for n, kind, data in members:
                    ti = tarfile.TarInfo(n)
                    if kind == "link":
                        ti.type, ti.linkname = tarfile.SYMTYPE, "/bin/sh"
                    ti.size = len(data)
                    ti.mode = 0o755
                    tf.addfile(ti, io.BytesIO(data))
            return path
        cases = (("traversal", [("pxqe-runtime/../../escaped.txt", "f", b"bad")], "unsafe path"),
                 ("absolute", [("/etc/escaped.txt", "f", b"bad")], "unsafe path"),
                 ("symlink", [("pxqe-runtime/lib/libcublas.so.12", "link", b"")], "link or special file"),
                 ("two tops", [("pxqe-runtime/runtime.json", "f", b"{}"), ("other/x", "f", b"x")], "more than one top-level folder"))
        for name, members, frag in cases:
            into = os.path.join(self.d, "into-" + name.replace(" ", "-"))
            os.makedirs(into)
            with self.assertRaises(PK.PackageError) as cm:
                PK._safe_extract_stream(pack(name + ".tar.xz", members), into)
            self.assertIn(frag, str(cm.exception), name)
            self.assertFalse(os.path.exists(os.path.join(self.d, "escaped.txt")))
        ok = os.path.join(self.d, "ok")
        os.makedirs(ok)
        PK._safe_extract_stream(pack("ok.tar.xz", [("pxqe-runtime/lib/libx.so.1", "f", b"abc"), ("pxqe-runtime/NOTICE.txt", "f", b"n")]), ok)
        self.assertEqual(sorted(os.listdir(ok)), ["NOTICE.txt", "lib"])                     # the top folder is stripped
        self.assertEqual(open(os.path.join(ok, "lib", "libx.so.1"), "rb").read(), b"abc")
        self.assertTrue(os.access(os.path.join(ok, "lib", "libx.so.1"), os.X_OK))
        many = os.path.join(self.d, "many.tar.xz")
        with tarfile.open(many, "w:xz") as tf:
            for i in range(30):
                ti = tarfile.TarInfo("pxqe-runtime/f%d" % i)
                tf.addfile(ti, io.BytesIO(b""))
        with self.assertRaises(PK.PackageError) as cm:
            PK._safe_extract_stream(many, os.path.join(self.d, "many-into"), max_files=10)
        self.assertIn("too many files", str(cm.exception))

    def test_disk_space_is_checked_with_the_numbers_before_the_download(self):
        m = PK.runtime_latest(self.lic.url)
        real = shutil.disk_usage
        shutil.disk_usage = lambda p: shutil._ntuple_diskusage(10 ** 12, 10 ** 12 - 5000, 5000)
        try:
            with self.assertRaises(PK.PackageError) as cm:
                PK.install_runtime_from_manifest(m, root=self.root, work=os.path.join(self.d, "w"))
        finally:
            shutil.disk_usage = real
        self.assertEqual(cm.exception.code, "disk_full")
        self.assertIn("GB free", str(cm.exception))
        self.assertEqual(self.lic.rt_downloads, 0)

    def test_the_download_resumes_with_range(self):
        m = PK.runtime_latest(self.lic.url)
        work = os.path.join(self.d, "w")
        os.makedirs(work)
        dest = os.path.join(work, "pxqe-runtime-%s.pack" % m["build_id"])
        with open(self.lic.runtime[1], "rb") as f:
            data = f.read()
        with open(dest + ".part", "wb") as f:
            f.write(data[:200])
        PK.install_runtime_from_manifest(m, root=self.root, work=work)
        self.assertTrue(any((r["headers"].get("range") or "") == "bytes=200-" for r in self.lic.requests))
        self.assertIsNotNone(PK.installed_runtime(self.root))

    # ---- the wrapper contract as the service reads it
    def test_runtime_block_is_parsed_and_an_older_wrappers_sentence_is_understood(self):
        new = AD._runtime({"lib": "unloadable", "detail": "libcublas.so.12, libcusolver.so.11 not found", "missing": ["libcublas.so.12", "libcusolver.so.11"], "fix": "runtime-pack",
                           "driver": True, "resolver": 1, "source": ""})
        self.assertEqual((new["fix"], new["missing"], new["resolver"]), ("runtime-pack", ["libcublas.so.12", "libcusolver.so.11"], 1))
        old = AD._runtime({"lib": "unloadable", "detail": "libcublas.so.12: cannot open shared object file: No such file or directory"})
        self.assertEqual((old["fix"], old["missing"], old["resolver"]), ("runtime-pack", ["libcublas.so.12"], 0))
        drv = AD._runtime({"lib": "unloadable", "detail": "libcuda.so.1: cannot open shared object file"})
        self.assertEqual((drv["fix"], drv["driver"], drv["missing"]), ("driver", False, ["libcuda.so.1"]))
        self.assertIn("NVIDIA driver", AD.runtime_missing_text(drv))
        self.assertIn("libcublas.so.12, libcusolver.so.11", AD.runtime_missing_text(new))
        self.assertEqual(AD._runtime({"lib": "loadable", "source": "pack", "resolver": 1})["source"], "pack")
        self.assertEqual(AD._runtime({"lib": "weird", "fix": "rm -rf"})["fix"], "")                              # only the words the page knows
        self.assertEqual(AD._runtime(None)["lib"], "unknown")

    def test_the_environment_that_lets_the_encoder_find_the_libraries(self):
        pack = {"dir": "/r/runtime/a", "lib_dir": "/r/runtime/a/lib"}
        env = AD.runtime_env({"edition": "pro", "runtime": {"resolver": 1}}, pack, "/eng")
        self.assertEqual(env, {"PXQE_RUNTIME_DIR": "/r/runtime/a", "PXQE_ENGINE": "/eng"})                       # a new wrapper loads them itself: no LD_LIBRARY_PATH
        env = AD.runtime_env({"edition": "pro", "runtime": {"resolver": 0}}, pack, None)
        self.assertEqual(env["PXQE_RUNTIME_DIR"], "/r/runtime/a")
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/r/runtime/a/lib"))                                    # an older one gets the loader path
        self.assertIn("LD_LIBRARY_PATH", AD.runtime_env(None, pack, None, probing=True))                          # a probe does not know the wrapper yet
        self.assertEqual(AD.runtime_env({"edition": "free"}, pack, None), {"PXQE_RUNTIME_DIR": "/r/runtime/a"})
        self.assertEqual(AD.runtime_env({"edition": "pro"}, None, None), {})
        self.assertNotIn("PXQE_KEY", json.dumps(AD.runtime_env({"edition": "pro"}, pack, "/e")))
        f = AD.explain_failure("pxqe: cannot load the encoder library. The Pro encoder needs the NVIDIA CUDA 12 libraries (cuBLAS and cuSOLVER) and could not find libcublas.so.12")
        self.assertEqual(f["code"], "runtime_missing")
        self.assertIn("Download the GPU runtime", f["hint"])
        self.assertEqual(AD.explain_failure("libcuda.so.1: cannot open shared object file")["code"], "no_driver")

    # ---- the service: need -> button -> install -> it works
    def test_missing_runtime_is_named_then_one_download_fixes_it_and_a_job_runs(self):
        r = self.rig()
        st = r.svc.state()
        rt = st["runtime"]
        self.assertTrue(rt["applies"] and rt["need"])
        self.assertEqual((rt["lib"], rt["fix"], rt["missing"]), ("unloadable", "runtime-pack", ["libcublas.so.12", "libcusolver.so.11"]))
        self.assertIn("libcublas.so.12, libcusolver.so.11", rt["text"])
        self.assertIsNone(rt["installed"])
        self.assertEqual(self.lic.requests, [])                                                                   # state() never touches the network
        self.assertFalse(AD.stages_not_ready(st["encoders"][0], ["encode"]) == [])
        # the checks say it once, with the button, and not again under Tools
        c = r.svc.checks(r.body("pxqn4"))
        self.assertFalse(c["can_start"])
        ck = {x["id"]: x for x in c["checks"]}
        self.assertEqual((ck["runtime"]["status"], ck["runtime"]["action"]), ("bad", "runtime"))
        self.assertIn("Download the GPU runtime once (about", ck["runtime"]["fix"])
        self.assertEqual(ck["tools"]["status"], "ok", ck["tools"])
        # the offer carries the size the button says
        v = r.svc.runtime_view(fetch=True)
        self.assertEqual((v["offer"]["id"], v["offer"]["cuda"]), ("cuda12.8.1-r1", "12.8.1"))
        self.assertTrue(v["offer"]["size_h"].endswith(("B", "KiB", "MiB", "GiB")))
        self.assertTrue(any(x["path"].startswith("/v1/runtime/latest") for x in self.lic.requests))
        # one click
        self.assertEqual(r.svc.get_runtime(), {"ok": True})
        job = self.wait_rt(r)
        self.assertEqual(job["phase"], "done", job)
        self.assertIn("The Pro encoder can start now", job["message"])
        rt = r.svc.state()["runtime"]
        self.assertTrue(rt["ok"] and not rt["need"])
        self.assertEqual((rt["source"], rt["installed"]["id"]), ("pack", "cuda12.8.1-r1"))
        self.assertEqual(r.svc.state()["encoders"][0]["runtime"]["lib"], "loadable")
        c = r.svc.checks(r.body("pxqn4"))
        self.assertTrue(c["can_start"], c["refusal"])
        self.assertNotIn("runtime", {x["id"]: x for x in c["checks"]})
        # and a job runs: the encoder it starts is given the folder
        j = r.svc.start(r.body("pxqn4"))
        self.assertEqual(r.wait(j["id"])["status"], "done")
        self.assertTrue(all("pxk1" not in x["path"] and "pxk1" not in x["body"] for x in self.lic.requests if "/v1/runtime/" in x["path"]))

    def test_an_older_wrapper_gets_the_folder_through_the_loader_path(self):
        r = self.rig(old_wrapper=True)
        rt = r.svc.state()["runtime"]
        self.assertTrue(rt["need"])
        self.assertEqual((rt["fix"], rt["missing"]), ("runtime-pack", ["libcublas.so.12"]))                      # all the loader said
        r.svc.get_runtime()
        self.assertEqual(self.wait_rt(r)["phase"], "done")
        rt = r.svc.state()["runtime"]
        self.assertTrue(rt["ok"], rt)
        self.assertEqual(r.svc.state()["encoders"][0]["runtime"]["resolver"], 0)
        j = r.svc.start(r.body("pxqn4"))
        self.assertEqual(r.wait(j["id"])["status"], "done")                                                        # LD_LIBRARY_PATH reached the encode run

    def test_the_engine_install_is_used_when_it_has_the_libraries(self):
        r = self.rig()
        os.makedirs(os.path.join(r.engine, "lib"))
        for n in ("libcublas.so.12", "libcusolver.so.11"):
            open(os.path.join(r.engine, "lib", n), "w").close()
        r.svc.rescan()
        rt = r.svc.state()["runtime"]
        self.assertTrue(rt["ok"] and not rt["need"], rt)
        self.assertEqual(rt["source"], "engine")                                                                   # no download offered
        self.assertEqual(self.lic.requests, [])
        self.assertTrue(r.svc.checks(r.body("pxqn4"))["can_start"])

    def test_no_driver_is_said_plainly_and_offers_no_download(self):
        r = Rig("pro", runtime={"lib": "unloadable", "detail": "libcuda.so.1: cannot open shared object file", "missing": ["libcuda.so.1"], "fix": "driver", "driver": False, "resolver": 1})
        rt = r.svc.state()["runtime"]
        self.assertFalse(rt["need"])
        self.assertIn("No NVIDIA driver was found", rt["text"])
        c = r.svc.checks(r.body("pxqn4"))
        ck = [x for x in c["checks"] if x["id"] == "runtime"][0]
        self.assertEqual((ck["label"], ck["status"], ck.get("action")), ("NVIDIA driver", "bad", None))
        self.assertIn("525", ck["fix"])
        self.assertFalse(c["can_start"])

    def test_free_has_no_use_for_it(self):
        r = Rig("free")
        rt = r.svc.state()["runtime"]
        self.assertEqual((rt["applies"], rt["need"], rt["ok"], rt["text"]), (False, False, False, ""))
        with self.assertRaises(E.EncodeError) as cm:
            r.svc.get_runtime()
        self.assertIn("Only the Pro encoder", str(cm.exception))

    def test_failures_are_plain_sentences_and_leave_the_need_in_place(self):
        r = self.rig()
        for mode, frag in (("tamper", "does not match its checksum"), ("bad_signature", "NOT installed"), ("bad_checksum", "damaged"), ("none", "no GPU runtime pack")):
            self.lic.rt_mode = mode
            self.lic.rebuild()
            r.svc._rt_offer = None
            r.svc.get_runtime()
            job = self.wait_rt(r)
            self.assertEqual(job["phase"], "failed", mode)
            self.assertIn(frag, job["message"], mode)
            self.assertTrue(r.svc.state()["runtime"]["need"], mode)
        self.lic.mode = "server_error"
        r.svc.get_runtime()
        self.assertIn("having trouble", self.wait_rt(r)["message"])
        self.lic.stop()
        r.svc.get_runtime()
        job = self.wait_rt(r)
        self.assertEqual((job["phase"], job["error"]["code"]), ("failed", "offline"))
        self.assertIn("Free encoder", job["message"])

    def test_only_one_runtime_download_at_a_time_and_cancel_resumes(self):
        r = self.rig()
        self.lic.rt_bulk, self.lic.rt_delay = 3 << 20, 0.01
        self.lic.rebuild()
        r.svc.get_runtime()
        with self.assertRaises(E.EncodeError):
            r.svc.get_runtime()
        time.sleep(0.3)
        r.svc.cancel_runtime()
        job = self.wait_rt(r)
        self.assertEqual((job["phase"], job["error"]["code"]), ("failed", "cancelled"))
        self.assertTrue(r.svc.state()["runtime"]["need"])
        self.lic.rt_delay = 0.0
        r.svc.get_runtime()                                                                                        # the same button again: it resumes
        self.assertEqual(self.wait_rt(r)["phase"], "done")

    def test_a_runtime_that_installs_but_does_not_help_says_so(self):
        r = self.rig(needs_runtime=False, runtime={"lib": "unloadable", "detail": "libcublas.so.12: cannot open shared object file"})
        r.svc.get_runtime()
        job = self.wait_rt(r)
        self.assertEqual(job["phase"], "installed_but", job)
        self.assertIn("still cannot load", job["message"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
