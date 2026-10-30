#!/usr/bin/env python3
"""Control-side patches from the 10-16 Control review (todo control-review-patches). No GPU, no model load:
a fake engine binary answers PXA_EXPLAIN, a fake GPU table stands in for the cards, tiny GGUFs for the files.

  #1  the engine's env for its own -sm pick rides with the argv (Flash-Next on a P100 pair)
  #9  expert-cache route + Overdrive class: --spec-type none is NOT forced (it defeated PXA_XCACHE_MTP); with
      PXA_NO_OVERDRIVE=1 or a non-Overdrive MTP file it is still forced (dry run, patched header, no model load)
  #12 .GGUF / .Gguf files are found by the model scan and accepted by validate_launch
  #8  the user's shell PXA_*/PXQ* wins over the launcher's env (Control Seat.start, the CLI/TUI merge helper)
  #7  a card below compute capability 6.0 is refused with a clear message (was 'Expect it to work')
  #6  build-linux.sh / build-macos.sh stamp the version from VERSION ('tag: v3.1' -> v3.1, not tagv3.1)
  #2  no Overdrive switch table in the launcher any more (the engine applies the PXQN library's recipe at startup,
      2026-10-19); every preset carries only the force switch, the box rows and the site rows
usage: python3 tests/test-pxa-launch-control-review.py    exit 0 = PASS
"""
import os, shutil, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = os.path.join(HERE, "..", "tools", "pxa-launch.py")

FAKE_ENGINE = """#!/bin/sh
case "$1" in --version) echo "fake 1"; exit 0;; esac
if [ "$PXA_EXPLAIN" = "1" ]; then
cat <<'J'
PXA_EXPLAIN_JSON {"levers_declared":9,"level":2,"picks":{"sm":{"value":"%(sm)s","status":"MEASURED","why":"fake"}},"env":{%(env)s}}
J
fi
exit 0
"""
FN_ENV = ("PXA_TSPLIT_QWEN4EXP_HC PXA_TSPLIT_UNPROVEN_ARCH PXA_TSPLIT_QWEN4EXP_PAIRLOCAL "
          "PXA_TSPLIT_WORKERS PXA_TSPLIT_REPLAY PXA_QWEN4EXP_HC_FUSED").split()


def tiny_gguf(path, arch="qwen35", blk=False):
    sys.path.insert(0, os.path.join(HERE, "..", "gguf-py"))
    import numpy as np
    import gguf
    w = gguf.GGUFWriter(path, arch)
    w.add_block_count(1); w.add_context_length(256); w.add_embedding_length(64)
    w.add_feed_forward_length(128); w.add_head_count(4); w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5); w.add_file_type(0)
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float32))
    if blk:
        w.add_tensor("blk.0.attn_norm.weight", np.zeros((64,), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def explain(tmp, gpus="2x600", sm="tensor", env=FN_ENV, args=(), extra_env=None, model=None):
    """run the launcher's --explain against the fake engine; -> (stdout+stderr, argv list, env dict from the 'env:' line)"""
    eng = os.path.join(tmp, "eng")
    os.makedirs(os.path.join(eng, "bin"), exist_ok=True)
    exe = os.path.join(eng, "bin", "llama-server")
    with open(exe, "w") as f:
        f.write(FAKE_ENGINE % {"sm": sm, "env": ",".join('"%s":"1"' % k for k in env)})
    os.chmod(exe, 0o755)
    mdl = model or os.path.join(tmp, "t.gguf")
    if not os.path.exists(mdl):
        tiny_gguf(mdl)
    e = {k: v for k, v in os.environ.items() if not k.startswith("PXA_") and k not in ("LD_LIBRARY_PATH",)}
    e.update(PXA_ENGINE_DIR=eng, PXA_LAUNCH_FAKE_GPUS=gpus, HOME=tmp)
    e.update(extra_env or {})
    r = subprocess.run([sys.executable, LAUNCH, "--explain", "--gpus", ",".join(str(i) for i in range(sum(int(x.split("x")[0]) if "x" in x else 1 for x in gpus.split(",")))),
                        "--model", mdl, "-c", "256"] + list(args),
                       env=e, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    cmd = next((l.split("command:", 1)[1].split() for l in out.splitlines() if l.strip().startswith("command:")), [])
    envs = {}
    for l in out.splitlines():
        if l.strip().startswith("env:"):
            for tok in l.split("env:", 1)[1].split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    envs[k] = v
    return out, cmd, envs


class Item1EngineEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxa-ctl-review-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_flashnext_p100_pair_gets_the_six_env_with_sm_tensor(self):
        out, cmd, envs = explain(self.tmp)
        self.assertIn("-sm", cmd, out)
        self.assertEqual(cmd[cmd.index("-sm") + 1], "tensor", out)
        for k in FN_ENV:
            self.assertEqual(envs.get(k), "1", f"{k} missing from the launcher env:\n{out}")

    def test_layer_pick_emits_none_of_them(self):
        out, cmd, envs = explain(self.tmp, sm="layer", env=[])
        self.assertEqual(cmd[cmd.index("-sm") + 1], "layer", out)
        for k in FN_ENV:
            self.assertNotIn(k, envs)

    def test_user_typed_layer_overrides_the_pick_and_the_env_stays_home(self):
        # the final -sm (layer, typed) is not the engine's pick (tensor): its env must not ride along
        out, cmd, envs = explain(self.tmp, args=["--sm", "layer"])
        self.assertEqual(cmd[cmd.index("-sm") + 1], "layer", out)
        for k in FN_ENV:
            self.assertNotIn(k, envs, out)


def load_launcher():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pxa_launch_ctl_review", LAUNCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Item2OverdriveRecipeNotInTheOpenTree(unittest.TestCase):
    """Since 2026-10-19 (todo overdrive-levers-into-closed-lib) the launcher carries no Overdrive switch table: the
    engine applies the PXQN library's recipe at startup. Every preset carries only the force switch, the box rows
    (CPU pool, NUMA bind) and the site rows - not one recipe row."""
    # PXA_XCACHE_RING_LEND=0 (one-V100 preset, 4.0 RC2): an open catalog lever set OFF for that class (a measured
    # per-class override: prefill), not a closed recipe row; the closed-lever text guard still covers every preset.
    ALLOWED = ("PXA_OVERDRIVE=1", "PXA_XCACHE_CPUPOOL_CPUS=", "PXA_NUMA_BIND=", "GGML_CUDA_NO_PINNED=", "PXA_PLE_MMAP=",
               "CUDA_DEVICE_ORDER=", "PXA_XCACHE_RING_LEND=0")

    def test_launcher_has_no_switch_table_and_presets_carry_only_the_switch(self):
        m = load_launcher()
        self.assertFalse(hasattr(m, "OVERDRIVE_ENV"))
        self.assertEqual(m.OVERDRIVE_SWITCH, "PXA_AUTO_OVERDRIVE")
        pdir = os.path.join(HERE, "..", "presets")
        presets = [f for f in os.listdir(pdir) if f.startswith("pxa-overdrive-flashnext") and f.endswith(".env")]
        self.assertTrue(presets)
        for f in presets:
            with open(os.path.join(pdir, f)) as fh:
                rows = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
            self.assertIn("PXA_OVERDRIVE=1", rows, f)
            for r in rows:
                self.assertTrue(r.startswith(self.ALLOWED), (f, r))


class Item6VersionStamp(unittest.TestCase):
    CASES = {"tag:              v3.1\n": "v3.1", "tag: v4.0-pre\nrest\n": "v4.0-pre", "v4.0\n": "v4.0",
             "TAG:v4.0 (beta)\n": "v4.0", "4.0.1\n": "4.0.1"}

    def test_both_build_scripts_stamp_the_version_token(self):
        for script in ("build-linux.sh", "build-macos.sh"):
            with open(os.path.join(HERE, "..", "packaging", "control", script)) as f:
                line = next(l for l in f if l.startswith("VER="))
            for text, want in self.CASES.items():
                tmp = tempfile.mkdtemp(prefix="pxa-ver-")
                try:
                    with open(os.path.join(tmp, "VERSION"), "w") as f:
                        f.write(text)
                    got = subprocess.run(["bash", "-c", f'W={tmp}; {line}\necho "$VER"'], capture_output=True,
                                         text=True).stdout.strip()
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                self.assertEqual(got, want, f"{script} {text!r}")


class Item7OldCards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxa-ctl-review-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sm52_is_refused_clearly(self):
        out, cmd, _ = explain(self.tmp, gpus="1x520", sm="layer", env=[])
        self.assertIn("[R-34]", out)
        self.assertIn("below compute capability 6.0", out)
        self.assertIn("5.2", out)
        self.assertNotIn("Expect it to work", out)
        self.assertEqual(cmd, [], "no command may be printed for a card the engine cannot run")

    def test_mixed_pair_with_one_old_card_is_refused(self):
        out, cmd, _ = explain(self.tmp, gpus="520,600", sm="layer", env=[])
        self.assertIn("[R-34]", out)
        self.assertEqual(cmd, [])

    def test_pascal_and_volta_still_plan(self):
        for g in ("1x600", "1x610", "1x700"):
            out, cmd, _ = explain(self.tmp, gpus=g, sm="layer", env=[])
            self.assertNotIn("[R-34]", out, g)
            self.assertTrue(cmd, g + "\n" + out)


class Item8UserEnvWins(unittest.TestCase):
    def run_seat(self, shell, launcher_env, lever_env):
        sys.path.insert(0, os.path.join(HERE, "..", "tools"))
        import pxa_control as C
        L = load_launcher()
        saved = dict(os.environ)
        try:
            os.environ.update(shell)
            seat = C.Seat(L)
            code = ("import os;print('E', *[k + '=' + os.environ.get(k, '-') for k in "
                    "('PXA_ENHANCE','PXA_XCACHE_ASYNC','PXA_SPEC','LD_LIBRARY_PATH')])")
            built = (L.Plan(), [sys.executable, "-u", "-c", code], launcher_env, "0", {}, 4096)
            seat.start(built, {"port": 8999, "model": "m", "gpus": [0]}, lever_env)
            seat.proc.wait(timeout=20)
            import time
            time.sleep(0.3)
            lines = [x[1] for x in seat.lines_since(0)]
            return lines, seat
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_shell_pxa_value_beats_the_launcher_and_says_so(self):
        lines, seat = self.run_seat({"PXA_ENHANCE": "0", "LD_LIBRARY_PATH": "/user/lib"},
                                    {"PXA_ENHANCE": "1", "PXA_XCACHE_ASYNC": "1", "LD_LIBRARY_PATH": "/engine/lib"}, {})
        e = next(l for l in lines if l.startswith("E "))
        self.assertIn("PXA_ENHANCE=0", e)               # the user's shell won
        self.assertIn("PXA_XCACHE_ASYNC=1", e)          # not set by the user: the launcher's pick
        self.assertIn("LD_LIBRARY_PATH=/engine/lib", e)  # not a PXA_* name: the launcher's
        self.assertTrue(any("your shell's PXA_ENHANCE=0 is kept" in l for l in lines), lines)
        self.assertEqual(seat.env_shown["PXA_ENHANCE"], "0")

    def test_a_kept_credential_is_named_never_shown(self):
        # review 10-18 night: the "is kept" line and env_shown carried the shell's value verbatim; a name that says
        # credential (TOKEN/KEY/SECRET/PASS/CRED/AUTH) is masked like pxa_explain_footer does for --explain
        sys.path.insert(0, os.path.join(HERE, "..", "tools"))
        import pxa_control as C
        self.assertEqual(C.env_value_shown("PXA_ENHANCE", "1"), "1")
        self.assertEqual(C.env_value_shown("PXA_API_KEY", "shell-secret-value"), "<set, hidden>")
        lines, seat = self.run_seat({"PXA_API_KEY": "shell-secret-value"}, {"PXA_API_KEY": "launcher-value"}, {})
        said = [l for l in lines if "is kept" in l]
        self.assertTrue(said and "PXA_API_KEY=<set, hidden> is kept" in said[0], lines)
        self.assertFalse(any("shell-secret-value" in l or "launcher-value" in l for l in said), said)
        self.assertEqual(seat.env_shown["PXA_API_KEY"], "<set, hidden>")

    def test_a_lever_clicked_in_the_ui_still_wins_over_the_shell(self):
        lines, _ = self.run_seat({"PXA_SPEC": "shell"}, {}, {"PXA_SPEC": "ui"})
        self.assertIn("PXA_SPEC=ui", next(l for l in lines if l.startswith("E ")))

    def test_merge_helper(self):
        L = load_launcher()
        base = {"PXA_A": "user", "PXQ_B": "user", "OTHER": "user", "PXA_SAME": "1"}
        kept = L.merge_env_user_wins(base, {"PXA_A": "x", "PXQ_B": "y", "OTHER": "launcher", "PXA_NEW": "n", "PXA_SAME": "1"})
        self.assertEqual(base, {"PXA_A": "user", "PXQ_B": "user", "OTHER": "launcher", "PXA_SAME": "1", "PXA_NEW": "n"})
        self.assertEqual(sorted(k for k, _, _ in kept), ["PXA_A", "PXQ_B"])   # PXA_SAME agreed: nothing to say


class Item12CaseInsensitiveGguf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxa-ctl-review-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scan_lists_upper_and_mixed_case_extensions(self):
        L = load_launcher()
        L.SCAN_CACHE = os.path.join(self.tmp, "modelscan.json")     # never the real state dir
        for n in ("Lower.gguf", "UPPER.GGUF", "Mixed.Gguf"):
            tiny_gguf(os.path.join(self.tmp, n), blk=True)
        entries, _notes = L.scan_models([self.tmp])
        names = {os.path.basename(e["path"]) for e in entries}
        self.assertEqual(names, {"Lower.gguf", "UPPER.GGUF", "Mixed.Gguf"}, names)

    def test_validate_launch_and_model_kind_accept_upper_case(self):
        sys.path.insert(0, os.path.join(HERE, "..", "tools"))
        import pxa_control as C
        L = load_launcher()
        f = os.path.join(self.tmp, "UPPER.GGUF")
        tiny_gguf(f)
        self.assertEqual(L.model_kind(f), "gguf")
        v = C.validate_launch({"gpus": [0], "model": f, "port": 8899}, {0}, set(), [self.tmp])
        self.assertEqual(v["model"], f)

    def test_split_gguf_shard_names_keep_the_case(self):
        L = load_launcher()
        self.assertEqual(L.gguf_shards("/m/X-00001-of-00002.GGUF"),
                         ["/m/X-00001-of-00002.GGUF", "/m/X-00002-of-00002.GGUF"])
        self.assertEqual(L.gguf_shards("/m/X-00001-of-00002.gguf")[1], "/m/X-00002-of-00002.gguf")


DRY9 = '''
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pxa_launch", sys.argv[1])
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
L.SCAN_CACHE = sys.argv[2]
TIER = sys.argv[3]
real = L.model_profile
def fake_profile(path, kind):
    p = real(path, kind)
    p.update(arch="qwen4exp", is_moe=True, deltanet=True, tier=TIER, mtp_tensors=3, n_expert=256,
             expert_bytes=24 * 2**30, n_block=40, ple_bytes=0, kv_bytes_tok=20000, hdr_err=None, n_ctx_train=262144)
    return p
L.model_profile = fake_profile
L.model_bytes = lambda path, kind: 30 * 2**30     # bigger than one 16 GiB card: the expert-cache route
L.host_memory = lambda: (256 * 2**30, 200 * 2**30)
sys.argv = ["pxa-launch", "--explain", "--gpus", "0", "--model", sys.argv[4], "-c", "16384"]
L.main()
'''


class ExplainHidesSecrets(unittest.TestCase):
    """--explain lists the PXA_*/PXQ_* levers set in the environment; a credential's value is never printed"""

    def test_token_value_hidden_lever_value_shown(self):
        import contextlib
        import io
        from unittest import mock
        sys.path.insert(0, os.path.join(HERE, "..", "tools"))
        import pxa_explain_footer as F
        env = {"PXA_BUS_TOKEN": "s3cr3t-value", "PXA_STORE_KEY": "k3y-value", "PXA_ENHANCE": "1"}
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
            F.print_footer(type("A", (), {"gpus": "0"})(), [(0, "Tesla P100", 60, 16000, 100)])
        text = out.getvalue()
        self.assertNotIn("s3cr3t-value", text)
        self.assertNotIn("k3y-value", text)
        self.assertIn("PXA_BUS_TOKEN=<set, hidden>", text)
        self.assertIn("PXA_ENHANCE=1", text)


class Item9XcacheSpec(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxa-ctl-review-")
        self.model = os.path.join(self.tmp, "t.gguf")
        tiny_gguf(self.model)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dry(self, tier="PXQN2", extra_env=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith("PXA_")}
        e.update(PXA_LAUNCH_FAKE_GPUS="1x600", HOME=self.tmp)
        e.update(extra_env or {})
        r = subprocess.run([sys.executable, "-c", DRY9, LAUNCH, os.path.join(self.tmp, "scan.json"), tier, self.model],
                           env=e, capture_output=True, text=True, timeout=300)
        out = r.stdout + r.stderr
        self.assertIn("route:  EXPERT CACHE", out)
        cmd = next(l for l in out.splitlines() if l.strip().startswith("command:"))
        env = next(l for l in out.splitlines() if l.strip().startswith("env:"))
        return out, cmd, env

    def test_overdrive_keeps_mtp_on_the_xcache_route(self):
        # 2026-10-19: the launcher exports the public switch PXA_OVERDRIVE=1 for the class and no recipe row (the
        # engine applies the PXQN library's set at startup); the decision not to force --spec-type none is unchanged
        out, cmd, env = self.dry()
        self.assertNotIn("--spec-type", cmd)
        self.assertIn("PXA_OVERDRIVE=1", env)
        self.assertIn("PXA_XCACHE_CPUPOOL_CPUS=", env)
        self.assertNotIn("PXA_XCACHE_MTP", env)
        self.assertNotIn("PXA_QWEN4EXP_HC_FUSED", env)
        self.assertNotIn("PXA_OVERDRIVE_LIB", env)

    def test_overdrive_off_still_forces_none(self):
        out, cmd, env = self.dry(extra_env={"PXA_NO_OVERDRIVE": "1"})
        self.assertIn("--spec-type none", cmd)
        self.assertNotIn("PXA_XCACHE_MTP", env)
        self.assertNotIn("PXA_OVERDRIVE=1", env)

    def test_a_non_overdrive_tier_still_forces_none(self):
        out, cmd, env = self.dry(tier="PXQ3")
        self.assertIn("--spec-type none", cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
