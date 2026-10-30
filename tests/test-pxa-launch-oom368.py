#!/usr/bin/env python3
"""bug #368: a 16 GB + 12 GB P100 pair serving the 35B MoE flagship with its vision projector crashed
at the first prompt (CUDA pool on card 0 could not grow; Beta 5). Two launcher picks caused it:

  1. the 2-card recipe row (measured on two EQUAL 16 GB cards) handed its even -ts 1,1 to an
     unequal pair under -sm layer, which partitions BYTES: the 12 GB card's spare room went unused
     and the 16 GB card was packed to its last few hundred MiB;
  2. with a projector on the GPU the engine runs -ub = -b (the multimodal batch rule), so the row's
     -b 8192 -ub 2048 ran a 8192-token ubatch: 4x the compute buffer and GEMM temporaries the row
     was measured with. Then the projector (858 MiB + its compute) loaded on card 0 after the
     placement had judged the fit without it.

The launcher now (a) keeps the row's -ub by emitting -b = -ub when a projector rides on the GPU,
and (b) gives an unequal pair the capacity split, with the projector charged to the first card.
Equal-card seats without a projector get the same command as before, byte for byte.
Pure-function test, no GPU.   usage: test-pxa-launch-oom368.py   exit 0 = PASS"""
import contextlib, importlib.util, io, os, shutil, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("pxl368", os.path.join(HERE, "..", "tools", "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

# the reporter's cards: total MiB, used MiB (free = 15933 / 11851, as in the report's load log)
P16 = (0, "Tesla P100-PCIE-16GB", 60, 16280, 347, "u0")
P12 = (1, "Tesla P100-PCIE-12GB", 60, 12198, 347, "u1")
P16B = (1, "Tesla P100-PCIE-16GB", 60, 16280, 347, "u1")


def recipe(key):
    for r in L.RECIPES:
        if r.key == key:
            return r
    raise KeyError(key)


class Seat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="oom368-")
        cls.model = os.path.join(cls.d, "PXA-Coder-35B-v2-PXQ4.gguf")
        open(cls.model, "wb").close()
        cls.mmproj = os.path.join(cls.d, "mmproj-PXA-Coder-35B-v2-f16.gguf")
        with open(cls.mmproj, "wb") as f:          # sparse: the size is what the launcher reads
            f.truncate(int(857.6 * 1048576))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def args(self, ub=0, b=0, ts=""):
        return type("A", (), {
            "ctx": 65535, "sm": "layer", "model": self.model, "host": "0.0.0.0", "port": 8081,
            "ngl": 999, "np": 1, "ctk": "q4_0", "ctv": "q4_0", "threads": 2, "emit_threads": False,
            "workload": "chat", "ub": ub, "b": b, "ts": ts, "spec": None, "draft_model": None,
            "hot_model": [], "no_mmap": False,
        })()

    def plan(self, engine_ub=None):
        p = L.Plan()
        p.fa = "on"
        p.recipe = recipe("2xpair-flagship-moe")
        if engine_ub:
            p.engine_ac = {"picks": {"b": {"value": "8192", "status": "MEASURED", "why": "cell"},
                                     "ub": {"value": str(engine_ub), "status": "MEASURED", "why": "cell"}}}
        return p

    def build(self, sel, mmproj, a=None, plan=None, env=None):
        buf = io.StringIO()
        old = {k: os.environ.get(k) for k in (env or {})}
        os.environ.update(env or {})
        try:
            with contextlib.redirect_stdout(buf):
                cmd, _env, _ctx = L.build_llama_cmd(plan or self.plan(), a or self.args(), sel,
                                                    {"arch": "qwen35moe", "tier": "PXQ4"}, 65535,
                                                    [2048], mmproj, explain=True)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return cmd, buf.getvalue()

    @staticmethod
    def val(cmd, flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    # ---- the reporter's seat --------------------------------------------------------------
    def test_report_seat_keeps_the_measured_ubatch(self):
        cmd, out = self.build([P16, P12], self.mmproj)
        self.assertEqual(self.val(cmd, "-ub"), "2048")
        self.assertEqual(self.val(cmd, "-b"), "2048", out)   # engine runs -ub = -b with a GPU projector
        self.assertIn("projector", out)

    def test_report_seat_gets_a_capacity_split(self):
        cmd, out = self.build([P16, P12], self.mmproj)
        ts = self.val(cmd, "-ts")
        self.assertIsNotNone(ts)
        self.assertNotEqual(ts, "1,1", out)
        s0, s1 = (int(x) for x in ts.split(","))
        self.assertGreater(s0, s1)

    def test_projector_is_charged_to_the_first_card(self):
        with_p, _ = L.auto_tensor_split([P16, P12], 65535, mmproj_mib=1500)
        without, _ = L.auto_tensor_split([P16, P12], 65535)
        a0 = int(with_p.split(",")[0])
        b0 = int(without.split(",")[0])
        self.assertLess(a0, b0)

    def test_engine_registry_ub_pick_is_capped_too(self):
        cmd, out = self.build([P16, P12], self.mmproj, plan=self.plan(engine_ub=2048))
        self.assertEqual((self.val(cmd, "-b"), self.val(cmd, "-ub")), ("2048", "2048"), out)

    # ---- what must NOT change --------------------------------------------------------------
    def test_equal_pair_without_projector_is_unchanged(self):
        cmd, _ = self.build([P16, P16B], None)
        self.assertEqual((self.val(cmd, "-b"), self.val(cmd, "-ub"), self.val(cmd, "-ts")),
                         ("8192", "2048", "1,1"))

    def test_unequal_pair_without_projector_keeps_the_row_batch(self):
        cmd, _ = self.build([P16, P12], None)
        self.assertEqual((self.val(cmd, "-b"), self.val(cmd, "-ub")), ("8192", "2048"))

    def test_equal_pair_with_projector_keeps_the_even_split(self):
        cmd, _ = self.build([P16, P16B], self.mmproj)
        self.assertEqual(self.val(cmd, "-ts"), "1,1")
        self.assertEqual(self.val(cmd, "-b"), "2048")

    def test_forced_batch_is_kept_and_warned(self):
        cmd, out = self.build([P16, P12], self.mmproj, a=self.args(ub=2048, b=8192))
        self.assertEqual((self.val(cmd, "-b"), self.val(cmd, "-ub")), ("8192", "2048"))
        self.assertIn("projector", out)

    def test_forced_ts_is_kept(self):
        cmd, _ = self.build([P16, P12], self.mmproj, a=self.args(ts="1,1"))
        self.assertEqual(self.val(cmd, "-ts"), "1,1")

    def test_off_switch_restores_the_old_batch(self):
        cmd, _ = self.build([P16, P12], self.mmproj, env={"PXA_LAUNCH_MMPROJ_BATCH": "0"})
        self.assertEqual((self.val(cmd, "-b"), self.val(cmd, "-ub")), ("8192", "2048"))

    def test_gemma_rows_keep_their_batch_with_a_projector(self):
        # a Gemma image is attended non-causally and must sit in one batch: -b 2048 stays
        for key, sel in (("1xp100-gemma4-moe", [P16]), ("2xp100-gemma4-moe", [P16, P16B])):
            p = self.plan()
            p.recipe = recipe(key)
            cmd, _ = self.build(sel, self.mmproj, plan=p)
            self.assertEqual((self.val(cmd, "-b"), self.val(cmd, "-ub")), ("2048", str(p.recipe.ub)), key)

    def test_cap_never_goes_under_the_default_batch(self):
        cmd = ["x", "-b", "8192", "-ub", "256"]
        L.mmproj_batch_cap(cmd, self.mmproj, False)
        self.assertEqual(cmd[2], "2048")

    def test_cpu_projector_keeps_the_row_batch(self):
        p = self.plan()
        p.recipe = recipe("2xpair-flagship-moe")
        old = list(p.recipe.extra)
        p.recipe.extra = old + ["--no-mmproj-offload"]
        try:
            cmd, _ = self.build([P16, P16B], self.mmproj, plan=p)
        finally:
            p.recipe.extra = old
        self.assertEqual(self.val(cmd, "-b"), "8192")


if __name__ == "__main__":
    unittest.main(verbosity=1)
