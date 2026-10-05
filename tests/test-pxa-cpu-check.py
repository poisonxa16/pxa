#!/usr/bin/env python3
"""The CPU preflight that picks lib/ (fast) or lib-compat/ (any x86-64 CPU): tools/pxa-cpu-check.sh (run-server.sh, pxa-launch, the
container entrypoint) and its Python twin in tools/pxa-launch.py (engine_ld_path), against the same cpuinfo fixtures.

No GPU, no engine, no model: a fake install tree with an empty libggml.so in lib/ and lib-compat/.

    python3 tests/test-pxa-cpu-check.py          (wired into CTest as test-pxa-cpu-check)
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_SH = os.path.join(ROOT, "tools", "pxa-cpu-check.sh")
LAUNCH_PY = os.path.join(ROOT, "tools", "pxa-launch.py")

BASE = "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx lm constant_tsc rep_good nopl cpuid pni"
# first line of each fixture is a different CPU's "flags" line, as /proc/cpuinfo writes it
CPUS = {
    "qemu64":      (BASE, ["avx", "avx2", "fma", "f16c"]),
    "nehalem":     (BASE + " ssse3 cx16 sse4_1 sse4_2 popcnt lahf_lm", ["avx", "avx2", "fma", "f16c"]),
    "sandybridge": (BASE + " ssse3 cx16 sse4_1 sse4_2 popcnt aes xsave avx lahf_lm", ["avx2", "fma", "f16c"]),
    "ivybridge":   (BASE + " ssse3 cx16 sse4_1 sse4_2 popcnt aes xsave avx f16c rdrand lahf_lm", ["avx2", "fma"]),
    "haswell":     (BASE + " ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm abm avx2 bmi1 bmi2", []),
    "zen4":        (BASE + " ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c avx2 bmi1 bmi2 avx512f avx512bw", []),
}


def cpuinfo_text(flags):
    return ("processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: test cpu\nflags\t\t: %s\nbogomips\t: 4000.00\n\n"
            "processor\t: 1\nflags\t\t: %s\n" % (flags, flags))


def load_launch():
    spec = importlib.util.spec_from_file_location("pxa_launch_cpu_test", LAUNCH_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class CpuCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pxa-cpucheck-")
        cls.info = {}
        for name, (flags, _miss) in CPUS.items():
            p = os.path.join(cls.tmp, "cpuinfo-" + name)
            with open(p, "w") as f:
                f.write(cpuinfo_text(flags))
            cls.info[name] = p
        cls.noflags = os.path.join(cls.tmp, "cpuinfo-arm")
        with open(cls.noflags, "w") as f:
            f.write("processor\t: 0\nBogoMIPS\t: 50.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n")
        # an install tree: E/lib/libggml.so and E/lib-compat/libggml.so
        cls.E = os.path.join(cls.tmp, "pkg")
        for d in ("lib", "lib-compat", "bin"):
            os.makedirs(os.path.join(cls.E, d))
        for d in ("lib", "lib-compat"):
            open(os.path.join(cls.E, d, "libggml.so"), "w").close()
        cls.E_nocompat = os.path.join(cls.tmp, "pkg-nocompat")
        for d in ("lib", "bin"):
            os.makedirs(os.path.join(cls.E_nocompat, d))
        open(os.path.join(cls.E_nocompat, "lib", "libggml.so"), "w").close()
        cls.L = load_launch()

    # ---- the shell function -------------------------------------------------------------
    def sh(self, cpuinfo, libdir, env_extra=None):
        """-> (rc, LD_LIBRARY_PATH after, stderr, PXA_CPU_LIB_PICKED)"""
        env = {"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": "/prior/lib"}
        if cpuinfo:
            env["PXA_CPUINFO"] = cpuinfo
        env.update(env_extra or {})
        script = '. "%s"; pxa_cpu_pick "%s"; rc=$?; echo "LDP=$LD_LIBRARY_PATH"; echo "PICKED=${PXA_CPU_LIB_PICKED:-}"; exit $rc' % (CHECK_SH, libdir)
        r = subprocess.run(["sh", "-c", script], capture_output=True, text=True, env=env)
        out = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
        return r.returncode, out.get("LDP"), r.stderr, out.get("PICKED")

    def test_sh_fast_cpu_changes_nothing(self):
        for name in ("haswell", "zen4"):
            rc, ldp, err, picked = self.sh(self.info[name], os.path.join(self.E, "lib-compat"))
            self.assertEqual(rc, 0, name)
            self.assertEqual(ldp, "/prior/lib", name)
            self.assertEqual(err, "", name)
            self.assertEqual(picked, "", name)

    def test_sh_old_cpu_picks_compat_with_one_line(self):
        cd = os.path.join(self.E, "lib-compat")
        for name in ("qemu64", "nehalem", "sandybridge", "ivybridge"):
            rc, ldp, err, picked = self.sh(self.info[name], cd)
            self.assertEqual(rc, 0, name)
            self.assertEqual(ldp, cd + ":/prior/lib", name)
            self.assertEqual(picked, "compat", name)
            self.assertEqual(len([l for l in err.splitlines() if l.strip()]), 1, "exactly one line for " + name)
            self.assertIn("lib-compat", err)
            for want in CPUS[name][1]:
                self.assertIn(want, err, name)
            self.assertNotIn("Traceback", err)

    def test_sh_missing_compat_dir_stops_with_one_plain_line(self):
        rc, ldp, err, picked = self.sh(self.info["nehalem"], os.path.join(self.E_nocompat, "lib-compat"))
        self.assertEqual(rc, 1)
        self.assertEqual(len([l for l in err.splitlines() if l.strip()]), 1)
        self.assertIn("no lib-compat folder", err)
        self.assertIn("Illegal instruction", err)

    def test_sh_cannot_tell_does_not_guess(self):
        for p in (self.noflags, os.path.join(self.tmp, "does-not-exist")):
            rc, ldp, err, picked = self.sh(p, os.path.join(self.E, "lib-compat"))
            self.assertEqual((rc, ldp, err, picked), (0, "/prior/lib", "", ""))

    def test_sh_overrides(self):
        cd = os.path.join(self.E, "lib-compat")
        rc, ldp, err, picked = self.sh(self.info["haswell"], cd, {"PXA_CPU_LIB": "compat"})
        self.assertEqual((rc, picked), (0, "compat"))
        self.assertTrue(ldp.startswith(cd + ":"))
        rc, ldp, err, picked = self.sh(self.info["nehalem"], cd, {"PXA_CPU_LIB": "fast"})
        self.assertEqual((rc, ldp, err), (0, "/prior/lib", ""))

    def test_sh_says_it_once_per_process_tree(self):
        cd = os.path.join(self.E, "lib-compat")
        rc, ldp, err, picked = self.sh(self.info["nehalem"], cd, {"PXA_CPU_NOTE_DONE": "1"})
        self.assertEqual((rc, picked, err), (0, "compat", ""))   # picked, silent: the parent already said it

    # ---- the Python twin ----------------------------------------------------------------
    def with_env(self, **kv):
        old = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) if v is not None else os.environ.pop(k, None) for k, v in old.items()])

    def test_py_missing_matches_the_table(self):
        self.with_env(PXA_CPU_LIB=None, PXA_CPU_NOTE_DONE=None)
        for name, (_f, miss) in CPUS.items():
            os.environ["PXA_CPUINFO"] = self.info[name]
            self.assertEqual(self.L.cpu_isa_missing(), miss, name)
        os.environ["PXA_CPUINFO"] = self.noflags
        self.assertEqual(self.L.cpu_isa_missing(), [])
        os.environ["PXA_CPUINFO"] = os.path.join(self.tmp, "nope")
        self.assertEqual(self.L.cpu_isa_missing(), [])

    def test_py_ld_path_puts_compat_first_on_old_cpus_only(self):
        self.with_env(PXA_CPU_LIB=None, PXA_CPU_NOTE_DONE="1", LD_LIBRARY_PATH="/prior/lib")
        os.environ["PXA_CPUINFO"] = self.info["haswell"]
        ldp, _ = self.L.engine_ld_path(self.E)
        self.assertEqual(ldp.split(":")[0], self.E + "/lib")
        self.assertNotIn("lib-compat", ldp)
        os.environ["PXA_CPUINFO"] = self.info["nehalem"]
        ldp, _ = self.L.engine_ld_path(self.E)
        self.assertEqual(ldp.split(":")[:2], [self.E + "/lib-compat", self.E + "/lib"])
        self.assertTrue(ldp.endswith("/prior/lib"))
        # no lib-compat in this install: unchanged (the launcher's engine_runs then reports the illegal instruction)
        ldp, _ = self.L.engine_ld_path(self.E_nocompat)
        self.assertNotIn("lib-compat", ldp)

    def test_py_shell_has_already_said_it_stays_silent(self):
        self.with_env(PXA_CPU_LIB=None, PXA_CPU_NOTE_DONE="1", PXA_CPUINFO=self.info["nehalem"])
        self.L._CPU_NOTE_SHOWN[0] = False
        r = subprocess.run([sys.executable, "-c",
                            "import importlib.util,sys;s=importlib.util.spec_from_file_location('l','%s');m=importlib.util.module_from_spec(s);"
                            "s.loader.exec_module(m);print(m.engine_ld_path('%s')[0])" % (LAUNCH_PY, self.E)],
                           capture_output=True, text=True, env=dict(os.environ))
        self.assertEqual(r.stderr, "")
        self.assertIn("lib-compat", r.stdout)

    def test_py_says_it_once_when_nobody_did(self):
        env = dict(os.environ, PXA_CPUINFO=self.info["nehalem"])
        env.pop("PXA_CPU_NOTE_DONE", None)
        env.pop("PXA_CPU_LIB", None)
        code = ("import importlib.util;s=importlib.util.spec_from_file_location('l','%s');m=importlib.util.module_from_spec(s);"
                "s.loader.exec_module(m);m.engine_ld_path('%s');m.engine_ld_path('%s')" % (LAUNCH_PY, self.E, self.E))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(len([l for l in r.stderr.splitlines() if l.strip()]), 1, r.stderr)
        self.assertIn("lib-compat", r.stderr)

    def test_py_and_sh_agree_on_every_cpu(self):
        self.with_env(PXA_CPU_LIB=None, PXA_CPU_NOTE_DONE="1")
        for name in CPUS:
            os.environ["PXA_CPUINFO"] = self.info[name]
            py_compat = self.L.engine_ld_path(self.E)[0].split(":")[0].endswith("lib-compat")
            _rc, _ldp, _err, picked = self.sh(self.info[name], os.path.join(self.E, "lib-compat"), {"PXA_CPU_NOTE_DONE": "1"})
            self.assertEqual(py_compat, picked == "compat", name)


if __name__ == "__main__":
    unittest.main(verbosity=1)
