#!/usr/bin/env python3
"""The compiled-Control stage (packaging/control/stage.py) carries what plugins and the speech service run on in a compiled
Control: the plugin kit (plugin-sdk.zip), on Windows python.org's embeddable Python (python-embed.zip: pinned by sha256,
downloaded once into a cache folder) with Microsoft's C++ runtime DLLs beside python.exe (the speech engines import them
and a Windows without the Visual C++ Redistributable has none), and a VERSION that names the release tag, so a 4.0 build
is not stamped with the repo's last tag (the store refuses a plugin whose min_host is newer than the Control).

    python3 tests/test-pxa-control-stage.py

No network: the stand-in embeddable Python and runtime wheel are served from file:// URLs.
"""
import hashlib
import importlib.util
import os
import pathlib
import re
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
CONTROL = os.path.join(ROOT, "packaging", "control")
TMP = tempfile.mkdtemp(prefix="pxa-control-stage-test-")
TAG = "v4.0.0-beta.3"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)

from pxa_control import core, pyfind  # noqa: E402
from pxa_control import store  # noqa: E402
from pxa_control.plugins import sdk  # noqa: E402


def load_stage():
    spec = importlib.util.spec_from_file_location("pxa_control_stage_t", os.path.join(CONTROL, "stage.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    return st


ST = load_stage()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stand_in_python(d):
    """a zip shaped like python.org's embeddable package -> (file:// url, sha256)."""
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "python-3.13.17-embeddable-amd64.zip")
    with zipfile.ZipFile(p, "w") as z:
        for n in ("python.exe", "python313._pth", "python313.zip", "python313.dll"):
            z.writestr(n, "stand-in %s\n" % n)
    return pathlib.Path(p).as_uri(), sha256(p)


def stand_in_vcrt(d, dlls=("msvcp140.dll", "msvcp140_1.dll")):
    """a zip shaped like the msvc-runtime wheel on PyPI (the DLLs in its data folder, other copies under Scripts) ->
    (file:// url, sha256)."""
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "msvc_runtime-14.51.36231-cp313-cp313-win_amd64.whl")
    data = "msvc_runtime-14.51.36231.data/data/"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("msvc_runtime.cp313-win_amd64.pyd", "stand-in pyd\n")
        for n in dlls + ("concrt140.dll",):
            z.writestr(zipfile.ZipInfo(data + n, date_time=(2026, 6, 10, 15, 49, 34)), "stand-in %s\n" % n)
            z.writestr(data + "Scripts/" + n, "a Scripts copy of %s\n" % n)
    return pathlib.Path(p).as_uri(), sha256(p)


def entries(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


class Stage(unittest.TestCase):
    """one full stage, as build-windows.ps1 runs it: a tag and the embeddable Python."""

    @classmethod
    def setUpClass(cls):
        cls.out = os.path.join(TMP, "stage")
        cls.cache = os.path.join(TMP, "cache")
        url, sha = stand_in_python(os.path.join(TMP, "python-org"))
        vurl, vsha = stand_in_vcrt(os.path.join(TMP, "pypi"))
        saved = ST.EMBED_URL, ST.EMBED_SHA256, ST.VCRT_URL, ST.VCRT_SHA256
        ST.EMBED_URL, ST.EMBED_SHA256, ST.VCRT_URL, ST.VCRT_SHA256 = url, sha, vurl, vsha
        try:
            ST.main([cls.out, "--tag", TAG, "--python-embed", cls.cache])
        finally:
            ST.EMBED_URL, ST.EMBED_SHA256, ST.VCRT_URL, ST.VCRT_SHA256 = saved
        cls.python_org = os.path.join(TMP, "python-org", "python-3.13.17-embeddable-amd64.zip")

    def test_the_stage_carries_the_plugin_kit(self):
        kit = os.path.join(self.out, sdk.KIT_ZIP)
        self.assertTrue(os.path.isfile(kit))
        self.assertEqual(sha256(kit), sdk.build(os.path.join(TMP, "kit-ref.zip")))      # the kit of this tree, as built
        self.assertEqual(ST.plugin_leaks(self.out), [])

    def test_the_version_names_the_release_tag(self):
        with open(os.path.join(self.out, "VERSION")) as f:
            token = core.version_token(f.read())
        self.assertEqual(token, TAG)
        self.assertFalse(store.host_too_old("4.0", version=token))                      # the min_host the store plugins carry

    def test_the_stage_carries_the_pinned_python_with_the_cpp_runtime_beside_python_exe(self):
        got, org = entries(os.path.join(self.out, pyfind.EMBED_ZIP)), entries(self.python_org)
        self.assertEqual({n: got[n] for n in org}, org)                                  # python.org's files, unchanged
        self.assertEqual({n: b for n, b in got.items() if n not in org},
                         {"msvcp140.dll": b"stand-in msvcp140.dll\n", "msvcp140_1.dll": b"stand-in msvcp140_1.dll\n"})


class Version(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)

    def test_without_a_tag_the_repo_version_is_kept(self):
        ST.write_version(self.d, None)
        with open(os.path.join(self.d, "VERSION"), "rb") as a, open(os.path.join(ROOT, "VERSION"), "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_a_tag_must_look_like_a_version(self):
        for bad in ("v4.0; rm -rf /", "latest", "4", "v4.0\nevil"):
            with self.assertRaises(SystemExit, msg=bad):
                ST.write_version(self.d, bad)
        self.assertFalse(os.path.exists(os.path.join(self.d, "VERSION")))


class Embed(unittest.TestCase):
    """python.org's embeddable Python: downloaded once into a cache folder, used only when its sha256 is the pinned one."""

    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.cache = os.path.join(self.d, "cache")
        self.url, self.sha = stand_in_python(os.path.join(self.d, "python-org"))

    def test_downloaded_once_and_checked(self):
        p = ST.fetch_embed(self.cache, url=self.url, sha256=self.sha)
        self.assertEqual(sha256(p), self.sha)
        shutil.rmtree(os.path.join(self.d, "python-org"))                                # python.org gone: the cache serves
        self.assertEqual(ST.fetch_embed(self.cache, url=self.url, sha256=self.sha), p)

    def test_a_download_that_does_not_match_the_pin_is_refused(self):
        with self.assertRaises(SystemExit):
            ST.fetch_embed(self.cache, url=self.url, sha256="0" * 64)
        self.assertEqual(os.listdir(self.cache) if os.path.isdir(self.cache) else [], [])  # nothing kept

    def test_a_cached_copy_that_changed_is_downloaded_again(self):
        p = ST.fetch_embed(self.cache, url=self.url, sha256=self.sha)
        with open(p, "ab") as f:
            f.write(b"changed")
        self.assertEqual(sha256(ST.fetch_embed(self.cache, url=self.url, sha256=self.sha)), self.sha)

    def test_the_pin_is_python_org_and_a_python_plugins_run_on(self):
        m = re.fullmatch(r"https://www\.python\.org/ftp/python/(3)\.(\d+)\.\d+/python-3\.\d+\.\d+-embeddable-amd64\.zip",
                         ST.EMBED_URL)
        self.assertTrue(m, ST.EMBED_URL)
        self.assertTrue(pyfind.PY_MIN <= (int(m.group(1)), int(m.group(2))) <= pyfind.PY_MAX)
        self.assertRegex(ST.EMBED_SHA256, r"^[0-9a-f]{64}$")


class CppRuntime(unittest.TestCase):
    """msvcp140.dll and msvcp140_1.dll (onnxruntime and espeak-ng import them; python.org's package carries only
    vcruntime140): from the msvc-runtime wheel on PyPI, pinned by sha256, put beside python.exe in python-embed.zip."""

    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.cache = os.path.join(self.d, "cache")
        url, sha = stand_in_python(os.path.join(self.d, "python-org"))
        self.embed = ST.fetch_embed(self.cache, url=url, sha256=sha)
        self.url, self.sha = stand_in_vcrt(os.path.join(self.d, "pypi"))

    def test_a_wheel_that_does_not_match_the_pin_is_refused(self):
        with self.assertRaises(SystemExit):
            ST.fetch_vcrt(self.cache, url=self.url, sha256="0" * 64)
        self.assertEqual(sorted(os.listdir(self.cache)), [os.path.basename(self.embed)])    # nothing kept

    def test_the_same_downloads_give_the_same_zip(self):
        w = ST.fetch_vcrt(self.cache, url=self.url, sha256=self.sha)
        a, b = os.path.join(self.d, "a.zip"), os.path.join(self.d, "b.zip")
        ST.embed_zip(self.embed, w, a)
        ST.embed_zip(self.embed, w, b)
        self.assertEqual(sha256(a), sha256(b))
        self.assertNotEqual(sha256(a), sha256(self.embed))

    def test_a_wheel_without_one_of_the_dlls_stops_the_stage(self):
        url, sha = stand_in_vcrt(os.path.join(self.d, "pypi-short"), dlls=("msvcp140.dll",))
        w = ST.fetch_vcrt(self.cache, url=url, sha256=sha)
        dest = os.path.join(self.d, "short.zip")
        with self.assertRaises(SystemExit):
            ST.embed_zip(self.embed, w, dest)
        self.assertFalse(os.path.exists(dest))

    def test_the_pin_is_pypi_msvc_runtime_for_the_bundled_python(self):
        m = re.fullmatch(r"https://files\.pythonhosted\.org/packages/[0-9a-f/]+/msvc_runtime-14\.\d+\.\d+-cp3(\d+)-cp3\d+"
                         r"-win_amd64\.whl", ST.VCRT_URL)
        self.assertTrue(m, ST.VCRT_URL)
        self.assertEqual("3." + m.group(1), re.search(r"python-(3\.\d+)\.", ST.EMBED_URL).group(1))
        self.assertRegex(ST.VCRT_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(ST.VCRT_DLLS, ("msvcp140.dll", "msvcp140_1.dll"))


class BuildScripts(unittest.TestCase):
    """every compiled build bundles what the stage made, and passes the release tag to the stage."""

    def read(self, name):
        with open(os.path.join(CONTROL, name), encoding="utf-8") as f:
            return f.read()

    def test_windows_bundles_the_kit_and_the_python_with_both_tools(self):
        t = self.read("build-windows.ps1")
        for line in ('--add-data "$W\\plugin-sdk.zip;."', '--add-data "$W\\python-embed.zip;."',
                     "--include-data-files=plugin-sdk.zip=plugin-sdk.zip",
                     "--include-data-files=python-embed.zip=python-embed.zip",
                     '$StageArgs = @($W, "--python-embed"', '$StageArgs += @("--tag", $Tag)', 'stage.py" @StageArgs'):
            self.assertTrue(line in t, "build-windows.ps1 lacks " + line)
        self.assertNotIn("3.1.0.0", t)                                                    # the file version follows the tag

    def test_the_ci_workflow_bundles_them_too(self):
        with open(os.path.join(ROOT, ".github", "workflows", "control-build.yml"), encoding="utf-8") as f:
            t = f.read()
        self.assertEqual(t.count("--include-data-files=plugin-sdk.zip=plugin-sdk.zip"), 2)          # Windows + macOS
        self.assertEqual(t.count("--include-data-files=python-embed.zip=python-embed.zip"), 1)      # Windows only
        self.assertIn('"--python-embed", "out/cache"', t)

    def test_linux_and_macos_bundle_the_kit(self):
        for name, lines in (("build-linux.sh", ('--include-data-files="$W/plugin-sdk.zip=plugin-sdk.zip"',
                                                '--add-data "$W/plugin-sdk.zip:."')),
                            ("build-macos.sh", ('--include-data-files="$W/plugin-sdk.zip=plugin-sdk.zip"',)),
                            ("build-macos-universal2.sh", ('--add-data "$W/plugin-sdk.zip:."',))):
            t = self.read(name)
            for line in lines + ('stage.py" "$W" ${TAG:+--tag "$TAG"}',):
                self.assertTrue(line in t, "%s lacks %s" % (name, line))


    def test_linux_and_macos_warn_loudly_without_a_tag(self):
        """no TAG: VERSION stays the repo's (v3.1) and every bundled plugin (min_host 4.0) is refused; a dev build still
        runs, with a warning on stderr like build-windows.ps1's."""
        import subprocess
        for name in ("build-linux.sh", "build-macos.sh", "build-macos-universal2.sh"):
            t = self.read(name)
            warn = [ln for ln in t.splitlines() if ln.startswith('[ -n "${TAG:-}" ] ||')]
            self.assertEqual(len(warn), 1, "%s has no TAG warning" % name)
            self.assertLess(t.index(warn[0]), t.index('stage.py" "$W"'), name)        # before the stage, not after it
            env = {k: v for k, v in os.environ.items() if k != "TAG"}
            for tag, quiet in ((None, False), ("v4.0.0", True)):
                e = dict(env, **({"TAG": tag} if tag else {}))
                r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + warn[0] + "\necho went-on"], env=e,
                                   capture_output=True, text=True, timeout=30)
                self.assertEqual((r.returncode, r.stdout), (0, "went-on\n"), name)   # a warning, not a stop
                if quiet:
                    self.assertEqual(r.stderr, "", name)
                else:
                    self.assertIn("WARNING", r.stderr)
                    self.assertIn("min_host", r.stderr)

if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
