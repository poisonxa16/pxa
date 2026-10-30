#!/usr/bin/env python3
"""Stage tools/ for a compiled PXA Control build, on any OS (BUILD-CONTROL.md): the package and its siblings as shipped
(no tests, no caches), the launcher under an importable name (pxa_launch.py), VERSION (the release tag when one is
given), the PUBLIC lever catalog inside the page folder, the entry script, third_party.zip: LGPL code (pystray,
python-xlib) as plain, replaceable .py, never compiled in, and what plugins and the speech service run on in a compiled
Control: the plugin kit (plugin-sdk.zip, pxa_control/plugins/sdk.py) and, for Windows, python.org's embeddable Python
(python-embed.zip, pinned below, downloaded once into a cache folder; pxa_control/pyfind.py unpacks it) with Microsoft's
C++ runtime DLLs beside python.exe (from the msvc-runtime wheel, pinned below).
    python packaging/control/stage.py <out dir> [--tag v4.0.0] [--python-embed <cache dir>]"""
import argparse
import hashlib
import importlib.util
import os
import re
import shutil
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = os.path.join(ROOT, "tools")
DIRS = ("pxa_control", "pxa_chat", "pxa_ctl", "pxa_control_ui")
FILES = ("pxa_thinking.py", "pxa_hf.py", "pxa_encode.py", "pxa_encode_adapter.py", "pxa_encode_pkg.py", "pxa_encode_plan.py",
         "pxa_telemetry.py", "pxa_mqtt.py", "pxa_lib_update.py", "pxa_engine_update.py", "pxa_expert_map.py", "pxa_explain_footer.py",
         "pxa_thinking_profiles.json", "pxa_encode_cells.json", "pxa-bench.py")
LGPL = ("pystray", "Xlib", "six")
# python.org's embeddable Python 3.13 for Windows x64: the sha256 python.org publishes for it in its Windows install index
# (https://www.python.org/ftp/python/index-windows.json, id pythonembed-3.13-64); a new pin = a new version and its sha
EMBED_URL = "https://www.python.org/ftp/python/3.13.17/python-3.13.17-embeddable-amd64.zip"
EMBED_SHA256 = "0ba1ee4bfcbdc3a559d27e2e9ce76d04a7839947a23f46375872c755c0d463b8"
# Microsoft's C++ runtime, which python.org's package does not carry (it has vcruntime140 only) and the speech engines
# import (onnxruntime for the voice, espeak-ng): a Windows without the Visual C++ Redistributable has neither DLL. From the
# msvc-runtime wheel on PyPI (the sha256 PyPI publishes for it), unmodified, put beside python.exe, where Windows looks
# before System32; Visual Studio's distributable-code terms allow shipping these files with the program that uses them
VCRT_URL = ("https://files.pythonhosted.org/packages/c2/6e/7ba5696e9d8c1bcb01352f06e441497c308e84f75aba598400646ab31ca7/"
            "msvc_runtime-14.51.36231-cp313-cp313-win_amd64.whl")
VCRT_SHA256 = "f3c3ac4f733a774f6ae209567e533bfde6f72dc1c10977733bfe09ce01e5b624"
VCRT_DLLS = ("msvcp140.dll", "msvcp140_1.dll")
FETCH_LIMIT = 64 << 20          # 11.4 MB and 2 MB today; anything far larger is not that file
TAG_RX = r"v?\d+\.\d+[A-Za-z0-9._-]*"


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage tools/ for a compiled PXA Control build")
    ap.add_argument("out")
    ap.add_argument("--tag", type=_tag, help="the release tag VERSION names (default: the repo's VERSION file)")
    ap.add_argument("--python-embed", metavar="CACHE", help="carry python.org's embeddable Python (Windows builds), "
                    "downloaded once into this folder")
    a = ap.parse_args(argv)
    embed = fetch_embed(a.python_embed) if a.python_embed else None       # before the stage: a bad download stops early
    vcrt = fetch_vcrt(a.python_embed) if a.python_embed else None
    out = a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "tests")
    for d in DIRS:
        shutil.copytree(os.path.join(T, d), os.path.join(out, d), ignore=ign)
    for f in FILES:
        if os.path.isfile(os.path.join(T, f)):
            shutil.copy2(os.path.join(T, f), os.path.join(out, f))
    shutil.copy2(os.path.join(T, "pxa-launch.py"), os.path.join(out, "pxa_launch.py"))
    write_version(out, a.tag)
    # the PUBLIC lever table (closed rows dropped, internal text scrubbed), never the raw tree catalog: the same
    # filter a release build embeds (scripts/pxa-lever-catalog.py --release-filter); no catalog = refuse to stage
    import subprocess
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "pxa-lever-catalog.py"), "--release-filter",
                    os.path.join(out, "pxa_control_ui", "pxa-lever-catalog.inc")], cwd=ROOT, check=True)
    shutil.copy2(os.path.join(ROOT, "packaging", "control", "pxa_control_entry.py"), os.path.join(out, "pxa_control_entry.py"))
    with zipfile.ZipFile(os.path.join(out, "third_party.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for name in LGPL:
            spec = importlib.util.find_spec(name)       # located, never imported (pystray opens the display on import)
            if spec is None or not spec.origin:
                continue
            src = spec.origin
            if os.path.basename(src) == "__init__.py":
                base = os.path.dirname(os.path.dirname(src))
                for dp, dn, fn in os.walk(os.path.dirname(src)):
                    dn[:] = [x for x in dn if x != "__pycache__"]
                    for f in fn:
                        if f.endswith(".py"):
                            p = os.path.join(dp, f)
                            z.write(p, os.path.relpath(p, base))
            else:
                z.write(src, os.path.basename(src))
    _sdk().build(os.path.join(out, "plugin-sdk.zip"), tools=T)
    if embed:
        embed_zip(embed, vcrt, os.path.join(out, "python-embed.zip"))
    leaks = plugin_leaks(out)
    if leaks:
        shutil.rmtree(out)
        sys.exit("stage.py: plugin files reached the Control stage (plugins ship only as signed .pxp Store downloads):\n  "
                 + "\n  ".join(leaks))
    print(out)


def _tag(text):
    if not re.fullmatch(TAG_RX, text):
        raise argparse.ArgumentTypeError("%r is not a version tag (v4.0.0, v4.0.0-beta.3)" % text)
    return text


def write_version(out, tag):
    """VERSION: the release tag in the packer's layout (Control reads the token after 'tag:'), or the repo's file."""
    if tag is None:
        shutil.copy2(os.path.join(ROOT, "VERSION"), os.path.join(out, "VERSION"))
        return
    if not re.fullmatch(TAG_RX, tag):
        sys.exit("stage.py: --tag %r is not a version tag (v4.0.0, v4.0.0-beta.3)" % tag)
    with open(os.path.join(out, "VERSION"), "w", newline="\n") as f:
        f.write("tag:              %s\n" % tag)


def _sdk():
    """pxa_control/plugins/sdk.py, loaded by path: its build() needs only the standard library."""
    spec = importlib.util.spec_from_file_location("pxa_plugin_sdk_stage", os.path.join(T, "pxa_control", "plugins", "sdk.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_embed(cache, url=None, sha256=None):
    """the pinned embeddable Python in cache (downloaded once; a copy whose sha256 is not the pin is downloaded again)
    -> its path. A download that does not match the pin stops the stage and leaves nothing behind."""
    return _fetch(cache, url or EMBED_URL, sha256 or EMBED_SHA256, "the embeddable Python")


def fetch_vcrt(cache, url=None, sha256=None):
    """the pinned msvc-runtime wheel in cache, as fetch_embed -> its path."""
    return _fetch(cache, url or VCRT_URL, sha256 or VCRT_SHA256, "the C++ runtime wheel")


def _fetch(cache, url, sha256, what):
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, url.rsplit("/", 1)[-1])
    if os.path.isfile(dest) and _sha256(dest) == sha256:
        return dest
    tmp = "%s.%d.part" % (dest, os.getpid())
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            n = 0
            for b in iter(lambda: r.read(1 << 20), b""):
                n += len(b)
                if n > FETCH_LIMIT:
                    raise ValueError("larger than %d bytes" % FETCH_LIMIT)
                f.write(b)
        got = _sha256(tmp)
        if got != sha256:
            raise ValueError("sha256 %s, pinned %s" % (got, sha256))
        os.replace(tmp, dest)
    except (OSError, ValueError) as e:
        for p in (tmp, dest):
            if os.path.exists(p):
                os.remove(p)
        sys.exit("stage.py: %s from %s did not verify: %s" % (what, url, e))
    return dest


def embed_zip(embed, vcrt, dest):
    """python.org's embeddable Python with VCRT_DLLS from the msvc-runtime wheel beside python.exe -> dest. python.org's
    bytes stay as they are (the DLLs are appended), and the DLLs go in stored, with the wheel's dates and one header for
    every OS, so the same two downloads give the same zip on any build machine."""
    shutil.copyfile(embed, dest)
    try:
        with zipfile.ZipFile(vcrt) as w, zipfile.ZipFile(dest, "a") as z:
            have = {n.rsplit("/", 1)[-1].lower(): n for n in w.namelist() if re.fullmatch(r"[^/]+\.data/data/[^/]+\.dll", n)}
            missing = [d for d in VCRT_DLLS if d not in have]
            if missing:
                raise ValueError("no %s in its data folder" % ", ".join(missing))
            for d in VCRT_DLLS:
                src = w.getinfo(have[d])
                zi = zipfile.ZipInfo(d, date_time=src.date_time)
                zi.create_system = 0
                z.writestr(zi, w.read(src), compress_type=zipfile.ZIP_STORED)
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        os.remove(dest)
        sys.exit("stage.py: the C++ runtime from %s could not be added: %s" % (vcrt, e))
    return dest


def plugin_leaks(out):
    """The release scan's built-in plugin-payload rule (scripts/release-scan.py), on every staged file."""
    spec = importlib.util.spec_from_file_location("pxa_release_scan", os.path.join(ROOT, "scripts", "release-scan.py"))
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    found = []
    for dp, dn, fn in os.walk(out):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            if rs.plugin_payload_name(rel, f) or rs.zip_has_plugin(p) or (
                    f.endswith(".py") and rs.PLUGIN_MODULE_RX.search(open(p, encoding="utf-8", errors="replace").read())):
                found.append(rel)
    return found


if __name__ == "__main__":
    main()
