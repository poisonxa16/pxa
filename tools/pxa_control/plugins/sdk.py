"""The plugin kit: the plain-source part of PXA Control that plugins and the speech service import (through
PXA_CONTROL_TOOLS / the speech bootstrap's path).

A source run hands them its own tools folder. A compiled Control has no Python sources to hand over, so its bundle
carries the kit as plugin-sdk.zip (stage.py writes it with build()); ensure() unpacks it once into
<config>/plugins/sdk/<sha16> and, once per process, checks every file against the kit's manifest (SDK.json): a changed or
missing file means the kit is unpacked again.

The kit's pxa_control/__init__.py is a stub: the real one loads all of Control (core, routes, server, ...), which a
plugin never needs. FILES is closed: tests/test-pxa-plugin-sdk.py imports every kit module in a Python that sees only
the kit, and checks that every plugin imports only kit modules.

Stdlib only at module level: stage.py loads this file by path, before anything is installed.
"""
import hashlib
import json
import os
import shutil
import zipfile

KIT_ZIP = "plugin-sdk.zip"
MANIFEST = "SDK.json"
TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAMP = (2020, 1, 1, 0, 0, 0)       # every member gets the same date: the same sources give the same zip
STUB_INIT = ('"""PXA Control\'s plugin kit: the parts of PXA Control that plugins and the speech service run on, as '
             'plain source.\nA compiled PXA Control unpacks it into its config folder and hands plugins its path."""\n')
FILES = (
    "pxa_control/mdns.py",
    "pxa_control/plugins/__init__.py",
    "pxa_control/plugins/child.py",
    "pxa_control/plugins/gate.py",
    "pxa_control/store/__init__.py",
    "pxa_control/store/pubkeys.json",
    "pxa_control/speech/__init__.py",
    "pxa_control/speech/__main__.py",
    "pxa_control/speech/client.py",
    "pxa_control/speech/engines.lock",              # setup.pins_id() reads both locks for every speech status
    "pxa_control/speech/kokoro.lock",
    "pxa_control/speech/models.json",
    "pxa_control/speech/registry.py",
    "pxa_control/speech/server.py",
    "pxa_control/speech/service.py",
    "pxa_control/speech/setup.py",
    "pxa_control/speech/stt.py",
    "pxa_control/speech/tts.py",
    "pxa_control/speech/vad.py",
    "pxa_encode_adapter.py",            # pxa_encode_pkg imports it
    "pxa_encode_pkg.py",                # the Ed25519 verifier the store and the plugin loader use (already plain source on Linux)
)

_checked = {}       # (config dir, kit zip) -> unpacked kit folder, checked in this process


def forget():
    _checked.clear()


def build(out_zip, tools=TOOLS):
    """write the kit (stub __init__, FILES, SDK.json) to out_zip -> the zip's sha256. Stored, sorted, one date and one
    mode for every member: the same sources give the same bytes on every build and every OS."""
    files = {"pxa_control/__init__.py": STUB_INIT.encode("utf-8")}
    for rel in FILES:
        with open(os.path.join(tools, *rel.split("/")), "rb") as f:
            files[rel] = f.read()
    man = {"kit": 1, "files": {rel: hashlib.sha256(b).hexdigest() for rel, b in sorted(files.items())}}
    files[MANIFEST] = (json.dumps(man, indent=1, sort_keys=True) + "\n").encode("utf-8")
    tmp = "%s.%d.tmp" % (out_zip, os.getpid())
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for rel in sorted(files):
                zi = zipfile.ZipInfo(rel, STAMP)
                zi.create_system = 3
                zi.external_attr = 0o644 << 16
                z.writestr(zi, files[rel])
        os.replace(tmp, out_zip)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return _sha256(out_zip)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _safe(name):
    n = name.replace("\\", "/")
    return bool(n) and not n.startswith("/") and ":" not in n and ".." not in n.split("/")


def _intact(home, files):
    for rel, sha in files.items():
        p = os.path.join(home, *rel.split("/"))
        if not os.path.isfile(p) or _sha256(p) != sha:
            return False
    return True


def ensure(cfg_dir, bundle):
    """<bundle>/plugin-sdk.zip unpacked into <config>/plugins/sdk/<sha16> (beside the old one, then moved in) -> that
    folder, or None (no kit, a bad kit, a full disk)."""
    z = os.path.join(bundle, KIT_ZIP) if bundle else ""
    if not cfg_dir or not z or not os.path.isfile(z):
        return None
    key = (cfg_dir, z)
    if key in _checked:
        return _checked[key]
    try:
        home = os.path.join(cfg_dir, "plugins", "sdk", _sha256(z)[:16])
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            if MANIFEST not in names or not all(_safe(n) for n in names):
                return None
            files = json.loads(zf.read(MANIFEST).decode("utf-8")).get("files")
            if not isinstance(files, dict) or set(files) != set(names) - {MANIFEST}:
                return None
            if not _intact(home, files):
                tmp = "%s.%d.tmp" % (home, os.getpid())
                _rmtree(tmp)
                try:
                    zf.extractall(tmp)
                    if not _intact(tmp, files):
                        return None
                    _rmtree(home)
                    try:
                        os.replace(tmp, home)
                    except OSError:
                        if not _intact(home, files):            # another Control finished the same unpack first
                            raise
                finally:
                    _rmtree(tmp)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    _checked[key] = home
    return home


def _rmtree(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def tools_dir(cfg_dir, frozen_=None, bundle=None):
    """the folder plugins and the speech service import pxa_control from: a source run's own tools folder; a compiled
    Control's unpacked kit (None when its bundle has no usable kit)."""
    from .. import pyfind
    if not (pyfind.frozen() if frozen_ is None else frozen_):
        return TOOLS
    return ensure(cfg_dir, pyfind.bundle_root() if bundle is None else bundle)
