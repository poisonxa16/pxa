"""Plugin packages (.pxp): how PXA Control installs and loads a first-party plugin it did not ship with.

A .pxp is a zip: plugin.json (the signed manifest, see plugins/__init__.py) at its root plus exactly the files the manifest
lists, nothing else. Supporter plugins are not in the open Control tree or the release package: their sources live in the
private tree and reach a computer only as a .pxp through the grant-bound Store download (licd download-token), so the
delivery is part of the gate (PLUGINS-PROTOTYPE-PLAN S8/S9, security review F4b).

install(): signature first (from the zip, before anything is written), then the zip's names must equal the listed files
(no links, no absolute or parent paths, no extras, size caps), extracted into a staging folder and checked again by
load_manifest (signature + every sha256), then moved into <config>/plugins/installed/<plugin_id>/<version>/.
load(): every time, the installed folder is re-verified (signature + hashes + no unlisted file in its Python package), a
supporter plugin must pass gate.require (the machine-bound grant), then its Python package is imported under a private
module name. The honest limit: the code runs on the person's own computer, so the gate decides what Control does, and the
grant-bound download decides who gets the package at all.
"""
import importlib.util
import json
import os
import secrets
import shutil
import stat
import sys
import threading
import zipfile
import zlib

from . import PluginError, load_manifest, verify_manifest, validate_manifest  # noqa: F401
from . import _VER_RE as P_VER
from . import _ID_RE as P_ID
from . import gate

PXP_EXT = ".pxp"
MAX_FILES = 1024
MAX_BYTES = 128 << 20         # all files together, uncompressed (the Store caps the package itself at 64 MiB on the wire)
MAX_MANIFEST = 1 << 20
MAX_RATIO = 200               # a member that inflates more than this (and past RATIO_FLOOR) is a zip bomb
RATIO_FLOOR = 64 << 10


def _err(msg, code, status=400):
    e = PluginError(msg, code)
    e.status = status
    return e


def ver_key(v):
    """x.y.z[-tag] -> a sortable tuple (the tag sorts before the plain release)."""
    import re
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[-+](.*))?$", str(v or ""))
    if not m:
        return (-1, -1, -1, "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), "~" if m.group(4) is None else m.group(4))


def _root(cfg_dir):
    return os.path.join(cfg_dir, "plugins", "installed")


def read_manifest(path, pubkeys=None):
    """the manifest inside a .pxp, verified (shape + first-party signature) before anything is extracted."""
    try:
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("plugin.json")
            if info.file_size > MAX_MANIFEST:
                raise PluginError("plugin.json is too large", "bad_package")
            m = json.loads(z.read(info).decode("utf-8"))
    except KeyError:
        raise PluginError("not a plugin package (no plugin.json)", "bad_package")
    except PluginError:
        raise
    except (OSError, ValueError, EOFError, RuntimeError, NotImplementedError, zipfile.BadZipFile, zlib.error):
        raise PluginError("not a plugin package (the zip or its plugin.json cannot be read)", "bad_package")
    return verify_manifest(m, pubkeys)


def _check_names(z, m):
    want = set(m["files"]) | {"plugin.json"}
    seen, total = set(), 0
    infos = z.infolist()
    if len(infos) > MAX_FILES:
        raise PluginError("too many files in the package", "bad_package")
    for i in infos:
        n = i.filename
        if i.is_dir():
            continue
        if n in seen:
            raise PluginError("%s is in the package twice" % n, "bad_package")
        seen.add(n)
        mode = (i.external_attr >> 16) & 0o170000
        if mode and not stat.S_ISREG(mode):
            raise PluginError("%s is not a plain file" % n, "bad_package")
        if i.file_size > RATIO_FLOOR and i.file_size > MAX_RATIO * max(i.compress_size, 1):
            raise PluginError("%s inflates too far (a zip bomb)" % n, "bad_package")
        if n not in want:
            raise PluginError("%s is not listed in the signed manifest" % n, "bad_package")
        total += i.file_size
        if total > MAX_BYTES:
            raise PluginError("the package is too large", "bad_package")
    missing = want - seen
    if missing:
        raise PluginError("listed but missing: " + ", ".join(sorted(missing)[:5]), "bad_package")


def current_version(cfg_dir, plugin_id):
    """the installed (current) version of plugin_id, or None."""
    d = folder(cfg_dir, plugin_id)
    return os.path.basename(d) if d else None


# one install or removal at a time in this process (eng review F8): the server is threaded, and two tabs (or Update plus
# Download) must never swap the same plugin folder at once. The work is local file moves; downloads happen before this.
_INSTALL_LOCK = threading.RLock()


def install(path, cfg_dir, pubkeys=None, item=None, expect_version=None, expect_kid=None, refuse_older=False):
    """a .pxp -> {plugin_id, version, edition, entitlement, name, dir}. Raises PluginError (code bad_package,
    unknown_key, bad_signature, bad_file, invalid_manifest) and leaves the installed plugins as they were.
    item: the store item this package was downloaded for; the signed manifest must name it (its entitlement, or its id for a
    free plugin), so a grant for one item never installs another item's package (security review P10 L2).
    expect_version / expect_kid: what the signed Store catalogue says this package is (version_mismatch 400, unknown_key 400).
    refuse_older (the Store path): an older version than the installed one is refused (downgrade 409); the same version again
    is a reinstall."""
    with _INSTALL_LOCK:
        return _install(path, cfg_dir, pubkeys, item, expect_version, expect_kid, refuse_older)


def _install(path, cfg_dir, pubkeys, item, expect_version, expect_kid, refuse_older):
    m = read_manifest(path, pubkeys)
    if item is not None and (m.get("entitlement") or m["plugin_id"]) != item:
        raise PluginError("the package is not the %s plugin" % item, "bad_package")
    pid, ver = m["plugin_id"], m["version"]
    if expect_version is not None and ver != expect_version:
        raise _err("the package is version %s, the Store lists %s" % (ver, expect_version), "version_mismatch")
    if expect_kid is not None and m["signature"]["kid"] != expect_kid:
        raise _err("the package is signed with another key than the Store lists", "unknown_key")
    have = current_version(cfg_dir, pid)
    if refuse_older and have and ver_key(ver) < ver_key(have):
        raise _err("%s %s is installed; %s is older" % (m["name"], have, ver), "downgrade", 409)
    base = os.path.join(_root(cfg_dir), pid)
    os.makedirs(base, mode=0o700, exist_ok=True)
    stage = os.path.join(base, ".stage-" + secrets.token_hex(6))
    os.makedirs(stage, mode=0o700)
    try:
        with zipfile.ZipFile(path) as z:
            _check_names(z, m)
            left = MAX_BYTES + MAX_MANIFEST                 # all files together, as written (not only as declared)
            for rel in list(m["files"]) + ["plugin.json"]:
                dest = os.path.join(stage, *rel.split("/"))
                os.makedirs(os.path.dirname(dest), mode=0o700, exist_ok=True)
                with z.open(rel) as src, open(dest, "wb") as out:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        left -= len(b)
                        if left < 0:
                            raise PluginError("the package is too large", "bad_package")
                        out.write(b)
        argv0 = m["entry"]["argv"][0]
        for rel in m["files"]:
            os.chmod(os.path.join(stage, *rel.split("/")), 0o700 if rel == argv0 else 0o600)
        load_manifest(stage, pubkeys)                       # signature and every sha256, on the extracted bytes
        final = os.path.join(base, ver)
        if os.path.exists(final):
            old = os.path.join(base, ".old-" + secrets.token_hex(6))
            os.replace(final, old)
            try:
                os.replace(stage, final)
            except OSError:
                os.replace(old, final)                  # the old install stays (review F1): swap back, then refuse
                raise
            shutil.rmtree(old, ignore_errors=True)      # removed only after the new one is in place
        else:
            os.replace(stage, final)
        tmp = os.path.join(base, ".current-%s.tmp" % secrets.token_hex(6))
        with open(tmp, "w") as f:
            f.write(ver + "\n")
        os.replace(tmp, os.path.join(base, "current"))
    except PluginError:
        raise
    except (EOFError, RuntimeError, NotImplementedError, ValueError, zipfile.BadZipFile, zlib.error) as e:
        # a corrupt, encrypted or oddly compressed member: plain words, never a 500 with a class name (review F5)
        raise PluginError("the package is damaged (a file in it cannot be read); nothing was changed", "bad_package") from e
    except OSError as e:
        # a disk error, or a folder in use (Windows): what was installed before stays (review F1, F5)
        raise _err("the plugin could not be installed (a file is in use or the disk refused); what was installed before "
                   "is unchanged. Close what uses it and try again", "install_failed", 500) from e
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)
    return {"plugin_id": pid, "version": ver, "edition": m["edition"], "entitlement": m.get("entitlement"),
            "name": m["name"], "dir": final}


def folder(cfg_dir, plugin_id):
    """the installed (current) folder of plugin_id, or None."""
    base = os.path.join(_root(cfg_dir), plugin_id)
    try:
        with open(os.path.join(base, "current")) as f:
            ver = f.read().strip()
    except OSError:
        return None
    if not P_VER.match(ver):                        # a version, never a path ("..", "/")
        return None
    d = os.path.join(base, ver)
    return d if os.path.isdir(d) else None


def installed(cfg_dir, pubkeys=None, cache=None):
    """[{plugin_id, name, version, edition, entitlement, ok, error?, gate?}] for every installed plugin; supporter plugins
    carry their gate state (from gate.state, the one gate)."""
    out = []
    root = _root(cfg_dir)
    for pid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = folder(cfg_dir, pid)
        if not d:
            continue
        row = {"plugin_id": pid}
        try:
            m = load_manifest(d, pubkeys)
            row.update(name=m["name"], version=m["version"], edition=m["edition"], entitlement=m.get("entitlement"), ok=True)
            if m["edition"] == "supporter":
                row["gate"] = gate.state(cfg_dir, m["entitlement"], "plugin", cache)["state"]
        except PluginError as e:
            row.update(ok=False, error=e.code)
        out.append(row)
    return out


# Finder / Explorer drop these when a user opens the plugin folder (review N2): skipped by the unlisted-file check only;
# never imported (not module names) and never served (the host serves signed, listed files only)
OS_METADATA = frozenset({".ds_store", "thumbs.db", "desktop.ini"})
_CODE_EXT = (".py", ".pyc", ".pyo", ".pyd", ".so", ".pth")


def os_metadata(name):
    """True for an OS metadata file name: .DS_Store, Thumbs.db, desktop.ini, or an AppleDouble ._* that is not code."""
    n = name.lower()
    return n in OS_METADATA or (n.startswith("._") and len(n) > 2 and not n.endswith(_CODE_EXT))


def unlisted_any(d, m, cfg_dir=None):
    """why the installed version folder is not exactly the signed plugin, or None (review F6, N1): the first file the
    manifest does not list (plugin.json aside), any __pycache__ folder or .pyc file (bytecode would run in place of the
    signed source), any symlink in the tree, or a folder whose real path is not its install path. The host refuses to
    start a plugin with one; a dropped bin/json.py or lib/urllib/ would otherwise run as the verified plugin. OS metadata
    files (os_metadata: .DS_Store, Thumbs.db, desktop.ini, AppleDouble ._* that is not code) are skipped (review N2)."""
    if cfg_dir is not None:
        want = os.path.join(os.path.realpath(_root(cfg_dir)), str(m["plugin_id"]), str(m["version"]))
        if os.path.realpath(d) != want:
            return "the plugin folder (not at its install path)"
    listed = {os.path.normpath(os.path.join(d, *rel.split("/"))) for rel in m["files"]}
    listed.add(os.path.normpath(os.path.join(d, "plugin.json")))
    for dirpath, dirs, names in os.walk(d, followlinks=False):
        for x in sorted(dirs):
            p = os.path.join(dirpath, x)
            if x == "__pycache__" or os.path.islink(p):
                return os.path.relpath(p, d)
        dirs.sort()
        for n in sorted(names):
            p = os.path.normpath(os.path.join(dirpath, n))
            if os.path.islink(p) or n.endswith((".pyc", ".pyo")):
                return os.path.relpath(p, d)
            if p not in listed and not (os_metadata(n) and os.path.isfile(p)):
                return os.path.relpath(p, d)
    return None


def load(cfg_dir, plugin_id, pubkeys=None, cache=None):
    """import an installed plugin's Python package, after the signature, the file hashes and (supporter) the grant.
    -> the module. The plugin's own code still asks the gate before each run (defence in depth)."""
    d = folder(cfg_dir, plugin_id)
    if not d:
        raise PluginError("%s is not installed" % plugin_id, "not_installed")
    m = load_manifest(d, pubkeys)
    if m["plugin_id"] != plugin_id:
        raise PluginError("the installed folder holds another plugin", "bad_file")
    if m["edition"] == "supporter":
        gate.require(cfg_dir, m["entitlement"], "plugin", cache, name=m["name"])
    py = m.get("python")
    if not py:
        raise PluginError("%s has no in-process package (it runs under the plugin host)" % plugin_id, "no_python")
    extra = unlisted_any(d, m, cfg_dir)
    if extra:
        raise PluginError("%s is not part of the signed plugin" % extra, "bad_file")
    pkg_dir = os.path.join(d, *py["path"].split("/"), py["package"])
    name = "pxa_plugin_" + py["package"]
    sys.modules.pop(name, None)
    for k in [k for k in sys.modules if k.startswith(name + ".")]:
        sys.modules.pop(k, None)
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # no __pycache__ beside the signed files, ever: it stays off for this process, since a plugin's later (lazy) imports
    # would otherwise write bytecode there and its next load is refused (review N1)
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    mod.PXA_PLUGIN_MANIFEST = m
    return mod


def uninstall(cfg_dir, plugin_id):
    """remove an installed plugin's files (every version). The caller stops it first; consent and keys stay until withdrawn.
    -> {plugin_id, removed}. The folder is moved aside first, so a half-removed plugin never loads."""
    if not P_ID.match(plugin_id or ""):
        raise _err("bad plugin id", "invalid")
    with _INSTALL_LOCK:                                     # never in the middle of an install (eng review F8)
        return _uninstall(cfg_dir, plugin_id)


def _uninstall(cfg_dir, plugin_id):
    base = os.path.join(_root(cfg_dir), plugin_id)
    if not os.path.isdir(base):
        raise _err("%s is not installed" % plugin_id, "not_installed", 404)
    gone = os.path.join(_root(cfg_dir), ".removed-%s-%s" % (plugin_id, secrets.token_hex(6)))
    os.replace(base, gone)
    shutil.rmtree(gone, ignore_errors=True)
    return {"plugin_id": plugin_id, "removed": True}
