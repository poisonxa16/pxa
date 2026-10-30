"""The Python that runs plugins and the speech service.

A source run uses its own interpreter. A compiled Control's sys.executable is Control itself, so it uses, in order:
PXA_CONTROL_PYTHON; the Python the bundle carries (Windows: python.org's embeddable package as python-embed.zip, unpacked
once into <config>/python/<sha16>); a Python 3.10 - 3.14 (64-bit) found on this computer (Windows: the entries python.org's
installer writes to the registry (PEP 514), PATH without the Microsoft Store aliases, the usual folders; Linux: python3.x
on PATH; macOS: Homebrew and python.org, never /usr/bin/python3, the stub that offers to install Apple's developer tools).
Each candidate is started once (-I -c, no secret in its environment) to read its version and width. A hit is kept for the
process; "none found" is asked again after MISS_TTL_S, so a Python installed later needs no restart.

Stdlib only.
"""
import hashlib
import ntpath
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
import zipfile

PY_MIN, PY_MAX = (3, 10), (3, 14)
MISS_TTL_S = 60.0
PROBE_TIMEOUT_S = 15.0
EMBED_ZIP = "python-embed.zip"
EMBED_EXE = "python.exe" if os.name == "nt" else "python"
OK = "pxa-unpacked"
NO_WINDOW = 0x08000000          # Windows CREATE_NO_WINDOW: no console flashes up for a probe
TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# what a probe inherits: enough for Python to start (Windows needs SYSTEMROOT), never a token or key
PROBE_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "LANG", "LC_ALL", "LC_CTYPE")
PROBE = "import struct, sys; print(sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8)"

_memo = {}      # (config dir, bundle) -> (clock, path or None)


def frozen():
    """a compiled Control (Nuitka sets __compiled__ on every module it compiles; PyInstaller sets sys.frozen)."""
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def bundle_root():
    """the folder a compiled Control keeps its data files in (desktop.bundle_root, without the desktop side's imports)."""
    return os.environ.get("PXA_BUNDLE_ROOT") or (TOOLS if frozen() else "")


def forget():
    _memo.clear()


def find(cfg_dir, frozen_=None, env=None, bundle=None, cands=None, clock=time.monotonic):
    """the Python for plugins and speech, or None (missing_words() says what the user can do)."""
    if not (frozen() if frozen_ is None else frozen_):
        return sys.executable
    bundle = bundle_root() if bundle is None else bundle
    key = (cfg_dir, bundle)
    hit = _memo.get(key)
    if hit is not None and (hit[1] or clock() - hit[0] < MISS_TTL_S):
        return hit[1]
    py = _search(cfg_dir, os.environ if env is None else env, bundle, cands)
    _memo[key] = (clock(), py)
    return py


def _search(cfg_dir, env, bundle, cands):
    over = env.get("PXA_CONTROL_PYTHON")
    if over and usable(probe(over)):
        return over
    if bundle and cfg_dir:
        exe = embedded(cfg_dir, bundle)
        if exe and usable(probe(exe)):
            return exe
    for p in candidates() if cands is None else cands:
        if usable(probe(p)):
            return p
    return None


def probe(path, timeout=PROBE_TIMEOUT_S):
    """-> (major, minor, bits) of the Python at path, or None (not there, not a Python, too slow)."""
    if not path or not os.path.isfile(path):
        return None
    env = {k: os.environ[k] for k in PROBE_ENV if k in os.environ}
    kw = {"creationflags": NO_WINDOW} if os.name == "nt" else {}
    try:
        out = subprocess.run([path, "-I", "-c", PROBE], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                             timeout=timeout, **kw)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    a = out.stdout.split()
    if out.returncode != 0 or len(a) != 3 or not all(x.isdigit() for x in a):
        return None
    return tuple(int(x) for x in a)


def usable(v):
    return bool(v) and PY_MIN <= tuple(v[:2]) <= PY_MAX and v[2] == 64


# ---- the bundle's own Python ------------------------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rmtree(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _safe(name):
    n = name.replace("\\", "/")
    return bool(n) and not n.startswith("/") and ":" not in n and ".." not in n.split("/")


def embedded(cfg_dir, bundle):
    """<bundle>/python-embed.zip unpacked once into <config>/python/<sha16> (beside the old one, then moved in; a folder
    without the finished mark is unpacked again) -> its python(.exe), or None (no zip, a bad zip, a full disk)."""
    z = os.path.join(bundle, EMBED_ZIP)
    if not os.path.isfile(z):
        return None
    try:
        sha = _sha256(z)
        home = os.path.join(cfg_dir, "python", sha[:16])
        exe = os.path.join(home, EMBED_EXE)
        if os.path.isfile(os.path.join(home, OK)) and os.path.isfile(exe):
            return exe
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            if EMBED_EXE not in names or not all(_safe(n) for n in names):
                return None
            tmp = "%s.%d.tmp" % (home, os.getpid())
            _rmtree(tmp)
            try:
                zf.extractall(tmp)
                if os.name != "nt":
                    os.chmod(os.path.join(tmp, EMBED_EXE), 0o700)         # a zip keeps no file modes
                with open(os.path.join(tmp, OK), "w") as f:
                    f.write(sha)
                _rmtree(home)
                try:
                    os.replace(tmp, home)
                except OSError:
                    if not os.path.isfile(os.path.join(home, OK)):     # another Control finished the same unpack first
                        raise
            finally:
                _rmtree(tmp)
        return exe if os.path.isfile(exe) else None
    except (OSError, ValueError, zipfile.BadZipFile):
        return None


# ---- Pythons on this computer -----------------------------------------------------------------------------------------
def pick_registry(rows):
    """(tag, install folder) rows of PEP 514 -> [python.exe], newest first: 3.10 - 3.14 only, no 32-bit (-32) or
    free-threaded (t) builds."""
    got = []
    for tag, folder in rows:
        m = re.fullmatch(r"3\.(\d+)", tag or "")
        if m and folder and PY_MIN <= (3, int(m.group(1))) <= PY_MAX:
            got.append(((3, int(m.group(1))), ntpath.join(folder, "python.exe")))
    out = []
    for _v, p in sorted(got, key=lambda x: x[0], reverse=True):
        if p not in out:
            out.append(p)
    return out


def _registry():
    """python.org's installs on Windows: HKCU, then HKLM (64-bit view, then 32-bit view)."""
    try:
        import winreg
    except ImportError:
        return []
    rows = []
    for hive, view in ((winreg.HKEY_CURRENT_USER, 0), (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
                       (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY)):
        try:
            core = winreg.OpenKey(hive, r"Software\Python\PythonCore", 0, winreg.KEY_READ | view)
        except OSError:
            continue
        with core:
            i = 0
            while True:
                try:
                    tag = winreg.EnumKey(core, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(core, tag + r"\InstallPath", 0, winreg.KEY_READ | view) as ip:
                        try:
                            folder = ntpath.dirname(winreg.QueryValueEx(ip, "ExecutablePath")[0])
                        except OSError:
                            folder = winreg.QueryValueEx(ip, "")[0]
                    rows.append((tag, folder))
                except OSError:
                    pass
    return pick_registry(rows)


def candidates(plat=None, env=None, which=shutil.which, registry=None, exists=os.path.isfile):
    """Pythons on this computer, the likeliest first (not started yet)."""
    plat = plat or sys.platform
    env = os.environ if env is None else env
    vers = ["3.%d" % m for m in range(PY_MAX[1], PY_MIN[1] - 1, -1)]
    isabs = ntpath.isabs if plat == "win32" else posixpath.isabs
    out = []

    def add(p):
        # absolute only: Windows' which() looks in the current folder first, and so does an empty or "." PATH entry
        if p and isabs(p) and p not in out and exists(p):
            out.append(p)
    if plat == "win32":
        for p in (registry or _registry)():
            add(p)
        for n in ("python", "python3"):
            p = which(n)
            if p and "\\windowsapps\\" not in p.lower():            # a Store alias opens the Store instead of running
                add(p)
        for v in vers:
            if env.get("LOCALAPPDATA"):
                add(ntpath.join(env["LOCALAPPDATA"], "Programs", "Python", "Python" + v.replace(".", ""), "python.exe"))
            if env.get("ProgramFiles"):
                add(ntpath.join(env["ProgramFiles"], "Python" + v.replace(".", ""), "python.exe"))
    elif plat == "darwin":
        add("/opt/homebrew/bin/python3")
        add("/usr/local/bin/python3")
        for v in vers:
            add("/Library/Frameworks/Python.framework/Versions/%s/bin/python3" % v)
        for n in ["python" + v for v in vers] + ["python3"]:
            p = which(n)
            if p != "/usr/bin/python3":
                add(p)
    else:
        for n in ["python" + v for v in vers] + ["python3"]:
            add(which(n))
        add("/usr/bin/python3")
        add("/usr/local/bin/python3")
    return out


def missing_words(plat=None):
    """what the user can do when no Python was found, in plain words."""
    plat = plat or sys.platform
    if plat == "win32":
        return ("PXA Control could not start its own Python, which plugins and speech run on. Reinstall PXA Control, or "
                "install Python 3.10 to 3.14 (64-bit) from python.org.")
    if plat == "darwin":
        return "Plugins and speech need Python 3.10 to 3.14: install it from python.org or with Homebrew, then try again."
    return ("Plugins and speech need Python 3.10 to 3.14: install python3 with your system's package manager, then try "
            "again.")
