"""What every program PXA Control starts inherits: one rule per platform, applied to every subprocess.Popen in the process.

Windows: Control is a windowed program (no console of its own), so each console program it starts (llama-server for a
version check, nvidia-smi, docker, a Python helper) would get a NEW console window that flashes up and closes. Every
child is started with CREATE_NO_WINDOW instead, unless the caller asked for a console on purpose (CREATE_NEW_CONSOLE or
DETACHED_PROCESS). Output Control reads still arrives through the pipes. PXA_CHILD_CONSOLE=1 turns this off.
Only a parent WITHOUT a console does this (GetConsoleWindow() == 0): a console program that imports this package
(pxa-launch's command line, pxa_encode, the bench handoff) shares its console with its children, which then print
there as before and open no window of their own.

Linux: the package's own libraries (lib/, lib-compat/: a libstdc++ older than a new distribution's, the CUDA runtime)
are for the engine programs only. A system program started with them first on LD_LIBRARY_PATH (the browser opener
xdg-open -> kde-open / Qt 6, built against the system's newer libstdc++) dies with "GLIBCXX_3.4.32 not found". So a
child that is not one of the package's own programs and inherits Control's environment (env=None) gets that
environment with the package's library folders taken out of LD_LIBRARY_PATH (and the compiled build's own unpack
folder, sys._MEIPASS). The engine's own launches build their environment explicitly (engine_ld_path) and are left as
they are. Since 4.0.1 the pxa wrapper no longer puts lib/ on Control's path at all; this rule keeps an older wrapper,
run-server.sh, or the compiled build from leaking it.

install() is idempotent and is called when the pxa_control package loads.
"""
import inspect
import os
import shutil
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008

_ORIG_INIT = subprocess.Popen.__init__
_SIG = inspect.signature(_ORIG_INIT)
_INSTALLED = [False]


def _real(p):
    try:
        return os.path.realpath(p)
    except (OSError, ValueError):
        return p


def bundled_lib_dirs(environ=None):
    """the folders whose libraries are the package's own: <engine>/lib, <engine>/lib-compat, the compiled build's unpack dir."""
    environ = os.environ if environ is None else environ
    out = []
    E = environ.get("PXA_ENGINE_DIR")
    if E:
        for sub in ("lib", "lib-compat"):
            out.append(_real(os.path.join(E, sub)))
    mei = getattr(sys, "_MEIPASS", None)
    if mei:
        out.append(_real(mei))
    return out


def _inside(path, dirs):
    rp = _real(path)
    return any(rp == d or rp.startswith(d.rstrip(os.sep) + os.sep) for d in dirs)


def strip_ld_path(value, dirs):
    """LD_LIBRARY_PATH without the entries that are (or are under) one of dirs; None when nothing is left."""
    kept = [x for x in (value or "").split(os.pathsep) if x and not _inside(x, dirs)]
    return os.pathsep.join(kept) if kept else None


def system_env(environ=None):
    """a copy of environ (default os.environ) fit for a system program: the package's library folders are not on
    LD_LIBRARY_PATH. When the compiled build saved the user's own value (LD_LIBRARY_PATH_ORIG), that value is used."""
    environ = os.environ if environ is None else environ
    env = dict(environ)
    if getattr(sys, "_MEIPASS", None) and "LD_LIBRARY_PATH_ORIG" in env:
        orig = env.pop("LD_LIBRARY_PATH_ORIG")
        val = strip_ld_path(orig, bundled_lib_dirs(environ))
    else:
        val = strip_ld_path(env.get("LD_LIBRARY_PATH"), bundled_lib_dirs(environ))
    if val:
        env["LD_LIBRARY_PATH"] = val
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def _program(args, executable, shell, env):
    """the file the child will run, resolved like the OS would (None when it cannot be told)."""
    if shell:
        return "/bin/sh"
    prog = executable
    if prog is None:
        if isinstance(args, (str, bytes, os.PathLike)):
            prog = args
        else:
            try:
                prog = list(args)[0]
            except (TypeError, IndexError):
                return None
    prog = os.fsdecode(prog) if isinstance(prog, (bytes, os.PathLike)) else prog
    if not isinstance(prog, str):
        return None
    if os.sep in prog or (os.altsep and os.altsep in prog):
        return prog
    path = (env or os.environ).get("PATH")
    return shutil.which(prog, path=path) or prog


def is_package_program(prog, environ=None):
    """a program of this package (its engine dir, or the compiled build's unpack dir): it may keep the bundled libraries."""
    environ = os.environ if environ is None else environ
    if not prog:
        return False
    roots = []
    E = environ.get("PXA_ENGINE_DIR")
    if E:
        roots.append(_real(E))
    mei = getattr(sys, "_MEIPASS", None)
    if mei:
        roots.append(_real(mei))
    return bool(roots) and _inside(prog, roots)


def _kernel32():
    """kernel32 on Windows, None elsewhere (the test hook)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        return ctypes.windll.kernel32
    except (ImportError, AttributeError, OSError):
        return None


def parent_has_console():
    """True when this process has a console window (a console program, or one started from a terminal): its children
    share it and need no flag. False for the windowed Control (and off Windows)."""
    k = _kernel32()
    if k is None:
        return False
    try:
        return bool(k.GetConsoleWindow())
    except Exception:
        return False


def windows_flags(flags):
    """creationflags with CREATE_NO_WINDOW added, unless the caller asked for a console (or none at all) itself."""
    flags = int(flags or 0)
    if flags & (CREATE_NEW_CONSOLE | DETACHED_PROCESS):
        return flags
    return flags | CREATE_NO_WINDOW


def adjust(arguments, windows=None, environ=None, has_console=None):
    """apply the rule to Popen's bound arguments (a dict, changed in place). Returned for tests."""
    windows = (os.name == "nt") if windows is None else windows
    environ = os.environ if environ is None else environ
    if windows:
        if environ.get("PXA_CHILD_CONSOLE") == "1":
            return arguments
        if not (parent_has_console() if has_console is None else has_console):
            arguments["creationflags"] = windows_flags(arguments.get("creationflags", 0))
        return arguments
    if arguments.get("env") is not None:
        return arguments                       # an environment the caller built on purpose (the engine's own launches)
    ld = environ.get("LD_LIBRARY_PATH")
    if not ld:
        return arguments
    dirs = bundled_lib_dirs(environ)
    if not dirs:
        return arguments
    saved = bool(getattr(sys, "_MEIPASS", None)) and "LD_LIBRARY_PATH_ORIG" in environ
    as_is = os.pathsep.join(x for x in ld.split(os.pathsep) if x) or None
    if strip_ld_path(ld, dirs) == as_is and not saved:
        return arguments                       # nothing of the package's on the path: leave the call untouched
    prog = _program(arguments.get("args"), arguments.get("executable"), arguments.get("shell"), None)
    if is_package_program(prog, environ):
        return arguments
    arguments["env"] = system_env(environ)
    return arguments


def _popen_init(self, *a, **kw):
    try:
        bound = _SIG.bind(self, *a, **kw)
    except TypeError:
        return _ORIG_INIT(self, *a, **kw)      # let Popen raise its own error for a bad call
    adjust(bound.arguments)
    return _ORIG_INIT(*bound.args, **bound.kwargs)


def install():
    """patch subprocess.Popen once for this process (every subprocess.run/call/check_output, os.popen and webbrowser
    goes through it)."""
    if _INSTALLED[0]:
        return False
    subprocess.Popen.__init__ = _popen_init
    _INSTALLED[0] = True
    return True


def open_url(url):
    """open url in the user's browser, with the system's own libraries (the rule above, applied explicitly)."""
    import webbrowser
    install()
    return webbrowser.open(url)
