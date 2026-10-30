"""PXA Control as a desktop app: the double-click entry of the compiled pxa-control binary (and `python -m pxa_control`).

No flag is needed for the common case. On start it:
  1. reuses this user's running Control if there is one (opens it signed in and exits: one Control per user);
  2. finds the engine itself: PXA_ENGINE_DIR if set, else an engine shipped inside or beside this app
     (<bundle>/engine, <app dir>/engine, the release layout <app dir>/bin), else the launcher's own search;
  3. remembers the models folder (control.json); on the very first run with none it uses ~/PXA/models;
  4. starts the server on 127.0.0.1 (the first free port from 7777), opens the browser on the signed-in address;
  5. puts an icon in the tray / menu bar with Open and Quit when a tray is available (pystray), else runs until
     Ctrl-C / SIGTERM / closing the terminal, exactly like `pxa --gui`.
GPUs are detected by Control itself (nvidia-smi / NVML; the Rig tab says what it found, a box without a GPU still opens).
"""
import importlib
import importlib.util
import os
import sys
import threading

from . import core as _core
from . import bridge as _bridge
from . import server as _server

APP_NAME = "PXA Control"


def frozen():
    """True in a compiled build (Nuitka sets __compiled__ on every module it compiles; PyInstaller sets sys.frozen)."""
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def app_dir():
    """where the user's copy of the app lives: the folder of the binary (a onefile build runs from a cache folder,
    so argv[0] names the file the user double-clicked), else tools/.. of a source checkout."""
    if frozen():
        return os.path.dirname(os.path.abspath(sys.argv[0] or sys.executable))
    return os.path.dirname(_core.HERE)


def bundle_root():
    """the folder the build put its data in (onefile: the extraction folder), or None in a source checkout."""
    env = os.environ.get("PXA_BUNDLE_ROOT")
    if env:
        return env
    return _core.HERE if frozen() else None


def _exe(d, name):
    for n in (name, name + ".exe"):
        p = os.path.join(d, "bin", n)
        if os.path.isfile(p):
            return p
    return None


def find_engine(environ=None):
    """-> (engine dir or None, why). An engine dir holds bin/llama-server (the launcher's own rule)."""
    env = environ if environ is not None else os.environ
    if env.get("PXA_ENGINE_DIR"):
        return env["PXA_ENGINE_DIR"], "PXA_ENGINE_DIR"
    cands = []
    b = bundle_root()
    if b:
        cands.append((os.path.join(b, "engine"), "inside this app"))
    a = app_dir()
    cands += [(os.path.join(a, "engine"), "beside this app"), (a, "this release folder")]
    for d, why in cands:
        exe = _exe(d, "llama-server")
        if exe:
            if os.name != "nt" and not os.access(exe, os.X_OK):     # a data file in a onefile payload may lose +x
                try:
                    for n in os.listdir(os.path.join(d, "bin")):
                        p = os.path.join(d, "bin", n)
                        os.chmod(p, os.stat(p).st_mode | 0o111)
                except OSError:
                    pass
            return d, why
    return None, "not shipped with this app: the launcher searches the usual places"


def default_models_dir():
    return os.path.join(os.path.expanduser("~"), "PXA", "models")


def models_dirs_for_first_run(environ=None):
    """[] when the user already has a models folder (control.json or PXA_MODELS_DIR); else [~/PXA/models], created."""
    env = environ if environ is not None else os.environ
    if env.get("PXA_MODELS_DIR") or _core.load_config().get("model_dirs"):
        return []
    d = default_models_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return []
    return [d]


def load_launcher():
    """the launcher module (pxa-launch.py): compiled builds carry it as pxa_launch, a checkout loads the file."""
    try:
        return importlib.import_module("pxa_launch")
    except ImportError:
        pass
    p = os.path.join(_core.HERE, "pxa-launch.py")
    spec = importlib.util.spec_from_file_location("pxa_launch", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pxa_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


def _log_to_file_when_headless():
    """a double-clicked app has no terminal: its output goes to control.log (0600) in the config folder."""
    try:
        if sys.stdout is not None and sys.stdout.isatty():
            return None
    except (AttributeError, ValueError):
        pass
    try:
        path = _bridge.control_log_path()
        f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        sys.stdout = sys.stderr = f
        return path
    except OSError:
        return None


def third_party_dir():
    """LGPL libraries (pystray, python-xlib) ship as plain .py files beside a compiled build, never compiled in, so a
    user can replace them (LGPL); this puts that folder on sys.path."""
    b = bundle_root()
    for d in ((os.path.join(b, "third_party.zip"), os.path.join(b, "third_party")) if b else ()):
        if os.path.exists(d):                     # a zip (pure-Python packages import from it) or a folder
            if d not in sys.path:
                sys.path.append(d)
            return d
    return None


def tray_available():
    if os.environ.get("PXA_CONTROL_TRAY", "").lower() in ("0", "off", "no"):
        return False
    third_party_dir()
    try:
        import pystray  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except Exception:          # noqa: BLE001 - no tray library, no display, no backend: run without a tray
        return False


def run_tray(url_fn, quit_fn, icon_path):
    """Open / Quit in the tray (Windows), menu bar (macOS) or status area (Linux). Runs on the main thread."""
    import pystray
    from PIL import Image
    img = Image.open(icon_path) if icon_path and os.path.isfile(icon_path) else Image.new("RGB", (64, 64), (224, 138, 60))

    def _open(icon, item):
        _server.open_in_browser(url_fn())

    def _quit(icon, item):
        quit_fn()
        icon.stop()
    icon = pystray.Icon("pxa-control", img, APP_NAME, menu=pystray.Menu(
        pystray.MenuItem("Open " + APP_NAME, _open, default=True), pystray.MenuItem("Quit", _quit)))
    icon.run()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="pxa-control", description=APP_NAME + ": start it, open it, nothing to set.")
    ap.add_argument("--port", type=int, default=0, help="listen here (default: the first free port from 7777)")
    ap.add_argument("--lan", action="store_true", help="also listen on the network, behind the access token")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--no-tray", action="store_true", help="no tray / menu-bar icon")
    ap.add_argument("--models-dir", action="append", default=[], help="a folder with .gguf models (remembered)")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    a = ap.parse_args(argv)
    if a.version:
        print(f"{APP_NAME} {_core.CONTROL_VERSION or 'dev'}{' (compiled)' if frozen() else ''}")
        return 0
    running = _bridge.find_running(lan=True if a.lan else None, port=a.port or None)
    if running:
        print(f"{APP_NAME}: already running, opening {_bridge.control_url(running['port'])}")
        if not a.no_browser:
            _server.open_in_browser(running["url"], wait=True)
        return 0
    log = _log_to_file_when_headless()
    eng, why = find_engine()
    if eng and not os.environ.get("PXA_ENGINE_DIR"):
        os.environ["PXA_ENGINE_DIR"] = eng
    print(f"{APP_NAME}: engine {eng or '(searching)'} ({why})" + (f"; log {log}" if log else ""))
    dirs = list(a.models_dir) or models_dirs_for_first_run()
    L = load_launcher()
    port = a.port or _core.free_port(_core.DEFAULT_PORT, host="0.0.0.0" if a.lan else "127.0.0.1")
    if a.no_tray or a.lan or not tray_available():
        return _server.serve(L, port=port, lan=a.lan, open_browser=not a.no_browser, models_dirs=dirs)
    rc = {}
    th = threading.Thread(target=lambda: rc.update(code=_server.serve(L, port=port, lan=a.lan, open_browser=not a.no_browser,
                                                                     models_dirs=dirs)), name="pxa-control-server", daemon=True)
    th.start()

    def _url():
        r = _bridge.find_running(port=port)
        return r["url"] if r else _bridge.control_url(port)

    def _quit():
        srv = _server.SERVING.get(port)
        if srv is not None:
            srv.shutdown()
        th.join(60)
    run_tray(_url, _quit, os.path.join(_core.UI_DIR, "mark.png"))
    th.join(60)
    return rc.get("code", 0)


if __name__ == "__main__":
    sys.exit(main())
