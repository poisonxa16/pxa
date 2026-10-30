"""PXA Control: the launcher bridge -- the auto-start rule, the run records a second `pxa` reads, and starting or
reusing a background Control (`ensure_control`). See the comment block in core above CONTROL_IDLE_S."""
from .core import *  # noqa: F401,F403
from .auth import *  # noqa: F401,F403


def control_env_mode(environ=None):
    """PXA_CONTROL -> 'on' | 'off' | 'auto' (unset, empty, 'auto' or anything else)."""
    v = ((environ if environ is not None else os.environ).get("PXA_CONTROL") or "").strip().lower()
    return "off" if v in _OFF_WORDS else "on" if v in _ON_WORDS else "auto"


def autostart_decision(no_control=False, environ=None, tty=False, container=None):
    """-> (on, why). Whether PXA Control comes up by itself (next to a server, or for a bare `pxa`).
    `why` is one short line for the user; None when they switched it off themselves."""
    env = environ if environ is not None else os.environ
    if no_control:
        return False, None
    if env.get("PXA_CONTROL_SPAWNED"):
        return False, "started by PXA Control"
    mode = control_env_mode(env)
    if mode == "off":
        return False, None
    if mode == "on":
        return True, "PXA_CONTROL=1"
    if container:
        return False, (f"inside a container ({container}): opt in with PXA_CONTROL=1 and publish "
                       f"the port, e.g. -p {DEFAULT_PORT}:{DEFAULT_PORT}")
    if not tty:
        return False, "not started from a terminal; PXA_CONTROL=1 turns it on"
    return True, "a terminal"


def write_run_record(port, lan, companion, token=None):
    """run/control-<pid>.json (0600 in a 0700 dir). It keeps the session token, so `pxa` run again opens this
    Control signed in; another user cannot read it (H1)."""
    rec = {"pid": os.getpid(), "start": proc_start_ticks(os.getpid()), "port": int(port), "lan": bool(lan),
           "companion": bool(companion), "since": time.time(), "version": CONTROL_VERSION}
    if token:
        rec["token"] = str(token)
    p = os.path.join(_private_dir("run"), f"control-{os.getpid()}.json")
    _write_private_json(p, rec)
    return p


def running_controls():
    """this user's Controls that are up, the ones a person opened first, then the oldest."""
    recs = [r for _p, r in _live_records("run", "control-") if isinstance(r.get("port"), int)]
    return sorted(recs, key=lambda r: (bool(r.get("companion")), float(r.get("since") or 0)))


def control_answers(port, token=None, timeout=1.5):
    """True when 127.0.0.1:port is a PXA Control (its /api/health answers with control_version; open on a
    loopback Control, token-guarded on a LAN one)."""
    from .core import _NO_PROXY_OPENER   # loopback: the token never goes through http_proxy / the system proxy
    try:
        rq = urllib.request.Request(f"http://127.0.0.1:{int(port)}/api/health",
                                    headers={"X-PXA-Token": token} if token else {})
        with _NO_PROXY_OPENER.open(rq, timeout=timeout) as r:
            return "control_version" in json.loads(r.read().decode("utf-8", "replace") or "{}")
    except Exception:       # noqa: BLE001
        return False


def control_url(port, lan=False, token=None):
    """the address that opens this Control signed in: the token rides in the query string once; the server
    swaps it for a cookie and redirects to the bare address (H1)."""
    return f"http://127.0.0.1:{int(port)}/" + (f"?token={token}" if token else "")


def find_running(lan=None, port=None, timeout=1.5):
    """The Control this user already has up, as {pid, port, lan, companion, url}, or None. lan=True
    asks for one that listens beyond this machine; port= for that port only."""
    for rec in running_controls():
        if lan and not rec.get("lan"):
            continue
        if port and rec["port"] != int(port):
            continue
        token = rec.get("token") or (load_or_make_token() if rec.get("lan") else None)
        if control_answers(rec["port"], token, timeout):
            return dict(rec, url=control_url(rec["port"], rec.get("lan"), token))
    return None


def control_log_path():
    d = config_dir()
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "control.log")
    try:
        if os.path.getsize(p) > (1 << 20):         # keep one older generation, a megabyte each
            os.replace(p, p + ".1")
    except OSError:
        pass
    return p


def start_detached(argv, env=None, log_path=None):
    """Run argv in its own session (no terminal, so closing the terminal or Ctrl-C on the server does
    not reach it), stdin /dev/null, output appended to log_path. Returns once the intermediate shell
    has exited: the program is then a child of init, not of the caller, which is about to exec into a
    server that would never reap it."""
    out = subprocess.DEVNULL
    if log_path:
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        out = os.fdopen(fd, "ab")
    try:
        p = subprocess.Popen(["/bin/sh", "-c", '"$@" &', "sh"] + list(argv), stdin=subprocess.DEVNULL,
                             stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                             close_fds=True)
        p.wait(timeout=15)
    finally:
        if out is not subprocess.DEVNULL:
            out.close()


def ensure_control(launcher_path, port=DEFAULT_PORT, lan=False, avoid=(), models_dirs=(), wait_s=8.0,
                   python=None):
    """Reuse this user's running Control, or start one in the background. -> {url, port, reused, lan}
    or {error, ...}. Never raises: a Control that cannot start must not stop the server."""
    try:
        r = find_running(lan=True if lan else None)
        if r:
            return {"url": r["url"], "port": r["port"], "reused": True, "lan": bool(r.get("lan"))}
        token = load_or_make_token() if lan else None
        p = free_port(int(port or DEFAULT_PORT), host="0.0.0.0" if lan else "127.0.0.1", avoid=set(avoid or ()))
        argv = [python or sys.executable, launcher_path, "--gui", "--control-companion", "--no-browser",
                "--port", str(p)] + (["--lan"] if lan else [])
        for d in models_dirs or ():
            argv += ["--models-dir", d]
        env = dict(os.environ)
        env["PXA_CONTROL_SPAWNED"] = "1"
        log = control_log_path()
        start_detached(argv, env, log)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if control_answers(p, token, timeout=0.5):
                rec = find_running(port=p)          # its run record holds the session token it made (H1)
                url = rec["url"] if rec else control_url(p, lan, token)
                return {"url": url, "port": p, "reused": False, "lan": bool(lan), "log": log}
            time.sleep(0.15)
        return {"error": f"it did not answer on port {p} within {wait_s:.0f} s (log: {log})", "port": p, "log": log}
    except Exception as e:      # noqa: BLE001
        return {"error": f"{e.__class__.__name__}: {e}"}


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]   # star-import carries the _private names too
