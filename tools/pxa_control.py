"""PXA Control - the launcher's browser front end (`pxa-launch --gui`).

A stdlib-only HTTP server plus one self-contained page (tools/pxa_control_ui/index.html). It is a
FRONT DOOR, like the terminal UI: it collects the same answers a command line would carry, builds
them into the launcher's own argparse namespace (build_parser), and runs the launcher's own
plan_and_build(), doctor(), scan_models() and vram_check(). It decides nothing about the engine.

What it adds on top of the CLI:
  * a live view of the rig (nvidia-smi + sysfs telemetry, --doctor findings);
  * a model library over folders you name, remembered in ~/.config/pxa/control.json;
  * several servers at once: named, persisted server profiles (model, cards, port, levers,
    extra engine args, engine build), each with Start / Stop / Restart, health, log and metrics,
    with port and card conflicts refused BEFORE anything is spawned;
  * a dashboard of every llama-server on the box: the ones it started, plus docker containers and
    processes it did not start (the seats), discovered and monitored READ-ONLY unless you adopt one
    and explicitly allow control;
  * the running server's /pxa/stats as charts (a local history of the GUI's own requests when the
    server is too old to have /pxa/stats), a short benchmark (tools/pxa-bench.py's prompts, REPS 3);
  * a Live tab: a rolling in-memory history of every card (nvidia-smi) and every running server
    (/slots, /props, /pxa/stats, /metrics when present) drawn as thin-line SVG charts, per-card strips and a
    request timeline; the sampler runs only while somebody is looking (class Live, GET /api/live);
  * a chat box that streams from the seat's OpenAI endpoint;
  * an Encode tab: a wizard that turns a Hugging Face model into a PXQ / PXQN file (tools/pxa_encode*.py). It drives the
    PXA Quantizer's command line as a subprocess and never touches its internals; the licence key it may hold is sent
    only to the licence server and never reaches a log, a bug report or a score.
  * a per-model Thinking switch (on / off / auto) and thinking-token budget, mapped to whatever
    mechanism the model's family uses (tools/pxa_thinking.py, tools/pxa_thinking_profiles.json):
    applied at launch as llama-server flags and per chat request through the engine proxy.
  * a Profiles tab (v3.1, tools/pxa_ctl/): per-card / per-group GPU profiles (power limit, clocks, persistence,
    thermal guard, auto-start of saved servers with health wait and restart policy, schedules, Quiet mode), live per-card
    readout. Changing a card needs "allow_gpu_control": true in control.json (default off) AND a typed confirm; a
    reserved card, a lock file or maintenance mode refuses changes and launches onto that card.

SAFETY (a web page drives GPUs, so every edge is closed on purpose):
  * binds 127.0.0.1 unless --lan; with --lan a random token is required (cookie after first visit);
  * Host header must name this server (DNS-rebinding guard); POSTs need a same-origin Origin;
  * no shell passthrough anywhere: the command is the launcher's, argv only, never a shell;
  * env overrides accept only names in the PXA lever catalog, values from a narrow charset;
  * every numeric input is range-checked; model paths must be .gguf files under a configured folder;
  * the engine proxy forwards a fixed list of paths to a LOOPBACK port only, and only to a port
    of a server this GUI knows (one it started, one it discovered, or the attach port);
  * extra engine args are an allow-list of llama-server flags (no --path, no --log-file, ...);
  * a server it did not start is never stopped or restarted unless it was adopted with control
    switched on, and every such action names the container it acts on.
  * the chat's host access (v3.1) is off by default, per session, allowlist-only (host-allow.json in the
    config dir), argv-only with no shell, and refused outright when Control is on the LAN unless the owner
    sets host_access_lan; every host call needs an approval card and writes a redacted audit record.
  * every POST/DELETE body must be a JSON object; every error is {"error", "code"} with a matching HTTP status;
    GET /api/health answers even when nvidia-smi is missing.
"""

import ctypes
import hmac
import shlex
import http.server
import importlib.util
import io
import json
import os
import re
import secrets
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

try:                                     # the Encode tab: a packaging without these modules loses the tab, not the GUI
    import pxa_encode as ENC
    import pxa_encode_pkg as ENCPK
    ENC_IMPORT_ERROR = None
except Exception as _enc_err:            # noqa: BLE001
    ENC = ENCPK = None
    ENC_IMPORT_ERROR = f"{_enc_err.__class__.__name__}: {_enc_err}"
import pxa_thinking as TH

try:                                     # the GPU profile backend (v3.1, tools/pxa_ctl/): without it, only the Profiles tab is lost
    import pxa_ctl as CTL
    from pxa_ctl import driver as CTL_DRV, service as CTL_SVC
    CTL_IMPORT_ERROR = None
    CTL_ERRORS = (CTL.CtlError,)
except Exception as _ctl_err:            # noqa: BLE001
    CTL = CTL_DRV = CTL_SVC = None
    CTL_IMPORT_ERROR = f"{_ctl_err.__class__.__name__}: {_ctl_err}"
    CTL_ERRORS = ()

try:                                     # the history store behind the Live tab (v3.1): without it, Live is memory-only as before
    import pxa_telemetry as TEL
    TEL_IMPORT_ERROR = None
except Exception as _tel_err:            # noqa: BLE001
    TEL = None
    TEL_IMPORT_ERROR = f"{_tel_err.__class__.__name__}: {_tel_err}"

try:                                     # the in-house chat agent (v3.1, tools/pxa_chat/): without it, Chat is plain chat as before
    import pxa_chat as CHAT
    CHAT_IMPORT_ERROR = None
except Exception as _chat_err:           # noqa: BLE001
    CHAT = None
    CHAT_IMPORT_ERROR = f"{_chat_err.__class__.__name__}: {_chat_err}"

try:                                     # the licensed library updater (v3.1, tools/pxa_lib_update.py): without it, that one panel is lost
    import pxa_lib_update as LU
    LIB_IMPORT_ERROR = None
except Exception as _lib_err:            # noqa: BLE001
    LU = None
    LIB_IMPORT_ERROR = f"{_lib_err.__class__.__name__}: {_lib_err}"
PK_ERRORS = (ENCPK.PackageError,) if ENCPK is not None else ()   # the updater raises the package verifier's own error

try:                                     # the optional MQTT publisher (v3.1, tools/pxa_mqtt.py): off unless the user turns it on
    import pxa_mqtt as MQTT
    MQTT_IMPORT_ERROR = None
except Exception as _mqtt_err:           # noqa: BLE001
    MQTT = None
    MQTT_IMPORT_ERROR = f"{_mqtt_err.__class__.__name__}: {_mqtt_err}"

DEFAULT_PORT = 7777
DEFAULT_SERVER_PORT = 8080
HERE = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.environ.get("PXA_CONTROL_UI_DIR") or os.path.join(HERE, "pxa_control_ui")
TOKEN_COOKIE = "pxa_control_token"
MAX_BODY = 1 << 20

# ---------------------------------------------------------------------------------------------
# config (~/.config/pxa/control.json), presets, local history
# ---------------------------------------------------------------------------------------------


def config_dir():
    d = os.environ.get("PXA_CONTROL_CONFIG_DIR")
    if d:
        return d
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "pxa")


def config_path():
    return os.path.join(config_dir(), "control.json")


_cfg_lock = threading.Lock()


def load_config():
    try:
        with open(config_path()) as f:
            c = json.load(f)
        if not isinstance(c, dict):
            c = {}
    except Exception:
        c = {}
    c.setdefault("model_dirs", [])
    c.setdefault("presets", {})
    c.setdefault("attach_port", 0)
    c.setdefault("user_name", "")
    c.setdefault("host_access_lan", False)     # chat host access over the network: off (see chat_host_policy)
    return c


def save_config(c):
    d = config_dir()
    os.makedirs(d, exist_ok=True)
    tmp = config_path() + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)
    os.replace(tmp, config_path())


def load_or_make_token():
    """The --lan access token: PXA_CONTROL_TOKEN if set, else the one kept in the config dir (0600),
    else a new random one saved there. It used to change at every restart, which broke every
    bookmark and phone home-screen shortcut to the GUI."""
    env = os.environ.get("PXA_CONTROL_TOKEN", "").strip()
    if env:
        if len(env) < 16 or not re.match(r"^[A-Za-z0-9_\-]+$", env):
            raise SystemExit("PXA_CONTROL_TOKEN must be at least 16 of A-Z a-z 0-9 _ -")
        return env
    p = os.path.join(config_dir(), "token")
    try:
        with open(p) as f:
            t = f.read().strip()
        if len(t) >= 16 and re.match(r"^[A-Za-z0-9_\-]+$", t):
            return t
    except OSError:
        pass
    t = secrets.token_urlsafe(18)
    try:
        os.makedirs(config_dir(), exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(t + "\n")
    except OSError:
        pass
    return t


def append_jsonl(name, rec, keep=20000):
    p = os.path.join(config_dir(), name)
    try:
        os.makedirs(config_dir(), exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError:
        pass


def read_jsonl(name, since=0.0, limit=5000):
    p = os.path.join(config_dir(), name)
    out = []
    try:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and float(r.get("ts", 0) or 0) >= since:
                    out.append(r)
    except OSError:
        pass
    return out[-limit:]


# ---------------------------------------------------------------------------------------------
# PXA Control alongside every server (2026-10-05). `pxa` with no arguments opens Control; a server
# started from the command line (pxa-launch with a model, run-server.sh) brings one up next to it.
# Three small records in the per-user config dir (0700) make that safe and single:
#   run/control-<pid>.json   a Control that is up: a second `pxa` finds it and reuses it
#   launched/<pid>.json      a server the launcher started: Control lists it as adopted and may stop it
#   control.log              what a Control started in the background printed
# Every record names its process by pid AND /proc start time, so a recycled pid never matches.
#
# THE RULE (also in docs/LAUNCHER.md "PXA Control opens by itself"):
#   --no-control or PXA_CONTROL=0       never
#   PXA_CONTROL=1                       always (a service, a container, a script: you asked for it)
#   a process PXA Control started       never (PXA_CONTROL_SPAWNED: no Control inside a Control)
#   inside a container                  no: 127.0.0.1 there is unreachable from the host
#   stdin is not a terminal             no: a service or a script must not open a web port on its own
#   otherwise (a person at a terminal)  yes, on 127.0.0.1 only; --lan is the only way off this machine
# A Control that started on its own closes itself once no server is running and no page has asked
# it anything for CONTROL_IDLE_S (10 min). One you opened yourself (pxa, pxa --gui) runs until Ctrl-C.
# ---------------------------------------------------------------------------------------------
CONTROL_IDLE_S = 600.0
CONTROL_IDLE_POLL_S = 15.0
_OFF_WORDS = ("0", "off", "no", "false", "never")
_ON_WORDS = ("1", "on", "yes", "true", "always")


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


def _private_dir(name):
    d = os.path.join(config_dir(), name)
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def _write_private_json(path, rec):
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)


def _proc_stat(pid):
    """-> (state, start_ticks) from /proc/<pid>/stat, or None (gone, or no /proc)."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as f:
            st = f.read().decode("utf-8", "replace")
        rest = st[st.rindex(")") + 2:].split()
        return rest[0], int(rest[19])
    except (OSError, ValueError, IndexError):
        return None


def proc_start_ticks(pid):
    s = _proc_stat(pid)
    return s[1] if s else None


def pid_alive(pid, start=None):
    """True while pid runs (not a zombie) and, given `start`, is still the same process."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    s = _proc_stat(pid)
    if s is None:
        if os.path.isdir("/proc/self"):
            return False
        try:                                # no /proc (not Linux): existence only
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False
    if s[0] in ("Z", "X"):
        return False
    return start is None or s[1] == start


def _live_records(sub, prefix=""):
    """[(path, record)] of the records in config_dir()/sub whose process still runs; stale ones are removed."""
    d = os.path.join(config_dir(), sub)
    try:
        names = os.listdir(d)
    except OSError:
        return []
    out = []
    for n in sorted(names):
        if not (n.startswith(prefix) and n.endswith(".json")):
            continue
        p = os.path.join(d, n)
        try:
            with open(p) as f:
                rec = json.load(f)
            ok = isinstance(rec, dict) and pid_alive(rec.get("pid"), rec.get("start"))
        except (OSError, ValueError):
            rec, ok = None, False
        if ok:
            out.append((p, rec))
        else:
            try:
                os.unlink(p)
            except OSError:
                pass
    return out


def write_run_record(port, lan, companion):
    rec = {"pid": os.getpid(), "start": proc_start_ticks(os.getpid()), "port": int(port), "lan": bool(lan),
           "companion": bool(companion), "since": time.time(), "version": CONTROL_VERSION}
    p = os.path.join(_private_dir("run"), f"control-{os.getpid()}.json")
    _write_private_json(p, rec)
    return p


def running_controls():
    """this user's Controls that are up, the ones a person opened first, then the oldest."""
    recs = [r for _p, r in _live_records("run", "control-") if isinstance(r.get("port"), int)]
    return sorted(recs, key=lambda r: (bool(r.get("companion")), float(r.get("since") or 0)))


def control_answers(port, token=None, timeout=1.5):
    """True when 127.0.0.1:port is a PXA Control (its /api/info answers with control_version)."""
    try:
        rq = urllib.request.Request(f"http://127.0.0.1:{int(port)}/api/info",
                                    headers={"X-PXA-Token": token} if token else {})
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return "control_version" in json.loads(r.read().decode("utf-8", "replace") or "{}")
    except Exception:       # noqa: BLE001
        return False


def control_url(port, lan=False, token=None):
    return f"http://127.0.0.1:{int(port)}/" + (f"?token={token}" if lan and token else "")


def find_running(lan=None, port=None, timeout=1.5):
    """The Control this user already has up, as {pid, port, lan, companion, url}, or None. lan=True
    asks for one that listens beyond this machine; port= for that port only."""
    for rec in running_controls():
        if lan and not rec.get("lan"):
            continue
        if port and rec["port"] != int(port):
            continue
        token = load_or_make_token() if rec.get("lan") else None
        if control_answers(rec["port"], token, timeout):
            return dict(rec, url=control_url(rec["port"], rec.get("lan"), token))
    return None


def record_launched(pid, port=None, model=None, gpus=None, by="pxa-launch", host=None):
    """The launcher (or run-server.sh) says: pid is a server I started for this user. Control lists it
    as adopted, with Stop. Written before the exec, so the pid is the server's own."""
    try:
        port = int(port) if port not in (None, "") else None
    except (TypeError, ValueError):
        port = None
    if isinstance(gpus, str):
        gpus = [int(x) for x in gpus.split(",") if x.strip().isdigit()]
    rec = {"pid": int(pid), "start": proc_start_ticks(pid), "port": port, "host": host,
           "model": model or None, "gpus": list(gpus) if gpus else None, "by": by, "since": time.time()}
    p = os.path.join(_private_dir("launched"), f"{int(pid)}.json")
    _write_private_json(p, rec)
    return p


def launched_servers():
    """{pid: record} of the servers the launcher started that still run."""
    return {int(r["pid"]): r for _p, r in _live_records("launched")}


def launched_signature():
    """the names in launched/ (cheap: one listdir), so a cached fleet notices a new or finished server."""
    try:
        return tuple(sorted(os.listdir(os.path.join(config_dir(), "launched"))))
    except OSError:
        return ()


def forget_launched(pid):
    try:
        os.unlink(os.path.join(config_dir(), "launched", f"{int(pid)}.json"))
    except (OSError, ValueError, TypeError):
        pass


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
                return {"url": control_url(p, lan, token), "port": p, "reused": False, "lan": bool(lan), "log": log}
            time.sleep(0.15)
        return {"error": f"it did not answer on port {p} within {wait_s:.0f} s (log: {log})", "port": p, "log": log}
    except Exception as e:      # noqa: BLE001
        return {"error": f"{e.__class__.__name__}: {e}"}


# ---------------------------------------------------------------------------------------------
# the lever catalog (common/pxa-lever-catalog.inc, generated from docs/LEVERS.md)
# ---------------------------------------------------------------------------------------------
_Q = r'"((?:[^"\\]|\\.)*)"'
CATALOG_RE = re.compile(r'^\{\s*' + r',\s*'.join([_Q] * 6) + r'\s*\}')
LEVER_NAME_RE = re.compile(r"^PX[AQ]_[A-Z0-9_]{1,64}$")
LEVER_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:=+\-/]{0,200}$")


def catalog_candidates():
    return [os.environ.get("PXA_LEVER_CATALOG", ""),
            os.path.join(HERE, "..", "common", "pxa-lever-catalog.inc"),
            os.path.join(UI_DIR, "pxa-lever-catalog.inc"),
            "/usr/local/share/pxa/pxa-lever-catalog.inc"]


def load_catalog(path=None):
    """[{name, default, scope, status, evidence, rule}] from the generated catalog; [] if absent."""
    for p in ([path] if path else catalog_candidates()):
        if not p or not os.path.isfile(p):
            continue
        rows = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = CATALOG_RE.match(line.strip())
                if not m:
                    continue
                name = m.group(1)
                if not LEVER_NAME_RE.match(name):
                    continue
                g = [x.replace('\\"', '"') for x in m.groups()]
                rows.append({"name": g[0], "default": g[1], "scope": g[2], "status": g[3],
                             "evidence": g[4], "rule": g[5]})
        return rows, os.path.abspath(p)
    return [], None


# Built-in levers (2026-10-08): a release catalog is the PUBLIC table, without the levers the closed PXQN library reads
# (the Overdrive set the Flash-Next preset turns on among them). Its last lines carry their hashes, no names
# ("// built-in:" every one, "// built-in-preset:" the preset's), so such a name is accepted (passed into the server's
# environment like any lever) and shown as built-in instead of being refused as unknown.
def lever_hash(name):
    """FNV-1a 64, 16 hex digits (scripts/pxa-lever-catalog.py lever_hash; the page computes the same)."""
    h = 0xcbf29ce484222325
    for b in name.encode("utf-8"):
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def load_builtin(path):
    """(every built-in hash, the preset's) from the catalog's trailing lines; empty sets for a full catalog."""
    every, preset = set(), set()
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                for tag, dst in (("// built-in:", every), ("// built-in-preset:", preset)):
                    if line.startswith(tag):
                        dst.update(x for x in line[len(tag):].split() if re.match(r"^[0-9a-f]{16}$", x))
    return every, preset


def builtin_kind(name, every, preset):
    """'preset' | 'builtin' | None for a lever name that is not in the catalog."""
    if not every or not isinstance(name, str) or not LEVER_NAME_RE.match(name):
        return None
    cands = [name] + [name[:k] + "*" for k in range(5, len(name))]
    hs = {lever_hash(c) for c in cands}
    if hs & preset:
        return "preset"
    if hs & every:
        return "builtin"
    return None


class LeverNames(set):
    """The catalog's names; `in` also admits the built-in levers (validate_levers takes it as catalog_names)."""

    def __init__(self, names=(), every=(), preset=()):
        super().__init__(names)
        self.every, self.preset = set(every), set(preset)

    def __contains__(self, name):
        return set.__contains__(self, name) or builtin_kind(name, self.every, self.preset) is not None

    def builtin(self, name):
        return None if set.__contains__(self, name) else builtin_kind(name, self.every, self.preset)


def validate_levers(levers, catalog_names):
    """{name: value} -> (clean dict, errors[]). Only catalog names, only a narrow value charset;
    an empty value means 'not overridden' and is dropped."""
    clean, errs = {}, []
    if levers in (None, ""):
        return clean, errs
    if not isinstance(levers, dict):
        return {}, ["levers must be an object of NAME: value"]
    if len(levers) > 64:
        return {}, ["at most 64 lever overrides"]
    for k, v in levers.items():
        if not isinstance(k, str) or not LEVER_NAME_RE.match(k):
            errs.append(f"not a lever name: {k!r}")
            continue
        if k not in catalog_names:
            errs.append(f"{k} is not in the PXA lever catalog")
            continue
        if isinstance(v, bool):
            v = "1" if v else "0"
        if isinstance(v, (int, float)):
            v = str(v)
        if not isinstance(v, str) or not LEVER_VALUE_RE.match(v):
            errs.append(f"{k}: value must be up to 200 of A-Z a-z 0-9 _ . , : = + - /")
            continue
        if v == "":
            continue
        clean[k] = v
    return clean, errs


_OFF_RE = re.compile(r"^\s*(off|0|unset|false|no|none|disabled)\b", re.I)
_ON_RE = re.compile(r"^\s*(on|1|true|yes|enabled)\b", re.I)
_BOOL_VALUES = {"0", "1", "on", "off", "true", "false", "yes", "no"}


def lever_default_state(row):
    """'off' | 'on' | 'value' | 'site': what the engine does when the lever is NOT set, read off the
    catalog's default column. 'site' = it depends on the card/model (see the rule)."""
    d = (row.get("default") or "").strip()
    if not d or d.lower().startswith(("see site", "rule", "detect", "auto", "table", "tuned", "sysfs")):
        return "site"
    if _OFF_RE.match(d):
        return "off"
    if _ON_RE.match(d):
        return "on"
    return "value"


def lever_kind(row):
    """'bool' | 'int' | 'text': the value shape the catalog implies (a hint, never a refusal)."""
    d = (row.get("default") or "").strip().lower()
    rule = (row.get("rule") or "").strip()
    if re.match(r"^(off|on)\b", d) or "(=1" in d or "(=0" in d or re.match(r"^[01] \((on|off)\)$", d):
        return "bool"
    if re.match(r"^-?\d+$", d):
        return "bool" if (d in ("0", "1") and rule.startswith(("=1", "=0"))) else "int"
    if rule.startswith(("=1:", "=0:", "=1 ", "=0 ")) and d in ("", "unset", "off", "on"):
        return "bool"
    return "text"


def lint_levers(levers, catalog_rows):
    """{name: value} -> [warning]: a value that does not look like what the catalog implies, and a
    lever set to its own default (which changes nothing). Never refuses: the charset check does."""
    by = {r["name"]: r for r in catalog_rows or []}
    out = []
    for k, v in sorted((levers or {}).items()):
        r = by.get(k)
        if not r:
            continue
        kind, sv = lever_kind(r), str(v).strip().lower()
        if kind == "bool" and sv not in _BOOL_VALUES:
            out.append(f"{k}={v}: this lever is an on/off switch (0 or 1)")
        elif kind == "int" and not re.match(r"^-?\d+$", sv):
            out.append(f"{k}={v}: this lever takes a whole number (default {r['default']})")
        st = lever_default_state(r)
        if (st == "off" and sv in ("0", "off", "false", "no")) or (st == "on" and sv in ("1", "on", "true", "yes")):
            out.append(f"{k}={v} is already the default ({r['default']}): setting it changes nothing")
        if r.get("status") in ("diagnostic", "site"):
            out.append(f"{k} is a {r['status']} lever: meant for measurement, not for serving")
    return out


# Extra llama-server arguments a profile may carry, appended after the launcher's own command (the
# last occurrence wins in llama.cpp's parser). An ALLOW-LIST, value count per flag: anything that
# reads or writes files the GUI does not vet (--path, --log-file, --ssl-*, --slot-save-path,
# --lora, -m, --mmproj ...) or changes who can reach the server (--host, --api-key*) is refused.
EXTRA_FLAGS = {
    "-t": 1, "--threads": 1, "-tb": 1, "--threads-batch": 1, "-b": 1, "--batch-size": 1,
    "-ub": 1, "--ubatch-size": 1, "-ngl": 1, "--n-gpu-layers": 1, "-ts": 1, "--tensor-split": 1,
    "-mg": 1, "--main-gpu": 1, "-ot": 1, "--override-tensor": 1, "-c": 1, "--ctx-size": 1,
    "-np": 1, "--parallel": 1, "-n": 1, "--n-predict": 1, "-fa": 1, "--flash-attn": 1,
    "-ctk": 1, "--cache-type-k": 1, "-ctv": 1, "--cache-type-v": 1, "--cache-ram": 1, "-cram": 1,
    "--alias": 1, "-a": 1, "--jinja": 0, "--no-jinja": 0, "--no-context-shift": 0, "--context-shift": 0,
    "--mlock": 0, "--no-mmap": 0, "--metrics": 0, "--slots": 0, "--no-slots": 0, "--no-webui": 0,
    "--cont-batching": 0, "--no-cont-batching": 0, "-cb": 0, "-nocb": 0, "--swa-full": 0, "-kvu": 0,
    "--kv-unified": 0, "--temp": 1, "--top-k": 1, "--top-p": 1, "--min-p": 1, "--repeat-penalty": 1,
    "--presence-penalty": 1, "--frequency-penalty": 1, "--seed": 1, "-s": 1,
    "--reasoning-format": 1, "--reasoning-budget": 1, "--chat-template": 1, "--ctx-checkpoints": 1,
    "-wgt": 1, "--rope-scaling": 1, "--rope-freq-base": 1, "--rope-freq-scale": 1, "--yarn-orig-ctx": 1,
    "--defrag-thold": 1, "-dt": 1, "--timeout": 1, "-to": 1, "--threads-http": 1, "--n-cpu-moe": 1,
    "-ncmoe": 1, "--cpu-moe": 0, "-cmoe": 0, "--draft-max": 1, "--draft-min": 1, "--draft-p-min": 1,
    "-md": 1, "--model-draft": 1, "--sleep-idle-seconds": 1,
    "--spec-replace": 2, "-sm": 1, "--split-mode": 1, "--numa": 1, "--prio": 1, "--poll": 1,
    "--keep": 1, "--no-warmup": 0, "--verbose": 0, "-v": 0, "--log-verbosity": 1, "-lv": 1,
}
EXTRA_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.,:=+\-/\\|*()\[\]^$?]{1,300}$")


def validate_extra_args(x):
    """A string or list -> [argv tokens]. Raises Invalid on a flag outside EXTRA_FLAGS, a missing
    value, or a character outside a conservative set (no quotes, spaces, ;, &, `, $(...) shells)."""
    if x in (None, "", []):
        return []
    if isinstance(x, str):
        try:
            toks = shlex.split(x)
        except ValueError as e:
            raise Invalid(f"extra args: {e}")
    elif isinstance(x, list) and all(isinstance(t, str) for t in x):
        toks = list(x)
    else:
        raise Invalid("extra args must be a string or a list of strings")
    if len(toks) > 64:
        raise Invalid("extra args: at most 64 tokens")
    out, i = [], 0
    while i < len(toks):
        f = toks[i]
        flag, eq, val = f.partition("=") if f.startswith("--") else (f, "", "")
        if flag not in EXTRA_FLAGS:
            raise Invalid(f"extra args: {flag} is not on the allowed list (it may read or write files, "
                          "or change who can reach the server); use a lever or the launcher's settings")
        n = EXTRA_FLAGS[flag]
        vals = [val] if eq else toks[i + 1:i + 1 + n]
        if (eq and n != 1) or len(vals) != n or any(v.startswith("-") and not re.match(r"^-\d", v) and n for v in vals):
            raise Invalid(f"extra args: {flag} takes {n} value(s)")
        for v in vals:
            if not EXTRA_TOKEN_RE.match(v) or "$(" in v:    # argv goes to exec, not a shell; still no $(...)
                raise Invalid(f"extra args: {flag} value {v!r} has characters outside A-Z a-z 0-9 _ . , : = + - / \\ | * ( ) [ ] ^ $ ?")
        out += [flag] + vals
        i += 1 + (0 if eq else n)
    return out


def bounded(fn, timeout, default=None):
    """fn() with a wall-clock budget: a hung nvidia-smi or a stuck engine must not hold the caller
    (the bug report has to work exactly when the rig is stuck). Returns default on timeout/error."""
    box = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as e:      # noqa: BLE001 - reported to the caller as the default
            box["e"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    return box.get("v", default) if not t.is_alive() else default


# ---------------------------------------------------------------------------------------------
# GPU list without nvidia-smi (a container with the compute driver but no utilities)
# ---------------------------------------------------------------------------------------------
_GPU_PROBE = r"""
import ctypes, json, os, sys
out = {"rows": [], "src": None, "err": None}
def nvml():
    for name in ("libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"):
        try:
            lib = ctypes.CDLL(name); break
        except OSError:
            lib = None
    if lib is None or lib.nvmlInit_v2() != 0:
        return False
    n = ctypes.c_uint(0); lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
    for i in range(n.value):
        h = ctypes.c_void_p(); lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
        nm = ctypes.create_string_buffer(96); lib.nvmlDeviceGetName(h, nm, 96)
        uu = ctypes.create_string_buffer(96); lib.nvmlDeviceGetUUID(h, uu, 96)
        ma, mi = ctypes.c_int(0), ctypes.c_int(0); lib.nvmlDeviceGetCudaComputeCapability(h, ctypes.byref(ma), ctypes.byref(mi))
        m = Mem(); lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m))
        out["rows"].append([i, nm.value.decode(), ma.value * 10 + mi.value, m.total >> 20, m.used >> 20, uu.value.decode()])
    out["src"] = "NVML (libnvidia-ml)"
    return True
def cuda():
    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            lib = ctypes.CDLL(name); break
        except OSError:
            lib = None
    if lib is None:
        out["err"] = "neither nvidia-smi, NVML nor libcuda is available"; return False
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    rc = lib.cuInit(0)
    if rc != 0:
        out["err"] = f"cuInit failed ({rc})"; return False
    n = ctypes.c_int(0); lib.cuDeviceGetCount(ctypes.byref(n))
    rows = []
    for i in range(n.value):
        d = ctypes.c_int(0); lib.cuDeviceGet(ctypes.byref(d), i)
        nm = ctypes.create_string_buffer(96); lib.cuDeviceGetName(nm, 96, d)
        ma, mi = ctypes.c_int(0), ctypes.c_int(0)
        lib.cuDeviceGetAttribute(ctypes.byref(ma), 75, d); lib.cuDeviceGetAttribute(ctypes.byref(mi), 76, d)
        tot = ctypes.c_size_t(0); (getattr(lib, "cuDeviceTotalMem_v2", None) or lib.cuDeviceTotalMem)(ctypes.byref(tot), d)
        bus = ctypes.create_string_buffer(32); lib.cuDeviceGetPCIBusId(bus, 32, d)
        rows.append([bus.value.decode(), nm.value.decode(), ma.value * 10 + mi.value, tot.value >> 20])
    rows.sort()
    out["rows"] = [[i, r[1], r[2], r[3], 0, "CUDA-" + r[0]] for i, r in enumerate(rows)]
    out["src"] = "the CUDA driver (libcuda), memory in use unknown"
    return True
try:
    (os.environ.get("PXA_CONTROL_GPU_PROBE") != "cuda" and nvml()) or cuda()
except Exception as e:
    out["err"] = f"{e.__class__.__name__}: {e}"
print(json.dumps(out))
"""


def probe_gpus_fallback(timeout=15):
    """-> (rows, source, error): the card list from NVML, else the CUDA driver API, in a child
    process (a driver call that hangs cannot hang the GUI). Rows have gpu_table()'s shape."""
    try:
        r = subprocess.run([sys.executable, "-c", _GPU_PROBE], capture_output=True, text=True, timeout=timeout)
        d = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception as e:      # noqa: BLE001
        return [], None, f"GPU probe failed: {e.__class__.__name__}"
    rows = []
    for x in d.get("rows") or []:
        try:
            rows.append((int(x[0]), str(x[1]), int(x[2]), int(x[3]), int(x[4]), str(x[5])))
        except (TypeError, ValueError, IndexError):
            continue
    return rows, d.get("src"), d.get("err")


# ---------------------------------------------------------------------------------------------
# launch request validation -> launcher argv
# ---------------------------------------------------------------------------------------------
KV_TYPES = ["auto", "q4_0", "q8_0", "q6_0", "q5_0", "f16"]   # auto = pass nothing; the engine registry picks the measured KV
SPLIT_MODES = ["auto", "layer", "tensor"]
FA_MODES = {"auto": None, "on": "chat", "off": "longdoc"}   # the launcher's FA regime IS the workload
PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]{1,48}$")


class Invalid(ValueError):
    pass


class Reply(object):
    """a route's answer that is not JSON (a CSV download, the Prometheus text)."""

    def __init__(self, body, ctype, code=200, extra=None):
        self.body, self.ctype, self.code, self.extra = body, ctype, code, extra or {}


def content_length(headers, limit):
    """The request's Content-Length as a checked int: a missing header is 0; a negative, non-numeric
    or oversized one is refused (a negative length made rfile.read(-1) block the thread for good)."""
    raw = (headers.get("Content-Length") or "0").strip()
    if not raw.isdigit():
        raise Invalid("bad Content-Length")
    n = int(raw)
    if n > limit:
        raise Invalid("request too large")
    return n


def free_port(start, host="127.0.0.1", avoid=None, span=200):
    avoid = set(avoid) if isinstance(avoid, (set, list, tuple)) else {avoid}
    for p in range(start, start + span):
        if p not in avoid and not port_in_use(p, host) and (host == "127.0.0.1" or not port_in_use(p)):
            return p
    raise Invalid(f"no free server port between {start} and {start + span - 1}: set one by hand")


def port_in_use(port, host="127.0.0.1"):
    """True when something already listens on host:port (the engine would refuse with PORT_GUARD a
    few seconds after Start; saying so before anything is spawned is the clearer answer)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR, as the engine's own listener sets it: a server we just stopped leaves its
    # connections in TIME_WAIT for up to a minute, and without this the probe reported our OWN
    # just-stopped port as "in use by another program" (user report 2026-09-30, Stop -> change model).
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        return False
    except OSError:
        pass
    finally:
        s.close()
    # The bind failed, but that alone does not mean a server is there. The engine (cpp-httplib) sets
    # SO_REUSEPORT, not SO_REUSEADDR, so the TIME_WAIT/FIN_WAIT sockets a stopped engine leaves on
    # its port (one per health poll) refuse a SO_REUSEADDR bind for up to a minute, while the next
    # engine binds there fine. Only a LISTENING socket blocks a server (user report 2026-10-07:
    # Stop, then Start on the same port refused until the port was changed or the GUI restarted).
    listening = _listening_ports()
    if listening is not None:
        return port in listening
    try:                                    # no /proc/net (not Linux): does anything answer?
        with socket.create_connection(("127.0.0.1" if host in ("0.0.0.0", "") else host, port), timeout=1):
            return True
    except OSError:
        return False


def _listening_ports():
    """{port} of every TCP socket in LISTEN state on this machine (Linux /proc/net/tcp{,6}), or None
    when that table cannot be read."""
    ports, seen = set(), False
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(f) as fh:
                next(fh, None)
                for ln in fh:
                    p = ln.split()
                    if len(p) > 3 and p[3] == "0A":         # TCP_LISTEN
                        ports.add(int(p[1].rsplit(":", 1)[1], 16))
            seen = True
        except (OSError, ValueError, IndexError):
            continue
    return ports if seen else None


def _int(body, key, lo, hi, default):
    v = body.get(key, default)
    if v in (None, ""):
        v = default
    if isinstance(v, bool):
        raise Invalid(f"{key} must be an integer")
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise Invalid(f"{key} must be an integer")
    if not lo <= v <= hi:
        raise Invalid(f"{key} must be between {lo} and {hi}")
    return v


def path_under(path, roots):
    rp = os.path.realpath(path)
    for r in roots:
        rr = os.path.realpath(os.path.expanduser(r))
        if rp == rr or rp.startswith(rr.rstrip("/") + "/"):
            return True
    return False


def _lexically_under(path, roots):
    """the path as written (.. folded, symlinks NOT followed) is inside a root, as written or resolved."""
    ap = os.path.abspath(path)
    for r in roots:
        for rr in {os.path.abspath(os.path.expanduser(r)), os.path.realpath(os.path.expanduser(r))}:
            if ap.startswith(rr.rstrip("/") + "/"):
                return True
    return False


def is_gguf_file(path):
    """a regular, readable file that starts with the GGUF magic (symlinks followed)."""
    try:
        import stat as _stat
        if not _stat.S_ISREG(os.stat(path).st_mode) or not os.access(path, os.R_OK):
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def model_path_ok(path, roots):
    """A model may be served when it resolves inside a model folder, or when it is a link that sits
    inside a model folder and points at a regular, readable GGUF somewhere else (models offloaded to
    another disk and linked back). Anything else outside the folders is refused."""
    if path_under(path, roots):
        return True
    return path.endswith(".gguf") and _lexically_under(path, roots) and is_gguf_file(path)


def annotate_plan_text(text, pairs):
    """The launcher explains its own flags; PXA Control appends more after it has planned (thinking
    flags, the operator's extra args). Make the explanation match: rewrite the row of a flag that is
    now passed, and add rows for flags the launcher does not list. pairs: [(flag, value, source)],
    in command order (the last one of a flag wins, as on the engine's command line)."""
    if not text or not pairs:
        return text
    final = {}
    for flag, value, src in pairs:
        final[flag] = (value, src)
    lines = text.split("\n")
    rowre = re.compile(r"^(\s*)(-[-\w]+)(\s+)(.*)$")
    done, anchor, indent = set(), None, "  "
    for i, ln in enumerate(lines):
        m = rowre.match(ln)
        if not m or len(m.group(1)) > 8:
            continue
        flag = m.group(2)
        if flag in ("--reasoning-format", "--reasoning-budget"):
            anchor, indent = i, m.group(1)
        if flag in final and flag not in done and len(m.group(2) + m.group(3)) >= 20:
            value, src = final[flag]
            v = value if len(str(value)) <= 44 else str(value)[:41] + "..."
            lines[i] = f"{m.group(1)}{flag:<20} {v:<44} {src}"
            done.add(flag)
            if flag in ("--reasoning-format", "--reasoning-budget"):
                anchor = i
    extra = []
    for flag, (value, src) in final.items():
        if flag in done:
            continue
        v = value if len(str(value)) <= 44 else str(value)[:41] + "..."
        extra.append(f"{indent}{flag:<20} {v:<44} {src}")
    if extra:
        if anchor is not None:
            lines[anchor + 1:anchor + 1] = extra
        else:
            lines += ["", "  added by PXA Control after the plan:"] + extra
    return "\n".join(lines)


def _flag_pairs(args, src):
    """['--a', 'x', '--b', '--c', '-1'] -> [(--a, x), (--b, ''), (--c, -1)] (a value may start with '-' if numeric)."""
    out, i = [], 0
    while i < len(args):
        a = str(args[i])
        if a.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", a):
            v = ""
            if i + 1 < len(args):
                n = str(args[i + 1])
                if not n.startswith("-") or re.fullmatch(r"-\d+(\.\d+)?", n):
                    v, i = n, i + 1
            out.append((a, v if v != "" else "on", src))
        i += 1
    return out


def check_engine_dir(d):
    """'' (= the launcher's own search) or a build dir holding an executable bin/llama-server; the
    bin dir or the binary itself are accepted and normalized to the build dir."""
    if d in (None, ""):
        return ""
    if not isinstance(d, str) or len(d) > 4096 or "\x00" in d:
        raise Invalid("engine build: not a path")
    p = os.path.abspath(os.path.expanduser(d.strip()))
    if os.path.isfile(p) and os.path.basename(p) == "llama-server":
        p = os.path.dirname(p)
    if os.path.isfile(os.path.join(p, "llama-server")) and not os.path.isfile(os.path.join(p, "bin", "llama-server")):
        p = os.path.dirname(p)
    exe = os.path.join(p, "bin", "llama-server")
    if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        raise Invalid(f"engine build: no executable bin/llama-server under {p}")
    return p


def validate_launch(body, gpu_indexes, catalog_names, model_roots, gui_port=None, check_model=True,
                    resolve_port=True, avoid_ports=()):
    """A launch/plan request -> a normalized dict. Raises Invalid with a user-facing reason.
    resolve_port=False keeps port 0 as 'auto' (a saved profile must not freeze today's free port);
    avoid_ports are ports other servers of this GUI hold or are about to bind."""
    if not isinstance(body, dict):
        raise Invalid("request body must be a JSON object")
    g = body.get("gpus")
    if isinstance(g, str):
        g = [x for x in re.split(r"[,\s]+", g) if x]
    if not isinstance(g, list) or not g or len(g) > 16:
        raise Invalid("choose at least one card")
    cards = []
    for x in g:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            raise Invalid(f"card {x!r} is not an index")
        if xi not in gpu_indexes:
            raise Invalid(f"card {xi} is not on this machine")
        if xi not in cards:
            cards.append(xi)
    model = body.get("model")
    if not isinstance(model, str) or not model or len(model) > 4096 or "\x00" in model:
        raise Invalid("choose a model")
    if not model.endswith(".gguf"):
        raise Invalid("the model must be a .gguf file")
    if check_model:
        if not os.path.isfile(model):
            raise Invalid(f"model file not found: {model}")
        if not model_path_ok(model, model_roots):
            raise Invalid("the model must be inside one of your model folders (Models tab)")
    kv = body.get("kv", "auto") or "auto"
    if kv not in KV_TYPES:
        raise Invalid(f"kv must be one of {', '.join(KV_TYPES)}")
    sm = body.get("sm", "auto") or "auto"
    if sm not in SPLIT_MODES:
        raise Invalid(f"split mode must be one of {', '.join(SPLIT_MODES)}")
    fa = body.get("fa", "auto") or "auto"
    if fa not in FA_MODES:
        raise Invalid("flash attention must be auto, on or off")
    port = _int(body, "port", 0, 65535, 0)
    if port == 0 and resolve_port:  # auto: the first free port from 8080 up (8080 is often taken already)
        port = free_port(DEFAULT_SERVER_PORT, "0.0.0.0" if body.get("expose") else "127.0.0.1",
                         avoid={gui_port} | set(avoid_ports or ()))
    elif 0 < port < 1024:
        raise Invalid("port must be between 1024 and 65535 (or 0 for auto)")
    if gui_port and port == gui_port:
        raise Invalid(f"port {port} is PXA Control's own port")
    levers, errs = validate_levers(body.get("levers") or {}, catalog_names)
    if errs:
        raise Invalid("; ".join(errs))
    out = {
        "gpus": cards, "model": model,
        "ctx": _int(body, "ctx", 0, 1 << 21, 0),
        "np": _int(body, "np", 0, 64, 0),
        "kv": kv, "sm": sm, "fa": fa, "port": port,
        "mtp": bool(body.get("mtp", False)),
        "expose": bool(body.get("expose", False)),
        "accept_unmeasured": bool(body.get("accept_unmeasured", False)),
        "allow_busy": bool(body.get("allow_busy", False)),
        "levers": levers,
        "extra_args": validate_extra_args(body.get("extra_args")),
        "engine_dir": check_engine_dir(body.get("engine_dir")),
        "thinking": clean_thinking(body.get("thinking")),
        "draft_model": _draft_model(body, model, model_roots, check_model),
        "hot_models": _hot_models(body, model, model_roots, check_model),
    }
    return out


def _draft_model(body, model, model_roots, check_model):
    draft = body.get("draft_model") or ""
    if draft in ("", None):
        return ""
    if not isinstance(draft, str) or not draft.endswith(".gguf") or len(draft) > 4096 or "\x00" in draft:
        raise Invalid("the drafter must be a .gguf file")
    if check_model:
        if not os.path.isfile(draft):
            raise Invalid("drafter file not found: %s" % draft)
        if not model_path_ok(draft, model_roots):
            raise Invalid("the drafter must be inside one of your model folders (Models tab)")
    return draft


def _hot_models(body, model, model_roots, check_model):
    """Extra models registered with --hot-model. Paths only; the launcher refuses Pascal and no-VMM cards."""
    hots = body.get("hot_models") or []
    if hots in ("", None):
        return []
    if isinstance(hots, str):
        hots = [hots]
    if not isinstance(hots, list) or len(hots) > 8:
        raise Invalid("at most 8 extra models on one server")
    out, seen = [], set()
    main = os.path.abspath(model) if isinstance(model, str) else ""
    for raw in hots:
        if not isinstance(raw, str) or not raw.endswith(".gguf") or len(raw) > 4096 or "\x00" in raw:
            raise Invalid("an extra model must be a .gguf path")
        if any(c in raw for c in " \t"):
            raise Invalid("an extra model path cannot contain spaces (the engine's --hot-model parser splits on them)")
        if check_model:
            if not os.path.isfile(raw):
                raise Invalid("extra model file not found: %s" % raw)
            if not model_path_ok(raw, model_roots):
                raise Invalid("an extra model must be inside one of your model folders (Models tab)")
        ap = os.path.abspath(raw)
        if ap == main:
            raise Invalid("an extra model is the one already chosen to start on the cards")
        if ap in seen:
            continue
        seen.add(ap)
        out.append(raw)
    return out


def hot_model_argv(paths):
    """NAME=PATH tokens for --hot-model. The name is the file stem; a stem with '=' gets m2, m3, ..."""
    used, out = set(), []
    for i, path in enumerate(paths, start=2):
        stem = os.path.splitext(os.path.basename(path))[0]
        name = stem if stem and "=" not in stem and " " not in stem else ("m%d" % i)
        if name in used:
            name = "%s-%d" % (name, i)
        used.add(name)
        out.append("%s=%s" % (name, path))
    return out


def clean_thinking(x):
    """{mode: auto|on|off, budget: None|-1..N, level: None|str} (tools/pxa_thinking.py) or Invalid."""
    try:
        return TH.clean_settings(x)
    except TH.Invalid as e:
        raise Invalid(str(e))


def thinking_is_default(t):
    return not t or (t.get("mode", "auto") == "auto" and t.get("budget") is None and not t.get("level")
                     and not t.get("effort"))


def launcher_argv(req):
    """The command line a user would have typed for this request (no shell; argv only)."""
    argv = ["--gpus", ",".join(str(x) for x in req["gpus"]), "--model", req["model"],
            "--port", str(req["port"]), "--host", "0.0.0.0" if req["expose"] else "127.0.0.1",
            "--sm", req["sm"]]
    if req["kv"] != "auto":          # auto: no -ctk/-ctv, the engine registry picks the KV type
        argv += ["--ctk", req["kv"], "--ctv", req["kv"]]
    argv += ["--yes", "--no-interactive", "--no-tui"]
    if req["ctx"]:
        argv += ["--ctx", str(req["ctx"])]
    if req["np"]:
        argv += ["--np", str(req["np"])]
    wl = FA_MODES[req["fa"]]
    if wl == "chat" and req["np"] > 1:
        wl = "serve"
    if wl:
        argv += ["--workload", wl]
    if req["mtp"]:
        argv += ["--spec", "mtp"]
    if req["accept_unmeasured"]:
        argv += ["--accept-unmeasured"]
    if req["allow_busy"]:
        argv += ["--allow-busy"]
    if req.get("draft_model"):
        argv += ["--draft-model", req["draft_model"]]
    for spec in hot_model_argv(req.get("hot_models") or []):
        argv += ["--hot-model", spec]
    return argv


def shell_line(env, cmd):
    """One line a POSIX shell runs as-is: the exact environment the GUI adds, then the exact argv."""
    return " ".join([f"{k}={shlex.quote(str(v))}" for k, v in sorted((env or {}).items())] +
                    [shlex.quote(str(a)) for a in cmd])


def cli_line(req):
    env = " ".join(f"{k}={v}" for k, v in sorted(req["levers"].items()))
    return ((env + " ") if env else "") + "pxa-launch " + " ".join(
        a if re.match(r"^[A-Za-z0-9_./,:=+-]+$", a) else "'" + a.replace("'", "'\\''") + "'"
        for a in launcher_argv(req))


# ---------------------------------------------------------------------------------------------
# the one seat this GUI runs
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# "Report a problem": a redacted bundle, shown to the user in full, sent only on a Send click
# ---------------------------------------------------------------------------------------------
def version_token(text):
    """First token of a VERSION file. A line that starts with 'tag:' yields the token after it.
    The same rule as tools/pxa-update.c version_token."""
    if not text:
        return ""
    line = str(text).splitlines()[0].strip()
    if line.lower().startswith("tag:"):
        line = line[4:].strip()
    parts = line.split()
    return parts[0] if parts else ""


def version_from_dir(d):
    """Package version in d: current/VERSION, then VERSION, then the current symlink's
    basename with a leading 'pxa-' removed. Empty when none of those exist."""
    if not d:
        return ""
    for rel in (os.path.join("current", "VERSION"), "VERSION"):
        try:
            with open(os.path.join(d, rel), encoding="utf-8") as f:
                tok = version_token(f.read(4096))
        except OSError:
            continue
        if tok:
            return tok
    cur = os.path.join(d, "current")
    if not (os.path.islink(cur) or os.path.isdir(cur)):
        return ""
    base = os.path.basename(os.path.realpath(cur))
    if base.startswith("pxa-"):
        base = base[4:]
    if base and base != os.path.basename(os.path.abspath(d)):
        return base
    return ""


def package_version():
    """Version of the package this Control is serving. The tree beside tools/ wins (a source
    checkout and an unpacked tarball both keep VERSION there), then PXA_INSTALL_DIR, then the
    per-user install. Empty when no package version can be read."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.path.dirname(here)]
    inst = os.environ.get("PXA_INSTALL_DIR")
    if inst:
        roots.append(inst)
    roots.append(os.path.join(os.path.expanduser("~"), ".local", "share", "pxa"))
    seen = set()
    for d in roots:
        d = os.path.abspath(d)
        if d in seen:
            continue
        seen.add(d)
        tok = version_from_dir(d)
        if tok:
            return tok
    return ""


CONTROL_VERSION = package_version()  # footer and reports: the token in the package VERSION file
BUG_URL_DEFAULT = "https://bugs.pxanetwork.com/v1/report"
REPORT_MAX = 256 * 1024                      # the intake refuses more than this
BANNER_RE = re.compile(r"PXA_(REGISTRY|TSPLIT|AUTO)")
BUILD_RE = re.compile(r"^(build|version|system_info|PXA[ _-]?(build|version))\b", re.I)


def bug_url():
    return os.environ.get("PXA_BUG_URL") or BUG_URL_DEFAULT


_SECRET_PATTERNS = [
    (re.compile(r"\bpxk1\.[A-Za-z0-9\-]{3,24}\.[A-Za-z0-9_\-]{8,}"), "<redacted-key>"),          # a PXA Quantizer licence key
    (re.compile(r"(?i)\b(PXQE_KEY|PXA_LICENCE_KEY|licen[cs]e[_-]?key)(\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|\S+)"), r"\1\2<redacted>"),
    (re.compile(r"(--api-key)(?:=|\s+)\S+", re.I), r"\1 <redacted>"),
    (re.compile(r"(?i)\b(authorization|proxy-authorization)\s*[:=]\s*(?:bearer|basic|bot)?\s*\S+"), r"\1: <redacted>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b([A-Za-z0-9_\-]*(?:api[_-]?key|token(?!s)|secret|passw(?:or)?d|credential)[A-Za-z0-9_\-]*)"
                r"(\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|\S{6,})"), r"\1\2<redacted>"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]*"), "<redacted-jwt>"),
    (re.compile(r"\b(?:hf|ghp|gho|ghs|ghu|github_pat|glpat|xox[abprs]|sk|pk|rk|AKIA|AIza)[_\-]?[A-Za-z0-9_\-]{16,}"), "<redacted-key>"),
    (re.compile(r"\b[A-Za-z0-9_\-]{23,28}\.[A-Za-z0-9_\-]{6,7}\.[A-Za-z0-9_\-]{27,}\b"), "<redacted-token>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S), "<redacted-private-key>"),
]
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b")
_HOME_RES = [re.compile(r"/home/[^/\s\"']+"), re.compile(r"/Users/[^/\s\"']+"), re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s\"']+"),
             re.compile(r"/root(?=/|\b)")]
_KEEP_IPS = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}
_HIGH_ENTROPY_RE = re.compile(r"\b(?=[A-Za-z0-9+/_\-]*[A-Z])(?=[A-Za-z0-9+/_\-]*[a-z])(?=[A-Za-z0-9+/_\-]*\d)[A-Za-z0-9+/_\-]{32,}={0,2}")


def _identity_words():
    """hostnames and user names that must not leave the machine."""
    hosts, users = set(), set()
    for h in (getattr(os, "uname", lambda: None)() and os.uname().nodename, socket.gethostname(),
              os.environ.get("HOSTNAME"), os.environ.get("COMPUTERNAME")):
        if h and len(h) >= 2:
            hosts.add(h)
            hosts.add(h.split(".")[0])
    for u in (os.environ.get("USER"), os.environ.get("LOGNAME"), os.environ.get("USERNAME")):
        if u and len(u) >= 3 and u.lower() not in ("root", "user", "admin", "pxa"):
            users.add(u)
    try:
        import getpass
        u = getpass.getuser()
        if u and len(u) >= 3 and u.lower() not in ("root", "user", "admin", "pxa"):
            users.add(u)
    except Exception:
        pass
    return {h for h in hosts if len(h) >= 2}, users


def _ip6_or_keep(t):
    import ipaddress
    try:
        ipaddress.IPv6Address(t)
    except ValueError:
        return t                     # a clock time, a "a:b:c" label, ...
    return t if t in ("::1", "::") else "<ip6>"


def redact_text(s, hosts=None, users=None):
    """One string in, the same string with home paths, hostnames, IPs, user names, e-mail, MACs and anything
    that looks like a token or key replaced by a placeholder. Idempotent."""
    if not isinstance(s, str) or not s:
        return s
    if hosts is None or users is None:
        h, u = _identity_words()
        hosts = h if hosts is None else hosts
        users = u if users is None else users
    for rx, rep in _SECRET_PATTERNS:
        s = rx.sub(rep, s)
    home = os.path.expanduser("~")
    if home and home not in ("/", "~") and len(home) > 1:
        s = s.replace(home, "~")
    for rx in _HOME_RES:
        s = rx.sub("~", s)
    s = _EMAIL_RE.sub("<email>", s)
    s = _MAC_RE.sub("<mac>", s)
    s = _IPV4_RE.sub(lambda m: m.group(0) if m.group(0) in _KEEP_IPS else "<ip>", s)
    s = _IPV6_RE.sub(lambda m: _ip6_or_keep(m.group(0)), s)
    for w in sorted(hosts, key=len, reverse=True):
        s = re.sub(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", "<host>", s, flags=re.I)
    for w in sorted(users, key=len, reverse=True):
        s = re.sub(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", "<user>", s)
    s = _HIGH_ENTROPY_RE.sub("<redacted-key>", s)
    return s


def redact_obj(o, hosts=None, users=None, depth=0):
    if hosts is None or users is None:
        h, u = _identity_words()
        hosts = h if hosts is None else hosts
        users = u if users is None else users
    if depth > 8:
        return None
    if isinstance(o, str):
        return redact_text(o, hosts, users)
    if isinstance(o, (list, tuple)):
        return [redact_obj(x, hosts, users, depth + 1) for x in o]
    if isinstance(o, dict):
        return {redact_text(str(k), hosts, users): redact_obj(v, hosts, users, depth + 1) for k, v in o.items()}
    return o


SCORE_NAME_BAD = ("fuck", "shit", "cunt", "bitch", "nigg", "fagg", "whore", "slut", "rape", "nazi", "hitler", "dick", "cock", "pussy", "asshole", "retard", "porn")
SCORE_URL_RE = re.compile(r"(https?:|www\.|discord\.gg|discord\.com|://|\.(com|net|org|io|gg|xyz|ru|cn|tk|ly|me)\b)", re.I)
QUANT_RE = re.compile(r"(PXQN\d+\w*|PXQ\d+\w*|PXQU\w*|IQ\d_\w+|Q\d_K(?:_[SML])?|Q\d_\d|MXFP4|F16|BF16)", re.I)


def clean_board_name(name):
    """the same rules the board applies: max 32 chars, no links/mentions/profanity; '' = anonymous."""
    n = re.sub(r"[\x00-\x1f\x7f]", "", str(name or "")).strip()
    flat = re.sub(r"[^a-z]", "", n.lower().translate(str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})))
    if not n or SCORE_URL_RE.search(n) or re.search(r"[@<`*]", n) or any(b in flat for b in SCORE_NAME_BAD):
        return ""
    return re.sub(r"[^\w .'\-]", "", n)[:32].strip()


def score_base_url():
    u = urllib.parse.urlparse(bug_url())
    return f"{u.scheme}://{u.netloc}"


class Seat(object):
    def __init__(self, L, sid="main"):
        self.L = L
        self.sid = sid
        self.last_line_ts = None
        self.env_shown = {}
        self.lock = threading.Lock()
        self.proc = None
        self.cmd = None
        self.req = None
        self.log = deque(maxlen=6000)
        self.build_lines = []          # the engine's first build/version lines, kept after the log ring drops them
        self.seq = 0
        self.phase = "stopped"
        self.started = None
        self.first_token = None
        self.stopping = False
        self.hidden = 0
        self.cond = threading.Condition()

    def _append(self, line):
        with self.cond:
            self.seq += 1
            self.log.append((self.seq, line))
            if len(self.build_lines) < 6 and BUILD_RE.match(line.strip()):
                self.build_lines.append(line)
            self.last_line_ts = time.time()
            self.cond.notify_all()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def port(self):
        return self.req["port"] if self.req else None

    def start(self, built, req, lever_env):
        plan, cmd, env, cv, prof, ctx = built
        with self.lock:
            if self.running():
                raise Invalid("a server is already running; stop it first")
            e = dict(os.environ)
            e.update(env)
            e.update(self.L.device_env(cv)[0])
            e.update(lever_env)
            self.log.clear()
            self.build_lines = []
            self.stopping, self.hidden = False, 0
            self._append("$ " + " ".join(f"{k}={v}" for k, v in sorted(lever_env.items()))
                         + (" " if lever_env else "") + " ".join(self.L.redact_cmd(cmd)))
            self.proc = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, text=True, bufsize=1,
                                         errors="replace")
            self.cmd, self.req = cmd, req
            self.env_shown = dict(env, **self.L.device_env(cv)[0], **lever_env)
            self.phase, self.started, self.first_token = "starting", time.time(), None
            proc = self.proc
            threading.Thread(target=self._pump, args=(proc,), daemon=True).start()

    def _pump(self, proc):
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if HEALTH_NOISE_RE.search(line):
                    self.hidden += 1        # PXA Control's own 3 s health polls: not the user's traffic
                    continue
                self._append(line)
                low = line.lower()
                if "loading model" in low or "llama_model_loader" in low:
                    self.phase = "loading model"
                elif "offloaded" in low and "layers to gpu" in low:
                    self.phase = "weights on the GPUs"
                elif "server is listening" in low or "http server listening" in low:
                    self.phase = "listening"
        except Exception:
            pass
        rc = proc.wait()
        try:
            proc.stdout.close()
        except Exception:
            pass
        self._append(f"[pxa-control] server process exited with code {rc}")
        if proc is self.proc and not self.stopping:
            self.phase = f"exited ({rc})"

    def stop(self):
        with self.lock:
            proc, cmd = self.proc, self.cmd
            if proc is None or proc.poll() is not None:
                return False
            self.phase, self.stopping = "stopping", True
            self.L.LaunchTUI._stop(proc, cmd)       # SIGTERM by PID, then kill; docker by name
            try:
                proc.wait(timeout=15)
            except Exception:
                pass
            if proc.poll() is None:
                # SIGKILL did not take: the process is stuck inside the driver (D state). Say so
                # instead of claiming 'stopped' and then refusing the next Start as 'running'.
                self.phase = f"stuck: pid {proc.pid} ignores SIGKILL (GPU driver busy); its cards stay held"
                self._append(f"[pxa-control] {self.phase}")
                return False
            self.phase = "stopped"
            return True

    def search(self, q, limit=2000, regex=False):
        """[(seq, line)] of the whole kept log matching q (case-insensitive; regex on request)."""
        with self.cond:
            items = list(self.log)
        if not q:
            return items[-limit:]
        if regex:
            try:
                rx = re.compile(q, re.I)
            except re.error as e:
                raise Invalid(f"bad search pattern: {e}")
            hit = [x for x in items if rx.search(x[1])]
        else:
            ql = q.lower()
            hit = [x for x in items if ql in x[1].lower()]
        return hit[-limit:]

    def lines_since(self, since, limit=2000):
        with self.cond:
            items = [x for x in self.log if x[0] > since]
        return items[-limit:]

    def status(self):
        r = self.running()
        d = {"running": r, "phase": self.phase if (r or self.proc) else "stopped",
             "pid": self.proc.pid if self.proc else None,
             "exit_code": (self.proc.returncode if self.proc and not r else None),
             "port": self.port(), "started": self.started, "log_seq": self.seq,
             "model": self.req["model"] if self.req else None,
             "gpus": self.req["gpus"] if self.req else None,
             "command": " ".join(self.L.redact_cmd(self.cmd)) if self.cmd else None,
             "request": self.req if r else None, "sid": self.sid,
             "last_line_age": (time.time() - self.last_line_ts) if (r and self.last_line_ts) else None,
             "health": None, "hidden_health_lines": self.hidden}
        return d


# ---------------------------------------------------------------------------------------------
# the fleet: every llama-server on this machine, started here or not
# ---------------------------------------------------------------------------------------------
SID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-()#:+]{1,48}$")
DOCKER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def _argv_value(argv, *flags, default=None):
    for i, a in enumerate(argv):
        for f in flags:
            if a == f and i + 1 < len(argv):
                return argv[i + 1]
            if f.startswith("--") and a.startswith(f + "="):
                return a.split("=", 1)[1]
    return default


def _is_llama_server(argv):
    return any(os.path.basename(a) == "llama-server" for a in (argv or [])[:3])


def proc_table():
    """{pid: (ppid, argv)} from /proc (Linux); {} elsewhere."""
    out = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for n in names:
        if not n.isdigit():
            continue
        try:
            with open(f"/proc/{n}/stat", "rb") as f:
                st = f.read().decode("utf-8", "replace")
            ppid = int(st[st.rindex(")") + 2:].split()[1])
            with open(f"/proc/{n}/cmdline", "rb") as f:
                argv = [x.decode("utf-8", "replace") for x in f.read().split(b"\0") if x]
        except (OSError, ValueError, IndexError):
            continue
        out[int(n)] = (ppid, argv)
    return out


def descendants(root, table):
    kids = {}
    for pid, (pp, _a) in table.items():
        kids.setdefault(pp, []).append(pid)
    seen, todo = {root}, [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in seen:
                seen.add(c)
                todo.append(c)
    return seen


def _in_container(pid):
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            cg = f.read()
    except OSError:
        return False
    return any(k in cg for k in ("docker", "containerd", "kubepods", "libpod", "lxc.payload"))


def _proc_env(pid, keys):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read().split(b"\0")
    except OSError:
        return {}
    out = {}
    for kv in raw:
        k, _, v = kv.decode("utf-8", "replace").partition("=")
        if k in keys:
            out[k] = v
    return out


def parse_visible(spec, gpu_rows):
    """'0,1' / 'GPU-uuid,...' / 'all' / '' -> [nvidia-smi index] (None = unknown)."""
    if spec is None:
        return None
    spec = str(spec).strip()
    if spec in ("", "void", "none", "NoDevFiles"):
        return []
    idx = [g[0] for g in gpu_rows or []]
    if spec == "all":
        return idx
    by_uuid = {g[5]: g[0] for g in gpu_rows or []}
    out = []
    for p in spec.split(","):
        p = p.strip()
        if p.isdigit():
            out.append(int(p))
        elif p in by_uuid:
            out.append(by_uuid[p])
        else:
            for u, i in by_uuid.items():
                if p and u.startswith(p):
                    out.append(i)
    return out


def _http_json(port, path, timeout=1.5):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace") or "null")


def probe_registered_models(port, timeout=1.5):
    """Models /v1/models lists when this server was started with --hot-model, else None.

    The engine already reports which one is on the cards (pxa_hot_swap.active). A server with a
    single model and no hot-swap object is left as None so the Servers tab does not grow a line.
    """
    try:
        data = _http_json(port, "/v1/models", timeout)
    except Exception:       # noqa: BLE001 - the fleet card still renders without this line
        return None
    rows = []
    for m in (data or {}).get("data") or []:
        if not isinstance(m, dict):
            continue
        hs = m.get("pxa_hot_swap")
        if not isinstance(hs, dict) or "active" not in hs:
            return None
        rows.append({"id": m.get("id"), "active": bool(hs.get("active"))})
    return rows or None


def probe_health(port, timeout=1.5):
    """-> (state, detail): 'ok' | 'loading' | 'down' | 'http N'."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            try:
                j = json.loads(r.read().decode("utf-8", "replace") or "{}")
            except ValueError:
                j = {}
            return ("ok" if r.status == 200 else f"http {r.status}"), j
    except urllib.error.HTTPError as e:
        return ("loading" if e.code == 503 else f"http {e.code}"), {}
    except Exception:       # noqa: BLE001
        return "down", {}


# ---- the context the ENGINE resolved (BigBair 2026-10-08: "auto" must show the number it became) ----------------------
CTX_LOG_RE = re.compile(r"\bn_ctx\s*[=:]\s*(\d+)")
CTX_SLOT_LOG_RE = re.compile(r"\bn_ctx_(?:per_seq|seq|slot)\s*[=:]\s*(\d+)")


def probe_ctx(port, timeout=1.5):
    """-> {"n_ctx", "per_slot", "slots"} from the engine's own /props, or None. Never guessed: a server whose /props has no
    n_ctx gives None. PXA engines put n_ctx at the top level; stock llama.cpp only has the slot's
    default_generation_settings.n_ctx (the per-slot context)."""
    try:
        props = _http_json(port, "/props", timeout=timeout)
    except Exception:       # noqa: BLE001
        return None
    if not isinstance(props, dict):
        return None
    top = _num(props.get("n_ctx"))
    dgs = props.get("default_generation_settings") if isinstance(props.get("default_generation_settings"), dict) else {}
    per = _num(dgs.get("n_ctx"))
    slots = _num(props.get("total_slots"))
    if not top and not per:
        return None
    return {"n_ctx": int(top) if top else None, "per_slot": int(per) if per else None, "slots": int(slots) if slots else None}


def ctx_from_log(lines):
    """-> (n_ctx, per_slot) from the engine's own load log (llama_context: n_ctx = 65536 / n_ctx_per_seq = 16384)."""
    n = per = None
    for ln in lines:
        m = CTX_SLOT_LOG_RE.search(ln)
        if m:
            per = int(m.group(1))
            continue
        m = CTX_LOG_RE.search(ln)
        if m:
            n = int(m.group(1))
    return n, per


def context_info(requested, engine_arg, probed=None, log=None, cards=None):
    """the Context line a server card shows: {mode, requested, launcher_arg, resolved, per_slot, slots, source, how}.
    requested: what the user typed (0 / None = auto); engine_arg: the -c the engine was started with (the launcher's
    pick when auto); probed: probe_ctx(); log: ctx_from_log()."""
    try:
        req = int(requested or 0)
    except (TypeError, ValueError):
        req = 0
    try:
        arg = int(engine_arg) if engine_arg not in (None, "") else None
    except (TypeError, ValueError):
        arg = None
    res = per = slots = None
    src = None
    if probed:
        res, per, slots = probed.get("n_ctx"), probed.get("per_slot"), probed.get("slots")
        src = "engine /props"
    if res is None and log and log[0]:
        res, per = log[0], (log[1] or per)
        src = "engine log"
    if res is None and per is not None and slots:
        src = src or "engine /props"
    mode = "set" if req > 0 else "auto"
    where = f" on card(s) {','.join(str(c) for c in cards)}" if cards else ""
    if res is None and per is None:
        how = "the engine has not reported its context yet (still loading, or its /props has no n_ctx)"
    elif mode == "auto":
        how = (f"Context was left on auto. The launcher sized it from the model's trained context and the VRAM free{where} "
               f"(its fit check: weights + KV cache must fit), "
               + (f"started the engine with -c {arg:,}, " if arg else "let the engine pick its default, ")
               + f"and the engine reports {(res or per):,} ({src}).")
    else:
        how = f"Context set by hand to {req:,}; the engine reports {(res or per):,} ({src})."
    if per and res and slots and slots > 1:
        how += f" {slots} parallel slots share it: {per:,} each."
    return {"mode": mode, "requested": req, "launcher_arg": arg, "resolved": res, "per_slot": per, "slots": slots,
            "source": src, "how": how}


PROM_RE = re.compile(r"^llamacpp:([a-z_]+)\s+([0-9.eE+\-]+)\s*$", re.M)


def probe_metrics(port, window=3600, timeout=2.0):
    """t/s for one server: its own /pxa/stats (median and last over the window), else the
    llama.cpp /metrics counters (--metrics), else {}."""
    now = time.time()
    try:
        d = _http_json(port, f"/pxa/stats?since={int(now - window)}", timeout)
        # a request that generated one token in ~0 s reports 1e6 t/s: not a speed, dropped
        recs = [r for r in (d or {}).get("records") or [] if isinstance(r, dict)
                and 0 < float(r.get("decode_tps") or 0) < 5000]
        dec = [r.get("decode_tps") for r in recs if r.get("decode_tps")]
        pre = [r.get("prefill_tps") for r in recs if r.get("prefill_tps")]
        last = recs[-1] if recs else {}
        return {"src": "/pxa/stats", "requests": len(recs), "decode_median": median(dec) or None,
                "prefill_median": median(pre) or None, "decode_last": last.get("decode_tps"),
                "prefill_last": last.get("prefill_tps"), "last_ts": last.get("ts"), "window_s": window}
    except Exception:       # noqa: BLE001
        pass
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as r:
            txt = r.read().decode("utf-8", "replace")
        m = {k: float(v) for k, v in PROM_RE.findall(txt)}
        return {"src": "/metrics", "decode_median": m.get("predicted_tokens_seconds"),
                "prefill_median": m.get("prompt_tokens_seconds"),
                "requests": int(m.get("n_decode_total", 0)) or None,
                "processing": m.get("requests_processing")}
    except Exception:       # noqa: BLE001
        return {}


def compute_apps(gpu_rows):
    """[(pid, gpu_index, MiB)] of every CUDA process (nvidia-smi), [] when unavailable."""
    if os.environ.get("PXA_LAUNCH_FAKE_GPUS") or not shutil_which("nvidia-smi"):
        return []
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except Exception:       # noqa: BLE001
        return []
    by_uuid = {g[5]: g[0] for g in gpu_rows or []}
    out = []
    for line in (r.stdout or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3 or p[2] not in by_uuid:
            continue
        try:
            out.append((int(p[0]), by_uuid[p[2]], int(p[1])))
        except ValueError:
            continue
    return out


def docker_bin():
    return shutil_which("docker")


def scan_docker(gpu_rows, extra_names=(), timeout=10):
    """llama-server containers: running ones, plus any named in extra_names (adopted, maybe stopped)."""
    d = docker_bin()
    if not d:
        return [], "docker CLI not found"
    try:
        r = subprocess.run([d, "ps", "-q", "--no-trunc"], capture_output=True, text=True, timeout=timeout)
        ids = r.stdout.split() if r.returncode == 0 else []
        if r.returncode != 0:
            return [], (r.stderr or "docker ps failed").strip()[:200]
    except Exception as e:      # noqa: BLE001
        return [], f"docker ps: {e.__class__.__name__}"
    ids += [n for n in extra_names if DOCKER_NAME_RE.match(n or "")]
    if not ids:
        return [], None
    try:
        r = subprocess.run([d, "inspect"] + ids, capture_output=True, text=True, timeout=timeout)
        info = json.loads(r.stdout or "[]")
    except Exception as e:      # noqa: BLE001
        return [], f"docker inspect: {e.__class__.__name__}"
    out, seen = [], set()
    for c in info:
        cfg, st, hc = c.get("Config") or {}, c.get("State") or {}, c.get("HostConfig") or {}
        name = (c.get("Name") or "").lstrip("/")
        if not name or name in seen:
            continue
        argv = list(cfg.get("Entrypoint") or []) + list(cfg.get("Cmd") or [])
        if not any("llama-server" in a for a in argv):
            continue
        seen.add(name)
        env = dict(x.split("=", 1) for x in (cfg.get("Env") or []) if "=" in x)
        port = int(_argv_value(argv, "--port", default="8080") or 8080)
        host_port, host_ip = None, None
        if hc.get("NetworkMode") == "host":
            host_port, host_ip = port, _argv_value(argv, "--host", default="127.0.0.1")
        else:
            m = ((c.get("NetworkSettings") or {}).get("Ports") or {}).get(f"{port}/tcp") or []
            if m:
                try:
                    host_port, host_ip = int(m[0].get("HostPort")), m[0].get("HostIp") or "0.0.0.0"
                except (TypeError, ValueError):
                    pass
        vis = env.get("NVIDIA_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES")
        for dr in hc.get("DeviceRequests") or []:
            if dr.get("DeviceIDs"):
                vis = ",".join(dr["DeviceIDs"])
            elif dr.get("Count") == -1 and vis is None:
                vis = "all"
        out.append({"key": "d:" + name, "kind": "docker", "name": name, "container": name,
                    "image": cfg.get("Image"), "running": bool(st.get("Running")),
                    "status": st.get("Status"), "pid": st.get("Pid") or None, "started": st.get("StartedAt"),
                    "port": host_port, "host": host_ip, "gpus": parse_visible(vis, gpu_rows),
                    "model": _argv_value(argv, "-m", "--model"), "alias": _argv_value(argv, "-a", "--alias"),
                    "ctx": _argv_value(argv, "-c", "--ctx-size"), "argv": argv})
    return out, None


def process_entry(pid, argv, gpu_rows):
    """one bare server process as a fleet instance (port: --port, else LLAMA_ARG_PORT, else 8080)."""
    argv = list(argv or [])
    env = _proc_env(pid, ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LLAMA_ARG_PORT"))
    try:
        port = int(_argv_value(argv, "--port", default=env.get("LLAMA_ARG_PORT") or "8080") or 8080)
    except ValueError:
        port = 8080
    try:
        started = os.stat(f"/proc/{pid}").st_mtime
    except OSError:
        started = None
    return {"key": f"p:{pid}", "kind": "process", "name": f"llama-server pid {pid}", "pid": pid,
            "running": True, "port": port, "host": _argv_value(argv, "--host", default="127.0.0.1"),
            "gpus": parse_visible(env.get("CUDA_VISIBLE_DEVICES", env.get("NVIDIA_VISIBLE_DEVICES")), gpu_rows),
            "model": _argv_value(argv, "-m", "--model"), "alias": _argv_value(argv, "-a", "--alias"),
            "ctx": _argv_value(argv, "-c", "--ctx-size"), "started_epoch": started, "argv": argv}


def scan_processes(gpu_rows, table, skip_pids=()):
    """bare llama-server processes on this host (not in a container, not started by this GUI)."""
    out = []
    skip = set(skip_pids)
    for pid, (_pp, argv) in table.items():
        if pid in skip or not argv or os.path.basename(argv[0]) != "llama-server" or _in_container(pid):
            continue
        out.append(process_entry(pid, argv, gpu_rows))
    return out


# ---------------------------------------------------------------------------------------------
# the application: everything the handlers call, testable without a socket
# ---------------------------------------------------------------------------------------------
HEALTH_NOISE_RE = re.compile(r'path="/health"|\] slot data \||\] all slots are idle')
ENGINE_GET = {"health", "props", "v1/models", "pxa/stats", "pxa/speed", "pxa/explain", "slots"}
ENGINE_POST = {"v1/chat/completions", "completion"}


class EncodeHost(object):
    """What the Encode tab's backend (pxa_encode.EncodeService) needs from the GUI, and nothing more: the launcher module, the
    cards, the config file, the engine folders, the model folders. Kept as a small object so the backend can be tested
    with a fake one."""

    def __init__(self, app):
        self.app = app
        self.L = app.L

    def gpus(self):
        return self.app.gpus()

    def cfg_load(self):
        return load_config()

    def cfg_update(self, fn):
        with _cfg_lock:
            c = load_config()
            fn(c)
            save_config(c)

    def engine_dirs(self):
        try:
            E = self.app.rig_static().get("engine_dir")
        except Exception:        # noqa: BLE001 - the tab then says the engine tools are missing
            E = None
        out = [E] if E else []
        try:
            for d in self.L.find_source_builds():
                if d not in out:
                    out.append(d)
        except Exception:        # noqa: BLE001
            pass
        return out

    def model_roots(self):
        return self.app.model_roots()

    def cuda_version(self):
        try:
            return self.app.rig_static().get("cuda")
        except Exception:        # noqa: BLE001
            return None


# ---------------------------------------------------------------------------------------------
# the licensed library as its own updatable artifact (v3.1, tools/pxa_lib_update.py)
# ---------------------------------------------------------------------------------------------
# `libggml-pxqn.so` is the closed half of the engine and the only part of an install that is fixed on
# its own schedule: a kernel fix, a card that now works, a bug in the PXQN runtime. Shipping that as a
# whole engine release means downloading the engine again for one file, so the licence server hands out
# the file itself - signed, checksummed, swapped by a symlink rename, and always reversible - and this
# is the page's side of it: ask what is on the channel (a check changes nothing on disk but the
# last-check stamp), say "library update available" the way the encoder says it, and apply on demand.
# Never while a server from this install is running: the library is replaced underneath it.
# Beta is a separate channel and the server, not this page, decides who may have it.
LIB_CHECK_S = 6 * 3600          # how often the poller asks the licence server by itself
LIB_PAGE_S = 60.0               # refresh the cached answer when the page asks and it is older than this
LIB_CHANNELS = ("stable", "beta")   # the channels the licence server publishes; mirrored from the updater so a page can be refused without it


class LibUpdate(object):
    """The licensed library beside the Encode tab's encoder: same page, different artifact. Wraps tools/pxa_lib_update.py,
    which does the signature, checksum, staging and swap; this only decides when to ask and what the page is told."""

    def __init__(self, app):
        self.app = app
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.thread = None
        self._cache = None              # the last check: {checked, latest, update, notice, error, beta}
        self._beta = None               # True/False once the server has said; None before the first beta attempt
        self._busy = ""                 # "" | "check" | "apply" | "rollback": what is in flight, for the page
        self._ask = 0.0
        if LU is None:
            self._cache = {"error": {"code": "unavailable", "message": "This PXA Control has no library updater (%s)." % (LIB_IMPORT_ERROR or "")}}

    # ---- where and what ------------------------------------------------------------------------
    def dir(self):
        """The install the library belongs to: the engine directory Control runs from, else the per-user default."""
        try:
            return self.app.rig_static().get("engine_dir") or None
        except Exception:        # noqa: BLE001
            return None

    def _cfg(self):
        c = (self.app.cfg_load() if hasattr(self.app, "cfg_load") else load_config()) or {}
        l = c.get("lib")
        return l if isinstance(l, dict) else {}

    def cfg_set(self, **kw):
        def fn(c):
            l = c.get("lib") if isinstance(c.get("lib"), dict) else {}
            for k, v in kw.items():
                if v is None:
                    l.pop(k, None)
                else:
                    l[k] = v
            c["lib"] = l
        # App has no cfg_update of its own (only EncodeHost does), so fall back to the same
        # load-modify-save the config helpers use, under the same lock: the page and the library
        # poller both write this file.
        if hasattr(self.app, "cfg_update"):
            self.app.cfg_update(fn)
        else:
            with _cfg_lock:
                c = load_config()
                fn(c)
                save_config(c)

    def channel(self):
        """The channel this install follows: the setting the user chose, else what the last install recorded, else stable.
        The setting comes first because the check path stamps the channel it asked about into the state file, so a
        state-first order would let one check pin the install and make the page's channel control a no-op."""
        if LU is None:
            return "stable"
        ch = str(self._cfg().get("channel") or "").strip()
        if not ch:
            d = self.dir()
            ch = str(LU.load_state(d).get("channel") or "").strip()
        return ch if ch in ("stable", "beta") else "stable"

    def auto(self):
        return self._cfg().get("auto") is True

    def installed(self):
        if LU is None:
            return {}
        d = self.dir()
        st = LU.load_state(d)
        return {"version": str(st.get("version") or ""), "lib_id": st.get("lib_id") or None, "prev": st.get("prev") or None,
                "prev_version": str(st.get("prev_version") or ""), "applied": st.get("applied"), "checked": st.get("checked"),
                "links": LU.installed_links(d)}

    # ---- what the page is told -----------------------------------------------------------------
    def status(self):
        """The cached answer, and never a network call: the page polls this. `check()` refreshes it."""
        if LU is None:
            return {"available": False, "error": (self._cache or {}).get("error"), "update": False, "running": False}
        d = self.dir()
        try:
            running = LU.server_running(d)
        except Exception:        # noqa: BLE001
            running = False
        out = {"available": True, "installed": self.installed(), "channel": self.channel(), "auto": self.auto(),
               "beta": self._beta, "running": running, "busy": self._busy, "error": None, "latest": "",
               "lib_id": None, "update": False, "notice": "", "min_engine": ""}
        with self.lock:
            c = dict(self._cache) if isinstance(self._cache, dict) else None
        if c:
            out["error"] = c.get("error")
            out["latest"] = str(c.get("latest") or "")
            out["lib_id"] = c.get("lib_id")
            out["update"] = bool(c.get("update")) and not c.get("error")
            out["notice"] = str(c.get("notice") or "")
            out["min_engine"] = str(c.get("min_engine") or "")
            if c.get("beta") is not None:
                out["beta"] = c["beta"]
            if c.get("checked"):
                out["installed"]["checked"] = c["checked"]
        return out

    # ---- asking, applying ----------------------------------------------------------------------
    def check(self, force=False, channel=None):
        """Ask the licence server what is on the channel (writes nothing but the state's last-check stamp) and keep the answer
        for the page. Does not raise: a server we cannot reach is a sentence in the payload, like everywhere else here."""
        if LU is None:
            return self.status()
        now = time.time()
        with self.lock:
            fresh = self._cache is not None and (now - self._ask) < (LIB_PAGE_S if force else LIB_CHECK_S)
            if fresh and not force:
                return self.status()
            self._busy = "check"
        try:
            d = self.dir()
            ch = channel or self.channel()
            rel, raw, cur = LU.check(d, channel=ch)
            newer = bool(rel["version"]) and rel["version"] != cur
            entry = {"checked": int(now), "latest": rel["version"], "lib_id": rel["lib_id"], "update": newer,
                     "notice": str((raw or {}).get("notice") or ""), "min_engine": str(rel.get("min_engine") or ""), "error": None}
            if ch == "beta":
                self._beta = True
        except PK_ERRORS as e:
            entry = {"checked": int(now), "latest": "", "update": False, "notice": "", "error": _pkg_error(e)}
            if getattr(e, "code", "") == "not_valued":
                self._beta = False
        except Exception as e:   # noqa: BLE001
            entry = {"checked": int(now), "latest": "", "update": False, "notice": "",
                     "error": {"code": "failed", "message": "%s: %s" % (e.__class__.__name__, e)}}
        with self.lock:
            self._cache = entry
            self._ask = now
            self._busy = ""
        if entry["update"] and self.auto():
            try:
                self.apply()                     # off by default; the spec's rule is the owner turns it on
            except Invalid:
                pass
        return self.status()

    def apply(self, channel=None):
        """Install the newest release on the channel. Raises Invalid with a plain sentence (the page shows it) when the server
        refuses, a server from this install is running, or the engine is too old for the release."""
        if LU is None:
            raise Invalid("This PXA Control has no library updater (%s)." % (LIB_IMPORT_ERROR or ""))
        return self._do("apply", lambda: self._apply(channel))

    def rollback(self):
        if LU is None:
            raise Invalid("This PXA Control has no library updater (%s)." % (LIB_IMPORT_ERROR or ""))
        return self._do("rollback", lambda: self._rollback())

    def _do(self, what, fn):
        with self.lock:
            if self._busy:
                raise Invalid("A library %s is already running; wait for it to finish." % self._busy)
            self._busy = what
        try:
            done = fn()
            self.check(force=True)
            return done
        except PK_ERRORS as e:
            # the updater raises its own verifier error; the page is promised a plain sentence and a
            # 4xx, so translate it here instead of letting the dispatcher report an internal 500.
            err = Invalid(str(e))
            err.code = str(getattr(e, "code", "") or "") or "lib_failed"
            raise err
        finally:
            with self.lock:
                self._busy = ""

    def _apply(self, channel=None):
        d, ch = self.dir(), channel or self.channel()
        done, rel, cur = LU.apply(d, channel=ch)
        if not done:
            return {"applied": False, "version": cur, "message": "The library is already up to date."}
        return {"applied": True, "version": rel["version"], "lib_id": rel["lib_id"],
                "message": "Library %s installed (%s). Restart a server for it to take effect." % (rel["version"], rel["lib_id"])}

    def _rollback(self):
        d = self.dir()
        ver, was = LU.rollback(d)
        return {"applied": True, "version": ver, "was": was,
                "message": ("Rolled back to %s." % ver) if ver else "Rolled back to the library the engine shipped."}

    # ---- the poller ----------------------------------------------------------------------------
    def touch(self):
        """Somebody is looking at the page: start the background checker if it is not running."""
        if LU is None:
            return
        with self.lock:
            if (self.thread is None or not self.thread.is_alive()) and not self.stop_ev.is_set():
                self.thread = threading.Thread(target=self._loop, daemon=True, name="pxa-lib")
                self.thread.start()

    def close(self):
        self.stop_ev.set()

    def _loop(self):
        while not self.stop_ev.is_set():
            try:
                if self._cache is None:
                    self.check()
            except Exception:        # noqa: BLE001 - a failed check is a sentence on the page, never a dead thread
                pass
            self.stop_ev.wait(LIB_CHECK_S)

    def _safe_check(self):
        try:
            self.check()
        except Exception:            # noqa: BLE001
            pass


def _pkg_error(e):
    return {"code": str(getattr(e, "code", "") or "failed"), "message": str(e)}


# ---------------------------------------------------------------------------------------------
# live history: the numbers behind the Live tab
# ---------------------------------------------------------------------------------------------
# One rolling sample per card and per server, taken by a small thread that runs while somebody is
# looking at PXA Control (a viewer asked within LIVE_IDLE_S) and, with the telemetry history on (v3.1,
# tools/pxa_telemetry.py, the default), all the time at its slower background pace: every sample then
# also goes to the on-disk store, so the Live tab can draw days and weeks, not only what a page saw. Everything is read from places that
# exist on every PXA engine build: nvidia-smi for the cards, and per server /slots, /props, /pxa/stats
# (and /metrics when the server was started with --metrics, which is only used for exact rates).
# Only numbers are stored (telemetry.db in the config dir) and nothing leaves this machine; a request's
# prompt text, which /slots carries, is dropped on the way in and never reaches the page or the store.
LIVE_KEEP_S = 3 * 3600          # history kept in memory
LIVE_FAST_S = 2.0               # sample spacing while the Live tab is open
LIVE_SLOW_S = 10.0              # ... while only another tab is
LIVE_VIEW_S = 30.0              # "the Live tab is open": it asked within this many seconds
LIVE_IDLE_S = 600.0             # stop sampling when nobody has asked for this long
LIVE_FLEET_S = 15.0             # how stale the server list may be
LIVE_REQ_S = 5.0                # how often a server's finished requests are fetched
LIVE_PALETTE = 8                # colour slots (categorical palette in the page)
CARD_COLS = ["t", "mem", "util", "temp", "power", "limit", "clock"]
SRV_COLS = ["t", "dec", "pre", "busy", "slots", "ctx", "xhit", "xswap", "acc", "ema"]
HOST_COLS = ["t", "ram", "cpu", "swap"]
REQ_COLS = ["t", "slot", "n_prompt", "n_cached", "prompt_n", "prompt_ms", "prefill_tps", "n_gen", "gen_ms",
            "decode_tps", "draft_n", "draft_acc"]
PROM_COUNTERS = ("tokens_predicted_total", "tokens_predicted_seconds_total", "prompt_tokens_total",
                 "prompt_seconds_total")


def _num(x):
    """a finite float, else None ("[N/A]", "", nan, text)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _round_row(r, nd=2):
    return [None if v is None else (round(v, 1) if i == 0 else round(v, nd)) for i, v in enumerate(r)]


def bucket_rows(rows, step):
    """Mean of every column over `step`-second buckets (None skipped); a bucket's time is its mean
    time. step <= 0 or a short list: the rows unchanged."""
    if step <= 0 or len(rows) < 3:
        return [list(r) for r in rows]
    out, cur, key = [], [], None

    def flush():
        if not cur:
            return
        r = [sum(x[0] for x in cur) / len(cur)]
        for c in range(1, len(cur[0])):
            v = [x[c] for x in cur if x[c] is not None]
            r.append(sum(v) / len(v) if v else None)
        out.append(r)
    for x in rows:
        k = int(x[0] // step)
        if k != key:
            flush()
            cur, key = [], k
        cur.append(x)
    flush()
    return out


class Live(object):
    """Rolling live history of every card and every running server (see the block comment above)."""

    def __init__(self, app):
        self.app = app
        self.lock = threading.Lock()
        self.tick_lock = threading.Lock()
        self.nrows = int(LIVE_KEEP_S / LIVE_FAST_S) + 16
        self.cards = {}             # index -> deque of CARD_COLS rows
        self.card_meta = {}         # index -> {name, class, mem_total_mib, throttle}
        self.host = deque(maxlen=self.nrows)      # HOST_COLS rows: this machine's memory and CPU
        self.host_meta = {}
        self._cpu_prev = None
        self.srv = {}               # key -> per-server record, see _server()
        self.slot_of = {}           # key -> colour slot, first seen first served, never repainted
        self.last_any = 0.0
        self.last_fast = 0.0
        self.ticks = 0
        self.last_tick = 0.0
        self.thread = None
        self.stop_ev = threading.Event()
        self._thr_field = None      # which nvidia-smi throttle-reason field this driver has
        self.background = False     # the telemetry history is on: keep sampling with nobody looking
        self.bg_every = LIVE_SLOW_S
        self.bg_since = None

    def start_background(self, every=None):
        """sample with nobody looking, every `every` seconds (the telemetry store's sample_s)."""
        with self.lock:
            self.background = True
            self.bg_every = max(LIVE_FAST_S, float(every or LIVE_SLOW_S))
            if self.bg_since is None:
                self.bg_since = time.time()
            if (self.thread is None or not self.thread.is_alive()) and not self.stop_ev.is_set():
                self.thread = threading.Thread(target=self._loop, daemon=True, name="pxa-live")
                self.thread.start()

    def stop_background(self):
        with self.lock:
            self.background = False
            self.bg_since = None

    # ---- life cycle ---------------------------------------------------------------------------
    def touch(self, fast=False):
        """somebody is looking: start sampling, and (the Live tab only) take the very first sample right now so the
        first answer is not empty."""
        now = time.time()
        with self.lock:
            self.last_any = now
            if fast:
                self.last_fast = now
            first = fast and self.ticks == 0
            if (self.thread is None or not self.thread.is_alive()) and not self.stop_ev.is_set():
                self.thread = threading.Thread(target=self._loop, daemon=True, name="pxa-live")
                self.thread.start()
        if first:
            self.tick(wait=True)

    def close(self):
        self.stop_ev.set()

    def _loop(self):
        if self.ticks == 0:
            self._safe_tick()
        while not self.stop_ev.is_set():
            t0 = time.time()
            with self.lock:
                idle = t0 - self.last_any > LIVE_IDLE_S
                if idle and not self.background:
                    self.thread = None
                    return
                fast = t0 - self.last_fast < LIVE_VIEW_S
                every = LIVE_FAST_S if fast else (self.bg_every if idle else min(LIVE_SLOW_S, self.bg_every))
            self.stop_ev.wait(max(0.2, every - (time.time() - t0)))
            if not self.stop_ev.is_set():
                self._safe_tick()

    def _safe_tick(self):
        try:
            self.tick()
        except Exception:        # noqa: BLE001 - a failed sample is a gap in the chart, never a crash
            pass

    def slot_for(self, key):
        with self.lock:
            if key not in self.slot_of:
                self.slot_of[key] = len(self.slot_of) % LIVE_PALETTE
            return self.slot_of[key]

    # ---- one sample ---------------------------------------------------------------------------
    def tick(self, now=None, wait=False):
        if not self.tick_lock.acquire(blocking=wait):
            return                    # a sample is already being taken
        try:
            now = time.time() if now is None else now
            if self.ticks and 0 <= now - self.last_tick < 0.5:
                return                # two callers in the same instant: one sample is enough
            self.last_tick = now
            self._sample_cards(now)
            self._sample_host(now)
            self._sample_servers(now)
            self._prune(now)
            with self.lock:
                self.ticks += 1
            try:
                self._feed(now)
            except Exception:        # noqa: BLE001 - the history store is never allowed to break the Live tab
                pass
        finally:
            self.tick_lock.release()

    def _sample_cards(self, now):
        rows = self.app.gpu_live()
        with self.lock:
            for r in rows:
                i = r["index"]
                dq = self.cards.setdefault(i, deque(maxlen=self.nrows))
                dq.append((now, r.get("mem"), r.get("util"), r.get("temp"), r.get("power"), r.get("limit"),
                           r.get("clock")))
                self.card_meta[i] = {"name": r.get("name"), "class": r.get("class"),
                                     "mem_total_mib": r.get("mem_total"), "throttle": r.get("throttle") or []}

    def _sample_host(self, now):
        """memory and CPU of this machine from /proc (Linux): where a model's host-side tables and offloaded experts live."""
        try:
            mem = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                        mem[k] = int(v.split()[0]) / 1024.0
            with open("/proc/stat") as f:
                vals = [int(x) for x in f.readline().split()[1:]]
            idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
            if "MemTotal" not in mem or "MemAvailable" not in mem:
                return
        except (OSError, ValueError, IndexError):
            return
        cpu = None
        if self._cpu_prev and total > self._cpu_prev[1]:
            cpu = max(0.0, min(100.0, 100.0 * (1.0 - (idle - self._cpu_prev[0]) / float(total - self._cpu_prev[1]))))
        self._cpu_prev = (idle, total)
        swap = (mem.get("SwapTotal", 0.0) - mem.get("SwapFree", 0.0)) if mem.get("SwapTotal") else None
        with self.lock:
            self.host.append((now, mem["MemTotal"] - mem["MemAvailable"], cpu, swap))
            self.host_meta = {"ram_total_mib": mem["MemTotal"], "swap_total_mib": mem.get("SwapTotal") or 0.0}

    def _sample_servers(self, now):
        f = self.app.fleet(max_age=LIVE_FLEET_S)
        insts = [x for x in f.get("instances", []) if x.get("running") and x.get("port")]
        up = {x["key"] for x in insts}
        work = []
        for x in insts:
            st = self._server(x)
            if x.get("health") != "ok" or st["busy"]:
                continue
            st["busy"] = True
            work.append((x, st))
        res = {}

        def one(x, st):
            try:
                res[x["key"]] = self._poll(x, st, now)
            except Exception:        # noqa: BLE001
                res[x["key"]] = None
            finally:
                st["busy"] = False
        ths = [threading.Thread(target=one, args=a, daemon=True) for a in work]
        for th in ths:
            th.start()
        end = time.time() + 5.0                      # one shared deadline: a few hung servers must not stall the sample
        for th in ths:
            th.join(max(0.05, end - time.time()))
        with self.lock:
            for key, st in self.srv.items():
                st["up"] = key in up
            for key, got in res.items():
                if not got:
                    continue
                st = self.srv[key]
                row, meta, reqs = got
                st["rows"].append(row)
                st["meta"].update(meta)
                for q in reqs:
                    st["reqs"].append(q)

    def _server(self, x):
        key = x["key"]
        with self.lock:
            st = self.srv.get(key)
            if st is None:
                st = {"rows": deque(maxlen=self.nrows), "reqs": deque(maxlen=4000), "meta": {}, "busy": False,
                      "up": True, "prev": {}}
                self.srv[key] = st
            if key not in self.slot_of:
                self.slot_of[key] = len(self.slot_of) % LIVE_PALETTE
            st["meta"].update({"key": key, "name": x.get("label") or x.get("name"), "kind": x.get("kind"),
                               "port": x.get("port"), "gpus": x.get("gpus") or sorted((x.get("vram") or {}).keys()),
                               "model_file": x.get("model_file"), "slot": self.slot_of[key]})
            return st

    # ---- one server -----------------------------------------------------------------------------
    @staticmethod
    def _prom(port, timeout=1.5):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as r:
            txt = r.read().decode("utf-8", "replace")
        m = {k: float(v) for k, v in PROM_RE.findall(txt)}
        return m if all(k in m for k in PROM_COUNTERS) else None

    def _poll(self, x, st, now):
        """-> (row, meta update, new finished-request rows) for one server, from /slots, /props, /pxa/stats and, when the
        server has it, /metrics. Rates are differences of cumulative counters over the time since the last sample."""
        port = x["port"]
        prev = st["prev"]
        slots = props = None
        try:
            j = _http_json(port, "/slots", 1.5)
            slots = j if isinstance(j, list) else None
        except Exception:        # noqa: BLE001 - --no-slots, an older build, a busy server: no slot numbers this time
            pass
        try:
            j = _http_json(port, "/props", 1.5)
            props = j if isinstance(j, dict) else None
        except Exception:        # noqa: BLE001
            pass
        counters = None
        if prev.get("m_off", 0) <= now:
            try:
                counters = self._prom(port)
                if counters is None:
                    prev["m_off"] = now + 120
            except urllib.error.HTTPError:
                prev["m_off"] = now + 120          # 501: the server was not started with --metrics
            except Exception:        # noqa: BLE001
                prev["m_off"] = now + 30
        t_prev = prev.get("t")
        dt = (now - t_prev) if t_prev else None
        if dt is not None and dt <= 0:
            dt = None
        # ---- slots: busy count, tokens written, draft acceptance, what each busy slot is doing
        cur, busy, d_tok, clean, d_tot, d_acc, ema, slot_rows = {}, 0, 0, 0, 0, 0, [], []
        for s in slots or []:
            if not isinstance(s, dict):
                continue
            sid, task = s.get("id"), s.get("id_task")
            nt = s.get("next_token") if isinstance(s.get("next_token"), dict) else {}
            nd = int(_num(nt.get("n_decoded")) or 0)
            remain = _num(nt.get("n_remain"))
            state = s.get("state")
            is_busy = bool(state) if state is not None else bool(s.get("is_processing"))
            dtot, dacc = int(_num(s.get("n_draft_total")) or 0), int(_num(s.get("n_draft_accepted")) or 0)
            p = (prev.get("slots") or {}).get(sid)
            same = bool(p) and p[0] == task and dt is not None
            if same:
                if nd >= p[1] and p[4] and is_busy and nd > 0:
                    d_tok += nd - p[1]          # a slot that was writing for the whole interval: its tokens count
                    clean += 1
                if dtot >= p[2]:
                    d_tot += dtot - p[2]
                    d_acc += max(0, dacc - p[3])
            began = (p[5] if (p and p[0] == task and p[4]) else None)
            dec_at = (p[6] if (p and p[0] == task) else None)
            if is_busy:
                busy += 1
                if began is None:
                    began = now - (dt or 0) / 2
                if nd > 0 and dec_at is None:
                    dec_at = now
                if dtot > 0 and _num(s.get("spec_accept_ema")) is not None:
                    ema.append(_num(s.get("spec_accept_ema")))
                slot_rows.append({"id": sid, "phase": "decode" if nd > 0 else "prefill", "n_decoded": nd,
                                  "n_remain": None if remain is None or remain < 0 else int(remain), "t0": round(began, 1),
                                  "t1": None if dec_at is None else round(dec_at, 1)})
            else:
                began = dec_at = None
            cur[sid] = (task, nd, dtot, dacc, is_busy, began, dec_at)
        prev["slots"] = cur
        # ---- exact engine counters when --metrics is on: tokens per BUSY second, not per wall second
        dec = pre = None
        m_prev = prev.get("m")
        if counters is not None:
            if m_prev is not None:
                dtok = counters["tokens_predicted_total"] - m_prev["tokens_predicted_total"]
                dsec = counters["tokens_predicted_seconds_total"] - m_prev["tokens_predicted_seconds_total"]
                dpt = counters["prompt_tokens_total"] - m_prev["prompt_tokens_total"]
                dps = counters["prompt_seconds_total"] - m_prev["prompt_seconds_total"]
                if dtok >= 0 and dsec > 0:
                    dec = dtok / dsec
                if dpt >= 0 and dps > 0:
                    pre = dpt / dps
            prev["m"] = counters
        if dec is None and slots is not None and dt is not None and clean:
            dec = d_tok / dt          # estimated from slot progress; no value (a gap) while nothing is being written
        # ---- speculation
        acc = (d_acc / d_tot) if d_tot > 0 else None
        ema_v = (sum(ema) / len(ema)) if ema else None
        has_spec = bool(prev.get("has_spec")) or any(v[2] > 0 for v in cur.values())
        prev["has_spec"] = has_spec
        # ---- the expert cache and the KV ring, from /props
        xhit = xswap = ctx = n_ctx = None
        xc = (props or {}).get("pxa_xcache")
        has_x = isinstance(xc, dict)
        if has_x:
            h, m_, sw = _num(xc.get("hits")) or 0, _num(xc.get("misses")) or 0, _num(xc.get("swaps_started")) or 0
            xp = prev.get("x")
            if xp and dt:
                dh, dm = h - xp[0], m_ - xp[1]
                if dh >= 0 and dm >= 0 and dh + dm > 0:
                    xhit = dh / (dh + dm)
                if sw >= xp[2]:
                    xswap = (sw - xp[2]) / dt
            prev["x"] = (h, m_, sw)
        if props:
            n_ctx = _num(props.get("n_ctx"))
            used = _num(props.get("kv_cache_used_cells"))
            if n_ctx and n_ctx > 0 and used is not None:
                ctx = min(1.0, used / n_ctx)
        prev["t"] = now
        row = (now, dec, pre, busy if slots is not None else None, len(slots) if slots is not None else None, ctx,
               xhit, xswap, acc, ema_v)
        phase = "idle" if not busy else ("decode" if any(r["phase"] == "decode" for r in slot_rows) else "prefill")
        meta = {"phase": phase, "active": slot_rows, "n_ctx": None if n_ctx is None else int(n_ctx),
                "has_xcache": has_x or bool(st["meta"].get("has_xcache")), "has_spec": has_spec,
                "source": "engine counters" if counters is not None else ("slot progress" if slots is not None else None),
                "total_slots": (int(props["total_slots"]) if props and _num(props.get("total_slots")) else
                                (len(slots) if slots is not None else None)),
                "model": os.path.basename(str((props or {}).get("model_alias") or (props or {}).get("model_name") or "")) or None}   # a name, never a path
        # ---- finished requests: the server's own record of each one (prefill, decode, draft counts)
        reqs = []
        if now - prev.get("req_t", 0) >= LIVE_REQ_S:
            prev["req_t"] = now
            since = prev.get("req_since") or (now - LIVE_KEEP_S)
            try:
                d = _http_json(port, f"/pxa/stats?since={since:.3f}&limit=500", 2.5)
                top = since
                for r in (d or {}).get("records") or []:
                    if not isinstance(r, dict) or not _num(r.get("ts")) or r["ts"] <= since:
                        continue
                    top = max(top, r["ts"])
                    reqs.append((r["ts"], int(_num(r.get("slot")) if _num(r.get("slot")) is not None else -1),
                                 _num(r.get("n_prompt")), _num(r.get("n_cached")), _num(r.get("prompt_n")),
                                 _num(r.get("prompt_ms")), _num(r.get("prefill_tps")), _num(r.get("n_gen")),
                                 _num(r.get("gen_ms")), _num(r.get("decode_tps")), _num(r.get("draft_n")),
                                 _num(r.get("draft_acc"))))
                prev["req_since"] = top
                meta["has_stats"] = True
            except Exception:        # noqa: BLE001 - an older build without /pxa/stats
                meta["has_stats"] = False
        return row, meta, reqs

    @staticmethod
    def store_key(key, port):
        """the history's name for a server: a bare process is p:<pid>, which changes at every restart, so the
        store files it under its port instead (the same server on the same port keeps one line)."""
        return f"p@{port}" if str(key).startswith("p:") and port else str(key)

    def _feed(self, now):
        """hand this tick's rows to the telemetry store (averaged there and written every sample_s)."""
        tel = getattr(self.app, "telemetry", None)
        if tel is None:
            return
        recent = now - max(30.0, 3 * self.bg_every)      # a server's first poll back-fills 3 h of requests: those
        cards, srv, host = [], [], None                  # go to the request table, not into this interval's sums
        with self.lock:
            for i, dq in self.cards.items():
                if dq and dq[-1][0] == now:
                    m = self.card_meta.get(i, {})
                    cards.append((str(i), m.get("name"), {"class": m.get("class"), "mem_total_mib": m.get("mem_total_mib")},
                                  list(dq[-1][1:])))
            if self.host and self.host[-1][0] == now:
                host = list(self.host[-1][1:])
            for key, st in self.srv.items():
                m = st["meta"]
                hw = st.get("fed_req", 0.0)
                new = [r for r in st["reqs"] if r[0] > hw]
                if new:
                    st["fed_req"] = max(r[0] for r in new)
                row = st["rows"][-1] if st["rows"] and st["rows"][-1][0] == now else None
                if row is None and not new:
                    continue
                info = {"kind": m.get("kind"), "port": m.get("port"), "gpus": m.get("gpus"),
                        "model_file": os.path.basename(str(m.get("model_file") or m.get("model") or "")) or None}
                srv.append((self.store_key(key, m.get("port")), m.get("name"), info, row, new))
        for k, name, info, vals in cards:
            tel.record("card", k, now, vals, label=name, info=info)
            try:
                reasons = (self.card_meta.get(int(k)) or {}).get("throttle") or []
            except (TypeError, ValueError):
                reasons = []
            if reasons:
                tel.note_throttle(k, now, reasons)
        if host is not None:
            tel.record("host", "host", now, host, label="this machine", info={"ram_total_mib": self.host_meta.get("ram_total_mib")})
        for k, name, info, row, new in srv:
            if new:
                tel.record_requests(k, new, label=name, info=info)
            if row is not None:
                fresh = [r for r in new if r[0] >= recent]
                sums = [float(len(fresh)), sum(r[7] or 0 for r in fresh), sum(r[4] or 0 for r in fresh),
                        sum(r[8] or 0 for r in fresh), sum(r[5] or 0 for r in fresh)]
                tel.record("server", k, now, list(row[1:]) + sums, label=name, info=info)
        tel.flush(now)

    def latest(self, max_age=60.0, with_meta=False):
        """the newest reading of every card, server and the host, for /metrics.

        `with_meta` also carries what the row alone cannot say -- the server's phase, model name and
        slot count, a card's total VRAM -- which is what the MQTT publisher needs to build entities.
        The default is off so /metrics keeps exactly the shapes it had."""
        now = time.time()
        with self.lock:
            cards = []
            for i in sorted(self.cards):
                dq = self.cards[i]
                if dq and now - dq[-1][0] <= max_age:
                    r = dq[-1]
                    m = self.card_meta.get(i) or {}
                    c = dict(zip(CARD_COLS, r), index=i, name=m.get("name"))
                    if with_meta:
                        c["mem_total_mib"] = m.get("mem_total_mib")
                    cards.append(c)
            servers = []
            for key, st in self.srv.items():
                if st["up"] and st["rows"] and now - st["rows"][-1][0] <= max_age:
                    m = st["meta"]
                    s = dict(zip(SRV_COLS, st["rows"][-1]), key=self.store_key(key, m.get("port")),
                             name=m.get("name"), port=m.get("port"))
                    if with_meta:
                        s.update({k: m.get(k) for k in ("phase", "total_slots", "model", "n_ctx",
                                                        "kind", "model_file")})
                    servers.append(s)
            host = dict(zip(HOST_COLS, self.host[-1])) if self.host and now - self.host[-1][0] <= max_age else None
        return cards, servers, host

    def _prune(self, now):
        cut = now - LIVE_KEEP_S
        with self.lock:
            for dq in list(self.cards.values()) + [self.host]:
                while dq and dq[0][0] < cut:
                    dq.popleft()
            for key in list(self.srv):
                st = self.srv[key]
                for dq in (st["rows"], st["reqs"]):
                    while dq and dq[0][0] < cut:
                        dq.popleft()
                if not st["up"] and not st["rows"]:
                    del self.srv[key]

    # ---- what the page asks for ----------------------------------------------------------------
    def snapshot(self, since=None, step=0.0):
        now = time.time()
        since = (now - 900) if since is None else since
        with self.lock:
            cards = []
            for i in sorted(self.cards):
                m = self.card_meta.get(i, {})
                cards.append({"index": i, "name": m.get("name"), "class": m.get("class"), "mem_total_mib": m.get("mem_total_mib"),
                              "throttle": list(m.get("throttle") or []), "slot": i % LIVE_PALETTE,
                              "rows": [r for r in self.cards[i] if r[0] > since]})
            servers = []
            for key, st in self.srv.items():
                s = dict(st["meta"])
                s["up"] = st["up"]
                s["rows"] = [r for r in st["rows"] if r[0] > since]
                s["reqs"] = [r for r in st["reqs"] if r[0] > since]
                servers.append(s)
            host = {"ram_total_mib": self.host_meta.get("ram_total_mib"), "swap_total_mib": self.host_meta.get("swap_total_mib"),
                    "rows": [r for r in self.host if r[0] > since]} if self.host else None
            first = min([dq[0][0] for dq in self.cards.values() if dq] + [d["rows"][0][0] for d in self.srv.values() if d["rows"]] or [now])
            ticks = self.ticks
        for c in cards:
            c["rows"] = [_round_row(r) for r in bucket_rows(c["rows"], step)]
        for s in servers:
            s["rows"] = [_round_row(r, 3) for r in bucket_rows(s["rows"], step)]
            s["reqs"] = [_round_row(r, 2) for r in s["reqs"]]
        if host:
            host["rows"] = [_round_row(r, 1) for r in bucket_rows(host["rows"], step)]
        servers.sort(key=lambda s: s.get("slot", 0))
        return {"ts": now, "since": since, "history_from": first, "every": LIVE_FAST_S, "ticks": ticks,
                "background": self.background, "background_since": self.bg_since,
                "cols": {"card": CARD_COLS, "server": SRV_COLS, "req": REQ_COLS, "host": HOST_COLS}, "palette": LIVE_PALETTE,
                "cards": cards, "servers": servers, "host": host}


class AppLauncher(object):
    """what the auto-start supervisor (tools/pxa_ctl/supervisor.py) may do: start / stop / ask about a SAVED server of
    this Control, through App.start / App.stop, so an auto-start gets the same validation, conflict check, card lock and
    plan as a click on Start."""

    def __init__(self, app):
        self.app = app

    def _settings(self, sid):
        prof = self.app.profiles()
        if sid not in prof:
            raise Invalid(f"no saved server '{sid}'")
        st = dict(prof[sid].get("settings") or {})
        if not st.get("model"):
            raise Invalid(f"server '{sid}' has no model saved: open it in Launch and Save")
        return st

    def start(self, sid, spec=None):
        body = self._settings(sid)
        body["sid"] = sid
        if spec:                                   # a power user's per-auto-start extras, validated by App.start like any launch
            if spec.get("extra_args"):
                body["extra_args"] = validate_extra_args(body.get("extra_args")) + list(spec["extra_args"])
            if spec.get("env"):
                body["levers"] = dict(body.get("levers") or {}, **spec["env"])
        r = self.app.start(body)
        if not r.get("ok"):
            tail = (r.get("text") or "").strip().splitlines()[-3:]
            raise Invalid(f"server '{sid}' did not start: " + " / ".join(tail)[:300])
        return r

    def stop(self, sid):
        return self.app.stop(sid)

    def status(self, sid):
        seat = self.app.seats.get(sid)
        if seat is None:
            return {"running": False, "healthy": False, "stopped_by_user": False, "exit_code": None}
        d = seat.status()
        healthy = False
        if d["running"] and d.get("port"):
            h, _detail = probe_health(d["port"], timeout=2)
            healthy = h == "ok"
        return {"running": d["running"], "healthy": healthy, "exit_code": d.get("exit_code"),
                "stopped_by_user": (not d["running"]) and d.get("phase") == "stopped" and seat.proc is not None}

    def cards(self, sid):
        return self.app.cards_by_index(self._settings(sid).get("gpus") or [])


class App(object):
    def __init__(self, L, port=DEFAULT_PORT, lan=False, token=None, models_dirs=None):
        self.L = L
        self.port = port
        self.lan = lan
        self.token = token if token is not None else (load_or_make_token() if lan else None)
        self.seat = Seat(L, "main")          # the first server (the single seat of older versions)
        self.seats = {"main": self.seat}
        self.seats_lock = threading.Lock()
        self.plan_lock = threading.Lock()
        self._gpu_cache = (0.0, None)
        self._gpu_lock = threading.Lock()
        self._fleet_cache = (0.0, None)
        self._fleet_lock = threading.Lock()
        self._launched_seen = None
        self._engines_cache = (0.0, None)
        self._default_limits = None
        self.catalog, self.catalog_src = load_catalog()
        self.catalog_names = LeverNames({r["name"] for r in self.catalog}, *load_builtin(self.catalog_src))
        self._rig_cache = (0.0, None)
        self._doctor_cache = (0.0, None)
        self._prof_cache = {}
        self._models_cache = None
        self._enc = None
        self._enc_lock = threading.Lock()
        self.lib = LibUpdate(self)
        self.live = Live(self)
        self.telemetry = None                     # the on-disk history (start_telemetry), None when off
        self.tel_cfg = TEL.settings(load_config()) if TEL is not None else {"enabled": False, "prometheus": False}
        self.tel_error = TEL_IMPORT_ERROR
        self.mqtt = None                          # the Home Assistant publisher thread (start_mqtt), None when off
        self.mqtt_cfg = MQTT.settings(load_config()) if MQTT is not None else {"enabled": False}
        self.mqtt_error = MQTT_IMPORT_ERROR
        self.bench = {"running": False, "lines": [], "result": None, "error": None}
        self.started_at = time.time()
        self.gpuctl = None                        # the Profiles backend (make_gpuctl), None when tools/pxa_ctl is missing
        self.gpuctl_error = CTL_IMPORT_ERROR
        self.make_gpuctl()
        self.companion = False                    # started in the background next to a server (serve(companion=True))
        self.idle_s = CONTROL_IDLE_S
        self.last_request = self.last_busy = time.time()
        with _cfg_lock:
            c = load_config()
            added = False
            for d in (models_dirs or []):
                d = os.path.abspath(os.path.expanduser(d))
                if d not in c["model_dirs"]:
                    c["model_dirs"].append(d)
                    added = True
            if added:
                save_config(c)

    # ---- telemetry history (v3.1) ---------------------------------------------------------
    def telemetry_path(self):
        return os.environ.get("PXA_CONTROL_TELEMETRY_DB") or os.path.join(config_dir(), "telemetry.db")

    def start_telemetry(self):
        """open the history store and keep the Live sampler running in the background. One writer per
        store file: a second PXA Control on the same config dir reads the history but does not record."""
        if TEL is None:
            return None
        with _cfg_lock:
            self.tel_cfg = TEL.settings(load_config())
        if not self.tel_cfg["enabled"] or self.telemetry is not None:
            return self.telemetry
        try:
            st = TEL.Store(self.telemetry_path(), self.tel_cfg)
        except Exception as e:        # noqa: BLE001 - an unwritable config dir: Live stays memory-only
            self.tel_error = f"{e.__class__.__name__}: {e}"
            return None
        if st.lock_writer():
            st.start()
            self.live.start_background(self.tel_cfg["sample_s"])
        self.telemetry = st
        self.tel_error = None
        return st

    def stop_telemetry(self):
        st, self.telemetry = self.telemetry, None
        self.live.stop_background()
        if st is not None:
            st.close()

    def telemetry_status(self):
        out = {"available": TEL is not None, "settings": dict(self.tel_cfg), "error": self.tel_error,
               "recording": False, "env": {k: os.environ.get(v) for k, v in (TEL._ENV.items() if TEL else ())
                                           if os.environ.get(v) is not None}}
        if self.telemetry is not None:
            out.update(self.telemetry.status())
            out["recording"] = self.telemetry.is_writer and out.get("writer_alive", False)
            out["writer_pid"] = self.telemetry.writer_pid()
        out["sampling_since"] = self.live.bg_since
        return out

    def set_telemetry(self, body):
        """POST /api/telemetry/settings: change and persist the history settings (an env var still wins)."""
        if TEL is None:
            raise Invalid("the telemetry module is missing from this PXA Control: " + (TEL_IMPORT_ERROR or ""))
        body = body or {}
        allowed = {k: body[k] for k in TEL.DEFAULTS if k in body}
        if not allowed:
            raise Invalid("nothing to change: send one of " + ", ".join(TEL.DEFAULTS))
        with _cfg_lock:
            c = load_config()
            t = dict(c.get("telemetry") or {})
            t.update(allowed)
            c["telemetry"] = TEL.settings({"telemetry": t}, env={})
            save_config(c)
            new = TEL.settings(c)
        was = self.tel_cfg
        self.tel_cfg = new
        if not new["enabled"] and self.telemetry is not None:
            self.stop_telemetry()
        elif new["enabled"] and self.telemetry is None:
            self.start_telemetry()
        elif self.telemetry is not None:
            self.telemetry.cfg.update(new)
            if new["sample_s"] != was.get("sample_s"):
                self.live.start_background(new["sample_s"])
            self.telemetry.maintain_now()
        return self.telemetry_status()

    def metrics_text(self):
        """GET /metrics (Prometheus text format, off unless prometheus is on): PXA Control's own per-card,
        per-server and host gauges, then every running server's own /metrics relabeled server="<key>"."""
        self.live.touch()
        cards, servers, host = self.live.latest()
        engine = {}
        ths = []

        def one(s):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{s['port']}/metrics", timeout=2.0) as r:
                    engine[s["key"]] = r.read(1 << 20).decode("utf-8", "replace")
            except Exception:        # noqa: BLE001 - started without --metrics: Control's gauges only
                pass
        for s in servers:
            if s.get("port"):
                th = threading.Thread(target=one, args=(s,), daemon=True)
                th.start()
                ths.append(th)
        end = time.time() + 3.0
        for th in ths:
            th.join(max(0.05, end - time.time()))
        counters = self.telemetry.counters if self.telemetry is not None else {}
        return TEL.prom_text(cards, servers, host, counters=dict(counters), engine=dict(engine))

    # ---- Home Assistant over MQTT (v3.1, tools/pxa_mqtt) ----------------------------------------
    def mqtt_host_label(self):
        """the name Home Assistant shows for this machine: the hostname, or PXA_CONTROL_MQTT_LABEL."""
        return (os.environ.get("PXA_CONTROL_MQTT_LABEL") or socket.gethostname() or "pxa-control")

    def mqtt_snapshot(self):
        """the Live sampler's newest reading, shaped for the publisher: every card, every running
        server (with the phase/model/slot count the row does not carry), the host, and the expert
        map's learned-session count."""
        self.live.touch()
        cards, servers, host = self.live.latest(with_meta=True)
        model_file = next((s.get("model_file") for s in servers if s.get("model_file")), None)
        snap = {"cards": cards, "servers": servers, "host": host or {}}
        try:
            import pxa_expert_map as M                       # optional, like everywhere else in Control
            st = M.status_dict(model_file)
            # the KEY's presence is what says "there is a map to report on": with no module there is no
            # map, and an "idle" published from an empty dict would claim one exists and is quiet.
            snap["expert"] = {"sessions": st.get("sessions"), "learning": bool(st.get("learning"))}
        except Exception:        # noqa: BLE001 - no expert map on this build/model: those entities are just absent
            pass
        return snap

    def start_mqtt(self):
        """start the Home Assistant publisher (off unless the settings say on). Publishing is
        one thread on a plain socket; a broker that is down is recorded, never raised."""
        if MQTT is None:
            return None
        with _cfg_lock:
            self.mqtt_cfg = MQTT.settings(load_config())
        if not self.mqtt_cfg["enabled"] or self.mqtt is not None:
            return self.mqtt
        if not self.mqtt_cfg.get("host"):
            self.mqtt_error = "no broker host set"
            return None
        try:
            p = MQTT.Publisher(self.mqtt_cfg, self.mqtt_snapshot, self.mqtt_host_label())
            p.start()
        except Exception as e:        # noqa: BLE001 - a bad setting must not take Control down
            self.mqtt_error = f"{e.__class__.__name__}: {e}"
            return None
        self.mqtt, self.mqtt_error = p, None
        return p

    def stop_mqtt(self):
        p, self.mqtt = self.mqtt, None
        if p is not None:
            p.stop()

    def mqtt_status(self):
        out = {"available": MQTT is not None, "settings": MQTT.public(self.mqtt_cfg) if MQTT else {},
               "error": self.mqtt_error, "host_label": self.mqtt_host_label(),
               "env": {k: os.environ.get(v) for k, v in (MQTT._ENV.items() if MQTT else ())
                       if os.environ.get(v) is not None}}
        if self.mqtt is not None:
            out.update(self.mqtt.status())
        return out

    def set_mqtt(self, body):
        """POST /api/mqtt/settings: change and persist the publisher's settings (an env var still wins)."""
        if MQTT is None:
            raise Invalid("the MQTT module is missing from this PXA Control: " + (MQTT_IMPORT_ERROR or ""))
        body = body or {}
        allowed = {k: body[k] for k in MQTT.DEFAULTS if k in body}
        if "password" in body and body["password"] == "" and "password" not in allowed:
            allowed["password"] = ""            # an explicit empty string clears the stored password
        if not allowed:
            raise Invalid("nothing to change: send one of " + ", ".join(MQTT.DEFAULTS))
        if "base" in allowed and not MQTT.valid_topic_root(allowed["base"]):
            raise Invalid("the base topic may not contain a space, '+' or '#', and may not start with '$'")
        with _cfg_lock:
            c = load_config()
            m = dict(c.get("mqtt") or {})
            m.update(allowed)
            c["mqtt"] = MQTT.settings({"mqtt": m}, env={})
            save_config(c)
            new = MQTT.settings(c)
        was = self.mqtt_cfg
        self.mqtt_cfg = new
        if not new["enabled"]:
            self.stop_mqtt()
        elif self.mqtt is None:
            self.start_mqtt()
        else:
            p = self.mqtt
            p.cfg = dict(new)
            if (new["host"], new["port"], new["user"], new["password"], new["tls"], new["base"],
                    new["prefix"]) != (was["host"], was["port"], was["user"], was["password"],
                                       was["tls"], was["base"], was["prefix"]):
                self.stop_mqtt()                # a changed link needs a fresh CONNECT: restart the thread
                self.start_mqtt()
        return self.mqtt_status()

    def mqtt_test(self, body):
        """POST /api/mqtt/test: connect to the broker as configured and prove the link with a
        PINGREQ, publishing nothing -- so a test cannot leave a retained entity behind. Settings in
        the body are tested instead of the saved ones, so the button works before you save."""
        if MQTT is None:
            raise Invalid("the MQTT module is missing from this PXA Control: " + (MQTT_IMPORT_ERROR or ""))
        body = body or {}
        cand = dict(self.mqtt_cfg)
        cand.update({k: body[k] for k in MQTT.DEFAULTS if k in body})
        if not body.get("password"):
            cand["password"] = self.mqtt_cfg.get("password", "")   # keep the stored one when the page sent none
        cand = MQTT.settings({"mqtt": cand}, env={})
        if not cand.get("host"):
            return {"ok": False, "error": "no broker host set"}
        out = MQTT.test_connection(cand)
        out["settings"] = MQTT.public(cand)
        return out

    # ---- GPU profiles (v3.1, tools/pxa_ctl) ------------------------------------------------------
    def make_gpuctl(self, adapter=None):
        """build the Profiles backend. It asks no driver anything until a page or a profile needs it."""
        if CTL is None:
            return None
        try:
            rows = None
            if os.environ.get("PXA_LAUNCH_FAKE_GPUS"):
                rows, _e = self.gpus()
            ad = adapter or CTL_DRV.make_adapter(rows=rows)
            self.gpuctl = CTL_SVC.GpuControl(config_dir(), CTL_SVC.settings_from(load_config()), adapter=ad,
                                             launcher=AppLauncher(self))
            self.gpuctl_error = None
        except Exception as e:        # noqa: BLE001 - an unreadable config: Control works, the tab says why
            self.gpuctl, self.gpuctl_error = None, f"{e.__class__.__name__}: {e}"
        return self.gpuctl

    def server_context(self, x=None, seat=None, healthy=False):
        """the resolved context of a fleet instance or a seat (cached per port + pid: it cannot change while the process
        lives). Read from the engine (/props, else its load log), never computed here."""
        cache = self.__dict__.setdefault("_ctx_cache", {})
        if seat is not None:
            port, pid = seat.port(), (seat.proc.pid if seat.proc else None)
            requested = (seat.req or {}).get("ctx")
            arg = _argv_value(seat.cmd or [], "-c", "--ctx-size")
            cards = (seat.req or {}).get("gpus")
        else:
            port, pid = x.get("port"), x.get("pid")
            seat = self.seats.get(x["sid"]) if x.get("kind") == "managed" and x.get("sid") else None
            if seat is not None and seat.running():
                return self.server_context(seat=seat, healthy=healthy)
            requested = x.get("ctx")
            arg = x.get("ctx")
            cards = x.get("gpus")
        key = (port, pid)
        probed = cache.get(key)
        if probed is None and port and healthy:
            probed = probe_ctx(port)
            if probed:
                if len(cache) > 64:
                    cache.clear()
                cache[key] = probed
        log = None
        if not probed and seat is not None:
            with seat.cond:
                lines = [ln for _q, ln in list(seat.log)[-4000:] if "n_ctx" in ln]
            log = ctx_from_log(lines)
        if not probed and not (log and log[0]) and not (seat is not None and seat.running()) and not port:
            return None
        return context_info(requested, arg, probed, log, cards)

    def need_gpuctl(self):
        if self.gpuctl is None:
            raise Invalid("GPU profiles are not available in this PXA Control: " + (self.gpuctl_error or "tools/pxa_ctl missing"))
        return self.gpuctl

    def cards_by_index(self, indexes):
        """[(uuid, index)] of the launcher's card indexes (a launch names cards by index)."""
        rows, _e = self.gpus()
        by = {g[0]: g[5] for g in rows}
        return [(by.get(i, f"index-{i}"), i) for i in indexes]

    def gpu_state(self):
        """GET /api/gpu/state: the backend's state plus what only Control knows: who runs on each card, the speed of
        the servers there (tok/s per watt), a power / temperature sparkline (the history store when it records, else
        the Live tab's memory) and the saved servers an auto-start can name."""
        g = self.need_gpuctl()
        st = g.state()
        self.live.touch()                         # keeps the in-memory sampler alive while the page is open (10 s steps)
        try:
            fl = self.fleet(max_age=15)
        except Exception:             # noqa: BLE001
            fl = {"cards": [], "instances": []}
        users = {c["index"]: c for c in fl.get("cards") or []}
        names = {x["key"]: (x.get("label") or x.get("name")) for x in fl.get("instances") or []}
        now = time.time()
        spark = self._gpu_spark([c["index"] for c in st["cards"]], now)
        speed = self._gpu_speed(st["cards"], now)
        for c in st["cards"]:
            u = users.get(c["index"]) or {}
            c["users"] = [{"name": names.get(x.get("key")) or x.get("name"), "mib": x.get("mib"), "key": x.get("key")}
                          for x in u.get("users") or []]
            c["other_mib"] = u.get("other_mib") or 0
            c["other_names"] = u.get("other_names") or []
            c["spark"] = spark.get(c["index"], [])
            c["tps"], c["tps_per_w"] = speed.get(c["index"], (None, None))
        prof = self.profiles()
        st["servers"] = [{"sid": k, "name": v.get("name", k), "model": os.path.basename(str((v.get("settings") or {}).get("model") or "")),
                          "gpus": (v.get("settings") or {}).get("gpus") or []} for k, v in sorted(prof.items())]
        st["telemetry"] = self.telemetry is not None
        return st

    def _gpu_spark(self, indexes, now, span=3600, points=60):
        out = {}
        tel = self.telemetry
        if tel is not None:
            try:
                d = tel.series("card", keys=[str(i) for i in indexes], since=now - span, until=now, max_points=points,
                               cols=["power", "temp"])
                for m in d.get("series") or []:
                    out[int(m["key"])] = [[r[0], r[1], r[2]] for r in m["rows"]]
            except Exception:         # noqa: BLE001 - no history: the in-memory rows below
                out = {}
        with self.live.lock:
            for i in indexes:
                if out.get(i):
                    continue
                rows = [r for r in self.live.cards.get(i, ()) if r[0] >= now - span]
                step = max(1, len(rows) // points)
                out[i] = [[round(r[0], 1), r[4], r[3]] for r in rows[::step]]
        return out

    def _gpu_speed(self, cards, now):
        """{index: (tok/s on this card, tok/s per watt)}: each running server's decode speed shared over its cards by
        their power draw (a server on two cards at 120 W and 80 W: 60 % / 40 %)."""
        power = {c["index"]: c.get("power_w") for c in cards}
        tps = {}
        with self.live.lock:
            for st in self.live.srv.values():
                if not st.get("up") or not st["rows"] or now - st["rows"][-1][0] > 60:
                    continue
                dec = st["rows"][-1][1]
                gp = []
                for x in st["meta"].get("gpus") or []:
                    try:
                        gp.append(int(x))
                    except (TypeError, ValueError):
                        continue
                tot = sum(power.get(i) or 0 for i in gp)
                if not dec or not gp or tot <= 0:
                    continue
                for i in gp:
                    tps[i] = tps.get(i, 0.0) + dec * (power.get(i) or 0) / tot
        out = {}
        for i, v in tps.items():
            p = power.get(i)
            out[i] = (round(v, 2), round(v / p, 4) if p else None)
        return out

    # ---- rig ------------------------------------------------------------------------------
    def gpus(self, max_age=1.5):
        """gpu_table() rows, shared for max_age seconds (the dashboard, the rig page and every open
        tab used to run nvidia-smi on their own every 3 s). Without nvidia-smi (a container with
        the compute driver only) the card list comes from NVML or the CUDA driver instead of being
        empty (shibi 2026-10-01-c2a51dfa)."""
        t, v = self._gpu_cache
        if v is not None and time.time() - t < max_age:
            return v
        with self._gpu_lock:
            t, v = self._gpu_cache
            if v is not None and time.time() - t < max_age:
                return v
            rows, err = self.L.gpu_table()
            if not rows and not os.environ.get("PXA_LAUNCH_FAKE_GPUS"):
                t0, fb = getattr(self, "_fallback_cache", (0.0, None))
                if fb is None or time.time() - t0 > 30:
                    fb = probe_gpus_fallback()
                    self._fallback_cache = (time.time(), fb)
                frows, src, ferr = fb
                if frows:
                    rows, err = frows, f"nvidia-smi unavailable ({err}); cards read from {src}"
                elif ferr:
                    err = f"{err}; fallback: {ferr}"
            v = (rows or [], err)
            self._gpu_cache = (time.time(), v)
            return v

    HOT_C = 80.0
    THROTTLE_WARN = ("sw_power_cap", "hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown")

    @classmethod
    def heat_warning(cls, index, temp_c, reasons):
        """One plain sentence for a card at/above 80 C or with an active thermal/power throttle, else None."""
        bad = [r for r in (reasons or []) if r in cls.THROTTLE_WARN]
        hot = temp_c is not None and temp_c >= cls.HOT_C
        if not hot and not bad:
            return None
        thermal = [r for r in bad if "thermal" in r or r == "hw_slowdown"]
        if hot and temp_c is not None:
            what = f"{temp_c:.0f} \u00b0C" + (", thermal throttling" if thermal else (", power throttling" if bad else ""))
        else:
            what = "thermal throttling" if thermal else "power throttling"
        return (f"Card {index} is running hot ({what}): expect much lower speed. "
                "Passive Tesla cards need forced airflow.")

    def rig_live(self):
        rows, err = self.gpus()
        throttle = {t["index"]: t["throttle_reasons"] for t in self.gpu_telemetry()}
        tele = {}
        if not os.environ.get("PXA_LAUNCH_FAKE_GPUS"):
            out = self.L._run(["nvidia-smi", "--query-gpu=index,temperature.gpu,power.draw,power.limit,"
                               "utilization.gpu,pcie.link.gen.current,pcie.link.width.current,"
                               "pcie.link.gen.max,pcie.link.width.max,fan.speed",
                               "--format=csv,noheader,nounits"], timeout=15) or ""
            for line in out.splitlines():
                f = [x.strip() for x in line.split(",")]
                if len(f) < 10:
                    continue

                def num(s):
                    try:
                        return float(s)
                    except ValueError:
                        return None
                try:
                    tele[int(f[0])] = {"temp_c": num(f[1]), "power_w": num(f[2]), "power_limit_w": num(f[3]),
                                       "util_pct": num(f[4]), "pcie_gen": num(f[5]), "pcie_width": num(f[6]),
                                       "pcie_gen_max": num(f[7]), "pcie_width_max": num(f[8]),
                                       "fan_pct": num(f[9])}
                except ValueError:
                    continue
        procs, _ok = self.L.resident_procs(rows)
        cards = []
        for g in rows:
            t = tele.get(g[0], {})
            cards.append(dict({"index": g[0], "name": g[1].replace("NVIDIA ", ""), "sm": g[2],
                               "mem_total_mib": g[3], "mem_used_mib": g[4], "uuid": g[5],
                               "class": self.L.CARD_CLASS.get(g[2], f"sm_{g[2]}"),
                               "procs": [{"pid": p, "name": n, "mib": m} for p, n, m in procs.get(g[0], [])],
                               "narrow": bool(t.get("pcie_width") and t["pcie_width"] < self.L.PCIE_NARROW_BELOW),
                               "throttle_reasons": throttle.get(g[0], []),
                               "heat_warning": self.heat_warning(g[0], t.get("temp_c"), throttle.get(g[0], []))},
                              **t))
        return {"cards": cards, "error": err, "ts": time.time()}

    def gpu_live(self):
        """One reading of every card for the Live tab: [{index, name, class, mem_total, mem, util, temp, power, limit,
        clock, throttle}] from ONE nvidia-smi call. A fake rig, or a machine without nvidia-smi, gives the card table's
        memory only (the other numbers are None and the page draws nothing for them)."""
        rows, _err = self.gpus(max_age=30)
        out = {}
        for g in rows:
            out[g[0]] = {"index": g[0], "name": g[1].replace("NVIDIA ", ""), "class": self.L.CARD_CLASS.get(g[2], f"sm_{g[2]}"),
                         "mem_total": g[3], "mem": g[4], "util": None, "temp": None, "power": None, "limit": None,
                         "clock": None, "throttle": []}
        if not out or os.environ.get("PXA_LAUNCH_FAKE_GPUS") or not shutil_which("nvidia-smi"):
            return [out[i] for i in sorted(out)]
        base = "index,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm"
        cands = ["clocks_throttle_reasons.active", "clocks_event_reasons.active", ""] if self.live._thr_field is None \
            else [self.live._thr_field]          # the field was renamed in newer drivers: find the one this driver has
        text = None
        for fld in cands:
            text = self.L._run(["nvidia-smi", "--query-gpu=" + base + ("," + fld if fld else ""),
                                "--format=csv,noheader,nounits"], timeout=8)
            if text:
                self.live._thr_field = fld
                break
        for line in (text or "").splitlines():
            f = [x.strip() for x in line.split(",")]
            try:
                r = out.get(int(f[0]))
            except (ValueError, IndexError):
                continue
            if r is None or len(f) < 7:
                continue
            r["mem"] = _num(f[1]) if _num(f[1]) is not None else r["mem"]
            r["util"], r["temp"], r["power"], r["limit"], r["clock"] = (_num(v) for v in f[2:7])
            if len(f) > 7:
                try:
                    mask = int(f[7], 16) if f[7].lower().startswith("0x") else int(f[7])
                    r["throttle"] = [n for b, n in self.THROTTLE_BITS if mask & b and n in self.THROTTLE_WARN]
                except ValueError:
                    pass
        return [out[i] for i in sorted(out)]

    def rig_static(self, force=False):
        t, d = self._rig_cache
        if d is not None and not force and time.time() - t < 60:
            return d
        drv = (self.L._run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "").strip()
        drv = drv.splitlines()[0] if drv else None
        head = self.L._run(["nvidia-smi"]) or ""
        m = re.search(r"CUDA Version:\s*([0-9.]+)", head)
        ck = self.L.container_kind()
        shm = self.L.shm_bytes()
        E, enote = self.L.resolve_engine_dir()
        d = {"driver": drv, "cuda": m.group(1) if m else None, "container": ck,
             "shm_bytes": shm, "shm_ok": shm >= self.L.SHM_MIN_BYTES,
             "engine_dir": E, "engine_note": enote,
             "engine_ok": bool(E) and "WILL NOT START" not in (enote or ""), "cpu_count": os.cpu_count(),
             "hostname": os.uname().nodename if hasattr(os, "uname") else ""}
        self._rig_cache = (time.time(), d)
        return d

    def doctor(self, force=False):
        t, d = self._doctor_cache
        if d is not None and not force and time.time() - t < 120:
            return d
        rows, err = self.gpus()
        a = self.L.build_parser().parse_args(["--doctor", "--no-sha"])
        with self.plan_lock:
            cap = self.L._Capture().run(self.L.doctor, a, rows, err)
        text = cap.text
        findings = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("!!"):
                findings.append({"level": "bad", "text": s[2:].strip()})
            elif s.startswith("OK:"):
                findings.append({"level": "ok", "text": s[3:].strip()})
            elif "WARNING" in s or "under 1 GiB" in s or "mixed card classes" in s:
                findings.append({"level": "warn", "text": s})
        d = {"text": text, "findings": findings, "exit": cap.value if cap.code == 0 else cap.code}
        self._doctor_cache = (time.time(), d)
        return d

    # ---- models ---------------------------------------------------------------------------
    def model_roots(self):
        c = load_config()
        roots = [os.path.abspath(os.path.expanduser(x)) for x in c["model_dirs"]]
        env = os.environ.get("PXA_MODELS_DIR", "")
        roots += [os.path.abspath(x) for x in env.split(os.pathsep) if x.strip()]
        out = []
        for r in roots:
            if r not in out:
                out.append(r)
        return out

    def set_model_dirs(self, dirs):
        if not isinstance(dirs, list) or len(dirs) > 32:
            raise Invalid("model_dirs must be a list of up to 32 folders")
        clean = []
        for d in dirs:
            if not isinstance(d, str) or not d.strip() or len(d) > 4096 or "\x00" in d:
                raise Invalid(f"not a folder: {d!r}")
            p = os.path.abspath(os.path.expanduser(d.strip()))
            if not os.path.isdir(p):
                raise Invalid(f"not a folder on this machine: {p}")
            if p not in clean:
                clean.append(p)
        with _cfg_lock:
            c = load_config()
            c["model_dirs"] = clean
            save_config(c)
        self._models_cache = None
        return clean

    def _extra_facts(self, path):
        """params and the dominant non-PXQ type, from the launcher's header reader, cached."""
        try:
            st = os.stat(path)
        except OSError:
            return {}
        key = (path, st.st_size, st.st_mtime_ns)
        hit = self._prof_cache.get(key)
        if hit is not None:
            return hit
        h = self.L.gguf_header(path)
        params = sum(t[2] for t in h.get("tensors") or [] if len(t) > 2)
        hist = {}
        for t in h.get("tensors") or []:
            hist[t[1]] = hist.get(t[1], 0) + (t[2] if len(t) > 2 else 1)
        dom = max(hist.items(), key=lambda kv: kv[1])[0] if hist else None
        kvs = h.get("kv") or {}
        name = self.L.PXQ_GGML_TYPE.get(dom) or self.L.NON_PXQ_GGML_TYPE.get(dom) or (
            f"ggml type {dom}" if dom is not None else None)
        quant = {k[len("pxa.quantizer."):]: v for k, v in kvs.items() if k.startswith("pxa.quantizer.")} or None
        d = {"params": params, "size_label": kvs.get("general.size_label"),
             "name": kvs.get("general.name"), "dominant_type": name, "quantizer": quant}
        try:
            tp = TH.detect(arch=kvs.get("general.architecture"), name=kvs.get("general.name"),
                           basename=kvs.get("general.basename"), template=kvs.get("tokenizer.chat_template"),
                           path=path, params=params, size_label=kvs.get("general.size_label"))
            d["thinking"] = {k: tp[k] for k in ("family", "label", "mechanism", "supported", "default", "can_disable",
                                                 "partial_off", "levels", "summary", "detected_by", "confidence",
                                                 "methods", "on_method", "off_method")}
            d["thinking"]["budget"] = tp["budget"]["suggested"]
            d["thinking"]["enforce"] = tp["budget"]["enforce"]
        except Exception as e:      # noqa: BLE001 - a profile table problem must not hide the model list
            d["thinking"] = {"family": "unknown", "supported": False, "summary": f"thinking profile error: {e}"}
        self._prof_cache[key] = d
        return d

    def models(self, force=False):
        if self._models_cache is not None and not force:
            return self._models_cache
        roots = self.model_roots()
        with self.plan_lock:
            entries, notes = self.L.scan_models([r for r in roots if os.path.isdir(r)])
        out = []
        for e in entries:
            x = self._extra_facts(e["path"])
            tier = e.get("tier") or ""
            codec = "PXQN" if tier.startswith("PXQN") else ("PXQ" if tier.startswith("PXQ") else "other")
            notice = ""
            if not e.get("err"):
                notice = self.L.standard_gguf_notice(
                    e["path"], tier=e.get("tier"), tier_kv=e.get("tier_kv"), inspected=True)
            out.append({"path": e["path"], "file": os.path.basename(e["path"]), "size": e.get("size"),
                        "size_h": self.L.human_bytes(e.get("size")), "family": self.L.family_label(e),
                        "arch": e.get("arch"), "tier": tier or None, "codec": codec,
                        "type": tier or x.get("dominant_type"), "params": x.get("params"),
                        "size_label": x.get("size_label"), "quantizer": x.get("quantizer"), "n_ctx_train": e.get("n_ctx_train"),
                        "vision": e.get("vision"), "err": e.get("err"),
                        "kv_bytes_tok": e.get("kv_bytes_tok"), "ple_bytes": e.get("ple_bytes"),
                        "ple": e.get("ple"), "shards_missing": e.get("shards_missing"),
                        "n_expert": e.get("n_expert") or 0, "expert_bytes": e.get("expert_bytes") or 0,
                        "thinking": x.get("thinking"), "notice": notice})
        self._models_cache = {"roots": roots, "models": out, "notes": notes}
        return self._models_cache

    def _quant_notice(self, model):
        """One line when this loaded model is a standard GGUF quant. Empty for a PXA quant
        and when the name alone does not say which kind of file it is."""
        if not isinstance(model, str) or not model.strip():
            return ""
        for e in ((self._models_cache or {}).get("models") or []):
            if e.get("path") == model:
                return e.get("notice") or ""
        return self.L.standard_gguf_notice(model)

    def fits(self, model_entry, card_rows, ctx=0):
        """'fits' | 'tight' | 'ram' | 'no' for this model on these cards, from the launcher's
        fit_verdict (its vram_check and expert-cache route), on IDLE cards (a busy card is shown on
        the Rig tab, not here). 'ram' = bigger than the cards, served by the expert cache with the
        cold experts in system RAM. ctx 0 = one 4096 slot."""
        if not card_rows:
            return {"verdict": "no", "why": "no cards"}
        sel = [(g[0], g[1], g[2], g[3], 0, g[5]) for g in card_rows]
        r = self.L.fit_verdict(sel, model_entry, ctx)
        return {"verdict": r["verdict"], "why": r["why"]}

    def fit_table(self, card_sets, ctx=0):
        rows, _ = self.gpus()
        by = {g[0]: g for g in rows}
        sets = []
        for cs in (card_sets or [])[:8]:
            if not isinstance(cs, list) or not cs:
                continue
            try:
                idx = [int(x) for x in cs][:16]
            except (TypeError, ValueError):
                raise Invalid("card sets must be lists of card indexes")
            sel = [by[i] for i in idx if i in by]
            if sel:
                sets.append(sel)
        res = {}
        for m in self.models()["models"]:
            res[m["path"]] = [dict(self.fits(m, s, ctx), cards=[g[0] for g in s]) for s in sets]
        return res

    # ---- server profiles (several servers, one GUI) -----------------------------------------
    def profiles(self):
        """{sid: {"name", "settings"}}: every server this GUI manages, persisted in control.json.
        'main' always exists (it is the single seat of older versions, and the attach fallback)."""
        c = load_config()
        sv = c.get("servers") if isinstance(c.get("servers"), dict) else {}
        out = {k: v for k, v in sv.items() if isinstance(v, dict) and SID_RE.match(k)}
        if "main" not in out:
            out["main"] = {"name": "Server 1", "settings": c.get("last_launch") or {}}
        for k, v in out.items():
            v.setdefault("name", k)
            v.setdefault("settings", {})
        return out

    def get_seat(self, sid, create=False):
        sid = sid or "main"
        if not isinstance(sid, str) or not SID_RE.match(sid):
            raise Invalid("server id: 1-32 of a-z 0-9 _ - (starting with a letter or digit)")
        with self.seats_lock:
            seat = self.seats.get(sid)
            if seat is None:
                if not create and sid not in self.profiles():
                    raise Invalid(f"no server '{sid}': create it first")
                seat = self.seats[sid] = Seat(self.L, sid)
            return seat

    def _new_sid(self, name):
        base = re.sub(r"[^a-z0-9]+", "-", (name or "server").lower()).strip("-")[:24] or "server"
        if not SID_RE.match(base):
            base = "s-" + base
        taken, sid, n = set(self.profiles()), base, 2
        while sid in taken:
            sid, n = f"{base}-{n}", n + 1
        return sid

    def save_profile(self, sid, name, settings):
        if not isinstance(name, str) or not SERVER_NAME_RE.match(name.strip() or "x" * 99):
            raise Invalid("server name: 1-48 of letters, digits, space and _ . - ( ) # : +")
        name = name.strip()
        settings = dict(settings or {})
        settings.pop("sid", None)
        req = None
        if settings.get("model") or settings.get("gpus"):
            req = self.validate(settings, check_model=False, resolve_port=False)
        with _cfg_lock:
            c = load_config()
            sv = c.get("servers") if isinstance(c.get("servers"), dict) else {}
            if not sid:
                sid = self._new_sid(name)
            if not SID_RE.match(sid):
                raise Invalid("server id: 1-32 of a-z 0-9 _ - (starting with a letter or digit)")
            if sid not in sv and len(sv) >= 32:
                raise Invalid("at most 32 servers")
            if "main" not in sv and sid != "main":
                sv["main"] = {"name": "Server 1", "settings": c.get("last_launch") or {}}
            sv[sid] = {"name": name, "settings": req if req is not None else (sv.get(sid, {}).get("settings") or {})}
            c["servers"] = sv
            save_config(c)
        self.get_seat(sid, create=True)
        return sid, sv[sid]

    def delete_profile(self, sid):
        if sid == "main":
            raise Invalid("the first server cannot be deleted (rename it instead)")
        seat = self.seats.get(sid)
        if seat is not None and seat.running():
            raise Invalid("stop that server first")
        with _cfg_lock:
            c = load_config()
            ok = (c.get("servers") or {}).pop(sid, None) is not None
            save_config(c)
        with self.seats_lock:
            self.seats.pop(sid, None)
        self._fleet_cache = (0.0, None)
        return ok

    def _remember(self, sid, body):
        """the settings a server last started with become its profile (port kept as typed)."""
        try:
            settings = dict(body)
            settings.pop("sid", None)
            req = self.validate(settings, check_model=False, resolve_port=False)
        except Invalid:
            return
        with _cfg_lock:
            c = load_config()
            sv = c.get("servers") if isinstance(c.get("servers"), dict) else {}
            ent = sv.get(sid) or {"name": "Server 1" if sid == "main" else sid}
            ent["settings"] = req
            sv[sid] = ent
            c["servers"] = sv
            if sid == "main":
                c["last_launch"] = req
            save_config(c)

    # ---- thinking (per model: tools/pxa_thinking.py) ------------------------------------------
    def thinking_for_model(self, path, verify=False):
        """the full thinking profile of a model file (header read once, cached by size and mtime)."""
        try:
            st = os.stat(path)
        except OSError:
            return TH.detect()
        key = ("think", path, st.st_size, st.st_mtime_ns)
        hit = self._prof_cache.get(key)
        if hit is None:
            h = self.L.gguf_header(path)
            kv = h.get("kv") or {}
            params = sum(t[2] for t in h.get("tensors") or [] if len(t) > 2) or None
            tmpl = kv.get("tokenizer.chat_template")
            hit = TH.detect(arch=kv.get("general.architecture"), name=kv.get("general.name"),
                            basename=kv.get("general.basename"), template=tmpl, path=path, params=params,
                            size_label=kv.get("general.size_label"))
            hit["_template"] = tmpl
            self._prof_cache[key] = hit
        return self._with_verify(hit, verify)

    def _with_verify(self, prof, verify):
        out = {k: v for k, v in prof.items() if k != "_template"}
        if verify:
            if "_verify" not in prof:
                prof["_verify"] = TH.verify_render(prof.get("_template"), prof) if prof.get("_template") else \
                    {"ok": False, "error": "the model file carries no chat template"}
            out["verify"] = prof["_verify"]
        out.pop("_verify", None)
        return out

    def thinking_for_port(self, port, verify=False):
        """the profile of a server this GUI did not start: its /props (template, model path)."""
        try:
            props = _http_json(port, "/props", timeout=3.0) or {}
        except Exception:       # noqa: BLE001 - an old or busy server: decide from nothing (toggle hidden)
            props = {}
        props = props if isinstance(props, dict) else {}
        tmpl = props.get("chat_template") if isinstance(props.get("chat_template"), str) else None
        mp = props.get("model_path") if isinstance(props.get("model_path"), str) else ""
        if mp.endswith(".gguf") and os.path.isfile(mp):
            prof = self.thinking_for_model(mp, verify=verify)
            if not tmpl or prof.get("template", {}).get("sha1") == TH.template_facts(tmpl)["sha1"]:
                return prof
            # the server renders another template (--chat-template-file): the server's own one wins
        key = ("think-port", port, TH.template_facts(tmpl).get("sha1") if tmpl else None, mp)
        hit = self._prof_cache.get(key)
        if hit is None:
            hit = TH.detect(template=tmpl, path=mp or None)
            hit["_template"] = tmpl
            self._prof_cache[key] = hit
        return self._with_verify(hit, verify)

    def thinking_overrides(self):
        c = load_config()
        t = c.get("thinking_models")
        return t if isinstance(t, dict) else {}

    def set_thinking_override(self, model, settings):
        if not isinstance(model, str) or not model.endswith(".gguf") or len(model) > 4096 or "\x00" in model:
            raise Invalid("thinking override: model must be a .gguf path")
        clean = None if settings is None else clean_thinking(settings)
        with _cfg_lock:
            c = load_config()
            t = c.get("thinking_models") if isinstance(c.get("thinking_models"), dict) else {}
            if clean is None or thinking_is_default(clean):
                t.pop(model, None)
            else:
                if model not in t and len(t) >= 512:
                    raise Invalid("at most 512 per-model thinking settings")
                t[model] = clean
            c["thinking_models"] = t
            save_config(c)
        return clean

    def thinking_settings_for(self, req):
        """what a launch uses: the server's own setting, else the model's remembered one, else auto."""
        t = req.get("thinking") or {}
        if not thinking_is_default(t):
            return t, "server"
        o = self.thinking_overrides().get(req.get("model") or "")
        if o:
            return clean_thinking(o), "model"
        return clean_thinking(None), "auto"

    def thinking_launch(self, req):
        try:
            prof = self.thinking_for_model(req["model"])
            st, src = self.thinking_settings_for(req)
            args, notes = TH.launch_args(prof, st)
        except Exception as e:      # noqa: BLE001 - never block a launch on the thinking profile
            return {"args": [], "notes": [f"profile error ({e.__class__.__name__}: {e}); no thinking flags"],
                    "family": "unknown", "settings": None, "source": "error"}
        if args:
            notes = [f"{prof['label']}: {' '.join(args)} (from the {src} setting)"] + notes
        return {"args": args, "notes": notes, "family": prof["family"], "settings": st, "source": src,
                "supported": prof["supported"]}

    def _seat_on_port(self, port):
        for sid, seat in list(self.seats.items()):
            if seat.req and seat.running() and seat.port() == port:
                return sid, seat
        return None, None

    def thinking_target(self, query, verify=False):
        """-> (profile, settings, source, model) for ?model= | ?sid= | ?port= (or main/attach)."""
        model = _q(query, "model")
        if model:
            if not os.path.isfile(model) or not model_path_ok(model, self.model_roots()):
                raise Invalid("the model must be a file inside one of your model folders")
            o = self.thinking_overrides().get(model)
            return self.thinking_for_model(model, verify), clean_thinking(o), ("model" if o else "auto"), model
        sid = _q(query, "sid")
        seat = self.seats.get(sid) if sid else None
        if seat is not None and seat.req and seat.running():
            st, src = self.thinking_settings_for(seat.req)
            return self.thinking_for_model(seat.req["model"], verify), st, src, seat.req["model"]
        if sid and sid in self.profiles():
            s = self.profiles()[sid].get("settings") or {}
            if s.get("model"):
                st, src = self.thinking_settings_for({"model": s["model"], "thinking": s.get("thinking")})
                return self.thinking_for_model(s["model"], verify), st, src, s["model"]
        port = self.target_port(query)
        if not port:
            raise Invalid("no server running")
        msid, mseat = self._seat_on_port(port)
        if mseat is not None:
            st, src = self.thinking_settings_for(mseat.req)
            return self.thinking_for_model(mseat.req["model"], verify), st, src, mseat.req["model"]
        prof = self.thinking_for_port(port, verify)
        return prof, clean_thinking(None), "auto", None

    def thinking_info(self, query):
        prof, st, src, model = self.thinking_target(query, verify=_q(query, "verify", "1") != "0")
        eff = dict(st)
        eff["budget_effective"] = TH.effective_budget(prof, st)
        return {"profile": prof, "settings": st, "source": src, "model": model, "effective": eff,
                "override": self.thinking_overrides().get(model) if model else None}

    def thinking_rewrite(self, data, query):
        """the engine proxy's hook: a chat body carrying pxa_thinking {mode, budget, level, fallback}
        -> the body this model needs (pxa_thinking removed), and notes for an X-PXA-Thinking header."""
        try:
            body = json.loads(data.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return data, None
        if not isinstance(body, dict):
            return data, None
        if "pxa_thinking" not in body:
            # the admin's effort selector (default off: no effort saved = byte-for-byte passthrough)
            try:
                prof, st, _src, _m = self.thinking_target(query)
            except Exception:       # noqa: BLE001 - never block a chat on the thinking profile
                return data, None
            st = st or {}
            if st.get("effort") and (st.get("lock") or not TH.client_thinking(body)):
                body, notes = TH.apply_effort(body, prof, st)
            elif "reasoning_effort" in body:
                body, notes = TH.map_client_effort(body, prof)
            else:
                return data, None
            return json.dumps(body).encode(), {"family": prof.get("family"), "label": prof.get("label"),
                                               "notes": notes}
        want = body.pop("pxa_thinking")
        try:
            want = want if isinstance(want, dict) else {}
            st = clean_thinking({k: want.get(k) for k in ("mode", "budget", "level")})
            prof, _st, _src, _m = self.thinking_target(query)
            body, notes = TH.apply_request(body, prof, st["mode"], st["budget"], st["level"],
                                           use_fallback=bool(want.get("fallback")))
            info = {"family": prof["family"], "label": prof["label"], "notes": notes}
        except Invalid as e:
            info = {"family": "unknown", "notes": [f"not applied: {e}"]}
        return json.dumps(body).encode(), info

    # ---- conflicts --------------------------------------------------------------------------
    def held_ports(self, except_sid=None):
        """{port: who} for every port another server of this GUI, or a discovered server, holds."""
        out = {}
        for sid, seat in list(self.seats.items()):
            if sid != except_sid and seat.running() and seat.port():
                out[seat.port()] = f"server '{self.profiles().get(sid, {}).get('name', sid)}' (this GUI)"
        for inst in self._fleet_external():
            if inst.get("port") and inst.get("running"):
                out.setdefault(inst["port"], f"{inst['kind']} {inst['name']}")
        return out

    def conflicts(self, req, sid):
        """{"errors": [...], "warnings": [...]}: port and card clashes with the other servers of this
        GUI, the containers/processes it discovered, and anything else listening, decided BEFORE a
        process is spawned (a clash used to surface as an engine PORT_GUARD or an OOM)."""
        errs, warns = [], []
        port = req["port"]
        held = self.held_ports(except_sid=sid)
        mine = self.seats.get(sid)
        mine_running = bool(mine and mine.running())
        if port in held:
            errs.append(f"port {port} is used by {held[port]}: pick another server port (or leave it on auto)")
        elif not (mine_running and mine.port() == port) and (
                port_in_use(port, "0.0.0.0" if req["expose"] else "127.0.0.1") or port_in_use(port)):
            errs.append(f"port {port} is already in use by another program: pick another server port")
        owners = {}
        for osid, seat in list(self.seats.items()):
            if osid != sid and seat.running() and seat.req:
                for g in seat.req.get("gpus") or []:
                    owners.setdefault(g, []).append(f"server '{self.profiles().get(osid, {}).get('name', osid)}'"
                                                    f" (:{seat.port()})")
        snap = self._fleet_cache[1]
        for inst in self._fleet_external():
            if not inst.get("running"):
                continue
            cards = set(inst.get("gpus") or []) | {int(k) for k, v in (inst.get("vram") or {}).items() if v}
            for g in cards:
                owners.setdefault(g, []).append(f"{inst['kind']} {inst['name']}" + (f" (:{inst['port']})" if inst.get("port") else ""))
        for g in req["gpus"]:
            if g in owners:
                msg = f"card {g} is in use by {', '.join(owners[g])}"
                if req["allow_busy"]:
                    warns.append(msg + " (allowed: 'Allow busy cards' is on; expect an out-of-memory)")
                else:
                    errs.append(msg + ": choose other cards, stop that server, or tick 'Allow busy cards'")
        if snap:
            for c in snap.get("cards") or []:
                if c["index"] in req["gpus"] and c.get("other_mib"):
                    warns.append(f"card {c['index']} has {c['other_mib']} MiB held by other programs "
                                 f"({', '.join(c.get('other_names') or [])})")
        return {"errors": errs, "warnings": warns}

    def fit_now(self, req):
        """the launcher's VRAM check for this model on these cards as they are NOW (memory other
        programs hold subtracted), next to the idle-card answer the Models table shows."""
        m = next((x for x in self.models()["models"] if x["path"] == req["model"]), None)
        if m is None:
            try:
                m = {"size": os.path.getsize(req["model"])}
            except OSError:
                return None
        rows, _ = self.gpus()
        sel = [g for g in rows if g[0] in req["gpus"]]
        if not sel:
            return None
        ctx = req["ctx"] or 0
        idle = self.fits(m, sel, ctx)
        r = self.L.fit_verdict(list(sel), m, ctx)
        notes = r["notes"]
        if r["verdict"] == "no":
            now = {"verdict": "no", "why": r["why"]}
        elif r["verdict"] == "ram":
            # the expert-cache route: what decides it right now is the RAM free, not the VRAM
            now = {"verdict": "tight" if r["busy"] else "ram", "why": r["why"]}
        else:
            why = " ".join(n for n in notes if n.startswith("VRAM"))
            now = {"verdict": "tight" if any("TIGHT FIT" in n for n in notes) else "fits", "why": why}
            # the launcher's check never blocks; when its own numbers say the estimate exceeds what is
            # free right now, say so plainly instead of "tight" (13.65 GiB needed vs 1.44 GiB free)
            mm = re.search(r"= ([\d.]+) GiB vs ([\d.]+) GiB free", why)
            if mm and float(mm.group(1)) > float(mm.group(2)):
                now["verdict"] = "no"
                now["why"] = (f"needs about {mm.group(1)} GiB but only {mm.group(2)} GiB is free on these cards "
                              "right now (another server holds the rest). " + why)
        free = sum(max(0, g[3] - g[4]) for g in sel)
        return {"idle": idle, "now": now, "model_gib": round((m.get("size") or 0) / 2 ** 30, 2),
                "free_gib_now": round(free / 1024.0, 2), "total_gib": round(sum(g[3] for g in sel) / 1024.0, 2)}

    # ---- plan / start / stop --------------------------------------------------------------
    def _build(self, req, explain):
        rows, _err = self.gpus(max_age=0.5)
        a = self.L.build_parser().parse_args(launcher_argv(req))
        a.explain = explain
        envset = dict(req["levers"])
        if req.get("engine_dir"):
            envset["PXA_ENGINE_DIR"] = req["engine_dir"]      # this profile's build, for the planner
        saved = {k: os.environ.get(k) for k in envset}
        with self.plan_lock:
            try:
                os.environ.update(envset)       # the planner reads PXA_* from the environment
                cap = self.L._Capture().run(self.L.plan_and_build, a, rows)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        cap.args = a
        cap.thinking = self.thinking_launch(req)
        tail = list(cap.thinking["args"]) + list(req.get("extra_args") or [])
        if tail:
            cap.text = annotate_plan_text(cap.text, _flag_pairs(cap.thinking["args"], "PXA Control (thinking)")
                                          + _flag_pairs(req.get("extra_args") or [], "YOURS (extra args)"))
        if cap.code == 0 and cap.value is not None and tail:
            plan, cmd, env, cv, prof, ctx = cap.value
            cap.value = (plan, list(cmd) + tail, env, cv, prof, ctx)
        return cap

    def validate(self, body, check_model=True, resolve_port=True, sid=None):
        rows, _ = self.gpus()
        avoid = set(self.held_ports(except_sid=sid)) if resolve_port else set()
        return validate_launch(body, {g[0] for g in rows}, self.catalog_names, self.model_roots(),
                               gui_port=self.port, check_model=check_model, resolve_port=resolve_port,
                               avoid_ports=avoid)

    def _plan_dict(self, req, cap, sid, conflicts=None):
        ok = cap.code == 0 and cap.value is not None
        d = {"ok": ok, "exit": cap.code, "text": cap.text, "cli": cli_line(req), "sid": sid,
             "lever_warnings": lint_levers(req["levers"], self.catalog)}
        if ok:
            plan, cmd, env, cv, prof, ctx = cap.value
            full_env = dict(env, **self.L.device_env(cv)[0], **req["levers"])
            d.update({"engine": plan.engine, "command": " ".join(self.L.redact_cmd(cmd)),
                      "shell": shell_line(full_env, self.L.redact_cmd(cmd)),
                      "env": dict(env, **req["levers"]), "cards": cv, "ctx": ctx,
                      "ctx_auto": not req.get("ctx"), "port": req["port"],
                      "notes": list(plan.notes), "blockers": list(plan.blockers),
                      "sm": cap.args.sm, "np": cap.args.np, "workload": cap.args.workload,
                      "extra_args": req.get("extra_args") or []})
        th = getattr(cap, "thinking", None)
        if th:
            d["thinking"] = th
            if ok:
                d["notes"] = d["notes"] + ["thinking: " + n for n in th["notes"]]
        if conflicts is not None:
            d["conflicts"] = conflicts
            if ok:
                d["blockers"] = d["blockers"] + conflicts["errors"]
                d["notes"] = d["notes"] + conflicts["warnings"]
        try:
            d["fit"] = self.fit_now(req)
        except Exception as e:      # noqa: BLE001 - the fit is advice, never a reason to fail the plan
            d["fit"] = {"error": f"{e.__class__.__name__}: {e}"}
        return d

    def plan(self, body):
        body = dict(body or {}) if isinstance(body, dict) else body
        sid = body.pop("sid", None) if isinstance(body, dict) else None
        sid = sid or "main"
        self.get_seat(sid, create=(sid == "main"))
        # a recent picture of the box BEFORE the port is picked and conflicts are checked: right
        # after the GUI starts the cache is empty, and the first plan saw no other server at all
        self.fleet(max_age=5)
        req = self.validate(body, sid=sid)
        cap = self._build(req, explain=True)
        return self._plan_dict(req, cap, sid, self.conflicts(req, sid))

    def start(self, body):
        body = dict(body or {}) if isinstance(body, dict) else body
        sid = (body.pop("sid", None) if isinstance(body, dict) else None) or "main"
        seat = self.get_seat(sid, create=(sid == "main"))
        if seat.running():
            raise Invalid("this server is already running; stop it first")
        self._fleet_cache = (0.0, None)             # decide on a fresh picture of the box
        self.fleet()                                # (before the auto port is chosen, too)
        req = self.validate(body, sid=sid)
        cf = self.conflicts(req, sid)
        if cf["errors"]:
            raise Invalid("; ".join(cf["errors"]))
        if self.gpuctl is not None and self.gpuctl.guard.locks():   # a reserved card, a race lock file, maintenance: 423
            self.gpuctl.check_launch(self.cards_by_index(req["gpus"]), f"start server '{sid}'")
        cap = self._build(req, explain=False)
        if cap.code != 0 or cap.value is None:
            return {"ok": False, "exit": cap.code, "text": cap.text, "sid": sid}
        cmd = cap.value[1]
        if not self.L.shutil.which(cmd[0]) and not os.path.exists(cmd[0]):
            return {"ok": False, "exit": 4, "sid": sid, "text": cap.text + f"\n{cmd[0]} not found: set this "
                    "server's engine build (Launch > Advanced) or PXA_ENGINE_DIR to the dir holding bin/llama-server"}
        seat.start(cap.value, req, req["levers"])
        st = self.L._load_state()
        st["last_command"] = self.L.redact_cmd(cmd)
        st["last_cards"] = cap.value[3]
        self.L._save_state(st)
        self._remember(sid, body)
        self._fleet_cache = (0.0, None)
        # the plan as it was when it started: planning again now would see the new seat on the cards
        # and refuse them as busy (R-20), which is what the page used to show after every Start
        started = self._plan_dict(req, cap, sid, cf)
        return {"ok": True, "text": cap.text, "plan": started, "status": self.status(sid), "sid": sid}

    def restart(self, body=None, sid=None):
        if isinstance(body, dict) and body.get("sid"):
            sid = body["sid"]
        sid = sid or "main"
        seat = self.get_seat(sid, create=(sid == "main"))
        b = {k: v for k, v in (body or {}).items() if k != "sid"} if isinstance(body, dict) else {}
        if not b.get("model"):
            b = dict(seat.req) if seat.req else dict(self.profiles().get(sid, {}).get("settings") or {})
        if not b.get("model"):
            raise Invalid("nothing to restart")
        seat.stop()
        if seat.running():
            raise Invalid(seat.phase)
        b["sid"] = sid
        return self.start(b)

    def stop(self, sid=None):
        seat = self.get_seat(sid or "main", create=(not sid or sid == "main"))
        ok = seat.stop()
        self._fleet_cache = (0.0, None)
        return ok

    def stop_all(self):
        for seat in list(self.seats.values()):
            if seat.running():
                seat.stop()

    def engine_port(self, sid=None):
        seat = self.seats.get(sid or "main")
        if seat is not None and seat.req and seat.running():
            return seat.req["port"]
        if sid and sid != "main":
            return None
        p = int(load_config().get("attach_port") or 0)
        return p or None

    def status(self, sid=None):
        sid = sid or "main"
        seat = self.get_seat(sid, create=(sid == "main"))
        d = seat.status()
        d["name"] = self.profiles().get(sid, {}).get("name", sid)
        port = self.engine_port(sid)
        d["engine_port"] = port
        d["attached"] = bool(port and not seat.running())
        if port:
            h, detail = probe_health(port, timeout=2)
            d["health"] = h
            d["health_detail"] = detail
            if h == "ok" and d["running"]:
                d["phase"] = "serving"
        try:
            d["context"] = self.server_context(seat=seat, healthy=d.get("health") == "ok") if d["running"] else None
        except Exception:            # noqa: BLE001
            d["context"] = None
        return d

    # ---- the fleet dashboard ----------------------------------------------------------------
    def adopted(self):
        a = load_config().get("adopted")
        return {k: v for k, v in (a or {}).items() if isinstance(v, dict)} if isinstance(a, dict) else {}

    def _fleet_external(self):
        snap = self._fleet_cache[1]
        if snap is None:
            return []
        return [x for x in snap["instances"] if x["kind"] != "managed"]

    def fleet(self, max_age=2.5):
        """Every llama-server on the box: this GUI's servers, docker containers and bare processes,
        each with health, t/s, and VRAM per card (nvidia-smi's compute apps mapped to its process
        tree). Cached for max_age seconds; one scan at a time."""
        sig = launched_signature()           # a server the launcher just started shows at once, not at the next scan
        t, v = self._fleet_cache
        if v is not None and time.time() - t < max_age and sig == self._launched_seen:
            return v
        with self._fleet_lock:
            t, v = self._fleet_cache
            if v is not None and time.time() - t < max_age and sig == self._launched_seen:
                return v
            v = self._fleet_scan()
            self._fleet_cache = (time.time(), v)
            self._launched_seen = sig
            return v

    def _fleet_scan(self):
        rows, gerr = self.gpus()
        table = proc_table()
        apps = compute_apps(rows)
        adopted = self.adopted()
        errors = [gerr] if gerr else []
        profiles = self.profiles()
        insts = []
        managed_pids = set()
        for sid in sorted(profiles, key=lambda k: (k != "main", profiles[k].get("name", k).lower())):
            seat = self.seats.get(sid) or Seat(self.L, sid)
            st = seat.status()
            prof = profiles[sid].get("settings") or {}
            pid = st["pid"] if st["running"] else None
            if pid:
                managed_pids |= descendants(pid, table)
            # a seat Control did not start itself can still be ATTACHED to a server the launcher started (attach_port):
            # status() and the header call that serving, so the pickers must not label it "stopped"
            aport = None if st["running"] else self.engine_port(sid)
            insts.append({"key": "m:" + sid, "kind": "managed", "sid": sid, "name": profiles[sid].get("name", sid),
                          "running": st["running"], "phase": st["phase"], "pid": pid, "attached": bool(aport),
                          "port": st["port"] if st["running"] else (aport or prof.get("port") or None),
                          "host": ("0.0.0.0" if (seat.req or prof).get("expose") else "127.0.0.1"),
                          "gpus": st["gpus"] if st["running"] else prof.get("gpus"),
                          "model": st["model"] if st["running"] else prof.get("model"),
                          "started_epoch": st["started"] if st["running"] else None,
                          "exit_code": st["exit_code"], "log_seq": st["log_seq"],
                          "last_line_age": st.get("last_line_age"), "control": True, "adopted": False})
        # PXA_CONTROL_DISCOVER=0: list only this GUI's own servers (tests; a box where the scan is unwanted)
        discover = os.environ.get("PXA_CONTROL_DISCOVER", "1") != "0"
        dock, derr = (scan_docker(rows, extra_names=[k[2:] for k in adopted if k.startswith("d:")])
                      if discover else ([], None))
        if derr and derr != "docker CLI not found":
            errors.append(derr)
        for x in dock:
            a = adopted.get(x["key"])
            x.update({"adopted": bool(a), "control": bool(a and a.get("control")),
                      "label": (a or {}).get("name") or x["name"]})
            insts.append(x)
        dock_pids = set()
        for x in dock:
            if x.get("pid"):
                dock_pids |= descendants(x["pid"], table)
        # servers the launcher started for this user (pxa-launch, run-server.sh): adopted with Stop, listed
        # even when discovery is off or the process is not a bare llama-server (a vLLM `docker run`)
        launched = {pid: r for pid, r in launched_servers().items() if pid not in managed_pids | dock_pids}
        for x in (scan_processes(rows, table, skip_pids=managed_pids | dock_pids) if discover else []):
            r = launched.pop(x["pid"], None)
            if r:
                self._mark_launched(x, r)
            else:
                x.update({"adopted": False, "control": False, "label": x["name"]})
            insts.append(x)
        for pid, r in sorted(launched.items()):
            x = process_entry(pid, (table.get(pid) or (0, []))[1], rows)
            self._mark_launched(x, r)
            insts.append(x)
        # VRAM per card per instance, and what nobody here owns
        claimed = set()
        for x in insts:
            pids = descendants(x["pid"], table) if x.get("pid") else set()
            vram = {}
            for pid, g, mib in apps:
                if pid in pids:
                    vram[g] = vram.get(g, 0) + mib
                    claimed.add((pid, g))
            x["vram"] = vram
            x["vram_mib"] = sum(vram.values())
            argv = x.pop("argv", None)
            if argv:
                x["cmd"] = " ".join(self.L.redact_cmd(argv))
            x["model_file"] = os.path.basename(x["model"]) if isinstance(x.get("model"), str) else None
            x["quant_notice"] = self._quant_notice(x.get("model"))
        names = {}
        for pid, _g, _m in apps:
            argv = (table.get(pid) or (0, []))[1]
            names[pid] = os.path.basename(argv[0]) if argv else "?"
        cards = []
        for g in rows:
            # a server holds a card when it has memory there OR was started on it (still loading,
            # or no per-process memory readout on this driver)
            users = [{"key": x["key"], "sid": x.get("sid"), "name": x.get("label") or x["name"],
                      "mib": x["vram"].get(g[0], 0)}
                     for x in insts if x["vram"].get(g[0]) or (x.get("running") and g[0] in (x.get("gpus") or []))]
            other = [(pid, mib) for pid, gg, mib in apps if gg == g[0] and (pid, gg) not in claimed]
            cards.append({"index": g[0], "name": g[1].replace("NVIDIA ", ""), "mem_total_mib": g[3],
                          "mem_used_mib": g[4], "users": users, "other_mib": sum(m for _p, m in other),
                          "other_names": sorted({f"{names.get(p, '?')} pid {p}" for p, _m in other})[:6]})
        # health + metrics, in parallel (a dead port costs its timeout once, not per instance)
        live = [x for x in insts if x.get("running") and x.get("port")]

        def probe(x):
            h, detail = probe_health(x["port"])
            x["health"], x["slots_idle"], x["slots_processing"] = h, detail.get("slots_idle"), detail.get("slots_processing")
            x["metrics"] = probe_metrics(x["port"]) if h == "ok" else {}
            x["models"] = probe_registered_models(x["port"]) if h == "ok" else None
            try:
                x["context"] = self.server_context(x, healthy=(h == "ok"))
            except Exception:        # noqa: BLE001 - the context line is information, never a reason to fail the scan
                x["context"] = None
            if h == "ok" and x["kind"] == "managed":
                x["phase"] = "serving"          # same word as /api/status uses
        ths = [threading.Thread(target=probe, args=(x,), daemon=True) for x in live]
        for th in ths:
            th.start()
        for th in ths:
            th.join(6)
        for x in insts:
            x.setdefault("health", None if not x.get("running") else "down")
            x.setdefault("metrics", {})
            x["actions"] = self._actions(x)
            x["slot"] = self.live.slot_for(x["key"])        # one colour per server, here and on the Live tab
        return {"instances": insts, "cards": cards, "errors": errors, "ts": time.time(),
                "docker": docker_bin() is not None}

    @staticmethod
    def _mark_launched(x, r):
        """a fleet instance that is a server the launcher started: adopted, with Stop. Its argv says
        the port when it is llama-server; otherwise (still exec'ing, or a vLLM `docker run`) the
        launcher's record does."""
        argv = x.get("argv") or []
        if not _is_llama_server(argv):
            x["port"] = r.get("port") or x.get("port")
            x["model"] = r.get("model") or x.get("model")
        if not x.get("gpus") and r.get("gpus"):
            x["gpus"] = r["gpus"]
        by = "run-server.sh" if r.get("by") == "run-server.sh" else "pxa"
        x.update({"adopted": True, "control": True, "origin": "launcher", "launched_by": by,
                  "label": f"started with {by} (:{x.get('port')})", "started_epoch": r.get("since") or x.get("started_epoch")})

    @staticmethod
    def _actions(x):
        if x["kind"] == "managed":
            return ["stop", "restart", "log"] if x["running"] else ["start", "log", "edit"] + ([] if x.get("sid") == "main" else ["delete"])
        if x.get("origin") == "launcher":
            return ["log", "stop"] if x.get("running") else ["log"]
        acts = ["log"] if x["kind"] == "docker" or x.get("pid") else []
        if x["kind"] == "docker":
            acts.append("unadopt" if x.get("adopted") else "adopt")
            if x.get("control"):
                acts += ["stop", "restart"] if x["running"] else ["start"]
        return acts

    def _instance(self, key):
        for x in self.fleet(max_age=5)["instances"]:
            if x["key"] == key:
                return x
        raise Invalid(f"no such server: {key}")

    def adopt(self, key, name=None, control=False):
        if not isinstance(key, str) or not key.startswith("d:") or not DOCKER_NAME_RE.match(key[2:]):
            raise Invalid("only a docker container can be adopted (a bare process has no stable name)")
        if name not in (None, "") and (not isinstance(name, str) or not SERVER_NAME_RE.match(name)):
            raise Invalid("name: 1-48 of letters, digits, space and _ . - ( ) # : +")
        self._instance(key)
        with _cfg_lock:
            c = load_config()
            a = c.get("adopted") if isinstance(c.get("adopted"), dict) else {}
            a[key] = {"name": name or key[2:], "control": bool(control)}
            c["adopted"] = a
            save_config(c)
        self._fleet_cache = (0.0, None)
        return a[key]

    def unadopt(self, key):
        with _cfg_lock:
            c = load_config()
            ok = (c.get("adopted") or {}).pop(key, None) is not None
            save_config(c)
        self._fleet_cache = (0.0, None)
        return ok

    def external_control(self, key, action, confirm):
        """stop / start / restart a container this GUI did not start. Refused unless it was adopted
        with control on AND the request repeats the container's name (a mis-tap on a phone must not
        take a seat down)."""
        if action not in ("stop", "start", "restart"):
            raise Invalid("action must be stop, start or restart")
        x = self._instance(key)
        if x.get("origin") == "launcher":
            if action != "stop":
                raise Invalid("a server started from the command line can only be stopped from here; "
                              "run the same command again to start it")
            if confirm != "stop":
                raise Invalid("type stop to confirm")
            return self.stop_launched(x["pid"])
        if x["kind"] != "docker":
            raise Invalid("only an adopted docker container can be controlled from here")
        if not (x.get("adopted") and x.get("control")):
            raise Invalid(f"{x['name']} is monitored read-only: adopt it and switch 'allow control' on first")
        if confirm != x["container"]:
            raise Invalid(f"type the container name ({x['container']}) to confirm")
        r = subprocess.run([docker_bin(), action, x["container"]], capture_output=True, text=True, timeout=120)
        self._fleet_cache = (0.0, None)
        if r.returncode != 0:
            raise Invalid(f"docker {action} {x['container']} failed: {(r.stderr or r.stdout).strip()[:300]}")
        return {"ok": True, "action": action, "container": x["container"]}

    def stop_launched(self, pid, grace_s=30.0):
        """SIGTERM a server the launcher started (the identity is re-checked against its record first:
        pid AND start time), then SIGKILL after grace_s. The server's own Ctrl-C path, from a page."""
        import signal
        r = launched_servers().get(int(pid))
        if not r:
            raise Invalid(f"pid {pid} is not a running server the launcher started")
        try:
            os.kill(r["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            raise Invalid(f"pid {pid} belongs to another user: stop it there")
        killed = False
        deadline = time.time() + grace_s
        while pid_alive(r["pid"], r.get("start")) and time.time() < deadline:
            time.sleep(0.2)
        if pid_alive(r["pid"], r.get("start")):
            try:
                os.kill(r["pid"], signal.SIGKILL)
                killed = True
            except OSError:
                pass
        forget_launched(r["pid"])
        self.last_busy = time.time()            # the idle clock of a background Control starts now
        self._fleet_cache = (0.0, None)
        return {"ok": True, "action": "stop", "pid": r["pid"], "killed": killed}

    def idle_seconds(self, now=None):
        """0 while a server runs (one this Control started, or one the launcher started for this user),
        else seconds since the last server stopped or the last request a page made, whichever is later."""
        now = time.time() if now is None else now
        busy = any(s.running() for s in list(self.seats.values())) or bool(launched_servers())
        if busy:
            self.last_busy = now
            return 0.0
        return max(0.0, now - max(self.last_busy, self.last_request))

    def external_log(self, key, tail=500, q=""):
        """the last lines of a server this GUI did not start: docker logs, or the file a bare
        process writes its stdout to. Read-only."""
        tail = max(1, min(int(tail or 500), 5000))
        x = self._instance(key)
        lines, src = [], None
        if x["kind"] == "docker":
            r = subprocess.run([docker_bin(), "logs", "--tail", str(tail), x["container"]],
                               capture_output=True, text=True, timeout=20, errors="replace")
            lines = ((r.stdout or "") + (r.stderr or "")).splitlines()
            src = f"docker logs --tail {tail} {x['container']}"
        elif x.get("pid"):
            try:
                path = os.readlink(f"/proc/{x['pid']}/fd/1")
            except OSError:
                path = None
            if path and os.path.isfile(path):
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 2_000_000))
                    lines = f.read().decode("utf-8", "replace").splitlines()[-tail:]
                src = path
            else:
                src = "this process writes to a terminal or pipe, not a file: no log to show"
        if q:
            ql = q.lower()
            lines = [ln for ln in lines if ql in ln.lower()]
        lines = [ln for ln in lines if not HEALTH_NOISE_RE.search(ln)]
        return {"key": key, "src": src, "lines": lines[-tail:]}

    def known_ports(self):
        ports = {s.port() for s in self.seats.values() if s.running() and s.port()}
        ports |= {x["port"] for x in self._fleet_external() if x.get("port")}
        ap = int(load_config().get("attach_port") or 0)
        if ap:
            ports.add(ap)
        return ports

    def target_port(self, query):
        """the engine port a proxied request goes to: ?sid= (a server of this GUI), ?port= (only a
        port of a server this GUI knows: never an arbitrary loopback service), else main/attach."""
        sid = _q(query, "sid")
        if sid:
            return self.engine_port(sid)
        p = _q(query, "port")
        if p:
            try:
                p = int(p)
            except ValueError:
                raise Invalid("port must be a number")
            if p not in self.known_ports():
                self.fleet(max_age=0)
                if p not in self.known_ports():
                    raise Invalid(f"port {p} is not a server this GUI knows")
            return p
        return self.engine_port()

    def engines(self, force=False):
        """the llama-server builds this GUI can see (for a per-server engine choice)."""
        t, v = self._engines_cache
        if v is not None and not force and time.time() - t < 300:
            return v
        cands = []
        for d in list(getattr(self.L, "ENGINE_DIR_CANDIDATES", [])) + (
                self.L.find_source_builds() if hasattr(self.L, "find_source_builds") else []):
            if d and os.path.isfile(os.path.join(d, "bin", "llama-server")):
                d = os.path.abspath(d)
                if d not in cands:
                    cands.append(d)
        out = []
        for d in cands[:16]:
            ok, why = bounded(lambda d=d: self.L.engine_runs(d), 30, (False, "timed out"))
            out.append({"dir": d, "runs": ok, "note": why,
                        "mtime": os.path.getmtime(os.path.join(d, "bin", "llama-server"))})
        E, note = bounded(self.L.resolve_engine_dir, 60, (None, "timed out"))
        v = {"builds": out, "default": E, "default_note": note, "env": os.environ.get("PXA_ENGINE_DIR")}
        self._engines_cache = (time.time(), v)
        return v

    # ---- report a problem -----------------------------------------------------------------
    THROTTLE_BITS = ((0x1, "gpu_idle"), (0x2, "applications_clocks_setting"), (0x4, "sw_power_cap"),
                     (0x8, "hw_slowdown"), (0x10, "sync_boost"), (0x20, "sw_thermal_slowdown"),
                     (0x40, "hw_thermal_slowdown"), (0x80, "hw_power_brake_slowdown"))

    def gpu_telemetry(self):
        """Per-GPU temperature, SM/memory clocks, power draw/limit and throttle reasons, read now."""
        if os.environ.get("PXA_LAUNCH_FAKE_GPUS"):
            return []
        base = "index,temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,power.limit,"
        out = None
        for fld in ("clocks_throttle_reasons.active", "clocks_event_reasons.active"):   # renamed in newer drivers
            out = self.L._run(["nvidia-smi", "--query-gpu=" + base + fld, "--format=csv,noheader,nounits"], timeout=15)
            if out:
                break
        rows = []

        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
        for line in (out or "").splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 9:
                continue
            try:
                idx = int(f[0])
            except ValueError:
                continue
            try:
                mask = int(f[8], 16) if f[8].lower().startswith("0x") else int(f[8])
            except ValueError:
                mask = None
            reasons = None if mask is None else [n for b, n in self.THROTTLE_BITS if mask & b and n != "gpu_idle"]
            rows.append({"index": idx, "temp_c": num(f[1]), "sm_clock_mhz": num(f[2]), "sm_clock_max_mhz": num(f[3]),
                         "mem_clock_mhz": num(f[4]), "mem_clock_max_mhz": num(f[5]), "power_w": num(f[6]),
                         "power_limit_w": num(f[7]), "throttle_mask": f[8],
                         "throttle_reasons": reasons if reasons is not None else [f[8]]})
        return rows

    REPORT_BUDGET_S = {"gpus": 8, "rig": 5, "telemetry": 8, "engine": 6}     # per slow source, seconds

    @staticmethod
    def parse_engine_version(text):
        """'4711 (abc1234)' from the engine's 'build: ...' / 'version: ...' lines, or None."""
        for ln in (text or "").splitlines():
            m = re.search(r"^\s*(?:build|version)\s*[:=]\s*(\S.*)", ln, re.I)
            if m:
                return m.group(1).strip()[:80]
        return None

    def engine_version_probe(self, sid="main"):
        """the version of the llama-server binary a seat runs (or the detected engine), from `--version`;
        cached per binary and mtime. For reports whose log no longer holds the build line and whose server is
        down or still loading: 9 of the 17 Discord reports read on 2026-10-06 said 'engine: unknown'."""
        seat = self.seats.get(sid) or self.seat
        exe = None
        if seat.cmd and isinstance(seat.cmd[0], str) and os.path.isfile(seat.cmd[0]):
            exe = seat.cmd[0]
        else:
            st = self._rig_cache[1] or {}
            E = st.get("engine_dir")
            if E and os.path.isfile(f"{E}/bin/llama-server"):
                exe = f"{E}/bin/llama-server"
        if not exe:
            return None
        try:
            key = (os.path.realpath(exe), os.path.getmtime(exe))
        except OSError:
            return None
        cache = self.__dict__.setdefault("_engine_ver_cache", {})
        if key in cache:
            return cache[key]
        env = dict(os.environ)
        try:
            E = os.path.dirname(os.path.dirname(os.path.realpath(exe)))
            env["LD_LIBRARY_PATH"], _ = self.L.engine_ld_path(E)
        except Exception:
            pass
        try:
            r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10, env=env,
                               stdin=subprocess.DEVNULL, errors="replace")
            ver = self.parse_engine_version((r.stdout or "") + "\n" + (r.stderr or ""))
        except Exception:
            ver = None
        if ver:
            cache[key] = ver
        return ver

    def report_bundle(self, sid=None):
        """The whole report, already redacted, exactly as the page will show it (and send it).
        Every slow source has a time budget: the report must come back while the server is stuck
        loading or the driver hangs nvidia-smi (shibi had to reboot to send one, 2026-09-30)."""
        sid = sid if (sid and sid in self.seats) else "main"
        seat = self.seats[sid]
        notes = []
        with seat.cond:
            log = [x[1] for x in seat.log]
        banner = [ln for ln in log if BANNER_RE.search(ln)][:40]
        build = list(seat.build_lines[:6]) or [ln for ln in log if BUILD_RE.match(ln.strip())][:6]
        version = "unknown"
        for ln in build:
            m = re.search(r"(?:build|version)\s*[:=]\s*(\S.*)", ln, re.I)
            if m:
                version = m.group(1).strip()[:80]
                break
        if version == "unknown":
            try:
                port = self.engine_port(sid)
                if port:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=2) as r:
                        bi = (json.loads(r.read().decode("utf-8", "replace")) or {}).get("build_info")
                        if bi:
                            version = str(bi)[:80]
            except Exception:
                pass
        if version == "unknown":        # the build line left the log ring and the server is down or still loading
            v = bounded(lambda: self.engine_version_probe(sid), self.REPORT_BUDGET_S.get("engine", 6), None)
            if v:
                version = v + " (from --version)"
        rows, _err = bounded(self.gpus, self.REPORT_BUDGET_S["gpus"], ([], "timed out"))
        if not rows and _err == "timed out":
            notes.append(f"nvidia-smi did not answer within {self.REPORT_BUDGET_S['gpus']} s (driver busy or hung)")
        t, st = self._rig_cache
        if st is None:      # never block the report on a fresh engine probe (it runs llama-server)
            st = bounded(self.rig_static, self.REPORT_BUDGET_S["rig"], None)
            if st is None:
                st = {}
                notes.append(f"rig facts (driver, CUDA, engine) did not come back within {self.REPORT_BUDGET_S['rig']} s")
        gpus = [{"index": g[0], "name": g[1].replace("NVIDIA ", ""), "cc": g[2], "vram_gb": round(g[3] / 1024.0, 1),
                 "driver": st.get("driver")} for g in rows]
        req = seat.req or {}
        mpath = req.get("model") if isinstance(req.get("model"), str) else None
        msize = None
        try:
            msize = os.path.getsize(mpath) if mpath else None
        except OSError:
            pass
        bench = read_jsonl("bench.jsonl", limit=3)[-3:]
        tele = bounded(self.gpu_telemetry, self.REPORT_BUDGET_S["telemetry"], None)
        if tele is None:
            tele = []
            notes.append(f"GPU telemetry timed out ({self.REPORT_BUDGET_S['telemetry']} s)")
        others = []
        for osid, s2 in list(self.seats.items()):
            if osid != sid and (s2.running() or s2.proc):
                st2 = s2.status()
                others.append({"sid": osid, "phase": st2["phase"], "gpus": st2["gpus"], "exit_code": st2["exit_code"],
                               "model": os.path.basename(st2["model"]) if st2["model"] else None})
        age = (time.time() - seat.last_line_ts) if (seat.running() and seat.last_line_ts) else None
        bundle = {
            "engine": {"version": version, "build_lines": build, "cuda": st.get("cuda"), "container": st.get("container")},
            "banner": banner or ["(no PXA_REGISTRY / PXA_TSPLIT / PXA_AUTO lines in this seat's log yet; start a server first)"],
            "flags": " ".join(self.L.redact_cmd(seat.cmd)) if seat.cmd else "",
            "request": {k: v for k, v in req.items() if k != "model"},
            "gpus": gpus,
            "gpu_telemetry": tele,
            "model": {"filename": os.path.basename(mpath) if mpath else None, "size_bytes": msize},
            "phase": seat.phase,
            "running_s": round(time.time() - seat.started) if (seat.running() and seat.started) else None,
            "silent_s": round(age) if age is not None else None,
            "exit_code": seat.proc.returncode if (seat.proc and not seat.running()) else None,
            "log_tail": log[-300:],
            "benchmarks": bench,
            "other_servers": others,
            "collection_notes": notes,
            "heat_24h": self._heat_24h(),
        }
        return {"version": CONTROL_VERSION, "app": "pxa-control", "bundle": redact_obj(bundle), "contact": ""}

    def _heat_24h(self):
        """Max temperature, minutes at or above 80 C, and throttle-reason counts, last 24 h.

        From the history store (the same samples the Live tab graphs). Empty when the store is off
        or has not been written yet. Bounded so a stuck disk does not stall Report a problem.
        """
        tel = getattr(self, "telemetry", None)
        if tel is None:
            return {"cards": [], "note": "history store is off"}
        summary = bounded(lambda: tel.heat_summary(), 1.5, None)
        if summary is None:
            return {"cards": [], "note": "history store did not answer within 1.5 s"}
        return summary

    def report_send(self, body):
        """POST the payload the user saw (and maybe edited) to the intake. Only ever called from a Send click."""
        payload = (body or {}).get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                raise Invalid("the report is not valid JSON; fix it in the box or use Copy report")
        if not isinstance(payload, dict):
            raise Invalid("no report to send")
        raw = json.dumps(payload).encode("utf-8")
        if len(raw) > REPORT_MAX:
            raise Invalid(f"the report is {len(raw) // 1024} KB; the limit is {REPORT_MAX // 1024} KB (trim the log tail)")
        url = bug_url()
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            return {"ok": False, "error": "PXA_BUG_URL must be an http(s) URL", "url": url}
        req = urllib.request.Request(url, data=raw, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "pxa-control/" + CONTROL_VERSION})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                j = json.loads(r.read().decode("utf-8", "replace") or "{}")
            return {"ok": bool(j.get("ok")), "id": j.get("id"), "url": url}
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8", "replace")).get("error")
            except Exception:
                msg = None
            return {"ok": False, "error": f"the report server said {e.code}: {msg or e.reason}", "url": url}
        except Exception as e:
            return {"ok": False, "error": f"could not reach the report server ({e.__class__.__name__}: {e})", "url": url}

    # ---- community high-score board ---------------------------------------------------------
    # ---- the saved name for the board ---------------------------------------------------------
    def user_name(self):
        """the name the user typed for the board the first time (control.json), cleaned by the board's rules."""
        return {"name": clean_board_name(load_config().get("user_name", "")), "config": config_path()}

    def set_user_name(self, body):
        """Remember a board name, or forget it (empty). Only a display name lives here: no keys, no secrets."""
        raw = (body or {}).get("name", "")
        if not isinstance(raw, str) or len(raw) > 200:
            raise Invalid("name must be text")
        name = clean_board_name(raw)
        if raw.strip() and not name:
            raise Invalid("that name is not allowed on the board (no links, @mentions or rude words)")
        with _cfg_lock:
            c = load_config()
            c["user_name"] = name
            save_config(c)
        return {"ok": True, "name": name}

    @staticmethod
    def _cmd_flags(tokens):
        """argv tokens -> {flag: value} for the flags the score params need ('-c 4096', '--ctx-size=4096')."""
        out, i = {}, 0
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("-") and "=" in t and not t.startswith("--spec-type"):
                k, _, v = t.partition("=")
                out[k] = v
            elif t.startswith("-"):
                nxt = tokens[i + 1] if i + 1 < len(tokens) and not (tokens[i + 1].startswith("-") and len(tokens[i + 1]) > 1 and not tokens[i + 1][1:].isdigit()) else None
                out[t] = nxt
                if nxt is not None:
                    i += 1
            i += 1
        return out

    _build_cache = {}

    def engine_build_string(self, exe):
        """'version: 6214 (c1d43e84b5)' from `llama-server --version`, once per binary (a server that has not printed a
        build line yet, or whose log scrolled away, still gets its build into the score)."""
        if not exe or not os.path.isfile(exe):
            return None
        hit = self._build_cache.get(exe)
        if hit is not None:
            return hit or None

        def probe():
            r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=40)
            blob = (r.stdout or "") + (r.stderr or "")
            m = re.search(r"version:\s*(\S.*)", blob)
            return m.group(1).strip()[:80] if m else ""
        v = bounded(probe, 45, "") or ""
        self._build_cache[exe] = v
        return v or None

    def score_params(self, res, seat, model, cards, sel, cmd_tokens, n, split, mtp):
        """The params that produced a score. Only facts about the rig, the engine build, the model FILE NAME and the
        launch settings: no paths, hosts, addresses, keys or tokens (the board refuses a submission that has any)."""
        fl = self._cmd_flags(cmd_tokens)

        def first(*names):
            for k in names:
                if fl.get(k) not in (None, ""):
                    return fl[k]
            return None

        def toint(v):
            try:
                return int(str(v))
            except (TypeError, ValueError):
                return None
        req = seat.req or {}
        spec = next((cmd_tokens[i + 1] for i, t in enumerate(cmd_tokens[:-1]) if t == "--spec-type"), "") or ""
        m = re.search(r"n_max=(\d+)", spec)
        n_max = toint(m.group(1)) if m else toint(first("--draft-max", "--spec-draft-n-max", "--draft-n-max"))
        try:
            size = os.path.getsize(model)
        except (OSError, TypeError):
            size = None
        base = os.path.basename(str(model))
        tier = codec = None
        for e in ((self._models_cache or {}).get("models") or []):
            if e.get("path") == model:
                tier, codec = e.get("tier") or None, e.get("codec") or None
                break
        if not tier:
            qm = QUANT_RE.search(base)
            tier = qm.group(1).upper() if qm else None
        if not codec and tier:
            codec = "PXQN" if tier.upper().startswith("PXQN") else ("PXQ" if tier.upper().startswith("PXQ") else "other")
        st = self._rig_cache[1] or {}
        ver = ""
        try:
            ver = self.report_bundle()["bundle"]["engine"]["version"]
        except Exception:
            pass
        if not ver or ver == "unknown":
            ver = self.engine_build_string(seat.cmd[0] if seat.cmd else "") or ""
        cm = re.search(r"\(([0-9a-f]{6,40})\)", ver or "")
        fa = first("-fa", "--flash-attn")
        fa = {"1": "on", "true": "on", "0": "off", "false": "off"}.get(str(fa).lower(), str(fa).lower()) if fa else None
        by = {c["class"]: c for c in res.get("classes", [])}
        p = {
            "cards": [{"name": re.sub(r"[^A-Za-z0-9 ._()+:,\-]", "", g[1].replace("NVIDIA ", ""))[:48] or "GPU", "vram_mb": int(g[3])} for g in sel[:16]],
            "card_count": n,
            "driver": st.get("driver"),
            "engine_version": re.sub(r"[^A-Za-z0-9 ._()+:,\-]", "", ver or "")[:80].strip() or None,
            "engine_commit": cm.group(1) if cm else None,
            "model_file": re.sub(r"[^A-Za-z0-9._+\-]", "_", base)[:128], "model_size_bytes": size, "codec": codec, "tier": tier,
            "ctx": toint(first("-c", "--ctx-size")) or (req.get("ctx") or None),
            "kv_k": first("-ctk", "--cache-type-k") or (None if req.get("kv", "auto") == "auto" else req.get("kv")),
            "kv_v": first("-ctv", "--cache-type-v") or (None if req.get("kv", "auto") == "auto" else req.get("kv")),
            "split_mode": split,
            "mtp": bool(mtp),
            "n_max": n_max,
            "batch": toint(first("-b", "--batch-size")),
            "ubatch": toint(first("-ub", "--ubatch-size")),
            "flash_attn": fa if fa in ("on", "off", "auto") else None,
            "prompt_class": "prose",              # the headline decode number is the prose class
            "classes": [c for c in ("prose", "edit", "long") if c in by],
            "reps": res.get("reps"),
            "greedy512_sha": res.get("greedy512_sha"),
            "ts": int(res.get("ts") or time.time()),
        }
        return {k: v for k, v in p.items() if v not in (None, "")}

    def score_payload(self, res, name=""):
        """bench result -> the exact /v1/score payload (redacted; nothing identifying beyond the chosen name)."""
        seat = next((x for x in self.seats.values() if x.req and x.req.get("port") == res.get("port")), self.seat)
        req = seat.req or {}
        model = res.get("model") or req.get("model") or ""
        base = os.path.basename(str(model))
        sha8 = "*"
        try:
            import hashlib
            with open(model, "rb") as f:
                sha8 = hashlib.sha256(f.read(16 << 20)).hexdigest()[:8]     # first 16 MiB: the header and the first tensors
        except (OSError, TypeError):
            pass
        rows, _e = self.gpus()
        cards = [c for c in (res.get("cards") or req.get("gpus") or [])]
        sel = [g for g in rows if g[0] in cards] or rows
        gname = (sel[0][1].replace("NVIDIA ", "") if sel else "unknown")
        toks = [os.path.basename(t) if (os.sep in t and not t.startswith("-")) else t for t in self.L.redact_cmd(seat.cmd)] if seat.cmd else []
        # the model path and any --host / --api-key value never go to the board: file name only
        keep, skip = [], False
        for t in toks:
            if skip:
                skip = False
                continue
            if t in ("--host", "--api-key", "-a", "--alias", "--api-key-file", "--hf-token", "--ssl-key-file", "--ssl-cert-file"):
                skip = True
                continue
            if re.match(r"^--(host|api-key|alias|hf-token)=", t):
                continue
            keep.append(t)
        cmd = " ".join(keep)
        low = cmd.lower()
        n = max(1, len(cards) or len(sel))
        split = "tensor" if re.search(r"(-sm|--split-mode)[ =]tensor", low) else ("layer" if n > 1 else "plain")
        mode = split + ("+mtp" if ("mtp" in low and split != "plain") else "")
        if split == "plain" and "mtp" in low:
            mode = "mtp"
        by = {c["class"]: c for c in res.get("classes", [])}
        prose, long_ = by.get("prose", {}), by.get("long", {})
        qm = QUANT_RE.search(base)
        payload = {
            "app": "pxa-control", "version": CONTROL_VERSION, "name": clean_board_name(name),
            "engine_version": self.report_bundle()["bundle"]["engine"]["version"],
            "bracket": {"model_file": base, "model_sha8": sha8, "quant": qm.group(1).lower() if qm else "unknown",
                        "gpu": gname, "gpu_count": n, "mode": mode},
            "metrics": {"decode_tps": round(prose.get("decode_tps") or 0, 2), "prefill_tps": round(long_.get("prefill_tps") or 0, 1) or None},
            "cmd": cmd,
            "raw": {"reps": res.get("reps"), "greedy512_sha": res.get("greedy512_sha"),
                    "classes": [{k: c.get(k) for k in ("class", "decode_tps", "prefill_tps", "n_prompt", "decode_all", "prefill_all")}
                                for c in res.get("classes", [])]},
        }
        payload["params"] = self.score_params(res, seat, model, cards, sel, keep, n, split, "mtp" in low and split != "plain" or mode == "mtp")
        return redact_obj(payload)

    def score_check(self):
        """after a benchmark: the payload we would send, the current record for its bracket and whether this run beats it."""
        res = self.bench.get("result")
        if not res or self.bench.get("running"):
            return {"available": False, "reason": "run a benchmark first"}
        p = self.score_payload(res)
        if not p["metrics"]["decode_tps"]:
            return {"available": False, "reason": "the benchmark has no decode number"}
        b = p["bracket"]
        key = "|".join([re.sub(r"[^a-z0-9.]+", "-", re.sub(r"(?i)\.gguf$", "", b["model_file"]).lower()).strip("-")[:64], b["model_sha8"],
                        re.sub(r"[^a-z0-9_.]+", "-", b["quant"].lower()).strip("-")[:24],
                        re.sub(r"[^a-z0-9]+", "-", re.sub(r"(?i)nvidia|geforce|tesla|-pcie|-sxm2|-\d+gb|\d+gb", " ", b["gpu"]).lower()).strip("-")[:32],
                        str(b["gpu_count"]), b["mode"]])
        rec, err, top = None, None, []
        try:
            # A User-Agent is required: the board sits behind Cloudflare, which answers Python's default
            # urllib identity with 403 (error 1010), and the check then failed without a word (user report
            # 2026-09-30: "hit high score, it vanishes and does nothing").
            q = urllib.request.Request(score_base_url() + "/v1/scores?bracket=" + urllib.parse.quote(key, safe=""),
                                       headers={"User-Agent": "pxa-control/" + CONTROL_VERSION})
            with urllib.request.urlopen(q, timeout=8) as r:
                j = json.loads(r.read().decode("utf-8", "replace")) or {}
                rec, top = j.get("record"), j.get("top") or []
        except Exception as e:
            err = f"could not reach the board ({e.__class__.__name__})"
        beats = err is None and (rec is None or p["metrics"]["decode_tps"] > rec["decode"])
        return {"available": True, "payload": p, "record": rec, "top": top, "beats": beats, "error": err, "board": score_base_url(),
                "name": self.user_name()["name"]}

    def score_detail(self, sid):
        """one score's stored params, from the board (the click on a score)."""
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            raise Invalid("score id must be a number")
        if not 0 < sid < 10**9:
            raise Invalid("score id out of range")
        q = urllib.request.Request(score_base_url() + f"/v1/score/{sid}", headers={"User-Agent": "pxa-control/" + CONTROL_VERSION})
        try:
            with urllib.request.urlopen(q, timeout=8) as r:
                return {"ok": True, "score": json.loads(r.read().decode("utf-8", "replace"))}
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": "no such score on the board" if e.code == 404 else f"the board said {e.code}"}
        except Exception as e:
            return {"ok": False, "error": f"could not reach the board ({e.__class__.__name__})"}

    def score_send(self, body):
        payload = (body or {}).get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                raise Invalid("the submission is not valid JSON")
        if not isinstance(payload, dict):
            raise Invalid("nothing to send")
        payload["name"] = clean_board_name(payload.get("name"))
        if payload["name"]:                       # saved after the first entry: the user never types it again
            with _cfg_lock:
                c = load_config()
                if c.get("user_name") != payload["name"]:
                    c["user_name"] = payload["name"]
                    save_config(c)
        raw = json.dumps(payload).encode("utf-8")
        if len(raw) > 32 * 1024:
            raise Invalid("the submission is over 32 KB")
        req = urllib.request.Request(score_base_url() + "/v1/score", data=raw, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "pxa-control/" + CONTROL_VERSION})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                j = json.loads(r.read().decode("utf-8", "replace") or "{}")
            return {"ok": bool(j.get("ok")), "id": j.get("id"), "record": j.get("record"), "rank": j.get("rank"), "status": j.get("status"), "note": j.get("note")}
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8", "replace")).get("error")
            except Exception:
                msg = None
            return {"ok": False, "error": f"the board said {e.code}: {msg or e.reason}"}
        except Exception as e:
            return {"ok": False, "error": f"could not reach the board ({e.__class__.__name__}: {e})"}

    # ---- presets --------------------------------------------------------------------------
    def presets(self):
        return load_config().get("presets", {})

    def save_preset(self, name, body):
        if not isinstance(name, str) or not PRESET_NAME_RE.match(name):
            raise Invalid("preset name: 1-48 of letters, digits, space, _ . -")
        req = self.validate(body, check_model=False, resolve_port=False)   # auto port stays auto
        with _cfg_lock:
            c = load_config()
            if name not in c["presets"] and len(c["presets"]) >= 100:
                raise Invalid("at most 100 presets")
            c["presets"][name] = req
            save_config(c)
        return req

    def delete_preset(self, name):
        with _cfg_lock:
            c = load_config()
            ok = c["presets"].pop(name, None) is not None
            save_config(c)
        return ok

    def set_attach_port(self, port):
        port = _int({"p": port}, "p", 0, 65535, 0)
        if port and port == self.port:
            raise Invalid("that is PXA Control's own port")
        with _cfg_lock:
            c = load_config()
            c["attach_port"] = port
            save_config(c)
        return port

    # ---- Encode tab ------------------------------------------------------------------------
    def encode(self):
        """The Encode tab's service (created on first use: it reads the jobs folder and may find encoders on disk)."""
        if ENC is None:
            raise Invalid("The Encode tab is not available in this install (%s). Reinstall PXA Control with its pxa_encode*.py files." % ENC_IMPORT_ERROR)
        with self._enc_lock:
            if self._enc is None:
                self._enc = ENC.EncodeService(EncodeHost(self))
            return self._enc

    def encode_shutdown(self):
        """Control is exiting: running encodes become resumable and their processes stop."""
        if self._enc is not None:
            self._enc.shutdown()

    def encode_call(self, name, *a, **kw):
        """Run one service method; its plain-sentence errors become 400s (never a traceback on the page)."""
        svc = self.encode()
        try:
            return getattr(svc, name)(*a, **kw)
        except ENC.EncodeError as e:
            raise Invalid(str(e))

    def encode_test(self, body):
        """'Test it': load the finished file in Control's own server + bench flow. The output folder joins the model folders,
        the file gets a server profile of its own ('Encode test'), and the existing Start is called; the page then waits for
        health and runs the existing benchmark, and 'Share your score' is the existing high-score panel."""
        svc = self.encode()
        try:
            t = svc.test_info((body or {}).get("id"))
        except ENC.EncodeError as e:
            raise Invalid(str(e))
        folder = os.path.dirname(t["model"])
        roots = self.model_roots()
        if not path_under(t["model"], roots):
            self.set_model_dirs(list(load_config().get("model_dirs", [])) + [folder])
        rows, _e = self.gpus()
        have = {g[0] for g in rows}
        cards = [c for c in ((body or {}).get("cards") or t["cards"] or []) if c in have]
        note = None
        if not cards:
            best = max(rows, key=lambda g: g[3] - g[4]) if rows else None
            if best is None:
                raise Invalid("No graphics card was found on this machine to test it on.")
            cards = [best[0]]
            note = "Tested on card %d of this machine." % best[0]
        settings = {"model": t["model"], "gpus": cards}
        sid, _p = self.save_profile("encode-test", "Encode test", settings)
        res = self.start(dict(settings, sid=sid))
        res["sid"] = sid
        res["cards"] = cards
        res["note"] = note
        return res

    # ---- bench ----------------------------------------------------------------------------
    def _bench_module(self):
        p = os.path.join(HERE, "pxa-bench.py")
        if not os.path.isfile(p):
            p2 = shutil_which("pxa-bench")
            p = p2 or p
        if not os.path.isfile(p):
            return None
        spec = importlib.util.spec_from_file_location("pxa_bench_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def start_bench(self, body):
        body = body or {}
        q = {k: [str(body[k])] for k in ("sid", "port") if body.get(k)}
        port = self.target_port(q)
        if not port:
            raise Invalid("no server: start one on the Launch tab (or attach to a running one)")
        cards = None
        if body.get("sid") and self.seats.get(body["sid"]) and self.seats[body["sid"]].req:
            cards = self.seats[body["sid"]].req.get("gpus")
        elif not body.get("sid") and not body.get("port") and self.seat.running() and self.seat.req:
            cards = self.seat.req.get("gpus")
        else:
            for x in self.fleet(max_age=10)["instances"]:
                if x.get("port") == port and x.get("running"):
                    cards = x.get("gpus") or sorted(int(k) for k in x.get("vram") or {})
                    break
        if self.bench["running"]:
            raise Invalid("a benchmark is already running")
        reps = _int(body, "reps", 1, 6, 3)
        warm = _int(body, "warmup", 0, 120, 10)
        mod = self._bench_module()
        if mod is None:
            raise Invalid("tools/pxa-bench.py is not installed next to the launcher")
        self.bench = {"running": True, "lines": [], "result": None, "error": None, "started": time.time(),
                      "port": port, "cards": cards}
        threading.Thread(target=self._bench_run, args=(mod, port, reps, warm, cards), daemon=True).start()
        return {"ok": True}

    def default_power_limits(self):
        """{index: stock power limit W} (nvidia-smi power.default_limit), read once."""
        if self._default_limits is None:
            out = {}
            if not os.environ.get("PXA_LAUNCH_FAKE_GPUS"):
                txt = self.L._run(["nvidia-smi", "--query-gpu=index,power.default_limit",
                                   "--format=csv,noheader,nounits"], timeout=10) or ""
                for line in txt.splitlines():
                    f = [x.strip() for x in line.split(",")]
                    try:
                        out[int(f[0])] = float(f[1])
                    except (ValueError, IndexError):
                        continue
            self._default_limits = out
        return self._default_limits

    @staticmethod
    def summarize_telemetry(samples, cards=None, stock=None):
        """per-card worst case over the samples taken DURING a benchmark (peak temperature, lowest
        SM clock, every throttle reason seen, power limit vs stock) + plain-language warnings. The
        report used to carry one idle snapshot taken at report time, which hid a card that
        throttled under load (tucsonjohn 2026-09-30-b4c7a346: 5.6 t/s, 53 C / 405 MHz at idle)."""
        stock = stock or {}
        per = {}
        for smp in samples:
            for r in smp:
                if cards and r["index"] not in cards:
                    continue
                c = per.setdefault(r["index"], {"index": r["index"], "samples": 0, "temp_max": None, "sm_clock_min": None,
                                                "sm_clock_max_seen": None, "sm_clock_rated": r.get("sm_clock_max_mhz"),
                                                "power_max_w": None, "power_limit_w": r.get("power_limit_w"),
                                                "power_stock_w": stock.get(r["index"]), "throttle": []})
                c["samples"] += 1
                for k, src, fn in (("temp_max", "temp_c", max), ("sm_clock_min", "sm_clock_mhz", min),
                                   ("sm_clock_max_seen", "sm_clock_mhz", max), ("power_max_w", "power_w", max)):
                    v = r.get(src)
                    if v is not None:
                        c[k] = v if c[k] is None else fn(c[k], v)
                for t in r.get("throttle_reasons") or []:
                    if t not in c["throttle"] and not str(t).startswith("0x"):
                        c["throttle"].append(t)
        warns = []
        for c in per.values():
            i = c["index"]
            if c["temp_max"] is not None and c["temp_max"] >= App.HOT_C:
                warns.append(f"card {i} reached {c['temp_max']:.0f} \u00b0C during the run")
            th = [t for t in c["throttle"] if t in App.THROTTLE_WARN]
            if th:
                warns.append(f"card {i} throttled during the run ({', '.join(th)})")
            if c["power_limit_w"] and c["power_stock_w"] and c["power_limit_w"] < 0.9 * c["power_stock_w"]:
                warns.append(f"card {i} power limit {c['power_limit_w']:.0f} W is below its stock {c['power_stock_w']:.0f} W")
            if c["sm_clock_max_seen"] and c["sm_clock_rated"] and c["sm_clock_max_seen"] < 0.6 * c["sm_clock_rated"]:
                warns.append(f"card {i} never clocked above {c['sm_clock_max_seen']:.0f} MHz (rated {c['sm_clock_rated']:.0f})")
        return {"cards": sorted(per.values(), key=lambda c: c["index"]), "samples": len(samples), "warnings": warns}

    def _bench_run(self, mod, port, reps, warm, cards=None):
        b = self.bench
        url = f"http://127.0.0.1:{port}"
        samples, stop_ev = [], threading.Event()

        def sampler():
            while not stop_ev.is_set():
                t = bounded(self.gpu_telemetry, 10, None)
                if t:
                    samples.append(t)
                stop_ev.wait(2.0)
        threading.Thread(target=sampler, daemon=True).start()

        def say(s):
            b["lines"].append(time.strftime("%H:%M:%S ") + s)
        try:
            props = {}
            try:
                props = mod._get_json(url, "/props", timeout=10)
            except Exception:
                pass
            model = props.get("model_path") or (self.seat.req or {}).get("model") or "?"
            if port != self.engine_port():
                model = props.get("model_path") or props.get("model_alias") or "?"
            if warm:
                say(f"warm-up {warm}s")
                end = time.time() + warm
                while time.time() < end:
                    mod.measure_once(url, mod.CLASS_PROMPTS["prose"]["prompt"], 32, timeout=120)
            classes = []
            for cls in ("prose", "edit", "long"):
                spec = mod.CLASS_PROMPTS[cls]
                dec, pre, npr = [], [], None
                for i in range(reps):
                    # pxa-bench.py's own request (greedy, seed 0, no prompt cache), read here so the
                    # prompt size goes into the history next to the speeds
                    r = mod._post_json(url, "/completion", {"prompt": spec["prompt"], "n_predict": spec["n_predict"],
                                                           "temperature": 0.0, "seed": 0, "cache_prompt": False},
                                       timeout=600)
                    t = r.get("timings") or {}
                    d_, p_ = t.get("predicted_per_second") or 0.0, t.get("prompt_per_second") or 0.0
                    npr = t.get("prompt_n") or r.get("tokens_evaluated")
                    dec.append(d_)
                    if p_:
                        pre.append(p_)
                    say(f"{cls} rep {i + 1}/{reps}: {npr or '?'} prompt tokens, decode {d_:.1f} t/s, "
                        f"prefill {p_:.1f} t/s")
                    append_jsonl("history.jsonl", {"ts": time.time(), "model": os.path.basename(model),
                                                   "source": "bench", "n_prompt": npr,
                                                   "decode_tps": d_, "prefill_tps": p_})
                classes.append({"class": cls, "decode_tps": median(dec), "prefill_tps": median(pre),
                                "reps": reps, "n_prompt": npr, "decode_all": dec, "prefill_all": pre})
            say("greedy512 identity check")
            g = mod.greedy512(url)
            stop_ev.set()
            tel = self.summarize_telemetry(samples, cards, self.default_power_limits())
            for w in tel["warnings"]:
                say("WARNING: " + w)
            res = {"ts": time.time(), "model": model, "cards": cards,
                   "port": port, "reps": reps, "warmup_s": warm, "classes": classes,
                   "greedy512_sha": g.get("sha256"), "greedy512_empty": g.get("empty"), "telemetry": tel}
            append_jsonl("bench.jsonl", res)
            b["result"] = res
            say("done")
        except Exception as e:
            b["error"] = f"{e.__class__.__name__}: {e}"
            say("failed: " + b["error"])
        finally:
            stop_ev.set()
            b["running"] = False

    # ---- local speed history (fallback when the server has no /pxa/stats) ------------------
    def record_timings(self, t, model):
        if not isinstance(t, dict):
            return
        rec = {"ts": time.time(), "model": model or "?", "source": "chat",
               "n_prompt": t.get("prompt_n"), "n_gen": t.get("predicted_n"),
               "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second")}
        append_jsonl("history.jsonl", rec)


def median(xs):
    xs = sorted(x for x in xs if x)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def shutil_which(x):
    import shutil
    return shutil.which(x)


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/encode.js": ("encode.js", "application/javascript; charset=utf-8"),
          "/encode.css": ("encode.css", "text/css; charset=utf-8"),
          "/live.js": ("live.js", "application/javascript; charset=utf-8"),
          "/live.css": ("live.css", "text/css; charset=utf-8"),
          "/profiles.js": ("profiles.js", "application/javascript; charset=utf-8"),
          "/profiles.css": ("profiles.css", "text/css; charset=utf-8"),
          "/mqtt.js": ("mqtt.js", "application/javascript; charset=utf-8"),
          "/mark.png": ("mark.png", "image/png"),
          "/favicon.png": ("mark.png", "image/png")}
# the Chat tab's Classic view: chat.js (the PXAChat core) and its plugins chat-<name>.js / .css
# (tools/pxa_control_ui/CHAT-PLUGINS.md). The Assistant view's agent-<name> plugins are registered
# by pxa_chat.register, not here.
CHAT_PLUGINS = ("history", "markdown", "actions", "stream", "attach", "params", "context", "ux")
for _n in ("chat",) + tuple("chat-" + p for p in CHAT_PLUGINS):
    STATIC["/" + _n + ".js"] = (_n + ".js", "application/javascript; charset=utf-8")
    STATIC["/" + _n + ".css"] = (_n + ".css", "text/css; charset=utf-8")


_REQ = threading.local()          # who is asking, for the GPU audit log (never a token)


def req_who():
    return getattr(_REQ, "who", None) or "api"


def err_body(msg, code="invalid", status=400, detail=None):
    """every error PXA Control answers: {"error": human text, "code": machine word, "status": http status}"""
    d = {"error": str(msg), "code": code, "status": status}
    if detail is not None:
        d["detail"] = detail
    return d


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "PXAControl/1"
    app = None          # set on the subclass made by make_handler

    def handle(self):
        # a client that drops the connection mid-reply (health checkers, a closed tab) is not an
        # error worth a traceback in the log
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        if os.environ.get("PXA_CONTROL_ACCESS_LOG"):
            sys.stderr.write("pxa-control: " + (fmt % args) + "\n")

    # ---- guards ---------------------------------------------------------------------------
    def _host_ok(self):
        host = (self.headers.get("Host") or "").strip().lower()
        if not host:
            return False
        if self.app.lan:
            return True            # the token is the guard on a LAN bind
        h = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        return h in ("127.0.0.1", "localhost", "[::1]")

    def _origin_ok(self):
        o = self.headers.get("Origin")
        if not o:
            return True            # same-origin fetch from older browsers / curl
        try:
            u = urllib.parse.urlparse(o)
        except ValueError:
            return False
        return u.netloc.lower() == (self.headers.get("Host") or "").lower()

    def _cookie_token(self):
        c = self.headers.get("Cookie") or ""
        for part in c.split(";"):
            k, _, v = part.strip().partition("=")
            if k == TOKEN_COOKIE:
                return v
        return None

    def _auth(self, query):
        """-> (ok, set_cookie). Token from cookie, X-PXA-Token header or ?token=."""
        if not self.app.token:
            return True, False
        auth = self.headers.get("Authorization") or ""
        bearer = auth[7:].strip() if auth[:7].lower() == "bearer " else None     # a Prometheus scrape job
        for cand, from_query in ((self._cookie_token(), False), (self.headers.get("X-PXA-Token"), False),
                                 (bearer, False), ((query.get("token") or [None])[0], True)):
            if cand and hmac.compare_digest(str(cand), self.app.token):
                return True, from_query
        return False, False

    # ---- responses ------------------------------------------------------------------------
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if ctype.startswith("text/html"):
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json_body(self):
        n = content_length(self.headers, MAX_BODY)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            raise Invalid("request body is not JSON")
        if not isinstance(body, dict):              # every route reads fields: a list or a number was a 500 before
            raise Invalid("request body must be a JSON object")
        return body

    def _login_page(self):
        return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                "<title>PXA Control</title><body style='font-family:system-ui;background:#111214;color:#ECEDEF;"
                "display:grid;place-items:center;height:100vh;margin:0'><form method=get "
                "style='display:grid;gap:10px;min-width:280px'><b>PXA Control</b><span style='color:#A0A4AC'>"
                "This rig's GUI is on the network; enter the token printed where it was started.</span>"
                "<input name=token autocomplete=off style='padding:8px;border-radius:8px;border:1px solid #33363C;"
                "background:#1D1F23;color:inherit'><button style='padding:8px;border-radius:8px;border:0;"
                "background:#E08A3C;color:#111214;font-weight:700'>Open</button></form>")

    # ---- routing --------------------------------------------------------------------------
    def do_HEAD(self):
        p = urllib.parse.urlparse(self.path).path
        if p not in STATIC:            # a HEAD on /api/log/stream used to hold a thread forever
            return self._send(405, {"error": "HEAD only on the page"})
        self.do_GET()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        u = urllib.parse.urlparse(self.path)
        path, query = u.path, urllib.parse.parse_qs(u.query)
        if not self._host_ok():
            return self._send(421, err_body("unexpected Host header", "bad_host", 421))
        ok, set_cookie = self._auth(query)
        if not ok:
            if path == "/metrics":
                return self._send(401, "token required (Authorization: Bearer <token>)\n", "text/plain; charset=utf-8")
            if method == "GET" and not path.startswith("/api/"):
                return self._send(401, self._login_page(), "text/html; charset=utf-8")
            return self._send(401, err_body("token required", "unauthorized", 401))
        self.app.last_request = time.time()
        _REQ.who = f"page {self.client_address[0]}" if self.client_address else "api"     # a page is looking: a background Control stays up
        if set_cookie and method == "GET":
            # drop the token from the address bar once it is in a cookie
            q = {k: v for k, v in query.items() if k != "token"}
            loc = path + (("?" + urllib.parse.urlencode(q, doseq=True)) if q else "")
            self.send_response(303)
            self.send_header("Location", loc)
            self.send_header("Set-Cookie", f"{TOKEN_COOKIE}={self.app.token}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Strict")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if method != "GET" and not self._origin_ok():
            return self._send(403, err_body("cross-origin request refused", "cross_origin", 403))
        try:
            if method == "GET" and path in STATIC:
                fn, ctype = STATIC[path]
                try:
                    with open(os.path.join(UI_DIR, fn), "rb") as f:
                        data = f.read()
                except OSError:
                    return self._send(404, {"error": f"{fn} missing from {UI_DIR}"})
                return self._send(200, data, ctype)
            if path.startswith("/api/engine/"):
                return self._engine_proxy(method, path[len("/api/engine/"):], u.query)
            if method == "GET" and path == "/api/log/stream":
                return self._log_stream(query)
            route = ROUTES.get((method, path))
            if route is None:
                if any(p == path for (_m, p) in ROUTES):
                    return self._send(405, err_body(f"{method} is not allowed on {path}", "method_not_allowed", 405))
                return self._send(404, err_body("not found", "not_found", 404))
            body = self._json_body() if method in ("POST", "DELETE") else None
            res = route(self.app, body, query)
            if hasattr(res, "stream_to"):          # a live event stream (the chat agent's /api/chat/events)
                return res.stream_to(self)
            if isinstance(res, Reply):
                return self._send(res.code, res.body, res.ctype, res.extra)
            return self._send(200, res)
        except Invalid as e:
            return self._send(getattr(e, "status", 400), err_body(e, getattr(e, "code", "invalid"), getattr(e, "status", 400)))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            if CTL_ERRORS and isinstance(e, CTL_ERRORS):      # the Profiles backend's refusals: locked 423, forbidden 403, ...
                return self._send(e.status, err_body(e, e.code, e.status, getattr(e, "detail", None)))
            import traceback
            traceback.print_exc()
            return self._send(500, err_body(f"{e.__class__.__name__}: {e}", "internal", 500))

    # ---- SSE log --------------------------------------------------------------------------
    def _log_stream(self, query):
        try:
            since = int((query.get("since") or ["0"])[0])
        except ValueError:
            since = 0
        seat = self.app.get_seat(_q(query, "sid") or "main", create=not _q(query, "sid"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_ping = time.time()
        try:
            while True:
                items = seat.lines_since(since, 500)
                if items:
                    since = items[-1][0]
                    payload = json.dumps({"seq": since, "lines": [x[1] for x in items]})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                elif time.time() - last_ping > 15:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_ping = time.time()
                with seat.cond:
                    seat.cond.wait(timeout=1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    # ---- engine proxy ---------------------------------------------------------------------
    def _engine_proxy(self, method, sub, qs):
        allowed = ENGINE_GET if method == "GET" else (ENGINE_POST if method == "POST" else set())
        if sub not in allowed:
            return self._send(404, {"error": f"{method} /{sub} is not proxied"})
        query = urllib.parse.parse_qs(qs or "")
        port = self.app.target_port(query)       # Invalid (400) for a port this GUI does not know
        if not port:
            return self._send(503, {"error": "no server running"})
        fwd = urllib.parse.urlencode({k: v for k, v in query.items() if k not in ("sid", "port", "token")}, doseq=True)
        url = f"http://127.0.0.1:{port}/{sub}" + (("?" + fwd) if fwd and method == "GET" else "")
        data = None
        think_info = None
        hdr = {"Accept": self.headers.get("Accept") or "*/*"}
        if method == "POST":
            try:
                n = content_length(self.headers, MAX_BODY * 8)
            except Invalid as e:
                return self._send(413 if "large" in str(e) else 400, {"error": str(e)})
            data = self.rfile.read(n)
            hdr["Content-Type"] = "application/json"
            if sub == "v1/chat/completions":
                data, think_info = self.app.thinking_rewrite(data, query)
        req = urllib.request.Request(url, data=data, headers=hdr, method=method)
        try:
            r = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as e:
            body = e.read()
            return self._send(e.code, body, e.headers.get("Content-Type") or "application/json")
        except Exception as e:
            return self._send(502, {"error": f"server unreachable: {e.__class__.__name__}"})
        ctype = r.headers.get("Content-Type") or "application/octet-stream"
        streaming = "event-stream" in ctype
        if not streaming:
            body = r.read()
            if sub in ("v1/chat/completions", "completion"):
                try:
                    j = json.loads(body)
                    self.app.record_timings(j.get("timings"), j.get("model"))
                except Exception:
                    pass
            if sub == "pxa/speed":
                ctype = "text/html; charset=utf-8"
            return self._send(r.status, body, ctype, extra=_think_header(think_info))
        self.send_response(r.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in _think_header(think_info).items():
            self.send_header(k, v)
        self.end_headers()
        tail = b""
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                tail = (tail + chunk)[-65536:]
        except (BrokenPipeError, ConnectionResetError):
            r.close()
            return
        for line in reversed(tail.split(b"\n")):
            if line.startswith(b"data:") and b'"timings"' in line:
                try:
                    j = json.loads(line[5:].strip())
                    self.app.record_timings(j.get("timings"), j.get("model"))
                except Exception:
                    pass
                break


def _q(query, k, default=None):
    return (query.get(k) or [default])[0]


def _think_header(info):
    """X-PXA-Thinking: what the proxy did to a chat body (ASCII JSON, bounded)."""
    if not info:
        return {}
    return {"X-PXA-Thinking": json.dumps(info, ensure_ascii=True, separators=(",", ":"))[:2000]}


def r_thinking(app, body, query):
    return app.thinking_info(query)


def r_thinking_override(app, body, query):
    body = body or {}
    return {"model": body.get("model"), "saved": app.set_thinking_override(body.get("model"), body.get("thinking"))}


def r_thinking_table(app, body, query):
    t = TH.load_table()
    return {"version": t.get("version"), "note": t.get("note"), "answer_reserve": t.get("answer_reserve"),
            "families": [{k: f.get(k) for k in ("id", "label", "mechanism", "default", "levels", "budget", "source")}
                         for f in t["families"]]}



def _pxa_update_bin():
    """the pxa-update binary shipped next to this install, or none."""
    env = os.environ.get("PXA_UPDATE_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    for c in (os.path.join(HERE, "pxa-update"), os.path.join(os.path.dirname(HERE), "bin", "pxa-update")):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _pxa_update_check(bin):
    p = subprocess.run([bin, "check"], capture_output=True, text=True, timeout=40)
    line = ""
    if p.stdout:
        lines = [x for x in p.stdout.splitlines() if x.startswith("current=")]
        line = lines[-1] if lines else ""
    info = {"ok": p.returncode == 0, "binary": True, "update": False}
    for part in line.split():
        if "=" in part:
            k, v = part.split("=", 1)
            if k in ("current", "latest"):
                info[k] = v
            elif k == "update":
                info["update"] = v == "yes"
    if p.returncode != 0:
        info["error"] = ((p.stderr or "could not check for an update").strip())[:300]
    return info


def r_pxa_update(app, body, query):
    """GET /api/update: ask the pxa-update binary. Python does not download anything."""
    n = len([s for s in app.seats.values() if s.running()])
    b = _pxa_update_bin()
    if not b:
        return {"ok": True, "binary": False, "update": False, "servers_running": n}
    info = _pxa_update_check(b)
    info["servers_running"] = n
    return info


def r_pxa_update_apply(app, body, query):
    """POST /api/update/apply: run pxa-update apply. Refuses while a server is running."""
    if any(s.running() for s in app.seats.values()):
        e = Invalid("Stop the running servers first. PXA will not restart them.")
        e.status, e.code = 409, "servers_running"
        raise e
    b = _pxa_update_bin()
    if not b:
        e = Invalid("pxa-update is not in this install")
        e.status, e.code = 404, "no_updater"
        raise e
    try:
        p = subprocess.run([b, "apply"], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        e = Invalid("The update took too long and was stopped. Try again.")
        e.status, e.code = 504, "update_timeout"
        raise e
    if p.returncode != 0:
        e = Invalid(((p.stderr or p.stdout or "update failed").strip())[:400])
        e.status, e.code = (409 if p.returncode == 2 else 500), "update_failed"
        raise e
    return {"ok": True, "output": (p.stdout or "").strip()[:400]}


def r_lib_update(app, body, query):
    """GET /api/lib/update: the licensed library's update state. ?check=1 asks the licence server now, otherwise the
    cached answer is returned and the poller refreshes it in the background."""
    app.lib.touch()
    if str(_q(query, "check", "")).lower() in ("1", "true", "yes", "on"):
        return app.lib.check(force=True)
    return app.lib.status()


def r_lib_update_apply(app, body, query):
    """POST /api/lib/update/apply: install the newest release on the channel. Refuses while a server is running, on a
    refused or unverifiable manifest, and on an engine older than the release asks for."""
    app.lib.touch()
    return app.lib.apply(channel=(_b(body).get("channel") or None))


def r_lib_update_rollback(app, body, query):
    """POST /api/lib/update/rollback: put back the previous library, or the one the engine shipped."""
    app.lib.touch()
    return app.lib.rollback()


def r_lib_update_settings(app, body, query):
    """POST /api/lib/update/settings {channel, auto}: the channel this install follows and whether a newer release is
    applied without asking. Both are off until set; the beta channel is still refused by the server for a key without
    the valued role."""
    app.lib.touch()
    b = _b(body)
    ch, auto = b.get("channel"), b.get("auto")
    if ch is not None and ch not in LIB_CHANNELS:
        raise Invalid("channel must be one of: %s" % ", ".join(LIB_CHANNELS))
    if auto is not None and not isinstance(auto, bool):
        raise Invalid("auto must be true or false")
    app.lib.cfg_set(channel=ch, auto=auto)
    return app.lib.status()


def r_expert_map(app, body, query):
    """GET /api/expert-map: the adaptive map, in plain words, and where it stands."""
    import pxa_expert_map as M
    model = _q(query, "model", "") or ""
    return M.status_dict(model or None)


def r_expert_map_rebuild(app, body, query):
    """POST /api/expert-map/rebuild: start from the curated csv. Calibration is the builder."""
    import pxa_expert_map as M
    body = body or {}
    model = body.get("model") or ""
    curated = body.get("curated") or ""
    dest = body.get("dest") or (M.beside_model(model) if model else "")
    if model and curated:
        try:
            M.curated_start(model, curated, dest)
        except ValueError as e:
            err = Invalid(str(e))
            err.status, err.code = 400, "bad_map"
            raise err
    out = M.status_dict(model or None, dest or None)
    out["ok"] = True
    out["note"] = (
        "The curated map is the starting point. Each calibration pass is measured. "
        "The builder stops when decode speed stops rising and keeps the fastest map. "
        "Leave online adapt on after that."
    )
    return out


def r_expert_map_reset(app, body, query):
    """POST /api/expert-map/reset: drop learned counts. The curated file stays."""
    import pxa_expert_map as M
    body = body or {}
    model = body.get("model") or ""
    if not model:
        err = Invalid("name the model")
        err.status, err.code = 400, "bad_map"
        raise err
    try:
        M.reset_learned([M.learned_beside(model), M.cache_learned(model)])
    except ValueError as e:
        err = Invalid(str(e))
        err.status, err.code = 400, "bad_map"
        raise err
    out = M.status_dict(model)
    out["ok"] = True
    out["note"] = "Learned counts cleared. The map that shipped with the model is unchanged."
    return out


def r_health(app, body, query):
    """GET /api/health: is this Control alive and what can it do. Cheap: no nvidia-smi call, no server probe."""
    seats = [s for s in app.seats.values() if s.running()]
    d = {"ok": True, "control_version": CONTROL_VERSION, "pid": os.getpid(), "uptime_s": round(time.time() - app.started_at, 1),
         "servers_running": len(seats), "nvidia_smi": shutil_which("nvidia-smi") is not None,
         "telemetry": {"available": TEL is not None, "recording": app.telemetry is not None},
         "encode": ENC is not None, "gpu_profiles": None, "errors": []}
    if app.gpuctl is not None:
        try:
            d["gpu_profiles"] = app.gpuctl.health()
        except Exception as e:        # noqa: BLE001 - health answers even when the backend is broken
            d["errors"].append(f"gpu profiles: {e.__class__.__name__}: {e}")
    elif app.gpuctl_error:
        d["errors"].append("gpu profiles: " + app.gpuctl_error)
    return d


def _limit(query, default=200, hi=2000):
    v = _q(query, "limit", str(default))
    if not str(v).isdigit():
        raise Invalid("limit: a whole number")
    return max(1, min(hi, int(v)))


def r_gpu_state(app, body, query):
    return app.gpu_state()


def r_gpu_profiles(app, body, query):
    g = app.need_gpuctl()
    return {"profiles": list(g.store.profiles().values()), "presets": CTL.store.PRESETS}


def r_gpu_profile_save(app, body, query):
    return {"profile": app.need_gpuctl().save_profile(body, who=req_who())}


def r_gpu_profile_delete(app, body, query):
    return app.need_gpuctl().delete_profile(body, who=req_who())


def r_gpu_plan(app, body, query):
    return app.need_gpuctl().plan(body)


def r_gpu_apply(app, body, query):
    return app.need_gpuctl().apply(body, who=req_who())


def r_gpu_reset(app, body, query):
    return app.need_gpuctl().reset(body, who=req_who())


def r_gpu_reserve(app, body, query):
    return app.need_gpuctl().reserve(body, who=req_who())


def r_gpu_maintenance(app, body, query):
    return app.need_gpuctl().maintenance(body, who=req_who())


def r_gpu_quiet(app, body, query):
    return app.need_gpuctl().quiet(body, who=req_who())


def r_gpu_audit(app, body, query):
    return app.need_gpuctl().audit_read(limit=_limit(query))


def r_gpu_export(app, body, query):
    d = app.need_gpuctl().export()
    return Reply(json.dumps(d, indent=1), "application/json; charset=utf-8",
                 extra={"Content-Disposition": 'attachment; filename="pxa-gpu-profiles.json"'})


def r_gpu_import(app, body, query):
    return app.need_gpuctl().import_(body, who=req_who())


def r_gpu_schedules(app, body, query):
    return app.need_gpuctl().set_schedules(body, who=req_who())


def r_gpu_supervisor(app, body, query):
    return app.need_gpuctl().supervisor_action(body, who=req_who())


def r_gpu_undo(app, body, query):
    return app.need_gpuctl().undo(body, who=req_who())


def r_gpu_ui(app, body, query):
    """POST /api/gpu/ui {"mode": "simple" | "advanced"}: the Profiles page's Simple / Advanced switch, remembered in
    control.json (the page also keeps it per browser)."""
    mode = (body or {}).get("mode")
    if mode not in ("simple", "advanced"):
        raise Invalid("mode: simple or advanced")
    with _cfg_lock:
        c = load_config()
        c["ui_mode"] = mode
        save_config(c)
    if app.gpuctl is not None:
        app.gpuctl.settings["ui_mode"] = mode
    return {"mode": mode}


def r_gpu_power(app, body, query):
    return app.need_gpuctl().set_power(body, who=req_who())


def r_gpu_clocks(app, body, query):
    return app.need_gpuctl().clocks(_q(query, "uuid"))


def r_info(app, body, query):
    return {"version": 1, "control_version": CONTROL_VERSION, "lan": app.lan, "port": app.port,
            "companion": app.companion, "idle_exit_s": app.idle_s if app.companion else None,
            "catalog_src": app.catalog_src, "levers": len(app.catalog), "kv_types": KV_TYPES,
            "split_modes": SPLIT_MODES, "config": config_path(), "docker": docker_bin() is not None,
            "extra_flags": sorted(EXTRA_FLAGS), "thinking": True}


def r_rig(app, body, query):
    return {"static": app.rig_static(force=_q(query, "force") == "1"), "live": app.rig_live()}


def r_rig_live(app, body, query):
    app.live.touch()
    return app.rig_live()


def r_doctor(app, body, query):
    return app.doctor(force=_q(query, "force") == "1")


def r_companions(app, body, query):
    """Drafter next to the chosen model, and whether --hot-model may be offered for these cards."""
    model = _q(query, "model") or ""
    raw = _q(query, "gpus") or ""
    want = set()
    for part in raw.replace(",", " ").split():
        try:
            want.add(int(part))
        except ValueError:
            raise Invalid("card %r is not an index" % part)
    rows, _err = app.gpus()
    sel = [g for g in rows if g[0] in want] if want else []
    drafters = []
    if isinstance(model, str) and model.endswith(".gguf") and os.path.isfile(model):
        sib = app.L.gemma4_assistant_siblings(model)
        if sib:
            drafters.append({"path": sib, "file": os.path.basename(sib)})
    return {"drafters": drafters, "vmm": bool(sel) and app.L.vmm_offered(sel)}


def r_models(app, body, query):
    return app.models(force=_q(query, "force") == "1")


def r_model_dirs(app, body, query):
    dirs = app.set_model_dirs((body or {}).get("model_dirs"))
    return {"model_dirs": dirs, "models": app.models(force=True)}


def r_fits(app, body, query):
    body = body or {}
    ctx = _int(body, "ctx", 0, 1 << 21, 0)
    return app.fit_table(body.get("card_sets"), ctx)


def r_levers(app, body, query):
    names = app.catalog_names
    every, preset = getattr(names, "every", set()), getattr(names, "preset", set())
    # built-in levers set where Control was started (the Flash-Next preset's env): counted, not named; they reach the
    # server through the inherited environment
    env_on = sum(1 for k, v in os.environ.items() if v and isinstance(names, LeverNames) and names.builtin(k))
    return {"src": app.catalog_src,
            "levers": [dict(r, kind=lever_kind(r), default_state=lever_default_state(r)) for r in app.catalog],
            "builtin": {"hashes": sorted(every), "preset": sorted(preset), "env_on": env_on}}


def r_levers_lint(app, body, query):
    levers, errs = validate_levers((body or {}).get("levers") or {}, app.catalog_names)
    return {"errors": errs, "warnings": lint_levers(levers, app.catalog)}


def r_plan(app, body, query):
    return app.plan(body)


def r_start(app, body, query):
    return app.start(body)


def r_stop(app, body, query):
    sid = (body or {}).get("sid") or _q(query, "sid") or "main"
    return {"stopped": app.stop(sid), "status": app.status(sid)}


def r_restart(app, body, query):
    return app.restart(body if body else None, sid=_q(query, "sid"))


def r_status(app, body, query):
    return app.status(_q(query, "sid"))


def r_log(app, body, query):
    try:
        since = int(_q(query, "since", "0"))
    except ValueError:
        since = 0
    seat = app.get_seat(_q(query, "sid") or "main", create=not _q(query, "sid"))
    q = _q(query, "q", "")
    if q:
        items = seat.search(q[:200], regex=_q(query, "regex") == "1")
        return {"seq": seat.seq, "lines": [x[1] for x in items], "q": q, "matches": len(items)}
    items = seat.lines_since(since)
    return {"seq": items[-1][0] if items else since, "lines": [x[1] for x in items]}


def r_servers(app, body, query):
    prof = app.profiles()
    out = []
    for sid in sorted(prof, key=lambda k: (k != "main", prof[k].get("name", k).lower())):
        st = app.seats[sid].status() if sid in app.seats else Seat(app.L, sid).status()
        out.append({"sid": sid, "name": prof[sid].get("name", sid), "settings": prof[sid].get("settings") or {},
                    "running": st["running"], "phase": st["phase"], "port": st["port"], "pid": st["pid"]})
    return {"servers": out}


def r_server_save(app, body, query):
    body = body or {}
    sid, ent = app.save_profile(body.get("sid"), body.get("name") or "", body.get("settings"))
    return {"sid": sid, "server": ent, "servers": r_servers(app, None, {})["servers"]}


def r_server_delete(app, body, query):
    return {"deleted": app.delete_profile((body or {}).get("sid")), "servers": r_servers(app, None, {})["servers"]}


def r_fleet(app, body, query):
    app.live.touch()
    return app.fleet(max_age=0 if _q(query, "force") == "1" else 2.5)


def r_live(app, body, query):
    """GET /api/live?since=<unix s>&step=<s>: the rolling history behind the Live tab. `since` (default 15 min) keeps
    a poll small: ask only for what is newer than the last row you have; `step` averages old rows into buckets."""
    app.live.touch(fast=True)
    now = time.time()
    since = _num(_q(query, "since"))
    since = None if since is None else min(max(since, now - LIVE_KEEP_S), now)
    return app.live.snapshot(since, max(0.0, min(_num(_q(query, "step")) or 0.0, 600.0)))


def _tel(app):
    if app.telemetry is None:
        if app.tel_cfg.get("enabled") and not app.tel_error:
            raise Invalid("this PXA Control does not record history (it was started without it)")
        raise Invalid("the telemetry history is off" + (f" ({app.tel_error})" if app.tel_error else "") +
                      ": turn it on with POST /api/telemetry/settings {\"enabled\": true} or unset PXA_CONTROL_TELEMETRY=0")
    return app.telemetry


def _tel_args(query):
    keys = [k for k in (_q(query, "key") or "").split(",") if k.strip()] or None
    return keys, _num(_q(query, "since")), _num(_q(query, "until")), _num(_q(query, "step"))


def r_tel_status(app, body, query):
    """GET /api/telemetry: is the history on, where it lives, how big, which series it holds."""
    st = app.telemetry_status()
    if app.telemetry is not None:
        try:
            st["series_list"] = app.telemetry.list_series()
        except Exception as e:        # noqa: BLE001
            st["series_list"] = []
            st["error"] = st.get("error") or str(e)
    return st


def r_tel_series(app, body, query):
    """GET /api/telemetry/series?kind=card|server|host&key=a,b&since=&until=&step=&points=&source=auto|raw|1m"""
    keys, since, until, step = _tel_args(query)
    kind = _q(query, "kind", "card")
    if kind not in ("card", "server", "host"):
        raise Invalid("kind must be card, server or host")
    pts = int(min(max(_num(_q(query, "points")) or 600, 10), 5000))
    src = _q(query, "source", "auto")
    if src not in ("auto", "raw", "1m"):
        raise Invalid("source must be auto, raw or 1m")
    return _tel(app).series(kind, keys, since, until, step, max_points=pts, source=src)


def r_tel_history(app, body, query):
    """GET /api/telemetry/history?since=&until=&points=: cards, servers and the host in the shape the Live tab draws."""
    st = _tel(app)
    _k, since, until, step = _tel_args(query)
    pts = int(min(max(_num(_q(query, "points")) or 600, 10), 5000))
    c = st.series("card", None, since, until, step, max_points=pts)
    s = st.series("server", None, since, until, step, max_points=pts)
    h = st.series("host", None, since, until, step, max_points=pts)
    return {"ts": time.time(), "since": c["since"], "until": c["until"], "step": c["step"], "source": c["source"],
            "cols": {"card": c["cols"], "server": s["cols"], "host": h["cols"]},
            "cards": c["series"], "servers": s["series"], "host": (h["series"] or [None])[0]}


def r_tel_requests(app, body, query):
    """GET /api/telemetry/requests?key=&since=&until=&limit=: every finished request the store kept (raw retention)."""
    keys, since, until, _s = _tel_args(query)
    return _tel(app).requests(keys, since, until, int(_num(_q(query, "limit")) or 2000))


def r_tel_csv(app, body, query):
    """GET /api/telemetry/csv?kind=card|server|host|requests&key=&since=&until=&step=: a spreadsheet download."""
    keys, since, until, step = _tel_args(query)
    kind = _q(query, "kind", "card")
    if kind not in ("card", "server", "host", "requests"):
        raise Invalid("kind must be card, server, host or requests")
    text = _tel(app).csv_text(kind, keys, since, until, step, source=_q(query, "source", "auto"))
    name = "pxa-telemetry-%s-%s.csv" % (kind, time.strftime("%Y%m%d-%H%M", time.localtime()))
    return Reply(text, "text/csv; charset=utf-8", extra={"Content-Disposition": f'attachment; filename="{name}"'})


def r_tel_settings(app, body, query):
    return app.set_telemetry(body)


def r_metrics(app, body, query):
    """GET /metrics: Prometheus text, only when the history's prometheus setting (or PXA_CONTROL_METRICS=1) is on."""
    if TEL is None or not app.tel_cfg.get("prometheus"):
        return Reply("PXA Control /metrics is off: set PXA_CONTROL_METRICS=1 or POST /api/telemetry/settings "
                     "{\"prometheus\": true}\n", "text/plain; charset=utf-8", code=404)
    return Reply(app.metrics_text(), "text/plain; version=0.0.4; charset=utf-8")


def r_mqtt_status(app, body, query):
    """GET /api/mqtt: is the Home Assistant publisher on, is it connected, what has it published."""
    return app.mqtt_status()


def r_mqtt_settings(app, body, query):
    """POST /api/mqtt/settings: turn the publisher on/off and set the broker. The password is stored
    and reported as `password_set` only, never echoed back."""
    return app.set_mqtt(body)


def r_mqtt_test(app, body, query):
    """POST /api/mqtt/test: connect to the broker and prove the link, publishing nothing."""
    return app.mqtt_test(body)


def r_fleet_adopt(app, body, query):
    body = body or {}
    return {"adopted": app.adopt(body.get("key"), body.get("name"), bool(body.get("control")))}


def r_fleet_unadopt(app, body, query):
    return {"unadopted": app.unadopt((body or {}).get("key"))}


def r_fleet_control(app, body, query):
    body = body or {}
    return app.external_control(body.get("key"), body.get("action"), body.get("confirm"))


def r_fleet_log(app, body, query):
    try:
        tail = int(_q(query, "tail", "500"))
    except ValueError:
        tail = 500
    return app.external_log(_q(query, "key"), tail, (_q(query, "q", "") or "")[:200])


def r_engines(app, body, query):
    return app.engines(force=_q(query, "force") == "1")


def r_presets(app, body, query):
    return app.presets()


def r_preset_save(app, body, query):
    body = body or {}
    return {"saved": app.save_preset(body.get("name"), body.get("settings")), "presets": app.presets()}


def r_preset_delete(app, body, query):
    return {"deleted": app.delete_preset((body or {}).get("name")), "presets": app.presets()}


def r_attach(app, body, query):
    return {"attach_port": app.set_attach_port((body or {}).get("port")), "status": app.status()}


def r_bench_start(app, body, query):
    return app.start_bench(body)


def r_bench(app, body, query):
    return {"job": app.bench, "history": read_jsonl("bench.jsonl", limit=50)}


def r_history(app, body, query):
    try:
        since = float(_q(query, "since", "0"))
    except ValueError:
        since = 0.0
    return {"records": read_jsonl("history.jsonl", since=since)}


def r_report_bundle(app, body, query):
    return app.report_bundle(_q(query, "sid"))


def r_score_check(app, body, query):
    return app.score_check()


def r_user(app, body, query):
    return app.user_name()


def r_user_set(app, body, query):
    return app.set_user_name(body)


def r_score_detail(app, body, query):
    return app.score_detail(_q(query, "id"))


def r_score_send(app, body, query):
    return app.score_send(body)


def r_report_send(app, body, query):
    return app.report_send(body)


def _b(body):
    return body if isinstance(body, dict) else {}


def r_enc_state(app, body, query):
    return app.encode_call("state", force=bool(_q(query, "force")))


def r_enc_rescan(app, body, query):
    return app.encode_call("rescan")


def r_enc_use(app, body, query):
    return app.encode_call("use", _b(body).get("path"))


def r_enc_add(app, body, query):
    return app.encode_call("add_encoder", _b(body).get("path"))


def r_enc_key(app, body, query):
    return app.encode_call("set_key", _b(body).get("key"))


def r_enc_get(app, body, query):
    b = _b(body)
    return app.encode_call("get_encoder", b.get("edition"), b.get("key"), bool(b.get("update")))


def r_enc_get_cancel(app, body, query):
    return app.encode_call("cancel_get")


def r_enc_runtime(app, body, query):
    """GET: the GPU runtime (cuBLAS / cuSOLVER) the Pro encoder needs: what is missing, what is installed, what the licence server offers."""
    return app.encode_call("runtime_view", None, True, bool(_q(query, "force")))


def r_enc_runtime_get(app, body, query):
    return app.encode_call("get_runtime")


def r_enc_runtime_cancel(app, body, query):
    return app.encode_call("cancel_runtime")


def r_enc_update(app, body, query):
    return app.encode_call("update_check", force=bool(_b(body).get("force")))


def r_enc_licence(app, body, query):
    return app.encode_call("licence_check")


def r_enc_inspect(app, body, query):
    return app.encode_call("inspect", _b(body).get("source"), force=bool(_b(body).get("force")))


def r_enc_plan(app, body, query):
    return app.encode_call("plan", _b(body))


def r_enc_checks(app, body, query):
    return app.encode_call("checks", _b(body))


def r_enc_start(app, body, query):
    return app.encode_call("start", _b(body))


def r_enc_job(app, body, query):
    since = _q(query, "since")
    return app.encode_call("job_view", _q(query, "id"), log_since=int(since) if since and since.isdigit() else None)


def r_enc_log(app, body, query):
    since = _q(query, "since")
    return app.encode_call("log_view", _q(query, "id"), int(since) if since and since.isdigit() else 0)


def r_enc_pause(app, body, query):
    return app.encode_call("pause", _b(body).get("id"))


def r_enc_resume(app, body, query):
    return app.encode_call("resume", _b(body).get("id"))


def r_enc_cancel(app, body, query):
    return app.encode_call("cancel", _b(body).get("id"))


def r_enc_discard(app, body, query):
    return app.encode_call("discard", _b(body).get("id"))


def r_enc_test(app, body, query):
    return app.encode_test(_b(body))


def r_enc_browse(app, body, query):
    return app.encode_call("browse", _q(query, "path"))


ROUTES = {
    ("GET", "/api/info"): r_info,
    ("GET", "/api/rig"): r_rig,
    ("GET", "/api/rig/live"): r_rig_live,
    ("GET", "/api/doctor"): r_doctor,
    ("GET", "/api/models"): r_models,
    ("GET", "/api/launch/companions"): r_companions,
    ("POST", "/api/models/dirs"): r_model_dirs,
    ("POST", "/api/models/fits"): r_fits,
    ("GET", "/api/levers"): r_levers,
    ("POST", "/api/levers/lint"): r_levers_lint,
    ("GET", "/api/servers"): r_servers,
    ("POST", "/api/servers"): r_server_save,
    ("DELETE", "/api/servers"): r_server_delete,
    ("GET", "/api/fleet"): r_fleet,
    ("GET", "/api/live"): r_live,
    ("GET", "/api/telemetry"): r_tel_status,
    ("GET", "/api/telemetry/series"): r_tel_series,
    ("GET", "/api/telemetry/history"): r_tel_history,
    ("GET", "/api/telemetry/requests"): r_tel_requests,
    ("GET", "/api/telemetry/csv"): r_tel_csv,
    ("POST", "/api/telemetry/settings"): r_tel_settings,
    ("GET", "/metrics"): r_metrics,
    ("GET", "/api/mqtt"): r_mqtt_status,
    ("POST", "/api/mqtt/settings"): r_mqtt_settings,
    ("POST", "/api/mqtt/test"): r_mqtt_test,
    ("POST", "/api/fleet/adopt"): r_fleet_adopt,
    ("DELETE", "/api/fleet/adopt"): r_fleet_unadopt,
    ("POST", "/api/fleet/control"): r_fleet_control,
    ("GET", "/api/fleet/log"): r_fleet_log,
    ("GET", "/api/engines"): r_engines,
    ("POST", "/api/plan"): r_plan,
    ("POST", "/api/start"): r_start,
    ("POST", "/api/stop"): r_stop,
    ("POST", "/api/restart"): r_restart,
    ("GET", "/api/status"): r_status,
    ("GET", "/api/log"): r_log,
    ("GET", "/api/presets"): r_presets,
    ("POST", "/api/presets"): r_preset_save,
    ("DELETE", "/api/presets"): r_preset_delete,
    ("POST", "/api/attach"): r_attach,
    ("POST", "/api/bench"): r_bench_start,
    ("GET", "/api/bench"): r_bench,
    ("GET", "/api/history"): r_history,
    ("GET", "/api/report/bundle"): r_report_bundle,
    ("POST", "/api/report/send"): r_report_send,
    ("GET", "/api/score/check"): r_score_check,
    ("GET", "/api/thinking"): r_thinking,
    ("POST", "/api/thinking/override"): r_thinking_override,
    ("GET", "/api/thinking/table"): r_thinking_table,
    ("POST", "/api/score/send"): r_score_send,
    ("GET", "/api/score/detail"): r_score_detail,
    ("GET", "/api/user"): r_user,
    ("POST", "/api/user"): r_user_set,
    ("GET", "/api/encode/state"): r_enc_state,
    ("POST", "/api/encode/rescan"): r_enc_rescan,
    ("POST", "/api/encode/use"): r_enc_use,
    ("POST", "/api/encode/add-encoder"): r_enc_add,
    ("POST", "/api/encode/key"): r_enc_key,
    ("POST", "/api/encode/get"): r_enc_get,
    ("POST", "/api/encode/get/cancel"): r_enc_get_cancel,
    ("GET", "/api/encode/runtime"): r_enc_runtime,
    ("POST", "/api/encode/runtime/get"): r_enc_runtime_get,
    ("POST", "/api/encode/runtime/cancel"): r_enc_runtime_cancel,
    ("POST", "/api/encode/update"): r_enc_update,
    ("POST", "/api/encode/licence"): r_enc_licence,
    ("POST", "/api/encode/inspect"): r_enc_inspect,
    ("POST", "/api/encode/plan"): r_enc_plan,
    ("POST", "/api/encode/checks"): r_enc_checks,
    ("POST", "/api/encode/start"): r_enc_start,
    ("GET", "/api/encode/job"): r_enc_job,
    ("GET", "/api/encode/log"): r_enc_log,
    ("POST", "/api/encode/pause"): r_enc_pause,
    ("POST", "/api/encode/resume"): r_enc_resume,
    ("POST", "/api/encode/cancel"): r_enc_cancel,
    ("POST", "/api/encode/discard"): r_enc_discard,
    ("POST", "/api/encode/test"): r_enc_test,
    ("GET", "/api/encode/browse"): r_enc_browse,
    ("GET", "/api/update"): r_pxa_update,
    ("GET", "/api/lib/update"): r_lib_update,
    ("POST", "/api/lib/update/apply"): r_lib_update_apply,
    ("POST", "/api/lib/update/rollback"): r_lib_update_rollback,
    ("POST", "/api/lib/update/settings"): r_lib_update_settings,
    ("GET", "/api/expert-map"): r_expert_map,
    ("POST", "/api/expert-map/rebuild"): r_expert_map_rebuild,
    ("POST", "/api/expert-map/reset"): r_expert_map_reset,
    ("POST", "/api/update/apply"): r_pxa_update_apply,
    ("GET", "/api/health"): r_health,
    ("GET", "/api/gpu/state"): r_gpu_state,
    ("GET", "/api/gpu/profiles"): r_gpu_profiles,
    ("POST", "/api/gpu/profiles"): r_gpu_profile_save,
    ("DELETE", "/api/gpu/profiles"): r_gpu_profile_delete,
    ("POST", "/api/gpu/plan"): r_gpu_plan,
    ("POST", "/api/gpu/apply"): r_gpu_apply,
    ("POST", "/api/gpu/reset"): r_gpu_reset,
    ("POST", "/api/gpu/reserve"): r_gpu_reserve,
    ("POST", "/api/gpu/maintenance"): r_gpu_maintenance,
    ("POST", "/api/gpu/quiet"): r_gpu_quiet,
    ("GET", "/api/gpu/audit"): r_gpu_audit,
    ("GET", "/api/gpu/export"): r_gpu_export,
    ("POST", "/api/gpu/import"): r_gpu_import,
    ("POST", "/api/gpu/schedules"): r_gpu_schedules,
    ("POST", "/api/gpu/supervisor"): r_gpu_supervisor,
    ("GET", "/api/gpu/clocks"): r_gpu_clocks,
    ("POST", "/api/gpu/power"): r_gpu_power,
    ("POST", "/api/gpu/undo"): r_gpu_undo,
    ("POST", "/api/gpu/ui"): r_gpu_ui,
}

def chat_host_policy(app):
    """Host access for the chat agent (v3.1). A Control bound to this machine may offer it; a Control on
    the LAN (--lan) refuses it unless the owner has set host_access_lan in the config, because on the LAN
    the person approving a command is not necessarily the person at the keyboard. The allowlist file and
    the audit log live in the config dir; the audit record is redacted like every other Control record."""
    def gate():
        c = load_config()
        if getattr(app, "lan", False) and not c.get("host_access_lan"):
            return False, ("host access is off over the network. Control is reachable from other machines, "
                           "so this stays off until you set host_access_lan in the Control config.")
        return True, ""
    return CHAT.HostPolicy(gate=gate, allow_path=os.path.join(config_dir(), "host-allow.json"),
                           audit=lambda rec: append_jsonl("host-audit.jsonl", redact_obj(rec)))


if CHAT is not None:                     # the chat agent's routes + agent.js/agent.css: one ROUTES.update()/STATIC.update() block
    try:
        CHAT.register(ROUTES, STATIC, Reply, config_dir, load_config, host=chat_host_policy)
    except Exception as _chat_err:       # noqa: BLE001
        CHAT, CHAT_IMPORT_ERROR = None, f"{_chat_err.__class__.__name__}: {_chat_err}"


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(app, host, port):
    H = type("PXAControlHandler", (Handler,), {"app": app})
    return _Server((host, port), H)


def _quiet(fn):
    try:
        fn()
    except Exception:
        pass


def _lan_addresses():
    out = []
    try:
        r = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
        out = [x for x in r.stdout.split() if ":" not in x]
    except Exception:
        pass
    return out or ["<this-machine>"]


SERVING = {}


def _mins(sec):
    return f"{sec / 60:.0f} min" if sec >= 60 else f"{sec:.0f} s"


def has_desktop(environ=None):
    env = environ if environ is not None else os.environ
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")) or sys.platform in ("darwin", "win32")


def headless_hint(port, lan=False, environ=None, user=None):
    """-> [lines]: the "no desktop here" help, or [] when there is a display or Control is already on
    the LAN. Headless = neither DISPLAY nor WAYLAND_DISPLAY (an SSH session without X forwarding is
    that case). The ssh line uses the server address SSH_CONNECTION carries (3rd field) and the
    login name; where they are unknown it keeps <user> / <this-machine> placeholders."""
    env = environ if environ is not None else os.environ
    if lan or has_desktop(env):
        return []
    parts = (env.get("SSH_CONNECTION") or "").split()
    host = parts[2] if len(parts) >= 4 else "<this-machine>"
    if user is None:
        try:
            import getpass
            user = getpass.getuser()
        except Exception:       # noqa: BLE001
            user = "<user>"
    port = int(port)
    return [
        "  No desktop here, and PXA Control listens on this machine only (127.0.0.1), so another computer cannot reach it as it is.",
        f"  A) rerun with `pxa --gui --lan`: it listens on every interface (0.0.0.0), port {DEFAULT_PORT} by default, plain http,",
        "     behind a random access token. Open the address it prints (it ends in ?token=...); a cookie remembers it after that.",
        "     No other approval step. Use it on a network you trust.",
        f"  B) or from your own computer: ssh -L {port}:127.0.0.1:{port} {user}@{host}   then open http://127.0.0.1:{port}",
        "     The text menu instead: pxa --tui",
    ]


def open_in_browser(url, wait=False):
    """open url in this desktop's browser (in the background unless wait); False when there is no
    desktop here."""
    if not has_desktop():
        return False

    def _open():
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:       # noqa: BLE001
            pass
    if wait:
        _open()
    else:
        threading.Thread(target=_open, daemon=True).start()
    return True


def serve(L, port=DEFAULT_PORT, lan=False, open_browser=True, models_dirs=None, companion=False, idle_s=None):
    """Entry point for `pxa-launch --gui` (and a bare `pxa`). Returns an exit code. companion=True: a
    Control the launcher started in the background next to a server; it closes itself after idle_s
    (CONTROL_IDLE_S) with no server running and no page asking."""
    app = App(L, port=port, lan=lan, models_dirs=models_dirs)
    app.companion = bool(companion)
    if idle_s is None:
        try:                                     # PXA_CONTROL_IDLE_S: the 10-minute rule, in seconds
            idle_s = float(os.environ.get("PXA_CONTROL_IDLE_S") or CONTROL_IDLE_S)
        except ValueError:
            idle_s = CONTROL_IDLE_S
    app.idle_s = idle_s
    host = "0.0.0.0" if lan else "127.0.0.1"
    try:
        srv = make_server(app, host, port)
    except OSError as e:
        print(f"pxa-launch --gui: cannot listen on {host}:{port} ({e}). Pick another with --port.",
              file=sys.stderr)
        return 2
    try:
        run_rec = write_run_record(port, lan, companion)
    except OSError:
        run_rec = None
    SERVING[port] = srv                          # who is listening where (tests stop one through it)
    url = control_url(port, lan, app.token)
    print("=" * 78)
    print(f"PXA Control: {url}")
    if companion:
        print(f"  started next to a server (pid {os.getpid()}); it closes itself {_mins(idle_s)} after the last "
              "server stops with no page open.")
    else:
        print("  the launcher in a browser. Ctrl-C stops it (and the servers it started).")
    if lan:
        print(f"  listening on every interface, port {port}. Access token (keep it private):")
        print(f"    {app.token}")
        ips = _lan_addresses()
        for ip in ips[:3]:
            print(f"  open  http://{ip}:{port}/?token={app.token}")
        if len(ips) > 3:
            print(f"        (or any of this machine's other {len(ips) - 3} addresses, same port and token)")
    else:
        print("  this machine only (--lan to share it)")
        if not companion:
            for _ln in headless_hint(port, lan):
                print(_ln)
    print(f"  config: {config_path()}   levers: {len(app.catalog)} from {app.catalog_src or 'NOT FOUND'}")
    tel = app.start_telemetry()                   # the Live tab's history on disk (PXA_CONTROL_TELEMETRY=0: off)
    if tel is not None:
        print(f"  history: {tel.path} (a sample every {app.tel_cfg['sample_s']:g} s, kept {app.tel_cfg['raw_days']:g} days, "
              f"minutes {app.tel_cfg['rollup_days']:g} days)" + ("" if tel.is_writer else
              f"; another PXA Control (pid {tel.writer_pid()}) records it, this one reads"))
    elif app.tel_error:
        print(f"  history: off ({app.tel_error})")
    mq = app.start_mqtt()                         # Home Assistant over MQTT (PXA_CONTROL_MQTT=1: on)
    if mq is not None:
        print(f"  home assistant: mqtt -> {app.mqtt_cfg['host']}:{app.mqtt_cfg['port']} "
              f"every {app.mqtt_cfg['interval']:g} s under '{app.mqtt_cfg['base']}'")
    elif app.mqtt_error:
        print(f"  home assistant: off ({app.mqtt_error})")
    if app.gpuctl is not None:                    # Profiles: background only when a switch needs it (default: none)
        g = app.gpuctl
        print(f"  GPU profiles: adapter {g.adapter.kind}; changes {'ALLOWED' if g.settings['allow_gpu_control'] else 'off'}"
              f", auto-start {'on' if g.settings['autostart'] else 'off'}, schedules {'on' if g.settings['schedules'] else 'off'}"
              + (f", {len(g.guard.locks())} lock(s) in force" if g.guard.locks() else ""))
        if g.needs_background():
            g.start_background()
            try:
                b = g.boot()
                if b["reapplied"] or b["queued"] or b["errors"]:
                    print(f"  GPU profiles at start: re-applied {len(b['reapplied'])}, queued {len(b['queued'])} auto-start(s)"
                          + (f", errors: {'; '.join(b['errors'])}" if b["errors"] else ""))
            except Exception as e:      # noqa: BLE001
                print(f"  GPU profiles at start: {e.__class__.__name__}: {e}")
    elif app.gpuctl_error:
        print(f"  GPU profiles: off ({app.gpuctl_error})")
    print("=" * 78)
    sys.stdout.flush()
    # read the model headers once in the background: the first scan of a spinning-disk library takes
    # tens of seconds, and the page should not wait on it at the first visit
    threading.Thread(target=lambda: _quiet(app.models), daemon=True).start()
    if open_browser and not lan:
        open_in_browser(f"http://127.0.0.1:{port}/")
    stop_ev = threading.Event()
    if companion:
        def _idle_watch():
            while not stop_ev.wait(min(CONTROL_IDLE_POLL_S, max(0.2, idle_s / 4))):
                try:
                    idle = app.idle_seconds()
                except Exception:       # noqa: BLE001
                    continue
                if idle >= idle_s:
                    print(f"PXA Control: no server running and no page open for {_mins(idle)}: closing.")
                    sys.stdout.flush()
                    srv.shutdown()
                    return
        threading.Thread(target=_idle_watch, daemon=True, name="pxa-control-idle").start()
    try:
        import signal

        def _term(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)      # `docker stop` / kill <pid>: stop the seat too
        # a GUI started in the background (`... &`, nohup) inherits SIGINT=ignored: take it back,
        # so kill -INT stops it (and its servers) the same way Ctrl-C does
        signal.signal(signal.SIGINT, _term)
    except (ValueError, OSError):
        pass
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\npxa-launch --gui: stopping.")
    finally:
        stop_ev.set()
        SERVING.pop(port, None)
        if app.gpuctl is not None:                # first: no auto-start may restart a server we are about to stop
            _quiet(app.gpuctl.stop)
        if run_rec:
            try:
                os.unlink(run_rec)
            except OSError:
                pass
        for seat in list(app.seats.values()):
            if seat.running():
                print("pxa-launch --gui: stopping the server it started (%s, pid %s)." % (seat.sid, seat.proc.pid))
                seat.stop()
        app.encode_shutdown()
        app.live.close()
        _quiet(app.stop_telemetry)
        _quiet(app.stop_mqtt)
        srv.server_close()
    return 0
