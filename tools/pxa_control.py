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
        return True
    finally:
        s.close()


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
        if not path_under(model, model_roots):
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
    }
    return out


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
CONTROL_VERSION = "v3"   # what the UI shows as "PXA Control v3" and what reports carry (the release is PXA v3)
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
# live history: the numbers behind the Live tab
# ---------------------------------------------------------------------------------------------
# One rolling sample per card and per server, taken by a small thread that only runs while somebody
# is looking at PXA Control (a viewer asked within LIVE_IDLE_S). Everything is read from places that
# exist on every PXA engine build: nvidia-smi for the cards, and per server /slots, /props, /pxa/stats
# (and /metrics when the server was started with --metrics, which is only used for exact rates).
# Nothing is stored on disk and nothing leaves this machine; a request's prompt text, which /slots
# carries, is dropped on the way in and never reaches the page.
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
                if t0 - self.last_any > LIVE_IDLE_S:
                    self.thread = None
                    return
                fast = t0 - self.last_fast < LIVE_VIEW_S
            self.stop_ev.wait(max(0.2, (LIVE_FAST_S if fast else LIVE_SLOW_S) - (time.time() - t0)))
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
                "cols": {"card": CARD_COLS, "server": SRV_COLS, "req": REQ_COLS, "host": HOST_COLS}, "palette": LIVE_PALETTE,
                "cards": cards, "servers": servers, "host": host}


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
        self.catalog_names = {r["name"] for r in self.catalog}
        self._rig_cache = (0.0, None)
        self._doctor_cache = (0.0, None)
        self._prof_cache = {}
        self._models_cache = None
        self._enc = None
        self._enc_lock = threading.Lock()
        self.live = Live(self)
        self.bench = {"running": False, "lines": [], "result": None, "error": None}
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
        d = {"params": params, "size_label": kvs.get("general.size_label"),
             "name": kvs.get("general.name"), "dominant_type": name}
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
            out.append({"path": e["path"], "file": os.path.basename(e["path"]), "size": e.get("size"),
                        "size_h": self.L.human_bytes(e.get("size")), "family": self.L.family_label(e),
                        "arch": e.get("arch"), "tier": tier or None, "codec": codec,
                        "type": tier or x.get("dominant_type"), "params": x.get("params"),
                        "size_label": x.get("size_label"), "n_ctx_train": e.get("n_ctx_train"),
                        "vision": e.get("vision"), "err": e.get("err"),
                        "kv_bytes_tok": e.get("kv_bytes_tok"), "ple_bytes": e.get("ple_bytes"),
                        "ple": e.get("ple"), "shards_missing": e.get("shards_missing"),
                        "n_expert": e.get("n_expert") or 0, "expert_bytes": e.get("expert_bytes") or 0})
        self._models_cache = {"roots": roots, "models": out, "notes": notes}
        return self._models_cache

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
        if cap.code == 0 and cap.value is not None and req.get("extra_args"):
            plan, cmd, env, cv, prof, ctx = cap.value
            cap.value = (plan, list(cmd) + list(req["extra_args"]), env, cv, prof, ctx)
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
                      "env": dict(env, **req["levers"]), "cards": cv, "ctx": ctx, "port": req["port"],
                      "notes": list(plan.notes), "blockers": list(plan.blockers),
                      "sm": cap.args.sm, "np": cap.args.np, "workload": cap.args.workload,
                      "extra_args": req.get("extra_args") or []})
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

    REPORT_BUDGET_S = {"gpus": 8, "rig": 5, "telemetry": 8}     # per slow source, seconds

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
        build = [ln for ln in log if BUILD_RE.match(ln.strip())][:6]
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
        }
        return {"version": CONTROL_VERSION, "app": "pxa-control", "bundle": redact_obj(bundle), "contact": ""}

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
          "/mark.png": ("mark.png", "image/png"),
          "/favicon.png": ("mark.png", "image/png")}


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
        for cand, from_query in ((self._cookie_token(), False), (self.headers.get("X-PXA-Token"), False),
                                 ((query.get("token") or [None])[0], True)):
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
            return json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            raise Invalid("request body is not JSON")

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
            return self._send(421, {"error": "unexpected Host header"})
        ok, set_cookie = self._auth(query)
        if not ok:
            if method == "GET" and not path.startswith("/api/"):
                return self._send(401, self._login_page(), "text/html; charset=utf-8")
            return self._send(401, {"error": "token required"})
        self.app.last_request = time.time()     # a page is looking: a background Control stays up
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
            return self._send(403, {"error": "cross-origin request refused"})
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
                return self._send(404, {"error": "not found"})
            body = self._json_body() if method in ("POST", "DELETE") else None
            res = route(self.app, body, query)
            return self._send(200, res)
        except Invalid as e:
            return self._send(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._send(500, {"error": f"{e.__class__.__name__}: {e}"})

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
        hdr = {"Accept": self.headers.get("Accept") or "*/*"}
        if method == "POST":
            try:
                n = content_length(self.headers, MAX_BODY * 8)
            except Invalid as e:
                return self._send(413 if "large" in str(e) else 400, {"error": str(e)})
            data = self.rfile.read(n)
            hdr["Content-Type"] = "application/json"
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
            return self._send(r.status, body, ctype)
        self.send_response(r.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
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


def r_info(app, body, query):
    return {"version": 1, "control_version": CONTROL_VERSION, "lan": app.lan, "port": app.port,
            "companion": app.companion, "idle_exit_s": app.idle_s if app.companion else None,
            "catalog_src": app.catalog_src, "levers": len(app.catalog), "kv_types": KV_TYPES,
            "split_modes": SPLIT_MODES, "config": config_path(), "docker": docker_bin() is not None,
            "extra_flags": sorted(EXTRA_FLAGS)}


def r_rig(app, body, query):
    return {"static": app.rig_static(force=_q(query, "force") == "1"), "live": app.rig_live()}


def r_rig_live(app, body, query):
    app.live.touch()
    return app.rig_live()


def r_doctor(app, body, query):
    return app.doctor(force=_q(query, "force") == "1")


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
    return {"src": app.catalog_src,
            "levers": [dict(r, kind=lever_kind(r), default_state=lever_default_state(r)) for r in app.catalog]}


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
    ("POST", "/api/models/dirs"): r_model_dirs,
    ("POST", "/api/models/fits"): r_fits,
    ("GET", "/api/levers"): r_levers,
    ("POST", "/api/levers/lint"): r_levers_lint,
    ("GET", "/api/servers"): r_servers,
    ("POST", "/api/servers"): r_server_save,
    ("DELETE", "/api/servers"): r_server_delete,
    ("GET", "/api/fleet"): r_fleet,
    ("GET", "/api/live"): r_live,
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
}


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
        if not companion and not has_desktop():
            print(f"  no desktop here: from your own computer, `ssh -L {port}:127.0.0.1:{port} <this machine>`, then open "
                  "the address above. The text menu: pxa --tui")
    print(f"  config: {config_path()}   levers: {len(app.catalog)} from {app.catalog_src or 'NOT FOUND'}")
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
        srv.server_close()
    return 0
