"""PXA Control - the launcher's browser front end (`pxa-launch --gui`).

A stdlib-only HTTP server plus one self-contained page (tools/pxa_control_ui/index.html). It is a
FRONT DOOR, like the terminal UI: it collects the same answers a command line would carry, builds
them into the launcher's own argparse namespace (build_parser), and runs the launcher's own
plan_and_build(), doctor(), scan_models() and vram_check(). It decides nothing about the engine.

What it adds on top of the CLI:
  * a live view of the rig (nvidia-smi + sysfs telemetry, --doctor findings);
  * a model library over folders you name, remembered in ~/.config/pxa/control.json;
  * Start / Stop / Restart of ONE seat, with its log streamed to the page (SSE, polling fallback);
  * the running server's /pxa/stats as charts (a local history of the GUI's own requests when the
    server is too old to have /pxa/stats), a short benchmark (tools/pxa-bench.py's prompts, REPS 3);
  * a chat box that streams from the seat's OpenAI endpoint.

SAFETY (a web page drives GPUs, so every edge is closed on purpose):
  * binds 127.0.0.1 unless --lan; with --lan a random token is required (cookie after first visit);
  * Host header must name this server (DNS-rebinding guard); POSTs need a same-origin Origin;
  * no shell passthrough anywhere: the command is the launcher's, argv only, never a shell;
  * env overrides accept only names in the PXA lever catalog, values from a narrow charset;
  * every numeric input is range-checked; model paths must be .gguf files under a configured folder;
  * the engine proxy forwards a fixed list of paths to a LOOPBACK port only.
"""

import hmac
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
    return c


def save_config(c):
    d = config_dir()
    os.makedirs(d, exist_ok=True)
    tmp = config_path() + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)
    os.replace(tmp, config_path())


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


# ---------------------------------------------------------------------------------------------
# launch request validation -> launcher argv
# ---------------------------------------------------------------------------------------------
KV_TYPES = ["f16", "q8_0", "q6_0", "q5_0", "q4_0"]
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
    for p in range(start, start + span):
        if p != avoid and not port_in_use(p, host):
            return p
    raise Invalid(f"no free server port between {start} and {start + span - 1}: set one by hand")


def port_in_use(port, host="127.0.0.1"):
    """True when something already listens on host:port (the engine would refuse with PORT_GUARD a
    few seconds after Start; saying so before anything is spawned is the clearer answer)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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


def validate_launch(body, gpu_indexes, catalog_names, model_roots, gui_port=None, check_model=True):
    """A launch/plan request -> a normalized dict. Raises Invalid with a user-facing reason."""
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
    kv = body.get("kv", "f16") or "f16"
    if kv not in KV_TYPES:
        raise Invalid(f"kv must be one of {', '.join(KV_TYPES)}")
    sm = body.get("sm", "auto") or "auto"
    if sm not in SPLIT_MODES:
        raise Invalid(f"split mode must be one of {', '.join(SPLIT_MODES)}")
    fa = body.get("fa", "auto") or "auto"
    if fa not in FA_MODES:
        raise Invalid("flash attention must be auto, on or off")
    port = _int(body, "port", 0, 65535, 0)
    if port == 0:                   # auto: the first free port from 8080 up (8080 is often taken already)
        port = free_port(DEFAULT_SERVER_PORT, "0.0.0.0" if body.get("expose") else "127.0.0.1", avoid=gui_port)
    elif port < 1024:
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
    }
    return out


def launcher_argv(req):
    """The command line a user would have typed for this request (no shell; argv only)."""
    argv = ["--gpus", ",".join(str(x) for x in req["gpus"]), "--model", req["model"],
            "--port", str(req["port"]), "--host", "0.0.0.0" if req["expose"] else "127.0.0.1",
            "--sm", req["sm"], "--ctk", req["kv"], "--ctv", req["kv"],
            "--yes", "--no-interactive", "--no-tui"]
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
CONTROL_VERSION = "2026.10.1"
BUG_URL_DEFAULT = "https://bugs.pxanetwork.com/v1/report"
REPORT_MAX = 256 * 1024                      # the intake refuses more than this
BANNER_RE = re.compile(r"PXA_(REGISTRY|TSPLIT|AUTO)")
BUILD_RE = re.compile(r"^(build|version|system_info|PXA[ _-]?(build|version))\b", re.I)


def bug_url():
    return os.environ.get("PXA_BUG_URL") or BUG_URL_DEFAULT


_SECRET_PATTERNS = [
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
    def __init__(self, L):
        self.L = L
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
            self.phase = "stopped"
            return True

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
             "request": self.req if r else None,
             "health": None, "hidden_health_lines": self.hidden}
        return d


# ---------------------------------------------------------------------------------------------
# the application: everything the handlers call, testable without a socket
# ---------------------------------------------------------------------------------------------
HEALTH_NOISE_RE = re.compile(r'path="/health"|\] slot data \||\] all slots are idle')
ENGINE_GET = {"health", "props", "v1/models", "pxa/stats", "pxa/speed", "pxa/explain", "slots"}
ENGINE_POST = {"v1/chat/completions", "completion"}


class App(object):
    def __init__(self, L, port=DEFAULT_PORT, lan=False, token=None, models_dirs=None):
        self.L = L
        self.port = port
        self.lan = lan
        self.token = token if token is not None else (secrets.token_urlsafe(18) if lan else None)
        self.seat = Seat(L)
        self.plan_lock = threading.Lock()
        self.catalog, self.catalog_src = load_catalog()
        self.catalog_names = {r["name"] for r in self.catalog}
        self._rig_cache = (0.0, None)
        self._doctor_cache = (0.0, None)
        self._prof_cache = {}
        self._models_cache = None
        self.bench = {"running": False, "lines": [], "result": None, "error": None}
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
    def gpus(self):
        rows, err = self.L.gpu_table()
        return rows or [], err

    def rig_live(self):
        rows, err = self.gpus()
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
                               "narrow": bool(t.get("pcie_width") and t["pcie_width"] < self.L.PCIE_NARROW_BELOW)},
                              **t))
        return {"cards": cards, "error": err, "ts": time.time()}

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
                        "ple": e.get("ple"), "shards_missing": e.get("shards_missing")})
        self._models_cache = {"roots": roots, "models": out, "notes": notes}
        return self._models_cache

    def fits(self, model_entry, card_rows, ctx=0):
        """'fits' | 'tight' | 'no' for this model on these cards, from the launcher's vram_check,
        on IDLE cards (a busy card is shown on the Rig tab, not here). ctx 0 = one 4096 slot."""
        if not card_rows:
            return {"verdict": "no", "why": "no cards"}
        sel = [(g[0], g[1], g[2], g[3], 0, g[5]) for g in card_rows]
        prof = {"kv_bytes_tok": model_entry.get("kv_bytes_tok") or 0,
                "ple_bytes": model_entry.get("ple_bytes") or 0,
                "ple_tensor": "per_layer_token_embd" if model_entry.get("ple") else None,
                "kv_bytes_src": "the launcher's KV/token estimate"}
        plan = self.L.Plan()
        notes = self.L.vram_check(plan, sel, model_entry.get("size") or 0,
                                  ctx or self.L.ANCHOR_CTX_PER_SLOT, prof, True)
        if plan.refusals:
            return {"verdict": "no", "why": plan.refusals[0][1]}
        if any("TIGHT FIT" in n for n in notes):
            return {"verdict": "tight", "why": " ".join(n for n in notes if n.startswith("VRAM"))}
        return {"verdict": "fits", "why": " ".join(n for n in notes if n.startswith("VRAM"))}

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

    # ---- plan / start / stop --------------------------------------------------------------
    def _build(self, req, explain):
        rows, _err = self.gpus()
        a = self.L.build_parser().parse_args(launcher_argv(req))
        a.explain = explain
        saved = {k: os.environ.get(k) for k in req["levers"]}
        with self.plan_lock:
            try:
                os.environ.update(req["levers"])       # the planner reads PXA_* from the environment
                cap = self.L._Capture().run(self.L.plan_and_build, a, rows)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        cap.args = a
        return cap

    def validate(self, body, check_model=True):
        rows, _ = self.gpus()
        return validate_launch(body, {g[0] for g in rows}, self.catalog_names, self.model_roots(),
                               gui_port=self.port, check_model=check_model)

    def plan(self, body):
        req = self.validate(body)
        cap = self._build(req, explain=True)
        ok = cap.code == 0 and cap.value is not None
        d = {"ok": ok, "exit": cap.code, "text": cap.text, "cli": cli_line(req)}
        if ok:
            plan, cmd, env, cv, prof, ctx = cap.value
            d.update({"engine": plan.engine, "command": " ".join(self.L.redact_cmd(cmd)),
                      "env": dict(env, **req["levers"]), "cards": cv, "ctx": ctx,
                      "notes": list(plan.notes), "blockers": list(plan.blockers) + (
                          [f"port {req['port']} is already in use by another program: pick another server port"]
                          if not self.seat.running() and port_in_use(req["port"]) else []),
                      "sm": cap.args.sm, "np": cap.args.np, "workload": cap.args.workload})
        return d

    def start(self, body):
        req = self.validate(body)
        if self.seat.running():
            raise Invalid("a server is already running; stop it first")
        if port_in_use(req["port"], "0.0.0.0" if req["expose"] else "127.0.0.1"):
            raise Invalid(f"port {req['port']} is already in use by another program: pick another server port")
        cap = self._build(req, explain=False)
        if cap.code != 0 or cap.value is None:
            return {"ok": False, "exit": cap.code, "text": cap.text}
        cmd = cap.value[1]
        if not self.L.shutil.which(cmd[0]) and not os.path.exists(cmd[0]):
            return {"ok": False, "exit": 4, "text": cap.text + f"\n{cmd[0]} not found"}
        self.seat.start(cap.value, req, req["levers"])
        st = self.L._load_state()
        st["last_command"] = self.L.redact_cmd(cmd)
        st["last_cards"] = cap.value[3]
        self.L._save_state(st)
        with _cfg_lock:
            c = load_config()
            c["last_launch"] = {k: v for k, v in req.items()}
            save_config(c)
        plan = cap.value[0]
        # the plan as it was when it started: planning again now would see the new seat on the cards
        # and refuse them as busy (R-20), which is what the page used to show after every Start
        started = {"ok": True, "exit": 0, "text": cap.text, "cli": cli_line(req), "engine": plan.engine,
                   "command": " ".join(self.L.redact_cmd(cmd)), "env": dict(cap.value[2], **req["levers"]),
                   "cards": cap.value[3], "ctx": cap.value[5], "notes": list(plan.notes),
                   "blockers": list(plan.blockers), "sm": cap.args.sm, "np": cap.args.np,
                   "workload": cap.args.workload}
        return {"ok": True, "text": cap.text, "plan": started, "status": self.status()}

    def restart(self, body=None):
        body = body or (self.seat.req and dict(self.seat.req))
        if not body:
            raise Invalid("nothing to restart")
        self.seat.stop()
        return self.start(body)

    def engine_port(self):
        if self.seat.req and self.seat.running():
            return self.seat.req["port"]
        p = int(load_config().get("attach_port") or 0)
        return p or None

    def status(self):
        d = self.seat.status()
        port = self.engine_port()
        d["engine_port"] = port
        d["attached"] = bool(port and not self.seat.running())
        if port:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    d["health"] = "ok" if r.status == 200 else f"http {r.status}"
                if d["health"] == "ok" and d["running"]:
                    d["phase"] = "serving"
            except urllib.error.HTTPError as e:
                d["health"] = "loading" if e.code == 503 else f"http {e.code}"
            except Exception:
                d["health"] = "down"
        return d

    # ---- report a problem -----------------------------------------------------------------
    def report_bundle(self):
        """The whole report, already redacted, exactly as the page will show it (and send it)."""
        seat = self.seat
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
                port = self.engine_port()
                if port:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=2) as r:
                        bi = (json.loads(r.read().decode("utf-8", "replace")) or {}).get("build_info")
                        if bi:
                            version = str(bi)[:80]
            except Exception:
                pass
        rows, _err = self.gpus()
        st = self.rig_static()
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
        bundle = {
            "engine": {"version": version, "build_lines": build, "cuda": st.get("cuda"), "container": st.get("container")},
            "banner": banner or ["(no PXA_REGISTRY / PXA_TSPLIT / PXA_AUTO lines in this seat's log yet; start a server first)"],
            "flags": " ".join(self.L.redact_cmd(seat.cmd)) if seat.cmd else "",
            "request": {k: v for k, v in req.items() if k != "model"},
            "gpus": gpus,
            "model": {"filename": os.path.basename(mpath) if mpath else None, "size_bytes": msize},
            "phase": seat.phase,
            "exit_code": seat.proc.returncode if (seat.proc and not seat.running()) else None,
            "log_tail": log[-300:],
            "benchmarks": bench,
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
    def score_payload(self, res, name=""):
        """bench result -> the exact /v1/score payload (redacted; nothing identifying beyond the chosen name)."""
        req = self.seat.req or {}
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
        cmd = " ".join(self.L.redact_cmd(self.seat.cmd)) if self.seat.cmd else ""
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
        rec, err = None, None
        try:
            with urllib.request.urlopen(score_base_url() + "/v1/scores?bracket=" + urllib.parse.quote(key, safe=""), timeout=8) as r:
                rec = (json.loads(r.read().decode("utf-8", "replace")) or {}).get("record")
        except Exception as e:
            err = f"could not reach the board ({e.__class__.__name__})"
        beats = err is None and (rec is None or p["metrics"]["decode_tps"] > rec["decode"])
        return {"available": True, "payload": p, "record": rec, "beats": beats, "error": err, "board": score_base_url()}

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
        req = self.validate(body, check_model=False)
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
        port = self.engine_port()
        if not port:
            raise Invalid("no server: start one on the Launch tab (or attach to a running one)")
        if self.bench["running"]:
            raise Invalid("a benchmark is already running")
        reps = _int(body, "reps", 1, 6, 3)
        warm = _int(body, "warmup", 0, 120, 10)
        mod = self._bench_module()
        if mod is None:
            raise Invalid("tools/pxa-bench.py is not installed next to the launcher")
        self.bench = {"running": True, "lines": [], "result": None, "error": None, "started": time.time()}
        threading.Thread(target=self._bench_run, args=(mod, port, reps, warm), daemon=True).start()
        return {"ok": True}

    def _bench_run(self, mod, port, reps, warm):
        b = self.bench
        url = f"http://127.0.0.1:{port}"

        def say(s):
            b["lines"].append(time.strftime("%H:%M:%S ") + s)
        try:
            props = {}
            try:
                props = mod._get_json(url, "/props", timeout=10)
            except Exception:
                pass
            model = props.get("model_path") or (self.seat.req or {}).get("model") or "?"
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
            res = {"ts": time.time(), "model": model, "cards": (self.seat.req or {}).get("gpus"),
                   "port": port, "reps": reps, "warmup_s": warm, "classes": classes,
                   "greedy512_sha": g.get("sha256"), "greedy512_empty": g.get("empty")}
            append_jsonl("bench.jsonl", res)
            b["result"] = res
            say("done")
        except Exception as e:
            b["error"] = f"{e.__class__.__name__}: {e}"
            say("failed: " + b["error"])
        finally:
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
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seat = self.app.seat
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
        port = self.app.engine_port()
        if not port:
            return self._send(503, {"error": "no server running"})
        url = f"http://127.0.0.1:{port}/{sub}" + (("?" + qs) if qs and method == "GET" else "")
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
    return {"version": 1, "lan": app.lan, "port": app.port, "catalog_src": app.catalog_src,
            "levers": len(app.catalog), "kv_types": KV_TYPES, "split_modes": SPLIT_MODES,
            "config": config_path()}


def r_rig(app, body, query):
    return {"static": app.rig_static(force=_q(query, "force") == "1"), "live": app.rig_live()}


def r_rig_live(app, body, query):
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
    return {"src": app.catalog_src, "levers": app.catalog}


def r_plan(app, body, query):
    return app.plan(body)


def r_start(app, body, query):
    return app.start(body)


def r_stop(app, body, query):
    return {"stopped": app.seat.stop(), "status": app.status()}


def r_restart(app, body, query):
    return app.restart(body if body else None)


def r_status(app, body, query):
    return app.status()


def r_log(app, body, query):
    try:
        since = int(_q(query, "since", "0"))
    except ValueError:
        since = 0
    items = app.seat.lines_since(since)
    return {"seq": items[-1][0] if items else since, "lines": [x[1] for x in items]}


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
    return app.report_bundle()


def r_score_check(app, body, query):
    return app.score_check()


def r_score_send(app, body, query):
    return app.score_send(body)


def r_report_send(app, body, query):
    return app.report_send(body)


ROUTES = {
    ("GET", "/api/info"): r_info,
    ("GET", "/api/rig"): r_rig,
    ("GET", "/api/rig/live"): r_rig_live,
    ("GET", "/api/doctor"): r_doctor,
    ("GET", "/api/models"): r_models,
    ("POST", "/api/models/dirs"): r_model_dirs,
    ("POST", "/api/models/fits"): r_fits,
    ("GET", "/api/levers"): r_levers,
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


def serve(L, port=DEFAULT_PORT, lan=False, open_browser=True, models_dirs=None):
    """Entry point for `pxa-launch --gui`. Returns an exit code."""
    app = App(L, port=port, lan=lan, models_dirs=models_dirs)
    host = "0.0.0.0" if lan else "127.0.0.1"
    try:
        srv = make_server(app, host, port)
    except OSError as e:
        print(f"pxa-launch --gui: cannot listen on {host}:{port} ({e}). Pick another with --port.",
              file=sys.stderr)
        return 2
    print("=" * 78)
    print("PXA Control - the launcher in a browser. Ctrl-C stops it (and the server it started).")
    if lan:
        print(f"  listening on every interface, port {port}. Access token (keep it private):")
        print(f"    {app.token}")
        ips = _lan_addresses()
        for ip in ips[:3]:
            print(f"  open  http://{ip}:{port}/?token={app.token}")
        if len(ips) > 3:
            print(f"        (or any of this machine's other {len(ips) - 3} addresses, same port and token)")
    else:
        print(f"  open  http://127.0.0.1:{port}/     (this machine only; --lan to share it)")
    print(f"  config: {config_path()}   levers: {len(app.catalog)} from {app.catalog_src or 'NOT FOUND'}")
    print("=" * 78)
    sys.stdout.flush()
    # read the model headers once in the background: the first scan of a spinning-disk library takes
    # tens of seconds, and the page should not wait on it at the first visit
    threading.Thread(target=lambda: _quiet(app.models), daemon=True).start()
    if open_browser and not lan:
        def _open():
            try:
                import webbrowser
                webbrowser.open(f"http://127.0.0.1:{port}/")
            except Exception:
                pass
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform == "darwin":
            threading.Thread(target=_open, daemon=True).start()
    try:
        import signal

        def _term(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)      # `docker stop` / kill <pid>: stop the seat too
    except (ValueError, OSError):
        pass
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\npxa-launch --gui: stopping.")
    finally:
        if app.seat.running():
            print("pxa-launch --gui: stopping the server it started (pid %s)." % app.seat.proc.pid)
            app.seat.stop()
        srv.server_close()
    return 0
