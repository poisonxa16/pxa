"""Profiles, presets, schedules, reserved cards and quiet-mode state, in ONE JSON file next to control.json
(gpu_profiles.json, 0600, atomic replace, the previous copy kept as .bak).

    {"schema": 1, "profiles": {id: profile}, "active": {uuid: id}, "schedules": [schedule], "reserved": {uuid: why},
     "maintenance": {"on": bool, "why": str, "since": ts}, "quiet": {...} | null, "boot": {"applied": ts}}

A profile:
    {"id", "name", "preset": "efficiency" | null, "targets": ["GPU-uuid", ...], "notes": "",
     "gpu": {"power": {"mode": "watts" | "pct_default" | "pct_max" | "default", "value": n} | null,
             "persistence": true | false | null,
             "app_clocks": {"mem": MHz, "sm": MHz} | "default" | null,
             "locked_clocks": {"min": MHz, "max": MHz} | "reset" | null},
     "thermal": {"warn_c": n | null, "act_c": n | null, "action": "alert" | "cap_power" | "stop_autostart", "cap_pct": n},
     "autostart": [{"server": sid, "order": n, "delay_s": n, "health_wait_s": n, "restart": "never" | "on-failure",
                    "max_retries": n, "backoff_s": n, "on_boot": bool}]}
null = "leave as it is". The file validates on every write; a file with a NEWER schema is read but never rewritten.
"""
import copy
import json
import os
import re
import threading
import time

from .driver import UUID_RE
from .errors import Invalid

SCHEMA = 1
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-()#:+/]{1,48}$")
SID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
TIME_RE = re.compile(r"^([01][0-9]|2[0-3]):([0-5][0-9])$")
POWER_MODES = ("watts", "pct_default", "pct_max", "default")
THERMAL_ACTIONS = ("alert", "cap_power", "stop_autostart")
RESTART = ("never", "on-failure")
MAX_PROFILES = 64
MAX_AUTOSTART = 16
MAX_SCHEDULES = 32
ENV_NAME_RE = re.compile(r"^PX[AQ]_[A-Z0-9_]{1,64}$")          # the same rule as PXA Control's lever catalog names
ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:=+\-/]{0,200}$")
ARG_RE = re.compile(r"^[A-Za-z0-9_.,:=+\-/\\|*()\[\]^$?]{1,300}$")

# Built-in presets: templates a profile starts from (a percentage, so one preset fits a P100, a V100 and a 1080 Ti).
# The three 'simple' ones are what a beginner sees first; the others live under Advanced.
PRESETS = {
    "quiet": {"name": "Quiet", "simple": True, "order": 1,
              "blurb": "Cooler, quieter and cheaper to run, a little slower. Uses 60 % of each card's stock power.",
              "gpu": {"power": {"mode": "pct_default", "value": 60}, "persistence": True, "app_clocks": None,
                      "locked_clocks": None},
              "thermal": {"warn_c": 75, "act_c": 83, "action": "cap_power", "cap_pct": 50}},
    "balanced": {"name": "Balanced", "simple": True, "order": 2,
                 "blurb": "Most of the speed for clearly less heat and noise. Uses 80 % of each card's stock power.",
                 "gpu": {"power": {"mode": "pct_default", "value": 80}, "persistence": True, "app_clocks": None,
                         "locked_clocks": None},
                 "thermal": {"warn_c": 78, "act_c": 85, "action": "cap_power", "cap_pct": 60}},
    "max": {"name": "Max Speed", "simple": True, "order": 3,
            "blurb": "The most power each card allows: fastest, but hottest and loudest.",
            "gpu": {"power": {"mode": "pct_max", "value": 100}, "persistence": True, "app_clocks": None,
                    "locked_clocks": None},
            "thermal": {"warn_c": 82, "act_c": 87, "action": "cap_power", "cap_pct": 80}},
    "race": {"name": "Benchmark", "simple": False, "order": 4,
             "blurb": "Factory settings on every card, so speed tests are fair and repeatable.",
             "gpu": {"power": {"mode": "default", "value": None}, "persistence": True, "app_clocks": "default",
                     "locked_clocks": "reset"},
             "thermal": {"warn_c": 80, "act_c": None, "action": "alert", "cap_pct": 80}},
    "overnight": {"name": "Overnight", "simple": False, "order": 5,
                  "blurb": "Extra quiet and cool while nobody is watching. Uses 55 % of stock power.",
                  "gpu": {"power": {"mode": "pct_default", "value": 55}, "persistence": True, "app_clocks": None,
                          "locked_clocks": None},
                  "thermal": {"warn_c": 75, "act_c": 82, "action": "cap_power", "cap_pct": 50}},
}


def _num(v, key, lo, hi, allow_none=True, integer=False):
    if v is None or v == "":
        if allow_none:
            return None
        raise Invalid(f"{key} is required")
    if isinstance(v, bool):
        raise Invalid(f"{key} must be a number")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise Invalid(f"{key} must be a number")
    if f != f or not lo <= f <= hi:
        raise Invalid(f"{key} must be between {lo:g} and {hi:g}")
    return int(round(f)) if integer else f


def _only(d, allowed, where):
    if not isinstance(d, dict):
        raise Invalid(f"{where} must be an object")
    extra = sorted(set(d) - set(allowed))
    if extra:
        raise Invalid(f"{where}: unknown field(s) {', '.join(extra)}")


def validate_targets(t):
    if not isinstance(t, list) or not t:
        raise Invalid("targets: pick at least one card (by UUID)")
    if len(t) > 64:
        raise Invalid("targets: at most 64 cards")
    out = []
    for u in t:
        if not isinstance(u, str) or not UUID_RE.match(u):
            raise Invalid(f"targets: {str(u)[:60]!r} is not a card UUID (GPU-...)")
        if u not in out:
            out.append(u)
    return out


def validate_gpu(g):
    if g is None:
        g = {}
    _only(g, ("power", "persistence", "app_clocks", "locked_clocks"), "gpu")
    out = {"power": None, "persistence": None, "app_clocks": None, "locked_clocks": None}
    p = g.get("power")
    if p is not None:
        _only(p, ("mode", "value"), "gpu.power")
        mode = p.get("mode")
        if mode not in POWER_MODES:
            raise Invalid("gpu.power.mode: one of " + ", ".join(POWER_MODES))
        if mode == "watts":
            val = _num(p.get("value"), "power limit (W)", 30, 1000, allow_none=False)
        elif mode in ("pct_default", "pct_max"):
            val = _num(p.get("value"), "power limit (%)", 10, 100, allow_none=False)
        else:
            val = None
        out["power"] = {"mode": mode, "value": val}
    pm = g.get("persistence")
    if pm is not None and not isinstance(pm, bool):
        raise Invalid("gpu.persistence: true, false or null (leave)")
    out["persistence"] = pm
    ac = g.get("app_clocks")
    if ac == "default":
        out["app_clocks"] = "default"
    elif ac is not None:
        _only(ac, ("mem", "sm"), "gpu.app_clocks")
        out["app_clocks"] = {"mem": _num(ac.get("mem"), "application memory clock (MHz)", 100, 20000, False, True),
                             "sm": _num(ac.get("sm"), "application SM clock (MHz)", 100, 5000, False, True)}
    lc = g.get("locked_clocks")
    if lc == "reset":
        out["locked_clocks"] = "reset"
    elif lc is not None:
        _only(lc, ("min", "max"), "gpu.locked_clocks")
        lo = _num(lc.get("min"), "locked clock min (MHz)", 100, 5000, False, True)
        hi = _num(lc.get("max"), "locked clock max (MHz)", 100, 5000, False, True)
        if lo > hi:
            raise Invalid("locked clocks: min must not exceed max")
        out["locked_clocks"] = {"min": lo, "max": hi}
    return out


def validate_thermal(t):
    if t is None:
        t = {}
    _only(t, ("warn_c", "act_c", "action", "cap_pct"), "thermal")
    warn = _num(t.get("warn_c"), "warning temperature (C)", 40, 100, integer=True)
    act = _num(t.get("act_c"), "action temperature (C)", 40, 105, integer=True)
    if warn is not None and act is not None and act < warn:
        raise Invalid("thermal: the action temperature must be at or above the warning one")
    action = t.get("action") or "alert"
    if action not in THERMAL_ACTIONS:
        raise Invalid("thermal.action: one of " + ", ".join(THERMAL_ACTIONS))
    cap = _num(t.get("cap_pct"), "thermal power cap (%)", 10, 100, integer=True)
    return {"warn_c": warn, "act_c": act, "action": action, "cap_pct": cap if cap is not None else 70}


def validate_autostart(a):
    if a is None:
        return []
    if not isinstance(a, list):
        raise Invalid("autostart must be a list")
    if len(a) > MAX_AUTOSTART:
        raise Invalid(f"autostart: at most {MAX_AUTOSTART} servers")
    out, seen = [], set()
    for i, s in enumerate(a):
        _only(s, ("server", "order", "delay_s", "health_wait_s", "restart", "max_retries", "backoff_s", "on_boot",
                  "extra_args", "env"), f"autostart[{i}]")
        xa = s.get("extra_args") or []
        if not isinstance(xa, list) or len(xa) > 32 or any(not isinstance(a, str) or not ARG_RE.match(a) for a in xa):
            raise Invalid(f"autostart[{i}].extra_args: up to 32 engine arguments, no spaces or shell characters "
                          "(PXA Control checks each against its allow-list at start)")
        env = s.get("env") or {}
        if not isinstance(env, dict) or len(env) > 32 or any(
                not ENV_NAME_RE.match(str(k)) or not ENV_VALUE_RE.match(str(v)) for k, v in env.items()):
            raise Invalid(f"autostart[{i}].env: up to 32 PXA_* / PXQ_* settings with plain values")
        sid = s.get("server")
        if not isinstance(sid, str) or not SID_RE.match(sid):
            raise Invalid(f"autostart[{i}].server: a server id from the Servers tab")
        if sid in seen:
            raise Invalid(f"autostart: server {sid} is listed twice")
        seen.add(sid)
        restart = s.get("restart") or "never"
        if restart not in RESTART:
            raise Invalid(f"autostart[{i}].restart: never or on-failure")
        on_boot = s.get("on_boot", False)
        if not isinstance(on_boot, bool):
            raise Invalid(f"autostart[{i}].on_boot: true or false")
        out.append({"server": sid,
                    "order": _num(s.get("order", i + 1), "order", 0, 999, False, True),
                    "delay_s": _num(s.get("delay_s", 0), "delay (s)", 0, 3600, False, True),
                    "health_wait_s": _num(s.get("health_wait_s", 600), "health wait (s)", 5, 7200, False, True),
                    "restart": restart,
                    "max_retries": _num(s.get("max_retries", 3), "max retries", 0, 50, False, True),
                    "backoff_s": _num(s.get("backoff_s", 15), "backoff (s)", 1, 3600, False, True),
                    "on_boot": on_boot, "extra_args": list(xa), "env": {str(k): str(v) for k, v in env.items()}})
    out.sort(key=lambda x: x["order"])
    return out


def validate_profile(p, pid=None):
    _only(p, ("id", "name", "preset", "targets", "notes", "gpu", "thermal", "autostart"), "profile")
    pid = pid or p.get("id")
    if not isinstance(pid, str) or not ID_RE.match(pid):
        raise Invalid("profile id: 1-32 of a-z 0-9 _ - (starting with a letter or digit)")
    name = p.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name.strip() or "!"):
        raise Invalid("profile name: 1-48 of letters, digits, space and _ . - ( ) # : + /")
    preset = p.get("preset")
    if preset is not None and preset not in PRESETS:
        raise Invalid("preset: one of " + ", ".join(PRESETS))
    notes = p.get("notes") or ""
    if not isinstance(notes, str) or len(notes) > 500:
        raise Invalid("notes: text, at most 500 characters")
    return {"id": pid, "name": name.strip(), "preset": preset, "targets": validate_targets(p.get("targets")),
            "notes": notes.replace("\x00", ""), "gpu": validate_gpu(p.get("gpu")),
            "thermal": validate_thermal(p.get("thermal")), "autostart": validate_autostart(p.get("autostart"))}


def validate_schedule(s, profiles):
    _only(s, ("id", "profile", "at", "days", "enabled"), "schedule")
    sid = s.get("id")
    if not isinstance(sid, str) or not ID_RE.match(sid):
        raise Invalid("schedule id: 1-32 of a-z 0-9 _ -")
    if s.get("profile") not in profiles:
        raise Invalid(f"schedule {sid}: no profile {str(s.get('profile'))[:40]!r}")
    at = s.get("at")
    if not isinstance(at, str) or not TIME_RE.match(at):
        raise Invalid(f"schedule {sid}: time as HH:MM (24 h)")
    days = s.get("days", "daily")
    if days != "daily":
        if not isinstance(days, list) or not days or any(not isinstance(d, int) or isinstance(d, bool) or not 0 <= d <= 6 for d in days):
            raise Invalid(f"schedule {sid}: days = 'daily' or a list of 0 (Mon) .. 6 (Sun)")
        days = sorted(set(days))
    en = s.get("enabled", True)
    if not isinstance(en, bool):
        raise Invalid(f"schedule {sid}: enabled = true or false")
    return {"id": sid, "profile": s["profile"], "at": at, "days": days, "enabled": en}


def empty():
    return {"schema": SCHEMA, "profiles": {}, "active": {}, "schedules": [], "reserved": {},
            "maintenance": {"on": False, "why": "", "since": None}, "quiet": None, "boot": {}}


def migrate(d):
    """-> (data, writable). Older files are upgraded in memory; a newer schema is used read-only."""
    if not isinstance(d, dict):
        return empty(), True
    v = d.get("schema", 0)
    if not isinstance(v, int) or v > SCHEMA:
        return d, False
    out = empty()
    for k in out:
        if k in d and k != "schema":
            out[k] = d[k]
    if v < 1:                                    # schema 0 (never shipped): profiles were a list
        if isinstance(d.get("profiles"), list):
            out["profiles"] = {p.get("id"): p for p in d["profiles"] if isinstance(p, dict) and p.get("id")}
    # drop whatever no longer validates instead of refusing to start
    good = {}
    for pid, p in (out["profiles"] or {}).items():
        try:
            good[pid] = validate_profile(p, pid)
        except Invalid:
            continue
    out["profiles"] = good
    out["active"] = {u: p for u, p in (out["active"] or {}).items() if isinstance(u, str) and UUID_RE.match(u) and p in good}
    sch = []
    for s in out["schedules"] or []:
        try:
            sch.append(validate_schedule(s, good))
        except Invalid:
            continue
    out["schedules"] = sch
    out["reserved"] = {u: (w if isinstance(w, dict) else {"why": str(w)[:200]}) for u, w in (out["reserved"] or {}).items()
                       if isinstance(u, str) and UUID_RE.match(u)}
    if not isinstance(out["maintenance"], dict):
        out["maintenance"] = {"on": False, "why": "", "since": None}
    out["schema"] = SCHEMA
    return out, True


class ProfileStore(object):
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.writable = True
        self.error = None
        self.data = self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            return empty()
        except (OSError, ValueError) as e:
            self.error = f"{os.path.basename(self.path)} unreadable ({e.__class__.__name__}); starting empty, the file is kept"
            self.writable = False
            return empty()
        d, self.writable = migrate(raw)
        if not self.writable:
            self.error = f"{os.path.basename(self.path)} has schema {raw.get('schema')!r}, newer than {SCHEMA}: read-only"
        return d

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.data)

    def save(self):
        with self.lock:
            if not self.writable:
                raise Invalid(self.error or "the profile file is read-only", code="read_only", status=409)
            d = os.path.dirname(self.path)
            os.makedirs(d, exist_ok=True)
            if os.path.exists(self.path):
                try:
                    with open(self.path, "rb") as f:
                        old = f.read()
                    fd = os.open(self.path + ".bak", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "wb") as f:
                        f.write(old)
                except OSError:
                    pass
            tmp = self.path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=1, sort_keys=True)
            os.replace(tmp, self.path)

    def mutate(self, fn):
        """fn(data) under the lock, then save; the data is restored if fn or the save fails."""
        with self.lock:
            before = copy.deepcopy(self.data)
            try:
                r = fn(self.data)
                self.save()
                return r
            except Exception:
                self.data = before
                raise

    # ---- profiles -----------------------------------------------------------------------------
    def profiles(self):
        with self.lock:
            return copy.deepcopy(self.data["profiles"])

    def get(self, pid):
        with self.lock:
            p = self.data["profiles"].get(pid)
            if p is None:
                raise Invalid(f"no profile {str(pid)[:40]!r}", code="not_found", status=404)
            return copy.deepcopy(p)

    def put(self, p):
        p = validate_profile(p)

        def f(d):
            if p["id"] not in d["profiles"] and len(d["profiles"]) >= MAX_PROFILES:
                raise Invalid(f"at most {MAX_PROFILES} profiles")
            d["profiles"][p["id"]] = p
        self.mutate(f)
        return p

    def delete(self, pid):
        def f(d):
            if pid not in d["profiles"]:
                raise Invalid(f"no profile {str(pid)[:40]!r}", code="not_found", status=404)
            used = [s["id"] for s in d["schedules"] if s["profile"] == pid]
            if used:
                raise Invalid(f"profile {pid} is used by schedule(s) {', '.join(used)}: remove those first", code="in_use",
                              status=409)
            del d["profiles"][pid]
            d["active"] = {u: x for u, x in d["active"].items() if x != pid}
        self.mutate(f)

    def new_id(self, name):
        base = re.sub(r"[^a-z0-9]+", "-", (name or "profile").lower()).strip("-")[:24] or "profile"
        if not ID_RE.match(base):
            base = "p-" + base
        pid, n = base, 2
        with self.lock:
            while pid in self.data["profiles"]:
                pid, n = f"{base}-{n}", n + 1
        return pid

    def set_active(self, uuids, pid):
        def f(d):
            for u in uuids:
                if pid:
                    d["active"][u] = pid
                else:
                    d["active"].pop(u, None)
        self.mutate(f)

    # ---- schedules, reserved cards, maintenance, quiet -----------------------------------------
    def set_schedules(self, items):
        if not isinstance(items, list) or len(items) > MAX_SCHEDULES:
            raise Invalid(f"schedules: a list of at most {MAX_SCHEDULES}")
        with self.lock:
            profs = self.data["profiles"]
            out = [validate_schedule(s, profs) for s in items]
        ids = [s["id"] for s in out]
        if len(set(ids)) != len(ids):
            raise Invalid("schedules: ids must be unique")

        def f(d):
            d["schedules"] = out
        self.mutate(f)
        return out

    def reserve(self, uuids, on, why="", by="", until=None):
        uuids = validate_targets(uuids)
        if not isinstance(why, str) or len(why) > 200:
            raise Invalid("reason: text, at most 200 characters")

        def f(d):
            for u in uuids:
                if on:
                    d["reserved"][u] = {"why": why or "reserved", "by": str(by)[:80], "since": time.time(), "until": until}
                else:
                    d["reserved"].pop(u, None)
        self.mutate(f)

    def set_maintenance(self, on, why="", by=""):
        if not isinstance(on, bool):
            raise Invalid("on: true or false")

        def f(d):
            d["maintenance"] = {"on": on, "why": str(why or "")[:200], "by": str(by)[:80], "since": time.time() if on else None}
        self.mutate(f)

    def set_quiet(self, q):
        def f(d):
            d["quiet"] = q
        self.mutate(f)

    def set_boot(self, rec):
        def f(d):
            d["boot"] = rec
        self.mutate(f)

    # ---- import / export -----------------------------------------------------------------------
    def export(self):
        with self.lock:
            return {"pxa_gpu_profiles": SCHEMA, "exported": time.time(),
                    "profiles": list(copy.deepcopy(self.data["profiles"]).values()),
                    "schedules": copy.deepcopy(self.data["schedules"])}

    def import_(self, blob, replace=False):
        if not isinstance(blob, dict) or blob.get("pxa_gpu_profiles") not in (1,):
            raise Invalid("not a PXA GPU profile export (pxa_gpu_profiles: 1 expected)")
        items = blob.get("profiles")
        if not isinstance(items, list) or len(items) > MAX_PROFILES:
            raise Invalid(f"profiles: a list of at most {MAX_PROFILES}")
        new = {}
        for p in items:
            v = validate_profile(p)
            new[v["id"]] = v
        sch_in = blob.get("schedules") or []

        def f(d):
            profs = {} if replace else dict(d["profiles"])
            profs.update(new)
            if len(profs) > MAX_PROFILES:
                raise Invalid(f"at most {MAX_PROFILES} profiles")
            sch = [] if replace else list(d["schedules"])
            have = {s["id"] for s in sch}
            for s in sch_in:
                v = validate_schedule(s, profs)
                if v["id"] in have:
                    sch = [x for x in sch if x["id"] != v["id"]]
                sch.append(v)
            d["profiles"] = profs
            d["schedules"] = sch[:MAX_SCHEDULES]
            d["active"] = {u: x for u, x in d["active"].items() if x in profs}
        self.mutate(f)
        return {"profiles": len(new), "schedules": len(sch_in)}
