"""GPU driver adapters. One shape for every card, whatever reads it:

    {index, uuid, name, sm, power_w, limit_w, min_w, max_w, default_w, persistence, sm_clock, mem_clock, max_sm, max_mem,
     app_sm, app_mem, def_app_sm, def_app_mem, locked, temp_c, slowdown_c, shutdown_c, fan_pct, util_pct,
     mem_used_mib, mem_total_mib, caps: {power_limit, persistence, app_clocks, locked_clocks, fan_control, fan_read}}

A number the driver does not report is None; a control the card does not support has caps[...] False and the page hides it
(P100: power, persistence, application clocks; V100: those plus locked clocks; GTX 1080 Ti: power limit only, no application
clocks, fan read-only). Writes address a card by UUID, never by index, so a reordered bus cannot hit the wrong card.

Every subprocess call is an argv list (never a shell) with a timeout. A missing nvidia-smi is not an error: NullAdapter
answers "no cards, nothing writable" and PXA Control keeps working.
"""
import os
import re
import shutil
import subprocess
import threading
import time

from .errors import DriverError, Invalid

UUID_RE = re.compile(r"^(GPU|MIG)-[A-Za-z0-9\-]{1,64}$")
CAPS_KEYS = ("power_limit", "persistence", "app_clocks", "locked_clocks", "fan_control", "fan_read")

FIELDS = ["index", "uuid", "name", "compute_cap", "power.draw", "power.limit", "power.min_limit", "power.max_limit",
          "power.default_limit", "persistence_mode", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
          "clocks.applications.graphics", "clocks.applications.memory", "clocks.default_applications.graphics",
          "clocks.default_applications.memory", "temperature.gpu", "fan.speed", "utilization.gpu", "memory.used",
          "memory.total"]
KEYS = ["index", "uuid", "name", "sm", "power_w", "limit_w", "min_w", "max_w", "default_w", "persistence", "sm_clock",
        "mem_clock", "max_sm", "max_mem", "app_sm", "app_mem", "def_app_sm", "def_app_mem", "temp_c", "fan_pct",
        "util_pct", "mem_used_mib", "mem_total_mib"]


def num(s):
    """a number from an nvidia-smi cell ('250.00 W', '1328 MHz', '[N/A]', '[Not Supported]') or None."""
    if s is None:
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    m = re.match(r"^\s*(-?[0-9]+(?:\.[0-9]+)?)", str(s))
    return float(m.group(1)) if m else None


def sm_from_name(name):
    n = (name or "").lower()
    for pat, sm in (("h100", 90), ("l40", 89), ("4090", 89), ("a100", 80), ("3090", 86), ("a6000", 86), ("t4", 75),
                    ("2080", 75), ("v100", 70), ("titan v", 70), ("p100", 60), ("p40", 61), ("1080", 61), ("1070", 61)):
        if pat in n:
            return sm
    return None


def detect_caps(c, linux=True):
    """what can be changed on this card, from what it reports (not from a model list)."""
    geforce = "geforce" in (c.get("name") or "").lower() or "titan" in (c.get("name") or "").lower()
    sm = c.get("sm") or 0
    return {
        "power_limit": c.get("min_w") is not None and c.get("max_w") is not None and c["max_w"] > c["min_w"],
        "persistence": bool(linux) and c.get("persistence") is not None,
        "app_clocks": c.get("app_sm") is not None and c.get("app_mem") is not None and not geforce,
        "locked_clocks": sm >= 70,
        "fan_control": False,              # nvidia-smi cannot set fans; a passive Tesla has none to read either
        "fan_read": c.get("fan_pct") is not None,
    }


def finish(c, linux=True):
    c.setdefault("locked", None)
    c.setdefault("slowdown_c", None)
    c.setdefault("shutdown_c", None)
    for k in KEYS:
        c.setdefault(k, None)
    c["caps"] = detect_caps(c, linux)
    return c


class Adapter(object):
    """base: read-only unless a subclass implements the setters."""
    kind = "base"
    available = False
    writable = False

    def __init__(self):
        self._lock = threading.Lock()
        self._cache = (0.0, None)
        self.error = None

    # ---- reads -------------------------------------------------------------------------------
    def _read(self):
        return []

    def list(self, max_age=1.0):
        t, v = self._cache
        if v is not None and time.time() - t < max_age:
            return [dict(x, caps=dict(x["caps"])) for x in v]
        with self._lock:
            v = self._read()
            self._cache = (time.time(), v)
        return [dict(x, caps=dict(x["caps"])) for x in v]

    def get(self, uuid, fresh=False):
        for c in self.list(max_age=0.0 if fresh else 1.0):
            if c["uuid"] == uuid:
                return c
        raise Invalid(f"no card with UUID {uuid}", code="no_card", status=404)

    def supported_clocks(self, uuid):
        """{mem_mhz: [sm_mhz, ...]} the card accepts as application clocks, or {} when unknown."""
        return {}

    def invalidate(self):
        self._cache = (0.0, None)

    def info(self):
        return {"kind": self.kind, "available": self.available, "writable": self.writable, "error": self.error}

    # ---- writes (by UUID) ----------------------------------------------------------------------
    def _no(self, what):
        raise DriverError(f"{what}: the {self.kind} adapter cannot change cards", code="unsupported", status=501)

    def set_power_limit(self, uuid, watts):
        self._no("power limit")

    def set_persistence(self, uuid, on):
        self._no("persistence mode")

    def set_app_clocks(self, uuid, mem, sm):
        self._no("application clocks")

    def reset_app_clocks(self, uuid):
        self._no("application clocks")

    def lock_clocks(self, uuid, lo, hi):
        self._no("locked clocks")

    def reset_locked_clocks(self, uuid):
        self._no("locked clocks")


class NullAdapter(Adapter):
    """no driver here (no nvidia-smi, no NVML): no cards, nothing writable, Control keeps working."""
    kind = "none"

    def __init__(self, why="no NVIDIA driver tools found"):
        super().__init__()
        self.error = why


class NvidiaSmiAdapter(Adapter):
    kind = "nvidia-smi"

    def __init__(self, exe=None, run=None, timeout=10.0, write_timeout=20.0):
        super().__init__()
        self.exe = exe or shutil.which("nvidia-smi") or "nvidia-smi"
        self._run = run or subprocess.run
        self.timeout, self.write_timeout = timeout, write_timeout
        self.available, self.writable = True, True
        self._fields = list(FIELDS)
        self._thresholds = {}          # uuid -> (slowdown, shutdown): static, read once
        self._locked = {}              # uuid -> (lo, hi) as last set here (the driver cannot report it)
        self._clocks = {}

    def _call(self, args, timeout):
        try:
            r = self._run([self.exe] + list(args), capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise DriverError("nvidia-smi not found", code="no_driver", status=503)
        except subprocess.TimeoutExpired:
            raise DriverError(f"nvidia-smi did not answer within {timeout:g} s", code="timeout", status=504)
        if r.returncode != 0:
            msg = ((r.stderr or "") + " " + (r.stdout or "")).strip().splitlines()
            raise DriverError("nvidia-smi: " + (msg[0] if msg else f"exit {r.returncode}")[:300], code="driver")
        return r.stdout or ""

    def _read(self):
        try:
            try:
                out = self._call(["--query-gpu=" + ",".join(self._fields), "--format=csv,noheader"], self.timeout)
            except DriverError as e:
                if "compute_cap" in self._fields and ("not a valid field" in str(e).lower() or "invalid" in str(e).lower()):
                    self._fields.remove("compute_cap")          # an old driver: guess the generation from the name
                    out = self._call(["--query-gpu=" + ",".join(self._fields), "--format=csv,noheader"], self.timeout)
                else:
                    raise
        except DriverError as e:
            self.error = str(e)
            return []
        self.error = None
        rows = []
        for line in out.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < len(self._fields):
                continue
            d = dict(zip(self._fields, f))
            try:
                idx = int(d["index"])
            except ValueError:
                continue
            cc = num(d.get("compute_cap"))
            c = {"index": idx, "uuid": d["uuid"], "name": d["name"].replace("NVIDIA ", ""),
                 "sm": int(round(cc * 10)) if cc else sm_from_name(d["name"]),
                 "power_w": num(d["power.draw"]), "limit_w": num(d["power.limit"]), "min_w": num(d["power.min_limit"]),
                 "max_w": num(d["power.max_limit"]), "default_w": num(d["power.default_limit"]),
                 "persistence": {"enabled": True, "disabled": False}.get(d["persistence_mode"].lower()),
                 "sm_clock": num(d["clocks.sm"]), "mem_clock": num(d["clocks.mem"]), "max_sm": num(d["clocks.max.sm"]),
                 "max_mem": num(d["clocks.max.mem"]), "app_sm": num(d["clocks.applications.graphics"]),
                 "app_mem": num(d["clocks.applications.memory"]),
                 "def_app_sm": num(d["clocks.default_applications.graphics"]),
                 "def_app_mem": num(d["clocks.default_applications.memory"]), "temp_c": num(d["temperature.gpu"]),
                 "fan_pct": num(d["fan.speed"]), "util_pct": num(d["utilization.gpu"]),
                 "mem_used_mib": num(d["memory.used"]), "mem_total_mib": num(d["memory.total"]),
                 "locked": self._locked.get(d["uuid"])}
            sd = self._thresholds.get(c["uuid"])
            if sd:
                c["slowdown_c"], c["shutdown_c"] = sd
            rows.append(finish(c, os.name == "posix"))
        if rows and not self._thresholds:
            self._read_thresholds(rows)
        return rows

    def _read_thresholds(self, rows):
        """the card's own slowdown / shutdown temperatures (static; one nvidia-smi -q per Control lifetime)."""
        try:
            out = self._call(["-q", "-d", "TEMPERATURE"], self.timeout)
        except DriverError:
            self._thresholds = {r["uuid"]: (None, None) for r in rows}
            return
        blocks = re.split(r"\n(?=GPU [0-9A-Fa-f]{4,8}:)", out)
        found = []
        for b in blocks:
            if not re.match(r"\s*GPU [0-9A-Fa-f]{4,8}:", b):
                continue
            sd = re.search(r"GPU Slowdown Temp\s*:\s*([0-9]+)", b)
            sh = re.search(r"GPU Shutdown Temp\s*:\s*([0-9]+)", b)
            found.append((num(sd.group(1)) if sd else None, num(sh.group(1)) if sh else None))
        by_index = sorted(rows, key=lambda r: r["index"])
        for r, t in zip(by_index, found):        # -q lists cards in the same (PCI bus) order as the index
            self._thresholds[r["uuid"]] = t
            r["slowdown_c"], r["shutdown_c"] = t
        for r in rows:
            self._thresholds.setdefault(r["uuid"], (None, None))

    def supported_clocks(self, uuid):
        if uuid in self._clocks:
            return self._clocks[uuid]
        self._uuid(uuid)
        out = {}
        try:
            text = self._call(["-i", uuid, "--query-supported-clocks=mem,gr", "--format=csv,noheader,nounits"], self.timeout)
            for line in text.splitlines():
                p = [num(x) for x in line.split(",")]
                if len(p) == 2 and p[0] and p[1]:
                    out.setdefault(int(p[0]), []).append(int(p[1]))
        except DriverError:
            out = {}
        self._clocks[uuid] = out
        return out

    @staticmethod
    def _uuid(uuid):
        if not isinstance(uuid, str) or not UUID_RE.match(uuid):
            raise Invalid("a card is named by its UUID (GPU-...)")
        return uuid

    def _w(self, args):
        self._call(args, self.write_timeout)
        self.invalidate()

    def set_power_limit(self, uuid, watts):
        self._w(["-i", self._uuid(uuid), "-pl", f"{float(watts):.0f}"])

    def set_persistence(self, uuid, on):
        self._w(["-i", self._uuid(uuid), "-pm", "1" if on else "0"])

    def set_app_clocks(self, uuid, mem, sm):
        self._w(["-i", self._uuid(uuid), "-ac", f"{int(mem)},{int(sm)}"])

    def reset_app_clocks(self, uuid):
        self._w(["-i", self._uuid(uuid), "-rac"])

    def lock_clocks(self, uuid, lo, hi):
        self._w(["-i", self._uuid(uuid), "-lgc", f"{int(lo)},{int(hi)}"])
        self._locked[uuid] = (int(lo), int(hi))

    def reset_locked_clocks(self, uuid):
        self._w(["-i", self._uuid(uuid), "-rgc"])
        self._locked.pop(uuid, None)


class NvmlAdapter(Adapter):
    """the same through pynvml (no subprocess per reading). Used only when pynvml imports and initialises."""
    kind = "nvml"

    def __init__(self, nv=None):
        super().__init__()
        if nv is None:
            import pynvml as nv          # noqa: F401 - ImportError tells make_adapter to fall back
        self.nv = nv
        nv.nvmlInit()
        self.available, self.writable = True, True
        self._locked = {}

    def _g(self, fn, *a):
        try:
            return getattr(self.nv, fn)(*a)
        except Exception:                # noqa: BLE001 - NVMLError_NotSupported and friends: the value is unknown
            return None

    @staticmethod
    def _s(v):
        return v.decode() if isinstance(v, bytes) else (str(v) if v is not None else None)

    def _read(self):
        nv = self.nv
        n = self._g("nvmlDeviceGetCount") or 0
        rows = []
        for i in range(n):
            h = self._g("nvmlDeviceGetHandleByIndex", i)
            if h is None:
                continue
            mw = lambda v: (v / 1000.0) if isinstance(v, (int, float)) else None       # noqa: E731
            cons = self._g("nvmlDeviceGetPowerManagementLimitConstraints", h) or (None, None)
            cc = self._g("nvmlDeviceGetCudaComputeCapability", h)
            util = self._g("nvmlDeviceGetUtilizationRates", h)
            mem = self._g("nvmlDeviceGetMemoryInfo", h)
            pm = self._g("nvmlDeviceGetPersistenceMode", h)
            uuid = self._s(self._g("nvmlDeviceGetUUID", h))
            G, M, SM = getattr(nv, "NVML_CLOCK_GRAPHICS", 0), getattr(nv, "NVML_CLOCK_MEM", 2), getattr(nv, "NVML_CLOCK_SM", 1)
            c = {"index": i, "uuid": uuid, "name": (self._s(self._g("nvmlDeviceGetName", h)) or "GPU").replace("NVIDIA ", ""),
                 "sm": (cc[0] * 10 + cc[1]) if cc else None,
                 "power_w": mw(self._g("nvmlDeviceGetPowerUsage", h)), "limit_w": mw(self._g("nvmlDeviceGetEnforcedPowerLimit", h)),
                 "min_w": mw(cons[0]), "max_w": mw(cons[1]), "default_w": mw(self._g("nvmlDeviceGetPowerManagementDefaultLimit", h)),
                 "persistence": None if pm is None else bool(pm),
                 "sm_clock": self._g("nvmlDeviceGetClockInfo", h, SM), "mem_clock": self._g("nvmlDeviceGetClockInfo", h, M),
                 "max_sm": self._g("nvmlDeviceGetMaxClockInfo", h, SM), "max_mem": self._g("nvmlDeviceGetMaxClockInfo", h, M),
                 "app_sm": self._g("nvmlDeviceGetApplicationsClock", h, G), "app_mem": self._g("nvmlDeviceGetApplicationsClock", h, M),
                 "def_app_sm": self._g("nvmlDeviceGetDefaultApplicationsClock", h, G),
                 "def_app_mem": self._g("nvmlDeviceGetDefaultApplicationsClock", h, M),
                 "temp_c": self._g("nvmlDeviceGetTemperature", h, getattr(nv, "NVML_TEMPERATURE_GPU", 0)),
                 "slowdown_c": self._g("nvmlDeviceGetTemperatureThreshold", h, getattr(nv, "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN", 1)),
                 "shutdown_c": self._g("nvmlDeviceGetTemperatureThreshold", h, getattr(nv, "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN", 0)),
                 "fan_pct": self._g("nvmlDeviceGetFanSpeed", h), "util_pct": getattr(util, "gpu", None),
                 "mem_used_mib": (mem.used >> 20) if mem is not None else None,
                 "mem_total_mib": (mem.total >> 20) if mem is not None else None, "locked": self._locked.get(uuid)}
            rows.append(finish(c, os.name == "posix"))
        return rows

    def _h(self, uuid):
        NvidiaSmiAdapter._uuid(uuid)
        try:
            return self.nv.nvmlDeviceGetHandleByUUID(uuid.encode() if hasattr(uuid, "encode") else uuid)
        except Exception as e:           # noqa: BLE001
            raise DriverError(f"NVML: no card {uuid} ({e.__class__.__name__})", code="no_card", status=404)

    def _do(self, fn, *a):
        try:
            getattr(self.nv, fn)(*a)
        except Exception as e:           # noqa: BLE001
            raise DriverError(f"NVML {fn}: {e}", code="driver")
        self.invalidate()

    def set_power_limit(self, uuid, watts):
        self._do("nvmlDeviceSetPowerManagementLimit", self._h(uuid), int(round(float(watts) * 1000)))

    def set_persistence(self, uuid, on):
        self._do("nvmlDeviceSetPersistenceMode", self._h(uuid), 1 if on else 0)

    def set_app_clocks(self, uuid, mem, sm):
        self._do("nvmlDeviceSetApplicationsClocks", self._h(uuid), int(mem), int(sm))

    def reset_app_clocks(self, uuid):
        self._do("nvmlDeviceResetApplicationsClocks", self._h(uuid))

    def lock_clocks(self, uuid, lo, hi):
        self._do("nvmlDeviceSetGpuLockedClocks", self._h(uuid), int(lo), int(hi))
        self._locked[uuid] = (int(lo), int(hi))

    def reset_locked_clocks(self, uuid):
        self._do("nvmlDeviceResetGpuLockedClocks", self._h(uuid))
        self._locked.pop(uuid, None)


# ---- the mock: tests and test instances of Control --------------------------------------------------------------
MOCK_MODELS = {
    "p100": dict(name="Tesla P100-PCIE-16GB", sm=60, min_w=125.0, max_w=250.0, default_w=250.0, max_sm=1328.0, max_mem=715.0,
                 app_sm=1189.0, app_mem=715.0, def_app_sm=1189.0, def_app_mem=715.0, mem_total_mib=16384.0, persistence=True,
                 slowdown_c=85.0, shutdown_c=88.0, fan_pct=None),
    "v100": dict(name="Tesla V100-PCIE-16GB", sm=70, min_w=100.0, max_w=250.0, default_w=250.0, max_sm=1380.0, max_mem=877.0,
                 app_sm=1245.0, app_mem=877.0, def_app_sm=1245.0, def_app_mem=877.0, mem_total_mib=16384.0, persistence=True,
                 slowdown_c=84.0, shutdown_c=87.0, fan_pct=None),
    "1080ti": dict(name="GeForce GTX 1080 Ti", sm=61, min_w=125.0, max_w=300.0, default_w=250.0, max_sm=1911.0, max_mem=5505.0,
                   app_sm=None, app_mem=None, def_app_sm=None, def_app_mem=None, mem_total_mib=11264.0, persistence=False,
                   slowdown_c=93.0, shutdown_c=96.0, fan_pct=50.0),
}
SM_TO_MODEL = {60: "p100", 61: "1080ti", 70: "v100"}
MOCK_CLOCKS = {"p100": {715: [1328, 1303, 1278, 1252, 1227, 1202, 1189, 1176, 1151, 1126, 1101, 1075, 1050, 1025, 999, 974, 949, 924, 898, 873, 848, 823, 797, 772, 747, 721, 696, 671, 645, 620, 595, 570, 544]},
               "v100": {877: [1380, 1372, 1365, 1357, 1350, 1342, 1335, 1327, 1320, 1312, 1305, 1297, 1290, 1282, 1275, 1267, 1260, 1252, 1245, 1237, 1230, 1222, 1215, 1207, 1200, 1192, 1185, 1177, 1170, 1162, 1155, 1147, 1140, 1132, 1125, 1117, 1110, 1102, 1095, 1087, 1080, 1072, 1065, 1057, 1050, 1042, 1035, 1027, 1020, 1012, 1005, 997, 990, 982, 975, 967, 960, 952, 945, 937]},
               "1080ti": {}}


class MockAdapter(Adapter):
    """an in-memory rig. Writes change it (or fail / drift on request), every call is recorded, nothing leaves Python."""
    kind = "mock"

    def __init__(self, models=None, rows=None, fail=None, drift=None):
        super().__init__()
        self.available, self.writable = True, True
        self.calls = []
        self.fail = dict(fail or {})          # {"power_limit": "message"}: that setter raises
        self.drift = dict(drift or {})        # {"power_limit": 5}: the readback differs by this much (a driver that ignored us)
        self.cards = []
        if rows:                              # from the launcher's (fake) card table: the same UUIDs Control already shows
            for r in rows:
                self.cards.append(self._card(r[0], SM_TO_MODEL.get(r[2], "p100"), r[5], r[1]))
        else:
            for i, m in enumerate(models or ["p100", "p100", "v100", "1080ti", "v100", "p100", "p100"]):
                self.cards.append(self._card(i, m, f"GPU-mock-{i:04d}-{m}"))

    @staticmethod
    def _card(i, model, uuid, name=None):
        base = dict(MOCK_MODELS[model])
        c = dict(base, index=i, uuid=uuid, model=model, power_w=40.0 + 9 * i, limit_w=base["default_w"],
                 sm_clock=base["max_sm"] * 0.4, mem_clock=base["max_mem"], temp_c=34.0 + 3 * i, util_pct=float((i * 17) % 100),
                 mem_used_mib=float(1024 * (i % 4)), locked=None)
        if name:
            c["name"] = name.replace("NVIDIA ", "")
        return c

    def _read(self):
        out = []
        for c in self.cards:
            d = {k: v for k, v in c.items() if k != "model"}
            out.append(finish(d))
        return out

    def supported_clocks(self, uuid):
        c = self._find(uuid)
        return {k: list(v) for k, v in MOCK_CLOCKS.get(c["model"], {}).items()}

    def _find(self, uuid):
        for c in self.cards:
            if c["uuid"] == uuid:
                return c
        raise DriverError(f"no card {uuid}", code="no_card", status=404)

    def _set(self, what, uuid, **vals):
        self.calls.append((what, uuid, dict(vals)))
        c = self._find(uuid)
        if what in self.fail:
            raise DriverError(f"mock {what}: {self.fail[what]}", code="driver")
        d = self.drift.get(what, 0)
        for k, v in vals.items():
            c[k] = (v + d) if (isinstance(v, (int, float)) and not isinstance(v, bool) and d) else v
        self.invalidate()

    def set_power_limit(self, uuid, watts):
        c = self._find(uuid)
        if not (c["min_w"] <= float(watts) <= c["max_w"]):
            raise DriverError(f"mock: {watts} W is outside {c['min_w']:g}..{c['max_w']:g} W", code="driver")
        self._set("power_limit", uuid, limit_w=float(watts))

    def set_persistence(self, uuid, on):
        self._set("persistence", uuid, persistence=bool(on))

    def set_app_clocks(self, uuid, mem, sm):
        if self._find(uuid)["app_sm"] is None:
            raise DriverError("mock: application clocks are not supported on this card", code="unsupported")
        self._set("app_clocks", uuid, app_mem=float(mem), app_sm=float(sm))

    def reset_app_clocks(self, uuid):
        c = self._find(uuid)
        self._set("app_clocks_reset", uuid, app_mem=c["def_app_mem"], app_sm=c["def_app_sm"])

    def lock_clocks(self, uuid, lo, hi):
        if (self._find(uuid).get("sm") or 0) < 70:
            raise DriverError("mock: locked clocks need Volta or newer", code="unsupported")
        self._set("locked_clocks", uuid, locked=(int(lo), int(hi)))

    def reset_locked_clocks(self, uuid):
        self._set("locked_clocks_reset", uuid, locked=None)


def make_adapter(kind=None, rows=None, environ=None):
    """the adapter this Control uses: PXA_CONTROL_GPU_ADAPTER=auto|smi|nvml|mock|none (auto: mock when the launcher's
    PXA_LAUNCH_FAKE_GPUS describes a fake rig, else NVML when pynvml works, else nvidia-smi, else none)."""
    env = os.environ if environ is None else environ
    kind = (kind or env.get("PXA_CONTROL_GPU_ADAPTER") or "auto").strip().lower()
    if kind == "auto" and env.get("PXA_LAUNCH_FAKE_GPUS", "").strip():
        kind = "mock"
    if kind == "mock":
        spec = env.get("PXA_CONTROL_GPU_MOCK", "").strip()
        if spec:
            return MockAdapter(models=[m.strip().lower() for m in spec.split(",") if m.strip().lower() in MOCK_MODELS] or None)
        return MockAdapter(rows=rows) if rows else MockAdapter()
    if kind == "none":
        return NullAdapter("GPU control switched off (PXA_CONTROL_GPU_ADAPTER=none)")
    if kind in ("auto", "nvml"):
        try:
            return NvmlAdapter()
        except Exception as e:           # noqa: BLE001 - no pynvml / no libnvidia-ml: try nvidia-smi
            if kind == "nvml":
                return NullAdapter(f"NVML unavailable: {e.__class__.__name__}")
    if shutil.which("nvidia-smi"):
        return NvidiaSmiAdapter()
    return NullAdapter("nvidia-smi not found and NVML unavailable: cards are read-only")
