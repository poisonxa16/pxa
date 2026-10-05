"""pxa_encode.py - the Encode tab's backend for PXA Control: turn a Hugging Face model into a PXQ / PXQN file in a few clicks.

PXA Control is open source and the quantizer is not, so this file only drives the quantizer's command line as a subprocess
(everything that depends on that command line is in pxa_encode_adapter.py) and the engine's open tools (the converter,
llama-quantize, llama-imatrix). The arithmetic is in pxa_encode_plan.py, the "Get the encoder" download in pxa_encode_pkg.py.

THE WIZARD (one HTTP call each, see the routes in pxa_control.py):
  inspect   source -> size, architecture, support, licence               (step 1)
  plan      cards  -> the tiers, sizes, context, recommendations          (step 2)
  checks    tier   -> disk, RAM, VRAM, tools, licence key, time estimate  (step 3)
  start     -> a job; then pause / resume / cancel / discard              (step 4)
  test      -> load the result in Control's server + bench flow           (step 5)

THE JOB: download -> convert -> reference (Q8_0) -> skeleton -> dump -> hessians -> encode -> verify. With an encoder that reports CLI
version 2 the whole job is ONE call, `pxqe make` (flow "make"): it prints `@pxqe ` JSON lines that drive the stage bars below, keeps its own
state in the work folder, and the SAME command again resumes it. An older encoder keeps the multi-command path (flow "legacy") below.
Every stage is idempotent:
its output is written under a temp name and renamed when whole, a finished stage is recorded in job.json (written atomically after
each change), and a resumed job skips what is done. Resume works after a crash, a reboot or a Control restart: a job that was
running when Control went away is marked `interrupted`, and its leftover process (if any) is stopped before the stage restarts.
Hessians and encode run in ONE encoder process (one licence job, one quota charge); the page still shows two bars.

SAFETY: argv only, never a shell; the licence key reaches the encoder through its environment and nothing else; every line that
reaches the log, the page or an error goes through _scrub (key, Hugging Face token); paths the page sends are checked; a refusal is
always one plain sentence (EncodeError), never a traceback.
"""

import collections
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error

import pxa_encode_adapter as AD
import pxa_encode_pkg as PK
import pxa_encode_plan as PL

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["download", "convert", "quantize", "reference", "skeleton", "dump", "hessians", "encode", "verify"]
STAGE_LABEL = {"download": "Download the model", "convert": "Convert to GGUF", "quantize": "Quantize (classic PXQ)",
               "reference": "Make the reference copy (Q8_0)",
               "skeleton": "Lay out the new file", "dump": "Run the calibration text through the model",
               "hessians": "Measure the weights", "encode": "Encode", "verify": "Check the result"}
LIVE_STATES = ("running", "paused", "queued")
NAME_RE = re.compile(r"^[A-Za-z0-9._\-]{1,96}$")
JOB_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


class EncodeError(ValueError):
    """One plain sentence for the user."""


class _Cancelled(Exception):
    pass


class _StageFailed(Exception):
    def __init__(self, code, message, hint=""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def _scrub(text, extra=()):
    s = PK._scrub(text)
    s = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "<hf-token>", s)
    for x in extra:
        if x and len(x) >= 8:
            s = s.replace(x, "<secret>")
    return s


def data_dir():
    d = os.environ.get("PXA_ENCODE_HOME")
    if d:
        return d
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "pxa", "encode")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _meminfo_available():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return None


def _nearest_existing(p):
    p = os.path.abspath(p)
    while p and not os.path.exists(p):
        n = os.path.dirname(p)
        if n == p:
            break
        p = n
    return p


def _free_bytes(p):
    try:
        return shutil.disk_usage(_nearest_existing(p)).free
    except OSError:
        return None


def _dev(p):
    try:
        return os.stat(_nearest_existing(p)).st_dev
    except OSError:
        return None


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _sha256_file(path, progress=None, cancelled=None):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            if cancelled and cancelled():
                raise _Cancelled()
            h.update(b)
            done += len(b)
            if progress:
                progress(done / float(size or 1))
    return h.hexdigest()


class Rt(object):
    """Runtime-only state of a job (never persisted)."""

    def __init__(self):
        self.thread = None
        self.proc = None
        self.cancel = threading.Event()
        self.paused = False
        self.log = collections.deque(maxlen=5000)
        self.seq = 0
        self.lock = threading.RLock()
        self.last_save = 0.0


class EncodeService(object):
    """Everything behind /api/encode/*. `host` supplies the launcher module and the config; see Host in pxa_control.py."""

    def __init__(self, host, data=None, cells_path=None):
        self.host = host
        self.data = data or data_dir()
        self.lock = threading.RLock()
        self.jobs = {}
        self.rt = {}
        self._found = (0.0, [])
        self._insp = {}
        self.pkg = {"running": False, "phase": "idle", "pct": 0.0, "message": "", "error": None, "edition": None}
        self.rtpkg = {"running": False, "phase": "idle", "pct": 0.0, "message": "", "error": None}      # the GPU runtime download (cuBLAS / cuSOLVER)
        self._rt_offer = None                  # (time, view | None, error | None) of the licence server's GPU runtime pack
        self.cells = PL.load_cells(cells_path or os.path.join(HERE, "pxa_encode_cells.json"))
        self._pymods = {}
        self._kstat = None
        self._lic_seen = {}                 # path -> (time, licence) as the licence server last reported it through the encoder
        self._closed = False
        self.recover()

    # ---- config -------------------------------------------------------------------------------
    def cfg(self):
        c = self.host.cfg_load().get("encode")
        return c if isinstance(c, dict) else {}

    def cfg_set(self, **kw):
        def fn(c):
            e = c.get("encode") if isinstance(c.get("encode"), dict) else {}
            for k, v in kw.items():
                if v is None:
                    e.pop(k, None)
                else:
                    e[k] = v
            c["encode"] = e
        self.host.cfg_update(fn)

    def _key(self):
        k = self.cfg().get("licence_key") or ""
        return k if PK.valid_key(k) else ""

    def _secrets(self):
        return [self._key(), PL.hf_token()]

    def jobs_dir(self):
        return os.path.join(self.data, "jobs")

    def default_work(self):
        return self.cfg().get("work_dir") or os.path.join(self.data, "work")

    def default_out(self):
        c = self.cfg().get("out_dir")
        if c:
            return c
        roots = self.host.model_roots()
        return roots[0] if roots else os.path.join(self.data, "out")

    # ---- encoders (detect, switch, add) -----------------------------------------------------------
    def encoders(self, force=False):
        t, found = self._found
        if found and not force and time.time() - t < 30:
            return found
        c = self.cfg()
        configured = [c.get("preferred")] + list(c.get("extra") or [])
        found = AD.discover([x for x in configured if x], self.host.engine_dirs(), env=self._probe_env())
        pref = c.get("preferred")
        sel = AD.choose(found, pref)
        for f in found:
            f["selected"] = bool(sel and f["path"] == sel["path"])
            seen = self._lic_seen.get(f["path"])
            if f.get("ok") and f.get("licence") and seen and time.time() - seen[0] < 900 and not f["licence"].get("checked"):
                f["licence"] = seen[1]      # a rescan asks the encoder offline: keep what the licence server said a few minutes ago
        self._found = (time.time(), found)
        return found

    def _probe_env(self):
        """PXQE_KEY / PXQE_SERVER for `info --json`: the CLI reads them to tell 'unchecked' from 'no key' (environment only, never argv).
        PXQE_PYTHON names the Python that has the converter's packages, so the `convert` stage's readiness is the one `make` will see."""
        env = {"PXQE_PYTHON": self.convert_python()}      # the interpreter Control hands to `make --python`: its stage readiness is judged on it
        env.update(self._rt_env(None, probing=True))        # where the CUDA libraries may be: the GPU runtime pack and the engine install
        k = self._key()
        if k:
            env.update({"PXQE_KEY": k, "PXQE_SERVER": self.licence_server()})
        return env

    def key_status(self, force=False):
        """The licence key's plan, asked of the licence server through the encoder (`pxqe status`, Pro only), cached for five minutes
        (a failure for one minute, so an offline machine is not made to wait at every step). -> {tier, allowed: set|None, remaining,
        unlimited, lock} or None when unknown. `allowed` lists the PXQN tiers the plan includes; None = no information. `lock` is the licence server's
        lock block (AD.lock_policy), None when it sends none (an older server)."""
        sel, key = self.selected(), self._key()
        if not sel or sel["edition"] != "pro" or not key:
            return None
        hit = self._kstat
        if hit and not force and hit[2] == sel["path"] and time.time() - hit[0] < (300 if hit[1] else 60):
            return hit[1]
        rc, so, se = AD.run_cli(sel["path"], ["status"], env=dict(AD.run_env(sel, key, self.licence_server()), **self._rt_env(sel)), timeout=30)
        out = None
        try:
            o = json.loads(so)
            if rc == 0 and isinstance(o, dict):
                al = o.get("allowed_tiers")
                out = {"tier": str(o.get("tier") or "")[:30], "allowed": {AD.norm_tier(t) for t in al if isinstance(t, str)} if isinstance(al, list) and al else None,
                       "remaining": o.get("remaining"), "unlimited": bool(o.get("unlimited")), "lock": AD.lock_policy(o.get("lock"))}
        except ValueError:
            out = None
        self._kstat = (time.time(), out, sel["path"])
        return out

    def lock_policy(self, info, ks=None):
        """What the licence server allows this key for locked files: AD.lock_policy() of its answer, AD.LOCK_NONE when it answered without a lock block (an
        older server: no locks, same as switched off), None when it could not be asked. Read through the encoder (`pxqe status`); the licence check of an
        encoder that has the block says the same."""
        ks = self.key_status() if ks is None else ks
        if ks:
            return ks.get("lock") or AD.LOCK_NONE
        return ((info or {}).get("licence") or {}).get("lock")

    def lock_view(self, info, ks=None):
        """The "Who can load this file" choices for the selected Pro encoder (AD.lock_view), None for anything else or when no key is saved."""
        if not info or info.get("edition") != "pro" or not self._key():
            return None
        return AD.lock_view(info, self.lock_policy(info, ks))

    def selected(self, force=False):
        for f in self.encoders(force):
            if f.get("selected"):
                return f
        return None

    def rescan(self):
        return self.state(force=True)

    def use(self, path):
        path = str(path or "")
        found = self.encoders(True)
        hit = next((f for f in found if os.path.realpath(f["path"]) == os.path.realpath(path) and f["ok"]), None)
        if not hit:
            raise EncodeError("That encoder is not in the list or does not start. Press Rescan, or use Get the encoder.")
        self.cfg_set(preferred=hit["path"])
        return self.state(force=True)

    def add_encoder(self, path):
        """'Use a Pro encoder...': a path to the command line (pxqe) or to a folder holding it. It is run with `info --json` and
        nothing else; if it answers, it is remembered and its tiers unlock now, with no restart."""
        p = os.path.expanduser(str(path or "").strip())
        if not p or "\x00" in p or len(p) > 4096:
            raise EncodeError("Give the path of the encoder: the pxqe file, or the folder you unpacked it into.")
        if os.path.isdir(p):
            q = AD.cli_in_dir(p)
            if not q:
                raise EncodeError("There is no pxqe command line in that folder.")
            p = q
        if not os.path.isfile(p):
            raise EncodeError("That path does not exist.")
        if p.endswith((".tar.gz", ".tgz", ".zip", ".tar")):
            raise EncodeError("That is a packed download. Unpack it first and pick the pxqe inside, or use Get the encoder, which "
                              "checks the package signature for you.")
        if not re.match(r"^pxqe([._\-][A-Za-z0-9._\-]*)?$", os.path.basename(p)):
            raise EncodeError("That file is not called pxqe, so it is not run. Pick the pxqe command line.")
        info = AD.probe(p, env=self._probe_env())
        if not info["ok"]:
            raise EncodeError("That does not look like a working PXA Quantizer: %s." % (info["error"] or "it did not answer").rstrip("."))
        c = self.cfg()
        extra = [x for x in (c.get("extra") or []) if os.path.realpath(x) != os.path.realpath(p)] + [p]
        self.cfg_set(extra=extra[-8:], preferred=p if info["edition"] == "pro" else c.get("preferred"))
        return self.state(force=True)

    # ---- the key and the licence server ------------------------------------------------------------
    def set_key(self, key):
        k = (key or "").strip()
        if not k:
            self.cfg_set(licence_key=None)
            return self.state(force=True)
        if not PK.valid_key(k):
            raise EncodeError("That does not look like a PXA Quantizer key. It starts with pxk1. and comes from the PXA Network "
                              "Discord (use /encoder).")
        self.cfg_set(licence_key=k)
        return self.state()

    def licence_server(self):
        return PK.licence_url(self.cfg())

    def get_encoder(self, edition, key=None, update=False):
        """Download + verify + install the Free or Pro package in the background; poll state()['pkg']."""
        if edition not in ("free", "pro"):
            raise EncodeError("Choose Free or Pro.")
        if edition == "pro":
            if key:
                self.set_key(key)
            if not self._key():
                raise EncodeError("Paste your key first (it starts with pxk1. and comes from the PXA Network Discord, /encoder).")
        with self.lock:
            if self.pkg.get("running"):
                raise EncodeError("An encoder download is already running.")
            self.pkg = {"running": True, "phase": "asking", "pct": 0.0, "message": "Asking the licence server ...", "error": None,
                        "edition": edition, "cancel": False}
        threading.Thread(target=self._get_encoder_thread, args=(edition,), daemon=True).start()
        return {"ok": True}

    def _set_pkg(self, **kw):
        with self.lock:
            self.pkg.update(kw)

    def _get_encoder_thread(self, edition):
        base = self.licence_server()
        key = self._key()
        try:
            plat, cuda = PK.detect_platform(self.host.cuda_version())
            m = PK.free_latest(base, plat, cuda) if edition == "free" else PK.request_package(base, key, plat, cuda)
            self._set_pkg(phase="downloading", message="Downloading %s %s ..." % (edition.title(), m.get("version") or m["build_id"]))

            def prog(done, total):
                self._set_pkg(pct=(done / float(total)) if total else 0.0)
            work = os.path.join(self.data, "pkg")
            self._set_pkg(phase="downloading")
            dest_info = PK.install_from_manifest(m, work=work, progress=prog, cancelled=lambda: self.pkg.get("cancel"))
            self.cfg_set(update=None)
            found = self.encoders(True)
            pref = dest_info["path"] if edition == "pro" or not any(f["ok"] and f["edition"] == "pro" for f in found) else None
            if pref:
                self.cfg_set(preferred=pref)
                self.encoders(True)
            self._set_pkg(phase="done", running=False, pct=1.0, message="%s encoder %s installed." % (edition.title(), dest_info["version"] or dest_info["build_id"]),
                          installed=dest_info["path"])
        except PK.PackageError as e:
            self._set_pkg(phase="failed", running=False, error={"code": e.code, "message": _scrub(str(e), self._secrets())},
                          message=_scrub(str(e), self._secrets()))
        except Exception as e:      # noqa: BLE001 - never a traceback to the page, and never the key
            self._set_pkg(phase="failed", running=False, error={"code": "failed", "message": "The install failed (%s)." % type(e).__name__},
                          message="The install failed (%s)." % type(e).__name__)

    def licence_check(self):
        """Ask the encoder (and through it the licence server) for the licence as it is now."""
        sel = self.selected()
        if not sel:
            raise EncodeError("No encoder is installed.")
        if sel["edition"] != "pro":
            return self.state()
        self._kstat = None
        r = AD.probe(sel["path"], check_licence=True, env=self._probe_env())
        if not r["ok"]:
            raise EncodeError("The encoder did not answer: %s." % (r["error"] or "no details").rstrip("."))
        sel["licence"] = r.get("licence")
        if r.get("licence") and r["licence"].get("checked"):
            self._lic_seen[sel["path"]] = (time.time(), r["licence"])
        return self.state()

    def cancel_get(self):
        self._set_pkg(cancel=True)
        return {"ok": True}

    # ---- the GPU runtime: the NVIDIA CUDA libraries (cuBLAS, cuSOLVER) the Pro encoder links -----------------------------------------
    def _rt_env(self, info=None, probing=False):
        engs = self.host.engine_dirs()
        return AD.runtime_env(info, PK.installed_runtime(), engs[0] if engs else None, probing)

    def _set_rt(self, **kw):
        with self.lock:
            self.rtpkg.update(kw)

    def runtime_offer(self, force=False):
        """The GPU runtime pack the licence server offers (one small signed call, cached an hour; a failure a minute). -> (view | None, error | None).
        The view is what the button says: {id, size, size_h, unpacked_h, cuda, notice}."""
        hit = self._rt_offer
        if hit and not force and time.time() - hit[0] < (3600 if hit[1] else 60):
            return hit[1], hit[2]
        try:
            plat, cuda = PK.detect_platform(self.host.cuda_version())
            m = PK.runtime_latest(self.licence_server(), plat, cuda)
            view = {"id": m["build_id"], "size": m["size"], "size_h": PL.human_bytes(m["size"]), "unpacked": m["unpacked"],
                    "unpacked_h": PL.human_bytes(m["unpacked"]) if m["unpacked"] else "", "cuda": m["cuda"], "notice": m["notice"], "libs": m["libs"]}
            self._rt_offer = (time.time(), view, None)
        except PK.PackageError as e:
            self._rt_offer = (time.time(), None, _scrub(str(e), self._secrets()))
        except Exception as e:      # noqa: BLE001 - never a traceback to the page
            self._rt_offer = (time.time(), None, "The GPU runtime could not be looked up (%s)." % type(e).__name__)
        return self._rt_offer[1], self._rt_offer[2]

    def runtime_view(self, sel=None, fetch=False, force=False):
        """What the Encode tab shows about the GPU runtime. No network unless `fetch` (then the offer is looked up when it is needed).
        {applies, need, ok, lib, source, missing[], fix, driver, detail, text, installed{id,size,size_h,cuda,dir}|None, offer{...}|None, offer_error, job{...}}"""
        sel = sel if sel is not None else self.selected()
        rt = (sel or {}).get("runtime") or {}
        pro = bool(sel and sel.get("edition") == "pro")
        lib = rt.get("lib") if pro else None
        need = pro and lib == "unloadable" and rt.get("fix") == "runtime-pack"
        pack = PK.installed_runtime()
        offer, oerr = (None, None)
        if fetch and (need or force):
            offer, oerr = self.runtime_offer(force)
        elif self._rt_offer:
            offer, oerr = self._rt_offer[1], self._rt_offer[2]
        job = dict(self.rtpkg)
        job.pop("cancel", None)
        return {"applies": pro, "need": need, "ok": bool(pro and lib == "loadable"), "lib": lib, "source": rt.get("source") or "", "missing": rt.get("missing") or [],
                "fix": rt.get("fix") or "", "driver": rt.get("driver") is not False, "detail": rt.get("detail") or "",
                "text": AD.runtime_missing_text(rt) if lib in ("unloadable", "missing") else "",
                "installed": ({"id": pack["id"], "size": pack["size"], "size_h": PL.human_bytes(pack["size"]), "cuda": pack["cuda"], "dir": pack["dir"]} if pack else None),
                "offer": offer, "offer_error": oerr, "job": job}

    def get_runtime(self):
        """Download + verify + install the GPU runtime in the background; poll state()['runtime']['job']."""
        sel = self.selected()
        if not sel or sel["edition"] != "pro":
            raise EncodeError("Only the Pro encoder needs the GPU runtime. Get the Pro encoder first (Get the encoder).")
        with self.lock:
            if self.rtpkg.get("running"):
                raise EncodeError("The GPU runtime download is already running.")
            self.rtpkg = {"running": True, "phase": "asking", "pct": 0.0, "message": "Asking the licence server ...", "error": None, "cancel": False}
        threading.Thread(target=self._get_runtime_thread, daemon=True).start()
        return {"ok": True}

    def _get_runtime_thread(self):
        try:
            plat, cuda = PK.detect_platform(self.host.cuda_version())
            m = PK.runtime_latest(self.licence_server(), plat, cuda)
            self._set_rt(phase="downloading", message="Downloading the GPU runtime (%s, one time) ..." % PL.human_bytes(m["size"]))

            def prog(done, total):
                if total and done >= total:
                    self._set_rt(phase="installing", pct=1.0, message="Checking and unpacking the GPU runtime ...")
                else:
                    self._set_rt(pct=(done / float(total)) if total else 0.0)
            work = os.path.join(self.data, "pkg")
            pack = PK.install_runtime_from_manifest(m, work=work, progress=prog, cancelled=lambda: self.rtpkg.get("cancel"))
            self._found = (0.0, [])
            sel = self.selected(True)
            rt = (sel or {}).get("runtime") or {}
            if sel and sel["edition"] == "pro" and rt.get("lib") != "loadable":
                self._set_rt(phase="installed_but", running=False, pct=1.0, message="The GPU runtime was installed, but the Pro encoder still cannot load: %s" % (AD.runtime_missing_text(rt)),
                             error={"code": "still_unloadable", "message": AD.runtime_missing_text(rt)})
            else:
                self._set_rt(phase="done", running=False, pct=1.0, message="The GPU runtime is installed (CUDA %s, %s). The Pro encoder can start now." % (pack["cuda"] or "12", PL.human_bytes(pack["size"])))
        except PK.PackageError as e:
            self._set_rt(phase="failed", running=False, error={"code": e.code, "message": _scrub(str(e), self._secrets())}, message=_scrub(str(e), self._secrets()))
        except Exception as e:      # noqa: BLE001 - never a traceback to the page
            self._set_rt(phase="failed", running=False, error={"code": "failed", "message": "The GPU runtime install failed (%s)." % type(e).__name__},
                         message="The GPU runtime install failed (%s)." % type(e).__name__)

    def cancel_runtime(self):
        self._set_rt(cancel=True)
        return {"ok": True}

    def update_check(self, force=False):
        """At most once a day: is there a newer build than the installed ones? -> {available: [{edition, build_id}], ...}"""
        c = self.cfg()
        u = c.get("update") if isinstance(c.get("update"), dict) else {}
        if not force and u.get("checked") and time.time() - float(u["checked"]) < 86400:
            return self._update_view(u)
        installed = self._installed_builds()
        base = self.licence_server()
        res = {"checked": time.time(), "free": None, "pro": None, "error": None}
        try:
            plat, cuda = PK.detect_platform(self.host.cuda_version())
            if "free" in installed:
                res["free"] = {"edition": "free", "build_id": PK.free_latest(base, plat, cuda)["build_id"]}
            key = self._key()
            if key and "pro" in installed:
                cur = sorted(installed["pro"])[-1]
                r = PK.latest_pro(base, key, cur)
                res["pro"] = {"edition": "pro", "build_id": r["latest"], "update_available": r["update_available"]}
        except PK.PackageError as e:
            res["error"] = _scrub(str(e), self._secrets())
        self.cfg_set(update=res)
        return self._update_view(res, installed)

    def _installed_builds(self):
        out = {}
        for f in self.encoders():
            if f["ok"]:
                out.setdefault(f["edition"], set()).add(f["build_id"])
        return out

    def _update_view(self, u, installed=None):
        installed = self._installed_builds() if installed is None else installed
        avail = []
        for ed in ("pro", "free"):
            m = u.get(ed)
            if not m or ed not in installed or m.get("build_id") in installed[ed]:
                continue
            if ed == "pro" and m.get("update_available") is False:
                continue
            avail.append({"edition": ed, "build_id": m["build_id"], "version": "%s-%s" % (ed, m["build_id"])})
        return {"available": avail, "checked": u.get("checked"), "error": u.get("error")}

    # ---- the page's one-call state -------------------------------------------------------------------
    def state(self, force=False):
        if force:
            self._kstat = None              # a Rescan: the licence server's lock switch can change at any time, so ask again
        found = self.encoders(force)
        sel = next((f for f in found if f.get("selected")), None)
        key = self._key()
        c = self.cfg()
        pkg = dict(self.pkg)
        pkg.pop("cancel", None)
        upd = c.get("update") if isinstance(c.get("update"), dict) else {}
        return {"encoders": [self._enc_view(f) for f in found], "selected": sel["path"] if sel else None,
                "edition": sel["edition"] if sel else None, "key_set": bool(key), "key_masked": PK.mask_key(key) if key else "",
                "licence_server": PK.licence_url(c).split("://", 1)[-1], "pkg": pkg, "update": self._update_view(upd) if upd else {"available": []},
                "defaults": {"work_dir": self.default_work(), "out_dir": self.default_out()}, "runtime": self.runtime_view(sel),
                "catalog": [{"key": t["key"], "name": t["name"], "family": t["family"], "bpw": t["bpw"], "cls": t["cls"]} for t in PL.TIERS],
                "hf_token": bool(PL.hf_token()), "jobs": self.job_list(), "how_to_get_pro": {
                    "kofi": "https://ko-fi.com/shatteredrealms1", "discord": "https://discord.gg/EqazvV9tf",
                    "text": "Pro is a supporter feature: support PXA on Ko-fi, then use /encoder in the PXA Network Discord to get your key."}}

    @staticmethod
    def _enc_view(f):
        return {k: f.get(k) for k in ("path", "ok", "error", "edition", "version", "build_id", "tiers", "tier_names", "features", "licence", "runtime", "selected", "cli", "stages")}

    # ---- step 1: inspect the source -------------------------------------------------------------------
    def inspect(self, source, force=False):
        try:
            kind, ident = PL.parse_source(source)
        except PL.SourceError as e:
            raise EncodeError(str(e))
        key = "%s:%s" % (kind, ident)
        hit = self._insp.get(key)
        if hit and not force and time.time() - hit[0] < 600:
            return hit[1]
        try:
            if kind == "hf":
                out = self._inspect_hf(ident)
            elif os.path.isdir(ident):
                out = self._inspect_folder(ident)
            elif os.path.isfile(ident) and ident.endswith(".gguf"):
                out = self._inspect_gguf(ident)
            else:
                raise EncodeError("Pick a folder with a Hugging Face model (config.json and .safetensors files) or a .gguf file.")
        except PL.SourceError as e:
            raise EncodeError(str(e))
        self._insp[key] = (time.time(), out)
        return out

    def converter_names(self):
        p = self.find_convert()
        return PL.converter_arch_names(p) if p else None

    def _common(self, out, cfg, license_id):
        facts = PL.facts_from_config(cfg) if cfg else {}
        out.update({"arch": facts.get("arch"), "layers": facts.get("layers"), "hidden": facts.get("hidden"), "inter": facts.get("inter"),
                    "kv_bytes_tok": facts.get("kv_bytes_tok"), "n_ctx_train": facts.get("n_ctx_train"), "tied": facts.get("tie", False),
                    "moe": bool(facts.get("experts"))})
        out["support"] = PL.arch_support(out["arch"], self.converter_names())
        out["licence"] = PL.classify_licence(license_id)
        out["param_label"] = PL.param_label(out.get("params"))
        out["size_h"] = PL.human_bytes(out.get("size_bytes"))
        return out

    def _inspect_hf(self, repo):
        hf = PL.HF()
        m = hf.model(repo)
        sib = [s for s in (m.get("siblings") or []) if isinstance(s, dict) and isinstance(s.get("rfilename"), str)]
        want = [(s["rfilename"], int(s.get("size") or 0)) for s in sib if "/" not in s["rfilename"] and PL.WANT_FILE_RE.search(s["rfilename"])]
        st = [w for w in want if w[0].endswith(".safetensors")]
        if not st:
            raise EncodeError("%s has no .safetensors weights, so PXA cannot convert it (it may be GGUF-only or an older format)." % repo)
        cfg_text = hf.text(repo, "config.json")
        try:
            cfg = json.loads(cfg_text) if cfg_text else {}
        except ValueError:
            cfg = {}
        lic = None
        cd = m.get("cardData") if isinstance(m.get("cardData"), dict) else {}
        lic = cd.get("license") or next((t.split(":", 1)[1] for t in (m.get("tags") or []) if isinstance(t, str) and t.startswith("license:")), None)
        if isinstance(lic, list):
            lic = lic[0] if lic else None
        dl = sum(s for _, s in want)
        total = (m.get("safetensors") or {}).get("total") if isinstance(m.get("safetensors"), dict) else None
        params = int(total) if isinstance(total, int) and total > 0 else int(sum(s for _, s in st) / 2)
        facts = PL.facts_from_config(cfg) if cfg else {}
        voc, hid = facts.get("vocab") or 0, facts.get("hidden") or 0
        emb = int(voc) * int(hid) if voc and hid else 0
        out = {"kind": "hf", "id": repo, "name": repo.split("/")[-1], "params": params, "emb": emb, "head": 0 if facts.get("tie") else emb,
               "size_bytes": dl, "download_bytes": dl, "files": want, "gated": bool(m.get("gated")), "private": bool(m.get("private")),
               "gguf_kind": None, "notes": []}
        if out["gated"]:
            out["notes"].append("This model is gated: your Hugging Face account must have accepted its licence (set HF_TOKEN).")
        if not cfg:
            out["notes"].append("config.json could not be read, so the architecture and context are unknown.")
        return self._common(out, cfg, lic)

    def _inspect_folder(self, d):
        cfgp = os.path.join(d, "config.json")
        files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
        if not os.path.isfile(cfgp) or not files:
            raise EncodeError("That folder does not hold a Hugging Face model: it needs config.json and .safetensors files.")
        cfg = _read_json(cfgp) or {}
        total = emb = head = 0
        size = 0
        for f in files[:4000]:
            hd = PL.read_safetensors_header(os.path.join(d, f))
            t, e, h = PL.params_from_headers(hd)
            total, emb, head = total + t, emb + e, head + h
            size += os.path.getsize(os.path.join(d, f))
        card = ""
        try:
            with open(os.path.join(d, "README.md"), errors="replace") as fh:
                card = fh.read(8192)
        except OSError:
            pass
        out = {"kind": "folder", "id": d, "name": os.path.basename(d.rstrip("/")) or "model", "params": total, "emb": emb, "head": head,
               "size_bytes": size, "download_bytes": 0, "src_dir": d, "gated": False, "private": False, "gguf_kind": None, "notes": []}
        return self._common(out, cfg, PL.licence_from_card(card))

    def _inspect_gguf(self, path):
        L = self.host.L
        h = L.gguf_header(path)
        if not h.get("ok"):
            raise EncodeError("That file is not a readable GGUF (%s)." % (h.get("err") or "bad header"))
        kv = h.get("kv") or {}
        hist = {}
        params = emb = head = 0
        for nm, ty, ne in h.get("tensors") or []:
            hist[ty] = hist.get(ty, 0) + ne
            params += ne
            if nm == "token_embd.weight":
                emb = ne
            elif nm == "output.weight":
                head = ne
        dom = max(hist.items(), key=lambda kv2: kv2[1])[0] if hist else None
        kind = {30: "bf16", 1: "f16", 8: "q8_0", 0: "f32"}.get(dom)
        if kind is None:
            name = L.PXQ_GGML_TYPE.get(dom) or L.NON_PXQ_GGML_TYPE.get(dom) or ("type %s" % dom)
            raise EncodeError("This GGUF is already quantized (%s). Encoding from it would copy its losses: start from the original "
                              "Hugging Face model, or from a BF16, F16 or Q8_0 GGUF." % name)
        arch = kv.get("general.architecture")
        kvb = None
        try:
            kvb = L.kv_bytes_per_token(kv, arch)[0] if arch else None
        except Exception:      # noqa: BLE001 - an estimate only
            kvb = None
        size = os.path.getsize(path)
        out = {"kind": "gguf", "id": path, "name": re.sub(r"(?i)[-_.](bf16|f16|f32|q8_0)$", "", os.path.basename(path)[:-5]) or "model",
               "params": params, "emb": emb, "head": head, "size_bytes": size, "download_bytes": 0, "gated": False, "private": False,
               "gguf_kind": kind, "notes": [], "arch": arch, "layers": kv.get("%s.block_count" % arch) if arch else None,
               "hidden": kv.get("%s.embedding_length" % arch) if arch else None, "inter": kv.get("%s.feed_forward_length" % arch) if arch else None,
               "kv_bytes_tok": kvb, "n_ctx_train": kv.get("%s.context_length" % arch) if arch else None, "tied": not head,
               "moe": bool(kv.get("%s.expert_count" % arch)) if arch else False}
        if isinstance(out["inter"], list):
            out["inter"] = max(out["inter"] or [0])
        out["support"] = PL.arch_support(arch, None)
        out["licence"] = PL.classify_licence(kv.get("general.license"))
        out["param_label"] = PL.param_label(params)
        out["size_h"] = PL.human_bytes(size)
        return out

    # ---- step 2: the tiers ------------------------------------------------------------------------------
    def cards(self):
        rows, err = self.host.gpus()
        out = []
        for g in rows or []:
            out.append({"index": g[0], "name": str(g[1]).replace("NVIDIA ", ""), "sm": g[2], "vram_mib": g[3], "used_mib": g[4],
                        "free_mib": max(0, g[3] - g[4]), "class": self.host.L.CARD_CLASS.get(g[2], "sm_%s" % g[2])})
        return out, err

    def _target(self, body):
        """The cards the finished model will run on: indexes of this machine's cards, or a typed amount of VRAM (another machine)."""
        local, _ = self.cards()
        if body.get("vram_gib") not in (None, ""):
            try:
                g = float(body["vram_gib"])
            except (TypeError, ValueError):
                raise EncodeError("The memory of the target cards must be a number of GiB.")
            if not 2 <= g <= 4096:
                raise EncodeError("The target memory must be between 2 and 4096 GiB.")
            return [{"vram_mib": int(g * 1024), "class": None, "index": None, "name": "another machine"}], local
        idx = body.get("cards")
        if not isinstance(idx, list) or not idx:
            raise EncodeError("Pick the cards that will run the model (or type another machine's memory).")
        try:
            idx = [int(i) for i in idx][:16]
        except (TypeError, ValueError):
            raise EncodeError("Card numbers must be numbers.")
        by = {c["index"]: c for c in local}
        sel = [by[i] for i in idx if i in by]
        if not sel:
            raise EncodeError("None of those cards exist on this machine.")
        return sel, local

    def plan(self, body):
        src = self.inspect((body or {}).get("source"))
        sel, local = self._target(body or {})
        info = self.selected()
        ks = self.key_status()
        allowed = ks["allowed"] if ks else None
        rows = PL.build_tiers(src, sel, info, self.cells, allowed)
        recs = PL.recommend(rows, info["edition"] if info else None)
        note = None
        if allowed:
            note = "Your key's plan%s includes these PXQN tiers: %s." % ((" (" + ks["tier"] + ")") if ks.get("tier") else "",
                                                                            ", ".join(PL.tier_entry(k)["name"] for k in sorted(allowed)))
        return {"plan_note": note, "source": src, "target": {"cards": [c.get("index") for c in sel], "vram_gib": round(sum(c["vram_mib"] for c in sel) / 1024.0, 1),
                "names": [c["name"] for c in sel]}, "tiers": rows, "recommended": recs,
                "encoder": self._enc_view(info) if info else None, "locked": [r["key"] for r in rows if r["locked"] and r["family"] == "pxqn"],
                "lock": self.lock_view(info, ks)}

    def browse(self, path=None):
        """Folders and .gguf files under `path` (names only; nothing is read), for the Source step's picker."""
        p = os.path.abspath(os.path.expanduser(str(path or "~")))
        if "\x00" in p:
            raise EncodeError("That is not a folder.")
        p = _nearest_existing(p)
        if not os.path.isdir(p):
            p = os.path.dirname(p)
        rows = []
        try:
            with os.scandir(p) as it:
                for e in sorted(it, key=lambda x: x.name.lower()):
                    if e.name.startswith("."):
                        continue
                    try:
                        if e.is_dir():
                            kind = "hfdir" if os.path.isfile(os.path.join(e.path, "config.json")) else "dir"
                            rows.append({"name": e.name, "kind": kind})
                        elif e.name.endswith(".gguf"):
                            rows.append({"name": e.name, "kind": "gguf", "size": e.stat().st_size})
                    except OSError:
                        continue
                    if len(rows) >= 600:
                        break
        except PermissionError:
            raise EncodeError("This folder cannot be read (permission denied).")
        except OSError as e:
            raise EncodeError("This folder cannot be read (%s)." % (e.strerror or "error"))
        here = os.path.isfile(os.path.join(p, "config.json")) and any(f.endswith(".safetensors") for f in os.listdir(p))
        return {"path": p, "parent": os.path.dirname(p) if p != "/" else None, "entries": rows, "is_model": here}

    # ---- step 3: checks ---------------------------------------------------------------------------------
    def stages_for(self, src, info, use_hess, classic=False):
        s = []
        if src["kind"] == "hf":
            s.append("download")
        if src["kind"] in ("hf", "folder"):
            s.append("convert")
        if classic:                                   # the classic tiers: one CPU quantize from the BF16 / source GGUF
            return s + ["quantize", "verify"]
        if src["kind"] != "gguf" or src.get("gguf_kind") != "q8_0":
            s.append("reference")
        s.append("skeleton")
        if use_hess:
            s += ["dump", "hessians"]
        s += ["encode", "verify"]
        return s

    def _use_hess(self, info, tier, body):
        if AD.is_classic(tier):
            return False
        if AD.supports_make(info):          # `make` always runs the calibration text and the statistics for a PXQN tier (there is no round-to-nearest mode)
            return True
        v = (body or {}).get("use_hessians")
        if isinstance(v, bool):
            return v and ("pxqn" in tier or "ldlq" in (info or {}).get("features", []))
        return bool(info and ("ldlq" in info.get("features", []) or (info["edition"] == "pro" and tier.startswith("pxqn"))))

    def find_tool(self, *names):
        dirs = []
        for d in self.host.engine_dirs():
            dirs += [os.path.join(d, "bin"), d]
        sel = self.selected()
        if sel:
            cd = os.path.dirname(os.path.realpath(sel["path"]))
            dirs += [cd, os.path.join(cd, "bin"), os.path.join(cd, "..", "bin")]
        dirs += [x for x in os.environ.get("PATH", "").split(os.pathsep) if x]
        for n in names:
            for d in dirs:
                p = os.path.join(d, n)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
        return None

    def find_convert(self):
        env = os.environ.get("PXA_CONVERT_SCRIPT")
        if env and os.path.isfile(env):
            return env
        cands = []
        for d in self.host.engine_dirs():
            cands += [os.path.join(d, "convert_hf_to_gguf.py"), os.path.join(d, "tools", "convert_hf_to_gguf.py")]
        cands.append(os.path.join(os.path.dirname(HERE), "convert_hf_to_gguf.py"))
        sel = self.selected()
        if sel:
            cd = os.path.dirname(os.path.realpath(sel["path"]))
            cands += [os.path.join(cd, "convert_hf_to_gguf.py"), os.path.join(cd, "tools", "convert_hf_to_gguf.py")]
        for c in cands:
            if os.path.isfile(c):
                return c
        return None

    def convert_python(self):
        return os.environ.get("PXA_CONVERT_PYTHON") or sys.executable

    def python_modules_ok(self):
        py = self.convert_python()
        mods = os.environ.get("PXA_CONVERT_MODULES")
        mods = [m for m in (mods.split(",") if mods is not None else ["torch", "numpy", "safetensors", "transformers"]) if m.strip()]
        if not mods:
            return True, ""
        hit = self._pymods.get((py, tuple(mods)))
        if hit and time.time() - hit[0] < 300:
            return hit[1], hit[2]
        try:
            r = subprocess.run([py, "-c", "import " + ", ".join(mods)], capture_output=True, text=True, timeout=90)
            ok = r.returncode == 0
            miss = ""
            if not ok:
                m = re.search(r"No module named '?([A-Za-z0-9_.]+)", r.stderr or "")
                miss = m.group(1) if m else "a Python package"
        except (OSError, subprocess.TimeoutExpired):
            ok, miss = False, "python"
        self._pymods[(py, tuple(mods))] = (time.time(), ok, miss)
        return ok, miss

    def _req(self, body):
        """Validate the wizard's choices -> a normalised request dict, or EncodeError."""
        body = body or {}
        src = self.inspect(body.get("source"))
        tier = PL.tier_key(body.get("tier") or "")
        if not tier:
            raise EncodeError("Pick a tier first.")
        info = self.selected()
        ks = self.key_status()
        row = next((r for r in PL.build_tiers(src, [{"vram_mib": 1, "class": None}], info, self.cells, ks["allowed"] if ks else None) if r["key"] == tier), None)
        if row is None:
            raise EncodeError("This Control does not know the tier %s." % str(body.get("tier"))[:20])
        work = os.path.abspath(os.path.expanduser(str(body.get("work_dir") or self.default_work())))
        out = os.path.abspath(os.path.expanduser(str(body.get("out_dir") or self.default_out())))
        for p in (work, out):
            if "\x00" in p or len(p) > 4000:
                raise EncodeError("That folder name is not valid.")
        local, _ = self.cards()
        enc_card = body.get("encode_card")
        if enc_card in (None, ""):
            tgt = [c for c in local if c["index"] in (body.get("cards") or [])] or local
            enc_card = max(tgt, key=lambda c: c["free_mib"])["index"] if tgt else None
        else:
            try:
                enc_card = int(enc_card)
            except (TypeError, ValueError):
                raise EncodeError("The encode card must be a card number.")
            if enc_card not in {c["index"] for c in local}:
                raise EncodeError("Card %s does not exist on this machine." % enc_card)
        use_hess = self._use_hess(info, tier, body)
        want = body.get("lock")
        if want not in (None, "") and want not in AD.LOCK_MODES:
            raise EncodeError("Pick who can load the file: only you, any PXA supporter, or anyone.")
        lview = None if AD.is_classic(tier) else self.lock_view(info, ks)
        lmode, lerr = AD.lock_mode_for_run(lview, tier, want or None)
        return {"src": src, "tier": tier, "row": row, "info": info, "work": work, "out": out, "enc_card": enc_card, "local": local,
                "lock_view": lview, "lock": lmode, "lock_error": lerr,
                "flow": "make" if AD.supports_make(info) else "legacy",
                "use_hess": use_hess, "keep": bool(body.get("keep")), "classic": AD.is_classic(tier),
                "stages": self.stages_for(src, info, use_hess, AD.is_classic(tier)),
                "dump_cards": [c["index"] for c in local if c["index"] in (body.get("cards") or [c["index"]])] or [c["index"] for c in local],
                "licence_ack": bool(body.get("licence_ack")), "targets": body.get("cards")}

    def checks(self, body, fresh=True):
        if fresh:
            self._kstat = None                      # the licence server's lock switch can change at any time: ask again at every check
        q = self._req(body)
        src, info, tier, row = q["src"], q["info"], q["tier"], q["row"]
        out = []

        def add(cid, label, status, text, fix="", action=None):
            out.append({"id": cid, "label": label, "status": status, "text": text, "fix": fix} if not action else
                       {"id": cid, "label": label, "status": status, "text": text, "fix": fix, "action": action})

        # the encoder and the licence
        if not info:
            add("encoder", "Encoder", "bad", "No PXA Quantizer is installed on this computer.", "Use Get the encoder (Free needs no key).")
        else:
            add("encoder", "Encoder", "ok", "%s %s (%s build)." % ("PXA Quantizer " + info["edition"].title(), info["version"] or "", info["build_id"]))
            if not row["available"]:
                if row["locked_reason"] == "Supporter feature":
                    add("tier", "Tier", "bad", "%s is a Supporter feature: the Free encoder only makes the classic PXQ tiers." % row["name"],
                        "Pick a classic tier, or get Pro.")
                else:
                    add("tier", "Tier", "bad", "%s: %s." % (row["name"], row["locked_reason"] or "not available"), "Pick another tier.")
            classic = q["classic"]
            if info["edition"] == "pro" and not classic:
                lic = self._licence_now(info)
                st = lic.get("state") if lic else None
                left = lic.get("encodes_left") if lic else None
                if not self._key():
                    add("licence", "Licence key", "bad", "No licence key is saved for the Pro encoder.", "Paste your key in Get the encoder.")
                elif st == "paused":
                    add("licence", "Licence key", "bad", "This key is paused because it was used from too many networks in one day.",
                        "Ask an admin in the PXA Network Discord to resume it. A new key does not help while this one is paused.")
                elif st in ("expired", "revoked", "suspended"):
                    add("licence", "Licence key", "bad", "The licence server refused this key: it is %s." % st,
                        "Get a fresh key from the PXA Network Discord (/encoder), or use the Free encoder.")
                elif st == "refused":
                    add("licence", "Licence key", "bad", "The licence server refused this key%s." % ((": " + lic["reason"]) if lic.get("reason") else ""),
                        "Check the key, or get a fresh one from the PXA Network Discord (/encoder).")
                elif st in ("no_key", "no_server"):
                    add("licence", "Licence key", "bad", "The encoder does not have your key." if st == "no_key" else "The encoder does not know the licence server.",
                        "Paste your key in Get the encoder.")
                elif (left == 0 and not lic.get("unlimited")) or st == "no_quota":
                    add("licence", "Licence key", "bad", "You have no encodes left this month.", "Quota resets monthly; the Free encoder still works.")
                elif st == "offline":
                    add("licence", "Licence key", "warn", "Could not reach the licence server to check your encodes. The run will ask again.", "Check your internet.")
                else:
                    add("licence", "Licence key", "ok", "Key accepted%s%s." % (" for %s" % lic["user"] if lic and lic.get("user") else "",
                        ", unlimited encodes" if lic.get("unlimited") else ((", %d encode%s left" % (left, "" if left == 1 else "s")) if left is not None else "")))
                lv = q.get("lock_view")
                if lv and self._key():
                    if q["lock_error"]:
                        add("lock", "File lock", "bad", q["lock_error"], "Go back to Target and pick one of the choices under Who can load this file.")
                    elif lv["state"] == "unknown":
                        add("lock", "File lock", "warn", "Could not ask the licence server which locks your plan allows. The encoder asks again when it starts; the Done page says "
                            "what lock the file got.", "Check your internet, or press Check again.")
                    elif lv["state"] == "off":
                        add("lock", "File lock", "ok", "This file will be written without a lock: anyone with a PXA engine can load it. %s." % AD.LOCK_OFF_TEXT)
                    else:
                        mode = q["lock"]
                        add("lock", "File lock", "ok", {"personal": "Only you will be able to load this file. It is tied to your PXA account.",
                                                         "supporters": "Any active PXA supporter will be able to load this file, and you can too.",
                                                         "open": "Anyone with a PXA engine will be able to load this file: it will not be locked."}[mode],
                            ("A locked file loads in PXA %s or newer." % AD.LOCK_MIN_ENGINE) if mode != "open" else "")
                rt = info.get("runtime") or {}
                if rt.get("lib") in ("unloadable", "missing"):
                    if rt.get("fix") == "runtime-pack":
                        offer, _err = self.runtime_offer()
                        add("runtime", "GPU runtime", "bad", AD.runtime_missing_text(rt),
                            "Download the GPU runtime once (%s) and it is used from then on, or install the CUDA 12 toolkit. A classic tier needs neither." % (
                                ("about " + offer["size_h"]) if offer else "about 1 GB"), action="runtime")
                    elif rt.get("fix") == "driver":
                        add("runtime", "NVIDIA driver", "bad", AD.runtime_missing_text(rt), "Install the NVIDIA driver (version 525 or newer), or pick a classic tier.")
                    else:
                        add("runtime", "Pro encoder", "bad", AD.runtime_missing_text(rt), "Reinstall the encoder (Get the encoder), or pick a classic tier.")
            elif info["edition"] == "pro":
                add("licence", "Licence", "ok", "The classic tiers need no licence and use no encode.")
            else:
                add("licence", "Licence", "ok", "The Free encoder needs no key.")
        fr = row.get("tier_fraction")
        if q["classic"] and fr is not None and fr < 0.55:
            add("composition", "Small model", "warn", "This model is small: about %d%% of the file would be in the %s tier, and the quantizer refuses a file under 50%%." % (round(100 * fr), row["name"]),
                "If it refuses, pick a higher tier or a larger model.")
        if q["classic"] and src["kind"] == "gguf" and src.get("gguf_kind") == "q8_0":
            add("double", "Source quality", "warn", "This GGUF is already Q8_0. Quantizing from it is a second lossy pass, so the result is a little worse than "
                "quantizing the original BF16 model once.", "For the best file, start from the Hugging Face model or a BF16 / F16 GGUF.")
        # the source's own licence
        sl = src["licence"]
        if sl["level"] == "noderivs":
            add("src_licence", "Model licence", "bad" if not q["licence_ack"] else "warn",
                "%s Making a copy for yourself may be fine; sharing it is not." % sl["text"], "Tick \"I will not share the result\" to continue.")
        elif sl["level"] in ("noncommercial", "unknown", "conditions"):
            add("src_licence", "Model licence", "warn", sl["text"])
        else:
            add("src_licence", "Model licence", "ok", sl["text"])
        if src["support"]["level"] == "unsupported":
            add("arch", "Architecture", "bad", src["support"]["text"], "Pick a Qwen, Gemma or GLM model.")
        elif src["support"]["level"] in ("untested", "unknown"):
            add("arch", "Architecture", "warn", src["support"]["text"])
        else:
            add("arch", "Architecture", "ok", "%s: %s" % (src["arch"] or "", src["support"]["text"]))
        # disk
        tb = row["size_bytes"] or 0
        sz = PL.stage_sizes(src, tb, q["stages"])
        peak, outb = sz["peak"], tb
        same = _dev(q["work"]) == _dev(q["out"])
        fw, fo = _free_bytes(q["work"]), _free_bytes(q["out"])
        if same:
            need, free = peak + outb, fw
            ok = free is None or free >= need * 1.05
            add("disk", "Disk space", "ok" if ok else "bad",
                "Needs about %s at the peak (download, intermediate files and the result), %s free on %s." % (PL.human_bytes(need), PL.human_bytes(free or 0), q["work"]) if not ok
                else "About %s needed at the peak; %s free." % (PL.human_bytes(need), PL.human_bytes(free or 0)),
                "Pick another work folder in Advanced, or free some space." if not ok else "")
        else:
            problems = []
            if fw is not None and fw < peak * 1.05:
                problems.append("work folder needs %s, has %s" % (PL.human_bytes(peak), PL.human_bytes(fw)))
            if fo is not None and fo < outb * 1.05:
                problems.append("output folder needs %s, has %s" % (PL.human_bytes(outb), PL.human_bytes(fo)))
            add("disk", "Disk space", "bad" if problems else "ok",
                ("Not enough space: " + "; ".join(problems) + ".") if problems else
                "Work folder %s of %s free; output folder %s." % (PL.human_bytes(fw or 0), PL.human_bytes(peak), PL.human_bytes(fo or 0)),
                "Pick another folder in Advanced." if problems else "")
        # RAM
        ram_free = _meminfo_available()
        ram_need = (3 << 30) if q["classic"] else PL.ram_need_bytes(src.get("hidden"), src.get("inter"))
        if ram_free is not None:
            st = "ok" if ram_free >= ram_need else ("warn" if ram_free >= 0.6 * ram_need else "bad")
            add("ram", "Memory (RAM)", st, "%s free, about %s needed." % (PL.human_bytes(ram_free), PL.human_bytes(ram_need)),
                "Close other programs." if st != "ok" else "")
        # VRAM on the encode card
        local = q["local"]
        need_v = PL.encode_vram_bytes(src.get("hidden"), src.get("inter"))
        card = next((c for c in local if c["index"] == q["enc_card"]), None)
        if q["classic"]:
            add("vram", "Graphics card", "ok", "Not needed: the classic quantizer runs on the CPU.")
        elif card is None:
            add("vram", "Graphics card", "bad", "No graphics card was found to run the encode on.", "The encoder needs an NVIDIA card with about %s free." % PL.human_bytes(need_v))
        else:
            free_b = card["free_mib"] * PL.MIB
            st = "ok" if free_b >= need_v * 1.15 else "bad"
            add("vram", "Graphics card", st, "Card %d (%s) has %s free; the encode needs about %s." % (card["index"], card["name"], PL.human_bytes(free_b), PL.human_bytes(need_v)),
                "Stop whatever is using the card (Servers tab) or pick another card in Advanced." if st == "bad" else "")
        # tools
        if q["flow"] == "make":
            miss = self._make_missing(q, info)
            add("tools", "Tools", "bad" if miss else "ok", ("These steps cannot run on this computer: " + "; ".join(miss) + ".") if miss else
                "Everything this tier needs is in the encoder package.",
                "Install what is listed, then press Rescan on the Encode tab." if miss else "")
        else:
            miss = []
            conv = None
            if "convert" in q["stages"]:
                conv = self.find_convert()
                if not conv:
                    miss.append("the converter (convert_hf_to_gguf.py)")
                else:
                    okm, mm = self.python_modules_ok()
                    if not okm:
                        miss.append("the Python package %s for the converter" % mm)
            if "reference" in q["stages"] or "skeleton" in q["stages"]:
                qt = self.find_tool("pxq-quantize", "llama-quantize")
                if not qt:
                    miss.append("llama-quantize")
                elif "skeleton" in q["stages"] and not AD.quantize_tool_knows_pxqn(qt):
                    miss.append("a quantize tool that can lay out PXQN files (the engine build that comes with the Pro encoder)")
            if "dump" in q["stages"]:
                if not self.find_tool("llama-imatrix"):
                    miss.append("llama-imatrix (the activation dump)")
                if not AD.calibration_file(info["path"] if info else None):
                    miss.append("the calibration text")
            add("tools", "Tools", "bad" if miss else "ok", ("Missing: " + ", ".join(miss) + ".") if miss else
                ("Converter found." if q["classic"] else "Converter, quantizer and dump tools found."),
                "These come with the PXA engine build and the encoder package; see the Encode page of the PXA docs." if miss else "")
        est = PL.time_estimate(src, q["stages"], n_gpu_cards=max(1, len(q["dump_cards"])))
        bad = [c for c in out if c["status"] == "bad"]
        return {"checks": out, "can_start": not bad, "refusal": bad[0]["text"] if bad else None,
                "estimate": {"low_s": est["low"], "high_s": est["high"], "text": "%s to %s" % (PL.human_dur(est["low"]), PL.human_dur(est["high"])),
                             "stages": {k: [v[0], v[1]] for k, v in est["stages"].items()}},
                "disk": {"peak": peak, "output": outb, "peak_h": PL.human_bytes(peak), "output_h": PL.human_bytes(outb), "intermediates_h": PL.human_bytes(sz["keep_extra"])},
                "stages": [{"id": s, "label": STAGE_LABEL[s]} for s in q["stages"]], "work_dir": q["work"], "out_dir": q["out"],
                "use_hessians": q["use_hess"], "encode_card": q["enc_card"], "lock": q["lock"]}

    def _licence_now(self, info):
        """The pro licence as the encoder itself reports it, asking the licence server (the one network call of the checks)."""
        r = AD.probe(info["path"], check_licence=True, env=self._probe_env())
        if r["ok"] and r.get("licence"):
            info["licence"] = r["licence"]
            if r["licence"].get("checked"):
                self._lic_seen[info["path"]] = (time.time(), r["licence"])
        return info.get("licence") or {}

    # ---- jobs: create, run, persist -----------------------------------------------------------------------
    def job_list(self):
        with self.lock:
            return [self._brief_job(j) for j in sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)[:12]]

    def _brief_job(self, j):
        return {"id": j["id"], "status": j["status"], "name": j["params"].get("name"), "tier": j["params"].get("tier_name"),
                "pct": self._overall(j)[0], "created": j["created"], "updated": j.get("updated"), "error": j.get("error"),
                "result": j.get("result")}

    def _jobfile(self, jid):
        return os.path.join(self.jobs_dir(), jid, "job.json")

    def _save(self, j, force=True):
        if self._closed:
            return
        rt = self.rt.get(j["id"])
        now = time.time()
        if rt and not force and now - rt.last_save < 1.5:
            return
        j["updated"] = now
        if rt:
            rt.last_save = now
        j["pct"], j["eta_s"] = self._overall(j)
        try:
            _write_json(self._jobfile(j["id"]), j)
        except OSError:
            pass

    def shutdown(self, kill=True):
        """Control is exiting (Ctrl-C, docker stop): every live job is written down as `interrupted` so the next start can
        resume it, and its encoder / tool process is stopped. Nothing writes after this."""
        with self.lock:
            live = [j for j in self.jobs.values() if j["status"] in LIVE_STATES]
        self._closed = True
        for j in live:
            rt = self.rt.get(j["id"])
            p = rt.proc if rt else None
            j["status"] = "interrupted"
            j["error"] = {"code": "interrupted", "message": "Control was closed while this was running.", "hint": "Resume continues from the last finished step."}
            for s in j["stages"]:
                if s["status"] == "running":
                    s["status"] = "pending"
            j["pid"] = None
            try:
                _write_json(self._jobfile(j["id"]), j)
            except OSError:
                pass
            if rt:
                rt.cancel.set()
            if kill and p is not None and p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGCONT)
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except OSError:
                    pass

    def recover(self):
        """Load every job on disk. One that was running when Control (or the machine) went away is `interrupted` and can be resumed."""
        try:
            names = os.listdir(self.jobs_dir())
        except OSError:
            return
        for n in names:
            j = _read_json(self._jobfile(n))
            if not isinstance(j, dict) or j.get("id") != n or not JOB_ID_RE.match(n):
                continue
            if j.get("status") in LIVE_STATES:
                j["status"] = "interrupted"
                j["error"] = {"code": "interrupted", "message": "Control was closed (or the computer restarted) while this was running.",
                              "hint": "Resume continues from the last finished step."}
                for s in j["stages"]:
                    if s["status"] == "running":
                        s["status"] = "pending"
            self.jobs[n] = j
            rt = self.rt[n] = Rt()
            try:                                            # the log of the earlier run stays in front of the new lines
                with open(os.path.join(self.jobs_dir(), n, "log.txt"), errors="replace") as f:
                    for ln in f.read().splitlines()[-3000:]:
                        rt.seq += 1
                        rt.log.append((rt.seq, ln))
            except OSError:
                pass

    def start(self, body):
        self._kstat = None                      # the lock the job is started with is the one the licence server allows right now
        q = self._req(body)
        chk = self.checks(body, fresh=False)
        if not chk["can_start"]:
            raise EncodeError(chk["refusal"])
        if q["src"]["licence"]["level"] == "noderivs" and not q["licence_ack"]:
            raise EncodeError("This model's licence does not allow sharing a quantized copy. Tick \"I will not share the result\" to make one for yourself.")
        with self.lock:
            if any(j["status"] in LIVE_STATES for j in self.jobs.values()):
                raise EncodeError("An encode is already running. Wait for it, or cancel it, then start another.")
        tname = q["row"]["name"]
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        name = re.sub(r"[^A-Za-z0-9._\-]+", "-", q["src"]["name"]).strip("-")[:80] or "model"
        est = chk["estimate"]["stages"]
        weights = {s: max(2.0, (est.get(s, [60, 60])[0] + est.get(s, [60, 60])[1]) / 2.0) for s in q["stages"]}
        if "verify" in weights:       # a slice of the bar, never most of it (its estimate has a fixed part that dwarfs a small model)
            weights["verify"] = max(2.0, 0.03 * sum(w for k, w in weights.items() if k != "verify"))
        j = {"id": jid, "created": time.time(), "updated": time.time(), "status": "queued",
             "params": {"source": {k: q["src"].get(k) for k in ("kind", "id", "name", "arch", "params", "layers", "hidden", "inter", "gguf_kind", "download_bytes", "files", "emb", "head", "tied", "src_dir")},
                        "tier": q["tier"], "tier_name": tname, "cls": q["row"]["cls"], "name": name, "work": q["work"], "out": q["out"], "enc_card": q["enc_card"],
                        "dump_cards": q["dump_cards"], "keep": q["keep"], "use_hess": q["use_hess"], "encoder": q["info"]["path"] if q["info"] else None,
                        "edition": q["info"]["edition"] if q["info"] else None, "size_estimate": q["row"]["size_bytes"], "targets": q["targets"],
                        "licence_level": q["src"]["licence"]["level"], "flow": q["flow"], "lock": q["lock"],
                        "out_file": self._pick_out(q["out"], name, tname) if q["flow"] == "make" else None},
             "stages": [{"id": s, "label": STAGE_LABEL[s], "status": "pending", "pct": 0.0, "detail": "", "weight": weights[s], "started": None, "finished": None}
                        for s in q["stages"]],
             "licence_jid": None, "error": None, "result": None, "pct": 0.0, "eta_s": None, "counts": {"encoded": 0, "exact": 0, "bad": 0, "hess": 0, "hess_total": 0}}
        with self.lock:
            self.jobs[jid] = j
            self.rt[jid] = Rt()
        self._save(j)
        self._launch(j)
        return self.job_view(jid)

    def _launch(self, j):
        rt = self.rt[j["id"]]
        rt.cancel.clear()
        rt.paused = False
        j["status"] = "running"
        j["error"] = None
        j["make_overall"] = None
        self._save(j)
        rt.thread = threading.Thread(target=self._run, args=(j,), daemon=True)
        rt.thread.start()

    def get(self, jid):
        j = self.jobs.get(str(jid))
        if j is None:
            raise EncodeError("There is no such encode job.")
        return j, self.rt[j["id"]]

    def job_view(self, jid, log_since=None):
        j, rt = self.get(jid)
        v = dict(j)
        v["pct"], v["eta_s"] = self._overall(j)           # fresh at every read (the saved copy is throttled)
        v["params"] = {k: j["params"].get(k) for k in ("name", "tier", "tier_name", "cls", "work", "out", "edition", "keep", "use_hess", "lock")}
        v["log_seq"] = rt.seq
        if log_since is not None:
            v["log"] = [ln for sq, ln in list(rt.log) if sq > int(log_since)]
        return v

    def log_view(self, jid, since=0):
        j, rt = self.get(jid)
        with rt.lock:
            items = [(sq, ln) for sq, ln in rt.log if sq > int(since or 0)]
        if not items and int(since or 0) == 0:
            try:
                with open(os.path.join(self.jobs_dir(), j["id"], "log.txt"), errors="replace") as f:
                    lines = f.read().splitlines()[-2000:]
                return {"seq": rt.seq, "lines": lines}
            except OSError:
                pass
        return {"seq": rt.seq, "lines": [ln for _, ln in items]}

    def _log(self, j, line):
        if self._closed:
            return
        rt = self.rt[j["id"]]
        line = _scrub(line.rstrip("\r\n"), self._secrets())[:2000]
        with rt.lock:
            rt.seq += 1
            rt.log.append((rt.seq, line))
        try:
            os.makedirs(os.path.join(self.jobs_dir(), j["id"]), exist_ok=True)
            with open(os.path.join(self.jobs_dir(), j["id"], "log.txt"), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # progress ------------------------------------------------------------------------------------------------
    def _stage(self, j, sid):
        return next(s for s in j["stages"] if s["id"] == sid)

    def _overall(self, j):
        tot = sum(s["weight"] for s in j["stages"]) or 1.0
        done = sum(s["weight"] * (1.0 if s["status"] in ("done", "skipped") else s["pct"]) for s in j["stages"])
        mo = j.get("make_overall")
        if mo is not None and j["status"] != "done" and j["params"].get("flow") == "make":
            done = mo * tot                                     # `pxqe make` reports its own whole-run percentage: it knows its stage weights
        eta = 0.0
        for s in j["stages"]:
            if s["status"] in ("done", "skipped"):
                continue
            if s["status"] == "running" and s.get("tool_eta") is not None:
                eta += s["tool_eta"]                            # the encoder's own estimate for the stage it is in
            elif s["status"] == "running" and s.get("started") and s["pct"] > 0.03:
                el = time.time() - s["started"]
                eta += max(0.0, el / s["pct"] - el)
            else:
                eta += s["weight"] * (1.0 - s["pct"])
        return round(min(0.999, done / tot) if j["status"] != "done" else 1.0, 4), (round(eta) if j["status"] == "running" else None)       # no ETA while paused: the clock keeps running

    def _set(self, j, sid, pct=None, detail=None, status=None):
        s = self._stage(j, sid)
        if pct is not None:
            s["pct"] = max(0.0, min(1.0, pct))
        if detail is not None:
            s["detail"] = detail
        if status:
            s["status"] = status
            if status == "running" and not s.get("started"):
                s["started"] = time.time()
            if status in ("done", "skipped"):
                s["pct"] = 1.0
                s["finished"] = time.time()
                s.pop("tool_eta", None)
        if s["status"] == "running" and s.get("started") and s["pct"] > 0.03:
            el = time.time() - s["started"]
            s["eta_s"] = round(max(0.0, el / s["pct"] - el))
        self._save(j, force=status is not None)

    # control --------------------------------------------------------------------------------------------------
    def pause(self, jid):
        j, rt = self.get(jid)
        if j["status"] != "running":
            raise EncodeError("This job is not running.")
        rt.paused = True
        j["status"] = "paused"
        p = rt.proc
        if p and p.poll() is None:
            self._signal(p, signal.SIGSTOP)
        self._save(j)
        return self.job_view(jid)

    def resume(self, jid):
        j, rt = self.get(jid)
        if j["status"] == "paused":
            rt.paused = False
            j["status"] = "running"
            p = rt.proc
            if p and p.poll() is None:
                self._signal(p, signal.SIGCONT)
            self._save(j)
            return self.job_view(jid)
        if j["status"] in ("interrupted", "failed", "cancelled"):
            with self.lock:
                if any(x["status"] in LIVE_STATES and x["id"] != jid for x in self.jobs.values()):
                    raise EncodeError("Another encode is running. Wait for it first.")
            if rt.thread and rt.thread.is_alive():
                rt.thread.join(8)                      # the status flips a moment before the runner thread has finished
                if rt.thread.is_alive():
                    raise EncodeError("The job is still shutting down; try again in a few seconds.")
            self._kill_orphan(j)
            for s in j["stages"]:
                if s["status"] in ("failed", "running"):
                    s["status"] = "pending"
            self._launch(j)
            return self.job_view(jid)
        raise EncodeError("This job cannot be resumed (it is %s)." % j["status"])

    def cancel(self, jid):
        j, rt = self.get(jid)
        if j["status"] not in ("running", "paused", "queued"):
            raise EncodeError("This job is not running.")
        rt.cancel.set()
        p = rt.proc
        if p and p.poll() is None:
            if rt.paused:
                self._signal(p, signal.SIGCONT)
            self._signal(p, signal.SIGINT)
            threading.Thread(target=self._escalate, args=(p,), daemon=True).start()
        rt.paused = False
        return {"ok": True}

    def _escalate(self, p):
        for sig, wait in ((signal.SIGTERM, 6), (signal.SIGKILL, 0)):
            t0 = time.time()
            while time.time() - t0 < (6 if sig == signal.SIGTERM else 0.1):
                if p.poll() is not None:
                    return
                time.sleep(0.2)
            self._signal(p, sig)
        return

    @staticmethod
    def _descendants(pid):
        """Every process below `pid` (children, their children, ...), from /proc. `pxqe make` starts each worker (converter, calibration tool,
        encoder) in a session of its own, so a signal to make's process group does not reach them."""
        kids = {}
        try:
            names = os.listdir("/proc")
        except OSError:
            return []
        for n in names:
            if not n.isdigit():
                continue
            try:
                with open("/proc/%s/stat" % n) as f:
                    raw = f.read()
                ppid = int(raw[raw.rindex(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(n))
        out, todo = [], [int(pid)]
        while todo:
            for c in kids.get(todo.pop(), []):
                if c not in out:
                    out.append(c)
                    todo.append(c)
        return out

    def _signal(self, p, sig):
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (OSError, ProcessLookupError):
            try:
                p.send_signal(sig)
            except OSError:
                pass
        if sig in (signal.SIGSTOP, signal.SIGCONT):             # pause / resume must reach the whole tree, not only the command that started it
            for c in self._descendants(p.pid):
                try:
                    os.kill(c, sig)
                except OSError:
                    pass

    def _kill_orphan(self, j):
        pid = j.get("pid")
        if pid and _pid_alive(pid):
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGCONT)
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
                for c in self._descendants(pid):               # a paused job's workers (own sessions) are stopped: they get the TERM too, then are woken to act on it
                    for sg in (signal.SIGTERM, signal.SIGCONT):
                        try:
                            os.kill(c, sg)
                        except OSError:
                            pass
                t0 = time.time()
                while _pid_alive(pid) and time.time() - t0 < 6:
                    time.sleep(0.2)
                if _pid_alive(pid):
                    os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            except OSError:
                pass
        j["pid"] = None

    def discard(self, jid):
        j, rt = self.get(jid)
        if j["status"] not in LIVE_STATES and rt.thread and rt.thread.is_alive():
            rt.thread.join(8)
        if j["status"] in LIVE_STATES or (rt.thread and rt.thread.is_alive()):
            raise EncodeError("Cancel the job first.")
        wd = os.path.join(j["params"]["work"], j["id"])
        shutil.rmtree(wd, ignore_errors=True)
        shutil.rmtree(os.path.join(self.jobs_dir(), j["id"]), ignore_errors=True)
        with self.lock:
            self.jobs.pop(j["id"], None)
            self.rt.pop(j["id"], None)
        return {"ok": True}

    # the pipeline ---------------------------------------------------------------------------------------------
    def _wait_unpaused(self, rt):
        while rt.paused and not rt.cancel.is_set():
            time.sleep(0.3)

    def _run(self, j):
        rt = self.rt[j["id"]]
        P = j["params"]
        wd = os.path.join(P["work"], j["id"])
        os.makedirs(wd, exist_ok=True)
        self._log(j, "--- %s: %s -> %s ---" % (time.strftime("%H:%M:%S"), P["source"].get("name"), P["tier_name"]))
        try:
            if P.get("flow") == "make":
                self._run_make(j, wd)
            else:
                for st in j["stages"]:
                    if st["status"] in ("done", "skipped"):
                        continue
                    if rt.cancel.is_set():
                        raise _Cancelled()
                    fn = getattr(self, "_s_" + st["id"])
                    self._set(j, st["id"], status="running", pct=0.0, detail="")
                    fn(j, st, wd)
                    if st["status"] == "running":
                        self._set(j, st["id"], status="done")
                    self._cleanup_after(j, st["id"], wd)
            j["status"] = "done"
            self._log(j, "--- finished ---")
        except _Cancelled:
            j["status"] = "cancelled"
            j["error"] = {"code": "cancelled", "message": "Cancelled. Finished steps are kept; Resume continues, Discard deletes the work files.", "hint": ""}
            for s in j["stages"]:
                if s["status"] == "running":
                    s["status"] = "pending"
            self._log(j, "--- cancelled ---")
        except _StageFailed as e:
            j["status"] = "failed"
            j["error"] = {"code": e.code, "message": e.message, "hint": e.hint}
            for s in j["stages"]:
                if s["status"] == "running":
                    s["status"] = "failed"
            self._log(j, "--- failed: %s ---" % e.message)
        except Exception as e:     # noqa: BLE001 - the page gets a sentence, the log gets the type
            j["status"] = "failed"
            msg = _scrub("%s: %s" % (type(e).__name__, e), self._secrets())[:300]
            j["error"] = {"code": "failed", "message": "Something unexpected stopped the job (%s)." % msg, "hint": "Resume tries again from the last finished step."}
            for s in j["stages"]:
                if s["status"] == "running":
                    s["status"] = "failed"
            self._log(j, "--- unexpected: %s ---" % msg)
        finally:
            j["pid"] = None
            rt.proc = None
            self._save(j)

    def _stage_done(self, j, sid):
        s = next((x for x in j["stages"] if x["id"] == sid), None)
        return None if s is None else s["status"] in ("done", "skipped")

    def _cleanup_after(self, j, sid, wd):
        """Free disk as soon as nothing later needs a file (the disk estimate assumes this)."""
        if j["params"].get("keep"):
            return
        rm = {"convert": ["src"], "reference": ["bf16.gguf"], "quantize": ["bf16.gguf"], "hessians": ["act", "calib.imatrix"],
              "verify": ["q8.gguf", "skel.gguf", "hess", "act"]}
        for n in rm.get(sid, []):
            p = os.path.join(wd, n)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                if n == "q8.gguf" and j["params"]["source"]["kind"] == "gguf" and j["params"]["source"].get("gguf_kind") == "q8_0":
                    continue
                try:
                    os.unlink(p)
                except OSError:
                    pass
        if sid == "verify":
            shutil.rmtree(wd, ignore_errors=True)       # the finished file was moved out; nothing else in here is needed

    # process helper
    def _exec(self, j, argv, env=None, parse=None, cwd=None):
        """Run argv, log every line (key and tokens scrubbed), keep the last 80 for the failure sentence. `parse(line)` returning True means
        the line was a machine line (the `@pxqe ` progress lines): it is neither logged nor kept."""
        rt = self.rt[j["id"]]
        e = dict(os.environ)
        e.update(env or {})
        self._log(j, "$ " + " ".join(_scrub(a) for a in argv))
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1,
                                 errors="replace", env=e, cwd=cwd, start_new_session=True)
        except OSError as ex:
            raise _StageFailed("missing_tool", "Cannot start %s (%s)." % (os.path.basename(argv[0]), ex.strerror or "error"),
                               "Check that the engine and the encoder are installed (Encode tab, Rescan).")
        rt.proc = p
        j["pid"] = p.pid
        self._save(j)
        if rt.paused:
            self._signal(p, signal.SIGSTOP)
        tail = collections.deque(maxlen=80)
        try:
            for line in p.stdout:
                line = line.rstrip("\n")
                if parse and parse(line) is True:
                    continue
                self._log(j, line)
                tail.append(line)
        finally:
            try:
                p.stdout.close()
            except OSError:
                pass
        rc = p.wait()
        rt.proc = None
        j["pid"] = None
        if rt.cancel.is_set():
            raise _Cancelled()
        return rc, "\n".join(tail)

    @staticmethod
    def _threads():
        """CPU threads for the quantize and dump tools: all but one core, or PXA_ENCODE_THREADS."""
        try:
            n = int(os.environ.get("PXA_ENCODE_THREADS", ""))
        except ValueError:
            n = 0
        return max(1, n) if n else max(1, (os.cpu_count() or 4) - 1)

    def _fail_from(self, j, rc, text):
        ex = AD.explain_failure(_scrub(text, self._secrets()), rc)
        raise _StageFailed(ex["code"], ex["message"], ex["hint"])

    def _tool_env(self):
        env = {}
        try:
            E = self.host.engine_dirs()[0] if self.host.engine_dirs() else None
            if E:
                env["LD_LIBRARY_PATH"] = self.host.L.engine_ld_path(E)[0]
        except Exception:      # noqa: BLE001
            pass
        return env

    # --- stages
    def _s_download(self, j, st, wd):
        rt, P = self.rt[j["id"]], j["params"]
        src_dir = os.path.join(wd, "src")
        os.makedirs(src_dir, exist_ok=True)
        files = [(n, int(s)) for n, s in (P["source"].get("files") or [])]
        total = sum(s for _, s in files) or 1
        hf = PL.HF()
        repo = P["source"]["id"]
        done_bytes = 0
        t0 = time.time()
        for name, size in files:
            if "/" in name or name.startswith(".") or "\\" in name:
                continue
            dest = os.path.join(src_dir, name)
            if os.path.isfile(dest) and (not size or os.path.getsize(dest) == size):
                done_bytes += os.path.getsize(dest)
                self._set(j, "download", pct=done_bytes / total, detail="%s of %s" % (PL.human_bytes(done_bytes), PL.human_bytes(total)))
                continue
            part = dest + ".part"
            have = os.path.getsize(part) if os.path.isfile(part) else 0
            url = "%s/%s/resolve/main/%s" % (hf.endpoint, repo, name)
            hdr = {"Range": "bytes=%d-" % have} if have else {}
            try:
                r = hf.open(url, hdr, timeout=60)
            except urllib.error.HTTPError as e:
                if e.code == 416 and have:
                    os.replace(part, dest)
                    done_bytes += os.path.getsize(dest)
                    continue
                if e.code in (401, 403):
                    raise _StageFailed("gated", "Hugging Face refused the download of %s (gated or private)." % repo,
                                       "Accept the model's licence on huggingface.co and set HF_TOKEN, then Resume.")
                raise _StageFailed("download", "Hugging Face answered %d for %s." % (e.code, name), "Resume tries again.")
            except (urllib.error.URLError, OSError, TimeoutError):
                raise _StageFailed("offline", "The download stopped: cannot reach Hugging Face.", "Check your connection and press Resume; it continues where it stopped.")
            if r.status == 200 and have:
                have = 0
            mode = "ab" if (have and r.status == 206) else "wb"
            got = have if mode == "ab" else 0
            try:
                with r, open(part, mode) as f:
                    last = 0.0
                    while True:
                        self._wait_unpaused(rt)
                        if rt.cancel.is_set():
                            raise _Cancelled()
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        if time.time() - last > 0.4:
                            last = time.time()
                            cur = done_bytes + got
                            rate = (cur) / max(1e-3, time.time() - t0)
                            self._set(j, "download", pct=cur / total, detail="%s of %s" % (PL.human_bytes(cur), PL.human_bytes(total)))
            except (urllib.error.URLError, OSError, TimeoutError):
                raise _StageFailed("offline", "The download was interrupted.", "Press Resume; it continues where it stopped.")
            os.replace(part, dest)
            done_bytes += os.path.getsize(dest)
        self._log(j, "downloaded %s" % PL.human_bytes(done_bytes))

    def _src_dir(self, j, wd):
        s = j["params"]["source"]
        return s.get("src_dir") if s["kind"] == "folder" else os.path.join(wd, "src")

    def _s_convert(self, j, st, wd):
        out = os.path.join(wd, "bf16.gguf")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return
        conv = self.find_convert()
        if not conv:
            raise _StageFailed("missing_tool", "The converter (convert_hf_to_gguf.py) was not found.", "It comes with the PXA engine build.")
        tmp = out + ".tmp"
        layers = j["params"]["source"].get("layers") or 0
        rx_pct = re.compile(r"(\d+)%\|")
        rx_blk = re.compile(r"\bblk\.(\d+)\.")

        def parse(line):
            m = rx_pct.search(line)
            if m:
                self._set(j, "convert", pct=int(m.group(1)) / 100.0, detail="")
                return
            m = rx_blk.search(line)
            if m and layers:
                self._set(j, "convert", pct=min(0.99, (int(m.group(1)) + 1) / float(layers)), detail="layer %s of %s" % (int(m.group(1)) + 1, layers))
        try:
            os.unlink(tmp)
        except OSError:
            pass
        rc, tail = self._exec(j, [self.convert_python(), conv, self._src_dir(j, wd), "--outtype", "bf16", "--outfile", tmp], self._tool_env(), parse)
        if rc != 0 or not os.path.isfile(tmp):
            self._fail_from(j, rc, tail)
        os.replace(tmp, out)

    def _quant(self, j, sid, src, dst, ftype, env=None, label=""):
        tool = self.find_tool("pxq-quantize", "llama-quantize")
        if not tool:
            raise _StageFailed("missing_tool", "llama-quantize was not found.", "It comes with the PXA engine build.")
        rx = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]")

        def parse(line):
            m = rx.match(line.strip())
            if m:
                self._set(j, sid, pct=int(m.group(1)) / float(m.group(2)), detail="tensor %s of %s" % (m.group(1), m.group(2)))
        tmp = dst + ".tmp"
        try:
            os.unlink(tmp)
        except OSError:
            pass
        e = self._tool_env()
        e.update(env or {})
        rc, tail = self._exec(j, [tool, src, tmp, ftype, str(self._threads())], e, parse)
        if rc != 0 or not os.path.isfile(tmp):
            self._fail_from(j, rc, tail)
        os.replace(tmp, dst)

    def _s_quantize(self, j, st, wd):
        """The classic tiers: `pxqe quantize SRC DST FTYPE THREADS` (the public CPU quantizer shipped in both editions)."""
        P = j["params"]
        s = P["source"]
        out = os.path.join(wd, "encoded.gguf")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return
        info = next((f for f in self.encoders(True) if f["path"] == P.get("encoder")), None)
        if not info:
            raise _StageFailed("no_encoder", "The encoder this job was started with is gone.", "Install an encoder (Get the encoder), then Resume.")
        if s["kind"] == "gguf":
            src, requant = s["id"], s.get("gguf_kind") == "q8_0"
        else:
            src, requant = os.path.join(wd, "bf16.gguf"), False
        rx = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]")

        def parse(line):
            m = rx.match(line.strip())
            if m:
                self._set(j, "quantize", pct=int(m.group(1)) / float(m.group(2)), detail="tensor %s of %s" % (m.group(1), m.group(2)))
        tmp = out + ".tmp"
        try:
            os.unlink(tmp)
        except OSError:
            pass
        argv = AD.argv_quantize(info["path"], src, tmp, AD.ftype_name(P["tier"]), self._threads(), requant)
        rc, tail = self._exec(j, argv, {}, parse)
        if rc != 0 or not os.path.isfile(tmp):
            self._fail_from(j, rc, tail)
        os.replace(tmp, out)

    def _q8_path(self, j, wd):
        s = j["params"]["source"]
        if s["kind"] == "gguf" and s.get("gguf_kind") == "q8_0":
            return s["id"]
        return os.path.join(wd, "q8.gguf")

    def _s_reference(self, j, st, wd):
        s = j["params"]["source"]
        out = self._q8_path(j, wd)
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return
        src = s["id"] if s["kind"] == "gguf" else os.path.join(wd, "bf16.gguf")
        self._quant(j, "reference", src, out, "Q8_0")

    def _s_skeleton(self, j, st, wd):
        P = j["params"]
        out = os.path.join(wd, "skel.gguf")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return
        text = AD.skeleton_tier_map(P["tier"])
        if text is None:
            raise _StageFailed("unsupported_tier", "This Control does not know how to lay out %s yet." % P["tier_name"],
                               "Update PXA Control, or pick another tier.")
        tf = os.path.join(wd, "tiers.txt")
        with open(tf, "w") as f:
            f.write(text)
        calib = AD.calibration_file(P.get("encoder"))
        calib_sha = _sha256_file(calib) if calib else ""
        info = next((f for f in self.encoders() if f["path"] == P.get("encoder")), None) or {}
        env = AD.skeleton_env(tf, calib_sha, info.get("build_id") or "pxa-control")
        self._quant(j, "skeleton", self._q8_path(j, wd), out, AD.ftype_name(P["tier"]), env)

    def _s_dump(self, j, st, wd):
        P = j["params"]
        if os.path.isfile(os.path.join(wd, "dump.done")):
            return
        tool = self.find_tool("llama-imatrix")
        calib = AD.calibration_file(P.get("encoder"))
        if not tool or not calib:
            raise _StageFailed("missing_tool", "The activation dump tool or the calibration text was not found.", "See the checks on the Encode page.")
        act = os.path.join(wd, "act")
        shutil.rmtree(act, ignore_errors=True)
        os.makedirs(act)
        q8 = self._q8_path(j, wd)
        total = {"n": 0}
        rx_total = re.compile(r"computing over (\d+) chunks")
        rx_chunk = re.compile(r"^\[(\d+)\][\d.]+,")

        def parse(line):
            m = rx_total.search(line)
            if m:
                total["n"] = int(m.group(1))
            m = rx_chunk.match(line.strip())
            if m and total["n"]:
                self._set(j, "dump", pct=min(0.99, int(m.group(1)) / float(total["n"])), detail="chunk %s of %s" % (m.group(1), total["n"]))
        local = {c["index"]: c for c in self.cards()[0]}
        cards = [i for i in P["dump_cards"] if i in local]
        free = sum(local[i]["free_mib"] for i in cards) * PL.MIB - PL.HEADROOM_MIB_PER_CARD * PL.MIB * max(1, len(cards))
        size = os.path.getsize(q8) if os.path.isfile(q8) else 0
        layers = int(P["source"].get("layers") or 0)
        ngl = 999 if (not size or free >= size * 1.05) else max(0, int(layers * max(0.0, free) / float(size)))
        env = self._tool_env()
        env.update(AD.dump_env(act))
        if cards:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in cards)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        argv = AD.dump_argv(tool, q8, calib, os.path.join(wd, "calib.imatrix"), self._threads(), ngl)
        rc, tail = self._exec(j, argv, env, parse)
        if rc != 0:
            self._fail_from(j, rc, tail)
        open(os.path.join(wd, "dump.done"), "w").close()

    def _s_hessians(self, j, st, wd):
        """Hessians and the encode are ONE encoder process (one licence job, one charge): this stage runs it, the next one only
        reports. When the first tensor line arrives the Hessians are complete, so this stage ends and `encode` begins."""
        self._run_encoder(j, wd)

    def _s_encode(self, j, st, wd):
        self._run_encoder(j, wd)

    def _run_encoder(self, j, wd):
        P = j["params"]
        rt = self.rt[j["id"]]
        info = next((f for f in self.encoders(True) if f["path"] == P.get("encoder")), None)
        if not info:
            raise _StageFailed("no_encoder", "The encoder this job was started with is gone.", "Install an encoder (Get the encoder), then Resume.")
        q8 = self._q8_path(j, wd)
        skel = os.path.join(wd, "skel.gguf")
        dst = os.path.join(wd, "encoded.gguf")
        if not os.path.isfile(dst):
            shutil.copyfile(skel, dst)
        hdir = AD.hess_dir_for(dst)                       # `run --act` writes the Hessians here (<folder of --dst>/hess)
        hess_done = os.path.isfile(os.path.join(wd, "hess.done"))
        use_h = P["use_hess"]
        counts = j["counts"]
        counts.update({"encoded": 0, "exact": 0, "bad": 0})
        state = {"total": None}
        if use_h:
            counts["hess_total"] = int(P["source"].get("layers") or 0) * 4 or 0
        has_hess_stage = use_h

        def parse(line):
            ev = AD.parse_line(line)
            if not ev:
                return
            k = ev["event"]
            if k == "job":
                j["licence_jid"] = ev["jid"]
                self._save(j)
            elif k == "total":
                state["total"] = ev["tensors"]
            elif k == "hess":
                counts["hess"] += 1
                tot = counts["hess_total"]
                if has_hess_stage and not self._stage_done(j, "hessians"):
                    self._set(j, "hessians", pct=min(0.99, counts["hess"] / float(tot)) if tot else 0.5, detail="%d Hessians" % counts["hess"])
            elif k == "tensor":
                if has_hess_stage and not self._stage_done(j, "hessians"):
                    open(os.path.join(wd, "hess.done"), "w").close()
                    self._set(j, "hessians", status="done")
                    self._set(j, "encode", status="running", pct=0.0)
                counts["encoded"] += 1
                counts["exact"] += 1 if ev["exact"] else 0
                counts["bad"] += 1 if ev["bad"] else 0
                tot = state["total"] or P["source"].get("n_tensors") or 0
                self._set(j, "encode", pct=min(0.99, counts["encoded"] / float(tot)) if tot else 0.5,
                          detail="tensor %d of %s" % (counts["encoded"], tot or "?"))
        env = AD.run_env(info, self._key(), self.licence_server())
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if P.get("enc_card") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(P["enc_card"])
        env.update(self._tool_env())
        rte = self._rt_env(info)
        if rte.get("LD_LIBRARY_PATH") and env.get("LD_LIBRARY_PATH"):          # an older wrapper: the engine's libraries AND the GPU runtime folder
            rte["LD_LIBRARY_PATH"] = rte["LD_LIBRARY_PATH"].split(os.pathsep)[0] + os.pathsep + env["LD_LIBRARY_PATH"]
        env.update(rte)
        resume = j.get("licence_jid") if info["edition"] == "pro" else None
        if use_h and not hess_done:
            argv = AD.argv_run(info["path"], q8, dst, act=os.path.join(wd, "act"), device=0, resume=resume)
        elif use_h:
            argv = AD.argv_run(info["path"], q8, dst, hdir=hdir, device=0, resume=resume)
        else:
            argv = AD.argv_run(info["path"], q8, dst, device=0, resume=resume)
        rc, tail = self._exec(j, argv, env, parse)
        if rc != 0:
            self._fail_from(j, rc, tail)
        if has_hess_stage and not self._stage_done(j, "hessians"):
            open(os.path.join(wd, "hess.done"), "w").close()
            self._set(j, "hessians", status="done")
        self._set(j, "encode", status="done")

    def _s_verify(self, j, st, wd):
        P = j["params"]
        rt = self.rt[j["id"]]
        enc = os.path.join(wd, "encoded.gguf")
        if not os.path.isfile(enc) or os.path.getsize(enc) == 0:
            raise _StageFailed("no_output", "The encoder finished but there is no output file.", "Resume re-runs the encode.")
        c = j["counts"]
        if c["bad"] and not AD.is_classic(P["tier"]):
            raise _StageFailed("verify", "%d tensor(s) did not decode back exactly, so the file was NOT delivered." % c["bad"],
                               "Resume re-runs the encode. If it repeats, report it (Report a problem).")
        L = self.host.L
        h = L.gguf_header(enc)
        if not h.get("ok"):
            raise _StageFailed("verify", "The output is not a readable GGUF (%s)." % (h.get("err") or "bad header"), "Resume re-runs the encode.")
        want = AD.ftype_name(P["tier"]).upper()
        types = {str(L.PXQ_GGML_TYPE.get(t[1])).upper() for t in h.get("tensors") or [] if L.PXQ_GGML_TYPE.get(t[1])}
        have = (want in types) if AD.norm_tier(P["tier"]) != "pxquniversal" else bool(types)      # universal = a mix: any PXQ type will do
        if not have and not os.environ.get("PXA_ENCODE_SKIP_TYPE_CHECK"):
            raise _StageFailed("verify", "The file holds no %s tensors, so it is not the tier you asked for." % P["tier_name"], "Resume re-runs the encode.")
        self._set(j, "verify", pct=0.1, detail="checksum")
        sha = _sha256_file(enc, lambda f: self._set(j, "verify", pct=0.1 + 0.9 * f, detail="checksum"), rt.cancel.is_set)
        os.makedirs(P["out"], exist_ok=True)
        final = self._pick_out(P["out"], P["name"], P["tier_name"])
        shutil.move(enc, final)
        pxq_counts = {}
        for t in h.get("tensors") or []:
            nm = L.PXQ_GGML_TYPE.get(t[1])
            if nm:
                pxq_counts[nm] = pxq_counts.get(nm, 0) + 1
        j["result"] = {"path": final, "sha256": sha, "size": os.path.getsize(final), "tier": P["tier_name"], "cls": P["cls"], "name": P["name"],
                       "encoded": c["encoded"], "exact": c["exact"], "tier_tensors": pxq_counts.get(want, sum(pxq_counts.values())), "types": pxq_counts}
        self._log(j, "sha256 %s  %s" % (sha, final))

    # ---- the one-command flow (`pxqe make`, CLI version 2) -----------------------------------------------------
    @staticmethod
    def _pick_out(out_dir, name, tier_name):
        """<out>/<name>-<tier>.gguf, or -2, -3 ... : a finished file is never overwritten."""
        base = "%s-%s" % (name, tier_name)
        final, n = os.path.join(out_dir, base + ".gguf"), 1
        while os.path.exists(final):
            n += 1
            final = os.path.join(out_dir, "%s-%d.gguf" % (base, n))
        return final

    def _make_missing(self, q, info):
        """The steps of this job that `info --json` says cannot run here, each as one phrase with its reason (the encoder's own sentence)."""
        need = [AD.JOB_STAGE_MAKE[x] for x in q["stages"] if x in AD.JOB_STAGE_MAKE]
        what = {r["name"]: r["does"] for r in info.get("stages") or []}
        rt_bad = (info.get("runtime") or {}).get("lib") in ("unloadable", "missing")          # the runtime check above already says that, once
        return ["%s (%s)" % (what.get(n) or n, why.rstrip(".")) for n, why in AD.stages_not_ready(info, need)
                if not (rt_bad and why.startswith("The encoder library cannot start"))]

    def _make_engine(self, j):
        """-> (engine folder, --gpu-layers) for the calibration run, or (None, None). An engine folder is offered to `make` only when it has a
        llama-imatrix; `make` itself checks that the binary carries the hook and falls back to the bundled CPU tool when it does not.
        --gpu-layers keeps the old rule: all layers when the Q8_0 copy fits in the cards' free memory, else as many as do."""
        P = j["params"]
        if AD.is_classic(P["tier"]):
            return None, None
        eng = next((d for d in self.host.engine_dirs() if os.path.isfile(os.path.join(d, "bin", "llama-imatrix"))), None)
        if not eng:
            return None, None
        local = {c["index"]: c for c in self.cards()[0]}
        cards = [i for i in P["dump_cards"] if i in local]
        free = sum(local[i]["free_mib"] for i in cards) * PL.MIB - PL.HEADROOM_MIB_PER_CARD * PL.MIB * max(1, len(cards))
        size = float(P["source"].get("params") or 0) * 8.5 / 8.0                      # the Q8_0 copy: 8.5 bits per weight
        layers = int(P["source"].get("layers") or 0)
        if not cards or not size or free >= size * 1.05:
            return eng, None
        return eng, max(0, int(layers * max(0.0, free) / size))

    def _make_event(self, j, ev, st):
        """One parsed `@pxqe ` line -> the page's stage bars (the log shows `make`'s own human lines, not these)."""
        k = ev["event"]
        if k == "plan":
            have = {s["id"]: s for s in j["stages"]}
            order = []
            for r in ev["stages"]:
                sid = AD.MAKE_STAGE_ID[r["name"]]
                order.append(have.get(sid) or {"id": sid, "label": STAGE_LABEL[sid], "status": "pending", "pct": 0.0, "detail": "", "weight": 10.0,
                                               "started": None, "finished": None})
            if order:
                j["stages"] = order                                   # the encoder's plan is what will really happen
            st["planned"] = True
        elif k == "stage":
            sid = AD.MAKE_STAGE_ID[ev["stage"]]
            if not any(s["id"] == sid for s in j["stages"]):
                return
            if ev["overall"] is not None:
                j["make_overall"] = ev["overall"]
            if ev["state"] == "start":
                self._set(j, sid, status="running", pct=0.0, detail="")
            elif ev["state"] == "progress":
                self._stage(j, sid)["tool_eta"] = ev["eta_s"]
                if self._stage(j, sid)["status"] != "running":
                    self._set(j, sid, status="running")
                self._set(j, sid, pct=ev["percent"] if ev["percent"] is not None else None, detail=ev["message"])
            elif ev["state"] in ("done", "skipped"):          # skipped = finished by an earlier run of the same command
                self._set(j, sid, status="done")
        elif k == "done":
            st["done"] = ev
        elif k == "error":
            st["err"] = ev

    def _run_make(self, j, wd):
        """The whole job as ONE `pxqe make` run. Started again with the same argv it resumes from its own state file in `wd`."""
        P = j["params"]
        rt = self.rt[j["id"]]
        info = next((f for f in self.encoders(True) if f["path"] == P.get("encoder")), None)
        if not info:
            raise _StageFailed("no_encoder", "The encoder this job was started with is gone.", "Install an encoder (Get the encoder), then Resume.")
        if not AD.supports_make(info):
            raise _StageFailed("old_encoder", "The encoder at %s was replaced by an older one that cannot continue this job." % P.get("encoder"),
                               "Put back the encoder this job was started with, or discard the job.")
        src = P["source"]
        tier = AD.make_tier(P["tier"])
        out_file = P["out_file"]
        os.makedirs(P["out"], exist_ok=True)
        local = {c["index"] for c in self.cards()[0]}
        cards = [i for i in ([P.get("enc_card")] + list(P.get("dump_cards") or [])) if i is not None and i in local]
        cards = list(dict.fromkeys(cards))                       # the encode card first: `--device 0` is then the encode card
        engine, ngl = self._make_engine(j)
        py = self.convert_python()
        requant = AD.is_classic(P["tier"]) and src["kind"] == "gguf" and src.get("gguf_kind") == "q8_0"       # a second lossy pass: the quantizer wants both flags
        argv = AD.argv_make(info["path"], src["id"], tier, out_file, wd, device=0 if cards and not AD.is_classic(P["tier"]) else None, python=py,
                            engine=engine, gpu_layers=ngl, threads=self._threads(), keep_work=bool(P.get("keep")),
                            quantizer_args=AD.REQUANTIZE_ARGS if requant else (), lock=P.get("lock") if AD.supports_lock(info) else None)
        env = AD.make_env(info, P["tier"], self._key(), self.licence_server(), py)
        env.update(self._rt_env(info))
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if cards:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in cards)
        st = {"err": None, "done": None, "planned": False}

        def parse(line):
            ev = AD.parse_make_line(line)
            if ev:
                self._make_event(j, ev, st)
                return True
            h = AD.parse_line(line)
            if h and h["event"] == "job":
                j["licence_jid"] = h["jid"]
                self._save(j)
            return False
        rc, tail = self._exec(j, argv, env, parse)
        if rc == 0 and st["done"]:
            return self._finish_make(j, st["done"], wd)
        if rc == 0:
            raise _StageFailed("no_result", "The encoder finished but did not say what it wrote.", "Press Resume: the same command again reports the finished file.")
        err = st["err"]
        if err:
            err = dict(err, message=_scrub(err["message"], self._secrets()))
        ex = AD.explain_make_failure(rc, err, _scrub(tail, self._secrets()))
        raise _StageFailed(ex["code"], ex["message"], ex["hint"])

    def _finish_make(self, j, ev, wd):
        """`make` said done: check the file is where and what it said, record the result, and clear the work folder."""
        P = j["params"]
        out = ev["out"]
        if not os.path.isfile(out) or os.path.getsize(out) != ev["size"]:
            raise _StageFailed("no_output", "The encoder said it finished but the file is not at %s." % out, "Press Resume: the same command again checks it.")
        L = self.host.L
        h = L.gguf_header(out)
        if not h.get("ok"):
            raise _StageFailed("verify", "The output is not a readable GGUF (%s)." % (h.get("err") or "bad header"), "Press Resume to run the check again.")
        want = AD.ftype_name(P["tier"]).upper()
        pxq_counts = {}
        for t in h.get("tensors") or []:
            nm = L.PXQ_GGML_TYPE.get(t[1])
            if nm:
                pxq_counts[nm] = pxq_counts.get(nm, 0) + 1
        if want not in {k.upper() for k in pxq_counts} and AD.norm_tier(P["tier"]) != "pxquniversal":
            self._log(j, "note: the file holds no %s tensors by its type ids; the encoder's own check passed" % P["tier_name"])
        for s in j["stages"]:
            if s["status"] not in ("done", "skipped"):
                self._set(j, s["id"], status="done")
        j["counts"].update({"encoded": 0, "exact": 0, "bad": 0})
        j["result"] = {"path": out, "sha256": ev["sha256"], "size": ev["size"], "tier": P["tier_name"], "cls": P["cls"], "name": P["name"],
                       "encoded": 0, "exact": 0, "tier_tensors": pxq_counts.get(want, sum(pxq_counts.values())), "types": pxq_counts,
                       "lock": self._lock_result(j, h, ev)}
        self._log(j, "sha256 %s  %s" % (ev["sha256"], out))
        if not P.get("keep"):
            shutil.rmtree(wd, ignore_errors=True)               # the finished file is already in the output folder; nothing else in here is needed

    def _lock_result(self, j, h, ev):
        """The lock of a finished PXQN file by the Pro encoder, for the Done screen: what the FILE'S OWN header says (pxa.lock.mode; a locked file's
        tensors carry type id 4096 + their type), not what the encoder's last line claimed. None for a classic tier or the Free encoder (no lock applies)."""
        P = j["params"]
        if AD.is_classic(P["tier"]) or P.get("edition") != "pro":
            return None
        kv = h.get("kv") or {}
        fm = kv.get("pxa.lock.mode")
        mode = fm if fm in ("personal", "supporters") else ("locked" if h.get("locked") else "open")
        said = ev.get("lock") or "open"
        if said != mode and not (mode == "locked" and said != "open"):
            self._log(j, "note: the encoder said the lock is '%s' but the file's own header says '%s'; the file is what counts" % (said, mode))
        fid = str(kv.get("pxa.lock.file_id") or ev.get("lock_file_id") or "")[:24]
        return AD.lock_done_view(mode, fid if mode != "open" else "")

    # ---- step 5: test it ------------------------------------------------------------------------------------
    def test_info(self, jid):
        j, _ = self.get(jid)
        if j["status"] != "done" or not j.get("result"):
            raise EncodeError("The encode has not finished yet.")
        r = j["result"]
        if not os.path.isfile(r["path"]):
            raise EncodeError("The output file is no longer at %s." % r["path"])
        return {"model": r["path"], "cards": j["params"].get("targets") or [], "name": r["name"], "tier": r["tier"]}
