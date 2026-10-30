"""Speech setup: the engines and voice models the shared speech service needs, installed by one button in PXA Control
(Plugins tab, "Speech on this computer") or by `python3 -m pxa_control.speech setup [--languages]`.

* Engines: pinned CPU wheels (pins(), per Python version; prebuilt wheels only, nothing is compiled on the user's computer),
  every dependency locked too, with each file's sha256 (engines.lock, kokoro.lock: pip --require-hashes refuses anything
  else), installed by a pinned, sha256-checked pip zipapp (PIP_*) into Control's own folder <config>/speech/site-py<maj><min>,
  never into the system Python: no admin rights, works on PEP 668 distributions, a Python upgrade gets its own folder.
  The new set is built beside the old one (.new), import-checked in a fresh process (`setup --check`), marked, then
  swapped in, so a failed run leaves the last good set in place.
* Models: `python -m pxa_control.speech fetch` in a fresh process that already sees the new engines (registry.py: licence
  allowlist, pinned revision, sha256); --languages adds the multilingual hearing model and the German voice.
* Progress: <config>/speech/setup.json {state running|done|failed|stopped, step, message, started, updated, languages},
  written atomically with a heartbeat (a running job that goes quiet for STALE_S reads as stopped). Output: setup.log (0600).
* add_site() puts a finished install ahead of the system's own packages for serve / fetch / bench (__main__.py).

Stdlib only: the plugin host imports it for the status.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

from . import registry

PIP_VERSION = "26.1.2"
PIP_URL = "https://bootstrap.pypa.io/pip/zipapp/pip-%s.pyz" % PIP_VERSION
PIP_SHA256 = "6ddc3444b803a48d83ccf1c4ad846717b42c8ffc9d74713a53ae829a97201365"
PIP_SIZE = 1756180
PY_MIN, PY_MAX = (3, 10), (3, 14)
# (sys.platform, machine) with a prebuilt wheel for every pin: none for Intel Macs or Windows on ARM (whisper.cpp, ONNX Runtime)
PLATFORMS = {("linux", "x86_64"), ("linux", "aarch64"), ("linux", "arm64"), ("win32", "amd64"), ("darwin", "arm64")}
COMMON = ("pywhispercpp==1.5.1", "huggingface_hub==2.2.0", "espeakng-loader==0.2.4", "phonemizer==3.4.0")
BY_PYTHON = {(3, 10): ("numpy==2.2.6", "onnxruntime==1.23.2"), (3, 11): ("numpy==2.4.4", "onnxruntime==1.31.0")}
NEWER = ("numpy==2.5.4", "onnxruntime==1.31.0")         # 3.12 - 3.14
# its own pip call with --no-deps: it declares python < 3.14, the tested 3.14 set runs it; its dependencies are pinned above
KOKORO = "kokoro-onnx==0.6.1"
# the hash locks pip installs from: lock_input() through `uv pip compile --universal --generate-hashes` (the recipe is in
# each lock's header); one file serves every pinned Python on Windows, Linux and Apple silicon
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINES_LOCK = os.path.join(HERE, "engines.lock")
KOKORO_LOCK = os.path.join(HERE, "kokoro.lock")
MODULES = ("numpy", "onnxruntime", "pywhispercpp", "huggingface_hub", "espeakng_loader", "phonemizer", "kokoro_onnx")
MARKER = "pxa-speech-setup.json"
STALE_S = 60.0
HEARTBEAT_S = 5.0
ENGINES_MB = 120            # the engines' download (measured), for the button's "about N MB"
ENGINES_DISK_MB = 700       # what they take on disk while a new set sits beside the old one
NO_WINDOW = 0x08000000      # Windows CREATE_NO_WINDOW: no console flashes up for pip or the checks
TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# a speech process starts as `python -c BOOT <tools> <args>`, not `python -m pxa_control.speech`: python.org's embeddable
# Python (the one a compiled Control carries) ignores PYTHONPATH and the current folder, so the bootstrap puts tools first
BOOT = ("import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
        "runpy.run_module('pxa_control.speech', run_name='__main__', alter_sys=True)")

HINTS = (
    ("DLL load failed", "win32", "A Windows part is missing: install the Microsoft Visual C++ Redistributable (x64) from "
                                 "Microsoft's website, then run the setup again."),
    ("No matching distribution found", None, "There is no ready-made speech package for this computer: speech needs 64-bit "
                                             "Python 3.10 to 3.14 on Windows, Linux or an Apple-silicon Mac."),
    ("Could not find a version that satisfies", None, "There is no ready-made speech package for this computer: speech needs "
                                                      "64-bit Python 3.10 to 3.14 on Windows, Linux or an Apple-silicon Mac."),
    ("CERTIFICATE_VERIFY_FAILED", "darwin", "Python on this Mac has no certificates yet: open the Python folder in "
                                            "Applications, run Install Certificates, then run the setup again."),
    ("CERTIFICATE_VERIFY_FAILED", None, "The secure connection could not be checked. If this computer uses a proxy with its "
                                        "own certificate, set SSL_CERT_FILE to it before starting PXA Control."),
    ("No space left on device", None, "The disk is full: free some space, then run the setup again."),
    ("Errno 28", None, "The disk is full: free some space, then run the setup again."),
    ("Temporary failure in name resolution", None, "This computer could not reach the internet: check the connection, then "
                                                   "run the setup again."),
    ("getaddrinfo failed", None, "This computer could not reach the internet: check the connection, then run the setup again."),
)


def version_of(version=None):
    return tuple(version or sys.version_info[:2])


def pins(version=None):
    """the first pip call's requirements for this Python (kokoro-onnx goes in its own call)."""
    return list(BY_PYTHON.get(version_of(version), NEWER)) + list(COMMON)


def lock_input():
    """the requirements engines.lock is made from: every Python's pins under its python_version marker, then COMMON."""
    lines = []
    for v in sorted(BY_PYTHON):
        lines += ['%s ; python_version == "%d.%d"' % (p, v[0], v[1]) for p in BY_PYTHON[v]]
    newer = max(BY_PYTHON)
    lines += ['%s ; python_version >= "%d.%d"' % (p, newer[0], newer[1] + 1) for p in NEWER]
    return "\n".join(lines + list(COMMON)) + "\n"


def pins_id(version=None):
    """names the installed set: a changed pin, pip or lock is a new set (the button offers the setup again)."""
    h = hashlib.sha256("\n".join(pins(version) + [KOKORO, PIP_VERSION]).encode())
    for p in (ENGINES_LOCK, KOKORO_LOCK):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _frozen():
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def supported(version=None, plat=None, machine=None, frozen=None):
    """-> (True, None) or (False, why in plain words)."""
    v = version_of(version)
    plat = plat or sys.platform
    machine = (machine or platform.machine() or "").lower()
    if _frozen() if frozen is None else frozen:
        return False, "This build of PXA Control cannot run speech yet."
    if not PY_MIN <= v <= PY_MAX:
        return False, "Speech needs Python 3.10 to 3.14; PXA Control runs on Python %d.%d." % v
    if ("linux" if plat.startswith("linux") else plat, machine) not in PLATFORMS or sys.maxsize <= 2 ** 32:
        return False, "Speech has no ready-made engines for this computer (%s, %s)." % (
            {"win32": "Windows", "darwin": "macOS"}.get(plat, plat.capitalize()), machine or "unknown")
    return True, None


def speech_dir(cfg=None):
    return os.path.join(cfg or registry.config_dir(), "speech")


def site_dir(cfg=None, version=None):
    return os.path.join(speech_dir(cfg), "site-py%d%d" % version_of(version))


def models_dir(cfg=None):
    return os.environ.get("PXA_SPEECH_MODELS") or os.path.join(cfg or registry.config_dir(), "models", "speech")


def _write_json(path, d):
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f)
    for i in range(20):                    # Windows: a reader holding the file open makes the replace fail for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 19:
                raise
            time.sleep(0.05)


def read_marker(path):
    try:
        with open(os.path.join(path, MARKER), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def write_marker(path):
    _write_json(os.path.join(path, MARKER), {"pins": pins_id(), "python": "%d.%d" % sys.version_info[:2],
                                             "pip": PIP_VERSION, "at": time.time()})


def add_site(path=None, force=False, syspath=None):
    """put Control's own speech packages ahead of the system's (after the standard library, before every site-packages),
    once. Only a finished, checked install (its marker), unless force (the check of a new set). -> the folder or None."""
    p = path or site_dir()
    sp = sys.path if syspath is None else syspath
    if not os.path.isdir(p) or not (force or read_marker(p)):
        return None
    if p not in sp:
        at = next((i for i, x in enumerate(sp)
                   if os.path.basename(str(x).rstrip("/\\")) in ("site-packages", "dist-packages")), len(sp))
        sp.insert(at, p)
    return p


# ---- the job's state file ---------------------------------------------------------------------------------------------
def job_path(cfg=None):
    return os.path.join(speech_dir(cfg), "setup.json")


def on_disk(cfg=None):
    """speech was set up here, or a setup ran: an engines marker in any site-py* folder, or the job's state file (no import,
    no process: the Plugins tab's refresh asks the speech Python for its status only then, or when a plugin needs speech)."""
    sd = speech_dir(cfg)
    try:
        names = os.listdir(sd)
    except OSError:
        return False
    return "setup.json" in names or any(n.startswith("site-py") and os.path.isfile(os.path.join(sd, n, MARKER))
                                        for n in names)


def write_job(cfg, d):
    os.makedirs(speech_dir(cfg), mode=0o700, exist_ok=True)
    _write_json(job_path(cfg), d)


def read_job(cfg=None, clock=time.time):
    """the last setup's state, or None. A running job silent for STALE_S (closed, killed, the computer slept), or stamped that
    far ahead of the clock, is stopped."""
    try:
        with open(job_path(cfg), encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    u = d.get("updated") or 0              # a hand-edited or damaged file reads as no job (never a 500, never "running" for ever)
    if isinstance(u, bool) or not isinstance(u, (int, float)) or not 0 <= u < 1e11:    # a real epoch in seconds: not inf, NaN, a ms stamp
        return None
    if d.get("state") == "running" and abs(clock() - u) > STALE_S:      # a stamp ahead of the clock (it jumped back) is stale too
        d.update(state="stopped", message="The setup stopped before it finished. Run it again.")
    return d


class _Job(object):
    def __init__(self, cfg, languages, clock, say=None):
        self.cfg, self.clock, self.say = cfg, clock, say
        now = clock()
        self.d = {"state": "running", "step": "start", "message": "Starting", "started": now, "updated": now,
                  "languages": bool(languages), "pid": os.getpid()}
        self.lock, self.quit = threading.Lock(), threading.Event()
        self.save()
        self.beat = threading.Thread(target=self._beat, name="speech-setup-beat", daemon=True)
        self.beat.start()

    def save(self, **change):
        with self.lock:
            self.d.update(change)
            self.d["updated"] = self.clock()
            write_job(self.cfg, self.d)
        if change.get("message") and self.say:
            try:
                self.say(change["message"])
            except Exception:  # noqa: BLE001 - the notify is a courtesy: its failure must never fail the job or stop the heartbeat
                pass

    def _beat(self):
        while not self.quit.wait(HEARTBEAT_S):
            try:
                self.save()
            except OSError:
                pass

    def step(self, step, message):
        self.save(step=step, message=message)

    def end(self, state, step, message):
        self.quit.set()
        self.beat.join(2)
        self.save(state=state, step=step, message=message)
        return 0 if state == "done" else 1


# ---- the work -------------------------------------------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rmtree(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def _short(e):
    return (str(e) or e.__class__.__name__).strip().splitlines()[-1][:300]


def hint(text, plat=None):
    """a known failure in the output -> what the user can do about it ("" when nothing known matches)."""
    plat = plat or sys.platform
    for needle, where, words in HINTS:
        if needle in text and (where is None or plat.startswith(where)):
            return words
    return ""


def _tail(path, n=16384, start=0):
    """the last n bytes of path, never before offset start (where this run's output begins)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(start, os.path.getsize(path) - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _download(url, dest, limit=PIP_SIZE * 2):
    req = urllib.request.Request(url, headers={"User-Agent": "pxa-control-speech-setup"})
    n = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        for b in iter(lambda: r.read(1 << 16), b""):
            n += len(b)
            if n > limit:
                raise ValueError("the download is larger than expected")
            f.write(b)


def speech_argv(py, *args, tools=TOOLS):
    """the command line that runs `pxa_control.speech <args>` (tools/pxa_control) in the Python py: no bytecode written,
    output unbuffered (the embeddable Python ignores PYTHON* variables)."""
    return [py, "-B", "-u", "-c", BOOT, tools] + list(args)


def _run(argv, env, cwd, log):
    kw = {"creationflags": NO_WINDOW} if os.name == "nt" else {}
    return subprocess.call(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kw)


def _models(cfg):
    """-> (english ready, languages ready, MB still to download for each)."""
    from .service import STT_MODELS, TTS_MODELS, TTS_EXTRAS
    reg = registry.load_registry()
    cache = models_dir(cfg)

    def have(mid):
        return registry.is_fetched(reg[mid], cache)

    def mb(mids):
        return sum(f.get("size", 0) for m in mids if not have(m) for f in reg[m]["files"]) // 1000000
    english = any(have(m) for m in STT_MODELS.values()) and have(TTS_MODELS["kokoro"])
    languages = (have(STT_MODELS["small"]) or have(STT_MODELS["base"])) and have(TTS_EXTRAS["piper-de"])
    return (english, languages, 0 if english else mb([STT_MODELS["tiny.en"], TTS_MODELS["kokoro"]]),
            0 if languages else mb([STT_MODELS["small"], TTS_EXTRAS["piper-de"]]))


def _find_spec(name):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None


def status(cfg=None, find_spec=None, clock=time.time):
    """what the Plugins tab shows: can this computer run speech, what is installed, what a setup would download, the job.
    engines: ready (this Control's set) | update (an older set, still used) | system (the user's own pip install) | missing."""
    cfg = cfg or registry.config_dir()
    ok, why = supported()
    mk = read_marker(site_dir(cfg))
    if mk:
        engines = "ready" if mk.get("pins") == pins_id() else "update"
    elif all((find_spec or _find_spec)(m) for m in MODULES):
        engines = "system"
    else:
        engines = "missing"
    english, languages, mb_en, mb_lang = _models(cfg)
    eng_mb = 0 if engines == "ready" else ENGINES_MB    # run() installs Control's own set whenever this one is not it
    return {"supported": ok, "why": why, "python": "%d.%d" % sys.version_info[:2], "engines": engines,
            "models": {"english": english, "languages": languages},
            "ready": bool(engines != "missing" and english and not _frozen()),
            # english: a setup run; languages: what the other languages add to it; add_languages: a languages run alone
            "download_mb": {"english": eng_mb + mb_en, "languages": mb_lang, "add_languages": eng_mb + mb_lang,
                            "engines": ENGINES_MB},
            "job": read_job(cfg, clock)}


def check(path, modules=MODULES, deep=True):
    """import every engine from `path` (ahead of the system's packages) in THIS process; deep: load the native parts too.
    -> {"ok": True, "versions": {module: version}} or {"ok": False, "error"}. `setup --check <folder>` runs it in a fresh
    process before a new set is swapped in."""
    add_site(path, force=True)
    importlib.invalidate_caches()
    versions = {}
    for m in modules:
        try:
            mod = importlib.import_module(m)
        except Exception as e:  # noqa: BLE001 - ImportError, or OSError from a native library
            return {"ok": False, "error": "%s: %s" % (m, _short(e))}
        versions[m] = str(getattr(mod, "__version__", "") or "")
    if deep:
        try:
            _deep_check()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": "engines: %s" % _short(e)}
    return {"ok": True, "versions": versions}


def _deep_check():
    """the native libraries the service loads: ONNX Runtime's CPU provider, whisper.cpp, espeak-ng (as tts.py wires it)."""
    import onnxruntime
    if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("ONNX Runtime has no CPU provider")
    import pywhispercpp.model  # noqa: F401
    from kokoro_onnx import Kokoro  # noqa: F401
    import espeakng_loader
    import phonemizer
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    if not str(phonemizer.phonemize("hello", "en-us", preserve_punctuation=True, with_stress=True)).strip():
        raise RuntimeError("espeak-ng gave no phonemes")


def _env(cfg):
    env = dict(os.environ)
    env["PYTHONPATH"] = TOOLS + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(PXA_CONTROL_CONFIG_DIR=cfg, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    # pip reads no config file at all: --isolated still loads the machine-wide one (Windows: C:\ProgramData\pip\pip.ini,
    # which a standard user can write), where an extra requirement and wheel folder would pass --require-hashes
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def _crashed(job, logp, start, e):
    """an unexpected error: the traceback to setup.log (0600), the job ended failed with its kind, a hint when this run's
    output or the error matches one, and where the details are (never the error's own text: the log has it)."""
    try:
        with open(logp, "ab") as log:
            log.write(("unexpected error:\n%s" % traceback.format_exc()).encode("utf-8", "replace"))
    except OSError:
        pass
    words = "The setup stopped on an unexpected error (%s)." % e.__class__.__name__
    more = hint(_tail(logp, start=start) + "\n" + str(e)) or "Run it again."
    try:
        return job.end("failed", job.d.get("step") or "start", " ".join((words, more, "Details: %s" % logp)))
    except Exception:  # noqa: BLE001 - the state file cannot be written either: it goes stale
        job.quit.set()
        return 1


def run(cfg=None, languages=False, force=False, runner=None, downloader=None, python=None, disk_free=None, clock=time.time,
        say=None):
    """the whole setup as one job (the module doc). -> 0 done, 1 failed (the state file says why), 2 another setup runs."""
    cfg = cfg or registry.config_dir()
    sd = speech_dir(cfg)
    os.makedirs(sd, mode=0o700, exist_ok=True)
    cur = read_job(cfg, clock)
    if cur and cur.get("state") == "running":
        return 2
    py = python or sys.executable
    runner, downloader = runner or _run, downloader or _download
    disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
    site = site_dir(cfg)
    env, logp = _env(cfg), os.path.join(sd, "setup.log")
    job = _Job(cfg, languages, clock, say)
    start = 0
    try:
        fd = os.open(logp, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        start = os.fstat(fd).st_size            # the log keeps earlier runs: a hint comes from this run's output only
        with os.fdopen(fd, "ab") as log:
            try:
                os.chmod(logp, 0o600)
            except OSError:
                pass

            def call(argv):
                log.write(("\n$ %s\n" % " ".join(argv)).encode("utf-8", "replace"))
                log.flush()
                rc = runner(argv, env, TOOLS, log)
                log.flush()
                return rc

            def failed(step, words):
                log.flush()                 # the log is buffered: the hint reads the file, so it needs the last lines on disk
                more = hint(_tail(logp, start=start))
                return job.end("failed", step, " ".join(x for x in (words, more, "Details: %s" % logp) if x))

            log.write(("\n==== speech setup %s (python %s, %s)\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), platform.python_version(),
                                                                   platform.platform())).encode("utf-8", "replace"))
            ok, why = supported()
            if not ok:
                return job.end("failed", "start", why)
            mk = read_marker(site)
            engines = force or not mk or mk.get("pins") != pins_id()
            _en, _lang, mb_en, mb_lang = _models(cfg)
            need = (ENGINES_DISK_MB if engines else 0) + mb_en + (mb_lang if languages else 0)
            free = disk_free(sd) // 1000000
            if free < need + 200:
                return job.end("failed", "start", "Not enough free disk space: speech needs about %d MB in %s and %d MB are free."
                               % (need + 200, sd, free))
            if engines:
                job.step("installer", "Downloading the installer")
                pyz = os.path.join(sd, "pip-%s.pyz" % PIP_VERSION)
                if not (os.path.isfile(pyz) and _sha256(pyz) == PIP_SHA256):
                    part = pyz + ".part"
                    try:
                        downloader(PIP_URL, part)
                    except Exception as e:  # noqa: BLE001
                        _rmtree(part)
                        log.write(("installer download: %s\n" % _short(e)).encode("utf-8", "replace"))
                        return failed("installer", "Could not download the installer.")
                    if not os.path.isfile(part) or os.path.getsize(part) != PIP_SIZE or _sha256(part) != PIP_SHA256:
                        _rmtree(part)
                        return job.end("failed", "installer", "The installer download did not match its fingerprint, so nothing "
                                       "was installed. Try again later.")
                    os.replace(part, pyz)
                job.step("engines", "Installing the speech engines (about %d MB)" % ENGINES_MB)
                new, old = site + ".new", site + ".old"
                _rmtree(new)
                base = [py, "-I", pyz, "install", "--isolated", "--target", new, "--only-binary=:all:", "--no-input",
                        "--disable-pip-version-check", "--no-warn-script-location", "--progress-bar", "off"]
                rc = call(base + ["--require-hashes", "-r", ENGINES_LOCK])
                if rc == 0:
                    rc = call(base + ["--no-deps", "--ignore-requires-python", "--require-hashes", "-r", KOKORO_LOCK])
                if rc != 0:
                    _rmtree(new)
                    return failed("engines", "Installing the speech engines failed.")
                job.step("verify", "Checking the speech engines")
                if call(speech_argv(py, "setup", "--check", new)) != 0:
                    _rmtree(new)
                    return failed("verify", "The speech engines did not start on this computer.")
                write_marker(new)
                try:
                    _rmtree(old)
                    if os.path.exists(site):
                        os.replace(site, old)
                    os.replace(new, site)
                except OSError as e:
                    if os.path.isdir(old) and not os.path.exists(site):
                        os.replace(old, site)
                    _rmtree(new)
                    log.write(("swap: %s\n" % _short(e)).encode("utf-8", "replace"))
                    return job.end("failed", "verify", "The speech engines are in use: turn off the speech plugins (or restart "
                                   "PXA Control), then run the setup again.")
                _rmtree(old)
            job.step("models", "Downloading the voice models (about %d MB)" % (mb_en + (mb_lang if languages else 0)))
            rc = call(speech_argv(py, "fetch"))
            if rc == 0 and languages:
                rc = call(speech_argv(py, "fetch", "--stt", "small", "--extra", "piper-de"))
            if rc != 0:
                return failed("models", "Downloading the voice models failed.")
            return job.end("done", "done", "Speech is ready.")
    except Exception as e:  # noqa: BLE001 - whatever it was, the job ends failed (never left "running" until STALE_S)
        return _crashed(job, logp, start, e)
