"""pxa_encode_adapter.py - the ONE file in PXA Control that knows how the PXA Quantizer command line works.

PXA Control is open source; the quantizer is not. Control never links to it and never reads its
files: it runs the quantizer's command line (`pxqe`) as a subprocess and reads what it prints.
Everything that depends on that command line lives HERE, so when the command line changes (a new
flag, a different progress line, a new verb) this is the only file to edit:

  * where to look for an installed encoder, and `info --json` (edition, build, tiers, licence);
  * the argv for each verb and the environment a run gets: `make` (CLI version 2: the whole flow in ONE command) or, for an encoder
    that reports a CLI version below 2, the older multi-command flow (`quantize`, `prep`, `run`);
  * the parsers for the progress lines the encoder prints (`@pxqe ` JSON lines from `make`, text lines from the older verbs);
  * the plain-language reading of a failure (licence refused, quota, offline, out of memory);
  * the skeleton step of today's recipe (a uniform tier map handed to the engine's quantize tool).

Contract (from the licence server's PACKAGE.md; each item below is one function here):
  `pxqe info --json [--check-licence]` prints one JSON object, no GPU, no network unless --check-licence:
      {"edition": "free"|"pro", "version", "build_id", "cli", "platform", "cuda_major", "tiers": [...], "features": [...],
       "runtime": {"lib": "none"|"loadable"|"unloadable"|"missing", "detail"?, "missing"?: ["libcublas.so.12", ...], "fix"?: "runtime-pack"|"driver"|"reinstall",
                   "driver"?: bool, "source"?: "pack"|"system"|"engine"|..., "resolver"?: 1},
       "stages": [{"name", "does", "edition", "ready", "reason"}...],  "make": {"progress_prefix", "sources", "classic_tiers", "exit_codes"},   (cli >= 2)
       "licence": {"state", "user", "key_id", "encodes_left", "unlimited", "expires", "checked", "reason"?}}   (licence: Pro only)
  `pxqe make SOURCE --tier T --out FILE [--work DIR] [--device N] [--python PATH] [--engine DIR] [--gpu-layers N] [--keep-work] [--lock MODE] ...`   CLI >= 2, both editions
      --lock auto|personal|supporters|open (Pro, PXQN tiers; an encoder that knows it lists `make.lock_modes` in `info --json`). What the licence server allows
      for the key comes back as `lock: {enabled, allowed_modes, default_mode, epoch}` in `pxqe status` and in `licence.lock` of `info --json --check-licence`;
      the `done` line then carries `lock`, `lock_file_id`, `lock_epoch` of a locked file.
      stdout lines that start with `@pxqe ` carry one JSON object each: plan / stage (start|progress|done|skipped) / done / error;
      exit codes 0 done, 1 a stage failed (state kept), 2 cannot run here / bad arguments, 3 refused by the licence server, 130 stopped.
      The SAME command again resumes from <work>/state.json. The key goes in the environment (PXQE_KEY / PXQE_SERVER), never argv.
  `pxqe quantize SRC DST FTYPE [THREADS]`   the classic tiers (both editions); arguments pass straight to the public quantizer
  `pxqe run --src Q8.gguf --dst SKEL.gguf (--hdir DIR | --act DIR) [--device N] [--resume JID]`   Pro: PXQN tiers (LDLQ). With --act
                                            the Hessians are made first, in the same licence job, into <dst folder>/hess
  `pxqe prep --src Q8.gguf --act DIR --out DIR`   Pro: Hessians only
  the licence key and the licence server reach it through PXQE_KEY / PXQE_SERVER (never argv: argv shows in `ps`).
  A refused run prints `pxqe: refused by the licence server (HTTP 402): no encodes left: ...` and exits non-zero.

Nothing in here may name or open the quantizer's internals, and nothing may print the licence key.
"""

import json
import os
import re
import subprocess
import sys

CLI_NAMES = ("pxqe", "pxqe.py")
INFO_ARGV = ["info", "--json"]
INFO_CHECK_ARGV = ["info", "--json", "--check-licence"]
PROBE_TIMEOUT_S = 25
PROBE_TIMEOUT_CHECK_S = 45

EDITIONS = ("free", "pro")
MAKE_MIN_CLI = 2                 # `info --json` "cli" from this version on runs the whole flow as `pxqe make`
MAKE_PREFIX = "@pxqe "
REQUANTIZE_ARGS = ("--allow-requantize", "--i-know-this-is-double-lossy")
# licence.state words the encoder may use -> the few states the page knows. Unknown words are "unknown":
# the page never trusts this for anything but wording (the encoder and the licence server enforce).
_STATE_MAP = {"valid": "valid", "ok": "valid", "active": "valid", "licensed": "valid",
              "expired": "expired", "revoked": "revoked", "disabled": "revoked", "banned": "revoked", "suspended": "suspended", "paused": "paused",
              "refused": "refused", "none": "no_key", "missing": "no_key", "no_key": "no_key", "nokey": "no_key", "unset": "no_key",
              "no_server": "no_server", "unchecked": "unchecked", "unknown": "unchecked", "offline": "offline", "unreachable": "offline",
              "no_quota": "no_quota", "exhausted": "no_quota", "quota": "no_quota"}


class AdapterError(Exception):
    """Carries a plain-language sentence for the user (never a traceback)."""


# ---------------------------------------------------------------------------------------------
# running the command line
# ---------------------------------------------------------------------------------------------
def cli_argv(path, args):
    """argv for `<path> <args...>`; a .py wrapper runs under the interpreter that runs Control."""
    if path.endswith(".py"):
        return [sys.executable, path] + list(args)
    return [path] + list(args)


def run_cli(path, args, env=None, timeout=PROBE_TIMEOUT_S):
    """-> (returncode, stdout, stderr). Never raises: a missing or hung binary is a returncode and a sentence."""
    e = dict(os.environ)
    e.update(env or {})
    try:
        r = subprocess.run(cli_argv(path, args), capture_output=True, text=True, timeout=timeout, env=e, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "the encoder did not answer within %d s" % timeout
    except OSError as e2:
        return 127, "", "cannot run the encoder: %s" % e2.strerror


# ---------------------------------------------------------------------------------------------
# info --json
# ---------------------------------------------------------------------------------------------
def norm_tier(name):
    """'PXQN-3bal', 'pxqn3_bal' -> 'pxqn3bal'. The key every tier lookup uses."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _licence(o):
    if not isinstance(o, dict):
        return None
    st = _STATE_MAP.get(str(o.get("state", "")).strip().lower().replace("-", "_"), "unknown")
    left = o.get("encodes_left")
    left = left if isinstance(left, int) and not isinstance(left, bool) and left >= 0 else None
    exp = o.get("expires")
    out = {"state": st, "user": str(o.get("user") or "")[:80], "encodes_left": left, "unlimited": o.get("unlimited") is True,
           "key_id": str(o.get("key_id") or "")[:24], "checked": o.get("checked") is True, "reason": str(o.get("reason") or "")[:160],
           "expires": exp if isinstance(exp, (str, int, float)) and not isinstance(exp, bool) else None}
    lk = lock_policy(o.get("lock"))             # only an encoder that knows locks, asked with --check-licence, has this
    if lk is not None:
        out["lock"] = lk
    return out


_RX_CUDA_LIB = re.compile(r"(lib(?:cublas|cublasLt|cusolver|cusparse|nvJitLink)\.so(?:\.\d+)*): cannot open shared object")
_RX_DRIVER_LIB = re.compile(r"libcuda\.so(?:\.\d+)*: cannot open shared object")
RUNTIME_FIXES = ("runtime-pack", "driver", "reinstall")


def _runtime(o):
    """runtime{}: lib none | loadable | unloadable | missing (else unknown), and for an encoder that cannot load: `missing` (the library names), `fix`
    (runtime-pack = Control can download the GPU runtime; driver = install the NVIDIA driver), `driver`, `source` (where the CUDA libraries were found),
    `resolver` (the wrapper searches for them itself; absent in older wrappers, which only say what the loader said)."""
    if not isinstance(o, dict):
        return {"lib": "unknown", "detail": "", "missing": [], "fix": "", "driver": True, "source": "", "resolver": 0}
    lib = str(o.get("lib") or "unknown")
    lib = lib if lib in ("none", "loadable", "unloadable", "missing") else "unknown"
    detail = str(o.get("detail") or "")[:160]
    miss = [str(x)[:40] for x in o.get("missing") or [] if isinstance(x, str)][:8]
    fix = str(o.get("fix") or "")
    fix = fix if fix in RUNTIME_FIXES else ""
    driver = o.get("driver") is not False
    r = o.get("resolver")
    out = {"lib": lib, "detail": detail, "missing": miss, "fix": fix, "driver": driver, "source": str(o.get("source") or "")[:12],
           "resolver": r if isinstance(r, int) and not isinstance(r, bool) and r > 0 else 0}
    if lib == "unloadable" and not fix:             # an older wrapper: only the loader's own sentence. Read it, so the page can still say what is missing.
        if _RX_DRIVER_LIB.search(detail):
            out.update(fix="driver", driver=False, missing=["libcuda.so.1"])
        else:
            m = _RX_CUDA_LIB.search(detail)
            if m:
                out.update(fix="runtime-pack", missing=[m.group(1)])
    return out


def runtime_missing_text(rt):
    """One plain sentence for what `runtime` says is missing (the checks and the Encode tab use it)."""
    rt = rt or {}
    if rt.get("fix") == "driver" or rt.get("driver") is False:
        return "No NVIDIA driver was found (libcuda.so.1 is missing). The Pro encoder needs an NVIDIA graphics card with its driver (version 525 or newer)."
    if rt.get("fix") == "runtime-pack":
        names = ", ".join(rt.get("missing") or []) or "libcublas.so.12, libcusolver.so.11"
        return "The Pro encoder needs the NVIDIA CUDA 12 libraries (cuBLAS and cuSOLVER) and they are not on this computer (missing: %s)." % names
    return "The Pro encoder cannot load on this machine%s." % ((": " + rt["detail"]) if rt.get("detail") else "")


def _cli_version(o):
    """`cli` in info --json: "2" (a string in the shipped encoder) or 2. Missing or unreadable = 1 (the older multi-command flow)."""
    v = o.get("cli") if isinstance(o, dict) else None
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        return 1
    return n if 0 <= n < 1000 else 1


def _stages(lst):
    """info --json `stages[]`: every stage this package can run and whether it can run on THIS machine, with a plain-sentence reason."""
    out = []
    for r in lst if isinstance(lst, list) else []:
        if isinstance(r, dict) and isinstance(r.get("name"), str) and re.match(r"^[a-z0-9_-]{1,20}$", r["name"]):
            out.append({"name": r["name"], "does": str(r.get("does") or "")[:80], "edition": str(r.get("edition") or "")[:8],
                        "ready": r.get("ready") is True, "reason": str(r.get("reason") or "")[:240]})
    return out


def _make(o):
    """info --json `make{}`: the facts about the one-command flow. {} when the encoder has none (CLI version 1)."""
    if not isinstance(o, dict):
        return {}
    ec = o.get("exit_codes") if isinstance(o.get("exit_codes"), dict) else {}
    return {"prefix": str(o.get("progress_prefix") or MAKE_PREFIX)[:12], "sources": [str(x)[:20] for x in o.get("sources") or [] if isinstance(x, str)][:8],
            "classic_tiers": [norm_tier(x) for x in o.get("classic_tiers") or [] if isinstance(x, str)][:16],
            "lock_modes": [x for x in o.get("lock_modes") or [] if x in LOCK_MODES][:4],
            "exit_codes": {str(k)[:4]: str(v)[:120] for k, v in ec.items()}, "error": str(o.get("error") or "")[:60]}


def supports_make(info):
    """True when this encoder runs the whole flow with `pxqe make` (CLI version 2 or later). An older one keeps the multi-command path."""
    return bool(info) and (info.get("cli") or 1) >= MAKE_MIN_CLI and not (info.get("make") or {}).get("error")


# ---------------------------------------------------------------------------------------------
# locked files: "Who can load this file" (Pro, PXQN tiers)
# ---------------------------------------------------------------------------------------------
LOCK_MODES = ("personal", "supporters", "open")      # `pxqe make --lock` values (besides auto), in the order the page lists them
LOCK_LABEL = {"personal": "Only me (recommended)", "supporters": "Any PXA supporter", "open": "Anyone (no lock)"}
LOCK_SHORT = {"personal": "only me", "supporters": "any PXA supporter", "open": "anyone (no lock)"}
LOCK_CAN = {"personal": "only you can load", "supporters": "any PXA supporter can load", "open": "anyone can load (no lock)"}
LOCK_NOTE = {"personal": "Only you can load it. It is tied to your PXA account, so nobody else can, even with a copy of the file.",
             "supporters": "Any active PXA supporter can load it, and you always can.",
             "open": "No lock. Anyone with a PXA engine can load it, and can pass it on."}
LOCK_MIN_ENGINE = "v3"                  # the first PXA build whose engine loads a locked file (older ones refuse it, they never load it wrongly)
LOCK_OFF_TEXT = "File locking turns on with the next encoder update"
LOCK_UNKNOWN_TEXT = "Could not reach the licence server to see which locks your plan allows. Press Rescan to try again."
LOCK_NONE = {"enabled": False, "allowed": ["open"], "default": "open", "epoch": ""}      # a licence server that sends no lock block (an older one): same as switched off


def lock_policy(o):
    """The licence server's `lock` block (`pxqe status`; `licence.lock` of `info --json --check-licence`): {enabled, allowed_modes, default_mode, epoch}
    -> {"enabled": bool, "allowed": [modes, in the page's order], "default": mode | "", "epoch": str}, or None when there is no such block.
    The page never trusts it for anything but wording and which choices to offer: the encoder and the licence server enforce."""
    if not isinstance(o, dict):
        return None
    al = o.get("allowed_modes")
    allowed = [m for m in LOCK_MODES if isinstance(al, list) and m in al]
    d = str(o.get("default_mode") or "").strip().lower()
    return {"enabled": o.get("enabled") is True, "allowed": allowed, "default": d if d in allowed else "", "epoch": str(o.get("epoch") or "")[:8]}


def supports_lock(info):
    """True when this encoder's `make` takes --lock (it lists the modes in `info --json` make.lock_modes). An older one is never given the flag."""
    return bool(info) and bool((info.get("make") or {}).get("lock_modes"))


def lock_view(info, policy):
    """What the Encode tab offers under "Who can load this file" for this encoder and this licence server's answer (`policy` = lock_policy() of it,
    LOCK_NONE for a server that sends no lock block, None when the server could not be asked). Pro only; the page shows it for a PXQN tier.
      {"state": "on" | "off" | "unknown", "reason": "encoder" | "server" | "", "choices": [{"id", "label", "note"}], "default": mode | "", "text": the one line
       shown while the choice is not available, "needs": the first PXA build that loads a locked file}
    off = the encoder cannot lock yet (no --lock) or the server's lock switch is off: the choice is shown disabled and the file is written without a lock."""
    base = {"state": "off", "reason": "", "choices": [], "default": "", "text": "", "needs": LOCK_MIN_ENGINE}
    if not info or info.get("edition") != "pro":
        return None
    if not supports_lock(info):
        return dict(base, reason="encoder", text=LOCK_OFF_TEXT)
    if policy is None:
        return dict(base, state="unknown", text=LOCK_UNKNOWN_TEXT)
    if not policy["enabled"]:
        return dict(base, reason="server", text=LOCK_OFF_TEXT)
    allowed = policy["allowed"]
    if not allowed:             # the switch is on but the server names no mode this Control knows
        return dict(base, state="unknown", text=LOCK_UNKNOWN_TEXT)
    return dict(base, state="on", choices=[{"id": m, "label": LOCK_LABEL[m], "note": LOCK_NOTE[m]} for m in allowed],
                default="personal" if "personal" in allowed else (policy["default"] or allowed[0]))


def lock_mode_for_run(view, tier, want=None):
    """The `--lock` value a run gets -> (mode | None, refusal sentence | None). None = pass no flag (an encoder that cannot lock, a classic tier, or a
    licence server that could not be asked: the encoder then asks for itself and uses the plan's default). While the licence server's switch is off the
    choice is not offered and the file is written open, so that is what is asked for (never a lock the page did not show)."""
    if not view or is_classic(tier):
        return None, None
    if view["state"] == "on":
        ids = [c["id"] for c in view["choices"]]
        if not want:
            return view["default"], None
        if want in ids:
            return want, None
        return None, "Your plan does not allow a file that %s. It allows: %s." % (LOCK_CAN.get(want, "way"), ", ".join(LOCK_SHORT[i] for i in ids))
    if view["state"] == "off" and view["reason"] == "server":
        return "open", None
    return None, None


def lock_done_view(mode, file_id=""):
    """The Done screen's reading of a finished PXQN file's lock. `mode` is what the file's own header says: personal | supporters | locked (locked, but the
    header does not say how) | open. -> {"mode", "locked", "title", "text", "needs", "file_id"} (plain sentences)."""
    needs = "It loads in PXA %s or newer. Older PXA builds refuse a locked file instead of loading it wrongly." % LOCK_MIN_ENGINE
    if mode == "personal":
        return {"mode": mode, "locked": True, "title": "Locked to you", "needs": needs, "file_id": file_id,
                "text": "Only you can load this file. PXA checks your PXA key when it loads the file; nobody else can, even with a copy of it."}
    if mode == "supporters":
        return {"mode": mode, "locked": True, "title": "Locked to PXA supporters", "needs": needs, "file_id": file_id,
                "text": "Any active PXA supporter can load this file, and so can you. PXA checks the supporter's key when it loads the file."}
    if mode == "locked":
        return {"mode": mode, "locked": True, "title": "Locked", "needs": needs, "file_id": file_id,
                "text": "This file is locked: PXA checks a PXA key when it loads the file."}
    return {"mode": "open", "locked": False, "title": "Not locked", "needs": "", "file_id": "",
            "text": "Anyone with a PXA engine can load this file."}


def parse_info(text):
    """The text `info --json` printed -> the normalised dict, or AdapterError with a plain sentence."""
    try:
        o = json.loads(text)
    except (TypeError, ValueError):
        raise AdapterError("it did not print valid JSON for `info --json`: this is not a PXA Quantizer, or it is too old")
    if not isinstance(o, dict):
        raise AdapterError("`info --json` did not print a JSON object")
    ed = str(o.get("edition", "")).strip().lower()
    if ed not in EDITIONS:
        raise AdapterError("it reports edition %r; expected free or pro" % ed[:20])
    tiers = o.get("tiers")
    if not isinstance(tiers, list) or not all(isinstance(t, str) for t in tiers):
        raise AdapterError("it did not list its quantization tiers")
    feats = o.get("features")
    feats = [str(f)[:40] for f in feats] if isinstance(feats, list) else []
    build = str(o.get("build_id") or "").strip()
    if not re.match(r"^[A-Za-z0-9._\-]{1,64}$", build):
        raise AdapterError("it did not report a usable build id")
    return {"edition": ed, "version": str(o.get("version") or "")[:40], "build_id": build,
            "tiers": [norm_tier(t) for t in tiers if norm_tier(t)], "tier_names": [t for t in tiers],
            "features": feats, "licence": _licence(o.get("licence")) if ed == "pro" else None, "runtime": _runtime(o.get("runtime")),
            "platform": str(o.get("platform") or "")[:40], "cuda_major": o.get("cuda_major") if isinstance(o.get("cuda_major"), int) else None,
            "cli": _cli_version(o), "stages": _stages(o.get("stages")), "make": _make(o.get("make"))}


def probe(path, check_licence=False, env=None):
    """Run `info --json` on one candidate. -> a dict that always has path/ok/error; ok means usable. `env` carries the licence
    key and server (PXQE_KEY / PXQE_SERVER): the CLI reads them to say `unchecked` instead of `none`, and to ask the server."""
    out = {"path": path, "ok": False, "error": None, "edition": None, "version": None, "build_id": None,
           "tiers": [], "tier_names": [], "features": [], "licence": None, "runtime": {"lib": "unknown", "detail": ""},
           "cli": 1, "stages": [], "make": {}}
    if not os.path.isfile(path):
        out["error"] = "that file does not exist"
        return out
    rc, so, se = run_cli(path, INFO_CHECK_ARGV if check_licence else INFO_ARGV, env=env,
                         timeout=PROBE_TIMEOUT_CHECK_S if check_licence else PROBE_TIMEOUT_S)
    if rc != 0 and not so.strip():
        out["error"] = (se.strip().splitlines() or ["it exited with an error"])[-1][:200]
        return out
    try:
        out.update(parse_info(so))
        out["ok"] = True
    except AdapterError as e:
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------------------------
# where an installed encoder may be
# ---------------------------------------------------------------------------------------------
def install_root():
    """Where Control installs the encoders it downloads: ~/.local/share/pxa/encoder/<edition>/<build_id>/."""
    d = os.environ.get("PXA_ENCODER_HOME")
    if d:
        return d
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "pxa", "encoder")


def cli_in_dir(d):
    """The command-line file inside a package directory, if any (a few usual layouts)."""
    for sub in ("", "bin", "wrapper"):
        for n in CLI_NAMES:
            p = os.path.join(d, sub, n)
            if os.path.isfile(p):
                return p
    return None


def candidate_paths(configured=(), engine_dirs=(), path_env=None):
    """Every place an encoder command line may be, in the order of trust: the ones the user set, PXA_ENCODER,
    the install root, the engine's folders, PATH. Existing regular files only, de-duplicated by real path."""
    cands = [c for c in configured if c]
    env = os.environ.get("PXA_ENCODER")
    if env:
        cands.append(env)
    root = install_root()
    try:
        for ed in sorted(os.listdir(root)):
            for b in sorted(os.listdir(os.path.join(root, ed))):
                p = cli_in_dir(os.path.join(root, ed, b))
                if p:
                    cands.append(p)
    except OSError:
        pass
    for d in engine_dirs:
        for sub in ("", "bin", "lib", "tools"):
            p = cli_in_dir(os.path.join(d, sub)) if sub else cli_in_dir(d)
            if p:
                cands.append(p)
    for d in (path_env if path_env is not None else os.environ.get("PATH", "")).split(os.pathsep):
        if d:
            for n in CLI_NAMES:
                cands.append(os.path.join(d, n))
    out, seen = [], set()
    for c in cands:
        c = os.path.expanduser(c)
        if os.path.isdir(c):
            c = cli_in_dir(c) or c
        if not os.path.isfile(c):
            continue
        rp = os.path.realpath(c)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(c)
    return out


def discover(configured=(), engine_dirs=(), path_env=None, probe_fn=None, env=None):
    """Probe every candidate. Usable ones first, Pro before Free; broken ones are kept (with their error) so the
    page can say 'found X but it did not answer' instead of silently showing nothing."""
    probe_fn = probe_fn or (lambda p: probe(p, env=env))
    found = [probe_fn(p) for p in candidate_paths(configured, engine_dirs, path_env)]
    rank = {"pro": 0, "free": 1}
    found.sort(key=lambda r: (0 if r["ok"] else 1, rank.get(r.get("edition"), 2)))
    return found


def choose(found, preferred=None):
    """The encoder a run uses: the user's pick if it is usable, else the first Pro, else the first Free."""
    usable = [f for f in found if f["ok"]]
    if preferred:
        for f in usable:
            if os.path.realpath(f["path"]) == os.path.realpath(preferred):
                return f
    return usable[0] if usable else None


# ---------------------------------------------------------------------------------------------
# argv + environment for the verbs
# ---------------------------------------------------------------------------------------------
def run_env(info, key=None, server=None):
    """The environment an encoder run gets. The key goes here and ONLY here (never argv, never a log line)."""
    env = {}
    if info and info.get("edition") == "pro":
        if key:
            env["PXQE_KEY"] = key
        if server:
            env["PXQE_SERVER"] = server
    return env


def runtime_env(info, pack=None, engine_dir=None, probing=False):
    """The environment that lets the encoder find the NVIDIA CUDA libraries (cuBLAS / cuSOLVER), no secrets. `pack` = pxa_encode_pkg.installed_runtime() (the GPU
    runtime Control downloaded), `engine_dir` = a PXA engine install the wrapper may also look in. The wrapper (resolver 1) reads PXQE_RUNTIME_DIR /
    PXQE_ENGINE and loads the libraries itself, only in the processes that need them. An older wrapper cannot, so it gets the folder on the loader's
    path (LD_LIBRARY_PATH: the process tree of `pxqe` only). A probe does not know the wrapper's age yet, so it always gets both."""
    env = {}
    if pack and pack.get("dir"):
        env["PXQE_RUNTIME_DIR"] = pack["dir"]
        old = probing or not ((info or {}).get("runtime") or {}).get("resolver")
        if old and (probing or (info and info.get("edition") == "pro")):
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = pack["lib_dir"] + ((os.pathsep + cur) if cur else "")
    if engine_dir:
        env["PXQE_ENGINE"] = engine_dir
    return env


def _dev(device):
    return ["--device", str(int(device))] if device is not None else []


def argv_prep(cli, src, act, out, device=None, resume=None):
    """Pro: the Hessians only (`run --act` makes them in the same job as the encode, which is what the pipeline uses)."""
    a = ["prep", "--src", src, "--act", act, "--out", out] + _dev(device)
    if resume:
        a += ["--resume", resume]
    return cli_argv(cli, a)


def hess_dir_for(dst):
    """Where `run --act` puts the Hessians it builds: <folder of the first --dst>/hess."""
    return os.path.join(os.path.dirname(os.path.abspath(dst)), "hess")


def argv_run(cli, src, dst, hdir=None, act=None, device=None, resume=None, layers=None):
    """Pro `run`: encode SRC into DST (a skeleton copy). With `act` the Hessians are made first, in the same licence job (one
    charge), into hess_dir_for(dst); with `hdir` they are read from a finished folder; with neither the encode is plain
    round-to-nearest."""
    a = ["run", "--src", src, "--dst", dst]
    if act:
        a += ["--act", act]
    elif hdir:
        a += ["--hdir", hdir]
    a += _dev(device)
    if layers:
        a += ["--layers", layers]
    if resume:
        a += ["--resume", resume]
    return cli_argv(cli, a)


def argv_quantize(cli, src, dst, ftype, threads, allow_requantize=False):
    """The classic tiers (both editions): `pxqe quantize [flags] SRC DST FTYPE THREADS`. Quantizing from a Q8_0 GGUF is a second lossy
    pass: the public quantizer refuses it unless BOTH --allow-requantize and --i-know-this-is-double-lossy are given (so the
    page warns before it asks for it)."""
    flags = ["--allow-requantize", "--i-know-this-is-double-lossy"] if allow_requantize else []
    return cli_argv(cli, ["quantize"] + flags + [src, dst, ftype, str(int(threads))])


# The classic tiers Control can run with just a type name. PXQ_UNIVERSAL is a per-tensor mix that needs a tier map: Control lists
# it (the build does) but does not offer it.
RUNNABLE_CLASSIC = ("pxq1", "pxq2", "pxq3", "pxq4", "pxq4hq", "pxq6")


def is_classic(tier):
    """Classic PXQ tiers go through `pxqe quantize`; PXQN tiers need the skeleton + Hessian + `run` pipeline."""
    t = norm_tier(tier)
    return t.startswith("pxq") and not t.startswith("pxqn")


# ---------------------------------------------------------------------------------------------
# `pxqe make` (CLI version 2): the whole flow in one command
# ---------------------------------------------------------------------------------------------
# make's stage names -> the ids of the stage bars on Control's page (the page's words are older than the encoder's).
MAKE_STAGE_ID = {"fetch": "download", "convert": "convert", "q8": "reference", "skeleton": "skeleton", "dump": "dump", "hess": "hessians",
                 "encode": "encode", "quantize": "quantize", "verify": "verify"}
JOB_STAGE_MAKE = {v: k for k, v in MAKE_STAGE_ID.items()}
_MAKE_STATES = ("start", "progress", "done", "skipped")


def make_tier(tier):
    """Control's normalised tier key -> the name `pxqe make --tier` takes (pxq_universal keeps its underscore)."""
    t = norm_tier(tier)
    return {"pxquniversal": "pxq_universal"}.get(t, t)


def make_env(info, tier, key=None, server=None, python=None):
    """The environment a `make` run gets. The key goes here and ONLY here (never argv, never a log line), and only for a run that needs the
    licence server: a PXQN tier on the Pro edition. The classic tiers need no licence, so they get no key. `python` is the interpreter that has
    torch + transformers for the converter (PXQE_PYTHON; also passed as --python)."""
    env = {}
    if info and info.get("edition") == "pro" and key and not is_classic(tier):
        env["PXQE_KEY"] = key
        if server:
            env["PXQE_SERVER"] = server
    if python:
        env["PXQE_PYTHON"] = python
    return env


def argv_make(cli, source, tier, out, work, device=None, python=None, engine=None, gpu_layers=None, threads=None, keep_work=False,
              overwrite=False, quantizer_args=(), lock=None):
    """`pxqe make SOURCE --tier T --out FILE --work DIR [...]`. SOURCE is a Hugging Face repo, a model folder or a .gguf. The SAME argv again
    resumes (state lives in <work>/state.json), so a job keeps its argv. `device` is the CUDA device for the GPU stages (an index into the
    job's CUDA_VISIBLE_DEVICES); `engine` is a PXA engine folder whose GPU calibration tool is used when it carries the hook. `lock` is who may load the
    finished file (personal | supporters | open; Pro, PXQN tiers, an encoder that lists make.lock_modes), or None for no flag. The lock is part of
    what a started job is, so a resume passes the same one."""
    a = ["make", source, "--tier", tier, "--out", out, "--work", work]
    if device is not None:
        a += ["--device", str(int(device))]
    if python:
        a += ["--python", python]
    if engine:
        a += ["--engine", engine]
        if gpu_layers is not None:
            a += ["--gpu-layers", str(int(gpu_layers))]
    if threads:
        a += ["--threads", str(int(threads))]
    if keep_work:
        a.append("--keep-work")
    if overwrite:
        a.append("--overwrite")
    for q in quantizer_args or ():
        a.append("--quantizer-arg=" + q)
    if lock in LOCK_MODES:
        a += ["--lock", lock]
    return cli_argv(cli, a)


def stages_not_ready(info, needed):
    """The stages in `needed` (make stage names) that `info --json` says cannot run on this machine -> [(name, plain reason)].
    A stage the encoder does not list is not held against it (older builds list fewer)."""
    by = {r["name"]: r for r in (info or {}).get("stages") or []}
    return [(n, by[n]["reason"] or "it cannot run on this machine") for n in needed if n in by and not by[n]["ready"]]


def _pct(v):
    """0..100 (any number) -> 0..1, clamped. Not a number -> None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        return None
    return max(0.0, min(1.0, float(v) / 100.0))


def _eta(v):
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < 10 ** 7 else None


def parse_make_line(line):
    """One output line of `pxqe make` -> an event dict, or None for everything that is not a progress line (human text, a broken line).
    Events (percentages are turned into 0..1):
      {"event": "plan", "stages": [{"name", "weight", "does"}], "tier", "out", "work", "source_kind"}
      {"event": "stage", "stage", "state": start|progress|done|skipped, "percent", "eta_s": int|None, "overall", "message"}
      {"event": "done", "out", "sha256", "size", "tier", "already", "lock", "lock_file_id", "lock_epoch"}
      {"event": "error", "stage", "code", "message", "resumable"}"""
    if not isinstance(line, str):
        return None
    t = line.lstrip()
    if not t.startswith(MAKE_PREFIX):
        return None
    try:
        o = json.loads(t[len(MAKE_PREFIX):])
    except ValueError:
        return None
    if not isinstance(o, dict):
        return None
    ev = o.get("event")
    if ev == "plan":
        st = []
        for r in o.get("stages") if isinstance(o.get("stages"), list) else []:
            if isinstance(r, dict) and isinstance(r.get("name"), str) and r["name"] in MAKE_STAGE_ID:
                w = r.get("weight")
                st.append({"name": r["name"], "weight": float(w) if isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0 else 1.0,
                           "does": str(r.get("does") or "")[:80]})
        return {"event": "plan", "stages": st, "tier": str(o.get("tier") or "")[:30], "out": str(o.get("out") or ""), "work": str(o.get("work") or ""),
                "source_kind": str(o.get("source_kind") or "")[:12]}
    if ev == "stage":
        name, state = o.get("stage"), o.get("state")
        if name not in MAKE_STAGE_ID or state not in _MAKE_STATES:
            return None
        return {"event": "stage", "stage": name, "state": state, "percent": _pct(o.get("percent")), "eta_s": _eta(o.get("eta_s")),
                "overall": _pct(o.get("overall_percent")), "message": str(o.get("message") or "")[:200]}
    if ev == "done":
        sha, size = o.get("sha256"), o.get("size")
        if not (isinstance(sha, str) and re.match(r"^[0-9a-f]{64}$", sha) and isinstance(size, int) and not isinstance(size, bool) and size > 0 and isinstance(o.get("out"), str)):
            return None
        lk = str(o.get("lock") or "").lower()
        return {"event": "done", "out": o["out"], "sha256": sha, "size": size, "tier": str(o.get("tier") or "")[:30], "already": o.get("already") is True,
                "lock": lk if lk in LOCK_MODES else "", "lock_file_id": str(o.get("lock_file_id") or "")[:24], "lock_epoch": str(o.get("lock_epoch") or "")[:8]}
    if ev == "error":
        return {"event": "error", "stage": str(o.get("stage") or "")[:20], "code": str(o.get("code") or "failed")[:30], "message": str(o.get("message") or "")[:400],
                "resumable": o.get("resumable") is True}
    return None


# exit code -> what it means for the person (the encoder's own words are in `info --json` make.exit_codes; these are Control's)
MAKE_EXIT_TEXT = {
    0: "The encode finished.",
    1: "A step failed. The work so far is kept; press Resume to continue from there.",
    2: "This encode cannot run here: a tool or package is missing, the disk is too full, or a setting is wrong.",
    3: "The licence server refused this encode.",
    130: "The encode was stopped. Press Resume to continue where it stopped.",
}

# error `code` -> what to do about it. The sentence itself is the encoder's (it prints one plain sentence); this is the next step.
_MAKE_HINT = {
    "offline": "Check your internet connection and press Resume; it continues where it stopped.",
    "interrupted": "Press Resume; it continues where it stopped.",
    "short": "Press Resume; it continues where it stopped.",
    "checksum": "Press Resume; the damaged file was removed and is downloaded again.",
    "http": "Press Resume to try again.",
    "gated": "Accept the model's licence on huggingface.co and set HF_TOKEN, then press Resume.",
    "not-found": "Check the model name.",
    "no-weights": "Pick a model that has .safetensors files, or a GGUF file.",
    "disk-full": "Free some space (or pick another work folder in Advanced) and press Resume.",
    "no-tool": "Encode tab, Rescan. If it stays, reinstall the encoder (Get the encoder).",
    "no-runtime": "Encode tab, Download the GPU runtime (one time), then press Resume.",
    "no-python-packages": "Install them with pip (the Checks list names them), then start again.",
    "convert-failed": "The bundled converter may not know this architecture yet. Convert it with the PXA engine's own tools and pick the .gguf file instead.",
    "q8-failed": "Press Resume to try again; if it repeats, open the log.",
    "encoder": "Stop whatever else is using the card (Servers tab) and press Resume. If it repeats, open the log.",
    "busy": "Another encode is using this work folder. Wait for it, or cancel it, then press Resume.",
    "exists": "Pick another output folder, or delete the old file.",
    "usage": "Pick another source or tier.",
    "edition": "Pick a classic tier, or get the Pro encoder.",
    "bad-gguf": "Pick a complete GGUF file.",
    "no-file": "Pick a file that exists.",
    "stopped": "Press Resume to continue where it stopped.",
}


def explain_make_failure(rc, err=None, text=""):
    """A `pxqe make` that did not finish -> {code, message, hint, resumable}. `err` is its last `error` event (None if it died without one),
    `text` the human lines it printed last. Plain sentences, no traceback, no key. A licence refusal (exit 3) is worded the way the
    older flow words it, so the page reads the same either way."""
    err = err or {}
    msg = (err.get("message") or "").strip()
    code = err.get("code") or ""
    blob = (msg + "\n" + (text or "")).strip()
    resumable = bool(err.get("resumable")) if err else rc in (1, 130)
    if rc == 3 or code == "refused":
        ex = explain_failure(blob, rc)
        if ex["code"] == "failed":
            ex = {"code": "refused", "message": "The licence server refused this encode%s" % ((": " + msg.rstrip(".") + ".") if msg else "."),
                  "hint": "Check your key and your plan (Encode tab, I have a key), or use the Free encoder for the classic tiers."}
        return dict(ex, resumable=False)
    if code == "stopped" or rc == 130:
        return {"code": "stopped", "message": MAKE_EXIT_TEXT[130], "hint": "", "resumable": True}
    if code == "quantize-failed" and _RX_COMPOSITION.search(blob):
        return dict(explain_failure(blob, rc), resumable=False)
    if msg:
        # the encoder's own sentence, plus Control's next step. Licence / memory wording is understood the same way the older flow does.
        pre = explain_failure(blob, rc)
        if pre["code"] in ("device_limit", "paused"):          # Resume on this machine cannot succeed until the key is moved or resumed
            return dict(pre, resumable=False)
        if pre["code"] in ("no_quota", "revoked", "bad_key", "old_build", "offline", "lib_unloadable", "runtime_missing", "no_driver", "oom", "tier_locked") + LOCK_CODES:
            return dict(pre, resumable=resumable)
        return {"code": code or "failed", "message": msg if msg.endswith((".", "!", "?")) else msg + ".",
                "hint": _MAKE_HINT.get(code) or ("Press Resume to try again." if resumable else "Open the log for details."), "resumable": resumable}
    ex = explain_failure(blob, rc)
    if isinstance(rc, int) and rc < 0:        # killed from outside (the kernel's out-of-memory killer, a kill -9, a crash): it left no sentence
        return {"code": "killed", "message": "The encoder was stopped from outside (signal %d), for example because the computer ran out of memory. "
                "The work so far is kept; press Resume to continue." % -rc, "hint": ex["hint"] if ex["code"] != "failed" else "Close other programs, then press Resume.",
                "resumable": True}
    base = MAKE_EXIT_TEXT.get(rc, "The encoder stopped (exit %s)." % rc)
    if ex["code"] == "failed":
        return {"code": "failed", "message": base, "hint": ex["hint"], "resumable": rc in (1, 130)}
    return dict(ex, resumable=resumable)


# ---------------------------------------------------------------------------------------------
# skeleton: the file layout the encoder fills in (today's recipe: the engine's quantize tool in skeleton mode)
# ---------------------------------------------------------------------------------------------
# A uniform tier map: the linear layers take the tier, the few sensitive tensors stay at 8 bit, the embedding
# table stays 6 bit, everything else is kept as it is (the byte layout of the uniform ladder files).
_UNIFORM_MAP = r"""# PXA Control: uniform {tier} layout
\.(attn_gate|attn_qkv|attn_q|attn_output|ssm_out|ffn_gate|ffn_up|ffn_down)\.weight$ {tier}
\.(attn_k|attn_v|ssm_alpha|ssm_beta)\.weight$ q8_0
nextn\.eh_proj\.weight$ q8_0
^output\.weight$ q8_0
^token_embd\.weight$ q6_K
. keep
"""


def skeleton_tier_map(tier):
    """Text of the tier-map file for a uniform `tier`, or None when this Control does not know the layout
    of that tier (balanced / allocator tiers need the encoder's own planner). None = the page marks it unavailable."""
    t = norm_tier(tier)
    if not re.match(r"^pxqn?[0-9][a-z0-9]*$", t) or t.endswith("bal"):
        return None
    return _UNIFORM_MAP.format(tier=t)


def skeleton_env(tier_file, calib_sha, encoder_id):
    return {"PXQN_SKELETON": "1", "PXQN_TIERS": tier_file, "PXQN_ROT_SITES": "attn_in,ffn_in,down_in,out_in",
            "PXQN_HEAD": "q8_0", "PXQN_CALIB": calib_sha, "PXQN_ENCODER_ID": encoder_id}


def ftype_name(tier):
    """The quantize tool's name for a tier (PXQN4S8 -> 'PXQN4S8', pxq4hq -> 'PXQ4-HQ', pxq_universal -> 'PXQ_UNIVERSAL')."""
    t = norm_tier(tier)
    return {"pxq4hq": "PXQ4-HQ", "pxquniversal": "PXQ_UNIVERSAL"}.get(t, t.upper())


def quantize_tool_knows_pxqn(tool):
    """The PXQN skeleton writer is the engine's quantize tool in skeleton mode. Release builds do not have it yet; the tool lists the
    types it accepts in --help, so ask it instead of guessing from the file name."""
    rc, so, se = run_cli(tool, ["--help"], timeout=20)
    return "PXQN4" in (so + se)


# The activation dump: the engine's imatrix tool, run on the reference copy with the dump switched on for the inputs of the
# linear layers the encoder will need Hessians for.
def dump_env(act_dir):
    return {"PXA_PPL_PARSE_SPECIAL": "1", "PXQN_DUMP_DIR": act_dir,
            "PXQN_DUMP_RE": r"blk\.[0-9]+\.(attn_qkv|attn_q|attn_output|ffn_gate|ffn_up|ffn_down|ssm_out)\.weight$"}


def dump_argv(tool, model, calib, imatrix_out, threads, ngl):
    return [tool, "-m", model, "-f", calib, "-c", "512", "-b", "512", "-ub", "512", "-t", str(int(threads)),
            "-ngl", str(int(ngl)), "-fa", "on", "-sm", "layer", "-o", imatrix_out]


def calibration_file(cli_path):
    """The calibration text the dump runs on: PXA_CALIB, else next to the encoder (the package ships it), else the install root."""
    cands = [os.environ.get("PXA_CALIB", "")]
    if cli_path:
        d = os.path.dirname(os.path.realpath(cli_path))
        for sub in ("", "data", "share", ".."):
            cands += [os.path.join(d, sub, "calib.txt")]
    cands.append(os.path.join(install_root(), "calib.txt"))
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


# ---------------------------------------------------------------------------------------------
# progress lines
# ---------------------------------------------------------------------------------------------
_RX_JOB = re.compile(r"^(?:pxqe|licence): job (\S+) started(?: \((\S+) (?:encodes? )?left(?: this month| after this one)?\))?")
_RX_JOB_END = re.compile(r"^(?:pxqe|licence): job (\S+) (ok|failed)(?: \((?:quota|encode) refunded\))?")
_RX_JOB_LIB = re.compile(r"^pxqe: job (J-[0-9a-f]+) \(key ")        # what the encoder library prints when it takes a ticket
_RX_TOTAL = re.compile(r"^pxqe \w+: .*?\btensors (\d+)\b")
_RX_TENSOR = re.compile(r"^(?:\S+\s+)?(blk\.\d+\.\S+|\S*(?:output|token_embd)\S*)\s+t\d+\s+R\d+\s+K\d+\s+(ldlq|rtn)\b(.*)$")
_RX_SUMMARY = re.compile(r"^SUMMARY\b")
_RX_DONE = re.compile(r"^ALL DONE total (\d+)s")
_RX_HESS = re.compile(r"(blk\.\d+\.\w+)\.hess\s+K=(\d+)")


def parse_line(line):
    """One output line -> an event dict or None. Events: job, job_end, total, tensor, hess, summary, done."""
    s = line.strip()
    if not s:
        return None
    m = _RX_JOB.match(s)
    if m:
        left = m.group(2)
        return {"event": "job", "jid": m.group(1), "left": int(left) if left and left.isdigit() else None, "unlimited": left == "unlimited"}
    m = _RX_JOB_LIB.match(s)
    if m:
        return {"event": "job", "jid": m.group(1), "left": None, "unlimited": False}
    m = _RX_JOB_END.match(s)
    if m:
        return {"event": "job_end", "jid": m.group(1), "ok": m.group(2) == "ok", "refunded": "refunded" in s}
    m = _RX_TOTAL.match(s)
    if m:
        return {"event": "total", "tensors": int(m.group(1))}
    m = _RX_TENSOR.match(s)
    if m:
        rest = m.group(3)
        bad = re.search(r"mismatch|MISMATCH|not exact|FAILED", rest) is not None
        return {"event": "tensor", "name": m.group(1), "mode": m.group(2), "exact": (not bad) and bool(re.search(r"decode-exact|bit-exact", rest)),
                "bad": bad}
    m = _RX_HESS.search(s)
    if m:
        return {"event": "hess", "name": m.group(1), "k": int(m.group(2))}
    if _RX_SUMMARY.match(s):
        return {"event": "summary", "text": s[:200]}
    m = _RX_DONE.match(s)
    if m:
        return {"event": "done", "seconds": int(m.group(1))}
    return None


# ---------------------------------------------------------------------------------------------
# failures in plain words
# ---------------------------------------------------------------------------------------------
# (regex over the encoder's output, code, the sentence shown, what to do). First match wins.
_FAILS = [
    # machine binding and the sharing pause come BEFORE the generic 403 pattern: the server sends them as "(HTTP 403): <sentence>",
    # and the user needs the real next step (/encoder reset, or an admin), not "get a fresh key" (e2e-v3 #15203)
    (r"already in use on \d+ machines?|registered to a different machine|device limit", "device_limit",
     "This key is already registered to other machines (two at most), so it will not run on this one.",
     "Move it to this computer with /encoder reset in the PXA Network Discord (once every 30 days), or ask an admin there."),
    (r"looks shared|key is paused|paused because|quantizer key is paused", "paused",
     "This key is paused because it was used from too many networks in one day.",
     "Ask an admin in the PXA Network Discord to resume it. A new key does not help while this one is paused."),
    (r"(?:said|HTTP) 402|no encodes left|quota", "no_quota",
     "You have no encodes left on this key.",
     "Encodes come back when your monthly quota resets. Meanwhile the Free encoder (classic PXQ tiers) still works."),
    (r"(?:said|HTTP) 403|revoked|has expired|key (is )?expired|suspended|account disabled", "revoked",
     "The licence server refused this key: it is revoked, expired or suspended.",
     "Check your supporter role in the PXA Network Discord (use /encoder) for a fresh key, or use the Free encoder."),
    (r"(?:said|HTTP) 401|invalid key|not recognised", "bad_key",
     "The licence server does not recognise this key.", "Check the key you pasted (it starts with pxk1.), or ask for a new one in the PXA Network Discord (/encoder)."),
    (r"(?:said|HTTP) 409|unknown build|build .* (not accepted|unknown)|not supported any more", "old_build",
     "The licence server no longer accepts this encoder build.",
     "Update the encoder (Encode tab, Get the encoder, Update available)."),
    (r"(?:said|HTTP) 429|too many running|rate limit", "busy",
     "The licence server says too many requests, or another encode with this key is still running.", "Wait a few minutes, or cancel the other encode, then resume."),
    (r"cannot reach the licence server|licence server.*(timed out|unreachable)", "offline",
     "Cannot reach the licence server.",
     "Check your internet connection and press Resume. The Free encoder needs no connection."),
    (r"PXQE_KEY|set PXQE_KEY|no (licence |quantizer )?key", "no_key",
     "No licence key is set for the Pro encoder.", "Paste your key under Get the encoder (I have a key), or switch to the Free encoder."),
    (r"No NVIDIA driver was found|libcuda\.so\.1.*cannot open", "no_driver",
     "No NVIDIA driver was found on this computer.", "The Pro encoder needs an NVIDIA graphics card with its driver (version 525 or newer). The Free encoder needs no card."),
    (r"needs the NVIDIA CUDA 12 libraries|Download the GPU runtime|lib(cublas|cusolver|cusparse|nvJitLink)\S*: cannot open shared object", "runtime_missing",
     "The Pro encoder needs the NVIDIA CUDA 12 libraries (cuBLAS and cuSOLVER) and they are not on this computer.",
     "Encode tab, Download the GPU runtime (one time), then press Resume. Or install the CUDA 12 toolkit."),
    (r"cannot load the encoder library|libpxqe.*(not found|cannot open)|cannot find lib/libpxqe", "lib_unloadable",
     "The Pro encoder cannot load on this machine.",
     "It needs an NVIDIA driver and the CUDA 12 runtime (cuBLAS and cuSOLVER): Encode tab, Download the GPU runtime. Or use the Free encoder."),
    (r"cannot find libpxqe|cannot run the encoder", "broken_install",
     "The encoder is installed but cannot start.", "Reinstall it from Encode tab, Get the encoder."),
    (r"out of memory|cudaErrorMemoryAllocation|CUDA error: out of memory|cusolver.*alloc", "oom",
     "The card ran out of memory during the encode.",
     "Stop any server using it (Servers tab) and Resume, or pick a card with more free memory."),
    (r"No space left on device|disk full", "disk_full", "The disk is full.", "Free some space (or pick another work folder) and Resume."),
    (r"tier .*not (allowed|licensed)|not allowed for this key", "tier_locked",
     "This key does not include that tier.", "Pick another tier, or upgrade (Valued Supporter has all tiers)."),
    (r"invalid ftype|unknown quant|unrecognized quantization|not a valid quantization", "bad_tier",
     "The quantize tool does not know this tier.", "Update the engine build, or pick another tier."),
]


_RX_COMPOSITION = re.compile(r"PXQ composition assertion: target (\S+).*?produced ([\d.]+)% PXQ-family bytes \(floor (\d+)%")
_RX_INFORMATIVE = re.compile(r"(failed to quantize:|error|assert|cannot|can't|refus|invalid|unsupported|not supported)", re.I)


# Lock refusals: the encoder's own sentences name its command-line flag (`--lock open`), which means nothing on this page, so they are re-worded here.
# They come BEFORE the generic licence patterns: a refusal from the licence server arrives as "(HTTP 403): <sentence>", and the 403 pattern would
# read a lock refusal as a revoked key.
LOCK_CODES = ("lock_off", "lock_not_allowed", "lock_changed", "lock_not_granted", "lock_server", "lock_old_encoder", "lock_ticket", "lock_resume", "lock_splice")
_RX_LK_OLD = re.compile(r"unrecognized arguments:.*--lock|--lock.*(invalid choice|unrecognized)|lock mode \(\S+\) this encoder does not know", re.I)
_RX_LK_OFF = re.compile(r"locked files are not switched on|locking models is not switched on|does not offer locked files", re.I)
_RX_LK_NOT_ALLOWED = re.compile(r"may not write an? (open|supporters|personal)\b", re.I)
_RX_LK_ALLOWED = re.compile(r"modes? (?:you|it) may use: ([a-z, ]+)", re.I)
_RX_LK_CHANGED = re.compile(r"started with lock '(\w+)', not '(\w+)'", re.I)
_RX_LK_GRANT = re.compile(r"did not grant an? (\w+) lock for this job \(it granted (\w+)\)", re.I)
_RX_LK_KEYS = re.compile(r"no lock keys", re.I)
_RX_LK_TICKET = re.compile(r"ticket asks for an? \w+ lock but carries no lock key|lock key in the job ticket|does not open the lock key", re.I)
_RX_LK_RESUME = re.compile(r"is a locked file \(|is an open skeleton|has locked tensors but no lock header|lock header is incomplete|does not open the file's lock", re.I)
_RX_LK_SPLICE = re.compile(r"locked file:.*cannot be spliced", re.I)


def _lk(mode):
    return LOCK_SHORT.get(str(mode or "").lower(), str(mode or "that choice"))


def explain_lock_failure(text):
    """A lock-related refusal from the encoder (or the licence server through it) -> {code, message, hint}, plain sentences with no command-line flags in
    them, or None when the text is not about a lock."""
    blob = text or ""
    if _RX_LK_OLD.search(blob):
        return {"code": "lock_old_encoder", "message": "This encoder is too old to lock files, or does not know the kind of lock your plan asks for.",
                "hint": "Update the encoder (Encode tab, Update available), then start again."}
    if _RX_LK_OFF.search(blob):
        return {"code": "lock_off", "message": "Locking files is not switched on at the PXA licence server right now, so a locked file cannot be made. Nothing was charged.",
                "hint": "Press Check again and start once more: while locking is off the file is written without a lock. %s." % LOCK_OFF_TEXT}
    m = _RX_LK_NOT_ALLOWED.search(blob)
    if m:
        ma = _RX_LK_ALLOWED.search(blob)
        allowed = [x.strip().lower() for x in (ma.group(1) if ma else "").split(",") if x.strip().lower() in LOCK_MODES]
        msg = "Your plan does not allow a file that %s." % LOCK_CAN.get(m.group(1).lower(), "way")
        if allowed:
            msg += " It allows: %s." % ", ".join(_lk(x) for x in LOCK_MODES if x in allowed)
        return {"code": "lock_not_allowed", "message": msg, "hint": "Go back to Target and pick one of the choices under Who can load this file."}
    m = _RX_LK_CHANGED.search(blob)
    if m:
        return {"code": "lock_changed", "message": "This encode was started with the choice \"%s\" and cannot continue with \"%s\"." % (_lk(m.group(1)), _lk(m.group(2))),
                "hint": "Press Resume (it keeps the first choice), or discard this encode and start a new one with the other choice."}
    m = _RX_LK_GRANT.search(blob)
    if m:
        return {"code": "lock_not_granted", "message": "The licence server did not give this file the lock you chose (you chose \"%s\", it gave \"%s\"), so the encode was stopped. Nothing was charged."
                % (_lk(m.group(1)), _lk(m.group(2))), "hint": "Press Check again and start once more. If it repeats, tell us in the PXA Network Discord."}
    if _RX_LK_KEYS.search(blob):
        return {"code": "lock_server", "message": "The licence server cannot lock files at the moment (its lock keys are not set up). Nothing was charged.",
                "hint": "Try again later, or ask in the PXA Network Discord."}
    if _RX_LK_TICKET.search(blob):
        return {"code": "lock_ticket", "message": "The lock key for this encode did not arrive from the licence server in one piece, or does not match your key, so the file could not be locked.",
                "hint": "Press Resume to ask again. If it repeats, enter your key again under Get the encoder."}
    if _RX_LK_SPLICE.search(blob):
        return {"code": "lock_splice", "message": "A locked file cannot be mixed with another tier afterwards.", "hint": "Make the mix in one encode instead."}
    if _RX_LK_RESUME.search(blob):
        return {"code": "lock_resume", "message": "The unfinished files of this encode were made with a different lock than the one it now asks for, so it cannot continue them.",
                "hint": "Discard this encode and start a new one."}
    return None


def explain_failure(text, rc=None):
    """The encoder's (or a tool's) output -> {code, message, hint}. Plain sentences, no traceback, no key."""
    blob = text or ""
    lk = explain_lock_failure(blob)
    if lk:
        return lk
    m = _RX_COMPOSITION.search(blob)
    if m:       # the public quantizer's own refusal on a small model (the embedding table is a big share of the file)
        return {"code": "composition", "message": "The quantizer refused to write this file: only %s%% of it would be in the %s tier (it needs at least %s%%), so the "
                "file would misrepresent what it contains. This happens on small models, where the embedding table is a large part of the file." % (m.group(2), m.group(1), m.group(3)),
                "hint": "Pick a higher tier, or a larger model."}
    for rx, code, msg, hint in _FAILS:
        if re.search(rx, blob, re.I):
            return {"code": code, "message": msg, "hint": hint}
    lines = [ln.strip() for ln in blob.splitlines() if ln.strip() and not ln.startswith("Traceback") and not ln.startswith("  File ")]
    # the most informative recent line: the tools end with a generic "failed to quantize model from ..." after the real reason
    good = [ln for ln in lines if _RX_INFORMATIVE.search(ln) and not ln.startswith("main: failed to quantize model from")]
    tail = (good[-1] if good else (lines[-1] if lines else ""))[:260]
    msg = "The step failed" + (" (exit %s)" % rc if rc is not None else "") + (": " + tail if tail else ".")
    return {"code": "failed", "message": msg, "hint": "Open the log for details. Resume tries the step again; finished steps are kept."}
