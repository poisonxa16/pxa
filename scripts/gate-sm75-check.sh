#!/usr/bin/env bash
# gate-sm75-check.sh -- release scan gate: every CUDA binary carries SASS for every release arch (+ the PTX arch).
#
# Every release build ships SASS for sm 60;61;70;75;80;86;89 (P100, GTX 10xx/P40, V100, Turing, A100, RTX 30xx,
# RTX 40xx: the release floor + 80;86;89) and PTX for compute_89 only. A package passes only when EVERY file that
# holds CUDA device code shows every release arch in `cuobjdump --list-elf` and compute_89 PTX in its fatbins, and
# every embedded fatbin in it (one per compiled .cu file) carries all the SASS archs: a single .cu built with a narrower
# list would run its sm_70 cubin on a 7.5 card (binary compatible within a major) under host decisions made for 7.5.
#
# usage: gate-sm75-check.sh [options] PATH...
#   PATH       a build/bin dir, an unpacked package, a file, a .tar/.tar.gz/.tgz (release tarball OR a
#              `docker save` image tarball: nested layer tars are walked in-stream), or a .zip (Windows)
#   --archs LIST            required SASS archs (default "60 61 70 75 80 86 89", the release list)
#   --ptx LIST              required PTX archs (default "89"; --ptx= or --ptx "" = none required)
#   --work DIR              where archive members with device code are written for cuobjdump
#                           (default: a temp dir under ${TMPDIR:-/tmp}; removed at exit unless --keep)
#   --json FILE             machine-readable result
#   --vendor                also list NVIDIA's own libs (cuBLAS, cudart, NCCL ...): informational only,
#                           they never fail the gate (not built by us; their arch lists are NVIDIA's)
#   --allow GLOB            a documented exception (basename glob), reported as ALLOWED, never fails;
#                           repeatable. Keep it empty unless the release owner signs an exception off.
#   --no-cuobjdump          fatbin header scan only (no CUDA toolkit and no docker on this host)
#   --keep                  keep --work
# cuobjdump: the host's if on PATH (or $CUDA_HOME/bin), else docker image $PXA_CUOBJDUMP_IMAGE
# (default pxa-release-build:cu128-u2204-cc, run with --network none, NVIDIA_VISIBLE_DEVICES=none).
# Windows .dll/.exe (PE) files: the Linux cuobjdump cannot read them, the fatbin header scan decides.
# The scan and cuobjdump must agree on ELF files (SCAN-DISAGREES fails the file).
# Exit: 0 PASS, 2 FAIL (a CUDA binary misses an arch), 3 nothing to check / no device code found / tool error.
# On a shared box wrap it in `taskset -c <build cores> nice -n 15 ionice -c3`.
set -euo pipefail

ARCHS="60 61 70 75 80 86 89"; PTX="89"; WORK=""; JSON=""; VENDOR=0; NOCUO=0; KEEP=0; ALLOW=()
paths=()
while [ $# -gt 0 ]; do
  case "$1" in
    --archs) ARCHS="${2:?}"; shift ;;
    --ptx) PTX="${2?}"; shift ;;
    --ptx=*) PTX="${1#--ptx=}" ;;
    --work) WORK="${2:?}"; shift ;;
    --json) JSON="${2:?}"; shift ;;
    --vendor) VENDOR=1 ;;
    --allow) ALLOW+=("${2:?}"); shift ;;
    --no-cuobjdump) NOCUO=1 ;;
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "gate-sm75-check: unknown option $1" >&2; exit 3 ;;
    *) paths+=("$1") ;;
  esac
  shift
done
[ ${#paths[@]} -gt 0 ] || { echo "gate-sm75-check: no PATH given (try --help)" >&2; exit 3; }
command -v python3 >/dev/null || { echo "gate-sm75-check: python3 is required" >&2; exit 3; }

CLEAN=0
if [ -z "$WORK" ]; then WORK=$(mktemp -d "${TMPDIR:-/tmp}/gate-sm75.XXXXXX"); CLEAN=1; fi
mkdir -p "$WORK"
cleanup() { if [ "$CLEAN" = 1 ] && [ "$KEEP" = 0 ]; then rm -rf -- "$WORK"; fi; }
trap cleanup EXIT

# cuobjdump runner: prints "== <file>" then the --list-elf output for each file given on stdin (NUL-separated)
CUO=""
if [ "$NOCUO" = 0 ]; then
  if command -v cuobjdump >/dev/null; then CUO=host
  elif [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/cuobjdump" ]; then CUO=host; PATH="$CUDA_HOME/bin:$PATH"
  elif command -v docker >/dev/null && docker image inspect "${PXA_CUOBJDUMP_IMAGE:-pxa-release-build:cu128-u2204-cc}" >/dev/null 2>&1; then CUO=docker
  else echo "gate-sm75-check: no cuobjdump (host or docker image); falling back to the fatbin header scan" >&2; fi
fi
export GATE_ARCHS="$ARCHS" GATE_PTX="$PTX" GATE_WORK="$WORK" GATE_JSON="$JSON" GATE_VENDOR="$VENDOR" GATE_CUO="$CUO" \
       GATE_IMAGE="${PXA_CUOBJDUMP_IMAGE:-pxa-release-build:cu128-u2204-cc}"
GATE_ALLOW=$(printf '%s\n' "${ALLOW[@]+"${ALLOW[@]}"}"); export GATE_ALLOW

exec python3 -I - "${paths[@]}" <<'PY'
import fnmatch, io, json, os, re, struct, subprocess, sys, tarfile, zipfile

ARCHS = [int(a) for a in os.environ["GATE_ARCHS"].replace(";", " ").replace(",", " ").split()]
PTX = [int(a) for a in os.environ.get("GATE_PTX", "").replace(";", " ").replace(",", " ").split()]
WORK = os.environ["GATE_WORK"]; JSON = os.environ.get("GATE_JSON") or ""
VENDOR = os.environ.get("GATE_VENDOR") == "1"; CUO = os.environ.get("GATE_CUO") or ""
IMAGE = os.environ["GATE_IMAGE"]; ALLOW = [g for g in os.environ.get("GATE_ALLOW", "").split("\n") if g]

VENDOR_GLOBS = ["libcublas*", "libcublasLt*", "libcudart*", "libnccl*", "libcusparse*", "libcufft*", "libcurand*",
                "libcusolver*", "libnvrtc*", "libnvJitLink*", "libcudnn*", "libnvblas*", "libcupti*", "libnvToolsExt*",
                "cublas64_*", "cublasLt64_*", "cudart64_*", "nvrtc64_*", "nvJitLink*", "cusparse64_*", "cufft64_*",
                "curand64_*", "cusolver64_*", "cudnn*", "nccl*"]
FATBIN_MAGIC = b"\x50\xed\x55\xba"   # 0xBA55ED50 little-endian (fatbin container header)


def base(name):
    return re.split(r"[\\/!]", name)[-1]     # zip members from Windows packages use backslashes


def is_vendor(name):
    b = base(name)
    return any(fnmatch.fnmatch(b, g) for g in VENDOR_GLOBS)


def allowed(name):
    b = base(name)
    return next((g for g in ALLOW if fnmatch.fnmatch(b, g)), None)


def scan_fatbins(data):
    """Every fatbin container in a host binary (ELF or PE): list of {(kind, arch)} sets, kind 1 = PTX, 2 = SASS cubin.
    Container header: u32 magic, u16 version (1), u16 header size (16), u64 payload size. Entry header: u16 kind,
    u16 version, u32 header size, u64 payload size, ..., u32 arch at +28. Entries follow back to back."""
    out = []; pos = 0; n = len(data)
    while True:
        pos = data.find(FATBIN_MAGIC, pos)
        if pos < 0 or pos + 16 > n:
            break
        ver, hsz, fsz = struct.unpack_from("<HHQ", data, pos + 4)
        if ver != 1 or hsz != 16 or fsz == 0 or pos + 16 + fsz > n:
            pos += 4; continue
        ents = set(); p = pos + 16; end = pos + 16 + fsz; ok = True
        while p + 32 <= end:
            kind, _v, ehs, esz = struct.unpack_from("<HHIQ", data, p)
            if kind not in (1, 2) or ehs < 32 or ehs > 0x200 or p + ehs + esz > end:
                ok = False; break
            arch = struct.unpack_from("<I", data, p + 28)[0]
            ents.add((kind, arch))
            p += ehs + esz
        if ok and ents:
            out.append(ents)
            pos = end
        else:
            pos += 4
    return out


results = []      # one dict per checked file
to_cuo = []       # (result index, path on disk)


def check_bytes(label, data, disk_path=None):
    if not (data[:4] == b"\x7fELF" or data[:2] == b"MZ"):
        return
    if is_vendor(label) and not VENDOR:
        return
    fbs = scan_fatbins(data)
    if not fbs:
        return
    sass = sorted({a for fb in fbs for k, a in fb if k == 2}); ptx = sorted({a for fb in fbs for k, a in fb if k == 1})
    per_fb_missing = {a: sum(1 for fb in fbs if (2, a) not in fb) for a in ARCHS}
    r = {"file": label, "fatbins": len(fbs), "sass": sass, "ptx": ptx,
         "fatbins_missing": {f"sm_{a}": c for a, c in per_fb_missing.items() if c},
         "vendor": is_vendor(label), "allowed": allowed(label), "cuobjdump": None}
    results.append(r)
    if data[:2] == b"MZ":
        r["cuobjdump"] = {"error": "PE file: the Linux cuobjdump cannot read it, fatbin scan only"}
    elif CUO:
        if disk_path is None:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)[-180:]
            disk_path = os.path.join(WORK, f"{len(results):04d}-{safe}")
            with open(disk_path, "wb") as f:
                f.write(data)
        to_cuo.append((len(results) - 1, os.path.abspath(disk_path)))


class Peek(io.RawIOBase):
    """A read-once stream with its first bytes already taken (lets tarfile reopen a nested stream)."""
    def __init__(self, head, rest):
        self.head, self.rest = head, rest
    def readable(self):
        return True
    def readinto(self, b):
        if self.head:
            n = min(len(b), len(self.head)); b[:n] = self.head[:n]; self.head = self.head[n:]; return n
        chunk = self.rest.read(len(b)); n = len(chunk); b[:n] = chunk; return n


def looks_tar(head):
    return head[:2] == b"\x1f\x8b" or (len(head) >= 262 and head[257:262] == b"ustar")


def walk_tar(fileobj, prefix):
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = f"{prefix}!{m.name}"
            b = base(m.name)
            f = tf.extractfile(m)
            head = f.read(512)
            if looks_tar(head) and (m.name.endswith((".tar", ".tar.gz", ".tgz")) or b == "layer.tar"
                                    or "/blobs/" in "/" + m.name):
                walk_tar(io.BufferedReader(Peek(head, f), 1 << 20), name)
                continue
            if not (head[:4] == b"\x7fELF" or head[:2] == b"MZ"):
                continue
            if is_vendor(b) and not VENDOR:
                continue
            check_bytes(name, head + f.read())


def walk(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(root, fn)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                with open(p, "rb") as f:
                    head = f.read(4)
                if not (head == b"\x7fELF" or head[:2] == b"MZ") or (is_vendor(fn) and not VENDOR):
                    continue
                with open(p, "rb") as f:
                    check_bytes(p, f.read(), p)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for zi in z.infolist():
                if zi.is_dir():
                    continue
                with z.open(zi) as f:
                    head = f.read(4)
                if not (head == b"\x7fELF" or head[:2] == b"MZ") or (is_vendor(zi.filename) and not VENDOR):
                    continue
                check_bytes(f"{path}!{zi.filename}", z.read(zi))
    elif os.path.isfile(path):
        with open(path, "rb") as f:
            head = f.read(512)
        if looks_tar(head):
            with open(path, "rb") as f:
                walk_tar(f, path)
        else:
            with open(path, "rb") as f:
                check_bytes(path, f.read(), path)
    else:
        print(f"gate-sm75-check: {path}: not found", file=sys.stderr); sys.exit(3)


for p in sys.argv[1:]:
    walk(p)

tool_err = None
if CUO and to_cuo:
    script = 'for f in "$@"; do echo "== $f"; cuobjdump --list-elf "$f" 2>&1 || echo "!! cuobjdump exit $?"; done'
    files = [p for _, p in to_cuo]
    if CUO == "host":
        cmd = ["bash", "-c", script, "x"] + files
    else:
        mounts = []
        for d in sorted({os.path.dirname(p) for p in files}):
            mounts += ["-v", f"{d}:{d}:ro"]
        cmd = (["docker", "run", "--rm", "--network", "none", "-e", "NVIDIA_VISIBLE_DEVICES=none"] + mounts
               + ["--entrypoint", "bash", IMAGE, "-c", script, "x"] + files)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800).stdout
    except Exception as e:
        out = ""; tool_err = f"cuobjdump run failed: {e}"
    blocks = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("== "):
            cur = line[3:]; blocks[cur] = []
        elif cur is not None:
            blocks[cur].append(line)
    for i, p in to_cuo:
        lines = blocks.get(p)
        if lines is None:
            continue
        if any(l.startswith("!! cuobjdump exit") for l in lines) and not any(".cubin" in l for l in lines):
            results[i]["cuobjdump"] = {"error": " ".join(l for l in lines if l.strip())[:300]}
            continue
        archs = sorted({int(a) for l in lines for a in re.findall(r"\.sm_(\d+)a?\.cubin", l)})
        results[i]["cuobjdump"] = {"sass": archs, "cubins": sum(1 for l in lines if ".cubin" in l)}

fail = 0; checked = 0
print(f"gate-sm75-check: required SASS archs {' '.join('sm_%d' % a for a in ARCHS)}; PTX {' '.join('compute_%d' % a for a in PTX) or '-'}; cuobjdump: {CUO or 'none (fatbin scan)'}")
for r in results:
    cu = r["cuobjdump"]
    sass_cu = cu.get("sass") if isinstance(cu, dict) and "sass" in cu else None
    have = set(sass_cu if sass_cu is not None else r["sass"])
    miss = [a for a in ARCHS if a not in have]
    miss_ptx = [a for a in PTX if a not in r["ptx"]]
    disagree = sass_cu is not None and set(sass_cu) != set(r["sass"])
    bad = bool(miss) or bool(miss_ptx) or bool(r["fatbins_missing"]) or disagree
    if r["vendor"]:
        verdict = "INFO"
    elif bad and r["allowed"]:
        verdict = f"ALLOWED({r['allowed']})"
    elif bad:
        verdict = "FAIL"; fail += 1
    else:
        verdict = "PASS"
    if not r["vendor"]:
        checked += 1
    r["verdict"] = verdict; r["missing"] = [f"sm_{a}" for a in miss] + [f"compute_{a}" for a in miss_ptx]
    src = "cuobjdump" if sass_cu is not None else (("scan (PE)" if cu["error"].startswith("PE file") else "scan; cuobjdump: "
                                                    + cu["error"][:80]) if isinstance(cu, dict) else "scan")
    extra = ""
    if r["fatbins_missing"]:
        extra += " fatbins-missing " + ",".join(f"{k}x{v}" for k, v in r["fatbins_missing"].items())
    if disagree:
        extra += f" SCAN-DISAGREES(scan {r['sass']})"
    print(f"{verdict:<5} {r['file']}: sass {' '.join('sm_%d' % a for a in sorted(have)) or '-'} | ptx "
          f"{' '.join('compute_%d' % a for a in r['ptx']) or '-'} | fatbins {r['fatbins']} [{src}]"
          + (f" MISSING {' '.join(r['missing'])}" if r["missing"] else "") + extra)

status = 0
if tool_err:
    print(f"gate-sm75-check: {tool_err}", file=sys.stderr); status = 3
if checked == 0:
    print("gate-sm75-check: NO CUDA device code found in the given paths (a CPU-only build, or the wrong path) -> exit 3")
    status = 3
elif fail:
    print(f"gate-sm75-check: FAIL {fail} of {checked} CUDA binaries miss a release arch"); status = 2
elif status == 0:
    nal = sum(1 for r in results if r["verdict"].startswith("ALLOWED"))
    print(f"gate-sm75-check: PASS {checked - nal} CUDA binaries carry {' '.join('sm_%d' % a for a in ARCHS)}{''.join(' compute_%d' % a for a in PTX)}"
          + (f"; {nal} ALLOWED exception(s) miss an arch (--allow)" if nal else ""))
if JSON:
    with open(JSON, "w") as f:
        json.dump({"archs": ARCHS, "ptx": PTX, "cuobjdump": CUO or None, "status": status, "checked": checked, "fail": fail,
                   "files": results}, f, indent=1)
sys.exit(status)
PY
