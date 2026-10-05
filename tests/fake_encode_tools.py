#!/usr/bin/env python3
"""Stand-ins for the tools the Encode tab drives, for tests. NOT the quantizer and containing nothing of it: they only print the
same shape of progress lines (the format the adapter parses) and write tiny valid GGUF files, so the wizard, the job runner, the
adapter and the page can be exercised without a GPU, a model or the real encoder.

    fake_encode_tools.py TOOL DIR [args...]      TOOL: pxqe | llama-quantize | convert_hf_to_gguf | llama-imatrix
DIR holds fake.json (the behaviour switches) and fake-state.json (the licence counters).
"""
import json
import os
import signal
import struct
import sys
import time

TYPE_OF = {"PXQN4": 259, "PXQN3": 257, "PXQN3S8": 258, "PXQN2": 260, "PXQN1": 261, "PXQN4S8": 262, "PXQN5": 263,
           "PXQ4": 252, "PXQ4-HQ": 253, "PXQ2": 254, "PXQ3": 255, "PXQ6": 256, "PXQ1": 248, "Q8_0": 8, "BF16": 30}
PER_LAYER = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]


def cfg(d):
    try:
        with open(os.path.join(d, "fake.json")) as f:
            return json.load(f)
    except OSError:
        return {}


def state_of(d):
    return state(d)


def state(d):
    try:
        with open(os.path.join(d, "fake-state.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"left": None, "jobs": 0}


def save_state(d, s):
    with open(os.path.join(d, "fake-state.json"), "w") as f:
        json.dump(s, f)


def _s(b):
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, n_layers, tier_type, arch="qwen3", base_type=8, embed_type=8, tmp_suffix="", lock_mode=None, lock_id="L-0123456789abcdef"):
    """A tiny GGUF. `lock_mode` (personal | supporters) writes it the way a locked Pro file is: the pxa.lock.* header keys, and the PXQN tensors listed
    under type id 4096 + their type."""
    ts = []
    for l in range(n_layers):
        for n in PER_LAYER:
            ty = tier_type if n not in ("attn_k", "attn_v") else base_type
            if lock_mode and ty == tier_type:
                ty += 4096
            ts.append(("blk.%d.%s.weight" % (l, n), ty, [64, 64]))
    ts += [("token_embd.weight", embed_type, [128, 64]), ("output.weight", base_type, [128, 64])]
    kvs = [("general.architecture", 8, _s(arch.encode())), ("%s.block_count" % arch, 4, struct.pack("<I", n_layers)),
           ("%s.embedding_length" % arch, 4, struct.pack("<I", 64)), ("%s.attention.head_count" % arch, 4, struct.pack("<I", 4)),
           ("%s.attention.head_count_kv" % arch, 4, struct.pack("<I", 2)), ("%s.context_length" % arch, 4, struct.pack("<I", 4096))]
    if lock_mode:
        kvs += [("pxa.lock.version", 4, struct.pack("<I", 1)), ("pxa.lock.mode", 8, _s(lock_mode.encode())), ("pxa.lock.file_id", 8, _s(lock_id.encode())),
                ("pxa.lock.cipher", 8, _s(b"chacha20"))]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(ts), len(kvs))
    for k, t, v in kvs:
        out += _s(k.encode()) + struct.pack("<I", t) + v
    off = 0
    for name, ty, dims in ts:
        out += _s(name.encode()) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
        off += 64
    out += b"\0" * ((-len(out)) % 32) + b"\0" * 256
    with open(path + tmp_suffix, "wb") as f:
        f.write(out)
    return len(ts)


def read_tensors(path):
    """-> [(name, type)] from a GGUF header written by write_gguf."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _v, nt, nkv = struct.unpack("<IQQ", f.read(20))

        def rs():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n)
        for _ in range(nkv):
            rs()
            t, = struct.unpack("<I", f.read(4))
            if t == 8:
                rs()
            else:
                f.read(4)
        out = []
        for _ in range(nt):
            nm = rs().decode()
            nd, = struct.unpack("<I", f.read(4))
            f.read(8 * nd)
            ty, _off = struct.unpack("<IQ", f.read(12))
            out.append((nm, ty))
        return out


def layers_of(model_dir):
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return int(json.load(f).get("num_hidden_layers", 4))
    except (OSError, ValueError):
        return 4


def sleep(d):
    t = float(cfg(d).get("delay", 0))
    if t:
        time.sleep(t)


def arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


# ----------------------------------------------------------------------------------------------------------------------
def tool_quantize(d, a):
    if "--help" in a:
        types = "Q8_0 PXQ1 PXQ2 PXQ3 PXQ4 PXQ6" + ("" if cfg(d).get("no_pxqn") else " PXQN1 PXQN2 PXQN3 PXQN3S8 PXQN4 PXQN4S8 PXQN5")
        print("usage: llama-quantize [options] input.gguf [output.gguf] type [nthreads]\nallowed quantization types: " + types)
        return 0
    src, dst, ftype = a[0], a[1], a[2]
    if not os.path.isfile(src):
        print("quantize: cannot open %s" % src)
        return 1
    if ftype != "Q8_0" and os.environ.get("PXQN_SKELETON") != "1":
        print("quantize: this tool only writes skeletons with PXQN_SKELETON=1 (the Encode tab must set it)")
        return 1
    if ftype != "Q8_0" and not os.path.isfile(os.environ.get("PXQN_TIERS", "/nonexistent")):
        print("quantize: PXQN_TIERS file missing")
        return 1
    tt = TYPE_OF.get(ftype)
    if tt is None:
        print("quantize: unknown ftype %s" % ftype)
        return 1
    ts = read_tensors(src)
    n_layers = len([1 for nm, _ in ts if nm.endswith("attn_q.weight")])
    n_t = len(ts)
    for i in range(n_t):
        print("[%4d/%4d] %-40s ... -> %s" % (i + 1, n_t, ts[i][0], ftype), flush=True)
        sleep(d)
    base = 8
    write_gguf(dst, n_layers, tt if ftype != "Q8_0" else 8, base_type=base, embed_type=14 if ftype != "Q8_0" else 8)
    print("quantize: done")
    return 0


def tool_convert(d, a):
    src = a[0]
    outfile = arg(a, "--outfile")
    if not os.path.isfile(os.path.join(src, "config.json")) or not any(f.endswith(".safetensors") for f in os.listdir(src)):
        print("convert: no config.json / safetensors in %s" % src)
        return 1
    n = layers_of(src)
    for l in range(n):
        print("INFO:hf-to-gguf:blk.%d.attn_q.weight, torch.bfloat16 --> BF16, shape = {64, 64}" % l, flush=True)
        sleep(d)
    for pct in (50, 100):
        print("Writing: %3d%%|########| 1.0M/1.0M" % pct, flush=True)
    write_gguf(outfile, n, 30, base_type=30, embed_type=30)
    return 0


def tool_imatrix(d, a):
    dump = os.environ.get("PXQN_DUMP_DIR")
    if not dump or not os.environ.get("PXQN_DUMP_RE"):
        print("imatrix: PXQN_DUMP_DIR / PXQN_DUMP_RE not set (the Encode tab must set them)")
        return 1
    m = arg(a, "-m")
    if not os.path.isfile(m) or not os.path.isfile(arg(a, "-f")):
        print("imatrix: model or calibration file missing")
        return 1
    n_layers = len([1 for nm, _ in read_tensors(m) if nm.endswith("attn_q.weight")])
    print("compute_imatrix: computing over 5 chunks with batch_size 512", flush=True)
    for i in range(1, 6):
        print("[%d]%.4f," % (i, 30.0 / i), flush=True)
        sleep(d)
    for l in range(n_layers):
        for n in ("attn_q", "ffn_gate", "ffn_down"):
            with open(os.path.join(dump, "blk.%d.%s.weight.f16" % (l, n)), "wb") as f:
                f.write(b"\0" * 64)
    print("Final estimate: PPL = 26.4257 +/- 2.18009")
    return 0


CLASSIC_NAMES = {"pxq1", "pxq2", "pxq3", "pxq4", "pxq4hq", "pxq6", "pxq_universal"}
PXQN_NAMES = {"pxqn1", "pxqn2", "pxqn3", "pxqn3s8", "pxqn4", "pxqn4s8", "pxqn5"}
CLASSIC_TYPES = {k: v for k, v in TYPE_OF.items() if k.startswith("PXQ") and not k.startswith("PXQN")}
CLASSIC_TYPES["PXQ_UNIVERSAL"] = 252


RT_LIBS = ("libcublas.so.12", "libcusolver.so.11")
RT_SENTENCE = ("The encoder library cannot start on this computer. The Pro encoder needs the NVIDIA CUDA 12 libraries (cuBLAS and cuSOLVER) and could not find "
               "libcublas.so.12, libcusolver.so.11 on this computer. In PXA Control, open the Encode tab and press \"Download the GPU runtime\" (one time).")


def runtime_state(c):
    """The fake's reading of the NVIDIA CUDA libraries (fake.json `needs_runtime`: this 'computer' has none until the GPU runtime is found).
    A new wrapper (resolver 1) looks at PXQE_CUDA_LIBS, PXQE_RUNTIME_DIR/lib and PXQE_ENGINE/lib; an older one (`old_wrapper`) only at LD_LIBRARY_PATH,
    like the loader does. -> None (not applicable) | {"lib": "loadable", ...} | the unloadable block."""
    if c.get("edition") != "pro" or not c.get("needs_runtime"):
        return None

    def has(d):
        return bool(d) and all(os.path.isfile(os.path.join(d, n)) for n in RT_LIBS)
    if c.get("old_wrapper"):
        if any(has(d) for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)):
            return {"lib": "loadable"}
        return {"lib": "unloadable", "detail": "libcublas.so.12: cannot open shared object file: No such file or directory"}
    for src, dirs in (("env", os.environ.get("PXQE_CUDA_LIBS", "").split(os.pathsep)), ("pack", [os.path.join(x, "lib") for x in [os.environ.get("PXQE_RUNTIME_DIR", "")] if x]),
                      ("engine", [os.path.join(os.environ.get("PXQE_ENGINE", ""), "lib")] if os.environ.get("PXQE_ENGINE") else [])):
        if any(has(x) for x in dirs):
            return {"lib": "loadable", "source": src, "resolver": 1}
    return {"lib": "unloadable", "detail": "libcublas.so.12, libcusolver.so.11 not found", "missing": list(RT_LIBS), "driver": True, "fix": "runtime-pack", "resolver": 1,
            "need": {"cuda_major": 12, "libs": list(RT_LIBS)}}


def tool_pxqe(d, a):
    c = cfg(d)
    edition = c.get("edition", "free")
    if a[:1] == ["info"]:
        if c.get("info_broken") == "garbage":
            print("this is not json")
            return 0
        if c.get("info_broken") == "crash":
            print("Segmentation fault", file=sys.stderr)
            return 139
        lic = None
        if edition == "pro":
            key = os.environ.get("PXQE_KEY")
            lic = c.get("licence") or {"state": "unchecked" if key else "none", "user": "tester", "key_id": "K-TEST0001", "encodes_left": None,
                                       "unlimited": None, "expires": None, "checked": False}
            if "--check-licence" in a:
                lic = dict(c.get("licence_check") or {"state": "valid" if key else "no-key", "user": "tester", "key_id": "K-TEST0001", "encodes_left": 4,
                                                      "unlimited": False, "expires": "2026-12-01", "checked": True})
                if c.get("status_fail"):                                            # the licence server cannot be reached: the check says so and has no lock block
                    lic.update(state="unreachable", reason="URLError")
                elif c.get("lock_support") and c.get("lock_server") is not None:   # an encoder that knows locks copies the server's lock block into the licence block
                    lic["lock"] = c["lock_server"]
        cli = str(c.get("cli", "2"))
        o = {"edition": edition, "version": c.get("version", "1.0.0"), "build_id": c.get("build_id", "b-test"), "cli": cli,
             "platform": "linux-x86_64", "cuda_major": 12,
             "tiers": c.get("tiers", ["pxq2", "pxq3", "pxq4"]), "features": c.get("features", []),
             "runtime": runtime_state(c) or c.get("runtime") or {"lib": "loadable" if edition == "pro" else "none"}}
        if int(cli) >= 2:       # the shape of the real encoder's `info --json` (PACKAGE.md): stages[] with readiness, and make{}
            nr = dict(c.get("not_ready") or {})
            rts = runtime_state(c)
            if rts and rts["lib"] != "loadable":
                nr.update({n: RT_SENTENCE for n in ("skeleton", "hess", "encode", "splice")})
            names = [("fetch", "download the model", "both"), ("convert", "convert it to GGUF", "both"), ("quantize", "quantize with a classic tier", "both"),
                     ("verify", "check the finished file", "both")]
            if edition == "pro":
                names += [("q8", "make the Q8_0 source", "pro"), ("skeleton", "write the quantized file's skeleton", "pro"), ("dump", "run the calibration text through the model", "pro"),
                          ("hess", "build the per-layer statistics", "pro"), ("encode", "encode the weights", "pro"), ("splice", "mix tiers", "pro")]
            o["stages"] = [{"name": n, "does": t, "edition": e, "ready": n not in nr, "reason": nr.get(n, "")} for n, t, e in names]
            o["make"] = {"progress_prefix": "@pxqe ", "sources": ["hf-repo", "hf-dir", "gguf"], "classic_tiers": sorted(CLASSIC_NAMES),
                         "exit_codes": {"0": "done", "1": "a stage failed, run again to resume", "2": "cannot run here / bad arguments", "3": "refused by the licence server",
                                        "130": "stopped"}}
            if c.get("lock_support"):
                o["make"]["lock_modes"] = ["auto", "open", "supporters", "personal"]
        if lic is not None:
            o["licence"] = lic
        print(json.dumps(o, sort_keys=True))
        return 0
    if a[:1] == ["quantize"]:
        return pxqe_quantize(d, a[1:])
    if a[:1] == ["fp"]:
        print("fp-0123456789abcdef")
        return 0
    if a[:1] == ["status"]:
        if not os.environ.get("PXQE_KEY"):
            print("pxqe: set PXQE_KEY (your quantizer key) and PXQE_SERVER")
            return 1
        o = {"tier": c.get("plan_tier", "valued"), "status": "active", "remaining": 4, "unlimited": False, "used": 1}
        if c.get("allowed_tiers") is not None:
            o["allowed_tiers"] = c["allowed_tiers"]
        if c.get("lock_server") is not None:           # what the licence server says about locks (it sends the block whatever the encoder's age)
            o["lock"] = c["lock_server"]
        if c.get("status_fail"):
            print("pxqe: cannot reach the licence server (URLError)", file=sys.stderr)
            return 1
        print(json.dumps(o))
        return 0
    if a[:1] in (["run"], ["prep"]):
        return pxqe_run(d, c, edition, a)
    if a[:1] == ["make"] and int(c.get("cli", "2")) >= 2:
        return pxqe_make(d, c, edition, a[1:])
    print("pxqe: '%s' is not part of this edition." % (a[:1] or [""])[0], file=sys.stderr)
    return 2


def pxqe_quantize(d, a):
    """`pxqe quantize [--allow-requantize] SRC DST FTYPE [THREADS]`: the classic tiers, any edition, no licence."""
    requant = "--allow-requantize" in a
    double = "--i-know-this-is-double-lossy" in a
    a = [x for x in a if x not in ("--allow-requantize", "--i-know-this-is-double-lossy")]
    src, dst, ftype = a[0], a[1], a[2]
    if not os.path.isfile(src):
        print("quantize: cannot open %s" % src)
        return 1
    tt = CLASSIC_TYPES.get(ftype)
    if tt is None:
        print("quantize: invalid ftype %s" % ftype)
        return 1
    ts = read_tensors(src)
    if all(ty == 8 for nm, ty in ts if nm.startswith("blk.")) and not (requant and double):
        print("quantize: this model is already quantized (Q8_0); requantizing needs --allow-requantize AND --i-know-this-is-double-lossy")
        return 1
    n_layers = len([1 for nm, _ in ts if nm.endswith("attn_q.weight")])
    for i in range(len(ts)):
        print("[%4d/%4d] %-40s ... -> %s" % (i + 1, len(ts), ts[i][0], ftype), flush=True)
        sleep(d)
    write_gguf(dst, n_layers, tt, base_type=8, embed_type=14)
    print("quantize: done")
    return 0


def pxqe_run(d, c, edition, a):
    st = state(d)
    resume = arg(a, "--resume")
    if edition == "pro":
        key, srv = os.environ.get("PXQE_KEY"), os.environ.get("PXQE_SERVER")
        if not key or not srv:
            print("pxqe: set PXQE_KEY (your quantizer key) and PXQE_SERVER")
            return 1
        if c.get("fail") == "402":
            print("pxqe: refused by the licence server (HTTP 402): no encodes left: monthly quota used")
            return 1
        if c.get("fail") == "403":
            print("pxqe: refused by the licence server (HTTP 403): key revoked")
            return 1
        if c.get("fail") == "lib":
            print("pxqe: cannot load the encoder library (libcusolver.so.12: cannot open shared object file). It needs an NVIDIA driver and the CUDA 12 runtime (cuBLAS, cuSOLVER).")
            return 1
        if (runtime_state(c) or {"lib": "loadable"})["lib"] != "loadable":
            print("pxqe: cannot load the encoder library. " + RT_SENTENCE)
            return 1
        if c.get("fail") == "offline":
            print("pxqe: cannot reach the licence server (URLError)")
            return 1
        if resume:
            jid = resume
            st.setdefault("resumes", []).append(resume)
        else:
            st["jobs"] = st.get("jobs", 0) + 1
            jid = "J%d" % st["jobs"]
            if st.get("left") is None:
                st["left"] = int(c.get("left", 5))
            st["left"] -= 1
        save_state(d, st)
        print("pxqe: job %s started (%s encodes left)" % (jid, st.get("left")), flush=True)
        if c.get("leak_key"):
            print("debug: PXQE_KEY=%s server=%s" % (key, srv), flush=True)
    if c.get("fail") == "oom":
        print("CUDA error: out of memory")
        return 1
    dst = arg(a, "--dst")
    src = arg(a, "--src")
    act, hdir = arg(a, "--act"), arg(a, "--hdir")
    if a[0] == "prep":
        return 0
    ts = read_tensors(dst)
    todo = [nm for nm, ty in ts if nm.startswith("blk.") and ty not in (8, 30)]
    if act:
        hd = os.path.join(os.path.dirname(dst), "hess")
        os.makedirs(hd, exist_ok=True)
        layers = sorted({int(nm.split(".")[1]) for nm in todo})
        for l in layers:
            for site in ("attn_in", "ffn_in", "out_in", "down_in"):
                p = os.path.join(hd, "blk.%d.%s.hess" % (l, site))
                with open(p, "wb") as f:
                    f.write(b"\0" * 64)
                print("%s K=64 ntok=10240 0.1s" % p, flush=True)
                sleep(d)
    elif hdir:
        if not os.path.isdir(hdir):
            print("pxqe: Hessian folder %s missing" % hdir)
            return 1
    prog = dst + ".progress"
    done = set()
    try:
        with open(prog) as f:
            done = set(f.read().split())
    except OSError:
        pass
    print("pxqe encode: 1 file(s) layers 0-99 tensors %d damp 0.3 minratio 1.5 ldlq 1 dev FakeCard sm_70" % len(todo), flush=True)
    n_new = 0
    for nm in todo:
        if nm in done:
            continue
        crash = c.get("crash_after")
        if crash and n_new >= int(crash) and not os.path.exists(os.path.join(d, "crashed.flag")):
            open(os.path.join(d, "crashed.flag"), "w").close()
            os.kill(os.getpid(), signal.SIGKILL)
        print("%s %s t259 R64 K64 ldlq rot 1 relmse 0.012345 pack decode-exact enc 0s" % (os.path.basename(dst), nm), flush=True)
        sleep(d)
        with open(prog, "a") as f:
            f.write(nm + "\n")
        n_new += 1
    with open(dst, "ab") as f:
        f.write(b"ENCODED")
    print("SUMMARY %s ldlq %d rtn 0 skip 0" % (os.path.basename(dst), len(todo)))
    print("ALL DONE total 1s", flush=True)
    if edition == "pro":
        print("pxqe: job %s ok" % jid)
    return 0


class _Stop(Exception):
    pass


def ev(**kw):
    print("@pxqe " + json.dumps(kw, sort_keys=True, separators=(",", ":")), flush=True)


def _wr(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def pxqe_make(d, c, edition, a):
    """`pxqe make SOURCE --tier T --out FILE [--work DIR] ...`: the shape of the real one-command flow (PACKAGE.md): `@pxqe ` JSON lines on stdout,
    exit codes 0 / 1 / 2 / 3 / 130, a state file in the work folder so the SAME command again resumes, one licence job for the whole run (PXQN
    tiers, Pro). Everything it writes is a tiny GGUF; nothing here is the quantizer."""
    import hashlib
    import shutil
    import urllib.error
    import urllib.request
    src = a[0]
    tier = (arg(a, "--tier") or "").lower()
    out = os.path.abspath(arg(a, "--out"))
    work = os.path.abspath(arg(a, "--work") or out + ".work")
    keep = "--keep-work" in a
    qargs = [x.split("=", 1)[1] for x in a if x.startswith("--quantizer-arg=")]
    pxqn = tier in PXQN_NAMES
    key, srv = os.environ.get("PXQE_KEY"), os.environ.get("PXQE_SERVER")
    st0 = state_of(d)
    st0.setdefault("makes", []).append({"argv": ["make"] + a, "key_in_env": bool(key), "server_in_env": bool(srv), "key_in_argv": bool(key) and any(key in x for x in a),
                                        "python_env": os.environ.get("PXQE_PYTHON", ""), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")})
    save_state(d, st0)

    def fail(code, msg, exit_code, resumable=False, stage="start"):
        ev(event="error", stage=stage, code=code, message=msg, resumable=resumable)
        print("pxqe: " + msg, file=sys.stderr)
        return exit_code
    if "--lock" in a and not c.get("lock_support"):          # an encoder from before locks: its argument parser has never heard of the flag
        print("usage: pxqe make [-h] --tier TIER --out OUT [--work WORK] source\npxqe make: error: unrecognized arguments: --lock %s" % (arg(a, "--lock") or ""), file=sys.stderr)
        return 2
    if tier not in CLASSIC_NAMES and not pxqn:
        return fail("usage", "'%s' is not a tier this tool writes." % tier, 2)
    if pxqn and edition != "pro":
        return fail("edition", "The %s tier belongs to the Pro edition of this tool. This package writes the classic tiers." % tier, 2)
    if pxqn:
        # the licence server answers BEFORE anything is written or downloaded: a refusal leaves no trace
        if not key or not srv:
            return fail("no-key", "There is no quantizer key. Enter your key in PXA Control, or set PXQE_KEY and PXQE_SERVER.", 3)
        if c.get("fail") == "402":
            return fail("refused", "No encodes left: 3 of 3 used.", 3)
        if c.get("fail") == "403":
            return fail("refused", "The licence server refused this encode (HTTP 403): key revoked", 3)
        if c.get("fail") == "lib":
            return fail("no-lib", "The encoder library cannot start on this computer (libcusolver.so.12: cannot open shared object file). It needs an NVIDIA driver and the CUDA 12 runtime (cuBLAS, cuSOLVER).", 2)
        if (runtime_state(c) or {"lib": "loadable"})["lib"] != "loadable":
            return fail("no-runtime", RT_SENTENCE, 2)
        if c.get("fail") == "offline":
            return fail("offline", "Cannot reach the licence server (URLError). Check the internet connection and run the same command again.", 1, True)
    lock_mode = "open"
    if pxqn and c.get("lock_support"):
        # the real encoder's resolve_lock(): the requested --lock against what the licence server's key status allows (the same sentences)
        want = (arg(a, "--lock") or "auto").lower()
        srv_lock = c.get("lock_server")
        if not srv_lock:
            if want not in ("auto", "open"):
                return fail("refused", "This licence server does not offer locked files yet, so the file would be written open. Run with --lock open, or try again later.", 3)
        else:
            allowed = srv_lock.get("allowed_modes") or ["open"]
            if want == "auto":
                lock_mode = srv_lock.get("default_mode") or allowed[0]
            elif want not in allowed:
                if not srv_lock.get("enabled"):
                    return fail("refused", "Locked files are not switched on yet, so this encode could only be written open (unlocked). Run it with --lock open, or wait until PXA turns locking on.", 3)
                return fail("refused", "Your tier may not write %s %s file. The modes you may use: %s." % ("an" if want == "open" else "a", want, ", ".join(allowed)), 3)
            else:
                lock_mode = want
        print("lock: %s" % {"open": "none (the file loads on any PXA engine)", "supporters": "supporters (only active PXA supporters can load the file)",
                            "personal": "personal (only your own key can load the file)"}[lock_mode], flush=True)
        if c.get("lock_refuse"):            # the licence server refuses the job start with its own sentence (a server-side lock refusal)
            return fail("refused", "The licence server refused this encode (HTTP 403): %s" % c["lock_refuse"], 3)
        if c.get("lock_grant") and c["lock_grant"] != lock_mode:
            return fail("refused", "The licence server did not grant a %s lock for this job (it granted %s). Nothing was charged." % (lock_mode, c["lock_grant"]), 3)
    kind = "gguf" if os.path.isfile(src) and src.endswith(".gguf") else ("hf-dir" if os.path.isdir(src) else "hf-repo")
    gguf_q8 = False
    if kind == "gguf":
        ts = read_tensors(src)
        gguf_q8 = all(ty == 8 for nm, ty in ts if nm.startswith("blk."))
        if gguf_q8 and not pxqn and not ("--allow-requantize" in qargs and "--i-know-this-is-double-lossy" in qargs):
            return fail("usage", "%s is quantized (q8_0); classic tiers are made from the original F16/BF16 weights." % src, 2)
    names = ["fetch"] if kind == "hf-repo" else []
    if kind in ("hf-repo", "hf-dir"):
        names.append("convert")
    if pxqn:
        if kind != "gguf" or not gguf_q8:
            names.append("q8")
        names += ["skeleton", "dump", "hess", "encode", "verify"]
    else:
        names += ["quantize", "verify"]
    W = {"fetch": 8, "convert": 6, "q8": 4, "skeleton": 4, "dump": 30, "hess": 10, "encode": 32, "verify": 4, "quantize": 80}
    total = float(sum(W[n] for n in names))
    os.makedirs(work, exist_ok=True)
    stp = os.path.join(work, "state.json")
    try:
        with open(stp) as f:
            S = json.load(f)
    except (OSError, ValueError):
        S = {}
    S.setdefault("done", {})
    run_key = hashlib.sha256(("%s|%s%s" % (src, tier, ("|lock=" + lock_mode) if c.get("lock_support") else "")).encode()).hexdigest()[:16]
    locked = {"lock": lock_mode, "lock_file_id": "L-0123456789abcdef", "lock_epoch": "2026-10" if lock_mode == "supporters" else ""} if lock_mode != "open" else {}
    res = (S.get("results") or {}).get(run_key)
    if res and os.path.isfile(out) and os.path.getsize(out) == res["size"]:
        ev(event="done", out=out, sha256=res["sha256"], size=res["size"], tier=tier, already=True, **locked)
        print("done: %s already made" % out)
        return 0
    if os.path.exists(out) and "--overwrite" not in a and not any(r.get("out") == out for r in (S.get("results") or {}).values()):
        return fail("exists", "%s already exists and was not made by this working folder." % out, 2)
    ev(event="plan", stages=[{"name": n, "weight": W[n], "does": n} for n in names], tier=tier, out=out, work=work, source=src, source_kind=kind)
    print("pxqe make: %s -> %s (%s). Stages: %s" % (src, out, tier, ", ".join(names)), flush=True)

    def term(signum, frame):
        raise _Stop()
    signal.signal(signal.SIGTERM, term)
    signal.signal(signal.SIGINT, term)
    P = lambda *x: os.path.join(work, *x)     # noqa: E731
    state = {"done_w": 0.0, "cur": None}

    def prog(stage, frac, msg=""):
        w = W[stage]
        ev(event="stage", stage=stage, state="progress", percent=round(frac * 100, 1), eta_s=None, message=msg,
           overall_percent=round(min(100.0, (state["done_w"] + w * frac) / total * 100), 1))
        sleep(d)

    def edge(stage, st, msg=""):
        w = W[stage]
        fr = 1.0 if st in ("done", "skipped") else 0.0
        ov = (state["done_w"] + (w if st in ("done", "skipped") else 0)) / total * 100
        ev(event="stage", stage=stage, state=st, percent=fr * 100, eta_s=0 if fr else None, message=msg, overall_percent=round(ov, 1))
    jid = None
    try:
        for n in names:
            state["cur"] = n
            if S["done"].get(n) and all(os.path.exists(P(x)) for x in S["done"][n].get("files", [])):
                edge(n, "skipped", "already done")
                state["done_w"] += W[n]
                continue
            edge(n, "start", n)
            files = []
            if pxqn and edition == "pro" and n in ("skeleton", "hess", "encode") and not jid:
                # one licence job for the whole make: started with the skeleton, resumed (no second charge) by the first stage that runs after a stop
                st = state_of(d)
                rec = S.get("job") or {}
                if rec.get("jid") and rec.get("status") == "started" and c.get("lock_support") and rec.get("lock", "open") != lock_mode:
                    return fail("usage", "This run's licence job was started with lock '%s', not '%s'. Run the same command with --lock %s, or add --restart to start over."
                                % (rec.get("lock", "open"), lock_mode, rec.get("lock", "open")), 2, False, n)
                if rec.get("jid") and rec.get("status") == "started":
                    jid = rec["jid"]
                    st.setdefault("resumes", []).append(jid)
                    save_state(d, st)
                else:
                    st["jobs"] = st.get("jobs", 0) + 1
                    jid = "J%d" % st["jobs"]
                    if st.get("left") is None:
                        st["left"] = int(c.get("left", 5))
                    st["left"] -= 1
                    S["job"] = {"jid": jid, "status": "started", "lock": lock_mode}
                    _wr(stp, S)
                    save_state(d, st)
                    print("licence: job %s started (%s encodes left after this one)" % (jid, st.get("left")), flush=True)
                    if c.get("leak_key"):
                        print("debug: PXQE_KEY=%s server=%s" % (key, srv), flush=True)
            if n == "fetch":
                repo = src.split("@")[0]
                ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
                hdr = {"User-Agent": "pxqe"}
                if os.environ.get("HF_TOKEN"):
                    hdr["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
                try:
                    info = json.loads(urllib.request.urlopen(urllib.request.Request("%s/api/models/%s/revision/main?blobs=true" % (ep, repo), headers=hdr), timeout=30).read())
                    os.makedirs(P("model"), exist_ok=True)
                    sibs = info["siblings"]
                    for i, sb in enumerate(sibs):
                        fn = sb["rfilename"]
                        dest = P("model", fn)
                        part = dest + ".part"
                        have = os.path.getsize(part) if os.path.exists(part) else 0
                        h2 = dict(hdr)
                        if have:
                            h2["Range"] = "bytes=%d-" % have
                        r = urllib.request.urlopen(urllib.request.Request("%s/%s/resolve/main/%s" % (ep, repo, fn), headers=h2), timeout=30)
                        with open(part, "ab" if (have and r.status == 206) else "wb") as f:
                            f.write(r.read())
                        os.replace(part, dest)
                        prog(n, (i + 1) / float(len(sibs)), "%s (%d B)" % (fn, os.path.getsize(dest)))
                    open(P("model", ".fetched"), "w").close()
                    files = ["model/.fetched"]
                except urllib.error.HTTPError as e:
                    if e.code in (401, 403):
                        return fail("gated", "Hugging Face says %s is private or gated. Accept its terms on huggingface.co and run again with your token in HF_TOKEN." % repo, 1, True, n)
                    return fail("http", "Hugging Face answered with an error (HTTP %d) when asked about %s." % (e.code, repo), 1, True, n)
                except (urllib.error.URLError, OSError) as e:
                    return fail("offline", "Cannot reach Hugging Face (%s). Check the internet connection and run again." % getattr(e, "reason", e), 1, True, n)
            elif n == "convert":
                md = P("model") if kind == "hf-repo" else src
                nl = layers_of(md)
                for l in range(nl):
                    print("INFO:hf-to-gguf:blk.%d.attn_q.weight, torch.bfloat16 --> BF16, shape = {64, 64}" % l, flush=True)
                    prog(n, (l + 1) / float(nl), "blk.%d" % l)
                write_gguf(P("model-bf16.gguf"), nl, 30, base_type=30, embed_type=30)
                files = ["model-bf16.gguf"]
            elif n == "q8":
                base = P("model-bf16.gguf") if kind != "gguf" else src
                ts = read_tensors(base)
                nl = len([1 for nm, _ in ts if nm.endswith("attn_q.weight")])
                for i in range(len(ts)):
                    prog(n, (i + 1) / float(len(ts)), "tensor %d of %d" % (i + 1, len(ts)))
                write_gguf(P("model-q8_0.gguf"), nl, 8, base_type=8, embed_type=8)
                files = ["model-q8_0.gguf"]
            elif n == "skeleton":
                q8 = P("model-q8_0.gguf") if os.path.exists(P("model-q8_0.gguf")) else src
                nl = len([1 for nm, _ in read_tensors(q8) if nm.endswith("attn_q.weight")])
                pass
                for i in range(3):
                    prog(n, (i + 1) / 3.0, "writing the skeleton")
                write_gguf(P("encoded.gguf"), nl, TYPE_OF[tier.upper()], base_type=8, embed_type=14, lock_mode=lock_mode if lock_mode != "open" else None)
                files = ["encoded.gguf"]
            elif n == "dump":
                nl = layers_of(P("model")) if kind == "hf-repo" else 4
                os.makedirs(P("act"), exist_ok=True)
                for i in range(5):
                    prog(n, (i + 1) / 5.0, "chunk %d of 5" % (i + 1))
                for l in range(nl):
                    with open(P("act", "blk.%d.attn_q.weight.f16" % l), "wb") as f:
                        f.write(b"\0" * 64)
                files = ["act"]
            elif n == "hess":
                os.makedirs(P("hess"), exist_ok=True)
                acts = sorted(os.listdir(P("act"))) if os.path.isdir(P("act")) else []
                for i, nm in enumerate(acts or ["x"]):
                    with open(P("hess", nm + ".hess"), "wb") as f:
                        f.write(b"\0" * 64)
                    prog(n, (i + 1) / float(len(acts) or 1), "building statistics")
                shutil.rmtree(P("act"), ignore_errors=True)
                files = ["hess"]
            elif n == "encode":
                if c.get("fail") == "oom":
                    return fail("encoder", "CUDA error: out of memory", 1, True, n)
                dst = P("encoded.gguf")
                todo = [nm for nm, ty in read_tensors(dst) if nm.startswith("blk.") and ty not in (8, 30)]
                done = set()
                try:
                    with open(dst + ".progress") as f:
                        done = set(f.read().split())
                except OSError:
                    pass
                n_new = 0
                for i, nm in enumerate(todo):
                    if nm in done:
                        continue
                    crash = c.get("crash_after")
                    if crash and n_new >= int(crash) and not os.path.exists(os.path.join(d, "crashed.flag")):
                        open(os.path.join(d, "crashed.flag"), "w").close()
                        os.kill(os.getpid(), signal.SIGKILL)
                    prog(n, (i + 1) / float(len(todo)), "encoding")
                    with open(dst + ".progress", "a") as f:
                        f.write(nm + "\n")
                    n_new += 1
                with open(dst, "ab") as f:
                    f.write(b"ENCODED")
                files = ["encoded.gguf"]
            elif n == "quantize":
                base = P("model-bf16.gguf") if kind != "gguf" else src
                ts = read_tensors(base)
                nl = len([1 for nm, _ in ts if nm.endswith("attn_q.weight")])
                for i in range(len(ts)):
                    prog(n, (i + 1) / float(len(ts)), "tensor %d of %d" % (i + 1, len(ts)))
                if c.get("composition") and "--pxq-composition-override" not in qargs:
                    return fail("quantize-failed", "The quantizer refused: PXQ composition assertion: target PXQ4 produced 41.3% PXQ-family bytes (floor 50%).", 1, False, n)
                write_gguf(P("result.gguf"), nl, CLASSIC_TYPES[{"pxq4hq": "PXQ4-HQ"}.get(tier, tier.upper())], base_type=8, embed_type=14)
                files = ["result.gguf"]
            elif n == "verify":
                res_file = P("encoded.gguf") if pxqn else P("result.gguf")
                read_tensors(res_file)
                prog(n, 1.0, "ok")
                files = []
            S["done"][n] = {"files": files}
            _wr(stp, S)
            edge(n, "done")
            state["done_w"] += W[n]
        res_file = P("encoded.gguf") if pxqn else P("result.gguf")
        h = hashlib.sha256(open(res_file, "rb").read()).hexdigest()
        size = os.path.getsize(res_file)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.move(res_file, out)
        S.setdefault("results", {})[run_key] = {"out": out, "sha256": h, "size": size}
        _wr(stp, S)
        if pxqn and edition == "pro" and jid:
            st = state_of(d)
            st.setdefault("finished", []).append([jid, "ok"])
            save_state(d, st)
            S["job"] = {"jid": jid, "status": "ok"}
            _wr(stp, S)
            print("licence: job %s ok" % jid)
        ev(event="done", out=out, sha256=h, size=size, tier=tier, **locked)
        print("done: %s sha256 %s" % (out, h))
        if not keep:
            for x in ("model", "model-bf16.gguf", "model-q8_0.gguf", "hess", "act"):
                shutil.rmtree(P(x), ignore_errors=True) if os.path.isdir(P(x)) else (os.path.exists(P(x)) and os.unlink(P(x)))
        return 0
    except _Stop:
        ev(event="error", stage=state["cur"] or "start", code="stopped", message="Stopped. Run the same command again to continue.", resumable=True)
        print("pxqe: stopped. Run the same command again to continue.", file=sys.stderr)
        return 130


def tool_server(d, a):
    """A stand-in llama-server: --version, then /health, /props and /completion (with timings) for the bench."""
    if "--version" in a:
        print("version: 1 (fakeengine)")
        return 0
    import http.server
    import socketserver
    port = int(arg(a, "--port", "8080"))
    model = arg(a, "-m") or arg(a, "--model") or "?"

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._send({"status": "ok"})
            if self.path.startswith("/props"):
                return self._send({"model_path": model, "build_info": "fakeengine", "default_generation_settings": {"n_ctx": 4096}})
            return self._send({"data": [{"id": os.path.basename(model)}]})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            time.sleep(0.05)
            self._send({"content": "fake output " * 12, "tokens_predicted": 24, "tokens_evaluated": 10,
                        "timings": {"predicted_per_second": 42.5, "prompt_per_second": 910.0, "prompt_n": 10}})

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    srv = S(("127.0.0.1", port), H)
    print("main: server is listening on http://127.0.0.1:%d - starting the loop" % port, flush=True)
    srv.serve_forever()
    return 0


def main():
    tool, d, a = sys.argv[1], sys.argv[2], sys.argv[3:]
    fn = {"pxqe": tool_pxqe, "llama-quantize": tool_quantize, "convert_hf_to_gguf": tool_convert, "llama-imatrix": tool_imatrix,
          "llama-server": tool_server}[tool]
    sys.exit(fn(d, a))


if __name__ == "__main__":
    main()
