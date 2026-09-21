#!/usr/bin/env python3
"""pxq-salience-gate.py — salience-aware quality gate for quantized GGUFs.

WHY THIS EXISTS (DS4-FP8-REQUANT-2026-08-02.md): global weight error (`wrel`, an
unweighted Frobenius norm) is BLIND to peak clipping. PXQ2-as-built had LOWER global
error than the coherent community IQ2XXS arm (0.3296 vs 0.3535) while being fully
degenerate, because 18.8% of its squared error sat in the top 1% of weights (vs 4.9%)
— its book/SUB16 composition structurally capped reconstruction at 0.6970 x the row
anchor. A tier can PASS a global-wrel gate and still be unusable.

This gate therefore measures, expert-tensor by expert-tensor, against the source file:
  * global wrel                       (reported, NOT gated)
  * relative error ON the top-1% and top-0.1% largest-|w| source weights
  * share of total squared error concentrated in those bands
  * the REALISED reconstruction ceiling: median over rows of max|recon| / row absmax

Verdict:
  FAIL (exit 1)  ceiling < 0.80  OR  top-1% rel err > 0.30
  WARN (exit 0)  ceiling < 0.95  OR  top-1% rel err > 0.20   (printed prominently)
  PASS (exit 0)  otherwise
(0.80, not a tighter bound, because the REALISED ceiling of a healthy 4-level book sits
below its theoretical one — per-block MSE deliberately under-scales the block max to fit
the other 15 elements; an absmax-1.0 LM4 realises ~0.86. Only structural clipping — a
book/SUB16 composition that cannot reach the anchor — drives it below 0.80. These match
the pxq_ceiling_check() thresholds in the quantizer.)

Thresholds validated against the three known cases (DS4-Flash, 512 rows x 9 tensors):
  PXQ4-core            ceiling 0.9878, top-1% 0.0343  -> PASS   (coherent, ppl 5.4097)
  community IQ2XXS     ceiling ~1.00,  top-1% 0.1631  -> PASS   (coherent, ppl 5.9171)
  PXQ2-as-built        ceiling 0.6970, top-1% 0.3085  -> FAIL   (degenerate)
A gate that does not fail the known-bad case is not a gate.

PXQ book provenance: a PXQ2/PXQ3 file carries the book it was built with in its
pxa.pxq2.book / pxa.pxq3.book KV. The runtime decoders default to the COMPILED book, so
this tool exports PXA_PXQ2_BOOK/PXA_PXQ3_BOOK from the file's own KV when invoking the
dequant helper — old-book files are judged as they actually decode under their book.

usage:
  pxq-salience-gate.py --src <source.gguf> --quant <quantized.gguf> --build <build-dir>
                       [--nrows 512] [--layers 0,21,42] [--report-only]
"""
import argparse, os, struct, subprocess, sys, tempfile
import numpy as np

GT_SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
GT_FM = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}

def gguf_dir(path):
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    def s_():
        n = u64(); return f.read(n).decode("utf-8", "replace")
    def val(t):
        if t == 8: return s_()
        if t == 9:
            et = u32(); n = u64(); return [val(et) for _ in range(n)]
        return struct.unpack("<" + GT_FM[t], f.read(GT_SZ[t]))[0]
    magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"not a GGUF: {path}")
    u32(); nt = u64(); nkv = u64(); kv = {}
    for _ in range(nkv):
        k = s_(); t = u32(); kv[k] = val(t)
    tens = {}
    for _ in range(nt):
        nm = s_(); nd = u32(); dims = [u64() for _ in range(nd)]
        tt = u32(); off = u64(); tens[nm] = (dims, tt, off)
    e = f.tell(); f.close()
    al = kv.get("general.alignment", 32)
    return tens, (e + al - 1)//al*al, kv

# panel-format PXQ types: slab bytes per 32-elem column block (+128 B header / 64-row panel)
PXQ_SLAB = {252:1088, 253:1152, 254:576, 255:832}
PXQ_BOOK_ENV = {254:("pxa.pxq2.book","PXA_PXQ2_BOOK"), 255:("pxa.pxq3.book","PXA_PXQ3_BOOK")}
# flat row-major types: bytes per row of k elems
def row_bytes(tt, k):
    return {39: k//32*17,    # MXFP4
            16: k//256*66,   # IQ2_XXS
            10: k//256*84,   # Q2_K
             8: k//32*34,    # Q8_0
             1: k*2,         # F16
            }[tt]

def slice_bytes(tt, nrows, k):
    if tt in PXQ_SLAB:
        return (nrows//64)*(128 + (k//32)*PXQ_SLAB[tt])
    return nrows*row_bytes(tt, k)

class Deq:
    def __init__(self, build, tmpdir):
        self.helper = os.path.join(build, "bin", "pxq-deq")
        self.tmp = tmpdir
        if not os.path.exists(self.helper):
            # compile on demand against the build's libggml
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pxq-deq.cpp")
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.helper = os.path.join(tmpdir, "pxq-deq")
            cmd = ["g++", "-O2", "-std=c++17", src, "-o", self.helper,
                   "-I", os.path.join(root, "ggml", "include"),
                   "-I", os.path.join(root, "ggml", "src"),
                   "-L", os.path.join(build, "ggml", "src"), "-lggml",
                   "-Wl,-rpath," + os.path.join(build, "ggml", "src")]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit("pxq-deq compile failed:\n" + r.stderr)
    def __call__(self, path, dirinfo, name, nrows, env_extra=None):
        tens, ds, _ = dirinfo
        if name not in tens:
            raise SystemExit(f"tensor {name} not in {path}")
        dims, tt, off = tens[name]
        ne0 = dims[0]
        raw = os.path.join(self.tmp, "_s.raw"); f32 = os.path.join(self.tmp, "_s.f32")
        with open(path, "rb") as f:
            f.seek(ds + off)
            blob = f.read(slice_bytes(tt, nrows, ne0))
        open(raw, "wb").write(blob)
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        r = subprocess.run([self.helper, str(tt), str(nrows), str(ne0), raw, f32],
                           capture_output=True, text=True, env=env)
        if r.returncode:
            raise SystemExit(f"pxq-deq failed on {name} (type {tt}):\n" + r.stderr)
        return np.fromfile(f32, dtype=np.float32).reshape(nrows, ne0), tt

def book_env_for(dirinfo, tt):
    """Decode a PXQ file under ITS OWN baked book, not the binary's compiled default."""
    _, _, kv = dirinfo
    if tt not in PXQ_BOOK_ENV:
        return None
    key, env = PXQ_BOOK_ENV[tt]
    book = kv.get(key)
    if not book:
        return None
    return {env: ",".join("%.10g" % v for v in book)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--build", required=True, help="engine build dir (for libggml / pxq-deq)")
    ap.add_argument("--nrows", type=int, default=512)
    ap.add_argument("--layers", default="0,21,42")
    ap.add_argument("--report-only", action="store_true", help="never exit non-zero")
    args = ap.parse_args()

    src_d = gguf_dir(args.src)
    qnt_d = gguf_dir(args.quant)
    layers = [int(x) for x in args.layers.split(",")]
    names = [f"blk.{l}.{g}.weight" for l in layers
             for g in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")]
    names = [n for n in names if n in src_d[0] and n in qnt_d[0]]
    if not names:
        raise SystemExit("no expert tensors found in both files — pass --layers explicitly")

    with tempfile.TemporaryDirectory() as tmp:
        deq = Deq(args.build, tmp)
        wrel_n, wrel_d = 0.0, 0.0
        top1_n, top1_d, top01_n, top01_d = 0.0, 0.0, 0.0, 0.0
        share1, share01, ceils = [], [], []
        print(f"pxq-salience-gate: {len(names)} expert tensors x {args.nrows} rows")
        for nm in names:
            A, _ = deq(args.src, src_d, nm, args.nrows)
            tt = qnt_d[0][nm][1]
            B, _ = deq(args.quant, qnt_d, nm, args.nrows, env_extra=book_env_for(qnt_d, tt))
            aa = np.abs(A)
            anc = aa.max(axis=1)
            ceil = float(np.median(np.abs(B).max(axis=1)/np.maximum(anc, 1e-30)))
            ceils.append(ceil)
            E2 = (A.astype(np.float64) - B.astype(np.float64))**2
            wrel_n += E2.sum(); wrel_d += (A.astype(np.float64)**2).sum()
            thr1  = np.percentile(aa, 99.0)
            thr01 = np.percentile(aa, 99.9)
            m1, m01 = aa >= thr1, aa >= thr01
            top1_n  += E2[m1].sum();  top1_d  += (A.astype(np.float64)[m1]**2).sum()
            top01_n += E2[m01].sum(); top01_d += (A.astype(np.float64)[m01]**2).sum()
            share1.append(float(E2[m1].sum()/E2.sum()))
            share01.append(float(E2[m01].sum()/E2.sum()))
            print(f"  {nm:34s} type {tt:3d}  ceiling {ceil:.4f}  "
                  f"top1%err-share {100*share1[-1]:5.1f}%")
        wrel  = (wrel_n/wrel_d) ** 0.5
        top1  = (top1_n/top1_d) ** 0.5
        top01 = (top01_n/top01_d) ** 0.5
        ceil_med = float(np.median(ceils))
        print()
        print(f"  global wrel                 : {wrel:.4f}   (reported, not gated — it is blind to clipping)")
        print(f"  rel err on top-1%  weights  : {top1:.4f}   (share of sq err: {100*np.mean(share1):.1f}%)")
        print(f"  rel err on top-0.1% weights : {top01:.4f}   (share of sq err: {100*np.mean(share01):.1f}%)")
        print(f"  realised ceiling (median)   : {ceil_med:.4f}   (max|recon| / row absmax)")

        fail = ceil_med < 0.80 or top1 > 0.30
        warn = ceil_med < 0.95 or top1 > 0.20
        if fail:
            print("\nVERDICT: FAIL — structural peak clipping / broken tail. "
                  "Do NOT ship this file; it can pass a global-wrel gate and still be degenerate "
                  "(that is exactly how PXQ2-as-built shipped once).")
            sys.exit(0 if args.report_only else 1)
        elif warn:
            print("\nVERDICT: WARN — elevated tail error. Usable tiers have shipped in this band "
                  "(PXQ3 ceiling 0.896), but gate the release on ppl/coherence, not on this alone.")
        else:
            print("\nVERDICT: PASS")
        sys.exit(0)

if __name__ == "__main__":
    main()
