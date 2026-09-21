#!/usr/bin/env python3
"""Reduction-order probe.

QUESTION. vLLM's native RMSNorm computes the variance as
``x.pow(2).mean(dim=-1, keepdim=True)`` on an fp32 tensor. A hand-fused kernel
has to produce the SAME fp32 bits for that scalar, or the fp16 output differs in
its last bit on a small fraction of elements, and a greedy byte-identity gate is
then a coin toss over 40 layers. torch's accumulation tree is an implementation
detail of TensorIterator; this probe finds out whether it is reproducible by a
one-block-per-row kernel, and if it is, which layout to write.

METHOD. Take torch's own fp32 result on the card. Recompute the row sum on the
host in float32 -- numpy float32 add is IEEE round-to-nearest, the same as a
CUDA fp32 add, so a host simulation of a given accumulation ORDER is bit-
comparable to a device kernel that uses that order. Search a space of layouts
for one that reproduces every row of every shape exactly.

Whatever the search says, part B measures the thing that actually decides the
gate: how many fp16 elements of the FINAL normed output differ when the
variance differs. Zero mismatches means the reduction order does not matter and
any sane fold ships default-on; a nonzero count is the honest reason to keep the
op behind an env flag until the 20-prompt gate has run on the pair.
"""
import itertools
import json
import sys

import numpy as np
import torch

F32 = np.float32


# ---------------------------------------------------------------- orders ----
# Every helper below is a VECTORISED simulation of one accumulation order.
# np.cumsum on a float32 array is a strictly sequential fp32 accumulation
# (pairwise summation is used by np.sum, never by cumsum), so its last element
# is bit-for-bit the same number a device thread would hold after adding the
# same values in the same order.
def _seq(v):
    v = np.asarray(v, dtype=F32)
    if v.size == 0:
        return F32(0.0)
    return np.cumsum(v, dtype=F32)[-1]


def _seq_rows(a):
    """Sequential fp32 sum along the last axis of a 2-D float32 array."""
    return np.cumsum(np.asarray(a, dtype=F32), axis=-1, dtype=F32)[..., -1]


def _tree_pairwise(p):
    """for (off = 1; off < n; off <<= 1) v += shfl_down(v, off) -- torch's
    reduce_kernel warp combine; the classic (0+1)+(2+3) pairing."""
    p = np.array(p, dtype=F32)
    n = p.size
    off = 1
    while off < n:
        q = p.copy()
        q[: n - off] = (p[: n - off] + p[off:]).astype(F32)
        p = q
        off <<= 1
    return p[0]


def _tree_halving(p):
    """for (off = n/2; off > 0; off >>= 1) v += v[t + off] -- the other form."""
    p = np.array(p, dtype=F32)
    n = p.size
    off = n // 2
    while off > 0:
        p[:off] = (p[:off] + p[off : 2 * off]).astype(F32)
        off >>= 1
    return p[0]


TREES = {"pairwise": _tree_pairwise, "halving": _tree_halving, "seq": _seq}


def partials_strided(sq, bw):
    """thread t accumulates sq[t], sq[t+bw], sq[t+2bw], ... in increasing order."""
    return _seq_rows(sq.reshape(-1, bw).T)


def partials_blocked(sq, bw):
    """thread t owns one contiguous chunk of the row."""
    return _seq_rows(sq.reshape(bw, -1))


def partials_vec(sq, bw, vec, merge):
    """Vectorised strided: vec independent accumulators per thread, merged at
    the end (torch's input_vec_size > 1 path)."""
    a = sq.reshape(-1, bw, vec)          # [k, bw, vec]
    acc = _seq_rows(np.moveaxis(a, 0, -1))   # [bw, vec]
    if merge == "seq":
        return _seq_rows(acc)
    return np.array([_tree_pairwise(r) for r in acc], dtype=F32)


def partials_vec_single(sq, bw, vec):
    """Vectorised strided with ONE accumulator, lanes added in index order."""
    a = sq.reshape(-1, bw, vec)              # [k, bw, vec]
    a = np.moveaxis(a, 1, 0).reshape(bw, -1)  # [bw, k*vec], k-major then lane
    return _seq_rows(a)


def partials_2d(sq, bw, bh):
    """Two-level: thread (tx,ty) takes tx + bw*(ty + bh*k). Reduce tx, then ty."""
    a = sq.reshape(-1, bh, bw)               # [k, bh, bw]
    return _seq_rows(np.moveaxis(a, 0, -1))  # [bh, bw]


def candidates(sq):
    """Yield (label, fp32 row sum) for every layout in the search space."""
    h = len(sq)
    widths = [w for w in (32, 64, 128, 256, 512, 1024, 2048)
              if w <= h and h % w == 0]
    for bw in widths:
        ps = partials_strided(sq, bw)
        pb = partials_blocked(sq, bw)
        for tname, tree in TREES.items():
            yield f"strided/bw{bw}/{tname}", tree(ps)
            yield f"blocked/bw{bw}/{tname}", tree(pb)
        for vec in (2, 4, 8):
            if h % (vec * bw):
                continue
            for merge in ("seq", "pairwise"):
                pv = partials_vec(sq, bw, vec, merge)
                for tname, tree in TREES.items():
                    yield f"vec{vec}m{merge}/bw{bw}/{tname}", tree(pv)
            pv1 = partials_vec_single(sq, bw, vec)
            for tname, tree in TREES.items():
                yield f"vec{vec}single/bw{bw}/{tname}", tree(pv1)
        for bh in (2, 4, 8, 16, 32):
            if h % (bw * bh):
                continue
            rows = partials_2d(sq, bw, bh)
            for tx_name, tx_tree in TREES.items():
                col = np.array([tx_tree(r) for r in rows], dtype=F32)
                for ty_name, ty_tree in TREES.items():
                    yield f"2d/bw{bw}x{bh}/{tx_name}+{ty_name}", ty_tree(col)
    # whole-row sequential, and the canonical fixed-chunk fold this package uses
    yield "row/seq", _seq(sq)
    for chunk in (32, 64, 128, 256):
        if h % chunk:
            continue
        n = h // chunk
        lvl = _seq_rows(sq.reshape(n, chunk))
        yield f"canon/chunk{chunk}/seq", _seq(lvl)
        yield f"canon/chunk{chunk}/pairwise", _tree_pairwise(lvl)


def finish(total, h, mode):
    return F32(total * F32(1.0 / h)) if mode == "recip" else F32(total / F32(h))


def bits(a):
    return np.asarray(a, dtype=F32).view(np.uint32)


# ------------------------------------------------------------------ part A ---
def part_a(shapes, seed=0, rows_per_shape=4):
    g = torch.Generator(device="cuda").manual_seed(seed)
    survivors = None
    detail = {}
    for (m, h) in shapes:
        x = torch.randn(m, h, generator=g, device="cuda", dtype=torch.float32)
        ref = x.pow(2).mean(dim=-1).cpu().numpy().astype(F32)
        sq = (x.cpu().numpy().astype(F32) * x.cpu().numpy().astype(F32)).astype(F32)
        ok = set()
        first = True
        for r in range(min(m, rows_per_shape)):
            hits = set()
            for label, total in candidates(sq[r]):
                for mode in ("recip", "div"):
                    if bits(finish(total, h, mode)) == bits(ref[r]):
                        hits.add(f"{label}|{mode}")
            ok = hits if first else (ok & hits)
            first = False
        detail[f"{m}x{h}"] = sorted(ok)
        print(f"  shape {m:5d}x{h:<5d}  layouts reproducing torch bit-exactly: "
              f"{len(ok)}" + (f"  e.g. {sorted(ok)[0]}" if ok else "  NONE"),
              flush=True)
        survivors = ok if survivors is None else (survivors & ok)
    return survivors, detail


# ------------------------------------------------------------------ part B ---
def native_gemma(x_f16, res_f32, w_f32, eps):
    """The exact native GemmaRMSNorm chain, on the card, as the reference."""
    weight = w_f32.float() + 1.0
    xf = x_f16.float() + res_f32.float()
    residual = xf
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * weight
    return out.to(torch.float16), residual


def part_b(shapes, eps=1e-6, seed=1):
    """How much damage does a DIFFERENT variance do to the fp16 output?
    Perturb the variance by exactly 1 fp32 ULP and count changed fp16 elements."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    for (m, h) in shapes:
        x = (torch.randn(m, h, generator=g, device="cuda", dtype=torch.float32)
             * 0.05).half()
        res = torch.randn(m, h, generator=g, device="cuda", dtype=torch.float32) * 0.05
        w = torch.randn(h, generator=g, device="cuda", dtype=torch.float32) * 0.02
        out0, _ = native_gemma(x, res, w, eps)

        weight = w.float() + 1.0
        xf = x.float() + res
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        for ulps in (1, 2, 4):
            v = var.view(torch.int32) + ulps
            v = v.view(torch.float32)
            o = ((xf * torch.rsqrt(v + eps)) * weight).to(torch.float16)
            diff = int((o != out0).sum().item())
            rows.append((m, h, ulps, diff, o.numel(),
                         100.0 * diff / o.numel()))
    print(f"  {'shape':>12} {'var ULP':>8} {'fp16 elems differing':>22} {'%':>8}")
    for m, h, u, d, n, pct in rows:
        print(f"  {m:5d}x{h:<6d} {u:8d} {d:>13d} / {n:<6d} {pct:7.3f}", flush=True)
    return rows


if __name__ == "__main__":
    assert torch.cuda.is_available(), "no cuda"
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0),
          "torch:", torch.__version__, flush=True)
    SHAPES = [(1, 2048), (2, 2048), (4, 2048), (8, 2048),
              (16, 256), (32, 128), (512, 2048), (2048, 2048)]
    print("\nPART A -- does any one-block-per-row layout reproduce torch's "
          "fp32 mean bit for bit?", flush=True)
    surv, detail = part_a(SHAPES)
    print(f"\n  layouts that work at EVERY shape: {sorted(surv) if surv else 'NONE'}",
          flush=True)
    print("\nPART B -- if the variance differs by k fp32 ULP, how much of the "
          "fp16 output changes?", flush=True)
    rows = part_b(SHAPES)
    json.dump({"survivors": sorted(surv or []), "per_shape": detail,
               "ulp_damage": rows},
              open("/work/bench/reduce_probe.json", "w"), indent=1)
    print("\nwrote /work/bench/reduce_probe.json", flush=True)
