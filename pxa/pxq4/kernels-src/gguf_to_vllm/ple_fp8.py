"""ple_fp8.py -- streaming q8_0 -> FP8 (e4m3fn, ONE global scale) requantizer for the Flash-Next
per-layer token embedding table (Blocker 2 of the qwen4exp conversion, plan of record #1090).

    python3 -m gguf_to_vllm.ple_fp8 --gguf <Flash-Next GGUF> --out <dir> [--threads 16]
                                    [--limit-shards N] [--sample-rows 4000] [--budget-rms-pct 5.0]

WHAT IT PRODUCES (what the fork's loader consumes, ple_layer.py:790-840, 2026-09-06 read):
  <out>/ple-fp8-000xx-of-000yy.safetensors      512 checkpoint shards named
        model.language_model.ple.ngram_embedding.shard_<k>.weight   F8_E4M3 [shard_rows, 160]
        with shard_rows = ceil(vocab / split_ngram_parts) = 625003 (vocab 320001536, 512 parts),
        plus ONE  model.language_model.ple.ngram_embedding.weight_scale  F32 [1]
  <out>/ple.safetensors.index.json               weight_map for the assembly step
  <out>/PLE-FP8-REPORT.json / .md                the pre-registered gate, per head, never pooled

WHY ONE GLOBAL SCALE: Qwen4ExpPLEFp8EmbeddingMethod creates a PerTensorScaleParameter (one
scale for the whole table); Qwen4ExpPinnedHostEmbedding refuses anything but FP8 storage. The
source is q8_0 (int8 + per-32-block fp16 scale). pxq23's step-0 kill test (#1096) measured the
16 head tables' absmax within 1.36x of each other, so one e4m3 scale keeps every head on
328-448 of 448 levels; this script measures the TRUE global absmax in a first pass (a sampled
maximum would risk saturation) and reports the per-head outcome after the second pass.

THE GATE (pre-registered here, before any number exists):
  per head table, on --sample-rows random rows (seeded), compare the q8_0 dequant against the
  FP8 value READ BACK FROM THE WRITTEN FILE times the written scale:
    * saturation count == 0                 (nothing clipped at 448)
    * rms relative error ||x-x^||/||x|| <= --budget-rms-pct (default 5 %: e4m3 carries 3 mantissa
      bits, so uniform rounding alone is ~3.6 % rms relative; a head far above that has lost
      levels to the global scale)
    * flushed-to-zero fraction < 1 % of the non-zero source values
  A failing head fails the run (exit 1) and is named; pooling across heads is exactly what would
  hide one flattened head.

CPU only, no GPU, no docker: numpy for the q8_0 decode, torch (CPU) for the e4m3 rounding
(round-to-nearest-even with explicit clamp to +-448 -- torch does NOT saturate on its own, it
produces NaN above the max). Peak RSS ~1 GB (one 625003-row shard at fp32 at a time).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import gguf_raw as G
from . import safetensors_io as ST


def disable_thp() -> None:
    """PR_SET_THP_DISABLE for this process. MEASURED 2026-09-06 on the box: THP enabled=always with
    defrag=madvise and a fragmented page cache (compact_stall 414k, thp_fault_fallback 7.3M) made
    every 100 MB numpy allocation cost 18-60 s of SYSTEM time in direct compaction (numpy madvises
    MADV_HUGEPAGE on large arrays; torch's allocator hits enabled=always). With THP disabled for
    the process the same allocation takes 0.1 s. No effect on any other process."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.prctl(41, 1, 0, 0, 0)          # PR_SET_THP_DISABLE = 41
        if rc != 0:
            print(f"warning: prctl(PR_SET_THP_DISABLE) rc={rc} errno={ctypes.get_errno()}", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"warning: could not disable THP: {e}", file=sys.stderr)
    try:
        np._core.multiarray._set_madvise_hugepage(False)   # belt and braces for numpy's own madvise
    except Exception:
        pass

TENSOR = "per_layer_token_embd.weight"
HF_PREFIX = "model.language_model.ple.ngram_embedding"
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0 ** -9           # smallest non-zero e4m3fn magnitude
SPLIT_PARTS_DEFAULT = 512                 # fork default split_ngram_parts
QK = 32                                   # q8_0 block
BLK_BYTES = 34                            # 2 B fp16 d + 32 int8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_BUF: dict = {}


def dequant_rows(mv: memoryview, row_bytes: int, r0: int, r1: int, K: int) -> np.ndarray:
    """Rows [r0, r1) of a q8_0 tensor as fp32 [r1-r0, K] (exact: y = d * q).

    Working buffers are allocated ONCE per shape and reused (np.copyto / out=): on this box a fresh
    100 MB allocation page-faults through THP compaction, so a per-call astype() would dominate.
    The returned array is a VIEW of the reusable buffer -- copy it if it must outlive the next call."""
    R = r1 - r0
    nb = K // QK
    raw = np.frombuffer(mv[r0 * row_bytes:r1 * row_bytes], dtype=np.uint8).reshape(R, nb, BLK_BYTES)
    b = _BUF.get((R, K))
    if b is None:
        b = {"d": np.empty((R, nb), np.float32), "q": np.empty((R, nb, QK), np.float32),
             "x": np.empty((R, nb, QK), np.float32), "d16": np.empty((R, nb, 2), np.uint8)}
        _BUF.clear(); _BUF[(R, K)] = b
    np.copyto(b["d16"], raw[:, :, :2])
    np.copyto(b["d"], b["d16"].view("<f2").reshape(R, nb), casting="unsafe")
    np.copyto(b["q"], raw[:, :, 2:].view(np.int8), casting="unsafe")
    np.multiply(b["q"], b["d"][:, :, None], out=b["x"])
    return b["x"].reshape(R, K)


_TBUF: dict = {}


def to_e4m3_bytes(x: np.ndarray, scale: float) -> np.ndarray:
    """e4m3fn bytes of x/scale, round-to-nearest-even, clamped to +-448 (torch does not saturate)."""
    import torch
    t = torch.from_numpy(x)
    tb = _TBUF.get(tuple(x.shape))
    if tb is None:
        tb = {"s": torch.empty(x.shape, dtype=torch.float32), "f8": torch.empty(x.shape, dtype=torch.float8_e4m3fn)}
        _TBUF.clear(); _TBUF[tuple(x.shape)] = tb
    torch.mul(t, 1.0 / scale, out=tb["s"])
    tb["s"].clamp_(-E4M3_MAX, E4M3_MAX)
    tb["f8"].copy_(tb["s"])
    return tb["f8"].view(torch.uint8).numpy()


def from_e4m3_bytes(b: np.ndarray, scale: float) -> np.ndarray:
    import torch
    return (torch.from_numpy(np.ascontiguousarray(b)).view(torch.float8_e4m3fn).float() * scale).numpy()


def read_rows_from_shard(path: str, name: str, rows: np.ndarray, K: int) -> np.ndarray:
    """Gather rows (u8 e4m3 bytes) of one tensor straight from a written safetensors file."""
    hdr = ST.read_header(path)
    meta = hdr[name]
    beg, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(0)
        n = int.from_bytes(f.read(8), "little")
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = np.empty((len(rows), K), dtype=np.uint8)
    for i, r in enumerate(rows):
        off = base + beg + int(r) * K
        out[i] = mm[off:off + K]
    del mm
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--split-parts", type=int, default=SPLIT_PARTS_DEFAULT)
    ap.add_argument("--file-shards", type=int, default=8, help="embedding shards per .safetensors file")
    ap.add_argument("--limit-shards", type=int, default=0, help="smoke test: only the first N shards (both passes)")
    ap.add_argument("--sample-rows", type=int, default=4000)
    ap.add_argument("--budget-rms-pct", type=float, default=5.0)
    ap.add_argument("--budget-ftz-pct", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args(argv)

    disable_thp()
    import torch
    torch.set_num_threads(max(1, a.threads))
    os.makedirs(a.out, exist_ok=True)

    g = G.GGUFFile(a.gguf)
    kv = g.kv
    arch = kv.get("general.architecture")
    ti = g.tensors[TENSOR]
    if ti.type != "q8_0":
        raise SystemExit(f"{TENSOR} is {ti.type}, this requantizer expects q8_0")
    K, V = ti.ne0, ti.ne1
    row_bytes = ti.nbytes // V
    assert row_bytes == (K // QK) * BLK_BYTES, (row_bytes, K)
    offsets = list(kv.get(f"{arch}.ple.head_offsets") or [])
    sizes = list(kv.get(f"{arch}.ple.head_vocab_sizes") or [])
    if not offsets or not sizes or len(offsets) != len(sizes):
        raise SystemExit("GGUF carries no ple.head_offsets / ple.head_vocab_sizes")
    used = offsets[-1] + sizes[-1]
    shard_rows = (V + a.split_parts - 1) // a.split_parts
    n_shards = (V + shard_rows - 1) // shard_rows
    if a.limit_shards:
        n_shards = min(n_shards, a.limit_shards)
    log(f"{TENSOR}: q8_0 ne=({K},{V}) = {ti.nbytes/1e9:.2f} GB; {len(sizes)} head tables, "
        f"{used} used rows + {V-used} padding rows; {a.split_parts} parts x {shard_rows} rows -> "
        f"{n_shards} shards of {shard_rows*K/1e6:.1f} MB fp8 each")
    mv = g.raw(TENSOR)

    # ---------------- pass 1: TRUE global absmax + per-head absmax/rms (streamed) ------------
    t0 = time.time()
    absmax = 0.0
    head_stats = [{"absmax": 0.0, "sumsq": 0.0, "n": 0} for _ in sizes]
    def head_of(row: int) -> int:
        # rows are laid out head after head at `offsets`; padding rows (>= used) belong to none
        lo, hi = 0, len(offsets) - 1
        if row >= used:
            return -1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= row:
                lo = mid
            else:
                hi = mid - 1
        return lo
    for s in range(n_shards):
        r0, r1 = s * shard_rows, min(V, (s + 1) * shard_rows)
        x = dequant_rows(mv, row_bytes, r0, r1, K)
        am = float(np.abs(x).max())
        absmax = max(absmax, am)
        # per-head accumulation within this shard (heads are contiguous row ranges)
        r = r0
        while r < r1:
            h = head_of(r)
            if h < 0:
                break
            hend = min(r1, offsets[h] + sizes[h])
            seg = x[r - r0:hend - r0]
            hs = head_stats[h]
            hs["absmax"] = max(hs["absmax"], float(np.abs(seg).max()))
            flat = seg.reshape(-1)
            hs["sumsq"] += float(np.dot(flat.astype(np.float64, copy=False), flat) if flat.size < 1 else np.einsum("i,i->", flat, flat, dtype=np.float64))
            hs["n"] += seg.size
            r = hend
        if s % 32 == 0 or s == n_shards - 1:
            log(f"PROGRESS pass1 shard {s+1}/{n_shards} absmax so far {absmax:.6f} ({time.time()-t0:.0f}s)")
    scale = absmax / E4M3_MAX
    log(f"pass 1 done in {time.time()-t0:.0f}s: global absmax {absmax:.6f} -> scale {scale:.6e}")
    for h, hs in enumerate(head_stats):
        rms = (hs["sumsq"] / hs["n"]) ** 0.5 if hs["n"] else 0.0
        levels = hs["absmax"] / scale if scale else 0.0
        log(f"  head {h:2d}: rows {sizes[h]:>9d} absmax {hs['absmax']:.6f} rms {rms:.6f} -> top level {levels:.0f}/448")

    # ---------------- pass 2: requantize + write, streaming, shard by shard ------------------
    t1 = time.time()
    files: list[str] = []
    weight_map: dict[str, str] = {}
    n_files = (n_shards + a.file_shards - 1) // a.file_shards + 1   # +1: the scale file
    meta = {"format": "pt", "pxa_ple_fp8": "q8_0->e4m3fn global scale", "source": os.path.basename(a.gguf),
            "scale": repr(scale), "absmax": repr(absmax), "split_ngram_parts": str(a.split_parts),
            "vocab_rows": str(V), "embedding_dim": str(K)}
    written = [0]
    def stream_shard(s: int):
        def w(f):
            r0, r1 = s * shard_rows, min(V, (s + 1) * shard_rows)
            x = dequant_rows(mv, row_bytes, r0, r1, K)
            f.write(to_e4m3_bytes(x, scale).tobytes())
            written[0] += 1
            if written[0] % 32 == 0 or written[0] == n_shards:
                log(f"PROGRESS pass2 shard {written[0]}/{n_shards} written ({time.time()-t1:.0f}s, "
                    f"{written[0]*shard_rows*K/1e9/(time.time()-t1+1e-9):.2f} GB/s)")
        return w
    fi = 0
    for s0 in range(0, n_shards, a.file_shards):
        fi += 1
        fname = f"ple-fp8-{fi:05d}-of-{n_files:05d}.safetensors"
        tensors = []
        for s in range(s0, min(n_shards, s0 + a.file_shards)):
            r0, r1 = s * shard_rows, min(V, (s + 1) * shard_rows)
            name = f"{HF_PREFIX}.shard_{s}.weight"
            tensors.append(ST.Tensor(name, "F8_E4M3", (r1 - r0, K), stream_shard(s), streaming=True))
            weight_map[name] = fname
        path = os.path.join(a.out, fname)
        ST.write_file(path, tensors, meta)
        files.append(path)
    # the scale, F32 [1], in its own small file
    fname = f"ple-fp8-{n_files:05d}-of-{n_files:05d}.safetensors"
    sname = f"{HF_PREFIX}.weight_scale"
    ST.write_file(os.path.join(a.out, fname), [ST.Tensor.from_numpy(sname, np.array([scale], dtype=np.float32))], meta)
    weight_map[sname] = fname
    total = sum(os.path.getsize(p) for p in files) + os.path.getsize(os.path.join(a.out, fname))
    with open(os.path.join(a.out, "ple.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total, **meta}, "weight_map": weight_map}, f, indent=1)
    log(f"pass 2 done in {time.time()-t1:.0f}s: {n_shards} shards, {total/1e9:.2f} GB in {len(files)+1} files")

    # ---------------- gate: read back, per head, pre-registered budgets ----------------------
    t2 = time.time()
    rng = np.random.default_rng(a.seed)
    report = {"gguf": a.gguf, "tensor": TENSOR, "scale": scale, "absmax": absmax, "shards": n_shards,
              "shard_rows": shard_rows, "budget_rms_pct": a.budget_rms_pct, "budget_ftz_pct": a.budget_ftz_pct,
              "sample_rows_per_head": a.sample_rows, "heads": [], "pass": True}
    limit_rows = n_shards * shard_rows
    for h, (off, sz) in enumerate(zip(offsets, sizes)):
        hi = min(off + sz, limit_rows)
        if off >= limit_rows:
            report["heads"].append({"head": h, "skipped": "outside --limit-shards"}); continue
        rows = np.sort(rng.choice(np.arange(off, hi), size=min(a.sample_rows, hi - off), replace=False))
        src = np.concatenate([dequant_rows(mv, row_bytes, int(r), int(r) + 1, K).copy() for r in rows])
        # gather from the written files by shard
        got = np.empty_like(src)
        for s in np.unique(rows // shard_rows):
            sel = rows // shard_rows == s
            name = f"{HF_PREFIX}.shard_{int(s)}.weight"
            b = read_rows_from_shard(os.path.join(a.out, weight_map[name]), name, rows[sel] - s * shard_rows, K)
            got[sel] = from_e4m3_bytes(b, scale)
        err = got - src
        nz = src != 0
        rms_rel = float(np.sqrt((err.astype(np.float64) ** 2).sum() / max(1e-30, (src.astype(np.float64) ** 2).sum()))) * 100
        sat = int((np.abs(src) / scale > E4M3_MAX * (1 + 1e-6)).sum())
        ftz = float(((got == 0) & nz).sum() / max(1, nz.sum())) * 100
        maxabs = float(np.abs(err).max())
        ok = sat == 0 and rms_rel <= a.budget_rms_pct and ftz < a.budget_ftz_pct
        report["heads"].append({"head": h, "rows_sampled": int(len(rows)), "rms_rel_err_pct": rms_rel,
                                "max_abs_err": maxabs, "saturated": sat, "flushed_to_zero_pct": ftz, "pass": ok})
        report["pass"] &= ok
        log(f"GATE head {h:2d}: rms_rel {rms_rel:.3f}% max_abs {maxabs:.2e} saturated {sat} ftz {ftz:.4f}% -> {'PASS' if ok else 'FAIL'}")
    report["gate_seconds"] = time.time() - t2
    with open(os.path.join(a.out, "PLE-FP8-REPORT.json"), "w") as f:
        json.dump(report, f, indent=1)
    md = ["# Flash-Next PLE q8_0 -> FP8 (e4m3fn, one global scale)", "",
          f"source `{a.gguf}` tensor `{TENSOR}` q8_0 ({K} x {V}); scale {scale:.6e} (global absmax {absmax:.6f}); "
          f"{n_shards} shards of {shard_rows} rows as `{HF_PREFIX}.shard_<k>.weight` F8_E4M3 + `{HF_PREFIX}.weight_scale` F32[1].",
          "", f"Pre-registered gate: saturation 0, rms relative error <= {a.budget_rms_pct} %, flushed-to-zero < {a.budget_ftz_pct} % -- per head, never pooled.", "",
          "| head | rows sampled | rms rel err % | max abs err | saturated | ftz % | verdict |", "|---|---|---|---|---|---|---|"]
    for hs in report["heads"]:
        if "skipped" in hs:
            md.append(f"| {hs['head']} | - | - | - | - | - | skipped ({hs['skipped']}) |"); continue
        md.append(f"| {hs['head']} | {hs['rows_sampled']} | {hs['rms_rel_err_pct']:.3f} | {hs['max_abs_err']:.2e} | {hs['saturated']} | {hs['flushed_to_zero_pct']:.4f} | {'PASS' if hs['pass'] else 'FAIL'} |")
    md += ["", f"**Overall: {'PASS' if report['pass'] else 'FAIL'}**", "",
           "Assembly: config.json must carry `ple_embedding_dtype: \"float8_e4m3fn\"` and `ple_offload_embedding: true` "
           "(the fork auto-offloads only on sm_70; P100s are sm_60), and the checkpoint's weight_map must include these shards "
           "beside the tensors convert.py emits for the rest of the model (Blocker 1 wiring)."]
    with open(os.path.join(a.out, "PLE-FP8-REPORT.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    log(f"GATE {'PASS' if report['pass'] else 'FAIL'}; report {a.out}/PLE-FP8-REPORT.md")
    print(f"FLASHNEXT PLE-FP8 {'DONE' if report['pass'] else 'FAILED-GATE'} {a.out}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
