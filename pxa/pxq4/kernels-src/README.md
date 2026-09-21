# PXQ4 kernel sources

## v12b

Opt-in env `PXQ4_MMV_MMA=1` routes M>=5 to the tensor-core (wmma) path;
M<=4 is untouched and bit-identical to v11. The shared mmv arenas are
sized from the shape and frozen on the first captured request. The shipped
library (`libpxq4_sm70_v12b.so`, md5 `2abdb4d8c4fb6017b24121d198ad5951`,
2715168 bytes) was built with `build_v12b.sh`. A rebuild reproduces every
host section byte for byte; only `.nv_fatbin` differs run to run because
`-lineinfo` device debug info is not deterministic in this toolchain, so
compare `readelf -SW` sections and exported symbols rather than the md5.

## v16 — the PXQ2/PXQ3 K-chunk-split decode family

`build_pxq_v18.sh` produces `libpxq_<arch>_v18.so`: v14's kernels plus, for PXQ2 and PXQ3 only,
the decode family PXQ4 has had since v3/v4/v6 — `k_pxq23_mmv_part`/`_reduce`, the single-launch
fused form, and the multi-token form, with persistent partials and arrival-counter arenas.
The PXQ4 half is untouched object code and the transcription self-test still gates it.

**Why it was missing, and why that reasoning expired.** The header of `pxq23_kernel.cuh` records
the family as PXQ4-only and gives the reason: the MoE decode path launches
`grid = (panels, S)` with S = tokens×top_k, i.e. 1024–2048 blocks, so "there is no starvation to
fix". That is true of the MoE file it was written against and false of a DENSE model whose
linears are pxq2/pxq3 — there `k_pxq23_mmv` launches `grid = (panels, M)`, which at decode
(M = 1) is 40–136 blocks of 256 threads on an 80-SM card: at most 12.5 % occupancy,
latency-bound. That is the mechanism behind a 3.25 bpw file decoding *slower* than the 4.25 bpw
one, and behind the collapse of per-stream throughput at concurrency 2. The same header names
the remedy — "port `k_pxq4_mmv_part`/`_reduce` here" — and this is that port.

**The claim is bit-exactness, not closeness.** Every arm preserves the monolithic kernel's fold
exactly: per-lane left-associated chunk chain, then ascending k-segment, one final
`__float2half_rn`. The atomic is an arrival counter, never an accumulator. So a difference is a
defect, and the gates demand max-abs-diff 0 rather than a tolerance —
`pxq23_selftest()`'s split differential, and `tests_pxq23/gpu_split_gate.py` on real checkpoint
tensors including from a captured CUDA graph.

**Levers.** `PXQ23_MMV_SPLIT=0` restores the pre-v16 dispatch for every shape and is the A/B
control. `PXQ23_MMV_MT=0` disables only the multi-token arm. `PXQ23_MMV_SPLIT_MAX_BLOCKS`
overrides the occupancy crossover (default `2 × multiProcessorCount`, device-derived).

**Two traps this build hit, recorded so the next one does not.** `libpxq_*_v15.so` was already
taken by a shipped image whose kernels are the v14 set, which is why this is v16 and why
`pxq_version()` returns 16 — test the ABI, not the file name. And on an image whose
`PYTHONPATH` carries the sidecar, `python -c 'import torch; print(...)'` prints an import banner
on **stdout**, so `TORCH=$(...)` in v13/v14 captures a wall of log text and every compile dies
with `nvcc fatal: A single input file is required`; v16 takes the last line.
