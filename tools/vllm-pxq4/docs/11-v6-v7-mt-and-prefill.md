# v6 multi-token mmv and v7 single-op dispatch

Two defects, both invisible to every offline gate we had, because both produced
**bit-exact correct output** and only cost time.

## v6 — the decode kernel re-read the weights once per token

`k_pxq4_mmv_fused` launched with `grid.z = M`, so each token in a batch got its own
block and every block read the full weight panel. Batch 2 therefore moved ~2x the
weight bytes of batch 1, on a kernel that is entirely weight-bandwidth-bound.

Measured effect, TP4, matched serving params:

| streams | AWQ agg (per-stream) | PXQ4 v5 agg (per-stream) |
|---|---|---|
| 1 | 62.18 | 62.00 |
| 2 | 110.76 (55.58) | 74.62 (37.31) |

Equal at M=1 and 33% down at M=2 is the fingerprint: the per-token cost was not the
math, it was re-reading the same weights.

Fix: `k_pxq4_mmv_fused_mt<VECX, MT>` with exact-M template instantiations for 1..8.
One block owns all M tokens of its (chunk, panel); weights are decoded once and folded
into M accumulators. Kill switch `PXQ4_MMV_MT=0`.

Result: conc2 74.62 -> 105.82 aggregate (37.31 -> 52.99 per stream), closing a 33% gap
to 4.5%. conc4 and conc8 crossed into a win over AWQ (37.30 vs 36.47, 74.09 vs 73.14).

## v7 — a Python branch got baked into the compiled prefill graph

`linear.py apply()` chose between the decode mmv path and dequant+GEMM with a plain
`if M <= 8`. That predicate is traced **once per compile range**, and this engine runs
backed dynamic shapes with `evaluate_guards=False`, so the mmv branch was frozen into
the whole `(1, 2048)` prefill range. Prefill ran the decode kernel: one full weight
read per token, 4.37 ms/token, linear in prompt length.

| prefill, 2790-token prompts, n=8 | tok/s |
|---|---|
| AWQ | 3159.8 |
| PXQ4 before | 229.1 |
| PXQ4 after | 3404.9 |

A 13.8x loss became a 7.8% win. Note how well-hidden this was: prefill throughput was
never the metric under test, the output was bit-exact throughout, and the standard
deviation of the broken measurement was 0.36 tok/s - it was *precisely* wrong.

Fix: move the mmv-vs-GEMM policy into a C++ custom op (`pxq4::linear_out`), which
torch.compile treats as opaque and therefore cannot constant-fold, plus a capture-safe
fp16 dequant arena.

Do NOT try to fix this with `compile_ranges_endpoints`. The engine's own
`sm70_gemma_long_prefill_fused_add_rms_norm` carries the same traced-branch pattern
(it requires >=256 tokens) and crashes on custom ranges.

## What still loses

- **conc16: -7 to -9%.** Not the kernel. At batch >=3 decode runs eager because only
  sizes (1,2) are cudagraph-captured; linears are under 10 ms of a ~119 ms step.
- **Extended capture sizes are not the fix yet.** Passing
  `cudagraph_capture_sizes=[1,2,4,8,16]` pins decode at a batch-independent ~121 ms/step
  (8.25 tok/s) at 100% GPU utilisation, reproduced on a clean uninstrumented boot.
  Root cause unknown.
- **First-token-latency spikes.** 3 of 12 short prompts deterministically pay ~0.9 s
  before the first token, on PXQ4 arms only, across every kernel version. Tokenizer
  ruled out (0.2-0.4 ms offline); survives the v7 prefill fix. Unresolved.

## Measurement notes

SM clocks droop 1530 -> 1380-1425 MHz under sustained decode and recover in gappy
phases - autoboost, not throttling (no throttle reasons, 210-240 W, <=67 C). This is
the source of the single-stream bimodality and it affects both arms equally.

On the in-process step harness, PXQ4 medians ahead (66.78 vs 65.48) but *means* behind
(63.66 vs 65.44): our distribution is bimodal (sd 5.66) where AWQ's is tight (sd 0.75).
That comparison should not be quoted as a win.

Cards 4-7 were checked for a hardware handicap and are clean: max clocks, no throttle
reasons, NVLink pattern identical to cards 0-3.

`VLLM_SM70_QUANT_BACKEND` routes only AWQ/GPTQ/compressed-tensors layers and is inert
for PXQ4, so marlin-vs-turbomind was never a live lever for this format.

## Gates

hostsim 17/17 legacy + split + MT parity, bit-exact including the fp32 `part[]`
accumulators; GPU parity across all TP4 shapes x M in 1..8; 400-launch stress, 0 fails.
