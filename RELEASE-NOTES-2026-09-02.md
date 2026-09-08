# Release notes — 2026-09-02

A 24-hour measurement pass across both rigs this project runs day to day: the 4× Tesla P100 rig
(llama.cpp-based engine, hybrid MoE, PXQ_UNIVERSAL 4-bit) and the 2× Tesla V100 rig (vLLM-based
sm_70 serving line, 27B dense-hybrid, PXQ4). Everything below traces to `bench/fair-battle.md`,
`docs/PXA-SM70-SERVING.md`, or the raw arms this document itself describes, protocol per
`bench/fair/protocol.md` unless stated otherwise (`llama-server`/vLLM `/completion`, temp 0, n=7
median, 1 warmup discarded, unique prompt per repeat).

Two sections: what's already live in the launcher, and what's queued in the next engine build
(**not yet in a tagged release** — everything under "Next engine build" needs the candidate
binary described there, not today's release).

## Launcher-level wins already live (2026-09-01)

- **4× P100 rig: `-ub 2048 -wgt 8`** (was `-ub 1024`). `-wgt 8` zeroes a 248k-vocab logits
  reservation, which is what lets the larger micro-batch fit at all — `-ub 2048` alone does not
  boot. Measured deep-fill (86,401-token prompt, 3 reps): prefill 223.4 → 229.6 t/s (+2.8%),
  decode flat. Boots clean at `-c 150016`.
- **2× V100 rig: `--max-num-batched-tokens 4096`** (was 2048). Six interleaved boots: prefill
  918–920 → 940–945 t/s @3k (+2.5%), 878–880 → 908–912 t/s @20k (+3.3%), decode and aggregate
  flat, KV pool unchanged. 8192 was tried too — no further prefill gain and a 15% smaller KV pool
  from the larger reserved scratch — so 4096 is the ceiling here, not a stop along the way.
- **2× V100 rig: `--gpu-memory-utilization 0.92`** (was 0.85). A capacity lever, not a speed one —
  roughly 44% more KV pool at the same accuracy, verified against the linear pool-vs-GMU
  relationship measured earlier. (The next-engine-build config below runs at 0.88, one notch back
  down — see the NCCL section.)

## Next engine build (not yet in a tagged release)

### Pipelined prefill

Prefill on the P100 rig runs mostly single-stream even though the scheduler supports a second
CUDA copy stream. Root cause was two bugs, not a missing feature: the graph allocator re-planned
its buffers on every prompt chunk instead of once per request (a stale reserve/real graph size
mismatch caused by state-reset nodes only emitted at position 0), and the batched MoE
row-mapping step forced a host sync inside the per-layer loop on every batched call. Fixed:
re-plans per 20k-token prefill went from 12 to 2, and the MoE row map moved fully on-device (next
item). With the fix, on top of the full ship set, prefill at `-c 98304` (`PXA_PIPELINE_PP=1
GGML_SCHED_MAX_COPIES=2`):

| arm | prefill @3,121 | prefill @20,801 |
|---|---|---|
| single-stream (ship set, PP off) | 487.06 | 410.61 |
| pipelined, two streams | 515.36 | **501.79 (+22.2% vs single-stream)** |

Byte-identical across 15 comparisons including sequence switches. **Known limit:** it does not fit
at the P100 rig's production `-c 150016` — compute buffer allocation fails on card 1 there, and
the no-PP fallback does not recover either. It ships as opt-in at reduced context (`-c 98304`)
today, not the default. Also fixed on this branch, unconditionally, regardless of
whether pipelining ships: a per-slot attention state window on the hybrid architecture that was
never reset between requests, so a second request could inherit the first request's window.

### GQA-packed attention

`PXA_FA_GQA_PACK=4` reads each attention key/value once per query group instead of once per
head. An earlier version of this same lever was measured as noise at low context fill and left
off. Re-measured at 86,401 tokens, where the re-read volume the lever removes is actually large:

| arm | decode @8 | decode @86,401 |
|---|---|---|
| control | 26.08 | 12.52 |
| `PXA_FA_GQA_PACK=4` | 26.22 | **17.54 (+40%)** |

Output-identical. Stacked with the host-overhead cuts below, the full shipped set measures
**+49% decode at 86,401 tokens, +2.3% at low fill**, over a bracketing control pair, output
identical in every arm.

### Host-overhead cuts

Four more bit-identical fixes, instrumented with a per-token host/GPU timing split
(`PXA_HOST_TIMING`): a struct-of-arrays KV-sequence mask, bounded top-k sampling off raw logits
instead of a full sort, a trimmed KQ-mask host-to-device upload, and a router aliasing barrier
for multi-row MoE dispatch (a correctness guard, zero measured cost, kept on). Together with
GQA-packed attention: measured host time per token at deep fill, **6.0 ms → 1.6 ms**; GPU submit
time per token, 73.6 ms → 48.7 ms. The full stack (these four plus GQA-pack plus five smaller
bit-identical prefill micro-fixes — narrowed get_rows, a fast-div copy kernel, a flattened concat
path, a register-cached norm, and lazy scheduler resets) measured **+2.6–4.6% prefill, +2.3%
low-fill decode, +49% deep-fill decode**, output-identical at temp 0 in every arm.

### Device-side MoE row map

The table mapping tokens to experts used to get built on the host and copied down once per
batched MoE layer, forcing a device-to-host sync (`prepare_row_mappigs`) on the critical path.
It now builds entirely on-device (hist/scan/fill in one pass); self-check mode compares the two
paths and found them bit-identical across 64 cases. This is the second half of the pipelined
prefill fix above — without it, the sync alone would have capped the pipelining win regardless of
the scheduler fix.

### PXQ on CPU

An AVX2 int8 dot product (16-entry pshufb lookup, `maddubs`/`madd`, per-32-block fp32
anchor/scale) makes CPU-only and partial-offload PXQ inference a real option instead of a
technically-correct fallback. It's panel-tiled to match PXQ's native 64-row panel layout rather
than going through a generic per-row vector-dot trait, which the panel layout can't address.
Measured on a 12-thread Xeon E5-2699 v3, PXQ4, 128-token prompt: **14.3 → 110.6 t/s prefill
(7.7×)**. Cross-checked against the CUDA decode path: 0 ULP difference across 196 tensors.
Default on (`PXA_PXQ_CPU_DOT`).

### Export and requantize

`llama-pxq-export in.gguf out.gguf [--type f16|f32] [--cpu]` streams a PXQ GGUF tensor-by-tensor,
decoding PXQ tensors in whole-panel chunks (GPU or CPU) and copying everything else byte for
byte. `llama-quantize --allow-requantize` now accepts a PXQ source and routes it through the same
decode path before re-quantizing — one step from PXQ straight to, say, Q4_K_M (an
`--i-know-this-is-double-lossy` flag is required for a PXQ or Q8_0 source, since that's quantizing
twice). Verified: 196 tensors CPU-vs-CUDA decode at 0 ULP, 114/114 non-PXQ tensors byte-identical
through the export, chunked output identical to whole-tensor output, and greedy generation
identical after a round trip.

One quantizer footgun found and not yet fixed: `--token-embedding-type pxq*` produces a model
that generates nothing, because the token-embedding lookup is a plain row-gather that never goes
through a PXQ decode path. The row-gather guard that should block this keys on tensor name and
size, not on quant type, so it doesn't catch it. Consumer-based guard queued; until then, don't
quantize the token embedding tensor to a PXQ type.

### Correctness fixes

- **get_rows grid overflow** — ported from ik_llama.cpp upstream (`78ce50c1`): a grid-size
  overflow in the get_rows kernel at large expert counts (relevant at 512-expert scale).
- **MMQ fusion-chain guard** — ported from upstream (`c49f7db3`): guards the MMQ fusion chain
  for non-MMQ quant types that shouldn't enter it.
- **Quantized cpy launch fix** — ported from upstream (`7642ac3e`): quantized cpy kernels were
  launched with 1-thread blocks; fixed.
- **PLE window reset** — found here, not upstream: the per-slot convolution window on the hybrid
  architecture's local-embedding path was never reset between requests, so a second request on a
  reused slot could inherit the first request's window. Fixed regardless of whether pipelined
  prefill ships, since it's a correctness bug on its own.
- **Token-embedding PXQ guard** — found, not yet fixed; see "Export and requantize" above.

## Rejected levers (so nobody re-tries them)

- **`PXA_FA_KEYS_PER_SPLIT` ladder** (1024/512/256) — negative on the GQA-packed D=256 decode
  kernel at 86k-token fill: more splits means more combine work once K/V reads are already shared
  across four heads. Stays off.
- **`PXA_GEMV_RPB=-1`** — −8% decode at low fill on this kernel family; the GPU-submit time got
  worse, not better. Off.
- **CUDA graph capture on P100** — hard-gated below Ampere in the backend; forcing it measured
  −2.8%. Dead, not revisited.
- **Fused TP2 all-reduce + RMSNorm fusion (V100)** — engagement confirmed in the boot log, but
  measured prefill −1.5…−1.9%, under the noise floor at best. Not shipped.
- **QPN v13 kernel** — a from-scratch PXQ4 GEMV rewrite, measured slower than the v12b tensor-core
  path at batch 16 on this rig's actual tensor shapes, and found on inspection to inherit the
  same split-K partials-arena capture-order hazard v12 had before the v12b fix (see
  `docs/PXA-SM70-SERVING.md`). Not shipped.
- **PXQ4 output head (4-bit lm_head)** — fails the 70-prompt same-top-token gate at 97.1%
  (bar is ≥98%); see `docs/PXA-SM70-SERVING.md`. Kept the fp16 head.
- **`PXA_X_CACHE`** (reused f32→f16 activation conversion across QKV matmuls) — bit-identical but
  measured null on this model's tensor shapes. Left in the tree, default off, not chased further.
- **CPU governor, `ondemand` → `performance`** — kept (it's harmless and already persisted), but
  the deep-fill A/B measured +0.9%, under this rig's ~3% noise floor. Not claimed as a win.

## Known limits

- **Pipelined prefill does not fit at the P100 rig's production context** — compute buffer
  allocation fails on card 1 at `-c 150016`, and the no-PP fallback doesn't recover either. It's
  real and byte-identical at reduced context (`-c 98304`); it isn't shipped at full context yet.
- **Volta dense decode still loses to MXFP4** — a structural DP4A-scale-fixup cost, not a tuning
  gap; see the codec-only table in `README.md` and `bench/fair-battle.md`.
- **A 4-bit output head fails the quality gate** — 97.1% same-top-token agreement against a
  98% bar. The fp16 head ships; a quantized head is still on the table at a different bit width
  or scheme.
