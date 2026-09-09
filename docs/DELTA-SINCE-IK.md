# The delta since ik_llama.cpp

pxa is not a general-purpose fork of ik_llama.cpp: it is a Pascal/Volta engine, built for
cards with no DP4A and no tensor cores (Tesla P100, Tesla V100, GTX 1080 Ti), plus a custom
weight codec (PXQ) that rides on top of it. ik_llama.cpp @ `1520eda98056` — now an ancestor in
this project's own history (`docs/README-upstream-ik_llama.md`) — is treated as a parts bin: its
architecture ports, its quant math, its bug fixes are pulled in when they are useful, and left
alone when they are not. Mainline llama.cpp plays a different role entirely — the compatibility
baseline neither fork tries to match feature-for-feature. Everything below is drawn from
`git log --oneline 1520eda98056..HEAD` (502 commits) and the repo's own measured docs; no number
appears here without one of those citations.

## Engine: what runs differently on Pascal/Volta

**The sm_60 fp16-GEMM path, and the trap avoided.** Upstream llama.cpp issue #25593 proposed
excluding cc 600 (P100) from `fast_fp16_available()`, the same way cc 610 (1080 Ti) is already
excluded, on the claim that it costs nothing. This tree built that carve-out and measured it
first: **−47% to −58% prefill** on a D=256 configuration, because `fattn-tile-f32` only supports head sizes
64/128, so the carve-out silently routes D=256 prefill to the single-column `vec_f32` kernel
(`docs/LEVERS.md`, "sm_60 FAST_FP16 carve-out: BUILT, MEASURED, REJECTED", 2026-08-16/17;
commits `a1f67063`, `e4d3c013`). The fix that shipped instead keeps P100 decode on `vec_f32` via
`pxq_use_sm60_vec_f32()` while prefill stays on the fast fp16 path — the two need different
answers, and upstream's proposal picked one answer for both.

**Flash-attention regime routing and the tile-f32 retile.** The engine dispatches a third
mask-skip kernel for the tile-f32 FA path used on sm_61 and forced-F32-precision archs
(`PXA_FA_MASK_SKIP_TILE_F32`, commit `7da043c8`), alongside the sm_60 tile-f16 skip
(`PXA_FA_MASK_SKIP_TILE`) and the sm_70 wmma skip already shipped. A fp32-accumulator variant of
the tile-f16 kernel followed (`PXA_FA_TILE_F32ACC`, commit `a6a969c9`).

**D=256 GQA-packed decode.** A head-packed D=256 vector flash-attention kernel for sm_60
(`PXA_FA_GQA_PACK`, commit `2ef1eb0b`) and a shared-memory staging variant for the packed query
rows (`PXA_FA_GQA_QSMEM`, commit `7c69551e`) target the wide-GQA decode shape directly, alongside
a 4-way ILP V-pass for the same D=256 decode kernel (commit `52f89a8d`).

**Wide/small f16 GEMVs.** A dedicated small-K/large-R f16 decode GEMV was built for the
hyper-connection up-projection (commit `8000f23d`); the general sm_60 GEMV space was explored and
narrowed with two negative results kept on record rather than hidden — a small-N R=1 route
measured **−28%** on deep decode (commit `736b3960`) and a dmmv route was replaced outright
(commit `af480b83`) before the K8-2D S-split decode mmv shipped (commits `06ccc205`, `1e26fc43`).

**MoE fused decode.** The fused up+gate MMVQ dispatch lets both projections walk the activation
once instead of twice (commits `eafcf2a0`, `7ccc4f32`): **+13.7%** dense Volta decode and
**+6.7%** MoE decode against the un-fused driver (`docs/LEVERS.md`, PXA_PXQ_MMVQ row). The K8-2D
S-way K-chunk-split decode mmv that shares the same driver family (commits `06ccc205`, `1e26fc43`)
is bit-exact against the unsplit kernel by construction (`PXA_PXQ4_2D_SPLIT`, `docs/LEVERS.md`),
measured **+35.1%** and **+14.7%** decode / **+57.5%** prefill on two P100 cells.

**Delta-net/GDN fusions and the correctness fix that matters most.** The DeltaNet decode
glue-kernel fusion (`PXA_FUSE_DELTANET=3`) measures **+3.7%** decode on P100 and **+2.8%**
combined with a q8_0 output head on V100 (`bench/README.md`). Separately, and more important:
`PXA_CKPT_HYBRID_ROLLBACK_v1` (commit `a8e8d0f9`) fixes a real contamination bug on hybrid
Gated-DeltaNet architectures (`qwen35moe`/`qwen3next`). The old checkpoint-restore gate compared
the attention side's KV position, which sits near zero on a unified hybrid cache; the per-sequence
recurrent state row (untouched by `seq_rm`) could sit at the *end* of a previous generation, so a
prefix-reuse request would decode fresh tokens against a recurrent state from the future of a
different generation — observed live as byte-identical greedy requests flipping between clean
tool calls and 4000-token loops. The fix rolls the state back whenever it sits ahead of the
re-entry point and matches checkpoints on `pos_max` instead of `pos_min`. The pre-fix mitigation
(`--ctx-checkpoints 0`) and the proper fix are both documented, with the warning that any binary
built before 2026-07-28 lacks it (`docs/LEVERS.md`, `--ctx-checkpoints` row). A related
correctness-only guard blocks `-sm graph` on the same hybrid architectures, which were measured
producing degenerate output (`PXA_ALLOW_GRAPH_SPLIT_HYBRID`, `docs/lab/LEVERS.md`). Bisecting that
on hardware in September moved the root cause off the cross-device all-reduce it was first
attributed to and onto the codec: graph split cuts the attention output and the expert down
projections along `K`, and a PXQ tensor cannot be cut on that axis — one fp16 anchor per row sits
in the 64-row panel header and covers all of `K` — so the slices were read at wrong byte offsets.
`-sm graph` is now refused on a PXQ file whatever the architecture, and `-sm layer` is the
supported split for PXQ files.

**Hybrid context shift.** Companion-shift mirroring for MTP/hybrid architectures was replaced with
`seq_rm` + rebuild (`PXA_HYBRID_CTX_SHIFT_v3`, `docs/LEVERS.md`, `PXA_MTP_DRAFT_RESERVE_CLAMP` row).

**`--kv-unified`.** One shared attention-KV ring across server slots, with admission control
(commit `f15bf696`).

**Elementwise chain fusion.** A generic straight-line elementwise-kernel fuser
(`PXA_EW_FUSE`, commit `844ac0e4`), built after profiling a 5-card decode configuration at ~1,340
kernel launches per token, 38% of them sub-4µs add/scale/mul/sigmoid kernels; the fuser collapses
sole-consumer, same-shape, same-device chains into one interpreter kernel, bit-identical by
construction.

**PLE/qwen4exp support.** A from-scratch port of the Qwen "4exp" hybrid architecture (Pipelined
Local Embedding side path, hyper-connection wide residual, GDN linear layers, gated attention,
MoE) across a dedicated converter and forward-graph build (commits `085b37d3`, `28b17b46`,
`3ef07892`, plus the PLE bring-up chain `edfa591c`, `6eb7badc`, `d8b64ad0`, `17d80af0`,
`b7ebb5b4`). ik_llama.cpp moves fast on new architecture ports of its own; this tree has not
diffed the two implementations, so treat this as "we ship one," not "ik doesn't."

**Scoped cancel / wedge exit server hardening.** Two bare-metal incidents drove hardening work: a
`ggml_abort` backtrace fork could duplicate a live server, holding a dup of the listening socket
while the original half-served (`PXA_BT_NOFORK_v1`, commit `97294087`); the wedge-exit contract
then had to branch on container vs bare-metal runtime, since exiting on bare metal just leaves a
dead port — bare metal now attempts a scoped `ret=-3` unwind that "releases slots honestly,
clients get errors" before a hard exit (`PXA_CONTAINER_AWARE_v1`, commit `e194df13`). A sampler
soft-fail (`PXA_SAMPLE_SOFTFAIL_v1`, `docs/LEVERS.md`) similarly scopes an unsampleable-logits
failure to the one request instead of aborting the whole process.

**Graph reuse.** Inside this engine, keyed CUDA-graph replay was tried and killed on both
architectures with captures verified firing (`PXA_CUDA_GRAPH_V2`, `docs/LEVERS.md`: **−3.9%**
P100, **−2..−4%** V100 — decode here is GPU-busy, not launch-bound, so replay bookkeeping is
pure tax). The win instead landed in the separate vLLM-PXQ4 sidecar tool, where CUDA graphs on
sm_60 move decode **3.9 → 14.9 tok/s** (commit `dd55c8d2`) — a different serving stack, cited here
so the number isn't confused with the llama.cpp-based engine above.

## Engine, v2026.09.02

Shipped and tagged. The ship list and full measurements are in
`RELEASE-NOTES-2026-09-02.md`. One line each:

- **Pipeline scheduler fixes.** The graph allocator now reserves against the real decode graph
  instead of re-planning on every prompt chunk, and the per-batch MoE row-mapping step no longer
  forces a host sync inside the layer loop — together these were the reason a second CUDA stream
  bought almost nothing; fixed, byte-identical, **+27.8% prefill @20,801** at `-c 98304` (does not
  fit at the rig's production `-c 150016`).
- **Device-side MoE row map.** The expert-routing table used to round-trip through the host once
  per batched MoE layer; it now builds on-device, self-checked bit-identical against the old path.
- **GQA-packed attention**, revisited. An earlier version of this same lever (see "D=256
  GQA-packed decode" above) measured as noise at low context fill; re-measured at 86,401 tokens
  it is **+40% decode**, output-identical — the mechanism (one key/value read per query group
  instead of per head) only pays for itself once the re-read volume is large.
- **Host-overhead cuts.** Bounded top-k sampling off raw logits, a struct-of-arrays KV-sequence
  mask, a trimmed KQ-mask host upload, and four more bit-identical prefill micro-fixes together
  cut measured per-token host time at deep fill from 6.0 ms to 1.6 ms.
- **PXQ on CPU.** An AVX2 int8 dot product (panel-tiled, matching PXQ's 64-row panel layout)
  makes CPU-only and partial-offload PXQ inference real: **7.7× prefill** on a 12-thread Xeon
  E5-2699 v3, cross-checked to 0 ULP against the CUDA decode path.
- **Export and requantize.** `llama-pxq-export` decodes a PXQ GGUF back to plain F16/F32
  tensor-by-tensor, and `llama-quantize --allow-requantize` now accepts a PXQ source — the
  lock-in objection in `README.md` ("Lock-in, stated plainly") has an answer now.
- **Product surface.** The server now prints its resolved codec, tensor census, and per-device
  kernel path at startup and in `--verbose` (`pxa: engine | codec=… | file: …`), plus a
  warning when a lab-only `PXA_*`/`PXQ_*` env var is set outside the two documented ones.

Three additional upstream fixes were ported alongside the above, folded into the same release: upstream `78ce50c1` (a get_rows grid-overflow bug at large expert
counts), `c49f7db3` (an MMQ fusion-chain guard for non-MMQ quant types), and `7642ac3e` (a
quantized-cpy kernel launched with 1-thread blocks). They are correctness fixes, not speed
levers, and they are in the same tagged release as the items above.

## Engine, 2026-09-03 build (not yet tagged)

Full detail and the correctness fixes are in `RELEASE-NOTES-2026-09-07.md`; one line each here.

- **Volta flash-attention, vendored from mainline.** The old wmma flash-attention kernel on sm_70
  ran about 2.7x slower than mainline llama.cpp's own `mma.sync m8n8k4` kernel at 20,801 tokens.
  Rather than write a second one, this build vendors mainline's kernel as of its commit
  `9400c8946e`, paired with an fp16 tensor-core GEMM route defaulted on for dense GEMMs on cc 7.0.
  Measured: prefill +36% at 20,801 tokens over the previously shipped tag, at `-ub 2048`.
- **A PXQ2 twin of `mmq.cuh`'s dp4a loop, for sm_61 prefill (N16).** Instead of a
  from-scratch int8 tile, the new sm_61 PXQ2 prefill compute loop mirrors the dp4a geometry
  llama.cpp's own `mmq.cuh` uses for its quantized tiles, carrying a q8_0 numeric contract and a
  PXQ2-specific slab loader on top of that geometry. It beats a fresh upstream ik_llama.cpp
  IQ2_KS build's cold prefill by 11.5% on the published Fusion2-35B PXQ2 tier, chat prefill at
  parity.
- **A scheduler difference from mainline, named.** Profiling why mainline still leads V100
  prefill at 20k tokens despite comparable per-device kernel time found that this fork's scheduler
  host-blocks before every layer split's input copy (`ggml_backend_sched_copy_inputs`, plus a
  second `cudaStreamSynchronize` on the tensor upload), so consecutive splits on a two-card layer
  split never overlap. Mainline's decode path returns without waiting on prompt batches that
  produce no output, which is how it keeps both cards busy at once at the same batch size. A fix
  for this fork (asynchronous host-to-device copies, one scheduler event per graph instead of per
  split, contiguous layer assignment) was built (`fffa96df62`) and measured unsafe in combination,
  so it ships off (`e00cb4fae9`). What did land is the input-view change below, which merges the
  splits themselves; the remaining gap is split shape and pipelining.
- **Two aliasing races fixed.** A delta-net state-aliasing race (rare garbage output, reproducible
  under pipeline parallelism) and an ADD+RMS-norm fusion aliasing race (greedy nondeterminism on
  mixed-format MoE files) are both fixed; both were live in `v2026.09.02`. See
  `docs/KNOWN-ISSUES.md` and `RELEASE-NOTES-2026-09-07.md`.

- **The delta-net fusion clusters are guarded to the graph shape they were written for**
  (`578719d3c0`, `6de128f68a`, `d5c95557d1`, `180524af13`, `211065fe1f`). Upstream has no
  equivalent of these fusions, so this is a divergence this project owns end to end: a 23-node
  delta-net window whose fusion wrote the convolution tail and the recurrent state row through the
  concat node's own data pointer, valid only on the in-place single-sequence shape, plus an
  out-gate fusion that skipped a write on an unverified sole-consumer claim. Both now fire only on
  the proven shape, under `PXA_FUSE_DELTANET_WAR` (default on). The default mask drops 15 → 7 → 3.
  Four node-adjacency fusions of this project's own — ADD+SOFTPLUS+MUL, MUL+EXP+MUL,
  CPY+VIEW+CPY, CONT+SUM_ROWS+TRANSPOSE — gained the write-after-read and same-device checks the
  older fusions already carried, under `PXA_G2_FUSE_WAR`. Two further delta-net per-micro-batch
  experiments were measured unsafe and ship off (`00405147ba`).
- **Per-sequence per-layer-embedding history on the Flash-Next hybrid** (`d719c02be7`,
  `e00ec388b6`, `ea42281fe9`). Neither upstream carries this architecture. The PLE convolution
  window was one history per context, so `-np 2 --kv-unified` aborted; it is now a per-sequence row
  bank keyed by `(seq, pos)`, gathered per token and written back in-graph, riding
  `llama_state_seq_get/set_data`, with a reset when a slot does not continue its own sequence, and
  the post-compute barrier restored for the micro-batch ordering the removed host readback used to
  provide. Multi-slot gauntlet 6/6 at `-np 2`; 112 MiB and 0.5% decode for the second slot.
- **One delta-net input view per graph, automatically, for single-sequence contexts**
  (`9f72492c72`, with `03cd2e860e` as its default-off predecessor). Each delta-net layer built its
  own `conv_seq_map` / `state_row_idx` / `state_mask` view of the same host tensor, and the
  `conv_seq_map` view keeps the full row stride, so every layer copied a 16 MiB span to carry
  400 KB — and 32 inputs fill a scheduler split. Building them once per graph merges the splits
  and is worth +30.5% prefill at 20,801 tokens on 2× V100 at `-ub 512`, taking summed device busy
  from 0.93 to 1.19. `PXA_DN_SHARE_INPUTS` is AUTO: on at `n_seq_max == 1`, off above it, because
  the merged layout makes a residual defect reachable at `-np 2`.
- **Nine Volta-only decode levers and a decode-graph default** (`f78bb8a539`, `028460e1fd`). M=1
  graph replay, delta-net safe-state fusions and sm_70 lever defaults: +3.5% decode at fill 8 and
  at 20,800 tokens of fill, byte-identical output. `PXA_CUDA_GRAPH_DECODE` is defaulted **off** in
  the same pass — it is slower and it aborts on the second request into a warm slot; the commit
  records what fired and what did not.
- **Two Volta kernels that lose, kept env-gated for the record** (`9f8a5d0e43`, `3ab48d21c5`,
  `192da85e31`, `5d81887e35`, `fab434cd16`, `219aba916c`, `2bf59b10f6`). A register-direct PXQ4
  `m8n8k4` prefill GEMM for sm_70 (`PXA_PXQ_GEMM_V70`) that is numerically clean and 40–52%
  slower than dequant + cuBLAS, and a wide-store full-matrix dequant (`PXA_PXQ_DQ_WIDE`) plus its
  side-stream prefetch arena (`PXA_DQC_MB`) that are bit-exact and, respectively, about 2× slower
  and −4.8% to −6.3% on prefill. Both ship default off with the bit-exactness harness that proves
  them correct. The measurement that matters for anyone attempting either: the incumbent
  `k_pxq6_dequant_matrix` already streams at 603–644 GB/s, roughly 70% of the V100's HBM peak.

## Codec: PXQ2/3/4/6/UNIVERSAL

PXQ is a family of CUDA-only slab-layout quant types (`README.md`, "Lock-in, stated plainly") built
around per-row E16 scales — a per-row fp16 anchor plus frozen sub-scales — decoded by a learned,
Lloyd-fit codebook that is compiled in and sha-pinned per tier (`docs/LEVERS.md`, `PXA_PXQ6R`,
`PXA_PXQ_BOOK` rows). PXQ_UNIVERSAL lets a per-tensor tier map mix PXQ1 through PXQ6 by an
importance knapsack for a fixed VRAM budget (`docs/LEVERS.md`, `PXA_PXQ1`/`--pxq-universal` rows).

Against MXFP4 at matched engine and cards: PXQ4 wins dense prefill **+19.2%** and dense decode
**+6.0%** on 2×P100, and MoE prefill **+18.9%** on 2×V100 (`README.md`, codec-only table). It also
wins **6.0% lower perplexity** at the same 4.25 bpw file size (6.9704 → 6.5527, paired, same
bytes). Against ikawrakow's own IQ_K quants, matched size/imatrix/corpus (measured 2026-07-18, while the
PXQ tiers still consumed an imatrix — they ignore one by default since 2026-08-24): PXQ wins speed
everywhere — decode **+3–15%**, prefill **+20–28%** — but IQ_K wins fidelity-per-byte at 3-bit and
4-bit (KLD 0.059 vs 0.076 at 3-bit; 0.028 vs 0.058 at 4-bit); PXQ2 flips that at 2-bit, beating
IQ2_KS on both speed (**+3–28%**) and fidelity (KLD 0.205 vs 0.284) (`bench/HEAD-TO-HEAD.md`).

The cell PXQ loses: Volta dense decode. MXFP4's block layout needs one DP4A scale fixup per 32
values; PXQ4's sub-scale hierarchy costs a second fixup and a second cache line, and at a kernel
already near HBM peak that is a structural tie-or-lose. Measured: **−7.0%**, MXFP4 wins
(`README.md`, codec-only table — "the Volta dense-decode loss is real and understood").

## Measured vs upstream

| comparison | result | source |
|---|---|---|
| Engine-only prefill, fixed weights, P100 | **~1.7×** (+59% `-fa on` serving, +88% `-fa off` batch) | `README.md` |
| Engine-only prefill, fixed weights, V100 | +12–13% | `bench/fair-battle.md` |
| Same-quant decode (IQ3_KS/IQ2_KS, all 3 cards) | **+2.7–3.3%**, V100 bit-identical output | `bench/fair-battle.md` |
| Codec-only, dense prefill, 2×P100 | **+19.2%** vs MXFP4 | `README.md` |
| Codec-only, dense decode, 2×P100 | **+6.0%** vs MXFP4 | `README.md` |
| Codec-only, dense decode, 2×V100 | **−7.0%**, MXFP4 wins | `README.md` |
| 1×P100, 35B MoE, PXQU-16 | ~62 t/s decode, 827–843 t/s prefill | `README.md`, `bench/README.md` |
| 4×P100, ~80B-A3B GDN-hybrid MoE, 3,121-tok fill | 475–490 t/s prefill, 27.5 t/s decode | README, "Scales up" (2026-09-01) |
| 4×P100, same model, 86,401-tok fill | 229.6 t/s prefill, 13.2 t/s decode | README, "Scales up" (2026-09-01) |

## What ik/mainline do better

ik_llama.cpp remains the better choice for CPU and CPU/GPU hybrid-offload serving — its
AVX-512/AVX2 GEMM paths and its `-ot`-style CPU-expert offload are not something this engine
tries to match, and it carries new SOTA CPU quant types this tree does not ship
(`README.md`, "Where this sits"). On Turing and newer it has working tensor-core MMA kernels this
engine has no reason to duplicate — this engine exists specifically for cards *without* that
hardware. Within the cards this engine does target, ik's IQ_K family beats PXQ3/PXQ4 on
fidelity-per-byte, as shown above, and MXFP4 beats PXQ4 on Volta dense decode by 7%
(`bench/HEAD-TO-HEAD.md`, `README.md`). Mainline llama.cpp's advantage is breadth and
compatibility — the widest architecture and backend coverage, and the reference CUDA
implementation everything else, including this tree, is ultimately checked against.

## Sync policy

This project does not track ik_llama.cpp or mainline llama.cpp release-for-release; it
cherry-picks model-graph ports and upstream bug fixes on its own schedule when they're useful on
Pascal/Volta (the `port:` commits throughout `git log 1520eda98056..HEAD` are exactly this, e.g.
commits `76b0c96d`, `c464871d`, `b4323fb2`). The three upstream fixes named just above
(`78ce50c1`, `c49f7db3`, `7642ac3e`) were queued as of the previous revision of this document;
they have since been ported and shipped in `v2026.09.02`.
