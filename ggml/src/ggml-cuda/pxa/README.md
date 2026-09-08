# `ggml-cuda/pxa/` — the PXA kernel suite

Everything in this directory is PXA-specific: it does not exist upstream, and removing the
directory (plus the `pxa/` globs in `ggml/src/CMakeLists.txt` and the `pxa/...` includes listed
at the bottom of this file) would leave a tree whose CUDA backend is the fork's ik_llama-derived
baseline. Files that merely *call* into this directory — `ggml-cuda.cu`, `mmq.cu`, `convert.cu`,
`cpy.cu`, `concat.cu`, `fattn.cu`, `fattn-tile-f16.cu`, `fattn-tile-f32.cu`,
`mmvq-templates.cuh` — stay where they are, because they are upstream files carrying PXA edits
rather than PXA files.

Not all of it is PXA original work, and the directory does not claim otherwise. `volta-mma/` is
vendored verbatim from mainline llama.cpp under MIT and carries its provenance in every file's
header, commit and all — read those headers before assuming a symbol there is ours.
`delta-net.cu` is ik_llama.cpp's DeltaNet linear-attention kernel for Qwen3-Next, also MIT, and
now carries the same kind of header; the PXA part of that file is the two `_ex` entry points the
fusion below drives, not the recurrence. Both live under `pxa/` because they are fork-specific —
nothing upstream of this fork contains them in this form — not because PXA claims authorship of
them.

**Reading order.** `pxa-enhance.cuh` first (it decides every default below), then `pxq4.cuh` →
`pxq6.cuh` → `pxq23.cuh` for the codec, then whichever kernel family you care about.

**Env-var convention.** Every lever is an environment variable. An explicit variable always
wins. When it is unset the value comes from the config *level* (`pxa-enhance.cuh`), and where a
level default is not simply on/off it is stated per file below. `PXA_REFERENCE=1` forces every
lever off — that is the bit-exact audit baseline, and it is the arm to compare against when a
lever is suspected.

---

## Config and dispatch

### `pxa-enhance.cuh`
The master config. Three tiers, resolved exactly once at startup:

| level | selected by | what it is |
|---|---|---|
| 0 REFERENCE | `PXA_REFERENCE=1` (wins over everything) | every PXA lever off — pure reference kernel and dispatch paths. The bit-exact audit baseline, and the arm to compare against when a lever is suspected. |
| 1 DEFAULT | `PXA_ENHANCE=0` | the pre-2026-09-03 shipped lever set. The rollback tier. |
| **2 ENHANCE** | **nothing set** (the default), or `PXA_ENHANCE=1` | DEFAULT plus every measured per-card lever: `PXA_PXQ_INT8_PREFILL=1` on sm_61 only, `PXA_ROUTER_FUSE=1` on sm_70 only, `PXA_SPEC_RELAXED=1` on spec lanes, `PXA_FA_GQA_PACK=4` and `PXA_MOE_DEVICE_MAP=1` on a multi-card sm_60 topology, and the nine arch-independent house levers below. |

**ENHANCE became the default on 2026-09-03.** Every number in the campaign chart was measured
with `PXA_ENHANCE=1` exported, so shipping level 1 meant the binary an operator ran was not the
binary that was measured — the levers self-armed on paper and never fired. `PXA_ENHANCE=1` is
still accepted and is now a no-op.

The nine **house levers** — `PXA_KQ_MASK_PAD1`, `PXA_KV_SEQ_SOA`, `PXA_TOPK_RAW`,
`PXA_TOPK_MOE_MULTIROW`, `PXA_GETROWS_NARROW`, `PXA_CPY_FASTDIV`, `PXA_CONCAT_FLAT`,
`PXA_NORM_REGCACHE`, `PXA_SCHED_RESET_LAZY` — are host-side or pure index-math changes (a
red-black-tree find replaced by an int compare, an integer division by a multiply-shift, a
wholesale memset by an exact dirty-set walk, a launch geometry flattened). They compute the same
values the paths they replace computed, so they follow the level, not the architecture:
`pxa_house_lever_default()` here, `pxa_cuda_house_lever()` in `../common.cuh`, `pxa_house_lever()`
in `ggml-backend.cpp`. Until 2026-09-03 they keyed on `ggml_pxa_cuda_is_volta_only()`, which left
them off on the 4× P100 seat they were measured on.

The file also holds the *auto* resolvers that combine the model profile with the detected device
topology — `pxa_pxq_mmvq_auto_default()` is the important one — and prints the whole resolution
as a `PXA_AUTO:` ledger at startup so a run's lever state is recoverable from its log.
`PXA_ENHANCE_DBG=1` makes the resolver verbose.

The level itself is **not** resolved here: `ggml_pxa_config_level()` (`ggml/src/ggml.c`, declared
in `ggml.h`) is the single definition, so the six non-CUDA translation units that need it —
`common/sampling.cpp`, `examples/server/server.cpp`, `examples/server/server-context.cpp`,
`src/llama.cpp` — read the same answer instead of hand-mirroring the logic.

`pxa-enhance.cuh` also owns the **per-card serve-flag defaults**, exposed to front ends as
`ggml_backend_cuda_pxa_suggest_batch()` (`ggml-cuda.h`): on a card set the campaign measured, the
server fills `-b`/`-ub` the user left unset — 2× sm_70 → `8192`/`2048`, 2× sm_60 → `8192`/`256`,
1× sm_61 → `2048`/`768`. An explicit `-b`/`-ub` always wins, and every other shape keeps the
adaptive-VRAM ladder.

---

## The PXQ codec

### `pxq4.cuh`
The PXQ4 format definition and the shared slab/panel machinery every other PXQ file reuses:
`PXQ4_QK=32`, `PXQ4_TYPE_SIZE=17`, 64-row panels (`PXQ4_BM`), 64-token tiles (`PXQ4_BN`),
1088-byte slabs, plus the tile descriptor (`pxq4_tile_info`), the row-map struct, the on-device
2D tile builder `k_pxq_tiles_2d`, and the activation gather. The layout is a slab of 64 bytes of
scale SoA followed by 64 rows × 16 bytes of nibble codes, slabs K-major inside a 64-row panel,
panels row-major, experts outermost. Numerics for the original tier are exactly MXFP4, so
MXFP4 ↔ PXQ4 is a lossless bit permutation. **Env:** `PXA_PXQ4=0` disables the fused kernels and
falls back to dequant→cuBLAS, which is bit-identical fp16 to the MXFP4 path; **default on**.

### `pxq6.cuh`
The largest file here and the heart of the codec: the E16-row two-level scale format
(per-row fp16 anchor in a 128-byte panel header × a 4-bit sub-scale per 16 elements for the core
tier, per 8 for HQ) and the policy-templated kernel family `k_pxq6_*` that PXQ4, PXQ4HQ, PXQ2,
PXQ3 and PXQ6R all instantiate. The fastpath levers are each separately gated and each documented
in the file with whether it is bit-exact: `PXA_PXQ6_KSPLIT` (K1, bit-exact decode K-split),
`PXA_PXQ6_KSPLIT_GEN=S` (K1b, deterministic but *not* bit-exact — different chain count),
`PXA_PXQ6_PAIRLUT` / `PXA_PXQ6_VECX` / `PXA_PXQ6_SHFL` (K2 family, all bit-exact),
`PXA_PXQ6_GUFUSE` / `PXA_PXQ6_SCATFUSE` (K3 prefill fusions, bit-exact),
`PXA_PXQ6_RAGTAIL` (K4, bit-exact ragged-tile FMA skip), `PXA_PXQ6_PIPE` (K5, bit-exact register
prefetch) and `PXA_PXQ6_WMMA=1|2|3` (K6, cc 7.0 only, **not** bit-exact). `PXA_PXQ6=0` is the
master off switch for the PXQ6/PXQ6HQ fused kernels. `PXA_PXQ6_FORCE_PREFILL=1` bypasses the
sm_60/70 arch gate for correctness A/Bs on other cards and must never be set in production.
Numeric tables come from `ggml/include/ggml-pxq6-tables.h`; `PXA_PXQ6_BOOK`, `PXA_PXQ6_SUB` and
`PXA_PXQ6_SUB_HQ` override them and must match the model file's `pxa.pxq6.*` provenance keys.
Ship defaults for the bit-exact levers are resolved through `pxa_gate_default()`, so REFERENCE
turns all of them off in one move.

### `pxq23.cuh`
PXQ2 (2-bit, LM4 book) and PXQ3 (3-bit, LM8, bit-plane packed) as one file, because the two tiers
differ only in book size and pair decode — every kernel they run on lives in `pxq6.cuh`. It is
`#include`d *from inside* `pxq6.cuh` at a load-bearing point: after the generalized
`pxq6_dot32` / `k_pxq6_dequant_matrix` templates and before the `PXQ6_PICK_FMT` pickers. Do not
move that include. PXQ2 packs 4 codes per byte (576-byte slab); PXQ3 uses three little-endian
words per code row, two low planes plus one high plane, decoded branch-free (832-byte slab).
**Env:** `PXA_PXQ2_BOOK` / `PXA_PXQ3_BOOK` override the frozen books (fp16-snapped, must match the
file's provenance keys); `PXA_PXQ3_PAIRLUT` enables PXQ3's own 64-entry pair LUT, **default off**.
The sub-scale LUT is shared with PXQ6, so `PXA_PXQ6_SUB` moves all three code widths at once. K6
WMMA is deliberately not extended to these tiers; the host driver masks it for `fmt >= P2`.

### `pxq6i8.cuh`
N13: the int8/DP4A MMQ-style **prefill** tile for the PXQ formats, written for sm_61 first. It is
an int8 twin of `k_pxq6_gemm_grouped` inside the same policy-templated family rather than a
repack to q8_0 or a trait inside `mmq.cuh` — the file's header explains why both alternatives
were rejected (VRAM for a q8_0 shadow; q8_0's per-32 scale cannot carry PXQ's per-16/per-8
sub-scales). The numeric contract is the frozen q3-s8 snap: book values snap to
`rint(book·127/absmax)`, and the `/127` and book absmax fold into the per-16 fp32 effective
scale, leaving the stored fp16 anchor untouched. Activations are quantized q8_1-style. **Not
bit-exact** versus the fused fp16 path, so a temp-0 sha is not a valid gate for it. **Env:**
`PXA_PXQ_INT8_PREFILL=0` **default off**, `=1` sm_61 only (this is the ship gate, armed
automatically at ENHANCE, measured +182% prefill on the 1080 Ti), `=2` all archs (test only —
sm_60 has no IDP4A and falls to emulation). `PXA_PXQ_I8_RAGTAIL` ports the bit-exact ragged-tile
skip to this tile, **default off**. See the negatives section for `PXA_PXQ_I8_DBUF` and
`PXA_PXQ_I8_BN`.

### `pxq4-mmq.cuh`
M1: a dp4a MMQ prefill tile for the 4-bit PXQ tiers on a **dense 2D** MUL_MAT, i.e. the shape
that otherwise dequantizes the whole weight matrix into an fp16 pool and runs cuBLAS HGEMM
against it. It keeps the format's native per-16 (PXQ4) / per-8 (PXQ4HQ) scales rather than
joint-snapping to a q8_0 shape, which is better numerics and close to free on Volta because the
extra work issues on the FP32 pipe while the dp4a chain sets the rate. Same frozen q3-s8 numeric
contract as `pxq6i8.cuh`; **not** bit-exact versus dequant+cuBLAS. **Env:** `PXA_PXQ4_MMQ=0`
**default off** (byte-identical dispatch to a build without it), `=1` on, `=2` forces cc 6.0
(test only — sm_60 has no IDP4A). `PXA_PXQ4_MMQ_NW` = 4 (default) or 8 warps;
`PXA_PXQ4_MMQ_MINNY` = 32 token floor. **This lever is a measured loss on sm_70 — see below.**

### `pxq4-v70.cuh`
V70: a register-direct PXQ4 prefill GEMM on Volta's native `mma.sync.aligned.m8n8k4` tile. It
decodes PXQ4 nibbles straight into the A register fragment, so the A operand never touches shared
memory and is never materialised as fp16 — no dequant kernel, no fp16 scratch, no smem staging,
no 60 KB carveout. The decoded weight value is bit-for-bit what `k_pxq6_dequant_matrix` writes
into the cuBLAS scratch (test #1 memcmps it), so the entire numeric difference against the
incumbent is tensor-core accumulation order. Two traps are called out in the header and are worth
repeating: the fragments must be addressed through `get_i`/`get_j` (never a hand-rolled
`threadIdx.x`, because `GGML_CUDA_MMA_NO_VOLTA_PERM` redefines them), and `blockDim.x` must be 32.
**Env:** `PXA_PXQ_GEMM_V70` (aliases `PXA_VOLTA_PXQ_GEMM`, `PXA_DEQUANT_ONCE`) **default off**;
`PXA_PXQ_GEMM_V70_CFG` tile config, default 0; `PXA_PXQ_GEMM_V70_MIN_NY` default 128 (inert at the
production `-ub 2048`); `PXA_PXQ_GEMM_V70_SKIP` takes an `R:K,R:K,...` per-shape denylist.
**This lever is a measured loss on sm_70 — see below.**

### `pxq-mmvq.cuh`
PXQ registered into the stock MMVQ int8/DP4A GEMV path — the kernel every stock 4-bit type rides
for **decode**. The motivation is structural, not incremental: profiling the bespoke
`k_pxq6_mmv` family reports Block Limit = Registers with 19.8–38.6% achieved occupancy, so it is
register-limited and latency-bound; MMVQ's one-row-per-block shape avoids exactly that. Scope is
deliberately narrow — `GGML_TYPE_PXQ4` and `GGML_TYPE_PXQ4HQ` only, both PX16-book tiers covered
by the snap validation — and the gate turns *itself* off if any `PXA_PXQ6_BOOK`/`_SUB`/`_SUB_HQ`
table override is set, because this translation unit carries its own frozen copies and takes no
runtime uploads. Not bit-exact versus the fused fp16 decode kernels. **Env:** `PXA_PXQ_MMVQ` =
`0` off, `1` sm_70+ (what it was built for), `2` any card with real DP4A (cc ≥ 6.1, test).
**Default is auto**: at DEFAULT or ENHANCE, on a PXQ4/PXQ4HQ-bearing model, it resolves to 1 when
an sm_70+ device is present and to 2 on an all-sm_61 fleet; it stays off on an sm_60-only fleet
(P100 has no DP4A) and at REFERENCE. The startup `PXA_AUTO:` line prints which arm fired and why.

### `pxq-moemap.cuh`
Device-side construction of the MoE expert row mapping and tile list. The host version of this
(`prepare_row_mappigs`) copies `ids` D2H and does a `cudaStreamSynchronize` inside graph compute,
once per layer per ubatch — which is precisely what stops the host from enqueuing ubatch *k+1*
while the devices still run ubatch *k*. The five kernels here (`k_pxq_moemap_hist`, `_scan_b`,
`_scan_e`, `_fill`, `_tiles`) produce byte-identical buffers with no readback and no synchronize.
Placement is decided by block prefix plus rank-within-block, never by atomics, so the buffers do
not depend on scheduling and the GEMM tiles see the same rows in the same order. What the host
loses is the two scalars it used for launch geometry; both are replaced by their worst case
(`n_rows`, `n_rows/PXQ4_BN + n_as`) and the unused blocks exit on a zeroed tile or a sentinel
mapping entry, which is why the tile and mapping buffers are cleared before the fill.

### `template-instances/mmvq-instance-pxq4.cu`, `mmvq-instance-pxq4hq.cu`
The two MMVQ template instantiations for `GGML_TYPE_PXQ4` and `GGML_TYPE_PXQ4HQ`. They exist as
their own translation units for the same reason as every other `mmvq-instance-*.cu`: to keep
compile time and register pressure per TU bounded. They are globbed only into the CUDA build, not
the HIP build — matching exactly what `ggml-cuda/template-instances/mmvq-instance*.cu` did before
the fence.

---

## Prefill and decode levers

### `pxa-dqcache.cuh`
K8-C: a weight-stationary prefetch arena for the dequant→cuBLAS prefill path. It is deliberately
**not** a cache — the per-device fp16 working set (~15–17 GB) dwarfs any affordable arena and the
access order is a clean cyclic scan, so an LRU over it would hit ~0%. It is a per-device fp16 ring
plus a two-node lookahead walker that runs the next MUL_MAT's `src0` dequant on a low-priority
side stream. The header documents the four safety arguments that make that legal — weights are
immutable for process lifetime and only `USAGE_WEIGHTS` non-split buffers are ever prefetched;
the key is the exact `(src0_dd_i, row_diff, ne00, type)` tuple compared in full, never hashed;
every write is on the dq stream and every read on the compute stream with an event between them
and a full-drain barrier on ring wrap; and `acquire()` declines while the consumer stream is
capturing, because a graph that replays a baked arena pointer reads a recycled region. **Env:**
`PXA_DQC_MB` **default 0 = off** (it changes allocation counts and stream structure, so it ships
armed only by env); `PXA_DQC_MIN_NY` default 64. **Measured loss — see below.**

### `pxa-smalln.cu` / `pxa-smalln.cuh`
`PXA_SPEC_SMALLN`, the B4 speculative-verify engine: a multi-column dequant-FMA GEMV for the
dense quantized backbone at `ne11 = 2..8` on pre-Volta cards. Speculative/MTP verify runs the
whole dense backbone at `ne11 = k+1`, and that shape otherwise rides int8 MMVQ whose emulated
dp4a compute scales with columns — measured `verify(4)/verify(1) = 1.646×` on P100, which eats
most of the acceptance win. A cuBLAS redirect was tried and measured *worse* (2.28×; per-call
dequant and setup dominate at tiny N). This kernel loads the weight once and does R column FMAs
instead, accumulating in fp32. MXFP4 scale decode is byte-identical to `dequantize_block_mxfp4`.
**Env:** `PXA_SPEC_SMALLN=1`, **default off**; G3-class versus the MMVQ path it replaces
(different reduction order), run-to-run deterministic.

### `pxa_expert_shard.cuh`
PXA-SHARD M2: shards the top-8-of-256 expert compute of one fused-MoE layer across a matched,
P2P-connected device group. Because top-8 routing writes one disjoint `dst` row per
(token, slot), each device owns a disjoint set of output rows and **there is no reduction** — each
device scatters straight into the home-device `dst` over P2P (UVA), with one event per peer. The
per-member accumulation is byte-identical to the per-token reference (it reuses
`grouped_moe_verify.cuh`'s q8_1 activations, MXFP4 dot and SILU epilogue); only addressing
differs, which is why the exact-equality shadow diff `PXA_MOE_GROUPED_VERIFY=1` must report zero
mismatches. **Env:** `PXA_EXPERT_SHARD` **default off** — with it unset nothing routes an expert
tensor into the shard buffer type, the op-path branch is guarded on
`pxa_buft_is_expert_shard(...)` which is then always false, and the binary is bit-identical to a
flag-off build. `PXA_SHARD_TIMING` prints per-wall timings. Peer copies and pool allocations are
not capture-safe, so inside a CUDA graph capture the path returns false and the caller runs the
per-token route.

### `pxa-deltanet-fuse.cuh`
`PXA_FUSE_DELTANET`: the R2 DeltaNet decode glue-kernel fusion for qwen35moe/qwen3next at bs=1,
where a chain of small kernels each pays a 3–10 µs launch-latency floor on Pascal and Volta. It
collapses SILU + two L2_NORMs into one kernel, CONT + CONCAT into one (with the delta-net core
writing its new ssm state directly into the destination, removing a 4.15 MB read + 4.15 MB write
per layer per token), and FUSED_RMS_NORM + FUSED_MUL_UNARY into one. **Env:** a **bitmask,
default 53** (bits 0|2|4|5); `=0` restores the fully eager path. bit0 (1) is the qk-norm/state-writeback
cluster and fires only on the exact in-place state alias; bit1 (2) is the out-gate fusion, **out of
the default since 2026-09-04** — see the negatives table below — and it *verifies* its sole-consumer
claim against the graph rather than asserting it, as well as declining on a shifted overlap between
its product and either input; all these guards are themselves under `PXA_FUSE_DELTANET_WAR`,
default on. bit2 (4) is the state-row gather+reset-mask
fusion, off by default because it was measured never to match the live decode graph. bit3 (8) is
the legacy row absorb, kept only as the historical reproducer. bit4 (16) is the safe-carry cluster
that writes into the graph's own allocated storage instead of the cache row. bit5 (32) is the
provable row absorb, admitted only when a reader-set scan proves no node between the cluster and
the SET_ROWS it absorbs touches the state storage; `PXA_DN_ROWSCAN=1` prints that scan and names
any blocking reader. The default pair is worth +3.7% P100 decode and is bit-exact against the
eager kernels; a pattern mismatch falls through to eager per node.

### `delta-net.cu` / `delta-net.cuh`
ik_llama.cpp's DeltaNet linear-attention kernel for Qwen3-Next (head dim 128, vendored at commit
`3c58ae37`, MIT — the file header carries the full note) with its state in
`[S_v, S_v·H_v, 1, n_seqs]` column-major layout, plus the two PXA entry points the fusion above
needs: `ggml_cuda_op_delta_net_ex` takes a `state_dst_override` so the new ssm state is written
straight into the recurrent cache row instead of through a CONCAT, and
`ggml_cuda_op_delta_net_ex2` additionally resolves the destination *row* on the device from the
index tensor the fused-away SET_ROWS would have read (`PXA_DN_SCATTER_FUSE`). The second form is
what lets the safe gather/scatter state path drop the SET_ROWS without reintroducing the read-side
aliasing that `PXA_DN_NP1_FASTPATH` was turned off for. Passing `nullptr` for the override gives
exactly the classic behaviour.

### `pxa-ew-fuse.cuh`
`PXA_EW_FUSE`: generic straight-line elementwise chain fusion. On the 5-card P100 decode seat the
host timeline shows ~1340 kernel launches per token costing ~12.4 ms of a ~37 ms token, and 38% of
all launches are four trivial elementwise kernels (add, scale, mul, sigmoid) whose GPU time is
1.6–3.2 µs each — pure launch overhead. This collapses a run of consecutive nodes, each the sole
consumer of the previous one's output and all same-shape contiguous f32 non-view non-OUTPUT, into
one interpreter kernel applying the same scalar formulas in the same order per element.
Elementwise ops carry no cross-element reduction, so the result is bit-identical by construction.
Covered ops: ADD/MUL (same shape, no broadcast), SCALE, SQRT, CLAMP, and UNARY
sigmoid/silu/abs/sgn/neg; anything else ends the chain. **Env:** `PXA_EW_FUSE=1` **default off**;
`PXA_EW_FUSE_MIN` default 2 (minimum run length); `PXA_EW_FUSE_LOG=1` prints each distinct chain
signature on first fire.

### `pxa-fastdiv.cuh`
Header-only 32-bit fast integer division by a loop-invariant divisor: `n/d` becomes
`(__umulhi(n, mp) + n) >> L` with `mp` and `L` precomputed on the host. Valid for
`1 ≤ d ≤ UINT32_MAX` and `0 ≤ n ≤ INT32_MAX`, which is the range every caller is gated on. It
exists because GP100 has no 64-bit integer divider, so a 64-bit division inside an index
expression costs tens of instructions where this costs two. It predates the upstream `ggml`
fastdiv helpers in `common.cuh`, and the copy in `solve_tri.cu` is a stub that still emits a real
division — hence the `pxa_` prefix, so it cannot collide with either. No env gate; used by
`cpy.cu` and `concat.cu`.

---

## Volta flash attention

### `fattn-volta-mma.cu` / `.cuh` and `fattn-volta-mma-inst-d{128,256}-n{1,2,4,8}.cu`
`PXA_FA_MMA_VOLTA`: sm_70 flash attention on mainline llama.cpp's MMA kernel. Profiling the
2×V100 pair on a 20801-token prefill showed both engines issuing the *same* 352 flash-attention
calls — 25.17 s on this fork's legacy `nvcuda::wmma` kernel against 9.22 s in mainline, a 2.73×
deficit that accounted for the entire GPU-compute gap (61.1 s vs 45.8 s). Mainline's Volta kernel
issues `m8n8k4` directly and, via `ncols2`, reads each K/V row once for several of the Q heads
that share it. Scope is narrow on purpose: **cc 7.0 only** (every other arch keeps its existing
dispatch untouched), head sizes 128 and 256 with `DKQ == DV`, **f16 K and V only** (mainline
stages a quantized KV cache through an allocator hook this fork does not have, so a non-f16 cache
would hand the kernel an unallocated pointer), and batch > 8 only — decode keeps the incumbent
path. Anything declined falls through to the tile route, which is itself a large win over the WMMA
kernel, so this can only ever change *which working kernel* runs. The header is intentionally thin
so none of the vendored macros escape into the dispatcher's translation unit. The eight
`-inst-` files are explicit instantiations (head dim × `ncols`) split out to bound per-TU compile
cost, all inside `namespace pxa_volta_fa`.

### `volta-mma/`
Five headers vendored verbatim from ggml-org/llama.cpp commit
`9400c8946e4da5e7694f2c26d6d4e50e14b690fa` (2026-09-02), MIT licensed, © 2023-2026 The ggml
authors. They are *vendored rather than merged* because this fork's `ggml-cuda` is ik_llama-derived
and its `mma.cuh` / `fattn-common.cuh` have diverged from mainline's: the two `launch_fattn`
template signatures and the two `fattn_kernel_t` ABIs are not compatible. Every symbol is renamed
into a `pxa_volta_*` namespace so both implementations coexist in one binary without ODR
conflicts, and every local edit is marked `PXA:`.
`mma-ml.cuh` is mainline's `mma.cuh` (the tensor-core tile primitives, including the sm_70
`mma.sync.m8n8k4` tiles — `pxq4-v70.cuh` uses these too). `fattn-mma-ml.cuh` is mainline's
`fattn-mma-f16.cuh`, the kernel itself. `launch-ml.cuh` is the *launch half only* of mainline's
`fattn-common.cuh` — the kernel ABI, causal `KV_max` bound, stream-k fixup, result combine and
`launch_fattn`; the vec-kernel dot products and quantized-KV dequantizers are deliberately not
vendored, because the MMA kernel does not reference them and this fork has its own.
`swizzle-ml.cuh` is mainline's `fattn-swizzle.cuh` (shared-memory XOR swizzle, Turing+ only).
`volta-mma-compat.cuh` is PXA's own shim: it maps mainline spellings onto this fork's
`common.cuh` and reproduces the few mainline `common.cuh` excerpts this fork lacks, under the same
licence.

---

## Documented negatives — do not re-run these blind

Each of the following was implemented, proven correct, measured, and left in the tree
**env-gated and default off**. They are kept because they are correct and because the conclusion
can invert on other hardware — not because they are pending. Re-running any of them costs GPU
hours and reproduces a known number.

| Lever | Env | Measured | Where |
|---|---|---|---|
| V70 native Volta PXQ4 prefill GEMM | `PXA_PXQ_GEMM_V70` | **−52.4% / −49.1%** prefill (3121 / 20801 tok) at `-ub 2048`; **−40.7% / −39.9%** at `-ub 512`; `CFG=2` at ub512 is worse still (−44.6% / −44.2%) | 2×V100-PCIE-16GB, Qwen3.8-27B-PXQ4, 2026-09-03 |
| PXQ4 dp4a MMQ prefill tile | `PXA_PXQ4_MMQ` | **−48.7% / −49.5%** prefill (3121 / 20801 tok) at `-ub 512`; device busy went *up* (1.26 → 1.37 summed over the pair) while throughput halved | same cell, 2026-09-03 |
| Dequant prefetch arena | `PXA_DQC_MB` | **−6.3% / −4.8%** prefill at `-ub 2048`; −0.3% (noise) / **−1.7%** at `-ub 512`; decode flat; +768 MiB VRAM per device | same cell, 768 MiB/device, 2026-09-03 |
| Wide-store dequant kernel | `PXA_PXQ_DQ_WIDE` | **~2× slower**: 290–325 GB/s against the incumbent's 604–644 GB/s | 1×V100, 2026-09-03 |
| int8 tile smem double-buffer | `PXA_PXQ_I8_DBUF` | **−0.4%** | 1080 Ti, PXQ2, ub768, 2026-07-22 |
| int8 tile 128-token tiles | `PXA_PXQ_I8_BN=128` | **−23%** (−21% stacked with DBUF) | 1080 Ti, same run |
| DeltaNet out-gate fusion | `PXA_FUSE_DELTANET` bit 1 (2) | **−0.6%** prefill, **+0.5…1.0%** decode — i.e. nothing — and it carried a write-after-read race that made the logits differ on **every** run | 2×V100 −sm layer, Qwable-27B-PXQ4core (qwen35), 2026-09-04 |

**Why V70 and PXQ4-MMQ both lose, in one line:** a V100 does 62.8 TOPS of dp4a against 125 TFLOPS
of HMMA, and the dequant pass a PXQ4-native prefill GEMM removes is only ~12% of the prefill wall.
A 2× slower multiply cannot be paid for out of a 12% saving. Two structurally different
PXQ4-native sm_70 prefill GEMMs — one dp4a, one register-direct `m8n8k4` — independently land at
roughly half of dequant+cuBLAS. The conclusion **inverts** on any card whose int8 rate beats its
fp16 rate (Turing and Ampere INT8 tensor cores are 2–4× fp16 there), which is the reason both
files are kept rather than deleted.

**Why the dequant arena loses:** the incumbent `k_pxq6_dequant_matrix` already streams at
604–644 GB/s, about 70% of the V100's 900 GB/s peak. There is no bandwidth headroom for overlap
to exploit, so the side-stream traffic plus per-op bookkeeping nets negative. The prefetch is
bit-identical (a determinism check gave the same sha256 on and off) — it is a pure performance
loss, not a correctness risk.

**Why the out-gate fusion is out:** the fusion removes the kernel boundary between reading `x`
and writing the product. Unfused, that boundary is a grid-wide barrier — `FUSED_RMS_NORM` has read
every byte of `x` before `FUSED_MUL_UNARY` writes one. Fused, the only barrier is the per-block
`__syncthreads()` inside the RMS reduction, which orders nothing *between* blocks, and this kernel
is one block per row reading its row twice. When ggml-alloc places the product on top of an input
at a **shifted** base, block *r*'s store is block *r'*'s not-yet-read input. Exact aliasing is
harmless (index-for-index); only a shifted overlap races, which is why it presented as a rare flake
that needed `-sm layer` and a long prompt. Measured as the spread of the token-0 top-1 probability
over 10 identical greedy requests: mask 55 → 1.26e-02 (9 distinct values in 10 runs), bit 1 alone →
3.56e-02 (10/10), masks 0 / 1 / 48 / 53 → 0.00e+00 and all the same value. The guard
(`pxa_g2_addfuse_no_shifted_overlap`, the predicate the ADD+`FUSED_RMS_NORM` fusion already uses)
is now applied here too, so the bit is safe when it is asked for — it simply buys nothing.

**Why the int8 tile variants lose:** the shipped tile already runs 223–228 registers with zero
spills at 4 blocks/SM, i.e. it is ILP-saturated rather than latency-bound, so the smem ping-pong
buys nothing; and at `-ub 768` this MoE routes ~96 tokens per expert per ubatch, so 128-token
tiles run mostly partial-fill. Stream-K was audited and declined without a build: the grid is
already wave-bound, not SM-starved.

---

## If you move or add a file here

The build picks this directory up through four globs in `ggml/src/CMakeLists.txt`
(`ggml-cuda/pxa/*.cu`, `ggml-cuda/pxa/*.cuh`, `ggml-cuda/pxa/volta-mma/*.cuh`, and
`ggml-cuda/pxa/template-instances/mmvq-instance*.cu` for the CUDA build; the first three plus
`*.cuh` for the HIP build, which never carried the mmvq instances). A new `.cu` here is compiled
automatically. Nothing outside this directory needs to change unless you add a new entry point,
in which case the includes that reach into `pxa/` are:

| Including file | Includes from `pxa/` |
|---|---|
| `ggml/src/ggml-cuda.cu` | `pxa-enhance.cuh`, `pxa-smalln.cuh`, `pxq-mmvq.cuh`, `delta-net.cuh`, `pxa-dqcache.cuh`, `pxq4.cuh`, `pxq-moemap.cuh`, `pxq6.cuh`, `pxq6i8.cuh`, `pxq4-v70.cuh`, `pxq4-mmq.cuh`, `pxa_expert_shard.cuh`, `pxa-deltanet-fuse.cuh`, `pxa-ew-fuse.cuh` |
| `ggml/src/ggml-cuda/mmq.cu` | `pxa-enhance.cuh` |
| `ggml/src/ggml-cuda/fattn-tile-f16.cu`, `fattn-tile-f32.cu` | `pxa-enhance.cuh` |
| `ggml/src/ggml-cuda/mmvq-templates.cuh` | `pxq-mmvq.cuh` |
| `ggml/src/ggml-cuda/convert.cu` | `pxq4.cuh`, `pxq6.cuh` |
| `ggml/src/ggml-cuda/cpy.cu`, `concat.cu` | `pxa-fastdiv.cuh` |
| `ggml/src/ggml-cuda/fattn.cu` | `fattn-volta-mma.cuh` |
| `tests/test-pxq-dq-wide.cu` | `pxq6.cuh`, `pxa-dqcache.cuh` |
| `tests/test-pxq4-mmq.cu` | `pxq6.cuh`, `pxq6i8.cuh`, `pxq4-mmq.cuh` |
| `tools/pxq4-v70-test.cu` | `pxq4-v70.cuh` |

`pxa_expert_shard.cuh` reaches back out to `../grouped_moe_verify.cuh`, and `pxa-dqcache.cuh` to
`../common.cuh` and `../convert.cuh`; those two files stay outside the fence because
`grouped_moe_verify.cuh` is also used directly by `ggml-cuda.cu` and the other two are upstream.
