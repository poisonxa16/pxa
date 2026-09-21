# PXQ4 CUDA kernel inventory + vLLM reusability verdict

Source tree: `/path/to/engine-repo` (branch fable9-swa-kv, HEAD acf8f245), read-only over ssh.
All line citations are from that tree. Nothing was built, run, or measured for this document.

---

## VERDICT (first, no preamble)

**No blocker.** The PXQ4 kernels are unusually portable — far more so than a ggml kernel family
normally is. FACT: `pxq6.cuh` is 3601 lines and contains exactly **10** occurrences of a `ggml_`
or `GGML_` token, of which 6 are in one host-side type-mapper (`pxa_pxq_fmt`, pxq6.cuh:3335-3345)
and 4 are in comments. The `__global__` kernels themselves touch **zero** ggml types: no
`ggml_tensor`, no `ggml_backend_cuda_context`, no `ggml_cuda_pool`, no ggml streams. They take raw
`const uint8_t* W`, `const half* A` / `const float* x`, `float* C`, plain ints, and a `cudaStream_t`
supplied by the caller.

The correct port strategy is **VENDOR-AND-WRAP, not rewrite**: copy the P6 (=PXQ4, id 252) slice of
`pxq6.cuh` verbatim into a torch CUDA extension, delete the ggml-side host drivers, and write a new
~350-line torch shim in their place. Numerics stay bit-identical to the llama.cpp engine because the
copied device code is byte-identical.

**LOC estimate (wrap path): ~500 lines copied verbatim + ~1,300-1,900 lines new/edited.**
Rewrite path: 2,500-4,000 lines and it throws away the frozen-numerics parity guarantee. Do not rewrite.

**Three caveats that shape the design, stated up front:**
1. The decode `mmv` family consumes **fp32 activations** and writes **fp32 output**
   (pxq6.cuh:635-637 `const float* xk`; pxq6.cuh:916-920, :968-969). vLLM hands fp16. Cheap to fix
   (one staging line, or a convert outside), but it is not a drop-in.
2. There is **no tensor-core PXQ4 GEMM currently reachable for a dense model**. `k_pxq6_gemm_grouped`
   (pxq6.cuh:2518) is a `__hfma2` half2 tile (~15-20 TF on V100, no HMMA); the WMMA kernel
   `k_pxq6_gemm_grouped_wmma` (pxq6.cuh:2913) exists and is sm_70-gated, but is wired **only** into
   the MoE prefill driver (ggml-cuda.cu:5056), and our model has no MoE. The dense 2D GEMM driver is
   clamped to sm_60 (ggml-cuda.cu:4533) after a measured **-18.6% on sm_70** regression
   (ggml-cuda.cu:4436-4444). Prefill in vLLM will therefore need either the WMMA kernel pointed at
   the one-expert 2D case (mechanically ~60 LOC, but UNMEASURED and NOT bit-exact) or a
   dequant→cuBLAS/torch.mm path.
3. The `PXA_PXQ_MMVQ` path (pxq-mmvq.cuh) is the **one** genuinely non-portable family: it lives
   inside ggml's `mmvq-templates.cuh` machinery, keys on `ggml_type` template parameters
   (mmvq-templates.cuh:21,39,42,136-137) and consumes `block_q8_1` (pxq-mmvq.cuh:134). Porting it
   means porting ggml's q8_1 activation quantizer too. Skip it in v1.

---

## 1. Which kernels exist for PXQ4

**Naming trap confirmed.** `pxq4.cuh` is 119 lines and contains **no PXQ4 (id 252) compute kernel at
all** — the id-250 legacy kernels were deleted (pxq4.cuh:59-60, :117-119). What survives there is
shared scaffolding only: `PXQ4_QK/BM/BN/SLAB_BYTES` (pxq4.cuh:21-25), `pxq4_tile_info`
(pxq4.cuh:30-35), `pxq4_rowmap` (pxq4.cuh:38), `k_pxq_tiles_2d` (pxq4.cuh:46-57),
`k_pxq4_gather_a_f16` (pxq4.cuh:68-80), `pxq4_glu_apply` / `k_pxq4_glu` (pxq4.cuh:87-110),
`PXQ4_MMV_KSEG 4` (pxq4.cuh:114). **All id-252 compute lives in `pxq6.cuh`**, selected by the policy
struct `pxq6_pol_p6` (pxq6.cuh:317-346), format tag `PXA_PXQ_FMT_P6 = 3` (pxq6.cuh:3323), mapped
from `GGML_TYPE_PXQ4` at pxq6.cuh:3337.

### (a) Pure dequant-to-fp16 (or f32)

| kernel | file:line |
|---|---|
| `k_pxq6_dequant_matrix<POL, dst_t>` | pxq6.cuh:681-726 |
| host wrapper `dequantize_row_pxq6_cuda<dst_t>` | pxq6.cuh:728-741 |

One and only one dequant kernel; `dst_t` templated (half or float). Launch is
`<<<nslabs, 64, 0, stream>>>` with `nslabs = (nrows/64)*kslabs` (pxq6.cuh:739-740). Includes the
2026-07-27 store-coalescing rewrite: decode row-major into a `__shared__ dst_t tile[64][34]`, then
store one warp per row along K (pxq6.cuh:695, :719-725). Aborts (not returns) on
`nrows%64 || n_per_row%32` (pxq6.cuh:731-735).

### (b) Fused dequant+GEMM (mmq-style prefill)

| kernel | file:line | notes |
|---|---|---|
| `k_pxq6_gemm_grouped<POL,RAG,PIPE>` | pxq6.cuh:2517-2625 | 64 thr, half2 FMA, **the workhorse** |
| `k_pxq6_gemm_gufuse<POL,dst_t,RAG,PIPE>` | pxq6.cuh:2631-2762 | K3: fused up+gate+GLU, MoE-shaped |
| `k_pxq6_gemm_down_scat<POL,RAG,PIPE>` | pxq6.cuh:2766-2880 | scatter-to-MoE-rows epilogue |
| `k_pxq6_gemm_grouped_wmma<POL,F32ACC>` | pxq6.cuh:2912-3010 | K6, sm_70 HMMA, **cc==700 only** |
| `k_pxq6_gemm_gufuse_wmma<POLU,POLG,dst_t>` | pxq6.cuh:3075-3207 | K6 v2 |
| `k_pxq6_gemm_down_scat_wmma<POL>` | pxq6.cuh:3212-3309 | K6 v2 |
| int8/DP4A prefill twin (`pxq6i8.cuh`) | pxq6i8.cuh:1-316 | sm_61-first, PXA_PXQ_INT8_PREFILL |

Only `k_pxq6_gemm_grouped` and `k_pxq6_gemm_grouped_wmma` are relevant to a plain vLLM linear; the
gufuse/down_scat variants are MoE-fused shapes we have no MoE for. (`k_pxq6_gemm_gufuse` is still
interesting later: a Qwen dense FFN *is* an up+gate pair with a SILU-swiglu epilogue, and
`pxq4_glu_apply` (pxq4.cuh:87-98) already implements the exact `unary==0` SILU-swiglu branch. That
is a v2 optimization, not v1.)

### (c) MMVQ / GEMV decode-path kernels

Two distinct families, do not conflate them:

**(c1) Bespoke PXA decode mmv** — the shipped default, entirely inside pxq6.cuh:

| kernel | file:line | thr |
|---|---|---|
| `k_pxq6_mmv<POL,MODE,VECX>` | pxq6.cuh:914-971 | 256 |
| `k_pxq6_mmv_ksplit` | pxq6.cuh:1163-1214 | 64 |
| `k_pxq6_mmv_ksplit_gen` | pxq6.cuh:1256-1337 | 256 |
| `k_pxq6_mmv_qp` (Q1 prefetch twin) | pxq6.cuh:1595-1659 | 256 |
| `k_pxq6_mmv_ksplit_gen_qp` | pxq6.cuh:1662-1750 | 256 |
| `k_pxq6_mmv_x2` (R2 two-rows-per-thread) | pxq6.cuh:1949-2016 | 128 |
| `k_pxq6_mmv_ksplit_gen_x2` | pxq6.cuh:2019-2103 | 128 |
| `k_pxq6_mmv_redfuse` | pxq6.cuh:984-1055 | 256 |
| reducers `k_pxq_mmv_reduce` / `_reduce_s` | pxq6.cuh:1218-1234 / 1342-1362 | |
| gateup twins `k_pxq6_gateup_mmv*` (5 variants) | pxq6.cuh:839-912, 1065-1122, 1381-1483, 1755-1863, 2106-2212, 2297-2366 | |

All share one device function: `pxq6_dot32<POL,MODE,VECX>` (pxq6.cuh:634-674) — 32 K-values, one
row, one thread. That function plus `pxq6_pol_p6` plus `pxq6_ldcodes` (pxq6.cuh:436-464) *is* the
PXQ4 decoder. Everything else is grid/occupancy variation around it.

**(c2) PXQ inside ggml's stock MMVQ** — `pxq-mmvq.cuh` (223 lines), `vec_dot_pxq_q8_1`
(pxq-mmvq.cuh:131-155), registered at mmvq.cu:109-112 and :359-362, block logic in
mmvq-templates.cuh:18-85, :552+. int8/dp4a against q8_1-quantized activations. Default OFF
(pxa_pxq_mmvq_mode, pxq-mmvq.cuh:167-190), explicitly **not bit-exact** vs the fp16 mmv
(pxq-mmvq.cuh:16-17). Arch gate `pxa_pxq_mmvq_on` (pxq-mmvq.cuh:219-223): mode 1 => `cc >= CC_VOLTA`,
mode 2 => `cc >= 610`.

### (d) The 2D variants

There are no separate 2D *kernels*. FACT: the "2D" paths are **host drivers in ggml-cuda.cu that
reuse the MoE kernels with a one-expert degenerate map**:

- `pxa_pxq_mmv_2d` (ggml-cuda.cu:4208-4410) — decode/GEMV, `ny <= PXA_PXQ4_2D_MAX_NY` (default 8,
  ggml-cuda.cu:4021). Feeds `k_pxq6_mmv*` a one-entry all-zero `ids` buffer with both strides 0
  (ggml-cuda.cu:4302-4310, :4404-4409), so `e` is always 0 and `pxq6_panel()` degenerates to
  `W + p*panel_stride`.
- `pxa_pxq_gemm_2d` (ggml-cuda.cu:4493-4570) — prefill, `ny > MAX_NY`. Builds a device-side tile map
  with `e==0` via `k_pxq_tiles_2d` (ggml-cuda.cu:4549-4551), converts src1 f32→f16 into a pool
  buffer (ggml-cuda.cu:4556-4559), then launches `k_pxq6_gemm_grouped` at
  `<<<dim3(R/64, ntiles), 64>>>` (ggml-cuda.cu:4563-4565).

**This is the single most important structural fact for the vLLM port**: our own codebase already
proved that the MoE-shaped kernels serve a plain 2D `y = x @ W^T` by supplying a trivial degenerate
map. vLLM needs exactly that 2D case, so the port reuses a call pattern we already ship.

---

## 2. Signatures, layout assumptions, arch gating

### Layout assumptions (identical across every kernel — one policy struct owns them)

`pxq6_pol_p6` (pxq6.cuh:317-346), constants from ggml-pxq6-tables.h:21-25:
`PXQ6_QK=32`, `PXQ6_BM=64`, `PXQ6_SLAB_BYTES=1088`, `PXQ6_HDR_BYTES=128`, `CODE_OFF=64`, `NEFF=2`,
`CODE_WORDS=4`, `CODE_BYTES=16`.

- `panel_stride = HDR + kslabs*SLAB` (pxq6.cuh:519-522); `panel(W,e,panels,p,kslabs) =
  W + ((e*panels)+p)*stride` (pxq6.cuh:523-527). Experts outermost, panels row-major, slabs K-major.
- anchor: `__half2float(((const half*)panel)[row])` (pxq6.cuh:325-327) — one fp16 per row in the
  128 B header.
- sub-scale: `eff[0]=anch*sub[slab[row]&0xf]` (elems 0-15), `eff[1]=anch*sub[slab[row]>>4]`
  (elems 16-31) (pxq6.cuh:329-334). Scale SoA is the first 64 B of the slab.
- codes: 16 B per row at `slab + 64 + row*16`, byte b = code(2b) | code(2b+1)<<4
  (pxq6.cuh:336-340). Loaded as one `uint4` (pxq6.cuh:438-441).
- hard geometry requirement: `rows % 64 == 0 && K % 32 == 0`, asserted host-side
  (ggml-cuda.cu:4237, :4514) and abort-checked in the dequant wrapper (pxq6.cuh:731-735).

**No cross-K coupling anywhere** — confirms the sharding conclusion in the brief. A K-subrange is a
slab subrange with a duplicated 128 B header; a row-subrange at a multiple of 64 is whole panels.

### Signatures

```
// (a) dequant
template <class POL, typename dst_t>
__global__ void k_pxq6_dequant_matrix(const uint8_t* wq, dst_t* y, int kslabs, int64_t K);
   grid = nslabs = (nrows/64)*kslabs ; block = 64 ; smem = static
   writes y[(p*64+r)*K + kb*32 + lane]   -> row-major [nrows][K]        pxq6.cuh:681-726

// (b) prefill GEMM
typedef void (*pxq6_gemm_fn)(const uint8_t*, const half*, float*, const float*, size_t,
                             const pxq4_tile_info*, int, int);                pxq6.cuh:3366-3367
template <class POL, bool RAG, bool PIPE>
__global__ void k_pxq6_gemm_grouped(const uint8_t* W, const half* A, float* C,
        const float* bias, size_t bias_nb1, const pxq4_tile_info* tiles, int R, int K);
   grid = dim3(R/64, ntiles) ; block = 64 ; smem static (sW[32][64]+sA[32][64] half = 8 KB)
   A  : row-major half [M][K],  At = A + tile.row0*K            pxq6.cuh:2524
   C  : row-major f32  [M][R],  Ct = C + tile.row0*R + p*64     pxq6.cuh:2525, store :2621-2623
   bias: optional f32, per output row, indexed [e][row]         pxq6.cuh:2617
   accumulate: half2 __hfma2, 8 rows x 8 tokens per thread      pxq6.cuh:2594-2611

// (b') WMMA twin, same typedef, block = 256
template <class POL, bool F32ACC>
__global__ void k_pxq6_gemm_grouped_wmma(... identical arg list ...);          pxq6.cuh:2912-2916

// (c) decode mmv
typedef void (*pxq6_mmv_fn)(const uint8_t*, const char*, size_t, size_t,
                            char*, size_t, size_t, const char*, size_t, size_t, int, int, int);
template <class POL, int MODE, bool VECX>
__global__ void k_pxq6_mmv(const uint8_t* W,
        const char* x_base, size_t x_tok_stride, size_t x_slot_stride,
        char* dst_base, size_t dst_tok_stride, size_t dst_slot_stride,
        const char* ids, size_t ids_nb0, size_t ids_nb1, int R, int K, int n_as);
   grid = dim3(R/64, 1, ny) ; block = 256 ; DYNAMIC smem = K*4 + KSEG*64*4    ggml-cuda.cu:4274,4404
   x   : cast to (const float*) -> **fp32 activations**                       pxq6.cuh:930
   dst : cast to (float*), out[p*64+row] = u -> **fp32 output**               pxq6.cuh:968-969
```

### Arch gating — all of it is HOST-SIDE dispatch policy, not device-side `#ifdef`

FACT: grepping `__CUDA_ARCH__` in pxq6.cuh returns exactly 4 hits — pxq6.cuh:2893, :2917, :3083,
:3218 — and **every one is a WMMA kernel** guarded `>= 700 && < 750`. The dequant, the decode mmv
family and `k_pxq6_gemm_grouped` carry **no `__CUDA_ARCH__` guard at all** and compile/run on any
arch that supports `__hfma2` (cc >= 5.3). There is no `CC_PASCAL`/`CC_VOLTA` reference anywhere in
pxq6.cuh's device code; the only cc test in that file is host-side
(`pxa_pxq6_ksplit_gen_eff`, pxq6.cuh:255-270: `cc == 700 || cc == 600` → default S=4).

Host-side gates that matter:

| gate | location | behaviour |
|---|---|---|
| WMMA prefill | ggml-cuda.cu:5032 | `cc == 700 ? pxa_pxq6_wmma() : 0` — Volta only, env default OFF (pxq6.cuh:233-237); reachable **only from the MoE driver** |
| dense 2D GEMM | ggml-cuda.cu:4533 | `arch_ok = cc < CC_VOLTA && fast_fp16_available(cc)` — sm_60 ONLY; mode 2 is clamped and announced on sm_70 (ggml-cuda.cu:4525-4531) |
| dense 2D GEMM default | ggml-cuda.cu:4468 | default 0 (OFF); ENHANCE auto-sets 1 only when an sm_60 device is present AND the model is dense |
| MMVQ | pxq-mmvq.cuh:219-223 | mode1 `cc >= CC_VOLTA`, mode2 `cc >= 610` |
| int8 prefill | pxq6i8.cuh:33 | sm_61 target; sm_60 falls to emulated dp4a — "never ship that" |
| **Volta cuBLAS routing** | **mmq.cu:250-267** | `if (cc >= CC_VOLTA && cc < CC_TURING)` and `ne11 >= pxa_volta_cublas_ne11()` (default 64 since 2026-07-21) → return false, i.e. route to dequant+cuBLAS fp16 instead of DP4A MMQ. Measured +9.4% prefill on 1xV100 (mmq.cu:253-262). |

**mmq.cu carries no PXQ-specific code.** The only PXA content is that Volta threshold. PXQ4 never
enters ggml's MMQ trait machinery — that was a deliberate design decision, argued explicitly at
pxq6i8.cuh:3-14 ("wedges panel-interleaved, row-meta'd PXQ layouts into trait machinery built for
standard ggml block quants"). **This is good news for the port**: there is no MMQ entanglement to
unpick, and the same argument that kept PXQ out of ggml's MMQ applies verbatim to keeping it out of
vLLM's GGUF loader.

---

## 3. ggml dependencies to strip or shim

Enumerated exhaustively. This list is short, which is the headline.

**Device code (the `__global__`/`__device__` functions): ZERO ggml dependencies.** No `ggml_tensor`,
no `ggml_backend_cuda_context`, no `ggml_cuda_pool`, no ggml stream type. Kernel args are raw
pointers and ints. Verified by grep: the 10 `ggml_`/`GGML_` hits in pxq6.cuh are at :9, :200, :381
(comments) and :3335-3342 (`pxa_pxq_fmt`, a host-side `ggml_type` → int switch).

**Things that must be stripped or shimmed:**

| # | dependency | where | fix |
|---|---|---|---|
| 1 | `#include "ggml.h"` via `pxa-enhance.cuh` | pxa-enhance.cuh:40 (for `ggml_pxa_model_profile`) | drop pxa-enhance.cuh entirely; replace `pxa_gate_default(x)` (used by the `PXA_PXQ6_GATE` macro, pxq6.cuh:125-135) with the literal default. ~20 lines. |
| 2 | `pxa_pxq_fmt(ggml_type)` | pxq6.cuh:3335-3345 | delete — the vLLM config knows it is PXQ4; hardcode `PXA_PXQ_FMT_P6`. |
| 3 | `pxq23.cuh` include for P1/P2/P3 policies | pxq6.cuh:69 (indirect), pxq23.cuh:199,231,266 | the `PXQ6_PICK*` macro pickers (pxq6.cuh:3375-3460) reference `pxq6_pol_p1/p2/p3/p6r`. Either vendor pxq23.cuh (373 lines, self-contained: only includes the tables headers, pxq23.cuh:41-43) or delete the pickers and instantiate `pxq6_pol_p6` directly. **Recommend: delete the pickers**, saves ~150 lines and all the P2/P3 tables. |
| 4 | `ggml_cuda_pool_alloc<T>` (A_f16 staging, tile map) | ggml-cuda.cu:4549-4557 | replace with `torch::empty({...}, options)` — vLLM's caching allocator. ~10 lines. |
| 5 | `ctx.stream()`, `ctx.device` | ggml-cuda.cu:4300, :4211 | `at::cuda::getCurrentCUDAStream()`, `x.device().index()`. ~4 lines. |
| 6 | `ggml_get_to_fp16_cuda(GGML_TYPE_F32)` f32→f16 converter | ggml-cuda.cu:4557-4559 | not needed — vLLM activations are **already** fp16. Deleting this is a saving, not a shim. |
| 7 | `pxa_pxq4_bufs_on_device()` | ggml-cuda.cu:4245, :4529 | delete; torch tensors carry their device. |
| 8 | `CUDA_CHECK` macro (common.cuh) | throughout the drivers | replace with `C10_CUDA_KERNEL_LAUNCH_CHECK()`. |
| 9 | `ggml_cuda_dp4a` (common.cuh) | pxq6i8.cuh:281, pxq-mmvq.cuh | only in the int8 families — **not ported in v1**. |
| 10 | `block_q8_1`, `ggml_type` templates, `quantize_row_q8_1_cuda` | pxq-mmvq.cuh:134, mmvq-templates.cuh:21,39,42 | **the hard one — do not port the MMVQ family in v1.** |
| 11 | env-var gates via `getenv` (~25 of them) | pxq6.cuh:96-247 | harmless but should be frozen to constants so a stray env var cannot change vLLM numerics silently. |
| 12 | `pxq6_maybe_upload_tables` (`cudaMemcpyToSymbol` + `cudaSetDevice`) | pxq6.cuh:274-306 | keep the `__device__ float pxq6_book_g[16]/pxq6_sub16_g[16]` symbols (pxq6.cuh:78-80) with their frozen `PXQ6_BOOK_INIT`/`PXQ6_SUB16_INIT` initializers and **delete the upload path** — it is only for env overrides, and it does `cudaSetDevice` mid-call, which is not CUDA-graph-capture safe. |
| 13 | `pxq6_ksplit_workspace` — raw `cudaMalloc`, declines under stream capture | pxq6.cuh:2480-2494 | **capture hazard.** vLLM runs `cudagraph_mode=FULL_AND_PIECEWISE`. Either preallocate the workspace at `process_weights_after_loading()` time, or skip the K-split kernels in v1 (they are bit-identical to the unsplit form — pxq6.cuh:26-31 — so skipping costs only occupancy). |

**Not a ggml dependency but a real constraint:** the mmv family stages the whole x vector in dynamic
shared memory, capped at 46 KB (ggml-cuda.cu:4262), i.e. `K <= 11264` for the plain S=1 form
(ggml-cuda.cu:4254-4257). Checked against this model at TP=4: ffn_down K=17408/4=4352 → 17.4 KB, OK;
ffn_gate/up K=5120 → 20.5 KB, OK; attn_output K=6144/4=1536, OK. **Every tensor fits at TP=4 and at
TP=2** (ffn_down 8704 → 34.8 KB). At TP=1, ffn_down (K=17408 → 69.6 KB) does **not** fit and needs
the S-split path. INFERENCE from the shape arithmetic, not measured.

---

## 4. Wrap or rewrite? — honest judgement + LOC

### Contract match

vLLM: `y = x @ W^T + bias`, `x` contiguous CUDA fp16 `[M, K]`, `W` stored `[N, K]`, out fp16 `[M, N]`,
weights created in `create_weights()` and consumed in `apply()`.

`k_pxq6_gemm_grouped` computes exactly `C[m][n] = sum_k A[m][k] * W[n][k]` with `A` = row-major half
`[M][K]` and `W` panel-interleaved over its N (=`R`) rows (pxq6.cuh:2524-2525, :2594-2611,
:2617-2623). **The shape contract is already the vLLM contract.** The only mismatches:

| mismatch | severity | fix |
|---|---|---|
| C is fp32, vLLM wants fp16 | trivial | template `dst_t` on the store (the sibling `k_pxq6_gemm_gufuse` is already `dst_t`-templated, pxq6.cuh:2631) — ~6 LOC |
| needs a `pxq4_tile_info` array | trivial | `k_pxq_tiles_2d` already builds it on-device (pxq4.cuh:46-57), capture-legal by design (pxq4.cuh:41-45) — reuse verbatim |
| bias is fp32 `[e][row]`, vLLM bias is fp16/fp32 `[N]` | trivial | pass `bias_nb1=0`; cast — ~4 LOC |
| decode mmv wants fp32 x | small | change `xs[idx] = x[idx]` → `__half2float(xh[idx])` at pxq6.cuh:931; dst cast at :968 — ~6 LOC per kernel |
| grid.y limit 65535 on tiles | none | `ntiles = ceil(M/64)`; M would have to exceed 4.19M tokens |

### Verdict: **WRAPPABLE — but by vendoring, not by linking.**

Be precise about what "wrap" can mean here. Every PXQ symbol is `static` inside a `.cuh` that is
`#include`d into ggml TUs (convert.cu:11, ggml-cuda.cu:46, :3966). There is **no exported symbol** to
link against — you cannot `dlopen` libggml-cuda and call `k_pxq6_gemm_grouped`. So "wrap" means:
copy the header slice into the vLLM extension's own `.cu` and compile it with vLLM's nvcc. That is
still a wrap in the sense that matters — **the device code is copied unchanged, so the numerics are
provably identical to the shipping engine** — but it is a source-level vendor, and it forks: future
llama.cpp-side kernel fixes will not propagate automatically. Budget a periodic re-sync, and put the
provenance canary comment (pxq6.cuh:1-3) in the vendored copy.

### LOC estimates

**Option A — vendor-and-wrap (RECOMMENDED)**

| component | LOC | kind |
|---|---|---|
| vendored `pxq4_pxa.cuh`: tables (ggml-pxq6-tables.h:21-30 subset), `pxq6_pol_p6`, `pxq6_ldcodes`, `pxq6_panel*`, `pxq6_acc2`, `pxq6_pairx`+mode structs, `pxq6_dot32`, `pxq6_deq_slab_cm`, `k_pxq6_dequant_matrix`, `k_pxq6_mmv`, `k_pxq6_gemm_grouped`, `pxq4_tile_info`, `k_pxq_tiles_2d` | ~500 | **copied verbatim** |
| edits to the vendored copy (deps 1,2,3,11,12 above; dst_t on the GEMM store; fp16 x staging) | ~60 | edited |
| optional: `k_pxq6_gemm_grouped_wmma` + `pxq6_wmma_*` helpers (pxq6.cuh:2884-3010, 3038-3070) | ~200 | copied verbatim |
| torch C++/CUDA shim: `torch::Tensor` entry points (`pxq4_dequant`, `pxq4_gemm`, `pxq4_gemv`), dtype/contiguity/shape checks, stream plumbing, `TORCH_LIBRARY` registration | ~350 | new |
| Python `PXQ4Config(QuantizationConfig)` + `PXQ4LinearMethod(LinearMethodBase)`: `create_weights` (raw uint8 panel blob + per-tensor metadata), `apply` (M-threshold dispatch gemv/gemm), `process_weights_after_loading` (no-op — no repack needed, the on-disk layout is the kernel layout), plus the **mixed-type dispatch** (PXQ4 / q8_0 / q6_k / f16 / f32 per LEVERS.md backbone rev2) | ~400 | new |
| offline GGUF→vLLM converter: read the mixed-type gguf, TP-shard PXQ4 by whole panels (column-parallel) or by slab byte-gather + header duplication (row-parallel), passthrough/dequant the q8_0 / q6_k / f16 tensors, emit safetensors + a `pxq4_meta.json` recording `pxa.pxq.backbone_rev`/`backbone_map` | ~600 | new |
| bit-exactness harness against the CPU reference (see §5) | ~150 | new |
| **TOTAL** | **~500 copied + ~1,560 new/edited** (range 1,300-1,900 depending on how much of the mixed-type converter is reused from existing tooling) | |

**Option B — rewrite the kernels from scratch against the format spec**

| component | LOC |
|---|---|
| PXQ4 dequant + GEMV + GEMM written fresh for a torch/Marlin-style layout | 1,200-2,000 |
| everything else in Option A that is not the vendored header | ~1,500 |
| **TOTAL** | **2,700-3,500** |

and it forfeits the thing that makes this cheap: **provable numeric identity with the llama.cpp
engine we already quality-gated**. A rewrite requires its own full G3/G4 quality gate on borrowed
GPUs we are not allowed to run on in this workflow. **Do not rewrite.**

### Where wrapping is *not* enough (be honest)

- **Prefill throughput.** `k_pxq6_gemm_grouped` is a scalar half2 tile with no tensor cores. Our own
  measurement note (ggml-cuda.cu:4436-4444) says that on sm_70, folding the dequant into this GEMM
  is **-18.6%** versus coalesced-dequant + cuBLAS HMMA, and spells out the corollary: *"a fix that
  coalesces the dequant and KEEPS cuBLAS's HMMA GEMM should dominate this on sm_70."* For vLLM the
  translation is: **v1 prefill should be `k_pxq6_dequant_matrix` → `torch.mm` (cuBLAS fp16)**, not
  the fused GEMM. That is also the *simplest* thing to write. The fused/WMMA GEMM is a v2 experiment
  requiring GPU time we do not have in this workflow.
- **Batched decode.** The mmv family is tuned for `ny <= 8` (ggml-cuda.cu:4021, :4239). vLLM with
  continuous batching routinely decodes M in the tens-to-hundreds. Above ~8 the port must fall to
  dequant+mm. The 110-120 tok/s projection in the brief is a **single-stream (M=1)** projection and
  the mmv path is the right kernel for exactly that case.
- **CUDA graph capture.** Plain `k_pxq6_mmv` at S=1 is capture-safe (no allocation, static+dynamic
  smem only). The K-split variants are not, because of `pxq6_ksplit_workspace`'s `cudaMalloc`
  (pxq6.cuh:2485-2492) — it explicitly declines under capture. Preallocate or omit.

---

## 5. CPU reference for bit-exact validation — YES

**`/path/to/engine-repo/ggml/src/pxq-cpu.c`**, 360 lines, plain C.

- `pxa_deq_row_pxq6(base, row, k, dst, hq=false)` — pxq-cpu.c:135-158. The PXQ4 (id 252) row dequant:
  panel stride pxq-cpu.c:140, fp16 anchor pxq-cpu.c:141, `eff = anchor * sub[nibble]` pxq-cpu.c:150-151,
  codes via `pxa_deq_pairs16` pxq-cpu.c:127-133, `o[i] = eff[..] * book[code]`.
- Public entry points: `pxa_pxq_dequant_row` (pxq-cpu.c:206-217, header pxq-cpu.h:53) and
  `pxa_pxq_dequant_2d` (pxq-cpu.c:219-225, header pxq-cpu.h:61) → row-major f32 `dst[row*k + col]`.
- The layout table at pxq-cpu.c:5-12 is the authoritative one-screen format summary
  (`PXQ4 (252): hdr 128, slab 1088, 1 scale B/row, 16 code B/row, code_off 64, eff per 16`).
- Reference matmul: `pxa_pxq_mul_mat_cpu` (pxq-cpu.c:343-360, header pxq-cpu.h:80).

**Portability**: it includes only `ggml-impl.h` (for `GGML_COMPUTE_FP16_TO_FP32`, pxq-cpu.c:30 —
replaceable with any correct fp16 converter, ~15 lines) plus the three frozen table headers. Compiles
standalone in an hour, or bind it with ctypes/pybind for a Python-side golden generator.

**What is and is not bit-exact — this is stated explicitly in our own source, do not get it wrong:**

- `pxq-cpu.h:16-18`: *"not required to be bit-exact with the CUDA GEMM kernels (which snap products
  to fp16 inside the MMA). **The dequant itself IS the parity-locked contract (fp32 eff/book
  products).**"*
- So: **`k_pxq6_dequant_matrix` vs `pxa_deq_row_pxq6` must match bit-for-bit in fp32** — both compute
  `eff = fp32(anchor) * SUB[s]` then `w = eff * book[c]` in the same order (pxq6.cuh:713-716 vs
  pxq-cpu.c:129-131). This is the gate to use for the port: **dequant bit-exactness proves the layout
  reader, the panel arithmetic, the TP byte-gather repack, and the table constants are all correct**,
  which is >90% of the risk in this port.
- The GEMM/mmv paths are *not* bit-exact against that reference (fp16 accumulation), and neither is
  the MMVQ path (s8 book snap, pxq-mmvq.cuh:16-17). Gate those with a tolerance/logprob-parity test,
  not with a hash.

**Recommended v1 validation ladder** (no GPU time required beyond a single tiny allocation):
1. Python: read a PXQ4 tensor out of the gguf, run the ctypes-bound `pxa_pxq_dequant_2d` → golden f32.
2. Run the ported `k_pxq6_dequant_matrix` on the same bytes → compare `torch.equal` in fp32. Must be
   exact.
3. Do (1)+(2) again on a **TP-sharded** slice (both a column split at a 64-row boundary and a K split
   at a 32-column boundary) → this is the test that proves the byte-gather repack is a permutation
   and not a requantization.
4. Only then compare `pxq4_gemm` against `dequant → torch.mm` with an fp16-appropriate tolerance.

---

## Appendix — one-line answers

- **Which file has the id-252 kernels?** `pxq6.cuh`. `pxq4.cuh` has none (pxq4.cuh:59-60, :117-119).
- **Do the kernels know about ggml?** No. 10 token hits in 3601 lines, all host-side or comments.
- **Is there PXQ code in mmq.cu?** No — only the Volta cuBLAS routing threshold (mmq.cu:250-267).
- **Is there a tensor-core PXQ4 GEMM?** Yes (pxq6.cuh:2912), sm_70-only, but only reachable from the
  MoE driver (ggml-cuda.cu:5056), env-default OFF, and explicitly not bit-exact (pxq6.cuh:53-59).
- **Cheapest correct v1 for vLLM?** `k_pxq6_dequant_matrix` → `torch.mm` for prefill,
  `k_pxq6_mmv` (fp16-staged) for M<=8 decode. Both are ~600 vendored LOC.
