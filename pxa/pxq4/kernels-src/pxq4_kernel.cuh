// pxq4_kernel.cuh — PXQ4 (ggml type id 252) device decode primitives for sm_70, vendored from
// the pxq_llama engine and re-addressed for vLLM's two-tensor parameter split.
//
// PROVENANCE. Every device function below is a line-for-line copy of the corresponding
// function in /path/to/engine-repo/ggml/src/ggml-cuda/pxq6.cuh (read-only; nothing in that
// tree was modified). The ONLY permitted edits are ADDRESSING edits, listed exhaustively here:
//
//   1. The engine stores one contiguous blob per tensor and computes
//        panel = W + (e*panels + p) * (HDR + kslabs*SLAB)
//        slab  = panel + HDR + kb*SLAB
//        anchor= ((const half*)panel)[row]
//      vLLM's stock weight loaders can only narrow() a single declared dim, so the converter
//      splits the blob into two tensors (plan §5.3):
//        slabs  uint8   [P, S, 1088]   P = N/64 panels, S = K/32 slabs
//        anchor float16 [P, 64]
//      giving
//        slab   = slabs  + (p*kslabs + kb) * SLAB
//        anchor = anchor[p*64 + row]
//      This is a pure re-addressing of the SAME bytes in the SAME order — the split is a
//      memcpy partition performed offline, asserted round-trip-exact by the converter.
//
//   2. The expert axis (e, ids[], n_as) is deleted. This model has no MoE: the artifact has
//      866 tensors, zero *_exps, zero expert KVs. e is always 0.
//
//   3. The mmv stages its activation slice per CANONICAL CHUNK instead of staging the whole
//      K vector once. See k_pxq4_mmv for why this is bit-exact and why it matters.
//
//   4. Activations arrive fp16 and results leave fp16 (vLLM's linear dtype) instead of fp32.
//      See k_pxq4_mmv for the exactness argument.
//
// NOT vendored (deliberately dropped): every MoE kernel (gufuse / down_scat / grouped / WMMA),
// the HQ policy pxq6_pol_p6hq, the PXQ2/PXQ3/PXQ6R policies, the int8/MMVQ family, the K-split
// workspace kernels (they cudaMalloc and decline under stream capture — pxq6.cuh:2480-2494 —
// which is disqualifying under vLLM's FULL_AND_PIECEWISE cuda graphs), and every decode MODE
// variant except MODE_TAB. All engine env gates that select the other modes default to OFF
// (pxq6.cuh:140-152), so MODE_TAB + VECX is exactly what the shipping engine runs on sm_70
// (selection logic: pxq6.cuh:3393-3400, VECX default true at :141).

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "pxq4_kernel_tables.h"

#ifndef PXQ4_EXTERN_SHARED
#define PXQ4_EXTERN_SHARED extern __shared__ __align__(16)
#endif

// ---------------------------------------------------------------------------------------------
// device-resident tables. One copy per translation unit, as in the engine (pxq6.cuh:79-81).
// Initialised from the frozen literals; pxq4_upload_tables() may overwrite them from the values
// recorded in the checkpoint's gguf KVs pxa.pxq6.book / pxa.pxq6.sub, which is what the engine
// does when PXA_PXQ6_BOOK / PXA_PXQ6_SUB were set at quantize time.
// ---------------------------------------------------------------------------------------------
static __device__ float pxq4_book_g[16]  = PXQ4_BOOK_INIT;
static __device__ float pxq4_sub16_g[16] = PXQ4_SUB16_INIT;

// ---------------------------------------------------------------------------------------------
// format policy — pxq6_pol_p6, pxq6.cuh:317-346, with the panel-relative anchor() accessor
// removed (the anchor now arrives as its own tensor; see addressing edit 1).
// ---------------------------------------------------------------------------------------------
struct pxq4_pol {
    static constexpr int SLAB       = PXQ4_SLAB_BYTES;
    static constexpr int CODE_OFF   = PXQ4_CODE_OFF;
    static constexpr int CODE_BYTES = PXQ4_CODE_BYTES;
    static constexpr int CODE_WORDS = 4;
    static constexpr int NEFF       = PXQ4_NEFF;

    // tid < 16 stages the book, 16 <= tid < 32 stages the sublevels. Callers must have at
    // least 32 threads and must __syncthreads() before reading tab/sub.
    // The table lives in shared memory as 16 floats and MUST stay 16 floats. The
    // MODE_TAB table-delivery angle is closed: staging book PAIRS (256 float2 entries,
    // so one LDS.64 fetches book[byte & 0xf] and book[byte >> 4] together) was built and
    // measured bit-exact -- 16.1% fewer thread-instructions, 163x the bank conflicts,
    // and 0.73-0.87x the speed. Warp-shuffle broadcast of the same values was also built:
    // bit-exact, and a wash (0.99-1.03x), because SHFL uses the same Volta MIO issue port
    // as LDS at the same rate, at 48 registers instead of 32. The array-REFERENCE
    // parameters below make widening the table a COMPILE ERROR rather than a silent
    // regression. See docs/10-kernel-speed.md, "The MODE_TAB table-delivery angle is
    // closed".
    __device__ static void stage_tabs(float (&tab)[16], float (&sub)[16], int tid) {
        static_assert(sizeof(tab) == 64 && sizeof(sub) == 64,
                      "pxq4: the book/sublevel tables must stay 16 floats = 64 bytes; a wider "
                      "table reintroduces shared-memory bank conflicts (measured 0.797x)");
        if      (tid < 16) tab[tid]      = pxq4_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq4_sub16_g[tid - 16];
    }

    // The two effective scales of one (row, 32-column block). The engine's parity-locked
    // dequant contract is eff = fp32(anchor_fp16) * SUB16[s4], then w = eff * fp32(book[c]);
    // the multiply ORDER is load-bearing for bit-exactness and must not be reassociated
    // (pxq-cpu.h:16-18, ggml-pxq6-tables.h:7-9).
    __device__ static void row_effs(const uint8_t * slab, int row, float anch,
                                    const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elements  0-15 of this 32-element block
        eff[1] = anch * sub[sb >> 4];     // elements 16-31
    }

    // one code byte -> (book[low nibble], book[high nibble]). b indexes the LE byte of the
    // 16-byte code row held in q[0..3]; element 2b takes the low nibble, 2b+1 the high.
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8 * (b & 3))) & 0xff;
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
};

// per-row code load: one 16-byte vector load. The address is
//   slab_base + 64 + row*16, and slab_base is a multiple of 1088 = 68*16 from a
// torch-allocated (>=256 B aligned) tensor base, so the uint4 load is always 16-B aligned.
// The launcher asserts contiguity, which is what makes that hold. (pxq6.cuh:436-441)
static __device__ __forceinline__ void pxq4_ldcodes(const uint8_t * p, uint32_t * q) {
    *(uint4 *)q = *(const uint4 *)p;
}

// The inner accumulation shape. PXQ_CANON_V2 == 0 is the shipping form; it emits
// FMUL+FFMA+FADD per pair, and BOTH forms are deterministic, but they are NOT bit-identical
// to each other, so this constant is a build-time re-baselining switch, never a runtime flag.
// (pxq6.cuh:575-608)
static __device__ __forceinline__ float pxq4_acc2(float acc, float a0, float x0,
                                                  float a1, float x1) {
#if PXQ4_CANON_V2
    return __fmaf_rn(a1, x1, __fmaf_rn(a0, x0, acc));
#else
    return acc + (a0 * x0 + a1 * x1);
#endif
}

// dot product of one weight row's 32-element block against 32 activations.
// (pxq6.cuh:634-674, MODE_TAB arm only)
template <bool VECX>
static __device__ __forceinline__ float pxq4_dot32(const uint8_t * __restrict__ slab, int row,
                                                   float anch,
                                                   const float * __restrict__ xk,
                                                   const float * __restrict__ tab,
                                                   const float * __restrict__ sub) {
    float eff[PXQ4_NEFF];
    pxq4_pol::row_effs(slab, row, anch, sub, eff);
    uint32_t q[4];
    pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + row * pxq4_pol::CODE_BYTES, q);

    // Two partial sums: t[0] accumulates the eff[0] half (elements 0-15, i.e. byte pairs
    // b = 0..7), t[1] the eff[1] half. `(b*NEFF) >> 4` is the engine's index expression;
    // with NEFF == 2 it is exactly (b >= 8).
    float t[PXQ4_NEFF];
#pragma unroll
    for (int i = 0; i < PXQ4_NEFF; ++i) t[i] = 0.f;

    if (VECX) {
        // float4 activation loads. Requires &xk[0] to be 16-B aligned, which holds because
        // every caller passes a 32-float-aligned offset into a 16-B-aligned base.
#pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2 * b];
            const float2 p0 = pxq4_pol::pair(q, b,     tab);
            const float2 p1 = pxq4_pol::pair(q, b + 1, tab);
            t[(b * PXQ4_NEFF) >> 4]       = pxq4_acc2(t[(b * PXQ4_NEFF) >> 4],       p0.x, xv.x, p0.y, xv.y);
            t[((b + 1) * PXQ4_NEFF) >> 4] = pxq4_acc2(t[((b + 1) * PXQ4_NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
        }
    } else {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = pxq4_pol::pair(q, b, tab);
            t[(b * PXQ4_NEFF) >> 4] = pxq4_acc2(t[(b * PXQ4_NEFF) >> 4], p.x, xk[2 * b], p.y, xk[2 * b + 1]);
        }
    }
    return eff[0] * t[0] + eff[1] * t[1];
}

// Canonical chunk count: a function of SHAPE ONLY. That is the property that makes a K-split
// launch bit-identical to an unsplit one in the engine, and it is why we reproduce it here —
// it is the only way our mmv output matches the shipping engine's byte-for-byte.
// (pxq6.cuh:826-833)
static __host__ __device__ __forceinline__ int pxq4_canon_nfix(int kslabs, int cmax) {
    int lim = kslabs / PXQ4_MMV_KSEG;   // >= KSEG slabs per chunk so every lane stays busy
    if (lim < 1)    lim = 1;
    if (lim > cmax) lim = cmax;
    int n = 1;
    while (n * 2 <= lim) n *= 2;        // largest power of two <= lim
    return n;
}

// Upper bound on the slabs in any one canonical chunk; the mmv's dynamic smem is sized from it.
static __host__ __forceinline__ int pxq4_canon_max_chunk(int kslabs) {
    const int nfix = pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    return (kslabs + nfix - 1) / nfix;
}

// ---------------------------------------------------------------------------------------------
// full-matrix dequant. One block per slab, 64 threads, one row each.
// (pxq6.cuh:680-726, with addressing edit 1 and the expert axis removed)
//
// out is [N, K] row-major fp16 — vLLM's weight layout for torch.mm(x, w.t()).
// ---------------------------------------------------------------------------------------------
template <typename dst_t>
static __global__ void k_pxq4_dequant_matrix(const uint8_t * __restrict__ slabs,
                                             const __half  * __restrict__ anchor,
                                             dst_t * __restrict__ y,
                                             const int kslabs, const int64_t K) {
    __shared__ float tab[16];
    __shared__ float sub[16];

    // STORE COALESCING (engine change of 2026-07-27, kept verbatim). Decoding is naturally
    // row-major — one thread owns one row and produces 32 consecutive outputs — but storing
    // that straight to y has 32 threads writing addresses K elements apart, so each store
    // instruction touches 32 sectors to deliver 64 useful bytes. Stage the 64x32 tile in smem
    // and write it back along K instead. Same values, same addresses, different instruction
    // mapping => bit-identical.
    //
    // The +2 pad makes the row stride 34, i.e. 17 four-byte banks for a 2-byte dst_t;
    // gcd(17, 32) == 1, so the column-major fill below is bank-conflict-free.
    __shared__ dst_t tile[PXQ4_BM][PXQ4_QK + 2];

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);
    __syncthreads();

    const int64_t slab_id = blockIdx.x;
    const int64_t p       = slab_id / kslabs;
    const int     kb      = (int)(slab_id % kslabs);
    const int     row     = threadIdx.x;

    const uint8_t * slab = slabs + ((size_t)p * kslabs + kb) * pxq4_pol::SLAB;
    const float     anch = __half2float(anchor[(size_t)p * PXQ4_BM + row]);

    float eff[PXQ4_NEFF];
    pxq4_pol::row_effs(slab, row, anch, sub, eff);
    uint32_t q[4];
    pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + row * pxq4_pol::CODE_BYTES, q);

#pragma unroll
    for (int b = 0; b < 16; ++b) {                 // b = element-pair index
        const float  e = eff[(b * PXQ4_NEFF) >> 4];
        const float2 v = pxq4_pol::pair(q, b, tab);
        tile[row][2 * b]     = (dst_t)(e * v.x);
        tile[row][2 * b + 1] = (dst_t)(e * v.y);
    }
    __syncthreads();

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x  >> 5;
    for (int r = warp; r < PXQ4_BM; r += nwarp) {
        y[(p * PXQ4_BM + r) * K + kb * PXQ4_QK + lane] = tile[r][lane];
    }
}

// ---------------------------------------------------------------------------------------------
// decode matrix-vector: out[M, N] = x[M, K] * W[N, K]^T, for small M.
// (pxq6.cuh:914-971, k_pxq6_mmv, with addressing edits 1-4)
//
// grid  = (N/64, M)      block = PXQ4_MMV_KSEG * 64 = 256 threads
// dynamic smem = pxq4_canon_max_chunk(K/32) * 32 floats
//
// EDIT 3 — CHUNKED ACTIVATION STAGING, and why it is bit-exact.
// The engine stages the whole K-vector in smem, which caps K at (46 KiB - 1 KiB)/4 = 11520
// floats and forces every wider node (a dense ffn_down is K = 17408) onto a K-split kernel
// that allocates a workspace with a raw cudaMalloc — which declines under stream capture and
// is therefore unusable inside vLLM's FULL_AND_PIECEWISE cuda graphs. We stage per CANONICAL
// CHUNK instead. The canonical chunk boundaries b0/b1 already exist in the engine's fold
// (they are what makes split == unsplit); staging inside that loop changes only WHEN an
// activation is copied to smem, never which activations are multiplied by which weights, in
// what order, or with what rounding. The fold is therefore byte-identical to the engine's
// unsplit kernel, and the smem bound drops to ceil(kslabs/nfix)*128 bytes — 4352 B at
// K = 17408, i.e. the cliff disappears entirely rather than being worked around.
//
// EDIT 4 — fp16 activation intake and fp16 result.
// The engine's mmv takes fp32 x and writes fp32. vLLM hands us fp16 activations and wants an
// fp16 result. Converting half->float on the smem staging write is EXACT (every fp16 is
// representable in fp32), so the values fed to the fold are identical to what an fp32 x
// holding the same numbers would have supplied. The result is folded in fp32 exactly as in
// the engine and rounded once, at the end, with __float2half_rn — the same single rounding
// vLLM would apply anyway when storing an fp32 result into an fp16 output tensor. Net effect:
// the fp16 x also halves the activation bytes read per block, which matters because every
// block re-reads the whole activation vector.
//
// NOTE ON M: the weight is re-read once per token (grid.y), so this kernel is only a win for
// very small M; the caller must cap M (PXQ4_MMV_MAX_M, default 8, matching the engine's
// PXA_PXQ4_2D_MAX_NY at ggml-cuda.cu:4019-4021) and use dequant + GEMM above that.
// ---------------------------------------------------------------------------------------------
template <bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq4_mmv(const uint8_t * __restrict__ slabs,
           const __half  * __restrict__ anchor,
           const __half  * __restrict__ x,          // [M, K]
           __half        * __restrict__ out,        // [M, N]
           const int R, const int K) {
    const int p  = blockIdx.x;                      // panel = 64 output rows
    const int iy = blockIdx.y;                      // token

    // Dynamic shared memory. Routed through a macro so the host simulator
    // (pxq4_kernel_hostsim.cpp, a test-only artifact) can bind the same name to a global
    // arena; under nvcc this expands to the ordinary CUDA declaration.
    PXQ4_EXTERN_SHARED float pxq4_xs[];

    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    // PXQ_CANON_v1: two-level fixed-chunk fold per lane.
    const int nfix = pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4_QK;

        // barrier 1 also covers the stage_tabs writes on the first iteration, and protects
        // the previous chunk's readers from this chunk's writers on every later iteration.
        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq4_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
        __syncthreads();

        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq4_dot32<VECX>(pan_slabs + (size_t)kb * pxq4_pol::SLAB, row, anch,
                                  pxq4_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
        }
        su += t;
    }

    red[kseg * PXQ4_BM + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}


// ---------------------------------------------------------------------------------------------
// K-chunk-split decode mmv: the same fold as k_pxq4_mmv, spread over grid.y = nfix blocks.
//
// WHY. k_pxq4_mmv launches one 256-thread block per (panel, token). At decode (M = 1) on this
// model's TP4 shapes that is 64-136 blocks on an 80-SM V100: at most 256 threads per SM, i.e.
// 12.5% occupancy, and the kernel is latency-bound, measured at 155-276 GB/s of weight traffic
// against ~700 GB/s achieved by a plain fp16 GEMV on the same card (which is why 4x fewer
// bytes was only buying ~1.2x less time). The canonical chunk fold already partitions K into
// nfix independent per-lane partial sums; putting each chunk in its OWN block multiplies the
// grid by nfix (8-16 for every real shape of this model) and restores enough parallelism to
// approach bandwidth-bound behaviour. The cost is a small fp32 partials tensor
// (M * panels * nfix * 256 floats, always < 1/4 of the weight bytes, usually L2-resident
// between the two kernels).
//
// BIT-EXACTNESS ARGUMENT. Per (row, kseg) lane, k_pxq4_mmv computes
//     su = ((0 + t_0) + t_1) + ... + t_{nfix-1},  t_c = 0 + dot32(kb=b0+kseg) + dot32(+KSEG)...
// and then folds across ksegs, in kseg order, in k_pxq4_mmv's tail:
//     u  = ((0 + su_0) + su_1) + su_2) + su_3;  out = __float2half_rn(u)
// k_pxq4_mmv_part computes exactly t_c (identical staging, identical dot32 calls, identical
// kb order and rounding) and stores it UNSUMMED. k_pxq4_mmv_reduce then performs literally
// the two folds above, in the same order, in fp32, and applies the same single final rounding.
// No addition is reassociated, no rounding point moves: the composition is bit-identical to
// the monolithic kernel, which the parity gates assert (GPU: mmv_out vs mmv_out_mono;
// hostsim: pxq4_hostsim_mmv_split_f16 vs pxq4_hostsim_mmv_f16).
//
// grid = (nfix, panels, M), block = 256; nfix MUST equal pxq4_canon_nfix(kslabs, CMAX).
//
// nfix and panels arrive as EXPLICIT ARGUMENTS rather than being read off gridDim. Reading
// them off gridDim also works, but the explicit form closes a latent hazard: the pre-swap
// kernel took nfix from gridDim.y while k_pxq4_mmv_reduce took it as an argument, with
// nothing asserting the two agreed. pxq4_launch_mmv_split_f16 now passes ONE value to both
// and asserts it equals pxq4_canon_nfix(kslabs, CMAX). Cost measured on sm_70: +8 integer
// instructions in the block prologue (328 -> 336), zero change to the 173-instruction dot32
// loop body and zero change to the FFMA/FMUL/FADD census.
//
// dynamic smem = pxq4_canon_max_chunk(K/32) * 32 floats, same bound as k_pxq4_mmv.
// Marker for out-of-tree consumers (benchmarks, hostsim shims) that must pick the right
// launch shape and argument list for k_pxq4_mmv_part / k_pxq4_mmv_fused.
#define PXQ4_MMV_PART_CHUNK_MAJOR 1
// ---------------------------------------------------------------------------------------------
template <bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq4_mmv_part(const uint8_t * __restrict__ slabs,
                const __half  * __restrict__ anchor,
                const __half  * __restrict__ x,        // [M, K]
                float         * __restrict__ part,     // [M, panels, nfix, KSEG*64]
                const int K, const int nfix, const int panels) {
    // GRID ORDER (chunk-major): c is the FASTEST-varying grid dimension, so a panel's nfix
    // chunk blocks are adjacent in launch order. Under the previous (panels, nfix, M) order a
    // panel's chunk blocks were strided by `panels` in linear block id, so the concurrently
    // resident set read one k-chunk of every panel -- `panels` weight streams strided by
    // kslabs*SLAB (174 KB on TP4 gate_up) -- instead of a few panels' full K range. This is a
    // PURE ADDRESSING CHANGE: part[] keeps the identical (iy, p, c, tid) layout, every lane
    // still visits the same kb in the same order, and k_pxq4_mmv_reduce is unchanged.
    const int c      = blockIdx.x;                     // canonical chunk (fastest-varying)
    const int p      = blockIdx.y;                     // panel
    const int iy     = blockIdx.z;                     // token

    PXQ4_EXTERN_SHARED float pxq4_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    // this block's canonical chunk -- the same b0/b1 the monolithic loop would compute for c
    const int b0 = (kslabs * c) / nfix;
    const int b1 = (kslabs * (c + 1)) / nfix;
    const int n  = (b1 - b0) * PXQ4_QK;

    __syncthreads();                                   // covers the stage_tabs writes
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        pxq4_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
    }
    __syncthreads();

    float t = 0.f;
    for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
        t += pxq4_dot32<VECX>(pan_slabs + (size_t)kb * pxq4_pol::SLAB, row, anch,
                              pxq4_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
    }
    // one fully-coalesced 1024-B store per block: threadIdx.x == kseg*64 + row, matching the
    // red[] layout of k_pxq4_mmv so the reduce below can replay its fold verbatim.
    part[(((size_t)iy * panels + p) * nfix + c) * (PXQ4_MMV_KSEG * PXQ4_BM)
         + threadIdx.x] = t;
}

// grid = (panels, M), block = 64 (one weight row each). Reads back the [nfix, KSEG*64] tile
// of one (panel, token) and performs k_pxq4_mmv's two folds in its exact order (see the
// bit-exactness argument above). All loads are coalesced: 64 consecutive floats per (c, s).
static __global__ void __launch_bounds__(PXQ4_BM)
k_pxq4_mmv_reduce(const float * __restrict__ part,    // [M, panels, nfix, KSEG*64]
                  __half      * __restrict__ out,     // [M, R]
                  const int nfix, const int R) {
    const int p   = blockIdx.x;
    const int iy  = blockIdx.y;
    const int row = threadIdx.x;

    const float * base = part + (((size_t)iy * gridDim.x + p) * nfix)
                              * (PXQ4_MMV_KSEG * PXQ4_BM);
    float su[PXQ4_MMV_KSEG];
#pragma unroll
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) su[s] = 0.f;
    for (int c = 0; c < nfix; ++c) {                  // chunk fold: su_s = ((0+t_0)+t_1)+...
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            su[s] += base[(size_t)c * (PXQ4_MMV_KSEG * PXQ4_BM) + s * PXQ4_BM + row];
        }
    }
    float u = 0.f;                                    // kseg fold, in kseg order
#pragma unroll
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += su[s];
    out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
}


// ---------------------------------------------------------------------------------------------
// v4: single-launch fused split mmv -- k_pxq4_mmv_part and k_pxq4_mmv_reduce in one kernel.
//
// WHY. The two-launch split pays a full device drain + refill between the part kernel and the
// reduce kernel. Measured on this card that gap is several microseconds per call and is larger
// than the entire decode instruction stream is worth (the zero-instruction floor for the part
// kernel is only 1.15-1.22x). This kernel does the reduce in whichever block of a
// (panel, token) arrives LAST, so there is one launch instead of two, and ~240 launches per
// decode step disappear from the CUDA graph.
//
// BIT-EXACTNESS. The atomic is an ARRIVAL COUNTER, never an accumulator: no floating-point
// value is ever atomically combined, so the expression tree is untouched. The winning block
// replays k_pxq4_mmv_reduce's fold verbatim -- chunks ascending c, then ksegs ascending s, one
// __float2half_rn at the end -- reading the same fp32 part[] words the two-launch reduce would
// have read. Composed with the argument at k_pxq4_mmv_part, this is bit-identical to the
// monolithic k_pxq4_mmv.
//
// WHAT WOULD NOT BE BIT-EXACT (rejected; do not "simplify" into these):
//   - one block summing a RANGE of chunks (grid-stride / persistent blocks). A left-associated
//     chain cannot be split: ((t0+t1)+t2) + (t3+t4) != ((((t0+t1)+t2)+t3)+t4).
//   - atomicAdd of floats into an accumulator: nondeterministic order, breaks the contract.
//   - cooperative_groups::grid_group::sync(): needs cudaLaunchCooperativeKernel, which caps the
//     grid at what fits resident; gate_up needs 2176 blocks. It would force the persistent-block
//     rewrite above, i.e. exactly the reassociation this avoids.
//
// FORWARD PROGRESS. No block ever spins. Every block increments and exits; only the one that
// observes old == nfix-1 continues. Deadlock is impossible regardless of block scheduling.
//
// ctr: M*panels unsigned, ZERO on entry. The winner rearms its slot to 0 before returning, so
// a completed launch leaves the buffer ready for the next one -- no memset launch is needed and
// nothing is allocated in-capture.
//
// CONCURRENCY. Exactly one PXQ4 mmv may be in flight per device at a time. This is NOT a new
// constraint -- the persistent part[] arena is already shared and reused across every PXQ4
// module and has the identical hazard -- but it is now a CORRECTNESS dependency of the barrier,
// not only a scratch-aliasing one, and it is documented at the arena as well.
//
// FAILURE MODE. A launch torn down mid-flight leaves ctr non-zero; no later block then observes
// old == nfix-1, out[] is never written, and the caller silently consumes stale fp16. The
// failure is SILENT. Mitigation: the arena zeroes the counter region once at allocation, and
// any error path that abandons a launch must reset the arena.
// ---------------------------------------------------------------------------------------------

#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
// sm_60 (P100): the PTX memory consistency model (.release/.acquire scopes, fence.acq_rel)
// is sm_70+; ptxas rejects atom.release.gpu below that. Fall back to the classic pre-Volta
// threadFenceReduction pattern: __threadfence() then a plain atomicAdd for the release side,
// and __threadfence() again for the acquire side. This is exactly the "separate __threadfence()
// before a plain atomicAdd" alternative the sm_70 comment below describes as bit-exact but
// slower -- the atomic is an ARRIVAL COUNTER, never an FP accumulator, so no floating-point
// expression tree is touched and the outputs stay bit-identical to the sm_70 build.
static __device__ __forceinline__ unsigned pxq4_arrive_release(unsigned * p) {
    __threadfence();                 // release: this thread's prior part[] stores are device-visible
    return atomicAdd(p, 1u);
}
static __device__ __forceinline__ void pxq4_fence_acq_rel() {
    __threadfence();                 // acquire: other blocks' part[] stores become visible
}
#else
// atom.release.gpu: the release fence is fused into the RMW. A separate __threadfence() before
// a plain atomicAdd is also bit-exact but measurably slower (every block pays a full membar.gl).
static __device__ __forceinline__ unsigned pxq4_arrive_release(unsigned * p) {
    unsigned old;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p) : "memory");
    return old;
}
static __device__ __forceinline__ void pxq4_fence_acq_rel() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
#endif
// __ldcg (ld.global.cg, L2-only) is available on every arch we build; on sm_60 L1 is not
// coherent across SMs either, so the same hint is wanted there.
static __device__ __forceinline__ float pxq4_ld_part(const float * p) { return __ldcg(p); }
#else
// hostsim: blocks run strictly sequentially, so the RMW degenerates to ++ and the fences are
// no-ops. This gates the VALUES the fused path computes; it can NEVER observe the race (see
// the device stress harness, which is mandatory rather than optional).
static inline unsigned pxq4_arrive_release(unsigned * p) { const unsigned old = *p; *p = old + 1u; return old; }
static inline void  pxq4_fence_acq_rel() {}
static inline float pxq4_ld_part(const float * p) { return *p; }
#endif

template <bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq4_mmv_fused(const uint8_t * __restrict__ slabs,
                 const __half  * __restrict__ anchor,
                 const __half  * __restrict__ x,        // [M, K]
                 float         * __restrict__ part,     // [M, panels, nfix, KSEG*64]
                 unsigned      * __restrict__ ctr,      // [M, panels], zero on entry and exit
                 __half        * __restrict__ out,      // [M, R]
                 const int R, const int K, const int nfix, const int panels) {
    const int c      = blockIdx.x;                      // canonical chunk (fastest-varying)
    const int p      = blockIdx.y;                      // panel
    const int iy     = blockIdx.z;                      // token

    PXQ4_EXTERN_SHARED float pxq4_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];
    __shared__ int   last;

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    const int b0 = (kslabs * c) / nfix;
    const int b1 = (kslabs * (c + 1)) / nfix;
    const int n  = (b1 - b0) * PXQ4_QK;

    __syncthreads();                                    // covers the stage_tabs writes
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        pxq4_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
    }
    __syncthreads();

    float t = 0.f;
    for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
        t += pxq4_dot32<VECX>(pan_slabs + (size_t)kb * pxq4_pol::SLAB, row, anch,
                              pxq4_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
    }

    const size_t  tile  = (size_t)(PXQ4_MMV_KSEG * PXQ4_BM);
    float * const pbase = part + (((size_t)iy * panels + p) * nfix) * tile;
    pbase[(size_t)c * tile + threadIdx.x] = t;

    // ---- arrival barrier over this (panel, token)'s nfix blocks --------------------------
    // The __syncthreads() below is LOAD-BEARING and is the easiest line in this kernel to omit.
    // The release orders only thread 0's OWN prior accesses, so without it lanes 1..255 may not
    // have issued their part[] stores when the counter is bumped, and the winner reads stale
    // words. Omitting it still passes a single-shot parity check, by luck. Do not remove it.
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = pxq4_arrive_release(&ctr[(size_t)iy * panels + p]);
        last = (old == (unsigned)(nfix - 1));
        if (last) {
            ctr[(size_t)iy * panels + p] = 0u;           // rearm for the next launch
            pxq4_fence_acq_rel();                        // acquire the other blocks' part[]
        }
    }
    __syncthreads();                                     // propagates the acquire to the block
    if (!last) return;

    // ---- k_pxq4_mmv_reduce's fold, verbatim ---------------------------------------------
    // __ldcg keeps the read off L1 (not coherent across SMs on sm_70); the loaded value is the
    // same fp32 word either way, so this is a cache-hint change only.
    float su = 0.f;
    for (int cc = 0; cc < nfix; ++cc) {                  // chunk fold, ascending c
        su += pxq4_ld_part(&pbase[(size_t)cc * tile + threadIdx.x]);
    }
    red[threadIdx.x] = su;                               // threadIdx.x == kseg*64 + row
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;                                   // kseg fold, ascending s
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}

// ---------------------------------------------------------------------------------------------
// v6: MULTI-TOKEN (MT) fused split mmv — the concurrency kernel.
//
// WHY. Every mmv variant above re-reads the whole weight tensor once PER TOKEN (grid.z = M).
// At decode that makes a batch of M cost ~M times the weight traffic of M = 1, which is why
// served per-stream throughput halved at concurrency 2 (16.1 ms -> 26.8 ms steps) while a
// kernel that amortizes weight bytes over the batch (AWQ/TurboMind) lost only 12%. This
// kernel gives one block ALL M tokens of its (chunk, panel): each weight byte is decoded
// once and folded into M accumulators, so the weight traffic of a decode step is ~constant
// in M for M <= 8 (activations and partials still scale with M, but they are KB not MB).
//
// BIT-EXACTNESS. Per token, the fold is UNCHANGED: pxq4_dot32_mt performs, for each token m,
// exactly the b-loop and pxq4_acc2 calls of pxq4_dot32 in the same order on t[m], the caller
// accumulates per-kb results in the same kb order, the partials keep the identical
// [M, panels, nfix, 256] layout, and the winner replays k_pxq4_mmv_reduce's fold verbatim per
// token. Interleaving tokens in the inner loop reorders nothing WITHIN any token's expression
// tree, so each token's output is bit-identical to the monolithic kernel's — asserted by the
// hostsim gate (test_pxq4_mmv_mt.py) and the GPU parity gate.
//
// GRID = (nfix, panels, 1), block = 256, template MT == M (1..8; the mmv ceiling is 8, so a
// dispatchable batch always fits one tile and there is no ragged tail).
// dynamic smem = pxq4_canon_max_chunk(kslabs) * 32 * MT floats: token m's chunk slice starts
// at pxq4_xs + m*n (n = this chunk's float count, a multiple of 32 so every token slice keeps
// the 16-B alignment the float4 loads need).
// ctr = panels unsigned (token axis collapsed), same zero-on-entry/exit contract as v4.
// ---------------------------------------------------------------------------------------------
template <bool VECX, int MT>
static __device__ __forceinline__ void pxq4_dot32_mt(const uint8_t * __restrict__ slab, int row,
                                                     float anch,
                                                     const float * __restrict__ xk,
                                                     const int xs_stride,
                                                     const float * __restrict__ tab,
                                                     const float * __restrict__ sub,
                                                     float (&res)[MT]) {
    float eff[PXQ4_NEFF];
    pxq4_pol::row_effs(slab, row, anch, sub, eff);
    uint32_t q[4];
    pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + row * pxq4_pol::CODE_BYTES, q);

    float t[MT][PXQ4_NEFF];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int i = 0; i < PXQ4_NEFF; ++i) t[m][i] = 0.f;
    }

    if (VECX) {
#pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float2 p0 = pxq4_pol::pair(q, b,     tab);
            const float2 p1 = pxq4_pol::pair(q, b + 1, tab);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const float4 xv = *(const float4 *)&xk[m * xs_stride + 2 * b];
                t[m][(b * PXQ4_NEFF) >> 4]       = pxq4_acc2(t[m][(b * PXQ4_NEFF) >> 4],       p0.x, xv.x, p0.y, xv.y);
                t[m][((b + 1) * PXQ4_NEFF) >> 4] = pxq4_acc2(t[m][((b + 1) * PXQ4_NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
            }
        }
    } else {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = pxq4_pol::pair(q, b, tab);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                t[m][(b * PXQ4_NEFF) >> 4] = pxq4_acc2(t[m][(b * PXQ4_NEFF) >> 4],
                                                       p.x, xk[m * xs_stride + 2 * b],
                                                       p.y, xk[m * xs_stride + 2 * b + 1]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) res[m] = eff[0] * t[m][0] + eff[1] * t[m][1];
}

template <bool VECX, int MT>
static __global__ void __launch_bounds__(256)
k_pxq4_mmv_fused_mt(const uint8_t * __restrict__ slabs,
                    const __half  * __restrict__ anchor,
                    const __half  * __restrict__ x,        // [MT, K]
                    float         * __restrict__ part,     // [MT, panels, nfix, KSEG*64]
                    unsigned      * __restrict__ ctr,      // [panels], zero on entry and exit
                    __half        * __restrict__ out,      // [MT, R]
                    const int R, const int K, const int nfix, const int panels) {
    const int c = blockIdx.x;                              // canonical chunk (fastest-varying)
    const int p = blockIdx.y;                              // panel

    PXQ4_EXTERN_SHARED float pxq4_xs[];
    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];
    __shared__ int   last;

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);

    const int b0 = (kslabs * c) / nfix;
    const int b1 = (kslabs * (c + 1)) / nfix;
    const int n  = (b1 - b0) * PXQ4_QK;

    __syncthreads();                                       // covers the stage_tabs writes
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const __half * xt = x + (size_t)m * K;
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq4_xs[m * n + idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
    }
    __syncthreads();

    float tacc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) tacc[m] = 0.f;
    for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
        float res[MT];
        pxq4_dot32_mt<VECX, MT>(pan_slabs + (size_t)kb * pxq4_pol::SLAB, row, anch,
                                pxq4_xs + (size_t)(kb - b0) * PXQ4_QK, n, tab, sub, res);
#pragma unroll
        for (int m = 0; m < MT; ++m) tacc[m] += res[m];
    }

    const size_t tile = (size_t)(PXQ4_MMV_KSEG * PXQ4_BM);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        part[(((size_t)m * panels + p) * nfix + c) * tile + threadIdx.x] = tacc[m];
    }

    // ---- arrival barrier over this panel's nfix blocks (see k_pxq4_mmv_fused) -----------
    __syncthreads();                                       // all lanes' part[] stores issued
    if (threadIdx.x == 0) {
        const unsigned old = pxq4_arrive_release(&ctr[p]);
        last = (old == (unsigned)(nfix - 1));
        if (last) {
            ctr[p] = 0u;                                   // rearm for the next launch
            pxq4_fence_acq_rel();                          // acquire the other blocks' part[]
        }
    }
    __syncthreads();                                       // propagates the acquire
    if (!last) return;

    // ---- k_pxq4_mmv_reduce's fold, verbatim, once per token ------------------------------
    for (int m = 0; m < MT; ++m) {
        const float * pbase = part + (((size_t)m * panels + p) * nfix) * tile;
        float su = 0.f;
        for (int cc = 0; cc < nfix; ++cc) {                // chunk fold, ascending c
            su += pxq4_ld_part(&pbase[(size_t)cc * tile + threadIdx.x]);
        }
        red[threadIdx.x] = su;
        __syncthreads();
        if (kseg == 0) {
            float u = 0.f;                                 // kseg fold, ascending s
#pragma unroll
            for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
            out[(size_t)m * R + p * PXQ4_BM + row] = __float2half_rn(u);
        }
        __syncthreads();                                   // red[] is reused by the next token
    }
}
