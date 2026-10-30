//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once
#include "pxa-env-flag.h"
#include "mmvq.cuh"
#include "vecdotq.cuh"
#include <type_traits>
#include <cstring>
#include "pxa/pxq-mmvq.cuh"
#include "mmvq-args.h"

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

// LANE PICK. A RUNTIME index into a per-thread array forces the whole array into local memory:
// one `dst[...] = tmp[j][threadIdx.x]` at the very end of the kernel costs
// `float tmp[ncols_y][rows_per_cuda_block]` its registers, and the accumulator then takes a local
// load and a local store on every k-block of the main loop. Measured with cudaFuncGetAttributes on
// this kernel's shape: localSizeBytes == ncols_y*rows_per_cuda_block*4 EXACTLY, and it is zero only
// when ncols_y == 1 or rows_per_cuda_block == 1 -- which is why plain single-column decode never
// paid it and every multi-column (spec-verify) width did. Picking the lane's element with an
// unrolled compare chain yields the same value and keeps the array in registers.
template <int N>
static __device__ __forceinline__ float pxa_lane_pick(const float (&v)[N], const int lane) {
    float r = v[0];
#pragma unroll
    for (int i = 1; i < N; ++i) {
        if (lane == i) {
            r = v[i];
        }
    }
    return r;
}

// PXQ tiers do not use the flat-block-index vec_dot: their panel/slab layout needs (row, kb) and
// the blocks-per-row count separately. Compile-time switch, so no other type's codegen changes.
template <ggml_type type>
static constexpr bool pxq_mmvq_layout = (type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQ4HQ ||
                                         type == GGML_TYPE_PXQ2 || type == GGML_TYPE_PXQ3 ||
                                         type == GGML_TYPE_PXQN4 || type == GGML_TYPE_PXQN4S8);   // + PXQN

// ROWS PER BLOCK. A PXQ slab holds 64 rows x 16 B of codes contiguously, so a k-block of ONE row
// is a 16 B island and the next k-block of that row is a whole slab (1088/1152 B) away: with the
// stock 1-row block a warp's 16 k-blocks touch 16 different 32 B sectors and use half of each,
// and the per-row scale bytes are 16 more sectors at ~3% use. Giving the block 16 ADJACENT rows
// makes the same warp read those rows' code rows (16 x 16 B = 256 B) and their scale bytes
// (16 x 2 B = 32 B) as contiguous runs -- full sectors, same total work, grid/16.
// ROWS is the PXQ tile height. It used to be clamped HERE at ncols_y > 2 ("so the accumulator
// file cannot explode"); that clamp now lives on the host, in mul_mat_vec_q_cuda_T, where it can
// be read off the width at launch time instead of being baked into every instantiation -- the
// value passed down is already clamped, so this template and the host's grid arithmetic agree
// with no rule to keep in sync, and the launch is unchanged for every caller that does not opt in.
// Non-PXQ types ignore ROWS entirely and keep the stock rule.
//
// What stays here is a compile-time CEILING, and it is a codegen constraint rather than a policy:
// the cross-warp staging buffers are (nwarps-1) x ncols_y x ROWS x WARP_SIZE floats, twice over in
// the up/gate twin, and the launcher's switch emits every (ncols_y, ROWS) pair whether or not it
// can select it -- ncols_y 8 at ROWS 16 asks the twin for 96 KB and ptxas refuses the whole build.
// The ceiling is set at the widest tile pxa_pxq_mmvq_tile can ask for at each width, so it never
// bites on a launch that actually happens; the worst surviving pair is 24 KB.
template <int ncols_y>
static constexpr int mmvq_rows_cap = ncols_y <= 2 ? 16 : (ncols_y <= 4 ? 8 : 4);

template <ggml_type type, int ncols_y, int ROWS>
static constexpr int mmvq_rows_per_block = pxq_mmvq_layout<type>
        ? (ROWS < mmvq_rows_cap<ncols_y> ? ROWS : mmvq_rows_cap<ncols_y>)
        : (ROWS > 1 ? ROWS : (ncols_y < 4 ? 1 : 2));   // ROWS > 1 on a stock type: the sm_70 tile, see mmvq_v100_rows

// PXA_PXQN_VERIFY_WIDE (2026-10-14): the PLAIN kernel's ceiling at 5..8 columns is 8, not 4. The 4 above is
// the up/gate twin's (two staging buffers: 48 KB at ncols_y 8 x ROWS 8); the plain kernel's single buffer at ncols_y 8 x
// ROWS 8 is (4-1) x 8 x 8 x 32 x 4 B = 24 KB. Only reachable through the PXA_PXQN_VERIFY_WIDE launch (the shipped tile
// table never asks for more than 4 rows above 4 columns), so every launch that exists today keeps its tile.
template <int ncols_y>
static constexpr int mmvq_rows_cap_plain = ncols_y <= 2 ? 16 : 8;

template <ggml_type type, int ncols_y, int ROWS>
static constexpr int mmvq_rows_per_block_plain = pxq_mmvq_layout<type>
        ? (ROWS < mmvq_rows_cap_plain<ncols_y> ? ROWS : mmvq_rows_cap_plain<ncols_y>)
        : mmvq_rows_per_block<type, ncols_y, ROWS>;

// The tier's policy, and the slab stride the panel arithmetic steps by. Both come off the policy
// so a new tier is one struct in pxq-mmvq.cuh and one arm here, with nothing to keep in sync.
template <ggml_type type>
using pxq_mmvq_pol =
    std::conditional_t<type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQN4, pxq_mmvq_pol_p6,
    std::conditional_t<type == GGML_TYPE_PXQ2, pxq_mmvq_pol_p2,
    std::conditional_t<type == GGML_TYPE_PXQ3, pxq_mmvq_pol_p3,
    std::conditional_t<type == GGML_TYPE_PXQN4S8, pxqn_mmvq_pol_n4s8, pxq_mmvq_pol_p6hq>>>>;

template <ggml_type type>
static constexpr int pxq_mmvq_slab = pxq_mmvq_pol<type>::SLAB;

template <ggml_type type, int VDR, int ROWS>
static __device__ __forceinline__ float pxq_mmvq_dot(
        const uint8_t * __restrict__ slab, const int r, const int i, const float anch,
        const float * __restrict__ sub, const pxq_mmvq_scales<pxq_mmvq_pol<type>, ROWS> & sc,
        const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    return vec_dot_pxq_q8_1<pxq_mmvq_pol<type>, VDR, ROWS>(slab, r, i, anch, sub, sc, bq8_1, iqs);
}

// The same dot in its two halves: decode one (row, k-block) of the weight once, then score that
// fragment against each activation column. Same arithmetic, same order, bit-identical result --
// see the note over pxq_mmvq_wfrag in pxa/pxq-mmvq.cuh.
template <ggml_type type, int VDR, int ROWS>
using pxq_mmvq_frag = pxq_mmvq_wfrag<pxq_mmvq_pol<type>, VDR, ROWS>;

template <ggml_type type, int VDR, int ROWS>
static __device__ __forceinline__ float pxq_mmvq_dot_f(
        const pxq_mmvq_frag<type, VDR, ROWS> & w, const float anch,
        const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    return pxq_mmvq_dot_frag<pxq_mmvq_pol<type>, VDR, ROWS>(w, anch, bq8_1, iqs);
}

// The 16-entry sub-scale table is the one lookup PXQ has that MXFP4's E8M0 exponent trick does
// not: its index is a 4-bit field of the weight stream, so it is divergent across the warp and
// the compiler cannot hoist it out of the k loop the way it hoists the uniform code book. Stage
// it in shared memory once per block and the per-k-block cost drops from an L1 global load to LDS.
template <ggml_type type>
static __device__ __forceinline__ const float * pxq_mmvq_stage_sub(float * s_sub, int tid) {
    if (tid < 16) {
        s_sub[tid] = pxq_mmvq_pol<type>::subtab()[tid];
    }
    __syncthreads();
    return s_sub;
}

// Per-block PXQ row invariants: every row of one block lives in the same 64-row panel (ROWS
// divides 64 and row0 is a multiple of ROWS), so the panel base is one address for the block and
// each row contributes exactly one fp16 anchor read for the whole K sweep. The 1/127 book fold
// rides along in the anchor so the inner loop carries one fewer multiply.
template <ggml_type type, int ROWS>
struct pxq_mmvq_rowbase {
    const uint8_t * pan;
    int   r0;
    float anch[ROWS];
    __device__ pxq_mmvq_rowbase(const void * __restrict__ vx, int row0, int kslabs) {
        pan = (const uint8_t *) vx
            + (size_t)(row0 >> 6) * ((size_t) PXQ6_HDR_BYTES + (size_t) kslabs * pxq_mmvq_slab<type>);
        r0  = row0 & 63;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            anch[i] = __half2float(((const half *) pan)[r0 + i]) * pxq_mmvq_pol<type>::BFOLD;
        }
    }
    __device__ const uint8_t * slab(int kbx) const {
        return pan + PXQ6_HDR_BYTES + (size_t) kbx * pxq_mmvq_slab<type>;
    }
};

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0   : return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1   : return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0   : return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1   : return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q6_0   : return vec_dot_q6_0_q8_1;
        case GGML_TYPE_Q8_0   : return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q2_K   : return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K   : return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K   : return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K   : return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K   : return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS : return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S  : return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S  : return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M  : return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL : return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_MXFP4  : return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_IQ4_XS : return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S  : return vec_dot_iq3_s_q8_1;
        default               : return nullptr;
    }
}

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0    : return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1    : return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0    : return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1    : return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q6_0    : return VDR_Q6_0_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0    : return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K    : return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K    : return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K    : return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K    : return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K    : return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS : return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS  : return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S   : return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS : return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S   : return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL  : return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4   : return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_PXQ4    : return VDR_PXQ_Q8_1_MMVQ;
        case GGML_TYPE_PXQ4HQ  : return VDR_PXQ_Q8_1_MMVQ;
        case GGML_TYPE_PXQ2    : return VDR_PXQ_Q8_1_MMVQ;
        case GGML_TYPE_PXQ3    : return VDR_PXQ_Q8_1_MMVQ;
        case GGML_TYPE_PXQN4   : return VDR_PXQ_Q8_1_MMVQ;
        case GGML_TYPE_PXQN4S8 : return VDR_PXQ_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS  : return VDR_IQ4_XS_Q8_1_MMVQ;
        default                : return 1;
    }
}

// HOIST: decode each weight fragment once per (row, k-block) and reuse it across the ncols_y
// activation columns, instead of once per (column, row, k-block). Bit-identical, on by default;
// HOIST=false reproduces the pre-2026-09-08 structure and exists so tests/test-pxq-mmvq-cols.cu
// can run both arms in one binary. PXQ tiers only -- no other type's codegen sees it.
// FAST (PXA_SPEC_FAST_VERIFY, ; PXQN4 at 2..8 columns only): pxq_mmvq_dot_frag_fast per column
// and the row anchor applied once to the finished sum. Deterministic, not bit-identical (see pxq-mmvq.cuh).
template <ggml_type type, int ncols_y, int nwarps, int ROWS, int VDRP, bool HOIST = true, bool FAST = false>
static __device__ void k_mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy,
    const float * bias, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const int bx = -1) {   // row-block index inside this matrix; -1 = blockIdx.x (every caller but the grouped launch)

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = pxq_mmvq_layout<type> ? VDRP : get_vdr_mmvq(type);

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    //int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < PXA_CC_RDNA2 ?
    //    ncols_y < 4 ? 1 : 2 : 1;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int rows_per_cuda_block = mmvq_rows_per_block_plain<type, ncols_y, ROWS>;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*(bx >= 0 ? bx : (int) blockIdx.x);
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;

    if constexpr (pxq_mmvq_layout<type>) {
        __shared__ float s_sub[16];
        const float * sub = pxq_mmvq_stage_sub<type>(s_sub, tid);
        const pxq_mmvq_rowbase<type, rows_per_cuda_block> rb(vx, row0, blocks_per_row_x);
        const int kqs = vdr * (tid % (qi/vdr));
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const uint8_t * slab = rb.slab(kbx);
            pxq_mmvq_scales<pxq_mmvq_pol<type>, rows_per_cuda_block> sc; sc.load(slab, rb.r0);
            if constexpr (HOIST && ncols_y > 1) {
                // row outer: unpack this (row, k-block) once, then run it against every column.
                // Each tmp[j][i] still takes exactly one addend per k-block, in k-block order.
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    pxq_mmvq_frag<type, VDRP, rows_per_cuda_block> w;
                    w.load(slab, rb.r0 + i, i, sub, sc, kqs);
#pragma unroll
                    for (int j = 0; j < ncols_y; ++j) {
                        if constexpr (FAST) {
                            tmp[j][i] += pxq_mmvq_dot_frag_fast<pxq_mmvq_pol<type>, VDRP, rows_per_cuda_block>(w, &y[j*blocks_per_col_y + kby], kqs);
                        } else {
                            tmp[j][i] += pxq_mmvq_dot_f<type, VDRP, rows_per_cuda_block>(w, rb.anch[i], &y[j*blocks_per_col_y + kby], kqs);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
                    for (int i = 0; i < rows_per_cuda_block; ++i) {
                        tmp[j][i] += pxq_mmvq_dot<type, VDRP, rows_per_cuda_block>(slab, rb.r0 + i, i, rb.anch[i], sub, sc, &y[j*blocks_per_col_y + kby], kqs);
                    }
                }
            }
        }
        if constexpr (FAST && HOIST && ncols_y > 1) {
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) tmp[j][i] *= rb.anch[i];
            }
        }
    } else { // braces: MSVC-mode nvcc loses the next declaration after an unbraced else-for
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
            }
        }
    }}

    // one scalar per thread: every thread only ever wrote and read element threadIdx.x of the
    // old array, and the runtime index was enough to put it in local memory (see pxa_lane_pick).
    float local_bias = 0.0f;
    if (bias && threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
        local_bias = bias[row0 + threadIdx.x];
    }
    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            dst[j*nrows_dst + row0 + threadIdx.x] = pxa_lane_pick<rows_per_cuda_block>(tmp[j], threadIdx.x) + local_bias;
        }
    }
}

template <ggml_type type, int ncols_y, int nwarps, int ROWS, int VDRP, bool HOIST = true>
static __device__ void k_fused_mul_mat_vec_q(
    const void * __restrict__ vup, const void * __restrict__ vgate,
    const float * __restrict__ bias_u, const float * __restrict__ bias_g,
    const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, ggml_unary_op unary_op, float limit) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = pxq_mmvq_layout<type> ? VDRP : get_vdr_mmvq(type);

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    //int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < PXA_CC_RDNA2 ?
    //    ncols_y < 4 ? 1 : 2 : 1;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int rows_per_cuda_block = mmvq_rows_per_block<type, ncols_y, ROWS>;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp_u[ncols_y][rows_per_cuda_block] = {0.0f};
    float tmp_g[ncols_y][rows_per_cuda_block] = {0.0f};
    // scalars, not arrays: the runtime index alone would spill both accumulator tiles to local
    // memory (see pxa_lane_pick).
    float local_bias_u = 0.0f;
    float local_bias_g = 0.0f;
    if (bias_u && threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
        local_bias_u = bias_u[row0 + threadIdx.x];
    }
    if (bias_g && threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
        local_bias_g = bias_g[row0 + threadIdx.x];
    }

    const block_q8_1 * y = (const block_q8_1 *) vy;

    if constexpr (pxq_mmvq_layout<type>) {
        __shared__ float s_sub[16];
        const float * sub = pxq_mmvq_stage_sub<type>(s_sub, tid);
        const pxq_mmvq_rowbase<type, rows_per_cuda_block> ru(vup,   row0, blocks_per_row_x);
        const pxq_mmvq_rowbase<type, rows_per_cuda_block> rg(vgate, row0, blocks_per_row_x);
        const int kqs = vdr * (tid % (qi/vdr));
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const uint8_t * su = ru.slab(kbx);
            const uint8_t * sg = rg.slab(kbx);
            pxq_mmvq_scales<pxq_mmvq_pol<type>, rows_per_cuda_block> scu; scu.load(su, ru.r0);
            pxq_mmvq_scales<pxq_mmvq_pol<type>, rows_per_cuda_block> scg; scg.load(sg, rg.r0);
            if constexpr (HOIST && ncols_y > 1) {
                // Up and gate are decoded in turn, not together, so only one fragment is live at
                // a time and the extra registers do not double for the fused twin.
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    {
                        pxq_mmvq_frag<type, VDRP, rows_per_cuda_block> w;
                        w.load(su, ru.r0 + i, i, sub, scu, kqs);
#pragma unroll
                        for (int j = 0; j < ncols_y; ++j) {
                            tmp_u[j][i] += pxq_mmvq_dot_f<type, VDRP, rows_per_cuda_block>(w, ru.anch[i], &y[j*blocks_per_col_y + kby], kqs);
                        }
                    }
                    {
                        pxq_mmvq_frag<type, VDRP, rows_per_cuda_block> w;
                        w.load(sg, rg.r0 + i, i, sub, scg, kqs);
#pragma unroll
                        for (int j = 0; j < ncols_y; ++j) {
                            tmp_g[j][i] += pxq_mmvq_dot_f<type, VDRP, rows_per_cuda_block>(w, rg.anch[i], &y[j*blocks_per_col_y + kby], kqs);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
                    for (int i = 0; i < rows_per_cuda_block; ++i) {
                        tmp_u[j][i] += pxq_mmvq_dot<type, VDRP, rows_per_cuda_block>(su, ru.r0 + i, i, ru.anch[i], sub, scu, &y[j*blocks_per_col_y + kby], kqs);
                        tmp_g[j][i] += pxq_mmvq_dot<type, VDRP, rows_per_cuda_block>(sg, rg.r0 + i, i, rg.anch[i], sub, scg, &y[j*blocks_per_col_y + kby], kqs);
                    }
                }
            }
        }
    } else { // braces: see above
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_u[j][i] += vec_dot_q_cuda(vup  , &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
                tmp_g[j][i] += vec_dot_q_cuda(vgate, &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
            }
        }
    }}

    __shared__ float tmp_shared_u[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    __shared__ float tmp_shared_g[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared_u[threadIdx.y-1][j][i][threadIdx.x] = tmp_u[j][i];
                tmp_shared_g[threadIdx.y-1][j][i][threadIdx.x] = tmp_g[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp_u[j][i] += tmp_shared_u[l][j][i][threadIdx.x];
                tmp_g[j][i] += tmp_shared_g[l][j][i][threadIdx.x];
            }
            tmp_u[j][i] = warp_reduce_sum(tmp_u[j][i]);
            tmp_g[j][i] = warp_reduce_sum(tmp_g[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            float u = pxa_lane_pick<rows_per_cuda_block>(tmp_u[j], threadIdx.x);
            float g = pxa_lane_pick<rows_per_cuda_block>(tmp_g[j], threadIdx.x);
            float r;
            switch (unary_op) {
                case GGML_UNARY_OP_SILU:
                    {
                        g = g/(1 + expf(-g));
                        g = min(g, limit);
                        r = max(-limit, min(limit, u))*g;
                    }break;
                case GGML_UNARY_OP_RELU: r = fmaxf(g, 0.0f) * u; break;
                case GGML_UNARY_OP_GELU: {
                    constexpr float GELU_COEF_A    = 0.044715f;
                    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
                    r = 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
                } break;
                // we assume that the supported ops have been checked by the caller
                default: {
                  constexpr float alpha = 1.702f;
                  constexpr float limit = 7.0f;
                  g += local_bias_g;
                  u += local_bias_u;
                  g = fminf(g, limit);
                  u = fmaxf(fminf(u, limit), -limit);
                  r = g / (1.0f + expf(-g * alpha)) * (1.0f + u);
                } break;
            }
            dst[j*nrows_dst + row0 + threadIdx.x] = r;
        }
    }
}

template <ggml_type type, int ncols_y, int nwarps, int ROWS, int VDRP, bool HOIST = true, bool FAST = false>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const char * __restrict__ ids_data, const void * __restrict__ bias,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1) {

    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    const float * b = (const float *)(bias ? ids_data ? (const char *)bias + i02*bias_nb1 : bias : nullptr);
    k_mul_mat_vec_q<type, ncols_y, nwarps, ROWS, VDRP, HOIST, FAST>(cx, cy, b, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// PXA_PXQN_VERIFY_WIDE (2026-10-14): mul_mat_vec_q with a MIN-BLOCKS launch bound. The body is the same
// k_mul_mat_vec_q instantiation (HOIST, not FAST) -- same per-thread k-block set (kbx = tid/(qi/vdr) + j*blocks_per_iter),
// same per-thread accumulation order, same cross-warp order and butterfly -- so every output is bit-identical to
// mul_mat_vec_q's; only ptxas's register budget changes. Why it exists: under __launch_bounds__(128, 1) the PXQ verify
// widths 6..8 take 141/154/163 registers (122 at 5), i.e. 3 blocks of 128 threads per SM instead of 4, and nvprof puts
// the whole w5 -> w6 verify step there (+27..33% per launch on every 27B shape at 6 columns). MINB 4 caps the kernel at
// 128 registers so 4 blocks stay resident.
template <ggml_type type, int ncols_y, int nwarps, int ROWS, int VDRP, int MINB>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, MINB)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q_mb(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const char * __restrict__ ids_data, const void * __restrict__ bias,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1) {

    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    const float * b = (const float *)(bias ? ids_data ? (const char *)bias + i02*bias_nb1 : bias : nullptr);
    k_mul_mat_vec_q<type, ncols_y, nwarps, ROWS, VDRP, true, false>(cx, cy, b, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// PXA_PXQN_VERIFY_WIDE (2026-10-14): the PXQ multi-column kernel with two bit-identical rewrites of
// k_mul_mat_vec_q's HOIST path, each a template switch so the harness can price them one at a time.
//  I2FX: the dp4a partial s (|s| <= 8*128*128 = 2^17) is converted to float as __int_as_float(s + 0x4B400000) - 1.5*2^23
//        (IADD + FADD, both full rate) instead of I2F (a quarter-rate conversion on sm_70: 16/clk/SM). Exact for |s| < 2^22,
//        so the float is the same float and the following FFMA is the same FFMA -- bit-identical.
//  EPI:  the cross-warp epilogue. k_mul_mat_vec_q hands warps 1..3's partials to warp 0, which alone runs ncols_y x ROWS
//        (up to 32) chains of 3 adds + a 5-step butterfly while the other warps exit. Here every warp stages its partials
//        and pair p = j*ROWS + i is finished by warp p % 4 in the SAME order (((w0 + w1) + w2) + w3, then the same xor
//        butterfly, whose lanes all hold the same sum) -- bit-identical, a quarter of the serial tail.
// Same lane mapping (kqs, kbx = tid/(qi/vdr) + j*blocks_per_iter), same per-thread k order, same dot arithmetic.
static __device__ __forceinline__ float pxa_i2f_exact(const int s) {   // == (float) s for |s| < 2^22
    return __int_as_float(s + 0x4B400000) - 12582912.0f;
}

template <class POL, int VDR, int ROWS, bool I2FX>
static __device__ __forceinline__ float pxq_mmvq_dot_frag_vw(
        const pxq_mmvq_wfrag<POL, VDR, ROWS> & w, const float anch,
        const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int * q8 = (const int *) bq8_1->qs;
    float sum = 0.0f;
#pragma unroll
    for (int l = 0; l < VDR; ++l) {
        const int m = iqs + l;
        int s = ggml_cuda_dp4a(w.v[l].x, q8[2*m + 0], 0);
        s     = ggml_cuda_dp4a(w.v[l].y, q8[2*m + 1], s);
        if constexpr (I2FX) {
            sum  += w.eff[l] * pxa_i2f_exact(s);
        } else {
            sum  += w.eff[l] * (float) s;
        }
    }
    return anch * __low2float(bq8_1->ds) * sum;
}

template <ggml_type type, int ncols_y, int ROWS, int VDRP, bool I2FX, bool EPI>
static __device__ void k_mul_mat_vec_q_vw(
    const void * __restrict__ vx, const void * __restrict__ vy,
    const float * bias, float * __restrict__ dst,
    const int ncols_x, const int nrows_y, const int nrows_dst) {
    static_assert(pxq_mmvq_layout<type> && ncols_y > 1, "PXQ multi-column only");
    constexpr int nwarps = 4;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = VDRP;
    constexpr int rows_per_cuda_block = mmvq_rows_per_block_plain<type, ncols_y, ROWS>;

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};
    const block_q8_1 * y = (const block_q8_1 *) vy;

    __shared__ float s_sub[16];
    const float * sub = pxq_mmvq_stage_sub<type>(s_sub, tid);
    const pxq_mmvq_rowbase<type, rows_per_cuda_block> rb(vx, row0, blocks_per_row_x);
    const int kqs = vdr * (tid % (qi/vdr));
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const uint8_t * slab = rb.slab(kbx);
        pxq_mmvq_scales<pxq_mmvq_pol<type>, rows_per_cuda_block> sc; sc.load(slab, rb.r0);
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            pxq_mmvq_frag<type, VDRP, rows_per_cuda_block> w;
            w.load(slab, rb.r0 + i, i, sub, sc, kqs);
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
                tmp[j][i] += pxq_mmvq_dot_frag_vw<pxq_mmvq_pol<type>, VDRP, rows_per_cuda_block, I2FX>(
                        w, rb.anch[i], &y[j*blocks_per_col_y + kby], kqs);
            }
        }
    }

    if constexpr (EPI) {
        constexpr int NP = ncols_y*rows_per_cuda_block;
        __shared__ float part[nwarps][NP][WARP_SIZE];
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                part[threadIdx.y][j*rows_per_cuda_block + i][threadIdx.x] = tmp[j][i];
            }
        }
        __syncthreads();
        for (int p = threadIdx.y; p < NP; p += nwarps) {
            float v = part[0][p][threadIdx.x];
#pragma unroll
            for (int l = 1; l < nwarps; ++l) {
                v += part[l][p][threadIdx.x];
            }
            v = warp_reduce_sum(v);
            const int j = p / rows_per_cuda_block, i = p % rows_per_cuda_block;
            if (threadIdx.x == 0 && row0 + i < nrows_dst) {
                dst[j*nrows_dst + row0 + i] = v + (bias ? bias[row0 + i] : 0.0f);
            }
        }
    } else {
        float local_bias = 0.0f;
        if (bias && threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
            local_bias = bias[row0 + threadIdx.x];
        }
        __shared__ float tmp_shared[nwarps-1][ncols_y][rows_per_cuda_block][WARP_SIZE];
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                }
                tmp[j][i] = warp_reduce_sum(tmp[j][i]);
            }
            if (threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
                dst[j*nrows_dst + row0 + threadIdx.x] = pxa_lane_pick<rows_per_cuda_block>(tmp[j], threadIdx.x) + local_bias;
            }
        }
    }
}

// PXA_PXQN_VERIFY_WIDE column split (2026-10-14). The block carries TWO full sets of k_mul_mat_vec_q's 128
// threads (blockDim 32 x 8): warps 0..3 own activation columns [0, NC0), warps 4..7 own [NC0, ncols_y). Inside each set,
// thread t = 32*(warp & 3) + lane is k_mul_mat_vec_q's thread t EXACTLY -- same k-blocks (kbx = t/(qi/vdr) + j*64), same
// half (kqs), same per-k-block dot, same accumulation order -- and each set reduces its own columns with k_mul_mat_vec_q's
// cross-warp order and butterfly, so every output is bit-identical to the one-set kernel (and to the 1-column launch).
// What changes is the per-thread footprint: a thread carries ROWS x ceil(ncols_y/2) accumulators and activation
// fragments instead of ROWS x ncols_y (at 8 columns: the 4-column kernel's ~112 registers instead of 163), so the SM
// keeps 16 warps in flight instead of 12; the activation reads are unchanged (each column read once per block) and the
// weight fragment is decoded by both sets (the second read is an L1 hit: same block, same addresses).
template <ggml_type type, int NC, int ROWS, int VDRP>
static __device__ __forceinline__ void pxq_mmvq_cs_body(
        const pxq_mmvq_rowbase<type, ROWS> & rb, const float * __restrict__ sub, const block_q8_1 * __restrict__ y,
        const int blocks_per_row_x, const int blocks_per_col_y, const int t, float (&tmp)[NC][ROWS]) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = VDRP;
    constexpr int blocks_per_iter = vdr * 4*WARP_SIZE / qi;          // k_mul_mat_vec_q at nwarps 4
    const int kqs = vdr * (t % (qi/vdr));
    for (int kbx = t / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const uint8_t * slab = rb.slab(kbx);
        pxq_mmvq_scales<pxq_mmvq_pol<type>, ROWS> sc; sc.load(slab, rb.r0);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            pxq_mmvq_frag<type, VDRP, ROWS> w;
            w.load(slab, rb.r0 + i, i, sub, sc, kqs);
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                tmp[j][i] += pxq_mmvq_dot_f<type, VDRP, ROWS>(w, rb.anch[i], &y[j*blocks_per_col_y + kby], kqs);
            }
        }
    }
}

// every warp stages its partials (no accumulator array lives across the barrier, so a thread's register footprint stays
// that of ITS column set); after the barrier warp 0 of the set finishes each (column, row) in k_mul_mat_vec_q's order:
// ((w0 + w1) + w2) + w3, then the same xor butterfly (every lane ends with the same sum).
template <int NC, int ROWS>
static __device__ __forceinline__ void pxq_mmvq_cs_stage(const float (&tmp)[NC][ROWS], float (*sh)[NC][ROWS][WARP_SIZE], const int wy) {
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            sh[wy][j][i][threadIdx.x] = tmp[j][i];
        }
    }
}
template <int NC, int ROWS>
static __device__ __forceinline__ void pxq_mmvq_cs_reduce(float (*sh)[NC][ROWS][WARP_SIZE], float * __restrict__ dst,
        const int col0, const int row0, const int nrows_dst, const float local_bias) {
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            float v = sh[0][j][i][threadIdx.x];
#pragma unroll
            for (int l = 1; l < 4; ++l) {
                v += sh[l][j][i][threadIdx.x];
            }
            v = warp_reduce_sum(v);
            if (threadIdx.x == i && row0 + i < nrows_dst) {
                dst[(col0 + j)*nrows_dst + row0 + i] = v + local_bias;
            }
        }
    }
}

template <ggml_type type, int ncols_y, int ROWS, int VDRP>
static __device__ void k_mul_mat_vec_q_cs(
    const void * __restrict__ vx, const void * __restrict__ vy,
    const float * bias, float * __restrict__ dst,
    const int ncols_x, const int nrows_y, const int nrows_dst) {
    static_assert(pxq_mmvq_layout<type> && ncols_y >= 2, "PXQ multi-column only");
    constexpr int NC0 = (ncols_y + 1)/2, NC1 = ncols_y/2;
    constexpr int R = ROWS;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int g   = threadIdx.y >> 2;                 // column set
    const int wy  = threadIdx.y & 3;                  // warp inside the set
    const int t   = WARP_SIZE*wy + threadIdx.x;       // k_mul_mat_vec_q's thread id
    const int row0 = R*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;

    __shared__ float s_sub[16];
    const float * sub = pxq_mmvq_stage_sub<type>(s_sub, tid);
    const pxq_mmvq_rowbase<type, R> rb(vx, row0, blocks_per_row_x);
    const block_q8_1 * y = (const block_q8_1 *) vy;

    float local_bias = 0.0f;
    if (bias && wy == 0 && threadIdx.x < R && row0 + threadIdx.x < nrows_dst) {
        local_bias = bias[row0 + threadIdx.x];
    }
    __shared__ float sh0[4][NC0][R][WARP_SIZE];
    __shared__ float sh1[4][NC1][R][WARP_SIZE];
    if (g == 0) {
        float tmp[NC0][R] = {0.0f};
        pxq_mmvq_cs_body<type, NC0, R, VDRP>(rb, sub, y, blocks_per_row_x, blocks_per_col_y, t, tmp);
        pxq_mmvq_cs_stage<NC0, R>(tmp, sh0, wy);
    } else {
        float tmp[NC1][R] = {0.0f};
        pxq_mmvq_cs_body<type, NC1, R, VDRP>(rb, sub, y + NC0*blocks_per_col_y, blocks_per_row_x, blocks_per_col_y, t, tmp);
        pxq_mmvq_cs_stage<NC1, R>(tmp, sh1, wy);
    }
    __syncthreads();
    if (wy != 0) {
        return;
    }
    if (g == 0) {
        pxq_mmvq_cs_reduce<NC0, R>(sh0, dst, 0,   row0, nrows_dst, local_bias);
    } else {
        pxq_mmvq_cs_reduce<NC1, R>(sh1, dst, NC0, row0, nrows_dst, local_bias);
    }
}

template <ggml_type type, int ncols_y, int ROWS, int VDRP, int MINB>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(8*WARP_SIZE, MINB)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q_cs(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const char * __restrict__ ids_data, const void * __restrict__ bias,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1) {
    GGML_UNUSED(nrows_x);
    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    const float * b = (const float *)(bias ? ids_data ? (const char *)bias + i02*bias_nb1 : bias : nullptr);
    k_mul_mat_vec_q_cs<type, ncols_y, ROWS, VDRP>(cx, cy, b, (float *)cdst, ncols_x, nrows_y, nrows_dst);
}

template <ggml_type type, int ncols_y, int ROWS, int VDRP, int MINB, bool I2FX, bool EPI>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(4*WARP_SIZE, MINB)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q_vw(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const char * __restrict__ ids_data, const void * __restrict__ bias,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1) {
    GGML_UNUSED(nrows_x);
    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    const float * b = (const float *)(bias ? ids_data ? (const char *)bias + i02*bias_nb1 : bias : nullptr);
    k_mul_mat_vec_q_vw<type, ncols_y, ROWS, VDRP, I2FX, EPI>(cx, cy, b, (float *)cdst, ncols_x, nrows_y, nrows_dst);
}

template <ggml_type type, int ncols_y, int nwarps, int ROWS, int VDRP, bool HOIST = true>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void fused_mul_mat_vec_q(
    const void * __restrict__ vup, const void * __restrict__ vgate,
    const void * __restrict__ vy, float * __restrict__ dst, const char * __restrict__ ids_data,
    const void * __restrict__ bias_u, const void * __restrict__ bias_g, const uint64_t bias_nb1,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, ggml_unary_op unary_op, float limit) {

    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        return;
    }
    const char * cx_u = (const char *)vup   + i02*nb02;
    const char * cx_g = (const char *)vgate + i02*nb02;
    const float * cx_u_b = bias_u ? (const float *)((const char *)bias_u + i02*bias_nb1) : nullptr;
    const float * cx_g_b = bias_g ? (const float *)((const char *)bias_g + i02*bias_nb1) : nullptr;
    const char * cy = (const char *)vy + i2*nb12;
    k_fused_mul_mat_vec_q<type, ncols_y, nwarps, ROWS, VDRP, HOIST>(cx_u, cx_g, cx_u_b, cx_g_b, cy, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst,
            unary_op, limit);
}

// K1 SINGLE-COLUMN PIPELINED PXQ MMVQ (PXA_PXQ_MMVQ_PF, 2026-10-14; see pxa_pxq_mmvq_pf()).
// The default single-column tile (4 rows, 4 warps, VDR 2) of k_mul_mat_vec_q with the next k-block's code words
// and scale word issued before this k-block's math, in a lean body (no ids / bias / multi-column plumbing) that
// the 64-register launch bound (8 blocks/SM) fits without a spill. Lane mapping (kqs, kbx = tid/2 + 64 j), the
// per-thread k order, the arithmetic (vec_dot_pxq_q8_1_qw, the one body vec_dot_pxq_q8_1 also runs), the
// cross-warp order and the warp reduction are k_mul_mat_vec_q's: bit-identical to it. PXQ rows come in 64-row
// panels, so a 4-row block never straddles a matrix end.
template <ggml_type type>
static __device__ __forceinline__ void k_pxq_mmvq1_pf(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const int ncols_x, const int row0) {
    static_assert(type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQN4, "K1 PF body: the p6 slab tiers only");
    using POL = pxq_mmvq_pol<type>;
    constexpr int NW = 4, ROWS = 4, qk = 32, qi = 4, vdr = 2, TPR = qi/vdr, BPI = vdr*NW*WARP_SIZE/qi;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int nkb = ncols_x / qk;
    float tmp[ROWS] = {0.0f};
    const block_q8_1 * y = (const block_q8_1 *) vy;
    __shared__ float s_sub[16];
    const float * sub = pxq_mmvq_stage_sub<type>(s_sub, tid);
    const pxq_mmvq_rowbase<type, ROWS> rb(vx, row0, nkb);
    const int kqs = vdr * (tid % TPR);
    int kbx = tid / TPR;
    uint32_t qw[ROWS][2];
    pxq_mmvq_scales<POL, ROWS> sc;
    if (kbx < nkb) {
        const uint8_t * slab = rb.slab(kbx);
        sc.load(slab, rb.r0);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) POL::template load_words<2>(slab, rb.r0 + i, kqs, qw[i]);
    }
    for (; kbx < nkb; kbx += BPI) {
        uint32_t qn[ROWS][2];
        pxq_mmvq_scales<POL, ROWS> scn;
        const int kn = kbx + BPI;
        if (kn < nkb) {
            const uint8_t * slab = rb.slab(kn);
            scn.load(slab, rb.r0);
#pragma unroll
            for (int i = 0; i < ROWS; ++i) POL::template load_words<2>(slab, rb.r0 + i, kqs, qn[i]);
        }
#pragma unroll
        for (int i = 0; i < ROWS; ++i) tmp[i] += vec_dot_pxq_q8_1_qw<POL, 2, ROWS>(qw[i], i, rb.anch[i], sub, sc, &y[kbx], kqs);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) { qw[i][0] = qn[i][0]; qw[i][1] = qn[i][1]; }
#pragma unroll
        for (int w = 0; w < (int) (sizeof(sc.w)/sizeof(sc.w[0])); ++w) sc.w[w] = scn.w[w];
    }
    __shared__ float tmp_shared[NW-1][ROWS][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < NW-1; ++l) tmp[i] += tmp_shared[l][i][threadIdx.x];
        tmp[i] = warp_reduce_sum(tmp[i]);
    }
    if (threadIdx.x < ROWS) {
        dst[row0 + threadIdx.x] = pxa_lane_pick<ROWS>(tmp, threadIdx.x) + 0.0f;   // + the no-bias local_bias
    }
}

template <ggml_type type>
__launch_bounds__(4*WARP_SIZE, 8)
static __global__ void mul_mat_vec_q_pf1(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const int ncols_x) {
    k_pxq_mmvq1_pf<type>(vx, vy, dst, ncols_x, 4*(int) blockIdx.x);
}

// GROUPED DECODE GEMV (PXA_MMVQ_GROUP, sm_70). Up to MMVQ_GROUP_MAX matrices of one type that read the
// same q8_1 activation (the DeltaNet qkv / z / alpha / beta chain, attention q / k / v) in ONE launch: the
// grid is the concatenation of each matrix's own row blocks and a block runs the unchanged single-matrix
// body on its matrix with its local block index. Every row is computed by exactly the code, lane mapping
// and reduction of its own launch, so the result is bit-identical to one launch per matrix; what goes is
// the per-launch floor of the small members (the 24-row alpha/beta and the 512-row k/v heads cost a full
// ~4-7 us launch each on a V100 for well under a microsecond of bytes) and the tail of each wave.
template <ggml_type type, int nwarps, int ROWS, int VDRP>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_group(const mmvq_group_args g) {
    const int b = blockIdx.x;
    int m = 0, b0 = 0;
#pragma unroll
    for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
        if (i < g.n && b >= g.blk_end[i-1]) { m = i; b0 = g.blk_end[i-1]; }
    }
    const void * vx = g.vx[0]; float * dst = g.dst[0]; int nr = g.nrows[0];
#pragma unroll
    for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
        if (m == i) { vx = g.vx[i]; dst = g.dst[i]; nr = g.nrows[i]; }
    }
    k_mul_mat_vec_q<type, 1, nwarps, ROWS, VDRP, true>(vx, g.vy, nullptr, dst, g.ncols_x, nr, g.nrows_y, nr, b - b0);
}

// K1 twin of mul_mat_vec_q_group: the same block -> matrix mapping, the pipelined single-column body.
template <ggml_type type>
__launch_bounds__(4*WARP_SIZE, 8)
static __global__ void mul_mat_vec_q_group_pf1(const mmvq_group_args g) {
    const int b = blockIdx.x;
    int m = 0, b0 = 0;
#pragma unroll
    for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
        if (i < g.n && b >= g.blk_end[i-1]) { m = i; b0 = g.blk_end[i-1]; }
    }
    const void * vx = g.vx[0]; float * dst = g.dst[0];
#pragma unroll
    for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
        if (m == i) { vx = g.vx[i]; dst = g.dst[i]; }
    }
    k_pxq_mmvq1_pf<type>(vx, g.vy, dst, g.ncols_x, 4*(b - b0));
}

// SM_70 ROW TILE FOR Q6_K (PXA_MMVQ_V100_ROWS). The single-column Q6_K block is 4 warps on ONE row: each
// thread issues a few narrow loads per 256-wide super-block and the SM runs out of loads in flight before
// it runs out of bandwidth. Two or four rows per block give every thread that many independent super-blocks
// per step. Each row keeps its own accumulator, lane mapping and warp reduction, so the output is
// bit-identical to the 1-row block. Measured on a V100 (standalone, ny = 1, weight bytes per second):
// 5120x5120 622 -> 674 GB/s (r2), 5120x6144 642 -> 699 (r4), 8704x5120 676 -> 717 (r2), 3072x5120
// 531 -> 611 (r4), the 248320-row head 713 -> 820 (r4); a 512-row matrix is flat and a 24-row one loses
// (r1 kept below 2048 rows); the fused up/gate twin is flat (752 vs 747) and keeps r1. The rule is
// cc == 700 only: sm_60 and every other arch launch exactly what they did before.
//   PXA_MMVQ_V100_ROWS  auto (default) | 0 or 1 = off | 2 | 4 (forced; still sm_70 and ny = 1 only)
static inline int mmvq_v100_rows_env() {
    static const int v = [](){
        const char * e = getenv("PXA_MMVQ_V100_ROWS");
        if (!e || !*e || !strcmp(e, "auto")) return -1;
        const int r = atoi(e);
        return r == 2 || r == 4 ? r : 1;
    }();
    return v;
}
// PXA_MMVQ_V100_ROWS_NY (ticket gemma-mtp-verify-kernel, 2026-10-02; default OFF): the same sm_70 Q6_K row
// tile at 2..3 columns, for a big (>= 65536-row) matrix only -- the LM head of an MTP verify. Without it
// the 2-column head runs the stock 1-row block: measured on a V100 (Gemma-4 262144 x 2816 Q6_K head)
// 1146 us at ny = 2 against 729 us at ny = 1 with the 4-row tile. Each row keeps its own accumulator,
// lane mapping and warp reduction (as at ny = 1), so the result is bit-identical.
//   PXA_MMVQ_V100_ROWS_NY  unset/0 = off | 2 | 4 (any other value = 4)
static inline int mmvq_v100_rows_ny_env() {
    static const int v = [](){
        const char * e = getenv("PXA_MMVQ_V100_ROWS_NY");
        if (!e || !*e) return 0;
        const int r = atoi(e);
        return r == 0 ? 0 : (r == 2 ? 2 : 4);
    }();
    return v;
}
template <ggml_type type>
static inline int mmvq_v100_rows(int cc, int ncols_y, int nrows, int ne2, bool fused) {
    if constexpr (type != GGML_TYPE_Q6_K) {
        return 1;
    } else {
        if (cc == PXA_CC_VOLTA && (ncols_y == 2 || ncols_y == 3) && ne2 <= 1 && !fused && nrows >= 65536) {
            const int r = mmvq_v100_rows_ny_env();
            return (r == 2 || r == 4) && nrows % r == 0 ? r : 1;
        }
        if (cc != PXA_CC_VOLTA || ncols_y != 1 || ne2 > 1 || fused) return 1;
        const int e = mmvq_v100_rows_env();
        const int r = e >= 0 ? e : (nrows >= 65536 ? 4 : (nrows >= 2048 ? 2 : 1));
        return (r == 2 || r == 4) && nrows % r == 0 ? r : 1;
    }
}

template <ggml_type type, int ROWS>
static void mul_mat_vec_q_cuda_rows1(const mmvq_args & args, cudaStream_t stream) {
    const dim3 block_nums((args.nrows_x + ROWS - 1)/ROWS, args.ne2, 1);
    const dim3 block_dims(WARP_SIZE, 4, 1);
    switch (args.ncols_y) {   // 2..3 only via PXA_MMVQ_V100_ROWS_NY
        case 2:
            mul_mat_vec_q<type, 2, 4, ROWS, 2><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            return;
        case 3:
            mul_mat_vec_q<type, 3, 4, ROWS, 2><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            return;
        default: break;
    }
    mul_mat_vec_q<type, 1, 4, ROWS, 2><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
            args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
}

template <ggml_type type, int ROWS, int VDRP>
static void mul_mat_vec_q_group_launch(mmvq_group_args & g, cudaStream_t stream) {
    constexpr int R = mmvq_rows_per_block<type, 1, ROWS>;
    int end = 0;
    for (int i = 0; i < g.n; ++i) { end += (g.nrows[i] + R - 1)/R; g.blk_end[i] = end; }
    for (int i = g.n; i < MMVQ_GROUP_MAX; ++i) { g.blk_end[i] = end; }
    if constexpr ((type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQN4) && ROWS == 4 && VDRP == 2) {
        if (pxa_pxq_mmvq_pf()) {   // K1: the pipelined single-column body (bit-identical; PXQ rows come in 64-row panels)
            mul_mat_vec_q_group_pf1<type><<<end, dim3(WARP_SIZE, 4, 1), 0, stream>>>(g);
            return;
        }
    }
    mul_mat_vec_q_group<type, 4, ROWS, VDRP><<<end, dim3(WARP_SIZE, 4, 1), 0, stream>>>(g);
}

// PXA_MMVQ_MIXGRP (2026-10-19): a PXQ-layout group on the pipelined one-column body (ga, blocks
// [0, ga.blk_end[ga.n-1])) and a stock-type group (gb, the rest: e.g. the Q8_0 DeltaNet alpha/beta that read the same q8_1
// activation right after the PXQN4 qkv/z) in ONE launch. Each block runs exactly the body and the local block index its own
// group launch gives it (mul_mat_vec_q_group_pf1 / mul_mat_vec_q_group): bit-identical, minus one launch.
template <ggml_type TA, ggml_type TB, int ROWSB>
__launch_bounds__(4*WARP_SIZE, 8)
static __global__ void mul_mat_vec_q_group_mix(const mmvq_group_args ga, const mmvq_group_args gb) {
    const int na = ga.blk_end[ga.n - 1];
    if ((int) blockIdx.x < na) {
        const int b = blockIdx.x;
        int m = 0, b0 = 0;
#pragma unroll
        for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
            if (i < ga.n && b >= ga.blk_end[i-1]) { m = i; b0 = ga.blk_end[i-1]; }
        }
        const void * vx = ga.vx[0]; float * dst = ga.dst[0];
#pragma unroll
        for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
            if (m == i) { vx = ga.vx[i]; dst = ga.dst[i]; }
        }
        k_pxq_mmvq1_pf<TA>(vx, ga.vy, dst, ga.ncols_x, 4*(b - b0));
    } else {
        const int b = (int) blockIdx.x - na;
        int m = 0, b0 = 0;
#pragma unroll
        for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
            if (i < gb.n && b >= gb.blk_end[i-1]) { m = i; b0 = gb.blk_end[i-1]; }
        }
        const void * vx = gb.vx[0]; float * dst = gb.dst[0]; int nr = gb.nrows[0];
#pragma unroll
        for (int i = 1; i < MMVQ_GROUP_MAX; ++i) {
            if (m == i) { vx = gb.vx[i]; dst = gb.dst[i]; nr = gb.nrows[i]; }
        }
        k_mul_mat_vec_q<TB, 1, 4, ROWSB, 2, true>(vx, gb.vy, nullptr, dst, gb.ncols_x, nr, gb.nrows_y, nr, b - b0);
    }
}

// true = ga (a PXQ-layout type TA that would take the pf1 group body) and gb (stock type TB, same q8_1 activation) ran as one launch
template <ggml_type TA, ggml_type TB>
static bool mul_mat_vec_q_group_mix_cuda(mmvq_group_args & ga, mmvq_group_args & gb, cudaStream_t stream) {
    if constexpr (!pxq_mmvq_layout<TA> || pxq_mmvq_layout<TB>) {
        GGML_UNUSED(ga); GGML_UNUSED(gb); GGML_UNUSED(stream);
        return false;
    } else {
        if (ga.n < 1 || ga.n > MMVQ_GROUP_MAX || gb.n < 1 || gb.n > MMVQ_GROUP_MAX || ga.vy != gb.vy) return false;
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        // ga: only the shape mul_mat_vec_q_group_cuda<TA> launches on the pf1 body
        if (pxa_pxq_mmvq_vdr() != 2 || pxa_pxq_mmvq_tile(1, false, cc) != 4 || !pxa_pxq_mmvq_pf()) return false;
        constexpr int RA = mmvq_rows_per_block<TA, 1, 4>;
        int end = 0;
        for (int i = 0; i < ga.n; ++i) { end += (ga.nrows[i] + RA - 1)/RA; ga.blk_end[i] = end; }
        for (int i = ga.n; i < MMVQ_GROUP_MAX; ++i) { ga.blk_end[i] = end; }
        // gb: the tile mul_mat_vec_q_group_cuda<TB> picks
        int big = 0;
        for (int i = 0; i < gb.n; ++i) big = gb.nrows[i] > big ? gb.nrows[i] : big;
        int r = mmvq_v100_rows<TB>(cc, 1, big, 1, false);
        for (int i = 0; i < gb.n; ++i) if (gb.nrows[i] % r) r = 1;
        auto fill_b = [&](int R) {
            int e = 0;
            for (int i = 0; i < gb.n; ++i) { e += (gb.nrows[i] + R - 1)/R; gb.blk_end[i] = e; }
            for (int i = gb.n; i < MMVQ_GROUP_MAX; ++i) { gb.blk_end[i] = e; }
            return e;
        };
        switch (r) {
            case 4:  { const int eb = fill_b(mmvq_rows_per_block<TB, 1, 4>); mul_mat_vec_q_group_mix<TA, TB, 4><<<end + eb, dim3(WARP_SIZE, 4, 1), 0, stream>>>(ga, gb); break; }
            case 2:  { const int eb = fill_b(mmvq_rows_per_block<TB, 1, 2>); mul_mat_vec_q_group_mix<TA, TB, 2><<<end + eb, dim3(WARP_SIZE, 4, 1), 0, stream>>>(ga, gb); break; }
            default: { const int eb = fill_b(mmvq_rows_per_block<TB, 1, 1>); mul_mat_vec_q_group_mix<TA, TB, 1><<<end + eb, dim3(WARP_SIZE, 4, 1), 0, stream>>>(ga, gb); break; }
        }
        return true;
    }
}

// one launch for g.n matrices at ny = 1 with nwarps 4 (the ny = 1 shape of mul_mat_vec_q_cuda);
// false = not served here, the caller launches them one by one
template <ggml_type type>
static bool mul_mat_vec_q_group_cuda(mmvq_group_args & g, cudaStream_t stream) {
    if (g.n < 1 || g.n > MMVQ_GROUP_MAX) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if constexpr (pxq_mmvq_layout<type>) {
        // the tile / code width the single-matrix launch picks at ny = 1; only the shipped default is instanced
        if (pxa_pxq_mmvq_vdr() != 2 || pxa_pxq_mmvq_tile(1, false, cc) != 4) return false;
        mul_mat_vec_q_group_launch<type, 4, 2>(g, stream);
        return true;
    } else {
        int big = 0;
        for (int i = 0; i < g.n; ++i) big = g.nrows[i] > big ? g.nrows[i] : big;
        int r = mmvq_v100_rows<type>(cc, 1, big, 1, false);
        for (int i = 0; i < g.n; ++i) if (g.nrows[i] % r) r = 1;
        switch (r) {
            case 4:  mul_mat_vec_q_group_launch<type, 4, 2>(g, stream); break;
            case 2:  mul_mat_vec_q_group_launch<type, 2, 2>(g, stream); break;
            default: mul_mat_vec_q_group_launch<type, 1, 2>(g, stream); break;
        }
        return true;
    }
}

// PXA_SPEC_FAST_VERIFY: PXQN4 plain decode GEMV at 2..8 columns through the FAST instantiation
template <ggml_type type, int VDRP>
static constexpr bool pxq_mmvq_fast_ok() { return type == GGML_TYPE_PXQN4 && VDRP % 2 == 0; }
template <ggml_type type, int nwarps, int ROWS, int VDRP>
static void mul_mat_vec_q_fast_launch(const mmvq_args & args, const dim3 block_nums, const dim3 block_dims, cudaStream_t stream) {
    if constexpr (pxq_mmvq_fast_ok<type, VDRP>()) {
#define PXA_MMVQ_FAST_CASE(N) case N: mul_mat_vec_q<type, N, nwarps, ROWS, VDRP, true, true><<<block_nums, block_dims, 0, stream>>>( \
            args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, \
            args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1); break;
        switch (args.ncols_y) {
            PXA_MMVQ_FAST_CASE(2) PXA_MMVQ_FAST_CASE(3) PXA_MMVQ_FAST_CASE(4) PXA_MMVQ_FAST_CASE(5)
            PXA_MMVQ_FAST_CASE(6) PXA_MMVQ_FAST_CASE(7) PXA_MMVQ_FAST_CASE(8)
            default: GGML_ABORT("fatal error");
        }
#undef PXA_MMVQ_FAST_CASE
    } else {
        GGML_UNUSED(args); GGML_UNUSED(block_nums); GGML_UNUSED(block_dims); GGML_UNUSED(stream);
        GGML_ABORT("fatal error");
    }
}

template <ggml_type type, int nwarps, int ROWS, int VDRP>
static void mul_mat_vec_q_cuda_R(const mmvq_args & args, cudaStream_t stream) {

    GGML_ASSERT(args.ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(args.ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < PXA_CC_RDNA2 ?
        args.ncols_y < 4 ? 1 : 2 : 1;
    if constexpr (pxq_mmvq_layout<type>) {                  // must match mmvq_rows_per_block(_plain)
        const bool fused_launch = args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT;
        const int cap = args.ncols_y <= 2 ? 16 : (args.ncols_y <= 4 || !fused_launch ? 8 : 4);
        rows_per_cuda_block = ROWS < cap ? ROWS : cap;
    }

    const int64_t nblocks = (args.nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, args.ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT) {
    switch (args.ncols_y) {
        case 1:
            fused_mul_mat_vec_q<type, 1, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 2:
            fused_mul_mat_vec_q<type, 2, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 3:
            fused_mul_mat_vec_q<type, 3, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 4:
            fused_mul_mat_vec_q<type, 4, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 5:
            fused_mul_mat_vec_q<type, 5, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 6:
            fused_mul_mat_vec_q<type, 6, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 7:
            fused_mul_mat_vec_q<type, 7, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 8:
            fused_mul_mat_vec_q<type, 8, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
    } else if (pxq_mmvq_fast_ok<type, VDRP>() && args.ncols_y >= 2 && !args.ids_data && pxa_spec_fast_verify(ggml_cuda_info().devices[id].cc)) {
        mul_mat_vec_q_fast_launch<type, nwarps, ROWS, VDRP>(args, block_nums, block_dims, stream);
    } else {
    switch (args.ncols_y) {
        case 1:
            if constexpr ((type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQN4) && nwarps == 4 && ROWS == 4 && VDRP == 2) {
                // K1: the pipelined single-column body (bit-identical) for the plain one-matrix, no-bias, no-ids launch
                if (pxa_pxq_mmvq_pf() && args.ne2 == 1 && !args.ids_data && !args.bias_u && args.nrows_x % 64 == 0 && args.nrows_dst >= args.nrows_x) {
                    mul_mat_vec_q_pf1<type><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ncols_x);
                    break;
                }
            }
            mul_mat_vec_q<type, 1, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 2:
            mul_mat_vec_q<type, 2, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 3:
            mul_mat_vec_q<type, 3, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 4:
            mul_mat_vec_q<type, 4, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 5:
            mul_mat_vec_q<type, 5, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 6:
            mul_mat_vec_q<type, 6, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 7:
            mul_mat_vec_q<type, 7, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 8:
            mul_mat_vec_q<type, 8, nwarps, ROWS, VDRP><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
    }
}

// PXQ tile-height dispatch. A PXQ k-block is a 16 B island inside a 64-row slab, so the stock
// 1-row block reads half of every 32 B sector it touches and the per-row scale bytes at ~3%;
// widening the block to R adjacent rows turns both into contiguous runs. R is bounded by the
// register file (measured: R=16 puts the plain kernel at 231 regs and makes the fused twin spill
// 384 B/thread, vs MXFP4's 40/59), hence the sweep.
// PXA_PXQN_VERIFY_WIDE (2026-10-14; unset = ON at sm_70, 0 = off, 1 = on at sm_70): the PXQN4 q8_1 MMVQ at
// the MTP verify widths where the shipped launch loses occupancy to registers, through mul_mat_vec_q_mb -- the SAME
// k_mul_mat_vec_q body with a min-blocks launch bound (and, at 6 columns, the plain kernel's 8-row tile), so every
// output is bit-identical (tests/test-pxqn-verify-wide.cu: 390/390 + 425/425 bitwise vs the shipped launch and vs the
// 1-column launch). Measured per 27B verify step (sum of the 5 verify GEMV shapes x their per-step counts, V100):
// 4 cols R4/MB4 -4.5%, 6 cols R8/MB3 -8.1%, 7 cols R4/MB4 -11.6%, 8 cols R4/MB4 -9.9%; 2, 3 and 5 columns keep the
// shipped launch (every candidate measured slower there). Not taken: ids / batched (ne2 > 1) / fused launches,
// PXA_SPEC_FAST_VERIFY, VDR 4, any card but sm_70 (the only one measured).
static inline int pxa_pxqn_verify_wide_env() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQN_VERIFY_WIDE");
        return e ? (atoi(e) != 0 ? 1 : 0) : -1;
    }();
    return v;
}
template <ggml_type type, int nwarps>
static bool pxa_pxqn_verify_wide_try(const mmvq_args & args, cudaStream_t stream, const int cc) {
    if constexpr (type != GGML_TYPE_PXQN4 || nwarps != 4) {
        GGML_UNUSED(args); GGML_UNUSED(stream); GGML_UNUSED(cc);
        return false;
    } else {
        if (cc != PXA_CC_VOLTA || pxa_pxqn_verify_wide_env() == 0) return false;
        const int ny = args.ncols_y;
        if (ny != 4 && ny != 6 && ny != 7 && ny != 8) return false;
        if (args.ne2 != 1 || args.ids_data || (args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT)) return false;
        if (pxa_pxq_mmvq_vdr() != 2 || pxa_spec_fast_verify(cc) || args.nrows_x % 64 != 0) return false;
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) {
            fprintf(stderr, "PXA_PXQN_VERIFY_WIDE: PXQN4 verify GEMVs at 4/6/7/8 columns on the min-blocks launch (bit-identical; =0 restores the old launch)\n");
        }
        const dim3 blk(WARP_SIZE, 4, 1);
#define PXA_VW_L(NY, R, MB) mul_mat_vec_q_mb<type, NY, 4, R, 2, MB><<<dim3((args.nrows_x + R - 1)/R, 1, 1), blk, 0, stream>>>( \
            args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, \
            args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1)
        switch (ny) {
            case 4:  PXA_VW_L(4, 4, 4); break;
            case 6:  PXA_VW_L(6, 8, 3); break;
            case 7:  PXA_VW_L(7, 4, 4); break;
            default: PXA_VW_L(8, 4, 4); break;
        }
#undef PXA_VW_L
        return true;
    }
}

// PXA_PXQN_VWPF (2026-10-16; unset = ON at sm_70, 0 = off): the PXQN4 q8_1 MMVQ at the verify widths
// 3/6/7 (where it measured faster) on the K1 pipelined body. Kernel, gate and per-width tile live in pxa/pxq-mmvq-pft.cuh, compiled into the PXQN4
// instance TU only (pxa/template-instances/mmvq-instance-pxqn4.cu defines this): true = launched there.
bool pxa_pxqn_vwpf_launch(const mmvq_args & args, cudaStream_t stream);

template <ggml_type type, int nwarps>
static void mul_mat_vec_q_cuda_T(const mmvq_args & args, cudaStream_t stream) {
    if constexpr (pxq_mmvq_layout<type>) {
        if constexpr (type == GGML_TYPE_PXQN4 && nwarps == 4) {
            if (pxa_pxqn_vwpf_launch(args, stream)) {
                return;
            }
        }
        if (pxa_pxqn_verify_wide_try<type, nwarps>(args, stream, ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
            return;
        }
        const int r = pxa_pxq_mmvq_tile(args.ncols_y,
                args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT,
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
        if (pxa_pxq_mmvq_vdr() == 4) {
            switch (r) {
                case 1:  mul_mat_vec_q_cuda_R<type, nwarps, 1, 4>(args, stream); return;
                case 2:  mul_mat_vec_q_cuda_R<type, nwarps, 2, 4>(args, stream); return;
                case 8:  mul_mat_vec_q_cuda_R<type, nwarps, 8, 4>(args, stream); return;
                default: mul_mat_vec_q_cuda_R<type, nwarps, 4, 4>(args, stream); return;
            }
        }
        switch (r) {
            case 1:  mul_mat_vec_q_cuda_R<type, nwarps, 1 , 2>(args, stream); return;
            case 2:  mul_mat_vec_q_cuda_R<type, nwarps, 2 , 2>(args, stream); return;
            case 8:  mul_mat_vec_q_cuda_R<type, nwarps, 8 , 2>(args, stream); return;
            case 16: mul_mat_vec_q_cuda_R<type, nwarps, 16, 2>(args, stream); return;
            default: mul_mat_vec_q_cuda_R<type, nwarps, 4 , 2>(args, stream); return;
        }
    }
    if constexpr (type == GGML_TYPE_Q6_K && nwarps == 4) {
        const bool fused = args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT;
        const int r = mmvq_v100_rows<type>(ggml_cuda_info().devices[ggml_cuda_get_device()].cc,
                                           args.ncols_y, args.nrows_x, args.ne2, fused);
        if (r == 4) { mul_mat_vec_q_cuda_rows1<type, 4>(args, stream); return; }
        if (r == 2) { mul_mat_vec_q_cuda_rows1<type, 2>(args, stream); return; }
    }
    mul_mat_vec_q_cuda_R<type, nwarps, 1, 2>(args, stream);
}

// PXA_MMVQ_NW4 (2026-10-16; unset = ON, 0 = off): 5..8-column launches keep nwarps 4 instead of
// dropping to 2. nwarps sets the k-stride of every thread and the cross-warp fold of every row, so at 2 a column's
// sum is a different association than the 1-column launch computes: the 27B's Q8_0 tensors (output head, ssm_alpha /
// ssm_beta, attn_k / attn_v) made a 5..8-column MTP verify forward differ from one-token decode (pxa-bi-probe K = 5:
// first divergent node beta-0 = MUL_MAT(blk.0.ssm_beta.weight [Q8_0], ...)). At 4 every width 1..8 shares the
// 1-column reduction, the contract PXQN already keeps below.
static inline bool pxa_mmvq_nw4() {
    static const bool v = [](){ const char * e = getenv("PXA_MMVQ_NW4"); return !(e && pxa_env_value_off(e)); }();
    return v;
}
template <ggml_type type>
static void mul_mat_vec_q_cuda(const mmvq_args & args, cudaStream_t stream) {
    int nwarps = 1;
    int id = ggml_cuda_get_device();
    if (args.ne2 < 2 && ggml_cuda_info().devices[id].cc < PXA_CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        nwarps = (args.ncols_y <= 4 || pxa_mmvq_nw4()) ? 4 : 2;
        // PXQN keeps ONE reduction shape for every width 1..8, so a column's result does not depend
        // on how many columns ride with it (the PXQN GEMV's batch-invariance contract, 1..8 on this route)
        if constexpr (type == GGML_TYPE_PXQN4 || type == GGML_TYPE_PXQN4S8) nwarps = 4;
    }
    // PXA 2026-07-11 lever-2: the MoE routed GEMV (ne2>=2) otherwise stays nwarps=1 because the
    // expert dim fills gridDim.y. The nwarps>1 reduction path is ALREADY shipped for the non-MoE case
    // above (same reassociation, ~1e-6 class), so extending it to MoE is low-risk. Env-gated for the A/B;
    // set PXA_MMVQ_MOE_NWARPS=2 or 4 to force it. Prove faster + shadow-clean before defaulting.
    if (args.ne2 >= 2 && ggml_cuda_info().devices[id].cc < PXA_CC_RDNA2) {
        static const int moe_nwarps = [](){ const char * e = getenv("PXA_MMVQ_MOE_NWARPS"); int w = e ? atoi(e) : 0; return (w == 2 || w == 4) ? w : 0; }();
        if (moe_nwarps) nwarps = moe_nwarps;
    }
    switch (nwarps) {
        case 1:
            mul_mat_vec_q_cuda_T<type, 1>(args, stream);
            break;
        case 2:
            mul_mat_vec_q_cuda_T<type, 2>(args, stream);
            break;
        default:
            mul_mat_vec_q_cuda_T<type, 4>(args, stream);
    }
}

extern void mul_mat_vec_q4_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q4_1_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q5_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q5_1_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q6_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q8_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q2_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q3_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q4_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q5_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q6_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_xxs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_xs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_xxs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_m_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_nl_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_mxfp4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_pxq4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_pxq4hq_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_pxqn4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);     // 
extern void mul_mat_vec_pxqn4s8_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_pxq2_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_pxq3_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_xs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern bool mul_mat_vec_q6_K_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);
extern bool mul_mat_vec_q8_0_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);
extern bool mul_mat_vec_pxqn4_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);
extern bool mul_mat_vec_pxqn4s8_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);
extern bool mul_mat_vec_pxqn4_q8_1_group_mix_cuda(mmvq_group_args & ga, mmvq_group_args & gb, ggml_type tb, cudaStream_t stream);

