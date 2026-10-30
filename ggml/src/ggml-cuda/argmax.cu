#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "argmax.cuh"
#include "common.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}


// PXA_ARGMAX_VAL: one block per row -> (index, max, sumexp). Deterministic: the (value, index)
// order is "larger value wins, equal values -> lower index", so the result does not depend on
// the thread schedule and matches the host greedy sampler's first-maximal-id rule.
static __device__ __forceinline__ void pxa_amv_merge(float & v, int & i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}

// OW = 4 (op_params[0] == 4, PXA_DRAFT_ARGMAX): a fourth float, max2 = the largest value at any column other than the argmax
// (= max on an exact tie), taken in the sum pass; the first three floats are the OW = 3 ones bit for bit.
template <int OW>
static __global__ void argmax_val_f32(const float * __restrict__ x, float * __restrict__ dst, const int64_t ncols, const int64_t row_stride) {
    const int64_t row = blockIdx.x;
    const float * rowx = x + row * row_stride;

    float maxval = -INFINITY;
    int   argmax = 0x7fffffff;
    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        pxa_amv_merge(maxval, argmax, rowx[col], (int) col);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float v2 = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   i2 = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        pxa_amv_merge(maxval, argmax, v2, i2);
    }
    __shared__ float s_v[32];
    __shared__ int   s_i[32];
    __shared__ float s_s[32];
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) { s_v[warp_id] = maxval; s_i[warp_id] = argmax; }
    __syncthreads();
    if (warp_id == 0) {
        maxval = lane_id < n_warps ? s_v[lane_id] : -INFINITY;
        argmax = lane_id < n_warps ? s_i[lane_id] : 0x7fffffff;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float v2 = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
            const int   i2 = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
            pxa_amv_merge(maxval, argmax, v2, i2);
        }
        if (lane_id == 0) { s_v[0] = maxval; s_i[0] = argmax; }
    }
    __syncthreads();
    maxval = s_v[0];
    argmax = s_i[0];

    float sum = 0.0f;
    float m2  = -INFINITY;
    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float xv = rowx[col];
        sum += expf(xv - maxval);
        if (OW == 4 && col != argmax) m2 = fmaxf(m2, xv);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        if (OW == 4) m2 = fmaxf(m2, __shfl_xor_sync(0xFFFFFFFF, m2, offset, WARP_SIZE));
    }
    __shared__ float s_m2[32];
    __syncthreads();
    if (lane_id == 0) { s_s[warp_id] = sum; if (OW == 4) s_m2[warp_id] = m2; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < n_warps; ++w) tot += s_s[w];
        dst[OW*row + 0] = (float) argmax;
        dst[OW*row + 1] = maxval;
        dst[OW*row + 2] = tot;
        if (OW == 4) {
            float mm = -INFINITY;
            for (int w = 0; w < n_warps; ++w) mm = fmaxf(mm, s_m2[w]);
            dst[OW*row + 3] = mm;
        }
    }
}

// PXA_ARGMAX_VAL_MB (K2 head; default 0 = off, =1 on): argmax_val_f32's (index, max, sumexp) on many
// blocks instead of ONE 1024-thread block per row (164 us per 248320-wide row on V100 = 0.7% of a 27B PXQN4 token), BIT FOR
// BIT the same three floats:
//   pass 1 (k_amv_mb_max, grid (nb, rows), 256 threads): a chunk's (max, index) with pxa_amv_merge. The merge is a total
//          order on (value, index) (larger value, then lower index; NaN never wins), so any partition reaches the
//          reference's (max, argmax);
//   pass 2 (k_amv_mb_sum, grid (32, rows), one warp each): every block merges the nb partials, then block w IS the
//          reference's warp w: lane l sums expf(x[c] - max) over c = 32w + l + 1024k in increasing k (the reference's
//          thread 32w + l at blockDim 1024) and runs the same xor butterfly;
//   pass 3 (k_amv_mb_fin, one thread per row): tot = 0 + s_0 + ... + s_31 in warp order, as the reference's thread 0.
// tests/test-argmax-val-mb.cu checks the three floats against argmax_val_f32 by bits.
#define PXA_AMV_REF_THREADS 1024
#define PXA_AMV_MB_CHUNK    8192
static __global__ void __launch_bounds__(256)
k_amv_mb_max(const float * __restrict__ x, float2 * __restrict__ part, const int64_t ncols, const int64_t row_stride) {
    const int64_t row = blockIdx.y;
    const int nb = gridDim.x;
    const float * rowx = x + row * row_stride;
    const int64_t c0 = (int64_t) blockIdx.x * PXA_AMV_MB_CHUNK;
    const int64_t c1 = c0 + PXA_AMV_MB_CHUNK < ncols ? c0 + PXA_AMV_MB_CHUNK : ncols;
    float v = -INFINITY;
    int   i = 0x7fffffff;
#pragma unroll 4
    for (int64_t c = c0 + threadIdx.x; c < c1; c += 256) {
        pxa_amv_merge(v, i, rowx[c], (int) c);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xFFFFFFFF, v, o, WARP_SIZE);
        const int   i2 = __shfl_xor_sync(0xFFFFFFFF, i, o, WARP_SIZE);
        pxa_amv_merge(v, i, v2, i2);
    }
    __shared__ float sv[8];
    __shared__ int   si[8];
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int k = 1; k < 8; ++k) pxa_amv_merge(v, i, sv[k], si[k]);
        part[row * nb + blockIdx.x] = make_float2(v, __int_as_float(i));
    }
}
template <int OW>
static __global__ void __launch_bounds__(32)
k_amv_mb_sum(const float * __restrict__ x, const float2 * __restrict__ part, float * __restrict__ ssum, float * __restrict__ dst,
             const int64_t ncols, const int64_t row_stride, const int nb) {
    const int64_t row = blockIdx.y;
    const int w = blockIdx.x, l = threadIdx.x;
    float v = -INFINITY;
    int   i = 0x7fffffff;
    for (int k = l; k < nb; k += 32) {
        const float2 p = part[row * nb + k];
        pxa_amv_merge(v, i, p.x, __float_as_int(p.y));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xFFFFFFFF, v, o, WARP_SIZE);
        const int   i2 = __shfl_xor_sync(0xFFFFFFFF, i, o, WARP_SIZE);
        pxa_amv_merge(v, i, v2, i2);
    }
    const float maxval = v;
    const float * rowx = x + row * row_stride;
    float sum = 0.0f;
    float m2  = -INFINITY;
#pragma unroll 8
    for (int64_t col = 32*w + l; col < ncols; col += PXA_AMV_REF_THREADS) {
        const float xv = rowx[col];
        sum += expf(xv - maxval);
        if (OW == 4 && col != i) m2 = fmaxf(m2, xv);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        if (OW == 4) m2 = fmaxf(m2, __shfl_xor_sync(0xFFFFFFFF, m2, offset, WARP_SIZE));
    }
    if (l == 0) {
        ssum[row * (OW == 4 ? 64 : 32) + w] = sum;
        if (OW == 4) ssum[row * 64 + 32 + w] = m2;
        if (w == 0) { dst[OW*row + 0] = (float) i; dst[OW*row + 1] = maxval; }
    }
}
template <int OW>
static __global__ void k_amv_mb_fin(const float * __restrict__ ssum, float * __restrict__ dst, const int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float tot = 0.0f;
    for (int w = 0; w < PXA_AMV_REF_THREADS/WARP_SIZE; ++w) tot += ssum[r * (OW == 4 ? 64 : 32) + w];
    dst[OW*r + 2] = tot;
    if (OW == 4) {
        float mm = -INFINITY;
        for (int w = 0; w < 32; ++w) mm = fmaxf(mm, ssum[r * 64 + 32 + w]);
        dst[OW*r + 3] = mm;
    }
}
// part: rows*nb float2, ssum: rows*32 floats (rows*64 at OW = 4) (caller-provided scratch)
template <int OW>
static void pxa_argmax_val_mb_launch(const float * x, float * dst, int64_t ncols, int64_t nrows, int64_t row_stride,
                                     float2 * part, float * ssum, cudaStream_t st) {
    const int nb = (int) ((ncols + PXA_AMV_MB_CHUNK - 1) / PXA_AMV_MB_CHUNK);
    k_amv_mb_max<<<dim3(nb, (unsigned) nrows, 1), 256, 0, st>>>(x, part, ncols, row_stride);
    k_amv_mb_sum<OW><<<dim3(32, (unsigned) nrows, 1), 32, 0, st>>>(x, part, ssum, dst, ncols, row_stride, nb);
    k_amv_mb_fin<OW><<<(unsigned) ((nrows + 127) / 128), 128, 0, st>>>(ssum, dst, (int) nrows);
}
static inline int pxa_argmax_val_mb_nb(int64_t ncols) { return (int) ((ncols + PXA_AMV_MB_CHUNK - 1) / PXA_AMV_MB_CHUNK); }
static bool pxa_argmax_val_mb_on() {
    static const bool v = [] { const char * e = getenv("PXA_ARGMAX_VAL_MB"); return e && *e && atoi(e) != 0; }();
    return v;
}

// PXA_ARGMAX_COMBINE (op_params[0] == 2): src[k] = the [3, rows] (index, max, sumexp) of vocab slice k,
// op_params[1 + k] = that slice's first column, op_params[5] = number of slices. dst I32 [rows] = the
// global argmax, slices scanned in column order with strictly-greater wins (lowest id on exact ties).
struct pxa_amc_args { const float * p[4]; int off[4]; int n; };
static __global__ void argmax_combine_f32(pxa_amc_args a, int32_t * __restrict__ dst, const int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float best_v = 0.0f; int best = -1;
    for (int k = 0; k < a.n; ++k) {
        const float * q = a.p[k] + 3*r;
        if (best < 0 || q[1] > best_v) { best_v = q[1]; best = a.off[k] + (int) q[0]; }
    }
    dst[r] = best;
}

// PXA_SAMPLED_TOPW (op_params[0] == 3, 2026-10-19): the top-W window of every row (W = op_params[1] <= 128),
// dst row = W values sorted descending, then W ids as floats; equal values keep the lower id first -- the CPU sampler's raw
// window (common/sampling.cpp pxa_topk_window) element for element. Order key = (orderable value << 32) | ~id, -0.0 keyed as
// +0.0 (the CPU compares them equal); the values written back are read from the row itself, so they are the logits' bits.
//   pass 1 (grid (ceil(n/1024), rows), 256 threads): a 1024-column chunk, bitonic-sorted in shared memory, its top W keys
//   merge passes (grid (ceil(lists/G), rows), G = 1024/W lists per block): the same sort over G lists, top W out, until one
//   list is left; the last pass writes the values and ids. A total order on (value, id): the result does not depend on the
//   chunking, the pass count or the block schedule.
#define PXA_TOPW_CH 1024
static __device__ __forceinline__ unsigned long long pxa_topw_key(float v, int id) {
    if (v == 0.0f) v = 0.0f;                                     // -0.0 -> +0.0
    const unsigned int u = __float_as_uint(v);
    const unsigned int o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) o << 32) | (unsigned long long) (0xFFFFFFFFu - (unsigned int) id);
}
static __device__ __forceinline__ void pxa_topw_bitonic_desc(unsigned long long * s) {   // N = PXA_TOPW_CH, 256 threads
    for (int k = 2; k <= PXA_TOPW_CH; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
            for (int t = 0; t < PXA_TOPW_CH/2/256; ++t) {
                const int p   = threadIdx.x + 256*t;                // pair index: i = the lower element of the pair
                const int i   = 2*j*(p / j) + (p % j);
                const int ixj = i + j;
                const unsigned long long a = s[i], b = s[ixj];
                const bool desc = (i & k) == 0;
                if (desc ? (a < b) : (a > b)) { s[i] = b; s[ixj] = a; }
            }
            __syncthreads();
        }
    }
}
static __global__ void __launch_bounds__(256)
k_topw_p1(const float * __restrict__ x, unsigned long long * __restrict__ part, const int64_t ncols, const int64_t row_stride, const int W) {
    __shared__ unsigned long long s[PXA_TOPW_CH];
    const int64_t row = blockIdx.y;
    const float * rowx = x + row * row_stride;
    const int64_t c0 = (int64_t) blockIdx.x * PXA_TOPW_CH;
    for (int i = threadIdx.x; i < PXA_TOPW_CH; i += 256) {
        const int64_t c = c0 + i;
        s[i] = c < ncols ? pxa_topw_key(rowx[c], (int) c) : 0ull;
    }
    __syncthreads();
    pxa_topw_bitonic_desc(s);
    for (int i = threadIdx.x; i < W; i += 256) part[(row * gridDim.x + blockIdx.x) * W + i] = s[i];
}
// lists of W keys in: [rows][nin][W]; out: [rows][gridDim.x][W], or the final dst when dst != nullptr (one block per row)
static __global__ void __launch_bounds__(256)
k_topw_merge(const unsigned long long * __restrict__ pin, unsigned long long * __restrict__ pout, const int nin, const int W,
             const float * __restrict__ x, const int64_t row_stride, float * __restrict__ dst) {
    __shared__ unsigned long long s[PXA_TOPW_CH];
    const int64_t row = blockIdx.y;
    const int G = PXA_TOPW_CH / W;
    const int l0 = blockIdx.x * G;
    for (int i = threadIdx.x; i < PXA_TOPW_CH; i += 256) {
        const int l = l0 + i / W;
        s[i] = l < nin ? pin[(row * nin + l) * W + i % W] : 0ull;
    }
    __syncthreads();
    pxa_topw_bitonic_desc(s);
    if (dst == nullptr) {
        for (int i = threadIdx.x; i < W; i += 256) pout[(row * gridDim.x + blockIdx.x) * W + i] = s[i];
        return;
    }
    const float * rowx = x + row * row_stride;
    float * d = dst + row * 2 * W;
    for (int i = threadIdx.x; i < W; i += 256) {
        const unsigned long long k = s[i];
        if (k == 0ull) { d[i] = -INFINITY; d[W + i] = -1.0f; continue; }
        const int id = (int) (0xFFFFFFFFu - (unsigned int) (k & 0xFFFFFFFFull));
        d[i] = rowx[id]; d[W + i] = (float) id;
    }
}
// scratch: 2 buffers of rows * ceil(ncols/1024) * W keys each (pxa_topw_scratch_keys)
static inline size_t pxa_topw_scratch_keys(int64_t ncols, int64_t nrows, int W) {
    return (size_t) nrows * (size_t) ((ncols + PXA_TOPW_CH - 1) / PXA_TOPW_CH) * (size_t) W;
}
static void pxa_topw_launch(const float * x, float * dst, int64_t ncols, int64_t nrows, int64_t row_stride, int W,
                            unsigned long long * b0, unsigned long long * b1, cudaStream_t st) {
    int n = (int) ((ncols + PXA_TOPW_CH - 1) / PXA_TOPW_CH);
    k_topw_p1<<<dim3(n, (unsigned) nrows, 1), 256, 0, st>>>(x, b0, ncols, row_stride, W);
    const int G = PXA_TOPW_CH / W;
    unsigned long long * a = b0, * b = b1;
    for (;;) {
        const int nout = (n + G - 1) / G;
        k_topw_merge<<<dim3(nout, (unsigned) nrows, 1), 256, 0, st>>>(a, b, n, W, x, row_stride, nout == 1 ? dst : nullptr);
        if (nout == 1) break;
        n = nout; unsigned long long * t = a; a = b; b = t;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    if (dst->op_params[0] == 2) {
        GGML_ASSERT(dst->type == GGML_TYPE_I32);
        pxa_amc_args a = {};
        a.n = dst->op_params[5];
        GGML_ASSERT(a.n >= 1 && a.n <= 4);
        for (int k = 0; k < a.n; ++k) {
            GGML_ASSERT(dst->src[k] && dst->src[k]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[k]));
            a.p[k] = (const float *) dst->src[k]->data;
            a.off[k] = dst->op_params[1 + k];
        }
        const int rows = (int) dst->ne[0];
        argmax_combine_f32<<<(rows + 127) / 128, 128, 0, ctx.stream()>>>(a, (int32_t *) dst->data, rows);
        return;
    }

    if (dst->op_params[0] == 3) {   // PXA_SAMPLED_TOPW
        GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src0->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
        const int W = dst->op_params[1];
        const int64_t nrows = src0->ne[1];
        GGML_ASSERT(W >= 1 && W <= 128 && (W & (W - 1)) == 0 && dst->ne[0] == 2*W && nrows <= 65535);
        const size_t nk = pxa_topw_scratch_keys(src0->ne[0], nrows, W);
        ggml_cuda_pool_alloc<unsigned long long> b0(ctx.pool(), nk), b1(ctx.pool(), nk);
        pxa_topw_launch((const float *) src0->data, (float *) dst->data, src0->ne[0], nrows, src0->nb[1] / sizeof(float), W,
                        b0.get(), b1.get(), ctx.stream());
        return;
    }

    if (dst->op_params[0] == 1 || dst->op_params[0] == 4) {   // 4: + max2 (PXA_DRAFT_ARGMAX)
        GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src0->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
        const bool w4 = dst->op_params[0] == 4;
        GGML_ASSERT(dst->ne[0] == (w4 ? 4 : 3));
        const int64_t nrows = src0->ne[1];
        if (pxa_argmax_val_mb_on() && src0->ne[0] >= 4*PXA_AMV_MB_CHUNK && nrows <= 65535) {   // PXA_ARGMAX_VAL_MB
            ggml_cuda_pool_alloc<float2> part(ctx.pool(), (size_t) nrows * pxa_argmax_val_mb_nb(src0->ne[0]));
            ggml_cuda_pool_alloc<float>  ssum(ctx.pool(), (size_t) nrows * (w4 ? 64 : 32));
            if (w4) {
                pxa_argmax_val_mb_launch<4>((const float *) src0->data, (float *) dst->data, src0->ne[0], nrows,
                                            src0->nb[1] / sizeof(float), part.get(), ssum.get(), ctx.stream());
            } else {
                pxa_argmax_val_mb_launch<3>((const float *) src0->data, (float *) dst->data, src0->ne[0], nrows,
                                            src0->nb[1] / sizeof(float), part.get(), ssum.get(), ctx.stream());
            }
            return;
        }
        if (w4) {
            argmax_val_f32<4><<<dim3(nrows, 1, 1), dim3(1024, 1, 1), 0, ctx.stream()>>>(
                (const float *) src0->data, (float *) dst->data, src0->ne[0], src0->nb[1] / sizeof(float));
        } else {
            argmax_val_f32<3><<<dim3(nrows, 1, 1), dim3(1024, 1, 1), 0, ctx.stream()>>>(
                (const float *) src0->data, (float *) dst->data, src0->ne[0], src0->nb[1] / sizeof(float));
        }
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
