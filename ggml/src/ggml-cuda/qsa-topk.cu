//
// PXA_QSA: the block top-k as a radix SELECT.
//
// WHAT IT REPLACES AND WHY
// ------------------------
// ggml_top_k in this tree is ggml_argsort(DESC) viewed to the first k, and above 1024 columns
// that argsort routes to a CUB radix sort of the WHOLE row. At the Flash-Next seat's 86k fill
// the QSA selection sorts 21,632 block scores per token per layer to keep 512 of them, and CUB
// spends an index-init kernel, a device-to-device copy, two pool allocations and a dozen or so
// upsweep/scan/downsweep launches doing it. This decode step is bound by per-launch HOST work
// (~1340 launches per token at 98.4% host time), so the launches are the cost, not the sort.
//
// THE ALGORITHM, which is ggml_qsa_topk_row_f32()'s, verbatim
// -----------------------------------------------------------
//  1. floats -> a monotone unsigned key (the IEEE-754 total order): ascending key order is
//     ascending float order, so -inf becomes 0 and a masked block sorts last.
//  2. four 8-bit histogram passes, most significant digit first, walking each histogram from
//     bin 255 down until k elements are accounted for. After four passes `thr` is the EXACT
//     key of the k-th largest element and `above` is how many are strictly greater.
//  3. take every element with key > thr, plus the first k - above with key == thr in ascending
//     column index -- which is exactly what the stable descending sort it replaces does.
//  4. sort those k survivors, and only those.
//
// ONE BLOCK PER ROW, so every pass is a __syncthreads() apart and there is no inter-block
// communication, no temporary allocation and no second kernel. Determinism is structural: the
// histograms are integer adds (order-free), and the survivor sort is a bitonic network over the
// TOTAL order (key descending, then column index ascending), so no two elements ever compare
// equal and the result cannot depend on thread scheduling.
//

#include "qsa-topk.cuh"
#include "../ggml-impl.h" // PXA: ggml_get_op_params_i32

#define QSA_TOPK_BLOCK 256

static __device__ __forceinline__ uint32_t qsa_topk_key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// (key descending, index ascending) as one ascending 64-bit order
static __device__ __forceinline__ uint64_t qsa_topk_ord(uint32_t key, int idx) {
    return (((uint64_t) (~key)) << 32) | (uint32_t) idx;
}

// exclusive prefix sum of `v` across the block; returns this thread's exclusive prefix and
// leaves the block total in *total. Hillis-Steele over shared memory: 8 steps at 256 threads.
static __device__ __forceinline__ int qsa_topk_scan(int v, int * s_scan, int * total) {
    const int tid = threadIdx.x;
    s_scan[tid] = v;
    __syncthreads();
    for (int off = 1; off < QSA_TOPK_BLOCK; off <<= 1) {
        int add = 0;
        if (tid >= off) {
            add = s_scan[tid - off];
        }
        __syncthreads();
        if (tid >= off) {
            s_scan[tid] += add;
        }
        __syncthreads();
    }
    const int incl = s_scan[tid];
    *total = s_scan[QSA_TOPK_BLOCK - 1];
    __syncthreads();
    return incl - v;
}

// k_pad is k rounded up to a power of two; shared memory holds k_pad ordering words
static __global__ void qsa_topk_f32(const float * __restrict__ src, int32_t * __restrict__ dst,
                                    const int ncols, const int k, const int k_pad,
                                    const int64_t src_row_stride, const int64_t dst_row_stride) {
    // declared as uint64_t so the dynamic shared block is 8-byte aligned for s_ord
    extern __shared__ uint64_t qsa_smem_u64[];
    uint64_t * s_ord  = qsa_smem_u64;                              // k_pad
    int      * s_hist = (int *) (s_ord + k_pad);                   // 256
    int      * s_scan = s_hist + 256;                              // QSA_TOPK_BLOCK
    int      * s_ctl  = s_scan + QSA_TOPK_BLOCK;                   // 4

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float * srow = src + (size_t) row*src_row_stride;
    int32_t     * drow = dst + (size_t) row*dst_row_stride;

    //
    // 1-2. four MSD histogram passes -> the exact threshold key
    //
    if (tid == 0) {
        s_ctl[0] = 0;   // prefix
        s_ctl[1] = 0;   // above
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += QSA_TOPK_BLOCK) {
            s_hist[b] = 0;
        }
        __syncthreads();

        const uint32_t prefix  = (uint32_t) s_ctl[0];
        const uint32_t mask_hi = shift >= 24 ? 0u : (0xffffffffu << (shift + 8));

        for (int i = tid; i < ncols; i += QSA_TOPK_BLOCK) {
            const uint32_t key = qsa_topk_key(srow[i]);
            if ((key & mask_hi) == (prefix & mask_hi)) {
                atomicAdd(&s_hist[(key >> shift) & 0xffu], 1);
            }
        }
        __syncthreads();

        // one thread walks 256 bins from the top; cheaper than any parallel scan at this size
        if (tid == 0) {
            int above = s_ctl[1];
            int d = 255;
            for (; d > 0; --d) {
                if (above + s_hist[d] >= k) {
                    break;
                }
                above += s_hist[d];
            }
            s_ctl[0] = (int) ((uint32_t) s_ctl[0] | ((uint32_t) d << shift));
            s_ctl[1] = above;
        }
        __syncthreads();
    }

    const uint32_t thr   = (uint32_t) s_ctl[0];
    const int      above = s_ctl[1];
    const int      n_tie = k - above;

    //
    // 3. collect the survivors: everything above the threshold, then the first n_tie ties, in
    //    ascending column index. Two block scans per chunk -- one to rank the ties, one to place
    //    the takes -- so the ties are taken in index order without any atomics deciding it.
    //
    if (tid == 0) {
        s_ctl[2] = 0;   // ties seen so far
        s_ctl[3] = 0;   // survivors written so far
    }
    for (int i = tid; i < k_pad; i += QSA_TOPK_BLOCK) {
        s_ord[i] = 0xffffffffffffffffull;   // padding sorts last
    }
    __syncthreads();

    for (int base = 0; base < ncols; base += QSA_TOPK_BLOCK) {
        const int i = base + tid;

        uint32_t key = 0;
        bool gt = false, eq = false;
        if (i < ncols) {
            key = qsa_topk_key(srow[i]);
            gt  = key > thr;
            eq  = key == thr;
        }

        int n_eq = 0;
        const int eq_rank = qsa_topk_scan(eq ? 1 : 0, s_scan, &n_eq) + s_ctl[2];

        const bool take = gt || (eq && eq_rank < n_tie);

        int n_take = 0;
        const int slot = qsa_topk_scan(take ? 1 : 0, s_scan, &n_take) + s_ctl[3];

        if (take && slot < k) {
            s_ord[slot] = qsa_topk_ord(key, i);
        }
        __syncthreads();
        if (tid == 0) {
            s_ctl[2] += n_eq;
            s_ctl[3] += n_take;
        }
        __syncthreads();
        if (s_ctl[3] >= k) {
            break;
        }
    }

    //
    // 4. sort the k survivors. Bitonic over k_pad, on a TOTAL order, so the network's result
    //    does not depend on which thread compares which pair.
    //
    for (int len = 2; len <= k_pad; len <<= 1) {
        for (int step = len >> 1; step > 0; step >>= 1) {
            for (int i = tid; i < k_pad; i += QSA_TOPK_BLOCK) {
                const int j = i ^ step;
                if (j > i) {
                    const bool up = ((i & len) == 0);
                    const uint64_t a = s_ord[i];
                    const uint64_t b = s_ord[j];
                    if ((a > b) == up) {
                        s_ord[i] = b;
                        s_ord[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < k; i += QSA_TOPK_BLOCK) {
        drow[i] = (int32_t) (uint32_t) (s_ord[i] & 0xffffffffull);
    }
}

void ggml_cuda_op_qsa_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int     k     = ggml_get_op_params_i32(dst, 0);
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0 && (int64_t) k <= ncols);

    int k_pad = 1;
    while (k_pad < k) {
        k_pad <<= 1;
    }

    const size_t smem = (size_t) k_pad*sizeof(uint64_t) +
                        (size_t) (256 + QSA_TOPK_BLOCK + 4)*sizeof(int);

    cudaStream_t stream = ctx.stream();
    qsa_topk_f32<<<(unsigned) nrows, QSA_TOPK_BLOCK, smem, stream>>>(
            (const float *) src0->data, (int32_t *) dst->data,
            (int) ncols, k, k_pad,
            src0->nb[1]/sizeof(float), dst->nb[1]/sizeof(int32_t));
}
