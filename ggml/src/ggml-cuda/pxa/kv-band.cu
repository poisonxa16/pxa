// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// kv-band.cu -- PXA_KV_BAND_EXACT, the P.V half (the softmax half lives in softmax.cu). See kv-band.cuh.
#include "pxa-env-flag.h"
#include "kv-band.cuh"

#include <climits>
#include <cstdlib>

bool ggml_cuda_pxa_kv_band_exact(void) {
    static const bool v = [](){
        const char * e = getenv("PXA_KV_BAND_EXACT");
        return e && *e ? atoi(e) != 0 : ggml_pxa_config_level() >= 1;
    }();
    return v;
}

// The soft_max half runs on the same cards as the P.V half (NVIDIA, cc >= 6.0). Pascal was thought placement-free
// (its fp16 GEMM has no tensor-core grain), but the one-token decode P.V (n = 1, k = the KV cells) takes a cuBLAS
// kernel that splits the cell axis, so a moved band was grouped differently there too (test-kv-band-align -gpu
// decode rows on a P100, 3e-3..8.7e-3). Both halves are identical to the old arithmetic when the
// band starts at cell 0. HIP keeps the old walk byte for byte. Declared next to its callers in softmax.cu.
static bool pxa_kv_band_cc(int cc) {
    return cc >= PXA_CC_PASCAL && cc < PXA_CC_OFFSET_AMD;
}

bool pxa_kv_band_softmax_applies(int cc) {
    return ggml_cuda_pxa_kv_band_exact() && pxa_kv_band_cc(cc);
}

// PXA_KV_BAND_MAX_MIB (default 64): the P.V half copies V with V's own strides, so the copy spans the layer's whole
// ring (n_ctx cells, not n_kv) and comes from the CUDA pool, which the placement planner does not count. A call whose
// copy would exceed the cap keeps the previous arithmetic (that call stays placement-dependent: long contexts on
// the unfused chain); the soft_max half is unaffected. 0 = no copy at all (P.V half off).
static size_t pxa_kv_band_cap_bytes(void) {
    static const size_t v = [](){
        const char * e = getenv("PXA_KV_BAND_MAX_MIB");
        const long mib = e && *e ? atol(e) : 64;
        return (size_t) (mib < 0 ? 0 : mib) << 20;
    }();
    return v;
}

bool pxa_kv_band_pv_applies(const ggml_tensor * src0, const ggml_tensor * src1, int cc) {
    if (!ggml_cuda_pxa_kv_band_exact() || !pxa_kv_band_cc(cc) || pxa_kv_band_cap_bytes() == 0) {
        return false;
    }
    // the same span ggml_cuda_mul_mat_batched_cublas_impl allocates for the copy: all of it on the strided route (one
    // host pointer) and on a tensor-core card; on Pascal's pointer route only the matrices from the first one a moved
    // read can take past the end of V (and its pad) on (review N3: the cap used to count the whole ring there)
    const size_t span = (src0->ne[0] - 1)*src0->nb[0] + (src0->ne[1] - 1)*src0->nb[1] +
                        (src0->ne[2] - 1)*src0->nb[2] + (src0->ne[3] - 1)*src0->nb[3] + src0->nb[0];
    size_t copy = span;
    const bool strided = src1->ne[2] == src0->ne[2] && src1->ne[3] == src0->ne[3] && ggml_is_contiguous_2(src0);
    if (cc < PXA_CC_VOLTA && !strided && src0->data != nullptr) {
        const int64_t lo = pxa_kv_band_copy_from(src0->ne, src0->nb, pxa_kv_band_limit(src0, cc), -1);
        copy = lo < 0 ? 0 : span - (size_t) lo;
    }
    if (copy > pxa_kv_band_cap_bytes()) {
        return false;
    }
    if (src1->op != GGML_OP_SOFT_MAX && src1->op != GGML_OP_SOFT_CAP_MAX) {
        return false;
    }
    const ggml_tensor * mask = src1->src[1];
    return mask != nullptr && (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) &&
           src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) &&
           mask->nb[0] == ggml_type_size(mask->type) && src1->ne[0] <= INT_MAX;
}

// open = anything but a masked -inf (the softmax walk in softmax.cu uses the same test)
template <typename T>
static __device__ __forceinline__ bool kv_band_open(T v) {
    return !((float) v == -INFINITY);
}
template <>
__device__ __forceinline__ bool kv_band_open<half>(half v) {
    return !(__half2float(v) == -INFINITY);
}

// one block per mask row (i1 < nrows, every i2/i3 of the mask): the first open column, folded with atomicMin
template <typename T>
static __global__ void k_kv_band_anchor(const char * __restrict__ mask, const int ncols, const int64_t nrows, const int64_t ne2,
                                        const size_t nb1, const size_t nb2, const size_t nb3, int * __restrict__ anchor) {
    __shared__ int smin;
    if (threadIdx.x == 0) {
        smin = INT_MAX;
    }
    __syncthreads();
    const int64_t i1 = blockIdx.x % nrows;
    const int64_t i2 = (blockIdx.x / nrows) % ne2;
    const int64_t i3 = blockIdx.x / (nrows*ne2);
    const T * row = (const T *) (mask + i1*nb1 + i2*nb2 + i3*nb3);
    for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
        if (kv_band_open(row[c])) {
            atomicMin(&smin, c); // the first open column of this thread's stride; later ones are larger
            break;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && smin != INT_MAX) {
        atomicMin(anchor, smin);
    }
}

void pxa_kv_band_anchor(const ggml_tensor * mask, int64_t ncols, int64_t nrows, int * anchor_d, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(anchor_d, 0x7f, sizeof(int), stream)); // 0x7f7f7f7f: "no open column" (>= ncols)
    const int64_t mcols = std::min<int64_t>(ncols, mask->ne[0]);
    const int64_t mrows = std::min<int64_t>(nrows, mask->ne[1]);
    const int64_t nblk  = mrows*mask->ne[2]*mask->ne[3];
    if (nblk <= 0 || mcols <= 0) {
        return;
    }
    if (mask->type == GGML_TYPE_F16) {
        k_kv_band_anchor<half><<<nblk, 256, 0, stream>>>((const char *) mask->data, (int) mcols, mrows, mask->ne[2],
                                                          mask->nb[1], mask->nb[2], mask->nb[3], anchor_d);
    } else {
        k_kv_band_anchor<float><<<nblk, 256, 0, stream>>>((const char *) mask->data, (int) mcols, mrows, mask->ne[2],
                                                           mask->nb[1], mask->nb[2], mask->nb[3], anchor_d);
    }
    CUDA_CHECK(cudaGetLastError());
}

static __device__ __forceinline__ int kv_band_read_anchor(const int * anchor, const int64_t ncols) {
    const int a = *anchor;
    return a > 0 && a < ncols ? a : 0;
}

// anchor_h >= 0: the anchor known on the host (pxa_kv_band_host_known, already read the way the line above reads it)
static __device__ __forceinline__ int kv_band_anchor_of(const int * anchor, const int anchor_h, const int64_t ncols) {
    return anchor_h >= 0 ? anchor_h : kv_band_read_anchor(anchor, ncols);
}

static int kv_band_blocks(const int64_t n) {
    return (int) std::max<int64_t>(1, std::min<int64_t>((n + 255)/256, 4096));
}

// ---- Pascal: cuBLAS's two kernel classes for the P.V product ----
// On sm_60 the attention P.V call (OP_T/OP_N, fp16, COMPUTE_16F) takes one cuBLAS kernel for k <= 511 and another
// for k >= 512 whenever n (query columns) <= 16; wider calls take one kernel at every k (10-22,
// cuBLAS 12.8, m 16..512, n 1..32, batch 4/32, k 64..32768). Each kernel alone ignores an exact-zero tail, so with
// the band anchored at k = 0 the only thing a band's placement can still change is WHICH kernel runs: k = n_kv
// grows with the cells in front of the band. So the call runs both: the k >= 512 kernel on the full width as
// before, and the k <= 511 kernel on the first 511 cells into a side buffer; a one-block pick keeps the latter
// exactly when the same band at cell 0 would have had n_kv < 512 (its span padded the way the cache pads n_kv:
// 256 cells with flash attention, whose mask is f16, else 32). A band at cell 0 is then answered byte for byte as
// before at every n_kv, and a moved band the same.
bool pxa_kv_band_pascal_dual(int cc, int64_t k, int64_t n) {
    return cc < PXA_CC_VOLTA && k >= PXA_KV_BAND_PASCAL_K && n <= PXA_KV_BAND_PASCAL_NMAX;
}

// The pad the pick assumes (pxa_kv_band_pick): 256 cells with flash attention (f16 mask), else 32.
static int pxa_kv_band_pad(const ggml_tensor * mask) {
    return mask->type == GGML_TYPE_F16 ? 256 : 32;
}

static bool pxa_kv_band_anchor0_on(void) {
    static const bool v = [](){ const char * e = getenv("PXA_KV_BAND_ANCHOR0"); return !(e && *e && pxa_env_value_off(e)); }();
    return v;
}

bool pxa_kv_band_anchor0_proven(const ggml_tensor * mask, int64_t ncols, int64_t nrows, bool dual, cudaStream_t stream) {
    if (mask == nullptr || !pxa_kv_band_anchor0_on() || mask->op != GGML_OP_NONE) {
        return false;
    }
    const int32_t * p = mask->op_params;
    // the proof covers exactly the rows and columns the band kernels read (min(nrows, ne[1]) rows, min(ncols, ne[0])
    // columns, every plane): the same counts on both sides, nothing more, nothing less
    if (p[0] != GGML_PXA_KQMASK_PROOF_MAGIC || p[GGML_PXA_KQMASK_PROOF_COL0] != 1 ||
        mask->ne[0] != ncols || p[GGML_PXA_KQMASK_PROOF_COLS] != ncols ||
        nrows > mask->ne[1] || p[GGML_PXA_KQMASK_PROOF_ROWS] != nrows) {
        return false;
    }
    if (dual && p[GGML_PXA_KQMASK_PROOF_LAST] < PXA_KV_BAND_PASCAL_K - pxa_kv_band_pad(mask)) {
        return false; // a band this short at cell 0 would have met the k <= 511 class: the pick decides (or unknown)
    }
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone) {
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

static bool pxa_kv_band_hostproof_on(void) {
    static const bool v = [](){ const char * e = getenv("PXA_KV_BAND_HOSTPROOF"); return !(e && *e && atoi(e) == 0); }();
    return v;
}

bool pxa_kv_band_host_known(const ggml_tensor * mask, int64_t ncols, int64_t nrows, cudaStream_t stream, int * anchor, int * last) {
    if (mask == nullptr || !pxa_kv_band_hostproof_on() || mask->op != GGML_OP_NONE) {
        return false;
    }
    const int32_t * p = mask->op_params;
    // the rows and columns k_kv_band_anchor / k_kv_band_last read (min(nrows, ne[1]) rows, min(ncols, ne[0]) columns,
    // every plane), exactly the ones llama scanned
    if (p[0] != GGML_PXA_KQMASK_PROOF_MAGIC || mask->ne[0] != ncols || p[GGML_PXA_KQMASK_PROOF_COLS] != ncols ||
        nrows > mask->ne[1] || p[GGML_PXA_KQMASK_PROOF_ROWS] != nrows) {
        return false;
    }
    const int32_t f = p[GGML_PXA_KQMASK_PROOF_FIRST];
    const int32_t e = p[GGML_PXA_KQMASK_PROOF_LAST];
    if (f < -1 || e < -1 || f >= ncols || e >= ncols) {
        return false; // not computed (a wide ubatch) or out of range
    }
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone) {
        (void) cudaGetLastError();
        return false; // a replayed graph would keep this call's anchor
    }
    *anchor = f > 0 ? f : 0; // kv_band_read_anchor: none open (-1) and 0 both read as 0
    *last   = e;
    return true;
}

bool pxa_kv_band_pick_small(int anchor, int last, const ggml_tensor * mask) {
    if (last < 0) {
        return false; // k_kv_band_pick keeps the full-width answer
    }
    const int64_t pad    = pxa_kv_band_pad(mask);
    const int64_t span   = (int64_t) last - anchor + 1;
    const int64_t padded = (span + pad - 1)/pad*pad;
    const int64_t nkv0   = padded > pad ? padded : pad;
    return nkv0 < PXA_KV_BAND_PASCAL_K;
}

template <typename T>
static __global__ void k_kv_band_last(const char * __restrict__ mask, const int ncols, const int64_t nrows, const int64_t ne2,
                                      const size_t nb1, const size_t nb2, const size_t nb3, int * __restrict__ last) {
    __shared__ int smax;
    if (threadIdx.x == 0) {
        smax = -1;
    }
    __syncthreads();
    const int64_t i1 = blockIdx.x % nrows;
    const int64_t i2 = (blockIdx.x / nrows) % ne2;
    const int64_t i3 = blockIdx.x / (nrows*ne2);
    const T * row = (const T *) (mask + i1*nb1 + i2*nb2 + i3*nb3);
    int m = -1;
    for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
        if (kv_band_open(row[c])) {
            m = c;
        }
    }
    if (m >= 0) {
        atomicMax(&smax, m);
    }
    __syncthreads();
    if (threadIdx.x == 0 && smax >= 0) {
        atomicMax(last, smax);
    }
}

void pxa_kv_band_last(const ggml_tensor * mask, int64_t ncols, int64_t nrows, int * last_d, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(last_d, 0xff, sizeof(int), stream)); // -1: no open column
    const int64_t mcols = std::min<int64_t>(ncols, mask->ne[0]);
    const int64_t mrows = std::min<int64_t>(nrows, mask->ne[1]);
    const int64_t nblk  = mrows*mask->ne[2]*mask->ne[3];
    if (nblk <= 0 || mcols <= 0) {
        return;
    }
    if (mask->type == GGML_TYPE_F16) {
        k_kv_band_last<half><<<nblk, 256, 0, stream>>>((const char *) mask->data, (int) mcols, mrows, mask->ne[2],
                                                        mask->nb[1], mask->nb[2], mask->nb[3], last_d);
    } else {
        k_kv_band_last<float><<<nblk, 256, 0, stream>>>((const char *) mask->data, (int) mcols, mrows, mask->ne[2],
                                                         mask->nb[1], mask->nb[2], mask->nb[3], last_d);
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void k_kv_band_pick(const half * __restrict__ small, half * __restrict__ dst, const int64_t n,
                                      const int * __restrict__ anchor, const int * __restrict__ last, const int64_t ncols,
                                      const int pad) {
    const int e = *last;
    if (e < 0) {
        return; // nothing open: keep the full-width answer (the previous arithmetic)
    }
    const int     a    = kv_band_read_anchor(anchor, ncols);
    const int64_t span = (int64_t) e - a + 1;
    const int64_t padded = (span + pad - 1)/pad*pad;
    const int64_t nkv0   = padded > pad ? padded : pad; // n_kv of the same band at cell 0
    if (nkv0 >= PXA_KV_BAND_PASCAL_K) {
        return;
    }
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        dst[i] = small[i];
    }
}

void pxa_kv_band_pick(const half * small, half * dst, int64_t n, const int * anchor_d, const int * last_d, int64_t ncols,
                      const ggml_tensor * mask, cudaStream_t stream) {
    const int pad = pxa_kv_band_pad(mask); // llama_kv_cache_get_padding: f16 mask = flash attention
    k_kv_band_pick<<<kv_band_blocks(n), 256, 0, stream>>>(small, dst, n, anchor_d, last_d, ncols, pad);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void k_kv_band_shift_p(const float * __restrict__ P, half * __restrict__ out, const int64_t ncols,
                                         const int64_t n, const int * __restrict__ anchor, const int anchor_h) {
    const int64_t a = kv_band_anchor_of(anchor, anchor_h, ncols);
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        const int64_t c = i % ncols;
        out[i] = c + a < ncols ? __float2half(P[i + a]) : __float2half(0.0f);
    }
}

void pxa_kv_band_shift_p(const float * P, half * out, int64_t ncols, int64_t nrows, const int * anchor_d, int anchor_h,
                         cudaStream_t stream) {
    const int64_t n = ncols*nrows;
    k_kv_band_shift_p<<<kv_band_blocks(n), 256, 0, stream>>>(P, out, ncols, n, anchor_d, anchor_h);
    CUDA_CHECK(cudaGetLastError());
}

// Pascal, pointer route: V read in place, each GEMM matrix moved by the anchor (no copy). Column c of the moved matrix
// reads cell c + a; past the row end that is a cell of the next row of the same V tensor. Every column c >= ne0 - a
// meets an exact-zero P column (pxa_kv_band_shift_p), so those reads add exact zeros, as the copy's zero tail does, and
// they are finite: the same cells the unmoved product multiplies by its masked zeros, or V cells of the KV buffer, which
// is cleared at allocation. A matrix moves only while its last read stays inside the V tensor (limit_off = bytes from
// the view to the tensor end, < 0 = never: tensor-core cards keep the copy). cuBLAS cannot see the device pointer
// array, so the kernel class (and the accumulation order) is the one the copy gets.
static __device__ __forceinline__ bool kv_band_v_inplace(const int64_t a, const int64_t off, const int64_t ne0,
                                                         const int64_t ne1, const size_t nb0, const size_t nb1,
                                                         const int64_t limit_off) {
    return limit_off >= 0 && off + (ne1 - 1)*(int64_t) nb1 + (a + ne0)*(int64_t) nb0 <= limit_off;
}

// blockIdx.y walks the rows (i1, i2, i3) of V from row0, threads walk the cells of a row: one index split per row.
// With ptr_route the rows of matrices read in place (anchor 0 or kv_band_v_inplace) are skipped.
static __global__ void k_kv_band_shift_v(const char * __restrict__ V, char * __restrict__ out,
                                         const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t row0,
                                         const int64_t nrows, const size_t nb0, const size_t nb1, const size_t nb2,
                                         const size_t nb3, const int * __restrict__ anchor, const int anchor_h,
                                         const bool ptr_route, const int64_t limit_off) {
    const int64_t a = kv_band_anchor_of(anchor, anchor_h, ne0);
    if (ptr_route && a == 0) {
        return;
    }
    for (int64_t r = row0 + blockIdx.y; r < nrows; r += gridDim.y) {
        const int64_t i1 = r % ne1;
        const int64_t i2 = (r / ne1) % ne2;
        const int64_t i3 = r / (ne1*ne2);
        if (ptr_route && kv_band_v_inplace(a, i2*nb2 + i3*nb3, ne0, ne1, nb0, nb1, limit_off)) {
            continue;
        }
        const char * src = V   + i1*nb1 + i2*nb2 + i3*nb3;
        char       * dst = out + i1*nb1 + i2*nb2 + i3*nb3;
        for (int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; c < ne0; c += (int64_t) gridDim.x*blockDim.x) {
            *(half *) (dst + c*nb0) = c + a < ne0 ? *(const half *) (src + (c + a)*nb0) : __float2half(0.0f);
        }
    }
}

// PXA_KV_BAND_V_BLOCKS (default 4096, the cap the flat copy had before the row grid): blocks per launch of the V copy;
// the rows past it are walked by the grid stride. A call at anchor 0 on the pointer route copies nothing, yet every block
// still reads the anchor, so such a launch costs what its blocks cost: one block row per V row (1024 rows x 25 blocks at
// n_kv 6400) put a V100's -fa off decode at -np 2 ~5% behind the flat copy. Same bytes at any value.
// 0 = one block row per V row (up to 65535).
static int64_t pxa_kv_band_v_blocks(void) {
    static const int64_t v = [](){
        const char * e = getenv("PXA_KV_BAND_V_BLOCKS");
        const long n = e && *e ? atol(e) : 4096;
        return (int64_t) (n < 0 ? 0 : n);
    }();
    return v;
}

// the first matrix (and the smallest offset of any matrix) a moved read takes past limit_off, for anchor a (a < 0: the
// worst anchor ne0 - 1); m0 = nmat and lo = -1 when every matrix is read in place
static void kv_band_first_copied(const int64_t ne[4], const size_t nb[4], int64_t limit_off, int64_t a, int64_t * m0, int64_t * lo) {
    const int64_t nmat = ne[2]*ne[3];
    const int64_t aw   = a < 0 ? ne[0] - 1 : a;
    *m0 = nmat;
    *lo = -1;
    for (int64_t m = 0; m < nmat; ++m) {
        const int64_t off = (m % ne[2])*(int64_t) nb[2] + (m / ne[2])*(int64_t) nb[3];
        if (off + (ne[1] - 1)*(int64_t) nb[1] + (aw + ne[0])*(int64_t) nb[0] > limit_off) {
            if (*m0 == nmat) {
                *m0 = m;
            }
            if (*lo < 0 || off < *lo) {
                *lo = off;
            }
        }
    }
}

int64_t pxa_kv_band_copy_from(const int64_t ne[4], const size_t nb[4], int64_t limit_off, int a) {
    if (a == 0) {
        return -1; // the pointer route reads the view itself
    }
    if (limit_off < 0) {
        return 0;
    }
    int64_t m0, lo;
    kv_band_first_copied(ne, nb, limit_off, a, &m0, &lo);
    return lo;
}

static bool pxa_kv_band_inplace_on(void) {
    static const bool v = [](){ const char * e = getenv("PXA_KV_BAND_INPLACE"); return !(e && *e && atoi(e) == 0); }();
    return v;
}

int64_t pxa_kv_band_limit(const ggml_tensor * src0, int cc) {
    if (cc >= PXA_CC_VOLTA || !pxa_kv_band_inplace_on() || src0->data == nullptr) {
        return -1;
    }
    const ggml_tensor * vb = src0->view_src ? src0->view_src : src0;
    int64_t lim = (int64_t) ((const char *) vb->data + ggml_nbytes(vb) - (const char *) src0->data);
    if (vb->op == GGML_OP_NONE && vb->type == GGML_TYPE_F16 && vb->op_params[0] == GGML_PXA_KV_VPAD_MAGIC &&
        vb->op_params[GGML_PXA_KV_VPAD_BYTES] > 0) {
        lim += vb->op_params[GGML_PXA_KV_VPAD_BYTES]; // llama's zero pad behind V (PXA_KV_BAND_VPAD)
    }
    return lim;
}

void pxa_kv_band_shift_v(const char * V, char * out, const int64_t ne[4], const size_t nb[4], const int * anchor_d,
                         int anchor_h, bool ptr_route, int64_t limit_off, cudaStream_t stream) {
    const int64_t nmat  = ne[2]*ne[3];
    const int64_t nrows = ne[1]*nmat;
    if (nrows <= 0 || ne[0] <= 0) {
        return;
    }
    if (ptr_route && anchor_h == 0) {
        return; // the view itself, nothing to copy (the kernel would exit at once)
    }
    int64_t row0 = 0;
    if (ptr_route && limit_off >= 0) {
        // matrices before m0 stay inside the tensor for this anchor (unknown: the worst one, a = ne0 - 1): no rows to
        // copy there
        int64_t m0, lo;
        kv_band_first_copied(ne, nb, limit_off, anchor_h, &m0, &lo);
        if (m0 == nmat) {
            return;
        }
        row0 = m0*ne[1];
    }
    const int64_t gx  = std::min<int64_t>((ne[0] + 255)/256, 64);
    const int64_t cap = pxa_kv_band_v_blocks();
    int64_t gy = std::min<int64_t>(nrows - row0, 65535);
    if (cap > 0) {
        gy = std::min<int64_t>(gy, std::max<int64_t>(1, cap/gx));
    }
    const dim3 grid((unsigned) gx, (unsigned) gy);
    k_kv_band_shift_v<<<grid, 256, 0, stream>>>(V, out, ne[0], ne[1], ne[2], row0, nrows, nb[0], nb[1], nb[2], nb[3],
                                                anchor_d, anchor_h, ptr_route, limit_off);
    CUDA_CHECK(cudaGetLastError());
}

// the arithmetic of k_compute_batched_ptrs (ggml-cuda.cu), with the src0 base picked by the anchor
static __global__ void k_kv_band_batched_ptrs(
        const char * src0_view, const char * src0_copy, const int * __restrict__ anchor, const int anchor_h, const int64_t ncols,
        const int64_t ne01, const size_t nb00, const size_t nb01, const int64_t limit_off, const char * src1, char * dst, const void ** ptrs_src, void ** ptrs_dst,
        const int64_t ne12, const int64_t ne13, const int64_t ne23, const size_t nb02, const size_t nb03,
        const size_t nb12, const size_t nb13, const size_t nbd2, const size_t nbd3, const int64_t r2, const int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }
    const int64_t a = kv_band_anchor_of(anchor, anchor_h, ncols);
    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;
    const int64_t off = i02*nb02 + i03*nb03;
    const char * src0 = a == 0 ? src0_view :
        kv_band_v_inplace(a, off, ncols, ne01, nb00, nb01, limit_off) ? src0_view + a*nb00 : src0_copy;
    ptrs_src[0*ne23 + i12 + i13*ne12] = src0 + off;
    ptrs_src[1*ne23 + i12 + i13*ne12] = src1 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = dst  + i12*nbd2 + i13*nbd3;
}

void pxa_kv_band_batched_ptrs(const void * src0_view, const void * src0_copy, const int * anchor_d, int anchor_h, int64_t ncols,
                              int64_t ne01, size_t nb00, size_t nb01, int64_t limit_off, const void * src1, void * dst, const void ** ptrs_src, void ** ptrs_dst,
                              int64_t ne12, int64_t ne13, int64_t ne23, size_t nb02, size_t nb03, size_t nb12, size_t nb13,
                              size_t nbd2, size_t nbd3, int64_t r2, int64_t r3, cudaStream_t stream) {
    const dim3 block_dims(16, 16);
    const dim3 grid_dims((ne13 + 15)/16, (ne12 + 15)/16);
    k_kv_band_batched_ptrs<<<grid_dims, block_dims, 0, stream>>>((const char *) src0_view, (const char *) src0_copy, anchor_d, anchor_h,
        ncols, ne01, nb00, nb01, limit_off, (const char *) src1, (char *) dst, ptrs_src, ptrs_dst, ne12, ne13, ne23, nb02, nb03, nb12, nb13, nbd2, nbd3, r2, r3);
    CUDA_CHECK(cudaGetLastError());
}
