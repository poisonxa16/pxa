// pxq4hq_mma.cu -- Volta (sm_70) tensor-core path for the PXQ4HQ multi-token decode GEMM.
//
//   out[M, N] = x[M, K] @ W[N, K]^T,  M = 1..16,  W in PXQ4HQ.
//
// RELATIONSHIP TO pxq4_mma.cu. That file is the shipped v12b PXQ4 arena path and is frozen.
// This is its transcription for the HQ tier, in its own translation unit for the same reason
// pxq4hq_kernel.cuh is: the PXQ4 kernel resolves ONE effective scale per 16-element k-tile and
// folds it into the fp16 weight before the fragment store, which is a shape and not a
// constant. PXQ4HQ has TWO per k-tile.
//
// WHAT CHANGES, exhaustively. Everything else -- the 32-way replicated book, the two
// round-robined accumulators, the global lookahead over three register slots, the block-wide
// activation staging, the split-K arena and its fp32 reduce -- is a line-for-line copy, and
// the device-vs-device arm of the gate is what proves it.
//   1. SLAB 1088 -> 1152, CODE_OFF 64 -> 128.
//   2. The scale byte a lane loads is slab[2*row + h], not slab[row]: PXQ4HQ's scale SoA is
//      two bytes per row, byte h covering the 16-element k-half this package already owns.
//   3. That byte's TWO nibbles are two effective scales inside the package's own half -- low
//      nibble for its elements 0..7, high for 8..15 -- so the fold is per code byte b
//      (b < 4 -> low) instead of one scalar for the whole fragment row.
//   4. The sub LUT is SUB8 (pxq4hq_mma_sub8_g), never the pxq4 SUB16 symbol.
//
// FORMAT ALIGNMENT, which is what still makes this cheap. A 16-element K half of a slab is
// EXACTLY 8 code bytes -- exactly the K extent of a wmma m16n16k16 fragment -- and PXQ4HQ does
// not change that: it subdivides the SCALE, not the codes. The scales therefore still fold
// into the fp16 weight before the fragment store and never touch the accumulator; there are
// simply two of them per fragment row instead of one. One 64-bit code load per lane still
// produces one complete half of a B fragment row.
//
// NUMERICS. Identical argument to the PXQ4 arena path: fp32 wmma accumulation across the whole
// of K, an fp32 split-K reduce, and a single __float2half_rn on the way out. The only
// departure from the SIMT kernel's arithmetic is that eff*BOOK[c] is rounded to fp16
// (relative error <= 2^-11) instead of staying fp32 -- smaller than the __float2half_rn the
// SIMT kernel already applies to its own result. Bit-identity with the SIMT kernel is NOT
// claimed and must never be gated on: the accumulation order differs by construction. The
// device-vs-device arm compares this path against dequant + cuBLAS on the same bytes.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

#include "pxq4hq_kernel_tables.h"

namespace wmma = nvcuda::wmma;

#define PXQ4HQ_MMA_WPAD  24                                   // smem ldm of a B tile, in halves
#define PXQ4HQ_MMA_GSLAB 4                                    // slabs of x staged per group
#define PXQ4HQ_MMA_XLD   (PXQ4HQ_MMA_GSLAB * PXQ4HQ_QK + 8)   // 136 halves = 272 B, 16-B rows

// TU-local table copies (static __device__ is TU-local, so a second TU cannot see the first
// one's symbols). Fanned out from the same uploads that feed pxq4hq_kernel.cu's copies.
static __device__ float pxq4hq_mma_book_g[16] = PXQ4HQ_BOOK_INIT;
static __device__ float pxq4hq_mma_sub8_g[16] = PXQ4HQ_SUB8_INIT;

// ---------------------------------------------------------------------------------------------
// grid = (nsplit, panels), block = 128 threads = 4 warps, one warp per 16 weight rows.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128)
k_pxq4hq_mma(const uint8_t * __restrict__ slabs,
             const __half  * __restrict__ anchor,
             const __half  * __restrict__ x,        // [M, K] fp16, M <= 16
             float         * __restrict__ part,     // [nsplit, panels, 64, 16] fp32
             const int kslabs, const int K, const int nsplit, const int panels, const int M)
{
    const int s    = blockIdx.x;
    const int p    = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = lane & 15;            // row inside this warp's 16-row group
    const int h    = lane >> 4;            // which 16-element K half of the slab

    // BOOK is replicated 32 ways so bookr[c][lane] always lands in bank `lane`, whatever the
    // code value. A bare 16-entry table indexed by data is a 2-8 way bank conflict on every
    // one of the 16 lookups per lane per slab.
    __shared__ float  bookr[16][32];
    __shared__ float  subt[16];
    __shared__ __half ws[4][2][16][PXQ4HQ_MMA_WPAD];
    __shared__ __align__(16) __half xs[16][PXQ4HQ_MMA_XLD];

    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) bookr[i >> 5][i & 31] = pxq4hq_mma_book_g[i >> 5];
    if (threadIdx.x < 16) subt[threadIdx.x] = pxq4hq_mma_sub8_g[threadIdx.x];

    const int   row  = warp * 16 + r;
    const float anch = __half2float(anchor[(size_t)p * PXQ4HQ_BM + row]);
    const uint8_t * pan = slabs + (size_t)p * kslabs * PXQ4HQ_SLAB_BYTES;

    // GLOBAL LOOKAHEAD: keep TWO slabs in flight in registers, so the LDG for slab t+2 is
    // issued before slab t's dequant and has two full slabs of arithmetic to hide behind.
    // Three register slots rotate; the rotation is by explicit named variables, never an
    // indexed array, because a dynamically indexed local array would spill to local memory and
    // cost more than the stall it removes.
    //
    // (SB) IS THE HQ CHANGE. PXQ4 reads one scale byte per row, sl_[row], and splits its two
    // nibbles across the two k-halves. PXQ4HQ reads TWO bytes per row and this package owns
    // exactly one of them -- byte h, covering this package's own k-half -- so the load is
    // sl_[2*row + h] and BOTH its nibbles belong to this package.
#define PXQ4HQ_MMA_LOAD_SLAB(TT, SB, Q)                                                        \
    do {                                                                                       \
        const uint8_t * sl_ = pan + (size_t)(TT) * PXQ4HQ_SLAB_BYTES;                          \
        (SB) = sl_[2 * row + h];                                                               \
        (Q)  = *(const uint2 *)(sl_ + PXQ4HQ_CODE_OFF + PXQ4HQ_CODE_BYTES * row + 8 * h);      \
    } while (0)

    const int b0 = (int)(((int64_t)kslabs * s)       / nsplit);
    const int b1 = (int)(((int64_t)kslabs * (s + 1)) / nsplit);

    // TWO accumulators, round-robined over the two k-tiles of a slab. Consecutive mma.sync
    // into one accumulator are a serial RAW chain and Volta HMMA latency is ~20 cycles.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    __syncthreads();                                        // covers the table staging

    uint32_t sb0 = 0u, sb1 = 0u, sb2 = 0u;
    uint2    q0  = make_uint2(0u, 0u), q1 = q0, q2 = q0;
    if (b0     < b1) PXQ4HQ_MMA_LOAD_SLAB(b0,     sb0, q0);
    if (b0 + 1 < b1) PXQ4HQ_MMA_LOAD_SLAB(b0 + 1, sb1, q1);

    for (int gs = b0; gs < b1; gs += PXQ4HQ_MMA_GSLAB) {
        const int gn = min(PXQ4HQ_MMA_GSLAB, b1 - gs);
        // Stage this group's activation slice ONCE for the whole block. Letting each warp pull
        // its own A fragment straight out of L2 was 4x the traffic and put ~250 cycles on the
        // mma critical path.
        __syncthreads();
        {
            const int nv = gn * PXQ4HQ_QK / 8;              // uint4 per row
            for (int idx = threadIdx.x; idx < 16 * nv; idx += 128) {
                const int m = idx / nv, u = idx - m * nv;
                // Rows >= M are the m16 padding: zeroed HERE, so the caller never has to
                // materialise a padded [16, K] copy of the activations.
                uint4 val = make_uint4(0u, 0u, 0u, 0u);
                if (m < M) val = *(const uint4 *)&x[(size_t)m * K + gs * PXQ4HQ_QK + u * 8];
                *(uint4 *)&xs[m][u * 8] = val;
            }
        }
        __syncthreads();

        for (int j = 0; j < gn; ++j) {
            const int t = gs + j;
            // Issue the load for slab t+2 FIRST, before anything that consumes slab t: that is
            // the whole point of the rotation. 32 lanes x 8 B covers 256 contiguous bytes of
            // the code block -- one LDG.64 each, perfectly coalesced, and each lane's 8 bytes
            // are exactly one B-fragment row half.
            if (t + 2 < b1) PXQ4HQ_MMA_LOAD_SLAB(t + 2, sb2, q2);
            // TWO effective scales inside this package's k-half: the low nibble covers its
            // elements 0..7 (code bytes b < 4), the high nibble its elements 8..15.
            const float efflo = anch * subt[sb0 & 0xf];
            const float effhi = anch * subt[sb0 >> 4];
            __half2 v[8];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const uint32_t w   = (b < 4) ? q0.x : q0.y;
                const uint32_t by  = (w >> (8 * (b & 3))) & 0xffu;
                const float    eff = (b < 4) ? efflo : effhi;
                v[b] = __floats2half2_rn(eff * bookr[by & 0xf][lane],
                                         eff * bookr[by >> 4  ][lane]);
            }
            *(uint4 *)&ws[warp][h][r][0] = *(const uint4 *)&v[0];
            *(uint4 *)&ws[warp][h][r][8] = *(const uint4 *)&v[4];
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b0f, b1f;
            wmma::load_matrix_sync(a0,  &xs[0][j * PXQ4HQ_QK],      PXQ4HQ_MMA_XLD);
            wmma::load_matrix_sync(a1,  &xs[0][j * PXQ4HQ_QK + 16], PXQ4HQ_MMA_XLD);
            wmma::load_matrix_sync(b0f, &ws[warp][0][0][0],         PXQ4HQ_MMA_WPAD);
            wmma::load_matrix_sync(b1f, &ws[warp][1][0][0],         PXQ4HQ_MMA_WPAD);
            wmma::mma_sync(acc0, a0, b0f, acc0);
            wmma::mma_sync(acc1, a1, b1f, acc1);
            __syncwarp();                                   // ws[] is reused next iteration
            // Rotate the in-flight slab registers. Slot 2 is only ever read after the guarded
            // load that filled it: slot 0 at iteration t was slot 2 at iteration t-2, whose
            // guard was (t-2)+2 < b1, i.e. t < b1 -- true for every iteration that runs.
            sb0 = sb1; q0 = q1;
            sb1 = sb2; q1 = q2;
        }
    }
#pragma unroll
    for (int i = 0; i < acc0.num_elements; ++i) acc0.x[i] += acc1.x[i];

    // part[(s, p, warp*16 + n, m)]: a col_major store puts element (m, n) at ptr[m + n*16].
    wmma::store_matrix_sync(part + ((size_t)(s * panels + p) * PXQ4HQ_BM + warp * 16) * 16,
                            acc0, 16, wmma::mem_col_major);
}

// fp32 split-K reduce -> fp16 out[M, N]. Ascending s, exactly as the SIMT mmv folds ascending
// chunks, so the fold order is a function of shape only.
__global__ void k_pxq4hq_mma_reduce(const float * __restrict__ part, __half * __restrict__ out,
                                    const int nsplit, const int panels, const int M, const int N)
{
    const int p = blockIdx.x;
    const int n = threadIdx.x;                              // 0..63
    const float * base = part + ((size_t)p * PXQ4HQ_BM + n) * 16;
    const size_t  step = (size_t)panels * PXQ4HQ_BM * 16;
    for (int m = 0; m < M; ++m) {
        float u = 0.f;
        for (int s = 0; s < nsplit; ++s) u += base[(size_t)s * step + m];
        out[(size_t)m * N + p * PXQ4HQ_BM + n] = __float2half_rn(u);
    }
}

// ---------------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------------
#define PXQ4HQ_MMA_CHECK(expr)                                                                \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "pxq4hq: %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                                \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

// CTA budget: ~512 resident blocks is the measured sweet spot on an 80-SM V100. Overshooting
// is the expensive direction (the PXQ4 sweep of nsplit in {2,4,8,16} on gate_up gave
// 176 / 184 / 203 / 494 us), and the HQ tier reads 6% more bytes per slab for the same shape,
// so the same budget is if anything slightly conservative here.
static int pxq4hq_mma_cta_budget() {
    static const int v = [] {
        const char * e = getenv("PXQ4HQ_MMA_CTAS");
        if (!(e && *e)) e = getenv("PXQ4_MMA_CTAS");        // one knob for both arena paths
        return (e && *e) ? atoi(e) : 512;
    }();
    return v;
}

static int pxq4hq_mma_nsplit(int panels, int kslabs) {
    int want = (pxq4hq_mma_cta_budget() + panels - 1) / panels;
    if (want < 1)  want = 1;
    if (want > 16) want = 16;
    while (want > 1 && kslabs / want < PXQ4HQ_MMA_GSLAB) --want;   // >= one full x-stage group
    return want;
}

int pxq4hq_mma_part_floats(int panels, int kslabs) {
    // SHAPE ONLY -- no M term. Every CUDA-graph capture size therefore asks the arena for the
    // identical allocation, so a capture can never be the first call that has to grow it.
    return pxq4hq_mma_nsplit(panels, kslabs) * panels * PXQ4HQ_BM * 16;
}

// The arch probe is asked ONCE, not once per call: this predicate sits in the mmv dispatch
// hook, so per-call it would run cudaGetDevice + cudaDeviceGetAttribute on every routed call,
// entirely on the host critical path and entirely redundant -- the answer is a property of the
// card. Same ONE-DEVICE-PER-PROCESS assumption the occupancy constant already makes.
static bool pxq4hq_mma_arch_ok() {
    static const bool v = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        return major >= 7;                                  // wmma m16n16k16 fp16 needs sm_70
    }();
    return v;
}

bool pxq4hq_mma_supported(int panels, int kslabs, int M) {
    if (M < 1 || M > 16) return false;
    if (panels < 1 || kslabs < 1) return false;
    if (panels > 65535) return false;                       // grid.y limit
    // Same static smem footprint as the PXQ4 arena path (the scale bytes live in registers,
    // not shared), well inside the 48 KiB no-opt-in budget.
    return pxq4hq_mma_arch_ok();
}

void pxq4hq_mma_upload_tables(const float * book16, const float * sub8) {
    PXQ4HQ_MMA_CHECK(cudaMemcpyToSymbol(pxq4hq_mma_book_g, book16, 16 * sizeof(float)));
    PXQ4HQ_MMA_CHECK(cudaMemcpyToSymbol(pxq4hq_mma_sub8_g, sub8,   16 * sizeof(float)));
}

void pxq4hq_launch_mma_f16(const uint8_t * slabs, const void * anchor, const void * x,
                           float * part, void * out, int M, int panels, int kslabs,
                           cudaStream_t stream) {
    const int N      = panels * PXQ4HQ_BM;
    const int K      = kslabs * PXQ4HQ_QK;
    const int nsplit = pxq4hq_mma_nsplit(panels, kslabs);
    if (M < 1 || M > 16) {
        fprintf(stderr, "pxq4hq: mma M out of range: %d\n", M);
        abort();
    }
    k_pxq4hq_mma<<<dim3((unsigned)nsplit, (unsigned)panels), 128, 0, stream>>>(
        slabs, (const __half *)anchor, (const __half *)x, part, kslabs, K, nsplit, panels, M);
    PXQ4HQ_MMA_CHECK(cudaGetLastError());
    k_pxq4hq_mma_reduce<<<(unsigned)panels, PXQ4HQ_BM, 0, stream>>>(
        part, (__half *)out, nsplit, panels, M, N);
    PXQ4HQ_MMA_CHECK(cudaGetLastError());
}
