// pxq4_mma.cu -- Volta (sm_70) tensor-core path for the PXQ4 multi-token decode GEMM.
//
//   out[M, N] = x[M, K] @ W[N, K]^T,  M = 1..16,  W in PXQ4.
//
// WHY THIS EXISTS. k_pxq4_mmv_fused_mt amortises the weight READ across the M tokens but not
// the ARITHMETIC: it still issues M fp32 FMA chains per decoded weight, so its cost is linear
// in M once the weight stream stops being the limit. Measured on V100-PCIe, gate_up
// (N=17408, K=5120, 47.3 MB): 100 us at M=1 (473 GB/s, near the bandwidth roof) but 298 us at
// M=8 (159 GB/s). The decode-side work is what scales, and on sm_70 it does not have to: the
// same weight stream feeds HMMA at 8x the fp32 FMA rate.
//
// The dequant + cuBLAS alternative was measured and is not competitive at these M:
// k_pxq4_dequant_matrix ALONE costs 305 us on gate_up -- more than the whole MT kernel at
// M=8 -- because it writes 178 MB of fp16 and reads 47 MB, 4x the byte traffic of the PXQ4
// stream, and it is already bandwidth-saturated at 748 GB/s. No dispatch threshold can fix
// that; only a fused kernel can.
//
// FORMAT ALIGNMENT, which is what makes this cheap. PXQ4 stores, per (panel of 64 rows,
// slab of 32 K-elements):
//     slab[0..63]               one scale byte per row: low nibble -> SUB16 index for
//                               elements 0..15, high nibble -> elements 16..31
//     slab[64 + 16*row .. +16]  16 code bytes, byte b = code(2b) | code(2b+1) << 4
//     w = fp32(anchor[row]) * SUB16[s4] * BOOK[c]
// so a 16-element K half of a slab is EXACTLY 8 code bytes under EXACTLY one effective
// scale -- which is exactly the K extent of a wmma m16n16k16 fragment. The scale therefore
// folds into the fp16 weight once per 16 values and never touches the accumulator, and one
// 64-bit code load per lane produces one complete half of a B fragment row.
//
// NUMERICS, and the PXQ4_GEMM2D lesson. gemm2d failed first-token quality at 87.5% because it
// carried a __half2 accumulator across the whole of K with no fp32 partials (relmean 0.9-1.2%);
// its codebook and scale handling were fp32 and correct. This kernel accumulates fp32 for the
// whole of K (wmma .f32.f16.f16.f32 chained across every k-tile, then an fp32 split-K reduce,
// then a single __float2half_rn on the way out). The ONLY departure from the SIMT kernel's
// arithmetic is that eff*BOOK[c] is rounded to fp16 (relative error <= 2^-11) instead of
// staying fp32 -- which is SMALLER than the __float2half_rn the SIMT kernel already applies to
// its own result. Measured against an fp32 reference over all five decode shapes and
// M in {1,2,4,8,16}: SIMT kernel rel-L2 2.07e-4, this kernel 2.92e-4, max-abs difference
// between the two <= 1.95e-3 (1-2 fp16 ULP). Bit-identity with the SIMT kernel is NOT claimed
// and must never be gated on: the accumulation order differs by construction.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

#include "pxq4_kernel_tables.h"

namespace wmma = nvcuda::wmma;

#define PXQ4_MMA_WPAD  24                       // smem ldm of a B tile, in halves (48 B)
#define PXQ4_MMA_GSLAB 4                        // slabs of x staged per group
#define PXQ4_MMA_XLD   (PXQ4_MMA_GSLAB * PXQ4_QK + 8)   // 136 halves = 272 B, 16-B rows

// TU-local table copies, exactly as pxq4_kernel.cu owns its own pxq4_book_g / pxq4_sub16_g
// (static __device__ is TU-local, so a second TU cannot see the first one's symbols).
// pxq4_mma_upload_tables is fanned out from the same set_tables that feeds pxq4_upload_tables.
static __device__ float pxq4_mma_book_g[16]  = PXQ4_BOOK_INIT;
static __device__ float pxq4_mma_sub16_g[16] = PXQ4_SUB16_INIT;
static __device__ uint32_t pxq4_mma_tab32_g[256 * 32];       // code byte -> (BOOK lo, BOOK hi) fp16 pair, x32 lanes

// ---------------------------------------------------------------------------------------------
// grid = (nsplit, panels), block = 128 threads = 4 warps, one warp per 16 weight rows.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128)
k_pxq4_mma(const uint8_t * __restrict__ slabs,
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
    // one of the 16 lookups per lane per slab; the engine's own note (pxq4_pol::stage_tabs)
    // records that WIDENING the table made this worse, but replicating it makes it free.
    __shared__ float  bookr[16][32];
    __shared__ float  subt[16];
    __shared__ __half ws[4][2][16][PXQ4_MMA_WPAD];
    __shared__ __align__(16) __half xs[16][PXQ4_MMA_XLD];

    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) bookr[i >> 5][i & 31] = pxq4_mma_book_g[i >> 5];
    if (threadIdx.x < 16) subt[threadIdx.x] = pxq4_mma_sub16_g[threadIdx.x];

    const int   row  = warp * 16 + r;
    const float anch = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const uint8_t * pan = slabs + (size_t)p * kslabs * PXQ4_SLAB_BYTES;

    // v14 GLOBAL LOOKAHEAD (smallm2, 2026-09-07). Every slab's weight bytes used to be loaded
    // on the line that consumed them, so the block ran one serial chain per slab:
    //     LDG.64 -> 16x LDS(book) + 16x FMUL -> STS.128 -> __syncwarp -> LDS(wmma) -> HMMA
    // and NCU's picture of that is a mainloop that is not bandwidth bound at all -- 171 us on
    // gate_up at M=8 is 277 GB/s against the 473 GB/s the SIMT kernel already reaches at M=1.
    // The fix is Kewaii's (1Cat sm70_awq_small_n_hmma_operator.md, GmemLookahead=2): keep TWO
    // slabs in flight in registers, so the LDG for slab t+2 is issued before slab t's dequant
    // and has two full slabs of arithmetic to hide behind. Three register slots rotate; the
    // rotation is by explicit named variables, never an indexed array, because a dynamically
    // indexed local array would spill to local memory and cost more than the stall it removes.
#define PXQ4_MMA_LOAD_SLAB(TT, SB, Q)                                                          \
    do {                                                                                       \
        const uint8_t * sl_ = pan + (size_t)(TT) * PXQ4_SLAB_BYTES;                            \
        (SB) = sl_[row];                                                                        \
        (Q)  = *(const uint2 *)(sl_ + PXQ4_CODE_OFF + PXQ4_CODE_BYTES * row + 8 * h);          \
    } while (0)

    const int b0 = (int)(((int64_t)kslabs * s)       / nsplit);
    const int b1 = (int)(((int64_t)kslabs * (s + 1)) / nsplit);

    // TWO accumulators, round-robined over the two k-tiles of a slab. Consecutive mma.sync
    // into one accumulator are a serial RAW chain and Volta HMMA latency is ~20 cycles;
    // splitting them was worth 355 -> 245 us on gate_up before the other fixes landed.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    __syncthreads();                                        // covers the table staging

    uint32_t sb0 = 0u, sb1 = 0u, sb2 = 0u;
    uint2    q0  = make_uint2(0u, 0u), q1 = q0, q2 = q0;
    if (b0     < b1) PXQ4_MMA_LOAD_SLAB(b0,     sb0, q0);
    if (b0 + 1 < b1) PXQ4_MMA_LOAD_SLAB(b0 + 1, sb1, q1);

    for (int gs = b0; gs < b1; gs += PXQ4_MMA_GSLAB) {
        const int gn = min(PXQ4_MMA_GSLAB, b1 - gs);
        // Stage this group's activation slice ONCE for the whole block. Letting each warp
        // pull its own A fragment straight out of L2 was 4x the traffic and put ~250 cycles
        // on the mma critical path: 130 us of the original 355 us, measured by ablation.
        __syncthreads();
        {
            const int nv = gn * PXQ4_QK / 8;                // uint4 per row
            for (int idx = threadIdx.x; idx < 16 * nv; idx += 128) {
                const int m = idx / nv, u = idx - m * nv;
                // Rows >= M are the m16 padding: zeroed HERE, so the caller never has to
                // materialise a padded [16, K] copy of the activations.
                uint4 val = make_uint4(0u, 0u, 0u, 0u);
                if (m < M) val = *(const uint4 *)&x[(size_t)m * K + gs * PXQ4_QK + u * 8];
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
            if (t + 2 < b1) PXQ4_MMA_LOAD_SLAB(t + 2, sb2, q2);
            const float eff = anch * subt[h ? (sb0 >> 4) : (sb0 & 0xf)];
            __half2 v[8];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const uint32_t w  = (b < 4) ? q0.x : q0.y;
                const uint32_t by = (w >> (8 * (b & 3))) & 0xffu;
                v[b] = __floats2half2_rn(eff * bookr[by & 0xf][lane],
                                         eff * bookr[by >> 4  ][lane]);
            }
            *(uint4 *)&ws[warp][h][r][0] = *(const uint4 *)&v[0];
            *(uint4 *)&ws[warp][h][r][8] = *(const uint4 *)&v[4];
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b0f, b1f;
            wmma::load_matrix_sync(a0,  &xs[0][j * PXQ4_QK],      PXQ4_MMA_XLD);
            wmma::load_matrix_sync(a1,  &xs[0][j * PXQ4_QK + 16], PXQ4_MMA_XLD);
            wmma::load_matrix_sync(b0f, &ws[warp][0][0][0],       PXQ4_MMA_WPAD);
            wmma::load_matrix_sync(b1f, &ws[warp][1][0][0],       PXQ4_MMA_WPAD);
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
    wmma::store_matrix_sync(part + ((size_t)(s * panels + p) * PXQ4_BM + warp * 16) * 16,
                            acc0, 16, wmma::mem_col_major);
}

// fp32 split-K reduce -> fp16 out[M, N]. Ascending s, exactly as k_pxq4_mmv_reduce folds
// ascending c, so the fold order is a function of shape only.
__global__ void k_pxq4_mma_reduce(const float * __restrict__ part, __half * __restrict__ out,
                                  const int nsplit, const int panels, const int M, const int N)
{
    const int p = blockIdx.x;
    const int n = threadIdx.x;                              // 0..63
    const float * base = part + ((size_t)p * PXQ4_BM + n) * 16;
    const size_t  step = (size_t)panels * PXQ4_BM * 16;
    for (int m = 0; m < M; ++m) {
        float u = 0.f;
        for (int s = 0; s < nsplit; ++s) u += base[(size_t)s * step + m];
        out[(size_t)m * N + p * PXQ4_BM + n] = __float2half_rn(u);
    }
}

// ---------------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------------
#define PXQ4_MMA_CHECK(expr)                                                                  \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "pxq4: %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                                \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

// CTA budget: ~512 resident blocks is the measured sweet spot on an 80-SM V100 (nsplit 2 for
// the 272-panel gate_up, 7 for the 80-panel out_proj). Sweeping nsplit in {2,4,8,16} on
// gate_up gave 176 / 184 / 203 / 494 us, so overshooting is the expensive direction.
static int pxq4_mma_cta_budget() {
    static const int v = [] {
        const char * e = getenv("PXQ4_MMA_CTAS");
        return (e && *e) ? atoi(e) : 512;
    }();
    return v;
}

static int pxq4_mma_nsplit(int panels, int kslabs) {
    int want = (pxq4_mma_cta_budget() + panels - 1) / panels;
    if (want < 1)  want = 1;
    if (want > 16) want = 16;
    while (want > 1 && kslabs / want < PXQ4_MMA_GSLAB) --want;   // >= one full x-stage group
    return want;
}

int pxq4_mma_part_floats(int panels, int kslabs) {
    // SHAPE ONLY -- no M term. Every CUDA-graph capture size therefore asks the arena for the
    // identical allocation, so a capture can never be the first call that has to grow it.
    return pxq4_mma_nsplit(panels, kslabs) * panels * PXQ4_BM * 16;
}

// v12b: the arch probe is asked ONCE, not once per call. This predicate sits in the mmv
// dispatch hook, so with the arm enabled it ran cudaGetDevice + cudaDeviceGetAttribute on
// EVERY routed call -- two driver round trips per PXQ4 module per token, ~240 modules per
// token on this model, entirely on the host critical path and entirely redundant: the answer
// is a property of the card. Cached the same way the split/mono occupancy constant already
// caches multiProcessorCount (pxq4_kernel_torch.cpp), which carries the same assumption:
// ONE DEVICE PER PROCESS. That is how the fork runs TP -- one rank, one process, one device,
// set before any PXQ4 op is reachable -- and a second device in the same process would have
// mis-answered the occupancy constant long before it mis-answered this.
static bool pxq4_mma_arch_ok() {
    static const bool v = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        return major >= 7;                                  // wmma m16n16k16 fp16 needs sm_70
    }();
    return v;
}

bool pxq4_mma_supported(int panels, int kslabs, int M) {
    if (M < 1 || M > 16) return false;
    if (panels < 1 || kslabs < 1) return false;
    if (panels > 65535) return false;                       // grid.y limit
    // 12,608 B of static smem, well inside the 48 KiB no-opt-in budget; 72 registers, no spill.
    return pxq4_mma_arch_ok();
}

// set false by pxq4_mma_upload_tables; the code->half2 table derives from the book
static bool g_m884_tab_ready = false;

void pxq4_mma_upload_tables(const float * book16, const float * sub16) {
    PXQ4_MMA_CHECK(cudaMemcpyToSymbol(pxq4_mma_book_g,  book16, 16 * sizeof(float)));
    PXQ4_MMA_CHECK(cudaMemcpyToSymbol(pxq4_mma_sub16_g, sub16,  16 * sizeof(float)));
    g_m884_tab_ready = false;                                 // the code->half2 table derives from the book
}


// =============================================================================================
// k_pxq4_mma884 -- raw mma.sync.m8n8k4 small-N path (2026-09-07). PXQ4 dequantised
// STRAIGHT INTO THE A FRAGMENT REGISTERS: no shared-memory staging of the weight at all.
//
// WHY. k_pxq4_mma above is bound by shared-memory wavefronts, not by DRAM and not by latency
// (smallm2, V100, gate_up M=8: 170 us = 278 GB/s while the SIMT kernel reaches 493 GB/s on the
// same bytes; a CTA sweep and a two-slab lookahead moved nothing). Every A and B fragment it feeds
// to wmma goes through smem with a stride wmma forces to a multiple of 16 B, so every fragment
// load is 4-way bank conflicted; 139 of the 170 us are LSU time. The wmma API gives no layout
// that avoids it. mma.sync.m8n8k4 documents its fragment layout (PTX ISA "Matrix Fragments for
// mma.m8n8k4 with .f16"), and that layout happens to be exactly the PXQ4 code layout:
//     A (8x4, row-major): each lane holds ONE ROW's 4 consecutive k values (a0..a3)
//     PXQ4: 16 code bytes per row per slab, byte b = code(2b) | code(2b+1) << 4
// so a lane's 4 k values are 2 consecutive code bytes of its row. One 16-B load per lane per slab
// (the whole row's 32 codes) feeds 8 mma instructions with zero smem traffic for A.
//
// SCALES ARE FOLDED INTO THE ACCUMULATE, NOT THE OPERAND. w = anchor[row] * SUB16[s4] * BOOK[c],
// and (anchor*SUB16) is constant per (row, 16-k half). So the mma accumulates sum_k BOOK[c]*x over
// each 16-k half into a FRESH fp32 fragment (4 instructions), and the lane then does
// acc += eff(row_of_this_c_element) * c. BOOK values are fp16-exact (PX16 book), fp16 x fp16
// products are exact in fp32, so the only roundings are the fp32 sums and the final
// __float2half_rn -- fewer than the wmma kernel, which rounds eff*BOOK to fp16 per weight.
// Bit-identity with the SIMT kernel is NOT claimed (fold order), same as k_pxq4_mma.
//
// Geometry: grid (nsplit, panels), block 128 = 4 warps. warp&1 -> which 32 rows of the panel,
// warp>>1 -> slab parity (the two parities are summed through smem at the end). Within a warp
// the four m8n8k4 "computations" (lane>>2 & 3) take 8 rows each; low/high lane groups take rows
// 0-3 / 4-7 of those 8 (A) and tokens 0-3 / 4-7 (B). NT = token blocks of 8 (M <= 8 -> 1, else 2).
// Partials: part[s][p][row 0..63][col 0..15] exactly as k_pxq4_mma, so k_pxq4_mma_reduce is
// shared. x is staged per 4-slab group into smem with a 136-half row stride: token rows land on
// banks 4r, so the 8-B B-fragment loads are conflict-free and the 4 computations broadcast.
// =============================================================================================
#define PXQ4_M884_GSLAB 1
#define PXQ4_M884_XLD   (PXQ4_M884_GSLAB * PXQ4_M884_PAR * PXQ4_QK + 8)   // 264 halves = 528 B rows
#define PXQ4_M884_PAR   4                                   // slab parities = warps per row half

__device__ __forceinline__ void pxq4_m884(float * d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
                   "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// v2 geometry (2026-09-07 03:20): the first cut (4 warps, one-slab lookahead, 32 KB table)
// was correct (1-2 ULP of SIMT, same rel-L2 vs fp32) but flat at ~150 us on gate_up: with 40 KB of
// smem per 128-thread block only 8 warps fit per SM, and one slab of lookahead (~300 cycles of work)
// does not cover a DRAM round trip. Now 256 threads = 8 warps per block: warp&1 -> row half,
// warp>>1 -> one of FOUR slab parities, TWO slabs of prefetch per parity stream (8 slabs ahead in
// panel order), activation stage shrunk to 2 slabs, and the four parities are summed through
// smem at the end. 46 KB static smem -> 2 blocks/SM on V100 = 16 warps/SM.
// v17 (2026-09-07): FUSE == true folds k_pxq4_mma_reduce's whole job into this
// kernel's epilogue and skips the second launch. It is only ever instantiated for nsplit == 1,
// where the "reduce" is a one-term sum -- see the dispatch below for why that is now the common
// case and pxq4_mma_reduce_fused() for why the result is bit-identical rather than merely close.
template <int NT, bool FUSE>
__global__ void __launch_bounds__(256)
k_pxq4_mma884(const uint8_t * __restrict__ slabs,
              const __half  * __restrict__ anchor,
              const __half  * __restrict__ x,        // [M, K] fp16, M <= 8*NT
              float         * __restrict__ part,     // [nsplit, panels, 64, 16] fp32
              __half        * __restrict__ out,      // [M, N] fp16; FUSE only, else nullptr
              const int kslabs, const int K, const int nsplit, const int panels, const int M)
{
    const int s    = blockIdx.x;
    const int p    = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q    = (lane >> 2) & 3;          // which of the 4 m8n8k4 computations
    const int g    = lane >> 4;                // low (0) / high (1) lane group
    const int l4   = lane & 3;
    const int rhalf = warp & 1;                // rows 0-31 or 32-63 of the panel
    const int par   = warp >> 1;               // slab parity this warp streams (0..3)
    const int rowA  = rhalf * 32 + q * 8 + l4 + 4 * g;             // A fragment row (weights)
    const int rowC0 = rhalf * 32 + q * 8 + (lane & 1) + 4 * g;     // C rows: rowC0 and rowC0 + 2
    const int rowC1 = rowC0 + 2;
    const int tB    = l4 + 4 * g;                                  // B fragment token (0..7)

    __shared__ float subt[16];
    __shared__ float red[2][64][16];                              // two-round parity fold (8 KB)
    // CODE BYTE -> A-FRAGMENT REGISTER, one LDS.32: entry b = (fp16 BOOK[b & 15], fp16 BOOK[b >> 4])
    // packed low/high = the (k, k+1) half2 the mma wants. Replicated 32 ways: lane l always reads
    // bank l whatever the code value (pxq4_mma's trick for its 16-entry book).
    __shared__ uint32_t tab[256][32];

    if (threadIdx.x < 16) subt[threadIdx.x] = pxq4_mma_sub16_g[threadIdx.x];
    {
        const uint4 * src = (const uint4 *)pxq4_mma_tab32_g;   // built by k_pxq4_m884_build_tab
        uint4 * dst = (uint4 *)&tab[0][0];
        for (int e = threadIdx.x; e < 256 * 32 / 4; e += blockDim.x) dst[e] = src[e];
    }
    const float anchC0 = __half2float(anchor[(size_t)p * PXQ4_BM + rowC0]);
    const float anchC1 = __half2float(anchor[(size_t)p * PXQ4_BM + rowC1]);
    const uint8_t * pan = slabs + (size_t)p * kslabs * PXQ4_SLAB_BYTES;

    // v3 (2026-09-07 03:25): NO shared-memory staging of x. v2 staged 4 slabs of activations
    // per block behind two __syncthreads and an unprefetched global load, and that wait -- not the
    // LSU, not the tensor pipe, not DRAM -- pinned the kernel at 330 GB/s. x is 8*NT tokens x K
    // halves (80 KB for gate_up): L2-resident for the whole kernel, and every lane needs exactly
    // its token's 64 contiguous bytes per slab, so each lane reads its own B fragments straight
    // from global (four LDG.128 per slab, one slab ahead) and the main loop has no barrier at all.
    // The 4 m8n8k4 computations read the same 64 B of x: L1 hits.
    const __half * xr[NT];
    bool xok[NT];
#pragma unroll
    for (int tb = 0; tb < NT; ++tb) {
        xok[tb] = (tB + 8 * tb) < M;
        xr[tb]  = x + (size_t)(xok[tb] ? (tB + 8 * tb) : 0) * K;
    }

    float acc[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;

    const int b0 = (int)(((int64_t)kslabs * s)       / nsplit);
    const int b1 = (int)(((int64_t)kslabs * (s + 1)) / nsplit);

    // prefetch: codes + scales two slabs deep, activations one slab deep (L2 latency), per parity stream
    uint4    cqA = make_uint4(0u, 0u, 0u, 0u), cqB = cqA;
    uint32_t sA0 = 0u, sA1 = 0u, sB0 = 0u, sB1 = 0u;
    uint4    xvA[NT][4];
#pragma unroll
    for (int tb = 0; tb < NT; ++tb)
#pragma unroll
        for (int u = 0; u < 4; ++u) xvA[tb][u] = make_uint4(0u, 0u, 0u, 0u);
    {
        const int t0 = b0 + par, t1 = b0 + par + PXQ4_M884_PAR;
        if (t0 < b1) {
            const uint8_t * sl = pan + (size_t)t0 * PXQ4_SLAB_BYTES;
            cqA = *(const uint4 *)(sl + PXQ4_CODE_OFF + PXQ4_CODE_BYTES * rowA); sA0 = sl[rowC0]; sA1 = sl[rowC1];
#pragma unroll
            for (int tb = 0; tb < NT; ++tb) if (xok[tb]) {
                const uint4 * xp = (const uint4 *)(xr[tb] + (size_t)t0 * PXQ4_QK);
#pragma unroll
                for (int u = 0; u < 4; ++u) xvA[tb][u] = xp[u];
            }
        }
        if (t1 < b1) { const uint8_t * sl = pan + (size_t)t1 * PXQ4_SLAB_BYTES;
            cqB = *(const uint4 *)(sl + PXQ4_CODE_OFF + PXQ4_CODE_BYTES * rowA); sB0 = sl[rowC0]; sB1 = sl[rowC1]; }
    }
    __syncthreads();                                               // tab / subt staged

    for (int t = b0 + par; t < b1; t += PXQ4_M884_PAR) {
        const uint4    q4  = cqA;
        const uint32_t sb0 = sA0, sb1 = sA1;
        uint4 xv[NT][4];
#pragma unroll
        for (int tb = 0; tb < NT; ++tb)
#pragma unroll
            for (int u = 0; u < 4; ++u) xv[tb][u] = xvA[tb][u];
        // rotate the code slots and refill the far one; refill the x slot for the next slab
        cqA = cqB; sA0 = sB0; sA1 = sB1;
        const int tn1 = t + PXQ4_M884_PAR, tn2 = t + 2 * PXQ4_M884_PAR;
        if (tn2 < b1) {
            const uint8_t * sl = pan + (size_t)tn2 * PXQ4_SLAB_BYTES;
            cqB = *(const uint4 *)(sl + PXQ4_CODE_OFF + PXQ4_CODE_BYTES * rowA);
            sB0 = sl[rowC0]; sB1 = sl[rowC1];
        }
        if (tn1 < b1) {
#pragma unroll
            for (int tb = 0; tb < NT; ++tb) if (xok[tb]) {
                const uint4 * xp = (const uint4 *)(xr[tb] + (size_t)tn1 * PXQ4_QK);
#pragma unroll
                for (int u = 0; u < 4; ++u) xvA[tb][u] = xp[u];
            }
        }
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const float eff0 = anchC0 * subt[hh ? (sb0 >> 4) : (sb0 & 0xfu)];
            const float eff1 = anchC1 * subt[hh ? (sb1 >> 4) : (sb1 & 0xfu)];
            float c[NT][8];
#pragma unroll
            for (int tb = 0; tb < NT; ++tb)
#pragma unroll
                for (int i = 0; i < 8; ++i) c[tb][i] = 0.f;
            const uint32_t cw0 = hh ? q4.z : q4.x;
            const uint32_t cw1 = hh ? q4.w : q4.y;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint32_t w   = (kk < 2) ? cw0 : cw1;
                const uint32_t by0 = (w >> (16 * (kk & 1)))     & 0xffu;
                const uint32_t by1 = (w >> (16 * (kk & 1) + 8)) & 0xffu;
                const uint32_t a0 = tab[by0][lane];
                const uint32_t a1 = tab[by1][lane];
#pragma unroll
                for (int tb = 0; tb < NT; ++tb) {
                    // halves [16hh + 4kk, +4) of this token's slab: uint4 index 2hh + kk/2, half (kk&1)
                    const uint4 xq = xv[tb][2 * hh + (kk >> 1)];
                    const uint32_t bx = (kk & 1) ? xq.z : xq.x;
                    const uint32_t by = (kk & 1) ? xq.w : xq.y;
                    pxq4_m884(c[tb], a0, a1, bx, by);
                }
            }
#pragma unroll
            for (int tb = 0; tb < NT; ++tb)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    acc[tb][i] = fmaf((i & 2) ? eff1 : eff0, c[tb][i], acc[tb][i]);
        }
    }

    // fold the four parities in two rounds through 8 KB of smem (the 12 KB one-round version put the
    // block over the 48 KB static limit): round 1  par 2,3 -> par 0,1;  round 2  par 1 -> par 0.
    const int rrb = rhalf * 32 + q * 8 + (lane & 1) + 4 * g;           // my C rows: rrb + (i & 2)
    if (par >= 2) {
#pragma unroll
        for (int tb = 0; tb < NT; ++tb)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                red[par - 2][rrb + (i & 2)][(i & 4) + (lane & 2) + (i & 1) + 8 * tb] = acc[tb][i];
    }
    __syncthreads();
    if (par < 2) {
#pragma unroll
        for (int tb = 0; tb < NT; ++tb)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[tb][i] += red[par][rrb + (i & 2)][(i & 4) + (lane & 2) + (i & 1) + 8 * tb];
    }
    __syncthreads();
    if (par == 1) {
#pragma unroll
        for (int tb = 0; tb < NT; ++tb)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                red[0][rrb + (i & 2)][(i & 4) + (lane & 2) + (i & 1) + 8 * tb] = acc[tb][i];
    }
    __syncthreads();
    if (par == 0) {
        float * pb = part + (size_t)(s * panels + p) * PXQ4_BM * 16;
        const int N = panels * PXQ4_BM;
#pragma unroll
        for (int tb = 0; tb < NT; ++tb)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = rrb + (i & 2);
                const int cc = (i & 4) + (lane & 2) + (i & 1) + 8 * tb;
                const float v = acc[tb][i] + red[0][rr][cc];
                if (FUSE) {
                    // The fp16 store k_pxq4_mma_reduce would have done, with the SAME
                    // arithmetic: its single-split loop is `u = 0.f; u += part[...]`, and
                    // `0.f + v` is not a no-op for v == -0.f (it yields +0.f, and the two
                    // have different fp16 bit patterns). __fadd_rn keeps that step explicit
                    // so no compiler flag can quietly delete it and break bit-identity.
                    // `cc` IS the token index m: part is [.., 64 rows, 16 token slots] and
                    // the reduce reads slot m. Slots >= M are padding and were never read.
                    if (cc < M)
                        out[(size_t)cc * N + p * PXQ4_BM + rr] = __float2half_rn(__fadd_rn(0.f, v));
                } else {
                    pb[(size_t)rr * 16 + cc] = v;
                }
            }
    }
}

// v17 (2026-09-07): BUDGET THE REDUCE ROUND, NOT THE CTAS.
//
// The old rule filled a fixed CTA budget of 160 (`want = ceil(160/panels)`), which is a wave
// heuristic for the split kernel alone and ignores that every extra split is a whole extra
// [nsplit, panels, 64, 16] fp32 slice for k_pxq4_mma_reduce to write and read back. At TP2 the
// rank-local panel counts are large enough (gate_up 272) that the budget mostly resolved to 1
// and the cost never showed; at TP4 every panel count halves, the rule starts handing out 2 and
// 3 splits, and the reduce round is what the extra parallelism is spent on.
//
// Measured on one V100 (GPU2), median of 5 x 200 iters, rank-local Qwen3.8-27B shapes, M=8, us:
//
//              panels  kslabs | nsplit 1   nsplit 2   nsplit 3        old rule -> new rule
//   TP4 gate_up   136     160 |   63.15      66.69          -         2 -> 1   -5.3 %
//   TP4 down       80     136 |   37.34      35.31          -         2 -> 2   unchanged
//   TP4 qkv_z      64     160 |   41.48      37.73      48.23         3 -> 2  -21.8 %
//   TP4 out_proj   80      48 |   21.35      18.31          -         2 -> 2   unchanged
//   TP2 gate_up   272     160 |  115.44          -          -         1 -> 1   unchanged
//   TP2 down       80     272 |   62.14      56.12          -         2 -> 2   unchanged
//   TP2 qkv_z     128     160 |   62.28      66.67          -         2 -> 1   -6.6 %
//   TP2 out_proj   80      96 |   29.96      28.29          -         2 -> 2   unchanged
//
// Sum of the four: TP4 168.55 -> 156.40 (-7.2 %), TP2 266.63 -> 262.18 (-1.7 %). Note the shape
// of the optimum: it is 2 at 64-127 panels (nsplit 1 is WORSE on qkv_z, down and out_proj) and 1
// at >= 128 panels, and it is never 3 or more -- pushing the budget the other way (320/480/640)
// was slower on every shape but one. So the rule is a budget of 128 with a hard cap of 2, which
// is the same thing as "one split per 128 panels, never more than two".
//
// PXQ4_MMA884_CTAS, when set, restores the old uncapped budget rule verbatim (=160 reproduces
// every v15/v16 number bit for bit), so the previous behaviour is one env var away.
#define PXQ4_M884_CTAS        128                           // panels per split, default rule
#define PXQ4_M884_NSPLIT_MAX  2                             // reduce rounds we are willing to pay

// v17: the single-split epilogue fusion (see the dispatch in pxq4_launch_mma_f16).
// PXQ4_MMA884_FUSE=0 forces the old kernel + k_pxq4_mma_reduce pair.
static bool pxq4_mma884_fuse_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXQ4_MMA884_FUSE");
        return !(e && *e && atoll(e) == 0);
    }();
    return v;
}
static int pxq4_mma884_cta_budget_override() {
    static const int v = [] {
        const char * e = getenv("PXQ4_MMA884_CTAS");
        return (e && *e) ? atoi(e) : 0;                     // 0 = unset = use the v17 rule
    }();
    return v;
}
static int pxq4_mma884_nsplit(int panels, int kslabs, int nsplit_max) {
    const int ov = pxq4_mma884_cta_budget_override();
    int want, cap;
    if (ov > 0) {                                           // legacy: fill a CTA budget, uncapped
        want = (ov + panels - 1) / panels;
        cap  = 16;
    } else {                                                // v17: budget the reduce round
        want = (PXQ4_M884_CTAS + panels - 1) / panels;
        cap  = PXQ4_M884_NSPLIT_MAX;
    }
    if (want < 1)   want = 1;
    if (want > cap) want = cap;
    if (want > nsplit_max) want = nsplit_max;               // never more partial slots than the arena has
    while (want > 1 && kslabs / want < 2 * PXQ4_M884_PAR) --want;
    return want;
}

// The 32 KB lane-replicated code->half2 table, built ONCE in global memory and copied per block
// with 128-bit stores (8 per thread) instead of 8192 scalar stores + 16 book conversions per block.
__global__ void k_pxq4_m884_build_tab() {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;      // 0 .. 8191
    if (e >= 256 * 32) return;
    const int b = e >> 5;
    const uint16_t lo = __half_as_ushort(__float2half_rn(pxq4_mma_book_g[b & 15]));
    const uint16_t hi = __half_as_ushort(__float2half_rn(pxq4_mma_book_g[b >> 4]));
    pxq4_mma_tab32_g[e] = (uint32_t)lo | ((uint32_t)hi << 16);
}

// v15 (2026-09-07): the m8n8k4 arm is ON BY DEFAULT under the small-M master switch
// on sm_70+. PXQ4_MMA884=0 forces the wmma arm back; PXQ4_MMA884=1 arms it even with the master
// switch off. It is a strict win over wmma at every M measured on a V100 (M=8 us, gate_up/down/
// qkv_z/out_proj: 117/58/68/30 vs wmma 159/87/87/39 vs SIMT 283/139/144/67) and 1-2 fp16 ULP
// from the SIMT path. On sm_60 pxq4_mma_arch_ok() is false and neither arm is ever reached.
static bool pxq4_mma884_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXQ4_MMA884");
        if (e && *e) return atoll(e) != 0;
        const char * sm = getenv("PXA_PXQ4_SMALLM");
        const bool smallm = !(sm && *sm && atoll(sm) == 0);
        return smallm && pxq4_mma_arch_ok();
    }();
    return v;
}

// Route policy for the caller: true when a routed MMA launch will take the m884 arm, which is
// flat in M and therefore wants a floor of 2 rather than the wmma arm's measured 5.
bool pxq4_mma884_active() { return pxq4_mma884_enabled(); }

void pxq4_launch_mma_f16(const uint8_t * slabs, const void * anchor, const void * x,
                         float * part, void * out, int M, int panels, int kslabs,
                         cudaStream_t stream) {
    const int N      = panels * PXQ4_BM;
    const int K      = kslabs * PXQ4_QK;
    const int nsplit = pxq4_mma_nsplit(panels, kslabs);
    if (M < 1 || M > 16) {
        fprintf(stderr, "pxq4: mma M out of range: %d\n", M);
        abort();
    }
    if (pxq4_mma884_enabled()) {
        // raw m8n8k4 path: same partials layout and reduce; its own (smaller) split count, never
        // above the arena's nsplit. The table build happens on the first call after (re)upload --
        // warmup, never inside a CUDA-graph capture.
        if (!g_m884_tab_ready) {
            k_pxq4_m884_build_tab<<<32, 256, 0, stream>>>();
            PXQ4_MMA_CHECK(cudaGetLastError());
            g_m884_tab_ready = true;
        }
        const int ns   = pxq4_mma884_nsplit(panels, kslabs, nsplit);
        // v17 (2026-09-07): SKIP THE REDUCE WHEN THERE IS NOTHING TO
        // REDUCE. The DGX TP4 k=7 profile put k_pxq4_mma_reduce at 1.11 ms/step -- 5.5 % of the
        // 20.1 ms target step -- across 256 launches, exactly one per m884 launch, at 4.3 us
        // each. At nsplit == 1 that whole launch does `u = 0.f + part[m]; out = (half)u`: a
        // full round trip of the [panels, 64, 16] fp32 partials through global memory to add
        // zero to each element. Fix 1 made nsplit == 1 the shape the big layers take (gate_up
        // at TP4, gate_up and qkv_z at TP2), so the epilogue writes the fp16 output itself.
        //
        // The result is BIT-IDENTICAL, not approximately equal: the accumulator values are
        // untouched by the epilogue, an fp32 store/load through `part` is exact, and the
        // epilogue reproduces the reduce's `0.f + v` and `__float2half_rn` verbatim.
        // PXQ4_MMA884_FUSE=0 forces the old two-launch path so the two can be A/B-ed on one
        // build; that is how the identity was gated.
        const bool fuse = (ns == 1) && pxq4_mma884_fuse_enabled();
        const dim3 grid((unsigned)ns, (unsigned)panels);
        if (fuse) {
            if (M <= 8)
                k_pxq4_mma884<1, true><<<grid, 256, 0, stream>>>(
                    slabs, (const __half *)anchor, (const __half *)x, part, (__half *)out,
                    kslabs, K, ns, panels, M);
            else
                k_pxq4_mma884<2, true><<<grid, 256, 0, stream>>>(
                    slabs, (const __half *)anchor, (const __half *)x, part, (__half *)out,
                    kslabs, K, ns, panels, M);
            PXQ4_MMA_CHECK(cudaGetLastError());
            return;
        }
        if (M <= 8)
            k_pxq4_mma884<1, false><<<grid, 256, 0, stream>>>(
                slabs, (const __half *)anchor, (const __half *)x, part, nullptr,
                kslabs, K, ns, panels, M);
        else
            k_pxq4_mma884<2, false><<<grid, 256, 0, stream>>>(
                slabs, (const __half *)anchor, (const __half *)x, part, nullptr,
                kslabs, K, ns, panels, M);
        PXQ4_MMA_CHECK(cudaGetLastError());
        k_pxq4_mma_reduce<<<(unsigned)panels, PXQ4_BM, 0, stream>>>(part, (__half *)out, ns, panels, M, N);
        PXQ4_MMA_CHECK(cudaGetLastError());
        return;
    } else {
        k_pxq4_mma<<<dim3((unsigned)nsplit, (unsigned)panels), 128, 0, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, part, kslabs, K, nsplit, panels, M);
    }
    PXQ4_MMA_CHECK(cudaGetLastError());
    k_pxq4_mma_reduce<<<(unsigned)panels, PXQ4_BM, 0, stream>>>(
        part, (__half *)out, nsplit, panels, M, N);
    PXQ4_MMA_CHECK(cudaGetLastError());
}
