// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxq4-mmq80.cu -- the sm_80+ prefill GEMM route for 4-bit E16 panel weights; contract and lever in pxq4-mmq80.cuh.
//
// SHAPE. One MMQ type instance (GGML_TYPE_PXQ4) on the stock tile machinery of mmq.cuh: mmq_y = 128 weight rows
// (two 64-row panels) x mmq_x tokens per block, 256 K per iteration = 8 slabs, one slab per warp in the loader.
// The x tile reuses the per-16-scale layout (MMQ_MMA_TILE_X_K_Q3_K: 64 ints of int8 + 16 fp32 scales per row),
// the y tile is the stock q8_1 D4 layout (one fp32 scale per 32 activations).
//
// LOADER. Warp w loads slab kbx0 + w for 32 consecutive rows per pass (4 passes): 32 x 16 B of codes = 512 B
// contiguous, 32 sub bytes and 32 anchors contiguous. Each 16 B code row decodes to 32 int8 with three byte
// permutes per 4 codes (the 16-entry s8 book lives in four registers). Slabs at or past K/32 (K % 256 != 0)
// are stored as zeros, so the tail never reads past a panel.
//
// EPILOGUE. The int32 accumulators start at the bit pattern of 1.5 * 2^23; for |dot| < 2^22 the result read as
// fp32 is 12582912 + dot EXACTLY (a 16-element group is at most 16 * 127 * 127 = 258064), so one FADD recovers
// the integer instead of an int->float conversion (16/clk/SM on GA10x: the conversions, not the tensor core,
// would set the rate otherwise). The value equals the conversion bit for bit.

#include "pxq4-mmq80.cuh"
#include "../mmq.cuh"
#include "../quantize.cuh"
#include "pxa-enhance.cuh"
#include "../../../include/ggml-pxq6-tables.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// the frozen q3-s8 book, entry i at byte (i & 3) of word i >> 2:
// {-125,-93,-71,-53,-38,-25,-12,0,11,22,33,46,60,76,97,127} = rint(PXQ6_BOOK_INIT * 127) (checked on the host)
#define PXQ4M80_B0 0xCBB9A383u
#define PXQ4M80_B1 0x00F4E7DAu
#define PXQ4M80_B2 0x2E21160Bu
#define PXQ4M80_B3 0x7F614C3Cu

#define PXQ4M80_MAGIC_I 0x4B400000          // fp32 1.5 * 2^23
#define PXQ4M80_MAGIC_F 12582912.0f

// global, not __constant__: each lane of a loader warp indexes its own row's sub byte, and divergent indices into the
// constant bank replay once per distinct address (up to 16x); the 64 B table sits in one L1 line, read with __ldg
static __device__ const float pxq4m80_sub16[16] = PXQ6_SUB16_INIT;

#if defined(INT8_MMA_AVAILABLE) && __CUDA_ARCH__ >= PXA_CC_AMPERE
#define PXQ4M80_DEVICE 1
#endif

// four 4-bit codes (nibbles 0-3 of sh) -> four int8 book entries; bit 3 of each nibble picks the upper words
static __device__ __forceinline__ int pxq4m80_lut(const uint32_t sh) {
    const uint32_t lo = __byte_perm(PXQ4M80_B0, PXQ4M80_B1, sh);
    const uint32_t hi = __byte_perm(PXQ4M80_B2, PXQ4M80_B3, sh);
    return (int) __byte_perm(lo, hi, 0x3210u | ((sh & 0x8888u) >> 1));
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_pxq4_mma80(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {
#ifdef PXQ4M80_DEVICE
    static_assert(nwarps*32 == MMQ_ITER_K, "one warp per 32-wide slab");
    static_assert(mmq_y % WARP_SIZE == 0, "whole passes");

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*WARP_SIZE);

    // stride = bytes per row inside a panel = (128 + 1088*nkb)/64 = 2 + 17*nkb
    const int     nkb          = (stride - 2) / 17;
    const int     kb           = kbx0 + threadIdx.y;
    const bool    live         = kb < nkb;
    const int64_t panel_stride = (int64_t) 64 * stride;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
        int i = i0 + threadIdx.x;

        if (need_check) {
            i = min(i, i_max);
        }

        const uint8_t * panel = (const uint8_t *) x + (i >> 6) * panel_stride;
        const int       r     = i & 63;

        int4   q0 = make_int4(0, 0, 0, 0);
        int4   q1 = make_int4(0, 0, 0, 0);
        float2 d  = make_float2(0.0f, 0.0f);

        if (live) {
            const uint8_t * slab = panel + PXQ6_HDR_BYTES + (int64_t) kb*PXQ6_SLAB_BYTES;
            const uint4     cw   = __ldg((const uint4 *) (slab + 64 + 16*r));
            const int       sb   = __ldg(slab + r);
            const float     a    = __half2float(__ldg((const half *) panel + r)) * (1.0f/127.0f);

            d.x = a * __ldg(pxq4m80_sub16 + (sb & 0xf));   // elements  0..15 of the slab
            d.y = a * __ldg(pxq4m80_sub16 + (sb >>  4));   // elements 16..31

            q0.x = pxq4m80_lut(cw.x); q0.y = pxq4m80_lut(cw.x >> 16);
            q0.z = pxq4m80_lut(cw.y); q0.w = pxq4m80_lut(cw.y >> 16);
            q1.x = pxq4m80_lut(cw.z); q1.y = pxq4m80_lut(cw.z >> 16);
            q1.z = pxq4m80_lut(cw.w); q1.w = pxq4m80_lut(cw.w >> 16);
        }

        int4 * dq = (int4 *) (x_qs + i*MMQ_MMA_TILE_X_K_Q3_K + 8*threadIdx.y);
        dq[0] = q0;
        dq[1] = q1;
        *(float2 *) (x_df + i*MMQ_MMA_TILE_X_K_Q3_K + 2*threadIdx.y) = d;
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(x_tile); GGML_UNUSED(kbx0); GGML_UNUSED(i_max); GGML_UNUSED(stride);
    NO_DEVICE_CODE;
#endif // PXQ4M80_DEVICE
}

// vec_dot_q8_0_16_q8_1_mma's tiling (per-16 x scales, per-32 y scales) with the magic-number epilogue
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_pxq4_q8_1_mma80(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef PXQ4M80_DEVICE
    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_A_I16K8 mma_A_K8;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity   = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx           = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + WARP_SIZE*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A A[ntx][8];
    float dA[ntx][mma_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            const int k0 = k00 + k01;

            ((mma_A_K8 *) A[n])[k01/8].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q3_K + k0, MMQ_MMA_TILE_X_K_Q3_K);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*MMQ_MMA_TILE_X_K_Q3_K + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            mma_B B[2];
            float dB[mma_C::ne/2];

            B[0].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),        MMQ_TILE_Y_K);
            B[1].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    C[0].x[l] = PXQ4M80_MAGIC_I;
                    C[1].x[l] = PXQ4M80_MAGIC_I;
                }
                C[0].mma_K4(A[n][k01/4 + 0], B[0]);
                C[1].mma_K4(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    const float c0 = __int_as_float(C[0].x[l]) - PXQ4M80_MAGIC_F;
                    const float c1 = __int_as_float(C[1].x[l]) - PXQ4M80_MAGIC_F;
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += dB[l%2]*(c0*dA[n][l/2][k01/4 + 0] + c1*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k00);
    NO_DEVICE_CODE;
#endif // PXQ4M80_DEVICE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_pxq4_q8_1_dp4a80(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k00);
    NO_DEVICE_CODE;   // the route runs only where the int8 tensor-core tile exists
}

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_PXQ4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_pxq4_mma80<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_pxq4_q8_1_mma80<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_pxq4_q8_1_dp4a80<mmq_x, mmq_y, nwarps>;
};

DECL_MMQ_CASE(GGML_TYPE_PXQ4);

// ------------------------------------------------------------------------------------------------------------------
// host side
// ------------------------------------------------------------------------------------------------------------------

static bool pxq4m80_lever() {
    static const bool v = [] {
        // OPT-IN (default off): measured slower than the fp16 route on sm_86 (rig RTX 3080 Ti, 10-22: pp512 N/O
        // 0.918 nano4b PXQ4 / 0.925 qwen4b PXQN4, pp4096 0.941 / 0.975), so it stays a lever until the tile beats it
        const char * e = getenv("PXA_PXQ_MMQ80");
        return e ? atoi(e) != 0 : false;
    }();
    return v;
}

// the s8 words above must be rint(book * 127) of the frozen book the PXQ4 kernels read; an env table override
// (PXA_PXQ6_BOOK / PXA_PXQ6_SUB) changes what a PXQ4 weight means, so the route then declines for PXQ4
static bool pxq4m80_book_ok() {
    static const bool v = [] {
        static const float book[16] = PXQ6_BOOK_INIT;
        const uint32_t words[4] = {PXQ4M80_B0, PXQ4M80_B1, PXQ4M80_B2, PXQ4M80_B3};
        for (int c = 0; c < 16; ++c) {
            const int q = (int) lrintf(book[c] * 127.0f);
            const int w = (int) (int8_t) ((words[c >> 2] >> (8*(c & 3))) & 0xffu);
            if (q != w) {
                fprintf(stderr, "PXA_PXQ_MMQ80: s8 book entry %d = %d, expected %d -- route disabled\n", c, w, q);
                return false;
            }
        }
        return true;
    }();
    return v;
}

static bool pxq4m80_tables_overridden() {
    static const bool v = getenv("PXA_PXQ6_BOOK") != nullptr || getenv("PXA_PXQ6_SUB") != nullptr;
    return v;
}

bool ggml_cuda_pxq4_mmq80_on(int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= PXA_CC_AMPERE && ggml_cuda_highest_compiled_arch(cc) >= PXA_CC_AMPERE &&
           pxq4m80_lever() && pxq4m80_book_ok();
}

int ggml_cuda_pxq4_mmq80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                         ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_pxq4_mmq80_on(cc)) return -1;
    if (src0->type != GGML_TYPE_PXQ4 && src0->type != GGML_TYPE_PXQN4) return -1;   // the 4-bit E16 panel types
    if (src0->type == GGML_TYPE_PXQ4 && pxq4m80_tables_overridden()) return -1;

    const int64_t K  = src0->ne[0];
    const int64_t R  = src0->ne[1];
    const int64_t ny = src1->ne[1];

    // dense 2D, f32 in/out, rows contiguous, the default precision, the whole tensor (no views into panels)
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return -1;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;
    if (src0->view_src != nullptr || src0->data == nullptr) return -1;
    if (src1->ne[0] != K || dst->ne[0] != R || dst->ne[1] != ny) return -1;
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return -1;
    if (dst->op_params[0] != GGML_PREC_DEFAULT) return -1;
    if (K % PXQ6_QK != 0 || R % PXQ6_BM != 0) return -1;
    if (ny <= 8) return -1;                       // the decode / verify window keeps its kernels
    if (K > INT32_MAX/4 || R > INT32_MAX/4 || ny > INT32_MAX/4) return -1;

    static std::atomic<uint32_t> logged{0};
    const uint32_t bit = 1u << (ctx.device & 31);
    if (!(logged.fetch_or(bit) & bit)) {
        fprintf(stderr, "PXA_PXQ_MMQ80: dev%d sm_%d 4-bit prefill GEMM route on (opt-in PXA_PXQ_MMQ80=1; unset or 0 = the fp16 route)\n",
                ctx.device, cc/10);
    }

    cudaStream_t stream = ctx.stream();

    const int64_t kpad   = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t  nbq8   = (size_t) ny*kpad*sizeof(block_q8_1)/QK8_1 + get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> q8(ctx.pool(), nbq8);

    quantize_mmq_q8_1_cuda((const float *) src1->data, q8.get(), K, ny, 1, kpad, GGML_TYPE_PXQ4, stream);
    CUDA_CHECK(cudaGetLastError());

    // stride01 = bytes per row inside a panel, so x + stride01*it*128 lands on panel 2*it (the MMQ tile origin)
    const int64_t stride01 = (PXQ6_HDR_BYTES + (K/PXQ6_QK)*PXQ6_SLAB_BYTES) / PXQ6_BM;

    const mmq_args args = {(const char *) src0->data, q8.get(), (float *) dst->data, K, R, stride01, kpad, ny, ny, R};
    mul_mat_q_case<GGML_TYPE_PXQ4>(ctx, args, stream);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
