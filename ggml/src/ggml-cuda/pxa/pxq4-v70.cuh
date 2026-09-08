// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxq4-v70.cuh -- V70: register-direct PXQ4 prefill GEMM on the native sm_70
//                 mma.sync.aligned.m8n8k4 tile.
//
// WHY THIS EXISTS
//   The incumbent Volta prefill route for a PXQ 2D MUL_MAT is
//   k_pxq6_dequant_matrix -> fp16 pool scratch -> cublasGemmEx (ggml-cuda.cu, the
//   ggml_cuda_op_mul_mat_cublas fp16 branch). That route writes and re-reads the whole
//   weight matrix as fp16 once per ubatch: measured at 31,983 dequants / 2,910 GiB of
//   scratch traffic in one bench run, and ablating it entirely is worth +6.9% prefill at
//   fill 3121 and +3.8% at fill 20801 on 2x V100 (layer-major prefill work, 2026-09-02,
//   commit 9e7b129432). Fusing the decode into the GEMM realises that exactly instead of
//   as a cache ceiling.
//
//   The prior native attempt (K7, the 2026-09-02 native-GEMM attempt, pxq4-mma70.cuh) proved
//   the fused decode is nearly free (0.3-6 TFLOPS vs a pre-dequantized control) and that
//   the LOSS was the tile: nvcuda::wmma m16n16k16 is software-decomposed on sm_70 and tops
//   out at 53-67 TFLOPS where cuBLAS reaches 83-93. K7's own postmortem names the fix --
//   "a CUTLASS-grade mma.m8n8k4 tile with swizzled operand layouts and register-level
//   fragment pipelining, not a bigger wmma tile". That primitive is already in-tree:
//   volta-mma/mma-ml.cuh issues mma.sync.aligned.m8n8k4 straight from register fragments.
//
//   V70 therefore decodes PXQ4 nibbles DIRECTLY INTO the A register fragment. The A operand
//   never touches shared memory and is never materialised as fp16 anywhere: no dequant
//   kernel, no fp16 scratch buffer, no smem staging round-trip, no 60 KB carveout and so no
//   cudaFuncSetAttribute opt-in (unlike K7).
//
// NUMERICS -- ONE ROUNDING, BYTE-IDENTICAL WEIGHTS
//   eff = fp32(anchor_fp16) * SUB16[nibble]        (hoisted, once per 16-elem sub-block)
//   w   = __float2half_rn(eff * fp32(book[code]))  (one rounding, exactly the reference)
//   This is bit-for-bit the value k_pxq6_dequant_matrix (pxq6.cuh) writes into the cuBLAS
//   scratch -- test #1 memcmps it. So the ENTIRE numeric difference against the incumbent
//   route is tensor-core accumulation order, which puts V70 in K7's measured parity class
//   (max rel-to-peak 1.3e-5). NOT bit-exact vs cuBLAS: gate class is K6/K7's -- G3 logprob
//   parity + a ppl regate. A temp-0 sha is NOT a valid gate here.
//
// VERIFIED FRAGMENT FOUNDATION (mma-ml.cuh, do not re-derive)
//   A = tile<32,4,half2>                            ne=4, get_i(l)=threadIdx.x, get_j(l)=l
//   B = tile<8,4,half2,I_MAJOR_MIRRORED>            ne=4, get_i(0)=(tid/16)*4+(tid%4), get_j(l)=l
//   D = tile<32,8,float>                            ne=8, get_i(l)=(l&2)+(tid&~2),
//                                                         get_j(l)=(tid&2)+(l&5)
//   mma(D,A,B) issues TWO m8n8k4 PTX ops (A.x[0:1]/B.x[0:1] then A.x[2:3]/B.x[2:3]), so one
//   mma() call consumes K=8 halves. A 32-elem PXQ slab is exactly 4 mma K-steps.
//
//   TRAP: the fragments are addressed through get_i/get_j, NEVER through a hand-rolled
//   threadIdx.x expression. GGML_CUDA_MMA_NO_VOLTA_PERM redefines get_i and a hand-rolled
//   index would then silently read the wrong weight rows. Under either perm the warp still
//   covers the same contiguous 32 rows, so coalescing is unaffected either way.
//   TRAP: blockDim.x MUST be 32 -- the tiles read threadIdx.x as the lane id.
//
// PXQ4 CORE-TIER ADDRESSING (pxq6_pol_p6, one panel = 64 rows)
//   panel  = 128 B fp16 row anchors + kslabs * 1088 B slabs
//   slab   = 64 B scale SoA (byte r = row r's two SUB16 nibbles: lo = K 0-15, hi = K 16-31)
//            + 64 rows x 16 B nibble code rows
//   one LDG.128 at slab+64+16*r   = that row's 32 codes = 4 mma K-steps
//   one LDG.U8  at slab+r         = that row's two sub-scale nibbles
//   the fp16 anchor is loop-invariant and is hoisted out of the K loop entirely.
//
// PREFETCH: sm_70 has no cp.async (see fattn-mma-ml.cuh). One-slab-deep manual register
//   double-buffer, exactly K7's scheme -- 512 tensor clocks of the warp's own work in
//   flight, enough to cover an HBM round trip without leaning on occupancy.
//
// GATE: PXA_PXQ_GEMM_V70 (alias PXA_VOLTA_PXQ_GEMM / PXA_DEQUANT_ONCE), DEFAULT OFF.
//       PXA_PXQ_GEMM_V70_CFG   tile config, default 0
//       PXA_PXQ_GEMM_V70_MIN_NY  default 128 (inert at the production -ub 2048)
//       PXA_PXQ_GEMM_V70_SKIP  "R:K,R:K,..." per-shape denylist (the route table's manual arm)
//
// ---- MEASURED: LOSS at every ubatch tested. DEFAULT OFF, and do NOT re-run this blind. ------
// 2026-09-03, 2x V100-PCIE-16GB, Qwen3.8-27B-PXQ4, ON confirmed in the server log
// ("PXA_PXQ_GEMM_V70: ON"), prefill t/s at fill 3121 / 20801:
//     ub 2048   OFF 876 / 782      ON 417 / 398        -52.4% / -49.1%
//     ub  512   OFF 623 / 603      ON 370 / 362        -40.7% / -39.9%
//     ub  512, PXA_PXQ_GEMM_V70_CFG=2                  -44.6% / -44.2%   (worse than CFG=0)
// Correctness is not the issue: the decode is byte-identical to k_pxq6_dequant_matrix (test #1
// memcmp) and the parity class is K7's (max rel-to-peak 1.3e-5). The tile is. Register-direct
// m8n8k4 runs at 12.5-18.75% occupancy by design (sm_70 has no cp.async, so the manual register
// double-buffer is the only latency tool) and cannot reach cuBLAS HGEMM's rate, while the dequant
// pass it removes is only ~12% of the prefill wall. No sub-config recovers it.
// Independently confirmed from the other direction by the dp4a tile (pxq4-mmq.cuh): -49% on the
// same cell. Two structurally different PXQ4-native sm_70 prefill GEMMs both land at half the
// incumbent -- replacing the dense GEMM is not where Volta prefill is won.
// KEPT because it is correct and because the conclusion inverts on any card whose tensor-core
// int8/low-precision rate beats its fp16 rate.
#pragma once

#include "pxq6.cuh"                  // pxq6_pol_p6, pxq6_book_g / pxq6_sub16_g, the reference dequant
#include "volta-mma/mma-ml.cuh"      // pxa_volta_mma:: m8n8k4 register tiles

#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------------------------
// tile configuration
//   WM x WN warps per block; each warp owns MW A-tiles (32 weight rows each) x NW B-tiles
//   (8 tokens each).  BM = WM*MW*32 weight rows, BN = WN*NW*8 tokens per block.
// ---------------------------------------------------------------------------------------------
template <int WM_, int WN_, int MW_, int NW_>
struct pxq4_v70_cfg {
    static constexpr int WM = WM_, WN = WN_, MW = MW_, NW = NW_;
    static constexpr int NWARPS  = WM*WN;
    static constexpr int NTHREAD = 32*NWARPS;
    static constexpr int BM = WM*MW*32;
    static constexpr int BN = WN*NW*8;
};

//  V0 (default): warp 32x64, block  64x128, 128 thr, 64 acc regs/thread, B straight from L1
//  V1          : warp 64x32, block 128x128, 128 thr, halves B traffic per MAC, needs R%128==0
//  V2          : warp 32x32, block  64x64,  128 thr, register relief valve
//  V3          : warp 32x32, block 128x64,  256 thr, needs R%128==0
typedef pxq4_v70_cfg<2,2,1,8> pxq4_v70_cfg0;
typedef pxq4_v70_cfg<2,2,2,4> pxq4_v70_cfg1;
typedef pxq4_v70_cfg<2,2,1,4> pxq4_v70_cfg2;
typedef pxq4_v70_cfg<4,2,1,4> pxq4_v70_cfg3;

// L2 rasterisation group: how many TOKEN tiles a wave of blocks works on at once while the
// weight-row tile advances slowest. GN consecutive blocks share one weight tile.
//
// THIS MAPPING IS A ONE-LINE TRAP, so state the arithmetic. Shape (5120,17408) at ny=2048,
// BM=64/BN=128: npm=272 row tiles, npn=16 token tiles. Weights are 46 MB, activations 20 MB.
//   - iterate token-tile fastest (naive)   : each weight tile re-read npn=16 times -> 736 MB
//   - iterate row-tile fastest, GN=1       : same 736 MB, activation tile (1.3 MB) stays in L2
//   - GN=4 (this)                          : weight stream re-read npn/GN = 4 times -> 184 MB,
//                                            GN*1.3 = 5.2 MB of activations resident in the
//                                            6 MB L2. 4x less HBM traffic than either.
// Getting it backwards turns a 20 MB activation read into ~5.4 GB; test #0 prints the
// implied traffic so the mapping is checked, not assumed.
#define PXQ4_V70_GN 4

// ---------------------------------------------------------------------------------------------
// THE decode. One code word (4 bytes = 8 codes = 8 halves of K) straight into 4 half2 register
// slots of an A fragment. ONE rounding: __float2half_rn(eff * book[c]) -- byte-for-byte the
// value k_pxq6_dequant_matrix writes. The book gather is SHFL.IDX against a lane-register image
// of pxq6_book_g, so an env-overridden book stays consistent and the LSU stays free.
// Called by BOTH the GEMM kernel and the byte-identity dump kernel, so test #1 checks the code
// that actually runs, not a transcription of it.
// ---------------------------------------------------------------------------------------------
template <class POL>
static __device__ __forceinline__ void pxq4_v70_decode_word(const uint32_t w, const float e,
                                                            const float bv, half2 * __restrict__ dst) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {                        // byte l == code-row byte, elems 2b, 2b+1
        const int   byte = (w >> (8*l)) & 0xff;
        const float b0 = __shfl_sync(0xffffffffu, bv, byte & 0xf);
        const float b1 = __shfl_sync(0xffffffffu, bv, byte >> 4);
        dst[l] = __floats2half2_rn(e*b0, e*b1);
    }
}

// Byte-identity harness (test #1): decode the whole matrix through pxq4_v70_decode_word and emit
// it in k_pxq6_dequant_matrix's exact [R][K] fp16 layout so the two can be memcmp'd.
// grid = R/32 blocks of one warp; never launched by the engine, so it is compiled ONLY for
// tools/pxq4-v70-test.cu (which defines PXQ4_V70_SELFTEST) and never emitted into libggml.
// It shares pxq4_v70_decode_word with the GEMM, so test #1 still checks the code that runs.
#ifdef PXQ4_V70_SELFTEST
template <class POL>
static __global__ void k_pxq4_v70_decode_dump(const uint8_t * __restrict__ W, half * __restrict__ y,
                                              const int K, const int kslabs) {
#ifdef VOLTA_MMA_AVAILABLE
    const int lane = threadIdx.x;
    const int grp  = blockIdx.x;                          // 32-row group
    const int row  = grp*32 + lane;
    const uint8_t * panel = W + (size_t)(grp >> 1)*((size_t)POL::HDR + (size_t)kslabs*POL::SLAB);
    const int rr = ((grp & 1)*32) + lane;
    const float anch = __half2float(((const half *)panel)[rr]);
    const float bv = pxq6_book_g [lane & 15];
    const float sv = pxq6_sub16_g[lane & 15];
    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = panel + POL::HDR + (size_t)kb*POL::SLAB;
        const uint4 q = *(const uint4 *)(slab + POL::CODE_OFF + rr*POL::CODE_BYTES);
        const uint8_t sc = slab[rr];
        float eff[2];
        eff[0] = anch * __shfl_sync(0xffffffffu, sv, sc & 0xf);
        eff[1] = anch * __shfl_sync(0xffffffffu, sv, sc >> 4);
        for (int s = 0; s < 4; ++s) {
            half2 h[4];
            pxq4_v70_decode_word<POL>(((const uint32_t *)&q)[s], eff[s >> 1], bv, h);
            for (int l = 0; l < 4; ++l) {
                y[(size_t)row*K + kb*PXQ6_QK + 8*s + 2*l + 0] = __low2half (h[l]);
                y[(size_t)row*K + kb*PXQ6_QK + 8*s + 2*l + 1] = __high2half(h[l]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, y, K, kslabs);
    NO_DEVICE_CODE;
#endif
}
#endif // PXQ4_V70_SELFTEST

// ---------------------------------------------------------------------------------------------
// k_pxq4_gemm_v70 -- C[ny][R] (f32, row-major over R) = X[ny][K] (f16) @ W[R][K] (PXQ4)^T
//   grid  = dim3(R/BM, ceil(ny/BN)); block = dim3(32, NWARPS)
// ---------------------------------------------------------------------------------------------
template <class POL, class CFG>
__launch_bounds__(CFG::NTHREAD, 1)
static __global__ void k_pxq4_gemm_v70(
        const uint8_t * __restrict__ W, const half * __restrict__ X, float * __restrict__ C,
        const int R, const int K, const int ny, const int kslabs) {
#ifdef VOLTA_MMA_AVAILABLE
    using namespace pxa_volta_mma;
    typedef tile<32, 4, half2>                                  tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>    tile_B;
    typedef tile<32, 8, float>                                  tile_D;

    static_assert(POL::CODE_WORDS == 4 && POL::NEFF == 2 && POL::CODE_BYTES == 16,
                  "V70 v1 covers the PXQ4 core tier only (PXQ4HQ / PXQ6R are later policies)");
    static_assert(tile_A::ne == 4 && tile_B::ne == 4 && tile_D::ne == 8, "unexpected Volta tile ne");

    // ---- grouped rasterisation (see PXQ4_V70_GN above) ----
    const int npm = gridDim.x, npn = gridDim.y;
    const int pid = blockIdx.x + npm*blockIdx.y;
    const int in_group = PXQ4_V70_GN * npm;
    const int first_n  = (pid / in_group) * PXQ4_V70_GN;
    const int gsz      = min(npn - first_n, PXQ4_V70_GN);
    const int r        = pid % in_group;                 // < gsz*npm for the (only) partial group
    const int pid_n    = first_n + (r % gsz);
    const int pid_m    = r / gsz;

    const int warp = threadIdx.y;
    const int wm   = warp % CFG::WM;
    const int wn   = warp / CFG::WM;

    // lane-register images of the (possibly env-overridden) tables: the book gather becomes
    // SHFL.IDX instead of an LSU load, which is where the decode budget comes from.
    const float bv = pxq6_book_g [threadIdx.x & 15];
    const float sv = pxq6_sub16_g[threadIdx.x & 15];

    const int lane_i = tile_A::get_i(0);                 // NEVER threadIdx.x -- see the perm trap
    const int lane_b = tile_B::get_i(0);

    const size_t pstride = (size_t)POL::HDR + (size_t)kslabs*POL::SLAB;

    // per A-tile row addressing, all loop-invariant
    const uint8_t * panel[CFG::MW];
    int             rr   [CFG::MW];                      // row inside the 64-row panel
    float           anch [CFG::MW];
#pragma unroll
    for (int mi = 0; mi < CFG::MW; ++mi) {
        const int rbase = pid_m*CFG::BM + (wm*CFG::MW + mi)*32;
        panel[mi] = W + (size_t)(rbase >> 6)*pstride;    // a 32-row group never straddles a panel
        rr   [mi] = (rbase & 63) + lane_i;
        anch [mi] = __half2float(((const half *)panel[mi])[rr[mi]]);
    }

    const int tok0 = pid_n*CFG::BN + wn*CFG::NW*8;       // this warp's first token
    tile_D D[CFG::MW][CFG::NW];                          // zero-initialised by the tile ctor

    // ---- manual one-slab register double buffer (no cp.async on sm_70) ----
    uint4 qc[CFG::MW], qn[CFG::MW];
    uint8_t sc[CFG::MW], sn[CFG::MW];
#pragma unroll
    for (int mi = 0; mi < CFG::MW; ++mi) {
        const uint8_t * slab = panel[mi] + POL::HDR;
        qc[mi] = *(const uint4 *)(slab + POL::CODE_OFF + rr[mi]*POL::CODE_BYTES);
        sc[mi] = slab[rr[mi]];
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        if (kb + 1 < kslabs) {
#pragma unroll
            for (int mi = 0; mi < CFG::MW; ++mi) {
                const uint8_t * slab = panel[mi] + POL::HDR + (size_t)(kb+1)*POL::SLAB;
                qn[mi] = *(const uint4 *)(slab + POL::CODE_OFF + rr[mi]*POL::CODE_BYTES);
                sn[mi] = slab[rr[mi]];
            }
        }

        // eff = anchor * SUB16[nibble], hoisted once per 16-elem sub-block (2 SHFL per slab)
        float eff[CFG::MW][2];
#pragma unroll
        for (int mi = 0; mi < CFG::MW; ++mi) {
            eff[mi][0] = anch[mi] * __shfl_sync(0xffffffffu, sv, sc[mi] & 0xf);
            eff[mi][1] = anch[mi] * __shfl_sync(0xffffffffu, sv, sc[mi] >> 4);
        }

#pragma unroll
        for (int s = 0; s < 4; ++s) {                    // 4 mma K-steps of 8 halves per slab
            tile_A A[CFG::MW];
#pragma unroll
            for (int mi = 0; mi < CFG::MW; ++mi) {
                const uint32_t w = ((const uint32_t *)&qc[mi])[s];
                const float    e = eff[mi][s >> 1];      // halves 8s..8s+7: s<2 -> elems 0-15
                pxq4_v70_decode_word<POL>(w, e, bv, A[mi].x);
            }
            const int k0 = kb*PXQ6_QK + 8*s;
#pragma unroll
            for (int nj = 0; nj < CFG::NW; ++nj) {
                tile_B B;
                const int tok = tok0 + nj*8 + lane_b;
                if (tok < ny) {
                    ggml_cuda_memcpy_1<16>(B.x, (const half2 *)(X + (size_t)tok*K + k0));
                } else {                                 // N tail: zero-fill, run the MMA full width
                    B.x[0] = B.x[1] = B.x[2] = B.x[3] = make_half2(0.0f, 0.0f);
                }
#pragma unroll
                for (int mi = 0; mi < CFG::MW; ++mi) {
                    mma(D[mi][nj], A[mi], B);
                }
            }
        }

        if (kb + 1 < kslabs) {                           // qn/sn are only written under the same
#pragma unroll                                           // predicate; copying them otherwise reads
            for (int mi = 0; mi < CFG::MW; ++mi) {       // uninitialised registers (and qc is dead)
                qc[mi] = qn[mi]; sc[mi] = sn[mi];
            }
        }
    }

    // ---- epilogue: direct f32 stores, predicated on the token tail ----
#pragma unroll
    for (int mi = 0; mi < CFG::MW; ++mi) {
        const int rbase = pid_m*CFG::BM + (wm*CFG::MW + mi)*32;
#pragma unroll
        for (int nj = 0; nj < CFG::NW; ++nj) {
#pragma unroll
            for (int l = 0; l < tile_D::ne; ++l) {
                const int row = rbase + tile_D::get_i(l);
                const int tok = tok0 + nj*8 + tile_D::get_j(l);
                if (tok < ny) C[(size_t)tok*R + row] = D[mi][nj].x[l];
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, X, C, R, K, ny, kslabs);
    NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------
// host side: env gates, route table, launcher
// ---------------------------------------------------------------------------------------------
static bool pxa_pxq_gemm_v70() {
    static const bool on = [] {
        const char * e = getenv("PXA_PXQ_GEMM_V70");
        if (!e) e = getenv("PXA_VOLTA_PXQ_GEMM");        // aliases named in the standing order
        if (!e) e = getenv("PXA_DEQUANT_ONCE");
        const bool v = e ? atoi(e) != 0 : false;         // DEFAULT OFF until Measure flips it
        if (v) fprintf(stderr, "PXA_PXQ_GEMM_V70: ON (sm_70 register-direct PXQ4 m8n8k4 prefill GEMM; "
                               "NOT bit-exact vs cuBLAS -- G3 logprob parity + ppl regate required)\n");
        return v;
    }();
    return on;
}

static int pxa_pxq_v70_cfg() {
    static const int v = [] { const char * e = getenv("PXA_PXQ_GEMM_V70_CFG"); return e ? atoi(e) : 0; }();
    return v;
}

static int pxa_pxq_v70_min_ny() {
    static const int v = [] { const char * e = getenv("PXA_PXQ_GEMM_V70_MIN_NY"); return e ? atoi(e) : 128; }();
    return v;
}

// ROUTE TABLE. The plan's contract: any (shape, M-bucket) that does not beat the FULL incumbent
// route (k_pxq6_dequant_matrix + cublasGemmEx, dequant included) is marked ROUTE_CUBLAS, so the
// worst case is zero regression rather than K7's -11..-36%. The static rows below are populated
// from test #5 of tools/pxq4-v70-test.cu at the Measure phase; until then the table is empty and
// the whole lane is held off by PXA_PXQ_GEMM_V70 defaulting to 0. PXA_PXQ_GEMM_V70_SKIP is the
// manual arm: "R:K,R:K,..." denylists shapes without a rebuild.
struct pxq4_v70_route { int R, K, min_ny, max_ny; };
// The leading { 0, 0, 0, 0 } is a SENTINEL, not a rule: a real R is never 0, and a zero-length
// array is a GNU extension rather than ISO C++. Add measured rows after it.
static const pxq4_v70_route pxq4_v70_route_deny[] = {
    { 0, 0, 0, 0 },
    // { R, K, min_ny, max_ny }  -- populated at Measure from the test #5 TFLOPS column
};

static bool pxa_pxq_v70_route_ok(int R, int K, int ny) {
    for (size_t i = 0; i < sizeof(pxq4_v70_route_deny)/sizeof(pxq4_v70_route_deny[0]); ++i) {
        const pxq4_v70_route & e = pxq4_v70_route_deny[i];
        if (e.R == 0) continue;                                  // sentinel
        if (e.R == R && e.K == K && ny >= e.min_ny && ny <= e.max_ny) return false;
    }
    static const std::string skip = [] {
        const char * e = getenv("PXA_PXQ_GEMM_V70_SKIP");
        return std::string(e ? e : "");
    }();
    if (!skip.empty()) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d:%d", R, K);
        const std::string needle(buf);
        size_t p = 0;
        while (p < skip.size()) {
            size_t q = skip.find(',', p);
            if (q == std::string::npos) q = skip.size();
            if (skip.compare(p, q - p, needle) == 0) return false;
            p = q + 1;
        }
    }
    return true;
}

// R must be a multiple of the config's block-row height; cfg0/cfg2 take R%64 (always true for a
// PXQ panel), cfg1/cfg3 need R%128.
static int pxa_pxq_v70_bm(int cfg) {
    switch (cfg) {
        case 1:  return pxq4_v70_cfg1::BM;
        case 2:  return pxq4_v70_cfg2::BM;
        case 3:  return pxq4_v70_cfg3::BM;
        default: return pxq4_v70_cfg0::BM;
    }
}

template <class CFG>
static void pxq4_v70_launch_cfg(const uint8_t * W, const half * X, float * C,
                                int R, int K, int ny, cudaStream_t stream) {
    const int kslabs = K / PXQ6_QK;
    const dim3 grid((unsigned)(R / CFG::BM), (unsigned)((ny + CFG::BN - 1) / CFG::BN));
    const dim3 blk(32, CFG::NWARPS);                      // blockDim.x MUST be 32
    k_pxq4_gemm_v70<pxq6_pol_p6, CFG><<<grid, blk, 0, stream>>>(W, X, C, R, K, ny, kslabs);
}

// returns 0 on launch, -1 if the config/shape pair is not launchable
static int pxa_pxq4_gemm_v70_launch(const uint8_t * W, const half * X, float * C,
                                    int R, int K, int ny, int cfg, cudaStream_t stream) {
    if (K % PXQ6_QK || R % pxa_pxq_v70_bm(cfg)) return -1;
    switch (cfg) {
        case 1:  pxq4_v70_launch_cfg<pxq4_v70_cfg1>(W, X, C, R, K, ny, stream); break;
        case 2:  pxq4_v70_launch_cfg<pxq4_v70_cfg2>(W, X, C, R, K, ny, stream); break;
        case 3:  pxq4_v70_launch_cfg<pxq4_v70_cfg3>(W, X, C, R, K, ny, stream); break;
        default: pxq4_v70_launch_cfg<pxq4_v70_cfg0>(W, X, C, R, K, ny, stream); break;
    }
    return 0;
}
