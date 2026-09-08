// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxq4-mmq.cuh -- M1: dp4a MMQ prefill tile for the 4-bit PXQ tiers, DENSE 2D MUL_MAT.
//
// WHY. On sm_70 a dense 2D PXQ4 MUL_MAT in prefill takes the generic ggml route: dequantize the
// WHOLE weight matrix into an fp16 pool scratch (k_pxq6_dequant_matrix) and then run a cuBLAS
// HGEMM against it. At -ub 512 over a 20k prompt that is 41 full dequants of every weight in the
// model -- ~60 ms per ubatch per GPU, ~12% of the prefill wall -- and it also inflates the GEMM's
// own weight traffic 4x (fp16 scratch vs the 4.27 bpw panels). Mainline's MXFP4 prefill on the
// same Volta pair pays neither: mmq.cuh reads the quantized weights directly and accumulates on
// dp4a. This file is that route for PXQ4/PXQ4HQ.
//
// FORM. The tile is the N16 twin (k_pxq2_mmq_grouped, pxq6i8.cuh) generalized off the 2-bit tier:
// mmq.cuh's block geometry (WARP_SIZE lanes over ROWS, nwarps over TOKENS, (MMQ_X/NWARPS) *
// (MMQ_Y/WS) accumulators per thread), mmq.cuh's padded x/y tiles, and a new W loader in front.
// The one structural difference from N16 is the scale granularity: PXQ2 rides N15's joint-snap
// table so one fp32 scale covers a whole 32-element slab (q8_0's shape), which for a 4-entry book
// costs <=1.81% of a group's absmax. A 16-entry book cannot be joint-snapped that cheaply (the
// (smax,s) table would be 256 x 16 bytes and every nibble would need a 4-way register byte
// select), so this tile keeps the format's NATIVE per-16 (PXQ4) / per-8 (PXQ4HQ) scales: POL::NEFF
// dp4a sub-chains per slab, one fp32 eff scale each. That is *better* numerics than the q8_0
// shape, and on Volta it is close to free -- the extra work is fp32 (2 I2F + 2 MUL + 1 ADD + 1
// FMA per row/token/slab) and Volta's FP32 pipe is physically separate from the INT32 pipe the
// dp4a chain issues on, so the 8 dp4a per slab still set the rate.
//
// NUMERIC CONTRACT (the frozen q3-s8 snap of pxq6i8.cuh:19-31, unchanged):
//   book snap   q_c = rint(book[c] * 127 / absmax(book))      (absmax == 1.0 for the PX16 book,
//               so q_c IS the frozen Q3 s8 book [-125,-93,-71,-53,-38,-25,-12,0,11,22,33,46,60,
//               76,97,127] -- the same values pxq6i8/pxq-mmvq already ship)
//   dequant     w = (anchor * SUB[s] * absmax/127) * q_c      (the /127 and the book absmax fold
//               into the per-16 fp32 eff scale; the stored fp16 row anchor is untouched)
//   activations q8_0-style per-32 absmax/127 round-to-nearest, the same class of activation
//               quantization stock MMQ applies to every quantized type. No zero point: the PX16
//               book is a symmetric codebook, so there is no q8_1 sum term to carry.
// NOT bit-exact vs the dequant+cuBLAS fp16 route (int8 x int8 -> fp32 rescale vs fp16 HGEMM), so
// a temp-0 sha is NOT a valid gate: G3 logprob parity + a perplexity regate are mandatory.
//
// ENV: PXA_PXQ4_MMQ=0 (default, OFF -> the incumbent dequant+cuBLAS route, byte-identical
//      dispatch to the old build) | 1 (ON). PXA_PXQ4_MMQ_NW = 4 (default) or 8 warps.
//      PXA_PXQ4_MMQ_MINNY (default 32) is the token floor: below it the tile's 64-token columns
//      are mostly padding and the incumbent wins.
// ARCH: dp4a is native on sm_61+ and on sm_70; sm_60 has no IDP4A and falls to the emulated
//      ggml_cuda_dp4a, so cc 600 is declined unless PXA_PXQ4_MMQ=2 forces it (TEST ONLY).
//
// ---- MEASURED: LOSS on sm_70. DEFAULT OFF, and do NOT re-run this blind. -------------------
// 2026-09-03, 2x V100-PCIE-16GB, Qwen3.8-27B-PXQ4, -ub 512 -b 2048, 2 runs/arm, the checkpoint
// and pipeline levers on in BOTH arms, needle recalled in both:
//     20801 tok   OFF (dequant+cuBLAS) 1069.1 / 1068.0 t/s   ON 544.4 / 539.1   -49.5%
//      3121 tok   OFF                  1081.8 / 1062.3       ON 549.4 / 548.6   -48.7%
// Device busy went UP (1.26 -> 1.37 summed over the pair) while throughput halved: both cards are
// saturated and simply doing more work per token. The tile claimed one shape class in this model
// (R=10240 K=5120 ny=512, the dense fused gate+up) and that single substitution cost 19 s of a
// 19.5 s prefill.
// WHY: a V100 does 62.8 TOPS of dp4a against 125 TFLOPS of HMMA, so a dp4a GEMM starts at HALF the
// ceiling of the fp16 tensor-core GEMM it replaces, while the dequant pass it removes is only ~12%
// of the prefill wall. A 2x slower multiply cannot be paid for out of a 12% saving. Independently
// confirmed by the V70 lane (pxq4-v70.cuh, register-direct m8n8k4): -40..-52% on the same cell.
// Two structurally different PXQ4-native sm_70 prefill GEMMs both land at ~half of dequant+cuBLAS.
// KEPT because it is correct (tests/test-pxq4-mmq.cu, 176 checks, 0 failures) and because the
// conclusion INVERTS on any card whose int8 rate beats its fp16 rate -- Turing/Ampere INT8 tensor
// cores are 2-4x fp16 there, and this is the only PXQ4 prefill path that would exploit it.
#pragma once

#include <string>

// requires: pxq6.cuh (pxq6_pol_p6 / pxq6_pol_p6hq, pxq6_panel, pxq6_ldcodes, pxq4_tile_info),
//           pxq6i8.cuh (pxqi8_quant_row_groups), common.cuh (ggml_cuda_dp4a).

#define PXQ4MMQ_WS  32
#define PXQ4MMQ_KSL 4       // slabs staged per K step (128 K-elements)

// Only the 4-bit nibble tiers: one code byte == two book indices (lo, hi), which is what the
// 256-entry pair table below is keyed on. Everything else keeps its existing route.
template <class POL> struct pxq4mmq_ok { static constexpr bool value = false; };
template <> struct pxq4mmq_ok<pxq6_pol_p6>   { static constexpr bool value = true; };
template <> struct pxq4mmq_ok<pxq6_pol_p6hq> { static constexpr bool value = true; };

// ---------------------------------------------------------------------------------------------
// env gates
// ---------------------------------------------------------------------------------------------
static inline int pxa_pxq4_mmq() {
    static const int m = [](){
        const char * e = getenv("PXA_PXQ4_MMQ");
        int v = e ? atoi(e) : 0;
        if (v < 0 || v > 2) v = 0;
        if (v) fprintf(stderr, "PXA_PXQ4_MMQ: mode %d (dp4a MMQ prefill tile for the 4-bit PXQ "
                       "tiers, dense 2D%s; s8 book snap -- NOT bit-exact vs dequant+cuBLAS, "
                       "G3 logprob parity + ppl regate required)\n",
                       v, v == 2 ? ", sm_60 FORCED (emulated dp4a, TEST ONLY)" : "");
        return v;
    }();
    return m;
}

static inline int pxa_pxq4_mmq_nw() {
    static const int nw = [](){
        const char * e = getenv("PXA_PXQ4_MMQ_NW");
        const int v = e ? atoi(e) : 4;
        return v == 8 ? 8 : 4;
    }();
    return nw;
}

// Per-shape route skip (main #282 note 1): a comma-separated list of "R:K" the MMQ tile must
// decline so the incumbent dequant+cuBLAS HGEMM stays selectable on shapes where dp4a really is
// slower. Same mechanism and same spelling as the V70 lane's PXA_PXQ_GEMM_V70_SKIP.
static inline bool pxa_pxq4_mmq_route_ok(int R, int K) {
    static const std::string skip = [] {
        const char * e = getenv("PXA_PXQ4_MMQ_SKIP");
        return std::string(e ? e : "");
    }();
    if (skip.empty()) return true;
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
    return true;
}

// Token floor. The tile's column axis is 64 tokens wide; the incumbent dequant+cuBLAS route
// amortizes its dequant over the whole ubatch, so below a few tile-columns of real tokens the
// MMQ tile is mostly running padding. 32 keeps it clear of the decode window (which the mmv
// driver owns below PXA_PXQ4_2D_MAX_NY anyway) without claiming a shape it cannot win.
static inline int pxa_pxq4_mmq_min_ny() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQ4_MMQ_MINNY");
        const int n = e ? atoi(e) : 32;
        return n < 1 ? 1 : n;
    }();
    return v;
}

// ---------------------------------------------------------------------------------------------
// activation stage: contiguous f32 [ny][K] -> s8 [ny][K] (packed LE u32 words) + per-32 scales
// f32 [ny][K/32]. One block per token row; pxqi8_quant_row_groups is the same per-32 absmax/127
// quantizer the N13/N16 tiles feed on, used here straight off the global row (the rows are
// contiguous, so there is nothing for a smem stage to coalesce that L1 does not already).
// ---------------------------------------------------------------------------------------------
static __global__ void k_pxq4mmq_quant_rows(const float * __restrict__ src1,
        uint8_t * __restrict__ Aq, float * __restrict__ Ad, const int K) {
    const int i = blockIdx.x;
    pxqi8_quant_row_groups(src1 + (size_t)i*K,
                           (uint32_t *)(Aq + (size_t)i*K),
                           Ad + (size_t)i*(K/PXQ4_QK),
                           K/PXQ4_QK, blockDim.x);
}

// ---------------------------------------------------------------------------------------------
// W loader: one (slab, row) -> 8 packed s8 words (32 codes, K order) + POL::NEFF fp32 scales.
//
// pairq[b] = (uint8)q(book[b & 0xf]) | ((uint8)q(book[b >> 4]) << 8) -- the s8 twin of pxq6.cuh's
// PAIRLUT, keyed on a whole code byte, so a 32-code row costs 8 table loads and zero float->int
// conversions. Element e of the slab lives in code byte e/2, nibble e&1 (POL::pair's convention),
// so out8[w] (elements 4w..4w+3, LE) is exactly pairq[byte 2w] | pairq[byte 2w+1] << 16.
// ---------------------------------------------------------------------------------------------
template <class POL>
static __device__ __forceinline__ void pxq4mmq_q8_slab(
        const uint8_t * __restrict__ slab, const int row, const float anch,
        const float * __restrict__ sub, const uint32_t * __restrict__ pairq, const float bfold,
        uint32_t * __restrict__ out8, float * __restrict__ eff) {
    POL::row_effs(slab, row, anch, sub, eff);
    #pragma unroll
    for (int n = 0; n < POL::NEFF; ++n) eff[n] *= bfold;

    uint32_t q[4];
    pxq6_ldcodes<POL>(slab + POL::CODE_OFF + (size_t)row*POL::CODE_BYTES, q);
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        const int b0 = (q[(2*w)   >> 2] >> (8*((2*w)   & 3))) & 0xff;
        const int b1 = (q[(2*w+1) >> 2] >> (8*((2*w+1) & 3))) & 0xff;
        out8[w] = pairq[b0] | (pairq[b1] << 16);
    }
}

// ---------------------------------------------------------------------------------------------
// the tile. grid = (R/MMQ_Y panels, ntiles token tiles); block = (32, NWARPS).
// ---------------------------------------------------------------------------------------------
template <class POL, int MMQ_X, int MMQ_Y, int NWARPS, bool RAG>
static __global__ void __launch_bounds__(PXQ4MMQ_WS*NWARPS)
k_pxq4_mmq_2d(const uint8_t * __restrict__ W,
              const uint8_t * __restrict__ Aq, const float * __restrict__ Ad,
              float * __restrict__ C,
              const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    constexpr int WS   = PXQ4MMQ_WS;
    constexpr int KSL  = PXQ4MMQ_KSL;
    constexpr int KW   = KSL*8;              // four-byte words per row per K step
    constexpr int XSTR = KW + 1;             // padded row stride (mmq.cuh's +1 idiom)
    constexpr int NTHR = WS*NWARPS;
    constexpr int ACC  = (MMQ_X/NWARPS)*(MMQ_Y/WS);
    constexpr int RPS  = NTHR / MMQ_Y;       // (row, slab) pairs staged per thread-row sweep
    constexpr int NEFF = POL::NEFF;
    constexpr int VDRW = 8 / NEFF;           // words per eff sub-chain

    const int panels = R / MMQ_Y, kslabs = K / PXQ4_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    if (tile.nrows <= 0) return;
    const uint8_t * pan = pxq6_panel<POL>(W, 0, panels, p, kslabs);
    const uint8_t * Aqt = Aq + (size_t)tile.row0*K;
    const float   * Adt = Ad + (size_t)tile.row0*kslabs;
    float         * Ct  = C  + (size_t)tile.row0*R + (size_t)p*MMQ_Y;

    __shared__ float    tab[32], sub[16];
    __shared__ uint32_t sPQ[256];
    __shared__ uint32_t x_qs[MMQ_Y*XSTR];
    // x_df is indexed [(slab, eff)][row], NOT [row][slab][eff]: the row index is the warp's
    // LANE index in both the stage and the read, so it must be the fastest-varying axis or all
    // 32 lanes land on 4 banks (an 8-way conflict on every scale load).
    __shared__ float    x_df[KSL*NEFF*MMQ_Y];
    __shared__ uint32_t y_qs[MMQ_X*XSTR];
    __shared__ float    y_df[MMQ_X*KSL];

    const int tid = threadIdx.y*WS + threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    __syncthreads();

    float bmax = 0.f;
    #pragma unroll
    for (int t = 0; t < 16; ++t) bmax = fmaxf(bmax, fabsf(tab[t]));
    const float bfold = bmax / 127.f;
    const float binv  = bmax > 0.f ? 127.f/bmax : 0.f;
    for (int e = tid; e < 256; e += NTHR) {          // s8 pair table, keyed on a code byte
        int q0 = (int)rintf(tab[e & 0xf]*binv);
        int q1 = (int)rintf(tab[e >> 4 ]*binv);
        q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
        q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
        sPQ[e] = (uint32_t)(uint8_t)q0 | ((uint32_t)(uint8_t)q1 << 8);
    }
    __syncthreads();

    const int lrow = tid % MMQ_Y;                    // row / token staged by this thread
    const int lsl  = tid / MMQ_Y;                    // its slab within the K step
    const float anch = POL::anchor(pan, lrow);
    const bool  aok  = lrow < tile.nrows;

    float sum[ACC];
    #pragma unroll
    for (int a = 0; a < ACC; ++a) sum[a] = 0.f;

    for (int kb0 = 0; kb0 < kslabs; kb0 += KSL) {
        #pragma unroll
        for (int s = lsl; s < KSL; s += RPS) {       // W: PXQ4 panel -> s8 x tile
            uint32_t w8[8] = {0,0,0,0,0,0,0,0};
            float ef[NEFF];
            #pragma unroll
            for (int n = 0; n < NEFF; ++n) ef[n] = 0.f;
            if (kb0 + s < kslabs) {
                pxq4mmq_q8_slab<POL>(pan + POL::HDR + (size_t)(kb0 + s)*POL::SLAB,
                                     lrow, anch, sub, sPQ, bfold, w8, ef);
            }
            #pragma unroll
            for (int w = 0; w < 8; ++w) x_qs[lrow*XSTR + s*8 + w] = w8[w];
            #pragma unroll
            for (int n = 0; n < NEFF; ++n) x_df[(s*NEFF + n)*MMQ_Y + lrow] = ef[n];
        }
        #pragma unroll
        for (int s = lsl; s < KSL; s += RPS) {       // A: already q8-per-32 from the stage kernel
            const bool ok = aok && (kb0 + s < kslabs);
            const uint8_t * arow = Aqt + (size_t)lrow*K + (size_t)(kb0 + s)*PXQ4_QK;
            uint4 v0 = make_uint4(0,0,0,0), v1 = v0;
            if (ok) { v0 = *(const uint4 *)arow; v1 = *(const uint4 *)(arow + 16); }
            uint32_t * d = &y_qs[lrow*XSTR + s*8];
            d[0]=v0.x; d[1]=v0.y; d[2]=v0.z; d[3]=v0.w;
            d[4]=v1.x; d[5]=v1.y; d[6]=v1.z; d[7]=v1.w;
            y_df[lrow*KSL + s] = ok ? Adt[(size_t)lrow*kslabs + kb0 + s] : 0.f;
        }
        __syncthreads();

        // Register-cached x panel (N16's finding: the naive mmq.cuh ordering re-reads this
        // thread's weight words once per token group and makes the loop LDS-bound long before
        // it is occupancy-bound). MMQ_Y/WS rows' 8 words live in registers; each slab reads x
        // once and y once. The NEFF sub-chains share the slab's single activation scale, so the
        // token-side smem traffic is identical to the q8_0-shaped N16 loop.
        for (int s = 0; s < KSL; ++s) {
            int   xw[MMQ_Y/WS][8];
            float xd[MMQ_Y/WS][NEFF];
            #pragma unroll
            for (int ii = 0; ii < MMQ_Y/WS; ++ii) {
                const int i = ii*WS + threadIdx.x;
                #pragma unroll
                for (int w = 0; w < 8; ++w) xw[ii][w] = (int)x_qs[i*XSTR + s*8 + w];
                #pragma unroll
                for (int n = 0; n < NEFF; ++n) xd[ii][n] = x_df[(s*NEFF + n)*MMQ_Y + i];
            }
            #pragma unroll
            for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
                const int j = j0 + threadIdx.y;
                if (RAG && j >= tile.nrows) continue;
                int yw[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) yw[w] = (int)y_qs[j*XSTR + s*8 + w];
                const float dy = y_df[j*KSL + s];
                #pragma unroll
                for (int ii = 0; ii < MMQ_Y/WS; ++ii) {
                    float acc = 0.f;
                    #pragma unroll
                    for (int n = 0; n < NEFF; ++n) {
                        int sumi = 0;
                        #pragma unroll
                        for (int w = 0; w < VDRW; ++w) {
                            sumi = ggml_cuda_dp4a(xw[ii][n*VDRW + w], yw[n*VDRW + w], sumi);
                        }
                        acc += xd[ii][n]*(float)sumi;
                    }
                    sum[j0/NWARPS*(MMQ_Y/WS) + ii] += dy*acc;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j = j0 + threadIdx.y;
        if (j >= tile.nrows) continue;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WS) {
            const int i = i0 + threadIdx.x;
            Ct[(size_t)j*R + i] = sum[j0/NWARPS*(MMQ_Y/WS) + i0/WS];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// launch descriptor + per-format picker. Unknown formats decline (nullptr) -> the caller keeps
// the incumbent dequant+cuBLAS route for the WHOLE op.
// ---------------------------------------------------------------------------------------------
typedef void (*pxq4mmq_fn)(const uint8_t *, const uint8_t *, const float *, float *,
                           const pxq4_tile_info *, int, int);
struct pxq4mmq_kern { pxq4mmq_fn fn; dim3 block; };

template <class POL, int NWARPS>
static inline pxq4mmq_fn pxq4mmq_pick_rag() {
    return k_pxq4_mmq_2d<POL, PXQ4_BN, PXQ4_BM, NWARPS, true>;
}

static inline pxq4mmq_kern pxq4mmq_pick(int fmt) {
    const int nw = pxa_pxq4_mmq_nw();
    switch (fmt) {
        case PXA_PXQ_FMT_P6:
            return { nw == 8 ? pxq4mmq_pick_rag<pxq6_pol_p6, 8>() : pxq4mmq_pick_rag<pxq6_pol_p6, 4>(),
                     dim3(PXQ4MMQ_WS, nw) };
        case PXA_PXQ_FMT_P6HQ:
            return { nw == 8 ? pxq4mmq_pick_rag<pxq6_pol_p6hq, 8>() : pxq4mmq_pick_rag<pxq6_pol_p6hq, 4>(),
                     dim3(PXQ4MMQ_WS, nw) };
        default:
            return { nullptr, dim3(PXQ4MMQ_WS, nw) };
    }
}
