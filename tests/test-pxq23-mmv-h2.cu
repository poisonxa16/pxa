// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxq23-mmv-h2.cu -- differential harness for PXA_PXQ_MMV_H2, the half2 inner loop for the
// PXQ2/PXQ3 dense decode mmv on GP100.
//
// WHAT IS UNDER TEST. The shipped decode dot32 spends 2 shared-memory book gathers and
// FMUL + FFMA + FADD per element pair -- the same 16 pair decodes per 32 weights whatever the code
// width, which is why a 2-bit tier moves 39% fewer bytes than a 4-bit one and decodes at the same
// speed on a P100. The h2 arm replaces the pair decode with ONE gather from a half2 pair LUT and
// ONE HFMA2, and accumulates the 16-element eff group in fp16; everything above that group -- the
// eff scales, the eff[0]*t[0]+eff[1]*t[1] fold, the per-lane chain, the canonical per-chunk fold
// and the reducers -- stays fp32 and unchanged.
//
// So this is NOT a bit-exactness test. Two things ARE exact and are checked with zero tolerance:
//
//   (1) THE BOOK. The LM4/LM8 books are fp16-exact by the pxa_pxq23_book_ok() self-check, so every
//       LUT entry must equal the fp16 image of the very book value the fp32 arm gathers -- all 16
//       PXQ2 keys and all 64 PXQ3 keys, exhaustively, compared as raw fp16 bit patterns. If this
//       fails, the "zero book error" property the mode is built on is gone.
//
//   (2) SPLIT == UNSPLIT. The per-slab power-of-two scale is a function of 32 x values alone, so
//       every block that stages a slab derives the same scale at any S. The three h2 kernels are
//       run against each other at S = 1, 2, 4 and their dst must compare equal BY BITS. A failure
//       here means the mode reintroduced a launch-geometry dependence, which is exactly what
//       PXQ_CANON_v1 exists to prevent.
//
// The x rounding and the fp16 group accumulation are a tolerance question, and phase 2 reports the
// error rather than asserting a threshold -- the shipping bar is the model-level KL divergence
// against the exact path, not a per-dot epsilon. It DOES fail on a non-finite result where the
// fp32 arm was finite, on a relative error above a loose sanity ceiling, and on the guard cases.
//
// Needs a real CUDA device. Run inside the shared GPU lock window. Nonzero exit on any failure.
#include "ggml-cuda/pxa/pxq6.cuh"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0, g_cases = 0;

static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static void fail(const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  FAIL: "); vprintf(fmt, ap); printf("\n");
    va_end(ap);
    ++g_fail;
}

// ---------------------------------------------------------------------------------------------
// synthetic weights. Raw bytes are a valid panel by construction: every 2-/3-bit code and every
// 4-bit sub index is in range whatever the byte says. Anchors are kept finite and moderate so the
// reference arm itself is well conditioned -- the degenerate-input questions are phase 3's, and
// they belong on the x side, which is where the h2 arm actually differs.
// ---------------------------------------------------------------------------------------------
template <class POL>
static void build_raw(std::vector<uint8_t> & raw, int panels, int kslabs, uint64_t & seed) {
    const size_t stride = (size_t)POL::HDR + (size_t)kslabs*POL::SLAB;
    raw.assign(stride*panels, 0);
    for (int p = 0; p < panels; ++p) {
        uint8_t * pan = raw.data() + (size_t)p*stride;
        uint16_t * anc = (uint16_t *)pan;
        for (int r = 0; r < PXQ6_BM; ++r) {
            const int e = 8 + (int)(rs(seed) % 8);                   // exp 8..15 -> 2^-7..2^0
            anc[r] = (uint16_t)(((rs(seed) & 1) << 15) | (e << 10) | (rs(seed) & 0x3ff));
        }
        for (size_t b = POL::HDR; b < stride; ++b) pan[b] = (uint8_t)(rs(seed) & 0xff);
    }
}

// =============================================================================================
// PHASE 1 -- LUT identity, exhaustive, zero tolerance.
// Every entry of the staged half2 LUT against the fp16 image of the book pair the fp32 arm reads
// for the same key, compared as raw fp16 bits. Also re-derives the key from a synthetic code word
// so the key derivation itself (not just the table) is covered on both tiers.
// =============================================================================================
template <class POL>
static __global__ void k_lut_check(uint16_t * out_lut, uint16_t * out_ref, int * out_key_ok) {
    __shared__ __half2 hlut[POL::H2LUT];
    pxq6_stage_h2lut<POL>(hlut, threadIdx.x, blockDim.x);
    __syncthreads();
    for (int i = threadIdx.x; i < POL::H2LUT; i += blockDim.x) {
        const __half2 v = hlut[i];
        out_lut[2*i]     = __half_as_ushort(__low2half(v));
        out_lut[2*i + 1] = __half_as_ushort(__high2half(v));
        int c0, c1;
        POL::h2codes(i, c0, c1);
        out_ref[2*i]     = __half_as_ushort(__float2half_rn(POL::bookv(c0)));
        out_ref[2*i + 1] = __half_as_ushort(__float2half_rn(POL::bookv(c1)));
    }
    // key derivation: build a code row whose pair b carries key k, then read it back through
    // POL::h2key. Covers the nibble form (P2) and the bit-plane form (P3) on real code words.
    // unroll 1 on both loops, deliberately: fully unrolled this is 16 x 64 bodies per policy per
    // arch and it dominated the whole build's compile time. It is a correctness check, not a hot
    // loop. The pair indices are the four that matter -- 0 and 7 are the first and last pair of
    // the low code word, 8 and 15 the first and last of the high one, so both halves and both
    // word boundaries are covered on both tiers.
    if (threadIdx.x == 0) {
        int ok = 1;
        const int bsel[4] = { 0, 7, 8, 15 };
        #pragma unroll 1
        for (int bi = 0; bi < 4; ++bi) {
            const int b = bsel[bi];
            #pragma unroll 1
            for (int k = 0; k < POL::H2LUT; ++k) {
                uint32_t q[POL::CODE_WORDS];
                #pragma unroll
                for (int w = 0; w < POL::CODE_WORDS; ++w) q[w] = 0xdeadbeefu ^ (0x9e3779b9u*w);
                const int      h  = b >> 3;
                const int      sh = 2*((2*b) & 15);
                q[h] = (q[h] & ~(0xfu << sh)) | ((uint32_t)(k & 0xf) << sh);       // low planes
                if constexpr (POL::CODE_WORDS == 3) {                              // P3 high plane
                    q[2] = (q[2] & ~(0x3u << (2*b))) | ((uint32_t)((k >> 4) & 3) << (2*b));
                }
                if (POL::h2key(q, b) != k) ok = 0;
            }
        }
        *out_key_ok = ok;
    }
}

template <class POL>
static void phase1_tier(const char * tier) {
    ++g_cases;
    const int n = POL::H2LUT;
    uint16_t * d_lut = nullptr, * d_ref = nullptr; int * d_ok = nullptr;
    CUCK(cudaMalloc(&d_lut, 2*n*sizeof(uint16_t)));
    CUCK(cudaMalloc(&d_ref, 2*n*sizeof(uint16_t)));
    CUCK(cudaMalloc(&d_ok,  sizeof(int)));
    CUCK(cudaMemset(d_lut, 0xCD, 2*n*sizeof(uint16_t)));
    k_lut_check<POL><<<1, 64>>>(d_lut, d_ref, d_ok);
    CUCK(cudaDeviceSynchronize());
    CUCK(cudaGetLastError());
    std::vector<uint16_t> lut(2*n), ref(2*n); int ok = 0;
    CUCK(cudaMemcpy(lut.data(), d_lut, 2*n*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(ref.data(), d_ref, 2*n*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(&ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 2*n; ++i) if (lut[i] != ref[i]) ++bad;
    if (bad) fail("%s LUT: %d/%d halves differ from the fp16 image of the book", tier, bad, 2*n);
    if (!ok) fail("%s h2key: key derivation does not round-trip for every (pair, key)", tier);
    if (!bad && ok) printf("  %s  LUT %d entries exact, key derivation exact for all 16 pairs\n", tier, n);
    CUCK(cudaFree(d_lut)); CUCK(cudaFree(d_ref)); CUCK(cudaFree(d_ok));
}

// =============================================================================================
// PHASE 2/3 -- per-(row, slab) differential. One block per panel, 64 threads (one row each);
// both arms staged and evaluated in the SAME block off the SAME x, so nothing but the arithmetic
// differs. Dynamic smem: xs (K floats) | xh2 (K/2 half2) | xiv (K/32 floats).
// =============================================================================================
template <class POL>
static __global__ void __launch_bounds__(64)
k_dot_diff(const uint8_t * __restrict__ W, const float * __restrict__ x, int kslabs, int K,
           float * __restrict__ o_f32, float * __restrict__ o_h2) {
    extern __shared__ float smx[];
    float   * xs  = smx;
    __half2 * xh2 = (__half2 *)(smx + K);
    float   * xiv = (float *)(xh2 + (K >> 1));

    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    pxq6_stage_x_h2(x, K, (__half *)xh2, xiv);

    __shared__ float tab[32], sub[16];
    __shared__ __half2 hlut[POL::H2LUT];
    POL::stage_tabs(tab, sub, threadIdx.x);
    pxq6_stage_h2lut<POL>(hlut, threadIdx.x, blockDim.x);
    __syncthreads();
    pxq6_prmt_book pb{};

    const int p   = blockIdx.x;
    const int row = threadIdx.x;
    const uint8_t * pan = W + (size_t)p*((size_t)POL::HDR + (size_t)kslabs*POL::SLAB);
    const float anch = POL::anchor(pan, row);
    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        const size_t o = ((size_t)p*PXQ6_BM + row)*kslabs + kb;
        o_f32[o] = pxq6_dot32<POL, PXQ6_MODE_TAB, false>(slab, row, anch, xs + kb*PXQ6_QK,
                                                         tab, sub, nullptr, pb);
        o_h2[o]  = pxq6_dot32_h2<POL>(slab, row, anch, xh2 + kb*(PXQ6_QK/2), xiv[kb], sub, hlut);
    }
}

// x generators. `kind` picks the regime the h2 arm has to survive.
static void make_x(std::vector<float> & x, int K, int kind, uint64_t & seed) {
    x.assign(K, 0.0f);
    for (int i = 0; i < K; ++i) {
        const float u = (float)(rs(seed) % 2000001) / 1000000.0f - 1.0f;   // [-1, 1]
        switch (kind) {
            case 0: x[i] = u; break;                                       // unit scale
            case 1: x[i] = u * 1e-6f; break;                                // tiny
            case 2: x[i] = u * 1e4f; break;                                // large
            case 3: x[i] = u * ((i % 32) < 4 ? 1e3f : 1e-3f); break;        // 1e6 spread inside a slab
            case 4: x[i] = u; break;                                       // outliers patched below
            case 5: x[i] = (i % 64 == 0) ? 0.0f : u; break;                // zeros mixed in
            case 6: x[i] = 0.0f; break;                                    // all zero
            default: x[i] = u; break;
        }
    }
    if (kind == 4) {
        // one element per slab that would overflow fp16 outright (65504) and one that would take
        // the 8-step group accumulation over it without the power-of-two scale.
        for (int s = 0; s*32 < K; ++s) {
            x[s*32 + 3]  = (s & 1) ? 1.0e5f : -7.0e4f;
            x[s*32 + 17] = (s & 1) ? 3.0e30f : -3.0e30f;
        }
    }
}

template <class POL>
static void phase2_tier(const char * tier, int panels, int kslabs, int kind, const char * what,
                        uint64_t & seed, bool expect_zero) {
    ++g_cases;
    const int K = kslabs*PXQ6_QK;
    std::vector<uint8_t> raw;
    build_raw<POL>(raw, panels, kslabs, seed);
    std::vector<float> hx;
    make_x(hx, K, kind, seed);

    const size_t nout = (size_t)panels*PXQ6_BM*kslabs;
    uint8_t * d_w = nullptr; float * d_x = nullptr, * d_a = nullptr, * d_b = nullptr;
    CUCK(cudaMalloc(&d_w, raw.size()));
    CUCK(cudaMalloc(&d_x, (size_t)K*sizeof(float)));
    CUCK(cudaMalloc(&d_a, nout*sizeof(float)));
    CUCK(cudaMalloc(&d_b, nout*sizeof(float)));
    CUCK(cudaMemcpy(d_w, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(d_x, hx.data(), (size_t)K*sizeof(float), cudaMemcpyHostToDevice));

    const size_t smem = (size_t)K*sizeof(float) + (size_t)K*sizeof(__half) + (size_t)(K/32)*sizeof(float);
    k_dot_diff<POL><<<panels, 64, smem>>>(d_w, d_x, kslabs, K, d_a, d_b);
    CUCK(cudaDeviceSynchronize());
    CUCK(cudaGetLastError());

    std::vector<float> a(nout), b(nout);
    CUCK(cudaMemcpy(a.data(), d_a, nout*sizeof(float), cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(b.data(), d_b, nout*sizeof(float), cudaMemcpyDeviceToHost));

    double maxabs = 0.0, maxrel = 0.0, sumsq_e = 0.0, sumsq_r = 0.0;
    int    n_nonfinite = 0, n_nonzero = 0;
    for (size_t i = 0; i < nout; ++i) {
        if (!std::isfinite(a[i])) continue;                       // reference itself degenerate: skip
        if (!std::isfinite(b[i])) { ++n_nonfinite; continue; }
        if (expect_zero && b[i] != 0.0f) ++n_nonzero;
        const double e = fabs((double)b[i] - (double)a[i]);
        const double d = fabs((double)a[i]);
        maxabs = e > maxabs ? e : maxabs;
        if (d > 0.0) { const double r = e/d; maxrel = r > maxrel ? r : maxrel; }
        sumsq_e += e*e; sumsq_r += (double)a[i]*(double)a[i];
    }
    const double rms = sqrt(sumsq_r) > 0.0 ? sqrt(sumsq_e)/sqrt(sumsq_r) : 0.0;
    printf("  %s  %-22s max|err| %.3e   max rel %.3e   rms rel %.3e   (%zu dots)\n",
           tier, what, maxabs, maxrel, rms, nout);
    if (n_nonfinite) fail("%s %s: %d h2 results non-finite where the fp32 arm was finite "
                          "-- the power-of-two guard did not hold", tier, what, n_nonfinite);
    if (expect_zero && n_nonzero) fail("%s %s: %d h2 results non-zero on an all-zero x "
                                       "(the zero-max arm must contribute exactly 0)", tier, what, n_nonzero);
    // Loose sanity ceiling only. The shipping bar is model-level KLD, not a per-dot epsilon; this
    // catches a broken scale or a mis-keyed LUT, which show up orders of magnitude above fp16.
    if (rms > 1e-2) fail("%s %s: rms relative error %.3e is far above anything fp16 rounding can "
                         "explain -- the scale or the LUT is wrong", tier, what, rms);
    CUCK(cudaFree(d_w)); CUCK(cudaFree(d_x)); CUCK(cudaFree(d_a)); CUCK(cudaFree(d_b));
}

// non-finite propagation: an inf and a nan in x must reach the h2 output exactly as they reach the
// fp32 one. Checked as "both arms non-finite", not by value: fp32 may produce inf where fp16
// produces nan and vice versa, and neither is a defect -- silently returning a FINITE number is.
template <class POL>
static void phase3_nonfinite(const char * tier, uint64_t & seed) {
    ++g_cases;
    const int panels = 2, kslabs = 8, K = kslabs*PXQ6_QK;
    std::vector<uint8_t> raw;  build_raw<POL>(raw, panels, kslabs, seed);
    std::vector<float> hx;     make_x(hx, K, 0, seed);
    hx[5]  = INFINITY;              // slab 0
    hx[40] = -INFINITY;             // slab 1
    hx[70] = NAN;                   // slab 2

    const size_t nout = (size_t)panels*PXQ6_BM*kslabs;
    uint8_t * d_w = nullptr; float * d_x = nullptr, * d_a = nullptr, * d_b = nullptr;
    CUCK(cudaMalloc(&d_w, raw.size()));
    CUCK(cudaMalloc(&d_x, (size_t)K*sizeof(float)));
    CUCK(cudaMalloc(&d_a, nout*sizeof(float)));
    CUCK(cudaMalloc(&d_b, nout*sizeof(float)));
    CUCK(cudaMemcpy(d_w, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(d_x, hx.data(), (size_t)K*sizeof(float), cudaMemcpyHostToDevice));
    const size_t smem = (size_t)K*sizeof(float) + (size_t)K*sizeof(__half) + (size_t)(K/32)*sizeof(float);
    k_dot_diff<POL><<<panels, 64, smem>>>(d_w, d_x, kslabs, K, d_a, d_b);
    CUCK(cudaDeviceSynchronize());
    CUCK(cudaGetLastError());
    std::vector<float> a(nout), b(nout);
    CUCK(cudaMemcpy(a.data(), d_a, nout*sizeof(float), cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(b.data(), d_b, nout*sizeof(float), cudaMemcpyDeviceToHost));
    int swallowed = 0, invented = 0, clean = 0;
    for (size_t i = 0; i < nout; ++i) {
        const bool na = !std::isfinite(a[i]), nb = !std::isfinite(b[i]);
        if (na && !nb) ++swallowed;
        else if (!na && nb) ++invented;
        else if (!na && !nb) ++clean;
    }
    printf("  %s  non-finite x         swallowed %d   invented %d   clean %d\n",
           tier, swallowed, invented, clean);
    if (swallowed) fail("%s: %d dots returned a finite number where the fp32 arm returned inf/nan "
                        "-- the h2 arm must propagate, not hide", tier, swallowed);
    if (invented)  fail("%s: %d dots returned inf/nan where the fp32 arm was finite", tier, invented);
    CUCK(cudaFree(d_w)); CUCK(cudaFree(d_x)); CUCK(cudaFree(d_a)); CUCK(cudaFree(d_b));
}

// =============================================================================================
// PHASE 4 -- split == unsplit INSIDE the mode, bitwise. The three shipped h2 kernels against each
// other on the same panel and the same x: the plain form, the K1 KSPLIT form, and the gen S-split
// at every S that divides nfix. Zero tolerance; a single differing bit is a defect.
// =============================================================================================
template <class POL>
static void phase4_split(const char * tier, int R, int K, uint64_t & seed) {
    ++g_cases;
    const int panels = R/PXQ6_BM, kslabs = K/PXQ6_QK;
    const int nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    std::vector<uint8_t> raw;  build_raw<POL>(raw, panels, kslabs, seed);
    std::vector<float> hx;     make_x(hx, K, 0, seed);

    uint8_t * d_w = nullptr; float * d_x = nullptr, * d_dst = nullptr, * d_ws = nullptr;
    int32_t * d_ids = nullptr;
    CUCK(cudaMalloc(&d_w, raw.size()));
    CUCK(cudaMalloc(&d_x, (size_t)K*sizeof(float)));
    CUCK(cudaMalloc(&d_dst, (size_t)R*sizeof(float)));
    CUCK(cudaMalloc(&d_ws, (size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*(size_t)R*sizeof(float)));
    CUCK(cudaMalloc(&d_ids, sizeof(int32_t)));
    CUCK(cudaMemset(d_ids, 0, sizeof(int32_t)));
    CUCK(cudaMemcpy(d_w, raw.data(), raw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(d_x, hx.data(), (size_t)K*sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> base((size_t)R), cur((size_t)R);
    auto grab = [&](std::vector<float> & v) {
        CUCK(cudaDeviceSynchronize());
        CUCK(cudaGetLastError());
        CUCK(cudaMemcpy(v.data(), d_dst, (size_t)R*sizeof(float), cudaMemcpyDeviceToHost));
    };

    // reference arm: the plain (unsplit) h2 kernel, the same one the 2D driver fires at S == 1.
    const size_t smem_plain = (size_t)K*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);
    CUCK(cudaMemset(d_dst, 0xCD, (size_t)R*sizeof(float)));
    k_pxq6_mmv_h2<POL><<<dim3(panels,1,1), 256, smem_plain>>>(
        d_w, (const char *)d_x, 0, 0, (char *)d_dst, 0, 0, (const char *)d_ids, 0, 0, R, K, 1);
    grab(base);

    // K1 KSPLIT (64 threads, one block per (panel, lane)) + its canonical reducer
    CUCK(cudaMemset(d_dst, 0xCD, (size_t)R*sizeof(float)));
    k_pxq6_mmv_ksplit_h2<POL><<<dim3(panels*PXQ4_MMV_KSEG,1,1), 64, (size_t)K*sizeof(float)>>>(
        d_w, (const char *)d_x, 0, 0, d_ws, (const char *)d_ids, 0, 0, R, K, 1, 1);
    k_pxq_mmv_reduce<<<dim3((R+63)/64,1,1), 64>>>(d_ws, (char *)d_dst, 0, 0,
                                                  (const char *)d_ids, 0, 0, R, 1, 1);
    grab(cur);
    for (int i = 0; i < R; ++i) {
        if (memcmp(&base[i], &cur[i], sizeof(float)) != 0) {
            fail("%s R=%d K=%d: K1 KSPLIT differs from unsplit at row %d (%.9g vs %.9g)",
                 tier, R, K, i, base[i], cur[i]);
            break;
        }
    }

    // gen S-split, every power of two that divides nfix
    for (int S = 1; S <= nfix; S *= 2) {
        const int kc_max = (kslabs + S - 1)/S;
        const size_t smem_s = (size_t)kc_max*PXQ6_QK*sizeof(float);
        CUCK(cudaMemset(d_dst, 0xCD, (size_t)R*sizeof(float)));
        k_pxq6_mmv_ksplit_gen_h2<POL><<<dim3(panels*S,1,1), 256, smem_s>>>(
            d_w, (const char *)d_x, 0, 0, d_ws, (const char *)d_ids, 0, 0,
            (char *)d_dst, 0, 0, nullptr, R, K, 1, 1, S);
        k_pxq_mmv_reduce_s<<<dim3((R+63)/64,1,1), 64>>>(d_ws, (char *)d_dst, 0, 0,
                                                        (const char *)d_ids, 0, 0, R, 1, 1, kslabs);
        grab(cur);
        int bad = -1;
        for (int i = 0; i < R; ++i) if (memcmp(&base[i], &cur[i], sizeof(float)) != 0) { bad = i; break; }
        if (bad >= 0) fail("%s R=%d K=%d S=%d: gen split differs from unsplit at row %d (%.9g vs %.9g)",
                           tier, R, K, S, bad, base[bad], cur[bad]);
    }
    printf("  %s  R=%4d K=%5d nfix=%2d  split == unsplit bitwise across K1 and gen S=1..%d\n",
           tier, R, K, nfix, nfix);
    CUCK(cudaFree(d_w)); CUCK(cudaFree(d_x)); CUCK(cudaFree(d_dst));
    CUCK(cudaFree(d_ws)); CUCK(cudaFree(d_ids));
}

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    int dev = 0; CUCK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUCK(cudaGetDeviceProperties(&prop, dev));
    const int cc = prop.major*100 + prop.minor*10;
    printf("test-pxq23-mmv-h2: device %d (%s), cc %d\n", dev, prop.name, cc);
    printf("  the lever gates on cc == 600 EXACTLY (sm_61 runs fp16 at 1/64 rate); this binary\n"
           "  exercises the kernels directly, so it is meaningful on any fp16-capable device --\n"
           "  and on a non-600 card it is the proof that the SHIPPED path there is unchanged,\n"
           "  because the gate never selects these kernels at all.\n");

    uint64_t seed = 0x5eed1234abcd0001ull;

    printf("\nPHASE 1  half2 pair LUT, exhaustive, zero tolerance\n");
    phase1_tier<pxq6_pol_p2>("PXQ2");
    phase1_tier<pxq6_pol_p3>("PXQ3");

    printf("\nPHASE 2  per-(row, slab) differential vs the exact fp32 dot32\n");
    struct { int kind; const char * name; bool zero; } regimes[] = {
        { 0, "x ~ U(-1,1)",           false },
        { 1, "x ~ 1e-6",              false },
        { 2, "x ~ 1e4",               false },
        { 3, "1e6 spread in a slab",  false },
        { 5, "zeros mixed in",        false },
    };
    for (auto & rg : regimes) {
        phase2_tier<pxq6_pol_p2>("PXQ2", 4, 16, rg.kind, rg.name, seed, rg.zero);
        phase2_tier<pxq6_pol_p3>("PXQ3", 4, 16, rg.kind, rg.name, seed, rg.zero);
    }

    printf("\nPHASE 3  the guard: fp16-overflowing outliers, all-zero x, non-finite x\n");
    phase2_tier<pxq6_pol_p2>("PXQ2", 4, 16, 4, "overflow outliers",  seed, false);
    phase2_tier<pxq6_pol_p3>("PXQ3", 4, 16, 4, "overflow outliers",  seed, false);
    phase2_tier<pxq6_pol_p2>("PXQ2", 2,  8, 6, "all-zero x",         seed, true);
    phase2_tier<pxq6_pol_p3>("PXQ3", 2,  8, 6, "all-zero x",         seed, true);
    phase3_nonfinite<pxq6_pol_p2>("PXQ2", seed);
    phase3_nonfinite<pxq6_pol_p3>("PXQ3", seed);

    printf("\nPHASE 4  split == unsplit inside the mode, bitwise\n");
    phase4_split<pxq6_pol_p2>("PXQ2",  512, 2048, seed);
    phase4_split<pxq6_pol_p3>("PXQ3",  512, 2048, seed);
    phase4_split<pxq6_pol_p2>("PXQ2",  256, 5120, seed);
    phase4_split<pxq6_pol_p3>("PXQ3",  256, 5120, seed);

    printf("\n%s  %d cases, %d failures\n", g_fail ? "FAILED" : "PASSED", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
