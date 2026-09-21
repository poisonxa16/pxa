// test-pxq-cpu-dot.cpp — accuracy + agreement self-test for the PXQ x Q8 integer dot
// (ggml/src/pxq-dot.c). NOT registered with CMake — standalone build (needs AVX2/FMA/F16C, x86),
// the same shape as tests/test-pxq-cpu-dequant.cpp:
//
//   cc  -O2 -std=c11   -march=native -Iggml/include -Iggml/src -c ggml/src/pxq-cpu.c -o /tmp/pxq-cpu.o
//   cc  -O2 -std=c11   -march=native -Iggml/include -Iggml/src -c ggml/src/pxq-dot.c -o /tmp/pxq-dot.o
//   c++ -O2 -std=c++17 -march=native -Iggml/include -Iggml/src tests/test-pxq-cpu-dot.cpp \
//       /tmp/pxq-cpu.o /tmp/pxq-dot.o -o /tmp/test-pxq-cpu-dot -lm -lpthread
//   /tmp/test-pxq-cpu-dot
//
// WHAT IT MEASURES. The test synthesises PXQ4 / PXQ4-HQ panels itself, so it holds the
// (anchor, sub, code) triple for every element and can build both the exact weight row and the
// weight row as the int8 book image sees it. That splits the error into its two independent
// sources instead of reporting one lump:
//
//   BOOK   = <w_i8, x>     vs <w, x>    -- cost of the 16-entry int8 codebook, activations exact
//   ACT    = <w, x_q8>     vs <w, x>    -- cost of the Q8 activation quantisation, book exact
//   SCALAR = pxa_pxq_dot_q8_ref         vs <w, x>   -- both, scalar arm
//   AVX2   = pxa_pxq_dot_q8             vs <w, x>   -- both, SIMD arm
//   AGREE  = AVX2 vs SCALAR             -- must be float-rounding noise, nothing more
//
// The point of the split is the design justification in pxq-dot.c: if BOOK is at or below ACT,
// a wider (fp16/int16) codebook cannot improve the dot, because the activation side dominates
// and costs 2x the kernel to widen.
//
// It also checks the synthesised panels against pxa_pxq_dequant_row() bit-exactly, so a layout
// mistake in the test shows up as a layout failure and not as an error number.

#include "pxq-cpu.h"
#include "pxq-dot.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <thread>
#include <immintrin.h>

// --- link stubs -------------------------------------------------------------------------------
// pxq-cpu.c is one translation unit and carries the reowned graph ops (pxa_hadamard,
// pxa_rms_rms_add, ...) alongside the dequant this test needs, so linking it standalone pulls
// their references to libggml. None of them is on any path this test executes; they are defined
// here only so the object links. ggml_table_f32_f16 is the exception -- it is DATA, reachable
// through GGML_FP16_TO_FP32 -- so it is filled with the real values in main() rather than left
// zero, which would turn a mistake into silent zeros instead of a link error.
extern "C" {
float ggml_table_f32_f16[1 << 16];
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    std::fprintf(stderr, "ggml_abort at %s:%d: %s\n", file, line, fmt);
    std::abort();
}
int64_t ggml_nrows(const struct ggml_tensor *) { std::abort(); }
bool ggml_is_contiguous(const struct ggml_tensor *) { std::abort(); }
bool ggml_are_same_shape(const struct ggml_tensor *, const struct ggml_tensor *) { std::abort(); }
ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type) { std::abort(); }
}

static inline float    f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t f32_to_f16(float f)    { return _cvtss_sh(f, 0); }

#define PXQ_HDR   128
#define PXQ_SLAB    1088
#define PXQHQ_SLAB  1152

struct Panels {
    std::vector<uint8_t> buf;          // the PXQ tensor slice
    std::vector<float>   w;            // exact weights, [nrows][k]
    std::vector<float>   wi8;          // weights as the int8 book image sees them
};

// Synthesise `nrows` x `k` of a 4-bit tier straight from the format spec and record both weight
// images. hq selects the 8-element sub granularity (PXQ4-HQ) over the 16-element one (PXQ4).
static Panels make_panels(bool hq, int64_t nrows, int64_t k, std::mt19937 & rng,
                          const float * book, const float * sub16, const float * sub8,
                          const int8_t * bi8, float bscale) {
    const int64_t KB    = k/32;
    const int     slabb = hq ? PXQHQ_SLAB : PXQ_SLAB;
    const int     coff  = hq ? 128 : 64;
    const int64_t pbytes = PXQ_HDR + KB*slabb;
    const int64_t np     = nrows/64;
    const float * sub    = hq ? sub8 : sub16;

    Panels P;
    P.buf.assign((size_t)np*pbytes, 0);
    P.w  .assign((size_t)nrows*k, 0.0f);
    P.wi8.assign((size_t)nrows*k, 0.0f);

    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_real_distribution<float> anc(-2.0f, 2.0f);

    for (int64_t p = 0; p < np; ++p) {
        uint8_t * panel = P.buf.data() + p*pbytes;
        for (int r = 0; r < 64; ++r) {
            const uint16_t ah = f32_to_f16(anc(rng));
            ((uint16_t *)panel)[r] = ah;
            const float anchor = f16_to_f32(ah);
            for (int64_t kb = 0; kb < KB; ++kb) {
                uint8_t * slab = panel + PXQ_HDR + kb*slabb;
                int s4[4];
                for (int i = 0; i < 4; ++i) s4[i] = nib(rng);
                if (hq) {
                    slab[2*r]     = (uint8_t)(s4[0] | (s4[1] << 4));   // elems 0-7 / 8-15
                    slab[2*r + 1] = (uint8_t)(s4[2] | (s4[3] << 4));   // elems 16-23 / 24-31
                } else {
                    s4[1] = s4[0];                                     // elems 0-15
                    s4[3] = s4[2];                                     // elems 16-31
                    slab[r] = (uint8_t)(s4[0] | (s4[2] << 4));
                }
                uint8_t * code = slab + coff + r*16;
                const int64_t row = p*64 + r;
                for (int b = 0; b < 16; ++b) {
                    const int c0 = nib(rng), c1 = nib(rng);            // elems 2b, 2b+1
                    code[b] = (uint8_t)(c0 | (c1 << 4));
                    const int e0 = 2*b, e1 = 2*b + 1;
                    const float eff0 = anchor*sub[s4[e0 >> 3]];
                    const float eff1 = anchor*sub[s4[e1 >> 3]];
                    const size_t o = (size_t)row*k + kb*32;
                    P.w  [o + e0] = eff0*book[c0];
                    P.w  [o + e1] = eff1*book[c1];
                    P.wi8[o + e0] = eff0*((float)bi8[c0]*bscale);
                    P.wi8[o + e1] = eff1*((float)bi8[c1]*bscale);
                }
            }
        }
    }
    return P;
}

// TWO normalisations, because neither one alone is honest here.
//
//   E_MAG = |err| / sum_j |w_j x_j|   -- error against the ARITHMETIC SCALE of the row. This is
//     the number a bound can be put on: a wrong nibble order or a swapped sub-scale moves it by
//     orders of magnitude, and it does not blow up when the dot itself cancels.
//   E_DOT = rms(err) / rms(dot)       -- error against the RESULT. This is the one that matters
//     downstream, and on this test data it is pessimistic on purpose: the codes are drawn
//     uniformly over the book, so a row's dot against random activations is nearly all
//     cancellation and the denominator is about as small as it can get.
struct Err { double maxmag = 0.0, sse = 0.0, sref = 0.0; int64_t n = 0; };
static void note(Err & e, double got, double ref, double mag) {
    const double d = got - ref;
    e.sse  += d*d;
    e.sref += ref*ref;
    e.n    += 1;
    const double rel = mag > 0.0 ? std::fabs(d)/mag : 0.0;
    if (rel > e.maxmag) e.maxmag = rel;
}
static double rms(const Err & e) { return e.sref > 0 ? std::sqrt(e.sse/e.sref) : 0.0; }

static int g_fail = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); g_fail = 1; }
}

static void run_tier(bool hq, int64_t nrows, int64_t k, uint32_t seed) {
    const char * name = hq ? "PXQ4-HQ (253)" : "PXQ4 (252)";
    const float * book = nullptr; const float * sub16 = nullptr; const float * sub8 = nullptr;
    pxa_pxq_float_tables(&book, &sub16, &sub8);
    float bscale = 0.0f;
    const int8_t * bi8 = pxa_pxq_dot_book_i8(&bscale);

    std::mt19937 rng(seed);
    Panels P = make_panels(hq, nrows, k, rng, book, sub16, sub8, bi8, bscale);
    const ggml_type type = hq ? GGML_TYPE_PXQ4HQ : GGML_TYPE_PXQ4;

    // layout self-check: our synthetic weights must be what the shipping dequant reads back
    {
        std::vector<float> d(k);
        int64_t bad = 0;
        for (int64_t r = 0; r < nrows; ++r) {
            pxa_pxq_dequant_row(type, P.buf.data(), r, k, d.data());
            for (int64_t j = 0; j < k; ++j) if (d[j] != P.w[(size_t)r*k + j]) ++bad;
        }
        check(bad == 0, "synthesised panel disagrees with pxa_pxq_dequant_row");
    }

    // activations: heavy-tailed on purpose (a real hidden state has outliers, which is what
    // sets the Q8 block scale and therefore the activation error)
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> x(k);
    for (int64_t j = 0; j < k; ++j) {
        float v = nrm(rng);
        if ((rng() & 63) == 0) v *= 8.0f;
        x[j] = v;
    }
    std::vector<pxa_pxq_q8> xq(k/32);
    pxa_pxq_quantize_row_q8(x.data(), xq.data(), k);
    std::vector<float> xdq(k);
    for (int64_t b = 0; b < k/32; ++b)
        for (int j = 0; j < 32; ++j) xdq[b*32 + j] = xq[b].d*(float)xq[b].qs[j];

    Err e_book, e_act, e_scalar, e_avx2, e_agree;
    for (int64_t r = 0; r < nrows; ++r) {
        const float * w  = P.w  .data() + (size_t)r*k;
        const float * wi = P.wi8.data() + (size_t)r*k;
        double ref = 0, bk = 0, ac = 0, mag = 0;
        for (int64_t j = 0; j < k; ++j) {
            ref += (double)w [j]*(double)x  [j];
            bk  += (double)wi[j]*(double)x  [j];
            ac  += (double)w [j]*(double)xdq[j];
            mag += std::fabs((double)w[j]*(double)x[j]);
        }
        const double sc = pxa_pxq_dot_q8_ref(type, P.buf.data(), r, k, xq.data());
        const double av = pxa_pxq_dot_q8    (type, P.buf.data(), r, k, xq.data());
        note(e_book,   bk, ref, mag);
        note(e_act,    ac, ref, mag);
        note(e_scalar, sc, ref, mag);
        note(e_avx2,   av, ref, mag);
        note(e_agree,  av, sc,  mag);
    }

    std::printf("%-14s nrows=%-5lld k=%-5lld  simd=%s\n", name,
                (long long)nrows, (long long)k, pxa_pxq_dot_has_simd() ? "AVX2" : "scalar");
    std::printf("   %-8s max_E_MAG=%.3e  rms_E_DOT=%.3e\n", "BOOK",   e_book  .maxmag, rms(e_book));
    std::printf("   %-8s max_E_MAG=%.3e  rms_E_DOT=%.3e\n", "ACT",    e_act   .maxmag, rms(e_act));
    std::printf("   %-8s max_E_MAG=%.3e  rms_E_DOT=%.3e\n", "SCALAR", e_scalar.maxmag, rms(e_scalar));
    std::printf("   %-8s max_E_MAG=%.3e  rms_E_DOT=%.3e\n", "AVX2",   e_avx2  .maxmag, rms(e_avx2));
    std::printf("   %-8s max_E_MAG=%.3e  rms_E_DOT=%.3e\n", "AGREE",  e_agree .maxmag, rms(e_agree));

    // Bound on E_MAG. Each codec error is ~0.5/127 of its own block scale and a k-term dot
    // averages them down, so the expected worst case is a few times 1e-4 at k=1024; 3e-3 is a
    // wide margin that still catches a wrong nibble order, a swapped sub or a dropped half-block.
    check(e_scalar.maxmag < 3e-3, "scalar dot outside the codec error bound");
    check(e_avx2  .maxmag < 3e-3, "AVX2 dot outside the codec error bound");
    // The two arms do the same integer sums and differ only in float accumulation order, so this
    // is fp32 reassociation noise over k/32 blocks and nothing else.
    check(e_agree .maxmag < 1e-6, "AVX2 and scalar arms disagree beyond float reassociation");
    // The design claim pxq-dot.c makes: the int8 codebook is not what limits this dot.
    check(rms(e_book) < rms(e_act), "int8 book error exceeds the activation error");
}


// ---------------------------------------------------------------------------------------------
// WRAPPER TEST: pxa_pxq_mul_mat_cpu / pxa_pxq_moe_up_gate_cpu.
//
// The unit test above exercises one row against one activation row. The wrappers are where the
// phase-2 restructure actually lives -- the loop order flips, activations are quantised in tiles
// of PXA_PXQ_NYT, and the routed-row indirection picks both the activation row and the
// destination row. A tiling or row-map mistake there is invisible to a single-row test, so both
// wrappers are run against a naive f32 evaluation built on pxa_pxq_dequant_row, at ny values
// that straddle the tile boundary and at nth > 1.
// ---------------------------------------------------------------------------------------------

static void run_wrappers(int64_t nr0, int64_t k, int64_t ny, int nth, bool routed, uint32_t seed) {
    const float * book = nullptr; const float * s16 = nullptr; const float * s8 = nullptr;
    pxa_pxq_float_tables(&book, &s16, &s8);
    float bscale = 0.0f;
    const int8_t * bi8 = pxa_pxq_dot_book_i8(&bscale);
    std::mt19937 rng(seed);

    Panels A = make_panels(false, nr0, k, rng, book, s16, s8, bi8, bscale);   // "up"   / mul_mat
    Panels G = make_panels(false, nr0, k, rng, book, s16, s8, bi8, bscale);   // "gate"

    const int ne11 = (int)ny;
    std::vector<float> x((size_t)ny*k);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    for (auto & v : x) v = nrm(rng);

    // routed mode: dst row is (i1, i2), activation row is (i1 % ne11, i2) -- with ne12 == 1 here
    // i2 is always 0, so i1 is the permutation under test.
    std::vector<pxa_pxq_rowmap> map((size_t)ny);
    std::vector<int64_t> perm((size_t)ny);
    for (int64_t i = 0; i < ny; ++i) perm[i] = i;
    for (int64_t i = ny - 1; i > 0; --i) std::swap(perm[i], perm[rng() % (uint64_t)(i + 1)]);
    for (int64_t i = 0; i < ny; ++i) { map[i].i1 = (int32_t)perm[i]; map[i].i2 = 0; }
    const pxa_pxq_rowmap * rows = routed ? map.data() : nullptr;

    const size_t nb11 = (size_t)k*sizeof(float);
    const size_t nb1  = (size_t)nr0*sizeof(float);

    // per-row term magnitude, for the E_MAG bound
    auto row_mag = [&](const Panels & P, int64_t ix, const float * xr) {
        double m = 0;
        for (int64_t j = 0; j < k; ++j) m += std::fabs((double)P.w[(size_t)ix*k + j]*(double)xr[j]);
        return m;
    };
    auto act_row = [&](int64_t iy) { return x.data() + (size_t)(rows ? (rows[iy].i1 % ne11) : iy)*k; };
    auto dst_idx = [&](int64_t iy) { return (size_t)(rows ? rows[iy].i1 : iy)*nr0; };

    // --- mul_mat -----------------------------------------------------------------------------
    {
        std::vector<float> got((size_t)ny*nr0, -1e30f);
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t) th.emplace_back([&, t]{
            pxa_pxq_mul_mat_cpu(GGML_TYPE_PXQ4, A.buf.data(), nr0, k,
                                (const char *)x.data(), nb11, 0,
                                (char *)got.data(), nb1, 0, rows, ne11, ny, t, nth);
        });
        for (auto & j : th) j.join();

        double worst = 0.0;
        for (int64_t iy = 0; iy < ny; ++iy) {
            const float * xr = act_row(iy);
            for (int64_t ix = 0; ix < nr0; ++ix) {
                double ref = 0;
                for (int64_t j = 0; j < k; ++j) ref += (double)A.w[(size_t)ix*k + j]*(double)xr[j];
                const double d = std::fabs(got[dst_idx(iy) + ix] - ref)/row_mag(A, ix, xr);
                if (d > worst) worst = d;
            }
        }
        std::printf("   mul_mat      ny=%-4lld nth=%-2d %-7s max_E_MAG=%.3e\n",
                    (long long)ny, nth, routed ? "routed" : "dense", worst);
        check(worst < 3e-3, "pxa_pxq_mul_mat_cpu outside the codec error bound");
    }

    // --- fused up/gate -------------------------------------------------------------------------
    {
        std::vector<float> ub((size_t)nr0), gb((size_t)nr0);
        for (auto & v : ub) v = nrm(rng)*0.1f;
        for (auto & v : gb) v = nrm(rng)*0.1f;
        const int  uop   = GGML_UNARY_OP_SILU;
        const float limit = 0.0f;                 // no clamp: exercise the plain silu-swiglu arm

        std::vector<float> got((size_t)ny*nr0, -1e30f);
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t) th.emplace_back([&, t]{
            pxa_pxq_moe_up_gate_cpu(GGML_TYPE_PXQ4, A.buf.data(), GGML_TYPE_PXQ4, G.buf.data(),
                                    nr0, k, ub.data(), gb.data(),
                                    (const char *)x.data(), nb11, 0,
                                    (char *)got.data(), nb1, 0, rows, ne11, ny,
                                    uop, limit, t, nth);
        });
        for (auto & j : th) j.join();

        double worst = 0.0;
        for (int64_t iy = 0; iy < ny; ++iy) {
            const float * xr = act_row(iy);
            for (int64_t ix = 0; ix < nr0; ++ix) {
                double uv = ub[ix], gv = gb[ix];
                for (int64_t j = 0; j < k; ++j) {
                    uv += (double)A.w[(size_t)ix*k + j]*(double)xr[j];
                    gv += (double)G.w[(size_t)ix*k + j]*(double)xr[j];
                }
                const double act = gv/(1.0 + std::exp(-gv));
                const double ref = uv*act;
                // the product of two dots, so scale the bound by the two operand magnitudes
                const double mag = std::fabs(uv)*std::fabs(act)
                                 + row_mag(A, ix, xr)*std::fabs(act)
                                 + row_mag(G, ix, xr)*std::fabs(uv);
                const double d = std::fabs(got[dst_idx(iy) + ix] - ref)/mag;
                if (d > worst) worst = d;
            }
        }
        std::printf("   up_gate      ny=%-4lld nth=%-2d %-7s max_E_MAG=%.3e\n",
                    (long long)ny, nth, routed ? "routed" : "dense", worst);
        check(worst < 3e-3, "pxa_pxq_moe_up_gate_cpu outside the codec error bound");
    }
}

static void bench(int64_t nrows, int64_t k) {
    const float * book = nullptr; const float * s16 = nullptr; const float * s8 = nullptr;
    pxa_pxq_float_tables(&book, &s16, &s8);
    float bscale = 0.0f;
    const int8_t * bi8 = pxa_pxq_dot_book_i8(&bscale);
    std::mt19937 rng(7);
    Panels P = make_panels(false, nrows, k, rng, book, s16, s8, bi8, bscale);

    std::vector<float> x(k);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    for (auto & v : x) v = nrm(rng);
    std::vector<pxa_pxq_q8> xq(k/32);
    pxa_pxq_quantize_row_q8(x.data(), xq.data(), k);

    const int reps = 20;
    volatile float sink = 0.0f;

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < reps; ++it)
        for (int64_t r = 0; r < nrows; ++r) sink += pxa_pxq_dot_q8(GGML_TYPE_PXQ4, P.buf.data(), r, k, xq.data());
    auto t1 = std::chrono::steady_clock::now();

    std::vector<float> w(k);
    for (int it = 0; it < reps; ++it)
        for (int64_t r = 0; r < nrows; ++r) {
            pxa_pxq_dequant_row(GGML_TYPE_PXQ4, P.buf.data(), r, k, w.data());
            double a = 0; for (int64_t j = 0; j < k; ++j) a += (double)w[j]*(double)x[j];
            sink += (float)a;
        }
    auto t2 = std::chrono::steady_clock::now();

    const double dot_s = std::chrono::duration<double>(t1 - t0).count();
    const double deq_s = std::chrono::duration<double>(t2 - t1).count();
    const double weights = (double)reps*nrows*k;
    std::printf("single-thread gemv, %lldx%lld, %d reps\n", (long long)nrows, (long long)k, reps);
    std::printf("   int8 dot        %7.1f Mweight/s   %.3f s\n", weights/dot_s/1e6, dot_s);
    std::printf("   dequant + f64   %7.1f Mweight/s   %.3f s   (%.1fx slower)\n",
                weights/deq_s/1e6, deq_s, deq_s/dot_s);
    (void)sink;
}

int main() {
    for (int i = 0; i < (1 << 16); ++i) ggml_table_f32_f16[i] = _cvtsh_ss((uint16_t)i);
    std::printf("PXQ CPU int8 dot self-test\n");
    float bscale = 0.0f;
    const int8_t * bi8 = pxa_pxq_dot_book_i8(&bscale);
    std::printf("PX16 book -> int8 image (scale %.9g):\n  ", bscale);
    for (int c = 0; c < 16; ++c) std::printf("%4d", (int)bi8[c]);
    std::printf("\n\n");

    run_tier(false, 256, 1024, 1234);
    run_tier(false, 128, 4096, 5678);
    run_tier(true,  256, 1024, 4321);
    run_tier(true,  128, 4096, 8765);

    std::printf("\nwrappers (vs a naive f32 evaluation on pxa_pxq_dequant_row)\n");
    run_wrappers(128, 512,  1, 1, false, 11);    // decode: one activation row, one tile
    run_wrappers(128, 512, 17, 4, false, 12);    // ny straddles PXA_PXQ_NYT = 16
    run_wrappers(128, 512, 40, 4, true,  13);    // routed rows, multi-tile, nth > 1
    std::printf("\n");
    bench(2048, 4096);

    std::printf("\n%s\n", g_fail ? "FAILED" : "OK");
    return g_fail;
}
