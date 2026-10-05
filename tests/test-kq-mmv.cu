// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-kq-mmv.cu -- correctness, invariance and determinism of the k-quant decode GEMV (ggml/src/ggml-cuda/pxa/
// kq-mmv.cuh) against a double reference and against the incumbent q8_1 MMVQ on the same data.
//
// Checks (exit 0 = every one passed):
//   (a) accuracy: Q4_K / Q5_K / Q6_K / Q8_0 x ny 1..8 x the 27B decode shapes (incl. the tensor-split halves
//       and the 248320-row head), per path (h2 = sm_60 half2, i8 = dp4a). Reference: the dequantized weights
//       (host decode written from the block layouts) times the f32 x, in double, on a sample of rows.
//       Errors are normalised per (row, column) by sqrt(sum_k (w_k x_k)^2). Pass: the new path's max and rms
//       error <= the incumbent q8_1 MMVQ's on the same data (h2: strictly; i8 shares the q8_1 quantizer, so it
//       may sit at the incumbent's level: <= 1.10x).
//   (b) group launch (4 matrices, different R, one launch) bit-identical to one launch per matrix
//   (c) batch invariance: column c at ny=k bit-identical to the ny=1 run of column c (every row)
//   (d) gate/up epilogue bit-identical to GEMV(up) + GEMV(gate) + the kernel's own GLU, limit 0 and 7
//   (e) determinism: 5 runs byte-equal (ny 1 and 8)
//   (g) the wide form (Q8_0, x does not fit the block) bit-identical to the chunked form
//   (f) -bench: GB/s of the weight bytes per shape at ny=1, incumbent (quantize + MMVQ) vs new (information)
//
// Build: in the tree (tests/CMakeLists.txt, target test-kq-mmv), or standalone:
//   nvcc -O3 -std=c++17 -use_fast_math -extended-lambda -arch=sm_60 -I ggml/src -I ggml/include -DGGML_USE_CUDA \
//        -o test-kq-mmv tests/test-kq-mmv.cu
// Run:   ./test-kq-mmv [-bench | -bench-only] [-h2 | -i8 (bench path)] [-quick]
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmvq-templates.cuh"
#include "ggml-cuda/pxa/kq-mmv.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

static uint64_t g_seed = 0x6b712d6d6d762d31ull;
static inline uint64_t rnd() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17; return g_seed; }
static inline float rndf() { return (float)(rnd() & 0xffffff)/16777216.f; }   // [0,1)

static int g_nsm = 56;
static int g_cc  = 600;

// ------------------------------------------------------------------------------------------------ types
struct tinfo { int kind; ggml_type gt; const char * name; int qk, bb; };
static const tinfo TYPES[4] = {
    { KQ_Q4_K, GGML_TYPE_Q4_K, "Q4_K", 256, 144 },
    { KQ_Q5_K, GGML_TYPE_Q5_K, "Q5_K", 256, 176 },
    { KQ_Q6_K, GGML_TYPE_Q6_K, "Q6_K", 256, 210 },
    { KQ_Q8_0, GGML_TYPE_Q8_0, "Q8_0", 32,  34  },
};

static inline uint16_t f2h(float f) { __half h = __float2half_rn(f); uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float h2f(const uint8_t * p) { uint16_t u = (uint16_t)(p[0] | (p[1] << 8)); __half h; memcpy(&h, &u, 2); return __half2float(h); }

// random blocks with sane scales (every other byte pattern is a valid code)
static void gen_weights(const tinfo & T, std::vector<uint8_t> & w, int64_t R, int64_t K) {
    const int64_t nb = K/T.qk;
    w.resize((size_t)(R*nb*T.bb));
    uint64_t * p = (uint64_t *) w.data();
    const size_t n8 = w.size()/8;
    for (size_t i = 0; i < n8; ++i) p[i] = rnd();
    for (size_t i = n8*8; i < w.size(); ++i) w[i] = (uint8_t) rnd();
    for (int64_t b = 0; b < R*nb; ++b) {
        uint8_t * blk = w.data() + b*T.bb;
        auto put = [](uint8_t * q, float f) { const uint16_t u = f2h(f); q[0] = u & 0xff; q[1] = u >> 8; };
        switch (T.kind) {
            case KQ_Q4_K: case KQ_Q5_K: put(blk, 1e-3f*(0.5f + rndf())); put(blk + 2, 2e-3f*(0.5f + rndf())); break;
            case KQ_Q6_K: put(blk + 208, 1e-3f*(0.5f + rndf())); break;
            case KQ_Q8_0: put(blk, 1e-2f*(0.5f + rndf())); break;
        }
    }
}

static void scale_min(const uint8_t * q, int j, int & sc, int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// host decode of one row, from the block layouts
static void deq_row(const tinfo & T, const uint8_t * row, int64_t K, std::vector<double> & y) {
    y.assign((size_t)K, 0.0);
    const int64_t nb = K/T.qk;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * B = row + b*T.bb;
        double * o = y.data() + b*T.qk;
        if (T.kind == KQ_Q8_0) {
            const double d = h2f(B);
            for (int i = 0; i < 32; ++i) o[i] = d*(int8_t)B[2 + i];
        } else if (T.kind == KQ_Q6_K) {
            const double d = h2f(B + 208);
            const uint8_t * ql = B, * qh = B + 128; const int8_t * sc = (const int8_t *)(B + 192);
            for (int n = 0; n < 2; ++n) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l/16;
                    const int q1 = ((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = ((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = ((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    o[l]      = d*sc[is + 0]*q1;
                    o[l + 32] = d*sc[is + 2]*q2;
                    o[l + 64] = d*sc[is + 4]*q3;
                    o[l + 96] = d*sc[is + 6]*q4;
                }
                o += 128; ql += 64; qh += 32; sc += 8;
            }
        } else {
            const double d = h2f(B), dmin = h2f(B + 2);
            const uint8_t * scl = B + 4;
            const bool q5 = T.kind == KQ_Q5_K;
            const uint8_t * qh = B + 16, * qs = B + (q5 ? 48 : 16);
            for (int j = 0; j < 4; ++j) {
                int sc1, m1, sc2, m2;
                scale_min(scl, 2*j, sc1, m1); scale_min(scl, 2*j + 1, sc2, m2);
                for (int l = 0; l < 32; ++l) {
                    const int h1 = q5 ? ((qh[l] >> (2*j)) & 1) << 4 : 0;
                    const int h2 = q5 ? ((qh[l] >> (2*j + 1)) & 1) << 4 : 0;
                    o[64*j + l]      = d*sc1*((qs[32*j + l] & 0xF) + h1) - dmin*m1;
                    o[64*j + 32 + l] = d*sc2*((qs[32*j + l] >>  4) + h2) - dmin*m2;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ incumbent
static __global__ void t_quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx,
                                       const int64_t kx0_padded, const int64_t sx) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
    if (ix0 >= kx0_padded) return;
    const int64_t ix1 = blockIdx.y;
    const int64_t i_padded = ix1*kx0_padded + ix0;
    block_q8_1 * y = (block_q8_1 *) vy;
    const int64_t ib = i_padded / QK8_1, iqs = i_padded % QK8_1;
    const float xi = ix0 < kx ? x[ix1*sx + ix0] : 0.0f;
    float amax = fabsf(xi), sum = xi;
    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);
    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;
    if (iqs > 0) return;
    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template <ggml_type type, int NY>
static void inc_launch_t(const void * w, const void * q8, float * dst, int R, int K, int Kpad) {
    constexpr int nw  = NY <= 4 ? 4 : 2;
    constexpr int rpb = NY < 4 ? 1 : 2;
    mul_mat_vec_q<type, NY, nw, 1, 2><<<(R + rpb - 1)/rpb, dim3(WARP_SIZE, nw, 1)>>>(
        w, q8, dst, nullptr, nullptr, K, R, Kpad, R, 0, 0, 0, 0, 0);
}
template <ggml_type type>
static void inc_launch_ny(int ny, const void * w, const void * q8, float * dst, int R, int K, int Kpad) {
    switch (ny) {
        case 1: inc_launch_t<type, 1>(w, q8, dst, R, K, Kpad); break;
        case 2: inc_launch_t<type, 2>(w, q8, dst, R, K, Kpad); break;
        case 3: inc_launch_t<type, 3>(w, q8, dst, R, K, Kpad); break;
        case 4: inc_launch_t<type, 4>(w, q8, dst, R, K, Kpad); break;
        case 5: inc_launch_t<type, 5>(w, q8, dst, R, K, Kpad); break;
        case 6: inc_launch_t<type, 6>(w, q8, dst, R, K, Kpad); break;
        case 7: inc_launch_t<type, 7>(w, q8, dst, R, K, Kpad); break;
        case 8: inc_launch_t<type, 8>(w, q8, dst, R, K, Kpad); break;
    }
}
static void inc_launch(const tinfo & T, int ny, const float * x, int64_t sx, void * q8, const void * w, float * dst,
                       int R, int K) {
    const int Kpad = (K + 511)/512*512;
    t_quantize_q8_1<<<dim3((Kpad + 255)/256, ny, 1), 256>>>(x, q8, K, Kpad, sx);
    switch (T.kind) {
        case KQ_Q4_K: inc_launch_ny<GGML_TYPE_Q4_K>(ny, w, q8, dst, R, K, Kpad); break;
        case KQ_Q5_K: inc_launch_ny<GGML_TYPE_Q5_K>(ny, w, q8, dst, R, K, Kpad); break;
        case KQ_Q6_K: inc_launch_ny<GGML_TYPE_Q6_K>(ny, w, q8, dst, R, K, Kpad); break;
        case KQ_Q8_0: inc_launch_ny<GGML_TYPE_Q8_0>(ny, w, q8, dst, R, K, Kpad); break;
    }
    CUCK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------ new path
struct mm { const void * w; float * dst; int R; int64_t sd; const float * bias; };

// the same launch builder the engine uses (kqmmv_run_mats in kq-mmv.cuh)
static void new_launch(const tinfo & T, int path, int ny, const float * x, int64_t sx, int K,
                       const mm * ms, int n, bool gu, float limit, bool wide = true) {
    kqmmv_mdesc d[KQMMV_MAX_MATS];
    const int64_t nbrow = (int64_t)(K/T.qk)*T.bb;
    for (int i = 0; i < n; ++i) d[i] = kqmmv_mdesc{ ms[i].w, ms[i].dst, nbrow, ms[i].sd, ms[i].R, ms[i].bias };
    CUCK(kqmmv_run_mats(T.kind, path, ny, x, sx, K, d, n, gu, limit, g_nsm, 0, wide));
}

// ------------------------------------------------------------------------------------------------ helpers
static std::vector<float> dl(const float * d, size_t n) {
    std::vector<float> h(n);
    CUCK(cudaDeviceSynchronize());
    CUCK(cudaMemcpy(h.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}
static bool bits_eq(const float * a, const float * b, size_t n, size_t * first = nullptr) {
    if (memcmp(a, b, n*sizeof(float)) == 0) return true;
    if (first) for (size_t i = 0; i < n; ++i) if (memcmp(a + i, b + i, 4) != 0) { *first = i; break; }
    return false;
}
static bool all_finite(const std::vector<float> & v) {
    for (float f : v) if (!std::isfinite(f)) return false;
    return true;
}

struct errs { double maxe = 0, se = 0, sr = 0; };
static void acc_err(errs & e, const std::vector<float> & y, int64_t R, int ny, const std::vector<int> & rows,
                    const std::vector<double> & ref, const std::vector<double> & mag) {
    for (size_t s = 0; s < rows.size(); ++s)
        for (int j = 0; j < ny; ++j) {
            const double r = ref[s*8 + j], m = mag[s*8 + j];
            const double d = (double) y[(size_t)j*R + rows[s]] - r;
            if (m > 0) e.maxe = std::max(e.maxe, fabs(d)/m);
            e.se += d*d; e.sr += r*r;
        }
}

static void fill_x(std::vector<float> & x, int64_t K) {
    x.resize((size_t)8*K);
    for (auto & v : x) v = (rndf()*2.f - 1.f)*((rnd() & 63) == 0 ? 24.f : 1.f);
}

// ------------------------------------------------------------------------------------------------ (a)(c)(e)
static void test_shape(const tinfo & T, int64_t K, int64_t R, bool quick) {
    std::vector<uint8_t> hw; gen_weights(T, hw, R, K);
    std::vector<float> hx; fill_x(hx, K);
    const int64_t nbrow = (K/T.qk)*T.bb;

    // sampled rows: first, last, and a stride
    std::vector<int> rows;
    const int64_t ns = std::min<int64_t>(R, quick ? 64 : 192);
    for (int64_t s = 0; s < ns; ++s) rows.push_back((int)(s*(R - 1)/std::max<int64_t>(ns - 1, 1)));
    std::vector<double> ref(rows.size()*8), mag(rows.size()*8), wr;
    for (size_t s = 0; s < rows.size(); ++s) {
        deq_row(T, hw.data() + (size_t)rows[s]*nbrow, K, wr);
        for (int j = 0; j < 8; ++j) {
            double acc = 0, m2 = 0;
            for (int64_t k = 0; k < K; ++k) { const double p = wr[k]*(double)hx[(size_t)j*K + k]; acc += p; m2 += p*p; }
            ref[s*8 + j] = acc; mag[s*8 + j] = sqrt(m2);
        }
    }

    void * dw; float * dx; void * dq8; float * dy0; float * dy1; float * dyc;
    const int64_t Kpad = (K + 511)/512*512;
    CUCK(cudaMalloc(&dw, hw.size() + 256));
    CUCK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&dx, hx.size()*sizeof(float)));
    CUCK(cudaMemcpy(dx, hx.data(), hx.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&dq8, (size_t)8*Kpad/32*sizeof(block_q8_1)));
    CUCK(cudaMalloc(&dy0, (size_t)8*R*sizeof(float)));
    CUCK(cudaMalloc(&dy1, (size_t)8*R*sizeof(float)));
    CUCK(cudaMalloc(&dyc, (size_t)8*R*sizeof(float)));

    // incumbent errors per ny
    errs inc[9];
    for (int ny = 1; ny <= 8; ++ny) {
        inc_launch(T, ny, dx, K, dq8, dw, dy0, (int)R, (int)K);
        acc_err(inc[ny], dl(dy0, (size_t)ny*R), R, ny, rows, ref, mag);
    }

    for (int path = 0; path < 2; ++path) {
        const char * pn = path == KQMMV_H2 ? "h2" : "i8";
        // ny=1 per column (the batch-invariance reference)
        std::vector<float> col1((size_t)8*R);
        for (int c = 0; c < 8; ++c) {
            mm m{dw, dyc, (int)R, R};
            new_launch(T, path, 1, dx + (size_t)c*K, K, (int)K, &m, 1, false, 0.f);
            auto h = dl(dyc, (size_t)R);
            memcpy(col1.data() + (size_t)c*R, h.data(), (size_t)R*sizeof(float));
        }
        double worst_max = 0, worst_rms = 0, n1max = 0, n1rms = 0;
        for (int ny = 1; ny <= 8; ++ny) {
            mm m{dw, dy1, (int)R, R};
            new_launch(T, path, ny, dx, K, (int)K, &m, 1, false, 0.f);
            auto y = dl(dy1, (size_t)ny*R);
            CHECK(all_finite(y), "%s %s K=%lld R=%lld ny=%d: non-finite output", T.name, pn, (long long)K, (long long)R, ny);
            errs e; acc_err(e, y, R, ny, rows, ref, mag);
            const double rms_n = sqrt(e.se/std::max(e.sr, 1e-300)), rms_i = sqrt(inc[ny].se/std::max(inc[ny].sr, 1e-300));
            const double slack = path == KQMMV_H2 ? 1.0 : 1.10;
            CHECK(e.maxe <= inc[ny].maxe*slack && rms_n <= rms_i*slack,
                  "%s %s K=%lld R=%lld ny=%d: max %.3e (inc %.3e) rms %.3e (inc %.3e)", T.name, pn,
                  (long long)K, (long long)R, ny, e.maxe, inc[ny].maxe, rms_n, rms_i);
            if (ny == 1) { n1max = e.maxe; n1rms = rms_n; }
            worst_max = std::max(worst_max, e.maxe/std::max(inc[ny].maxe, 1e-300));
            worst_rms = std::max(worst_rms, rms_n/std::max(rms_i, 1e-300));
            // (c) batch invariance
            for (int c = 0; c < ny; ++c) {
                size_t f = 0;
                const bool eq = bits_eq(y.data() + (size_t)c*R, col1.data() + (size_t)c*R, (size_t)R, &f);
                CHECK(eq, "%s %s K=%lld R=%lld ny=%d col %d: differs from ny=1 at row %zu (%.9g vs %.9g)", T.name, pn,
                      (long long)K, (long long)R, ny, c, f, y[(size_t)c*R + f], col1[(size_t)c*R + f]);
            }
            // (e) determinism
            // (g) the wide form (k_kq_mmv_rw) equals the chunked k_kq_mmv form bit for bit
            if (kqmmv_rw_fits(T.kind, path, ny, (int)K)) {
                new_launch(T, path, ny, dx, K, (int)K, &m, 1, false, 0.f, false);
                auto yc = dl(dy1, (size_t)ny*R);
                CHECK(bits_eq(y.data(), yc.data(), y.size()), "%s %s K=%lld R=%lld ny=%d: wide form != chunked form",
                      T.name, pn, (long long)K, (long long)R, ny);
            }
            if (ny == 1 || ny == 8) {
                for (int rep = 0; rep < 4; ++rep) {
                    new_launch(T, path, ny, dx, K, (int)K, &m, 1, false, 0.f);
                    auto y2 = dl(dy1, (size_t)ny*R);
                    CHECK(bits_eq(y.data(), y2.data(), y.size()), "%s %s K=%lld R=%lld ny=%d: run %d not byte-equal",
                          T.name, pn, (long long)K, (long long)R, ny, rep + 2);
                }
            }
        }
        printf("  %s %s K=%5lld R=%6lld  err/incumbent worst over ny: max %.3f rms %.3f  (ny=1: new max %.2e rms %.2e | inc max %.2e rms %.2e)\n",
               T.name, pn, (long long)K, (long long)R, worst_max, worst_rms, n1max, n1rms, inc[1].maxe, sqrt(inc[1].se/inc[1].sr));
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dq8); cudaFree(dy0); cudaFree(dy1); cudaFree(dyc);
}

// ------------------------------------------------------------------------------------------------ (b) group
static void test_group(const tinfo & T) {
    const int K = 5120; const int Rs[4] = { 10240, 6144, 48, 1024 };
    std::vector<float> hx; fill_x(hx, K);
    float * dx; CUCK(cudaMalloc(&dx, hx.size()*4)); CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
    void * dw[4]; float * da[4]; float * db[4];
    for (int i = 0; i < 4; ++i) {
        std::vector<uint8_t> hw; gen_weights(T, hw, Rs[i], K);
        CUCK(cudaMalloc(&dw[i], hw.size() + 256)); CUCK(cudaMemcpy(dw[i], hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CUCK(cudaMalloc(&da[i], (size_t)8*Rs[i]*4)); CUCK(cudaMalloc(&db[i], (size_t)8*Rs[i]*4));
    }
    for (int path = 0; path < 2; ++path)
        for (int ny = 1; ny <= 8; ++ny) {
            mm ms[4];
            for (int i = 0; i < 4; ++i) { ms[i] = mm{dw[i], da[i], Rs[i], Rs[i]}; new_launch(T, path, ny, dx, K, K, &ms[i], 1, false, 0.f); }
            for (int i = 0; i < 4; ++i) ms[i].dst = db[i];
            new_launch(T, path, ny, dx, K, K, ms, 4, false, 0.f);
            for (int i = 0; i < 4; ++i) {
                auto a = dl(da[i], (size_t)ny*Rs[i]), b = dl(db[i], (size_t)ny*Rs[i]);
                CHECK(bits_eq(a.data(), b.data(), a.size()), "group %s %s ny=%d matrix %d (R=%d): grouped != per-matrix",
                      T.name, path ? "i8" : "h2", ny, i, Rs[i]);
            }
        }
    for (int i = 0; i < 4; ++i) { cudaFree(dw[i]); cudaFree(da[i]); cudaFree(db[i]); }
    cudaFree(dx);
}

// ------------------------------------------------------------------------------------------------ (d) GU
static void test_gu(const tinfo & T) {
    const int K = 5120, R = 8704;
    std::vector<float> hx; fill_x(hx, K);
    for (auto & v : hx) v *= 4.f;   // push the gate into both silu tails and the limit clamp
    float * dx; CUCK(cudaMalloc(&dx, hx.size()*4)); CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
    void * du; void * dg;
    std::vector<uint8_t> hw;
    gen_weights(T, hw, R, K); CUCK(cudaMalloc(&du, hw.size() + 256)); CUCK(cudaMemcpy(du, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    gen_weights(T, hw, R, K); CUCK(cudaMalloc(&dg, hw.size() + 256)); CUCK(cudaMemcpy(dg, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    float * yu; float * yg; float * yo; float * yf;
    CUCK(cudaMalloc(&yu, (size_t)8*R*4)); CUCK(cudaMalloc(&yg, (size_t)8*R*4));
    CUCK(cudaMalloc(&yo, (size_t)8*R*4)); CUCK(cudaMalloc(&yf, (size_t)8*R*4));
    const float limits[2] = { 0.0f, 7.0f };
    for (int path = 0; path < 2; ++path)
        for (int li = 0; li < 2; ++li)
            for (int ny = 1; ny <= 8; ++ny) {
                mm mu{du, yu, R, R}, mg{dg, yg, R, R};
                new_launch(T, path, ny, dx, K, K, &mu, 1, false, 0.f);
                new_launch(T, path, ny, dx, K, K, &mg, 1, false, 0.f);
                k_kq_glu<<<(ny*R + 255)/256, 256>>>(yg, yu, yo, ny*R, limits[li]);
                mm two[2] = { mm{du, yf, R, R}, mm{dg, yf, R, R} };
                new_launch(T, path, ny, dx, K, K, two, 2, true, limits[li]);
                auto a = dl(yo, (size_t)ny*R), b = dl(yf, (size_t)ny*R);
                size_t f = 0;
                const bool eq = bits_eq(a.data(), b.data(), a.size(), &f);
                CHECK(eq && all_finite(b), "GU %s %s limit %g ny=%d: fused != GEMV+GEMV+GLU (first diff %zu: %.9g vs %.9g)",
                      T.name, path ? "i8" : "h2", limits[li], ny, f, a[f], b[f]);
            }
    cudaFree(dx); cudaFree(du); cudaFree(dg); cudaFree(yu); cudaFree(yg); cudaFree(yo); cudaFree(yf);
}

// ------------------------------------------------------------------------------------------------ (g) bias epilogue
// the fused per-row ADD (PXA_KQMMV_BIAS) must equal the plain GEMV followed by an f32 add, bit for bit
static __global__ void k_t_add_rows(const float * y, const float * b, float * o, int R, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) o[i] = y[i] + b[i % R];
}
static void test_bias(const tinfo & T) {
    const int64_t shapes[3][2] = { {3072, 5120}, {8704, 5120}, {5120, 5120} };
    for (const auto & sh : shapes) {
        const int K = (int) sh[0], R = (int) sh[1];
        std::vector<float> hx; fill_x(hx, K);
        std::vector<float> hb(R); for (auto & v : hb) v = (rndf()*2.f - 1.f)*8.f;
        std::vector<uint8_t> hw; gen_weights(T, hw, R, K);
        float * dx; float * db; void * dw; float * y; float * o; float * f;
        CUCK(cudaMalloc(&dx, hx.size()*4)); CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
        CUCK(cudaMalloc(&db, (size_t)R*4)); CUCK(cudaMemcpy(db, hb.data(), (size_t)R*4, cudaMemcpyHostToDevice));
        CUCK(cudaMalloc(&dw, hw.size() + 256)); CUCK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CUCK(cudaMalloc(&y, (size_t)8*R*4)); CUCK(cudaMalloc(&o, (size_t)8*R*4)); CUCK(cudaMalloc(&f, (size_t)8*R*4));
        for (int path = 0; path < 2; ++path)
            for (int ny = 1; ny <= 2; ++ny) {
                mm m{dw, y, R, R};
                new_launch(T, path, ny, dx, K, K, &m, 1, false, 0.f);
                k_t_add_rows<<<(ny*R + 255)/256, 256>>>(y, db, o, R, ny*R);
                mm mb{dw, f, R, R, db};
                new_launch(T, path, ny, dx, K, K, &mb, 1, false, 0.f);
                auto a = dl(o, (size_t)ny*R), b = dl(f, (size_t)ny*R);
                size_t fi = 0;
                const bool eq = bits_eq(a.data(), b.data(), a.size(), &fi);
                CHECK(eq, "bias %s %s K=%d R=%d ny=%d: fused add != GEMV + ADD (first diff %zu: %.9g vs %.9g)",
                      T.name, path ? "i8" : "h2", K, R, ny, fi, a[fi], b[fi]);
            }
        cudaFree(dx); cudaFree(db); cudaFree(dw); cudaFree(y); cudaFree(o); cudaFree(f);
    }
}

// ------------------------------------------------------------------------------------------------ (f) bench
static void bench_shape(const tinfo & T, int64_t K, int64_t R, int path, int ny = 1) {
    std::vector<uint8_t> hw; gen_weights(T, hw, R, K);
    std::vector<float> hx; fill_x(hx, K);
    const int64_t Kpad = (K + 511)/512*512;
    void * dw; float * dx; void * dq8; float * dy;
    CUCK(cudaMalloc(&dw, hw.size() + 256)); CUCK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&dx, hx.size()*4)); CUCK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
    CUCK(cudaMalloc(&dq8, (size_t)8*Kpad/32*sizeof(block_q8_1))); CUCK(cudaMalloc(&dy, (size_t)8*R*4));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int it = 50;
    float t_inc = 1e30f, t_new = 1e30f;
    for (int rep = 0; rep < 3; ++rep) {
        float ms;
        cudaEventRecord(e0);
        for (int i = 0; i < it; ++i) inc_launch(T, ny, dx, K, dq8, dw, dy, (int)R, (int)K);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); t_inc = std::min(t_inc, ms/it);
        mm m{dw, dy, (int)R, R};
        cudaEventRecord(e0);
        for (int i = 0; i < it; ++i) new_launch(T, path, ny, dx, K, (int)K, &m, 1, false, 0.f);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); t_new = std::min(t_new, ms/it);
    }
    const double gb = (double)hw.size()/1e9;
    printf("  bench %s %s ny=%d K=%5lld R=%6lld  incumbent %7.1f us %6.1f GB/s | new %7.1f us %6.1f GB/s  (x%.2f)\n",
           T.name, path ? "i8" : "h2", ny, (long long)K, (long long)R, t_inc*1e3, gb/(t_inc*1e-3), t_new*1e3, gb/(t_new*1e-3), t_inc/t_new);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dw); cudaFree(dx); cudaFree(dq8); cudaFree(dy);
}

int main(int argc, char ** argv) {
    bool bench = false, quick = false, only_bench = false, width = false;
    int only_type = -1; long long only_K = 0, only_R = 0;
    int bench_path = -1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-bench")) bench = true;
        else if (!strcmp(argv[i], "-bench-only")) { bench = true; only_bench = true; }
        else if (!strcmp(argv[i], "-quick")) quick = true;
        else if (!strcmp(argv[i], "-width")) { width = true; bench = true; only_bench = true; }   // verify-width table
        else if (!strcmp(argv[i], "-h2")) bench_path = KQMMV_H2;
        else if (!strcmp(argv[i], "-i8")) bench_path = KQMMV_I8;
        else if (!strcmp(argv[i], "-shape") && i + 3 < argc) {   // -shape <type index 0..3> K R: bench that one only
            only_type = atoi(argv[i + 1]); only_K = atoll(argv[i + 2]); only_R = atoll(argv[i + 3]); i += 3;
            bench = true; only_bench = true;
        }
    }
    cudaDeviceProp prop; CUCK(cudaGetDeviceProperties(&prop, 0));
    g_nsm = prop.multiProcessorCount; g_cc = prop.major*100 + prop.minor*10;
    printf("test-kq-mmv on %s (cc %d, %d SMs)%s\n", prop.name, g_cc, g_nsm, quick ? " [quick]" : "");

    // K x R: the 27B decode shapes incl. the tensor-split halves and the 248320-row head
    const int64_t shapes[][2] = { {5120, 10240}, {5120, 6144}, {5120, 12288}, {5120, 1024}, {5120, 48},
                                  {5120, 8704}, {8704, 5120}, {3072, 5120}, {17408, 5120}, {5120, 248320} };
    const int nshapes = quick ? 3 : (int)(sizeof(shapes)/sizeof(shapes[0]));
    for (const tinfo & T : TYPES) {
        printf("%s\n", T.name);
        if (!only_bench) for (int s = 0; s < nshapes; ++s) test_shape(T, shapes[s][0], shapes[s][1], quick);
        if (!only_bench) test_group(T);
        if (!only_bench) test_gu(T);
        if (!only_bench) test_bias(T);
    }
    if (bench) {
        const int path = bench_path >= 0 ? bench_path : (g_cc < 610 ? KQMMV_H2 : KQMMV_I8);
        if (width) {
            // verify widths: the Q8_0 LM head (whole / tensor-split half), attention k/v, DeltaNet beta/alpha
            const int64_t ws[][2] = { {5120, 248320}, {5120, 124160}, {5120, 1024}, {5120, 512}, {5120, 48}, {5120, 24}, {17408, 5120} };
            for (const auto & s : ws)
                for (int ny : {1, 2, 3, 4, 5, 6, 8}) if (!getenv("KQW_NY") || atoi(getenv("KQW_NY")) == ny) bench_shape(TYPES[KQ_Q8_0], s[0], s[1], path, ny);
        } else if (only_type >= 0) bench_shape(TYPES[only_type & 3], only_K, only_R, path);
        else {
            for (const tinfo & T : TYPES)
                for (const auto & s : shapes) bench_shape(T, s[0], s[1], path);
            // verify widths on the widest per-card shape
            for (const tinfo & T : TYPES)
                for (int ny : {2, 4, 8}) bench_shape(T, 5120, 10240, path, ny);
        }
    }
    printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
