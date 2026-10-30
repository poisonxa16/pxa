// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxq-mmvq-pft.cu -- 2026-10-16: PXA_PXQN_VWPF, the K1 pipelined PXQ MMVQ body at the verify
// widths (pxa/pxq-mmvq-pft.cuh), on the real 27B PXQN4 shapes, sm_70.
//
// For every width 1..8 it runs the shipped launch (mul_mat_vec_q<PXQN4, NY, 4, ROWS, 2>), the one-column K1 launch
// (mul_mat_vec_q_pf1, the T=1 decode path) on each column, and the pipelined candidates (rows per block x launch-bound
// min blocks x I2F form), checks EVERY output BY BITS against (a) the shipped launch and (b) the one-column K1 launch of
// the same column, and times each one (cuda events, median of TRIALS x REPS launches, us per launch, weight GB/s).
//
// Build (sm_70; the engine TU's nvcc flags), from the repo root:
//   nvcc -DGGML_USE_CUDA -DNDEBUG -Iggml/src -Iggml/include -I<build>/pxa-catalog -O3 -std=c++20 -use_fast_math
//        -extended-lambda --generate-code=arch=compute_70,code=[compute_70,sm_70] -include ggml/src/pxqn-host-isa.h
//        -x cu tests/test-pxq-mmvq-pft.cu -o test-pxq-mmvq-pft -lcuda
// Run on one V100 inside a card lock: test-pxq-mmvq-pft [-q] [reps] | -only <shape 0..4> <ny> <variant> (profiler mode).
// Nonzero exit on any bit difference.
#include "ggml-cuda/common.cuh"

template<> struct ggml_cuda_type_traits<GGML_TYPE_PXQN4> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
};

#include "ggml-cuda/mmvq-templates.cuh"
#include "ggml-cuda/pxa/pxq-mmvq-pft.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static int g_fail = 0, g_checks = 0, REPS = 30, TRIALS = 5;
static bool QUICK = false;

struct shp { int R, K; const char * name; int per_tok; };
// the 27B PXQN4 verify GEMVs on the q8_1 MMVQ route (QPN takes the up/gate): count per verify step in per_tok
static const shp SHAPES[] = {
    {  5120, 17408, "down 5120x17408",  64 },
    {  5120,  6144, "out  5120x6144 ",  64 },
    {  6144,  5120, "z    6144x5120 ",  48 },
    { 10240,  5120, "qkv 10240x5120 ",  48 },
    { 12288,  5120, "q   12288x5120 ",  16 },
};

struct bufs {
    uint8_t * W = nullptr; block_q8_1 * Y = nullptr; float * ref = nullptr; float * one = nullptr; float * out = nullptr;
    size_t wbytes = 0; int R = 0, K = 0;
};

static void fill(bufs & B, int R, int K, uint64_t seed) {
    const int kslabs = K/32, panels = R/64;
    const size_t pstride = (size_t)PXQ6_HDR_BYTES + (size_t)kslabs*pxq_mmvq_slab<GGML_TYPE_PXQN4>;
    B.wbytes = (size_t)panels*pstride; B.R = R; B.K = K;
    std::vector<uint8_t> W(B.wbytes);
    for (size_t i = 0; i < W.size(); ++i) W[i] = (uint8_t)(rs(seed) & 0xff);
    for (int p = 0; p < panels; ++p) {
        uint16_t * an = (uint16_t *)&W[(size_t)p*pstride];
        for (int r = 0; r < 64; ++r) {
            const int e = 10 + (int)(rs(seed) % 6);
            an[r] = (uint16_t)(((rs(seed) & 1) << 15) | (e << 10) | (rs(seed) & 0x3ff));
        }
    }
    std::vector<uint8_t> Y((size_t)8*kslabs*sizeof(block_q8_1));
    for (int t = 0; t < 8; ++t) for (int g = 0; g < kslabs; ++g) {
        uint8_t * b = &Y[((size_t)t*kslabs + g)*sizeof(block_q8_1)];
        const int e = 6 + (int)(rs(seed) % 4);
        ((uint16_t *)b)[0] = (uint16_t)((e << 10) | (rs(seed) & 0x3ff));
        ((uint16_t *)b)[1] = 0;
        for (int j = 0; j < 32; ++j) b[4 + j] = (uint8_t)(rs(seed) & 0xff);
    }
    CUCK(cudaMalloc(&B.W, B.wbytes)); CUCK(cudaMalloc(&B.Y, Y.size()));
    CUCK(cudaMalloc(&B.ref, (size_t)8*R*4)); CUCK(cudaMalloc(&B.one, (size_t)8*R*4)); CUCK(cudaMalloc(&B.out, (size_t)8*R*4));
    CUCK(cudaMemcpy(B.W, W.data(), B.wbytes, cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(B.Y, Y.data(), Y.size(), cudaMemcpyHostToDevice));
}
static void freeb(bufs & B) { cudaFree(B.W); cudaFree(B.Y); cudaFree(B.ref); cudaFree(B.one); cudaFree(B.out); }

template <class F>
static float time_us(F f) {
    cudaEvent_t a, b; CUCK(cudaEventCreate(&a)); CUCK(cudaEventCreate(&b));
    f(); CUCK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int tr = 0; tr < TRIALS; ++tr) {
        CUCK(cudaEventRecord(a));
        for (int r = 0; r < REPS; ++r) f();
        CUCK(cudaEventRecord(b)); CUCK(cudaEventSynchronize(b));
        float ms = 0; CUCK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms*1000.f/REPS);
    }
    std::sort(t.begin(), t.end());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return t[t.size()/2];
}

static bool same_bits(const float * d0, const float * d1, size_t n, const char * what, const char * tag, int ny) {
    std::vector<float> a(n), b(n);
    CUCK(cudaMemcpy(a.data(), d0, n*4, cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(b.data(), d1, n*4, cudaMemcpyDeviceToHost));
    size_t bad = 0; double amax = 0;
    for (size_t i = 0; i < n; ++i) { if (memcmp(&a[i], &b[i], 4)) ++bad; amax = fmax(amax, fabs((double)a[i])); if (!std::isfinite(a[i])) ++bad; }
    ++g_checks;
    if (bad || amax == 0) { ++g_fail; printf("  FAIL %s %s ny=%d: %zu/%zu differ |max| %g\n", what, tag, ny, bad, n, amax); return false; }
    return true;
}

template <class F>
static void attrs(F kern, int & regs, int & lmem, int & occ) {
    cudaFuncAttributes fa; CUCK(cudaFuncGetAttributes(&fa, (const void *) kern));
    regs = fa.numRegs; lmem = (int) fa.localSizeBytes;
    CUCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void *) kern, 128, 0));
}


template <int NY> static constexpr int SHIP_ROWS = NY <= 2 ? 4 : (NY <= 4 ? 8 : 4);

template <int NY>
static void launch_ship(const bufs & B, float * dst) {
    constexpr int rpb = mmvq_rows_per_block_plain<GGML_TYPE_PXQN4, NY, SHIP_ROWS<NY>>;
    mul_mat_vec_q<GGML_TYPE_PXQN4, NY, 4, SHIP_ROWS<NY>, 2><<<(B.R + rpb - 1)/rpb, dim3(WARP_SIZE, 4, 1)>>>(
        B.W, B.Y, dst, nullptr, nullptr, B.K, B.R, B.K, B.R, 0, 0, 0, 0, 0);
}
// the engine's launch at 4/6/7/8 columns today (PXA_PXQN_VERIFY_WIDE, mmvq-templates.cuh pxa_pxqn_verify_wide_try)
template <int NY> static constexpr int VW_ROWS = NY == 6 ? 8 : 4;
template <int NY> static constexpr int VW_MB   = NY == 6 ? 3 : 4;
template <int NY>
static void launch_vw(const bufs & B, float * dst) {
    constexpr int R = VW_ROWS<NY>;
    mul_mat_vec_q_mb<GGML_TYPE_PXQN4, NY, 4, R, 2, VW_MB<NY>><<<(B.R + R - 1)/R, dim3(WARP_SIZE, 4, 1)>>>(
        B.W, B.Y, dst, nullptr, nullptr, B.K, B.R, B.K, B.R, 0, 0, 0, 0, 0);
}
static void launch_pf1(const bufs & B, const block_q8_1 * y, float * dst) {
    mul_mat_vec_q_pf1<GGML_TYPE_PXQN4><<<B.R/4, dim3(WARP_SIZE, 4, 1)>>>(B.W, y, dst, B.K);
}
template <int NY, int ROWS, int MINB, bool X, bool PY>
static void launch_pft(const bufs & B, float * dst) {
    mul_mat_vec_q_pft<GGML_TYPE_PXQN4, NY, ROWS, MINB, X, PY><<<B.R/ROWS, dim3(WARP_SIZE, 4, 1)>>>(B.W, B.Y, dst, B.K, B.K, B.R);
}

static double gbs(const bufs & B, float us) { return (double) B.wbytes / (us * 1e3); }

template <int NY, int ROWS, int MINB, bool X, bool PY = false>
static void variant_pft(const bufs & B, const shp & S, float t_ship) {
    const size_t n = (size_t)NY*B.R;
    CUCK(cudaMemset(B.out, 0x5a, n*4));
    launch_pft<NY, ROWS, MINB, X, PY>(B, B.out); CUCK(cudaGetLastError()); CUCK(cudaDeviceSynchronize());
    char tag[64]; snprintf(tag, sizeof tag, "pft R%d/MB%d%s%s", ROWS, MINB, X ? "/X" : "", PY ? "/PY" : "");
    const bool ok1 = same_bits(B.ref, B.out, n, S.name, tag, NY);
    const bool ok2 = same_bits(B.one, B.out, n, S.name, tag, NY);
    const float t = time_us([&]{ launch_pft<NY, ROWS, MINB, X, PY>(B, B.out); });
    int regs, lmem, occ; attrs(mul_mat_vec_q_pft<GGML_TYPE_PXQN4, NY, ROWS, MINB, X, PY>, regs, lmem, occ);
    printf("    %-16s %8.2f us %6.1f GB/s %+6.1f%%  regs %3d lmem %4d B  blk/SM %2d  %s\n", tag, t, gbs(B, t),
           100.0*(t - t_ship)/t_ship, regs, lmem, occ, ok1 && ok2 ? "bits=ship=pf1" : "BITS DIFFER");
}

template <int NY>
static void variant_vw(const bufs & B, const shp & S, float t_ship) {
    const size_t n = (size_t)NY*B.R;
    CUCK(cudaMemset(B.out, 0x5a, n*4));
    launch_vw<NY>(B, B.out); CUCK(cudaGetLastError()); CUCK(cudaDeviceSynchronize());
    char tag[64]; snprintf(tag, sizeof tag, "vw R%d/MB%d", VW_ROWS<NY>, VW_MB<NY>);
    const bool ok1 = same_bits(B.ref, B.out, n, S.name, tag, NY);
    const bool ok2 = same_bits(B.one, B.out, n, S.name, tag, NY);
    const float t = time_us([&]{ launch_vw<NY>(B, B.out); });
    int regs, lmem, occ; attrs(mul_mat_vec_q_mb<GGML_TYPE_PXQN4, NY, 4, VW_ROWS<NY>, 2, VW_MB<NY>>, regs, lmem, occ);
    printf("    %-16s %8.2f us %6.1f GB/s %+6.1f%%  regs %3d lmem %4d B  blk/SM %2d  %s\n", tag, t, gbs(B, t),
           100.0*(t - t_ship)/t_ship, regs, lmem, occ, ok1 && ok2 ? "bits=ship=pf1" : "BITS DIFFER");
}

template <int NY>
static void width(const bufs & B, const shp & S) {
    const size_t n = (size_t)NY*B.R;
    CUCK(cudaMemset(B.ref, 0xa5, n*4));
    launch_ship<NY>(B, B.ref); CUCK(cudaGetLastError()); CUCK(cudaDeviceSynchronize());
    // the T=1 decode path on each column: the identity every verify column must meet
    CUCK(cudaMemset(B.one, 0x33, n*4));
    for (int j = 0; j < NY; ++j) launch_pf1(B, B.Y + (size_t)j*(B.K/32), B.one + (size_t)j*B.R);
    CUCK(cudaGetLastError()); CUCK(cudaDeviceSynchronize());
    same_bits(B.one, B.ref, n, S.name, "ship-vs-pf1", NY);
    const float t_ship = time_us([&]{ launch_ship<NY>(B, B.ref); });
    const float t_pf1  = time_us([&]{ launch_pf1(B, B.Y, B.out); });
    int regs, lmem, occ; attrs(mul_mat_vec_q<GGML_TYPE_PXQN4, NY, 4, SHIP_ROWS<NY>, 2>, regs, lmem, occ);
    printf("  ny=%d  ship R%d %8.2f us %6.1f GB/s  regs %3d blk/SM %2d | pf1 x1 %7.2f us %6.1f GB/s\n", NY, SHIP_ROWS<NY>,
           t_ship, gbs(B, t_ship), regs, occ, t_pf1, gbs(B, t_pf1));
    if constexpr (NY == 4 || NY == 6 || NY == 7 || NY == 8) variant_vw<NY>(B, S, t_ship);
    variant_pft<NY, 2, 4, false>(B, S, t_ship);
    variant_pft<NY, 2, 8, false>(B, S, t_ship);
    variant_pft<NY, 4, 2, false>(B, S, t_ship);
    variant_pft<NY, 4, 3, false>(B, S, t_ship);
    variant_pft<NY, 4, 4, false>(B, S, t_ship);
    variant_pft<NY, 4, 6, false>(B, S, t_ship);
    variant_pft<NY, 8, 1, false>(B, S, t_ship);
    variant_pft<NY, 8, 2, false>(B, S, t_ship);
    variant_pft<NY, 8, 3, false>(B, S, t_ship);
    variant_pft<NY, 4, 4, true >(B, S, t_ship);
    variant_pft<NY, 8, 2, true >(B, S, t_ship);
    variant_pft<NY, 4, 3, false, true>(B, S, t_ship);
    variant_pft<NY, 4, 4, false, true>(B, S, t_ship);
    variant_pft<NY, 8, 2, false, true>(B, S, t_ship);
    variant_pft<NY, 4, 4, true,  true>(B, S, t_ship);
}

template <int NY>
static void only(const bufs & B, int v) {
    for (int r = 0; r < 3; ++r) {
        switch (v) {
            case 0:  launch_ship<NY>(B, B.ref); break;
            case 1:  launch_pft<NY, 4, 4, false, false>(B, B.out); break;
            case 2:  launch_pft<NY, 8, 2, false, false>(B, B.out); break;
            case 3:  launch_pft<NY, 4, 4, false, true >(B, B.out); break;
            default: launch_pft<NY, 2, 8, false, false>(B, B.out); break;
        }
    }
    CUCK(cudaDeviceSynchronize());
}

int main(int argc, char ** argv) {
    int only_s = -1, only_ny = 0, only_v = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-q")) QUICK = true;
        else if (!strcmp(argv[i], "-only") && i + 3 < argc) { only_s = atoi(argv[i+1]); only_ny = atoi(argv[i+2]); only_v = atoi(argv[i+3]); i += 3; }
        else REPS = atoi(argv[i]);
    }
    if (only_s >= 0) {   // profiler mode: three launches of one kernel on one shape at one width
        bufs B; fill(B, SHAPES[only_s].R, SHAPES[only_s].K, 0x9e3779b97f4a7c15ull);
        switch (only_ny) { case 1: only<1>(B, only_v); break; case 2: only<2>(B, only_v); break; case 3: only<3>(B, only_v); break;
            case 4: only<4>(B, only_v); break; case 5: only<5>(B, only_v); break; case 6: only<6>(B, only_v); break;
            case 7: only<7>(B, only_v); break; default: only<8>(B, only_v); break; }
        freeb(B); return 0;
    }
    cudaDeviceProp pr; CUCK(cudaGetDeviceProperties(&pr, 0));
    printf("test-pxq-mmvq-pft: %s sm_%d%d, %d SMs, REPS %d TRIALS %d\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount, REPS, TRIALS);
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    for (const shp & S : SHAPES) {
        bufs B; fill(B, S.R, S.K, seed);
        printf("== %s (%.1f MB weights)\n", S.name, B.wbytes/1e6);
        width<1>(B, S); width<2>(B, S); width<3>(B, S); width<4>(B, S);
        width<5>(B, S); width<6>(B, S); width<7>(B, S); width<8>(B, S);
        freeb(B);
        if (QUICK) break;
    }
    printf("%s: %d bit checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
