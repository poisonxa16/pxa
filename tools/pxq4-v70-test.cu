// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxq4-v70-test.cu -- standalone correctness + ceiling harness for the sm_70 register-direct
//                     PXQ4 m8n8k4 prefill GEMM (ggml/src/ggml-cuda/pxa/pxq4-v70.cuh).
//
// Build (no llama.cpp link):
//   nvcc -std=c++17 -O3 -arch=sm_70 -lineinfo -Xptxas -v \
//        -I<wt>/ggml/include -I<wt>/ggml/src -I<wt>/ggml/src/ggml-cuda \
//        tools/pxq4-v70-test.cu -o pxq4-v70-test -lcublas
// Run (V100 only, always under the lock):
//   flock -w 3600 /tmp/pxa-v100-bench.lock -c './pxq4-v70-test'
//
// PHASES
//   #1 decode byte-identity   -- memcmp of the decoded fp16 weights against
//                                k_pxq6_dequant_matrix (the cuBLAS route's own dequant).
//                                ZERO tolerance. Passing it reduces the whole numeric
//                                argument against the incumbent to accumulation order.
//   #2 reference GEMM         -- CPU double-precision reference, ragged N, small shapes.
//   #3 bit-stability          -- five runs from identical inputs, all five memcmp'd.
//   #4 guard halo / tail      -- guard floats fore and aft, NaN-filled interior, ragged M.
//                                the specific defence against the blockDim.x != 32 and the
//                                get_i-perm traps, both of which corrupt silently.
//   #5 full-size parity       -- six PXQ4 shapes x M in {32,64,512,2048} against the
//                                incumbent route reproduced exactly (dequant -> cublasGemmEx).
//   #0 tile ceiling            -- --ablate: V70 (decode INCLUDED) against a cublasGemmEx that is
//                                handed already-dequantized fp16 weights, i.e. cuBLAS gets its
//                                dequant for free. So the column is V70's tile+decode against a
//                                pure cuBLAS tile: the ceiling question, K7's table format.
//                                It does NOT bypass V70's decode -- there is no fp16 A path.
//   timing                    -- --time: the TFLOPS column that populates the route table.
// Correctness phases abort before any timing loop. Default run is correctness only.

#define PXQ4_V70_SELFTEST   // compiles k_pxq4_v70_decode_dump, which libggml must not carry
#include "pxa/pxq4-v70.cuh"

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2);} } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %s:%d status=%d\n", __FILE__, __LINE__, (int)s_); exit(2);} } while (0)

static int g_fail = 0;
static void report(const char * name, bool ok, const char * detail) {
    printf("%-34s %s   %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) g_fail++;
}

// ---------------------------------------------------------------------------------------------
// host-side PXQ4 core-tier packer (the layout under test, written independently of the kernel)
//   panel = 128 B fp16 anchors + kslabs * 1088 B slabs; slab = 64 B scale SoA + 64 x 16 B codes
// ---------------------------------------------------------------------------------------------
struct pxq4_host {
    int R, K, kslabs, panels;
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> code;     // [R][K] raw 4-bit codes, kept for the CPU reference
    std::vector<uint8_t> sub;      // [R][K/16] sub-scale nibbles
    std::vector<float>   anchor;   // [R]
};

static const float k_book[16]  = PXQ6_BOOK_INIT;
static const float k_sub16[16] = PXQ6_SUB16_INIT;

static pxq4_host make_pxq4(int R, int K, uint64_t seed) {
    pxq4_host w;
    w.R = R; w.K = K; w.kslabs = K/32; w.panels = R/64;
    const size_t pstride = 128 + (size_t)w.kslabs*1088;
    w.bytes.assign((size_t)w.panels*pstride, 0);
    w.code.assign((size_t)R*K, 0);
    w.sub.assign((size_t)R*(K/16), 0);
    w.anchor.assign(R, 0.f);
    std::mt19937_64 rng(seed);
    for (int r = 0; r < R; ++r) {
        // anchors are fp16 in the file: snap so the host reference and the kernel agree exactly
        const float a = 0.02f + 0.98f*((rng() % 10007) / 10007.0f);
        w.anchor[r] = __half2float(__float2half_rn(a));
        ((half *)(w.bytes.data() + (size_t)(r/64)*pstride))[r%64] = __float2half_rn(a);
    }
    for (int r = 0; r < R; ++r) {
        for (int kb = 0; kb < w.kslabs; ++kb) {
            uint8_t * slab = w.bytes.data() + (size_t)(r/64)*pstride + 128 + (size_t)kb*1088;
            const int rr = r % 64;
            const int s0 = (int)(rng() % 16), s1 = (int)(rng() % 16);
            slab[rr] = (uint8_t)(s0 | (s1 << 4));
            w.sub[(size_t)r*(K/16) + 2*kb + 0] = (uint8_t)s0;
            w.sub[(size_t)r*(K/16) + 2*kb + 1] = (uint8_t)s1;
            uint8_t * cr = slab + 64 + rr*16;
            for (int b = 0; b < 16; ++b) {
                const int c0 = (int)(rng() % 16), c1 = (int)(rng() % 16);
                cr[b] = (uint8_t)(c0 | (c1 << 4));
                w.code[(size_t)r*K + kb*32 + 2*b + 0] = (uint8_t)c0;
                w.code[(size_t)r*K + kb*32 + 2*b + 1] = (uint8_t)c1;
            }
        }
    }
    return w;
}

// the parity-locked dequant contract, on the host, with the SAME single rounding
static half host_w(const pxq4_host & w, int r, int k) {
    const float eff = w.anchor[r] * k_sub16[w.sub[(size_t)r*(w.K/16) + k/16]];
    return __float2half_rn(eff * k_book[w.code[(size_t)r*w.K + k]]);
}

// ---------------------------------------------------------------------------------------------
// the incumbent route, reproduced exactly as ggml_cuda_op_mul_mat_cublas does it:
//   ggml_get_to_fp16_cuda(PXQ4) == dequantize_row_pxq6_cuda  ->  cublasGemmEx(OP_T, OP_N, 32F)
// ---------------------------------------------------------------------------------------------
static void incumbent_route(cublasHandle_t h, const uint8_t * dW, half * dWf16, const half * dX,
                            float * dC, int R, int K, int ny, cudaStream_t stream) {
    dequantize_row_pxq6_cuda<half>(dW, dWf16, R, K, stream);
    const float alpha = 1.0f, beta = 0.0f;
    CB(cublasSetStream(h, stream));
    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, R, ny, K,
                    &alpha, dWf16, CUDA_R_16F, K,
                            dX,    CUDA_R_16F, K,
                    &beta,  dC,    CUDA_R_32F, R,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------------------------------------
// #1 decode byte-identity
// ---------------------------------------------------------------------------------------------
static void test1_byte_identity() {
    const int R = 256, K = 512;
    pxq4_host w = make_pxq4(R, K, 0xA1A1);
    uint8_t * dW; half * dRef; half * dV70;
    CK(cudaMalloc(&dW,  w.bytes.size()));
    CK(cudaMalloc(&dRef, (size_t)R*K*sizeof(half)));
    CK(cudaMalloc(&dV70, (size_t)R*K*sizeof(half)));
    CK(cudaMemcpy(dW, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice));
    dequantize_row_pxq6_cuda<half>(dW, dRef, R, K, 0);
    k_pxq4_v70_decode_dump<pxq6_pol_p6><<<R/32, 32>>>(dW, dV70, K, K/32);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<half> a((size_t)R*K), b((size_t)R*K);
    CK(cudaMemcpy(a.data(), dRef, a.size()*sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), dV70, b.size()*sizeof(half), cudaMemcpyDeviceToHost));
    const bool same = memcmp(a.data(), b.data(), a.size()*sizeof(half)) == 0;
    // also cross-check the host packer's own decode, so a shared kernel bug cannot hide
    size_t hostdiff = 0;
    for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) {
        const half hw = host_w(w, r, k);
        if (memcmp(&hw, &a[(size_t)r*K + k], sizeof(half)) != 0) hostdiff++;
    }
    char d[160];
    snprintf(d, sizeof(d), "%zu/%zu bytes differ vs k_pxq6_dequant_matrix; host-packer mismatches %zu",
             same ? (size_t)0 : (size_t)1, a.size()*sizeof(half), hostdiff);
    report("#1 decode byte-identity", same && hostdiff == 0, d);
    CK(cudaFree(dW)); CK(cudaFree(dRef)); CK(cudaFree(dV70));
}

// ---------------------------------------------------------------------------------------------
// #2 CPU double reference, ragged N
// ---------------------------------------------------------------------------------------------
static void test2_reference(cublasHandle_t h, int cfg) {
    const int Rs[] = {64, 192, 256};
    const int Ns[] = {17, 64, 129, 255, 256, 257, 1023};
    const int K = 256;
    double worst_v70 = 0.0, worst_inc = 0.0, worst_peak = 0.0;
    bool ok = true;
    char detail[256] = {0};
    for (int ri = 0; ri < 3; ++ri) {
        const int R = Rs[ri];
        if (R % pxa_pxq_v70_bm(cfg)) continue;
        pxq4_host w = make_pxq4(R, K, 0xB200u + ri);
        std::vector<half> hW((size_t)R*K);
        for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) hW[(size_t)r*K + k] = host_w(w, r, k);
        uint8_t * dW; CK(cudaMalloc(&dW, w.bytes.size()));
        CK(cudaMemcpy(dW, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice));
        half * dWf16; CK(cudaMalloc(&dWf16, (size_t)R*K*sizeof(half)));
        for (int ni = 0; ni < 7; ++ni) {
            const int ny = Ns[ni];
            std::vector<half>  hX((size_t)ny*K);
            std::mt19937_64 rng(0xC300u + ri*13 + ni);
            for (size_t i = 0; i < hX.size(); ++i)
                hX[i] = __float2half_rn(-1.0f + 2.0f*((rng() % 20011) / 20011.0f));
            half * dX; CK(cudaMalloc(&dX, hX.size()*sizeof(half)));
            CK(cudaMemcpy(dX, hX.data(), hX.size()*sizeof(half), cudaMemcpyHostToDevice));
            float * dC; CK(cudaMalloc(&dC, (size_t)ny*R*sizeof(float)));
            float * dCi; CK(cudaMalloc(&dCi, (size_t)ny*R*sizeof(float)));
            CK(cudaMemset(dC, 0, (size_t)ny*R*sizeof(float)));
            if (pxa_pxq4_gemm_v70_launch(dW, dX, dC, R, K, ny, cfg, 0) != 0) { ok = false; }
            CK(cudaGetLastError());
            incumbent_route(h, dW, dWf16, dX, dCi, R, K, ny, 0);
            CK(cudaDeviceSynchronize());
            std::vector<float> hC((size_t)ny*R), hCi((size_t)ny*R);
            CK(cudaMemcpy(hC.data(),  dC,  hC.size()*sizeof(float),  cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hCi.data(), dCi, hCi.size()*sizeof(float), cudaMemcpyDeviceToHost));
            double peak = 0.0;
            for (int t = 0; t < ny; ++t) for (int r = 0; r < R; ++r) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += (double)__half2float(hW[(size_t)r*K + k])
                                                 * (double)__half2float(hX[(size_t)t*K + k]);
                const double ev = fabs(acc - (double)hC [(size_t)t*R + r]);
                const double ei = fabs(acc - (double)hCi[(size_t)t*R + r]);
                if (fabs(acc) > peak)     peak     = fabs(acc);
                if (ev > worst_v70) worst_v70 = ev;
                if (ei > worst_inc) worst_inc = ei;
            }
            if (peak > worst_peak) worst_peak = peak;
            CK(cudaFree(dX)); CK(cudaFree(dC)); CK(cudaFree(dCi));
        }
        CK(cudaFree(dW)); CK(cudaFree(dWf16));
    }
    const double rel_v70 = worst_v70 / (worst_peak > 0 ? worst_peak : 1.0);
    // v70 must be within 5e-4 of peak, and no worse than 1.5x the incumbent's own error
    ok = ok && rel_v70 <= 5e-4 && worst_v70 <= 1.5*worst_inc + 1e-6;
    snprintf(detail, sizeof(detail),
             "max|v70-ref|=%.3e (rel-to-peak %.2e), incumbent %.3e, peak %.3e, ratio %.2f",
             worst_v70, rel_v70, worst_inc, worst_peak, worst_inc > 0 ? worst_v70/worst_inc : 0.0);
    report("#2 double reference, ragged N", ok, detail);
}

// ---------------------------------------------------------------------------------------------
// #3 bit-stability -- five runs, identical inputs, five buffers, all memcmp'd
// ---------------------------------------------------------------------------------------------
static void test3_bit_stability(int cfg) {
    const int K = 5120, R = 17408;
    const int Ms[] = {65, 2048};
    bool ok = true; int flips = 0;
    pxq4_host w = make_pxq4(R, K, 0xD400);
    uint8_t * dW; CK(cudaMalloc(&dW, w.bytes.size()));
    CK(cudaMemcpy(dW, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice));
    for (int mi = 0; mi < 2; ++mi) {
        const int ny = Ms[mi];
        std::vector<half> hX((size_t)ny*K);
        std::mt19937_64 rng(0xE500u + mi);
        for (size_t i = 0; i < hX.size(); ++i)
            hX[i] = __float2half_rn(-1.0f + 2.0f*((rng() % 20011) / 20011.0f));
        half * dX; CK(cudaMalloc(&dX, hX.size()*sizeof(half)));
        CK(cudaMemcpy(dX, hX.data(), hX.size()*sizeof(half), cudaMemcpyHostToDevice));
        float * dC[5];
        for (int i = 0; i < 5; ++i) {
            CK(cudaMalloc(&dC[i], (size_t)ny*R*sizeof(float)));
            CK(cudaMemset(dC[i], 0xCD, (size_t)ny*R*sizeof(float)));
            if (pxa_pxq4_gemm_v70_launch(dW, dX, dC[i], R, K, ny, cfg, 0) != 0) ok = false;
            CK(cudaGetLastError());
        }
        CK(cudaDeviceSynchronize());
        std::vector<float> a((size_t)ny*R), b((size_t)ny*R);
        CK(cudaMemcpy(a.data(), dC[0], a.size()*sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 1; i < 5; ++i) {
            CK(cudaMemcpy(b.data(), dC[i], b.size()*sizeof(float), cudaMemcpyDeviceToHost));
            if (memcmp(a.data(), b.data(), a.size()*sizeof(float)) != 0) { ok = false; flips++; }
        }
        for (int i = 0; i < 5; ++i) CK(cudaFree(dC[i]));
        CK(cudaFree(dX));
    }
    CK(cudaFree(dW));
    char d[160];
    snprintf(d, sizeof(d), "5 runs x M in {65,2048} on (%d,%d): %d differing pairs "
                           "(dense PXQ4, no recurrent state -- a flip here is ours)", K, R, flips);
    report("#3 bit-stability", ok, d);
}

// ---------------------------------------------------------------------------------------------
// #4 guard halo / ragged M -- guards fore and aft, NaN-filled interior
// ---------------------------------------------------------------------------------------------
static void test4_guard(int cfg) {
    const int K = 256, R = 256;
    const int Ms[] = {65, 127, 129};
    const int GUARD = 4096;
    bool ok = true; int guard_hits = 0, nan_hits = 0;
    pxq4_host w = make_pxq4(R, K, 0xF600);
    uint8_t * dW; CK(cudaMalloc(&dW, w.bytes.size()));
    CK(cudaMemcpy(dW, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice));
    for (int mi = 0; mi < 3; ++mi) {
        const int ny = Ms[mi];
        const size_t body = (size_t)ny*R;
        std::vector<half> hX((size_t)ny*K);
        std::mt19937_64 rng(0x1700u + mi);
        for (size_t i = 0; i < hX.size(); ++i)
            hX[i] = __float2half_rn(-1.0f + 2.0f*((rng() % 20011) / 20011.0f));
        half * dX; CK(cudaMalloc(&dX, hX.size()*sizeof(half)));
        CK(cudaMemcpy(dX, hX.data(), hX.size()*sizeof(half), cudaMemcpyHostToDevice));
        float * dBuf; CK(cudaMalloc(&dBuf, (body + 2*GUARD)*sizeof(float)));
        std::vector<float> host(body + 2*GUARD);
        for (int i = 0; i < GUARD; ++i) { host[i] = 1.2345678e30f; host[GUARD + body + i] = -9.8765432e30f; }
        for (size_t i = 0; i < body; ++i) host[GUARD + i] = nanf("");
        CK(cudaMemcpy(dBuf, host.data(), host.size()*sizeof(float), cudaMemcpyHostToDevice));
        if (pxa_pxq4_gemm_v70_launch(dW, dX, dBuf + GUARD, R, K, ny, cfg, 0) != 0) ok = false;
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host.data(), dBuf, host.size()*sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < GUARD; ++i) {
            if (host[i] != 1.2345678e30f) guard_hits++;
            if (host[GUARD + body + i] != -9.8765432e30f) guard_hits++;
        }
        for (size_t i = 0; i < body; ++i) if (isnan(host[GUARD + i])) nan_hits++;
        CK(cudaFree(dX)); CK(cudaFree(dBuf));
    }
    CK(cudaFree(dW));
    ok = ok && guard_hits == 0 && nan_hits == 0;
    char d[160];
    snprintf(d, sizeof(d), "M in {65,127,129}: %d guard floats clobbered, %d surviving NaNs",
             guard_hits, nan_hits);
    report("#4 guard halo / ragged M", ok, d);
}

// ---------------------------------------------------------------------------------------------
// #5 full-size parity vs the incumbent route (+ optional TFLOPS -- the route table)
// six PXQ4 shapes covering the dense attn / ffn mix of a Qwable-27B-class model
// ---------------------------------------------------------------------------------------------
struct shape { int K, R; const char * name; };
static const shape g_shapes[] = {
    { 5120,  5120, "attn qkv/out    " },
    { 5120, 17408, "ffn up          " },
    {17408,  5120, "ffn down        " },
    { 5120, 10240, "ffn gate        " },
    { 5120,  4096, "attn kv proj    " },
    { 4096,  5120, "attn q proj     " },
};

static void test5_parity(cublasHandle_t h, int cfg, bool do_time) {
    const int Ms[] = {32, 64, 512, 2048};
    double worst_rel = 0.0;
    bool ok = true;
    if (do_time) printf("\n  shape                 M      route(cuBLAS)   V70    ratio   TFLOPS route / V70\n");
    for (int si = 0; si < 6; ++si) {
        const int K = g_shapes[si].K, R = g_shapes[si].R;
        if (R % pxa_pxq_v70_bm(cfg)) continue;
        pxq4_host w = make_pxq4(R, K, 0x2800u + si);
        uint8_t * dW; CK(cudaMalloc(&dW, w.bytes.size()));
        CK(cudaMemcpy(dW, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice));
        half * dWf16; CK(cudaMalloc(&dWf16, (size_t)R*K*sizeof(half)));
        for (int mi = 0; mi < 4; ++mi) {
            const int ny = Ms[mi];
            std::vector<half> hX((size_t)ny*K);
            std::mt19937_64 rng(0x2900u + si*7 + mi);
            for (size_t i = 0; i < hX.size(); ++i)
                hX[i] = __float2half_rn(-1.0f + 2.0f*((rng() % 20011) / 20011.0f));
            half * dX; CK(cudaMalloc(&dX, hX.size()*sizeof(half)));
            CK(cudaMemcpy(dX, hX.data(), hX.size()*sizeof(half), cudaMemcpyHostToDevice));
            float * dC; CK(cudaMalloc(&dC,  (size_t)ny*R*sizeof(float)));
            float * dCi; CK(cudaMalloc(&dCi, (size_t)ny*R*sizeof(float)));
            if (pxa_pxq4_gemm_v70_launch(dW, dX, dC, R, K, ny, cfg, 0) != 0) ok = false;
            CK(cudaGetLastError());
            incumbent_route(h, dW, dWf16, dX, dCi, R, K, ny, 0);
            CK(cudaDeviceSynchronize());
            std::vector<float> a((size_t)ny*R), b((size_t)ny*R);
            CK(cudaMemcpy(a.data(), dC,  a.size()*sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), dCi, b.size()*sizeof(float), cudaMemcpyDeviceToHost));
            double peak = 0.0, maxd = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double d = fabs((double)a[i] - (double)b[i]);
                if (d > maxd) maxd = d;
                if (fabs((double)b[i]) > peak) peak = fabs((double)b[i]);
            }
            const double rel = peak > 0 ? maxd/peak : 0.0;
            if (rel > worst_rel) worst_rel = rel;
            if (do_time) {
                cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                const double flop = 2.0*(double)R*(double)K*(double)ny;
                float t_inc = 0, t_v70 = 0;
                for (int w2 = 0; w2 < 3; ++w2) incumbent_route(h, dW, dWf16, dX, dCi, R, K, ny, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for (int it = 0; it < 10; ++it) incumbent_route(h, dW, dWf16, dX, dCi, R, K, ny, 0);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&t_inc, e0, e1));
                for (int w2 = 0; w2 < 3; ++w2) pxa_pxq4_gemm_v70_launch(dW, dX, dC, R, K, ny, cfg, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for (int it = 0; it < 10; ++it) pxa_pxq4_gemm_v70_launch(dW, dX, dC, R, K, ny, cfg, 0);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&t_v70, e0, e1));
                printf("  %s %5d   %10.1f %10.1f   %5.2fx   (route %.3f ms / v70 %.3f ms)\n",
                       g_shapes[si].name, ny,
                       flop*10.0/(t_inc*1e9), flop*10.0/(t_v70*1e9), t_inc/t_v70,
                       t_inc/10.0, t_v70/10.0);
                CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            }
            CK(cudaFree(dX)); CK(cudaFree(dC)); CK(cudaFree(dCi));
        }
        CK(cudaFree(dW)); CK(cudaFree(dWf16));
    }
    ok = ok && worst_rel <= 2e-5;
    char d[160];
    snprintf(d, sizeof(d), "6 shapes x M in {32,64,512,2048}: max rel-to-peak %.2e vs the "
                           "dequant+cublasGemmEx route (K7 measured 1.3e-5)", worst_rel);
    report("#5 full-size parity", ok, d);
}

// ---------------------------------------------------------------------------------------------
// #0 tile ceiling -- V70 (decode included) vs cuBLAS handed pre-dequantized fp16 weights, so
// the dequant is free for cuBLAS and this is tile+decode against a pure tile. The codec cost is
// NOT removed from V70 (there is no fp16 A path into the register-direct kernel); the ablation
// that isolates it is the cuBLAS column's missing dequant, which the #5 route column carries.
// NOT run in a correctness pass (--ablate to run it), and it TIMES -- never in a correctness window.
// ---------------------------------------------------------------------------------------------
static void test0_ablation(cublasHandle_t h, int cfg) {
    printf("\n#0 TILE CEILING (V70 with decode vs cuBLAS on ALREADY-dequantized fp16 weights:\n"
           "   cuBLAS gets its dequant for free, so this is tile+decode vs pure tile; K7 format)\n");
    printf("  shape                 M      cuBLAS      V70(+decode)   delta\n");
    const int Ms[] = {512, 2048};
    for (int si = 0; si < 6; ++si) {
        const int K = g_shapes[si].K, R = g_shapes[si].R;
        if (R % pxa_pxq_v70_bm(cfg)) continue;
        pxq4_host w = make_pxq4(R, K, 0x3000u + si);
        uint8_t * dW; CK(cudaMalloc(&dW, w.bytes.size()));
        CK(cudaMemcpy(dW, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice));
        half * dWf16; CK(cudaMalloc(&dWf16, (size_t)R*K*sizeof(half)));
        dequantize_row_pxq6_cuda<half>(dW, dWf16, R, K, 0);
        for (int mi = 0; mi < 2; ++mi) {
            const int ny = Ms[mi];
            half * dX; CK(cudaMalloc(&dX, (size_t)ny*K*sizeof(half)));
            CK(cudaMemset(dX, 0x11, (size_t)ny*K*sizeof(half)));
            float * dC; CK(cudaMalloc(&dC, (size_t)ny*R*sizeof(float)));
            const double flop = 2.0*(double)R*(double)K*(double)ny;
            const float alpha = 1.0f, beta = 0.0f;
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CB(cublasSetStream(h, 0));
            for (int it = 0; it < 3; ++it)
                CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, R, ny, K, &alpha,
                                dWf16, CUDA_R_16F, K, dX, CUDA_R_16F, K, &beta,
                                dC, CUDA_R_32F, R, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int it = 0; it < 10; ++it)
                CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, R, ny, K, &alpha,
                                dWf16, CUDA_R_16F, K, dX, CUDA_R_16F, K, &beta,
                                dC, CUDA_R_32F, R, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float t_cb = 0; CK(cudaEventElapsedTime(&t_cb, e0, e1));
            for (int it = 0; it < 3; ++it) pxa_pxq4_gemm_v70_launch(dW, dX, dC, R, K, ny, cfg, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int it = 0; it < 10; ++it) pxa_pxq4_gemm_v70_launch(dW, dX, dC, R, K, ny, cfg, 0);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float t_v = 0; CK(cudaEventElapsedTime(&t_v, e0, e1));
            printf("  %s %5d   %9.1f   %9.1f   %+6.1f%%\n", g_shapes[si].name, ny,
                   flop*10.0/(t_cb*1e9), flop*10.0/(t_v*1e9), 100.0*(t_cb/t_v - 1.0));
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            CK(cudaFree(dX)); CK(cudaFree(dC));
        }
        CK(cudaFree(dW)); CK(cudaFree(dWf16));
    }
}

int main(int argc, char ** argv) {
    bool do_time = false, do_ablate = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--time"))   do_time   = true;
        if (!strcmp(argv[i], "--ablate")) do_ablate = true;
    }
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    const int cfg = pxa_pxq_v70_cfg();
    printf("pxq4-v70-test  device=%s  cc=%d.%d  SMs=%d  cfg=%d (BM=%d)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, cfg, pxa_pxq_v70_bm(cfg));
    if (prop.major != 7 || prop.minor != 0) {
        printf("NOTE: VOLTA_MMA_AVAILABLE is __CUDA_ARCH__ == 700 exactly; this kernel is a no-op "
               "on cc %d.%d and every phase below will fail by construction.\n", prop.major, prop.minor);
    }
    // L2 traffic sanity for the grouped rasterisation (the one-line trap, see PXQ4_V70_GN)
    {
        const int K = 5120, R = 17408, ny = 2048;
        const int BM = pxa_pxq_v70_bm(cfg);
        const double wbytes = (double)R*K*17.0/32.0;              // 1088 B per 64x32 slab
        const double abytes = (double)ny*K*2.0;
        const double npn = ceil(ny/128.0);
        printf("L2 rasterisation check (5120x17408, M=2048, BM=%d, GN=%d): weight stream re-read "
               "%.0fx -> %.0f MB, activation resident %.1f MB\n", BM, PXQ4_V70_GN,
               ceil(npn/PXQ4_V70_GN), wbytes*ceil(npn/PXQ4_V70_GN)/1e6,
               abytes*PXQ4_V70_GN/npn/1e6);
    }
    cublasHandle_t h; CB(cublasCreate(&h));
    printf("\n");
    test1_byte_identity();
    test2_reference(h, cfg);
    test3_bit_stability(cfg);
    test4_guard(cfg);
    test5_parity(h, cfg, do_time);
    if (do_ablate) test0_ablation(h, cfg);
    CB(cublasDestroy(h));
    printf("\n%s (%d failures)\n", g_fail ? "CORRECTNESS FAILED" : "ALL CORRECTNESS PHASES PASS", g_fail);
    return g_fail ? 1 : 0;
}
