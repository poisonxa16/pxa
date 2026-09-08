// pxq_q8.cu -- launchers and self-test for the int8 per-row-scale head kernels.
//
// Additive TU, following this tree's integration rule: it adds no
// symbol to and edits no line of pxq4_kernel.cu, pxq23_kernel.cu or their headers.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pxq_q8.cuh"
#include "pxq_q8_launch.h"

#define PXQ_Q8_CHECK(expr)                                                                    \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "pxq_q8: %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                                \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

static inline int pxq_q8_row_blocks(int N) {
    return (N + PXQ_Q8_WARPS_PER_BLOCK - 1) / PXQ_Q8_WARPS_PER_BLOCK;
}

void pxq_q8_launch_mmv_f16(const int8_t * W, const void * scale, const void * x, void * out,
                           int M, int N, int K, cudaStream_t stream) {
    const int blocks = pxq_q8_row_blocks(N);
    if (blocks > 2147483647 || M > 65535) {
        fprintf(stderr, "pxq_q8: mmv grid limit exceeded: N=%d M=%d\n", N, M);
        abort();
    }
    const dim3 grid((unsigned)blocks, (unsigned)M, 1u);
    k_pxq_q8_mmv<<<grid, PXQ_Q8_WARPS_PER_BLOCK * PXQ_Q8_LANES, 0, stream>>>(
        W, (const __half *)scale, (const __half *)x, (__half *)out, N, K);
    PXQ_Q8_CHECK(cudaGetLastError());
}

void pxq_q8_launch_dequant_f16(const int8_t * W, const void * scale, void * y,
                               int N, int K, cudaStream_t stream) {
    const dim3 grid((unsigned)pxq_q8_row_blocks(N), 1u, 1u);
    k_pxq_q8_dequant<<<grid, PXQ_Q8_WARPS_PER_BLOCK * PXQ_Q8_LANES, 0, stream>>>(
        W, (const __half *)scale, (__half *)y, N, K);
    PXQ_Q8_CHECK(cudaGetLastError());
}

// ===================================================================================== SELF-TEST
namespace {
uint32_t q8_lcg(uint32_t & s) { s = s * 1664525u + 1013904223u; return s; }
}  // namespace

int pxq_q8_selftest() {
    // A vocabulary-shaped but small case: N not a multiple of the 8-row block, K not a
    // multiple of 128, so both tails are exercised. Neither is a hypothetical -- a padded
    // vocab is normal and a non-2048 hidden size is one model away.
    const int N = 200, K = 300, M = 3;
    std::vector<int8_t>   h_w((size_t)N * K);
    std::vector<uint16_t> h_s(N), h_x((size_t)M * K);
    uint32_t s = 0x51EEDu;
    for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = (int8_t)((q8_lcg(s) >> 11) & 0xff);
    for (int n = 0; n < N; ++n) {
        const float v = 0.001f + (float)(q8_lcg(s) % 1000) / 100000.f;   // small, positive
        h_s[n] = __half_as_ushort(__float2half_rn(v));
    }
    for (size_t i = 0; i < h_x.size(); ++i) {
        const float v = ((float)(int)(q8_lcg(s) % 401) - 200.f) / 256.f;
        h_x[i] = __half_as_ushort(__float2half_rn(v));
    }

    int8_t * d_w = nullptr; __half * d_s = nullptr; __half * d_x = nullptr;
    __half * d_o = nullptr; __half * d_y = nullptr;
    PXQ_Q8_CHECK(cudaMalloc(&d_w, h_w.size()));
    PXQ_Q8_CHECK(cudaMalloc(&d_s, (size_t)N * 2));
    PXQ_Q8_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
    PXQ_Q8_CHECK(cudaMalloc(&d_o, (size_t)M * N * 2));
    PXQ_Q8_CHECK(cudaMalloc(&d_y, (size_t)N * K * 2));
    PXQ_Q8_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice));
    PXQ_Q8_CHECK(cudaMemcpy(d_s, h_s.data(), (size_t)N * 2, cudaMemcpyHostToDevice));
    PXQ_Q8_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));

    int rc = 0;

    // ---- dequant: no accumulation, so this one IS bit-exact and a mismatch is a real bug.
    pxq_q8_launch_dequant_f16(d_w, d_s, d_y, N, K, 0);
    PXQ_Q8_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got((size_t)N * K);
    PXQ_Q8_CHECK(cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N && rc == 0; ++n) {
        const float sc = __half2float(__ushort_as_half(h_s[n]));
        for (int k = 0; k < K; ++k) {
            const uint16_t want = __half_as_ushort(__float2half_rn((float)h_w[(size_t)n*K+k] * sc));
            const uint16_t g = got[(size_t)n * K + k];
            if (g != want) {
                fprintf(stderr, "pxq_q8_selftest: DEQUANT MISMATCH at [%d, %d]: device 0x%04x "
                                "host 0x%04x\n", n, k, g, want);
                rc = 1; break;
            }
        }
    }

    // ---- GEMV against a host replay of the kernel's OWN order: lane-strided partials in the
    // same 4-wide grouping, folded by the same halving tree, scaled once, rounded once. The
    // allowance is the same 1 ULP the PXQ self-test carries and for the same reason -- fp32
    // contraction is a freedom both compilers take and neither is obliged to take identically.
    if (rc == 0) {
        pxq_q8_launch_mmv_f16(d_w, d_s, d_x, d_o, M, N, K, 0);
        PXQ_Q8_CHECK(cudaDeviceSynchronize());
        std::vector<uint16_t> h_o((size_t)M * N);
        PXQ_Q8_CHECK(cudaMemcpy(h_o.data(), d_o, h_o.size() * 2, cudaMemcpyDeviceToHost));
        int ulp_off = 0, worst = 0;
        const int kmain = (K / 128) * 128;
        for (int m = 0; m < M && rc == 0; ++m) {
            for (int n = 0; n < N; ++n) {
                float part[32];
                for (int l = 0; l < 32; ++l) {
                    float a = 0.f;
                    for (int k = l * 4; k < kmain; k += 32 * 4) {
                        for (int j = 0; j < 4; ++j) {
                            a += (float)h_w[(size_t)n * K + k + j]
                               * __half2float(__ushort_as_half(h_x[(size_t)m * K + k + j]));
                        }
                    }
                    for (int k = kmain + l; k < K; k += 32) {
                        a += (float)h_w[(size_t)n * K + k]
                           * __half2float(__ushort_as_half(h_x[(size_t)m * K + k]));
                    }
                    part[l] = a;
                }
                for (int off = 16; off > 0; off >>= 1) {
                    for (int l = 0; l < off; ++l) part[l] += part[l + off];
                }
                const float sc = __half2float(__ushort_as_half(h_s[n]));
                const uint16_t want = __half_as_ushort(__float2half_rn(part[0] * sc));
                const uint16_t g = h_o[(size_t)m * N + n];
                if (g != want) {
                    const int d = (int)g - (int)want, ad = d < 0 ? -d : d;
                    if (ad > worst) worst = ad;
                    ++ulp_off;
                    if (ad > 1) {
                        fprintf(stderr, "pxq_q8_selftest: MMV MISMATCH at m=%d n=%d: device "
                                        "0x%04x host 0x%04x (%d ULP)\n", m, n, g, want, ad);
                        rc = 2; break;
                    }
                }
            }
        }
        if (rc == 0 && ulp_off * 20 > M * N) {
            fprintf(stderr, "pxq_q8_selftest: MMV: %d of %d rows differ by 1 ULP (>5%%) -- too "
                            "many for contraction alone\n", ulp_off, M * N);
            rc = 2;
        } else if (rc == 0 && ulp_off) {
            fprintf(stderr, "pxq_q8_selftest: MMV: %d of %d off by 1 ULP (contraction "
                            "allowance), max %d\n", ulp_off, M * N, worst);
        }
    }

    cudaFree(d_w); cudaFree(d_s); cudaFree(d_x); cudaFree(d_o); cudaFree(d_y);
    if (rc == 0) fprintf(stderr, "pxq_q8_selftest: PASS (dequant bit-exact, mmv within the "
                                 "1 ULP contraction allowance; N=%d K=%d exercises both tails)\n",
                         N, K);
    return rc;
}
