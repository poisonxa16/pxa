// pxq4hq_kernel.cu -- host launchers, table upload and the self-test for the PXQ4HQ kernels.
//
// This TU owns the only copies of pxq4hq_book_g / pxq4hq_sub8_g (they are `static __device__`
// and therefore TU-local, exactly as in the engine), so the table upload/download helpers must
// live here.
//
// It also owns the SELF-TEST. The rule this package works to is that the kernels must be
// provable on synthetic data before a multi-GB model is loaded, and for a tier whose whole
// reason to exist is fidelity that matters more than usual: a PXQ4HQ tensor decoded with the
// PXQ4 stride, or against the PXQ4 sub table, loads, shards, passes every shape assertion and
// is quietly wrong. The oracle below is a HOST re-derivation from the prose format spec in
// pxq4hq_kernel_tables.h -- deliberately written from the prose, not by calling the device
// policy -- so agreement is evidence and not a tautology.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "pxq4hq_kernel.cuh"
#include "pxq4hq_kernel_launch.h"
#include "pxq4hq_mma.h"

#define PXQ4HQ_CUDA_CHECK(expr)                                                               \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "pxq4hq: %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                                \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

static void pxq4hq_bad_tier(int tier, const char * what) {
    fprintf(stderr, "pxq4hq: %s does not serve tier %d. This TU serves tier %d (pxq4hq) only; "
                    "pxq4 is served by the frozen pxq4_kernel.cu and pxq2/pxq3 by pxq23_kernel.cu. "
                    "Serving a tensor with the wrong slab stride or the wrong sub table "
                    "produces a well-formed, completely wrong result that nothing downstream "
                    "can detect, so this is fatal rather than a fallback.\n",
            what, tier, PXQ_TIER_PXQ4HQ);
    abort();
}

int pxq4hq_slab_bytes(int tier) { return pxq4hq_tier_slab_bytes(tier); }
int pxq4hq_book_n(int tier)     { return pxq4hq_tier_book_n(tier); }

int pxq4hq_mmv_smem_bytes(int kslabs) {
    return pxq4hq_canon_max_chunk(kslabs) * PXQ4HQ_QK * (int)sizeof(float);
}

bool pxq4hq_mmv_supported(int kslabs) {
    // 48 KiB is the per-block static+dynamic shared budget available WITHOUT the
    // cudaFuncSetAttribute opt-in on sm_60 and sm_70 alike. Staying under it keeps occupancy;
    // chunked staging means no real shape needs more (K = 17408 costs 4352 B).
    // Static: 2*16 table floats + the KSEG*BM reduce tile.
    const int stat_smem = (int)(2 * 16 * sizeof(float) + PXQ4_MMV_KSEG * PXQ4HQ_BM * sizeof(float));
    return pxq4hq_mmv_smem_bytes(kslabs) + stat_smem <= 48 * 1024;
}

// ---------------------------------------------------------------------------------------- launchers
void pxq4hq_launch_dequant_f16(int tier, const uint8_t * slabs, const void * anchor, void * out,
                               int panels, int kslabs, cudaStream_t stream) {
    if (tier != PXQ_TIER_PXQ4HQ) pxq4hq_bad_tier(tier, "pxq4hq_launch_dequant_f16");
    const int64_t nslabs = (int64_t)panels * kslabs;
    const int64_t K      = (int64_t)kslabs * PXQ4HQ_QK;
    k_pxq4hq_dequant_matrix<__half><<<(unsigned)nslabs, PXQ4HQ_BM, 0, stream>>>(
        slabs, (const __half *)anchor, (__half *)out, kslabs, K);
    PXQ4HQ_CUDA_CHECK(cudaGetLastError());
}

void pxq4hq_launch_mmv_f16(int tier, const uint8_t * slabs, const void * anchor, const void * x,
                           void * out, int M, int panels, int kslabs, bool vecx,
                           cudaStream_t stream) {
    if (tier != PXQ_TIER_PXQ4HQ) pxq4hq_bad_tier(tier, "pxq4hq_launch_mmv_f16");
    if (panels > 65535 || M > 65535) {
        fprintf(stderr, "pxq4hq: mmv grid limit exceeded: panels=%d M=%d\n", panels, M);
        abort();
    }
    const int    R    = panels * PXQ4HQ_BM;
    const int    K    = kslabs * PXQ4HQ_QK;
    const size_t smem = (size_t)pxq4hq_mmv_smem_bytes(kslabs);
    const dim3   grid((unsigned)panels, (unsigned)M, 1u);
    if (vecx) {
        k_pxq4hq_mmv<true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    } else {
        k_pxq4hq_mmv<false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    }
    PXQ4HQ_CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------- tables
void pxq4hq_upload_book(int tier, const float * book, int n) {
    if (tier != PXQ_TIER_PXQ4HQ) pxq4hq_bad_tier(tier, "pxq4hq_upload_book");
    if (n != PXQ4HQ_BOOK_N) {
        fprintf(stderr, "pxq4hq: the book is %d entries, got %d\n", PXQ4HQ_BOOK_N, n);
        abort();
    }
    PXQ4HQ_CUDA_CHECK(cudaMemcpyToSymbol(pxq4hq_book_g, book, n * sizeof(float)));
    // The tensor-core TU keeps its own copies (static __device__ is TU-local), so every upload
    // must fan out to it or the two paths decode the same bytes differently.
    float sub[16];
    PXQ4HQ_CUDA_CHECK(cudaMemcpyFromSymbol(sub, pxq4hq_sub8_g, 16 * sizeof(float)));
    pxq4hq_mma_upload_tables(book, sub);
}

void pxq4hq_upload_sub(const float * sub8) {
    PXQ4HQ_CUDA_CHECK(cudaMemcpyToSymbol(pxq4hq_sub8_g, sub8, 16 * sizeof(float)));
    float book[PXQ4HQ_BOOK_N];
    PXQ4HQ_CUDA_CHECK(cudaMemcpyFromSymbol(book, pxq4hq_book_g, PXQ4HQ_BOOK_N * sizeof(float)));
    pxq4hq_mma_upload_tables(book, sub8);
}

void pxq4hq_download_book(int tier, float * book, int n) {
    if (tier != PXQ_TIER_PXQ4HQ || n != PXQ4HQ_BOOK_N) {
        pxq4hq_bad_tier(tier, "pxq4hq_download_book");
    }
    PXQ4HQ_CUDA_CHECK(cudaMemcpyFromSymbol(book, pxq4hq_book_g, n * sizeof(float)));
}

void pxq4hq_download_sub(float * sub8) {
    PXQ4HQ_CUDA_CHECK(cudaMemcpyFromSymbol(sub8, pxq4hq_sub8_g, 16 * sizeof(float)));
}

// ===================================================================================== SELF-TEST
//
// HOST ORACLE. Written from the prose spec in pxq4hq_kernel_tables.h, not from the device
// policy, so agreement between the two is evidence about the FORMAT rather than a restatement
// of one implementation. It reproduces the parity-locked contract exactly:
//     eff = fp32(anchor_fp16) * SUB8[nibble] ;  w = eff * fp32(book[code])
// and the fp16 store the dequant kernel performs, so the dequant comparison is BIT-EXACT, not
// a tolerance.
namespace {

const float host_book_hq[PXQ4HQ_BOOK_N] = PXQ4HQ_BOOK_INIT;
const float host_sub8[16]               = PXQ4HQ_SUB8_INIT;

// The effective scale index for element j of a 32-element block: bs8, so j >> 3.
inline int host_eff_idx(int j) { return j >> 3; }

// The four scale nibbles of one (row, 32-column block), straight from the packing spec:
// slab[2r] low -> 0..7, slab[2r] high -> 8..15, slab[2r+1] low -> 16..23, high -> 24..31.
inline int host_sub_nibble(const uint8_t * slab, int row, int blk) {
    const uint8_t byte = slab[2 * row + (blk >> 1)];
    return (blk & 1) ? (byte >> 4) : (byte & 0xf);
}

// code(j) for element j (0..31) of one 16-byte code row: byte b = j/2 holds code(2b) in the
// low nibble and code(2b+1) in the high nibble.
inline int host_code(const uint8_t * row, int j) {
    const uint8_t byte = row[j >> 1];
    return (j & 1) ? (byte >> 4) : (byte & 0xf);
}

// deterministic LCG so a failure is reproducible from nothing but this file
uint32_t lcg(uint32_t & s) { s = s * 1664525u + 1013904223u; return s; }

}  // namespace

int pxq4hq_selftest(int tier) {
    if (tier != PXQ_TIER_PXQ4HQ) {
        fprintf(stderr, "pxq4hq_selftest: unknown tier %d (this TU serves %d only)\n",
                tier, PXQ_TIER_PXQ4HQ);
        return 10;
    }
    const int panels = 3, kslabs = 5;                         // 192 rows x 160 columns: not a
    const int N = panels * PXQ4HQ_BM, K = kslabs * PXQ4HQ_QK; // power of two anywhere, nfix > 1
    const size_t nslab_bytes = (size_t)panels * kslabs * PXQ4HQ_SLAB_BYTES;

    std::vector<uint8_t>  h_slabs(nslab_bytes);
    std::vector<uint16_t> h_anchor((size_t)panels * PXQ4HQ_BM);
    uint32_t s = 0xC0FFEEu + (uint32_t)tier;
    for (size_t i = 0; i < nslab_bytes; ++i) h_slabs[i] = (uint8_t)(lcg(s) >> 13);
    // anchors: small positive-and-negative fp16 values, never inf/nan
    for (size_t i = 0; i < h_anchor.size(); ++i) {
        const float v = ((float)(int)(lcg(s) % 2001) - 1000.f) / 1024.f;
        h_anchor[i] = __half_as_ushort(__float2half_rn(v));
    }

    // ---- host oracle -> expected fp16 bit patterns, [N, K] row-major
    std::vector<uint16_t> want((size_t)N * K);
    for (int p = 0; p < panels; ++p) {
        for (int kb = 0; kb < kslabs; ++kb) {
            const uint8_t * slab = h_slabs.data()
                                 + ((size_t)p * kslabs + kb) * PXQ4HQ_SLAB_BYTES;
            for (int r = 0; r < PXQ4HQ_BM; ++r) {
                const float anch = __half2float(__ushort_as_half(h_anchor[(size_t)p * PXQ4HQ_BM + r]));
                float eff[PXQ4HQ_NEFF];
                for (int blk = 0; blk < PXQ4HQ_NEFF; ++blk) {
                    eff[blk] = anch * host_sub8[host_sub_nibble(slab, r, blk)];
                }
                const uint8_t * row = slab + PXQ4HQ_CODE_OFF + r * PXQ4HQ_CODE_BYTES;
                for (int j = 0; j < PXQ4HQ_QK; ++j) {
                    const float w = eff[host_eff_idx(j)] * host_book_hq[host_code(row, j)];
                    want[(size_t)(p * PXQ4HQ_BM + r) * K + kb * PXQ4HQ_QK + j] =
                        __half_as_ushort(__float2half_rn(w));
                }
            }
        }
    }

    // ---- device run
    uint8_t * d_slabs = nullptr; __half * d_anchor = nullptr; __half * d_out = nullptr;
    PXQ4HQ_CUDA_CHECK(cudaMalloc(&d_slabs, nslab_bytes));
    PXQ4HQ_CUDA_CHECK(cudaMalloc(&d_anchor, h_anchor.size() * 2));
    PXQ4HQ_CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * K * 2));
    PXQ4HQ_CUDA_CHECK(cudaMemcpy(d_slabs, h_slabs.data(), nslab_bytes, cudaMemcpyHostToDevice));
    PXQ4HQ_CUDA_CHECK(cudaMemcpy(d_anchor, h_anchor.data(), h_anchor.size() * 2, cudaMemcpyHostToDevice));

    int rc = 0;
    pxq4hq_launch_dequant_f16(tier, d_slabs, d_anchor, d_out, panels, kslabs, 0);
    PXQ4HQ_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got((size_t)N * K);
    PXQ4HQ_CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            fprintf(stderr, "pxq4hq_selftest: DEQUANT MISMATCH at [%zu, %zu]: device 0x%04x "
                            "host 0x%04x\n", i / K, i % K, got[i], want[i]);
            rc = 1; break;
        }
    }

    // ---- mmv arm: out = x @ W^T for M = 2, checked against a HOST REPLAY of the exact fold
    // (canonical chunks -> per-kseg partial -> kseg-ordered fold -> ONE rounding).
    //
    // WHY THIS ARM IS 1-ULP AND NOT BIT-EXACT, while the dequant arm above IS bit-exact. The
    // dequant arm performs no accumulation: every output is one multiply chain with a single
    // rounding, so host and device must agree to the bit and any disagreement is a decode bug.
    // This arm sums 160 products. Contracting `a*b + c` into an FMA is an implementation
    // freedom BOTH compilers take by default -- nvcc via --fmad=true for the device fold, the
    // host compiler via -ffp-contract -- and they are not obliged to make the same choices at
    // the same points; demanding bit-equality here produced a false failure on a sibling tier
    // (one row in 192, off by one ULP) while every device-vs-device arm was clean.
    //
    // The oracle is still pinned as hard as it can be: this TU's host code is compiled with
    // -ffp-contract=off, so the replay below is a fixed, reproducible sequence of separately
    // rounded operations. What is relaxed is only the comparison, and only by ONE unit in the
    // last place of fp16 -- about 0.05% -- with the count of differing rows reported. A real
    // decode bug does not produce one row off by one ULP; it produces many rows off by a lot,
    // and that still fails here.
    if (rc == 0 && pxq4hq_mmv_supported(kslabs)) {
        const int M = 2;
        std::vector<uint16_t> h_x((size_t)M * K);
        for (size_t i = 0; i < h_x.size(); ++i) {
            const float v = ((float)(int)(lcg(s) % 401) - 200.f) / 256.f;
            h_x[i] = __half_as_ushort(__float2half_rn(v));
        }
        __half * d_x = nullptr; __half * d_o = nullptr;
        PXQ4HQ_CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
        PXQ4HQ_CUDA_CHECK(cudaMalloc(&d_o, (size_t)M * N * 2));
        PXQ4HQ_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));
        pxq4hq_launch_mmv_f16(tier, d_slabs, d_anchor, d_x, d_o, M, panels, kslabs, true, 0);
        PXQ4HQ_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint16_t> h_o((size_t)M * N);
        PXQ4HQ_CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, h_o.size() * 2, cudaMemcpyDeviceToHost));

        const int nfix = pxq4hq_canon_nfix(kslabs, PXQ4_CANON_CMAX);
        int ulp_off = 0, worst_ulp = 0;
        for (int m = 0; m < M && rc == 0; ++m) {
            for (int p = 0; p < panels && rc == 0; ++p) {
                for (int r = 0; r < PXQ4HQ_BM; ++r) {
                    const float anch = __half2float(__ushort_as_half(h_anchor[(size_t)p * PXQ4HQ_BM + r]));
                    float su[PXQ4_MMV_KSEG] = {0.f, 0.f, 0.f, 0.f};
                    for (int c = 0; c < nfix; ++c) {
                        const int b0 = (kslabs * c) / nfix, b1 = (kslabs * (c + 1)) / nfix;
                        for (int kseg = 0; kseg < PXQ4_MMV_KSEG; ++kseg) {
                            float t = 0.f;
                            for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
                                const uint8_t * slab = h_slabs.data()
                                    + ((size_t)p * kslabs + kb) * PXQ4HQ_SLAB_BYTES;
                                float eff[PXQ4HQ_NEFF];
                                for (int blk = 0; blk < PXQ4HQ_NEFF; ++blk) {
                                    eff[blk] = anch * host_sub8[host_sub_nibble(slab, r, blk)];
                                }
                                const uint8_t * row = slab + PXQ4HQ_CODE_OFF + r * PXQ4HQ_CODE_BYTES;
                                // one partial per bs8 block, then eff-weighted -- the kernel's
                                // order, not a plain running dot product
                                float tb[PXQ4HQ_NEFF] = {0.f, 0.f, 0.f, 0.f};
                                for (int b = 0; b < 16; ++b) {
                                    const float a0 = host_book_hq[host_code(row, 2 * b)];
                                    const float a1 = host_book_hq[host_code(row, 2 * b + 1)];
                                    const float x0 = __half2float(__ushort_as_half(h_x[(size_t)m * K + kb * PXQ4HQ_QK + 2 * b]));
                                    const float x1 = __half2float(__ushort_as_half(h_x[(size_t)m * K + kb * PXQ4HQ_QK + 2 * b + 1]));
                                    float & acc = tb[(b * PXQ4HQ_NEFF) >> 4];
                                    acc = acc + (a0 * x0 + a1 * x1);
                                }
                                float u = eff[0] * tb[0];
                                for (int i = 1; i < PXQ4HQ_NEFF; ++i) u += eff[i] * tb[i];
                                t += u;
                            }
                            su[kseg] += t;
                        }
                    }
                    float u = 0.f;
                    for (int k = 0; k < PXQ4_MMV_KSEG; ++k) u += su[k];
                    const uint16_t w = __half_as_ushort(__float2half_rn(u));
                    const uint16_t g = h_o[(size_t)m * N + p * PXQ4HQ_BM + r];
                    if (w != g) {
                        // ULP distance in the fp16 encoding. Both values come from the same dot
                        // product, so they share a sign and an exponent neighbourhood; comparing
                        // the raw encodings as integers is exact for same-sign finite values.
                        const int d  = (int)g - (int)w;
                        const int ad = d < 0 ? -d : d;
                        if (ad > worst_ulp) worst_ulp = ad;
                        if (ad > 1) {
                            fprintf(stderr, "pxq4hq_selftest: MMV MISMATCH at [%d, %d]: device "
                                            "0x%04x host 0x%04x (%d ULP)\n",
                                    m, p * PXQ4HQ_BM + r, g, w, ad);
                            rc = 3; break;
                        }
                        ++ulp_off;
                    }
                }
            }
        }
        if (rc == 0 && ulp_off) {
            fprintf(stderr, "pxq4hq_selftest: mmv arm OK with %d/%d rows 1 ULP from the host "
                            "replay (worst %d ULP) -- FMA contraction, see the note in "
                            "pxq4hq_kernel.cu\n", ulp_off, M * N, worst_ulp);
        }
        cudaFree(d_x); cudaFree(d_o);
    }

    cudaFree(d_slabs); cudaFree(d_anchor); cudaFree(d_out);
    return rc;
}
