// pxq23_kernel.cu -- host launchers, table upload and the self-test for the PXQ2/PXQ3 kernels.
//
// This TU owns the only copies of pxq2_book_g / pxq3_book_g / pxq23_sub16_g (they are
// `static __device__` and therefore TU-local, exactly as in the engine), so the table
// upload/download helpers must live here.
//
// It also owns the SELF-TEST, which main required before any window: the window must be able
// to prove the kernels on synthetic data before a 10 GB model is loaded. The oracle is a HOST
// re-derivation from the format spec in pxq23_kernel_tables.h -- deliberately written from the
// prose, not by calling the device policies -- so agreement is evidence and not a tautology.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "pxq23_kernel.cuh"
#include "pxq_moe_fused.cuh"
#include "pxq23_kernel_launch.h"
#include "pxq_moe_fused_launch.h"
#include "pxq4_kernel_launch.h"     // pxq4_launch_dequant_f16, for the tier-252 gate

#define PXQ23_CUDA_CHECK(expr)                                                                \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "pxq23: %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                                                \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

static void pxq23_bad_tier(int tier, const char * what) {
    fprintf(stderr, "pxq23: %s does not serve tier %d (%s). Serving a tensor with the wrong "
                    "slab stride produces a well-formed, completely wrong result that nothing "
                    "downstream can detect, so this is fatal rather than a fallback.\n",
            what, tier, pxq_tier_name(tier));
    abort();
}

int pxq23_slab_bytes(int tier) { return pxq_tier_slab_bytes(tier); }
int pxq23_book_n(int tier)     { return pxq_tier_book_n(tier); }

int pxq23_mmv_smem_bytes(int kslabs) {
    return pxq23_canon_max_chunk(kslabs) * PXQ4_QK * (int)sizeof(float);
}

bool pxq23_mmv_supported(int kslabs) {
    // 48 KiB is the per-block static+dynamic shared budget available WITHOUT the
    // cudaFuncSetAttribute opt-in on sm_60 and sm_70 alike. Staying under it keeps occupancy;
    // chunked staging means no real shape needs more (K = 17408 costs 4352 B).
    // Static: 2*16 table floats + the KSEG*BM reduce tile.
    const int stat_smem = (int)(2 * 16 * sizeof(float) + PXQ4_MMV_KSEG * PXQ4_BM * sizeof(float));
    return pxq23_mmv_smem_bytes(kslabs) + stat_smem <= 48 * 1024;
}

// ---------------------------------------------------------------------------------------- launchers
template <class POL>
static void pxq23_dequant_tpl(const uint8_t * slabs, const void * anchor, void * out,
                              int panels, int kslabs, cudaStream_t stream) {
    const int64_t nslabs = (int64_t)panels * kslabs;
    const int64_t K      = (int64_t)kslabs * PXQ4_QK;
    k_pxq23_dequant_matrix<POL, __half><<<(unsigned)nslabs, PXQ4_BM, 0, stream>>>(
        slabs, (const __half *)anchor, (__half *)out, kslabs, K);
    PXQ23_CUDA_CHECK(cudaGetLastError());
}

void pxq23_launch_dequant_f16(int tier, const uint8_t * slabs, const void * anchor, void * out,
                              int panels, int kslabs, cudaStream_t stream) {
    switch (tier) {
        case PXQ_TIER_PXQ2: pxq23_dequant_tpl<pxq2_pol>(slabs, anchor, out, panels, kslabs, stream); return;
        case PXQ_TIER_PXQ3: pxq23_dequant_tpl<pxq3_pol>(slabs, anchor, out, panels, kslabs, stream); return;
        // tier 252 is served by the FROZEN pxq4_kernel.cu launcher, never by the templated
        // copy -- the templated PXQ4 instantiation exists only inside pxq23_selftest().
        case PXQ_TIER_PXQ4: pxq4_launch_dequant_f16(slabs, anchor, out, panels, kslabs, stream); return;
        default: pxq23_bad_tier(tier, "pxq23_launch_dequant_f16");
    }
}

template <class POL>
static void pxq23_mmv_tpl(const uint8_t * slabs, const void * anchor, const void * x, void * out,
                          int M, int panels, int kslabs, bool vecx, cudaStream_t stream) {
    const int    R    = panels * PXQ4_BM;
    const int    K    = kslabs * PXQ4_QK;
    const size_t smem = (size_t)pxq23_mmv_smem_bytes(kslabs);
    const dim3   grid((unsigned)panels, (unsigned)M, 1u);
    if (vecx) {
        k_pxq23_mmv<POL, true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    } else {
        k_pxq23_mmv<POL, false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    }
    PXQ23_CUDA_CHECK(cudaGetLastError());
}

void pxq23_launch_mmv_f16(int tier, const uint8_t * slabs, const void * anchor, const void * x,
                          void * out, int M, int panels, int kslabs, bool vecx,
                          cudaStream_t stream) {
    if (panels > 65535 || M > 65535) {
        fprintf(stderr, "pxq23: mmv grid limit exceeded: panels=%d M=%d\n", panels, M);
        abort();
    }
    switch (tier) {
        case PXQ_TIER_PXQ2: pxq23_mmv_tpl<pxq2_pol>(slabs, anchor, x, out, M, panels, kslabs, vecx, stream); return;
        case PXQ_TIER_PXQ3: pxq23_mmv_tpl<pxq3_pol>(slabs, anchor, x, out, M, panels, kslabs, vecx, stream); return;
        default: pxq23_bad_tier(tier, "pxq23_launch_mmv_f16");
    }
}

template <class POL>
static void pxq23_moe_mmv_tpl(const uint8_t * slabs, const void * anchor, const void * x,
                              const int32_t * ids, void * out, int S, int E, int panels,
                              int kslabs, bool vecx, cudaStream_t stream) {
    const int    R    = panels * PXQ4_BM;
    const int    K    = kslabs * PXQ4_QK;
    const size_t smem = (size_t)pxq23_mmv_smem_bytes(kslabs);
    const dim3   grid((unsigned)panels, (unsigned)S, 1u);
    if (vecx) {
        k_pxq23_moe_mmv<POL, true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, ids, (__half *)out, R, K, E, panels);
    } else {
        k_pxq23_moe_mmv<POL, false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, ids, (__half *)out, R, K, E, panels);
    }
    PXQ23_CUDA_CHECK(cudaGetLastError());
}

void pxq23_launch_moe_mmv_f16(int tier, const uint8_t * slabs, const void * anchor,
                              const void * x, const int32_t * ids, void * out, int S, int E,
                              int panels, int kslabs, bool vecx, cudaStream_t stream) {
    if (panels > 65535 || S > 65535) {
        fprintf(stderr, "pxq23: moe mmv grid limit exceeded: panels=%d S=%d\n", panels, S);
        abort();
    }
    switch (tier) {
        case PXQ_TIER_PXQ2: pxq23_moe_mmv_tpl<pxq2_pol>(slabs, anchor, x, ids, out, S, E, panels, kslabs, vecx, stream); return;
        case PXQ_TIER_PXQ3: pxq23_moe_mmv_tpl<pxq3_pol>(slabs, anchor, x, ids, out, S, E, panels, kslabs, vecx, stream); return;
        default: pxq23_bad_tier(tier, "pxq23_launch_moe_mmv_f16");
    }
}

// ------------------------------------------------------------------------------------- tables
void pxq23_upload_book(int tier, const float * book, int n) {
    if (n != pxq_tier_book_n(tier)) {
        fprintf(stderr, "pxq23: tier %s wants a %d-entry book, got %d\n",
                pxq_tier_name(tier), pxq_tier_book_n(tier), n);
        abort();
    }
    switch (tier) {
        case PXQ_TIER_PXQ2: PXQ23_CUDA_CHECK(cudaMemcpyToSymbol(pxq2_book_g, book, n * sizeof(float))); return;
        case PXQ_TIER_PXQ3: PXQ23_CUDA_CHECK(cudaMemcpyToSymbol(pxq3_book_g, book, n * sizeof(float))); return;
        default: pxq23_bad_tier(tier, "pxq23_upload_book");
    }
}

void pxq23_upload_sub(const float * sub16) {
    PXQ23_CUDA_CHECK(cudaMemcpyToSymbol(pxq23_sub16_g, sub16, 16 * sizeof(float)));
}

void pxq23_download_book(int tier, float * book, int n) {
    if (n != pxq_tier_book_n(tier)) { pxq23_bad_tier(tier, "pxq23_download_book"); }
    switch (tier) {
        case PXQ_TIER_PXQ2: PXQ23_CUDA_CHECK(cudaMemcpyFromSymbol(book, pxq2_book_g, n * sizeof(float))); return;
        case PXQ_TIER_PXQ3: PXQ23_CUDA_CHECK(cudaMemcpyFromSymbol(book, pxq3_book_g, n * sizeof(float))); return;
        default: pxq23_bad_tier(tier, "pxq23_download_book");
    }
}

void pxq23_download_sub(float * sub16) {
    PXQ23_CUDA_CHECK(cudaMemcpyFromSymbol(sub16, pxq23_sub16_g, 16 * sizeof(float)));
}

// ===================================================================================== SELF-TEST
//
// HOST ORACLE. Written from the prose spec in pxq23_kernel_tables.h, not from the device
// policies, so agreement between the two is evidence about the format rather than a restatement
// of one implementation. It reproduces the parity-locked contract exactly:
//     eff = fp32(anchor_fp16) * SUB16[nibble] ;  w = eff * fp32(book[code])
// and the fp16 store the dequant kernel performs, so the comparison is BIT-EXACT, not a
// tolerance.
namespace {

struct HostTier {
    int slab, code_off, code_bytes, book_n;
    const float * book;
};

float host_sub16[16] = PXQ4_SUB16_INIT;
const float host_book2[PXQ2_BOOK_N] = PXQ2_BOOK_INIT;
const float host_book3[PXQ3_BOOK_N] = PXQ3_BOOK_INIT;
const float host_book4[16]          = PXQ4_BOOK_INIT;

bool host_tier(int tier, HostTier & t) {
    switch (tier) {
        case PXQ_TIER_PXQ2: t = {PXQ2_SLAB_BYTES, 64, 8,  PXQ2_BOOK_N, host_book2}; return true;
        case PXQ_TIER_PXQ3: t = {PXQ3_SLAB_BYTES, 64, 12, PXQ3_BOOK_N, host_book3}; return true;
        case PXQ_TIER_PXQ4: t = {1088,            64, 16, 16,          host_book4}; return true;
        default: return false;
    }
}

// code(j) for element j (0..31) of one row of one slab, straight from the packing spec.
int host_code(int tier, const uint8_t * row, int j) {
    if (tier == PXQ_TIER_PXQ2) {
        uint32_t w;  memcpy(&w, row + 4 * (j >> 4), 4);          // LE u32, half j>>4
        return (int)((w >> (2 * (j & 15))) & 3u);
    }
    if (tier == PXQ_TIER_PXQ3) {
        uint32_t lo, hi;
        memcpy(&lo, row + 4 * (j >> 4), 4);                      // low plane of this half
        memcpy(&hi, row + 8, 4);                                 // w2, both halves' high plane
        return (int)((lo >> (2 * (j & 15))) & 3u) | (int)(((hi >> j) & 1u) << 2);
    }
    // PXQ4: byte b = j/2 holds code(2b) in the low nibble and code(2b+1) in the high nibble.
    const uint8_t byte = row[j >> 1];
    return (j & 1) ? (byte >> 4) : (byte & 0xf);
}

// deterministic LCG so a failure is reproducible from the tier alone
uint32_t lcg(uint32_t & s) { s = s * 1664525u + 1013904223u; return s; }

}  // namespace

int pxq23_selftest(int tier) {
    HostTier T;
    if (!host_tier(tier, T)) {
        fprintf(stderr, "pxq23_selftest: unknown tier %d\n", tier);
        return 10;
    }
    const int panels = 3, kslabs = 5;                 // 192 rows x 160 columns: not a power of
    const int N = panels * PXQ4_BM, K = kslabs * PXQ4_QK;   // two anywhere, and nfix > 1
    const size_t nslab_bytes = (size_t)panels * kslabs * T.slab;

    std::vector<uint8_t> h_slabs(nslab_bytes);
    std::vector<uint16_t> h_anchor((size_t)panels * PXQ4_BM);
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
            const uint8_t * slab = h_slabs.data() + ((size_t)p * kslabs + kb) * T.slab;
            for (int r = 0; r < PXQ4_BM; ++r) {
                const float anch = __half2float(__ushort_as_half(h_anchor[(size_t)p * PXQ4_BM + r]));
                const int   sb   = slab[r];
                const float eff0 = anch * host_sub16[sb & 0xf];
                const float eff1 = anch * host_sub16[sb >> 4];
                const uint8_t * row = slab + T.code_off + r * T.code_bytes;
                for (int j = 0; j < PXQ4_QK; ++j) {
                    const float e = (j < 16) ? eff0 : eff1;
                    const float w = e * T.book[host_code(tier, row, j)];
                    want[(size_t)(p * PXQ4_BM + r) * K + kb * PXQ4_QK + j] =
                        __half_as_ushort(__float2half_rn(w));
                }
            }
        }
    }

    // ---- device run
    uint8_t * d_slabs = nullptr; __half * d_anchor = nullptr; __half * d_out = nullptr;
    PXQ23_CUDA_CHECK(cudaMalloc(&d_slabs, nslab_bytes));
    PXQ23_CUDA_CHECK(cudaMalloc(&d_anchor, h_anchor.size() * 2));
    PXQ23_CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * K * 2));
    PXQ23_CUDA_CHECK(cudaMemcpy(d_slabs, h_slabs.data(), nslab_bytes, cudaMemcpyHostToDevice));
    PXQ23_CUDA_CHECK(cudaMemcpy(d_anchor, h_anchor.data(), h_anchor.size() * 2, cudaMemcpyHostToDevice));

    int rc = 0;
    if (tier == PXQ_TIER_PXQ4) {
        // TRANSCRIPTION GATE. Run the TEMPLATED PXQ4 instantiation from pxq23_kernel.cuh and
        // require it to equal both the host oracle and the SHIPPED pxq4 kernel on the same
        // bytes. If the policy transcription in pxq23_kernel.cuh ever drifts, this fails here,
        // before a model is loaded.
        const int64_t nsl = (int64_t)panels * kslabs;
        k_pxq23_dequant_matrix<pxq4_selftest_pol, __half><<<(unsigned)nsl, PXQ4_BM>>>(
            d_slabs, d_anchor, d_out, kslabs, (int64_t)K);
        PXQ23_CUDA_CHECK(cudaGetLastError());
    } else {
        pxq23_launch_dequant_f16(tier, d_slabs, d_anchor, d_out, panels, kslabs, 0);
    }
    PXQ23_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got((size_t)N * K);
    PXQ23_CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            fprintf(stderr, "pxq23_selftest(%s): DEQUANT MISMATCH at [%zu, %zu]: device 0x%04x "
                            "host 0x%04x\n", pxq_tier_name(tier), i / K, i % K, got[i], want[i]);
            rc = 1; break;
        }
    }

    if (rc == 0 && tier == PXQ_TIER_PXQ4) {
        // second arm of the gate: the shipped kernel on the same bytes
        std::vector<uint16_t> ship((size_t)N * K);
        pxq4_launch_dequant_f16(d_slabs, d_anchor, d_out, panels, kslabs, 0);
        PXQ23_CUDA_CHECK(cudaDeviceSynchronize());
        PXQ23_CUDA_CHECK(cudaMemcpy(ship.data(), d_out, ship.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < ship.size(); ++i) {
            if (ship[i] != want[i]) {
                fprintf(stderr, "pxq23_selftest(pxq4): SHIPPED KERNEL vs HOST ORACLE MISMATCH at "
                                "[%zu, %zu]: 0x%04x vs 0x%04x\n", i / K, i % K, ship[i], want[i]);
                rc = 2; break;
            }
        }
    }

    // ---- mmv arm: out = x @ W^T for M = 2, checked against a HOST REPLAY of the exact fold
    // (canonical chunks -> per-kseg partial -> kseg-ordered fold -> ONE rounding).
    //
    // WHY THIS ARM IS 1-ULP AND NOT BIT-EXACT, while the dequant arm above IS bit-exact.
    // The dequant arm performs no accumulation: every output is one multiply chain with a
    // single rounding, so host and device must agree to the bit and any disagreement is a
    // decode bug. This arm sums 160 products. Contracting `a*b + c` into an FMA is an
    // implementation freedom that BOTH compilers take by default -- nvcc via --fmad=true for
    // the device fold, and the host compiler via -ffp-contract -- and they are not obliged to
    // make the same choices at the same points. The engine's own build exercises that freedom,
    // so a host replay that is bit-exact against a contracted device fold is not achievable in
    // general; demanding it produced a false failure on PXQ3 (one row in 192, 0x20fc vs
    // 0x20fd) while every device-vs-device arm was clean.
    //
    // The oracle is still pinned as hard as it can be: this TU's host code is compiled with
    // -ffp-contract=off (build_pxq_v13.sh), so the replay below is a fixed, reproducible
    // sequence of separately-rounded operations rather than whatever the host compiler felt
    // like fusing. What is relaxed is only the comparison, and only by ONE unit in the last
    // place of fp16 -- about 0.05% -- with the count of differing rows reported. A real decode
    // bug does not produce one row off by one ULP; it produces many rows off by a lot, and
    // that still fails here. The bit-exact guarantees live where they are meaningful: the
    // dequant arm above, and the device-vs-device arms in gpu_selftest.py (mmv vs
    // dequant+cuBLAS, moe_mmv vs per-expert mmv).
    if (rc == 0 && tier != PXQ_TIER_PXQ4 && pxq23_mmv_supported(kslabs)) {
        const int M = 2;
        std::vector<uint16_t> h_x((size_t)M * K);
        for (size_t i = 0; i < h_x.size(); ++i) {
            const float v = ((float)(int)(lcg(s) % 401) - 200.f) / 256.f;
            h_x[i] = __half_as_ushort(__float2half_rn(v));
        }
        __half * d_x = nullptr; __half * d_o = nullptr;
        PXQ23_CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
        PXQ23_CUDA_CHECK(cudaMalloc(&d_o, (size_t)M * N * 2));
        PXQ23_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));
        pxq23_launch_mmv_f16(tier, d_slabs, d_anchor, d_x, d_o, M, panels, kslabs, true, 0);
        PXQ23_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint16_t> h_o((size_t)M * N);
        PXQ23_CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, h_o.size() * 2, cudaMemcpyDeviceToHost));

        const int nfix = pxq23_canon_nfix(kslabs, PXQ4_CANON_CMAX);
        int    ulp_off = 0, worst_ulp = 0;
        for (int m = 0; m < M && rc == 0; ++m) {
            for (int p = 0; p < panels && rc == 0; ++p) {
                for (int r = 0; r < PXQ4_BM; ++r) {
                    const float anch = __half2float(__ushort_as_half(h_anchor[(size_t)p * PXQ4_BM + r]));
                    float su[PXQ4_MMV_KSEG] = {0.f, 0.f, 0.f, 0.f};
                    for (int c = 0; c < nfix; ++c) {
                        const int b0 = (kslabs * c) / nfix, b1 = (kslabs * (c + 1)) / nfix;
                        for (int kseg = 0; kseg < PXQ4_MMV_KSEG; ++kseg) {
                            float t = 0.f;
                            for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
                                const uint8_t * slab = h_slabs.data()
                                    + ((size_t)p * kslabs + kb) * T.slab;
                                const int   sb   = slab[r];
                                const float eff0 = anch * host_sub16[sb & 0xf];
                                const float eff1 = anch * host_sub16[sb >> 4];
                                const uint8_t * row = slab + T.code_off + r * T.code_bytes;
                                // two partial sums, one per eff half, then eff-weighted -- the
                                // kernel's order, not a plain running dot product
                                float t0 = 0.f, t1 = 0.f;
                                for (int b = 0; b < 16; ++b) {
                                    const float a0 = T.book[host_code(tier, row, 2 * b)];
                                    const float a1 = T.book[host_code(tier, row, 2 * b + 1)];
                                    const float x0 = __half2float(__ushort_as_half(h_x[(size_t)m * K + kb * PXQ4_QK + 2 * b]));
                                    const float x1 = __half2float(__ushort_as_half(h_x[(size_t)m * K + kb * PXQ4_QK + 2 * b + 1]));
                                    float & acc = (b < 8) ? t0 : t1;
                                    acc = acc + (a0 * x0 + a1 * x1);
                                }
                                t += eff0 * t0 + eff1 * t1;
                            }
                            su[kseg] += t;
                        }
                    }
                    float u = 0.f;
                    for (int k = 0; k < PXQ4_MMV_KSEG; ++k) u += su[k];
                    const uint16_t w = __half_as_ushort(__float2half_rn(u));
                    const uint16_t g = h_o[(size_t)m * N + p * PXQ4_BM + r];
                    if (w != g) {
                        // ULP distance in the fp16 encoding. Both values come from the same
                        // dot product, so they share a sign and an exponent neighbourhood;
                        // comparing the raw encodings as integers is the standard trick and is
                        // exact for same-sign finite values.
                        const int d = (int)g - (int)w;
                        const int ad = d < 0 ? -d : d;
                        if (ad > worst_ulp) worst_ulp = ad;
                        ++ulp_off;
                        if (ad > 1) {
                            fprintf(stderr, "pxq23_selftest(%s): MMV MISMATCH at m=%d row=%d: "
                                            "device 0x%04x host 0x%04x (%d ULP -- more than "
                                            "the 1 ULP fp32 contraction allowance)\n",
                                    pxq_tier_name(tier), m, p * PXQ4_BM + r, g, w, ad);
                            rc = 3; break;
                        }
                    }
                }
            }
        }
        // A handful of 1-ULP rows is the contraction allowance; a large fraction is not, and
        // would mean the two folds are structurally different rather than differently fused.
        const int nrows = M * N;
        if (rc == 0 && ulp_off * 20 > nrows) {
            fprintf(stderr, "pxq23_selftest(%s): MMV: %d of %d rows differ by 1 ULP (>5%%). "
                            "That is too many for fp32 contraction alone -- the host replay and "
                            "the device fold are probably not the same fold.\n",
                    pxq_tier_name(tier), ulp_off, nrows);
            rc = 3;
        } else if (rc == 0 && ulp_off) {
            fprintf(stderr, "pxq23_selftest(%s): MMV: %d of %d rows off by 1 ULP "
                            "(fp32 contraction allowance), max %d ULP\n",
                    pxq_tier_name(tier), ulp_off, nrows, worst_ulp);
        }
        cudaFree(d_x); cudaFree(d_o);
    }

    cudaFree(d_slabs); cudaFree(d_anchor); cudaFree(d_out);
    if (rc == 0) {
        fprintf(stderr, "pxq23_selftest(%s): PASS (dequant bit-exact vs host oracle%s)\n",
                pxq_tier_name(tier),
                tier == PXQ_TIER_PXQ4 ? " and vs the shipped pxq4 kernel"
                                      : ", mmv within the 1 ULP contraction allowance");
    }
    return rc;
}

// =============================================================================================
// FUSED MoE decode block, PXQ2/PXQ3 instantiations (pxq_moe_fused.cuh), plus the two
// tier-independent helpers.
//
// Same TU-ownership rule as the pxq4 half: pxq2_book_g / pxq3_book_g / pxq23_sub16_g are
// TU-local, so the fused kernels that read them have to be instantiated here or they would
// silently bind to a second, never-uploaded copy of the tables. pxq4_selftest_pol is
// deliberately NOT instantiated -- pxq23_upload_book refuses tier 252, so that policy's book
// is whatever the header's initialiser left there, which is right up until a checkpoint ships
// a custom one.
// =============================================================================================

int pxq_moe_smem_bytes(int kslabs) {
    return pxq_moe_max_chunk_floats(kslabs) * (int)sizeof(float);
}

bool pxq_moe_supported(int kslabs) {
    // PURE ARITHMETIC, NO DRIVER CALL, and that is a performance requirement rather than a
    // style preference. This predicate runs inside every fused op, i.e. twice per MoE layer
    // and 80 times per decode step on the 35B. The first draft asked the driver
    // (cudaGetDevice + cudaDeviceGetAttribute) every single time; at the ~2 us a driver call
    // costs on this rig that is ~0.16 ms per token of pure overhead, roughly a tenth of the
    // win the fused block is supposed to deliver -- a kernel that pays for its own speedup at
    // the door. pxq23_mmv_supported already had this right and this now matches it.
    //
    // 48 KiB is the per-block static+dynamic shared budget available WITHOUT the
    // cudaFuncSetAttribute opt-in on sm_60 and sm_70 alike, so it is a constant of the two
    // architectures this library serves and needs no query. Static budget is taken from the
    // LARGER of the two fused kernels (the gateup, which carries two reduce halves because it
    // folds gate and up in one block) so one predicate is honest about both.
    const int stat_smem = (int)(2 * 16 * sizeof(float)
                                + 2 * PXQ4_MMV_KSEG * PXQ4_BM * sizeof(float));
    return kslabs > 0 && pxq_moe_smem_bytes(kslabs) + stat_smem <= 48 * 1024;
}

// v1: first shipped form. Bump on ANY change to the fused numerics or to the launcher ABI.
int pxq_moe_fused_version() { return 1; }

template <bool VECX>
struct pxq23_moe_dot_p2 {
    static __device__ __forceinline__ float dot(const uint8_t * __restrict__ slab, int row,
                                                float anch, const float * __restrict__ xk,
                                                const float * __restrict__ tab,
                                                const float * __restrict__ sub) {
        return pxq23_dot32<pxq2_pol, VECX>(slab, row, anch, xk, tab, sub);
    }
};
template <bool VECX>
struct pxq23_moe_dot_p3 {
    static __device__ __forceinline__ float dot(const uint8_t * __restrict__ slab, int row,
                                                float anch, const float * __restrict__ xk,
                                                const float * __restrict__ tab,
                                                const float * __restrict__ sub) {
        return pxq23_dot32<pxq3_pol, VECX>(slab, row, anch, xk, tab, sub);
    }
};

template <class POL, template <bool> class DOT>
static void pxq23_moe_gateup_tpl(const uint8_t * slabs, const void * anchor, const void * x,
                                 const int32_t * ids, void * act, int S, int Ip, int E,
                                 int panels13, int kslabs, int top_k, bool vecx,
                                 cudaStream_t stream) {
    const size_t smem = (size_t)pxq_moe_smem_bytes(kslabs);
    const dim3 grid((unsigned)(panels13 / 2), (unsigned)S, 1u);
    if (vecx) {
        k_pxq_moe_gateup_glu<POL, DOT<true>, true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, ids, (__half *)act,
            Ip, kslabs * PXQ4_QK, E, panels13, top_k);
    } else {
        k_pxq_moe_gateup_glu<POL, DOT<false>, false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, ids, (__half *)act,
            Ip, kslabs * PXQ4_QK, E, panels13, top_k);
    }
    PXQ23_CUDA_CHECK(cudaGetLastError());
}

void pxq23_moe_launch_gateup_glu(int tier, const uint8_t * slabs, const void * anchor,
                                 const void * x, const int32_t * ids, void * act, int S, int Ip,
                                 int E, int panels13, int kslabs, int top_k, bool vecx,
                                 cudaStream_t stream) {
    if ((panels13 & 1) || (panels13 / 2) * PXQ4_BM != Ip || S < 1 || S > 65535 ||
        top_k < 1 || S % top_k) {
        fprintf(stderr, "pxq23: fused moe gateup geometry invalid (panels13=%d Ip=%d S=%d "
                        "top_k=%d)\n", panels13, Ip, S, top_k);
        abort();
    }
    switch (tier) {
        case PXQ_TIER_PXQ2:
            pxq23_moe_gateup_tpl<pxq2_pol, pxq23_moe_dot_p2>(
                slabs, anchor, x, ids, act, S, Ip, E, panels13, kslabs, top_k, vecx, stream);
            return;
        case PXQ_TIER_PXQ3:
            pxq23_moe_gateup_tpl<pxq3_pol, pxq23_moe_dot_p3>(
                slabs, anchor, x, ids, act, S, Ip, E, panels13, kslabs, top_k, vecx, stream);
            return;
        default: pxq23_bad_tier(tier, "pxq23_moe_launch_gateup_glu");
    }
}

template <class POL, template <bool> class DOT>
static void pxq23_moe_down_fold_tpl(const uint8_t * slabs, const void * anchor, const void * act,
                                    const int32_t * ids, const float * wts, void * out, int M,
                                    int E, int panels2, int kslabs, int top_k, bool vecx,
                                    cudaStream_t stream) {
    const size_t smem = (size_t)pxq_moe_smem_bytes(kslabs);
    const dim3 grid((unsigned)panels2, (unsigned)M, 1u);
    if (vecx) {
        k_pxq_moe_down_fold<POL, DOT<true>, true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)act, ids, wts, (__half *)out,
            panels2 * PXQ4_BM, kslabs * PXQ4_QK, E, panels2, top_k);
    } else {
        k_pxq_moe_down_fold<POL, DOT<false>, false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)act, ids, wts, (__half *)out,
            panels2 * PXQ4_BM, kslabs * PXQ4_QK, E, panels2, top_k);
    }
    PXQ23_CUDA_CHECK(cudaGetLastError());
}

void pxq23_moe_launch_down_fold(int tier, const uint8_t * slabs, const void * anchor,
                                const void * act, const int32_t * ids, const float * wts,
                                void * out, int M, int E, int panels2, int kslabs, int top_k,
                                bool vecx, cudaStream_t stream) {
    if (panels2 < 1 || panels2 > 65535 || M < 1 || M > 65535 || top_k < 1) {
        fprintf(stderr, "pxq23: fused moe down geometry invalid (panels2=%d M=%d top_k=%d)\n",
                panels2, M, top_k);
        abort();
    }
    switch (tier) {
        case PXQ_TIER_PXQ2:
            pxq23_moe_down_fold_tpl<pxq2_pol, pxq23_moe_dot_p2>(
                slabs, anchor, act, ids, wts, out, M, E, panels2, kslabs, top_k, vecx, stream);
            return;
        case PXQ_TIER_PXQ3:
            pxq23_moe_down_fold_tpl<pxq3_pol, pxq23_moe_dot_p3>(
                slabs, anchor, act, ids, wts, out, M, E, panels2, kslabs, top_k, vecx, stream);
            return;
        default: pxq23_bad_tier(tier, "pxq23_moe_launch_down_fold");
    }
}

template <class POL, template <bool> class DOT>
static void pxq23_moe_down_part_tpl(const uint8_t * slabs, const void * anchor, const void * act,
                                    const int32_t * ids, void * dn, int S, int E, int panels2,
                                    int kslabs, bool vecx, cudaStream_t stream) {
    const size_t smem = (size_t)pxq_moe_smem_bytes(kslabs);
    const dim3 grid((unsigned)panels2, (unsigned)S, 1u);
    if (vecx) {
        k_pxq_moe_down_part<POL, DOT<true>, true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)act, ids, (__half *)dn,
            panels2 * PXQ4_BM, kslabs * PXQ4_QK, E, panels2);
    } else {
        k_pxq_moe_down_part<POL, DOT<false>, false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)act, ids, (__half *)dn,
            panels2 * PXQ4_BM, kslabs * PXQ4_QK, E, panels2);
    }
    PXQ23_CUDA_CHECK(cudaGetLastError());
}

void pxq23_moe_launch_down_part(int tier, const uint8_t * slabs, const void * anchor,
                                const void * act, const int32_t * ids, void * dn, int S, int E,
                                int panels2, int kslabs, bool vecx, cudaStream_t stream) {
    if (panels2 < 1 || panels2 > 65535 || S < 1 || S > 65535) {
        fprintf(stderr, "pxq23: fused moe down-part geometry invalid (panels2=%d S=%d)\n",
                panels2, S);
        abort();
    }
    switch (tier) {
        case PXQ_TIER_PXQ2:
            pxq23_moe_down_part_tpl<pxq2_pol, pxq23_moe_dot_p2>(
                slabs, anchor, act, ids, dn, S, E, panels2, kslabs, vecx, stream);
            return;
        case PXQ_TIER_PXQ3:
            pxq23_moe_down_part_tpl<pxq3_pol, pxq23_moe_dot_p3>(
                slabs, anchor, act, ids, dn, S, E, panels2, kslabs, vecx, stream);
            return;
        default: pxq23_bad_tier(tier, "pxq23_moe_launch_down_part");
    }
}

// Tier-independent: no tables, no policy. Lives here so form B needs no third TU.
void pxq_moe_launch_slot_fold(const void * dn, const float * wts, void * out, int M, int R,
                              int top_k, cudaStream_t stream) {
    const long long n = (long long)M * R;
    if (n < 1 || top_k < 1) {
        fprintf(stderr, "pxq23: slot fold geometry invalid (M=%d R=%d top_k=%d)\n", M, R, top_k);
        abort();
    }
    const unsigned blocks = (unsigned)((n + 255) / 256);
    k_pxq_moe_slot_fold<<<blocks, 256, 0, stream>>>(
        (const __half *)dn, wts, (__half *)out, M, R, top_k);
    PXQ23_CUDA_CHECK(cudaGetLastError());
}
