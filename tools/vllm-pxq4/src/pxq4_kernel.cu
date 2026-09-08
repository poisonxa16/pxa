// pxq4_kernel.cu — host launchers for the vendored PXQ4 sm_70 kernels.
//
// This TU owns the only copies of pxq4_book_g / pxq4_sub16_g, so the table upload/download
// helpers must live here (they are `static __device__` and therefore TU-local, exactly as in
// the engine, pxq6.cuh:79-81).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "pxq4_kernel.cuh"
#include "pxq4_kernel_launch.h"

#define PXQ4_CUDA_CHECK(expr)                                                                 \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "pxq4: %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                                \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

int pxq4_mmv_smem_bytes(int kslabs) {
    return pxq4_canon_max_chunk(kslabs) * PXQ4_QK * (int)sizeof(float);
}

bool pxq4_mmv_supported(int kslabs) {
    // 48 KiB is the sm_70 per-block static+dynamic shared budget available WITHOUT the
    // cudaFuncSetAttribute opt-in. We deliberately stay under it: opting in to the 96 KiB
    // Volta maximum would drop the kernel to one block per SM, and chunked staging (see
    // k_pxq4_mmv) means no real shape ever needs it — K = 17408 costs 4352 B.
    // 2*16 floats of tables + the KSEG*BM reduce tile + the fused kernel's `last` flag
    // (k_pxq4_mmv_fused: ptxas reports 1168 B static, vs 1152 B for k_pxq4_mmv). Budget for
    // the largest of the three, so a shape can never be admitted that the fused path cannot run.
    const int stat_smem = (int)(2 * 16 * sizeof(float) + PXQ4_MMV_KSEG * PXQ4_BM * sizeof(float)
                                + 16);
    return pxq4_mmv_smem_bytes(kslabs) + stat_smem <= 48 * 1024;
}

void pxq4_launch_dequant_f16(const uint8_t * slabs, const void * anchor, void * out,
                             int panels, int kslabs, cudaStream_t stream) {
    const int64_t nslabs = (int64_t)panels * kslabs;
    const int64_t K      = (int64_t)kslabs * PXQ4_QK;
    // one block per slab, 64 threads (one weight row each)
    k_pxq4_dequant_matrix<__half><<<(unsigned)nslabs, PXQ4_BM, 0, stream>>>(
        slabs, (const __half *)anchor, (__half *)out, kslabs, K);
    PXQ4_CUDA_CHECK(cudaGetLastError());
}

void pxq4_launch_mmv_f16(const uint8_t * slabs, const void * anchor, const void * x, void * out,
                         int M, int panels, int kslabs, bool vecx, cudaStream_t stream) {
    const int    R     = panels * PXQ4_BM;
    const int    K     = kslabs * PXQ4_QK;
    const size_t smem  = (size_t)pxq4_mmv_smem_bytes(kslabs);
    const dim3   grid((unsigned)panels, (unsigned)M, 1u);
    if (vecx) {
        k_pxq4_mmv<true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    } else {
        k_pxq4_mmv<false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    }
    PXQ4_CUDA_CHECK(cudaGetLastError());
}

int pxq4_mmv_nfix(int kslabs) {
    return pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX);
}

void pxq4_launch_mmv_split_f16(const uint8_t * slabs, const void * anchor, const void * x,
                               float * part, void * out, int M, int panels, int kslabs,
                               bool vecx, cudaStream_t stream) {
    const int    R    = panels * PXQ4_BM;
    const int    K    = kslabs * PXQ4_QK;
    const int    nfix = pxq4_mmv_nfix(kslabs);
    const size_t smem = (size_t)pxq4_mmv_smem_bytes(kslabs);
    // ONE nfix, passed to BOTH kernels. Pre-swap, k_pxq4_mmv_part read nfix off gridDim.y
    // while k_pxq4_mmv_reduce took it as an argument and nothing asserted they agreed; a
    // mismatch produced silently wrong output. Now there is a single source of truth and it is
    // checked here, since neither kernel can check it for itself. panels is gridDim.y under the
    // chunk-major order, so it must clear the 65535 limit (136 at TP4, 272 at TP1 for gate_up).
    if (nfix != pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX) || nfix < 1 || panels < 1) {
        fprintf(stderr, "pxq4: split mmv nfix/panels inconsistent (nfix=%d panels=%d "
                        "kslabs=%d) at %s:%d\n", nfix, panels, kslabs, __FILE__, __LINE__);
        abort();
    }
    if (panels > 65535 || M > 65535) {
        fprintf(stderr, "pxq4: split mmv grid limit exceeded: panels=%d M=%d\n", panels, M);
        abort();
    }
    const dim3   grid((unsigned)nfix, (unsigned)panels, (unsigned)M);
    if (vecx) {
        k_pxq4_mmv_part<true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, part, K, nfix, panels);
    } else {
        k_pxq4_mmv_part<false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, part, K, nfix, panels);
    }
    PXQ4_CUDA_CHECK(cudaGetLastError());
    const dim3 rgrid((unsigned)panels, (unsigned)M, 1u);
    k_pxq4_mmv_reduce<<<rgrid, PXQ4_BM, 0, stream>>>(part, (__half *)out, nfix, R);
    PXQ4_CUDA_CHECK(cudaGetLastError());
}

// Single-launch fused split mmv. Identical values to pxq4_launch_mmv_split_f16 (see
// k_pxq4_mmv_fused). `ctr` is M*panels unsigned, zeroed once at allocation; every COMPLETED
// launch leaves it zeroed, so steady state needs no memset and nothing is allocated in-capture.
void pxq4_launch_mmv_fused_f16(const uint8_t * slabs, const void * anchor, const void * x,
                               float * part, unsigned * ctr, void * out, int M, int panels,
                               int kslabs, bool vecx, cudaStream_t stream) {
    const int    R    = panels * PXQ4_BM;
    const int    K    = kslabs * PXQ4_QK;
    const int    nfix = pxq4_mmv_nfix(kslabs);
    const size_t smem = (size_t)pxq4_mmv_smem_bytes(kslabs);
    if (nfix < 2) {
        // nfix == 1 makes the barrier degenerate; the monolithic path owns that case and the
        // dispatch guard already excludes it. Keep the refusal explicit rather than implicit.
        fprintf(stderr, "pxq4: fused split mmv requires nfix >= 2 (got %d)\n", nfix);
        abort();
    }
    if (nfix != pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX) || panels < 1) {
        fprintf(stderr, "pxq4: fused mmv nfix/panels inconsistent (nfix=%d panels=%d "
                        "kslabs=%d)\n", nfix, panels, kslabs);
        abort();
    }
    if (panels > 65535 || M > 65535) {
        fprintf(stderr, "pxq4: fused mmv grid limit exceeded: panels=%d M=%d\n", panels, M);
        abort();
    }
    const dim3 grid((unsigned)nfix, (unsigned)panels, (unsigned)M);   // c fastest
    if (vecx) {
        k_pxq4_mmv_fused<true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, part, ctr, (__half *)out,
            R, K, nfix, panels);
    } else {
        k_pxq4_mmv_fused<false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, part, ctr, (__half *)out,
            R, K, nfix, panels);
    }
    PXQ4_CUDA_CHECK(cudaGetLastError());
}

void pxq4_upload_tables(const float * book16, const float * sub16) {
    PXQ4_CUDA_CHECK(cudaMemcpyToSymbol(pxq4_book_g,  book16, 16 * sizeof(float)));
    PXQ4_CUDA_CHECK(cudaMemcpyToSymbol(pxq4_sub16_g, sub16,  16 * sizeof(float)));
}

void pxq4_download_tables(float * book16, float * sub16) {
    PXQ4_CUDA_CHECK(cudaMemcpyFromSymbol(book16, pxq4_book_g,  16 * sizeof(float)));
    PXQ4_CUDA_CHECK(cudaMemcpyFromSymbol(sub16,  pxq4_sub16_g, 16 * sizeof(float)));
}

// v6: multi-token fused split mmv. One block owns all M tokens of its (chunk, panel); weight
// bytes are decoded once per block instead of once per (block, token). M must be 1..8 and is
// dispatched to an exact-MT template so every inner loop fully unrolls. Values are per-token
// bit-identical to pxq4_launch_mmv_f16 (see k_pxq4_mmv_fused_mt). `ctr` is panels unsigned,
// zero on entry and left zero by every completed launch.
int pxq4_mmv_mt_smem_bytes(int kslabs, int M) {
    return pxq4_canon_max_chunk(kslabs) * PXQ4_QK * (int)sizeof(float) * M;
}

bool pxq4_mmv_mt_supported(int kslabs, int M) {
    if (M < 1 || M > 8) return false;
    const int stat_smem = (int)(2 * 16 * sizeof(float) + PXQ4_MMV_KSEG * PXQ4_BM * sizeof(float)
                                + 16);
    return pxq4_mmv_mt_smem_bytes(kslabs, M) + stat_smem <= 48 * 1024;
}

template <bool VECX>
static void pxq4_dispatch_mmv_fused_mt(const uint8_t * slabs, const __half * anchor,
                                       const __half * x, float * part, unsigned * ctr,
                                       __half * out, int M, int R, int K, int nfix,
                                       int panels, size_t smem, cudaStream_t stream) {
    const dim3 grid((unsigned)nfix, (unsigned)panels, 1u);
    switch (M) {
#define PXQ4_MT_CASE(MTV)                                                                     \
        case MTV:                                                                             \
            k_pxq4_mmv_fused_mt<VECX, MTV><<<grid, 256, smem, stream>>>(                      \
                slabs, anchor, x, part, ctr, out, R, K, nfix, panels);                        \
            break;
        PXQ4_MT_CASE(1) PXQ4_MT_CASE(2) PXQ4_MT_CASE(3) PXQ4_MT_CASE(4)
        PXQ4_MT_CASE(5) PXQ4_MT_CASE(6) PXQ4_MT_CASE(7) PXQ4_MT_CASE(8)
#undef PXQ4_MT_CASE
        default:
            fprintf(stderr, "pxq4: fused mt mmv M out of range: %d\n", M);
            abort();
    }
}

void pxq4_launch_mmv_fused_mt_f16(const uint8_t * slabs, const void * anchor, const void * x,
                                  float * part, unsigned * ctr, void * out, int M, int panels,
                                  int kslabs, bool vecx, cudaStream_t stream) {
    const int    R    = panels * PXQ4_BM;
    const int    K    = kslabs * PXQ4_QK;
    const int    nfix = pxq4_mmv_nfix(kslabs);
    const size_t smem = (size_t)pxq4_mmv_mt_smem_bytes(kslabs, M);
    if (nfix < 2) {
        fprintf(stderr, "pxq4: fused mt mmv requires nfix >= 2 (got %d)\n", nfix);
        abort();
    }
    if (!pxq4_mmv_mt_supported(kslabs, M)) {
        fprintf(stderr, "pxq4: fused mt mmv unsupported (kslabs=%d M=%d)\n", kslabs, M);
        abort();
    }
    if (nfix != pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX) || panels < 1 || panels > 65535) {
        fprintf(stderr, "pxq4: fused mt mmv nfix/panels inconsistent (nfix=%d panels=%d "
                        "kslabs=%d)\n", nfix, panels, kslabs);
        abort();
    }
    if (vecx) {
        pxq4_dispatch_mmv_fused_mt<true>(slabs, (const __half *)anchor, (const __half *)x,
                                         part, ctr, (__half *)out, M, R, K, nfix, panels,
                                         smem, stream);
    } else {
        pxq4_dispatch_mmv_fused_mt<false>(slabs, (const __half *)anchor, (const __half *)x,
                                          part, ctr, (__half *)out, M, R, K, nfix, panels,
                                          smem, stream);
    }
    PXQ4_CUDA_CHECK(cudaGetLastError());
}
