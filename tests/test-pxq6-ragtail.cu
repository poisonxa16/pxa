// test-pxq6-ragtail.cu -- bug #227: the PXQ grouped prefill GEMM / SCATFUSE down pickers
// (PXQ6_PICK_FMT) hard-wired RAG=false for PXQ2 / PXQ3 / PXQ6R / PXQ1, so the default-on K4
// ragged-tile FMA skip (PXA_PXQ6_RAGTAIL) never ran on those tiers.
//
//   PHASE 1 (host only, no card needed): for every tier, pxq6_pick_gemm(fmt, rag, pipe) and
//           pxq6_pick_down_scat(fmt, rag, pipe) must return the <POL, rag, pipe> instantiation.
//           On the unfixed tree the rag=true picks of the four low tiers return the rag=false
//           kernel and this phase fails.
//   PHASE 2 (needs a card): K4 is bit-exact by construction -- it skips only FMAs whose results
//           are never stored. Every tier's RAG=true kernel is run against its RAG=false kernel
//           on ragged tiles (nrows 1, 5, 17, 40, 64) and the outputs are compared BY BITS,
//           including the untouched guard rows of a pre-poisoned C.
//
// Nonzero exit on any failure.
#include "ggml-cuda/pxa/pxq6.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0;

template <class POL>
static void phase1_tier(int fmt, const char * name) {
    for (int pipe = 0; pipe < 2; ++pipe) {
        for (int rag = 0; rag < 2; ++rag) {
            const pxq6_gemm_fn want_g = PXQ6_PICK2(k_pxq6_gemm_grouped, POL, rag != 0, pipe != 0);
            const pxq6_scat_fn want_s = PXQ6_PICK2(k_pxq6_gemm_down_scat, POL, rag != 0, pipe != 0);
            if (pxq6_pick_gemm(fmt, rag != 0, pipe != 0) != want_g) {
                printf("  FAIL %-5s pxq6_pick_gemm(rag=%d pipe=%d) is not <POL,%d,%d>\n", name, rag, pipe, rag, pipe);
                ++g_fail;
            }
            if (pxq6_pick_down_scat(fmt, rag != 0, pipe != 0) != want_s) {
                printf("  FAIL %-5s pxq6_pick_down_scat(rag=%d pipe=%d) is not <POL,%d,%d>\n", name, rag, pipe, rag, pipe);
                ++g_fail;
            }
        }
    }
    printf("  %-5s pickers checked\n", name);
}

static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

template <class POL>
static void phase2_tier(int fmt, const char * name, uint64_t & seed) {
    const int panels = 2, kslabs = 4;
    const int R = panels * PXQ6_BM, K = kslabs * PXQ6_QK;
    const int nrows_list[] = { 1, 5, 17, 40, 64 };
    const int n_tiles = (int)(sizeof(nrows_list) / sizeof(nrows_list[0]));
    const int n_exp = 2;

    // weights: finite fp16 anchors in the header (as the h2 harness does), random code bytes
    const size_t stride = (size_t)POL::HDR + (size_t)kslabs*POL::SLAB;
    std::vector<uint8_t> W(stride * panels * n_exp);
    for (size_t pe = 0; pe < (size_t)panels*n_exp; ++pe) {
        uint8_t * pan = W.data() + pe*stride;
        for (int b = 0; b < POL::HDR; ++b) pan[b] = 0;
        if (POL::HDR >= 2*PXQ6_BM) {
            uint16_t * anc = (uint16_t *)pan;
            for (int r = 0; r < PXQ6_BM; ++r) {
                const int e = 8 + (int)(rs(seed) % 8);
                anc[r] = (uint16_t)(((rs(seed) & 1) << 15) | (e << 10) | (rs(seed) & 0x3ff));
            }
        }
        for (size_t b = POL::HDR; b < stride; ++b) pan[b] = (uint8_t)(rs(seed) & 0xff);
    }

    std::vector<pxq4_tile_info> tiles(n_tiles);
    int total = 0;
    for (int t = 0; t < n_tiles; ++t) {
        tiles[t].e = t % n_exp; tiles[t].row0 = total; tiles[t].nrows = nrows_list[t]; tiles[t]._pad = 0;
        total += nrows_list[t];
    }
    const int rows_alloc = total + 64;   // the kernels read A rows only for tid < nrows; slack anyway
    std::vector<half> A((size_t)rows_alloc * K);
    for (auto & v : A) v = __float2half((float)((int)(rs(seed) % 2001) - 1000) / 1000.0f);

    uint8_t * dW; half * dA; float * dC0; float * dC1; pxq4_tile_info * dT;
    CUCK(cudaMalloc(&dW, W.size()));
    CUCK(cudaMalloc(&dA, A.size()*sizeof(half)));
    const size_t csz = (size_t)rows_alloc * R * sizeof(float);
    CUCK(cudaMalloc(&dC0, csz)); CUCK(cudaMalloc(&dC1, csz));
    CUCK(cudaMalloc(&dT, tiles.size()*sizeof(pxq4_tile_info)));
    CUCK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(dA, A.data(), A.size()*sizeof(half), cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(dT, tiles.data(), tiles.size()*sizeof(pxq4_tile_info), cudaMemcpyHostToDevice));

    bool ok = true;
    for (int pipe = 0; pipe < 2; ++pipe) {
        CUCK(cudaMemset(dC0, 0xA5, csz)); CUCK(cudaMemset(dC1, 0xA5, csz));
        const pxq6_gemm_fn k0 = pxq6_pick_gemm(fmt, false, pipe != 0);
        const pxq6_gemm_fn k1 = pxq6_pick_gemm(fmt, true,  pipe != 0);
        dim3 grid((unsigned)panels, (unsigned)n_tiles);
        k0<<<grid, 64>>>(dW, dA, dC0, nullptr, 0, dT, R, K);
        k1<<<grid, 64>>>(dW, dA, dC1, nullptr, 0, dT, R, K);
        CUCK(cudaGetLastError());
        CUCK(cudaDeviceSynchronize());
        std::vector<uint8_t> h0(csz), h1(csz);
        CUCK(cudaMemcpy(h0.data(), dC0, csz, cudaMemcpyDeviceToHost));
        CUCK(cudaMemcpy(h1.data(), dC1, csz, cudaMemcpyDeviceToHost));
        if (memcmp(h0.data(), h1.data(), csz) != 0) {
            printf("  FAIL %-5s gemm RAG on/off differ (pipe=%d)\n", name, pipe);
            ok = false;
        }
    }
    printf("  %s %-5s gemm RAG on == RAG off by bits (ragged nrows 1/5/17/40/64, pipe 0/1)\n", ok ? "ok  " : "FAIL", name);
    if (!ok) ++g_fail;
    cudaFree(dW); cudaFree(dA); cudaFree(dC0); cudaFree(dC1); cudaFree(dT);
}

int main() {
    printf("test-pxq6-ragtail (bug #227)\nPHASE 1  picker honours RAG on every tier\n");
    phase1_tier<pxq6_pol_p6>  (PXA_PXQ_FMT_P6,   "P6");
    phase1_tier<pxq6_pol_p6hq>(PXA_PXQ_FMT_P6HQ, "P6HQ");
    phase1_tier<pxq6_pol_p2>  (PXA_PXQ_FMT_P2,   "P2");
    phase1_tier<pxq6_pol_p3>  (PXA_PXQ_FMT_P3,   "P3");
    phase1_tier<pxq6_pol_p6r> (PXA_PXQ_FMT_P6R,  "P6R");
    phase1_tier<pxq6_pol_p1>  (PXA_PXQ_FMT_P1,   "P1");

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        (void)cudaGetLastError();
        printf("PHASE 2  SKIP: no CUDA device\n");
    } else {
        printf("PHASE 2  RAG on/off bit-identity on the device\n");
        uint64_t seed = 0x227ull * 0x9e3779b97f4a7c15ull;
        phase2_tier<pxq6_pol_p6>  (PXA_PXQ_FMT_P6,   "P6",   seed);
        phase2_tier<pxq6_pol_p2>  (PXA_PXQ_FMT_P2,   "P2",   seed);
        phase2_tier<pxq6_pol_p3>  (PXA_PXQ_FMT_P3,   "P3",   seed);
        phase2_tier<pxq6_pol_p6r> (PXA_PXQ_FMT_P6R,  "P6R",  seed);
        phase2_tier<pxq6_pol_p1>  (PXA_PXQ_FMT_P1,   "P1",   seed);
    }
    printf("%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}
