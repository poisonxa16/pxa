// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// test-pxq-mmvq-cols.cu -- differential harness for the PXQ MMVQ weight-decode hoist.
//
// WHAT IS UNDER TEST. k_mul_mat_vec_q / k_fused_mul_mat_vec_q (mmvq-templates.cuh) used to unpack
// a weight fragment -- the 8 B/16 B code load, the byte-perm book gather and the per-code-word
// sub-scale lookup -- once per (activation column, row, k-block), although none of it depends on
// the column. They now unpack it once per (row, k-block) and reuse it across all ncols_y columns
// (pxq_mmvq_wfrag / pxq_mmvq_dot_frag in pxa/pxq-mmvq.cuh). The per-column arithmetic and its
// order are untouched, so the restructure must be BITWISE identical, not merely close.
//
// HOW. The pre-hoist structure is still reachable as the HOIST=false instantiation of the same
// kernels, calling the untouched vec_dot_pxq_q8_1(). Both arms run in this one binary on the same
// random PXQ4/PXQ4HQ panels and the same q8_1 activations, and every output float must compare
// equal BY BITS. Any difference is a defect in the restructure, not a tolerance question -- there
// is no epsilon in this test.
//
// COVERAGE. All four MMVQ-registered PXQ tiers -- the two 4-bit ones (which differ in sub-scale
// granularity and therefore in the scale-byte indexing the hoist has to carry along) and, since
// 2026-09-09, PXQ2 and PXQ3 (which differ in the code load and the book gather) -- every tile
// height the launcher can pick
// (PXA_PXQ_MMVQ_ROWS 1|2|4|8|16), both code-words-per-thread settings (PXA_PXQ_MMVQ_VDR 2|4),
// ncols_y in {1,2,4,8} -- 1 is the single-column path that must NOT take the hoisted branch at
// all, 2/4/8 are the spec-verify widths the hoist exists for -- and several (rows, K) shapes.
// The fused up/gate twin is covered at the default tile.
//
// PLUS, since 2026-09-09, ONE THING ONLY A DEVICE CAN SAY. The PXQ2/PXQ3 weight decode is written
// once, in ggml-cuda/pxa/pxq23-mmvq.h, and compiles for both sides: PRMT on the device, an exact
// emulation of PRMT on the host. tests/test-pxq23-mmvq-snap.cpp proves the HOST arm against the
// format for every bit pattern either tier admits; the section at the end of this file runs the
// DEVICE arm over those same patterns -- all 2^16 for PXQ2, all 2^24 for PXQ3 -- and compares by
// bits. Between the two, the decode is proved exhaustively on the hardware it ships on.
//
// Needs a real CUDA device (~300 MB for the exhaustive PXQ3 sweep). Run inside the shared GPU
// lock window. Nonzero exit on any mismatch, on a non-finite output, or if an arm produced
// nothing to compare.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmvq-templates.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static int g_fail = 0, g_cases = 0;
static long long g_vals = 0;

static inline uint64_t rs(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// ---------------------------------------------------------------------------------------------
// synthetic inputs. Raw bytes only: the panel/slab layout is the contract, and a random slab is a
// perfectly valid one -- every 4-bit code and every 4-bit sub index is in range by construction.
// ---------------------------------------------------------------------------------------------
struct shapebuf {
    int R, K, ny, kslabs, panels;
    size_t pstride, wbytes;
    std::vector<uint8_t> Wu, Wg;   // two independent weight matrices (up / gate)
    std::vector<uint8_t> Y;        // ny * kslabs block_q8_1
};

static void make_shape(shapebuf & S, int slab_bytes, int R, int K, int ny, uint64_t & seed) {
    S.R = R; S.K = K; S.ny = ny;
    S.kslabs = K/32;
    S.panels = R/64;
    S.pstride = (size_t)PXQ6_HDR_BYTES + (size_t)S.kslabs*slab_bytes;
    S.wbytes  = (size_t)S.panels*S.pstride;

    for (std::vector<uint8_t> * W : { &S.Wu, &S.Wg }) {
        W->assign(S.wbytes, 0);
        for (size_t i = 0; i < S.wbytes; ++i) (*W)[i] = (uint8_t)(rs(seed) & 0xff);
        for (int p = 0; p < S.panels; ++p) {            // sane fp16 anchors: no inf/nan/denormal
            uint16_t * an = (uint16_t *)&(*W)[(size_t)p*S.pstride];
            for (int r = 0; r < 64; ++r) {
                const int e = 10 + (int)(rs(seed) % 6);              // exp 10..15 -> 2^-5..2^0
                an[r] = (uint16_t)(((rs(seed) & 1) << 15) | (e << 10) | (rs(seed) & 0x3ff));
            }
        }
    }

    // block_q8_1 = { half d; half s; int8 qs[32]; }. The PXQ dot reads d and qs only.
    S.Y.assign((size_t)ny*S.kslabs*sizeof(block_q8_1), 0);
    for (int t = 0; t < ny; ++t) {
        for (int g = 0; g < S.kslabs; ++g) {
            uint8_t * b = &S.Y[((size_t)t*S.kslabs + g)*sizeof(block_q8_1)];
            const int e = 6 + (int)(rs(seed) % 4);                   // exp 6..9 -> 2^-9..2^-6
            ((uint16_t *)b)[0] = (uint16_t)((e << 10) | (rs(seed) & 0x3ff));   // d, positive
            ((uint16_t *)b)[1] = 0;                                            // s, unused here
            for (int j = 0; j < 32; ++j) b[4 + j] = (uint8_t)(rs(seed) & 0xff);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// one (type, ncols_y, nwarps, ROWS, VDR) case: launch both arms, compare by bits
// ---------------------------------------------------------------------------------------------
struct devbufs { void * wu; void * wg; void * y; float * d0; float * d1; };

static void verdict(const char * what, ggml_type type, int ny, int nw, int rows, int vdr,
                    int R, int K, const std::vector<float> & a, const std::vector<float> & b) {
    ++g_cases;
    int bad = 0, nonfinite = 0;
    double amax = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        if (ua != ub) ++bad;
        if (!std::isfinite(a[i])) ++nonfinite;
        amax = fmax(amax, fabs((double)a[i]));
    }
    g_vals += (long long)a.size();
    const char * tn = type == GGML_TYPE_PXQ4   ? "PXQ4  "
                    : type == GGML_TYPE_PXQ4HQ ? "PXQ4HQ"
                    : type == GGML_TYPE_PXQ2   ? "PXQ2  " : "PXQ3  ";
    if (bad || nonfinite || amax == 0.0) {
        ++g_fail;
        printf("  FAIL %-6s %s ny=%d nw=%d rows=%2d vdr=%d R=%3d K=%4d : %d/%zu bits differ, "
               "%d non-finite, |max|=%g\n", tn, what, ny, nw, rows, vdr, R, K, bad, a.size(),
               nonfinite, amax);
        for (size_t i = 0, shown = 0; i < a.size() && shown < 4; ++i) {
            uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
            if (ua != ub) { printf("      [%zu] ref %.9g (0x%08x) vs new %.9g (0x%08x)\n",
                                   i, a[i], ua, b[i], ub); ++shown; }
        }
    }
}

template <ggml_type type, int NY, int NW, int ROWS, int VDR>
static void case_plain(const shapebuf & S, const devbufs & D) {
    constexpr int rpb = mmvq_rows_per_block<type, NY, ROWS>;
    const dim3 grid((S.R + rpb - 1)/rpb, 1, 1), blk(WARP_SIZE, NW, 1);
    const size_t nout = (size_t)NY*S.R;

    CUCK(cudaMemset(D.d0, 0xa5, nout*sizeof(float)));
    CUCK(cudaMemset(D.d1, 0x5a, nout*sizeof(float)));
    mul_mat_vec_q<type, NY, NW, ROWS, VDR, false><<<grid, blk>>>(
        D.wu, D.y, D.d0, nullptr, nullptr, S.K, S.R, S.K, S.R, 0, 0, 0, 0, 0);
    mul_mat_vec_q<type, NY, NW, ROWS, VDR, true ><<<grid, blk>>>(
        D.wu, D.y, D.d1, nullptr, nullptr, S.K, S.R, S.K, S.R, 0, 0, 0, 0, 0);
    CUCK(cudaGetLastError());
    CUCK(cudaDeviceSynchronize());

    std::vector<float> a(nout), b(nout);
    CUCK(cudaMemcpy(a.data(), D.d0, nout*sizeof(float), cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(b.data(), D.d1, nout*sizeof(float), cudaMemcpyDeviceToHost));
    verdict("mmvq ", type, NY, NW, ROWS, VDR, S.R, S.K, a, b);
}

// TILE INVARIANCE. The multi-column tile height decides which OUTPUT ROWS a block owns; it does
// not decide which k-blocks a thread accumulates (that is vdr*nwarps*WARP_SIZE/qi) nor the order
// it accumulates them, so every output must be bit-identical at every tile height. That is
// exactly the claim PXA_PXQ_MMVQ_COLS makes when it widens the tile at ncols_y > 2, and it is
// what makes an armed verify step bit-identical to an unarmed one. Checked here per tier rather
// than argued, on both the plain kernel and the up/gate twin, against the tile that ships today.
template <ggml_type type, int NY, int NW, int ROWS, int VDR, bool FUSED>
static void case_tile(const shapebuf & S, const devbufs & D) {
    constexpr int rpb0 = mmvq_rows_per_block<type, NY, 2>;      // the shipped clamp at ncols_y > 2
    constexpr int rpb1 = mmvq_rows_per_block<type, NY, ROWS>;
    const dim3 g0((S.R + rpb0 - 1)/rpb0, 1, 1), g1((S.R + rpb1 - 1)/rpb1, 1, 1), blk(WARP_SIZE, NW, 1);
    const size_t nout = (size_t)NY*S.R;

    CUCK(cudaMemset(D.d0, 0xa5, nout*sizeof(float)));
    CUCK(cudaMemset(D.d1, 0x5a, nout*sizeof(float)));
    if constexpr (FUSED) {
        fused_mul_mat_vec_q<type, NY, NW, 2, VDR, true><<<g0, blk>>>(
            D.wu, D.wg, D.y, D.d0, nullptr, nullptr, nullptr, 0,
            S.K, S.R, S.K, S.R, 0, 0, 0, 0, GGML_UNARY_OP_SILU, 1.0e30f);
        fused_mul_mat_vec_q<type, NY, NW, ROWS, VDR, true><<<g1, blk>>>(
            D.wu, D.wg, D.y, D.d1, nullptr, nullptr, nullptr, 0,
            S.K, S.R, S.K, S.R, 0, 0, 0, 0, GGML_UNARY_OP_SILU, 1.0e30f);
    } else {
        mul_mat_vec_q<type, NY, NW, 2, VDR, true><<<g0, blk>>>(
            D.wu, D.y, D.d0, nullptr, nullptr, S.K, S.R, S.K, S.R, 0, 0, 0, 0, 0);
        mul_mat_vec_q<type, NY, NW, ROWS, VDR, true><<<g1, blk>>>(
            D.wu, D.y, D.d1, nullptr, nullptr, S.K, S.R, S.K, S.R, 0, 0, 0, 0, 0);
    }
    CUCK(cudaGetLastError());
    CUCK(cudaDeviceSynchronize());

    std::vector<float> a(nout), b(nout);
    CUCK(cudaMemcpy(a.data(), D.d0, nout*sizeof(float), cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(b.data(), D.d1, nout*sizeof(float), cudaMemcpyDeviceToHost));
    verdict(FUSED ? "tile-f" : "tile-p", type, NY, NW, ROWS, VDR, S.R, S.K, a, b);
}

template <ggml_type type, int NY, int NW, int ROWS, int VDR>
static void case_fused(const shapebuf & S, const devbufs & D) {
    constexpr int rpb = mmvq_rows_per_block<type, NY, ROWS>;
    const dim3 grid((S.R + rpb - 1)/rpb, 1, 1), blk(WARP_SIZE, NW, 1);
    const size_t nout = (size_t)NY*S.R;

    CUCK(cudaMemset(D.d0, 0xa5, nout*sizeof(float)));
    CUCK(cudaMemset(D.d1, 0x5a, nout*sizeof(float)));
    fused_mul_mat_vec_q<type, NY, NW, ROWS, VDR, false><<<grid, blk>>>(
        D.wu, D.wg, D.y, D.d0, nullptr, nullptr, nullptr, 0,
        S.K, S.R, S.K, S.R, 0, 0, 0, 0, GGML_UNARY_OP_SILU, 1.0e30f);
    fused_mul_mat_vec_q<type, NY, NW, ROWS, VDR, true ><<<grid, blk>>>(
        D.wu, D.wg, D.y, D.d1, nullptr, nullptr, nullptr, 0,
        S.K, S.R, S.K, S.R, 0, 0, 0, 0, GGML_UNARY_OP_SILU, 1.0e30f);
    CUCK(cudaGetLastError());
    CUCK(cudaDeviceSynchronize());

    std::vector<float> a(nout), b(nout);
    CUCK(cudaMemcpy(a.data(), D.d0, nout*sizeof(float), cudaMemcpyDeviceToHost));
    CUCK(cudaMemcpy(b.data(), D.d1, nout*sizeof(float), cudaMemcpyDeviceToHost));
    verdict("fused", type, NY, NW, ROWS, VDR, S.R, S.K, a, b);
}

// nwarps follows mul_mat_vec_q_cuda(): 4 up to ncols_y 4, 2 above.
template <ggml_type type, int ROWS, int VDR>
static void sweep_plain(const shapebuf & S, const devbufs & D) {
    case_plain<type, 1, 4, ROWS, VDR>(S, D);
    case_plain<type, 2, 4, ROWS, VDR>(S, D);
    case_plain<type, 4, 4, ROWS, VDR>(S, D);
    case_plain<type, 8, 2, ROWS, VDR>(S, D);
}

template <ggml_type type, int ROWS, int VDR>
static void sweep_fused(const shapebuf & S, const devbufs & D) {
    case_fused<type, 1, 4, ROWS, VDR>(S, D);
    case_fused<type, 2, 4, ROWS, VDR>(S, D);
    case_fused<type, 4, 4, ROWS, VDR>(S, D);
    case_fused<type, 8, 2, ROWS, VDR>(S, D);
}

template <ggml_type type>
static void run_type(int R, int K, uint64_t & seed) {
    constexpr int slab = pxq_mmvq_slab<type>;

    shapebuf S;
    make_shape(S, slab, R, K, 8, seed);

    devbufs D{};
    CUCK(cudaMalloc(&D.wu, S.wbytes));
    CUCK(cudaMalloc(&D.wg, S.wbytes));
    CUCK(cudaMalloc(&D.y,  S.Y.size()));
    CUCK(cudaMalloc(&D.d0, (size_t)8*R*sizeof(float)));
    CUCK(cudaMalloc(&D.d1, (size_t)8*R*sizeof(float)));
    CUCK(cudaMemcpy(D.wu, S.Wu.data(), S.wbytes, cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(D.wg, S.Wg.data(), S.wbytes, cudaMemcpyHostToDevice));
    CUCK(cudaMemcpy(D.y,  S.Y.data(),  S.Y.size(), cudaMemcpyHostToDevice));

    // every tile height the launcher offers, both VDR settings
    sweep_plain<type,  1, 2>(S, D);
    sweep_plain<type,  2, 2>(S, D);
    sweep_plain<type,  4, 2>(S, D);
    sweep_plain<type,  8, 2>(S, D);
    sweep_plain<type, 16, 2>(S, D);
    sweep_plain<type,  4, 4>(S, D);
    // the fused twin at the default tile
    sweep_fused<type,  4, 2>(S, D);

    // tile invariance at the widths PXA_PXQ_MMVQ_COLS widens, against the tile that ships today.
    // Every distinct height the ceiling in mmvq_rows_per_block leaves reachable above two
    // columns: 4 and 8 at four columns, 4 at eight, on the plain kernel and on the up/gate twin,
    // and the wide one at both code-word settings.
    case_tile<type, 4, 4, 4, 2, false>(S, D);
    case_tile<type, 4, 4, 8, 2, false>(S, D);
    case_tile<type, 4, 4, 8, 4, false>(S, D);
    case_tile<type, 8, 2, 4, 2, false>(S, D);
    case_tile<type, 4, 4, 4, 2, true >(S, D);
    case_tile<type, 4, 4, 8, 2, true >(S, D);
    case_tile<type, 8, 2, 4, 2, true >(S, D);

    CUCK(cudaFree(D.wu)); CUCK(cudaFree(D.wg)); CUCK(cudaFree(D.y));
    CUCK(cudaFree(D.d0)); CUCK(cudaFree(D.d1));
}

// ---------------------------------------------------------------------------------------------
// device vs host, exhaustively: the PXQ2/PXQ3 group decode over every bit pattern either format
// admits. The host arm here is the same source with PRMT emulated, and test-pxq23-mmvq-snap.cpp
// has already checked that arm against the format for all of these patterns -- so agreement here
// carries the format proof onto the device's real PRMT.
// ---------------------------------------------------------------------------------------------
__global__ static void k_decode2(uint32_t n, int2 * __restrict__ out) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const pxq_mmvq_g8 g = pxq2_mmvq_gather8(i);              // i == the group's 16 code bits
    out[i] = make_int2(g.x, g.y);
}

__global__ static void k_decode3(uint32_t n, int2 * __restrict__ out) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const pxq_mmvq_g8 g = pxq3_mmvq_gather8(i & 0xFFFFu, i >> 16);   // low plane, high plane
    out[i] = make_int2(g.x, g.y);
}

static void decode_exhaustive(bool p3) {
    const uint32_t n = p3 ? (1u << 24) : (1u << 16);
    const char * name = p3 ? "PXQ3" : "PXQ2";

    int2 * d = nullptr;
    CUCK(cudaMalloc(&d, (size_t)n*sizeof(int2)));
    const int threads = 256;
    if (p3) k_decode3<<<(n + threads - 1)/threads, threads>>>(n, d);
    else    k_decode2<<<(n + threads - 1)/threads, threads>>>(n, d);
    CUCK(cudaGetLastError());
    CUCK(cudaDeviceSynchronize());

    std::vector<int2> h((size_t)n);
    CUCK(cudaMemcpy(h.data(), d, (size_t)n*sizeof(int2), cudaMemcpyDeviceToHost));
    CUCK(cudaFree(d));

    ++g_cases;
    int bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const pxq_mmvq_g8 g = p3 ? pxq3_mmvq_gather8(i & 0xFFFFu, i >> 16) : pxq2_mmvq_gather8(i);
        if (g.x != h[i].x || g.y != h[i].y) {
            if (bad < 4) {
                printf("      pattern 0x%08x: device {0x%08x,0x%08x} host {0x%08x,0x%08x}\n",
                       i, (unsigned)h[i].x, (unsigned)h[i].y, (unsigned)g.x, (unsigned)g.y);
            }
            ++bad;
        }
    }
    g_vals += 8LL*n;
    if (bad) { ++g_fail; printf("  FAIL %s device decode: %d/%u patterns differ\n", name, bad, n); }
    else     { printf("  %s device decode: %u patterns (%u values) bit-identical to the host arm\n",
                      name, n, 8*n); }
}

int main(int argc, char ** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);
    CUCK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUCK(cudaGetDeviceProperties(&prop, dev));
    printf("device %d: %s (cc %d.%d)\n", dev, prop.name, prop.major, prop.minor);
    printf("PXQ MMVQ column-reuse differential: HOIST=false (pre-2026-09-08) vs HOIST=true, "
           "bitwise\n");

    struct shp { int R, K; };
    const shp shapes[] = { { 64, 128 }, { 128, 256 }, { 256, 512 }, { 128, 32 }, { 64, 1024 } };

    uint64_t seed = 0x243f6a8885a308d3ull;
    for (const shp & s : shapes) {
        printf("shape R=%d K=%d\n", s.R, s.K);
        run_type<GGML_TYPE_PXQ4  >(s.R, s.K, seed);
        run_type<GGML_TYPE_PXQ4HQ>(s.R, s.K, seed);
        run_type<GGML_TYPE_PXQ2  >(s.R, s.K, seed);
        run_type<GGML_TYPE_PXQ3  >(s.R, s.K, seed);
    }

    printf("\nPXQ2/PXQ3 group decode, device PRMT vs the host arm, every bit pattern:\n");
    decode_exhaustive(false);
    decode_exhaustive(true);

    printf("\n%d cases, %lld values compared, %d failed\n", g_cases, g_vals, g_fail);
    if (g_cases == 0 || g_vals == 0) { printf("NOTHING COMPARED -- harness defect\n"); return 3; }
    printf(g_fail ? "FAIL\n" : "PASS (bit-identical)\n");
    return g_fail ? 1 : 0;
}
