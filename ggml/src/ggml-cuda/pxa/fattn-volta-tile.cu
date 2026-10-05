#include "fattn-volta-tile.cuh"
#include "volta-tile/fattn-tile-ml.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// -----------------------------------------------------------------------------
// PXA_FA_D512_VOLTA=2 (2026-09-20) -- the second candidate for a 512-wide head on
// sm_70, and the schedule upstream itself selects for this head size at decode on
// this architecture.
//
// WHY A SECOND ONE. The MMA kernel next door compiles itself out below 32 columns
// per block on sm_70, so at a GQA packing of 8 the smallest tile it can run is
// four query columns -- a decode step computes four tokens' worth of arithmetic
// for one token. This kernel has no such floor: its 512/512 config row runs at
// eight columns per block with two blocks per multiprocessor and a 64-row KV
// batch, which is the minimum work for one token with eight query heads packed
// onto one KV head.
//
// SCOPE, and it declines everything else: cc 7.0, DKQ == DV == 512, f16 K and V,
// a mask with ne2 == ne3 == 1, no attention sinks, no ALiBi, KV padded to 256,
// and a GQA ratio that is a multiple of 8 (the 512 family of this kernel is only
// implemented with the GQA optimization). Default off.
// -----------------------------------------------------------------------------

static std::atomic<long> g_pxa_d512_tile_calls{0};

// Query batch width of every node the kernel served, one bucket per width. The cut that keeps wide
// batches on the unfused chain (PXA_FA_D512_FUSED_MAXCOLS) lives in the graph builder, so this is
// the only place that can say which widths actually reached the kernel: a width present here took
// the kernel, a width absent from an otherwise-busy run took the chain. Widths above the last
// bucket are counted in it; ne1 is a ubatch width and 512 is the usual ceiling.
static constexpr int PXA_D512_NE1_BUCKETS = 513;
static std::atomic<long> g_pxa_d512_tile_ne1[PXA_D512_NE1_BUCKETS];

static void pxa_d512_tile_report(void) {
    const long n = g_pxa_d512_tile_calls.load();
    fprintf(stderr, "PXA_FA_D512_VOLTA=2: %ld node(s) served by the 512/512 tile kernel\n", n);
    if (n == 0) {
        fprintf(stderr, "PXA_FA_D512_VOLTA=2: armed but never engaged -- every 512/512 node was "
                        "declined or kept on the unfused chain\n");
        return;
    }
    fprintf(stderr, "PXA_FA_D512_VOLTA=2: by query batch width:");
    for (int w = 0; w < PXA_D512_NE1_BUCKETS; ++w) {
        const long c = g_pxa_d512_tile_ne1[w].load();
        if (c) {
            fprintf(stderr, " %s%d=%ld", w == PXA_D512_NE1_BUCKETS - 1 ? ">=" : "", w, c);
        }
    }
    fprintf(stderr, "\n");
}

void ggml_cuda_fattn_volta_tile_d512_arm_report(void) {
    static const bool once = [] {
        atexit(pxa_d512_tile_report);
        return true;
    }();
    (void) once;
}

bool ggml_cuda_fattn_volta_tile_d512_supported(const ggml_tensor * dst, int cc) {
    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V) {
        return false;
    }
    if (Q->ne[0] != 512 || K->ne[0] != 512 || V->ne[0] != 512) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }
    if (!mask || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (dst->src[4]) {
        return false; // attention sinks: not exercised, do not claim it
    }
    if (K->ne[1] % 256 != 0) {
        return false; // the vendored launcher's FATTN_KQ_STRIDE padding
    }
    if (Q->ne[3] != 1) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0 || (Q->ne[2] / K->ne[2]) % 8 != 0) {
        return false; // at DV > 256 only the GQA-optimized variant is implemented
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return false; // ALiBi turns the GQA optimization off, which this family needs
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    return true;
}

void ggml_cuda_flash_attn_ext_volta_tile_d512(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t ne1 = dst->src[0]->ne[1];
    g_pxa_d512_tile_ne1[ne1 < PXA_D512_NE1_BUCKETS ? ne1 : PXA_D512_NE1_BUCKETS - 1].fetch_add(1);

    if (g_pxa_d512_tile_calls.fetch_add(1) == 0) {
        const ggml_tensor * Q = dst->src[0];
        fprintf(stderr, "PXA_FA_D512_VOLTA=2: engaged (sm_70 head 512/512 -> vendored tile kernel; "
                        "first node ne1=%d heads=%d/%d)\n",
                (int) Q->ne[1], (int) Q->ne[2], (int) dst->src[1]->ne[2]);
        ggml_cuda_fattn_volta_tile_d512_arm_report(); // harmless if the lever already armed it
    }
    pxa_volta_fa::ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
}

// -----------------------------------------------------------------------------
// PXA_FA_D256_VOLTA_TILE (2026-09-21) -- head 256/256 at SPECULATIVE VERIFY width
// on sm_70, on the same vendored kernel.
//
// WHY. At head 256 with an f16 cache this fork serves query width 1 from the
// vector kernel, and that kernel is NO_DEVICE_CODE above one column on CUDA. A
// verify step is two to four columns wide, so it falls through to the legacy
// nvcuda::wmma tile -- which on this architecture spills to local memory and
// fits two blocks per multiprocessor. Measured on one V100, one FLASH_ATTN_EXT
// node, 12 q-heads / 2 kv-heads, f16 K and V, microseconds per node:
//
//   n_kv    width 2          width 4
//   16384   388 (wmma)       387 (wmma)
//   32768   754 (wmma)       754 (wmma)
//
// against 89 / 122 and 161 / 223 for the very kernel vendored in this directory,
// which upstream selects for exactly this shape (its rule sends
// ne1 * gqa_ratio <= 16 to the tile kernel, and a width-4 verify at GQA 6 is 24
// -- but at ncols2 = 2 the schedule below still runs it with four real query
// columns per block). That is a factor of 4.4 at 16k on a node a decode step
// pays sixteen times, once per full-attention layer.
//
// SCOPE, and it declines everything else: cc 7.0, DKQ == DV == 256, f16 K and V,
// a mask with ne2 == ne3 == 1, no attention sinks, no ALiBi, no logit softcap,
// KV padded to 256 (which -fa guarantees), and an even GQA ratio so the vendored
// selector takes a packed (ncols2 >= 2) schedule rather than the unpacked one.
// The query-width window and the KV floor are the lever's business, not the
// kernel's, and live in the route planner.
// -----------------------------------------------------------------------------

static std::atomic<long> g_pxa_d256_tile_calls{0};

// Same instrument as the 512 path above: one bucket per query batch width, so a run can say which
// widths actually reached the kernel rather than which ones the lever would have allowed.
static constexpr int PXA_D256_NE1_BUCKETS = 33;
static std::atomic<long> g_pxa_d256_tile_ne1[PXA_D256_NE1_BUCKETS];

static void pxa_d256_tile_report(void) {
    const long n = g_pxa_d256_tile_calls.load();
    fprintf(stderr, "PXA_FA_D256_VOLTA_TILE: %ld node(s) served by the 256/256 tile kernel\n", n);
    if (n == 0) {
        fprintf(stderr, "PXA_FA_D256_VOLTA_TILE: armed but never engaged -- every 256/256 node was "
                        "declined, below the KV floor, or outside the query-width window\n");
        return;
    }
    fprintf(stderr, "PXA_FA_D256_VOLTA_TILE: by query batch width:");
    for (int w = 0; w < PXA_D256_NE1_BUCKETS; ++w) {
        const long c = g_pxa_d256_tile_ne1[w].load();
        if (c) {
            fprintf(stderr, " %s%d=%ld", w == PXA_D256_NE1_BUCKETS - 1 ? ">=" : "", w, c);
        }
    }
    fprintf(stderr, "\n");
}

void ggml_cuda_fattn_volta_tile_d256_arm_report(void) {
    static const bool once = [] {
        atexit(pxa_d256_tile_report);
        return true;
    }();
    (void) once;
}

bool ggml_cuda_fattn_volta_tile_d256_supported(const ggml_tensor * dst, int cc) {
    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }
    if (!mask || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (dst->src[4]) {
        return false; // attention sinks: not exercised, do not claim it
    }
    if (K->ne[1] % 256 != 0) {
        return false; // the vendored launcher's FATTN_KQ_STRIDE padding
    }
    if (Q->ne[3] != 1) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }

    // The vendored ncols2 selector: an even GQA ratio takes a packed schedule, an odd one falls to
    // the unpacked ncols2 == 1 variant, which is a different kernel shape and a different question.
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio % 2 != 0) {
        return false;
    }
    // ... and the selector withdraws the packed schedule above a query width of 16 when the GQA
    // ratio is 4 or less (its `gqa_limit`), which would silently drop this node to ncols2 == 1.
    if (gqa_ratio <= 4 && Q->ne[1] > 16) {
        return false;
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return false; // ALiBi turns the GQA packing off, and that shape is unmeasured here
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false; // the kernel implements it at DV == 256; this fork has not measured it
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    return true;
}

void ggml_cuda_flash_attn_ext_volta_tile_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t ne1 = dst->src[0]->ne[1];
    g_pxa_d256_tile_ne1[ne1 < PXA_D256_NE1_BUCKETS ? ne1 : PXA_D256_NE1_BUCKETS - 1].fetch_add(1);

    if (g_pxa_d256_tile_calls.fetch_add(1) == 0) {
        const ggml_tensor * Q = dst->src[0];
        fprintf(stderr, "PXA_FA_D256_VOLTA_TILE: engaged (sm_70 head 256/256 -> vendored tile kernel; "
                        "first node ne1=%d heads=%d/%d n_kv=%d)\n",
                (int) Q->ne[1], (int) Q->ne[2], (int) dst->src[1]->ne[2], (int) dst->src[1]->ne[1]);
        ggml_cuda_fattn_volta_tile_d256_arm_report(); // harmless if the lever already armed it
    }
    pxa_volta_fa::ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
}
