#include "fattn-volta-mma.cuh"
#include "volta-mma/fattn-mma-ml.cuh"
#include "core/levers.cuh"
#include "../convert.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// PXA_FA_MMA_VOLTA (2026-09-02) -- sm_70 flash-attention on mainline llama.cpp's
// MMA kernel, vendored under volta-mma/ (see the provenance headers there).
//
// WHY: nvprof on the 2xV100 pair, Qwable-27B-PXQ4 at a 20801-token prefill,
// showed both engines issuing the SAME 352 flash-attention calls -- 25.17 s here
// on this fork's legacy nvcuda::wmma kernel against 9.22 s in mainline, a 2.73x
// deficit that accounted for the entire GPU-compute gap (61.1 s vs 45.8 s).
// mainline's Volta kernel uses the m8n8k4 tensor-core instruction directly and,
// via ncols2, reads each K/V row once for several of the Q heads that share it.
//
// SCOPE:
//   * cc == 7.0 only. Every other arch keeps its existing dispatch untouched.
//   * head sizes 128 and 256 (DKQ == DV). 128 covers the MoE and P100-seat
//     models if they are ever run on Volta; 256 is this rig's model.
//   * f16 K and V, plus a MATCHED q8_0 K/V pair at head size 256 on the prefill
//     widths -- see PXA_FA_MMA_VOLTA_Q8 below. mainline's launch_fattn staged a
//     quantized cache into scratch taken from a backend hook this fork does not
//     have; the vendored launcher now takes that scratch from the CUDA pool
//     instead, so the f16-only restriction is a choice rather than a limit.
//   * batch > 8 only; decode keeps the incumbent path until measured.
// Anything declined falls through to the tile route, which is itself a large win
// over the WMMA kernel, so this can only change which working kernel runs.

// mainline fattn.cu:183-194. ncols2 > 1 packs several Q heads into one K/V read
// and is only valid with a mask, no ALiBi, K padded to FATTN_KQ_STRIDE, and
// 16-byte-aligned higher-dimensional strides. Otherwise ncols2 must stay 1.
static bool pxa_volta_mma_use_gqa_opt(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % 256 == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }
    return use_gqa_opt;
}

// PXA_FA_MMA_VOLTA_Q8 (2026-09-21, default ON) -- a matched q8_0 K/V cache at head size 256 also
// reaches this kernel, on the prefill widths only.
//
// WHY IT IS WORTH DOING: q8_0 K/V is what buys context on a 16 GB card, and it was exactly the
// configuration this kernel refused. A q8_0 cache fell all the way through to the legacy WMMA
// kernel -- the kernel this file exists to get away from -- so the card paid the 2.7x attention
// deficit precisely when the user had asked for the long context that makes attention dominate.
// Both routes pay the SAME whole-tensor f16 conversion, so admitting the cache changes which
// attention kernel runs and nothing else about the work done.
//
// WHY THE SCRATCH IS THE INTERESTING PART: the conversion writes a full f16 copy of that layer's K
// and V, held for the duration of the call. On a 16 GB card at a context large enough to have
// wanted q8_0 in the first place, that copy is the difference between booting and not. It comes
// from the CUDA pool, and the pool will grow the allocation if it can -- so the admission asks
// first, below, and declines when the copy would not fit with room to spare. Declining is free:
// this whole route is a dispatch-time SUBSTITUTION, and every shape it turns down is served by the
// route the support query already named.
//
// NUMERICS: the conversion is the backend's own to_fp16 for q8_0, i.e. bit for bit what the WMMA
// route already feeds itself. What changes is the attention kernel, and a different kernel combines
// softmax partials in a different order -- an fp32-round-off-class change, not a bit-identical one.
static bool pxa_fa_mma_volta_q8() {
    return pxa_lever(PXA_LEVER_FA_MMA_VOLTA_Q8) != 0;
}

// Would the f16 staging for this node fit? The pool hands back what it already holds, so the only
// question that costs anything is the FIRST request at a given size on a given device; after that
// the high-water mark answers it. The margin is deliberately generous: this runs while the rest of
// the graph is still allocating, and a route that fits by a hair is a route that fails later.
static constexpr size_t PXA_VOLTA_Q8_SCRATCH_MARGIN = 256ull << 20; // 256 MiB

static bool pxa_volta_mma_q8_scratch_fits(const ggml_tensor * K, const ggml_tensor * V) {
    const bool V_is_K_view = V->view_src && (V->view_src == K ||
                                             (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need = ggml_nelements(K)*sizeof(half);
    if (!V_is_K_view) {
        need += ggml_nelements(V)*sizeof(half);
    }

    const int dev = ggml_cuda_get_device();
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    static std::atomic<size_t> staged_hwm[GGML_CUDA_MAX_DEVICES];
    if (need <= staged_hwm[dev].load(std::memory_order_relaxed)) {
        return true; // the pool is already holding at least this much for this route
    }

    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (need + PXA_VOLTA_Q8_SCRATCH_MARGIN > free_bytes) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            fprintf(stderr, "PXA_FA_MMA_VOLTA_Q8: declining the q8_0 K/V cache on device %d -- the f16 staging "
                            "needs %.0f MiB and only %.0f MiB is free; attention keeps its previous route\n",
                    dev, need/1048576.0, free_bytes/1048576.0);
        }
        return false;
    }

    // Publish the new high-water mark; a concurrent larger value must win.
    size_t seen = staged_hwm[dev].load(std::memory_order_relaxed);
    while (seen < need && !staged_hwm[dev].compare_exchange_weak(seen, need, std::memory_order_relaxed)) {
    }
    return true;
}

bool ggml_cuda_fattn_volta_mma_supported(const ggml_tensor * dst, int cc) {
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
    const int64_t DKQ = Q->ne[0];
    if (DKQ != 128 && DKQ != 256) {
        return false; // only these two families are instantiated
    }
    if (K->ne[0] != DKQ || V->ne[0] != DKQ) {
        return false; // DKQ == DV only
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        // PXA_FA_MMA_VOLTA_Q8: the one quantized shape that has been measured here. Matched q8_0
        // K and V, head size 256, prefill widths only. Everything else keeps the old refusal --
        // a mixed pair is declined rather than guessed at, because the launcher converts K and V
        // through the same code path and aliases V onto K when V is a view of it.
        if (!pxa_fa_mma_volta_q8()) {
            return false; // see SCOPE above
        }
        if (DKQ != 256 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0) {
            return false;
        }
        if (Q->ne[1] <= 8) {
            return false; // decode is not measured on this route and the staging is not worth it there
        }
        // Require the converter the launcher will actually pick, rather than finding out inside
        // the launch. Contiguous and strided are two different functions.
        if (ggml_is_contiguously_allocated(K) ? (ggml_get_to_fp16_cuda(K->type)    == nullptr)
                                              : (ggml_get_to_fp16_nc_cuda(K->type) == nullptr)) {
            return false;
        }
        if (ggml_is_contiguously_allocated(V) ? (ggml_get_to_fp16_cuda(V->type)    == nullptr)
                                              : (ggml_get_to_fp16_nc_cuda(V->type) == nullptr)) {
            return false;
        }
        if (!pxa_volta_mma_q8_scratch_fits(K, V)) {
            return false;
        }
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            fprintf(stderr, "PXA_FA_MMA_VOLTA_Q8: engaged (sm_70 head-256 prefill with a q8_0 K/V cache -> vendored "
                            "m8n8k4 MMA kernel, staged to f16 from the CUDA pool; not bit-identical to the previous "
                            "route -- override PXA_FA_MMA_VOLTA_Q8=0)\n");
        }
    }
    if (!mask) {
        return false; // the vendored kernel's KV_max bound reads the mask
    }
    if (mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false; // mainline rejects these in ggml_cuda_get_best_fattn_kernel
    }
    if (dst->src[4]) {
        return false; // attention sinks: not exercised here, do not claim it
    }
    if (K->ne[1] % 256 != 0) {
        return false; // FATTN_KQ_STRIDE padding the vendored launcher assumes
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    // NB: the batch-size gate lives in pxa_fa_mma_volta_take() in fattn.cu, so
    // that PXA_FA_MMA_VOLTA=2 can lift it for a decode measurement without this
    // structural predicate having to know about the lever.
    return true;
}

// mainline's ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1, specialised for
// Volta: turing_mma_available() is false there, so the 8/ncols2 rung is skipped.
template <int DKQ, int DV, int ncols2>
static void pxa_volta_mma_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] <= 16/ncols2) {
        pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }
    if (Q->ne[1] <= 32/ncols2) {
        pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }
    pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

// mainline's switch_ncols2, Volta branch (fattn.cu:200-222). Upstream comment:
// "On Volta the GQA optimizations aren't as impactful vs. minimizing wasted
// compute", hence the descending 8/4/2/1 ladder rather than a fixed ncols2.
template <int DKQ, int DV>
static void pxa_volta_mma_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const bool use_gqa_opt = pxa_volta_mma_use_gqa_opt(dst);
    const int  gqa_ratio   = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        pxa_volta_mma_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        pxa_volta_mma_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        pxa_volta_mma_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }
    pxa_volta_mma_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

void ggml_cuda_flash_attn_ext_volta_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[0]) {
        case 128: pxa_volta_mma_switch_ncols2<128, 128>(ctx, dst); return;
        case 256: pxa_volta_mma_switch_ncols2<256, 256>(ctx, dst); return;
        default:  GGML_ABORT("PXA: unreachable, guarded by ggml_cuda_fattn_volta_mma_supported");
    }
}

// -----------------------------------------------------------------------------
// PXA_FA_D512_VOLTA (2026-09-20) -- head size 512/512 on sm_70.
//
// WHY THIS EXISTS. A 512-wide head has no fused CUDA kernel on sm_70 in this
// tree: the vector kernels stop at 256, the WMMA support table lists
// 64/80/96/112/128/256, and the predicate above refuses any DKQ that is not 128
// or 256. ggml_cuda_fattn_is_supported() therefore declines the node, and the
// graph builder replaces it with the unfused KQ/soft_max/KQV chain on the card
// (PXA_FA_GPU_FALLBACK). That chain reads K and V once each AND writes and
// re-reads an f32 score matrix, in three launches instead of one, and its cost
// grows with the whole context -- which is exactly what a full-attention layer
// sees.
//
// The vendored kernel under volta-mma/ already carries the complete 512/512
// family: a Volta config row at every column count, the mask-driven KV_max
// bound, and the extern declarations. Nothing was missing but the instantiation
// and the two gates.
//
// THE COLUMN FLOOR. On sm_70 the vendored kernel compiles itself out below 32
// columns per block. With ncols2 = 8 (the GQA packing this shape wants) that
// makes the 16-column rung a stub, so the ladder here starts at ncols1 = 4.
// A decode step (one token) therefore computes a 4-token tile and throws three
// quarters of the arithmetic away -- the K and V traffic, which is what decode
// is bound by, is unchanged.
//
// SCOPE, and it declines everything else: cc 7.0, DKQ == DV == 512, f16 K and V,
// a mask with ne2 == ne3 == 1, no attention sinks, no ALiBi, KV padded to 256,
// and a GQA ratio that is a multiple of 8. Which value is the default is decided in fattn.cu.
// -----------------------------------------------------------------------------

static std::atomic<long> g_pxa_d512_volta_calls{0};

static void pxa_d512_mma_report(void) {
    const long n = g_pxa_d512_volta_calls.load();
    fprintf(stderr, "PXA_FA_D512_VOLTA=1: %ld node(s) served by the 512/512 MMA kernel\n", n);
    if (n == 0) {
        fprintf(stderr, "PXA_FA_D512_VOLTA=1: armed but never engaged -- every 512/512 node was "
                        "declined or kept on the unfused chain\n");
    }
}

// Called once when the lever selects this route, whether or not a node ever reaches it, so that
// the exit report exists on an armed run that served nothing and can therefore say zero.
void ggml_cuda_fattn_volta_mma_d512_arm_report(void) {
    static const bool once = [] {
        atexit(pxa_d512_mma_report);
        return true;
    }();
    (void) once;
}

bool ggml_cuda_fattn_volta_mma_d512_supported(const ggml_tensor * dst, int cc) {
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
        return false; // this is the 512/512 family and nothing else
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false; // the vendored launcher has no quantized-KV staging here
    }
    if (!mask) {
        return false; // the KV_max bound reads the mask
    }
    if (mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (dst->src[4]) {
        return false; // attention sinks: not exercised, do not claim it
    }
    if (K->ne[1] % 256 != 0) {
        return false; // FATTN_KQ_STRIDE padding the vendored launcher assumes
    }
    if (Q->ne[3] != 1) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    // ncols2 == 8 is the only packing instantiated for this family, and the
    // 512/512 rows of the kernel are implemented only with the GQA optimization.
    if ((Q->ne[2] / K->ne[2]) % 8 != 0) {
        return false;
    }
    if (!pxa_volta_mma_use_gqa_opt(dst)) {
        return false;
    }
    return true;
}

// The 512 ladder. The 16-column rung of pxa_volta_mma_switch_ncols1() is a stub
// on sm_70 for ncols2 == 8, so it is not used here.
void ggml_cuda_flash_attn_ext_volta_mma_d512(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (g_pxa_d512_volta_calls.fetch_add(1) == 0) {
        fprintf(stderr, "PXA_FA_D512_VOLTA: engaged (sm_70 head 512/512 -> vendored m8n8k4 MMA kernel, "
                        "ncols2=8; first node ne1=%d heads=%d/%d)\n",
                (int) Q->ne[1], (int) Q->ne[2], (int) dst->src[1]->ne[2]);
        ggml_cuda_fattn_volta_mma_d512_arm_report(); // harmless if the lever already armed it
    }

    if (Q->ne[1] <= 4) {
        pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 4, 8>(ctx, dst);
        return;
    }
    pxa_volta_fa::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 8, 8>(ctx, dst);
}
