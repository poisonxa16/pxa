//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fa-route.cuh"
#include "levers.cuh"

#include "../../fattn-tile-f16.cuh"
#include "../../fattn-tile-f32.cuh"
#include "../../fattn-vec-f16-interface.cuh"
#include "../../fattn-vec-f32-interface.cuh"
#include "../../fattn-wmma-f16-interface.cuh"
#include "../../fattn-mma-f16-interface.cuh"
#include "../../fattn-new-mma.cuh"
#include "../../dsa_attn.cuh"
#include "../fattn-volta-mma.cuh"
#include "../fattn-volta-tile.cuh"
#include "../fattn-tile-v2.cuh"
#include "../fattn-tile-big.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define FATTN_KQ_STRIDE 256

// =============================================================================================
// The levers. Every PXA_* switch that decides a flash-attention route is DECLARED in levers.cu
// (the registry) and read here through pxa_lever(); ggml-cuda/fattn.cu, which is an
// upstream-shaped file, no longer reads any of them, and neither does this file call getenv().
// The comment blocks below came with the levers and stay with the code that uses them -- they are
// the record of why each default is what it is; the registry row carries the one-line version a
// user is shown, and the PXA_AUTO report is generated from it.
// =============================================================================================


static inline bool mma_better_than_turing(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) > CC_TURING;
}

// PXQ port of upstream ik_llama.cpp PR #2144 (merged 2026-07-17):
// on sm_60 (P100/CC_PASCAL) the fp16 FA vec kernel accumulates the online-softmax
// denominator and the P.V product in fp16, flipping ~3-4% of decode top-1 tokens vs an
// all-fp32 reference. P100 decode is memory-bandwidth-bound, so routing decode
// (batch <= 8) to the fp32 vec kernel is measured-free upstream (tg128 96.79 vs 96.61
// t/s; neutral-or-better at long context). Prefill and the D=256 vec path stay on
// vec_f16. sm_61 (1080Ti) is unaffected (no fast fp16 -> already vec_f32).
// Env kill-switch: PXQ_SM60_FA_VEC_F32=0 restores the old fp16-accumulating route.
static bool pxq_sm60_fa_vec_f32_enabled() {
    return pxa_lever(PXA_LEVER_SM60_FA_VEC_F32) != 0;
}

static inline bool pxq_use_sm60_vec_f32(const int cc, const ggml_tensor * Q) {
    return cc == CC_PASCAL && Q->ne[1] <= 8 && pxq_sm60_fa_vec_f32_enabled();
}

// PXA_FA_TILE256 (2026-08-03): on pre-Volta cards the D=256 head had NO tile/mma prefill kernel,
// so `Q->ne[0] == 256` forced the single-column VEC kernel at ANY batch size. Profiled on the
// 122B-A10B (qwen35moe, head 256) 4xP100 rig at 8881-token fill: flash_attn_vec_ext_f16 was 52.7%
// of prefill GPU time (60 launches x 281 ms avg) — every query column re-streams the whole KV
// extent with no tile reuse. The D=256 tile-f16 (ncols=16) restores KQ-tile data reuse.
// Default ON. PXA_FA_TILE256=0 routes D=256 prefill back to the vec-f16 kernel: that is a
// KERNEL-SELECTION rollback, NOT a return to correct arithmetic. NP-DET-2026-09-09 measured the
// vec route returning the empty answer at the aligned offset on this same seat and fill -- both
// routes carry half accumulators, so the rollback swaps one too-coarse rounding realisation for
// another. The lever that fixes it is PXA_FA_TILE_F32ACC (see below), and it does not exist on
// the vec path. Decode (ne1 <= 8) is untouched either way.
// PXA_FA_TILE_VOLTA — sm_70 flash-attention kernel choice.
//
// DEFAULT OFF -- DO NOT TURN THIS ON. It was briefly defaulted to AUTO on the strength
// of its prefill numbers alone; a greedy-32 parity capture then showed the tile route
// EMITS CORRUPT OUTPUT on sm_70. Same binary, same prompts, temp 0, first 32 tokens:
//     base (WMMA route)  @3121  sha ee47a1641232110f  " memory memory stream memory..."
//     base (WMMA route)  @20801 sha 6d01ae25ffc202f6  " memory distributed memory..."
//     tile route         @3121  sha 539b277a38895ea1  "!!!!!!!!!!!!!!!!"
//     tile route         @20801 sha 539b277a38895ea1  "!!!!!!!!!!!!!!!!"
// Identical sha at two different prompt lengths, and the "!!!!" degenerate token run, are
// the signature of non-finite logits -- not a numerics difference. The speed was real
// (863.02 / 693.01 t/s against 804.78 / 621.56) which is exactly why speed alone is not a
// ship gate. The MMA route in fattn-volta-mma.cu is both FASTER (877.63 / 769.89) and
// parity-clean, so nothing is lost by leaving this off.
//
// NOTE -- RESOLVED; the capture this asked for was taken. NP-DET-2026-09-09 ran the head-256
// seat and found the sm_60 D=256 tile-f16 PREFILL path returns the right token with
// PXA_FA_TILE_F32ACC ON (the shipped default): what was too coarse was the fp16 accumulation,
// not the tile instance as such, and the vec route PXA_FA_TILE256=0 falls back to is wrong at
// the same fill. The two greedy captures of 2026-09-02 do not bear on this -- the first never
// started its containers (docker bind failure, so no arm ran) and neither recorded the model or
// the head-dim, so identical shas across their arms cannot be told apart from an inert lever.
// PXA_FA_MASK_SKIP_TILE, auto-armed in those runs, still has its own A/B outstanding -- see its
// HONESTY GATE in ggml-cuda/pxa/pxa-enhance.cuh.
//
// PXA_FA_TILE_VOLTA=1 or =2 still routes sm_70 to the tile kernel for A/B work.
//
// WHY THE FLIP. nvprof on 2xV100, Qwable-27B-PXQ4 (head dim 256, GQA 24/4, 16 attention
// layers), per-kernel GPU time with memcpy excluded: flash-attention is 9.9% of prefill
// compute at 3121 tokens but 41.2% at 20801, and the kernel serving it is
// flash_attn_ext_f16<256,256,cols_per_block=32,nwarps=4,...>, the legacy nvcuda::wmma
// tile. ptxas -v for that instance on sm_70: 255 registers, 2488 B stack frame,
// 2924 B spill stores / 4424 B spill loads, 33808 B smem — 2 blocks/SM and a local-memory
// round trip per K tile. Measured 25.17 s of GPU time over 352 calls at 20801 against
// 9.22 s for mainline llama.cpp's Volta kernel on exactly the same 352 calls (2.73x).
// The tile-f16 kernel has no wmma fragments to spill and carries a D=256 ncols=16 variant
// (PXA_FA_TILE256, added 2026-08-03), so on Volta it beats the WMMA tile outright:
//
//   2xV100, Qwable-27B-PXQ4, -b 2048 -ub 2048 -fa on, n=5 medians, spread <=0.30%:
//     prefill @3121   838.00 -> 863.32 t/s  (+3.0%)
//     prefill @20801  586.74 -> 681.40 t/s  (+16.1%)
//
// The gain tracks the attention share exactly, which is the signature of a real attention
// fix rather than a measurement artifact. Decode is EXCLUDED from the auto flip: at
// Q->ne[1] <= 8 the tile kernel would waste 16 query columns on 1, and decode is a
// different (bandwidth-bound) regime that this measurement does not cover.
//
//   unset / auto : tile for Q->ne[1] > 8 on sm_70, WMMA for decode   (default)
//   =1           : tile for EVERY batch size on sm_70 (the old A/B arm, incl. decode)
//   =0           : WMMA everywhere on sm_70 (the pre-2026-09-02 default)
//
// If tile cannot handle the shape the dispatcher falls through to the normal selector, so
// this can only ever change WHICH working kernel runs, never whether one runs at all.
// PXA_FA_MMA_VOLTA (2026-09-02) — sm_70 large-batch flash-attention on the
// vendored mainline MMA kernel (mma.sync.m8n8k4 + ncols2 GQA packing); see
// fattn-volta-mma.cu for the measurement that motivated it and for the scope of
// the shapes it accepts. Default ON at cc 7.0 for Q->ne[1] > 8; the tile route
// below stays as the fallback for every shape it declines, and decode is
// untouched. PXA_FA_MMA_VOLTA=0 disables it (tile route takes over),
// PXA_FA_MMA_VOLTA=1 is the default made explicit.
// PXA_FA_D512_VOLTA (2026-09-20) -- see the block comment in pxa/fattn-volta-mma.cu. DEFAULT 2:
// unset means the tile kernel, which on Gemma 4 26B-A4B turns the one decode cell this engine lost
// (87.2 tok/s against 93.8 at a 6.4k-token context, two V100s) into 103.0, and 72.7 into 98.2 at
// 12.7k. What it serves is narrowed in the graph builder, which marks the nodes it may take (see
// pxa_fa_d512_volta_takes below): the one-column decode step only, contexts of 1,280 cells and up
// only, the Gemma 4 family only unless the lever is set by hand, every layer offloaded, and only
// when every card of the context has the kernel (PXA_FA_D512_FUSED_MAXCOLS / PXA_FA_D512_FUSED_MINKV in src/llama-build-context.cpp say
// why). PXA_FA_D512_VOLTA=0 restores the unfused chain everywhere, byte for byte.
// One truthful banner line is printed by the kernel wrapper the first time it actually serves a
// node; the exit report is armed only when the lever was SET, so a default boot of a model with no
// 512-wide head says nothing.
//   0 : off - the unfused attention chain
//   1 : the vendored m8n8k4 MMA kernel (four query columns per block minimum)
//   2 : the vendored no-tensor-core tile kernel (eight columns per block), which is the
//       schedule upstream itself selects for this head size at decode on this architecture
static int pxa_fa_d512_volta_mode() {
    static const int mode = [] {
        const int m = (int) pxa_lever(PXA_LEVER_FA_D512_VOLTA);
        // Arm the exit report as soon as the lever is set by hand, so a run that arms it and then
        // serves nothing -- a model with no 512-wide head, a card that is not Volta, a shape the
        // kernel declines -- says so with a zero instead of saying nothing at all.
        if (m && pxa_lever_set_by_user(PXA_LEVER_FA_D512_VOLTA)) {
            if (m == 1) {
                ggml_cuda_fattn_volta_mma_d512_arm_report();
            } else {
                ggml_cuda_fattn_volta_tile_d512_arm_report();
            }
        }
        return m;
    }();
    return mode;
}

// True when SOME 512/512 route is both armed and structurally able to serve this node. The
// dispatcher and the support query must agree on this, so both ask exactly this question.
static bool pxa_fa_d512_volta_takes(const ggml_tensor * dst, int cc) {
    // The one door: only a node the graph builder marked (after its width cut, context floor,
    // every-card probe, architecture and offload checks) is served. An unmarked 512/512 node -
    // any other builder's, a probe's, a test's - keeps the route it always had.
    if (((const int32_t *) dst->op_params)[5] != 1) {
        return false;
    }
    switch (pxa_fa_d512_volta_mode()) {
        case 1:  return ggml_cuda_fattn_volta_mma_d512_supported(dst, cc);
        case 2:  return ggml_cuda_fattn_volta_tile_d512_supported(dst, cc);
        default: return false;
    }
}

// PXA_FA_D256_VOLTA_TILE (2026-09-21) -- see the block comment in pxa/fattn-volta-tile.cu for the
// per-node measurement that motivates it. At head 256 with an f16 cache this fork serves query
// width 1 from the vector kernel and everything above it from the legacy WMMA tile, because the
// vector kernel is NO_DEVICE_CODE above one column on CUDA. A speculative verify step is two to
// four columns wide, so every verify tick pays the WMMA tile -- 388 us per node at n_kv 16384
// against 89 for the kernel vendored in pxa/volta-tile, which is the one upstream selects for this
// shape. This lever routes that window to it.
//
// A SUBSTITUTION, dispatch side only: every shape the kernel declines falls through to the route
// the support query names, so ggml_backend_supports_op() answers exactly what it answered before
// and the graph is built the same way whether the lever is on or off.
//
//   0 : off - the route this fork always had (vector at width 1, WMMA above it)
//   1 : the tile kernel for query widths 2 .. PXA_FA_D256_VOLTA_TILE_MAXCOLS
//   2 : width 1 as well, i.e. plain decode
//
// The KV floor (PXA_FA_D256_VOLTA_TILE_MINKV, default 1280) is deliberate and not cosmetic: the
// gain is at depth, and the 512-wide sibling of this kernel has an unexplained short-KV defect
// (bug #189: below ~64 cells the fused route can flip the top-1 token). The floor is the same fence
// that path carries, set at the same value, until the 256 path has its own short-KV evidence.
static int pxa_fa_d256_tile_mode() {
    static const int mode = [] {
        const int m = (int) pxa_lever(PXA_LEVER_FA_D256_VOLTA_TILE);
        // Arm the exit report as soon as the lever is set by hand, so a run that arms it and serves
        // nothing says so with a zero instead of saying nothing at all.
        if (m && pxa_lever_set_by_user(PXA_LEVER_FA_D256_VOLTA_TILE)) {
            ggml_cuda_fattn_volta_tile_d256_arm_report();
        }
        return m;
    }();
    return mode;
}

static bool pxa_fa_d256_tile_takes(const ggml_tensor * dst, int cc) {
    const int mode = pxa_fa_d256_tile_mode();
    if (mode == 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const int64_t ne1 = Q->ne[1];
    if (ne1 < (mode >= 2 ? 1 : 2)) {
        return false;
    }
    if (ne1 > pxa_lever(PXA_LEVER_FA_D256_VOLTA_TILE_MAXCOLS)) {
        return false;
    }
    if (K->ne[1] < pxa_lever(PXA_LEVER_FA_D256_VOLTA_TILE_MINKV)) {
        return false;
    }
    return ggml_cuda_fattn_volta_tile_d256_supported(dst, cc);
}

static bool pxa_fa_mma_volta_take(const ggml_tensor * Q) {
    const bool enabled = pxa_lever(PXA_LEVER_FA_MMA_VOLTA) != 0;
    if (!enabled) {
        return false;
    }
    // PXA_FA_MMA_VOLTA=2 also routes DECODE (Q->ne[1] <= 8) here, purely so the
    // question can be measured. Note upstream does NOT choose this kernel at that
    // batch size on Volta: ggml_cuda_get_best_fattn_kernel sends
    // Q->ne[1]*gqa_ratio_eff <= 2 to the vec kernel and <= 16 to the tile kernel,
    // so for a GQA-6 model MMA only starts at Q->ne[1] > 8. Default stays > 8.
    // 2026-09-20: =2 is DISABLED. The decode-width instances of this kernel are stubs on sm_70 (only the
    // large-batch column counts are built), so lifting the gate faults the card with an unspecified
    // launch failure at widths 1, 2 and 4. The value is accepted and treated as =1, loudly.
    const bool take = Q->ne[1] > 8;
    if (take) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            fprintf(stderr, "PXA_FA_MMA_VOLTA: engaged (sm_70 large-batch flash-attention -> vendored mainline m8n8k4 MMA kernel; "
                            "first node ne1=%d head=%d; PXA_FA_MMA_VOLTA=0 reverts to the tile route)\n",
                    (int)Q->ne[1], (int)Q->ne[0]);
        }
    }
    return take;
}

enum pxa_fa_tile_volta_mode { PXA_FA_TILE_VOLTA_OFF = 0, PXA_FA_TILE_VOLTA_ALL = 1, PXA_FA_TILE_VOLTA_AUTO = 2 };

static int pxa_fa_tile_volta_mode() {
    return (int) pxa_lever(PXA_LEVER_FA_TILE_VOLTA);
}

static bool pxa_fa_tile_volta_take(const ggml_tensor * Q) {
    const int mode = pxa_fa_tile_volta_mode();
    if (mode == PXA_FA_TILE_VOLTA_OFF) return false;
    if (mode == PXA_FA_TILE_VOLTA_ALL) return true;
    // AUTO: prefill / large batch only, decode keeps the WMMA route.
    const bool take = Q->ne[1] > 8;
    if (take) { // announce once so a run can be shown to have taken this route
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            fprintf(stderr, "PXA_FA_TILE_VOLTA: AUTO engaged (sm_70 large-batch flash-attention -> tile kernel; "
                            "first node ne1=%d head=%d; decode keeps WMMA; PXA_FA_TILE_VOLTA=0 reverts)\n",
                    (int)Q->ne[1], (int)Q->ne[0]);
        }
    }
    return take;
}

static bool pxa_fa_tile256_enabled() {
    return pxa_lever(PXA_LEVER_FA_TILE256) != 0;
}

// =============================================================================================
// Node preparation (dispatch side only).
// =============================================================================================

// PXA_NANFIX_SWA_SLICE (2026-08-16): the windowed SWA slice below assumed KV-cache cell
// INDEX order matches POSITION order, so that the last pad(max(ntokens,256)+n_swa) cells of
// the unified cache contain every in-window cell. That holds only for a single sequence laid
// down into an empty cache. With np>1 or any slot reuse, a request's cells can sit at LOW
// cell indices while another sequence's long state raises K->ne[1]; once
// first = K->ne[1] - nton > 0 the slice cuts the query's own cells OUT of the view, every
// sliced mask row is all -inf, and the fully-masked flash-attention output is NaN -> ALL
// logits non-finite (the V100 seat's NaN cascade). Lab repro: any truncating-reuse/short decode
// while any sequence holds more than n_swa+256 cells; single-ubatch vs multi-ubatch and
// n_ctx are irrelevant. The slice is therefore DISABLED by default; PXA_FA_SWA_SLICE=1
// restores the old behavior for single-sequence benchmarking only.
// In its place op_params[4] now KEEPS n_swa (it used to be zeroed here in all cases), so the
// kernels' mask-driven KV_min_max scan — which reads the actual mask and is correct for any
// cell layout — bounds the KV iteration for SWA decode instead. PXA_FA_SWA_KEEP=0 zeroes it
// again (old dispatch behavior, full-range iteration) as a fallback lever.
ggml_tensor * pxa_fa_prepare_node(ggml_tensor * dst, pxa_fa_scratch_t & scratch) {
    // A node carrying a DSA index list is routed before any rewrite could apply to it, which is
    // what the dispatcher did when this block sat inline above the DSA branch. Keep it that way:
    // the sparse kernel must see the node it was given.
    if (ggml_cuda_dsa_attn_requested(dst)) {
        return dst;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int32_t n_swa = ((const int32_t *) dst->op_params)[4];
    if (n_swa <= 0) {
        return dst;
    }

    const bool pxa_swa_slice_on = pxa_lever(PXA_LEVER_FA_SWA_SLICE) != 0;
    const bool pxa_swa_keep_on  = pxa_lever(PXA_LEVER_FA_SWA_KEEP)  != 0;

    if (pxa_swa_slice_on) {
        int ntokens = std::max(FATTN_KQ_STRIDE, int(Q->ne[1]));
        int nton = FATTN_KQ_STRIDE*((ntokens + n_swa + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE);
        int first = K->ne[1] - nton;
        scratch.dst = *dst;
        scratch.dst.op_params[4] = 0;
        if (first > 0) { // UNSOUND with np>1 / slot reuse — see PXA_NANFIX_SWA_SLICE above
            scratch.dst = *dst;
            scratch.K = *K; scratch.K.ne[1] = nton; scratch.K.data = (char *)K->data + K->nb[1]*first;
            scratch.V = *V; scratch.V.ne[1] = nton; scratch.V.data = (char *)V->data + V->nb[1]*first;
            scratch.M = *mask; scratch.M.ne[0] = nton; scratch.M.data = (char *)mask->data + mask->nb[0]*first;
            scratch.dst.src[1] = &scratch.K;
            scratch.dst.src[2] = &scratch.V;
            scratch.dst.src[3] = &scratch.M;
            scratch.dst.op_params[4] = 0;
        }
        return &scratch.dst;
    }

    if (!pxa_swa_keep_on) {
        scratch.dst = *dst;
        scratch.dst.op_params[4] = 0;
        return &scratch.dst;
    }

    // else: dst untouched — n_swa flows through to the mask-driven KV_min_max scan.
    return dst;
}

// =============================================================================================
// The planner.
// =============================================================================================

static pxa_fa_plan_t pxa_fa_take(pxa_fa_route route, bool supported, const char * why) {
    pxa_fa_plan_t plan;
    plan.route     = route;
    plan.supported = supported;
    plan.why       = why;
    return plan;
}

pxa_fa_plan_t pxa_fa_plan_node(ggml_backend_cuda_context & ctx, const pxa_fa_query_t & q) {
    const ggml_tensor * dst  = q.node;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // The card being ASKED, by device id -- never the card that happens to be current. A support
    // query for a P100 issued while a V100 is the current device must answer for the P100. On the
    // dispatch side ggml_cuda_set_device(ctx.device) has already run, so this is the same value
    // the dispatcher used to read out of ggml_cuda_get_device().
    const int     cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int32_t precision = ((const int32_t *) dst->op_params)[3];
    const int32_t n_swa     = q.n_swa;
    const bool    dispatch  = q.kind == PXA_FA_QUERY_DISPATCH;

    // ---- sparse attention over an index list ------------------------------------------------
    // The only CUDA attention path that accepts head dim 512 by index list; on sm_70 it is what
    // keeps DeepSeek-V4 attention off the CPU backend.
    if (ggml_cuda_dsa_attn_supported(dst, cc)) {
        return pxa_fa_take(PXA_FA_ROUTE_DSA, true, "DSA index list");
    }
    // A node WITH src[5] must not fall through. The DSA path is the only kernel that reads the
    // index list; once it has declined -- PXA_DSA_ATTN off, or a shape it does not serve -- nothing
    // below honours the request, so the node belongs on a backend that computes the node it was
    // given rather than on a dense kernel that answers a different question. The support query
    // therefore says no and the dispatcher aborts rather than silently ignoring src[5].
    if (ggml_cuda_dsa_attn_requested(dst)) {
        return pxa_fa_take(PXA_FA_ROUTE_DSA_UNSERVED, false, "DSA index list no dense kernel honours");
    }

    // ---- AMD ---------------------------------------------------------------------------------
    // The two texts differed here and the difference is PRESERVED: the dispatcher also asked
    // fast_fp16_available(cc) before choosing the f16 vec kernel, the support query did not.
    // Unreachable in this build (no HIP target), kept so the move changes nothing.
    if (cc >= CC_OFFSET_AMD) {
        const bool f16_route = precision == GGML_PREC_DEFAULT && fast_fp16_available(cc);
        const bool supported = precision == GGML_PREC_DEFAULT
                             ? ggml_cuda_fattn_vec_f16_is_supported(ctx, dst)
                             : ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
        return pxa_fa_take(f16_route ? PXA_FA_ROUTE_VEC_F16 : PXA_FA_ROUTE_VEC_F32, supported, "AMD vec");
    }

    // ---- PXA_FA_TILE_512: head 512/512 and 576/512 on sm_60 / sm_70 -------------------------
    // The kernel declines everything else, so arming the lever can only ever change whether a
    // 512-wide node runs fused, never which kernel a 64/128/256 node gets.
    if (ggml_cuda_fattn_tile_big_armed() && ggml_cuda_fattn_tile_big_is_supported(ctx, dst)) {
        return pxa_fa_take(PXA_FA_ROUTE_TILE_BIG, true, "PXA_FA_TILE_512");
    }

    // ---- PXA_FA_D512_VOLTA: head 512/512 on sm_70 -------------------------------------------
    // With the lever unset (below ENHANCE) the 512/512 node is declined here, no GPU backend
    // claims it, and llm_build_kqv() builds the unfused chain instead. The predicate is fenced by
    // the graph builder's mark in op_params[5] -- see pxa_fa_d512_volta_takes().
    if (pxa_fa_d512_volta_takes(dst, cc)) {
        const bool mma = pxa_fa_d512_volta_mode() == 1;
        return pxa_fa_take(mma ? PXA_FA_ROUTE_D512_MMA : PXA_FA_ROUTE_D512_TILE, true, "PXA_FA_D512_VOLTA");
    }

    // ---- PXA_FA_D256_VOLTA_TILE (dispatch only) ----------------------------------------------
    // A SUBSTITUTION for the vec / WMMA rows further down, which is where the support answer keeps
    // coming from. Placed above PXA_FA_MMA_VOLTA because the two windows do not overlap -- that one
    // starts at Q->ne[1] > 8 and this one ends at PXA_FA_D256_VOLTA_TILE_MAXCOLS (8 by default) --
    // and if a user widens this window past it, the narrower, measured route is the one that wins.
    if (dispatch && pxa_fa_d256_tile_takes(dst, cc)) {
        return pxa_fa_take(PXA_FA_ROUTE_D256_TILE, true, "PXA_FA_D256_VOLTA_TILE");
    }

    // ---- PXA_FA_MMA_VOLTA (dispatch only) ----------------------------------------------------
    // A SUBSTITUTION: every shape it declines falls through to the route the support query names,
    // so the support answer does not depend on it and never has.
    if (dispatch && ggml_cuda_fattn_volta_mma_supported(dst, cc) && pxa_fa_mma_volta_take(Q)) {
        return pxa_fa_take(PXA_FA_ROUTE_VOLTA_MMA, true, "PXA_FA_MMA_VOLTA");
    }

    // ---- PXA_FA_TILE_VOLTA (dispatch only) ---------------------------------------------------
    // Same kind of substitution: sm_70 borrows the P100 tile route for A/B work. Default OFF.
    if (dispatch && pxa_fa_tile_volta_take(Q) && fp16_mma_available(cc) && !new_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT && ggml_cuda_fattn_tile_f16_is_supported(ctx, dst)) {
            return pxa_fa_take(PXA_FA_ROUTE_TILE_F16, true, "PXA_FA_TILE_VOLTA");
        }
        if (ggml_cuda_fattn_tile_f32_is_supported(ctx, dst)) {
            return pxa_fa_take(PXA_FA_ROUTE_TILE_F32, true, "PXA_FA_TILE_VOLTA");
        }
    }

    // ---- no fast fp16 (sm_61 and older) ------------------------------------------------------
    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            return pxa_fa_take(PXA_FA_ROUTE_VEC_F32, ggml_cuda_fattn_vec_f32_is_supported(ctx, dst),
                               "no fast fp16, narrow or D=256");
        }
        return pxa_fa_take(PXA_FA_ROUTE_TILE_F32, ggml_cuda_fattn_tile_f32_is_supported(ctx, dst),
                           "no fast fp16, wide");
    }

    // ---- fast fp16 but no fp16 mma (sm_60, P100) --------------------------------------------
    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            // PXA_FA_TILE256: D=256 no longer forces the vec kernel for batch > 8.
            if (Q->ne[1] <= 8 || (Q->ne[0] == 256 && !pxa_fa_tile256_enabled())) {
                if (pxq_use_sm60_vec_f32(cc, Q)) { // PR #2144: sm_60 decode -> fp32 accumulation
                    return pxa_fa_take(PXA_FA_ROUTE_VEC_F32, ggml_cuda_fattn_vec_f32_is_supported(ctx, dst),
                                       "sm_60 decode, PXQ_SM60_FA_VEC_F32");
                }
                return pxa_fa_take(PXA_FA_ROUTE_VEC_F16, ggml_cuda_fattn_vec_f16_is_supported(ctx, dst),
                                   "no fp16 mma, narrow");
            }
            // PXA_FA_TILE_V2 (dispatch only): same arithmetic, a different tile schedule, and it
            // declines any shape it does not serve -- so the support answer is the tile-f16 one
            // either way, which is what the support text said before the merge.
            if (dispatch && ggml_cuda_fattn_tile_v2_armed() && ggml_cuda_fattn_tile_v2_is_supported(ctx, dst)) {
                return pxa_fa_take(PXA_FA_ROUTE_TILE_V2, true, "PXA_FA_TILE_V2");
            }
            return pxa_fa_take(PXA_FA_ROUTE_TILE_F16, ggml_cuda_fattn_tile_f16_is_supported(ctx, dst),
                               "no fp16 mma, wide");
        }
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            return pxa_fa_take(PXA_FA_ROUTE_VEC_F32, ggml_cuda_fattn_vec_f32_is_supported(ctx, dst),
                               "f32 precision, narrow or D=256");
        }
        return pxa_fa_take(PXA_FA_ROUTE_TILE_F32, ggml_cuda_fattn_tile_f32_is_supported(ctx, dst),
                           "f32 precision, wide");
    }

    // ---- Turing and newer --------------------------------------------------------------------
    // Two decode shortcuts the dispatcher has and the support query never had; they are
    // substitutions for the rows below, which is where the support answer keeps coming from.
    if (dispatch && new_mma_available(cc) && K->ne[0] == 128 && V->ne[0] == 128 && Q->ne[0] == 128 && Q->ne[1] == 1 &&
            (Q->ne[2] / K->ne[2] == 12 || Q->ne[2] / K->ne[2] == 6 || Q->ne[2] / K->ne[2] == 10)) {
        return pxa_fa_take(PXA_FA_ROUTE_MMA_NEW, true, "new mma, D=128 decode");
    }
    if (dispatch && new_mma_available(cc) && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[0] == 256 && Q->ne[1] == 1 &&
            Q->ne[2] / K->ne[2] == 6) {
        return pxa_fa_take(PXA_FA_ROUTE_MMA_NEW, true, "new mma, D=256 decode");
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // the mma kernels have GQA-specific optimizations
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        return pxa_fa_take(PXA_FA_ROUTE_VEC_F32, ggml_cuda_fattn_vec_f32_is_supported(ctx, dst), "bs1 vector kernel");
    }

    // The 576/512, 320/256, 512/512 and 192/128 family. The two texts phrase the SAME family
    // differently -- the dispatcher by the (K,V) head-size pair, the support query by Q->ne[0]
    // plus a "gqa ratio is a multiple of 4" rule the dispatcher does not repeat. Both are kept
    // verbatim; unifying them would be a behaviour change and belongs to its own measurement.
    if (dispatch) {
        if (new_mma_available(cc) &&
                ((K->ne[0] == 576 && V->ne[0] == 512) ||
                 (K->ne[0] == 320 && V->ne[0] == 256) ||
                 (K->ne[0] == 512 && V->ne[0] == 512) ||
                 (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
            return pxa_fa_take(PXA_FA_ROUTE_MMA_NEW, true, "new mma, large head");
        }
    } else {
        if (new_mma_available(cc) &&
                (Q->ne[0] == 576 || Q->ne[0] == 320 || Q->ne[0] == 512 ||
                 (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
            if (Q->ne[0] == 576 || Q->ne[0] == 512 || Q->ne[0] == 320) {
                const int gqa_ratio = Q->ne[2]/K->ne[2];
                return pxa_fa_take(PXA_FA_ROUTE_MMA_NEW, (gqa_ratio % 4) == 0, "new mma, large head");
            }
            return pxa_fa_take(PXA_FA_ROUTE_MMA_NEW, true, "new mma, large head");
        }
    }

    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        // Attention-sink models (e.g. gpt-oss): the wmma kernel does not implement attention
        // sinks, so on sm_70 the prefill path would otherwise drop them and corrupt the context.
        // The tile kernel is sink-aware for head sizes 64/128 -> route there. Dispatch only: the
        // support query answers from ggml_cuda_fattn_wmma_f16_is_supported() as it always has.
        if (dispatch && dst->src[4] != nullptr && K->ne[0] == V->ne[0] && (K->ne[0] == 64 || K->ne[0] == 128)) {
            if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
                return pxa_fa_take(PXA_FA_ROUTE_TILE_F16, true, "attention sinks");
            }
            return pxa_fa_take(PXA_FA_ROUTE_TILE_F32, true, "attention sinks");
        }
        return pxa_fa_take(PXA_FA_ROUTE_WMMA_F16, ggml_cuda_fattn_wmma_f16_is_supported(ctx, dst), "wmma");
    }

    return pxa_fa_take(PXA_FA_ROUTE_MMA_F16, ggml_cuda_fattn_mma_f16_is_supported(ctx, dst), "mma f16");
}

const char * pxa_fa_route_name(pxa_fa_route route) {
    switch (route) {
        case PXA_FA_ROUTE_NONE:         return "none";
        case PXA_FA_ROUTE_DSA:          return "dsa";
        case PXA_FA_ROUTE_DSA_UNSERVED: return "dsa-unserved";
        case PXA_FA_ROUTE_VEC_F16:      return "vec-f16";
        case PXA_FA_ROUTE_VEC_F32:      return "vec-f32";
        case PXA_FA_ROUTE_TILE_F16:     return "tile-f16";
        case PXA_FA_ROUTE_TILE_F32:     return "tile-f32";
        case PXA_FA_ROUTE_TILE_V2:      return "tile-v2";
        case PXA_FA_ROUTE_TILE_BIG:     return "tile-big-512";
        case PXA_FA_ROUTE_D512_MMA:     return "d512-volta-mma";
        case PXA_FA_ROUTE_D512_TILE:    return "d512-volta-tile";
        case PXA_FA_ROUTE_D256_TILE:    return "d256-volta-tile";
        case PXA_FA_ROUTE_VOLTA_MMA:    return "volta-mma";
        case PXA_FA_ROUTE_WMMA_F16:     return "wmma-f16";
        case PXA_FA_ROUTE_MMA_F16:      return "mma-f16";
        case PXA_FA_ROUTE_MMA_NEW:      return "mma-new";
        default:                        return "?";
    }
}

// =============================================================================================
// The route census. One counter per (device, route), printed at exit when PXA_CORE_ROUTES=1.
// This is the first thing in the engine that can answer "did the kernel we shipped actually run,
// and on which card" without a fprintf inside the kernel.
// =============================================================================================

static std::atomic<uint64_t> g_pxa_fa_census[GGML_CUDA_MAX_DEVICES][PXA_FA_ROUTE_COUNT];

static void pxa_fa_route_census_report() {
    fprintf(stderr, "PXA_CORE_ROUTES: flash-attention nodes dispatched, by device and route\n");
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES; ++dev) {
        uint64_t total = 0;
        for (int r = 0; r < PXA_FA_ROUTE_COUNT; ++r) {
            total += g_pxa_fa_census[dev][r].load(std::memory_order_relaxed);
        }
        if (total == 0) {
            continue;
        }
        for (int r = 0; r < PXA_FA_ROUTE_COUNT; ++r) {
            const uint64_t n = g_pxa_fa_census[dev][r].load(std::memory_order_relaxed);
            if (n) {
                fprintf(stderr, "PXA_CORE_ROUTES:   device %d  %-16s %llu\n",
                        dev, pxa_fa_route_name((pxa_fa_route) r), (unsigned long long) n);
            }
        }
        fprintf(stderr, "PXA_CORE_ROUTES:   device %d  %-16s %llu\n", dev, "TOTAL", (unsigned long long) total);
    }
}

static bool pxa_fa_route_census_on() {
    // Armed once, and the report is registered at the same moment, so a run that arms the lever
    // and dispatches nothing still prints a header instead of saying nothing at all.
    static const bool on = [] {
        const bool armed = pxa_lever(PXA_LEVER_CORE_ROUTES) != 0;
        if (armed) {
            atexit(pxa_fa_route_census_report);
        }
        return armed;
    }();
    return on;
}

void pxa_fa_route_census(int device, pxa_fa_route route) {
    if (!pxa_fa_route_census_on()) {
        return;
    }
    const int r = (int) route;
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || r < 0 || r >= (int) PXA_FA_ROUTE_COUNT) {
        return;
    }
    g_pxa_fa_census[device][r].fetch_add(1, std::memory_order_relaxed);
}
