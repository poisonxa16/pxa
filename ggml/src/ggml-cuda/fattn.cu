//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16-interface.cuh"
#include "fattn-vec-f32-interface.cuh"
#include "fattn-wmma-f16-interface.cuh"
#include "pxa/fattn-volta-mma.cuh"
#include "fattn-mma-f16-interface.cuh"
#include "fattn-new-mma.cuh"
#include "fattn.cuh"
#include "convert.cuh"
#include "dsa_attn.cuh"

#include <atomic>
#include <cstdint>
#include <cstdlib>

#define FATTN_KQ_STRIDE 256

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
    static const bool enabled = [] {
        const char * v = getenv("PXQ_SM60_FA_VEC_F32");
        return !(v && v[0] == '0');
    }();
    return enabled;
}

static inline bool pxq_use_sm60_vec_f32(const int cc, const ggml_tensor * Q) {
    return cc == CC_PASCAL && Q->ne[1] <= 8 && pxq_sm60_fa_vec_f32_enabled();
}

// PXA_FA_TILE256 (2026-08-03): on pre-Volta cards the D=256 head had NO tile/mma prefill kernel,
// so `Q->ne[0] == 256` forced the single-column VEC kernel at ANY batch size. Profiled on the
// 122B-A10B (qwen35moe, head 256) 4xP100 rig at 8881-token fill: flash_attn_vec_ext_f16 was 52.7%
// of prefill GPU time (60 launches x 281 ms avg) — every query column re-streams the whole KV
// extent with no tile reuse. The D=256 tile-f16 (ncols=16) restores KQ-tile data reuse.
// Default ON; PXA_FA_TILE256=0 restores the vec route. Decode (ne1 <= 8) is untouched.
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
// NOTE FOR WHOEVER OWNS THE TILE KERNEL: the same D=256 ncols=16 tile-f16 instance is the
// DEFAULT prefill path on sm_60 via PXA_FA_TILE256. This capture does not prove it is
// broken there -- sm_70 takes different branches, and PXA_FA_MASK_SKIP_TILE auto-armed in
// these runs is tile-only and a prime suspect -- but the P100 seat deserves the same
// greedy-32 capture before anyone trusts it.
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
static bool pxa_fa_mma_volta_take(const ggml_tensor * Q) {
    static const bool enabled = [] {
        const char * v = getenv("PXA_FA_MMA_VOLTA");
        const bool on = !(v && v[0] == '0');
        if (!on) {
            fprintf(stderr, "PXA_FA_MMA_VOLTA=0: sm_70 flash-attention will not use the vendored mainline MMA kernel\n");
        }
        return on;
    }();
    if (!enabled) {
        return false;
    }
    // PXA_FA_MMA_VOLTA=2 also routes DECODE (Q->ne[1] <= 8) here, purely so the
    // question can be measured. Note upstream does NOT choose this kernel at that
    // batch size on Volta: ggml_cuda_get_best_fattn_kernel sends
    // Q->ne[1]*gqa_ratio_eff <= 2 to the vec kernel and <= 16 to the tile kernel,
    // so for a GQA-6 model MMA only starts at Q->ne[1] > 8. Default stays > 8.
    static const bool include_decode = [] {
        const char * v = getenv("PXA_FA_MMA_VOLTA");
        return v && v[0] == '2';
    }();
    const bool take = include_decode || Q->ne[1] > 8;
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
    static const int mode = [] {
        const char * v = getenv("PXA_FA_TILE_VOLTA");
        if (v && v[0] == '0') {
            fprintf(stderr, "PXA_FA_TILE_VOLTA=0: sm_70 flash-attention stays on the legacy WMMA kernel at every batch size\n");
            return (int) PXA_FA_TILE_VOLTA_OFF;
        }
        if (v && v[0] == '1') {
            fprintf(stderr, "PXA_FA_TILE_VOLTA=1: sm_70 flash-attention -> tile kernel at EVERY batch size (decode included)\n");
            return (int) PXA_FA_TILE_VOLTA_ALL;
        }
        if (v && v[0] == '2') {
            fprintf(stderr, "PXA_FA_TILE_VOLTA=2: sm_70 large-batch flash-attention -> tile kernel "
                            "(KNOWN TO PRODUCE CORRUPT OUTPUT ON THIS ARCH -- benchmarking only)\n");
            return (int) PXA_FA_TILE_VOLTA_AUTO;
        }
        return (int) PXA_FA_TILE_VOLTA_OFF; // see the correctness note above
    }();
    return mode;
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
    static const bool enabled = [] {
        const char * v = getenv("PXA_FA_TILE256");
        const bool on = !(v && v[0] == '0');
        fprintf(stderr, "PXA_FA_TILE256: %s (pre-Volta D=256 batch>8 attention -> tile-f16 ncols=16; PXA_FA_TILE256=0 reverts to vec)\n",
                on ? "ON" : "OFF");
        return on;
    }();
    return enabled;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];

    // DSA sparse attention. A node carrying an index list in src[5] attends only the
    // listed KV rows; this is the only CUDA attention path that accepts head dim 512,
    // so on sm_70 it is what keeps DeepSeek-V4 attention off the CPU backend.
    // ggml_cuda_fattn_is_supported() consults the same predicate -- keep them paired.
    if (ggml_cuda_dsa_attn_supported(dst, cc)) {
        ggml_cuda_dsa_attn_ext(ctx, dst);
        return;
    }

    ggml_tensor local_dst, Kl, Vl, Ml;
    // PXA_NANFIX_SWA_SLICE (2026-08-16): the windowed SWA slice below assumed KV-cache cell
    // INDEX order matches POSITION order, so that the last pad(max(ntokens,256)+n_swa) cells of
    // the unified cache contain every in-window cell. That holds only for a single sequence laid
    // down into an empty cache. With np>1 or any slot reuse, a request's cells can sit at LOW
    // cell indices while another sequence's long state raises K->ne[1]; once
    // first = K->ne[1] - nton > 0 the slice cuts the query's own cells OUT of the view, every
    // sliced mask row is all -inf, and the fully-masked flash-attention output is NaN -> ALL
    // logits non-finite (the Alina NaN cascade). Lab repro: any truncating-reuse/short decode
    // while any sequence holds more than n_swa+256 cells; single-ubatch vs multi-ubatch and
    // n_ctx are irrelevant. The slice is therefore DISABLED by default; PXA_FA_SWA_SLICE=1
    // restores the old behavior for single-sequence benchmarking only.
    // In its place op_params[4] now KEEPS n_swa (it used to be zeroed here in all cases), so the
    // kernels' mask-driven KV_min_max scan — which reads the actual mask and is correct for any
    // cell layout — bounds the KV iteration for SWA decode instead. PXA_FA_SWA_KEEP=0 zeroes it
    // again (old dispatch behavior, full-range iteration) as a fallback lever.
    if (n_swa > 0) {
        static const bool pxa_swa_slice_on = []() {
            const char * e = getenv("PXA_FA_SWA_SLICE");
            return e && e[0] == '1';
        }();
        static const bool pxa_swa_keep_on = []() {
            const char * e = getenv("PXA_FA_SWA_KEEP");
            return !(e && e[0] == '0');
        }();
        if (pxa_swa_slice_on) {
            int ntokens = std::max(FATTN_KQ_STRIDE, int(Q->ne[1]));
            int nton = FATTN_KQ_STRIDE*((ntokens + n_swa + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE);
            int first = K->ne[1] - nton;
            local_dst = *dst;
            local_dst.op_params[4] = 0;
            if (first > 0) { // UNSOUND with np>1 / slot reuse — see PXA_NANFIX_SWA_SLICE above
                local_dst = *dst;
                Kl = *K; Kl.ne[1] = nton; Kl.data = (char *)K->data + K->nb[1]*first;
                Vl = *V; Vl.ne[1] = nton; Vl.data = (char *)V->data + V->nb[1]*first;
                Ml = *mask; Ml.ne[0] = nton; Ml.data = (char *)mask->data + mask->nb[0]*first;
                local_dst.src[1] = &Kl;
                local_dst.src[2] = &Vl;
                local_dst.src[3] = &Ml;
                local_dst.op_params[4] = 0;
                dst = &local_dst;
            }
            dst = &local_dst;
        } else if (!pxa_swa_keep_on) {
            local_dst = *dst;
            local_dst.op_params[4] = 0;
            dst = &local_dst;
        }
        // else: dst untouched — n_swa flows through to the mask-driven KV_min_max scan.
    }

    // On AMD the tile kernels perform poorly, use the vec kernel instead:
    if (cc >= CC_OFFSET_AMD) {
        if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    // PXA_FA_TILE_VOLTA (experimental, default OFF). sm_70 HAS working WMMA, so the selector below
    // sends Volta to the WMMA kernel and the tile kernels only ever serve the no-mma cards (sm_60
    // P100). That makes tile-vs-WMMA on Volta untestable. This lever routes sm_70 -- and only sm_70
    // (fp16 mma present, Turing+ mma absent) -- down the same tile path the P100s take, so both can
    // be A/B'd inside ONE binary. If tile cannot handle the shape we fall through to the normal
    // selector rather than failing. With the env unset this file behaves exactly as before, so the
    // WMMA path and PXA_FA_FA_MASK_SKIP_v1 remain the default untouched.
    // sm_70: prefer the vendored mainline MMA kernel; it declines any shape it was
    // not instantiated for and control falls through to the tile route below.
    // the support check runs first so the one-shot "engaged" line is only printed on a device that will use the kernel
    if (ggml_cuda_fattn_volta_mma_supported(dst, cc) && pxa_fa_mma_volta_take(Q)) {
        ggml_cuda_flash_attn_ext_volta_mma(ctx, dst);
        return;
    }

    if (pxa_fa_tile_volta_take(Q) && fp16_mma_available(cc) && !new_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT && ggml_cuda_fattn_tile_f16_is_supported(ctx, dst)) {
            ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            return;
        }
        if (ggml_cuda_fattn_tile_f32_is_supported(ctx, dst)) {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            return;
        }
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            // PXA_FA_TILE256: D=256 no longer forces the vec kernel for batch > 8 — the tile-f16
            // kernel now carries a ncols=16 D=256 variant (see fattn-tile-f16.cu).
            if (Q->ne[1] <= 8 || (Q->ne[0] == 256 && !pxa_fa_tile256_enabled())) {
                if (pxq_use_sm60_vec_f32(cc, Q)) { // PR #2144: sm_60 decode -> fp32 accumulation
                    ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                }
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 128 && V->ne[0] == 128 && Q->ne[0] == 128 && Q->ne[1] == 1 &&
            (Q->ne[2] / K->ne[2] == 12 || Q->ne[2] / K->ne[2] == 6 || Q->ne[2] / K->ne[2] == 10)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[0] == 256 && Q->ne[1] == 1 && Q->ne[2] / K->ne[2] == 6) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        return;
    }

    //
    // It turns out the new new MMA implementation is slower than the
    // previous MMA implementation.
    // Hence, we use it only for DeepSeek with MLA enabled, where head sizes are 576, 512,
    // so no other implementation works.
    //

    if (new_mma_available(cc) &&
            ((K->ne[0] == 576 && V->ne[0] == 512) ||
             (K->ne[0] == 320 && V->ne[0] == 256) ||
             (K->ne[0] == 512 && V->ne[0] == 512) ||
             (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        //printf("Using ggml_cuda_flash_attn_ext_mma_new\n");
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    //
    // We need this because I haven't adapted new MMA kernels to work for different
    // K and V head sizes.
    // We also need it if the new MMA is not available
    //
    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        // Attention-sink models (e.g. gpt-oss): the wmma kernel does not implement attention
        // sinks, so on sm_70 (Volta/V100) the prefill path would otherwise drop them and corrupt
        // the context. The tile kernel is now sink-aware for head sizes 64/128 -> route there.
        if (dst->src[4] != nullptr && K->ne[0] == V->ne[0] && (K->ne[0] == 64 || K->ne[0] == 128)) {
            if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
            return;
        }
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }

    // As mentioned above, the new-new MMA is slower then the new MMA.
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
    //ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
}

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];

    // Must mirror the first branch of ggml_cuda_flash_attn_ext() exactly: same
    // predicate, same position (before every head-dim check), so the scheduler never
    // routes a DSA node to a backend that cannot run it -- and never rejects one the
    // dispatcher would have accepted. Nodes WITHOUT src[5] fall through unchanged and
    // are still rejected at head 512 on pre-Ampere, as before.
    if (ggml_cuda_dsa_attn_supported(dst, cc)) {
        return true;
    }

    if (cc >= CC_OFFSET_AMD) {
        return precision == GGML_PREC_DEFAULT ? ggml_cuda_fattn_vec_f16_is_supported(ctx, dst)
                                              : ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
        } else {
            return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
        }
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || (Q->ne[0] == 256 && !pxa_fa_tile256_enabled())) {
                if (pxq_use_sm60_vec_f32(cc, Q)) { // PR #2144: keep supported-check in lockstep
                    return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
                }
                return ggml_cuda_fattn_vec_f16_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f16_is_supported(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
            }
        }
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (new_mma_available(cc) &&
            (Q->ne[0] == 576 || Q->ne[0] == 320 || Q->ne[0] == 512 || (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        if (Q->ne[0] == 576 || Q->ne[0] == 512 || Q->ne[0] == 320) {
            int gqa_ratio = Q->ne[2]/K->ne[2];
            return (gqa_ratio % 4) == 0;
        }
        return true;
    }

    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        return ggml_cuda_fattn_wmma_f16_is_supported(ctx, dst);
    }

    return ggml_cuda_fattn_mma_f16_is_supported(ctx, dst);
}
