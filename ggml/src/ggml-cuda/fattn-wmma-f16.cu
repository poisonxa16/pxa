#include "fattn-wmma-f16.cuh"
#include "fattn-wmma-f16-interface.cuh"

// PXA_FA_WMMA_CPB (2026-09-02) — cols_per_block for the LARGE-BATCH (prefill) WMMA
// flash-attention tile. sm_70 is the only arch that reaches this kernel for prefill
// (fp16_mma_available && !new_mma_available; see fattn.cu), so this is a Volta lever.
//
// WHY: the incumbent picks cols_per_block = 32 for every Q->ne[1] > 32, which at head
// dim 256 is a register catastrophe. `frag_b Q_b[Dk/16][ncols/frag_n]` is held in
// registers by EVERY warp, so at Dk=256 it is 16*2 = 32 m16n16k16 fragments = 128
// registers per thread before any accumulator. ptxas -v on sm_70 (nvcc 12.8, -O3),
// flash_attn_ext_f16<256,256,cpb,4,64,1,half,*>:
//     cpb=32 : 255 regs, 2488 B stack, 2924 B spill-st / 4424 B spill-ld, 33808 B smem
//     cpb=16 : 255 regs, 1000 B stack, 1020 B spill-st / 1112 B spill-ld, 16912 B smem
//     cpb= 8 : (frag_m=32, VKQ_stride=128) — smaller still
// cpb=32 therefore runs at 2 blocks/SM (32768 of 65536 regs) while spilling ~3x more
// local traffic per K-tile than cpb=16, and burns 2x the smem for no extra reuse.
// Head dims <= 128 are NOT affected: they were the shapes cpb=32 was tuned for and they
// keep it (Q_b there is at most 8*2 fragments).
//
// AUTO (default): Dk >= 256 -> 16, everything else -> the incumbent 32.
// Env PXA_FA_WMMA_CPB = 8 | 16 | 32 forces a value for A/B; 0/unset = auto.
// This is a DISPATCH-only change: all three <256,256,cpb,half> template instances
// already exist (template-instances/fattn-wmma-f16-instance-kqhalf-cpb{8,16,32}.cu),
// so nothing new is compiled and no numerics change per-instance. Different cpb tiles
// the same math differently, so the online-softmax reduction order changes and results
// are NOT bit-identical to the cpb=32 build.
static int pxa_fa_wmma_cpb(int64_t Dk) {
    static const int forced = [] {
        const char * e = getenv("PXA_FA_WMMA_CPB");
        const int v = e ? atoi(e) : 0;
        if (v == 8 || v == 16 || v == 32) {
            fprintf(stderr, "PXA_FA_WMMA_CPB: forced cols_per_block=%d for large-batch WMMA flash-attention\n", v);
            return v;
        }
        return 0;
    }();
    if (forced) {
        return forced;
    }
    return Dk >= 256 ? 16 : 32;
}

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // PXA_FA_WMMA_MASK_STRIDE_FIX detector: report (once) when the half-accumulate branch would
    // have used a WRONG mask stride under the old code (nb31/sizeof(half) != ne11). PREC_F32
    // (float branch) was always correct; this fires only for the exposed default-precision path.
    {
        const ggml_tensor * m = dst->src[3];
        const ggml_tensor * K = dst->src[1];
        const int32_t prec = dst->op_params[3];
        if (m && K && prec == GGML_PREC_DEFAULT) {
            const int64_t stride_h = (int64_t)m->nb[1] / (int64_t)sizeof(half);
            if (stride_h != K->ne[1]) {
                static bool warned = false;
                if (!warned) { warned = true;
                    fprintf(stderr, "PXA_FA_WMMA_MASK_STRIDE: bug-active shape seen (mask row stride %lld half != K->ne[1] %lld); the ne11 stride was WRONG here, now corrected\n",
                            (long long)stride_h, (long long)K->ne[1]);
                }
            }
        }
    }

    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * V   = dst->src[2];

    if (Q->ne[0] != V->ne[0]) {
        if (!((Q->ne[0] == 192 && V->ne[0] == 128) || (Q->ne[0] == 576 && V->ne[0] == 512))) {
            fprintf(stderr, "======================= %s: Unhandled head size combination %d, %d\n", __func__, (int)Q->ne[0], (int)V->ne[0]);
            GGML_ABORT("fatal error");
        }
    }

    const int32_t precision = KQV->op_params[3];

    if (precision != GGML_PREC_DEFAULT) {
        if (Q->ne[1] <= 32 || Q->ne[0] > 128) {
            constexpr int cols_per_block = 16;
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, float>(ctx, dst);
                    break;
                case 256:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, float>(ctx, dst);
                    break;
                case 192:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, float>(ctx, dst);
                    break;
                default:
                    fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                    GGML_ABORT("fatal error");
                    break;
            }
        } else {
            constexpr int cols_per_block = 32;
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, float>(ctx, dst);
                    break;
                case 192:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, float>(ctx, dst);
                    break;
                // case 256:
                //     ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                //     break;
                default:
                    fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                    GGML_ABORT("fatal error");
                    break;
            }
        }
        return;
    }

    if ((Q->ne[1] <= 8 || pxa_fa_wmma_cpb(Q->ne[0]) == 8) && Q->ne[0] % WARP_SIZE == 0) {
        constexpr int cols_per_block = 8;
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, half>(ctx, dst);
                break;
            case 192:
                ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, half>(ctx, dst);
                break;
            default:
                fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }

    if (Q->ne[1] <= 32 || pxa_fa_wmma_cpb(Q->ne[0]) == 16) {
        constexpr int cols_per_block = 16;
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, half>(ctx, dst);
                break;
            case 80:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, half>(ctx, dst);
                break;
            case 112:
                ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, half>(ctx, dst);
                break;
            case 192:
                ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, half>(ctx, dst);
                break;
            default:
                fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }
    constexpr int cols_per_block = 32;
    switch (Q->ne[0]) {
        case 64:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, half>(ctx, dst);
            break;
        case 80:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, half>(ctx, dst);
            break;
        case 96:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, half>(ctx, dst);
            break;
        case 112:
            ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, half>(ctx, dst);
            break;
        case 128:
            ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, half>(ctx, dst);
            break;
        case 192:
            ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, half>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, half>(ctx, dst);
            break;
        default:
            fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
            GGML_ABORT("fatal error");
            break;
    }
}

bool ggml_cuda_fattn_wmma_f16_is_supported([[maybe_unused]] ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    auto K = dst->src[1];
    auto V = dst->src[2];
    if (K->ne[0] != V->ne[0]) return K->ne[0] == 192 && V->ne[0] == 128;
    return K->ne[0] == 64 || K->ne[0] == 80 || K->ne[0] == 96 || K->ne[0] == 112 || K->ne[0] == 128 || K->ne[0] == 256;
}
