#pragma once
// PXA_PROFILE_KEEP_TARGET (2026-10-24; default ON, =0 off).
//
// The PXA model profile (ggml_pxa_set_model_profile) is ONE per process, and every model load used to
// overwrite it. The Gemma 4 assistant drafter (arch gemma4_mtp: a dense 4-layer companion that reads the
// target's K/V and never runs alone) loads AFTER its target, so it replaced the target's profile
// (gemma4 / moe / 128 experts) with its own (gemma4_mtp / dense / 0 experts), and the CUDA side re-resolved
// every model-aware lever for the WHOLE process, the target's graph included. Measured on one P100,
// Gemma 4 26B-A4B PXQN4 + drafter, 8k prompt (run-20261024-0223 / -0239): ROUTER_FUSE 3 -> 0, so the
// target's 30 router GEMVs per token went from the 6.0 us router kernel to cuBLAS gemv2T at 22.7 us
// (nvprof), and a plain round with the drafter armed ran 49.40 t/s vs 51.45 t/s true plain. On V100 the
// same overwrite turns ROUTER_FUSE mode 1 off.
//
// Rule (PXA_PROFILE_KEEP_TARGET, default 1): a model load never replaces a profile already registered when
//   1 (default): the model is a companion (gemma4_mtp), the measured cell;
//   2 (opt-in):  it is a companion OR any separate-file draft model the server loads (-md / --model-draft,
//                llama_pxa_set_loading_draft around that load): a dense or non-PXQ draft would otherwise flip the
//                target's ROUTER_FUSE / MOE_DEVICE_MAP / PXQ_MMVQ / PXQ_GEMM_2D. Unmeasured on those pairs, so opt-in;
//   0: off, the old last-writer-wins.
// The first load with no profile registered always registers its own. Built-in nextn/MTP heads (GLM, Qwen) load no
// second model (common/speculative.cpp builds ctx_mtp on the target model), so the profile is set once there.
#include <cstdlib>

static inline bool llama_pxa_profile_keep_target(bool loading_companion, bool loading_draft, bool profile_registered,
                                                 const char * env) {
    const int mode = (env && *env) ? std::atoi(env) : 1;
    if (mode <= 0 || !profile_registered) {
        return false;
    }
    return loading_companion || (mode >= 2 && loading_draft);
}
