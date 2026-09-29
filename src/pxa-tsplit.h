#pragma once

// Levers for the PXA tensor split ('-sm tensor', LLAMA_SPLIT_MODE_TENSOR).
//
// Kept in one header because the admission (src/llama.cpp) and the tensor policy
// (src/llama-load-tensors.cpp) both read them and a drifted default would be invisible.
// Every one of them is documented in docs/LEVERS.md.

#include <cstdlib>

static inline bool pxa_tsplit_env_flag(const char * name, bool def) {
    const char * v = getenv(name);
    if (v == nullptr) {
        return def;
    }
    return atoi(v) != 0;
}

// PXA_TSPLIT_FALLBACK -- default OFF. '-sm tensor' is an explicit request; when it cannot be
// honoured the default is to say why and stop, not to quietly run something else. Set it to 1 to
// get the old behaviour of falling back to '-sm layer' (the refusal is still printed).
static inline bool pxa_tsplit_fallback_to_layer(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_FALLBACK", false);
}

// PXA_TSPLIT_ALLOW_4WAY -- default ON since 2026-09-27 (was OFF from 2026-09-25, bug #206).
// A tensor split over MORE than two devices served wrong tokens on 4x P100 while perplexity matched
// the layer split. The root cause was not the split: NCCL's SHM transport could not fit a 4-rank
// group in docker's 64 MB /dev/shm, ncclGroupEnd failed and its status was thrown away, so every
// card kept its own partial. That failure is now checked and demotes the NCCL route to the in-tree
// peer route for the rest of the process (ggml-cuda/reduce.cu, pxa_nccl_route_fail), and the 4-way
// arms are byte-identical to the NCCL-off route. Measured 2026-09-27 on 4x P100, dense
// 27B PXQN4, one binary, layer/tensor/layer: tensor decode +4.6..5.6%, prefill @22.6k +95%; dense 27B PXQ4:
// decode +38.5..41.1%, prefill +95.6% (lane
// launch-defaults, impl tsplit-4card-default). Set it to 0 to get the old guard back: a request for a
// >2-device split (typed or auto-picked) is warned about and runs '-sm layer' instead.
static inline bool pxa_tsplit_allow_4way(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_ALLOW_4WAY", true);
}

// PXA_TSPLIT_FORCE_FA -- default ON. The split attention builder requires flash attention: with the
// weights carrying ->extra and flash attention off, the builder falls through to a generic path
// that reads the split tensor's dummy base pointer. That is undefined behaviour, not an error. So
// under a tensor split flash attention is turned ON and the boot says so. Set it to 0 to get a hard
// refusal instead of the force.
static inline bool pxa_tsplit_force_fa(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_FORCE_FA", true);
}

// PXA_TSPLIT_UNPROVEN_ARCH -- default OFF. '-sm tensor' admits an arch on two separate facts: that
// its split graph builders are complete, and that the arch has been RUN through this split here.
// The second fact is what a hybrid arch needs most: the reduce-delivery defect that the DeltaNet
// fixes address produced degenerate output at temp 0, and "the fixes are armed" is not evidence
// about an arch nobody has run. So a hybrid with a complete builder and no recorded measurement is
// refused by name. Set this to 1 to run it anyway (for the measurement that would make the row):
// the boot then says, loudly, that nothing here has seen this arch through this mode.
static inline bool pxa_tsplit_allow_unproven_arch(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_UNPROVEN_ARCH", false);
}

// PXA_TSPLIT_LMHEAD -- default ON (2026-09-27; was OFF). Splits the LM head vocab-parallel
// across the participating devices instead of leaving it, and the final projection of every token, on one
// card: the head is a serial tail after the last all-reduce (ledger tsplit-k-head-device), so halving it
// is the only way to shorten it. Each logit row is still one device's full-K dot product, so the values
// are the single-device head's (greedy sha identical, ledger pxa-tsplit-lmhead-vocab-parallel). Its one
// measured cost was the concat gather in build_output; PXA_TSPLIT_LMHEAD_DIRECT below removes it. The
// loader's guards (arch list, MTP, -ot, granularity) still decide; =0 keeps the head on one device.
static inline bool pxa_tsplit_lmhead_split(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_LMHEAD", true);
}

// PXA_TSPLIT_LMHEAD_DIRECT -- default ON. With a vocab-parallel head, build_output no
// longer concatenates the per-device logit slices into one tensor on one device (a cross-device copy of
// half the logits plus a concat kernel, every token, and a second full-width logits buffer in the compute
// reserve). Each slice stays on its own device as a graph output and the host read-back copies every
// slice straight into its column range of the logits buffer. =0 restores the concat.
#define PXA_HEAD_PART_NAME "pxa_head_part"
static inline bool pxa_tsplit_lmhead_direct(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_LMHEAD_DIRECT", true);
}

// PXA_TSPLIT_GEMMA4 -- default OFF. Gemma 4 is the one arch whose split graph builder has never
// executed: the capability check refuses it by name, and '-sm graph'/'-sm attn' demote it to
// '-sm layer' before the builder is reached. Both of those doors are held shut by this one flag,
// so a probe run needs exactly one hand-set variable and a default run cannot reach the builder by
// accident. Setting it does not claim the arch is measured -- it is what makes the measurement
// possible, and the boot says so out loud.
static inline bool pxa_tsplit_gemma4(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_GEMMA4", false);
}
