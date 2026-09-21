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

// PXA_TSPLIT_LMHEAD -- default OFF in this wave. Splits the LM head vocab-parallel across the
// participating devices instead of leaving it, and the final projection of every token, on one
// card. It CHANGES OUTPUT: each device produces its own slice of the vocabulary and the slices are
// concatenated, so the logits are computed in a different order from the single-device head. Off by
// default until the decode number and the fidelity reading are on the ledger.
static inline bool pxa_tsplit_lmhead_split(void) {
    return pxa_tsplit_env_flag("PXA_TSPLIT_LMHEAD", false);
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
