// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// PXA_PLE_PIPELINE: may llama_graph_compute skip the qwen4exp per-graph PLE barrier?
//
// The barrier (ggml_backend_sched_synchronize after every graph while the PLE conv window exists) orders one
// ubatch's window write against the next graph's read. With scheduler copy slots (n_copies > 1, PXA_PIPELINE_PP)
// the write and the read sit on one device stream and consecutive graphs' inputs live in different slots, so the
// barrier only serialises the cards of a -sm layer prefill (device-busy <= 1.0 of 4).
//
// Measured (impl pp-next-ple-pipeline, 2026-09-25): Flash-Next PXQU on 4x P100 (sm_60) -sm layer, PXA_PIPELINE_PP=1
// copies 4, -b 8192: pp20801 398.3 -> 492.3 t/s (+23.6%), pp3121 517.9 -> 592.2 (+14.3%); the barrier's own flip
// test (needle3121 = two ubatches) 4/4 identical and correct, greedy sha identical to the barrier path.
//
// Default since 2026-10-19 (DEFAULTS sweep): auto = on where it was measured, i.e. the model has
// copy slots AND runs on exactly 4 devices that are all sm_60 (compute capability 600). Copy slots on qwen4exp
// still need PXA_PIPELINE_PP=1 (the 150k-context seat cannot afford their compute buffers), so a bare launch is
// unchanged; the lever no longer has to be named next to it.
//   PXA_PLE_PIPELINE=0   off (keep the barrier: the pre-2026-10-19 default)
//   PXA_PLE_PIPELINE=1   on wherever there are copy slots (any card count / arch)
//   PXA_REFERENCE=1      off unless PXA_PLE_PIPELINE=1 is named
// Without copy slots it never acts (the barrier is then the only thing ordering the window).
#pragma once

#include <cstdlib>

// env      = getenv("PXA_PLE_PIPELINE") (nullptr or "" = auto)
// n_copies = ggml_backend_sched_get_n_copies(sched)
// cc       = compute capability per model device (CUDA: 600 = P100), n_dev entries; nullptr / n_dev 0 = no GPU
static inline bool pxa_ple_pipeline_decide(const char * env, int n_copies, const int * cc, int n_dev, bool reference) {
    if (n_copies <= 1) {
        return false;
    }
    if (env != nullptr && *env) {
        return atoi(env) != 0;
    }
    if (reference) {
        return false;
    }
    if (cc == nullptr || n_dev != 4) {
        return false;
    }
    for (int i = 0; i < n_dev; ++i) {
        if (cc[i] != 600) {
            return false;
        }
    }
    return true;
}
