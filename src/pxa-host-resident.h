#pragma once
// PXA_HOST_RESIDENT gate helpers (pass 8e review R6/R7), header-only so tests/test-pxa-host-resident.cpp can check them
// on the CPU. The loader (src/llama.cpp llm_load_tensors) copies a small host weight context into its pinned host
// buffer only on full offload and only at or below the cap.
#include <algorithm>
#include <cstddef>
#include <cstdlib>

// every repeating layer and the output head on the cards (the documented scope of the rule)
inline bool pxa_host_resident_full_offload(int i_gpu_start, int n_gpu_layers, int n_layer) {
    return i_gpu_start == 0 && n_gpu_layers > n_layer;
}

// PXA_HOST_RESIDENT_MAX_MIB -> bytes. Unset: 2048 MiB. A positive integer: that many MiB, clamped to 64 GiB.
// Anything else (negative, zero, empty, junk, trailing characters): 0 = the rule is off, and *bad is set so the
// caller can say so once (it used to wrap a negative value to an unlimited cap).
inline size_t pxa_host_resident_cap_bytes(const char * e, bool * bad = nullptr) {
    if (bad) *bad = false;
    if (!e) return (size_t) 2048 << 20;
    char * end = nullptr;
    const long long v = strtoll(e, &end, 10);
    if (end == e || *end != '\0' || v <= 0) {
        if (bad) *bad = true;
        return 0;
    }
    return (size_t) std::min<long long>(v, 65536) << 20;
}
