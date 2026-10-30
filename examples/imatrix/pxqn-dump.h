#pragma once
// PXQN activation dump: fp32 rows -> fp16 for the encoder's statistics (H = X^T X).
// fp16 has no finite value above 65504; a larger activation (the "massive activations" a few channels of a few tokens carry) used to be
// written as +-inf, which made the layer's statistics non-finite and stopped the encode at the Cholesky. Such a value is SATURATED to
// +-65504 instead (NaN, which carries nothing, to 0) and counted. Saturating keeps the row, and with it the weight the outlier channel
// has in H: on a synthetic massive-activation set the LDLQ output error with saturated rows matched the fp32 truth within 0.2 %, while
// dropping those rows was 22-34 % worse (scripts/sat_vs_skip.py). Every value of magnitude <= 65504 converts exactly
// as before (the fast path is the same single ggml_fp32_to_fp16_row call), so a dump without overflow is byte-identical.
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <vector>

// convert n floats; returns how many were saturated (|x| > 65504) or zeroed (NaN). tmp is scratch (only used when something is changed).
static inline int64_t pxqn_dump_to_fp16(const float * x, ggml_fp16_t * y, int64_t n, std::vector<float> & tmp) {
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) bad += !(std::fabs(x[i]) <= 65504.0f);       // NaN fails the comparison too
    if (bad == 0) { ggml_fp32_to_fp16_row(x, y, n); return 0; }
    tmp.assign(x, x + n);
    for (int64_t i = 0; i < n; ++i) {
        const float v = tmp[i];
        if (std::isnan(v)) tmp[i] = 0.0f;
        else if (v > 65504.0f) tmp[i] = 65504.0f;
        else if (v < -65504.0f) tmp[i] = -65504.0f;
    }
    ggml_fp32_to_fp16_row(tmp.data(), y, n);
    return bad;
}
