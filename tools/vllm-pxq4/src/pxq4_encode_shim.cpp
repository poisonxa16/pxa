// pxq4_encode_shim.cpp — C-ABI wrapper exposing the file-static pxq6_quantize_expert
// (from mgv-wt src/pxq6-quantize.inc.cpp, shipped verbatim) as the frozen ABI required by
// gguf_to_vllm/encoder.py:
//
//     int pxq4_encode(const float *src, uint8_t *dst, int R, int K, const float *imx_or_null)
//
// Tier: GGML_TYPE_PXQ4 (id 252) is the CORE tier (tier = 0, bs16 subs, 1088 B slabs).
// Determined from the dispatch in mgv-wt src/llama-quantize.cpp:2624
//     const int tier = tgt == GGML_TYPE_PXQ4HQ ? 1 : 0;
// and confirmed by the ABI's own panel size formula 128 + (K/32)*1088 == PXQ6_HDR_BYTES +
// (K/32)*PXQ6_SLAB_BYTES (tier 1 / HQ would be 1152 B slabs).
//
// row0 is ALWAYS absolute (0 for the whole tensor). Threading goes through the upstream
// pxq6_quantize_tensor with E=1, which chunks panels and passes each chunk's ABSOLUTE row
// offset, so the bytes are identical at every thread count (see P15 note in the .inc.cpp).
//
// Build: g++ -O2 -std=c++17 -fPIC -shared -mf16c -pthread -o libpxq4_encode.so pxq4_encode_shim.cpp
// (NO -ffast-math: the codec is parity-locked to exact fp32 rounding.)

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <immintrin.h>

// ggml fp16 shims: identical to ggml-impl.h on x86 with F16C
// (GGML_COMPUTE_FP16_TO_FP32 = _cvtsh_ss, GGML_COMPUTE_FP32_TO_FP16 = _cvtss_sh(x, 0),
// i.e. IEEE binary16 with round-to-nearest-even — same as the scalar fallback).
typedef uint16_t ggml_fp16_t;
static inline float ggml_fp16_to_fp32(ggml_fp16_t h) { return _cvtsh_ss(h); }
static inline ggml_fp16_t ggml_fp32_to_fp16(float f) { return (ggml_fp16_t)_cvtss_sh(f, 0); }

// Copied verbatim from mgv-wt src/llama-quantize.cpp:1049-1062 (needed by
// pxq6_quantize_tensor's imx_for lambda).
static std::atomic<int64_t> g_pxq_imx_dead_cols{0};

static bool pxq_imatrix_column_usable(const float * w, int64_t K) {
    if (!w) return false;
    double s = 0.0;
    for (int64_t i = 0; i < K; ++i) {
        if (!std::isfinite(w[i]) || w[i] < 0.0f) { g_pxq_imx_dead_cols.fetch_add(1); return false; }
        s += (double) w[i];
    }
    if (s > 0.0) return true;
    g_pxq_imx_dead_cols.fetch_add(1);
    return false;
}

#include "pxq6-quantize.inc.cpp"

extern "C" int pxq4_encode(const float * src, uint8_t * dst, int R, int K,
                           const float * imx_or_null) {
    if (!src || !dst) return 2;
    if (R <= 0 || K <= 0 || (R % 64) != 0 || (K % 32) != 0) return 1;
    int nth = (int)std::thread::hardware_concurrency();
    if (const char * e = getenv("PXQ4_ENCODE_THREADS")) { const int v = atoi(e); if (v > 0) nth = v; }
    if (nth < 1) nth = 1;
    if (nth > 64) nth = 64;
    pxq6_quantize_tensor(src, dst, (int64_t)R, (int64_t)K, /*E=*/1,
                         imx_or_null, imx_or_null ? (int64_t)K : 0, nth, /*tier=*/0);
    return 0;
}

// Native reference dequant (tier 0), exposed for validation only: cross-checking it against
// gguf_to_vllm/reference.dequant on real artifact bytes proves tables+layout agreement
// independently of encode idempotence.
extern "C" int pxq4_decode(const uint8_t * src, float * dst, int R, int K) {
    if (!src || !dst) return 2;
    if (R <= 0 || K <= 0 || (R % 64) != 0 || (K % 32) != 0) return 1;
    pxq6_dequant_expert(src, dst, (int64_t)R, (int64_t)K, /*tier=*/0);
    return 0;
}
