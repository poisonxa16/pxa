#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>
typedef uint16_t ggml_fp16_t;
static inline float    test_fp16_to_fp32(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t test_fp32_to_fp16(float f)    { return _cvtss_sh(f, 0); }
#define ggml_fp16_to_fp32 test_fp16_to_fp32
#define ggml_fp32_to_fp16 test_fp32_to_fp16
// pre-empt the shared helper with the LEGACY always-lowest semantics (pre-change behavior)
#define PXQ_TIE_BREAK_DEFINED
static inline bool pxq_tie_take_hi(int64_t, int64_t) { return false; }
#include "../src/pxq6r-quantize.inc.cpp"
extern "C" void q_old(const float * src, uint8_t * dst, int64_t R, int64_t K) { pxq6r_quantize_expert(src, dst, R, K, nullptr); }
extern "C" void d_old(const uint8_t * src, float * dst, int64_t R, int64_t K) { pxq6r_dequant_expert(src, dst, R, K); }
