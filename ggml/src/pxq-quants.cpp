//
// PXA-owned quant codecs.
//
// These are the only per-type codecs that survived the phase-3 prune of the 53 ik-only quant
// types (2026-08-21). They were relocated here VERBATIM from ggml/src/iqk/iqk_quantize.cpp so
// that the whole ggml/src/iqk directory could be deleted; the arithmetic is byte-for-byte the
// code that was running before, because the CPU gauntlet compares token streams and any edit
// here would show up as a divergence rather than as a compile error.
//
// What is here and why each one stays:
//   mxfp4        -- ours (type 39). Consumed by our own PXQ4 files; ik-off vec_dot_type is
//                   stock q8_0 and the dot is pxa_mxfp4_dot_q8_0.
//   q8_0_x4      -- NOT dead: it is the live .vec_dot_type of GGML_TYPE_Q1_0_G128, whose
//                   scalar dot reads block_q8_0_x4 fields directly.
//   q8_KV        -- user-selectable KV cache (-ctk / -ctv q8_KV) with a working scalar dot
//                   (pxa_q8_KV_dot_q8_KV, pxq-cpu.c). Its 8-row-interleaved _R8 sibling went.
//   q1_0_g128    -- Bonsai 1-bit files; real scalar codec, no accelerator dependency.
//   ms_i2s       -- Microsoft BitNet I2_S files; real scalar dequant.
//   pxa_quantize_any  -- generic dequantise->requantise for a quantized mul_mat src1 and for
//                   llama.cpp's tensor-type conversion. Was iqk_quantize_any; the row
//                   interleave factor it used to look up is now always 1, because every type
//                   that made it anything else has been removed.
//   pxa_validate_tensor -- the --check-tensors / validate_quants scan. Was iqk_validate_tensor,
//                   minus the arms for the removed types.
//
// Provenance: derived from work by Iwan Kawrakow, MIT licensed. See the file it came from in
// the history of this repository.
//

#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "pxq-quants.h"
#include "pxq-cpu.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstring>
#include <memory>

namespace {


inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#if defined __AVX2__
// Horizontal reductions used by the q8_0_x4 and q8_KV row quantizers below. Relocated from
// ggml/src/iqk/iqk_common.h together with the codecs that call them.
inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64  = _mm_add_epi32(hi64, sum128);
    const __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
inline float hmax_f32_8(__m256 x) {
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    return _mm_cvtss_f32(max4);
}
#endif

}   // anonymous namespace


// ---------------------------------------------------------------------------------------
// Generic dequantise -> requantise. Was iqk_quantize_any(). The row-interleave lookup it used
// to do (num_rows()) returned something other than 1 only for the _R4/_R8/_R16 repacked types
// and for two ik activation formats, all of which are gone, so the factor is now a constant 1
// and the loop is a plain row loop.
// ---------------------------------------------------------------------------------------
void pxa_quantize_any(int from_type, int to_type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                      const void * x, void * y, void * work_buffer,
                      pxa_to_float_t to_float, pxa_from_float_t from_float, int ith, int nth) {
    auto type_x = ggml_type(from_type);
    GGML_ASSERT(ggml_type_size(type_x) == nb0);
    auto type_y = ggml_type(to_type);
    auto row_size_y = ggml_row_size(type_y, ne0);
    int64_t nrows = ne1*ne2*ne3;
    int64_t nrows_per_thread = (nrows + nth - 1)/nth;
    int64_t first_row = nrows_per_thread*ith;
    if (first_row >= nrows) return;
    int64_t last_row = std::min(first_row + nrows_per_thread, nrows);
    for (int64_t row = first_row; row < last_row; ++row) {
        int64_t i3 = row/(ne1*ne2);
        int64_t i2 = (row - i3*ne1*ne2)/ne1;
        int64_t i1 = row - i3*ne1*ne2 - i2*ne1;
        auto cx = (const char *)x + i1*nb1 + i2*nb2 + i3*nb3;
        auto cy = (char *)y + (i3*ne1*ne2 + i2*ne1 + i1)*row_size_y;
        if (type_x != GGML_TYPE_F32) {
            to_float((const void *)cx, (float *)work_buffer, ne0);
            from_float((const float *)work_buffer, (void *)cy, ne0);
        } else {
            from_float((const float *)cx, (void *)cy, ne0);
        }
    }
}


// ============================== q8_0_x4 (activation format for q1_0_g128)

void quantize_row_q8_0_x4(const float * x, void * vy, int64_t k) {
    const int nb = k / QK8_0;
    const int nb4 = 4*(nb/4);

    block_q8_0    * y  = (block_q8_0    *)vy;
    block_q8_0_x4 * y4 = (block_q8_0_x4 *)vy;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[32*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[32*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[32*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[32*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }
        }
    }
#elif defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        const float d = maxScalar / 127.f;
        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#else
    // Portable scalar path -- same reason as quantize_row_q8_1_x4_T below: the branch above was
    // an #else on __aarch64__, so plain x86-64 without AVX2 landed in AVX2 intrinsics. Semantics
    // taken from the NEON branch: amax over the block, d = amax/127, round-to-nearest-even
    // quants written in natural order at qs + 32*ir.
    for (int i = 0; i < nb; i++) {
        const int i4 = i/4, ir = i%4;
        const float * xb = x + i*QK8_0;
        float amax = 0.f;
        for (int j = 0; j < QK8_0; ++j) { const float a = fabsf(xb[j]); if (a > amax) amax = a; }
        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        if (i < nb4) y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        else         y[i].d       = GGML_FP32_TO_FP16(d);
        int8_t * qs = (i < nb4) ? (int8_t *)y4[i4].qs + 32*ir : (int8_t *)y[i].qs;
        for (int j = 0; j < QK8_0; ++j) qs[j] = (int8_t)nearbyintf(xb[j]*id);
    }
#endif
}

// ============================== q8_KV

void iqk_quantize_row_q8_KV(const float * x, void * vy, int64_t k) {
    assert(k % 32 == 0);
    auto dptr = (float *)vy;
    auto q8 = (int8_t *)(dptr + 2);
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256 maxAbs = _mm256_setzero_ps();
    for (int ib = 0; ib < k/8; ++ib) {
        const __m256 v = _mm256_loadu_ps(x + 8*ib);
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
    }
    const float maxScalar = hmax_f32_8(maxAbs);
    if (!maxScalar) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = maxScalar / 127.f;
    auto mul = _mm256_set1_ps(1/dptr[0]);
    auto isum = _mm256_setzero_si256();
    for (int i = 0; i < k/32; i++) {
        __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  0));
        __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  8));
        __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 16));
        __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 24));
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        isum = _mm256_add_epi32(isum, _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        i0 = _mm256_packs_epi16( i0, i2 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );
        _mm256_storeu_si256((__m256i *)q8, i0);
        q8 += 32;
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = hsum_i32_8(isum);
#elif defined __ARM_NEON
    int32x4_t ival[8];
    auto vmax = vdupq_n_f32(0.f);
    for (int j = 0; j < k; j += 4) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + j)));
    }
    auto smax = vmaxvq_f32(vmax);
    if (!smax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = smax/127;
    auto vid = vdupq_n_f32(1/dptr[0]);
    auto isum = vdupq_n_s32(0);
    for (int ib = 0; ib < k/32; ++ib) {
        auto xb = x + 32*ib;
        for (int k = 0; k < 8; ++k) {
            auto val = vld1q_f32(xb + 4*k);
            ival[k] = vcvtnq_s32_f32(vmulq_f32(val, vid));
            isum = vaddq_s32(isum, ival[k]);
        }
        for (int k = 0; k < 4; ++k) {
            auto i16 = vcombine_s16(vmovn_s32(ival[2*k+0]), vmovn_s32(ival[2*k+1]));
            vst1_s8(q8, vmovn_s16(i16));
            q8 += 8;
        }
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = vaddvq_s32(isum);
#else
    float amax = 0;
    for (int j = 0; j < k; ++j) {
        float ax = std::abs(x[j]);
        amax = std::max(amax, ax);
    }
    if (!amax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = amax/127;
    float id = 1/dptr[0];
    int isum = 0;
    for (int i = 0; i < k; i++) {
        q8[i] = nearest_int(id*x[i]);
        isum += q8[i];
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = isum;
#endif
}
void quantize_row_q8_KV(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_KV(x, vy, k);
}

void quantize_row_q8_KV_ref(const float * x, void * y, int64_t k) {
    quantize_row_q8_KV(x, y, k);
}

size_t quantize_q8_KV(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    (void)imatrix;
    auto row_size = ggml_row_size(GGML_TYPE_Q8_KV, n_per_row);
    auto q = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        quantize_row_q8_KV(src, q, n_per_row);
        src += n_per_row;
        q += row_size;
    }
    return row_size*nrows;
}

void dequantize_row_q8_KV(const void * x, float * y, int64_t k) {
    auto dptr = (const float *)x;
    float d = dptr[0];
    auto q8 = (const int8_t *)(dptr + 2);
    for (int j = 0; j < k; ++j) y[j] = d * q8[j];
}

void vec_dot_q8_KV_q8_KV(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    // The body used to end here. Without the accelerated matmul the whole #if above vanishes,
    // both asserts pass (n is a head size, a multiple of 32), and the function returned having
    // never assigned *s -- the same shape as vec_dot_mxfp4_q8_0_x4 and ggml_vec_dot_q6_0_q8_0.
    // q8_KV is user-selectable (-ctk q8_KV / -ctv q8_KV, common.cpp), and it is read by BOTH
    // the generic flash-attention loop -- where the caller's `float s;` is an uninitialized
    // stack slot that then goes through expf(), so garbage can be inf/NaN and not merely a
    // wrong logit -- and the generic mul_mat chunk loop against the K cache, where it is a
    // stale tmp[] instead. No repack and no interleaved layout is involved: q8_KV's
    // .vec_dot_type is q8_KV itself, so this is a plain per-row int8 dot.
    //
    // Note this is NOT only a no-accelerator fix: iqk_mul_mat above returns bool and declines
    // shapes it does not support, and until now that fell through to the same empty body on an
    // AVX2 build too. Measured on the AVX2 reference build after this change, calling this
    // function directly with nrc == 1: n = 32/64/128 now return the correct dot, and the only
    // code that could have produced it is the scalar body below -- i.e. the accelerated matmul
    // had been declining those shapes and writing nothing. The scalar body therefore sits below
    // the #if unconditionally, not inside an #else.
    //
    // For n >= 256 the same call does NOT reach here: it aborts inside the accelerated kernel
    // (GGML_ASSERT(nrc_x%4 == 0), iqk_gemm_kquants.cpp mul_mat_q8_KV_q8_KV) because a vec_dot is
    // by definition one row. That is a separate, pre-existing defect on the accelerated side and
    // is reported, not fixed here -- this file cannot reach past the early return.
    //
    // The implementation lives in our own translation unit and reads the STOCK row layout
    // (see pxa_q8_KV_dot_q8_KV in pxq-cpu.c for the layout derivation), so nothing on this
    // path is ik's -- not the format, not the code.
    pxa_q8_KV_dot_q8_KV(n, s, vx, vy);
}


namespace {
inline int best_index_mxfp4(float d, const int8_t * values, float x) {
    float best = std::abs(x - d*values[0]);
    int index = 0;
    for (int j = 1; j < 16; ++j) {
        float diff = std::abs(x - d*values[j]);
        if (diff < best) { best = diff; index = j; }
    }
    return index;
}
static void quantize_row_mxfp4_impl(int n_per_row, const float * x, char * cy,
        float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    GGML_ASSERT(n_per_row % QK_MXFP4 == 0);

    block_mxfp4 * y = (block_mxfp4 *)cy;

    if (!quant_weights) {
        //
        // Legacy RTN path — BIT-IDENTICAL to the pre-patch quantizer when no imatrix
        // is supplied (quantize_row_mxfp4 / quantize_row_mxfp4_ref always land here).
        //
        for (int ib = 0; ib < n_per_row/QK_MXFP4; ++ib) {
            memset(&y[ib], 0, sizeof(block_mxfp4));
            const float * xb = x + ib*QK_MXFP4;
            float amax = 0;
            for (int j = 0; j < QK_MXFP4; ++j) {
                float ax = fabsf(xb[j]);
                amax = std::max(amax, ax);
            }
            if (!amax) {
                continue;
            }
            const uint8_t e = (uint8_t) (floorf(log2f(amax)) - 2 + 127);
            const float d = GGML_E8M0_TO_FP32_HALF(e);
            y[ib].e = e;
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                uint8_t v0 = best_index_mxfp4(d, values, xb[j]);
                uint8_t v1 = best_index_mxfp4(d, values, xb[j+QK_MXFP4/2]);
                y[ib].qs[j] = v0 | (v1 << 4);
            }
        }
        return;
    }

    //
    // Imatrix-weighted path (PXA4). Same block_mxfp4 wire format (type 38), so zero
    // runtime/kernel change — only WHICH e8m0 exponent + nibbles get written differs.
    //
    // Per-element note: for a FIXED scale d the reconstruction grid d*values[] is fixed,
    // so nearest-value rounding already minimizes w*(x - d*q)^2 for any w > 0 — the index
    // choice is imatrix-independent. The imatrix therefore acts on the SCALE: we search
    // the e8m0 exponent in {e0-1, e0, e0+1} around the legacy floor(log2(amax))-2 pick
    // and keep the exponent with the smallest imatrix-weighted squared error.
    //
    // Weights follow the ik convention used by the sibling quants (e.g. IQ4_KS):
    //     weight[j] = qw[j] * sqrtf(sigma2 + x[j]*x[j]),  sigma2 = 2*mean(x^2)
    // with sigma2 computed once per QK_K super-block.
    //
    int last_ibl = -1;
    float sigma2 = 0;

    const int span = ntry > 0 ? 1 : 0;   // e8m0 exponent search half-width

    for (int ib = 0; ib < n_per_row/QK_MXFP4; ++ib) {
        memset(&y[ib], 0, sizeof(block_mxfp4));
        const float * xb = x + ib*QK_MXFP4;
        if (int ibl = ib/(QK_K/QK_MXFP4); ibl != last_ibl) {
            int n = std::min(QK_K, n_per_row - ibl*QK_K);
            const float * xbl = x + ibl*QK_K;
            float sumx2 = 0;
            for (int j = 0; j < n; ++j) sumx2 += xbl[j]*xbl[j];
            sigma2 = 2.0f*sumx2/n;
            last_ibl = ibl;
        }
        const float * qw = quant_weights + ib*QK_MXFP4;
        for (int j = 0; j < QK_MXFP4; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        float amax = 0;
        for (int j = 0; j < QK_MXFP4; ++j) {
            float ax = fabsf(xb[j]);
            amax = std::max(amax, ax);
        }
        if (!amax) {
            continue;
        }
        int e0 = (int) floorf(log2f(amax)) - 2 + 127;
        e0 = std::max(0, std::min(254, e0));   // keep e8m0 valid; legacy uint8 wrap never useful here
        float   best_err = -1;
        uint8_t best_e   = (uint8_t) e0;
        uint8_t best_qs[QK_MXFP4/2];
        uint8_t qs[QK_MXFP4/2];
        for (int ie = -span; ie <= span; ++ie) {
            const int ei = e0 + ie;
            if (ei < 0 || ei > 254) continue;
            const uint8_t e = (uint8_t) ei;
            const float d = GGML_E8M0_TO_FP32_HALF(e);
            float err = 0;
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                const int i1 = best_index_mxfp4(d, values, xb[j]);
                const int i2 = best_index_mxfp4(d, values, xb[j+QK_MXFP4/2]);
                const float diff1 = xb[j            ] - d*values[i1];
                const float diff2 = xb[j+QK_MXFP4/2] - d*values[i2];
                err += weight[j]*diff1*diff1 + weight[j+QK_MXFP4/2]*diff2*diff2;
                qs[j] = (uint8_t)(i1 | (i2 << 4));
            }
            if (best_err < 0 || err < best_err) {
                best_err = err;
                best_e   = e;
                std::memcpy(best_qs, qs, sizeof(qs));
            }
        }
        y[ib].e = best_e;
        std::memcpy(y[ib].qs, best_qs, sizeof(best_qs));
    }
}
}

void quantize_row_mxfp4_ref(const float * x, block_mxfp4 * y, int64_t k) {
    quantize_mxfp4(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_mxfp4(const float * x, void * y, int64_t k) {
    quantize_mxfp4(x, (void *)y, 1, k, nullptr, nullptr);
}

size_t quantize_mxfp4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
         [[maybe_unused]] const quantize_user_data * user_data) {
    // PXA_PXQ_IMX (DEFAULT OFF, 2026-08-24): the MXFP4 weighted-exponent imatrix path is
    // measured net-negative under the PXQ backbone (bisect: removing it alone recovered
    // 8.3420 -> 8.3231 on Ornith-9B, still above the 8.3076 no-imatrix artifact). An offered
    // imatrix is ignored unless PXA_PXQ_IMX=1 (same opt-in as the PXQ tiers; notice printed
    // once by the PXQ gate or here, whichever consumes first).
    if (imatrix && !(getenv("PXA_PXQ_IMX") && atoi(getenv("PXA_PXQ_IMX")) != 0)) {
        static const bool warned = [](){
            fprintf(stderr, "mxfp4: imatrix IGNORED (measured net-negative under the PXQ backbone; PXA_PXQ_IMX=1 to consume it)\n");
            return true;
        }();
        (void)warned;
        imatrix = nullptr;
    }

    constexpr int kBlockSize = QK_MXFP4;
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    auto row_size = ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
    char * qrow = (char *)dst;
    float weight[kBlockSize];
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_mxfp4_impl(n_per_row, src, qrow, weight, kvalues_mxfp4, imatrix, 7);
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_mxfp4(const block_mxfp4 * x, float * y, int64_t k) {
    constexpr int kBlockSize = QK_MXFP4;
    GGML_ASSERT(k%kBlockSize == 0);
    int nblock = k/kBlockSize;
    for (int ib = 0; ib < nblock; ++ib) {
        float d = GGML_E8M0_TO_FP32_HALF(x[ib].e);
        for (int j = 0; j < kBlockSize/2; ++j) {
            y[j             ] = d * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            y[j+kBlockSize/2] = d * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        y  += kBlockSize;
    }
}

void  vec_dot_mxfp4_q8_0_x4(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    GGML_ASSERT(n%QK_MXFP4 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // PXA 2026-08-20: this function had NO implementation without ik. The body below the ik
    // early-return was entirely commented out, so it fell off the end WITHOUT EVER ASSIGNING
    // *s -- the caller then read an uninitialized stack slot. That is why the quant harness
    // reported "mxfp4 dot product error: FAILED (inf)" and why a no-AVX2 build produced
    // non-finite logits: not a wrong answer, no answer at all. Our PXA-Coder-35B-v2 and
    // PXA-Agent-9B PXQ4 files carry 30 and 24 mxfp4 tensors respectively, so this was reachable
    // with our own shipped models on any AVX2-less CPU path.
    //
    // Semantics taken from dequantize_row_mxfp4 above (the authoritative reference in this file):
    //   d           = GGML_E8M0_TO_FP32_HALF(e)
    //   value[j]           = kvalues_mxfp4[qs[j] & 0xf]      (low nibble  -> first half)
    //   value[j+QK/2]      = kvalues_mxfp4[qs[j] >>  4]      (high nibble -> second half)
    // The activation side is Q8_0_X4: four blocks interleaved as { half d[4]; int8 qs[4*32] },
    // with any tail past 4*(nb/4) stored as plain block_q8_0 -- the same split the x4 quantizers
    // in this file write.
    // Without ik this routes to OUR implementation in pxq-cpu.c, against STOCK block_q8_0
    // activations (see the GGML_TYPE_MXFP4 traits in ggml.c). Nothing on this path is ik's:
    // not the format, not the code. That matters because our own PXQ4 files carry mxfp4
    // tensors, so we were carrying the exposure for a dependency we are removing.
    pxa_mxfp4_dot_q8_0(n, s, vx, vy);
    return;
    //const block_mxfp4 * x = (const block_mxfp4 *)vx;
    //const block_q8_K  * y = (const block_q8_K    *)vy;
    //int nblock = n/QK_MXFP4;
    //float sumf = 0;
    //for (int ibl = 0; ibl < nblock; ++ibl) {
    //    //int sumi = 0;
    //    auto qy = y[ibl].qs;
    //    auto qx = x[ibl].qs;
    //    float db = d * y[ibl].d;
    //    for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
    //        float dl = db * ((x[ibl].scales[ib] & 254) - 127);
    //        //int ls = (x[ibl].scales[ib] & 254) - 127;
    //        const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
    //        int suml = 0;
    //        for (int j = 0; j < kBlockSize/2; ++j) {
    //            suml += qy[j               ] * values[qx[j] & 0xf]
    //                  + qy[j + kBlockSize/2] * values[qx[j] >>  4];
    //        }
    //        sumf += dl * suml;
    //        //sumi += ls * suml;
    //        qy += kBlockSize;
    //        qx += kBlockSize/2;
    //    }
    //    //sumf += d * y[ibl].d * sumi;
    //}
    //*s = sumf;
}

// ============================== Microsoft BitNet I2_S

void dequantize_row_ms_i2s(const void * vx, float * y, int64_t k) {
    constexpr int kBlockSize = 128;
    constexpr int kGroupSize = kBlockSize/4;
    GGML_ASSERT(k % kBlockSize == 0);
    const uint8_t * x = (const uint8_t *)vx;
    const float * dptr = (const float *)(x + k/4);
    const float d = dptr[0];
    int nb = k/kBlockSize;
    for (int ib = 0; ib < nb; ++ib) {
        for (int ig = 0; ig < kBlockSize/kGroupSize; ++ig) {
            int shift = 6 - 2*ig;
            for (int j = 0; j < kGroupSize; ++j) {
                y[j] = d * (((x[j] >> shift) & 3) - 1);
            }
            y += kGroupSize;
        }
        x += kGroupSize;
    }
}

// ============================== Bonsai q1_0_g128

void quantize_row_q1_0_g128_ref(const float * x, block_q1_0_g128  * y, int64_t k) {
    quantize_row_q1_0_g128(x, y, k);
}

void quantize_row_q1_0_g128(const float * x, void * vy, int64_t k) {
    assert(k % QK1_0_G128 == 0);
    int nb = k / QK1_0_G128;
    auto y = (block_q1_0_g128 *)vy;
    for (int ib = 0; ib < nb; ++ib) {
        float sum = 0;
        for (int j = 0; j < QK1_0_G128; ++j) sum += std::abs(x[j]);
        float d = sum / QK1_0_G128;
        y[ib].d = GGML_FP32_TO_FP16(d);
        std::memset(y[ib].qs, 0, QK1_0_G128/8);
        for (int j = 0; j < QK1_0_G128; ++j) {
            if (x[j] >= 0.0f) {
                y[ib].qs[j / 8] |= (1 << (j % 8));
            }
        }
        x += QK1_0_G128;
    }
}

size_t quantize_q1_0_g128(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row % QK1_0_G128 == 0);
    int64_t ntot = nrows * n_per_row;
    quantize_row_q1_0_g128(src, dst, ntot);
    int64_t nblock = ntot / QK1_0_G128;
    return nblock * sizeof(block_q1_0_g128);
}

void dequantize_row_q1_0_g128(const block_q1_0_g128  * x, float * y, int64_t k) {
    assert(k % QK1_0_G128 == 0);
    constexpr uint8_t k_mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    int nb = k / QK1_0_G128;
    for (int ib = 0; ib < nb; ++ib) {
        float d = GGML_FP16_TO_FP32(x[ib].d);
        for (int i = 0; i < QK1_0_G128/8; ++i) {
            for (int j = 0; j < 8; ++j) {
                *y++ = x[ib].qs[i] & k_mask[j] ? d : -d;
            }
        }
    }
}

void vec_dot_q1_0_g128_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK1_0_G128 == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int nb = n / QK1_0_G128;

    constexpr uint8_t k_mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

    constexpr int n4 = QK1_0_G128 / QK8_0;

    auto x = (const block_q1_0_g128 *)vx;
    auto y = (const block_q8_0_x4 *)vy;
    int16_t sumi[QK1_0_G128/8];
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        auto dx = GGML_FP16_TO_FP32(x[ib].d);
        auto qx = x[ib].qs;
        auto qy = y[ib].qs;
        for (int k = 0; k < QK1_0_G128/8; ++k) {
            uint8_t bits = qx[k];
            int16_t s = 0;
            for (int j = 0; j < 8; ++j) {
                s += (bits & k_mask[j] ? qy[j] : -qy[j]);
            }
            qy += 8;
            sumi[k] = s;
        }
        auto s = sumi;
        for (int k = 0; k < n4; ++k) {
            float dy = GGML_FP16_TO_FP32(y[ib].d[k]);
            sumf += dx*dy*(s[0] + s[1] + s[2] + s[3]);
            s += 4;
        }
    }
    *s = sumf;
}

// ============================== tensor validation (--check-tensors)

namespace {
template <typename Block>
inline int check_row_for_blocks_256_fp16(int nblock, const Block * x) {
    int nbad = 0;
    for (int ib = 0; ib < nblock; ++ib) {
        float d = GGML_FP16_TO_FP32(x[ib].d);
        if (isnan(d)) ++nbad;
    }
    return nbad;
}
template <typename Block>
bool check_tensor_for_blocks_256_fp16(const ggml_tensor * tensor) {
    int nblock = tensor->ne[0]/QK_K;
    int nbad = 0;
    for (int row = 0; row < ggml_nrows(tensor); ++row) {
        auto x = (const Block *)((const char *)tensor->data + tensor->nb[1]*row);
        nbad += check_row_for_blocks_256_fp16(nblock, x);
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN block scales out of %g blocks in tensor %s\n", __func__,
                nbad, 1.*ggml_nrows(tensor)*nblock, tensor->name);
        if (tensor->ne[2] > 1) {
            int nb = tensor->ne[0]/QK_K;
            for (int64_t i02 = 0; i02 < tensor->ne[2]; ++i02) {
                int nbad_expert = 0;
                auto xex = (const char *)((const char *)tensor->data + i02*tensor->nb[2]);
                for (int64_t i01 = 0; i01 < tensor->ne[1]; ++i01) {
                    auto xr = (const Block *)(xex + i01*tensor->nb[1]);
                    nbad_expert += check_row_for_blocks_256_fp16(nb, xr);
                }
                if (nbad_expert > 0) fprintf(stderr,"    there are %d NaN block scales for expert %g\n", nbad_expert, 1.*i02);
            }
        }
        return false;
    }
    return true;
}
template <typename Block>
inline int check_row_for_blocks_256_fp16(int nblock, const Block * x, int nr) {
    int nbad = 0;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int j = 0; j < nr; ++j) {
            if (!isfinite(GGML_FP16_TO_FP32(x[ib].d[j]))) ++nbad;
        }
    }
    return nbad;
}
template <typename Block, int nr>
bool check_tensor_for_blocks_256_fp16_repacked(const ggml_tensor * tensor) {
    int nblock = tensor->ne[0]/QK_K;
    int nbad = 0;
    for (int row = 0; row < ggml_nrows(tensor); row += nr) {
        auto x = (const Block *)((const char *)tensor->data + tensor->nb[1]*row);
        nbad += check_row_for_blocks_256_fp16(nblock, x, nr);
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN block scales out of %g blocks in tensor %s\n", __func__,
                nbad, 1.*ggml_nrows(tensor)*nblock, tensor->name);
        if (tensor->ne[2] > 1) {
            int nb = tensor->ne[0]/QK_K;
            for (int64_t i02 = 0; i02 < tensor->ne[2]; ++i02) {
                int nbad_expert = 0;
                auto xex = (const char *)((const char *)tensor->data + i02*tensor->nb[2]);
                for (int64_t i01 = 0; i01 < tensor->ne[1]; i01 += nr) {
                    auto xr = (const Block *)(xex + i01*tensor->nb[1]);
                    nbad_expert += check_row_for_blocks_256_fp16(nb, xr, nr);
                }
                if (nbad_expert > 0) fprintf(stderr,"    there are %d NaN block scales for expert %g\n", nbad_expert, 1.*i02);
            }
        }
        return false;
    }
    return true;
}
struct F32Scale {
    static inline int check_row(const char * data) {
        float d = *(const float *)data;
        return isfinite(d) ? 0 : 1;
    }
};
struct F16Scale {
    static inline int check_row(const char * data) {
        float d = GGML_FP16_TO_FP32(*(const ggml_half *)data);
        return isfinite(d) ? 0 : 1;
    }
};
template <int nr>
struct F32ScaleRX {
    static inline int check_row(const char * data) {
        auto d = (const float *)data;
        int nbad = 0;
        for (int i = 0; i < nr; ++i) {
            if (!isfinite(d[i])) ++nbad;
        }
        return nbad;
    }
};
template <int nr>
struct F16ScaleRX {
    static inline int check_row(const char * data) {
        auto d = (const ggml_half *)data;
        int nbad = 0;
        for (int i = 0; i < nr; ++i) {
            if (!isfinite(GGML_FP16_TO_FP32(d[i]))) ++nbad;
        }
        return nbad;
    }
};
template <typename RS>
bool check_tensor_row_scales(const ggml_tensor * tensor) {
    auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    int num_rows = ggml_nrows(tensor);
    auto data = (const char *)tensor->data;
    int nbad = 0;
    for (int row = 0; row < num_rows; ++row) {
        nbad += RS::check_row(data);
        data += row_size;
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN row scales out of %d rows in tensor %s\n", __func__,
                nbad, num_rows, tensor->name);
        return false;
    }
    return true;
}
}

// PXQ slab types: one fp16 ROW ANCHOR per row, packed as a 128 B header at the head of every
// 64-row panel. The anchor is a row |absmax|, so it is finite and non-negative by construction --
// anything else means the panel was never written by the codec (stale buffer) or a scale
// overflowed fp16, and the rows in it decode to NaN. There was no PXQ arm here at all, which is
// why --check-tensors passed the corrupt 2026-07-22 PXA-Fusion2-35B tiers. Matching guard on the
// producing side: pxa_pxq_check_anchors() in src/llama-quantize.cpp.
static bool pxa_check_pxq_anchors(const ggml_tensor * tensor) {
    const int64_t K = tensor->ne[0];
    const int64_t R = tensor->ne[1];
    const int64_t E = tensor->ne[2]*tensor->ne[3];
    if (R % 64 != 0 || K % ggml_blck_size(tensor->type) != 0) return true;   // not slab geometry
    const int64_t rms         = ggml_internal_get_type_traits(tensor->type).row_meta_size;
    const int64_t panel_bytes = 64*rms + (K/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type)*64;
    const int64_t panels      = (R/64)*E;
    const char *  base        = (const char *) tensor->data;
    int64_t nbad = 0;
    for (int64_t p = 0; p < panels; ++p) {
        const ggml_half * hdr = (const ggml_half *)(base + p*panel_bytes);
        for (int r = 0; r < 64; ++r) {
            const float a = GGML_FP16_TO_FP32(hdr[r]);
            if (!isfinite(a) || a < 0.0f) ++nbad;
        }
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %lld non-finite/negative fp16 row anchors out of %lld rows in "
                        "PXQ tensor %s -- those panels decode to NaN\n", __func__,
                (long long) nbad, (long long) panels*64, tensor->name);
        return false;
    }
    return true;
}

bool pxa_validate_tensor(const ggml_tensor * tensor) {
    if (!tensor) return true;
    if (!ggml_is_contiguous(tensor)) return true;

    switch (tensor->type) {
        case GGML_TYPE_PXQ1:
        case GGML_TYPE_PXQ2:
        case GGML_TYPE_PXQ3:
        case GGML_TYPE_PXQ4:
        case GGML_TYPE_PXQ4HQ:
        case GGML_TYPE_PXQ6:       return pxa_check_pxq_anchors(tensor);
        case GGML_TYPE_IQ2_XXS:    return check_tensor_for_blocks_256_fp16<block_iq2_xxs>(tensor);
        case GGML_TYPE_IQ2_XS:     return check_tensor_for_blocks_256_fp16<block_iq2_xs>(tensor);
        case GGML_TYPE_IQ2_S:      return check_tensor_for_blocks_256_fp16<block_iq2_s>(tensor);
        case GGML_TYPE_IQ3_XXS:    return check_tensor_for_blocks_256_fp16<block_iq3_xxs>(tensor);
        case GGML_TYPE_IQ3_S:      return check_tensor_for_blocks_256_fp16<block_iq3_s>(tensor);
        case GGML_TYPE_IQ4_XS:     return check_tensor_for_blocks_256_fp16<block_iq4_xs>(tensor);
        default: break;
    }
    return true;
}

