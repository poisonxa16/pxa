// pxq-simd.h -- the VNNI intrinsic aliases ggml-quants.c uses, kept after ggml/src/iqk was
// deleted (ik separation, phase 3). This is all that was still live in iqk/iqk_config.h: the
// IQK_IMPLEMENT / HAVE_FANCY_SIMD / IQK_API / IQK_NOINLINE macros it also defined had no
// remaining users once the accelerator and the ik-only quant types were gone.
//
// Provenance: derived from ggml/src/iqk/iqk_config.h by Iwan Kawrakow, MIT licensed.

#pragma once

#if defined __x86_64__ || defined _M_X64
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    #define ggml_mm256_dpbusd_epi32 _mm256_dpbusd_epi32
    #define ggml_mm256_dpwssd_epi32 _mm256_dpwssd_epi32
    #define ggml_mm_dpbusd_epi32    _mm_dpbusd_epi32
#elif defined(__AVXVNNI__)
    #define ggml_mm256_dpbusd_epi32 _mm256_dpbusd_avx_epi32
    #define ggml_mm256_dpwssd_epi32 _mm256_dpwssd_avx_epi32
    #define ggml_mm_dpbusd_epi32    _mm_dpbusd_avx_epi32
#endif
#endif
