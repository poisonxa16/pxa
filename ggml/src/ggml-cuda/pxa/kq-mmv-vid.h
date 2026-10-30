#pragma once
// The KQMMV verify-identity width rule as pure host functions (no CUDA), shared by kq-mmv.cu and tests/test-pxa-kqmmv-vid.cpp.
//
// PXA_KQMMV_VERIFY_ID_SMALL is an ON/OFF switch (kernel default 0; the registry sets 1 for the archs where it was measured exact:
// nemotron_h_moe, qwen4exp, qwen35 with an sm_60 device). When on:
//   - a small grouped launch (wide-form widths, < KQMMV_WIDE_MIN_ROWS rows, attention k / v) keeps KQMMV at verify widths instead
//     of declining to the q8_1 MMVQ, and
//   - a launch of at most PXA_KQMMV_VID_SMALL_ROWS weight rows takes the verify-identity rule up to PXA_KQMMV_VERIFY_ID_SMALL_W
//     columns (default 8, never below PXA_KQMMV_VERIFY_ID's cap; 0 = that cap).
// When off, every launch keeps PXA_KQMMV_VERIFY_ID's cap and the small grouped launches decline as before (the 27B W3 MTP text is
// byte-for-byte the previous build's, #28441). The switch's value is never read as a width: =1 means on, not a cap of 1.

#include <cstdint>
#include <cstdlib>

#define PXA_KQMMV_VID_SMALL_ROWS 1024

// PXA_KQMMV_VERIFY_ID_SMALL: any non-zero integer = on; unset / empty / 0 = off
static inline bool pxa_kqmmv_vid_small_parse(const char * e) {
    return e && *e && atoi(e) != 0;
}

// PXA_KQMMV_VERIFY_ID_SMALL_W: unset / empty = 8; <= 0 = 0 (the base cap); clamped to max_ny
static inline int pxa_kqmmv_vid_small_w_parse(const char * e, int max_ny) {
    const int x = e && *e ? atoi(e) : 8;
    return x <= 0 ? 0 : x > max_ny ? max_ny : x;
}

// the widest verify batch the identity rule serves for a launch of R weight rows (R < 0 = unknown)
static inline int pxa_kqmmv_vid_cap(int base, bool small_on, int small_w, int64_t R) {
    return small_on && base > 0 && R > 0 && R <= PXA_KQMMV_VID_SMALL_ROWS && small_w > base ? small_w : base;
}

// a small grouped launch in the wide-form widths keeps KQMMV at verify width ny (the x-staging test is the caller's)
static inline bool pxa_kqmmv_vid_small_grouped(bool small_on, int cap, int64_t ny) {
    return small_on && ny >= 2 && ny <= cap;
}
