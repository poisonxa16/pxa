#pragma once

// PXA: 32-bit fast integer division by a loop-invariant divisor (multiply + shift).
//
// This branch predates the upstream ggml fastdiv helpers in common.cuh, and the copy that
// exists in solve_tri.cu is a stub that still emits a real division.  These are the real
// thing, PXA-prefixed so they cannot collide with either.
//
// n/d == (__umulhi(n, mp) + n) >> L, with mp and L precomputed on the host.
// Valid for 1 <= d <= UINT32_MAX and 0 <= n <= INT32_MAX, which is the range every caller
// here is gated on.  GP100 has no 64-bit integer divider, so a 64-bit division in an index
// expression costs tens of instructions; this costs two.

#include <cstdint>

static inline uint3 pxa_init_fastdiv_values(uint32_t d) {
    // L = ceil(log2(d))
    uint32_t L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }
    const uint32_t mp = (uint32_t) ((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
    // the divisor is packed alongside so the modulo helper needs one argument
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t pxa_fastdiv(uint32_t n, const uint3 fdv) {
    return (__umulhi(n, fdv.x) + n) >> fdv.y;
}

static __device__ __forceinline__ uint32_t pxa_fastmodulo(uint32_t n, const uint3 fdv) {
    return n - pxa_fastdiv(n, fdv) * fdv.z;
}

// returns {n / d, n % d}
static __device__ __forceinline__ uint2 pxa_fast_div_modulo(uint32_t n, const uint3 fdv) {
    const uint32_t q = pxa_fastdiv(n, fdv);
    return make_uint2(q, n - q * fdv.z);
}
