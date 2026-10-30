#pragma once
// Host-side atomics for the small lock-free caches in the header-only PXA kernels (kq-mmv.cuh).
// GCC/clang: the __atomic builtins, token for token what the callers used before (identical code on Linux).
// MSVC (Windows build): the Interlocked intrinsics (full barriers, at least as strong as acquire/acq_rel).
#include <stdint.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define PXA_ATOMIC_LOAD_ACQ_U64(p) ((uint64_t)_InterlockedCompareExchange64((volatile __int64 *)(p), 0, 0))
static inline bool pxa_atomic_cas_u64_msvc(uint64_t * p, uint64_t * expected, uint64_t desired) {
    const __int64 e = (__int64)*expected;
    const __int64 o = _InterlockedCompareExchange64((volatile __int64 *)p, (__int64)desired, e);
    if (o == e) return true;
    *expected = (uint64_t)o;
    return false;
}
#define PXA_ATOMIC_CAS_U64(p, expp, desired) pxa_atomic_cas_u64_msvc((p), (expp), (desired))
#else
#define PXA_ATOMIC_LOAD_ACQ_U64(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define PXA_ATOMIC_CAS_U64(p, expp, desired) __atomic_compare_exchange_n((p), (expp), (desired), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
#endif
