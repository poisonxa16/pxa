// win-compat/pxqn-msvc-compat.h -- force-included (/FI) into the C/C++ sources of ggml-pxqn.dll on MSVC only.
// GNU attributes are dropped (exports come from ggml-pxqn.def; the one remaining use is flatten, an inlining hint),
// and clock_gettime(CLOCK_MONOTONIC) is provided over QueryPerformanceCounter.
#pragma once
#if defined(_MSC_VER) && !defined(__clang__)
#define __attribute__(x)
#include <time.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#ifdef __cplusplus
extern "C" {
#endif
__declspec(dllimport) int __stdcall QueryPerformanceCounter(union _LARGE_INTEGER * c);
__declspec(dllimport) int __stdcall QueryPerformanceFrequency(union _LARGE_INTEGER * f);
#ifdef __cplusplus
}
#endif
static __inline int pxa_clock_gettime_mono(struct timespec * ts) {
    static long long freq = 0;
    long long c = 0;
    if (!freq) QueryPerformanceFrequency((union _LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((union _LARGE_INTEGER *) &c);
    ts->tv_sec  = (time_t) (c / freq);
    ts->tv_nsec = (long) ((c % freq) * 1000000000LL / freq);
    return 0;
}
#define clock_gettime(clk, ts) pxa_clock_gettime_mono(ts)
#endif
#endif
