#pragma once
// PXA_JIT_FIRST_START_v1: does this build carry native GPU code (SASS) a device can run, or will
// the CUDA driver JIT-compile every kernel from the embedded PTX on first start?
//
// A release build ships SASS for a fixed arch list plus PTX for the newest one. A card newer than
// every SASS arch (H100 sm_90, B200 sm_100, RTX 50 sm_120) gets its kernels compiled by the driver
// from PTX: minutes on the first start, cached afterwards (CUDA_CACHE_PATH, CUDA_CACHE_MAXSIZE).
// The server's wedge watchdog must not mistake that compile for a hung forward pass.
//
// Header-only and CUDA-free so tests/test-pxa-jit-wedge.cpp checks it without a GPU.

#include <cstdlib>

// The SASS arch list this build compiled, as one token: the "real" entries of
// CMAKE_CUDA_ARCHITECTURES joined by '_' ("60_61_70_75_86_89"; "90a" kept with its suffix).
// Empty (macro undefined) = unknown ("native", "all", a hand-written -gencode build): then only
// CUDA_FORCE_PTX_JIT=1 can say a device runs JIT code.
#define PXA_JIT_STR2(x) #x
#define PXA_JIT_STR(x)  PXA_JIT_STR2(x)
#ifdef PXA_CUDA_SASS_ARCHS
#define PXA_CUDA_SASS_ARCHS_STR PXA_JIT_STR(PXA_CUDA_SASS_ARCHS)
#else
#define PXA_CUDA_SASS_ARCHS_STR ""
#endif

// "none" = the build has PTX only ("60-virtual"): every card JITs.
// cc = 100*major + 10*minor (ggml's convention; 900 = sm_90, 1200 = sm_120).
// -> 1  no SASS in `sass_list` runs on this device: the driver JITs from PTX
//    0  a SASS arch runs natively (same major, minor <= the device's; an arch-specific "90a"
//       only on exactly that cc)
//   -1  unknown (empty list or a bad cc)
static inline int pxa_jit_needs_ptx(const char * sass_list, int cc) {
    if (sass_list == nullptr || *sass_list == '\0' || cc <= 0) {
        return -1;
    }
    if (sass_list[0] == 'n' && sass_list[1] == 'o' && sass_list[2] == 'n' && sass_list[3] == 'e' && sass_list[4] == '\0') {
        return 1;
    }
    const int dev_major = cc / 100;
    const int dev_minor = (cc % 100) / 10;
    bool any = false;
    const char * p = sass_list;
    while (*p) {
        while (*p && (*p < '0' || *p > '9')) {
            ++p;
        }
        if (!*p) {
            break;
        }
        int v = 0;
        while (*p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            ++p;
        }
        bool arch_specific = false;   // sm_90a / sm_100a / sm_120f: the suffixed forms
        while (*p && *p != '_' && *p != ',' && *p != ';' && *p != ' ') {
            arch_specific = true;
            ++p;
        }
        if (v < 10) {
            continue;
        }
        any = true;
        const int a_major = v / 10;
        const int a_minor = v % 10;
        if (a_major != dev_major) {
            continue;
        }
        if (arch_specific ? (a_minor == dev_minor) : (a_minor <= dev_minor)) {
            return 0;
        }
    }
    return any ? 1 : -1;
}

// CUDA_FORCE_PTX_JIT=1 makes the driver ignore SASS and JIT everything (how a box without a
// newer card reproduces the first-start case). Any non-empty value other than "0" counts.
static inline bool pxa_jit_forced(const char * force_env) {
    return force_env != nullptr && *force_env != '\0' && !(force_env[0] == '0' && force_env[1] == '\0');
}
