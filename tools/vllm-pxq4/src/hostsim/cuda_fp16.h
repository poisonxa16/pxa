// hostsim/cuda_fp16.h — CUDA shim so that pxq4_kernel.cuh can be compiled AND EXECUTED by a
// plain host C++ compiler, with no CUDA toolkit and no GPU.
//
// WHY THIS EXISTS. The single largest risk in this port is a silent transcription error in the
// vendored device code: PXQ4's panel/slab addressing has no redundancy, so a wrong nibble
// order or a wrong eff index produces plausible-looking numbers rather than a crash. The
// project's gates G1/G3 are CPU-only by design, but they only exercise a numpy reference —
// they cannot see the .cuh. This shim closes that hole: the *actual* kernel source is compiled
// and run on the CPU, block by block, with real threads and a real barrier, so a numpy-vs-
// kernel-source bit-exactness test can run on any machine before the GPU lease is ever taken.
//
// It is a TEST harness. It is never compiled into the shipped .so.
//
// Placed on the include path ahead of the real CUDA headers via -Ihostsim, so
// `#include <cuda_fp16.h>` inside pxq4_kernel.cuh resolves here.

#pragma once

#ifdef __CUDACC__
#error "hostsim/cuda_fp16.h must never be on the include path of a real nvcc build"
#endif

#include <cstdint>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------------------------
// qualifiers
// ---------------------------------------------------------------------------------------------
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __restrict__ __restrict
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)

// __shared__ becomes a function-local static. Correct here because the emulator runs exactly
// one block at a time (blocks are sequential, threads within a block are concurrent), which is
// also why a single static instance can stand in for per-block storage.
#define __shared__ static

// ---------------------------------------------------------------------------------------------
// vector types
// ---------------------------------------------------------------------------------------------
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct uint4  { uint32_t x, y, z, w; };
struct dim3   { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };

static inline float2 make_float2(float a, float b) { float2 r; r.x = a; r.y = b; return r; }

// ---------------------------------------------------------------------------------------------
// fp16. Software conversion rather than _Float16 so the harness has no ISA or compiler-version
// dependency; both directions implement IEEE-754 binary16 with round-to-nearest-even, which is
// what the hardware instructions do.
// ---------------------------------------------------------------------------------------------
struct __half {
    uint16_t x_;
    __half() = default;
    // real CUDA __half has a converting constructor from float; k_pxq4_dequant_matrix relies
    // on it via the `(dst_t)(e * v.x)` cast when dst_t == __half
    __half(float f);
};

static inline float __half2float(__half h) {
    const uint32_t s = (uint32_t)(h.x_ >> 15) & 0x1u;
    const uint32_t e = (uint32_t)(h.x_ >> 10) & 0x1fu;
    const uint32_t m = (uint32_t)(h.x_) & 0x3ffu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s << 31;                                  // +/- 0
        } else {
            // subnormal half -> normal float: renormalise
            uint32_t mm = m, ee = 0;
            while ((mm & 0x400u) == 0) { mm <<= 1; ++ee; }
            mm &= 0x3ffu;
            bits = (s << 31) | ((127u - 15u - ee + 1u) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        bits = (s << 31) | 0x7f800000u | (m << 13);          // inf / nan
    } else {
        bits = (s << 31) | ((e + 127u - 15u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline __half __float2half_rn(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t s = (bits >> 31) & 0x1u;
    int32_t        e = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t       m = bits & 0x7fffffu;
    __half out;
    if (((bits >> 23) & 0xffu) == 0xffu) {                   // inf / nan
        out.x_ = (uint16_t)((s << 15) | 0x7c00u | (m ? 0x200u | (m >> 13) : 0u));
        return out;
    }
    if (e >= 31) {                                           // overflow -> inf
        out.x_ = (uint16_t)((s << 15) | 0x7c00u);
        return out;
    }
    if (e <= 0) {                                            // subnormal or zero
        if (e < -10) { out.x_ = (uint16_t)(s << 15); return out; }
        m |= 0x800000u;                                      // restore implicit 1
        const uint32_t shift = (uint32_t)(14 - e);           // 24-bit significand -> 10 bits
        const uint32_t half  = 1u << (shift - 1);
        const uint32_t lower = m & ((1u << shift) - 1);
        uint32_t q = m >> shift;
        if (lower > half || (lower == half && (q & 1u))) ++q; // round half to even
        out.x_ = (uint16_t)((s << 15) | q);
        return out;
    }
    const uint32_t lower = m & 0x1fffu;                      // 13 dropped bits
    uint32_t q = m >> 13;
    if (lower > 0x1000u || (lower == 0x1000u && (q & 1u))) { // round half to even
        ++q;
        if (q == 0x400u) { q = 0; ++e; if (e >= 31) { out.x_ = (uint16_t)((s << 15) | 0x7c00u); return out; } }
    }
    out.x_ = (uint16_t)((s << 15) | ((uint32_t)e << 10) | q);
    return out;
}

inline __half::__half(float f) : x_(__float2half_rn(f).x_) {}

static inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }

// The kernel's dynamic shared-memory declaration. Real CUDA spells it
// `extern __shared__ __align__(16) float pxq4_xs[];`, which cannot survive `__shared__ ->
// static`, so the .cuh emits it through this macro instead. The GPU spelling lives in
// pxq4_kernel.cuh; the host spelling is a plain extern reference to a global arena defined by
// the simulator.
#define PXQ4_EXTERN_SHARED extern __align__(16)

// ---------------------------------------------------------------------------------------------
// launch context. threadIdx/blockIdx are per-thread; blockDim/gridDim are per-launch.
// pxq4_hostsim_launch() (pxq4_kernel_hostsim.cpp) sets these up.
// ---------------------------------------------------------------------------------------------
extern thread_local dim3 threadIdx;
extern thread_local dim3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;

void __syncthreads();
