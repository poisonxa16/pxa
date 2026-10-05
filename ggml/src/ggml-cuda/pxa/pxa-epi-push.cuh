// pxa-epi-push.cuh -- PXA_TSPLIT_EPI producer push plumbing, shared by the fused reduce (reduce.cu) and the
// GEMVs that produce a tensor-split partial (moved out of pxqn-kern.cuh unchanged).
#pragma once

#include <cstdint>

// PXA_TSPLIT_EPI producer push: the GEMV that produces a tensor-split partial also writes it -- plus
// the folded ADD's other operand, rounded exactly as that ADD node rounds it -- into the peer's staging slot of the
// upcoming fused reduce (reduce.cu), so the link write overlaps the GEMV instead of sitting on the critical path.
// The slot is ctr % PXQN_PUSH_POOL, read from the device-side reduce counter the next reduce kernel will read.
// A separate template instance (PUSH): the plain GEMV instances are untouched.
#define PXQN_PUSH_POOL 4
struct pxqn_push_args {
    float *       y2          = nullptr;   // the peer's staging area reserved for my rank, slot 0
    const float * add         = nullptr;   // the folded ADD's other operand (R floats) or nullptr
    const int *   ctr         = nullptr;   // this device's reduce counter
    long long     slot_floats = 0;         // staging stride per slot, floats
};
static __device__ __forceinline__ void pxqn_push_store(const pxqn_push_args & p, int64_t r, float v) {
    const unsigned c = (unsigned)*(const volatile int *)p.ctr;
    float * d = p.y2 + (long long)(c % (unsigned)PXQN_PUSH_POOL) * p.slot_floats;
    d[r] = p.add ? __fadd_rn(v, p.add[r]) : v;
    __threadfence_system();   // performed on the peer before this kernel ends, so before my arrival token
}
