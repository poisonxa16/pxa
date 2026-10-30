// ggml-cuda-p60.h -- test hooks of the P60 (sm_60) persistent layer halves (ggml-cuda/pxa/p60-mega*.h, pxqn-mega.cu).
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// The one-token PXQN4 MoE half of PXA_P60_MEGA (closed library) against the reference reimplementation of the normal
// decode path on pseudo-random inputs of the 35B-A3B shape (seed), bitwise: returns the differing words of x_out
// (0 = bit-identical), -1 = no PXQN library in this tree, -2 = a non-finite output value, -3 = a CUDA error.
// *v1_diff = the same count for the PXA_P60_MEGA_V1 kernel (information); out_us[3] = microseconds per call of
// default / V1 / reference over reps, out_us[3 + 7 w + i] = phase i (P0, P1, P2pro, P2, P3pro, P3, P4) of kernel w in us.
// Needs a CUDA device (tests/test-pxa-p60-halves.cpp).
GGML_API GGML_CALL int ggml_backend_cuda_p60_mega_selftest(int device, unsigned seed, int reps, int * v1_diff, float * out_us /* [24] */);

// The verify-width check: is token t of an n_tokens-token (2..6, speculative verify) call of a P60
// half bit-identical to the plain one-token decode of that token? op 0 = the PXQN4 MoE half (pxt2 at n_tokens and its
// ADD-folded call against the one-token instance), op 1 = the PXQN4 GDN half (v2 and v1 multi-token kernels against
// n_tokens chained one-token calls: x_out, final state, per-step conv / ssm captures). diffs[8] = differing words per
// output (layout in pxqn-mega.cu), out_us[32] = microseconds per call and per phase. Returns the sum of diffs (0 =
// bit-identical), -1 = no PXQN library, -2 = non-finite output, -3 = CUDA error, -4 = not an sm_60 card / bad op or width.
GGML_API GGML_CALL int ggml_backend_cuda_p60_vw_selftest(int device, int op, unsigned seed, int n_tokens, int reps, int * diffs /* [8] */, float * out_us /* [32] */);

#ifdef __cplusplus
}
#endif
