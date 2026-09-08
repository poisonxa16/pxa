// pxq_moe_fused_launch.h -- CUDA-free launcher declarations for the fused MoE decode block,
// consumed by pxq_moe_fused_torch.cpp. Nothing here names a __global__ or a device type, so
// the torch binding TU is compiled by the host compiler only -- the same split
// pxq4_kernel_launch.h and pxq23_kernel_launch.h already use.
//
// THERE ARE TWO SETS, NOT ONE, and the reason is a correctness hazard rather than style.
// pxq4_book_g/pxq4_sub16_g live in pxq4_kernel.cu and pxq2_book_g/pxq3_book_g/pxq23_sub16_g
// live in pxq23_kernel.cu, each as `static __device__` -- i.e. TU-local. A single dispatcher
// compiled into a THIRD TU would silently get its own zero-initialised copies of whichever
// tables its header pulled in, and a checkpoint that uploads a custom book would then decode
// correctly through the old ops and wrongly through the new ones with no error anywhere. So
// the pxq4 instantiations are launched from pxq4_kernel.cu and the pxq2/pxq3 ones from
// pxq23_kernel.cu, and the caller picks by tier. pxq23_kernel.cuh's pxq4_selftest_pol is NOT
// usable here for the same reason: pxq23_upload_book explicitly refuses tier 252
// (pxq23_kernel.cu:153), so that policy reads a book nothing ever writes to.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// ---------------------------------------------------------------------------------------------
// Shapes, once, so the three call sites cannot disagree:
//   M       tokens in this decode step (<= PXQ4_MMV_MAX_M)
//   top_k   experts per token
//   S       = M * top_k, the flattened (token, slot) row count
//   E       experts
//   H       hidden size          -- w13's contraction axis, w2's output axis
//   Ip      per-rank intermediate -- w13's output half, w2's contraction axis
//   w13     slabs [E, 2*Ip/64, H/32,  SLAB], anchor [E, 2*Ip/64, 64]   (gate half then up half)
//   w2      slabs [E, H/64,    Ip/32, SLAB], anchor [E, H/64,    64]
//   x       [M, H] fp16      act [S, Ip] fp16      out [M, H] fp16      ids [S] int32
//   wts     [S] fp32 router weights, already normalised by select_experts
// ---------------------------------------------------------------------------------------------

// gate GEMV + up GEMV + SwiGLU, one launch. grid = (Ip/64, S).
void pxq4_moe_launch_gateup_glu(const uint8_t * slabs, const void * anchor, const void * x,
                                const int32_t * ids, void * act, int S, int Ip, int E,
                                int panels13, int kslabs, int top_k, bool vecx,
                                cudaStream_t stream);
void pxq23_moe_launch_gateup_glu(int tier, const uint8_t * slabs, const void * anchor,
                                 const void * x, const int32_t * ids, void * act, int S, int Ip,
                                 int E, int panels13, int kslabs, int top_k, bool vecx,
                                 cudaStream_t stream);

// FORM A: down GEMV + router-weighted ascending top_k fold, one launch. grid = (H/64, M).
void pxq4_moe_launch_down_fold(const uint8_t * slabs, const void * anchor, const void * act,
                               const int32_t * ids, const float * wts, void * out, int M,
                               int E, int panels2, int kslabs, int top_k, bool vecx,
                               cudaStream_t stream);
void pxq23_moe_launch_down_fold(int tier, const uint8_t * slabs, const void * anchor,
                                const void * act, const int32_t * ids, const float * wts,
                                void * out, int M, int E, int panels2, int kslabs, int top_k,
                                bool vecx, cudaStream_t stream);

// FORM B: the same values, slot axis kept in the grid. grid = (H/64, S), then a fold pass.
// Bit-identical to form A by construction (same dot, same ascending fp32 fold, same two
// roundings); it exists only so the window can measure blocks-versus-launches instead of
// guessing.
void pxq4_moe_launch_down_part(const uint8_t * slabs, const void * anchor, const void * act,
                               const int32_t * ids, void * dn, int S, int E, int panels2,
                               int kslabs, bool vecx, cudaStream_t stream);
void pxq23_moe_launch_down_part(int tier, const uint8_t * slabs, const void * anchor,
                                const void * act, const int32_t * ids, void * dn, int S, int E,
                                int panels2, int kslabs, bool vecx, cudaStream_t stream);

// tier-independent fold pass for form B. Lives in the pxq23 TU; it touches no tables.
void pxq_moe_launch_slot_fold(const void * dn, const float * wts, void * out, int M, int R,
                              int top_k, cudaStream_t stream);

// dynamic shared-memory bytes for this K, and whether the current device can serve it.
// Tier-independent: it counts SLABS, and every tier has 32 columns per slab.
int  pxq_moe_smem_bytes(int kslabs);
bool pxq_moe_supported(int kslabs);

// bumped whenever the fused kernels' numerics or ABI change, so a PXQ4_LIB mismatch is
// diagnosable rather than mysterious.
int  pxq_moe_fused_version();
