// -----------------------------------------------------------------------------
// Vendored from ik_llama.cpp, commit 3c58ae37, file
// ggml/src/ggml-cuda/delta-net.cuh. MIT licensed:
//   MIT License, Copyright (c) 2023-2026 the ik_llama.cpp authors
//                Copyright (c) 2023-2026 The ggml authors
// Upstream declares ggml_cuda_op_delta_net; the two _ex entry points below are
// PXA's, and are the interface the DeltaNet decode fusion uses. See delta-net.cu
// for the full provenance note.
// -----------------------------------------------------------------------------
#include "../common.cuh"

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// PXA_FUSE_DELTANET (R2): optional redirect of the new ssm-state write straight into the
// recurrent cache row (fuses away the CONCAT state copy). nullptr = the classic behavior.
void ggml_cuda_op_delta_net_ex(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override);

// PXA_DN_SCATTER_FUSE (2026-09-03): as above, but the destination ROW inside the recurrent state
// buffer is resolved on the DEVICE from state_dst_row_idx (the very index tensor the fused-away
// SET_ROWS would have read), with state_dst_row_stride in ELEMENTS. This lets the safe
// gather/scatter state path write its new state straight into the cache row without the separate
// SET_ROWS copy, while still reading a private gathered copy -- so it does NOT reintroduce the
// read-side aliasing that PXA_DN_NP1_FASTPATH was turned off for. state_dst_row_idx == nullptr is
// exactly ggml_cuda_op_delta_net_ex.
void ggml_cuda_op_delta_net_ex2(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override,
                                const int32_t * state_dst_row_idx, int64_t state_dst_row_stride);
