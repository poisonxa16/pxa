#include "common.cuh"

// PXA: true when ggml_cuda_flash_attn_ext_mma_new() runs dst without reaching one of its asserts
// (or the launcher's, or a NO_DEVICE_CODE instance). pxa/core/fa-route.cu asks it before every new-mma pick.
bool ggml_cuda_fattn_new_mma_serves(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_mma_new(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// PXA: false when no instance of this family that the dispatcher can pick for dst fits the
// per-block shared memory of `device` (fa-route.cu declines the node then). See fattn-new-mma.cu.
bool ggml_cuda_fattn_new_mma_fits(int device, const ggml_tensor * dst);
