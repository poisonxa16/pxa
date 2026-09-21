// pxq4_mma.h -- host-side interface to the sm_70 tensor-core multi-token PXQ4 GEMM.
//
// ADDITIVE. Nothing in pxq4_kernel.cu / pxq4_kernel.cuh is touched by this path; the only
// edit outside this pair of files is the dispatch hook in pxq4_kernel_torch.cpp's mmv_out
// and the table fan-out in set_tables.
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

// fp32 partial words this shape needs in the split-K arena. Depends on SHAPE ONLY (panels,
// kslabs) and NOT on M, so every CUDA-graph capture size asks for the identical allocation.
int  pxq4_mma_part_floats(int panels, int kslabs);

// Shape/arch admissibility. M must be 1..16; the caller owns the M >= 5 performance policy.
bool pxq4_mma_supported(int panels, int kslabs, int M);

void pxq4_mma_upload_tables(const float * book16, const float * sub16);

void pxq4_launch_mma_f16(const uint8_t * slabs, const void * anchor, const void * x,
                         float * part, void * out, int M, int panels, int kslabs,
                         cudaStream_t stream);
