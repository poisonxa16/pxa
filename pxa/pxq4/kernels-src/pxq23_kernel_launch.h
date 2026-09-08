// pxq23_kernel_launch.h -- CUDA-free launcher declarations for the PXQ2/PXQ3 kernels,
// consumed by pxq23_torch.cpp. Nothing here names a __global__ or a device type, so the
// torch binding TU is compiled by the host compiler only (same split as
// pxq4_kernel_launch.h).
//
// `tier` is always the GGML TYPE ID (PXQ_TIER_PXQ2 = 254, PXQ_TIER_PXQ3 = 255). Passing a
// tier these functions do not serve is a hard abort, never a silent fallback: a wrong slab
// stride produces a well-formed, completely wrong tensor and nothing downstream can see it.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// dequantize a whole tensor: out[N, K] fp16 row-major from the two-tensor split.
// N = panels*64, K = kslabs*32. `anchor` points at fp16 data ([panels, 64]).
void pxq23_launch_dequant_f16(int tier, const uint8_t * slabs, const void * anchor, void * out,
                              int panels, int kslabs, cudaStream_t stream);

// out[M, N] fp16 = x[M, K] fp16 * W[N, K]^T. Small M only (see PXQ4_MMV_MAX_M).
void pxq23_launch_mmv_f16(int tier, const uint8_t * slabs, const void * anchor, const void * x,
                          void * out, int M, int panels, int kslabs, bool vecx,
                          cudaStream_t stream);

// MoE expert-indexed mmv: out[S, N] = x[S, K] @ W[ids[s]]^T, ids resident on DEVICE (no host
// sync; capture-safe). slabs [E, panels, kslabs, SLAB], anchor [E, panels, 64]. Rows whose id
// is outside [0, E) produce zeros. Values per row are bit-identical to pxq23_launch_mmv_f16
// on that expert's 2-D slice.
void pxq23_launch_moe_mmv_f16(int tier, const uint8_t * slabs, const void * anchor,
                              const void * x, const int32_t * ids, void * out, int S, int E,
                              int panels, int kslabs, bool vecx, cudaStream_t stream);

// dynamic shared-memory bytes the mmv needs for this K, and whether that fits the device.
// Tier-independent (it counts slabs, and every tier has 32 columns per slab), but taken as an
// argument so callers cannot accidentally ask the question without knowing the tier.
int  pxq23_mmv_smem_bytes(int kslabs);
bool pxq23_mmv_supported(int kslabs);

// slab stride in bytes for a tier, or 0 if this build does not serve it.
int  pxq23_slab_bytes(int tier);
// entries in this tier's book (4 for PXQ2, 8 for PXQ3), or 0 for an unknown tier.
int  pxq23_book_n(int tier);

// overwrite this tier's device-resident book on the CURRENT device. `n` must equal
// pxq23_book_n(tier). EAGER ONLY -- cudaMemcpyToSymbol; must run before any graph capture.
void pxq23_upload_book(int tier, const float * book, int n);
// overwrite the SHARED sub-scale LUT (16 floats). Shared by PXQ2/PXQ3 in this TU, exactly as
// the engine shares PXQ6's SUB16 across all three code widths.
void pxq23_upload_sub(const float * sub16);
void pxq23_download_book(int tier, float * book, int n);
void pxq23_download_sub(float * sub16);

// Self-test: build a deterministic synthetic panel set for `tier`, decode it on the host from
// the format spec, run the device kernels on the same bytes, and require BIT-EXACT agreement.
// Returns 0 on pass; a nonzero code (and a stderr line naming the first mismatch) on failure.
// tier 252 is accepted and checks the TEMPLATED PXQ4 instantiation against the shipped
// pxq4_launch_dequant_f16, which is the transcription gate for this whole header family.
int  pxq23_selftest(int tier);
