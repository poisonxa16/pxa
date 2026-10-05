// pxq4hq_kernel_launch.h -- CUDA-free launcher declarations for the PXQ4HQ kernels, consumed
// by pxq4hq_torch.cpp. Nothing here names a __global__ or a device type, so the torch binding
// TU is compiled by the host compiler only (the same split as pxq4_kernel_launch.h and
// pxq23_kernel_launch.h).
//
// `tier` is always the GGML TYPE ID (PXQ_TIER_PXQ4HQ = 253) and is taken as an argument even
// though this TU serves exactly one tier, so that a caller cannot reach these functions with a
// tier it never checked: passing anything else is a hard abort, never a silent fallback. A
// wrong slab stride produces a well-formed, completely wrong tensor and nothing downstream can
// see it.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// dequantize a whole tensor: out[N, K] fp16 row-major from the two-tensor split.
// N = panels*64, K = kslabs*32. `anchor` points at fp16 data ([panels, 64]).
void pxq4hq_launch_dequant_f16(int tier, const uint8_t * slabs, const void * anchor, void * out,
                               int panels, int kslabs, cudaStream_t stream);

// out[M, N] fp16 = x[M, K] fp16 * W[N, K]^T. Small M only (see the linear method's cap).
void pxq4hq_launch_mmv_f16(int tier, const uint8_t * slabs, const void * anchor, const void * x,
                           void * out, int M, int panels, int kslabs, bool vecx,
                           cudaStream_t stream);

// dynamic shared-memory bytes the mmv needs for this K, and whether that fits the device.
// Tier-independent (it counts slabs, and every tier has 32 columns per slab).
int  pxq4hq_mmv_smem_bytes(int kslabs);
bool pxq4hq_mmv_supported(int kslabs);

// slab stride in bytes for a tier, or 0 if this build does not serve it.
int  pxq4hq_slab_bytes(int tier);
// entries in this tier's book (16), or 0 for an unknown tier.
int  pxq4hq_book_n(int tier);

// overwrite this tier's device-resident book / sub LUT on the CURRENT device.
// EAGER ONLY -- cudaMemcpyToSymbol; must run before any graph capture.
//
// THE SUB IS THIS TIER'S OWN. PXQ2/PXQ3/PXQ4/PXQ6 share one SUB16 table and one upload path;
// PXQ4HQ indexes a different 16-entry fit (SUB8) because its sub-scale blocks are half the
// size, so uploading a checkpoint's pxq4 sub here would be silent, uniform weight error. The
// two are separate device symbols in separate translation units and neither upload touches
// the other.
void pxq4hq_upload_book(int tier, const float * book, int n);
void pxq4hq_upload_sub(const float * sub8);
void pxq4hq_download_book(int tier, float * book, int n);
void pxq4hq_download_sub(float * sub8);

// Self-test: build a deterministic synthetic panel set, decode it on the HOST from the format
// spec in pxq4hq_kernel_tables.h, run the device kernels on the same bytes and require
// BIT-EXACT agreement on the dequant arm (and 1-ULP on the accumulating mmv arm, for the
// FMA-contraction reason written out at the call site). Returns 0 on pass; a nonzero code and
// a stderr line naming the first mismatch on failure. Cheap: a few hundred KB, two launches.
int  pxq4hq_selftest(int tier);
