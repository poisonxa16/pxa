// pxq4_kernel_launch.h — the CUDA-free launcher declarations consumed by pxq4_kernel_torch.cpp.
// Keeping these in a plain header lets the torch binding TU be compiled by the host compiler
// only; nothing here mentions a __global__ or a device type.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// dequantize the whole tensor: out[N, K] fp16 (row-major) from the PXQ4 two-tensor split.
// N = panels*64, K = kslabs*32. `anchor` points at fp16 data ([panels, 64]).
void pxq4_launch_dequant_f16(const uint8_t * slabs, const void * anchor, void * out,
                             int panels, int kslabs, cudaStream_t stream);

// out[M, N] fp16 = x[M, K] fp16 * W[N, K]^T. Small M only (see PXQ4_MMV_MAX_M).
void pxq4_launch_mmv_f16(const uint8_t * slabs, const void * anchor, const void * x, void * out,
                         int M, int panels, int kslabs, bool vecx, cudaStream_t stream);

// K-chunk-split mmv (decode fast path): identical values to pxq4_launch_mmv_f16 -- same
// per-lane fold, same add order, same single rounding (see k_pxq4_mmv_part) -- but with
// grid.y = nfix so small-N shapes stop starving the SMs. `part` is caller-provided fp32
// scratch of pxq4_mmv_nfix(kslabs) * panels * M * 256 floats.
void pxq4_launch_mmv_split_f16(const uint8_t * slabs, const void * anchor, const void * x,
                               float * part, void * out, int M, int panels, int kslabs,
                               bool vecx, cudaStream_t stream);

// Single-launch fused twin of pxq4_launch_mmv_split_f16: the reduce runs in whichever block of
// a (panel, token) arrives last, so there is one launch instead of two. Identical values (the
// atomic is an arrival counter, never an accumulator -- see k_pxq4_mmv_fused). `ctr` is
// caller-provided scratch of M * panels unsigned, ZERO on entry; a completed launch leaves it
// zero. Requires nfix >= 2 and exactly one PXQ4 mmv in flight per device.
void pxq4_launch_mmv_fused_f16(const uint8_t * slabs, const void * anchor, const void * x,
                               float * part, unsigned * ctr, void * out, int M, int panels,
                               int kslabs, bool vecx, cudaStream_t stream);

// canonical chunk count for this K (= grid.y of the split mmv; sizes `part`).
int  pxq4_mmv_nfix(int kslabs);

// dynamic shared-memory bytes the mmv needs for this K, and whether that fits the device.
int  pxq4_mmv_smem_bytes(int kslabs);
bool pxq4_mmv_supported(int kslabs);

// overwrite the device-resident book / sublevel tables (16 floats each) on the CURRENT device.
// Must be called in eager mode, before any cuda-graph capture: it is a cudaMemcpyToSymbol.
void pxq4_upload_tables(const float * book16, const float * sub16);

// read the tables back for validation (G6).
void pxq4_download_tables(float * book16, float * sub16);

// v6: multi-token fused split mmv — one block owns all M tokens of its (chunk, panel), so a
// decode batch of M <= 8 reads each weight byte once instead of M times. Values per token are
// bit-identical to pxq4_launch_mmv_f16. `part` as the split mmv (M*panels*nfix*256 floats);
// `ctr` is panels unsigned, zero on entry and exit. Requires nfix >= 2 and
// pxq4_mmv_mt_supported(kslabs, M).
void pxq4_launch_mmv_fused_mt_f16(const uint8_t * slabs, const void * anchor, const void * x,
                                  float * part, unsigned * ctr, void * out, int M, int panels,
                                  int kslabs, bool vecx, cudaStream_t stream);
int  pxq4_mmv_mt_smem_bytes(int kslabs, int M);
bool pxq4_mmv_mt_supported(int kslabs, int M);
