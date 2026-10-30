// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxq4-mmq80.cuh -- the sm_80+ prefill GEMM route for 4-bit E16 panel weights (PXQ4), declaration.
//
// ROUTE. On sm_80+ a dense 2D PXQ4 MUL_MAT wider than the decode window used to take the generic route:
// dequantize the WHOLE weight into an fp16 pool scratch, convert src1 to fp16, run an fp16 GEMM, convert the
// output back. This route reads the panels directly inside the stock MMQ tile machinery (mmq.cuh: q8_1 activation
// tiles, stream-k, write-back), decoding each 32-wide slab of codes to int8 in the tile loader and running the
// products on the int8 tensor-core tile.
//
// NUMERIC CONTRACT (the frozen q3-s8 snap the PXQ4 decode kernels already ship, pxq4-mmq.cuh):
//   book      q_c = rint(book[c] * 127) (absmax 1.0), SUB16 frozen, fp16 row anchor untouched
//   weights   w   = (anchor * (1/127) * SUB16[s]) * q_c         (one fp32 scale per 16-element group)
//   activations   q8_1 per 32 (absmax/127, round to nearest), the stock MMQ activation tiles
//   products      exact int32 per 16-element group, then fp32: sum += dB * (c0*dA0 + c1*dA1)
// NOT bit-identical to the fp16 route (different arithmetic). Deterministic run to run at a fixed shape.
//
// ARCH: sm_80+ only (int8 m16n8k16). The device bodies are empty below sm_80 and the host entry declines unless
// the device is NVIDIA cc >= 800 AND the binary carries sm_80+ code for it, so sm_60/61/70/75 keep their route.
//
// LEVER: PXA_PXQ_MMQ80 = 0 (default: the previous route, unchanged) | 1 = this route (opt-in; slower than the fp16 route
// on sm_86 as measured 2026-10-22, kept as the base for tile work).
#pragma once

#include "../common.cuh"

// Returns 0 when it computed dst = src0 x src1, -1 when it declined (nothing launched). src0 must hold 4-bit
// E16 panels (64-row panels: 128 B of fp16 row anchors, then K/32 slabs of 64 sub bytes + 64 x 16 B code rows)
// whose int8 book is the frozen q3-s8 book; the caller checks the buffers are plain CUDA buffers of ctx.device.
int ggml_cuda_pxq4_mmq80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                         ggml_tensor * dst);

// the lever (PXA_PXQ_MMQ80) and the device gate, for callers that want to skip work before the call
bool ggml_cuda_pxq4_mmq80_on(int cc);
