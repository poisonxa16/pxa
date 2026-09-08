#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// PXA-SHARD (M1): expert-shard buffer type — shards a 3D expert tensor on ne[2]
// (expert-id) across a matched device group. Distinct from the row-split
// CUDA_Split type so the MoE op path (M2) can take the disjoint-write path.
// Instantiated ONLY by the M3 loader when PXA_EXPERT_SHARD is set.
GGML_API GGML_CALL ggml_backend_buffer_type_t pxa_expert_shard_buffer_type(const int * group, int n_shard);
// Predicate: is this buffer type an expert-shard type? False for all others
// (incl. the stock CUDA_Split type), so it is a no-op when the flag is off.
GGML_API GGML_CALL bool pxa_buft_is_expert_shard(ggml_backend_buffer_type_t buft);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
// raw compute capability (100*major + 10*minor, e.g. 610 for sm_61); -1 if device is out of range.
GGML_API GGML_CALL int  ggml_backend_cuda_get_device_cc(int device);

// name of the arch dispatch path the PXA tier logic selects for this device (e.g. "sm_61 dp4a");
// reporting only, decides nothing. "" if device is out of range.
GGML_API GGML_CALL const char * ggml_backend_cuda_get_device_pxa_path(int device);

// PXA per-card serve-flag defaults (2026-09-03). Answers "what -b/-ub did this exact card set
// measure fastest at", for the topologies the campaign actually measured. The caller passes the
// values it would otherwise use and only applies what comes back for the flags the USER LEFT
// UNSET -- an explicit -b/-ub must always win.
//
// The table is per CARD SET, measured on the campaign's own cells (dense 27B PXQ4-core on the two
// pairs, PXQ2 on the 1080 Ti). -ub does not transfer between models, which is exactly why the
// caller must only apply this where the user passed nothing: a recipe that states its own -ub
// measured it on its own model and must keep it.
//
// Returns 1 and writes *n_batch / *n_ubatch when the detected topology is a measured cell:
//   2x sm_70 (V100 pair)   -> -b 8192 -ub 2048
//   2x sm_60 (P100 pair)   -> -b 8192 -ub 256
//   1x sm_61 (1080 Ti)     -> -b 2048 -ub 768
// Returns 0 and touches nothing for every other topology (including the 4x P100 seat, which
// passes -b/-ub explicitly), at PXA_REFERENCE=1, and at PXA_ENHANCE=0. *why, if non-NULL, is
// pointed at a short static string naming the cell.
GGML_API GGML_CALL int ggml_backend_cuda_pxa_suggest_batch(int * n_batch, int * n_ubatch, const char ** why);

// Offline PXQ slab dequant (llama-pxq-export). Decodes a contiguous run of 64-row PXQ panels
// from HOST memory to HOST memory with the SAME device kernels the runtime uses
// (ggml_get_to_fp16_cuda / ggml_get_to_fp32_cuda), so an export is bit-identical to what a
// dequant->cuBLAS fallback would have fed the GEMM.
//   src        base of the panel run: tensor data + (row0/64)*panel_stride
//   src_bytes  byte length of that run
//   nrows      rows in the run, multiple of 64 (experts are just more panels: a 3-D PXQ
//              tensor is E * (ne1/64) contiguous panels, so nrows = ne1*ne2*ne3 decodes whole)
//   n_per_row  ne[0], multiple of 32
//   dst_type   GGML_TYPE_F16 or GGML_TYPE_F32; dst holds nrows*n_per_row elements
// Returns false (without touching dst) for a type with no CUDA dequant, a bad device, a
// non-slab-aligned shape, or a device allocation failure.
GGML_API GGML_CALL bool pxa_pxq_dequant_host(int device, enum ggml_type src_type, enum ggml_type dst_type,
                                             const void * src, size_t src_bytes,
                                             int64_t nrows, int64_t n_per_row, void * dst);

// true if every ordered GPU pair can peer-access (P2P) each other, or if there is <=1 device.
// read-only probe (cudaDeviceCanAccessPeer only, NO EnablePeerAccess) — safe to call at model-load time. cached.
GGML_API GGML_CALL bool ggml_backend_cuda_all_pairs_can_peer(void);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);
#ifdef  __cplusplus
}
#endif
