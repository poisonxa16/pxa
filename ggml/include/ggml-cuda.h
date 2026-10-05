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

// PXA_STREAM_WEIGHTS: weights pinned in host RAM, streamed to `device` through a VRAM ring once
// per graph evaluation. set_ring raises the ring size requested for the device (bytes; call before
// the first stream buffer of that device is allocated; PXA_STREAM_RING_MB overrides).
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_stream_buffer_type(int device);
GGML_API GGML_CALL void ggml_backend_cuda_stream_set_ring(int device, size_t bytes);
// true for a CUDA<d>_Stream buffer (pinned host bytes presented as a CUDA buffer)
GGML_API GGML_CALL bool ggml_backend_cuda_buffer_is_stream(ggml_backend_buffer_t buffer);

// Bug #266: size and pre-grow the per-device GEMM temporary pool at context creation.
//   backend_device: the device a CUDA backend runs on (-1 if not a CUDA backend)
//   mul_mat_pool_need: worst-case pool bytes one dense weight's GEMM draws at n_tokens columns
//     (holds_f32_out: the dense PXQ up/gate pair keeps the f32 `up` result in the pool meanwhile)
//   pool_reserve: grow the backend's pool so `bytes` are free; false (no abort) if the card lacks them
GGML_API GGML_CALL int    ggml_backend_cuda_backend_device(ggml_backend_t backend);
GGML_API GGML_CALL size_t ggml_backend_cuda_mul_mat_pool_need(int device, const struct ggml_tensor * w, int64_t n_tokens, bool holds_f32_out);
GGML_API GGML_CALL bool   ggml_backend_cuda_pool_reserve(ggml_backend_t backend, size_t bytes, size_t * pool_bytes);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
// raw compute capability (100*major + 10*minor, e.g. 610 for sm_61); -1 if device is out of range.
GGML_API GGML_CALL int  ggml_backend_cuda_get_device_cc(int device);
// PCIe link width in lanes (Linux sysfs current_link_width; PXA_PCIE_LINK_WIDTH overrides), 0 = unknown
GGML_API GGML_CALL int  ggml_backend_cuda_get_device_pcie_width(int device);

// PXA_P2P_SELFTEST: run the peer-copy self-test for devices a and b now (uncached): 1 = pattern
// intact, 0 = corrupted (PXA_P2P_SELFTEST_CORRUPT=1 forces this), -1 = not testable.
GGML_API int  ggml_backend_cuda_p2p_selftest(int a, int b);
// The cached per-process verdict the peer-access enable path uses (tests each pair once).
GGML_API bool ggml_backend_cuda_p2p_pair_trusted(int a, int b);

// name of the arch dispatch path the PXA tier logic selects for this device (e.g. "sm_61 dp4a");
// reporting only, decides nothing. "" if device is out of range.
GGML_API GGML_CALL const char * ggml_backend_cuda_get_device_pxa_path(int device);

// PXA test hook (bug pxqn-test-fused-blindspot): a GGML_OP_FUSED_UP_GATE graph node can legally
// decompose to per-operand MUL_MAT + GLU AT COMPUTE TIME (a declined shape, PXA_PXQN_GU_MAXNY, the
// sm_70 mma path, ...) while the graph still carries the fused op type -- so a test that only
// inspects the graph (node count / op type) cannot tell whether the real PXQN fused up/gate kernel
// (the PXQN fused up/gate kernel of libggml-pxqn) actually ran. Call
// reset() then read the counts after one ggml_backend_graph_compute(): reporting only, decides
// nothing, and costs one atomic add per real launch (no-op on every other backend/op).
GGML_API GGML_CALL void     ggml_backend_cuda_pxqn_gu_counters_reset(void);
GGML_API GGML_CALL uint64_t ggml_backend_cuda_pxqn_gu_decode_count(void);   // dense GU, ny <= 32 (decode GEMV kernel)
GGML_API GGML_CALL uint64_t ggml_backend_cuda_pxqn_gu_prefill_count(void);  // dense GU, ny > 32 (prefill GEMM kernel)

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
//   2x sm_60 (P100 pair)   -> -b 8192 -ub  256
//   1x sm_61 (1080 Ti)     -> -b 2048 -ub  768
//   4x sm_60 (P100 quad)   -> -b 2048 -ub  256 on a dense file, -ub 2048 on an expert file
// Returns 0 and touches nothing for every other topology, at PXA_REFERENCE=1, and at
// PXA_ENHANCE=0. *why, if non-NULL, is pointed at a short static string naming the cell and,
// where the answer depends on the file, the branch taken and the measurement behind it.
//
// n_expert is the model's expert count read from the file header BEFORE the model is loaded:
// >0 an expert (MoE) file, 0 a dense file, <0 "could not be asked" (the caller must pass -1
// rather than 0 when it does not know, because the two answers differ).
GGML_API GGML_CALL int ggml_backend_cuda_pxa_suggest_batch(int n_expert, int * n_batch, int * n_ubatch, const char ** why);

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

// PXA hot swap -- residency groups (core: ggml/src/pxa-residency.h). While a group is active
// (process-wide: the model on the cards), every weight, KV and compute buffer the CUDA backend
// allocates (and the scratch pools of backends created then) lives in the group, each in its own
// reserved address range. park() frees the VRAM and keeps what must survive in pinned host RAM (weights mirrored
// once, everything that is not a compute buffer copied every time); unpark() maps it back at the
// SAME addresses, one host thread per card. Needs CUDA VMM on every device (supported() says).
struct ggml_cuda_residency_stats {
    double   ms_total;                          // wall time of the call
    double   ms_map;                            // slowest device: create+map (unpark) / unmap (park)
    double   ms_copy;                           // slowest device: copy time
    double   ms_dev[GGML_CUDA_MAX_DEVICES];
    uint64_t bytes_weights;                     // held by the group, per class
    uint64_t bytes_state;
    uint64_t bytes_scratch;
    uint64_t bytes_copied;                      // moved over PCIe by this call
    uint64_t bytes_dev[GGML_CUDA_MAX_DEVICES];
    uint64_t bytes_mirror_new;                  // weight bytes mirrored for the first time
    uint64_t bytes_released;                    // park: physical bytes given back to the cards
    uint64_t bytes_resident;                    // bytes of the group still on the cards after the call
    uint64_t verify_bad;                        // PXA_SWAP_VERIFY=1: weight bytes that changed
    int      n_dev;
    char     err[256];
};
GGML_API GGML_CALL bool   ggml_backend_cuda_residency_supported(void);
GGML_API GGML_CALL void * ggml_backend_cuda_residency_new(const char * name);
GGML_API GGML_CALL void   ggml_backend_cuda_residency_free(void * r);
GGML_API GGML_CALL void * ggml_backend_cuda_residency_bind(void * r);        // make r the active group (process-wide); returns the previous
GGML_API GGML_CALL int    ggml_backend_cuda_residency_load_phase(int on);    // this thread; returns the previous
GGML_API GGML_CALL bool   ggml_backend_cuda_residency_prime(void * r, struct ggml_cuda_residency_stats * st);  // mirror weights now, keep everything mapped
GGML_API GGML_CALL bool   ggml_backend_cuda_residency_park(void * r, struct ggml_cuda_residency_stats * st);
GGML_API GGML_CALL bool   ggml_backend_cuda_residency_unpark(void * r, struct ggml_cuda_residency_stats * st);
// partial park: release only what `next` needs to come in (plus headroom bytes per card it uses)
GGML_API GGML_CALL bool   ggml_backend_cuda_residency_park_for(void * r, void * next, size_t headroom, struct ggml_cuda_residency_stats * st);
GGML_API GGML_CALL void   ggml_backend_cuda_residency_need(void * r, uint64_t * per_dev, int n);   // bytes unpark would map, per card
GGML_API GGML_CALL void   ggml_backend_cuda_residency_sizes(void * r, uint64_t * weights, uint64_t * state, uint64_t * scratch, uint64_t * pinned, uint64_t * resident);
GGML_API GGML_CALL void   ggml_backend_cuda_residency_set_pin_budget(uint64_t bytes);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);
#ifdef  __cplusplus
}
#endif
