#pragma once

// PXA_XCACHE online adaptation (src/llama-pxa-xcache.cpp): copy stream / events / pinned staging / link probe of the CUDA backend.
// All of it runs on a stream of its own; the compute stream is never touched, so a swap copy never blocks a decode.
// (A header of its own, not part of ggml-cuda.h: common.cuh includes that one, so every CUDA translation unit depends on it.)

#include "ggml.h"

#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

// ---- PXA_XCACHE_ASYNC: the rendezvous slot of one split layer (src/llama-pxa-xcache-async.cpp, GGML_OP_MOE_SPLIT_IDS sides 2 / 3) ----
// One block of pinned, device-mapped host memory per slot; the GPU's submit kernel fills ids / cur and raises `req`, the host worker
// computes the cold experts into `out` and raises `done`, the GPU's wait kernel spins on `done`. Every word the two sides share is
// volatile and sits on a cache line of its own. `seq` counts requests on the device (the submit kernel increments it, the wait kernel
// reads it), so the nodes carry no per-token state and a replayed graph needs no new parameters.
struct ggml_cuda_cold_slot {
    int  device;
    int  n_embd, n_used, n_tok_max;
    // the host's view of the block
    volatile uint64_t * h_req;      // GPU -> host: low 32 bits = number of the last request that has cold work (0 = none yet), high 32 = its width in tokens
    volatile uint32_t * h_done;     // host -> GPU: number of the last request the worker finished
    int32_t           * h_ids;      // ids_cold, compact [n_used x n_tok]
    float             * h_cur;      // activation, compact [n_embd x n_tok]
    float             * h_out;      // result, [n_embd x n_used x n_tok]: row (k, t) is valid when h_ids[k + t*n_used] >= 0
    // the device's view of the same block, and device-memory counters
    volatile uint64_t * d_req;
    volatile uint32_t * d_done;
    int32_t           * d_ids;
    float             * d_cur;
    float             * d_out;
    uint32_t          * d_seq;      // device memory: running request number
    unsigned long long * d_stat;    // device memory: see GGML_CUDA_COLD_STAT_*
};
enum {
    GGML_CUDA_COLD_STAT_WAIT_CLK = 0,   // sum over waits of the SM clocks the wait kernel spent before the host's flag arrived (GPU idle, CPU-bound)
    GGML_CUDA_COLD_STAT_WAITS    = 1,   // waits executed (one per cold-layer node pair)
    GGML_CUDA_COLD_STAT_SKIPPED  = 2,   // requests with no cold slot at all (the host was never involved)
    GGML_CUDA_COLD_STAT_TIMEOUTS = 3,   // waits that gave up (must stay 0: the result of such a layer is wrong)
    GGML_CUDA_COLD_STAT_MAX_CLK  = 4,   // longest single wait
    GGML_CUDA_COLD_STAT_CHK_ROWS = 5,   // PXA_XCACHE_ASYNC_CHECK: rows compared (worker vs scheduler split)
    GGML_CUDA_COLD_STAT_CHK_BAD  = 6,   // rows with any differing bit
    GGML_CUDA_COLD_STAT_CHK_MAXD = 7,   // largest |a - b| seen, as the bits of a positive float
    GGML_CUDA_COLD_STAT_N        = 8
};

// NULL when the device cannot map host memory (or on any allocation failure)
GGML_API GGML_CALL struct ggml_cuda_cold_slot * ggml_backend_cuda_cold_slot_new(int device, int n_embd, int n_used, int n_tok_max);
GGML_API GGML_CALL void   ggml_backend_cuda_cold_slot_free(struct ggml_cuda_cold_slot * slot);
// the device counters (synchronizes: call between decodes / at exit); false on failure
GGML_API GGML_CALL bool   ggml_backend_cuda_cold_slot_stats(struct ggml_cuda_cold_slot * slot, unsigned long long out[GGML_CUDA_COLD_STAT_N]);
// SM clock rate of the device in kHz (to turn the stat clocks into time)
GGML_API GGML_CALL int    ggml_backend_cuda_cold_clock_khz(int device);

GGML_API GGML_CALL void * ggml_backend_cuda_xcache_copy_stream(int device);
// copy n bytes (host pinned <-> device, any direction, UVA addresses) on the device's copy stream, in `chunk`-byte pieces (0 = one piece)
GGML_API GGML_CALL bool   ggml_backend_cuda_xcache_copy_async(int device, void * dst, const void * src, size_t n, size_t chunk);
GGML_API GGML_CALL void * ggml_backend_cuda_xcache_event_record(int device);   // an event behind everything queued on the copy stream so far
GGML_API GGML_CALL int    ggml_backend_cuda_xcache_event_done(void * ev);      // 1 completed, 0 pending, -1 error
GGML_API GGML_CALL void   ggml_backend_cuda_xcache_event_sync(void * ev);
GGML_API GGML_CALL void   ggml_backend_cuda_xcache_event_free(void * ev);
GGML_API GGML_CALL void * ggml_backend_cuda_xcache_pinned_alloc(size_t n);
GGML_API GGML_CALL void   ggml_backend_cuda_xcache_pinned_free(void * p);
// one-time link probe: pinned host -> device and device -> host GB/s (best of a few 16 MiB copies), 0 on failure
GGML_API GGML_CALL void   ggml_backend_cuda_xcache_probe_link(int device, double * h2d_gbs, double * d2h_gbs);

#ifdef  __cplusplus
}
#endif
