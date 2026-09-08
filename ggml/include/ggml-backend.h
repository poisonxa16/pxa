#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
    typedef struct ggml_backend_event * ggml_backend_event_t;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;

    //
    // Backend buffer
    //

    // buffer type
    GGML_API           const char *          ggml_backend_buft_name            (ggml_backend_buffer_type_t buft);
    GGML_API GGML_CALL ggml_backend_buffer_t ggml_backend_buft_alloc_buffer    (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API           size_t                ggml_backend_buft_get_alignment   (ggml_backend_buffer_type_t buft);
    GGML_API           size_t                ggml_backend_buft_get_max_size    (ggml_backend_buffer_type_t buft);
    GGML_API GGML_CALL size_t                ggml_backend_buft_get_alloc_size  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    GGML_API           bool                  ggml_backend_buft_is_host         (ggml_backend_buffer_type_t buft);

    // buffer
    enum ggml_backend_buffer_usage {
        GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    GGML_API           const char *                   ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
    GGML_API           void                           ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
    GGML_API           void *                         ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
    GGML_API           size_t                         ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
    GGML_API GGML_CALL void                           ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API           size_t                         ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
    GGML_API           size_t                         ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);
    GGML_API           size_t                         ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
    GGML_API           void                           ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
    GGML_API           bool                           ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
    GGML_API           void                           ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
    GGML_API           enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage     (ggml_backend_buffer_t buffer);
    GGML_API           ggml_backend_buffer_type_t     ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
    GGML_API           void                           ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);

    //
    // Backend
    //

    GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);
    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
    GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
    GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);

    GGML_API void ggml_backend_tensor_set_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API GGML_CALL void ggml_backend_tensor_set(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API GGML_CALL void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API void ggml_backend_synchronize(ggml_backend_t backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

    GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
    GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst);

    // events
    GGML_API ggml_backend_event_t   ggml_backend_event_new        (ggml_backend_t backend);
    GGML_API void                   ggml_backend_event_free       (ggml_backend_event_t event);
    GGML_API void                   ggml_backend_event_record     (ggml_backend_event_t event);
    GGML_API void                   ggml_backend_event_synchronize(ggml_backend_event_t event);
    GGML_API void                   ggml_backend_event_wait       (ggml_backend_t backend, ggml_backend_event_t event);

    //
    // CPU backend
    //

    GGML_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_API GGML_CALL bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_API           void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_API           void ggml_backend_cpu_set_moe_expert_prefetch(ggml_backend_t backend_cpu, bool enable);
    GGML_API           void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Create a backend buffer from an existing pointer
    GGML_API GGML_CALL ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

    GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);

#ifdef GGML_USE_CPU_HBM
    GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void);
#endif

    //
    // Backend registry
    //

    // The backend registry is a registry of all the available backends, and allows initializing backends in a generic way

    GGML_API size_t                     ggml_backend_reg_get_count(void);
    GGML_API size_t                     ggml_backend_reg_find_by_name(const char * name);
    GGML_API ggml_backend_t             ggml_backend_reg_init_backend_from_str(const char * backend_str); // str is backend_name:params (params is optional)
    GGML_API const char *               ggml_backend_reg_get_name(size_t i);
    GGML_API ggml_backend_t             ggml_backend_reg_init_backend(size_t i, const char * params); // params is backend-specific
    GGML_API ggml_backend_buffer_type_t ggml_backend_reg_get_default_buffer_type(size_t i);
    GGML_API ggml_backend_buffer_t      ggml_backend_reg_alloc_buffer(size_t i, size_t size);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backends to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferrably to run on the same backend as the buffer
        ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
        ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched);
        ggml_backend_sched_graph_compute(sched, graph);

        // if there are graph inputs:
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, graph);
        ggml_backend_tensor_set(input_tensor, ...);
        ggml_backend_sched_graph_compute(sched, graph);
    }
    */

    struct ggml_backend_sched;
    typedef struct ggml_backend_sched * ggml_backend_sched_t;

    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

    // Initialize a backend scheduler
    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel);
    GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);

    GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
    GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);
    GGML_API int                  ggml_backend_sched_get_backend_idx(ggml_backend_sched_t sched, ggml_backend_buffer_t buffer);

    // Get the number of splits of the last graph
    GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_replans(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
    // true when PXA_SCHED_ASYNC_INPUTS is armed: split inputs are copied stream-ordered and
    // the host does NOT block per split, so a graph submitted earlier may still be running on
    // the devices after ggml_backend_sched_graph_compute_async returns, exactly as under
    // n_copies > 1. Anything that reallocates or rewrites what a submitted graph reads must
    // drain first.
    GGML_API bool                 ggml_backend_sched_async_inputs(void);

    // ---- PXA_SYNC_LOG (default OFF, PXA_SYNC_LOG=1) -------------------------------------
    // Host-blocking-sync census. Every blocking sync (ggml_backend_synchronize,
    // ggml_backend_sched_synchronize, ggml_backend_event_synchronize) is timed and charged to
    // the innermost tag pushed by its caller, so one prefill boot says exactly WHICH function
    // blocks the host per ubatch, how often, and for how long. Nested syncs are charged once
    // (the outermost), so sched_synchronize is not double counted against its per-backend calls.
    GGML_API void ggml_backend_sync_tag_push(const char * tag);
    GGML_API void ggml_backend_sync_tag_pop (void);
    GGML_API void ggml_backend_sync_epoch   (void);   // one scheduler graph == one ubatch
    GGML_API void ggml_backend_sync_dump    (const char * label);
    GGML_API bool ggml_backend_sync_log_enabled(void);
    GGML_API void ggml_backend_sync_reset  (void);

    GGML_API size_t               ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);

    GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
    GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);

    // Allocate and compute graph on the backend scheduler
    GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);

    // Reset all assignments and allocators - must be called before changing the node backends
    GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);

    // ---- PXA_KQ_MASK_PAD1 (house custom, DEFAULT OFF) ---------------------------------------
    // The KQ mask input tensor is [n_kv, GGML_PAD(n_tokens,16)] and carries GGML_TENSOR_FLAG_INPUT,
    // so ggml_backend_sched_copy_inputs copies the WHOLE tensor, synchronously, once per split
    // that consumes it. At decode (n_tokens == 1) rows 1..15 are constant -inf padding that the
    // ncols==1 flash-attention vec kernel never reads. These entry points let llama.cpp tell the
    // scheduler how many mask rows are actually live for the ubatch being computed, so the copy
    // can be trimmed to that prefix. Enabled only when PXA_KQ_MASK_PAD1=1 (or 2) is set AND the
    // live row count is exactly 1. With the env unset every one of these is inert.
    GGML_API void                 ggml_backend_pxa_set_kqmask_live_rows(int64_t rows);
    GGML_API int                  ggml_backend_pxa_kqmask_mode(void);        // 0=off 1=trim copy 2=trim copy + skip host pad fill
    GGML_API int64_t              ggml_backend_pxa_kqmask_trim_count(void);  // firing counter: trimmed copies so far
    GGML_API int64_t              ggml_backend_pxa_kqmask_trim_saved_bytes(void);

    // Set a callback to be called for each resulting node during graph compute
    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);

    // enable or disable op offload for a given op
    GGML_API void                 ggml_backend_sched_set_op_offload(ggml_backend_sched_t sched, enum ggml_op op, bool on_or_off);
    GGML_API void                 ggml_backend_sched_set_only_active_experts(ggml_backend_sched_t sched, bool on_or_off);
    GGML_API void                 ggml_backend_sched_set_split_mode_graph(ggml_backend_sched_t sched, bool on_or_off, bool async);
    GGML_API void                 ggml_backend_sched_set_max_extra_alloc(ggml_backend_sched_t sched, int extra_alloc_MiB);

    // prefetch mmap'd MoE expert weights into the page cache
    GGML_API bool                 ggml_backend_prefetch_init(int n_threads);
    GGML_API void                 ggml_backend_prefetch_register_mapping(const void * addr, size_t size);
    GGML_API void                 ggml_backend_prefetch_unregister_mapping(const void * addr);

    //
    // Utils
    //

    struct ggml_backend_graph_copy {
        ggml_backend_buffer_t buffer;
        struct ggml_context * ctx_allocated;
        struct ggml_context * ctx_unallocated;
        struct ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
    GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);

    typedef bool (*GGML_CALL ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data);

    // Tensor initialization
    GGML_API void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
    GGML_API void ggml_backend_view_init(struct ggml_tensor * tensor);

    // ---- PXA arch-defaulting hook -----------------------------------------------------------
    // 1 iff every CUDA device VISIBLE TO THIS PROCESS is Volta-class (7.0 <= cc < 8.0); 0 otherwise
    // (P100/1080 Ti/Ampere+ seats, mixed sets, and CUDA-less builds). Lets a lever measured a win
    // only on sm_70 default ON for the V100 seat while staying OFF everywhere else, from files
    // that have no CUDA device handle. The CUDA backend supplies the real answer; a weak fallback
    // in ggml-backend.cpp returns 0 when the backend is not linked in.
    //
    // NOTE (2026-09-03): no lever currently keys on this. The nine house levers that did now
    // follow ggml_pxa_config_level() instead -- they are host-side and index-math changes with no
    // architecture in them, and this hook was excluding the 4x P100 seat they were measured on.
    // The hook is kept because it is the correct question for a genuinely sm_70-only lever.
    GGML_API int ggml_pxa_cuda_is_volta_only(void);


#ifdef  __cplusplus
}
#endif
