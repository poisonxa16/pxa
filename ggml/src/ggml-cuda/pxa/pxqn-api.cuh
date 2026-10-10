// pxqn-api.cuh -- the versioned function table libggml-pxqn hands the open CUDA backend.
// One member per PXQN entry point; the open forwards (pxqn-stub.cu) call through it. The table is refused (and PXQN
// stays unavailable) unless version, size and the host-side layouts the entry points share (the backend context,
// ggml_tensor) match this build exactly, so a library from a different engine build can never be half-used.
#pragma once

#include "pxqn.cuh"
#include "pxq4-rb.cuh"
#include "pxa-tsplit-epi.cuh"
#include "pxa-cold-dev.cuh"
#include "../mmvq-args.h"
#include "../mmvq.cuh"
#include "../../../include/ggml-cuda.h"

void mul_mat_vec_pxqn4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
void mul_mat_vec_pxqn4s8_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
bool mul_mat_vec_pxqn4_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);
bool mul_mat_vec_pxqn4s8_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);

#define GGML_PXQN_CUDA_API_VERSION 2u

#define GGML_PXQN_CUDA_FNS(X) \
    X(ggml_cuda_pxqn_ctx_init) X(ggml_cuda_pxqn_mul_mat) X(ggml_cuda_pxqn_mul_mat_pair) X(ggml_cuda_pxqn_up_gate) \
    X(ggml_cuda_pxqn_rht_mul_mat) X(ggml_cuda_pxqn_rht_ranges_place) X(ggml_cuda_pxqn_rht_supported) \
    X(ggml_cuda_op_pxqn_rht) X(ggml_cuda_pxqn_rms_rht_ok) X(ggml_cuda_pxqn_q8_lookup) X(ggml_cuda_pxqn_rms_rht) \
    X(ggml_cuda_pxqn_xmax_buf) X(ggml_cuda_pxqn_xmax_stamp) X(ggml_cuda_pxqn_q8sc_active) X(ggml_cuda_pxqn_q8sc_buf) \
    X(ggml_cuda_pxqn_q8sc_stamp) X(ggml_cuda_pxqn_mul_mat_id) X(ggml_cuda_pxqn_moe_up_gate) \
    X(ggml_cuda_pxqn_epi_rht_launch) \
    X(dequantize_row_pxqn3_f16) X(dequantize_row_pxqn3_f32) X(dequantize_row_pxqn3s8_f16) X(dequantize_row_pxqn3s8_f32) \
    X(dequantize_row_pxqn4_f16) X(dequantize_row_pxqn4_f32) X(dequantize_row_pxqn2_f16) X(dequantize_row_pxqn2_f32) \
    X(dequantize_row_pxqn1_f16) X(dequantize_row_pxqn1_f32) X(dequantize_row_pxqn4s8_f16) X(dequantize_row_pxqn4s8_f32) \
    X(dequantize_row_pxqn5_f16) X(dequantize_row_pxqn5_f32) \
    X(dequantize_row_pxa4_f16) X(dequantize_row_pxa4_f32) \
    X(ggml_backend_cuda_pxqn_gu_counters_reset) X(ggml_backend_cuda_pxqn_gu_decode_count) \
    X(ggml_backend_cuda_pxqn_gu_prefill_count) \
    X(ggml_cuda_pxq4_rb_mul_mat) X(ggml_cuda_pxq4_rb_up_gate) X(ggml_cuda_pxq4_rb_on) \
    X(mul_mat_vec_pxqn4_q8_1_cuda) X(mul_mat_vec_pxqn4s8_q8_1_cuda) \
    X(mul_mat_vec_pxqn4_q8_1_group_cuda) X(mul_mat_vec_pxqn4s8_q8_1_group_cuda) \
    X(ggml_cuda_pxqn_cold_wait)

struct ggml_pxqn_cuda_api {
    uint32_t version;       // GGML_PXQN_CUDA_API_VERSION
    uint32_t size;          // sizeof(ggml_pxqn_cuda_api)
    uint32_t ctx_size;      // sizeof(ggml_backend_cuda_context)
    uint32_t task_size;     // sizeof(pxa_ts_task)
#define GGML_PXQN_CUDA_MEMBER(f) decltype(&::f) p_##f;
    GGML_PXQN_CUDA_FNS(GGML_PXQN_CUDA_MEMBER)
#undef GGML_PXQN_CUDA_MEMBER
};
