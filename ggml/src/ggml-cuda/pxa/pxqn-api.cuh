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
#include "p60-mega.h"

void mul_mat_vec_pxqn4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
void mul_mat_vec_pxqn4s8_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
bool mul_mat_vec_pxqn4_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);
bool mul_mat_vec_pxqn4s8_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream);
bool mul_mat_vec_pxqn4_q8_1_group_mix_cuda(mmvq_group_args & ga, mmvq_group_args & gb, ggml_type tb, cudaStream_t stream);

// Table history. A number names exactly ONE table: never reuse a number for a different member list (v4 and v5 were
// each cut twice on separate branches; the union below takes the next free number). Add new members at the END only.
//   3: + ggml_cuda_pxqn_mega_moe_selftest (2026-10-19)
//   4: + ggml_cuda_pxqn_dn_outgate_rr, ggml_cuda_pxqn_up_gate_rht, ggml_cuda_pxqn_gate_tail_rht (rc2.2)
//  (4 on a 35B branch = 3 + the 35B halves; 5 on a 35B verify branch = that + ggml_cuda_pxqn_mega_vw_selftest;
//   5 on a V100 27B branch = rc2.2's 4 + the V100 27B entries: superseded, never ship a library with those tables)
//   6: rc2.2's 4 + ggml_cuda_pxqn_mega_gdn_t, ggml_cuda_pxqn_mega_attn_t, ggml_cuda_pxqn_mega_moe_t3,
//      ggml_cuda_pxqn_mega_attn_tail, ggml_cuda_pxqn_mega_vw_selftest (2026-10-21: union of rc2.2 + the 35B verify branch; the 35B branch is inside it)
//   7: 6 + ggml_cuda_pxqn_dn_outgate, ggml_cuda_pxqn_gate_tail, mul_mat_vec_pxqn4_q8_1_group_mix_cuda (the V100 27B port,
//      merged 2026-10-22); 8: reserved, not cut (p01-35b adds nothing beyond 6)
//   8 (cut 2026-10-22): 7 + ggml_cuda_pxqn_up_gate_down (PXA_PXQN_GU_DOWN; cut on a stream-K branch as "5" = rc2.2's 4 + it,
//      renumbered to the next free number 2026-10-22; never ship a library with that "5" table)
#define GGML_PXQN_CUDA_API_VERSION 8u

#define GGML_PXQN_CUDA_FNS(X) \
    X(ggml_cuda_pxqn_ctx_init) X(ggml_cuda_pxqn_mul_mat) X(ggml_cuda_pxqn_mul_mat_pair) X(ggml_cuda_pxqn_up_gate) \
    X(ggml_cuda_pxqn_rht_mul_mat) X(ggml_cuda_pxqn_rht_ranges_place) X(ggml_cuda_pxqn_rht_supported) \
    X(ggml_cuda_op_pxqn_rht) X(ggml_cuda_pxqn_rms_rht_ok) X(ggml_cuda_pxqn_q8_lookup) X(ggml_cuda_pxqn_rms_rht) \
    X(ggml_cuda_pxqn_xmax_buf) X(ggml_cuda_pxqn_xmax_stamp) X(ggml_cuda_pxqn_q8sc_active) X(ggml_cuda_pxqn_q8sc_buf) \
    X(ggml_cuda_pxqn_q8sc_stamp) X(ggml_cuda_pxqn_mul_mat_id) X(ggml_cuda_pxqn_moe_up_gate) \
    X(ggml_cuda_pxqn_epi_rht_launch) X(ggml_cuda_pxqn_epi_rht_mt_launch) \
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
    X(ggml_cuda_pxqn_cold_wait) X(ggml_cuda_pxqn_mega_moe) X(ggml_cuda_pxqn_mega_gdn) X(ggml_cuda_pxqn_mega_attn) X(ggml_cuda_pxqn_mega_moe_t) \
    X(ggml_cuda_pxqn_pre_vw) X(ggml_cuda_pxqn_post_vw) X(ggml_cuda_pxqn_mega_moe_t_fold) \
    X(ggml_cuda_pxqn_mega_moe_selftest) X(ggml_cuda_pxqn_dn_outgate_rr) X(ggml_cuda_pxqn_up_gate_rht) \
    X(ggml_cuda_pxqn_gate_tail_rht) \
    X(ggml_cuda_pxqn_mega_gdn_t) X(ggml_cuda_pxqn_mega_attn_t) X(ggml_cuda_pxqn_mega_moe_t3) \
    X(ggml_cuda_pxqn_mega_attn_tail) X(ggml_cuda_pxqn_mega_vw_selftest) \
    X(ggml_cuda_pxqn_dn_outgate) X(ggml_cuda_pxqn_gate_tail) X(mul_mat_vec_pxqn4_q8_1_group_mix_cuda) \
    X(ggml_cuda_pxqn_up_gate_down)

struct ggml_pxqn_cuda_api {
    uint32_t version;       // GGML_PXQN_CUDA_API_VERSION
    uint32_t size;          // sizeof(ggml_pxqn_cuda_api)
    uint32_t ctx_size;      // sizeof(ggml_backend_cuda_context)
    uint32_t task_size;     // sizeof(pxa_ts_task)
#define GGML_PXQN_CUDA_MEMBER(f) decltype(&::f) p_##f;
    GGML_PXQN_CUDA_FNS(GGML_PXQN_CUDA_MEMBER)
#undef GGML_PXQN_CUDA_MEMBER
};
