// pxqn-stub.cu -- the open side of the PXQN split: every PXQN entry point of the CUDA backend
// forwards through the table of the closed libggml-pxqn (pxqn-api.cuh, loaded by ggml-pxqn-loader.cpp). Without the
// library each entry point declines exactly like a declined shape, so the callers' incumbent routes run; the few that
// have no incumbent route abort with ONE message (unreachable in practice: PXQN files are refused at load).
#include "pxqn-api.cuh"
#include "../../ggml-pxqn-api.h"
#include "../../../include/ggml-cuda-p60.h"

#include <cstdio>
#include <cstdlib>

static const ggml_pxqn_cuda_api * pxqn_cuda_load() {
    const ggml_pxqn_lib_api * lib = ggml_pxqn_lib();
    if (!lib || !lib->cuda) return nullptr;
    const ggml_pxqn_cuda_api * t = (const ggml_pxqn_cuda_api *) lib->cuda;
    if (t->version != GGML_PXQN_CUDA_API_VERSION || t->size != sizeof(ggml_pxqn_cuda_api) ||
        t->ctx_size != sizeof(ggml_backend_cuda_context) || t->task_size != sizeof(pxa_ts_task)) {
        // the four leading uint32 fields are the same in every table version, so they can be read from any library
        fprintf(stderr, "pxqn: libggml-pxqn CUDA table v%u (size %u, ctx %u, task %u) does not match this engine's v%u "
                        "(size %u, ctx %u, task %u): the library comes from a different engine build -- PXQN unavailable on "
                        "CUDA; install the libggml-pxqn built with this engine\n",
                (unsigned) t->version, (unsigned) t->size, (unsigned) t->ctx_size, (unsigned) t->task_size,
                (unsigned) GGML_PXQN_CUDA_API_VERSION, (unsigned) sizeof(ggml_pxqn_cuda_api),
                (unsigned) sizeof(ggml_backend_cuda_context), (unsigned) sizeof(pxa_ts_task));
        return nullptr;
    }
    return t;
}

static inline const ggml_pxqn_cuda_api * T() {
    static const ggml_pxqn_cuda_api * t = pxqn_cuda_load();
    return t;
}

[[noreturn]] static void pxqn_missing(const char * where) {
    fprintf(stderr, "%s (%s)\n", GGML_PXQN_MISSING_MSG, where);
    abort();
}

bool ggml_cuda_pxqn_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQN3: case GGML_TYPE_PXQN3S8: case GGML_TYPE_PXQN4: case GGML_TYPE_PXQN2:
        case GGML_TYPE_PXQN1: case GGML_TYPE_PXQN4S8: case GGML_TYPE_PXQN5: case GGML_TYPE_PXA4:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_pxqn_ctx_init(int device) { if (T()) T()->p_ggml_cuda_pxqn_ctx_init(device); }

int ggml_cuda_pxqn_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return T() ? T()->p_ggml_cuda_pxqn_mul_mat(ctx, src0, src1, dst) : -1;
}
int ggml_cuda_pxqn_mul_mat_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0a, ggml_tensor * dsta,
                                const ggml_tensor * src0b, ggml_tensor * dstb, const ggml_tensor * src1) {
    return T() ? T()->p_ggml_cuda_pxqn_mul_mat_pair(ctx, src0a, dsta, src0b, dstb, src1) : -1;
}
int ggml_cuda_pxqn_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * rht, const ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_up_gate(ctx, dst, rht, mm) : -1;
}
int ggml_cuda_pxqn_rht_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * rht, ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_rht_mul_mat(ctx, rht, mm) : -1;
}
int ggml_cuda_pxqn_rht_ranges_place(ggml_backend_cuda_context & ctx, const float * x, int n, const int64_t * off,
                                    const int64_t * k0, uint64_t seed, int layer, int site, ggml_tensor * cat,
                                    const ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_rht_ranges_place(ctx, x, n, off, k0, seed, layer, site, cat, mm) : -1;
}
int ggml_cuda_pxqn_up_gate_rht(ggml_backend_cuda_context & ctx, ggml_tensor * gu, ggml_tensor * rht, const ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_up_gate_rht(ctx, gu, rht, mm) : -1;
}
int ggml_cuda_pxqn_up_gate_down(ggml_backend_cuda_context & ctx, ggml_tensor * gu, ggml_tensor * rht, ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_up_gate_down(ctx, gu, rht, mm) : -1;
}
int ggml_cuda_pxqn_gate_tail_rht(ggml_backend_cuda_context & ctx, const ggml_tensor * gview, const ggml_tensor * a,
                                 ggml_tensor * mu, ggml_tensor * rht) {
    return T() ? T()->p_ggml_cuda_pxqn_gate_tail_rht(ctx, gview, a, mu, rht) : -1;
}
int ggml_cuda_pxqn_dn_outgate_rr(ggml_backend_cuda_context & ctx, const float * x, const float * w, const float * z, float * dst,
                                 int ncols, int nrows, float eps, int block_size, int n, const int64_t * off, const int64_t * k0,
                                 uint64_t seed, int layer, int site, ggml_tensor * cat, const ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_dn_outgate_rr(ctx, x, w, z, dst, ncols, nrows, eps, block_size, n, off, k0, seed, layer, site,
                                                     cat, mm) : -1;
}
bool ggml_cuda_pxqn_rht_supported(const ggml_tensor * op) {
    return T() ? T()->p_ggml_cuda_pxqn_rht_supported(op) : false;
}
void ggml_cuda_op_pxqn_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!T()) pxqn_missing("PXQN_RHT");
    T()->p_ggml_cuda_op_pxqn_rht(ctx, dst);
}
bool ggml_cuda_pxqn_mega_moe(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_moe(ctx, args) : false;
}
int ggml_cuda_pxqn_mega_moe_selftest(int device, unsigned seed, int reps, int * v1_diff, float * out_us) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_moe_selftest(device, seed, reps, v1_diff, out_us) : -1;
}
// the test hook (ggml-cuda-p60.h): -1 without the library
int ggml_backend_cuda_p60_mega_selftest(int device, unsigned seed, int reps, int * v1_diff, float * out_us) {
    return ggml_cuda_pxqn_mega_moe_selftest(device, seed, reps, v1_diff, out_us);
}
int ggml_cuda_pxqn_mega_vw_selftest(ggml_backend_cuda_context & ctx, int op, unsigned seed, int n_tokens, int reps, int * diffs, float * out_us) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_vw_selftest(ctx, op, seed, n_tokens, reps, diffs, out_us) : -1;
}
// the test hook (ggml-cuda-p60.h): a backend context of its own for the call; -1 without the library
int ggml_backend_cuda_p60_vw_selftest(int device, int op, unsigned seed, int n_tokens, int reps, int * diffs, float * out_us) {
    if (!T()) return -1;
    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    const int r = ggml_cuda_pxqn_mega_vw_selftest(*ctx, op, seed, n_tokens, reps, diffs, out_us);
    delete ctx;
    return r;
}
bool ggml_cuda_pxqn_mega_gdn(ggml_backend_cuda_context & ctx, const pxa_p60_gdn_args & args) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_gdn(ctx, args) : false;
}
bool ggml_cuda_pxqn_mega_gdn_t(ggml_backend_cuda_context & ctx, const pxa_p60_gdn_t_args & args, int n_tokens) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_gdn_t(ctx, args, n_tokens) : false;
}
bool ggml_cuda_pxqn_mega_moe_t3(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_moe_t3(ctx, args, n_tokens) : false;
}
bool ggml_cuda_pxqn_mega_attn_t(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args, int n_tokens) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_attn_t(ctx, args, n_tokens) : false;
}
bool ggml_cuda_pxqn_mega_attn_tail(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args, int n_tokens,
                                   const float * q, const float * qf) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_attn_tail(ctx, args, n_tokens, q, qf) : false;
}
bool ggml_cuda_pxqn_mega_attn(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_attn(ctx, args) : false;
}
bool ggml_cuda_pxqn_mega_moe_t(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_moe_t(ctx, args, n_tokens) : false;
}
bool ggml_cuda_pxqn_mega_moe_t_fold(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens,
                                    const float * add_a, const float * add_b, float * add_out) {
    return T() ? T()->p_ggml_cuda_pxqn_mega_moe_t_fold(ctx, args, n_tokens, add_a, add_b, add_out) : false;
}
bool ggml_cuda_pxqn_pre_vw(ggml_backend_cuda_context & ctx, const pxa_p60_pre_args & args, int n_tokens) {
    return T() ? T()->p_ggml_cuda_pxqn_pre_vw(ctx, args, n_tokens) : false;
}
bool ggml_cuda_pxqn_post_vw(ggml_backend_cuda_context & ctx, const pxa_p60_post_args & args, int n_tokens) {
    return T() ? T()->p_ggml_cuda_pxqn_post_vw(ctx, args, n_tokens) : false;
}
bool ggml_cuda_pxqn_rms_rht_ok(const ggml_tensor * add, const ggml_tensor * norm, const ggml_tensor * rht) {
    return T() ? T()->p_ggml_cuda_pxqn_rms_rht_ok(add, norm, rht) : false;
}
const char * ggml_cuda_pxqn_q8_lookup(int device, const ggml_tensor * src1, int64_t ne10_padded) {
    return T() ? T()->p_ggml_cuda_pxqn_q8_lookup(device, src1, ne10_padded) : nullptr;
}
void ggml_cuda_pxqn_rms_rht(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * norm, ggml_tensor * rht,
                            bool want_q8, bool want_x8) {
    if (!T()) pxqn_missing("fused norm + PXQN_RHT");
    T()->p_ggml_cuda_pxqn_rms_rht(ctx, add, norm, rht, want_q8, want_x8);
}
float * ggml_cuda_pxqn_xmax_buf(int device, int64_t ncols, int64_t nrows) {
    return T() ? T()->p_ggml_cuda_pxqn_xmax_buf(device, ncols, nrows) : nullptr;
}
void ggml_cuda_pxqn_xmax_stamp(int device, const ggml_tensor * rht) { if (T()) T()->p_ggml_cuda_pxqn_xmax_stamp(device, rht); }
bool ggml_cuda_pxqn_q8sc_active(int device) { return T() ? T()->p_ggml_cuda_pxqn_q8sc_active(device) : false; }
void * ggml_cuda_pxqn_q8sc_buf(int device, int64_t ncols, int64_t * kpad) {
    return T() ? T()->p_ggml_cuda_pxqn_q8sc_buf(device, ncols, kpad) : nullptr;
}
void ggml_cuda_pxqn_q8sc_stamp(int device, const ggml_tensor * rht) { if (T()) T()->p_ggml_cuda_pxqn_q8sc_stamp(device, rht); }
int ggml_cuda_pxqn_dn_outgate(ggml_backend_cuda_context & ctx, const float * x, const float * w, const float * z, float * dst,
                              int ncols, int nrows, float eps, int block_size, const ggml_tensor * rht, const ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_dn_outgate(ctx, x, w, z, dst, ncols, nrows, eps, block_size, rht, mm) : -1;
}
int ggml_cuda_pxqn_gate_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * gview, const ggml_tensor * a, ggml_tensor * y,
                             const ggml_tensor * rht, const ggml_tensor * mm) {
    return T() ? T()->p_ggml_cuda_pxqn_gate_tail(ctx, gview, a, y, rht, mm) : -1;
}
int ggml_cuda_pxqn_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    return T() ? T()->p_ggml_cuda_pxqn_mul_mat_id(ctx, dst) : -1;
}
int ggml_cuda_pxqn_moe_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * down, ggml_tensor * kpad) {
    return T() ? T()->p_ggml_cuda_pxqn_moe_up_gate(ctx, dst, down, kpad) : -1;
}
void ggml_cuda_pxqn_epi_rht_launch(cudaStream_t stream, size_t smem, const float * pa, const float * pb, float * xout,
                                   const pxa_ts_task & task, int token, int * err_flag, const float * w, float * y,
                                   float * mx, int ncols, float eps, uint64_t seed, int layer, int site, int64_t k0,
                                   int push_payload, void * q8, int kpad) {
    if (!T()) pxqn_missing("tensor-split epilogue");
    T()->p_ggml_cuda_pxqn_epi_rht_launch(stream, smem, pa, pb, xout, task, token, err_flag, w, y, mx, ncols, eps, seed,
                                       layer, site, k0, push_payload, q8, kpad);
}
void ggml_cuda_pxqn_epi_rht_mt_launch(cudaStream_t stream, size_t smem, const float * pa, const float * pb, float * xout,
                                      const pxa_ts_task & task, int token, int * err_flag, const float * w, float * y,
                                      float * mx, int ncols, int nrows, float eps, uint64_t seed, int layer, int site,
                                      int64_t k0) {
    if (!T()) pxqn_missing("tensor-split epilogue");
    T()->p_ggml_cuda_pxqn_epi_rht_mt_launch(stream, smem, pa, pb, xout, task, token, err_flag, w, y, mx, ncols, nrows, eps,
                                          seed, layer, site, k0);
}

#define PXQN_DEQ_FWD(name, T_OUT) \
    void name(const void * vx, T_OUT * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream) { \
        if (!T()) pxqn_missing(#name); \
        T()->p_##name(vx, y, nrows, n_per_row, stream); \
    }
PXQN_DEQ_FWD(dequantize_row_pxqn3_f16,   half)
PXQN_DEQ_FWD(dequantize_row_pxqn3_f32,   float)
PXQN_DEQ_FWD(dequantize_row_pxqn3s8_f16, half)
PXQN_DEQ_FWD(dequantize_row_pxqn3s8_f32, float)
PXQN_DEQ_FWD(dequantize_row_pxqn4_f16,   half)
PXQN_DEQ_FWD(dequantize_row_pxqn4_f32,   float)
PXQN_DEQ_FWD(dequantize_row_pxqn2_f16,   half)
PXQN_DEQ_FWD(dequantize_row_pxqn2_f32,   float)
PXQN_DEQ_FWD(dequantize_row_pxqn1_f16,   half)
PXQN_DEQ_FWD(dequantize_row_pxqn1_f32,   float)
PXQN_DEQ_FWD(dequantize_row_pxqn4s8_f16, half)
PXQN_DEQ_FWD(dequantize_row_pxqn4s8_f32, float)
PXQN_DEQ_FWD(dequantize_row_pxqn5_f16,   half)
PXQN_DEQ_FWD(dequantize_row_pxqn5_f32,   float)
PXQN_DEQ_FWD(dequantize_row_pxa4_f16,    half)
PXQN_DEQ_FWD(dequantize_row_pxa4_f32,    float)
#undef PXQN_DEQ_FWD

GGML_CALL void ggml_backend_cuda_pxqn_gu_counters_reset(void) { if (T()) T()->p_ggml_backend_cuda_pxqn_gu_counters_reset(); }
GGML_CALL uint64_t ggml_backend_cuda_pxqn_gu_decode_count(void) {
    return T() ? T()->p_ggml_backend_cuda_pxqn_gu_decode_count() : 0;
}
GGML_CALL uint64_t ggml_backend_cuda_pxqn_gu_prefill_count(void) {
    return T() ? T()->p_ggml_backend_cuda_pxqn_gu_prefill_count() : 0;
}

// PXA_PXQ4_RB: the sm_60 one-column PXQ4 row-block GEMV ships in the library too; without it PXQ4 keeps its
// open decode path (correct, the pre-RB speed)
int ggml_cuda_pxq4_rb_mul_mat(int device, cudaStream_t stream, const ggml_tensor * src0, const ggml_tensor * src1,
                              ggml_tensor * dst) {
    return T() ? T()->p_ggml_cuda_pxq4_rb_mul_mat(device, stream, src0, src1, dst) : -1;
}
int ggml_cuda_pxq4_rb_up_gate(int device, cudaStream_t stream, ggml_tensor * dst) {
    return T() ? T()->p_ggml_cuda_pxq4_rb_up_gate(device, stream, dst) : -1;
}
bool ggml_cuda_pxq4_rb_on() { return T() ? T()->p_ggml_cuda_pxq4_rb_on() : false; }

void mul_mat_vec_pxqn4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    if (!T()) pxqn_missing("PXQN4 MMVQ");
    T()->p_mul_mat_vec_pxqn4_q8_1_cuda(args, stream);
}
void mul_mat_vec_pxqn4s8_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    if (!T()) pxqn_missing("PXQN4S8 MMVQ");
    T()->p_mul_mat_vec_pxqn4s8_q8_1_cuda(args, stream);
}
bool mul_mat_vec_pxqn4_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream) {
    return T() ? T()->p_mul_mat_vec_pxqn4_q8_1_group_cuda(g, stream) : false;
}
bool mul_mat_vec_pxqn4s8_q8_1_group_cuda(mmvq_group_args & g, cudaStream_t stream) {
    return T() ? T()->p_mul_mat_vec_pxqn4s8_q8_1_group_cuda(g, stream) : false;
}
bool mul_mat_vec_pxqn4_q8_1_group_mix_cuda(mmvq_group_args & ga, mmvq_group_args & gb, ggml_type tb, cudaStream_t stream) {
    return T() ? T()->p_mul_mat_vec_pxqn4_q8_1_group_mix_cuda(ga, gb, tb, stream) : false;
}

bool ggml_cuda_pxqn_cold_wait(const pxa_cold_dev & s, const char * ids, size_t inb0, size_t inb1, float * dst, size_t dnb1, size_t dnb2,
                              int n_embd, int n_used, int n_tok, long long timeout_clk, const int32_t * ticket, cudaStream_t stream) {
    return T() ? T()->p_ggml_cuda_pxqn_cold_wait(s, ids, inb0, inb1, dst, dnb1, dnb2, n_embd, n_used, n_tok, timeout_clk, ticket, stream) : false;
}
