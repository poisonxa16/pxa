// pxqn.cuh -- the PXQN entry points of the CUDA backend (ggml-cuda.cu, reduce.cu, convert.cu, mmvq.cu call these).
// PXA_CLOSED_HOOK: declarations only (the forwards are pxqn-stub.cu; the closed library implements them).
// In the open engine they are thin forwards (pxqn-stub.cu) into the closed libggml-pxqn, loaded at run time next to
// libggml; without it every entry point declines (or aborts with ONE clear message where a decline is impossible),
// PXQN files are refused at load, and every other type runs exactly as before. The closed library implements them.
#pragma once

#include "../common.cuh"

#define PXQN_MMV_MAX_NY 32

bool ggml_cuda_pxqn_type(ggml_type t);

void ggml_cuda_pxqn_ctx_init(int device);

int  ggml_cuda_pxqn_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst);

int  ggml_cuda_pxqn_mul_mat_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0a, ggml_tensor * dsta,
                                 const ggml_tensor * src0b, ggml_tensor * dstb, const ggml_tensor * src1);

// rht/mm (PXA_PXQN_RQ8): the down_in PXQN_RHT -> ffn_down MUL_MAT that reads dst, or nullptr
int  ggml_cuda_pxqn_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * rht = nullptr,
                            const ggml_tensor * mm = nullptr);

int  ggml_cuda_pxqn_rht_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * rht, ggml_tensor * mm);
int  ggml_cuda_pxqn_rht_ranges_place(ggml_backend_cuda_context & ctx, const float * x, int n, const int64_t * off,
                                     const int64_t * k0, uint64_t seed, int layer, int site, ggml_tensor * cat,
                                     const ggml_tensor * mm);

// PXA_PXQN_GU_RHT: FUSED_UP_GATE + the down_in PXQN_RHT reading it in one sm_60 launch (0 = both done, -1 = nothing launched)
int  ggml_cuda_pxqn_up_gate_rht(ggml_backend_cuda_context & ctx, ggml_tensor * gu, ggml_tensor * rht, const ggml_tensor * mm);
// PXA_PXQN_GU_DOWN: sm_60 prefill FUSED_UP_GATE -> down_in PXQN_RHT -> down MUL_MAT, the GLU folded into the down GEMM's x pre-pass
// (0 = all three done, 1 = only the up/gate done, -1 = nothing launched)
int  ggml_cuda_pxqn_up_gate_down(ggml_backend_cuda_context & ctx, ggml_tensor * gu, ggml_tensor * rht, ggml_tensor * mm);
// PXA_PXQN_GATE_RHT: the gated attention tail CONT(gate view) + SIGMOID + MUL(attn, .) + out_in PXQN_RHT in one launch (sm_60/61)
int  ggml_cuda_pxqn_gate_tail_rht(ggml_backend_cuda_context & ctx, const ggml_tensor * gview, const ggml_tensor * a,
                                  ggml_tensor * mu, ggml_tensor * rht);
// PXA_PXQN_OG_RR: DeltaNet out-gate + the head-split out_in RHT ranges reading its output in one launch (sm_60
// row-block route); 0 = launched (M, cat and the max sidecar written), -1 = declined (nothing launched)
int  ggml_cuda_pxqn_dn_outgate_rr(ggml_backend_cuda_context & ctx, const float * x, const float * w, const float * z, float * dst,
                                  int ncols, int nrows, float eps, int block_size, int n, const int64_t * off, const int64_t * k0,
                                  uint64_t seed, int layer, int site, ggml_tensor * cat, const ggml_tensor * mm);
bool ggml_cuda_pxqn_rht_supported(const ggml_tensor * op);
void ggml_cuda_op_pxqn_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_pxqn_rms_rht_ok(const ggml_tensor * add, const ggml_tensor * norm, const ggml_tensor * rht);
const char * ggml_cuda_pxqn_q8_lookup(int device, const ggml_tensor * src1, int64_t ne10_padded);
void ggml_cuda_pxqn_rms_rht(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * norm, ggml_tensor * rht,
                            bool want_q8 = true, bool want_x8 = false);   // want_x8: PXA_PXQN_GU_GX

float * ggml_cuda_pxqn_xmax_buf(int device, int64_t ncols, int64_t nrows);
void    ggml_cuda_pxqn_xmax_stamp(int device, const ggml_tensor * rht);
bool    ggml_cuda_pxqn_q8sc_active(int device);
void *  ggml_cuda_pxqn_q8sc_buf(int device, int64_t ncols, int64_t * kpad);
void    ggml_cuda_pxqn_q8sc_stamp(int device, const ggml_tensor * rht);
// PXA_PXQN_RQ8 / PXA_ATTN_TAIL_FUSE producers (pxqn-rq8.cuh): 0 = launched, -1 = the caller runs its own kernels
int     ggml_cuda_pxqn_dn_outgate(ggml_backend_cuda_context & ctx, const float * x, const float * w, const float * z, float * dst,
                                  int ncols, int nrows, float eps, int block_size, const ggml_tensor * rht, const ggml_tensor * mm);
int     ggml_cuda_pxqn_gate_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * gview, const ggml_tensor * a, ggml_tensor * y,
                                 const ggml_tensor * rht, const ggml_tensor * mm);

int  ggml_cuda_pxqn_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// PXQN-C (the codebook expert tier, GGML_TYPE_PXQNC): the library's role "cbook" (ggml-pxqn-ext.h). Without it the
// type is never on a card (the loader refuses the file); a decline at compute time aborts with one message.
bool ggml_cuda_pxqnc_supported(const ggml_tensor * op);
int  ggml_cuda_pxqnc_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
int  ggml_cuda_pxqnc_rht_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * rht, ggml_tensor * const * mm, int n_mm);
// A graph window at node i (a MUL_MAT): the library may compute node i and the nodes after it in one go. It returns the
// index of the last node it computed (their outputs as their own kernels write them), or i (nothing launched: node i runs
// as usual). host = what the engine offers it (below).
struct ggml_cuda_window_host {
    uint32_t size;   // sizeof(struct ggml_cuda_window_host)
    // nodes i and j run on the same device (the engine's split check)
    bool   (*same_device)(const ggml_cgraph * cgraph, int i, int j);
    // t lives in a CUDA buffer of this context's device
    bool   (*on_device)(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
    // the q8_1 buffer the GEMV right after node j would read t's quantized rows from (the engine's record), NULL = none;
    // kx / rows = its row length and row count
    void * (*q8_out)(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int j, const ggml_tensor * t,
                     int64_t * kx, int64_t * rows);
    // t's q8_1 rows were written into the buffer q8_out gave: the engine records them for that GEMV
    void   (*q8_note)(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int64_t kx, int64_t rows);
};
int  ggml_cuda_pxqn_window(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, const ggml_cuda_window_host * host);
// The narrow attention route (decode / verify widths) of one FLASH_ATTN_EXT node: D = head size, G = query heads per KV head,
// NC = columns, type = the K/V type. 0 = the library launched it (dst written), -1 = declined (the engine's own launch runs).
int  ggml_cuda_pxqn_attn_narrow(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int D, int G, int NC, int type);
// PXA_CLOSED_HOOK (2026-11-02): the closed wide (prompt) attention route; -1 = the caller's own route
int  ggml_cuda_pxqn_attn_wide(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// PXA_CLOSED_HOOK (2026-11-03): PF_SLICE producers on the dequant -> f16 cuBLAS route, the library's rule.
//   what 0: may a MUL_MAT whose src0 has ggml_type a be a planned producer -> 1 / 0
//   what 1: slice width of a cuBLAS producer of a reduce of a tokens at compute capability b, c = the engine's forced width
//           (0 = none), d = the most slices the engine can push -> tokens (0 = not sliced)
// Without the library every answer is 0: no cuBLAS producer is planned and its GEMM runs unsliced.
int64_t ggml_cuda_pxqn_pfs_cublas_plan(int what, int64_t a, int64_t b, int64_t c, int64_t d);
struct ggml_cuda_pfs_host {
    uint32_t size;   // sizeof(struct ggml_cuda_pfs_host)
    void *   user;
    // the engine's own GEMM + output conversion on token columns [t0, t0 + n); algo = a cublasGemmAlgo_t, -1 = its default
    void (*gemm_cols)(void * user, int64_t t0, int64_t n, int algo);
    // slice s is final: the engine runs the reduce chain on it and pushes it to the peer
    void (*slice_done)(void * user, int s);
};
// run one planned cuBLAS producer (nslice slices of `slice` tokens over ncols columns; f16out = the f16-output GEMM) on the
// host's callbacks: 0 = done, -1 = declined (nothing launched)
int  ggml_cuda_pxqn_pfs_cublas_run(int cc, int nslice, int64_t slice, int64_t ncols, bool f16out, const ggml_cuda_pfs_host * host);
// PXA_CLOSED_HOOK (2026-11-03): the host role's KQ_MASK_COPY answer for one host -> device copy of a graph
// input (the wide graph's attention mask). mode 1: enqueued on ctx's stream after its earlier work (the host does not block,
// so the copies to the cards overlap) -> 0 = enqueued, -1 = declined (nothing enqueued; the engine copies). The fence makes
// the host wait until every copy enqueued for the device has read its source and returns how many it waited for.
int  ggml_cuda_pxqn_input_copy(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const void * src, size_t size, int mode);
int  ggml_cuda_pxqn_input_fence(int device);
// mode 2 (the copy enqueued on the device's side stream after the card's earlier work): before node i of cgraph runs on ctx's
// stream, the stream waits for every such copy when the graph's first reader of it is near (that node or one of the next few)
void ggml_cuda_pxqn_input_wait(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// A GET_ROWS node: ctx == nullptr asks only (0 = the library serves this node's types, -1 = no; the backend's supports_op);
// otherwise 0 = launched (dst written), -1 = declined.
int  ggml_cuda_pxqn_get_rows(ggml_backend_cuda_context * ctx, const ggml_tensor * dst);
int  ggml_cuda_pxqn_moe_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * down, ggml_tensor * kpad);

struct pxa_ts_task;
// the fused tensor-split reduce epilogue with the PXQN tail (norm + RHT of the next site), launched by the library
void ggml_cuda_pxqn_epi_rht_launch(cudaStream_t stream, size_t smem, const float * pa, const float * pb, float * xout,
                                   const pxa_ts_task & task, int token, int * err_flag, const float * w, float * y,
                                   float * mx, int ncols, float eps, uint64_t seed, int layer, int site, int64_t k0,
                                   int push_payload, void * q8, int kpad);
// PXA_TSPLIT_EPI_MT: the same epilogue at the verify widths (nrows 2..4 rows, no q8_1 sidecar)
void ggml_cuda_pxqn_epi_rht_mt_launch(cudaStream_t stream, size_t smem, const float * pa, const float * pb, float * xout,
                                      const pxa_ts_task & task, int token, int * err_flag, const float * w, float * y,
                                      float * mx, int ncols, int nrows, float eps, uint64_t seed, int layer, int site,
                                      int64_t k0);

// dequant converters (panel -> row-major; nrows multiple of 64, n_per_row of the slab K)
void dequantize_row_pxqn3_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn3_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn3s8_f16(const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn3s8_f32(const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn2_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn2_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn1_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn1_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4s8_f16(const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn4s8_f32(const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn5_f16  (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxqn5_f32  (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxa4_f16   (const void * vx, half  * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_pxa4_f32   (const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
