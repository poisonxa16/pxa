// pxa / PXA kernel suite -- p60-mega.h: host-side declarations of the sm_60 persistent layer-half kernels
// (PXA_P60_MEGA) and the fp16 expert GEMM (PXA_P60_MOE_GEMM). Kept free of kernel code so that ggml-cuda.cu does not
// rebuild when the kernels change.
//
// Portions of the sm_60 kernels declared here:
//
//   Copyright: see NOTICE at the root of this tree.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
//   permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
//   the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//   THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
//   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//   IN THE SOFTWARE.
#pragma once

#include <cstdint>

struct ggml_backend_cuda_context;

// MoE half of a Qwen3.5/3.6 MoE layer, one token:
//   x_out = x_in + sum_{e in top8} w_e * down_e(silu(gate_e x) * up_e x) + sigmoid(sgate . x) * down_s(silu(g_s x) * u_s x)
//   with x = rms_norm(x_in) * norm, w = softmax(router x) renormalized over the top 8.
// gate/up types: Q4_K or Q5_K (K = 2048); down: Q5_K or Q6_K (K = 512); shared expert Q8_0; router, sgate f32.
struct pxa_p60_moe_args {
    const float * x_in;   // [2048]
    float *       x_out;  // [2048]
    const float * norm;   // [2048]
    float         eps;
    const float * router; // [256 rows of 2048] f32
    const float * sgate;  // [2048] f32
    const char *  gate;   // [256][512 rows of 2048]
    const char *  up;
    const char *  down;   // [256][2048 rows of 512]
    const char *  sg;     // Q8_0 [512 rows of 2048]
    const char *  su;     // Q8_0
    const char *  sd;     // Q8_0 [2048 rows of 512]
    int           gu_type; // GGML_TYPE_Q4_K / GGML_TYPE_Q5_K / GGML_TYPE_PXQN4
    int           dn_type; // GGML_TYPE_Q5_K / GGML_TYPE_Q6_K / GGML_TYPE_PXQN4
    int64_t       gu_nb2;  // bytes per expert of gate / up (PXQN4: 8 panels)
    int64_t       dn_nb2;  // bytes per expert of down
};

// returns false when the kernel cannot run (unsupported type combination)
bool pxa_p60_mega_moe(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args);
// the same MoE half for 2..6 tokens (pxa/p60-mega-t.cu; x_in / x_out hold n_tokens rows): fp16 activations, HFMA2,
// the union of the tokens' experts computed once per weight row. Q5_K gate/up only together with Q6_K down.
bool pxa_p60_mega_moe_t(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens);
// f32 copy of a BF16 [n] weight on this device (converted once, cached by source pointer); nullptr under capture
// before the copy exists
const float * pxa_p60_bf16_to_f32(ggml_backend_cuda_context & ctx, const void * src, int64_t n);

// Gated-DeltaNet half of a Qwen3.5/3.6 MoE layer, one token, one sequence, no recurrent snapshots, PXA DELTA_NET
// semantics: x = rms_norm(x_in) * attn_norm; qkv = conv4(state_conv * mask, qkv_w x) -> silu; q, k = l2_norm per head
// (rsqrt(sum + eps)), q also * 1/sqrt(128); g = softplus(alpha x + dt) * a, decay = exp(min(g, 50)), beta = sigmoid(beta x);
// S[k][v] (v contiguous) <- clamp(decay * S + k (beta (v - decay S^T k)), +-1e6), o = decay S^T q + beta(..) (k.q);
// y = rms_norm(o per head) * ssm_norm * silu(z x); x_out = x_in + ssm_out y.
struct pxa_p60_gdn_args {
    const float * x_in;      // [2048]
    float *       x_out;     // [2048]
    const float * attn_norm; // [2048]
    float         eps_norm;
    const char *  qkv;       // Q8_0 [8192 rows of 2048]
    const char *  z;         // Q8_0 [4096 rows of 2048]
    const char *  out;       // Q8_0 [2048 rows of 4096]
    const float * alpha;     // f32 [32 rows of 2048]
    const float * beta;      // f32 [32 rows of 2048]
    const float * dt;        // [32]
    const float * a;         // [32]
    const float * conv;      // [8192][4]
    const float * snorm;     // [128]
    float         eps_l2q, eps_l2k, eps_gn;
    int           repeat_type; // DELTA_NET op_params[0]: 0 = kq head h / 2, 1 = h % 16
    // recurrent cache rows [conv 3*8192 | ssm 32*128*128]: read row = s_rd_base + rd_ids[0] * rd_stride (bytes),
    // write row = s_wr_base + wr_ids[0] * wr_stride (bytes); the ids live on the device, the rows may coincide
    const char *    s_rd_base;
    const int32_t * rd_ids;
    int64_t         rd_stride;
    char *          s_wr_base;
    const int32_t * wr_ids;
    int64_t         wr_stride;
    const float *   mask;      // [1] state keep mask (0 = reset)
    // PXQN4 variant (closed library): qkv / z / out are PXQN4 and read rotated activations, alpha / beta Q8_0 read the
    // same rotated x; RHT sign words of (seed, layer, site) at global K index k0 + k
    int             proj_type; // GGML_TYPE_Q8_0 (open kernel) or GGML_TYPE_PXQN4
    uint64_t        rht_seed;
    int             rht_layer_in, rht_site_in, rht_layer_out, rht_site_out;
    int64_t         rht_k0_in, rht_k0_out;
    const char *    alpha_q8;  // Q8_0 [32 rows of 2048] (PXQN4 variant)
    const char *    beta_q8;
};

bool pxa_p60_mega_gdn(ggml_backend_cuda_context & ctx, const pxa_p60_gdn_args & args);
// the GDN half with PXQN4 projections (pxa/pxqn-mega.cu, closed library); false = declined
bool ggml_cuda_pxqn_mega_gdn(ggml_backend_cuda_context & ctx, const pxa_p60_gdn_args & args);
// the same half for 2..6 tokens of one sequence (speculative verify; pxa/pxqn-mega-gdnt.cuh, closed library).
// Per token the single-token kernel's arithmetic (token t == the t-th of T chained single-token calls, bit for bit). g.x_in /
// g.x_out hold n_tokens rows of 2048; the state row read once, the final state written once; conv_steps / ssm_steps: the
// dense per-step captures of SSM_CONV (all T steps, [3 * 8192] each) and DELTA_NET (t < T - 1, [32 * 128 * 128] each), or null.
struct pxa_p60_gdn_t_args {
    pxa_p60_gdn_args g;
    float * conv_steps; int64_t conv_step;   // elements between two steps
    float * ssm_steps;  int64_t ssm_step;
};
bool ggml_cuda_pxqn_mega_gdn_t(ggml_backend_cuda_context & ctx, const pxa_p60_gdn_t_args & args, int n_tokens);

// fp16 expert GEMM for MoE prefill (pxa/p60-moe-gemm.cu, PXA_P60_MOE_GEMM=1, >= PXA_P60_MOE_GEMM_MIN tokens, default 32):
// GGML_OP_MUL_MAT_ID and GGML_OP_MOE_FUSED_UP_GATE (silu, no clamp, no biases) with q4_K / q5_K / q6_K experts.
// Return false (nothing launched) when the node does not qualify.
struct ggml_tensor;
bool pxa_p60_moe_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool pxa_p60_moe_gemm_up_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Full-attention half of a Qwen3.5/3.6 MoE layer (pxa/p60-attn.cu, PXA_P60_MEGA_ATTN under PXA_P60_MEGA; T = 1 for now):
// attn_norm -> q (+gate) / k / v projections (Q8_0) -> q / k RMS norm -> M-RoPE -> this token's f16 k / v cache rows ->
// flash attention over the cache (K/V split across blocks) -> sigmoid gate -> output projection (Q8_0) -> residual.
struct pxa_p60_attn_args {
    const float * x_in;      // [T][2048]
    float *       x_out;     // [T][2048]
    const float * attn_norm; // [2048]
    float         eps_norm;
    const char *  wq;        // Q8_0 [8192 rows of 2048]: per head 256 query rows, then 256 gate rows
    const char *  wk;        // Q8_0 [512 rows of 2048]
    const char *  wv;        // Q8_0 [512 rows of 2048]
    const char *  wo;        // Q8_0 [2048 rows of 4096]
    const float * q_norm;    // [256]
    const float * k_norm;    // [256]
    float         eps_q, eps_k;
    const int32_t * pos;     // [4][T] M-RoPE positions
    int           n_dims;    // rotated dims
    int           sections[4];
    bool          imrope;
    float         freq_base, freq_scale, attn_factor;
    void *        k_dst;     // this token's cache rows (f16, 2 heads x 256), the destinations of the graph's CPY nodes
    void *        v_dst;
    int64_t       kv_t;      // halves between the rows of consecutive tokens (T > 1)
    const void *  K;         // f16 cache view read by flash attention: position stride k_p, head stride k_h (halves)
    int64_t       k_p, k_h;
    const void *  V;
    int64_t       v_p, v_h;
    int           n_kv;      // positions attended
    const void *  mask;      // f16 [T rows of >= n_kv]
    int64_t       mask_row;  // halves
    float         scale;     // kq scale
    // PXQN4 variant (closed library, one token): wq / wo PXQN4 reading rotated activations, wk / wv Q8_0 reading the
    // same rotated x; RHT sign words of (seed, layer, site) at global K index k0 + k
    int           proj_type; // GGML_TYPE_Q8_0 (open kernels) or GGML_TYPE_PXQN4
    uint64_t      rht_seed;
    int           rht_layer_in, rht_site_in, rht_layer_out, rht_site_out;
    int64_t       rht_k0_in, rht_k0_out;
};
bool pxa_p60_mega_attn(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args, int n_tokens);
// the attention half with PXQN4 q / output projections (pxa/pxqn-mega.cu, closed library), one token; false = declined
bool ggml_cuda_pxqn_mega_attn(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args);
// the same half for 2..6 tokens of one sequence (closed library): x_in / x_out [T][2048], pos [4][T],
// k_dst / v_dst token 0's cache rows (+ t * kv_t halves), mask rows mask_row apart; per token the single-token arithmetic
bool ggml_cuda_pxqn_mega_attn_t(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args, int n_tokens);
// PXA_P60_MEGA_ATTN_T=2: only the tail of that half -- flash attention per token (attn_fa_t) + merge, sigmoid gate,
// output projection, residual (attn_post_t) -- on the verify path's own q (q [T][16][256], normed + roped) and q projection
// rows (qf [T][8192], the gate read from it); the cache rows are already written. Uses x_in, x_out, wo, pos, K, V, mask,
// scale and the out RHT fields of args.
bool ggml_cuda_pxqn_mega_attn_tail(ggml_backend_cuda_context & ctx, const pxa_p60_attn_args & args, int n_tokens,
                                   const float * q, const float * qf);

// the same MoE half with PXQN4 routed experts (pxa/pxqn-mega.cu in the closed library, reached through the PXQN CUDA
// table; the open forward declines when the library is absent). One token; false = declined.
bool ggml_cuda_pxqn_mega_moe(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args);
// its selftest (tests/test-pxa-p60-halves.cpp through ggml_backend_cuda_p60_mega_selftest, ggml-cuda-p60.h): the
// default one-token kernel against the reference reimplementation on pseudo-random inputs, bitwise; the differing words
// of x_out (0 = identical), < 0 = could not run; *v1_diff the same for PXA_P60_MEGA_V1, out_us[3] = us per call
int ggml_cuda_pxqn_mega_moe_selftest(int device, unsigned seed, int reps, int * v1_diff, float * out_us);
// the verify-width selftest (ggml_backend_cuda_p60_vw_selftest in ggml-cuda-p60.h): token t of an
// n_tokens-token call of a P60 half against the plain one-token computation of that token, bitwise (op 0 MoE, op 1 GDN)
int ggml_cuda_pxqn_mega_vw_selftest(ggml_backend_cuda_context & ctx, int op, unsigned seed, int n_tokens, int reps, int * diffs, float * out_us);
// ... and for 2..6 tokens (speculative verify): union of the tokens' experts, each slab decoded once per token set
bool ggml_cuda_pxqn_mega_moe_t(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens);
//  PXA_P60_MEGA_MOE_INV: the same half per token bit-identical to the single-token kernel (pxa/pxqn-mega-moet3.cuh)
bool ggml_cuda_pxqn_mega_moe_t3(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens);
//  PXA_P60_MEGA_ADDFOLD: the same half with the residual ADD before it folded in (x = add_a + add_b)
bool ggml_cuda_pxqn_mega_moe_t_fold(ggml_backend_cuda_context & ctx, const pxa_p60_moe_args & args, int n_tokens,
                                    const float * add_a, const float * add_b, float * add_out);

// PXA_P60_PRE_VW: the input side of a 2..5-token GDN / attention layer (rms norm + RHT of attn_in + every projection that
// reads the rotated x) as one launch (pxqn-vw.cuh). PXQN4 matrices [R rows of 2048], Q8_0 matrices [R rows of 2048].
struct pxa_p60_pre_args {
    const float * x;  int64_t ldx;     // [T][2048], column stride in floats
    const float * norm; float eps;
    uint64_t rht_seed; int rht_layer, rht_site; int64_t rht_k0;
    int np; const char * pw[2]; int pr[2]; float * py[2]; int64_t pldy[2];
    int nq; const char * qw[2]; int qr[2]; float * qy[2]; int64_t qldy[2]; int64_t qnb1[2];
};
bool ggml_cuda_pxqn_pre_vw(ggml_backend_cuda_context & ctx, const pxa_p60_pre_args & args, int n_tokens);

// PXA_P60_POST_VW: the output side of a 2..5-token GDN layer (gated rms norm * silu(z), RHT of out_in, ssm_out, residual)
// as one launch (pxqn-vw.cuh k_pxqn_post).
struct pxa_p60_post_args {
    const float * o; const float * z; int64_t ldo, ldz;   // [T][4096] (token stride in floats)
    const float * snorm; float eps;                       // [128]
    uint64_t rht_seed; int rht_layer, rht_site; int64_t rht_k0;
    const char * w;                                       // PXQN4 ssm_out [2048 rows of 4096]
    const float * res; int64_t ldr;                       // residual [T][2048]
    float * y; int64_t ldy;                               // [T][2048]
};
bool ggml_cuda_pxqn_post_vw(ggml_backend_cuda_context & ctx, const pxa_p60_post_args & args, int n_tokens);

