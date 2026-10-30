// DeepSeek-V4 fused hyper-connection ops.
//
// Transcribed from llama.cpp `ggml/src/ggml-cuda/dsv4.cu` @ upstream commit
// 44c7b01de (deepseek-v4-flash CUDA branch). Copyright (c) 2023-2026 The ggml
// authors. MIT.
//
// The kernel arithmetic is upstream's, deliberately unaltered — in particular
// the Sinkhorn normalizations multiply by a reciprocal (1/(sum+eps)) rather
// than dividing, and the iteration/eps placement matches the disproof harness
// (brain-notes/sinkconv.py): softmax over the contiguous comb index (+eps),
// one strided-axis normalization, then (iters-1) alternating contiguous/strided
// normalizations. 20 iterations on a 4x4 do NOT converge early; do not lower
// sinkhorn_iters (iter 16 is 3.34e-2 off worst-case vs f32 eps 1.19e-7).
//
// NOTE on the comb buffer naming: upstream's producer (this sinkhorn kernel)
// calls the contiguous index src_hc, its consumer (hc_expand) indexes the same
// axis as dst_hc. Both ends are transcribed verbatim, so the composition is
// identical to upstream regardless of which name is right.

#include "common.cuh"
#include "dsv4.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int DSV4_HC_MAX = 16;

struct ggml_cuda_kargs_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ggml_cuda_kargs_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
};

struct ggml_cuda_kargs_dsv4_hc_weighted_sum {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};

// Sinkhorn de-spill (todo glm-sinkhorn-despill, SQUEEZE row 3).
//
// One templated body serves both paths. HC_T == 0 is the generic kernel: HC comes from args.n_hc
// at run time, the comb lives in c[16*16] indexed at run time, which nvcc puts in a 1 KiB local
// (stack) frame -> every Sinkhorn step is local-memory traffic (99 us x 90 calls/token on P100).
// HC_T > 0 makes HC a compile-time constant: c[] is HC_T*HC_T floats and every HC loop has a
// constant trip count that nvcc unrolls, so all indices are constants and c[] is promoted to
// registers (0 B stack, 0 B spill for HC_T == 4 on sm_60/sm_70, checked by
// scripts/glm-sinkhorn/ptxas-check.sh). No unroll pragmas on purpose: with none, the HC_T == 0
// instantiation compiles to PTX identical to the pre-template kernel (pragmas would add
// nounroll hints to the generic path). The float expressions and their evaluation order are
// the SAME source text in both instantiations (same sequential row/column sums, same reciprocal
// multiplies, same eps placement), so the result is bit-identical by construction; only storage
// and loop control change. tests/test-dsv4-sinkhorn-despill.cpp checks the host-side model of
// both orders bitwise, scripts/glm-sinkhorn/microbench.sh checks it on a GPU.
// PXA_DSV4_SINKHORN_HC4=1 selects the hc=4 kernel (default off until measured; =0/unset = the generic kernel).
template <int HC_T>
static __global__ void kernel_dsv4_hc_split_sinkhorn(
        const ggml_cuda_kargs_dsv4_hc_split_sinkhorn args,
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst) {
    static_assert(HC_T >= 0 && HC_T <= DSV4_HC_MAX, "HC_T out of range");
    constexpr int C_N = HC_T > 0 ? HC_T * HC_T : DSV4_HC_MAX * DSV4_HC_MAX;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int64_t) tid >= args.n_rows) {
        return;
    }

    const int HC = HC_T > 0 ? HC_T : args.n_hc;
    if (HC <= 0 || HC > DSV4_HC_MAX) {
        return;
    }

    const float * mix = mixes + ((int64_t) tid) * args.mix_hc;
    float * out = dst + ((int64_t) tid) * args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[C_N];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const int off = 2 * HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc * HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc * HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC * HC; ++i) {
        out[2 * HC + i] = c[i];
    }
}

static __global__ void kernel_dsv4_hc_expand(
        const ggml_cuda_kargs_dsv4_hc_expand args,
        const char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        char * dst) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    const float block_v = *((const float *) (block_out + d * args.nb_block0 + t * args.nb_block1));
    const float post_v  = *((const float *) (post      + dst_hc * args.nb_post0 + t * args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((const float *) (comb     + dst_hc * args.nb_comb0 + src_hc * args.nb_comb1 + t * args.nb_comb2));
        const float res_v  = *((const float *) (residual + d       * args.nb_res0  + src_hc * args.nb_res1  + t * args.nb_res2));
        acc += comb_v * res_v;
    }

    *((float *) (dst + d * args.nb0 + dst_hc * args.nb1 + t * args.nb2)) = acc;
}

static __global__ void kernel_dsv4_hc_weighted_sum(
        const ggml_cuda_kargs_dsv4_hc_weighted_sum args,
        const char * x,
        const char * weights,
        char * dst) {
    const int64_t n_elem = args.n_embd * args.n_tokens;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((const float *) (x     + d * args.nb_x0 + h * args.nb_x1 + t * args.nb_x2));
        const float wv = *((const float *) (weights + h * args.nb_w0 + t * args.nb_w1));
        acc += xv * wv;
    }

    *((float *) (dst + d * args.nb0 + t * args.nb1)) = acc;
}

} // namespace

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1);
    GGML_ASSERT(src0->ne[3] == 1);

    const int32_t n_hc           = ((const int32_t *) dst->op_params)[0];
    const int32_t sinkhorn_iters = ((const int32_t *) dst->op_params)[1];
    float eps;
    memcpy(&eps, (const int32_t *) dst->op_params + 2, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t n_rows = ne01 * ne02 * ne03;

    const float * mixes_d = (const float *) src0->data;
    const float * scale_d = (const float *) src1->data;
    const float * base_d  = (const float *) src2->data;
    float * dst_d = (float *) dst->data;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_rows));
    const int n_tg = (n_rows + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_split_sinkhorn args = {
        /*.n_hc            =*/ n_hc,
        /*.sinkhorn_iters  =*/ sinkhorn_iters,
        /*.n_rows          =*/ n_rows,
        /*.mix_hc          =*/ ne00,
        /*.nb01            =*/ src0->nb[1],
        /*.nb1             =*/ dst->nb[1],
        /*.eps             =*/ eps,
    };

    const cudaStream_t stream = ctx.stream();

    // PXA_DSV4_SINKHORN_HC4 (default ON since 10-24; =0 = the generic kernel): hc=4
    // (DeepSeek-V4 / GLM-5.x hc_mult) takes the register-resident specialization; any other hc, or the switch =0, takes the
    // generic kernel (same arithmetic, see above; bit-identical on V100, greedy sha equal on GLM-5.3-Flash).
    static const bool use_hc4 = [] {
        const char * e = getenv("PXA_DSV4_SINKHORN_HC4");
        const bool v = e == nullptr || atoi(e) != 0;
        if (v && e) fprintf(stderr, "PXA_DSV4_SINKHORN_HC4=%s: hc=4 Sinkhorn in registers (bit-identical de-spill)\n", e);
        if (!v) fprintf(stderr, "PXA_DSV4_SINKHORN_HC4=0: generic Sinkhorn kernel (off switch)\n");
        return v;
    }();
    if (n_hc == 4 && use_hc4) {
        kernel_dsv4_hc_split_sinkhorn<4><<<n_tg, nth, 0, stream>>>(args, mixes_d, scale_d, base_d, dst_d);
    } else {
        kernel_dsv4_hc_split_sinkhorn<0><<<n_tg, nth, 0, stream>>>(args, mixes_d, scale_d, base_d, dst_d);
    }
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * block_out = dst->src[0];
    ggml_tensor * residual  = dst->src[1];
    ggml_tensor * post      = dst->src[2];
    ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];

    const int64_t n_elem = ne0 * ne1 * ne2;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_elem));
    const int n_tg = (n_elem + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_expand args = {
        /*.n_embd    =*/ ne0,
        /*.n_hc      =*/ ne1,
        /*.n_tokens  =*/ ne2,
        /*.nb_block0 =*/ block_out->nb[0],
        /*.nb_block1 =*/ block_out->nb[1],
        /*.nb_res0   =*/ residual->nb[0],
        /*.nb_res1   =*/ residual->nb[1],
        /*.nb_res2   =*/ residual->nb[2],
        /*.nb_post0  =*/ post->nb[0],
        /*.nb_post1  =*/ post->nb[1],
        /*.nb_comb0  =*/ comb->nb[0],
        /*.nb_comb1  =*/ comb->nb[1],
        /*.nb_comb2  =*/ comb->nb[2],
        /*.nb0       =*/ dst->nb[0],
        /*.nb1       =*/ dst->nb[1],
        /*.nb2       =*/ dst->nb[2],
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_hc_expand<<<n_tg, nth, 0, stream>>>(
        args,
        (const char *) block_out->data,
        (const char *) residual->data,
        (const char *) post->data,
        (const char *) comb->data,
        (char *) dst->data);
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[3]       == 1);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(weights->ne[3] == 1);
    GGML_ASSERT(dst->ne[2]     == 1);
    GGML_ASSERT(dst->ne[3]     == 1);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_elem   = n_embd * n_tokens;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_elem));
    const int n_tg = (n_elem + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_weighted_sum args = {
        /*.n_embd  =*/ n_embd,
        /*.n_hc    =*/ n_hc,
        /*.n_tokens =*/ n_tokens,
        /*.nb_x0   =*/ x->nb[0],
        /*.nb_x1   =*/ x->nb[1],
        /*.nb_x2   =*/ x->nb[2],
        /*.nb_w0   =*/ weights->nb[0],
        /*.nb_w1   =*/ weights->nb[1],
        /*.nb0     =*/ dst->nb[0],
        /*.nb1     =*/ dst->nb[1],
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_hc_weighted_sum<<<n_tg, nth, 0, stream>>>(
        args,
        (const char *) x->data,
        (const char *) weights->data,
        (char *) dst->data);
}
