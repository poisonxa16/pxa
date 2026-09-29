// qwen4exp hyper-connection glue, fused (PXA_QWEN4EXP_HC_FUSED; 2026-09-25).
//
// The qwen4exp mixer + combine is ~20 tiny kernels per sublayer per device (scale, sigmoid,
// scale, repeat, mul, add for the combine; rms_norm, mul, ..., sigmoid, mul, cont, add x3, scale
// for the mixer around its three GEMVs). At batch 1 each is a few microseconds of device time and
// one host launch. These two kernels replace 13 of them.
//
// BIT-IDENTITY INTENT. Every expression below is written the way the unfused kernels write it
// (scale_f32: s*x + 0; sigmoid_f32: 1/(1+expf(-x)); binbcast mul/add; rms_norm_f32<1024>'s
// per-thread strided partial, warp_reduce_sum, 32-slot shared reduce, tmp/ncols, rsqrtf) and the
// stream mean sums c = 0..hc-1 in order before the 1/hc scale, exactly like cont + add + add + add
// + scale. __fmul_rn / __fadd_rn pin the steps the unfused chain rounds separately so the compiler
// cannot contract them into an FMA. The greedy sha of a fused arm against the unfused arm is the
// test of this claim.
#include "hc-fused.cuh"

template <int block_size>
static __global__ void k_hc_combine_norm(
        const char * __restrict__ r, const char * __restrict__ b, const char * __restrict__ inj,
        const float * __restrict__ gamma, float * __restrict__ dst,
        const int n_embd, const int n_hc,
        const int64_t r_nb1, const int64_t r_nb2, const int64_t b_nb1, const int64_t j_nb0, const int64_t j_nb1,
        const int64_t d_nb3, const float inv_hc, const float eps) {
    const int c   = blockIdx.x % n_hc;
    const int t   = blockIdx.x / n_hc;
    const int tid = threadIdx.x;

    const float * rr = (const float *) (r + c*r_nb1 + t*r_nb2);
    const float * br = (const float *) (b + t*b_nb1);
    const float   jv = *(const float *) (inj + c*j_nb0 + t*j_nb1);

    // scale(inject, 1/hc) -> sigmoid -> scale(., 2)
    const float js = __fmul_rn(inv_hc, jv);
    const float sg = 1.0f / (1.0f + expf(-js));
    const float w  = __fmul_rn(2.0f, sg);

    float * y = dst + ((int64_t) t*n_hc + c)*n_embd;

    float tmp = 0.0f;
    for (int col = tid; col < n_embd; col += block_size) {
        const float xi = __fadd_rn(rr[col], __fmul_rn(br[col], w));
        y[col] = xi;
        tmp += xi * xi;
    }
    if (gamma == nullptr) {
        return;
    }
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }
    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);

    float * xn = (float *) ((char *) y + d_nb3);
    const float * gm = gamma + (int64_t) c*n_embd;
    for (int col = tid; col < n_embd; col += block_size) {
        const float v = __fmul_rn(scale, y[col]);
        xn[col] = __fmul_rn(v, gm[col]);
    }
}

static __global__ void k_hc_gate_mix(const char * __restrict__ x, const char * __restrict__ g, float * __restrict__ dst,
        const int n_embd, const int n_hc, const int nt, const int64_t x_nb1, const int64_t g_nb1, const float inv_hc) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) n_embd*nt) {
        return;
    }
    const int e = (int) (i % n_embd);
    const int t = (int) (i / n_embd);
    const float * xr = (const float *) (x + t*x_nb1);
    const float * gr = (const float *) (g + t*g_nb1);
    float acc = 0.0f;
    for (int c = 0; c < n_hc; ++c) {
        const float s  = 1.0f / (1.0f + expf(-gr[c*n_embd + e]));
        const float gc = __fmul_rn(xr[c*n_embd + e], s);
        acc = c == 0 ? gc : __fadd_rn(acc, gc);
    }
    dst[i] = __fmul_rn(inv_hc, acc);
}

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * r = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * j = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    const int n_embd = (int) r->ne[0], n_hc = (int) r->ne[1], nt = (int) r->ne[2];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    const float inv_hc = 1.0f/(float) n_hc;
    cudaStream_t stream = ctx.stream();
    const dim3 grid(n_hc*nt);
    if (n_embd < 1024) {
        k_hc_combine_norm<256><<<grid, 256, 0, stream>>>((const char *) r->data, (const char *) b->data, (const char *) j->data,
            g ? (const float *) g->data : nullptr, (float *) dst->data, n_embd, n_hc,
            r->nb[1], r->nb[2], b->nb[1], j->nb[0], j->nb[1], dst->nb[3], inv_hc, eps);
    } else {
        k_hc_combine_norm<1024><<<grid, 1024, 0, stream>>>((const char *) r->data, (const char *) b->data, (const char *) j->data,
            g ? (const float *) g->data : nullptr, (float *) dst->data, n_embd, n_hc,
            r->nb[1], r->nb[2], b->nb[1], j->nb[0], j->nb[1], dst->nb[3], inv_hc, eps);
    }
}

void ggml_cuda_op_hc_gate_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = dst->src[1];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    int n_hc;
    memcpy(&n_hc, dst->op_params, sizeof(int));
    const int n_embd = (int) dst->ne[0], nt = (int) dst->ne[1];
    const int64_t n = (int64_t) n_embd*nt;
    k_hc_gate_mix<<<(unsigned) ((n + 255)/256), 256, 0, ctx.stream()>>>((const char *) x->data, (const char *) g->data,
        (float *) dst->data, n_embd, n_hc, nt, x->nb[1], g->nb[1], 1.0f/(float) n_hc);
}
