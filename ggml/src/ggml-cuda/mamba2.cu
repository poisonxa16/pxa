// PXA Nemotron-H phase 2: in-place Mamba-2 mixer kernels (GGML_OP_MAMBA2_CONV, GGML_OP_MAMBA2_SCAN) and the
// relu^2 pair fusion.
//
// Why: on Pascal the decode of a Mamba-2 hybrid is launch-bound (CUDA graphs are off below sm_70), and the
// unfused mixer is ~20 nodes per layer that also move the whole recurrent state (4 MB per layer for
// Nemotron-3-Nano-4B) through gather, mask, concat and scatter copies. These two ops read the state row in
// place and write it back in place: per layer 2 launches instead of ~16, and the state crosses DRAM once each
// way.
//
// Bit-identity: every value is computed with the same expression the unfused kernels use (ssm-conv.cu
// ssm_conv_single_seq_f32_nc4 / multi-seq nc4 kernels, binbcast op_add / op_mul, unary silu_f32, ssm-scan2.cu
// ssm_scan2_f32_group, pxa_dn_gather_mask_f32x4). Where the unfused graph rounds an intermediate to float in
// memory and a fused expression could otherwise be contracted into an FMA, __fadd_rn / __fmul_rn pin the
// rounding.
#include "mamba2.cuh"
#include "pxa/pxa-enhance.cuh"   // pxa_gate_default() -- level-aware lever defaults

// ------------------------------------------------------------------------------------------------------------
// conv: y = silu(conv1d(mask*window | x) + b), new window = last 3 inputs. One thread per (channel, token block,
// sequence); the token-block-0 thread is the only reader of the old window and the only writer of the new one.
// ------------------------------------------------------------------------------------------------------------
template <int split_n_t>
static __global__ void mamba2_conv_f32_nc4(
        float * __restrict__ rs, const int64_t rs_s1,
        const float * __restrict__ x, const int64_t x_s1,
        const float * __restrict__ w, const float * __restrict__ b,
        const int32_t * __restrict__ ids, const float * __restrict__ mask, const int64_t mask_s1,
        float * __restrict__ dst, const int nr, const int nst) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nr) {
        return;
    }
    const int t0 = blockIdx.y*split_n_t;
    if (t0 >= nst) {
        return;
    }
    const int s = blockIdx.z;

    const float   m   = mask[(int64_t) s*mask_s1];
    float *       win = rs + (int64_t) ids[s]*rs_s1 + (int64_t) c*3;
    const float * xs  = x + (int64_t) s*nst*x_s1 + c;
    float *       ds  = dst + (int64_t) s*nst*nr + c;

    const float c0 = w[(int64_t) c*4 + 0];
    const float c1 = w[(int64_t) c*4 + 1];
    const float c2 = w[(int64_t) c*4 + 2];
    const float c3 = w[(int64_t) c*4 + 3];

    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
    float x0, x1, x2;
    if (t0 == 0) {
        w0 = win[0] * m;
        w1 = win[1] * m;
        w2 = win[2] * m;
        x0 = w0; x1 = w1; x2 = w2;
    } else {
        x0 = xs[(int64_t) (t0 - 3)*x_s1];
        x1 = xs[(int64_t) (t0 - 2)*x_s1];
        x2 = xs[(int64_t) (t0 - 1)*x_s1];
    }

#pragma unroll
    for (int it = 0; it < split_n_t; ++it) {
        const int t = t0 + it;
        if (t >= nst) {
            break;
        }
        const float x3 = xs[(int64_t) t*x_s1];
        const float sumf = x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3;
        const float v = b ? __fadd_rn(sumf, b[c]) : sumf;
        ds[(int64_t) t*nr] = v / (1.0f + expf(-v));
        x0 = x1; x1 = x2; x2 = x3;
    }

    if (t0 == 0) {
        // entry e of the (masked window | x) sequence is w_e for e < 3, x_{e-3} after
        float nw[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int e = nst + k;
            nw[k] = e >= 3 ? xs[(int64_t) (e - 3)*x_s1] : (e == 0 ? w0 : e == 1 ? w1 : w2);
        }
        win[0] = nw[0];
        win[1] = nw[1];
        win[2] = nw[2];
    }
}

void ggml_cuda_op_mamba2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * rs   = dst->src[0];
    const ggml_tensor * x    = dst->src[1];
    const ggml_tensor * w    = dst->src[2];
    const ggml_tensor * ids  = dst->src[3];
    const ggml_tensor * mask = dst->src[4];
    const ggml_tensor * b    = dst->src[5];

    // in place: the rows must be device memory of this backend, never a scheduler copy
    GGML_ASSERT(rs->buffer && !ggml_backend_buffer_is_host(rs->buffer));
    GGML_ASSERT(w->ne[0] == 4 && ggml_is_contiguous(w) && ggml_is_contiguous(dst));
    GGML_ASSERT(rs->nb[1] % sizeof(float) == 0 && x->nb[1] % sizeof(float) == 0 && mask->nb[1] % sizeof(float) == 0);

    const int nr  = (int) w->ne[1];
    const int nt  = (int) x->ne[1];
    const int ns  = (int) ids->ne[0];
    const int nst = nt / ns;
    GGML_ASSERT(nst*ns == nt && ns <= 65535);

    constexpr int block     = 256;
    constexpr int split_n_t = 32;
    const dim3 grid((nr + block - 1)/block, (nst + split_n_t - 1)/split_n_t, ns);
    mamba2_conv_f32_nc4<split_n_t><<<grid, block, 0, ctx.stream()>>>(
            (float *) rs->data, (int64_t) (rs->nb[1]/sizeof(float)),
            (const float *) x->data, (int64_t) (x->nb[1]/sizeof(float)),
            (const float *) w->data, b ? (const float *) b->data : nullptr,
            (const int32_t *) ids->data, (const float *) mask->data, (int64_t) (mask->nb[1]/sizeof(float)),
            (float *) dst->data, nr, nst);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------------------
// scan: one warp per (head, head-dim row), d_state/32 state values per lane (the ssm_scan2_f32_group shape);
// the gate epilogue (y + D*x) * silu(z) runs on lane 0 after the warp reduction.
// ------------------------------------------------------------------------------------------------------------
template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    mamba2_scan_f32_group(
        float * __restrict__ rs, const int64_t rs_s1, const int64_t s_off,
        const int32_t * __restrict__ ids, const float * __restrict__ mask, const int64_t mask_s1,
        const float * __restrict__ x, const int64_t x_s2, const int64_t x_s3,
        const float * __restrict__ dt, const int64_t dt_s1, const int64_t dt_s2,
        const float * __restrict__ A, const float * __restrict__ dt_b, const float * __restrict__ D,
        const float * __restrict__ B, const int64_t B_s2, const int64_t B_s3,
        const float * __restrict__ C, const int64_t C_s2, const int64_t C_s3,
        const float * __restrict__ z, const int64_t z_s2, const int64_t z_s3,
        float * __restrict__ dst,
        const int n_head, const int d_head, const int n_group, const int nst) {
    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x * c_factor + warp;

    if (warp_idx >= n_head * d_head) {
        return;
    }

    const int head_idx = warp_idx / d_head;
    const int seq_idx  = blockIdx.y;
    const int g_off    = (head_idx / (n_head / n_group)) * d_state;

    const float m = mask[(int64_t) seq_idx*mask_s1];
    float * s_warp = rs + (int64_t) ids[seq_idx]*rs_s1 + s_off + (int64_t) warp_idx*d_state;

    const float * x_warp  = x  + (int64_t) seq_idx*x_s3  + warp_idx;
    const float * z_warp  = z  + (int64_t) seq_idx*z_s3  + warp_idx;
    const float * dt_warp = dt + (int64_t) seq_idx*dt_s2 + head_idx;
    const float * B_warp  = B  + (int64_t) seq_idx*B_s3  + g_off;
    const float * C_warp  = C  + (int64_t) seq_idx*C_s3  + g_off;
    float *       y_warp  = dst + (int64_t) seq_idx*nst*n_head*d_head + warp_idx;
    const int64_t stride_y = (int64_t) n_head*d_head;

    float state[c_factor];

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s_warp[WARP_SIZE * j + lane] * m;
    }

    const float A_h  = A[head_idx];
    const float dtb  = dt_b[head_idx];
    const float D_h  = D[head_idx];

    for (int i = 0; i < nst; i++) {
        const float dt_raw = __fadd_rn(dt_warp[(int64_t) i * dt_s1], dtb);
        const float dt_sp  = dt_raw <= 20.0f ? log1pf(expf(dt_raw)) : dt_raw;
        const float dA     = expf(dt_sp * A_h);
        const float xv     = x_warp[(int64_t) i * x_s2];
        const float x_dt   = xv * dt_sp;

        float state_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[(int64_t) i * B_s2 + WARP_SIZE * j + lane];
            const float C_val = C_warp[(int64_t) i * C_s2 + WARP_SIZE * j + lane];
            state[j] = state[j] * dA + B_val * x_dt;
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum(state_sum);

        if (lane == 0) {
            const float zv = z_warp[(int64_t) i * z_s2];
            const float yv = __fadd_rn(state_sum, __fmul_rn(xv, D_h));
            const float sg = zv / (1.0f + expf(-zv));
            y_warp[i * stride_y] = __fmul_rn(yv, sg);
        }
    }

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

// ------------------------------------------------------------------------------------------------------------
// scan, batched form (2026-10-16; PXA_MAMBA2_SCAN_CHUNK=0 restores the warp kernel above for
// every batch). Same grid, same warp-per-row arithmetic and the same warp_reduce_sum as mamba2_scan_f32_group; the
// per-token work that does not depend on the state moves to one lane per token in 32-token chunks:
//  * lane l computes dt_softplus, dA and x*dt for token i0+l once (the warp kernel computes them on all 32 lanes
//    for every token) and the token loop takes them with two broadcast shuffles;
//  * lane t keeps token t's reduced sum and the gate epilogue (y + D*x) * silu(z) runs once per chunk on all
//    lanes (the warp kernel runs it on lane 0 for every token).
// Every value is produced by the same expression on the same operands (only the lane that evaluates it changes),
// so the output and the written-back state are bitwise those of mamba2_scan_f32_group.
// ------------------------------------------------------------------------------------------------------------
// Level-aware since merge pass 8 (bug #281 audit): REFERENCE takes the warp kernel (bitwise the same output).
static bool mamba2_scan_chunk_on() {
    static const bool on = [] { const char * e = getenv("PXA_MAMBA2_SCAN_CHUNK"); return e ? atoi(e) != 0 : pxa_gate_default(true); }();
    return on;
}

template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    mamba2_scan_f32_lanes(
        float * __restrict__ rs, const int64_t rs_s1, const int64_t s_off,
        const int32_t * __restrict__ ids, const float * __restrict__ mask, const int64_t mask_s1,
        const float * __restrict__ x, const int64_t x_s2, const int64_t x_s3,
        const float * __restrict__ dt, const int64_t dt_s1, const int64_t dt_s2,
        const float * __restrict__ A, const float * __restrict__ dt_b, const float * __restrict__ D,
        const float * __restrict__ B, const int64_t B_s2, const int64_t B_s3,
        const float * __restrict__ C, const int64_t C_s2, const int64_t C_s3,
        const float * __restrict__ z, const int64_t z_s2, const int64_t z_s3,
        float * __restrict__ dst,
        const int n_head, const int d_head, const int n_group, const int nst) {
    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x * c_factor + warp;

    if (warp_idx >= n_head * d_head) {
        return;
    }

    const int head_idx = warp_idx / d_head;
    const int seq_idx  = blockIdx.y;
    const int g_off    = (head_idx / (n_head / n_group)) * d_state;

    const float m = mask[(int64_t) seq_idx*mask_s1];
    float * s_warp = rs + (int64_t) ids[seq_idx]*rs_s1 + s_off + (int64_t) warp_idx*d_state;

    const float * x_warp  = x  + (int64_t) seq_idx*x_s3  + warp_idx;
    const float * z_warp  = z  + (int64_t) seq_idx*z_s3  + warp_idx;
    const float * dt_warp = dt + (int64_t) seq_idx*dt_s2 + head_idx;
    const float * B_warp  = B  + (int64_t) seq_idx*B_s3  + g_off;
    const float * C_warp  = C  + (int64_t) seq_idx*C_s3  + g_off;
    float *       y_warp  = dst + (int64_t) seq_idx*nst*n_head*d_head + warp_idx;
    const int64_t stride_y = (int64_t) n_head*d_head;

    float state[c_factor];

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s_warp[WARP_SIZE * j + lane] * m;
    }

    const float A_h  = A[head_idx];
    const float dtb  = dt_b[head_idx];
    const float D_h  = D[head_idx];

    for (int i0 = 0; i0 < nst; i0 += WARP_SIZE) {
        const int n_in = min(WARP_SIZE, nst - i0);

        // lane l owns token i0 + l
        float my_dA = 0.0f, my_xdt = 0.0f, my_xv = 0.0f;
        if (lane < n_in) {
            const int i = i0 + lane;
            const float dt_raw = __fadd_rn(dt_warp[(int64_t) i * dt_s1], dtb);
            const float dt_sp  = dt_raw <= 20.0f ? log1pf(expf(dt_raw)) : dt_raw;
            my_dA  = expf(dt_sp * A_h);
            my_xv  = x_warp[(int64_t) i * x_s2];
            my_xdt = my_xv * dt_sp;
        }

        float my_sum = 0.0f;
        for (int t = 0; t < n_in; t++) {
            const int   i    = i0 + t;
            const float dA   = __shfl_sync(0xffffffff, my_dA,  t, WARP_SIZE);
            const float x_dt = __shfl_sync(0xffffffff, my_xdt, t, WARP_SIZE);

            float state_sum = 0.0f;
#pragma unroll
            for (int j = 0; j < c_factor; j++) {
                const float B_val = B_warp[(int64_t) i * B_s2 + WARP_SIZE * j + lane];
                const float C_val = C_warp[(int64_t) i * C_s2 + WARP_SIZE * j + lane];
                state[j] = state[j] * dA + B_val * x_dt;
                state_sum += state[j] * C_val;
            }

            state_sum = warp_reduce_sum(state_sum);
            if (lane == t) {
                my_sum = state_sum;
            }
        }

        if (lane < n_in) {
            const int i = i0 + lane;
            const float zv = z_warp[(int64_t) i * z_s2];
            const float yv = __fadd_rn(my_sum, __fmul_rn(my_xv, D_h));
            const float sg = zv / (1.0f + expf(-zv));
            y_warp[i * stride_y] = __fmul_rn(yv, sg);
        }
    }

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

template <int d_state>
static void mamba2_scan_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * rs   = dst->src[0];
    const ggml_tensor * x    = dst->src[1];
    const ggml_tensor * dt   = dst->src[2];
    const ggml_tensor * A    = dst->src[3];
    const ggml_tensor * B    = dst->src[4];
    const ggml_tensor * C    = dst->src[5];
    const ggml_tensor * dt_b = dst->src[6];
    const ggml_tensor * D    = dst->src[7];
    const ggml_tensor * z    = dst->src[8];
    const ggml_tensor * ids  = dst->src[9];
    const ggml_tensor * mask = dst->src[10];

    const int64_t s_off = (int64_t) (uint32_t) dst->op_params[0] | ((int64_t) dst->op_params[1] << 32);

    const int d_head  = (int) x->ne[0];
    const int n_head  = (int) x->ne[1];
    const int nst     = (int) x->ne[2];
    const int ns      = (int) x->ne[3];
    const int n_group = (int) B->ne[1];
    GGML_ASSERT(ns <= 65535);

    constexpr int fs = sizeof(float);
    GGML_ASSERT(x->nb[2] % fs == 0 && x->nb[3] % fs == 0 && z->nb[2] % fs == 0 && z->nb[3] % fs == 0);
    GGML_ASSERT(dt->nb[1] % fs == 0 && dt->nb[2] % fs == 0 && B->nb[2] % fs == 0 && B->nb[3] % fs == 0);
    GGML_ASSERT(C->nb[2] % fs == 0 && C->nb[3] % fs == 0 && rs->nb[1] % fs == 0 && mask->nb[1] % fs == 0);

    constexpr int threads   = d_state;
    constexpr int num_warps = threads / WARP_SIZE;
    const dim3 blocks((n_head * d_head + (num_warps - 1)) / num_warps, ns, 1);
    auto kern = (nst >= 8 && mamba2_scan_chunk_on()) ? mamba2_scan_f32_lanes<d_state / WARP_SIZE, d_state>
                                                      : mamba2_scan_f32_group<d_state / WARP_SIZE, d_state>;
    kern<<<blocks, threads, 0, ctx.stream()>>>(
            (float *) rs->data, (int64_t) (rs->nb[1]/fs), s_off,
            (const int32_t *) ids->data, (const float *) mask->data, (int64_t) (mask->nb[1]/fs),
            (const float *) x->data, (int64_t) (x->nb[2]/fs), (int64_t) (x->nb[3]/fs),
            (const float *) dt->data, (int64_t) (dt->nb[1]/fs), (int64_t) (dt->nb[2]/fs),
            (const float *) A->data, (const float *) dt_b->data, (const float *) D->data,
            (const float *) B->data, (int64_t) (B->nb[2]/fs), (int64_t) (B->nb[3]/fs),
            (const float *) C->data, (int64_t) (C->nb[2]/fs), (int64_t) (C->nb[3]/fs),
            (const float *) z->data, (int64_t) (z->nb[2]/fs), (int64_t) (z->nb[3]/fs),
            (float *) dst->data, n_head, d_head, n_group, nst);
}

void ggml_cuda_op_mamba2_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * rs = dst->src[0];
    GGML_ASSERT(rs->buffer && !ggml_backend_buffer_is_host(rs->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    switch (dst->src[4]->ne[0]) {
        case  64: mamba2_scan_launch< 64>(ctx, dst); break;
        case 128: mamba2_scan_launch<128>(ctx, dst); break;
        case 256: mamba2_scan_launch<256>(ctx, dst); break;
        default: GGML_ABORT("mamba2_scan: d_state must be 64, 128 or 256");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------------------
// relu^2 pair
// ------------------------------------------------------------------------------------------------------------
static __global__ void relu_sqr_f32(const float * x, float * dst_relu, float * dst_sqr, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const float r = fmaxf(x[i], 0);
    dst_relu[i] = r;
    dst_sqr[i]  = r * r;
}

bool ggml_cuda_relu_sqr_fusable(const ggml_tensor * relu, const ggml_tensor * sqr) {
    // level-aware default (bug #281): off at PXA_REFERENCE=1, on otherwise; PXA_RELU_SQR_FUSE=0/1 overrides
    static const bool on = [] { const char * e = getenv("PXA_RELU_SQR_FUSE"); return e ? atoi(e) != 0 : pxa_gate_default(true); }();
    return on && sqr && sqr->op == GGML_OP_SQR && sqr->src[0] == relu &&
           relu->type == GGML_TYPE_F32 && sqr->type == GGML_TYPE_F32 && relu->src[0]->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(relu->src[0]) && ggml_is_contiguous(relu) && ggml_is_contiguous(sqr) &&
           ggml_nelements(relu) == ggml_nelements(sqr) && ggml_nelements(relu) < INT_MAX;
}

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu, ggml_tensor * sqr) {
    const int k = (int) ggml_nelements(relu);
    constexpr int block = 256;
    relu_sqr_f32<<<(k + block - 1)/block, block, 0, ctx.stream()>>>(
            (const float *) relu->src[0]->data, (float *) relu->data, (float *) sqr->data, k);
    CUDA_CHECK(cudaGetLastError());
}
