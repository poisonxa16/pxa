// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-hccn-pf.cuh -- PXA_VFUSE bit 256 (2026-10-26): k_hc_combine_norm (hc-fused.cu) with its values kept in flight.
// The qwen4exp combine + next norm runs 2x per layer at the verify (one 1024-thread block per stream and token, n_embd 2560:
// 3 columns per thread). The shipped kernel writes y, reduces, then reads y back and only then loads gamma: two dependent
// global round trips after the barrier. Here each thread keeps its combined values in registers and loads its gamma values
// before the reduction; the arithmetic is the shipped kernel's expression for expression in the same order (the column loop
// unrolled to its exact trip count NJ = ceil(n_embd / block_size)), so every output bit is the same
// (tests/test-pxa-hccn.cu compares bytes against a verbatim copy of k_hc_combine_norm).
#pragma once

template <int block_size, int NJ>
static __global__ void k_hc_combine_norm_pf(
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

    const float js = __fmul_rn(inv_hc, jv);
    const float sg = 1.0f / (1.0f + expf(-js));
    const float w  = __fmul_rn(2.0f, sg);

    float * y = dst + ((int64_t) t*n_hc + c)*n_embd;

    float xv[NJ];
    float tmp = 0.0f;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int col = tid + j*block_size;
        xv[j] = 0.0f;
        if (col < n_embd) {
            const float xi = __fadd_rn(rr[col], __fmul_rn(br[col], w));
            y[col] = xi;
            tmp += xi * xi;
            xv[j] = xi;
        }
    }
    if (gamma == nullptr) {
        return;
    }
    const float * gm = gamma + (int64_t) c*n_embd;
    float gv[NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int col = tid + j*block_size;
        gv[j] = col < n_embd ? gm[col] : 0.0f;
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
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int col = tid + j*block_size;
        if (col < n_embd) {
            const float v = __fmul_rn(scale, xv[j]);
            xn[col] = __fmul_rn(v, gv[j]);
        }
    }
}

// true = launched (1024-thread form, n_embd in (1024, 4096]); false = the caller runs the shipped kernel
static bool pxa_hc_combine_norm_pf_launch(const char * r, const char * b, const char * inj, const float * gamma, float * dst,
        int n_embd, int n_hc, int nt, int64_t r_nb1, int64_t r_nb2, int64_t b_nb1, int64_t j_nb0, int64_t j_nb1, int64_t d_nb3,
        float inv_hc, float eps, cudaStream_t stream) {
    if (n_embd < 1024) return false;
    const dim3 grid(n_hc*nt);
    switch ((n_embd + 1023)/1024) {
        case 2: k_hc_combine_norm_pf<1024, 2><<<grid, 1024, 0, stream>>>(r, b, inj, gamma, dst, n_embd, n_hc, r_nb1, r_nb2, b_nb1, j_nb0, j_nb1, d_nb3, inv_hc, eps); return true;
        case 3: k_hc_combine_norm_pf<1024, 3><<<grid, 1024, 0, stream>>>(r, b, inj, gamma, dst, n_embd, n_hc, r_nb1, r_nb2, b_nb1, j_nb0, j_nb1, d_nb3, inv_hc, eps); return true;
        case 4: k_hc_combine_norm_pf<1024, 4><<<grid, 1024, 0, stream>>>(r, b, inj, gamma, dst, n_embd, n_hc, r_nb1, r_nb2, b_nb1, j_nb0, j_nb1, d_nb3, inv_hc, eps); return true;
        default: return false;
    }
}
