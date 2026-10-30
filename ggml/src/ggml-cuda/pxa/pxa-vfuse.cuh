// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxa-vfuse.cuh -- PXA_VFUSE (2026-10-26): the verify-width small-kernel chains of the qwen4exp (Flash-Next)
// layer in one launch each, bit-identical to the chains they replace. Included at the end of quantize.cu (the TU of
// quantize_q8_1, same CUDA flags as unary.cu / scale.cu / binbcast.cu); the host entries are declared in quantize.cuh.
//
// The verify pass (2..8 token columns) runs every small elementwise node as its own launch: on 2x V100 the ~1060
// small kernels of one verify cost ~3.9 ms of GPU time (2-3 us each, almost all launch floor). Three chains repeat
// every layer:
//   HC lo     SCALE(1/hc) -> SILU -> [quantize_q8_1] -> hc_*_up MMVQ           (2 per layer)
//   out gate  SIGMOID(z)  -> MUL(o, sig) -> [quantize_q8_1] -> wo / ssm_out MMVQ (1 per layer)
//   shexp     SIGMOID(g)  -> MUL(shexp_out, sig) -> ADD(routed, shexp)       (1 per layer)
// k_pxa_vf_ew_q8_1 runs the first two (both elementwise nodes + the q8_1 of the result in one launch; the MMVQ reads the
// q8_1 from a per-device record instead of quantizing again, ggml-cuda.cu pxa_vf_lookup) and k_pxa_vf_sig_mul_add the
// third. Every intermediate tensor is still written with the bytes its own kernel writes, so no consumer analysis
// is needed. The arithmetic is the replaced kernels' own, expression for expression:
//   scale_f32:   dst = scale * x + bias                (one FMA under the default -fmad=true, as in scale.cu)
//   silu_f32:    dst = x / (1.0f + expf(-x))
//   sigmoid_f32: dst = 1.0f / (1.0f + expf(-x))
//   op_mul / k_mul_fast: a * b (one rounding; fp32 multiply is bitwise commutative)
//   k_add_same:  a + b (one rounding; commutative)
//   quantize_q8_1: amax / sum over the 32-lane warp (warp_reduce_max / warp_reduce_sum, same lanes, same order),
//                  d = amax / 127, q = amax == 0 ? 0 : roundf(x / d), ds = (half d, half sum); padding lanes read 0.
// The MUL and the ADD of the shexp chain run on __fmul_rn / __fadd_rn: in one expression the compiler would contract
// them into an FMA, which the two separate kernels never do. Unit test: tests/test-pxa-vfuse.cu (byte compare against
// the stock kernels compiled from their own sources).
#pragma once

// OP 0 = HC lo (o1 = scale(a), o2 = silu(o1)); OP 1 = out gate (o1 = sigmoid(a), o2 = b * o1)
template <int OP>
static __global__ void k_pxa_vf_ew_q8_1(const float * __restrict__ a, const float * __restrict__ b,
                                        float * __restrict__ o1, float * __restrict__ o2, void * __restrict__ vy,
                                        const float p0, const float p1, const int64_t kx, const int64_t kx0_padded) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = blockIdx.y;

    const int64_t i_padded = ix1*kx0_padded + ix0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    float xi = 0.0f;
    if (ix0 < kx) {
        const int64_t e = ix1*kx + ix0;
        if constexpr (OP == 0) {
            const float t = p0 * a[e] + p1;              // scale_f32
            o1[e] = t;
            xi = t / (1.0f + expf(-t));                  // silu_f32
            o2[e] = xi;
        } else {
            const float s = 1.0f / (1.0f + expf(-a[e])); // sigmoid_f32
            o1[e] = s;
            xi = __fmul_rn(b[e], s);                     // op_mul
            o2[e] = xi;
        }
    }
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

// shexp tail: sig[row] = sigmoid(g[row]); mul[i] = o[i] * sig[i / ne0]; add[i] = r[i] + mul[i]
static __global__ void k_pxa_vf_sig_mul_add(const float * __restrict__ g, const float * __restrict__ o, const float * __restrict__ r,
                                            float * __restrict__ sig, float * __restrict__ mul, float * __restrict__ add,
                                            const int ne0, const int nelem) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    const int row = i / ne0;
    const float s = 1.0f / (1.0f + expf(-g[row]));       // sigmoid_f32
    if (i - row*ne0 == 0) {
        sig[row] = s;
    }
    const float m = __fmul_rn(o[i], s);                  // k_mul_fast: x[i] * y[i1]
    mul[i] = m;
    add[i] = __fadd_rn(r[i], m);                         // k_add_same
}

// shexp tail with its 1-row gate GEMV (PXA_VFUSE bit 64): every block (one row: ne0 % 256 == 0) first computes the gate logit of
// its column with mul_mat_row<float, 256>'s arithmetic line for line (per-thread stride-256 FMA chain, warp_reduce_sum, the
// 8-partial second reduce), the row's first block stores it, then the tail as k_pxa_vf_sig_mul_add.
// gout and sig may be the same address (an in-place SIGMOID): no __restrict__ on them; thread 0 of the row's first block stores
// the logit, then (after the barrier) sig, in program order.
static __global__ void k_pxa_vf_gate_tail(const float * __restrict__ w, const float * __restrict__ x, const size_t nb11,
                                          float * gout, const size_t nb1, const int n,
                                          const float * __restrict__ o, const float * __restrict__ r, float * sig,
                                          float * __restrict__ mul, float * __restrict__ add, const int ne0, const int nelem) {
    constexpr int block_size = 256;
    const int i0  = blockIdx.x*block_size;
    const int row = i0 / ne0;
    const float * ycol = (const float *)((const char *) x + (size_t) row * nb11);
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += block_size) {
        float xi = w[i];
        sum += xi * ycol[i];
    }
    sum = warp_reduce_sum(sum);
    __shared__ float tmp[block_size/WARP_SIZE];
    __shared__ float gsh;
    if (threadIdx.x % WARP_SIZE == 0) {
        tmp[threadIdx.x / WARP_SIZE] = sum;
    }
    __syncthreads();
    sum = threadIdx.x < block_size / WARP_SIZE ? tmp[threadIdx.x] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        gsh = sum;
        if (i0 - row*ne0 == 0) {
            *(float *)((char *) gout + (size_t) row * nb1) = sum;
        }
    }
    __syncthreads();
    const float g = gsh;
    const int i = i0 + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    const float s = 1.0f / (1.0f + expf(-g));            // sigmoid_f32
    if (i - row*ne0 == 0) {
        sig[row] = s;
    }
    const float m = __fmul_rn(o[i], s);                  // k_mul_fast
    mul[i] = m;
    add[i] = __fadd_rn(r[i], m);                         // k_add_same
}

void pxa_vf_gate_tail_cuda(const float * w, const float * x, size_t nb11, float * gout, size_t nb1, int n,
                           const float * o, const float * r, float * sig, float * mul, float * add, int64_t ne0, int64_t nelem,
                           cudaStream_t stream) {
    const int nblock = (int) ((nelem + 255) / 256);
    k_pxa_vf_gate_tail<<<nblock, 256, 0, stream>>>(w, x, nb11, gout, nb1, n, o, r, sig, mul, add, (int) ne0, (int) nelem);
}

void pxa_vf_ew_q8_1_cuda(int op, const float * a, const float * b, float * o1, float * o2, void * vy, float p0, float p1,
                         int64_t kx, int64_t kx1, int64_t kx0_padded, cudaStream_t stream) {
    GGML_ASSERT(kx0_padded % QK8_1 == 0);
    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    if (op == 0) {
        k_pxa_vf_ew_q8_1<0><<<num_blocks, block_size, 0, stream>>>(a, b, o1, o2, vy, p0, p1, kx, kx0_padded);
    } else {
        k_pxa_vf_ew_q8_1<1><<<num_blocks, block_size, 0, stream>>>(a, b, o1, o2, vy, p0, p1, kx, kx0_padded);
    }
}

void pxa_vf_sig_mul_add_cuda(const float * g, const float * o, const float * r, float * sig, float * mul, float * add,
                             int64_t ne0, int64_t nelem, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const int nblock = (int) ((nelem + kBlockSize - 1) / kBlockSize);
    k_pxa_vf_sig_mul_add<<<nblock, kBlockSize, 0, stream>>>(g, o, r, sig, mul, add, (int) ne0, (int) nelem);
}

// PXA_VFUSE bit 128: the qwen4exp DeltaNet gated norm. FUSED_RMS_NORM(x, w) [128, rows] -> SIGMOID(z) -> MUL(norm, sig) ->
// [quantize_q8_1] -> ssm_out in one launch. One block of 128 threads per 128-column row, as fused_rms_norm_f32<128> launches it
// (norm.cu, f32 source, no register cache): tmp += x*x, warp_reduce_sum, the 32-slot shared second reduce, mean = tmp / ncols,
// scale = rsqrtf(mean + eps), dst = scale * w * x; then sigmoid_f32, op_mul and the q8_1 of the product (with 128 % 32 == 0
// and an unpadded consumer row the warp of element e holds exactly q8_1 block e / 32, lane = e % 32). Every thread reads all
// of its inputs before it writes, so exact aliases stay index-for-index.
static __global__ void k_pxa_vf_gnorm_q8(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ z,
                                         float * __restrict__ fo, float * __restrict__ sig, float * __restrict__ mul,
                                         block_q8_1 * __restrict__ q8, const int ncols, const float eps) {
    constexpr int block_size = 128;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t e = (int64_t) row*ncols + tid;

    const float xi = x[e];
    float tmp = 0.0f;
    tmp += xi * xi;

    tmp = warp_reduce_sum(tmp);
    {
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
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    const float o  = scale * w[tid] * xi;                // fused_rms_norm_f32: scale * y[col] * x
    const float s  = 1.0f / (1.0f + expf(-z[e]));        // sigmoid_f32
    const float xm = __fmul_rn(o, s);                    // op_mul
    fo[e]  = o;
    sig[e] = s;
    mul[e] = xm;

    float amax = fabsf(xm);
    float sum  = xm;
    amax = warp_reduce_max(amax);
    sum  = warp_reduce_sum(sum);
    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xm / d);
    const int64_t ib = e / QK8_1, iqs = e % QK8_1;
    q8[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    reinterpret_cast<half&>(q8[ib].ds.x) = d;
    reinterpret_cast<half&>(q8[ib].ds.y) = sum;
}

void pxa_vf_gnorm_q8_cuda(const float * x, const float * w, const float * z, float * fo, float * sig, float * mul, void * q8,
                          int64_t nrows, float eps, cudaStream_t stream) {
    k_pxa_vf_gnorm_q8<<<(unsigned) nrows, 128, 0, stream>>>(x, w, z, fo, sig, mul, (block_q8_1 *) q8, 128, eps);
}
