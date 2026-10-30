// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// kq-vw.cuh -- Q8_0 GEMV at speculative-verify widths (2..6 columns) on sm_60 (PXA_KQMMV_VW, default OFF).
//
// The LM head (248k rows) at verify widths ran the chunked k_kq_mmv form at 1.5 / 2.0 / 2.5 / 3.0 ms for 2 / 3 / 4 / 5
// columns on a P100 (1.07 ms at one column): the h2 dot re-reads x from shared memory per row group and column. Here
// every block stages ALL columns once (fp16 with a power-of-two scale per 32-block, |x| < 2^5 so a chain of 8
// int8 x fp16 products stays < 2^15; each 32-block padded to 80 B so the lanes' 128-bit x reads are conflict-free) and a warp streams whole rows: each row as 1088 B chunks (32 Q8_0 blocks) staged
// through shared memory with the next two chunks in registers across row boundaries, lane = one block, every column
// dotted from the same decoded half2 weights, per-16 flush to fp32. The weight stream is read once for all columns
// (memory-bound target ~1 ms for the head at any width <= 6). Up to 4 matrices per launch (rows concatenated).
// Not bit-identical to the k_kq_mmv columns (staging granularity / summation order): opt-in.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#define KQVW_NT    512
#define KQVW_NW    (KQVW_NT / 32)
#define KQVW_CH    1088            // 32 Q8_0 blocks = 1024 values
#define KQVW_XMAX  10240           // staged values (ny * K)
#define KQVW_XB    40              // halves per staged 32-block (+16 B pad: the 8 lanes of a 128-bit quarter-warp hit 8 bank groups)

struct kqvw_args {
    const uint8_t * w[4];
    float *         y[4];
    int64_t         nb01[4];       // row stride (bytes)
    int64_t         sy[4];         // dst column stride (floats)
    int             r0[5];         // prefix of rows
    int             nm, K;
    const float *   x;
    int64_t         sx;            // x column stride (floats)
};

static __device__ __forceinline__ float kqvw_wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int NY>
static __global__ void __launch_bounds__(KQVW_NT, 1) k_kqvw(const kqvw_args a) {
    __shared__ __align__(16) __half xh[KQVW_XMAX / 32 * KQVW_XB];
    __shared__ float xsc[KQVW_XMAX / 32];
    __shared__ __align__(16) uint8_t stage[KQVW_NW][KQVW_CH];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int K = a.K, nb = K / 32;
    // stage x: a warp item = one 32-block of one column
    for (int it0 = warp; it0 < NY * nb; it0 += 8 * KQVW_NW) {
      float vv[8];
#pragma unroll
      for (int q = 0; q < 8; q++) {
          const int it = it0 + q * KQVW_NW;
          vv[q] = it < NY * nb ? a.x[(int64_t) (it / nb) * a.sx + 32 * (it % nb) + lane] : 0.f;
      }
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int it = it0 + q * KQVW_NW;
        if (it >= NY * nb) break;
        const int c = it / nb, b = it % nb;
        const float v = vv[q];
        float am = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
        float inv = 1.f, s = 1.f;
        if (am > 0.f) {
            int ex = ((__float_as_int(am) >> 23) & 0xFF) - 126;
            ex = ex < -100 ? -100 : (ex > 100 ? 100 : ex);
            inv = __int_as_float((127 + 5 - ex) << 23);
            s   = __int_as_float((127 + ex - 5) << 23);
        }
        xh[(c * nb + b) * KQVW_XB + lane] = __float2half_rn(v * inv);
        if (lane == 0) xsc[c * nb + b] = s;
      }
    }
    __syncthreads();

    const int R = a.r0[a.nm];
    const int gw = warp * gridDim.x + blockIdx.x, nwg = KQVW_NW * gridDim.x;
    const int nch = K / 1024;                                       // chunks per row
    const int nrow = gw < R ? (R - 1 - gw) / nwg + 1 : 0, nt = nrow * nch;
    uint8_t * st = stage[warp];
    auto rowp = [&](int r) -> const uint8_t * {
        int m = 0;
        while (m + 1 < a.nm && r >= a.r0[m + 1]) m++;
        return a.w[m] + (int64_t) (r - a.r0[m]) * a.nb01[m];
    };
    int4 b0[3], b1[3];
    auto ld = [&](int t, int4 * bb) {
        const int r = gw + (t / nch) * nwg;
        const int4 * p = (const int4 *) (rowp(r) + (t % nch) * KQVW_CH);
        bb[0] = __ldg(p + lane);
        bb[1] = __ldg(p + 32 + lane);
        if (lane < 4) bb[2] = __ldg(p + 64 + lane);
    };
    if (nt > 0) ld(0, b0);
    if (nt > 1) ld(1, b1);
    float acc[NY];
#pragma unroll
    for (int c = 0; c < NY; c++) acc[c] = 0.f;
    const __half2 k1152 = __half2half2(__ushort_as_half((unsigned short) 0x6480u));
    for (int t = 0; t < nt; t++) {
        int4 * s = (int4 *) st;
        s[lane] = b0[0];
        s[32 + lane] = b0[1];
        if (lane < 4) s[64 + lane] = b0[2];
        __syncwarp();
        b0[0] = b1[0]; b0[1] = b1[1]; b0[2] = b1[2];
        if (t + 2 < nt) ld(t + 2, b1);
        {
            const uint8_t * bp = st + 34 * lane;
            const float d = __half2float(*(const __half *) bp);
            const unsigned short * q = (const unsigned short *) (bp + 2);
            __half2 wv[16];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint32_t v = (q[2 * i] | ((uint32_t) q[2 * i + 1] << 16)) ^ 0x80808080u;
                const uint32_t u01 = __byte_perm(v, 0x64646464u, 0x5140);   // bytes 0,1 -> 0x64b0, 0x64b1
                const uint32_t u23 = __byte_perm(v, 0x64646464u, 0x5342);
                wv[2 * i]     = __hsub2(*(const __half2 *) &u01, k1152);
                wv[2 * i + 1] = __hsub2(*(const __half2 *) &u23, k1152);
            }
            const int xb = 32 * (t % nch) + lane;                      // 32-block of x
#pragma unroll
            for (int c = 0; c < NY; c++) {
                const float4 * xp = (const float4 *) (xh + (c * nb + xb) * KQVW_XB);
                float part = 0.f;
#pragma unroll
                for (int g = 0; g < 2; g++) {
                    const float4 x0 = xp[2 * g], x1 = xp[2 * g + 1];
                    __half2 h = __hmul2(wv[8 * g + 0], *(const __half2 *) &x0.x);
                    h = __hfma2(wv[8 * g + 1], *(const __half2 *) &x0.y, h);
                    h = __hfma2(wv[8 * g + 2], *(const __half2 *) &x0.z, h);
                    h = __hfma2(wv[8 * g + 3], *(const __half2 *) &x0.w, h);
                    h = __hfma2(wv[8 * g + 4], *(const __half2 *) &x1.x, h);
                    h = __hfma2(wv[8 * g + 5], *(const __half2 *) &x1.y, h);
                    h = __hfma2(wv[8 * g + 6], *(const __half2 *) &x1.z, h);
                    h = __hfma2(wv[8 * g + 7], *(const __half2 *) &x1.w, h);
                    const float2 f = __half22float2(h);
                    part += f.x + f.y;
                }
                acc[c] += part * d * xsc[c * nb + xb];
            }
        }
        __syncwarp();
        if (t % nch == nch - 1) {
            const int r = gw + (t / nch) * nwg;
            int m = 0;
            while (m + 1 < a.nm && r >= a.r0[m + 1]) m++;
#pragma unroll
            for (int c = 0; c < NY; c++) {
                const float v = kqvw_wsum(acc[c]);
                if (lane == 0) a.y[m][(int64_t) c * a.sy[m] + (r - a.r0[m])] = v;
                acc[c] = 0.f;
            }
        }
    }
}

// small launches (<= KQVW_SMALL_R rows, e.g. the delta-net alpha / beta pair: 64 rows): one block per 2 rows, the 16 warps
// split K (lane = 8 values of one row), fp32 dot straight from global x, warp shuffles + a fixed-order 16-way smem
// reduce. No block-wide x staging: one load round trip instead of the staged form's latency chain.
#define KQVW_SMALL_R 256
template <int NY>
static __global__ void __launch_bounds__(512) k_kqvw_small(const kqvw_args a) {
    __shared__ float red[16][2][NY];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int rs = lane >> 4, j = (warp << 4) | (lane & 15);       // row in the pair, 8-value slot of K = 2048
    const int r = 2 * blockIdx.x + rs, R = a.r0[a.nm];
    float acc[NY];
#pragma unroll
    for (int c = 0; c < NY; c++) acc[c] = 0.f;
    if (r < R && 8 * j < a.K) {
        int m = 0;
        while (m + 1 < a.nm && r >= a.r0[m + 1]) m++;
        const uint8_t * bp = a.w[m] + (int64_t) (r - a.r0[m]) * a.nb01[m] + 34 * (j >> 2);
        const float d = __half2float(*(const __half *) bp);
        const unsigned short * q = (const unsigned short *) (bp + 2 + 8 * (j & 3));
        float qf[8];
#pragma unroll
        for (int i = 0; i < 4; i++) { const unsigned short u = q[i]; qf[2 * i] = (float) (int8_t) (u & 0xff); qf[2 * i + 1] = (float) (int8_t) (u >> 8); }
#pragma unroll
        for (int c = 0; c < NY; c++) {
            const float4 * xp = (const float4 *) (a.x + (int64_t) c * a.sx + 8 * j);
            const float4 x0 = xp[0], x1 = xp[1];
            acc[c] = d * (qf[0] * x0.x + qf[1] * x0.y + qf[2] * x0.z + qf[3] * x0.w + qf[4] * x1.x + qf[5] * x1.y + qf[6] * x1.z + qf[7] * x1.w);
        }
    }
#pragma unroll
    for (int c = 0; c < NY; c++) {
        float v = acc[c];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if ((lane & 15) == 0) red[warp][rs][c] = v;
    }
    __syncthreads();
    if (threadIdx.x < 2 * NY) {
        const int rr = threadIdx.x / NY, c = threadIdx.x % NY, row = 2 * blockIdx.x + rr;
        if (row < R) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < 16; w++) v += red[w][rr][c];
            int m = 0;
            while (m + 1 < a.nm && row >= a.r0[m + 1]) m++;
            a.y[m][(int64_t) c * a.sy[m] + (row - a.r0[m])] = v;
        }
    }
}

// PXA_KQMMV_VW=2: the same streaming schedule (persistent, 16 warps per block, whole rows as 1088 B chunks
// staged per warp, the next two chunks in registers) with k_kq_mmv's per-row arithmetic: x staged by kq_stage (fp16 with a
// power-of-two scale per 32-block, padded 40 halves per block), lane = task (one Q8_0 block) of each chunk, kq_task's
// two 16-element half2 parts and fmaf(f, d * scale, acc), the same xor reduce. Column c at any width is bit-identical to the
// one-column k_kq_mmv launch of column c (the decode head): the verify head becomes width-invariant. Needs kq-mmv.cuh first.
#ifdef KQMMV_CHUNK
template <int NY>
static __global__ void __launch_bounds__(KQVW_NT, 1) k_kqhv(const kqvw_args a) {
    extern __shared__ __align__(16) unsigned char kqhv_smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int K = a.K;
    unsigned char * xq = kqhv_smem;                                            // [NY][K + K/4] halves
    float * xs = (float *) (kqhv_smem + (size_t) NY * (K + K / 4) * 2);        // [NY][K/32]
    uint8_t * st = (uint8_t *) (xs + NY * (K / 32)) + (size_t) warp * KQVW_CH; // per-warp chunk
    kq_stage<NY, KQMMV_H2, false>(a.x, a.sx, K, 0, K, xq, xs, nullptr, lane, warp, KQVW_NW, 8);
    __syncthreads();
    const kq_xview<KQMMV_H2> X{xq, xs, nullptr, K, 8};

    const int R = a.r0[a.nm];
    const int gw = warp * gridDim.x + blockIdx.x, nwg = KQVW_NW * gridDim.x;
    const int nch = K / 1024;
    const int nrow = gw < R ? (R - 1 - gw) / nwg + 1 : 0, nt = nrow * nch;
    auto rowp = [&](int r) -> const uint8_t * {
        int m = 0;
        while (m + 1 < a.nm && r >= a.r0[m + 1]) m++;
        return a.w[m] + (int64_t) (r - a.r0[m]) * a.nb01[m];
    };
    int4 b0[3], b1[3];
    auto ld = [&](int t, int4 * bb) {
        const int r = gw + (t / nch) * nwg;
        const int4 * p = (const int4 *) (rowp(r) + (t % nch) * KQVW_CH);
        bb[0] = __ldg(p + lane);
        bb[1] = __ldg(p + 32 + lane);
        if (lane < 4) bb[2] = __ldg(p + 64 + lane);
    };
    if (nt > 0) ld(0, b0);
    if (nt > 1) ld(1, b1);
    float acc[NY];
#pragma unroll
    for (int c = 0; c < NY; c++) acc[c] = 0.f;
    for (int t = 0; t < nt; t++) {
        int4 * s = (int4 *) st;
        s[lane] = b0[0];
        s[32 + lane] = b0[1];
        if (lane < 4) s[64 + lane] = b0[2];
        __syncwarp();
        b0[0] = b1[0]; b0[1] = b1[1]; b0[2] = b1[2];
        if (t + 2 < nt) ld(t + 2, b1);
        kq_task<KQ_Q8_0, NY, KQMMV_H2>(st, 0, lane, (t % nch) * KQMMV_CHUNK, X, acc);
        __syncwarp();
        if (t % nch == nch - 1) {
            const int r = gw + (t / nch) * nwg;
            int m = 0;
            while (m + 1 < a.nm && r >= a.r0[m + 1]) m++;
#pragma unroll
            for (int c = 0; c < NY; c++) {
                float v = acc[c];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
                if (lane == 0) a.y[m][(int64_t) c * a.sy[m] + (r - a.r0[m])] = v;
                acc[c] = 0.f;
            }
        }
    }
}
// + 16 B: kq_task's unaligned 16 B reads (kq_ld16) of the last block of a chunk run 4 B past it
static size_t kqhv_smem_bytes(int ny, int K) { return (size_t) ny * (K + K / 4) * 2 + (size_t) ny * (K / 32) * 4 + (size_t) KQVW_NW * KQVW_CH + 16; }
// launch: ny 2..5, K % 1024 == 0, ny * K <= KQVW_XMAX, nm <= 4 (the whole K staged per block, < 48 KiB)
static cudaError_t kqhv_launch(const kqvw_args & a, int ny, int nsm, cudaStream_t stream) {
    const size_t sm = kqhv_smem_bytes(ny, a.K);
    if (sm > 48 * 1024) return cudaErrorInvalidValue;
    const dim3 grid(nsm), block(KQVW_NT);
    switch (ny) {
        case 2: k_kqhv<2><<<grid, block, sm, stream>>>(a); break;
        case 3: k_kqhv<3><<<grid, block, sm, stream>>>(a); break;
        case 4: k_kqhv<4><<<grid, block, sm, stream>>>(a); break;
        case 5: k_kqhv<5><<<grid, block, sm, stream>>>(a); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
#endif

// launch: ny 2..5, K % 1024 == 0, ny * K <= KQVW_XMAX, nm <= 4
static cudaError_t kqvw_launch(const kqvw_args & a, int ny, int nsm, cudaStream_t stream) {
    if (a.r0[a.nm] <= KQVW_SMALL_R && a.K == 2048 && ((uintptr_t) a.x) % 16 == 0 && a.sx % 4 == 0) {
        const dim3 sg((a.r0[a.nm] + 1) / 2), sb(512);
        switch (ny) {
            case 2: k_kqvw_small<2><<<sg, sb, 0, stream>>>(a); break;
            case 3: k_kqvw_small<3><<<sg, sb, 0, stream>>>(a); break;
            case 4: k_kqvw_small<4><<<sg, sb, 0, stream>>>(a); break;
            case 5: k_kqvw_small<5><<<sg, sb, 0, stream>>>(a); break;
            default: return cudaErrorInvalidValue;
        }
        return cudaGetLastError();
    }
    const dim3 grid(nsm), block(KQVW_NT);
    switch (ny) {
        case 2: k_kqvw<2><<<grid, block, 0, stream>>>(a); break;
        case 3: k_kqvw<3><<<grid, block, 0, stream>>>(a); break;
        case 4: k_kqvw<4><<<grid, block, 0, stream>>>(a); break;
        case 5: k_kqvw<5><<<grid, block, 0, stream>>>(a); break;
        case 6: k_kqvw<6><<<grid, block, 0, stream>>>(a); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
