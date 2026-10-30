// Copyright (c) 2026 PXA Network.
//
// PXA_FA_LONGPREFILL (2026-10-15; opt-in, default OFF until measured + KLD-gated).
// sm_60 prompt attention for head 256 over an f16 K/V cache. The tile-f16 route gives a block 16 tokens of ONE head and
// reads K, V, Q and the scores from shared memory with 32-bit loads (S: 0.5 shared loads per MAC, P*V: 0.31): on sm_60
// (16 LD/ST units vs 64 fp16x2 lanes per SM) those loops are load-bound. Here a block owns 32 rows = (32 / G) tokens x
// the G query heads of one KV head (GQA packing: the K/V tile is shared by every row), a warp owns 4 rows, and the
// operands are laid out so one 128-bit shared load feeds 4 rows:
//   Qt[d-pair][row]  (Q * scale, transposed)       S = Q K^T: lane = key; per d-pair one 128-bit Qt load (4 rows)
//   Kt[d-pair][key]  (K tile transposed)                + one 32-bit Kt load -> 4 HFMA2 (S: 0.25 loads per MAC)
//   Vp[key-pair][d]  (two keys interleaved per half2)   O += P V: lane = 8 output dims; per key pair one 128-bit P
//   Pp[key-pair][row]                                   load + two 128-bit V loads -> 32 HFMA2 (0.05 loads per MAC)
// S and the online softmax stay in registers (fp32, warp shuffles); fp16 products are summed in short chains (S: 4 chains
// of 8 dims per row, each chain converted to fp32 on its own; P V: 8 key pairs) and accumulated in fp32. The next K tile
// and the current V tile are loaded into registers while the math runs. Fully masked tiles are skipped. Not bit-identical
// to the tile route (different summation order and fp16 chains): opt-in, KLD-gated.
#include "fattn-longpf.cuh"
#include <cmath>

namespace pxa_lpf {

constexpr int D = 256, DP = D / 2, R = 32, KT = 32, KTP = KT / 2, NT = 256;

struct Args {
    const float * Q; int64_t q_t, q_h;      // Q f32 [D, T, Hq]: token / head strides (floats)
    const half  * K; int64_t k_p, k_h;      // K f16 [D, L, Hkv]: key / head strides (halves)
    const half  * V; int64_t v_p, v_h;
    const half  * M; int64_t m_t;           // mask f16 [L, T]: token stride (halves)
    float * O; int64_t o_t, o_h;            // dst f32 [D, Hq, T]: token / head strides (floats)
    int T, L, G, TB;                        // tokens, keys, GQA ratio, tokens per block (32 / G)
    float scale;
};

static __device__ __forceinline__ half2 u2h2(uint32_t u) { return *reinterpret_cast<half2 *>(&u); }

// LEAN (PXA_FA_LONGPREFILL=3): the same tile walk with fewer quarter-rate conversions on sm_60
// (F2F runs at 16 results/clk/SM there, HFMA2 at 128): S folds its 4 fp16 chains per row with 3 HADD2 and converts
// once per 32 dims (4 F2F pairs per row per 128 dims-chunk instead of 16); P V keeps one fp16 chain per output over
// the whole 32-key tile (one conversion per output per tile instead of two) and the exponentials use ex2.approx.
template <bool LEAN>
__global__ void __launch_bounds__(NT, 1) k_fa_longpf(const Args a) {
#if __CUDA_ARCH__ >= 600
    __shared__ __align__(16) half2 Qt[DP][R];        // 16 KB
    __shared__ __align__(16) half2 KVs[DP * KT];     // 16 KB: Kt[DP][KT] for S, then Vp[KTP][D] for P V
    __shared__ __align__(16) half2 Pp[KTP][R];       //  2 KB
    half2 (*Kt)[KT] = reinterpret_cast<half2 (*)[KT]>(KVs);
    half2 (*Vp)[D]  = reinterpret_cast<half2 (*)[D]>(KVs);

    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int t0 = blockIdx.x * a.TB, kvh = blockIdx.y, nrows = a.TB * a.G;

    {   // Q -> Qt: thread = (row tid>>3, 32 dims part tid&7); scale applied in fp32 before the fp16 rounding
        const int row = tid >> 3, part = tid & 7, t = t0 + row / a.G, h = kvh * a.G + row % a.G;
        const bool ok = row < nrows && t < a.T;
        const float * src = a.Q + (int64_t) t * a.q_t + (int64_t) h * a.q_h + 32 * part;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float2 v = ok ? *reinterpret_cast<const float2 *>(src + 2 * i) : make_float2(0.0f, 0.0f);
            Qt[16 * part + i][row] = __floats2half2_rn(v.x * a.scale, v.y * a.scale);
        }
    }
    const half * Kh = a.K + (int64_t) kvh * a.k_h;
    const half * Vh = a.V + (int64_t) kvh * a.v_h;

    float m[4], l[4], o[4][8];
    int   trow[4];                                   // token of each of the warp's rows (-1: idle row)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = 4 * w + i, t = t0 + row / a.G;
        trow[i] = (row < nrows && t < a.T) ? t : -1;
        m[i] = -INFINITY; l[i] = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) o[i][d] = 0.0f;
    }
    __syncthreads();

    uint4 kreg[4], vreg[4];
    int kreg_tile = -1;
    auto load_k = [&](int k0) {                      // lane = key k0 + lane, warp w = dims 32w .. 32w + 31
        const int key = k0 + lane;
        const uint4 * src = reinterpret_cast<const uint4 *>(Kh + (int64_t) key * a.k_p + 32 * w);
#pragma unroll
        for (int c = 0; c < 4; ++c) kreg[c] = key < a.L ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
        kreg_tile = k0;
    };
    auto load_v = [&](int k0) {                      // thread = (key pair tid>>4, 16 dims (tid&15)*16)
        const int kp = tid >> 4, dc = (tid & 15) * 16, ka = k0 + 2 * kp, kb = ka + 1;
        vreg[0] = vreg[1] = vreg[2] = vreg[3] = make_uint4(0, 0, 0, 0);
        if (ka < a.L) { const uint4 * p = reinterpret_cast<const uint4 *>(Vh + (int64_t) ka * a.v_p + dc); vreg[0] = __ldg(p); vreg[1] = __ldg(p + 1); }
        if (kb < a.L) { const uint4 * p = reinterpret_cast<const uint4 *>(Vh + (int64_t) kb * a.v_p + dc); vreg[2] = __ldg(p); vreg[3] = __ldg(p + 1); }
    };
    load_k(0);

    for (int k0 = 0; k0 < a.L; k0 += KT) {
        const int key = k0 + lane;
        float mk[4];
        bool any = false;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            mk[i] = (trow[i] >= 0 && key < a.L) ? __half2float(a.M[(int64_t) trow[i] * a.m_t + key]) : -INFINITY;
            any |= mk[i] != -INFINITY;
        }
        if (!__syncthreads_or(any)) {
            continue;                                // every row masks every key of this tile
        }
        if (kreg_tile != k0) {
            load_k(k0);
        }
        {
            const half2 * kh = reinterpret_cast<const half2 *>(kreg);
#pragma unroll
            for (int i = 0; i < 16; ++i) Kt[16 * w + i][lane] = kh[i];
        }
        load_v(k0);                                  // in flight during S
        __syncthreads();

        float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
        for (int c = 0; c < DP; c += 16) {
            half2 h[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int u = 0; u < 4; ++u) h[i][u] = __float2half2_rn(0.0f);
#pragma unroll
            for (int dp = c; dp < c + 16; ++dp) {
                const uint4 q4 = *reinterpret_cast<const uint4 *>(&Qt[dp][4 * w]);
                const half2 kk = Kt[dp][lane];
                const int u = dp & 3;
                h[0][u] = __hfma2(u2h2(q4.x), kk, h[0][u]);
                h[1][u] = __hfma2(u2h2(q4.y), kk, h[1][u]);
                h[2][u] = __hfma2(u2h2(q4.z), kk, h[2][u]);
                h[3][u] = __hfma2(u2h2(q4.w), kk, h[3][u]);
            }
            if (LEAN) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {        // fold the 4 chains in fp16, one conversion per row
                    const float2 f = __half22float2(__hadd2(__hadd2(h[i][0], h[i][1]), __hadd2(h[i][2], h[i][3])));
                    sc[i] += f.x + f.y;
                }
            } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int u = 0; u < 4; ++u) {        // each 8-product chain to fp32 on its own
                    const float2 f = __half22float2(h[i][u]);
                    sc[i] += f.x + f.y;
                }
            }
        }

        float alpha[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float s = sc[i] + mk[i];
            float mx = s;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float mn = fmaxf(m[i], mx);
            const float p  = mn == -INFINITY ? 0.0f : (LEAN ? __expf(s - mn) : expf(s - mn));
            alpha[i]       = mn == -INFINITY ? 1.0f : (LEAN ? __expf(m[i] - mn) : expf(m[i] - mn));
            float sum = p;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            l[i] = l[i] * alpha[i] + sum;
            m[i] = mn;
            const float pn = __shfl_down_sync(0xffffffff, p, 1);
            if ((lane & 1) == 0) Pp[lane >> 1][4 * w + i] = __floats2half2_rn(p, pn);
        }
        __syncthreads();                             // Kt reads done before Vp overwrites the buffer
        {
            const int kp = tid >> 4, dc = (tid & 15) * 16;
            const half * a0 = reinterpret_cast<const half *>(&vreg[0]), * a1 = reinterpret_cast<const half *>(&vreg[1]);
            const half * b0 = reinterpret_cast<const half *>(&vreg[2]), * b1 = reinterpret_cast<const half *>(&vreg[3]);
            half2 vv[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) { vv[e] = __halves2half2(a0[e], b0[e]); vv[8 + e] = __halves2half2(a1[e], b1[e]); }
#pragma unroll
            for (int e = 0; e < 16; e += 4) *reinterpret_cast<uint4 *>(&Vp[kp][dc + e]) = *reinterpret_cast<const uint4 *>(&vv[e]);
        }
        if (k0 + KT < a.L) {
            load_k(k0 + KT);                         // in flight during P V
        }
        __syncthreads();

        if (LEAN) {                                  // one fp16 chain per output over the 16 key pairs
            half2 h[4][8];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int d = 0; d < 8; ++d) h[i][d] = __float2half2_rn(0.0f);
#pragma unroll 4
            for (int kp = 0; kp < KTP; ++kp) {
                const uint4 p4 = *reinterpret_cast<const uint4 *>(&Pp[kp][4 * w]);
                const uint4 v0 = *reinterpret_cast<const uint4 *>(&Vp[kp][8 * lane]);
                const uint4 v1 = *reinterpret_cast<const uint4 *>(&Vp[kp][8 * lane + 4]);
                const half2 ph[4] = { u2h2(p4.x), u2h2(p4.y), u2h2(p4.z), u2h2(p4.w) };
                const half2 vh[8] = { u2h2(v0.x), u2h2(v0.y), u2h2(v0.z), u2h2(v0.w), u2h2(v1.x), u2h2(v1.y), u2h2(v1.z), u2h2(v1.w) };
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int d = 0; d < 8; ++d) h[i][d] = __hfma2(ph[i], vh[d], h[i][d]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int d = 0; d < 8; ++d) { const float2 f = __half22float2(h[i][d]); o[i][d] = o[i][d] * alpha[i] + (f.x + f.y); }
            __syncthreads();                         // Vp / Pp reads done before the next tile
            continue;
        }
        float acc[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int d = 0; d < 8; ++d) acc[i][d] = 0.0f;
#pragma unroll
        for (int kc = 0; kc < KTP; kc += 8) {
            half2 h[4][8];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int d = 0; d < 8; ++d) h[i][d] = __float2half2_rn(0.0f);
#pragma unroll
            for (int kp = kc; kp < kc + 8; ++kp) {
                const uint4 p4 = *reinterpret_cast<const uint4 *>(&Pp[kp][4 * w]);
                const uint4 v0 = *reinterpret_cast<const uint4 *>(&Vp[kp][8 * lane]);
                const uint4 v1 = *reinterpret_cast<const uint4 *>(&Vp[kp][8 * lane + 4]);
                const half2 ph[4] = { u2h2(p4.x), u2h2(p4.y), u2h2(p4.z), u2h2(p4.w) };
                const half2 vh[8] = { u2h2(v0.x), u2h2(v0.y), u2h2(v0.z), u2h2(v0.w), u2h2(v1.x), u2h2(v1.y), u2h2(v1.z), u2h2(v1.w) };
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int d = 0; d < 8; ++d) h[i][d] = __hfma2(ph[i], vh[d], h[i][d]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int d = 0; d < 8; ++d) { const float2 f = __half22float2(h[i][d]); acc[i][d] += f.x + f.y; }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int d = 0; d < 8; ++d) o[i][d] = o[i][d] * alpha[i] + acc[i][d];
        __syncthreads();                             // Vp / Pp reads done before the next tile
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (trow[i] < 0) continue;
        const int row = 4 * w + i, h = kvh * a.G + row % a.G;
        const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
        float * out = a.O + (int64_t) trow[i] * a.o_t + (int64_t) h * a.o_h + 8 * lane;
        *reinterpret_cast<float4 *>(out)     = make_float4(o[i][0] * inv, o[i][1] * inv, o[i][2] * inv, o[i][3] * inv);
        *reinterpret_cast<float4 *>(out + 4) = make_float4(o[i][4] * inv, o[i][5] * inv, o[i][6] * inv, o[i][7] * inv);
    }
#else
    NO_DEVICE_CODE;
#endif
}

} // namespace pxa_lpf

bool ggml_cuda_pxa_fa_longpf_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // =1: P100 (cc 600) only; =2: also sm_70 (a correctness check on a V100 -- the V100 route is the MMA kernel)
    // =3: the LEAN variant on P100 (cc 600); =4: LEAN also on sm_70 (correctness check)
    static const int on = [](){ const char * e = getenv("PXA_FA_LONGPREFILL"); return e ? atoi(e) : 0; }();
    if (on <= 0) return false;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool lean = on >= 3;
    if (!(cc == 600 || ((on == 2 || on == 4) && cc == 700))) return false;
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    if (mask == nullptr || dst->src[4] != nullptr) return false;           // needs the KQ mask; no attention sinks
    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f || dst->op_params[3] != GGML_PREC_DEFAULT) return false;
    const int64_t T = Q->ne[1];
    if (T <= 16 || Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) return false;
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) return false;
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || K->ne[2] <= 0) return false;
    if (Q->ne[2] % K->ne[2] != 0) return false;
    const int G = (int) (Q->ne[2] / K->ne[2]);
    if (G < 2 || G > 16) return false;
    if (mask->ne[0] < K->ne[1] || mask->ne[1] < T || mask->ne[2] != 1 || mask->ne[3] != 1) return false;
    // =3/=4: only at long KV windows (PXA_FA_LONGPREFILL_MINKV, default 8192): below it the tile route (with
    // PXA_FA_TILE_H2CHUNK when set) was faster on 2x P100 (35B pp512 -6%, pp4096 -1.5%; pp16384 +5%, bracket b1 2026-10-15)
    static const int64_t minkv = [](){ const char * e = getenv("PXA_FA_LONGPREFILL_MINKV"); return e ? (int64_t) atoll(e) : (int64_t) 8192; }();
    if (lean && K->ne[1] < minkv) return false;
    if (dst->ne[0] != 256 || dst->ne[1] != Q->ne[2] || dst->ne[2] != T) return false;
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 8 || Q->nb[2] % 8 || ((uintptr_t) Q->data) % 8) return false;
    if (K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || K->nb[1] % 16 || K->nb[2] % 16 || V->nb[1] % 16 || V->nb[2] % 16) return false;
    if (((uintptr_t) K->data) % 16 || ((uintptr_t) V->data) % 16 || mask->nb[0] != sizeof(half)) return false;
    if (dst->nb[0] != sizeof(float) || dst->nb[1] % 16 || dst->nb[2] % 16 || ((uintptr_t) dst->data) % 16) return false;
    pxa_lpf::Args a = {};
    a.Q = (const float *) Q->data; a.q_t = Q->nb[1] / sizeof(float); a.q_h = Q->nb[2] / sizeof(float);
    a.K = (const half *) K->data;  a.k_p = K->nb[1] / sizeof(half);  a.k_h = K->nb[2] / sizeof(half);
    a.V = (const half *) V->data;  a.v_p = V->nb[1] / sizeof(half);  a.v_h = V->nb[2] / sizeof(half);
    a.M = (const half *) mask->data; a.m_t = mask->nb[1] / sizeof(half);
    a.O = (float *) dst->data; a.o_h = dst->nb[1] / sizeof(float); a.o_t = dst->nb[2] / sizeof(float);
    a.T = (int) T; a.L = (int) K->ne[1]; a.G = G; a.TB = pxa_lpf::R / G; a.scale = scale;
    const dim3 grid((unsigned) ((T + a.TB - 1) / a.TB), (unsigned) K->ne[2]);
    if (lean) {
        pxa_lpf::k_fa_longpf<true><<<grid, pxa_lpf::NT, 0, ctx.stream()>>>(a);
    } else {
        pxa_lpf::k_fa_longpf<false><<<grid, pxa_lpf::NT, 0, ctx.stream()>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
    static bool logged = false;
    if (!logged) { logged = true; fprintf(stderr, "PXA_FA_LONGPREFILL: engaged (sm_60 head-256 prompt attention, GQA %d, 32-row blocks, %s; opt-in, not bit-identical)\n", G, lean ? "lean conversions" : "short chains"); }
    return true;
}
