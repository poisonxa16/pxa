// grouped_moe_verify.cuh — A1 batched-MoE-verify: expert-GROUPED up+gate for
// Ny>=2 MTP/spec-decode verify batches. Read each routed expert's MXFP4 weights
// ONCE across the Ny verify tokens instead of once per token.
//
// v1 (2026-07-06): scalar-F32 Milestone A kernel, host-sync plan readback.
// v2 (2026-07-07): HALF2 kernel (smem-staged coalesced u32 weight loads,
//     __hfma2 with FP32-promote-per-block), NO host sync (grid oversubscribed on
//     a device n_union), shadow-verify mode, and the v1 OPERAND-ORDER FIX:
//
// ⚠⚠ v1 BUG (fixed in v2): the engine's fused per-token path binds
//     src0_dd_u = src0_1 (UP) and src0_dd_g = src0_2 (GATE) — see the
//     `ggml_cuda_op_fused_mul_mat_vec_q_id(..., src0_dd_u, src0_dd_g, ...)`
//     signature (mmvq.cu) and the callsite passing (src0_1->data, src0_2->data).
//     v1 passed src0_1->data into its `gate_base` parameter — SWAPPED. With the
//     asymmetric SILU epilogue (r = clamp(u) * silu(g), iqk_mmvq_templates.cuh)
//     that computes silu(up)*clamp(gate) = plausible garbage. v2 binds by name:
//     up = src0_1, gate = src0_2, and the shadow mode exists to prove it in-graph.
//
// v3 / gk6-iqk (2026-07-08, Tier 2 — the BIT-EXACT path): a grouped
//     instantiation of the ENGINE's OWN fused MMVQ. Per distinct expert bin it
//     calls the identical vec_dot_mxfp4_q8_1 q8_1-integer-dot against the SAME
//     src1_quantized activations the per-token path uses, with the SAME nwarps=1
//     lane/block assignment, the SAME warp_reduce_sum reduction, and the SAME
//     SILU epilogue. No f16, no from-scratch numerics. Result is memcmp==0 vs the
//     per-token reference BY CONSTRUCTION (only weight/y/dst ADDRESSING differs;
//     the float accumulation sequence is identical). This is the only kernel that
//     can pass the exact-equality G2 gate. (Internal fix-guide §2.2.)
//
// ENV (read once, default OFF — one-line rollback):
//   PXA_MOE_BATCHED_VERIFY (preferred; legacy alias PXA_MOE_GROUPED):
//     bit0 (1) = grouped up+gate ON. Kernel selected by the high bits:
//       (default, bit0 only) = gk6-iqk BIT-EXACT engine-vec_dot path (the ship path)
//       bit3 (8) = gk5 half2 kernel (speed A/B only; tolerance-G2, never prod default)
//       bit2 (4) = v1-style scalar-F32 kernel (debug)
//   PXA_MOE_GROUPED_VERIFY=1 = shadow mode: grouped writes a private scratch,
//     the per-token path still produces dst, a diff kernel device-printfs any
//     mismatch. For gk6 the diff is EXACT-equality (diff==0 required); for the
//     half2/scalar kernels it is the 2%-relative + isfinite tolerance (the G2 gate).
//     ~2x cost; debug only.
//
// Correctness provenance (microbench, 2026-07-07, kernel-dev/moe_verify_bench_v5.cu
// at real 122B-class MoE dims (256-expert/top-8/d_model 3072/d_ff 1024):
//   grouped vs per-application on the SAME kernel = BIT-EXACT (memcmp==0);
//   half2 vs FP32 golden rel_to_scale ~1.2e-3 (budget 1e-2). (Internal
//   A1 MoE-verify results, 2026-07-07.)
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

#define PXA_GK5_WPB 4     // warps (rows) per block

static inline int pxa_moe_grouped_flags() {
    static const int f = [](){
        const char* e = getenv("PXA_MOE_BATCHED_VERIFY");
        if (!e) e = getenv("PXA_MOE_GROUPED");
        return e ? atoi(e) : 0;
    }();
    return f;
}
static inline bool pxa_moe_grouped_shadow() {
    // PXA_TYPEARG_STRICT_v1-adjacent fix (2026-07-30): this was PRESENCE-tested, so
    // PXA_MOE_GROUPED_VERIFY=0 still ENABLED shadow-verify — the only PXA_* lever whose
    // =0 meant ON (flagged by the 2026-07-29 A14 lever inventory). Value-test like every
    // other lever: unset or =0 -> off, any nonzero -> on.
    static const bool v = [](){
        const char * e = getenv("PXA_MOE_GROUPED_VERIFY");
        return e && atoi(e) != 0;
    }();
    return v;
}

// exact engine MXFP4 scale: d(e) = 2^(e-128) incl. e<2 subnormals (convert.cu)
static __device__ __forceinline__ float pxa_mxfp4_d(uint8_t e) {
    const unsigned u = (e >= 2) ? ((unsigned)(e-1) << 23)
                                : (e ? 0x00400000u : 0x00200000u);
    return __int_as_float((int)u);
}

// ---------------------------------------------------------------------------
// Plan builder (device, single block): CSR grouping of the Ny*n_ids routed
// selections by distinct expert. Emits g_experts/g_offsets CSR + per-member
// token index and RT-1 scatter key (token*n_ids + slot). n_union stays ON
// DEVICE (kernels read it; grid is oversubscribed to Ny*n_ids bins) — no
// host readback, no stream stall.
// ---------------------------------------------------------------------------
__global__ void pxa_build_plan(const int* __restrict__ ids, int64_t ids_stride_row /*ints*/,
                               int Ny, int n_ids,
                               int* g_experts, int* g_offsets, int* g_tok, int* g_slot,
                               int* n_union_out) {
    __shared__ int uniq[64];             // NN = Ny*n_ids <= 64 (Ny<=8, n_ids<=8)
    const int N = Ny * n_ids;
    if (threadIdx.x != 0) return;        // serial: N<=64, cheaper than atomics
    int nu = 0;
    for (int i = 0; i < N; ++i) {
        const int t = i / n_ids, s = i % n_ids;
        const int e = ids[(int64_t)t*ids_stride_row + s];
        if (e < 0) continue;                 // SER (-ser) emits negative ids: no bin, no member
        int u = -1;
        for (int j = 0; j < nu; ++j) if (uniq[j] == e) { u = j; break; }
        if (u < 0) { uniq[nu] = e; ++nu; }
    }
    int off = 0;
    for (int u = 0; u < nu; ++u) {
        g_experts[u] = uniq[u];
        g_offsets[u] = off;
        for (int i = 0; i < N; ++i) {
            const int t = i / n_ids, s = i % n_ids;
            if (ids[(int64_t)t*ids_stride_row + s] == uniq[u]) {
                g_tok[off]  = t;
                g_slot[off] = (t << 16) | s;   // RT-1 key, packed (t,s)
                ++off;
            }
        }
    }
    g_offsets[nu] = off;
    *n_union_out = nu;
}

// F32 (strided) -> compact half [Ny][d_model]
__global__ void pxa_f32_to_half_rows(const float* __restrict__ src, int64_t src_stride_f,
                                     __half* __restrict__ dst, int d_model, int Ny) {
    const int t = blockIdx.y;
    for (int k = blockIdx.x*blockDim.x + threadIdx.x; k < d_model; k += gridDim.x*blockDim.x)
        dst[(size_t)t*d_model + k] = __float2half(src[(int64_t)t*src_stride_f + k]);
    (void)Ny;
}

// ---------------------------------------------------------------------------
// gk5 (v2, half2): grouped up+gate. grid=(ceil(d_ff/WPB), Ny*n_ids [oversub]),
// block=(32, WPB), dynamic smem = WPB*2*WROW u32 (WROW = d_model/32*17/4).
// Each warp owns one d_ff row f of one distinct-expert bin u:
//   stage gate+up MXFP4 rows to smem with coalesced u32 loads -> dequant each
//   32-block ONCE (smem LUT; qs bytes assembled to registers) -> __hfma2 across
//   all member token-columns (m <= Ny) -> FP32-promote per block -> warp-reduce
//   -> engine SILU epilogue -> scatter r to dst[tok*dst_nb2_f + slot_key*d_ff... ]
// RT-1: the scatter key is token*n_ids+slot (g_slot), matching the engine's
// dst layout out(t,s,f) = dst + t*nb2/4 + s*d_ff + f.
// ---------------------------------------------------------------------------
template<int MAXC>
__global__ void pxa_gk5_gateup_h2(
        const void* __restrict__ up_base,      // src0_1->data  (UP — see header note)
        const void* __restrict__ gate_base,    // src0_2->data  (GATE)
        int64_t w_nb1, int64_t w_nb2,          // bytes
        const __half* __restrict__ Xh,         // compact [Ny][d_model]
        const int* __restrict__ g_experts, const int* __restrict__ g_offsets,
        const int* __restrict__ g_tok, const int* __restrict__ g_slot,
        const int* __restrict__ n_union,       // device
        float* __restrict__ dst_base, int64_t dst_nb2_f /*floats per token*/,
        int d_model, int d_ff, float limit) {
    const int u    = blockIdx.y;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int NBI  = d_model >> 5;             // 32-blocks per row
    const int WROW = (NBI*17) >> 2;            // u32 words per row

    extern __shared__ unsigned pxa_smem[];     // [WPB][2*WROW]
    __shared__ int   s_tok[PXA_GK5_WPB][MAXC];
    __shared__ int   s_dst[PXA_GK5_WPB][MAXC];
    __shared__ float s_lut[16];
    if (threadIdx.y == 0 && lane < 16) s_lut[lane] = (float)kvalues_mxfp4[lane];
    __syncthreads();                            // before ANY divergent early-out

    if (u >= *n_union) return;                  // oversubscribed grid tail
    const int m0 = g_offsets[u], m = g_offsets[u+1] - m0;
    if (m <= 0) return;
    const int e = g_experts[u];
    const int f = blockIdx.x*PXA_GK5_WPB + warp;
    if (f >= d_ff) return;

    if (lane < m && lane < MAXC) {
        s_tok[warp][lane] = g_tok[m0+lane];
        s_dst[warp][lane] = g_slot[m0+lane];
    }
    {   // coalesced u32 staging of this row's up+gate blocks
        const unsigned* gu = (const unsigned*)((const char*)up_base   + (int64_t)e*w_nb2 + (int64_t)f*w_nb1);
        const unsigned* gg = (const unsigned*)((const char*)gate_base + (int64_t)e*w_nb2 + (int64_t)f*w_nb1);
        unsigned* su = pxa_smem + (size_t)warp*(2*WROW);
        unsigned* sg = su + WROW;
        #pragma unroll 4
        for (int w = lane; w < WROW; w += 32) { su[w] = __ldg(gu+w); sg[w] = __ldg(gg+w); }
    }
    __syncwarp();

    const unsigned char* sub = (const unsigned char*)(pxa_smem + (size_t)warp*(2*WROW));
    const unsigned char* sgb = sub + (size_t)WROW*4;
    float accg[MAXC], accu[MAXC];
    #pragma unroll
    for (int c = 0; c < MAXC; ++c) { accg[c]=0.f; accu[c]=0.f; }

    for (int ib = lane; ib < NBI; ib += 32) {
        const unsigned char* bu = sub + ib*17;
        const unsigned char* bg = sgb + ib*17;
        const float du = pxa_mxfp4_d(bu[0]);
        const float dg = pxa_mxfp4_d(bg[0]);
        unsigned quw[4], qgw[4];                // qs bytes -> registers once
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned char* u4 = bu + 1 + 4*w;
            const unsigned char* g4 = bg + 1 + 4*w;
            quw[w] = (unsigned)u4[0] | ((unsigned)u4[1]<<8) | ((unsigned)u4[2]<<16) | ((unsigned)u4[3]<<24);
            qgw[w] = (unsigned)g4[0] | ((unsigned)g4[1]<<8) | ((unsigned)g4[2]<<16) | ((unsigned)g4[3]<<24);
        }
        half2 hg[MAXC], hu[MAXC];
        #pragma unroll
        for (int c = 0; c < MAXC; ++c) { hg[c] = __float2half2_rn(0.f); hu[c] = hg[c]; }
        // 4 half2 pairs per step; X via one uint4 (4 half2) load per member per
        // step — the member-dimension load cost bounds the grouped win (measured).
        #pragma unroll
        for (int p4 = 0; p4 < 4; ++p4) {
            const bool hi = (p4 >= 2);              // pairs 0..7 lo-nibble, 8..15 hi
            half2 wg2[4], wu2[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int p  = p4*4 + q;
                const int j0 = hi ? 2*p-16 : 2*p;
                const unsigned uword = quw[j0>>2], gword = qgw[j0>>2];
                const int sh = (j0 & 3)*8;
                const unsigned ub0 = (uword >> sh) & 0xff, ub1 = (uword >> (sh+8)) & 0xff;
                const unsigned gb0 = (gword >> sh) & 0xff, gb1 = (gword >> (sh+8)) & 0xff;
                wu2[q] = __floats2half2_rn(du * s_lut[hi ? (ub0>>4) : (ub0&0xf)],
                                           du * s_lut[hi ? (ub1>>4) : (ub1&0xf)]);
                wg2[q] = __floats2half2_rn(dg * s_lut[hi ? (gb0>>4) : (gb0&0xf)],
                                           dg * s_lut[hi ? (gb1>>4) : (gb1&0xf)]);
            }
            #pragma unroll
            for (int c = 0; c < MAXC; ++c) {
                if (c >= m) break;
                const uint4 xw = __ldg((const uint4*)((const half2*)(Xh + (size_t)s_tok[warp][c]*d_model) + ib*16) + p4);
                const half2* xv = (const half2*)&xw;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    hu[c] = __hfma2(wu2[q], xv[q], hu[c]);
                    hg[c] = __hfma2(wg2[q], xv[q], hg[c]);
                }
            }
        }
        #pragma unroll
        for (int c = 0; c < MAXC; ++c) {        // FP32-promote per 32-block
            if (c >= m) break;
            const float2 fg = __half22float2(hg[c]);
            const float2 fu = __half22float2(hu[c]);
            accg[c] += fg.x + fg.y;
            accu[c] += fu.x + fu.y;
        }
    }
    #pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        if (c >= m) break;
        for (int o = 16; o > 0; o >>= 1) {
            accg[c] += __shfl_down_sync(0xffffffff, accg[c], o);
            accu[c] += __shfl_down_sync(0xffffffff, accu[c], o);
        }
    }
    if (lane == 0) {
        for (int c = 0; c < m; ++c) {
            // ENGINE SILU epilogue VERBATIM (iqk_mmvq_templates.cuh): u=up, g=gate
            float g = accg[c], uu = accu[c];
            g = g/(1.0f + expf(-g));
            g = fminf(g, limit);
            const float r = fmaxf(-limit, fminf(limit, uu)) * g;
            // RT-1 scatter: out(t,s,f) = dst + t*dst_nb2_f + s*d_ff + f;
            // key packs (t<<16)|s (built in pxa_build_plan).
            const int key = s_dst[warp][c];
            dst_base[(size_t)(key >> 16) * dst_nb2_f + (size_t)(key & 0xffff) * d_ff + f] = r;
        }
    }
}

// ---------------------------------------------------------------------------
// v1-style scalar-F32 fallback kernel (bit2): reads F32 activations directly,
// operands bound CORRECTLY (up/gate by name). Kept for A/B and half2 debugging.
// ---------------------------------------------------------------------------
template<int WPB>
__global__ void pxa_gk_gateup_scalar(
        const void* __restrict__ up_base, const void* __restrict__ gate_base,
        int64_t w_nb1, int64_t w_nb2 /*bytes*/,
        const float* __restrict__ act_base, int64_t act_nb2_f /*floats per token*/,
        const int* __restrict__ g_experts, const int* __restrict__ g_offsets,
        const int* __restrict__ g_tok, const int* __restrict__ g_slot,
        const int* __restrict__ n_union,
        float* __restrict__ dst_base, int64_t dst_nb2_f /*floats per token*/,
        int d_model, int d_ff, float limit) {
    const int u = blockIdx.y;
    if (u >= *n_union) return;
    const int m0 = g_offsets[u], m = g_offsets[u+1]-m0;
    if (m <= 0) return;
    const int e = g_experts[u];
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int f = blockIdx.x*WPB + warp;
    if (f >= d_ff) return;
    const int NB = d_model >> 5;
    typedef struct { unsigned char e; unsigned char qs[16]; } blk_t;
    const blk_t* ru = (const blk_t*)((const char*)up_base   + (int64_t)e*w_nb2 + (int64_t)f*w_nb1);
    const blk_t* rg = (const blk_t*)((const char*)gate_base + (int64_t)e*w_nb2 + (int64_t)f*w_nb1);
    __shared__ int s_tok[8][WPB];
    __shared__ int s_key[8][WPB];
    if (lane < m) { s_tok[lane][warp]=g_tok[m0+lane]; s_key[lane][warp]=g_slot[m0+lane]; }
    __syncwarp();
    float hg[8], hu[8];
    #pragma unroll
    for (int c=0;c<8;c++){hg[c]=0.f;hu[c]=0.f;}
    for (int ib = lane; ib < NB; ib += 32) {
        const blk_t bu = ru[ib], bg = rg[ib];
        const float du = pxa_mxfp4_d(bu.e);
        const float dg = pxa_mxfp4_d(bg.e);
        #pragma unroll
        for (int j=0;j<32;++j){
            const int b=j&15;
            const float wu=du*(float)kvalues_mxfp4[(j<16)?(bu.qs[b]&0xf):(bu.qs[b]>>4)];
            const float wg=dg*(float)kvalues_mxfp4[(j<16)?(bg.qs[b]&0xf):(bg.qs[b]>>4)];
            const int k = (ib<<5)+j;
            for (int c=0;c<m;++c){ const float xv = act_base[(int64_t)s_tok[c][warp]*act_nb2_f + k]; hg[c]+=wg*xv; hu[c]+=wu*xv; }
        }
    }
    #pragma unroll
    for (int c=0;c<8;c++){ if(c>=m)break;
        for(int o=16;o>0;o>>=1){ hg[c]+=__shfl_down_sync(0xffffffff,hg[c],o); hu[c]+=__shfl_down_sync(0xffffffff,hu[c],o);} }
    if (lane==0) for(int c=0;c<m;++c){
        float g=hg[c], uu=hu[c];
        g = g/(1.0f+expf(-g));
        g = fminf(g, limit);
        const float r = fmaxf(-limit, fminf(limit, uu)) * g;
        const int key = s_key[c][warp];
        dst_base[(size_t)(key >> 16) * dst_nb2_f + (size_t)(key & 0xffff) * d_ff + f] = r;
    }
}

// ---------------------------------------------------------------------------
// gk6-iqk (v3, BIT-EXACT): a grouped instantiation of the engine's OWN fused
// MMVQ (mmvq-templates.cuh k_fused_mul_mat_vec_q<MXFP4, ncols_y, nwarps=1>).
// The float accumulation is IDENTICAL to the per-token reference:
//   * SAME q8_1 activations: vy_base = the engine's src1_quantized; member j's
//     column = token g_tok[m0+j] at &y[tok*blocks_per_col_y + kby] — byte-identical
//     to the reference's local_src1 for that token (which sets nb12=0 => single col).
//   * SAME dot: vec_dot_mxfp4_q8_1 (the very function the engine dispatches for
//     GGML_TYPE_MXFP4), same weight-block index (row0)*blocks_per_row_x+kbx.
//   * SAME lane/block assignment: nwarps=1 (the MoE id path has ne2=n_ids>=2 =>
//     mul_mat_vec_q_cuda picks nwarps=1), rows_per_cuda_block=1 (ncols_y=1 ref);
//     kbx start = tid/(qi/vdr), stride = vdr*32/qi, kqs = vdr*(tid%(qi/vdr)).
//   * SAME reduction: warp_reduce_sum; SAME SILU epilogue (sanitized limit).
// The ONLY differences vs stock are ADDRESSING: blockIdx.y = union bin (not i2),
// expert e = g_experts[u], per-member y-column via g_tok, and the RT-1 scatter
// (g_slot key) instead of the contiguous dst[j*nrows_dst+row] write. None of
// those touch the accumulation => shadow diff must be EXACTLY 0.
// rows_per_cuda_block is fixed to 1 (mirrors the ncols_y=1 reference); MAXC only
// sizes the per-member accumulator arrays. One MAXC=4 instantiation covers
// MTP-n3 (Ny=4 => m<=4); the host falls back to the per-token path for m>4.
// ---------------------------------------------------------------------------
template<int MAXC>
__global__ void pxa_gk6_gateup_iqk(
        const void* __restrict__ up_base,      // src0_1->data (UP)
        const void* __restrict__ gate_base,    // src0_2->data (GATE)
        int64_t w_nb1 /*row_size bytes*/, int64_t w_nb2 /*expert stride bytes*/,
        const void* __restrict__ vy_base,      // src1_quantized (block_q8_1), all Ny cols
        int blocks_per_col_y,                  // block_q8_1 stride between token columns
        const int* __restrict__ g_experts, const int* __restrict__ g_offsets,
        const int* __restrict__ g_tok, const int* __restrict__ g_slot,
        const int* __restrict__ n_union,       // device
        float* __restrict__ dst_base, int64_t dst_nb2_f /*floats per token*/,
        int ncols_x /*d_model*/, int d_ff, float limit) {

    constexpr int qk  = ggml_cuda_type_traits<GGML_TYPE_MXFP4>::qk;   // 32
    constexpr int qi  = ggml_cuda_type_traits<GGML_TYPE_MXFP4>::qi;   // QI4_NL
    constexpr int vdr = VDR_MXFP4_Q8_1_MMVQ;                          // 2
    constexpr int nwarps = 1;                                         // MoE id path (ne2>=2)
    constexpr int rows_per_cuda_block = 1;                            // ncols_y=1 reference

    const int u = blockIdx.y;
    if (u >= *n_union) return;                    // oversubscribed grid tail
    const int m0 = g_offsets[u], m = g_offsets[u+1] - m0;
    if (m <= 0) return;
    const int e = g_experts[u];
    if (e < 0) return;                            // SER guard (parity w/ fused kernel i02<0)

    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;   // nwarps=1 => threadIdx.y==0
    const int row0 = rows_per_cuda_block*blockIdx.x;        // = blockIdx.x = output row f
    if (row0 >= d_ff) return;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_iter  = vdr*nwarps*WARP_SIZE / qi;

    const char* vup = (const char*)up_base   + (int64_t)e*w_nb2 + (int64_t)row0*w_nb1;
    const char* vgt = (const char*)gate_base + (int64_t)e*w_nb2 + (int64_t)row0*w_nb1;
    const block_q8_1* y = (const block_q8_1*)vy_base;

    // per-member accumulators (only [0,m) are used/reduced/stored)
    float tmp_u[MAXC]; float tmp_g[MAXC];
    #pragma unroll
    for (int j = 0; j < MAXC; ++j) { tmp_u[j] = 0.0f; tmp_g[j] = 0.0f; }

    for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx*(qk/QK8_1);            // qk==QK8_1 => kby==kbx
        const int kqs = vdr*(tid % (qi/vdr));
        #pragma unroll
        for (int j = 0; j < MAXC; ++j) {
            if (j >= m) break;
            const int t = g_tok[m0+j];
            const block_q8_1* yj = y + (int64_t)t*blocks_per_col_y + kby;
            tmp_u[j] += vec_dot_mxfp4_q8_1((const void*)vup, yj, kbx, kqs);
            tmp_g[j] += vec_dot_mxfp4_q8_1((const void*)vgt, yj, kbx, kqs);
        }
    }
    // nwarps==1: no cross-warp shared reduction; warp_reduce_sum broadcasts to all lanes
    #pragma unroll
    for (int j = 0; j < MAXC; ++j) {
        if (j >= m) break;
        tmp_u[j] = warp_reduce_sum(tmp_u[j]);
        tmp_g[j] = warp_reduce_sum(tmp_g[j]);
    }
    if (threadIdx.x == 0) {                         // rows_per_cuda_block==1 => lane 0 writes
        #pragma unroll
        for (int j = 0; j < MAXC; ++j) {
            if (j >= m) break;
            // ENGINE SILU epilogue VERBATIM (k_fused_mul_mat_vec_q): u=up, g=gate
            float uu = tmp_u[j], g = tmp_g[j];
            g = g/(1.0f + expf(-g));
            g = fminf(g, limit);
            const float r = fmaxf(-limit, fminf(limit, uu)) * g;
            const int key = g_slot[m0+j];           // (t<<16)|s
            dst_base[(size_t)(key >> 16) * dst_nb2_f + (size_t)(key & 0xffff) * d_ff + row0] = r;
        }
    }
}

// shadow diff: per-token dst vs grouped scratch; printf on violation (capped).
// exact=1 (gk6): require diff==0 exactly. exact=0 (half2/scalar): 2%-rel + isfinite.
__global__ void pxa_gk5_shadow_diff(const float* __restrict__ dst, const float* __restrict__ scratch,
                                    int64_t dst_nb2_f, int n_ids, int d_ff, int Ny, int* viol, int exact) {
    const int t = blockIdx.y;
    const int total = n_ids*d_ff;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < total; i += gridDim.x*blockDim.x) {
        const float a = dst    [(size_t)t*dst_nb2_f + i];
        const float b = scratch[(size_t)t*dst_nb2_f + i];
        const bool bad = exact ? (a != b)
                               : (!(fabsf(a-b) <= 0.02f*fmaxf(1.0f, fabsf(a))) || !isfinite(b));
        if (bad) {
            if (atomicAdd(viol, 1) < 8)
                printf("PXA_MOE_GROUPED_VERIFY MISMATCH t=%d i=%d pertoken=%.9g grouped=%.9g diff=%.3g\n",
                       t, i, a, b, fabsf(a-b));
        }
    }
    (void)Ny;
}

// ---------------------------------------------------------------------------
// Host entry. Returns true if the grouped path produced dst (skip the iy-loop).
// In shadow mode: grouped writes a private scratch, returns false (per-token
// path still runs), and pxa_moe_grouped_shadow_check() diffs afterwards.
// ---------------------------------------------------------------------------
struct pxa_moe_shadow_state {
    float* scratch = nullptr; size_t cap = 0;
    int*   viol    = nullptr;
    bool   pending = false;
    bool   exact   = false;   // gk6 => exact-equality diff; half2/scalar => tolerance
    int Ny = 0, n_ids = 0, d_ff = 0; int64_t dst_nb2_f = 0;
};
static inline pxa_moe_shadow_state& pxa_moe_shadow_st(int device) {
    static pxa_moe_shadow_state s[16];
    return s[device & 15];
}

static inline bool pxa_moe_grouped_gateup(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_1 /*UP*/, const ggml_tensor * src0_2 /*GATE*/,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        int Ny, int n_ids, float limit,
        const char * src1_q /*engine q8_1 activations*/, size_t src1_ddq_size /*bytes/col*/,
        cudaStream_t stream) {

    const int flags = pxa_moe_grouped_flags();
    if (!(flags & 1)) return false;
    // Capture safety (guide §2.2-7): shadow mode's cudaMalloc/printf and the pool
    // allocs are not capture-safe. Inside a CUDA-graph capture, fall back to the
    // per-token path. Costs nothing today (verify steps aren't captured).
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) return false;
    }
    // Kernel selection: default (bit0 only) = gk6-iqk bit-exact; bit3 = half2; bit2 = scalar.
    const bool use_half2  = (flags & 8) != 0;
    const bool use_scalar = !use_half2 && (flags & 4) != 0;
    const bool use_gk6    = !use_half2 && !use_scalar;
    // gk5 Tier-1 fix (2026-07-08): mmvq.cu:45 parity. The per-token path sanitizes limit
    // DOWNSTREAM of this hook; qwen35moe never sets op_params[1] (only STEP35 does), so the
    // raw limit here is 0.0f -> SILU epilogue clamps every routed-expert output to +-0 (garble,
    // accept 0.24). Sanitize exactly as mmvq.cu:45 does (covers both half2 + scalar kernels).
    limit = limit > 1e-6f ? limit : INFINITY;
    const int d_model = (int)src0_1->ne[0];
    const int d_ff    = (int)dst->ne[0];
    const int NN      = Ny * n_ids;
    // geometry / alignment gates (fall back silently to the per-token path)
    if (Ny < 2 || Ny > 8 || n_ids < 1 || n_ids > 8 || NN > 64) return false;
    if (d_model % 128 != 0) return false;                          // u32-aligned rows
    if (src0_1->nb[1] != src0_2->nb[1] || src0_1->nb[2] != src0_2->nb[2]) return false;
    if ((src0_1->nb[1] & 3) || ((uintptr_t)src0_1->data & 3) || ((uintptr_t)src0_2->data & 3)) return false;
    if (Ny*(size_t)n_ids >= (1u<<16)) return false;                // packed key range
    const int WROW = (d_model/32)*17/4;
    const size_t smem = (size_t)PXA_GK5_WPB*2*WROW*sizeof(unsigned);
    if (smem > 45*1024) return false;

    // plan (device, no host sync)
    ggml_cuda_pool_alloc<int> pl_exp(ctx.pool(), NN);
    ggml_cuda_pool_alloc<int> pl_off(ctx.pool(), NN+1);
    ggml_cuda_pool_alloc<int> pl_tok(ctx.pool(), NN);
    ggml_cuda_pool_alloc<int> pl_key(ctx.pool(), NN);
    ggml_cuda_pool_alloc<int> pl_nu (ctx.pool(), 1);
    pxa_build_plan<<<1, 32, 0, stream>>>((const int*)ids->data, (int64_t)(ids->nb[1]/sizeof(int)),
            Ny, n_ids, pl_exp.get(), pl_off.get(), pl_tok.get(), pl_key.get(), pl_nu.get());

    // output target: dst directly, or a private scratch in shadow mode
    float* out = (float*)dst->data;
    const int64_t dst_nb2_f = (int64_t)(dst->nb[2]/sizeof(float));
    auto & sh = pxa_moe_shadow_st(ctx.device);
    if (pxa_moe_grouped_shadow()) {
        const size_t need = (size_t)Ny*dst_nb2_f*sizeof(float);
        if (sh.cap < need) {
            if (sh.scratch) cudaFree(sh.scratch);
            cudaMalloc(&sh.scratch, need); sh.cap = need;
        }
        if (!sh.viol) { cudaMalloc(&sh.viol, sizeof(int)); }
        cudaMemsetAsync(sh.viol, 0, sizeof(int), stream);
        out = sh.scratch;
        sh.pending = true; sh.Ny = Ny; sh.n_ids = n_ids; sh.d_ff = d_ff; sh.dst_nb2_f = dst_nb2_f;
        sh.exact = use_gk6;   // gk6 must match the per-token path bit-for-bit
    }

    // ---- gk6-iqk BIT-EXACT path (default) ----------------------------------
    if (use_gk6) {
        // engine q8_1 activations: token-column stride in block_q8_1 units
        const int blocks_per_col_y = (int)(src1_ddq_size / sizeof(block_q8_1));
        // rows_per_cuda_block=1 (ncols_y=1 ref) => grid.x = d_ff; nwarps=1 => block=(32,1)
        dim3 grid6(d_ff, NN);
        dim3 blk6(WARP_SIZE, 1);
        auto launch6 = [&](auto maxc_tag) {
            constexpr int MAXC = decltype(maxc_tag)::value;
            pxa_gk6_gateup_iqk<MAXC><<<grid6, blk6, 0, stream>>>(
                    src0_1->data /*UP*/, src0_2->data /*GATE*/,
                    (int64_t)src0_1->nb[1] /*row_size*/, (int64_t)src0_1->nb[2] /*expert stride*/,
                    (const void*)src1_q, blocks_per_col_y,
                    pl_exp.get(), pl_off.get(), pl_tok.get(), pl_key.get(), pl_nu.get(),
                    out, dst_nb2_f, d_model, d_ff, limit);
        };
        if      (Ny <= 2) launch6(std::integral_constant<int,2>{});   // m<=Ny per bin
        else if (Ny <= 4) launch6(std::integral_constant<int,4>{});
        else              launch6(std::integral_constant<int,8>{});
        CUDA_CHECK(cudaGetLastError());
        return !pxa_moe_grouped_shadow();
    }

    dim3 blk(32, PXA_GK5_WPB);
    dim3 grid((d_ff + PXA_GK5_WPB - 1)/PXA_GK5_WPB, NN);           // oversubscribed on n_union

    if (use_scalar) {
        // scalar-F32 fallback path (reads src1 F32 directly)
        pxa_gk_gateup_scalar<PXA_GK5_WPB><<<grid, blk, 0, stream>>>(
                src0_1->data, src0_2->data, (int64_t)src0_1->nb[1], (int64_t)src0_1->nb[2],
                (const float*)src1->data, (int64_t)(src1->nb[2]/sizeof(float)),
                pl_exp.get(), pl_off.get(), pl_tok.get(), pl_key.get(), pl_nu.get(),
                out, dst_nb2_f, d_model, d_ff, limit);
        CUDA_CHECK(cudaGetLastError());
        return !pxa_moe_grouped_shadow();
    }

    // half2 path (bit3): convert the Ny activation rows to compact f16 once
    ggml_cuda_pool_alloc<__half> xh(ctx.pool(), (size_t)Ny*d_model);
    {
        dim3 cgrid((d_model + 255)/256, Ny);
        pxa_f32_to_half_rows<<<cgrid, 256, 0, stream>>>(
                (const float*)src1->data, (int64_t)(src1->nb[2]/sizeof(float)),
                xh.get(), d_model, Ny);
    }
    auto launch = [&](auto maxc_tag) {
        constexpr int MAXC = decltype(maxc_tag)::value;
        pxa_gk5_gateup_h2<MAXC><<<grid, blk, smem, stream>>>(
                src0_1->data /*UP*/, src0_2->data /*GATE*/,
                (int64_t)src0_1->nb[1], (int64_t)src0_1->nb[2],
                xh.get(), pl_exp.get(), pl_off.get(), pl_tok.get(), pl_key.get(), pl_nu.get(),
                out, dst_nb2_f, d_model, d_ff, limit);
    };
    if      (Ny <= 2) launch(std::integral_constant<int,2>{});
    else if (Ny <= 4) launch(std::integral_constant<int,4>{});
    else              launch(std::integral_constant<int,8>{});
    CUDA_CHECK(cudaGetLastError());
    return !pxa_moe_grouped_shadow();
}

// call AFTER the per-token iy-loop (no-op unless shadow mode armed)
static inline void pxa_moe_grouped_shadow_check(ggml_backend_cuda_context & ctx,
        ggml_tensor * dst, cudaStream_t stream) {
    auto & sh = pxa_moe_shadow_st(ctx.device);
    if (!sh.pending) return;
    sh.pending = false;
    dim3 grid(64, sh.Ny);
    pxa_gk5_shadow_diff<<<grid, 256, 0, stream>>>(
            (const float*)dst->data, sh.scratch, sh.dst_nb2_f, sh.n_ids, sh.d_ff, sh.Ny, sh.viol,
            sh.exact ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
}
