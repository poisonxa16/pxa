// pxa / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// pxq-mmvq-pft.cuh -- PXA_PXQN_VWPF (2026-10-16): the K1 pipelined single-column PXQ MMVQ body
// (k_pxq_mmvq1_pf) at the speculative verify widths, 2..8 activation columns, on sm_70.
//
// WHY. The verify-width launches (mul_mat_vec_q / mul_mat_vec_q_mb, HOIST body) are latency-bound, not issue- or
// DRAM-bound: ncu on down 5120x17408 shows 657 GB/s at 2 columns, 431 at 5 and 269 at 8 with the
// issue slots 38-44% busy, 0.5-0.8 eligible warps per scheduler and a third of every stall on an L1TEX scoreboard.
// Their loop loads a k-block's code + scale words and decodes them in the same iteration, so every iteration waits
// out a DRAM round trip. The one-column launch stopped doing that on 2026-10-14 (K1: issue the next k-block's words
// before this k-block's math) and runs at 731-752 GB/s. This is the same pipeline with NY columns per weight fragment.
//
// IDENTITY. Every output is bit-identical to the one-column launch of the same column (and so to plain decode):
// same 128-thread lane mapping (kqs = 2*(tid % 2), kbx = tid/2 + 64 n), same per-thread k-block order, the same
// per-(row, column, k-block) arithmetic (pxq_mmvq_frag_from_words + pxq_mmvq_dot_frag == vec_dot_pxq_q8_1_qw, the
// split the HOIST body has shipped since 2026-09-08), the same cross-warp fold ((w0 + w1) + w2) + w3 and the same
// xor butterfly. The fold is spread over the 4 warps (pair p = j*ROWS + i finished by warp p % 4, lane 0 stores), the
// EPI epilogue, which changes who adds, not what is added in which order. ROWS (rows per block) does not
// enter any row's arithmetic. tests/test-pxq-mmvq-pft.cu checks every output by bits against the one-column launch.
#pragma once

// pxq_mmvq_wfrag::load() minus its load_words: the same fragment (same bytes, same floats) from code words the caller
// already holds -- here loaded one k-block ahead. Kept out of pxq-mmvq.cuh, which every CUDA TU includes via common.cuh.
template <class POL, int VDR, int ROWS>
static __device__ __forceinline__ void pxq_mmvq_frag_from_words(
        pxq_mmvq_wfrag<POL, VDR, ROWS> & w, const uint32_t * qw, const int i,
        const float * __restrict__ sub, const pxq_mmvq_scales<POL, ROWS> & sc, const int iqs) {
#pragma unroll
    for (int l = 0; l < VDR; ++l) {
        const int m = iqs + l;                             // code group == 8-element k group
        w.v  [l] = POL::template group<VDR>(qw, l, iqs);
        w.eff[l] = sub[POL::snib(sc.byte(POL::sidx(i, m)), m)];
    }
}

// pxq_mmvq_dot_frag on activation words the caller already holds (PY: the column's q8_1 ints and scale are loaded one
// k-block ahead, like the code words): q[0..3] = qs ints 2*iqs .. 2*iqs + 3, d = __low2float(ds). Same dp4a chain,
// same float conversion, same fma order, same (anch * d) * sum -- the same float as the pointer form.
template <class POL, int ROWS, bool X>
static __device__ __forceinline__ float pxq_mmvq_dot_frag_r(
        const pxq_mmvq_wfrag<POL, 2, ROWS> & w, const float anch, const int * q, const float d) {
    float sum = 0.0f;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        int s = ggml_cuda_dp4a(w.v[l].x, q[2*l + 0], 0);
        s     = ggml_cuda_dp4a(w.v[l].y, q[2*l + 1], s);
        if constexpr (X) {
            sum  += w.eff[l] * pxa_i2f_exact(s);
        } else {
            sum  += w.eff[l] * (float) s;
        }
    }
    return anch * d * sum;
}

template <ggml_type type, int NY, int ROWS, bool X, bool PY>
static __device__ __forceinline__ void k_pxq_mmvq_pft(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_y, const int nrows_dst) {
    static_assert(type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQN4, "PXA_PXQN_VWPF: the p6 slab tiers only");
    static_assert(NY >= 1 && NY <= 8 && (ROWS == 2 || ROWS == 4 || ROWS == 8), "PXA_PXQN_VWPF: 1..8 columns, 2/4/8 rows");
    using POL = pxq_mmvq_pol<type>;
    constexpr int NW = 4, qk = 32, qi = 4, vdr = 2, TPR = qi/vdr, BPI = vdr*NW*WARP_SIZE/qi;
    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int row0 = ROWS*(int) blockIdx.x;
    const int nkb  = ncols_x / qk;
    const int bpc  = nrows_y / QK8_1;
    float tmp[NY][ROWS];
#pragma unroll
    for (int j = 0; j < NY; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) tmp[j][i] = 0.0f;
    }
    const block_q8_1 * y = (const block_q8_1 *) vy;
    __shared__ float s_sub[16];
    const float * sub = pxq_mmvq_stage_sub<type>(s_sub, tid);
    const pxq_mmvq_rowbase<type, ROWS> rb(vx, row0, nkb);
    const int kqs = vdr * (tid % TPR);
    int kbx = tid / TPR;
    uint32_t qw[ROWS][2];
    pxq_mmvq_scales<POL, ROWS> sc;
    constexpr int NYP = PY ? NY : 1;
    int   yq[NYP][4];
    float yd[NYP];
    if (kbx < nkb) {
        const uint8_t * slab = rb.slab(kbx);
        sc.load(slab, rb.r0);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) POL::template load_words<2>(slab, rb.r0 + i, kqs, qw[i]);
        if constexpr (PY) {
#pragma unroll
            for (int j = 0; j < NY; ++j) {
                const block_q8_1 * b = &y[j*bpc + kbx];
                const int * q8 = (const int *) b->qs;
#pragma unroll
                for (int t = 0; t < 4; ++t) yq[j][t] = q8[2*kqs + t];
                yd[j] = __low2float(b->ds);
            }
        }
    }
    for (; kbx < nkb; kbx += BPI) {
        uint32_t qn[ROWS][2];
        pxq_mmvq_scales<POL, ROWS> scn;
        int   yqn[NYP][4];
        float ydn[NYP];
        const int kn = kbx + BPI;
        if (kn < nkb) {
            const uint8_t * slab = rb.slab(kn);
            scn.load(slab, rb.r0);
#pragma unroll
            for (int i = 0; i < ROWS; ++i) POL::template load_words<2>(slab, rb.r0 + i, kqs, qn[i]);
            if constexpr (PY) {
#pragma unroll
                for (int j = 0; j < NY; ++j) {
                    const block_q8_1 * b = &y[j*bpc + kn];
                    const int * q8 = (const int *) b->qs;
#pragma unroll
                    for (int t = 0; t < 4; ++t) yqn[j][t] = q8[2*kqs + t];
                    ydn[j] = __low2float(b->ds);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            pxq_mmvq_frag<type, 2, ROWS> w;
            pxq_mmvq_frag_from_words(w, qw[i], i, sub, sc, kqs);
#pragma unroll
            for (int j = 0; j < NY; ++j) {
                if constexpr (PY) {
                    tmp[j][i] += pxq_mmvq_dot_frag_r<POL, ROWS, X>(w, rb.anch[i], yq[j], yd[j]);
                } else if constexpr (X) {   // pxa_i2f_exact: the same float as (float) s, see mul_mat_vec_q_vw
                    tmp[j][i] += pxq_mmvq_dot_frag_vw<POL, 2, ROWS, true>(w, rb.anch[i], &y[j*bpc + kbx], kqs);
                } else {
                    tmp[j][i] += pxq_mmvq_dot_f<type, 2, ROWS>(w, rb.anch[i], &y[j*bpc + kbx], kqs);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < ROWS; ++i) { qw[i][0] = qn[i][0]; qw[i][1] = qn[i][1]; }
#pragma unroll
        for (int w = 0; w < (int) (sizeof(sc.w)/sizeof(sc.w[0])); ++w) sc.w[w] = scn.w[w];
        if constexpr (PY) {
#pragma unroll
            for (int j = 0; j < NY; ++j) {
#pragma unroll
                for (int t = 0; t < 4; ++t) yq[j][t] = yqn[j][t];
                yd[j] = ydn[j];
            }
        }
    }
    constexpr int NP = NY*ROWS;
    __shared__ float part[NW][NP][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < NY; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) part[threadIdx.y][j*ROWS + i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
#pragma unroll
    for (int p = threadIdx.y; p < NP; p += NW) {
        float v = part[0][p][threadIdx.x];
#pragma unroll
        for (int l = 1; l < NW; ++l) v += part[l][p][threadIdx.x];
        v = warp_reduce_sum(v);
        const int j = p / ROWS, i = p % ROWS;
        if (threadIdx.x == 0 && row0 + i < nrows_dst) {
            dst[j*nrows_dst + row0 + i] = v + 0.0f;   // + the no-bias local_bias of the reference store
        }
    }
}

template <ggml_type type, int NY, int ROWS, int MINB, bool X, bool PY = false>
__launch_bounds__(4*WARP_SIZE, MINB)
static __global__ void mul_mat_vec_q_pft(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                         const int ncols_x, const int nrows_y, const int nrows_dst) {
    k_pxq_mmvq_pft<type, NY, ROWS, X, PY>(vx, vy, dst, ncols_x, nrows_y, nrows_dst);
}

// Host side (the PXQN4 instance TU only). PXA_PXQN_VWPF: unset = on at sm_70, 0 = off.
static inline int pxa_pxqn_vwpf_env() {
    static const int v = [](){
        const char * e = getenv("PXA_PXQN_VWPF");
        return e ? (atoi(e) != 0 ? 1 : 0) : -1;
    }();
    return v;
}
// Taken: sm_70, 3/6/7 columns, one matrix (ne2 == 1), no ids / bias / fused gate, VDR 2, not PXA_SPEC_FAST_VERIFY, rows in
// whole 64-row panels. Everything else keeps its launch.
template <ggml_type type>
static bool pxa_pxqn_vwpf_launch_t(const mmvq_args & args, cudaStream_t stream) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc != PXA_CC_VOLTA || pxa_pxqn_vwpf_env() == 0) return false;
    const int ny = args.ncols_y;
    if (ny != 3 && ny != 6 && ny != 7) return false;   // the widths where it measured faster (table below)
    if (args.ne2 != 1 || args.ids_data || args.bias_u || (args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT)) return false;
    if (pxa_pxq_mmvq_vdr() != 2 || pxa_spec_fast_verify(cc) || args.nrows_x % 64 != 0 || args.nrows_dst < args.nrows_x) return false;
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) {
        fprintf(stderr, "PXA_PXQN_VWPF: PXQN4 verify GEMVs at 3/6/7 columns on the pipelined body (bit-identical; =0 restores the old launch)\n");
    }
    const dim3 blk(WARP_SIZE, 4, 1);
#define PXA_VWPF_L(NY, R, MB, X, PY) mul_mat_vec_q_pft<type, NY, R, MB, X, PY><<<dim3(args.nrows_x/R, 1, 1), blk, 0, stream>>>( \
        args.vx_u, args.vy, args.dst, args.ncols_x, args.nrows_y, args.nrows_dst)
    // PXA_VWPF_TABLE (tests/test-pxq-mmvq-pft.cu, V100 card 2, 2026-10-16 16:2x, 1280/1280 bitwise vs the shipped launch and
    // the 1-column K1 launch; ms per 27B verify step = 5 GEMV shapes x per-step counts, vs the engine's launch today):
    // 3 cols R4/MB4/PY 12.02 -> 11.76 (-2.2%), 6 cols R4/MB4/X 18.04 (verify_wide R8/MB3) -> 16.83 (-6.7%),
    // 7 cols R4/MB4 20.28 (verify_wide R4/MB4) -> 19.17 (-5.5%). 2/5 cols (vs the shipped launch) and 4/8 cols (vs verify_wide)
    // measured slower or tied for every tile, so they keep their launch.
    switch (ny) {
        case 3:  PXA_VWPF_L(3, 4, 4, false, true ); break;
        case 6:  PXA_VWPF_L(6, 4, 4, true,  false); break;
        default: PXA_VWPF_L(7, 4, 4, false, false); break;
    }
#undef PXA_VWPF_L
    return true;
}
