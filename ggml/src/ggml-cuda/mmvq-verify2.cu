// mmvq-verify2.cu -- PXA_MOE_VERIFY2, see mmvq-verify2.cuh for the contract.
#include "mmvq-verify2.cuh"
#include "mmvq-templates.cuh"

#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>

bool pxa_moe_verify2_enabled() {
    static const bool v = [](){
        const char * e = getenv("PXA_MOE_VERIFY2");
        if (!(e && atoi(e) != 0)) return false;
        // PXA_MMVQ_MOE_NWARPS=2/4 changes the per-token id-GEMV's reduction shape; this path keeps the
        // nwarps = 1 shape, so it stands aside rather than silently differ from the path it replaces
        const char * w = getenv("PXA_MMVQ_MOE_NWARPS");
        const int nw = w ? atoi(w) : 0;
        if (nw == 2 || nw == 4) {
            fprintf(stderr, "PXA_MOE_VERIFY2: off because PXA_MMVQ_MOE_NWARPS=%d is set\n", nw);
            return false;
        }
        return true;
    }();
    return v;
}

int pxa_moe_verify2_max_ny() {
    static const int v = [](){
        const char * e = getenv("PXA_MOE_VERIFY2_MAX_NY");
        int n = e ? atoi(e) : PXA_MV2_MAXC;
        if (n < 2) n = 2;
        if (n > PXA_MV2_MAXC) n = PXA_MV2_MAXC;
        return n;
    }();
    return v;
}

void pxa_moe_verify2_log(const char * site, ggml_type type, int ny, int n_ids, int nrows) {
    static const bool on = [](){ const char * e = getenv("PXA_MOE_VERIFY2_LOG"); return e && atoi(e) != 0; }();
    if (!on) return;
    static std::mutex m;
    static std::set<std::string> seen;
    std::string key = std::string(site) + ":" + ggml_type_name(type) + ":" + std::to_string(ny);
    std::lock_guard<std::mutex> lk(m);
    if (seen.insert(key).second) {
        fprintf(stderr, "PXA_MOE_VERIFY2 engaged site=%s type=%s ny=%d n_ids=%d nrows=%d\n",
                site, ggml_type_name(type), ny, n_ids, nrows);
    }
}

// One row, NC columns. Each column: its own q8_1 row y[j] and its own output d[j]. Same k-block
// walk, vec_dot and warp reduction as k_mul_mat_vec_q / k_fused_mul_mat_vec_q at nwarps = 1 and
// rows_per_cuda_block = 1 (the shape the id-GEMV launches at n_ids >= 2).
template <ggml_type type, int NC, bool FUSED>
static __device__ __forceinline__ void pxa_mv2_cols(
        const char * __restrict__ xu, const char * __restrict__ xg,
        const block_q8_1 * const (&y)[NC], float * const (&d)[NC], const float bias,
        const int ncols_x, const int row, const int unary_op, const float limit) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid = threadIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;

    float tu[NC];
    float tg[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) { tu[j] = 0.0f; tg[j] = 0.0f; }

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            tu[j] += vec_dot_q_cuda(xu, &y[j][kby], row*blocks_per_row_x + kbx, kqs);
            if constexpr (FUSED) {
                tg[j] += vec_dot_q_cuda(xg, &y[j][kby], row*blocks_per_row_x + kbx, kqs);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        tu[j] = warp_reduce_sum(tu[j]);
        if constexpr (FUSED) {
            tg[j] = warp_reduce_sum(tg[j]);
        }
        if (tid == 0) {
            if constexpr (FUSED) {
                const float u = tu[j];
                float g = tg[j];
                float r;
                switch (unary_op) {
                    case GGML_UNARY_OP_SILU: {
                        g = g/(1 + expf(-g));
                        g = min(g, limit);
                        r = max(-limit, min(limit, u))*g;
                    } break;
                    case GGML_UNARY_OP_RELU: r = fmaxf(g, 0.0f) * u; break;
                    default: { // GGML_UNARY_OP_GELU (the host admits only SILU / RELU / GELU)
                        constexpr float GELU_COEF_A    = 0.044715f;
                        constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
                        r = 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
                    } break;
                }
                d[j][row] = r;
            } else {
                d[j][row] = tu[j] + bias;
            }
        }
    }
}

template <ggml_type type, int NC, bool FUSED>
static __device__ __forceinline__ void pxa_mv2_group(const pxa_mv2_args & a, const int (&cols)[PXA_MV2_MAXC], const int c0,
        const char * xu, const char * xg, const float bias, const int row) {
    const block_q8_1 * y[NC];
    float * d[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const int s = cols[c0 + j];
        const int t = s / a.n_ids;
        const int k = s - t*a.n_ids;
        y[j] = (const block_q8_1 *)(a.vy + t*a.y_st + k*a.y_sk);
        d[j] = (float *)(a.dst + t*a.d_st + k*a.d_sk);
    }
    pxa_mv2_cols<type, NC, FUSED>(xu, xg, y, d, bias, a.ncols_x, row, a.unary_op, a.limit);
}

// NCMAX = widest column group compiled into this instance (= the verify width it is launched for), so the
// register budget of a width-2 launch is not set by the width-4 path
template <ggml_type type, bool FUSED, int NCMAX>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void pxa_mv2_kernel(const pxa_mv2_args a) {
    const int row  = blockIdx.x;
    const int s    = blockIdx.y;               // flattened (token, slot)
    const int ns   = a.ny * a.n_ids;           // <= WARP_SIZE (host-checked)
    const int lane = threadIdx.x;
    // every lane reads one (token, slot) id at once: one load latency, then the grouping is warp votes
    int my = -1;
    if (lane < ns) {
        const int t = lane / a.n_ids;
        const int k = lane - t*a.n_ids;
        my = *(const int *)(a.ids + t*a.ids_nb1 + k*a.ids_nb0);
    }
    const int e = __shfl_sync(0xffffffff, my, s);
    if (e < 0) {
        return;
    }
    unsigned mask = __ballot_sync(0xffffffff, lane < ns && my == e);
    // the first occurrence of e in (token, slot) order owns every occurrence
    if (__ffs(mask) - 1 != s) {
        return;
    }
    const char * xu = (const char *) a.vx_u + e*a.nb02;
    const char * xg = FUSED ? (const char *) a.vx_g + e*a.nb02 : nullptr;
    float bias = 0.0f;
    if (!FUSED && a.bias) {
        bias = ((const float *)(a.bias + e*a.bias_nb1))[row];
    }
    while (mask) {
        int cols[PXA_MV2_MAXC];
        int nc = 0;
        while (mask && nc < NCMAX) {
            const int b = __ffs(mask) - 1;
            cols[nc++] = b;
            mask &= mask - 1;
        }
        if (nc == 1) {
            pxa_mv2_group<type, 1, FUSED>(a, cols, 0, xu, xg, bias, row);
        } else if (NCMAX >= 2 && nc == 2) {
            pxa_mv2_group<type, NCMAX >= 2 ? 2 : 1, FUSED>(a, cols, 0, xu, xg, bias, row);
        } else if (NCMAX >= 3 && nc == 3) {
            pxa_mv2_group<type, NCMAX >= 3 ? 3 : 1, FUSED>(a, cols, 0, xu, xg, bias, row);
        } else if (NCMAX >= 4) {
            pxa_mv2_group<type, NCMAX >= 4 ? 4 : 1, FUSED>(a, cols, 0, xu, xg, bias, row);
        }
    }
}

// ---- PXA_MOE_VERIFY2_TILE (default OFF): W warps per block x R rows per warp ----------------------------
// The 1-warp/1-row blocks above cap a V100 SM at 32 resident warps (the 32-blocks/SM limit) and give each warp
// one short dependent load chain (the down projection's K=704 is 22 q4_0 blocks = 2 k-iterations). Here each
// warp computes R rows interleaved (R independent load streams) and W warps share a block. Per (row, column)
// every lane walks the SAME k-blocks in the SAME order into its own accumulator, then the same warp_reduce_sum,
// so each output is bit-identical to pxa_mv2_cols (and therefore to the per-token loop it replaces).
template <ggml_type type, int NC, bool FUSED, int R>
static __device__ __forceinline__ void pxa_mv2t_cols(
        const char * __restrict__ xu, const char * __restrict__ xg,
        const block_q8_1 * const (&y)[NC], float * const (&d)[NC], const float * __restrict__ biasp,
        const int ncols_x, const int row0, const int unary_op, const float limit) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid = threadIdx.x % WARP_SIZE;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;

    float tu[R][NC];
    float tg[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) { tu[r][j] = 0.0f; tg[r][j] = 0.0f; }
    }

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int ib = (row0 + r)*blocks_per_row_x + kbx;
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                tu[r][j] += vec_dot_q_cuda(xu, &y[j][kby], ib, kqs);
                if constexpr (FUSED) {
                    tg[r][j] += vec_dot_q_cuda(xg, &y[j][kby], ib, kqs);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            tu[r][j] = warp_reduce_sum(tu[r][j]);
            if constexpr (FUSED) {
                tg[r][j] = warp_reduce_sum(tg[r][j]);
            }
            if (tid == 0) {
                if constexpr (FUSED) {
                    const float u = tu[r][j];
                    float g = tg[r][j];
                    float res;
                    switch (unary_op) {
                        case GGML_UNARY_OP_SILU: {
                            g = g/(1 + expf(-g));
                            g = min(g, limit);
                            res = max(-limit, min(limit, u))*g;
                        } break;
                        case GGML_UNARY_OP_RELU: res = fmaxf(g, 0.0f) * u; break;
                        default: {
                            constexpr float GELU_COEF_A    = 0.044715f;
                            constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
                            res = 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
                        } break;
                    }
                    d[j][row] = res;
                } else {
                    d[j][row] = tu[r][j] + (biasp ? biasp[row] : 0.0f);
                }
            }
        }
    }
}

template <ggml_type type, int NC, bool FUSED, int R>
static __device__ __forceinline__ void pxa_mv2t_group(const pxa_mv2_args & a, const int (&cols)[PXA_MV2_MAXC],
        const char * xu, const char * xg, const float * biasp, const int row0) {
    const block_q8_1 * y[NC];
    float * d[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const int s = cols[j];
        const int t = s / a.n_ids;
        const int k = s - t*a.n_ids;
        y[j] = (const block_q8_1 *)(a.vy + t*a.y_st + k*a.y_sk);
        d[j] = (float *)(a.dst + t*a.d_st + k*a.d_sk);
    }
    pxa_mv2t_cols<type, NC, FUSED, R>(xu, xg, y, d, biasp, a.ncols_x, row0, a.unary_op, a.limit);
}

template <ggml_type type, bool FUSED, int NCMAX, int R>
__launch_bounds__(8*WARP_SIZE)
static __global__ void pxa_mv2t_kernel(const pxa_mv2_args a) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x*(blockDim.x/WARP_SIZE) + warp)*R;   // host guarantees row0 + R <= nrows
    const int s    = blockIdx.y;
    const int ns   = a.ny * a.n_ids;
    const int lane = threadIdx.x % WARP_SIZE;
    int my = -1;
    if (lane < ns) {
        const int t = lane / a.n_ids;
        const int k = lane - t*a.n_ids;
        my = *(const int *)(a.ids + t*a.ids_nb1 + k*a.ids_nb0);
    }
    const int e = __shfl_sync(0xffffffff, my, s);
    if (e < 0) {
        return;
    }
    unsigned mask = __ballot_sync(0xffffffff, lane < ns && my == e);
    if (__ffs(mask) - 1 != s) {
        return;
    }
    const char * xu = (const char *) a.vx_u + e*a.nb02;
    const char * xg = FUSED ? (const char *) a.vx_g + e*a.nb02 : nullptr;
    const float * biasp = (!FUSED && a.bias) ? (const float *)(a.bias + e*a.bias_nb1) : nullptr;
    while (mask) {
        int cols[PXA_MV2_MAXC];
        int nc = 0;
        while (mask && nc < NCMAX) {
            const int b = __ffs(mask) - 1;
            cols[nc++] = b;
            mask &= mask - 1;
        }
        if (nc == 1) {
            pxa_mv2t_group<type, 1, FUSED, R>(a, cols, xu, xg, biasp, row0);
        } else if (NCMAX >= 2 && nc == 2) {
            pxa_mv2t_group<type, NCMAX >= 2 ? 2 : 1, FUSED, R>(a, cols, xu, xg, biasp, row0);
        } else if (NCMAX >= 3 && nc == 3) {
            pxa_mv2t_group<type, NCMAX >= 3 ? 3 : 1, FUSED, R>(a, cols, xu, xg, biasp, row0);
        } else if (NCMAX >= 4) {
            pxa_mv2t_group<type, NCMAX >= 4 ? 4 : 1, FUSED, R>(a, cols, xu, xg, biasp, row0);
        }
    }
}

// "WxR" (W warps per block in {1,2,4,8}, R rows per warp in {1,2,4}); unset / "0" / malformed = off
struct pxa_mv2_tile { int w = 0, r = 0; };
static pxa_mv2_tile pxa_mv2_parse_tile(const char * name, const pxa_mv2_tile & dflt) {
    const char * e = getenv(name);
    if (!e) return dflt;
    pxa_mv2_tile t;
    int w = 0, r = 0;
    if (sscanf(e, "%dx%d", &w, &r) == 2 && (w == 1 || w == 2 || w == 4 || w == 8) && (r == 1 || r == 2 || r == 4)) {
        t.w = w; t.r = r;
    }
    return t;
}
static pxa_mv2_tile pxa_mv2_tile_for(bool fused) {
    static const pxa_mv2_tile all  = pxa_mv2_parse_tile("PXA_MOE_VERIFY2_TILE", pxa_mv2_tile{});
    static const pxa_mv2_tile up   = pxa_mv2_parse_tile("PXA_MOE_VERIFY2_TILE_UP", all);
    static const pxa_mv2_tile down = pxa_mv2_parse_tile("PXA_MOE_VERIFY2_TILE_DOWN", all);
    static const bool logged = [](){
        if (up.w || down.w) fprintf(stderr, "PXA_MOE_VERIFY2_TILE: up/gate %dx%d, down %dx%d (WxR, 0x0 = untiled)\n",
                                    up.w, up.r, down.w, down.r);
        return true;
    }();
    (void) logged;
    return fused ? up : down;
}

template <ggml_type type, bool FUSED, int NCMAX>
static void pxa_mv2t_launch_r(const pxa_mv2_args & a, const pxa_mv2_tile & t, cudaStream_t stream) {
    const dim3 grid(a.nrows_x/(t.w*t.r), a.ny*a.n_ids, 1);
    const dim3 block(WARP_SIZE*t.w, 1, 1);
    switch (t.r) {
        case 1:  pxa_mv2t_kernel<type, FUSED, NCMAX, 1><<<grid, block, 0, stream>>>(a); break;
        case 2:  pxa_mv2t_kernel<type, FUSED, NCMAX, 2><<<grid, block, 0, stream>>>(a); break;
        default: pxa_mv2t_kernel<type, FUSED, NCMAX, 4><<<grid, block, 0, stream>>>(a); break;
    }
}

template <ggml_type type>
static bool pxa_mv2t_launch_t(const pxa_mv2_args & a, cudaStream_t stream) {
    const bool fused = a.vx_g != nullptr;
    const pxa_mv2_tile t = pxa_mv2_tile_for(fused);
    if (t.w == 0 || a.nrows_x % (t.w*t.r) != 0) return false;
#define PXA_MV2T_K(N) do { if (fused) pxa_mv2t_launch_r<type, true, N>(a, t, stream); \
                          else       pxa_mv2t_launch_r<type, false, N>(a, t, stream); } while (0)
    switch (a.ny) {
        case 1:  PXA_MV2T_K(1); break;
        case 2:  PXA_MV2T_K(2); break;
        case 3:  PXA_MV2T_K(3); break;
        default: PXA_MV2T_K(4); break;
    }
#undef PXA_MV2T_K
    return true;
}

template <ggml_type type>
static bool pxa_mv2_launch_t(const pxa_mv2_args & a, cudaStream_t stream) {
    if (pxa_mv2t_launch_t<type>(a, stream)) {
        return true;
    }
    if (a.ny == 1) {
        return false;
    }
    const dim3 grid(a.nrows_x, a.ny*a.n_ids, 1);
    const dim3 block(WARP_SIZE, 1, 1);
#define PXA_MV2_K(N) do { if (a.vx_g) pxa_mv2_kernel<type, true, N><<<grid, block, 0, stream>>>(a); \
                         else        pxa_mv2_kernel<type, false, N><<<grid, block, 0, stream>>>(a); } while (0)
    switch (a.ny) {
        case 2:  PXA_MV2_K(2); break;
        case 3:  PXA_MV2_K(3); break;
        default: PXA_MV2_K(4); break;
    }
#undef PXA_MV2_K
    return true;
}

bool pxa_moe_verify2_type_ok(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

bool pxa_moe_verify2_tile_w1() {
    static const bool v = [](){
        const char * e = getenv("PXA_MOE_VERIFY2_TILE_W1");
        const bool on = e && atoi(e) != 0 && pxa_moe_verify2_enabled() && pxa_mv2_tile_for(true).w && pxa_mv2_tile_for(false).w;
        if (on) fprintf(stderr, "PXA_MOE_VERIFY2_TILE_W1: one-token decode MoE also served by the tiled kernel\n");
        return on;
    }();
    return v;
}

bool pxa_moe_verify2_launch(ggml_type type, const pxa_mv2_args & a, cudaStream_t stream) {
    if (a.ny == 1 && !pxa_moe_verify2_tile_w1()) return false;
    if (a.ny < 1 || a.ny > PXA_MV2_MAXC || a.n_ids < 1 || a.nrows_x < 1) return false;
    if (a.ny*a.n_ids > WARP_SIZE) return false;   // one id per lane in the grouping vote
    if (a.vx_g && a.bias) return false;     // fused + bias is SWIGLU_OAI: not served here
    switch (type) {
        case GGML_TYPE_Q4_0: return pxa_mv2_launch_t<GGML_TYPE_Q4_0>(a, stream);
        case GGML_TYPE_Q4_1: return pxa_mv2_launch_t<GGML_TYPE_Q4_1>(a, stream);
        case GGML_TYPE_Q5_0: return pxa_mv2_launch_t<GGML_TYPE_Q5_0>(a, stream);
        case GGML_TYPE_Q5_1: return pxa_mv2_launch_t<GGML_TYPE_Q5_1>(a, stream);
        case GGML_TYPE_Q6_0: return pxa_mv2_launch_t<GGML_TYPE_Q6_0>(a, stream);
        case GGML_TYPE_Q8_0: return pxa_mv2_launch_t<GGML_TYPE_Q8_0>(a, stream);
        case GGML_TYPE_Q2_K: return pxa_mv2_launch_t<GGML_TYPE_Q2_K>(a, stream);
        case GGML_TYPE_Q3_K: return pxa_mv2_launch_t<GGML_TYPE_Q3_K>(a, stream);
        case GGML_TYPE_Q4_K: return pxa_mv2_launch_t<GGML_TYPE_Q4_K>(a, stream);
        case GGML_TYPE_Q5_K: return pxa_mv2_launch_t<GGML_TYPE_Q5_K>(a, stream);
        case GGML_TYPE_Q6_K: return pxa_mv2_launch_t<GGML_TYPE_Q6_K>(a, stream);
        case GGML_TYPE_IQ4_NL: return pxa_mv2_launch_t<GGML_TYPE_IQ4_NL>(a, stream);
        case GGML_TYPE_IQ4_XS: return pxa_mv2_launch_t<GGML_TYPE_IQ4_XS>(a, stream);
        case GGML_TYPE_MXFP4: return pxa_mv2_launch_t<GGML_TYPE_MXFP4>(a, stream);
        default: return false;
    }
}
