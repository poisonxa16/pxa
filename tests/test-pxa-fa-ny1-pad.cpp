// test-pxa-fa-ny1-pad.cpp -- PXA_FA_VEC_NY1_PAD identity (2026-10-20): with PXA_FA_VEC_NY1_SPLIT=1
// every column of a 2..8-column speculative verify batch must be BITWISE the one-token decode of that column, also when the
// batch straddles a 256-row KV pad (verify n_kv = pad(last column), an earlier column's decode ran at 256 keys fewer).
// The KV layout is the llama cache's ([D x n_head_kv] rows per cell, viewed as [D, n_kv, n_head_kv]), the mask is the causal
// single-sequence mask padded to GGML_KQ_MASK_PAD rows. Needs a CUDA device (ctest -L cuda; PXA_TEST_DEVICE=N picks it);
// the identity target is the sm_60 f32 vec route (P100). A CPU build SKIPs.
// Exit 0 = every column of every case bitwise equal. Run with PXA_FA_VEC_NY1_PAD=0 to see the old split differ.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "pxa-test-portable.h"   // setenv / unsetenv on native Windows too

static uint32_t rng_state = 12345u;
static float frand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

struct shape { int D, H, Hkv; };

int main() {
#ifndef GGML_USE_CUDA
    printf("test-pxa-fa-ny1-pad: SKIP (no CUDA backend in this build)\n");
    return 0;
#else
    setenv("PXA_FA_VEC_NY1_SPLIT", "1", 0);
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("test-pxa-fa-ny1-pad: SKIP (no CUDA device)\n");
        return 0;
    }
    const char * dv = getenv("PXA_TEST_DEVICE");
    ggml_backend_t be = ggml_backend_cuda_init(dv ? atoi(dv) : 0, nullptr);
    if (!be) {
        printf("test-pxa-fa-ny1-pad: SKIP (CUDA init failed)\n");
        return 0;
    }
    // The identity holds where the verify widths 2..8 and the one-token decode run the SAME kernel: the sm_60/61 vec route.
    // From sm_70 up the verify batch takes another kernel by design (V100: PXA_FA_D256_VOLTA_TILE at head 256, WMMA/MMA
    // elsewhere), so even a column that straddles nothing differs from its decode there: nothing to assert, exit 77 (SKIP).
    const int cc = ggml_backend_cuda_get_device_cc(dv ? atoi(dv) : 0);
    if (cc >= 700) {
        printf("test-pxa-fa-ny1-pad: SKIP (device cc %d: the verify batch does not run the decode's vec kernel; target is sm_60/61)\n", cc);
        ggml_backend_free(be);
        return 77;
    }

    // The split differs only at a few n_kv for a given (occupancy, column tiles): sweep head layouts (column tiles = heads / GQA
    // pack) and every boundary m*256 up to 4096 keys; the straddling case is the one that can differ, m*256+17 is the control.
    const shape shapes[] = { {256, 24, 4}, {256, 24, 24}, {256, 48, 48}, {256, 16, 2}, {128, 32, 8}, {128, 64, 8},
                             {128, 48, 48}, {128, 64, 64}, {64, 64, 64} };
    const int widths[] = { 2, 4, 8 };
    int pads[15];
    for (int i = 0; i < 15; ++i) pads[i] = i + 1;   // straddle the m*256 boundary
    int ncases = 0, ncols_bad = 0, ncols_tot = 0, nstraddle_bad = 0;

    for (const shape & sh : shapes) {
        for (int w : widths) {
            for (int m : pads) {
                for (int straddle = (m == 1 ? 0 : 1); straddle < 2; ++straddle) {
                    // columns at positions p..p+w-1; straddle: half before the boundary m*256, half after
                    const int p = straddle ? m*256 - w/2 : m*256 + 17;
                    const int n_kv_v = GGML_PAD(p + w, 256);
                    const int n_cells = n_kv_v;
                    const float scale = 1.0f / sqrtf((float) sh.D);

                    ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, true };
                    ggml_context * ctx = ggml_init(ip);
                    ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, sh.D*sh.Hkv, n_cells);
                    ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, sh.D*sh.Hkv, n_cells);
                    ggml_tensor * qb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.D*sh.H, w);
                    const int rowv = GGML_PAD(w, GGML_KQ_MASK_PAD);
                    ggml_tensor * mv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_v, rowv);

                    auto kview = [&](ggml_tensor * c, int n_kv) {
                        return ggml_view_3d(ctx, c, sh.D, n_kv, sh.Hkv, c->nb[1], ggml_row_size(c->type, sh.D), 0);
                    };
                    // verify graph
                    ggml_tensor * qv = ggml_view_3d(ctx, qb, sh.D, w, sh.H, qb->nb[1], ggml_row_size(qb->type, sh.D), 0);
                    ggml_tensor * ov = ggml_flash_attn_ext(ctx, qv, kview(kc, n_kv_v), kview(vc, n_kv_v), mv, scale, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(ov, GGML_PREC_F32);
                    ggml_cgraph * gv = ggml_new_graph(ctx);
                    ggml_build_forward_expand(gv, ov);
                    // decode graphs
                    std::vector<ggml_tensor *> md(w), od(w);
                    std::vector<ggml_cgraph *> gd(w);
                    std::vector<int> nkd(w);
                    for (int c = 0; c < w; ++c) {
                        nkd[c] = GGML_PAD(p + c + 1, 256);
                        md[c] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkd[c], GGML_KQ_MASK_PAD);
                        ggml_tensor * qd = ggml_view_3d(ctx, qb, sh.D, 1, sh.H, qb->nb[1], ggml_row_size(qb->type, sh.D), c*qb->nb[1]);
                        od[c] = ggml_flash_attn_ext(ctx, qd, kview(kc, nkd[c]), kview(vc, nkd[c]), md[c], scale, 0.0f, 0.0f);
                        ggml_flash_attn_ext_set_prec(od[c], GGML_PREC_F32);
                        gd[c] = ggml_new_graph(ctx);
                        ggml_build_forward_expand(gd[c], od[c]);
                    }
                    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

                    std::vector<ggml_fp16_t> h((size_t) sh.D*sh.Hkv*n_cells);
                    for (auto & x : h) x = ggml_fp32_to_fp16(frand());
                    ggml_backend_tensor_set(kc, h.data(), 0, ggml_nbytes(kc));
                    for (auto & x : h) x = ggml_fp32_to_fp16(frand());
                    ggml_backend_tensor_set(vc, h.data(), 0, ggml_nbytes(vc));
                    std::vector<float> hq((size_t) sh.D*sh.H*w);
                    for (auto & x : hq) x = 2.0f*frand();
                    ggml_backend_tensor_set(qb, hq.data(), 0, ggml_nbytes(qb));
                    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
                    std::vector<ggml_fp16_t> hm((size_t) n_kv_v*rowv, ninf);
                    for (int c = 0; c < w; ++c) for (int i = 0; i <= p + c; ++i) hm[(size_t) c*n_kv_v + i] = z;
                    ggml_backend_tensor_set(mv, hm.data(), 0, ggml_nbytes(mv));
                    for (int c = 0; c < w; ++c) {
                        std::vector<ggml_fp16_t> hd((size_t) nkd[c]*GGML_KQ_MASK_PAD, ninf);
                        for (int i = 0; i <= p + c; ++i) hd[i] = z;
                        ggml_backend_tensor_set(md[c], hd.data(), 0, ggml_nbytes(md[c]));
                    }

                    ggml_backend_graph_compute(be, gv);
                    std::vector<float> rv(ggml_nelements(ov));
                    ggml_backend_tensor_get(ov, rv.data(), 0, ggml_nbytes(ov));
                    int bad = 0;
                    for (int c = 0; c < w; ++c) {
                        ggml_backend_graph_compute(be, gd[c]);
                        std::vector<float> rd(ggml_nelements(od[c]));
                        ggml_backend_tensor_get(od[c], rd.data(), 0, ggml_nbytes(od[c]));
                        const size_t colf = (size_t) sh.D*sh.H;   // ov = [D, H, w]
                        bool fin = true;
                        for (float x : rd) fin = fin && std::isfinite(x);
                        if (!fin || memcmp(rd.data(), rv.data() + c*colf, colf*sizeof(float)) != 0) {
                            ++bad;
                        }
                    }
                    ++ncases;
                    ncols_tot += w;
                    ncols_bad += bad;
                    if (bad) {
                        if (straddle) ++nstraddle_bad;
                        printf("  D=%d H=%d Hkv=%d w=%d p=%d n_kv=%d%s: %d/%d columns differ from their decode\n",
                               sh.D, sh.H, sh.Hkv, w, p, n_kv_v, straddle ? " (straddles a 256 pad)" : "", bad, w);
                    }
                    ggml_backend_buffer_free(buf);
                    ggml_free(ctx);
                }
            }
        }
    }
    const char * pad = getenv("PXA_FA_VEC_NY1_PAD");
    printf("test-pxa-fa-ny1-pad: %s (%d cases, %d/%d verify columns bitwise == decode, %d straddling cases differ; PXA_FA_VEC_NY1_PAD=%s)\n",
           ncols_bad ? "FAIL" : "OK", ncases, ncols_tot - ncols_bad, ncols_tot, nstraddle_bad, pad ? pad : "default");
    ggml_backend_free(be);
    return ncols_bad ? 1 : 0;
#endif
}
