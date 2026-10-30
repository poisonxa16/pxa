// test-fa-mma-empty-slice.cpp -- FA stream-k with KV bounds (2026-10-22).
// fattn-mma-f16 / fattn-new-mma split the KV range of a column tile over several CUDA blocks (stream-k) and clamp each
// block's slice to the tile's unmasked range: [KV_min, KV_max) for SWA (n_swa in op_params[4]) and [0, KV_max) when
// ne01 >= 1024. A slice clamped EMPTY used to run process_tile's unconditional last iteration at kb0_stop-1, a tile
// another block owns, so that tile was counted twice (tiny Flint, window 33: GPU KLD 0.044 vs 1e-5 without SWA).
// Each case runs one FLASH_ATTN_EXT node on the CUDA backend and compares it with a double-precision host reference
// (causal + sliding-window mask, GQA, optional sinks). Needs a card (ctest -L cuda; PXA_TEST_DEVICE=N); a CPU build SKIPs.
// Exit 0 = every case within nmse 5e-4 (the test-backend-ops FA bar) and finite.
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

static uint32_t rng_state = 20261022u;
static float frand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

struct fa_case {
    const char * name;
    int D, DV, H, Hkv, n_kv, nb, p0, n_swa;
    bool sinks;
};

int main() {
#ifndef GGML_USE_CUDA
    printf("test-fa-mma-empty-slice: SKIP (no CUDA backend in this build)\n");
    return 0;
#else
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("test-fa-mma-empty-slice: SKIP (no CUDA device)\n");
        return 0;
    }
    const char * dv = getenv("PXA_TEST_DEVICE");
    const int dev = dv ? atoi(dv) : 0;
    ggml_backend_t be = ggml_backend_cuda_init(dev, nullptr);
    if (!be) {
        printf("test-fa-mma-empty-slice: SKIP (CUDA init failed)\n");
        return 0;
    }
    printf("test-fa-mma-empty-slice: device %d cc %d\n", dev, ggml_backend_cuda_get_device_cc(dev));

    const fa_case cases[] = {
        // SWA (KV_min_max scan), small grids -> stream-k
        { "swa33-gqa8-sinks",   128, 128,  8, 1, 1024,  128,  512,   33, true  },  // tiny Flint
        { "swa33-gqa8",         128, 128,  8, 1, 1024,  128,  896,   33, false },
        { "swa513-flint",       128, 128, 32, 4, 2048,  512, 1024,  513, true  },  // Flint layout
        { "swa513-flint-ub128", 128, 128, 32, 4, 2048,  128, 1536,  513, true  },
        { "swa200-d256",        256, 256, 16, 8, 1024,  128,  768,  200, false },  // Gemma-like head 256
        { "swa128-d64-sinks",    64,  64, 64, 8, 1536,  256, 1024,  128, true  },  // gpt-oss-like
        // KV_max only (ne01 >= 1024, causal, no SWA); ne01 > 2048 forces stream-k on fattn-mma-f16
        { "kvmax-causal-2304",  128, 128,  8, 1, 2560, 2304,  256,    0, false },
        { "kvmax-causal-1024",  128, 128,  4, 4, 2048, 1024, 1024,    0, false },
        // new-mma family (K 192 / V 128) with KV_max
        { "kvmax-k192v128",     192, 128,  4, 4, 2048, 1024, 1024,    0, false },
        // control: no bounds engaged (short batch, no SWA)
        { "control-causal",     128, 128,  8, 1, 1024,  128,  896,    0, false },
    };

    int nbad = 0, nskip = 0;
    for (const fa_case & c : cases) {
        const float scale = 1.0f / sqrtf((float) c.D);
        const int rows = GGML_PAD(c.nb, GGML_KQ_MASK_PAD);

        ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.D,  c.nb,   c.H);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.D,  c.n_kv, c.Hkv);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.DV, c.n_kv, c.Hkv);
        ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, rows);
        ggml_tensor * s = c.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.H) : nullptr;
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        if (s) ggml_flash_attn_ext_add_sinks(o, s);
        ((int32_t *) o->op_params)[4] = c.n_swa;   // as the graph builders do for SWA layers
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, o);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

        std::vector<float> hq((size_t) c.D*c.nb*c.H), hk((size_t) c.D*c.n_kv*c.Hkv), hv((size_t) c.DV*c.n_kv*c.Hkv), hs(c.H);
        for (auto & x : hq) x = 2.0f*frand();
        std::vector<ggml_fp16_t> t16(hk.size());
        for (size_t i = 0; i < hk.size(); ++i) { t16[i] = ggml_fp32_to_fp16(frand()); hk[i] = ggml_fp16_to_fp32(t16[i]); }
        ggml_backend_tensor_set(k, t16.data(), 0, ggml_nbytes(k));
        t16.resize(hv.size());
        for (size_t i = 0; i < hv.size(); ++i) { t16[i] = ggml_fp32_to_fp16(frand()); hv[i] = ggml_fp16_to_fp32(t16[i]); }
        ggml_backend_tensor_set(v, t16.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(q, hq.data(), 0, ggml_nbytes(q));
        for (auto & x : hs) x = 1.5f*frand();
        if (s) ggml_backend_tensor_set(s, hs.data(), 0, ggml_nbytes(s));

        auto allowed = [&](int i, int j) {
            const int p = c.p0 + i;
            return j <= p && (c.n_swa == 0 || p - j < c.n_swa);
        };
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> hm((size_t) c.n_kv*rows, ninf);
        for (int i = 0; i < c.nb; ++i) for (int j = 0; j < c.n_kv; ++j) if (allowed(i, j)) hm[(size_t) i*c.n_kv + j] = z;
        ggml_backend_tensor_set(m, hm.data(), 0, ggml_nbytes(m));

        // A node the backend declines never reaches this card in a real graph (the scheduler keeps it on another
        // backend), and computing it here anyway runs into the dispatcher's abort: sm_60/61 have no kernel for
        // K 192 / V 128 (P100 ctest). Nothing to compare, so the case is skipped, not failed.
        if (!ggml_backend_supports_op(be, o)) {
            printf("  %-20s D=%d/%d H=%d/%d n_kv=%d nb=%d: SKIP (the CUDA backend declines this node on this device)\n",
                   c.name, c.D, c.DV, c.H, c.Hkv, c.n_kv, c.nb);
            ++nskip;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }
        ggml_backend_graph_compute(be, g);
        std::vector<float> ho(ggml_nelements(o));   // [DV, H, nb]
        ggml_backend_tensor_get(o, ho.data(), 0, ggml_nbytes(o));

        double se = 0.0, sr = 0.0;
        bool fin = true;
        std::vector<double> sc(c.n_kv), acc(c.DV);
        const int gqa = c.H / c.Hkv;
        for (int h = 0; h < c.H; ++h) {
            const int hk_i = h / gqa;
            for (int i = 0; i < c.nb; ++i) {
                const float * qi = hq.data() + ((size_t) h*c.nb + i)*c.D;
                double mx = c.sinks ? (double) hs[h] : -INFINITY;
                for (int j = 0; j < c.n_kv; ++j) {
                    if (!allowed(i, j)) { sc[j] = -INFINITY; continue; }
                    const float * kj = hk.data() + ((size_t) hk_i*c.n_kv + j)*c.D;
                    double d = 0.0;
                    for (int e = 0; e < c.D; ++e) d += (double) qi[e]*kj[e];
                    sc[j] = d*scale;
                    if (sc[j] > mx) mx = sc[j];
                }
                double den = c.sinks ? exp((double) hs[h] - mx) : 0.0;
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int j = 0; j < c.n_kv; ++j) {
                    if (sc[j] == -INFINITY) continue;
                    const double w = exp(sc[j] - mx);
                    den += w;
                    const float * vj = hv.data() + ((size_t) hk_i*c.n_kv + j)*c.DV;
                    for (int e = 0; e < c.DV; ++e) acc[e] += w*vj[e];
                }
                const float * oi = ho.data() + ((size_t) i*c.H + h)*c.DV;
                for (int e = 0; e < c.DV; ++e) {
                    const double r = acc[e]/den;
                    fin = fin && std::isfinite(oi[e]);
                    se += (oi[e] - r)*(oi[e] - r);
                    sr += r*r;
                }
            }
        }
        const double nmse = se / (sr > 0 ? sr : 1.0);
        const bool ok = fin && nmse < 5e-4;
        nbad += !ok;
        printf("  %-20s D=%d/%d H=%d/%d n_kv=%d nb=%d p0=%d n_swa=%d sinks=%d: nmse %.3e %s\n",
               c.name, c.D, c.DV, c.H, c.Hkv, c.n_kv, c.nb, c.p0, c.n_swa, (int) c.sinks, nmse, ok ? "OK" : (fin ? "FAIL" : "FAIL (non-finite)"));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    printf("test-fa-mma-empty-slice: %s (%d bad, %d skipped, of %d)\n", nbad ? "FAIL" : "OK", nbad, nskip,
           (int) (sizeof(cases)/sizeof(cases[0])));
    return nbad ? 1 : 0;
#endif
}
