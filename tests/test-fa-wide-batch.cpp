// Large-head flash attention (576/512 MLA, 320/256 MLA, 512/512) at PREFILL widths with GQA, against a
// double-precision host reference (2026-10-21).
//
// Why: on sm_75 the new-MMA kernel (fattn-new-mma.cu) compiles every variant with ncols1*ncols2 > 32 to
// NO_DEVICE_CODE, but the host picked the 64-column tile for any batch wider than 32/ncols2 -- a 576/512
// node at gqa 8 or 16 (DeepSeek-class MLA) or a 320/256 node at gqa 16 trapped on a Turing card at the
// first prompt. The host now keeps the 32-column tile on Turing. The same cases run on every arch that
// reaches the route (sm_75+); on sm_80+ they pin the existing 64-column path to the same reference.
//
// Run it under a compute_75 PTX-only build on an Ampere card (CMAKE_CUDA_ARCHITECTURES=75-virtual) to
// execute the Turing device variants on hardware without a Turing card: highest compiled arch = 750 on
// the host, __CUDA_ARCH__ = 750 in the JIT-compiled device code.
//
// Pass: every case the CUDA backend accepts lands within tolerance; a trap aborts the process (the
// regression this guards). Exit 77 without a card, or when the card takes none of the cases (pre-Turing:
// these head sizes are not served by the new-MMA route there).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; }
    float gauss() { float x = 0.0f; for (int i = 0; i < 4; ++i) x += unit(); return x*0.8660254f; }
};

struct fa_case {
    int64_t dk, dv;   // K/Q head size, V head size
    bool    mla;      // V is the last dv elements of each K row (how the MLA kernels read it)
    int64_t gqa;      // query heads per KV head
};

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- cannot run\n");
        return 77;
    }

    const fa_case cases[] = {
        { 576, 512, true,  16 }, { 576, 512, true,  8 }, { 576, 512, true,  4 },
        { 320, 256, true,  16 },
        { 512, 512, false, 16 }, { 512, 512, false, 8 },
    };
    // 1..16 = the decode and narrow-verify widths; 33, 64, 100 = prompt widths past the 32-column tile.
    const int64_t widths[] = { 1, 2, 5, 16, 33, 64, 100 };
    const int64_t n_kv   = 512;  // a multiple of FATTN_KQ_STRIDE; every query sees a causal prefix of it
    const int64_t nh_kv  = 1;

    int n_fail = 0, n_run = 0, n_skip = 0;
    for (const fa_case & fc : cases)
    for (int64_t nb : widths) {
        const int64_t nh = fc.gqa*nh_kv;
        const int64_t DK = fc.dk, DV = fc.dv;

        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, nb,   nh,    1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, n_kv, nh_kv, 1);
        ggml_tensor * v = fc.mla
            ? ggml_view_4d(ctx, k, DV, n_kv, nh_kv, 1, k->nb[1], k->nb[2], k->nb[3], (size_t) (DK - DV)*ggml_element_size(k))
            : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, n_kv, nh_kv, 1);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, 1);
        const float scale = 1.0f/sqrtf((float) DK);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);

        if (!ggml_backend_supports_op(cuda, o)) {
            printf("  D=%lld/%lld gqa=%2lld nb=%3lld  SKIPPED (backend declines)\n",
                   (long long) DK, (long long) DV, (long long) fc.gqa, (long long) nb);
            n_skip++;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);

        lcg r(0xfa3a000ull + (uint64_t) (DK*1009 + DV*17 + fc.gqa*131 + nb));
        std::vector<float> qf((size_t) ggml_nelements(q)), kf((size_t) ggml_nelements(k));
        std::vector<float> vf(fc.mla ? 0 : (size_t) ggml_nelements(v));
        for (float & x : qf) x = r.gauss();
        for (float & x : kf) x = r.gauss();
        for (float & x : vf) x = r.gauss();
        std::vector<ggml_fp16_t> kh(kf.size()), vh(vf.size());
        ggml_fp32_to_fp16_row(kf.data(), kh.data(), (int64_t) kf.size());
        for (size_t i = 0; i < kf.size(); ++i) kf[i] = ggml_fp16_to_fp32(kh[i]); // the reference sees the cache as stored
        if (!fc.mla) {
            ggml_fp32_to_fp16_row(vf.data(), vh.data(), (int64_t) vf.size());
            for (size_t i = 0; i < vf.size(); ++i) vf[i] = ggml_fp16_to_fp32(vh[i]);
        }

        // Causal prompt mask: query column j sits at position n_kv - nb + j.
        const int64_t mrows = m->ne[1];
        std::vector<float> mf((size_t) (n_kv*mrows));
        for (int64_t j = 0; j < mrows; ++j) {
            const int64_t pos = n_kv - nb + j;
            for (int64_t c = 0; c < n_kv; ++c) {
                mf[(size_t) (j*n_kv + c)] = (j < nb && c <= pos) ? 0.0f : -INFINITY;
            }
        }
        std::vector<ggml_fp16_t> mh(mf.size());
        ggml_fp32_to_fp16_row(mf.data(), mh.data(), (int64_t) mf.size());

        ggml_backend_tensor_set(q, qf.data(), 0, qf.size()*sizeof(float));
        ggml_backend_tensor_set(k, kh.data(), 0, kh.size()*sizeof(ggml_fp16_t));
        if (!fc.mla) {
            ggml_backend_tensor_set(v, vh.data(), 0, vh.size()*sizeof(ggml_fp16_t));
        }
        ggml_backend_tensor_set(m, mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        const bool ok = ggml_backend_graph_compute(cuda, gf) == GGML_STATUS_SUCCESS;

        std::vector<float> res((size_t) ggml_nelements(o)); // [DV, nh, nb]
        if (ok) {
            ggml_backend_tensor_get(o, res.data(), 0, res.size()*sizeof(float));
        }

        double err2 = 0.0, ref2 = 0.0, maxerr = 0.0;
        std::vector<double> lg((size_t) n_kv), acc((size_t) DV);
        for (int64_t j = 0; j < nb && ok; ++j) {
            const int64_t pos = n_kv - nb + j;
            for (int64_t h = 0; h < nh; ++h) {
                const int64_t hk = h/fc.gqa;
                const float * qr = qf.data() + (size_t) ((h*nb + j)*DK);
                double mx = -INFINITY;
                for (int64_t c = 0; c <= pos; ++c) {
                    const float * kr = kf.data() + (size_t) ((hk*n_kv + c)*DK);
                    double s = 0.0;
                    for (int64_t d = 0; d < DK; ++d) s += (double) qr[d]*kr[d];
                    lg[(size_t) c] = s*scale;
                    mx = std::max(mx, lg[(size_t) c]);
                }
                double den = 0.0;
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int64_t c = 0; c <= pos; ++c) {
                    const double p = exp(lg[(size_t) c] - mx);
                    den += p;
                    const float * vr = fc.mla ? kf.data() + (size_t) ((hk*n_kv + c)*DK + (DK - DV))
                                              : vf.data() + (size_t) ((hk*n_kv + c)*DV);
                    for (int64_t d = 0; d < DV; ++d) acc[(size_t) d] += p*vr[d];
                }
                const float * out = res.data() + (size_t) ((j*nh + h)*DV);
                for (int64_t d = 0; d < DV; ++d) {
                    const double ref = acc[(size_t) d]/den;
                    const double e   = (double) out[d] - ref;
                    if (!std::isfinite(out[d])) { maxerr = INFINITY; }
                    err2 += e*e; ref2 += ref*ref;
                    maxerr = std::max(maxerr, fabs(e));
                }
            }
        }
        const double nmse = ref2 > 0.0 ? err2/ref2 : err2;
        // f16 K/V and f16 probabilities with f32 accumulation: a correct kernel sits near 1e-6 NMSE; a
        // wrong column tiling or a missed KV block is an O(1) error on the affected rows.
        const bool pass = ok && std::isfinite(maxerr) && nmse < 1e-4 && maxerr < 5e-2;
        printf("  D=%lld/%lld gqa=%2lld nb=%3lld  nmse %.3e maxerr %.3e  %s\n",
               (long long) DK, (long long) DV, (long long) fc.gqa, (long long) nb, nmse, maxerr, pass ? "ok" : "FAIL");
        n_run++;
        if (!pass) n_fail++;

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(cuda);
    printf("test-fa-wide-batch: %d/%d cases pass, %d declined\n", n_run - n_fail, n_run, n_skip);
    if (n_run == 0) {
        return 77;
    }
    return n_fail ? 1 : 0;
}
