// Head 512/512 flash attention on the device at SHORT KV lengths, against a double-precision host
// reference (bug gemma4-d512-fused-short-prefill).
//
// The fused 512/512 route on sm_70 (PXA_FA_D512_VOLTA) is only reached by a node the graph builder
// marked (op_params[5] == 1), so this harness marks its nodes the same way. Every case is one
// decode-shaped node: a KV range padded to FATTN_KQ_STRIDE (256) cells, of which only the first
// n_real are live and the rest are masked with -inf -- exactly what a short context presents to
// the kernel. The padded cells carry non-zero data, as a reused KV cache does.
//
// The unfused chain the engine builds for the same node is computed alongside and held to the same
// tolerance: it was the chain, not the fused kernels, that was wrong at a short context -- its two
// products ran with fp16 accumulation and fp16 scores (see PXA_FA_D512_CHAIN_F32).
//
// Pass: every case, fused and chain, within tolerance of the reference. On a card that never reaches the fused
// route the cases still run (and must still pass on whatever route serves them); the driver
// script says whether the fused kernel was actually engaged.

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
    // Roughly unit-variance, like an RMS-normalised activation.
    float gauss() { float x = 0.0f; for (int i = 0; i < 4; ++i) x += unit(); return x*0.8660254f; }
};

static constexpr int64_t D = 512;

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- cannot run\n");
        return 77;
    }

    // n_real live cells; the node's KV range is n_real padded up to 256.
    // The long rows (1024, 1500) are the regime the bug report calls exact-grade; the short rows
    // are the ones it calls wrong. Same kernel, same data statistics, so the two must agree.
    const int64_t n_reals[] = { 1, 2, 6, 7, 16, 31, 32, 48, 63, 64, 65, 100, 128, 200, 256, 300, 1024, 1500 };
    // Logit spread: the product of the scale and the q/k magnitudes. Gemma 4 attends with scale
    // 1.0 over RMS-normalised q and k, so its logits are far wider than 1/sqrt(D) would give.
    const float sigmas[] = { 1.0f, 8.0f, 24.0f };
    // (A logit spread of 48 is past anything Gemma 4 produces -- its k_norm folds the score scale
    // to ~0.06 per element -- and there the tile kernel's half2 accumulator reaches NMSE ~1e-5.)
    const int64_t widths[] = { 1, 2 };
    const int64_t nh = 16, nh_kv = 2; // GQA 8, the shape the fused route requires

    int n_fail = 0, n_run = 0;
    for (int64_t nb : widths)
    for (float sigma : sigmas)
    for (int64_t n_real : n_reals) {
        if (n_real < nb) {
            continue; // the first query column would see no key at all
        }
        const int64_t n_kv = (n_real + 255)/256*256;

        ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nb,   nh,    1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, nh_kv, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, nh_kv, 1);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, 1);
        const float scale = sigma/sqrtf((float) D);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
        ((int32_t *) o->op_params)[5] = 1; // the graph builder's mark: this node may take the fused route

        // The unfused chain on the same inputs: what the engine computes when the node is not fused
        // (llm_build_kqv's chain with an FA-layout V cache). It is held to the same tolerance.
        // The chain requests fp32 exactly as llm_build_kqv does for a 512/512 head, under the same
        // lever: PXA_FA_D512_CHAIN_F32=0 builds the old fp16 chain, which this harness fails.
        static const bool chain_f32 = !(getenv("PXA_FA_D512_CHAIN_F32") && strcmp(getenv("PXA_FA_D512_CHAIN_F32"), "0") == 0);
        ggml_tensor * kq0 = ggml_mul_mat(ctx, k, q);
        if (chain_f32) ggml_mul_mat_set_prec(kq0, GGML_PREC_F32);
        ggml_tensor * kq  = ggml_soft_max_ext(ctx, kq0, m, scale, 0.0f);
        ggml_tensor * vt  = ggml_cont(ctx, ggml_transpose(ctx, v));
        ggml_tensor * kqv0 = ggml_mul_mat(ctx, vt, kq);
        if (chain_f32) ggml_mul_mat_set_prec(kqv0, GGML_PREC_F32);
        ggml_tensor * kqv = ggml_cont(ctx, ggml_permute(ctx, kqv0, 0, 2, 1, 3));

        if (!ggml_backend_supports_op(cuda, o)) {
            printf("  nb=%lld sigma=%5.1f n_real=%4lld  SKIPPED (backend declines)\n",
                   (long long) nb, sigma, (long long) n_real);
            ggml_free(ctx);
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);

        lcg r(0xd512000ull + (uint64_t) (n_real*131 + nb*7) + (uint64_t) sigma);
        std::vector<float> qf((size_t) ggml_nelements(q)), kf((size_t) ggml_nelements(k)), vf((size_t) ggml_nelements(v));
        for (float & x : qf) x = r.gauss();
        for (float & x : kf) x = r.gauss();
        for (float & x : vf) x = r.gauss();
        std::vector<ggml_fp16_t> kh(kf.size()), vh(vf.size());
        ggml_fp32_to_fp16_row(kf.data(), kh.data(), (int64_t) kf.size());
        ggml_fp32_to_fp16_row(vf.data(), vh.data(), (int64_t) vf.size());
        for (size_t i = 0; i < kf.size(); ++i) kf[i] = ggml_fp16_to_fp32(kh[i]); // reference sees the cache as stored
        for (size_t i = 0; i < vf.size(); ++i) vf[i] = ggml_fp16_to_fp32(vh[i]);

        // Decode-shaped causal mask: query j sits at position n_real - nb + j.
        const int64_t mrows = m->ne[1];
        std::vector<float> mf((size_t) (n_kv*mrows));
        for (int64_t j = 0; j < mrows; ++j) {
            const int64_t pos = n_real - nb + j;
            for (int64_t c = 0; c < n_kv; ++c) {
                mf[(size_t) (j*n_kv + c)] = (j < nb && c <= pos) ? 0.0f : -INFINITY;
            }
        }
        std::vector<ggml_fp16_t> mh(mf.size());
        ggml_fp32_to_fp16_row(mf.data(), mh.data(), (int64_t) mf.size());

        ggml_backend_tensor_set(q, qf.data(), 0, qf.size()*sizeof(float));
        ggml_backend_tensor_set(k, kh.data(), 0, kh.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, vh.data(), 0, vh.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(m, mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_build_forward_expand(gf, kqv);
        bool ok = ggml_backend_graph_compute(cuda, gf) == GGML_STATUS_SUCCESS;

        std::vector<float> res((size_t) ggml_nelements(o)); // [D, nh, nb]
        std::vector<float> chn((size_t) ggml_nelements(kqv));
        if (ok) {
            ggml_backend_tensor_get(o,   res.data(), 0, res.size()*sizeof(float));
            ggml_backend_tensor_get(kqv, chn.data(), 0, chn.size()*sizeof(float));
        }

        // Reference in double.
        double err2 = 0.0, ref2 = 0.0, maxerr = 0.0, cerr2 = 0.0;
        std::vector<double> lg((size_t) n_kv), acc((size_t) D);
        for (int64_t j = 0; j < nb && ok; ++j) {
            const int64_t pos = n_real - nb + j;
            for (int64_t h = 0; h < nh; ++h) {
                const int64_t hk = h/(nh/nh_kv);
                const float * qr = qf.data() + (size_t) ((h*nb + j)*D);
                double mx = -INFINITY;
                for (int64_t c = 0; c <= pos; ++c) {
                    const float * kr = kf.data() + (size_t) ((hk*n_kv + c)*D);
                    double s = 0.0;
                    for (int64_t d = 0; d < D; ++d) s += (double) qr[d]*kr[d];
                    lg[(size_t) c] = s*scale;
                    mx = std::max(mx, lg[(size_t) c]);
                }
                double den = 0.0;
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int64_t c = 0; c <= pos; ++c) {
                    const double p = exp(lg[(size_t) c] - mx);
                    den += p;
                    const float * vr = vf.data() + (size_t) ((hk*n_kv + c)*D);
                    for (int64_t d = 0; d < D; ++d) acc[(size_t) d] += p*vr[d];
                }
                const float * out = res.data() + (size_t) ((j*nh + h)*D);
                const float * cho = chn.data() + (size_t) ((j*nh + h)*D);
                for (int64_t d = 0; d < D; ++d) {
                    const double ref = acc[(size_t) d]/den;
                    const double e   = (double) out[d] - ref;
                    if (!std::isfinite(out[d])) { maxerr = INFINITY; }
                    err2 += e*e; ref2 += ref*ref;
                    cerr2 += ((double) cho[d] - ref)*((double) cho[d] - ref);
                    maxerr = std::max(maxerr, fabs(e));
                }
            }
        }
        const double nmse  = ref2 > 0.0 ? err2/ref2  : err2;
        const double cnmse = ref2 > 0.0 ? cerr2/ref2 : cerr2;
        // f16 K/V, f16 probabilities and a half2 accumulator: a correct kernel lands at or below
        // ~1e-6 NMSE at any KV length. A wrongly weighted key at a short context is a percent-level
        // error on the output, i.e. NMSE 1e-4 and up.
        const bool pass = ok && std::isfinite(maxerr) && nmse < 1e-5 && maxerr < 5e-2 && cnmse < 1e-5;
        printf("  nb=%lld sigma=%5.1f n_real=%5lld n_kv=%5lld  fused nmse %.3e maxerr %.3e | chain nmse %.3e  %s\n",
               (long long) nb, sigma, (long long) n_real, (long long) n_kv, nmse, maxerr, cnmse, pass ? "ok" : "FAIL");
        n_run++;
        if (!pass) n_fail++;

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(cuda);
    printf("test-fa-d512-short-kv: %d/%d cases pass\n", n_run - n_fail, n_run);
    return n_fail ? 1 : 0;
}
