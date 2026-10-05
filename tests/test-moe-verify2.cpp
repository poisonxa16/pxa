//
// PXA_MOE_VERIFY2 parity harness (ggml/src/ggml-cuda/mmvq-verify2.cuh).
//
// The lever serves a 2..4-token MoE verify batch as ONE expert-grouped id-GEMV launch per projection
// instead of one launch per token. It claims:
//
//   (a) fused MoE up/gate (+ the down that follows it): BIT-IDENTICAL to the per-token fast-TG loop;
//   (b) standalone MUL_MAT_ID at 2..4 tokens: correct (it replaces the MMQ-id GEMM there, so this is
//       a tolerance claim against a host reference, not an identity claim).
//
// Per case and per verify width Ny = 1..5 this prints
//   NMSE  of every CUDA result against a double-precision host reference over the dequantized weights
//   HASH-EXACT <case> ny=<n> <fnv>   for the fused graph (up/gate output AND down output)
//   HASH-INFO  <case> ny=<n> <fnv>   for the unfused graph (MMQ-id when the lever is off)
//
// The lever is read once per process, so the identity proof is a diff of two runs:
//
//   PXA_MOE_VERIFY2=0 ./test-moe-verify2 | grep HASH-EXACT > off.txt
//   PXA_MOE_VERIFY2=1 ./test-moe-verify2 | grep HASH-EXACT > on.txt
//   diff off.txt on.txt      # must be empty
//
// Exit 0 = every NMSE under its bar, 1 = a failure, 77 = no CUDA device.
//
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
#include <random>
#include <string>
#include <vector>

struct moe_case {
    const char * name;
    ggml_type    type;
    int          K;       // n_embd (also the down output rows)
    int          FF;      // expert ffn width
    int          E;       // experts
    int          used;    // experts per token
    bool         merged;  // one [gate | up] stack
    ggml_unary_op op;
};

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static double nmse(const std::vector<float> & a, const std::vector<double> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = a[i] - ref[i]; num += d*d; den += ref[i]*ref[i]; }
    return den > 0 ? num/den : num;
}

static double act(ggml_unary_op op, double g, double u) {
    switch (op) {
        case GGML_UNARY_OP_SILU: return g/(1 + std::exp(-g)) * u;
        case GGML_UNARY_OP_RELU: return std::max(g, 0.0) * u;
        default: return 0.5*g*u*(1.0 + std::tanh(0.79788456080286535587989211986876*g*(1.0 + 0.044715*g*g)));
    }
}

// fill a quantized 3-D weight [ncols, nrows, E] from random floats; keep the dequantized copy for the reference
static void fill_weight(ggml_tensor * t, std::mt19937 & rng, std::vector<float> & deq) {
    const int64_t ncols = t->ne[0], nrows = t->ne[1]*t->ne[2];
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(ncols*nrows);
    for (auto & x : f) x = u(rng);
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, nrows, ncols, nullptr, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
    deq.resize(ncols*nrows);
    ggml_type_traits_t tt = ggml_internal_get_type_traits(t->type);
    const size_t rs = ggml_row_size(t->type, ncols);
    for (int64_t r = 0; r < nrows; ++r) tt.to_float(q.data() + r*rs, deq.data() + r*ncols, ncols);
}

static std::vector<float> get_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size()*sizeof(float));
    return v;
}

static bool run_case(ggml_backend_t cuda, const moe_case & c, uint32_t seed) {
    std::mt19937 rng(seed);
    bool ok = true;

    // ---- weights, resident on the CUDA device (so the graph builds the fused node)
    ggml_init_params wp = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * gu   = c.merged ? ggml_new_tensor_3d(wctx, c.type, c.K, 2*c.FF, c.E) : nullptr;
    ggml_tensor * up   = c.merged ? nullptr : ggml_new_tensor_3d(wctx, c.type, c.K, c.FF, c.E);
    ggml_tensor * gate = c.merged ? nullptr : ggml_new_tensor_3d(wctx, c.type, c.K, c.FF, c.E);
    ggml_tensor * down = ggml_new_tensor_3d(wctx, c.type, c.FF, c.K, c.E);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, cuda);
    if (!wbuf) { printf("%s: weight alloc failed\n", c.name); ggml_free(wctx); return false; }

    std::vector<float> dq_gu, dq_up, dq_gate, dq_down;
    if (c.merged) fill_weight(gu, rng, dq_gu); else { fill_weight(up, rng, dq_up); fill_weight(gate, rng, dq_gate); }
    fill_weight(down, rng, dq_down);
    // expert e, row r of the gate / up halves
    auto Wg = [&](int e, int r) -> const float * {
        return c.merged ? &dq_gu[((size_t)e*2*c.FF + r)*c.K] : &dq_gate[((size_t)e*c.FF + r)*c.K];
    };
    auto Wu = [&](int e, int r) -> const float * {
        return c.merged ? &dq_gu[((size_t)e*2*c.FF + c.FF + r)*c.K] : &dq_up[((size_t)e*c.FF + r)*c.K];
    };
    auto Wd = [&](int e, int r) -> const float * { return &dq_down[((size_t)e*c.K + r)*c.FF]; };

    for (int ny = 1; ny <= 5; ++ny) {
        // ---- routing: token 0 random; later tokens share about half of token 0's experts
        std::vector<int32_t> ids(c.used*ny);
        {
            std::vector<int> perm(c.E);
            for (int i = 0; i < c.E; ++i) perm[i] = i;
            for (int t = 0; t < ny; ++t) {
                std::shuffle(perm.begin(), perm.end(), rng);
                std::vector<char> taken(c.E, 0);
                int n = 0;
                if (t > 0) {
                    for (int k = 0; k < c.used; ++k) {
                        if (rng() & 1) { const int e = ids[k]; ids[t*c.used + n++] = e; taken[e] = 1; }
                    }
                }
                for (int i = 0; n < c.used; ++i) if (!taken[perm[i]]) { ids[t*c.used + n++] = perm[i]; taken[perm[i]] = 1; }
                // shuffle the slot order so shared experts sit at different slots
                std::shuffle(ids.begin() + t*c.used, ids.begin() + (t+1)*c.used, rng);
            }
        }
        std::vector<float> x((size_t)c.K*ny);
        { std::uniform_real_distribution<float> u(-1.0f, 1.0f); for (auto & v : x) v = u(rng); }

        // ---- host reference
        std::vector<double> par_ref((size_t)c.FF*c.used*ny), out_ref((size_t)c.K*c.used*ny);
        for (int t = 0; t < ny; ++t) for (int k = 0; k < c.used; ++k) {
            const int e = ids[t*c.used + k];
            const float * xt = &x[(size_t)t*c.K];
            double * p = &par_ref[((size_t)t*c.used + k)*c.FF];
            for (int r = 0; r < c.FF; ++r) {
                double g = 0, u = 0;
                const float * wg = Wg(e, r); const float * wu = Wu(e, r);
                for (int i = 0; i < c.K; ++i) { g += (double)wg[i]*xt[i]; u += (double)wu[i]*xt[i]; }
                p[r] = act(c.op, g, u);
            }
            double * o = &out_ref[((size_t)t*c.used + k)*c.K];
            for (int r = 0; r < c.K; ++r) {
                const float * wd = Wd(e, r);
                double s = 0;
                for (int i = 0; i < c.FF; ++i) s += (double)wd[i]*p[i];
                o[r] = s;
            }
        }

        for (int variant = 0; variant < 2; ++variant) {
            if (variant == 1 && c.merged) continue;   // the unfused graph is built over the separate stacks
            ggml_init_params ap = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
            ggml_context * actx = ggml_init(ap);
            ggml_tensor * b   = ggml_new_tensor_3d(actx, GGML_TYPE_F32, c.K, 1, ny);
            ggml_tensor * tid = ggml_new_tensor_2d(actx, GGML_TYPE_I32, c.used, ny);
            ggml_tensor * par;
            if (variant == 0) {
                par = c.merged ? ggml_moe_up_gate(actx, gu, nullptr, b, tid, c.op) : ggml_moe_up_gate(actx, up, gate, b, tid, c.op);
            } else {
                ggml_tensor * mu = ggml_mul_mat_id(actx, up,   b, tid);
                ggml_tensor * mg = ggml_mul_mat_id(actx, gate, b, tid);
                par = ggml_fused_mul_unary(actx, mg, mu, c.op);
            }
            ggml_tensor * out = ggml_mul_mat_id(actx, down, par, tid);
            ggml_cgraph * gf = ggml_new_graph(actx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(actx, cuda);
            ggml_backend_tensor_set(b, x.data(), 0, x.size()*sizeof(float));
            ggml_backend_tensor_set(tid, ids.data(), 0, ids.size()*sizeof(int32_t));
            if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
                printf("%s ny=%d v%d: compute failed\n", c.name, ny, variant); ok = false;
            }
            const std::vector<float> pv = get_f32(par), ov = get_f32(out);
            const double e_par = nmse(pv, par_ref), e_out = nmse(ov, out_ref);
            const double bar = 2e-3;
            const bool pass = std::isfinite(e_par) && std::isfinite(e_out) && e_par < bar && e_out < bar;
            ok = ok && pass;
            const uint64_t h = fnv(ov.data(), ov.size()*sizeof(float), fnv(pv.data(), pv.size()*sizeof(float)));
            printf("%-10s %s ny=%d graph=%s par_op=%s nmse_par=%.3e nmse_out=%.3e %s\n", pass ? "PASS" : "FAIL", c.name, ny,
                   variant == 0 ? "fused" : "unfused", ggml_op_name(par->op), e_par, e_out, pass ? "" : "<<<");
            printf("%s %s ny=%d %016llx\n", variant == 0 ? "HASH-EXACT" : "HASH-INFO", c.name, ny, (unsigned long long) h);
            ggml_backend_buffer_free(abuf);
            ggml_free(actx);
        }
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return ok;
}

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) { printf("no CUDA device\n"); return 77; }
    const char * lev = getenv("PXA_MOE_VERIFY2");
    printf("PXA_MOE_VERIFY2=%s\n", lev ? lev : "(unset)");

    const moe_case cases[] = {
        // Gemma 4 26B-A4B routed-expert shape (K 2816, expert FF 704, top-8, merged gate|up, GELU), 32 of its 128 experts
        { "gemma-q4_0", GGML_TYPE_Q4_0, 2816, 704, 32, 8, true,  GGML_UNARY_OP_GELU },
        { "q4_0-sep",   GGML_TYPE_Q4_0,  512, 256, 16, 4, false, GGML_UNARY_OP_GELU },
        { "q8_0-silu",  GGML_TYPE_Q8_0,  512, 256, 16, 4, false, GGML_UNARY_OP_SILU },
        { "q4_K-sep",   GGML_TYPE_Q4_K,  512, 256, 16, 4, false, GGML_UNARY_OP_SILU },
        { "q6_K-mrg",   GGML_TYPE_Q6_K,  512, 256,  8, 4, true,  GGML_UNARY_OP_GELU },
    };
    bool ok = true;
    uint32_t seed = 1234;
    for (const auto & c : cases) ok = run_case(cuda, c, seed++) && ok;
    ggml_backend_free(cuda);
    printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
