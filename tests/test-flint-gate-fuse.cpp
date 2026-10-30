// PXA_FLINT_GATE_FUSE (ggml-cuda, default OFF since #354, =1 opt-in): the norm_w sigmoid router tail of build_moe_ffn without a selection
// bias (Flint) - SIGMOID -> RESHAPE -> ARGSORT DESC -> VIEW top-k -> GET_ROWS -> RESHAPE -> SUM_ROWS -> DIV - runs as one
// launch (k_sigmoid_topk_norm_f32) and must write exactly what the op chain writes.
//
// The reference is the same chain computed one stage per graph from host copies (sigmoid alone; argsort alone; get_rows +
// sum_rows/div, which the backend fuses as it does in the engine), so no stage can be fused with its neighbour. The fused
// run is the whole chain in one graph, the shape build_moe_ffn emits. Compared bit for bit: the probs, the full argsort,
// the gathered weights and the normalized weights. Logits are drawn to make ties (equal sigmoids, saturated 1.0f) and
// all-zero rows (sum 0 -> norm 0) common. Run it as is (the test sets PXA_FLINT_GATE_FUSE=1 when it is unset, so the
// fused path runs; stderr shows the opt-in banner) and with PXA_FLINT_GATE_FUSE=0 (op chain): both must print OK with the
// same HASH-ALL. Exit 77 without a CUDA device.
//
// ALIAS part (#354): the engine's graph allocator may place SUM_ROWS/DIV (or ARGSORT) inside a buffer the chain already
// freed in op order (probs after GET_ROWS, logits after SIGMOID). The fused chain then runs with tensors placed by hand
// in such layouts, several times each, and every tensor live after the tail (ids, w, wn) must match the separate-buffer
// run bit for bit. Not compared: probs/logits (dead there) and wsum (read only by the DIV; the op chain's fused SUM_ROWS+DIV
// never writes it). Before the #354 guard the dead-probs layout raced.

#include "pxa-env-flag.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

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
};

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const unsigned char * c = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ULL; }
    return h;
}

template <typename T>
static std::vector<T> get(ggml_tensor * t) {
    std::vector<T> v(ggml_nbytes(t) / sizeof(T));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static bool run(ggml_backend_t be, ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    return ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
}

// ALIAS part: the fused chain with hand-placed buffers. layout 0 = all separate (the reference), 1 = wsum/wn inside the dead
// probs (probs not in place), 2 = probs in place on logits + wsum/wn inside them, 3 = ids on the dead logits, 4 = wn in place
// on w, 5 = layout 1 with wn in place on w. Returns false on a graph failure; fills ids/w/wsum/wn.
static bool run_alias(ggml_backend_t be, const std::vector<float> & x, int ne, int nr, int k, int layout,
                      std::vector<int32_t> & a_out, std::vector<float> & g_out, std::vector<float> & s_out, std::vector<float> & d_out) {
    ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * L  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nr);
    ggml_tensor * P  = ggml_sigmoid(ctx, L);
    ggml_tensor * R3 = ggml_reshape_3d(ctx, P, 1, ne, nr);
    ggml_tensor * T  = ggml_top_k(ctx, P, k);
    ggml_tensor * A  = T->view_src;
    ggml_tensor * G  = ggml_get_rows(ctx, R3, T);
    ggml_tensor * R  = ggml_reshape_2d(ctx, G, k, nr);
    ggml_tensor * SR = ggml_sum_rows(ctx, R);
    ggml_tensor * D  = ggml_div(ctx, R, SR);
    ggml_tensor * D4 = ggml_reshape_3d(ctx, D, 1, k, nr);

    auto up = [](size_t n) { return (n + 255) & ~(size_t) 255; };
    const size_t nL = up(ggml_nbytes(L)), nA = up(ggml_nbytes(A)), nG = up(ggml_nbytes(G)), nS = up(ggml_nbytes(SR)), nD = up(ggml_nbytes(D));
    // regions: [L][P][A][G][SR][D] when separate; the alias layouts move SR/D (or A, or P) into a dead region
    size_t oL = 0, oP = nL, oA = oP + nL, oG = oA + nA, oS = oG + nG, oD = oS + nS;
    const size_t total = oD + nD;
    if (layout == 1 || layout == 5) { oD = oP + nL - nD - nS; oS = oP + nL - nS; }           // tail of the probs buffer
    if (layout == 2) { oP = oL; oD = oL + nL - nD - nS; oS = oL + nL - nS; }                 // probs in place on logits
    if (layout == 3) { oA = oL; }                                                             // ids on the dead logits
    if (layout == 4 || layout == 5) { oD = oG; }                                              // wn in place on w
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(be, total);
    if (!buf) { ggml_free(ctx); return false; }
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    ggml_backend_tensor_alloc(buf, L,  base + oL);
    ggml_backend_tensor_alloc(buf, P,  base + oP);
    ggml_backend_tensor_alloc(buf, A,  base + oA);
    ggml_backend_tensor_alloc(buf, G,  base + oG);
    ggml_backend_tensor_alloc(buf, SR, base + oS);
    ggml_backend_tensor_alloc(buf, D,  base + oD);
    for (ggml_tensor * v : { R3, T, R, D4 }) ggml_backend_view_init(v);

    ggml_backend_tensor_set(L, x.data(), 0, ggml_nbytes(L));
    bool ok = run(be, ctx, D4);
    if (ok) {
        a_out = get<int32_t>(A);
        g_out = get<float>(G);
        s_out = get<float>(SR);
        d_out = get<float>(D);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    // the lever is default OFF (opt-in =1): this test exists for the fused path, so it opts in unless the caller set the
    // variable (=0 runs the op chain for the HASH-ALL cross-check). Set before the backend reads it (first use, cached).
    if (!getenv("PXA_FLINT_GATE_FUSE")) {
        setenv("PXA_FLINT_GATE_FUSE", "1", 0);
    }
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- skipped\n");
        return 77;
    }
    const char * e = getenv("PXA_FLINT_GATE_FUSE");
    printf("PXA_FLINT_GATE_FUSE=%s (lever %s)\n", e ? e : "(unset)", pxa_env_flag("PXA_FLINT_GATE_FUSE", 0) ? "ON (fused)" : "OFF (op chain)");

    struct shape { int ne; int k; };
    const shape shapes[] = { {256, 6}, {256, 8}, {128, 8}, {64, 4}, {32, 2}, {160, 6}, {512, 10}, {1000, 32}, {48, 6} };
    const int rows[] = { 1, 3, 64, 512 };
    int rc = 0, ncase = 0;
    uint64_t all = 1469598103934665603ULL;

    for (const shape & sh : shapes) {
        for (int nr : rows) {
            for (int dist = 0; dist < 3; ++dist) {   // 0 = router-like spread, 1 = heavy ties + saturation, 2 = 0-rows mixed in
                const int ne = sh.ne, k = sh.k;
                lcg r(0xF1A7ull + (uint64_t) ne*131 + (uint64_t) nr*7 + dist*1000003ull);
                std::vector<float> x((size_t) ne*nr);
                for (int ir = 0; ir < nr; ++ir) {
                    const bool zero_row = dist == 2 && ir % 3 == 1;
                    for (int c = 0; c < ne; ++c) {
                        const uint32_t v = r.next();
                        float f;
                        if (zero_row)       f = -200.0f - (float) (v % 7);                       // sigmoid underflows to 0
                        else if (dist == 0) f = ((float) (v % 20001) - 10000.0f) / 2500.0f;      // [-4, 4]
                        else                f = (float) ((int) (v % 9) - 4) * 0.5f + ((v >> 8) % 4 == 0 ? 40.0f : 0.0f);
                        x[(size_t) ir*ne + c] = f;
                    }
                }

                ggml_init_params ip = { ggml_tensor_overhead()*64 + 4*ggml_graph_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                // reference leaves: one stage per graph
                ggml_tensor * L1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nr);
                ggml_tensor * P1 = ggml_sigmoid(ctx, L1);
                ggml_tensor * P2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nr);
                ggml_tensor * A2 = ggml_argsort(ctx, P2, GGML_SORT_ORDER_DESC);
                ggml_tensor * P3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nr);
                ggml_tensor * A3 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne, nr);
                ggml_tensor * V3 = ggml_view_2d(ctx, A3, k, nr, A3->nb[1], 0);
                ggml_tensor * G3 = ggml_get_rows(ctx, ggml_reshape_3d(ctx, P3, 1, ne, nr), V3);
                ggml_tensor * R3 = ggml_reshape_2d(ctx, G3, k, nr);
                ggml_tensor * D3 = ggml_div(ctx, R3, ggml_sum_rows(ctx, R3));
                // fused: the whole chain in one graph, as build_moe_ffn builds it
                ggml_tensor * L  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nr);
                ggml_tensor * P  = ggml_sigmoid(ctx, L);
                ggml_tensor * T  = ggml_top_k(ctx, P, k);
                ggml_tensor * A  = T->view_src;
                ggml_tensor * G  = ggml_get_rows(ctx, ggml_reshape_3d(ctx, P, 1, ne, nr), T);
                ggml_tensor * R  = ggml_reshape_2d(ctx, G, k, nr);
                ggml_tensor * D  = ggml_div(ctx, R, ggml_sum_rows(ctx, R));
                ggml_tensor * D4 = ggml_reshape_3d(ctx, D, 1, k, nr);

                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
                bool ok = buf != nullptr;
                ok = ok && (ggml_backend_tensor_set(L1, x.data(), 0, ggml_nbytes(L1)), run(cuda, ctx, P1));
                std::vector<float> p_ref = ok ? get<float>(P1) : std::vector<float>();
                ok = ok && (ggml_backend_tensor_set(P2, p_ref.data(), 0, ggml_nbytes(P2)), run(cuda, ctx, A2));
                std::vector<int32_t> a_ref = ok ? get<int32_t>(A2) : std::vector<int32_t>();
                if (ok) {
                    ggml_backend_tensor_set(P3, p_ref.data(), 0, ggml_nbytes(P3));
                    ggml_backend_tensor_set(A3, a_ref.data(), 0, ggml_nbytes(A3));
                }
                ok = ok && run(cuda, ctx, D3);
                ok = ok && (ggml_backend_tensor_set(L, x.data(), 0, ggml_nbytes(L)), run(cuda, ctx, D4));
                if (!ok) {
                    printf("  ne=%-5d k=%-3d rows=%-4d dist=%d FAIL (graph compute)\n", ne, k, nr, dist);
                    rc = 1;
                } else {
                    const std::vector<float>   g_ref = get<float>(G3), d_ref = get<float>(D3);
                    const std::vector<float>   p_got = get<float>(P),  g_got = get<float>(G), d_got = get<float>(D);
                    const std::vector<int32_t> a_got = get<int32_t>(A);
                    const bool sp = memcmp(p_ref.data(), p_got.data(), p_ref.size()*4) == 0;
                    const bool sa = memcmp(a_ref.data(), a_got.data(), a_ref.size()*4) == 0;
                    const bool sg = memcmp(g_ref.data(), g_got.data(), g_ref.size()*4) == 0;
                    const bool sd = memcmp(d_ref.data(), d_got.data(), d_ref.size()*4) == 0;
                    const bool good = sp && sa && sg && sd;
                    if (!good) rc = 1;
                    uint64_t h = fnv(p_got.data(), p_got.size()*4);
                    h = fnv(a_got.data(), a_got.size()*4, h);
                    h = fnv(g_got.data(), g_got.size()*4, h);
                    h = fnv(d_got.data(), d_got.size()*4, h);
                    all = fnv(&h, sizeof(h), all);
                    ncase++;
                    printf("  ne=%-5d k=%-3d rows=%-4d dist=%d %s probs %s ids %s w %s wn %s  HASH %016llx\n", ne, k, nr, dist,
                           good ? "OK  " : "FAIL", sp ? "=" : "X", sa ? "=" : "X", sg ? "=" : "X", sd ? "=" : "X",
                           (unsigned long long) h);
                }
                if (buf) ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
        }
    }
    printf("HASH-ALL %016llx over %d cases\n", (unsigned long long) all, ncase);

    // ALIAS part (#354)
    const char * lname[] = { "separate", "wsum/wn in dead probs", "probs on logits + wsum/wn in them", "ids on dead logits",
                             "wn in place on w", "dead probs + wn on w" };
    for (int nr : { 825, 4096 }) {
        for (int k : { 6, 8 }) {
            const int ne = 256;
            lcg r(0x354ull + (uint64_t) nr*31 + (uint64_t) k);
            std::vector<float> x((size_t) ne*nr);
            for (float & f : x) f = ((float) (r.next() % 20001) - 10000.0f) / 2500.0f;
            std::vector<int32_t> a0; std::vector<float> g0, s0, d0;
            if (!run_alias(cuda, x, ne, nr, k, 0, a0, g0, s0, d0)) { printf("  alias rows=%d k=%d FAIL (graph compute)\n", nr, k); rc = 1; continue; }
            for (int layout = 1; layout <= 5; ++layout) {
                int bad = 0, reps = 0;
                for (int rep = 0; rep < 8; ++rep, ++reps) {
                    std::vector<int32_t> a; std::vector<float> g, sm, d;
                    if (!run_alias(cuda, x, ne, nr, k, layout, a, g, sm, d)) { bad = -1; break; }
                    const bool w_live = layout != 4 && layout != 5;          // w holds wn after an in-place DIV
                    const bool same = a == a0 && (!w_live || memcmp(g.data(), g0.data(), g0.size()*4) == 0) &&
                                      memcmp(d.data(), d0.data(), d0.size()*4) == 0;   // wsum is dead (see the header)
                    bad += same ? 0 : 1;
                }
                if (bad != 0) rc = 1;
                printf("  alias rows=%-4d k=%d layout %d (%s): %s %d/%d runs equal to the separate-buffer run\n", nr, k, layout,
                       lname[layout], bad == 0 ? "OK  " : "FAIL", bad < 0 ? 0 : reps - bad, reps);
            }
        }
    }
    ggml_backend_free(cuda);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
