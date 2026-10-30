// PXA_GLM_KDA_FUSE pattern test (CPU only, no GPU; 2026-10-18).
//
// The CUDA fusion (ggml/src/ggml-cuda/pxa/pxa-glm-kda-fuse.cuh) is only as good as its matcher: a match that never fires
// is a dead lever, a match with the wrong index mapping is a silent wrong answer. Both are checked here on the CPU with
// the very matcher the CUDA backend calls (pxa-glm-kda-match.h):
//
//  A) synthetic graphs that copy build_glm5next.cpp's two chains op for op, computed by the ggml CPU backend:
//     - the matcher fires (decode width 1 and a prefill width), and the fused kernel's mapping reproduces the CONT
//       output BIT FOR BIT from the chain's own last node (no arithmetic involved: a mapping check);
//     - the fused arithmetic, emulated on the host, agrees with the CPU chain (tolerance: the CPU sigmoid is a SIMD
//       exp, not the CUDA one -- the GPU identity is the kernel's construction plus the SASS census, not this test);
//     - mutations decline: an extra reader of a skipped intermediate, an OUTPUT flag, a non per-head multiplier, a
//       broadcast out-gate operand.
//  B) with a tiny glm5next GGUF (tests/gen_tiny_glm5next.py): the REAL prefill and decode graphs, every node recorded
//     through the eval callback; the gate chain must match once per KDA block and the out-gate at least once per KDA
//     block, and every match's mapping must reproduce the recorded CONT bit for bit.
//
//   test-glm-kda-fuse-match [tiny-glm5next.gguf [n_kda=6]]
#include "ggml.h"
#include "llama.h"
#include "ggml-cuda/pxa/pxa-glm-kda-match.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    float * p = (float *) t->data;
    for (int64_t k = 0; k < ggml_nelements(t); ++k) p[k] = d(rng);
}

static int find_node(const ggml_cgraph * g, const ggml_tensor * t) {
    for (int i = 0; i < g->n_nodes; ++i) if (g->nodes[i] == t) return i;
    return -1;
}

static float host_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static bool close_enough(float a, float b) {
    return std::fabs(a - b) <= 2e-6f * std::max(1.0f, std::fabs(b));
}

// mapping (bitwise) + arithmetic (tolerance) of one gate match, from the values the chain nodes hold
static void verify_gate(const pxa_glm_kda_gate_match & m, const float * cont, const float * s2,
                        const float * g, const float * dtb, const float * A, const char * tag) {
    int64_t bad_map = 0, bad_math = 0;
    for (int64_t k = 0; k < m.n; ++k) {
        const int64_t e = pxa_glm_kda_gate_src_index(m, k);
        if (memcmp(&cont[k], &s2[e], sizeof(float)) != 0) ++bad_map;
        float v = g[e] + dtb[e % m.n0];
        v = v * A[(e / m.hd) % m.nh];
        v = std::fmaf(m.s1_scale, v, m.s1_bias);
        v = host_sigmoid(v);
        v = std::fmaf(m.s2_scale, v, m.s2_bias);
        if (!close_enough(v, cont[k])) ++bad_math;
    }
    CHECK(bad_map == 0, "%s: gate mapping differs at %lld of %lld elements", tag, (long long) bad_map, (long long) m.n);
    CHECK(bad_math == 0, "%s: gate arithmetic differs at %lld of %lld elements", tag, (long long) bad_math, (long long) m.n);
}

static void verify_outgate(const pxa_glm_kda_outgate_match & m, const float * cont, const float * mul,
                           const float * o, const float * g, const char * tag) {
    int64_t bad_map = 0, bad_math = 0;
    for (int64_t k = 0; k < m.n; ++k) {
        if (memcmp(&cont[k], &mul[k], sizeof(float)) != 0) ++bad_map;
        if (!close_enough(o[k] * host_sigmoid(g[k]), cont[k])) ++bad_math;
    }
    CHECK(bad_map == 0, "%s: outgate mapping differs at %lld of %lld", tag, (long long) bad_map, (long long) m.n);
    CHECK(bad_math == 0, "%s: outgate arithmetic differs at %lld of %lld", tag, (long long) bad_math, (long long) m.n);
}

// ---------------------------------------------------------------------------------------------- A) synthetic graphs
enum gate_mut { GM_NONE, GM_EXTRA_READER, GM_OUTPUT_FLAG, GM_A_NOT_PER_HEAD, GM_NO_PERMUTE };

static void synth_gate(int64_t hd, int64_t nh, int64_t T, gate_mut mut) {
    char tag[96];
    snprintf(tag, sizeof(tag), "synth gate hd=%lld nh=%lld T=%lld mut=%d", (long long) hd, (long long) nh, (long long) T, (int) mut);
    ggml_init_params ip = { 64u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    std::mt19937 rng(1234 + (unsigned) (hd*31 + nh*7 + T + mut));
    const int64_t d_inner = hd*nh;
    ggml_tensor * g1  = ggml_new_tensor_2d(c, GGML_TYPE_F32, d_inner, T);
    ggml_tensor * dtb = ggml_new_tensor_1d(c, GGML_TYPE_F32, d_inner);
    ggml_tensor * sa  = mut == GM_A_NOT_PER_HEAD ? ggml_new_tensor_2d(c, GGML_TYPE_F32, hd, nh) : ggml_new_tensor_1d(c, GGML_TYPE_F32, nh);
    fill(g1, rng, -4, 4); fill(dtb, rng, -1, 1); fill(sa, rng, -3, -0.01f);
    // build_glm5next.cpp, the bounded forget gate, op for op
    ggml_tensor * add = ggml_add(c, g1, dtb);
    ggml_tensor * A   = mut == GM_A_NOT_PER_HEAD ? ggml_reshape_3d(c, sa, hd, nh, 1) : ggml_reshape_3d(c, sa, 1, nh, 1);
    ggml_tensor * x   = ggml_reshape_3d(c, add, hd, nh, T);
    ggml_tensor * mul = ggml_mul(c, x, A);
    ggml_tensor * s1  = ggml_scale(c, mul, -1.0f);
    ggml_tensor * sig = ggml_sigmoid(c, s1);
    ggml_tensor * s2  = ggml_scale(c, sig, -5.0f);
    ggml_tensor * out = mut == GM_NO_PERMUTE ? ggml_cont(c, s2)
                                              : ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, s2, hd, nh, T, 1), 0, 2, 1, 3));
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    if (mut == GM_EXTRA_READER) ggml_build_forward_expand(gf, ggml_add(c, s1, s1));
    if (mut == GM_OUTPUT_FLAG) sig->flags |= GGML_TENSOR_FLAG_OUTPUT;
    ggml_graph_compute_with_ctx(c, gf, 2);

    const int i = find_node(gf, add);
    CHECK(i >= 0, "%s: ADD not in graph", tag);
    pxa_glm_kda_gate_match m;
    const char * why = nullptr;
    const bool ok = i >= 0 && pxa_glm_kda_match_gate(gf, i, m, pxa_glm_kda_consumers_scan, &why);
    if (mut == GM_NONE || mut == GM_NO_PERMUTE) {
        CHECK(ok, "%s: no match (%s)", tag, why ? why : "?");
        if (ok) {
            CHECK(gf->nodes[m.last] == out, "%s: match does not end at the CONT", tag);
            verify_gate(m, (const float *) out->data, (const float *) s2->data, (const float *) g1->data,
                        (const float *) dtb->data, (const float *) sa->data, tag);
        }
    } else {
        CHECK(!ok, "%s: matched a mutated chain", tag);
        if (!ok) printf("  %s declined: %s\n", tag, why ? why : "?");
    }
    ggml_free(c);
}

enum out_mut { OM_NONE, OM_SWAPPED, OM_BROADCAST, OM_EXTRA_READER };

static void synth_outgate(int64_t hd, int64_t nh, int64_t T, out_mut mut) {
    char tag[96];
    snprintf(tag, sizeof(tag), "synth outgate hd=%lld nh=%lld T=%lld mut=%d", (long long) hd, (long long) nh, (long long) T, (int) mut);
    ggml_init_params ip = { 64u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    std::mt19937 rng(99 + (unsigned) (hd + nh*5 + T*11 + mut));
    ggml_tensor * xin = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, nh, T);
    ggml_tensor * w   = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd);
    ggml_tensor * g2  = mut == OM_BROADCAST ? ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, 1, T) : ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, nh, T);
    fill(xin, rng, -2, 2); fill(w, rng, 0.5f, 1.5f); fill(g2, rng, -6, 6);
    // build_glm5next.cpp output gate: o = rms_norm(.)*w; gated = o * sigmoid(g2); cont_2d
    ggml_tensor * o   = ggml_mul(c, ggml_rms_norm(c, xin, 1e-6f), w);
    ggml_tensor * sig = ggml_sigmoid(c, g2);
    ggml_tensor * mul = mut == OM_SWAPPED ? ggml_mul(c, sig, o) : ggml_mul(c, o, sig);
    ggml_tensor * out = ggml_cont_2d(c, mul, hd*nh, T);
    ggml_cgraph * gf = ggml_new_graph(c);
    // swapped operands: the graph must still visit o before the sigmoid (as build_glm5next's order does), else a
    // non-noop node sits between SIGMOID and MUL and the chain is (correctly) not contiguous
    if (mut == OM_SWAPPED) ggml_build_forward_expand(gf, o);
    ggml_build_forward_expand(gf, out);
    if (mut == OM_EXTRA_READER) ggml_build_forward_expand(gf, ggml_scale(c, sig, 2.0f));
    ggml_graph_compute_with_ctx(c, gf, 2);
    const int i = find_node(gf, sig);
    pxa_glm_kda_outgate_match m;
    const char * why = nullptr;
    const bool ok = i >= 0 && pxa_glm_kda_match_outgate(gf, i, m, pxa_glm_kda_consumers_scan, &why);
    if (mut == OM_NONE || mut == OM_SWAPPED) {
        CHECK(ok, "%s: no match (%s)", tag, why ? why : "?");
        if (ok) {
            CHECK(gf->nodes[m.last] == out, "%s: match does not end at the CONT", tag);
            verify_outgate(m, (const float *) out->data, (const float *) mul->data, (const float *) o->data, (const float *) g2->data, tag);
        }
    } else {
        CHECK(!ok, "%s: matched a mutated chain", tag);
        if (!ok) printf("  %s declined: %s\n", tag, why ? why : "?");
    }
    ggml_free(c);
}

// ---------------------------------------------------------------------------------------------- B) the real graph
struct rec_t {
    bool on = false;
    std::vector<ggml_tensor *> nodes;
    std::unordered_map<const ggml_tensor *, std::vector<float>> snap;   // node outputs at their own compute time
};

static void snap_tensor(rec_t & r, const ggml_tensor * t) {
    if (!t || !t->data || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || r.snap.count(t)) return;
    std::vector<float> v(ggml_nelements(t));
    memcpy(v.data(), t->data, v.size()*sizeof(float));
    r.snap.emplace(t, std::move(v));
}

static bool rec_cb(ggml_tensor * t, bool ask, void * ud) {
    rec_t & r = *(rec_t *) ud;
    if (!r.on) return false;
    if (ask) return true;                    // observe every node: each runs on its own, so its output is still intact here
    r.nodes.push_back(t);
    snap_tensor(r, t);
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = t->src[s];
        if (src && src->op == GGML_OP_NONE) snap_tensor(r, src);   // weights / inputs
    }
    return true;
}

static const float * snapped(rec_t & r, const ggml_tensor * t) {
    auto it = r.snap.find(t);
    return it == r.snap.end() ? nullptr : it->second.data();
}

static void scan_real_graph(rec_t & r, const char * what, int n_kda, int & n_gate_out, int & n_out_out) {
    ggml_cgraph g = {};
    g.n_nodes = (int) r.nodes.size();
    g.size    = g.n_nodes;
    g.nodes   = r.nodes.data();
    int n_gate = 0, n_out = 0;
    for (int i = 0; i < g.n_nodes; ++i) {
        const ggml_tensor * t = g.nodes[i];
        const char * why = nullptr;
        if (t->op == GGML_OP_ADD) {
            pxa_glm_kda_gate_match m;
            if (pxa_glm_kda_match_gate(&g, i, m, pxa_glm_kda_consumers_scan, &why)) {
                ++n_gate;
                const float * cont = snapped(r, m.cont), * s2 = snapped(r, m.s2), * gg = snapped(r, m.add->src[0]);
                const float * dtb = snapped(r, m.add->src[1]), * A = snapped(r, m.mul->src[1]);
                CHECK(cont && s2 && gg && dtb && A, "%s: gate match at %s without recorded values", what, t->name);
                if (cont && s2 && gg && dtb && A) verify_gate(m, cont, s2, gg, dtb, A, what);
            } else if (why && strstr(t->name, "kda")) {
                printf("  %s: ADD %s not matched: %s\n", what, t->name, why);
            }
        } else if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID) {
            pxa_glm_kda_outgate_match m;
            if (pxa_glm_kda_match_outgate(&g, i, m, pxa_glm_kda_consumers_scan, &why)) {
                ++n_out;
                const ggml_tensor * o = m.mul->src[0] == m.sig ? m.mul->src[1] : m.mul->src[0];
                const float * cont = snapped(r, m.cont), * mul = snapped(r, m.mul), * oo = snapped(r, o), * gg = snapped(r, m.sig->src[0]);
                CHECK(cont && mul && oo && gg, "%s: outgate match at %s without recorded values", what, t->name);
                if (cont && mul && oo && gg) verify_outgate(m, cont, mul, oo, gg, what);
            }
        }
    }
    printf("%s: %d nodes, gate chain matched %d (expect %d), out-gate matched %d (expect >= %d)\n",
           what, g.n_nodes, n_gate, n_kda, n_out, n_kda);
    CHECK(n_gate == n_kda, "%s: gate chain matched %d times, want %d", what, n_gate, n_kda);
    CHECK(n_out >= n_kda, "%s: out-gate matched %d times, want >= %d", what, n_out, n_kda);
    n_gate_out += n_gate; n_out_out += n_out;
}

static int real_model(const char * path, int n_kda) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }
    rec_t r;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 1;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.flash_attn = false; cp.mla_attn = 1;
    cp.cb_eval = rec_cb; cp.cb_eval_user_data = &r;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const int n_vocab = llama_n_vocab(model);
    const int n_prompt = 7;
    llama_batch b = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        b.token[i] = (llama_token) (5 + (i*37) % (n_vocab - 8)); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == n_prompt - 1;
    }
    b.n_tokens = n_prompt;
    int n_gate = 0, n_out = 0;
    r.on = true;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
    r.on = false;
    scan_real_graph(r, "prefill (7 tokens)", n_kda, n_gate, n_out);
    llama_pos pos = n_prompt;
    for (int step = 0; step < 3; ++step) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        for (int k = 1; k < n_vocab; ++k) if (lg[k] > lg[best]) best = k;
        llama_batch d = llama_batch_init(1, 0, 1);
        d.token[0] = best; d.pos[0] = pos++; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0; d.logits[0] = 1; d.n_tokens = 1;
        r.nodes.clear(); r.snap.clear();
        r.on = true;
        if (llama_decode(ctx, d) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        r.on = false;
        char what[48]; snprintf(what, sizeof(what), "decode step %d", step);
        scan_real_graph(r, what, n_kda, n_gate, n_out);
        llama_batch_free(d);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    CHECK(pxa_glm_kda_fuse_parse(nullptr) == 0 && pxa_glm_kda_fuse_parse("0") == 0 && pxa_glm_kda_fuse_parse("1") == 3 &&
          pxa_glm_kda_fuse_parse("2") == 2 && pxa_glm_kda_fuse_parse("3") == 3 && pxa_glm_kda_fuse_parse("-1") == 0,
          "env parse");
    for (int64_t T : { 1, 5 }) {
        synth_gate(16, 4, T, GM_NONE);
        synth_gate(128, 3, T, GM_NONE);
        synth_gate(16, 4, T, GM_NO_PERMUTE);
        synth_outgate(16, 4, T, OM_NONE);
        synth_outgate(32, 2, T, OM_SWAPPED);
    }
    synth_gate(16, 4, 3, GM_EXTRA_READER);
    synth_gate(16, 4, 3, GM_OUTPUT_FLAG);
    synth_gate(16, 4, 3, GM_A_NOT_PER_HEAD);
    synth_outgate(16, 4, 3, OM_BROADCAST);
    synth_outgate(16, 4, 3, OM_EXTRA_READER);
    if (argc > 1) {
        const int n_kda = argc > 2 ? atoi(argv[2]) : 6;
        if (real_model(argv[1], n_kda) != 0) ++g_fail;
    } else {
        printf("no tiny glm5next GGUF given: synthetic part only\n");
    }
    if (g_fail) { fprintf(stderr, "test-glm-kda-fuse-match: %d FAIL\n", g_fail); return 1; }
    printf("test-glm-kda-fuse-match: OK\n");
    return 0;
}
