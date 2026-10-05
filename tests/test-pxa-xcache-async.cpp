// test-pxa-xcache-async.cpp -- PXA_XCACHE_ASYNC: the cold path's host worker is bit-identical to the plain CPU graph.
//
// The lever hands the cold half of a split expert layer to a worker thread that runs the layer's CPU sub-graph (up, gate, unary, down
// over the cold stack) on the slot's host buffers. This test builds the same cold stack twice over (weights shared), runs
//   reference : the same nodes as one plain ggml graph on the CPU (what the scheduler's cold CPU split runs), and
//   candidate : the worker of src/llama-pxa-xcache-async-core.cpp, driven through the slot protocol
// for hundreds of random (width, ids, activation) cases, and compares the output tensors BYTE for byte (memcmp, every row, hot-less ids
// included). Two drivers:
//   host  (default)  the "GPU" is the main thread: it writes ids / activation / request flag into the slot exactly as the submit kernel
//                    does and waits on the done flag as the wait kernel does -- no CUDA needed
//   --gpu N          the real submit / wait kernels on CUDA device N, in a ggml graph (submit -> a hot stand-in -> wait -> merge) run
//                    by the CUDA backend, compared with the same reference
// Weights: random Q8_0 (up, gate) and Q4_0 (down), or with --gguf FILE --layer L the first --experts experts of a real file's layer
// (any type the CPU can read, PXQN included when libggml-pxqn.so is found), so the closed CPU kernels run too.
//
// build: tests/run-xcache-async-test.sh [--gpu N] [--gguf FILE --layer L]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda-xcache.h"
#include "llama-pxa-xcache-async-core.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

// ---- host slots: plain aligned memory behind the same struct the CUDA backend fills ----------------------------------------------
static struct ggml_cuda_cold_slot * host_slot_new(int dev, int n_embd, int n_used, int n_tok_max) {
    const size_t line = 128;
    auto up = [&](size_t n) { return (n + line - 1)/line*line; };
    const size_t o_ids = 2*line, o_cur = o_ids + up((size_t) n_used*n_tok_max*4), o_out = o_cur + up((size_t) n_embd*n_tok_max*4);
    const size_t total = o_out + up((size_t) n_embd*n_used*n_tok_max*4);
    char * h = (char *) aligned_alloc(4096, (total + 4095)/4096*4096);
    memset(h, 0, total);
    auto * s = new ggml_cuda_cold_slot();
    memset(s, 0, sizeof(*s));
    s->device = dev; s->n_embd = n_embd; s->n_used = n_used; s->n_tok_max = n_tok_max;
    s->h_req = s->d_req = (volatile uint64_t *) h;
    s->h_done = s->d_done = (volatile uint32_t *) (h + line);
    s->h_ids = s->d_ids = (int32_t *) (h + o_ids);
    s->h_cur = s->d_cur = (float *) (h + o_cur);
    s->h_out = s->d_out = (float *) (h + o_out);
    return s;
}
static void host_slot_free(struct ggml_cuda_cold_slot * s) { if (s) { free((void *) s->h_req); delete s; } }

// ---- the cold stack the CPU reads ---------------------------------------------------------------------------------------------------
struct stack_t {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *up = nullptr, *gate = nullptr, *down = nullptr, *up_gate = nullptr;
    bool merged = false;   // the layer's up/gate are ONE stack (rows [0, n_ff) gate, [n_ff, 2 n_ff) up): up_gate is used, up / gate are null
    int n_embd = 0, n_ff = 0, n_slots = 0;
};

static void quantize_rows(ggml_type type, ggml_tensor * t, std::mt19937 & rng, float scale) {
    const int64_t nrows = t->ne[1]*t->ne[2], k = t->ne[0];
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> row((size_t) k);
    std::vector<uint8_t> q(ggml_row_size(type, k));
    for (int64_t r = 0; r < nrows; ++r) {
        for (auto & v : row) v = nd(rng);
        ggml_quantize_chunk(type, row.data(), q.data(), 0, 1, k, nullptr, nullptr);
        ggml_backend_tensor_set(t, q.data(), (size_t) r*q.size(), q.size());
    }
}

static bool make_synthetic_stack(stack_t & st, int n_embd, int n_ff, int n_slots, std::mt19937 & rng, bool merged) {
    st.n_embd = n_embd; st.n_ff = n_ff; st.n_slots = n_slots; st.merged = merged;
    ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
    st.ctx = ggml_init(ip);
    if (merged) {
        st.up_gate = ggml_new_tensor_3d(st.ctx, GGML_TYPE_Q8_0, n_embd, 2*n_ff, n_slots);
        ggml_set_name(st.up_gate, "t.ffn_up_gate_exps.xc");
    } else {
        st.up   = ggml_new_tensor_3d(st.ctx, GGML_TYPE_Q8_0, n_embd, n_ff, n_slots);
        st.gate = ggml_new_tensor_3d(st.ctx, GGML_TYPE_Q8_0, n_embd, n_ff, n_slots);
        ggml_set_name(st.up, "t.ffn_up_exps.xc"); ggml_set_name(st.gate, "t.ffn_gate_exps.xc");
    }
    st.down = ggml_new_tensor_3d(st.ctx, GGML_TYPE_Q4_0, n_ff, n_embd, n_slots);
    ggml_set_name(st.down, "t.ffn_down_exps.xc");
    st.buf = ggml_backend_alloc_ctx_tensors_from_buft(st.ctx, ggml_backend_cpu_buffer_type());
    if (!st.buf) return false;
    if (merged) {
        quantize_rows(GGML_TYPE_Q8_0, st.up_gate, rng, 0.05f);
    } else {
        quantize_rows(GGML_TYPE_Q8_0, st.up, rng, 0.05f);
        quantize_rows(GGML_TYPE_Q8_0, st.gate, rng, 0.05f);
    }
    quantize_rows(GGML_TYPE_Q4_0, st.down, rng, 0.05f);
    return true;
}

// the first n_exp experts of a real file's layer (type and shape as the file has them): reads nb[2]*n_exp bytes of each expert tensor
static bool make_real_stack(stack_t & st, const char * path, int layer, int n_exp) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "cannot open %s\n", path); return false; }
    const size_t data_off = gguf_get_data_offset(g);
    char nm[160];
    ggml_tensor * m[3] = {};
    const char * kinds[3] = { "ffn_up_exps", "ffn_gate_exps", "ffn_down_exps" };
    for (int i = 0; i < 3; ++i) {
        snprintf(nm, sizeof(nm), "blk.%d.%s.weight", layer, kinds[i]);
        m[i] = ggml_get_tensor(meta, nm);
        if (!m[i]) { fprintf(stderr, "tensor %s not in the file\n", nm); gguf_free(g); return false; }
        if (m[i]->ne[2] < n_exp) { fprintf(stderr, "%s has %lld experts\n", nm, (long long) m[i]->ne[2]); gguf_free(g); return false; }
    }
    st.n_embd = (int) m[0]->ne[0]; st.n_ff = (int) m[0]->ne[1]; st.n_slots = n_exp;
    ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
    st.ctx = ggml_init(ip);
    ggml_tensor ** dst[3] = { &st.up, &st.gate, &st.down };
    for (int i = 0; i < 3; ++i) {
        *dst[i] = ggml_new_tensor_3d(st.ctx, m[i]->type, m[i]->ne[0], m[i]->ne[1], n_exp);
        for (int d = 1; d < 3; ++d) (*dst[i])->nb[d] = m[i]->nb[d];
        snprintf(nm, sizeof(nm), "blk.%d.%s.weight.xc", layer, kinds[i]);
        ggml_set_name(*dst[i], nm);
    }
    st.buf = ggml_backend_alloc_ctx_tensors_from_buft(st.ctx, ggml_backend_cpu_buffer_type());
    if (!st.buf) { gguf_free(g); return false; }
    FILE * f = fopen(path, "rb");
    bool ok = f != nullptr;
    for (int i = 0; ok && i < 3; ++i) {
        snprintf(nm, sizeof(nm), "blk.%d.%s.weight", layer, kinds[i]);
        const size_t off = data_off + gguf_get_tensor_offset(g, gguf_find_tensor(g, nm));
        const size_t n = (size_t) m[i]->nb[2]*n_exp;
        std::vector<uint8_t> buf(n);
        ok = fseeko(f, (off_t) off, SEEK_SET) == 0 && fread(buf.data(), 1, n, f) == n;
        if (ok) ggml_backend_tensor_set(*dst[i], buf.data(), 0, n);
    }
    if (f) fclose(f);
    printf("real stack: layer %d, %d experts, n_embd %d n_ff %d, types up %s gate %s down %s\n", layer, n_exp, st.n_embd, st.n_ff,
           ggml_type_name(st.up->type), ggml_type_name(st.gate->type), ggml_type_name(st.down->type));
    gguf_free(g);
    return ok;
}

// ---- reference: the same nodes as one plain CPU graph ------------------------------------------------------------------------------
struct ref_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *cur = nullptr, *ids = nullptr, *out = nullptr;
    std::vector<uint8_t> mem;
    int nth = 4;
};
static void make_ref(ref_t & r, const stack_t & st, int n_used, int n_tok, int nth) {
    r.nth = nth;
    r.mem.assign(64u << 20, 0);
    ggml_init_params ip = { r.mem.size(), r.mem.data(), false };
    r.ctx = ggml_init(ip);
    r.cur = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, st.n_embd, 1, n_tok);
    r.ids = ggml_new_tensor_2d(r.ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * par = nullptr;
    if (getenv("XCA_LITERAL") && !st.merged) {
        // the literal else-branch of llm_build_moe_ffn's build_experts (separate up / gate / unary nodes), not through ggml_moe_up_gate
        ggml_tensor * up   = ggml_mul_mat_id(r.ctx, st.up,   r.cur, r.ids);
        ggml_tensor * gate = ggml_mul_mat_id(r.ctx, st.gate, r.cur, r.ids);
        par = ggml_fused_mul_unary(r.ctx, gate, up, GGML_UNARY_OP_SILU);
    } else {
        par = st.merged ? ggml_moe_up_gate(r.ctx, st.up_gate, nullptr, r.cur, r.ids, GGML_UNARY_OP_SILU)
                        : ggml_moe_up_gate(r.ctx, st.up, st.gate, r.cur, r.ids, GGML_UNARY_OP_SILU);
    }
    r.out = ggml_mul_mat_id(r.ctx, st.down, par, r.ids);
    r.gf = ggml_new_graph(r.ctx);
    ggml_build_forward_expand(r.gf, r.out);
}

static void make_case(std::mt19937 & rng, int n_embd, int n_used, int n_tok, int n_slots, std::vector<int32_t> & ids, std::vector<float> & cur, float p_cold) {
    ids.assign((size_t) n_used*n_tok, -1);
    cur.resize((size_t) n_embd*n_tok);
    std::normal_distribution<float> nd(0.0f, 0.7f);
    for (auto & v : cur) v = nd(rng);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    for (int t = 0; t < n_tok; ++t) {
        std::vector<int> perm(n_slots);
        for (int i = 0; i < n_slots; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        int nxt = 0;
        for (int k = 0; k < n_used; ++k) if (ud(rng) < p_cold) ids[(size_t) k + (size_t) t*n_used] = perm[nxt++];
    }
}

static bool any_cold(const std::vector<int32_t> & ids) { for (int32_t e : ids) if (e >= 0) return true; return false; }

// ---- host driver: this thread plays the GPU ----------------------------------------------------------------------------------------
static int run_host(pxa_xca_core & core, const stack_t & st, int n_used, int n_cases, int nth, int n_layers, int ref_nth) {
    std::mt19937 rng(1234);
    const int max_tok = 8;
    std::vector<ggml_cuda_cold_slot *> slots;
    ggml_cuda_cold_slot * s0 = nullptr;
    for (int l = 0; l < n_layers; ++l) {
        ggml_cuda_cold_slot * s = core.slot(l, 0, st.n_embd, n_used, max_tok);
        CHECK(s != nullptr, "slot %d", l);
        if (!s) return 1;
        slots.push_back(s);
        for (int w = 1; w <= max_tok; ++w) CHECK(core.add_width(s, w, st.up, st.gate, st.up_gate, st.down, (int) GGML_UNARY_OP_SILU, nth), "add_width %d", w);
        if (l == 0) s0 = s;
    }
    (void) s0;
    std::vector<ref_t> refs(max_tok + 1);
    for (int w = 1; w <= max_tok; ++w) make_ref(refs[w], st, n_used, w, ref_nth);
    std::vector<uint32_t> seq(n_layers, 0);
    int n_bad = 0, n_skipped = 0, n_ran = 0;
    size_t n_nonzero = 0;
    for (int c = 0; c < n_cases; ++c) {
        const int l = c % n_layers;
        const int w = 1 + (c*7 + c/3) % max_tok;
        std::vector<int32_t> ids; std::vector<float> cur;
        make_case(rng, st.n_embd, n_used, w, st.n_slots, ids, cur, (c % 5 == 0) ? 0.05f : (c % 5 == 1 ? 1.0f : 0.5f));
        ggml_cuda_cold_slot * s = slots[l];
        // submit
        memcpy(s->h_ids, ids.data(), ids.size()*4);
        memcpy(s->h_cur, cur.data(), cur.size()*4);
        const uint32_t q = ++seq[l];
        if (any_cold(ids)) {
            std::atomic_thread_fence(std::memory_order_release);
            *s->h_req = (uint64_t) q | ((uint64_t) w << 32);
            // wait
            const auto t0 = std::chrono::steady_clock::now();
            while (*s->h_done != q) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) { CHECK(false, "case %d timed out", c); return 1; }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            ++n_ran;
        } else { *s->h_done = q; ++n_skipped; continue; }
        // reference
        ref_t & r = refs[w];
        memcpy(r.ids->data, ids.data(), ids.size()*4);
        memcpy(r.cur->data, cur.data(), cur.size()*4);
        ggml_graph_compute_with_ctx(r.ctx, r.gf, ref_nth);
        const size_t nb = (size_t) st.n_embd*n_used*w*4;
        for (size_t i = 0; i < nb/4; ++i) n_nonzero += ((const float *) s->h_out)[i] != 0.0f;
        if (memcmp(r.out->data, s->h_out, nb) != 0) {
            ++n_bad;
            if (n_bad <= 3) {
                size_t first = 0; while (first < nb && ((const char *) r.out->data)[first] == ((const char *) s->h_out)[first]) ++first;
                fprintf(stderr, "  case %d (layer %d, width %d): first differing byte %zu of %zu\n", c, l, w, first, nb);
            }
        }
    }
    printf("host driver: %d cases (%d with cold experts, %d without) over %d layers, widths 1..%d, worker %d CPU threads, reference %d: %d mismatching outputs\n",
           n_cases, n_ran, n_skipped, n_layers, max_tok, nth, ref_nth, n_bad);
    printf("  (%zu non-zero output floats compared)\n", n_nonzero);
    CHECK(n_nonzero > 1000, "the compared outputs are all zero");
    CHECK(n_bad == 0, "%d outputs differ from the plain CPU graph", n_bad);
    return n_bad;
}

#ifdef GGML_USE_CUDA
// ---- GPU driver: the real kernels in a ggml graph on the CUDA backend ----------------------------------------------------------------
static int run_gpu(pxa_xca_core & core, const stack_t & st, int n_used, int n_cases, int nth, int dev) {
    ggml_backend_t be = ggml_backend_cuda_init(dev, nullptr);
    if (!be) { fprintf(stderr, "no CUDA device %d\n", dev); return 1; }
    std::mt19937 rng(777);
    const int max_tok = 8;
    ggml_cuda_cold_slot * slot = core.slot(0, dev, st.n_embd, n_used, max_tok);
    CHECK(slot != nullptr, "cuda slot");
    if (!slot) return 1;
    for (int w = 1; w <= max_tok; ++w) CHECK(core.add_width(slot, w, st.up, st.gate, st.up_gate, st.down, (int) GGML_UNARY_OP_SILU, nth), "add_width %d", w);
    std::vector<ref_t> refs(max_tok + 1);
    for (int w = 1; w <= max_tok; ++w) make_ref(refs[w], st, n_used, w, nth);

    struct gw_t { ggml_context * ctx; ggml_cgraph * gf; ggml_backend_buffer_t buf; ggml_tensor *cur, *ids_cold, *ids_hot, *hot, *merged; };
    std::vector<gw_t> gs(max_tok + 1);
    for (int w = 1; w <= max_tok; ++w) {
        gw_t & g = gs[w];
        ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        g.ctx = ggml_init(ip);
        g.cur      = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, st.n_embd, 1, w);
        g.ids_cold = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_used, w);
        g.ids_hot  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_used, w);
        ggml_set_input(g.cur); ggml_set_input(g.ids_cold); ggml_set_input(g.ids_hot);
        ggml_tensor * ticket = ggml_moe_cold_submit(g.ctx, g.cur, g.ids_cold, slot);
        ggml_tensor * rep = ggml_repeat(g.ctx, g.cur, ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, st.n_embd, n_used, w));
        g.hot = ggml_scale(g.ctx, rep, 2.0f);                    // stands in for the hot experts: GPU work between submit and wait
        ggml_tensor * cold = ggml_moe_cold_wait(g.ctx, ticket, g.ids_cold, st.n_embd, slot);
        g.merged = ggml_moe_merge(g.ctx, g.hot, cold, g.ids_hot, g.ids_cold);
        g.gf = ggml_new_graph(g.ctx);
        ggml_build_forward_expand(g.gf, ticket);                 // the engine orders it with ids_hot->src[2] = ticket; here by expansion order
        ggml_build_forward_expand(g.gf, g.hot);
        ggml_build_forward_expand(g.gf, cold);
        ggml_build_forward_expand(g.gf, g.merged);
        g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
        CHECK(g.buf != nullptr, "alloc width %d", w);
        if (!g.buf) return 1;
    }
    int n_bad = 0, n_skipped = 0, n_ran = 0;
    double us_total = 0;
    for (int c = 0; c < n_cases; ++c) {
        const int w = 1 + (c*7 + c/3) % max_tok;
        std::vector<int32_t> ids; std::vector<float> cur;
        make_case(rng, st.n_embd, n_used, w, st.n_slots, ids, cur, (c % 5 == 0) ? 0.05f : (c % 5 == 1 ? 1.0f : 0.5f));
        // the hot ids: every slot the cold side does not hold is "served by the hot stack" (hot id 0), the others -1
        std::vector<int32_t> ids_hot(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) ids_hot[i] = ids[i] >= 0 ? -1 : ((c + (int) i) % 3 == 0 ? -1 : 0);
        gw_t & g = gs[w];
        ggml_backend_tensor_set(g.cur, cur.data(), 0, cur.size()*4);
        ggml_backend_tensor_set(g.ids_cold, ids.data(), 0, ids.size()*4);
        ggml_backend_tensor_set(g.ids_hot, ids_hot.data(), 0, ids_hot.size()*4);
        const auto t0 = std::chrono::steady_clock::now();
        ggml_status stt = ggml_backend_graph_compute(be, g.gf);
        CHECK(stt == GGML_STATUS_SUCCESS, "graph compute");
        us_total += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> got((size_t) st.n_embd*n_used*w);
        ggml_backend_tensor_get(g.merged, got.data(), 0, got.size()*4);
        // expected: hot rows 2*cur, cold rows from the plain CPU graph, neither: zero
        ref_t & r = refs[w];
        memcpy(r.ids->data, ids.data(), ids.size()*4);
        memcpy(r.cur->data, cur.data(), cur.size()*4);
        const bool cold_any = any_cold(ids);
        if (cold_any) { ggml_graph_compute_with_ctx(r.ctx, r.gf, nth); ++n_ran; } else ++n_skipped;
        int bad_rows = 0;
        for (int t = 0; t < w; ++t) for (int k = 0; k < n_used; ++k) {
            const size_t i = (size_t) k + (size_t) t*n_used;
            const float * row = got.data() + i*st.n_embd;
            std::vector<float> want(st.n_embd, 0.0f);
            if (ids_hot[i] >= 0) for (int j = 0; j < st.n_embd; ++j) want[j] = 2.0f*cur[(size_t) t*st.n_embd + j];
            else if (ids[i] >= 0) memcpy(want.data(), (const float *) r.out->data + i*st.n_embd, st.n_embd*4);
            if (memcmp(row, want.data(), st.n_embd*4) != 0) ++bad_rows;
        }
        if (bad_rows) { ++n_bad; if (n_bad <= 3) fprintf(stderr, "  case %d (width %d): %d rows differ\n", c, w, bad_rows); }
    }
    printf("gpu driver (device %d): %d cases (%d with cold experts, %d without), widths 1..%d: %d mismatching outputs; %.0f us per graph (submit + hot stand-in + wait + merge + the host handshake, includes graph launch and sync)\n",
           dev, n_cases, n_ran, n_skipped, max_tok, n_bad, us_total/n_cases);
    unsigned long long stt[GGML_CUDA_COLD_STAT_N] = {};
    if (ggml_backend_cuda_cold_slot_stats(slot, stt)) {
        const double khz = ggml_backend_cuda_cold_clock_khz(dev);
        printf("gpu counters: %llu waits, %llu no-cold submits, %llu timeouts, wait %.1f us avg / %.1f us max\n", stt[GGML_CUDA_COLD_STAT_WAITS],
               stt[GGML_CUDA_COLD_STAT_SKIPPED], stt[GGML_CUDA_COLD_STAT_TIMEOUTS],
               stt[GGML_CUDA_COLD_STAT_WAITS] ? (double) stt[GGML_CUDA_COLD_STAT_WAIT_CLK]*1000.0/khz/stt[GGML_CUDA_COLD_STAT_WAITS] : 0.0,
               (double) stt[GGML_CUDA_COLD_STAT_MAX_CLK]*1000.0/khz);
        CHECK(stt[GGML_CUDA_COLD_STAT_TIMEOUTS] == 0, "wait timeouts");
        CHECK(stt[GGML_CUDA_COLD_STAT_WAITS] == (unsigned long long) n_cases, "wait count %llu != %d", stt[GGML_CUDA_COLD_STAT_WAITS], n_cases);
        CHECK(stt[GGML_CUDA_COLD_STAT_SKIPPED] == (unsigned long long) n_skipped, "skipped %llu != %d", stt[GGML_CUDA_COLD_STAT_SKIPPED], n_skipped);
    }
    CHECK(n_bad == 0, "%d outputs differ", n_bad);
    for (int w = 1; w <= max_tok; ++w) { ggml_backend_buffer_free(gs[w].buf); ggml_free(gs[w].ctx); }
    ggml_backend_free(be);
    return n_bad;
}
#endif

int main(int argc, char ** argv) {
    bool merged = false;
    int ref_nth = 0;
    int gpu = -1, n_cases = 600, nth = 4, n_layers = 3, real_layer = -1, real_experts = 24, n_used = 10;
    const char * gguf_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cases") && i + 1 < argc) n_cases = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) n_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gguf") && i + 1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) real_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) real_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--used") && i + 1 < argc) n_used = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--merged")) merged = true;
        else if (!strcmp(argv[i], "--ref-threads") && i + 1 < argc) ref_nth = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--gpu N] [--cases N] [--threads N] [--layers N] [--gguf FILE --layer L [--experts N] [--used K]] [--merged]\n", argv[0]); return 2; }
    }
    std::mt19937 rng(42);
    stack_t st;
    if (gguf_path) {
        if (!make_real_stack(st, gguf_path, real_layer < 0 ? 3 : real_layer, real_experts)) { fprintf(stderr, "real stack failed\n"); return 1; }
        n_used = std::min(n_used, st.n_slots);
    } else if (!make_synthetic_stack(st, 512, 256, 64, rng, merged)) { fprintf(stderr, "stack alloc failed\n"); return 1; }
    int rc = 0;
    if (gpu < 0) {
        pxa_xca_core core(host_slot_new, host_slot_free);
        rc = run_host(core, st, n_used, n_cases, nth, n_layers, ref_nth > 0 ? ref_nth : nth);
        core.stop();
        const pxa_xca_totals t = core.totals(false);
        printf("worker: %llu requests, %.0f us avg per request, %llu us max, %llu errors\n", (unsigned long long) t.requests,
               (double) t.compute_us/std::max<uint64_t>(1, t.requests), (unsigned long long) t.max_us, (unsigned long long) t.errors);
        CHECK(t.errors == 0, "worker errors");
    } else {
#ifdef GGML_USE_CUDA
        pxa_xca_core core(ggml_backend_cuda_cold_slot_new, ggml_backend_cuda_cold_slot_free, nullptr);
        rc = run_gpu(core, st, n_used, n_cases, nth, gpu);
        core.stop();
        const pxa_xca_totals t = core.totals(false);
        printf("worker: %llu requests, %.0f us avg per request, %llu us max, %llu errors\n", (unsigned long long) t.requests,
               (double) t.compute_us/std::max<uint64_t>(1, t.requests), (unsigned long long) t.max_us, (unsigned long long) t.errors);
        CHECK(t.errors == 0, "worker errors");
#else
        fprintf(stderr, "built without CUDA\n"); return 2;
#endif
    }
    ggml_backend_buffer_free(st.buf);
    ggml_free(st.ctx);
    printf("%s\n", (g_fail || rc) ? "TEST FAILED" : "TEST PASSED");
    return (g_fail || rc) ? 1 : 0;
}
