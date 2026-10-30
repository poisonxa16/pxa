// pxa / PXA expert-cache cold path without host round trips -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache-async-core.cpp -- see llama-pxa-xcache-async.h for the design and llama-pxa-xcache-async-core.h for the interface.
#include "llama-pxa-xcache-async-core.h"

#include <cmath>
#include "ggml-pxqn-api.h"
#include "llama-impl.h"
#include "llama-pxa-thread-place.h"
// PXA_XCACHE_MISS_TOKEN (src/llama-pxa-xcache.cpp)
bool llama_pxa_tokmiss_on();
void llama_pxa_tokmiss_note_request(const int32_t * ids, int n_used, int ntok, uint64_t us);

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define PXA_XCA_RELAX() _mm_pause()
#else
#define PXA_XCA_RELAX() do { } while (0)
#endif

// PXA_XCACHE_IDLE_BACKOFF_MS (default 0 = off). An idle cold-path thread waits longer between checks.
// Within a request the path is unchanged. Nothing is computed differently.
// PXA_XCACHE_ASYNC_TIMEOUT_S (default 30): the slow-step line. The card waits for every cold layer however long the worker takes (a slow
// worker changes the time, never the output; ggml-cuda pxa-cold-dev.cuh); a step slower than this is logged (rate-limited) and counted.
static uint64_t pxa_xca_slow_us(void) {
    static const uint64_t v = [] { const char * e = getenv("PXA_XCACHE_ASYNC_TIMEOUT_S"); const double s = e && atof(e) > 0 ? atof(e) : 30.0;
                                   return (uint64_t) (s*1e6); }();
    return v;
}
static std::atomic<uint64_t> pxa_xca_n_slow{0};

static int64_t pxa_xca_idle_backoff_us(void) {
    static const int64_t v = [] { const char * e = getenv("PXA_XCACHE_IDLE_BACKOFF_MS"); const long n = e ? atol(e) : 0; return (int64_t) (n > 0 ? n : 0) * 1000; }();
    return v;
}

// one (layer, width) CPU graph: the nodes the scheduler's cold split would have run, over the slot's own host buffers
struct pxa_xca_sub {
    ggml_context *        ctx = nullptr;      // a shared arena of the core (not owned)
    ggml_cgraph *         gf  = nullptr;
    std::vector<uint8_t>  scratch;     // the intermediates (up, gate, activated)
    std::vector<uint8_t>  work;        // the CPU plan's work buffer
    ggml_cplan            plan;
    // PXA_XCACHE_LEAN: the same nodes, as the lean executor runs them (null when the sub-graph is not the up/gate/unary/down shape)
    ggml_tensor *         l_up = nullptr, * l_gate = nullptr, * l_unary = nullptr, * l_down = nullptr;
    bool                  lean_ok = false;
};

struct pxa_xca_layer {
    int                           il = -1, device = -1;
    struct ggml_cuda_cold_slot *  slot = nullptr;
    std::atomic<pxa_xca_sub *>    subs[PXA_XCA_MAX_TOK + 1];
    uint32_t                      seen = 0;                 // worker only
    uint64_t                      n_req = 0, us_total = 0, us_max = 0, n_err = 0;   // worker only (read after stop())
    pxa_xca_layer() { for (auto & s : subs) s.store(nullptr, std::memory_order_relaxed); }
};

// per-width CPU cold-path accounting (diagnostic, printed in the summary line)
static std::atomic<uint64_t> pxa_xca_w_req[PXA_XCA_MAX_TOK + 1], pxa_xca_w_us[PXA_XCA_MAX_TOK + 1];

static inline int64_t pxa_xca_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int pxa_xca_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

// PXA_XCACHE_PATHLOG unset or 0: off. "1" writes stderr. Any other value is a file path.
// One line per cold layer step: the experts the CPU computed. Experts absent from the
// list were not on this cold path (GPU hit, or no routing into that slot).
static FILE * pxa_xca_pathlog() {
    static FILE * f = []() -> FILE * {
        const char * e = getenv("PXA_XCACHE_PATHLOG");
        if (!e || !*e || std::strcmp(e, "0") == 0) return nullptr;
        if (std::strcmp(e, "1") == 0) return stderr;
        return std::fopen(e, "a");
    }();
    return f;
}

static void pxa_xca_pathlog_line(const pxa_xca_layer * L, uint32_t req, int ntok, bool lean, bool graph) {
    FILE * f = pxa_xca_pathlog();
    if (!f || !L || !L->slot || !L->slot->h_ids || ntok < 1) return;
    const char * path = lean ? "cpu-miss" : (graph ? "cpu-graph" : "none");
    std::fprintf(f, "layer=%d req=%u ntok=%d path=%s", L->il, req, ntok, path);
    const int n_used = L->slot->n_used;
    for (int t = 0; t < ntok; ++t) {
        std::fprintf(f, " t%d=", t);
        for (int k = 0; k < n_used; ++k) {
            const int32_t e = L->slot->h_ids[k + t*n_used];
            if (e >= 0) std::fprintf(f, "%d,", e);
        }
    }
    std::fputc('\n', f);
    std::fflush(f);
}


// PXA_XCACHE_CPUPOOL (default off). The library runs the step. No library: off.
bool pxa_xca_lean_wanted(void) {
    static const bool on = [] {
        const char * e = getenv("PXA_XCACHE_CPUPOOL");
        if (!(e && *e && atoi(e) != 0)) return false;
        if (ggml_pxqn_xcache_cpupool_create_get() != nullptr) return true;
        LLAMA_LOG_WARN("PXA_XCACHE: CPU pool was requested but the library is not loaded; staying off\n");
        return false;
    }();
    return on;
}

bool pxa_xca_core::lean_run(pxa_xca_sub * sb, struct ggml_cuda_cold_slot * slot, int ntok) {
    if (!sb || !sb->lean_ok || !lean_ || !slot) return false;
    ggml_pxqn_xcache_cpupool_step_fn step = ggml_pxqn_xcache_cpupool_step_get();
    if (!step) return false;
    return step(lean_, sb->l_up, sb->l_gate, sb->l_unary, sb->l_down, slot->h_ids, slot->n_used, ntok) == 0;
}

pxa_xca_core::pxa_xca_core(pxa_xca_slot_new_fn nf, pxa_xca_slot_free_fn ff, pxa_xca_slot_stats_fn sf)
    : new_fn_(nf), free_fn_(ff), stats_fn_(sf) {
    spin_us_ = std::max(0, pxa_xca_env_int("PXA_XCACHE_ASYNC_SPIN_US", 3000));
}

// bytes of ggml context one sub-graph needs (about ten tensors and a 32-node graph; generous)
static size_t pxa_xca_sub_mem() { return 24*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false) + 4096; }
constexpr int PXA_XCA_SUBS_PER_ARENA = 16;

struct ggml_context * pxa_xca_core::sub_ctx() {
    if (arenas_.empty() || arena_used_ >= PXA_XCA_SUBS_PER_ARENA) {
        ggml_init_params ip = { /*.mem_size =*/ pxa_xca_sub_mem()*PXA_XCA_SUBS_PER_ARENA, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ggml_context * c = ggml_init(ip);
        if (!c) return nullptr;
        arenas_.push_back(c);
        arena_used_ = 0;
    }
    ++arena_used_;
    return arenas_.back();
}

pxa_xca_core::~pxa_xca_core() {
    stop();
    if (lean_) {
        if (ggml_pxqn_xcache_cpupool_destroy_fn d = ggml_pxqn_xcache_cpupool_destroy_get()) d(lean_);
        lean_ = nullptr;
    }
    const int n = n_layers_.load();
    for (int i = 0; i < n; ++i) {
        pxa_xca_layer * L = layers_[i];
        for (auto & s : L->subs) delete s.load();
        if (L->slot && free_fn_) free_fn_(L->slot);
        delete L;
    }
    for (ggml_context * c : arenas_) ggml_free(c);
}

struct ggml_cuda_cold_slot * pxa_xca_core::slot(int il, int device, int n_embd, int n_used, int n_tok_max) {
    const int n = n_layers_.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        if (layers_[i]->il == il && layers_[i]->device == device) {
            return layers_[i]->slot;
        }
    }
    if (n >= PXA_XCA_MAX_LAYERS || !new_fn_) return nullptr;
    struct ggml_cuda_cold_slot * s = new_fn_(device, n_embd, n_used, n_tok_max);
    if (!s) return nullptr;
    pxa_xca_layer * L = new pxa_xca_layer();
    L->il = il; L->device = device; L->slot = s;
    layers_[n] = L;
    n_layers_.store(n + 1, std::memory_order_release);
    start();
    return s;
}

bool pxa_xca_core::add_width(struct ggml_cuda_cold_slot * slot, int n_tok, struct ggml_tensor * up, struct ggml_tensor * gate,
                             struct ggml_tensor * up_gate, struct ggml_tensor * down, int unary_op, int n_threads, float clamp_limit) {
    const bool gateless = unary_op == PXA_XCA_GATELESS_RELU || unary_op == PXA_XCA_GATELESS_RELU_SQR;
    if (gateless ? (!up || gate || up_gate || clamp_limit > 0.0f) : !(up_gate || (up && gate))) return false;
    if (!slot || n_tok < 1 || n_tok > PXA_XCA_MAX_TOK || n_tok > slot->n_tok_max || !down || n_threads < 1) return false;
    pxa_xca_layer * L = nullptr;
    const int n = n_layers_.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) if (layers_[i]->slot == slot) { L = layers_[i]; break; }
    if (!L) return false;
    if (L->subs[n_tok].load(std::memory_order_acquire)) return true;          // built already

    const int n_embd = slot->n_embd, n_used = slot->n_used;
    auto * sb = new pxa_xca_sub();
    sb->ctx = sub_ctx();
    if (!sb->ctx) { delete sb; return false; }
    ggml_context * ctx = sb->ctx;

    // the inputs ARE the slot's host buffers: what the GPU's submit kernel wrote is what the CPU nodes read
    ggml_tensor * cur_h = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tok);
    ggml_tensor * ids_h = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    cur_h->data = slot->h_cur;
    ids_h->data = slot->h_ids;
    ggml_set_name(cur_h, "xca_cur");
    ggml_set_name(ids_h, "xca_ids");

    // exactly the calls llm_build_moe_ffn makes for a cold stack the CPU owns: with host weights ggml_moe_up_gate decomposes to
    // mul_mat_id(up), mul_mat_id(gate), fused_mul_unary (and splits a merged up/gate stack into its halves), then the down matmul
    // A gate-less chain is build_experts' gate-less branch: mul_mat_id(up), relu, (sqr), mul_mat_id(down).
    ggml_tensor * par = nullptr;
    if (gateless) {
        par = ggml_relu(ctx, ggml_mul_mat_id(ctx, up, cur_h, ids_h));
        if (unary_op == PXA_XCA_GATELESS_RELU_SQR) par = ggml_sqr(ctx, par);
    } else if (clamp_limit > 0.0f) {
        // PXA_XCACHE_ASYNC_CLAMP: the DeepSeek-V4 / GLM-5.x clamped SwiGLU, node for node the dsv4_clamp branch of build_experts
        // (llama-build-context.cpp): up clamped to [-limit, limit], gate to [-inf, limit], then silu(gate) * up. No fused form exists.
        if (up_gate || !up || !gate) { delete sb; return false; }
        ggml_tensor * u = ggml_mul_mat_id(ctx, up,   cur_h, ids_h);
        ggml_tensor * g = ggml_mul_mat_id(ctx, gate, cur_h, ids_h);
        u   = ggml_clamp(ctx, u, -clamp_limit, clamp_limit);
        g   = ggml_clamp(ctx, g, -INFINITY,    clamp_limit);
        par = ggml_swiglu_split(ctx, g, u);
    } else {
        par = up_gate ? ggml_moe_up_gate(ctx, up_gate, nullptr, cur_h, ids_h, (enum ggml_unary_op) unary_op)
                      : ggml_moe_up_gate(ctx, up, gate, cur_h, ids_h, (enum ggml_unary_op) unary_op);
    }
    if (par->op == GGML_OP_MOE_FUSED_UP_GATE || down->ne[0] != par->ne[0]) {      // a fused node has no CPU kernel; a K-straddled down needs the pad
        delete sb;
        return false;
    }
    ggml_tensor * out = ggml_mul_mat_id(ctx, down, par, ids_h);
    if (out->ne[0] != n_embd || out->ne[1] != n_used || out->ne[2] != n_tok) { delete sb; return false; }
    ggml_set_name(out, "xca_out");
    sb->gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(sb->gf, out);

    size_t off = 0;
    for (int i = 0; i < sb->gf->n_nodes; ++i) {
        ggml_tensor * t = sb->gf->nodes[i];
        if (t == out || t->view_src) continue;
        off += GGML_PAD(ggml_nbytes(t), 64);
    }
    sb->scratch.assign(off + 64, 0);
    uint8_t * base = (uint8_t *) (((uintptr_t) sb->scratch.data() + 63) & ~(uintptr_t) 63);
    off = 0;
    for (int i = 0; i < sb->gf->n_nodes; ++i) {
        ggml_tensor * t = sb->gf->nodes[i];
        if (t == out) { t->data = slot->h_out; continue; }
        if (t->view_src) continue;
        t->data = base + off;
        off += GGML_PAD(ggml_nbytes(t), 64);
    }
    // in-place nodes are views of their input (ggml_clamp in the clamped chain): they were created before their source had storage,
    // so point them at it now (a view of a weight recomputes the address it already has)
    for (int i = 0; i < sb->gf->n_nodes; ++i) {
        ggml_tensor * t = sb->gf->nodes[i];
        if (t != out && t->view_src && t->view_src->data) t->data = (char *) t->view_src->data + t->view_offs;
    }
    sb->plan = ggml_graph_plan(sb->gf, n_threads);
    sb->work.assign(sb->plan.work_size + 64, 0);
    sb->plan.work_data = (uint8_t *) (((uintptr_t) sb->work.data() + 63) & ~(uintptr_t) 63);

    if (pxa_xca_lean_wanted()) {
        // the shape the pool accepts
        std::vector<ggml_tensor *> mm; ggml_tensor * un = nullptr; bool other = false;
        ggml_tensor * g_relu = nullptr, * g_sqr = nullptr;      // gate-less chain: the activation nodes
        for (int i = 0; i < sb->gf->n_nodes; ++i) {
            ggml_tensor * t = sb->gf->nodes[i];
            switch (t->op) {
                case GGML_OP_MUL_MAT_ID:      mm.push_back(t); break;
                case GGML_OP_FUSED_MUL_UNARY: if (un) other = true; un = t; break;
                case GGML_OP_UNARY:           if (!gateless || g_relu || ggml_get_unary_op(t) != GGML_UNARY_OP_RELU) other = true; g_relu = t; break;
                case GGML_OP_SQR:             if (!gateless || g_sqr) other = true; g_sqr = t; break;
                case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: break;
                default: other = true; break;
            }
        }
        // gate-less: up -> relu (-> sqr) -> down, the activation computed by the pool's lean relu (the library says it has one)
        ggml_tensor * g_act = g_sqr ? g_sqr : g_relu;
        if (gateless && !other && g_relu && mm.size() == 2 && mm[1] == out && mm[1]->src[1] == g_act && g_relu->src[0] == mm[0] &&
            (!g_sqr || g_sqr->src[0] == g_relu) && (g_sqr != nullptr) == (unary_op == PXA_XCA_GATELESS_RELU_SQR) &&
            mm[0]->src[2] == ids_h && mm[1]->src[2] == ids_h && ggml_pxa_lean_mmid_ok(mm[0]) && ggml_pxa_lean_mmid_ok(mm[1]) &&
            mm[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(mm[0]) && ggml_is_contiguous(g_act) && ggml_pxqn_xcache_cpupool_gateless_get() != nullptr) {
            sb->l_up = mm[0]; sb->l_gate = nullptr; sb->l_unary = g_act; sb->l_down = mm[1];
            sb->lean_ok = true;
        }
        if (!gateless && !other && un && mm.size() == 3 && mm[2] == out && mm[2]->src[1] == un && mm[0]->src[2] == ids_h && mm[1]->src[2] == ids_h &&
            mm[2]->src[2] == ids_h && ggml_pxa_lean_mmid_ok(mm[0]) && ggml_pxa_lean_mmid_ok(mm[1]) && ggml_pxa_lean_mmid_ok(mm[2]) &&
            ((un->src[0] == mm[1] && un->src[1] == mm[0]) || (un->src[0] == mm[0] && un->src[1] == mm[1])) &&
            un->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_1(un->src[0])) {
            sb->l_gate = un->src[0]; sb->l_up = un->src[1]; sb->l_unary = un; sb->l_down = mm[2];
            sb->lean_ok = true;
        }
        if (sb->lean_ok) {
            if (!lean_) {
                ggml_pxqn_xcache_cpupool_create_fn create = ggml_pxqn_xcache_cpupool_create_get();
                if (create) {
                    pxa_thread_place_scope unpinned;      // PXA_THREAD_PLACE: workers beyond the CPU list must not inherit the decode core
                    lean_ = create(n_threads, spin_us_);
                }
                if (!lean_) sb->lean_ok = false;
                else pool_n_.store(pxa_thread_place_pool_threads(n_threads), std::memory_order_release);
            }
        }
    }

    L->subs[n_tok].store(sb, std::memory_order_release);
    return true;
}

void pxa_xca_core::start() {
    if (started_) return;
    started_ = true;
    stop_.store(false);
    th_ = std::thread([this] { run(); });
    // PXA_XCACHE_HANGDUMP_S=N (diagnostic, default off): every N s, from host memory only (no CUDA call, so it still works while the
    // compute stream is wedged), print the worker's current layer and every layer whose request / done / seen flags disagree.
    static const int dump_s = [] { const char * e = getenv("PXA_XCACHE_HANGDUMP_S"); return e ? atoi(e) : 0; }();
    if (dump_s > 0 && !dump_th_.joinable()) {
        dump_th_ = std::thread([this] {
            uint64_t last_req = 0;
            while (!stop_.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::seconds(dump_s));
                const int n = n_layers_.load(std::memory_order_acquire);
                uint64_t tot = 0;
                for (int i = 0; i < n; ++i) tot += layers_[i]->n_req;
                const int il = cur_il_.load();
                const int64_t t0 = cur_t0_.load();
                fprintf(stderr, "PXA_XCACHE_HANGDUMP: served %llu (+%llu), worker %s il %d for %lld us\n", (unsigned long long) tot,
                        (unsigned long long) (tot - last_req), il >= 0 ? "busy" : "idle", il, il >= 0 ? (long long) (pxa_xca_now_us() - t0) : 0LL);
                int shown = 0;
                for (int i = 0; i < n && shown < 12; ++i) {
                    pxa_xca_layer * L = layers_[i];
                    const uint64_t r = *L->slot->h_req; const uint32_t d = *L->slot->h_done;
                    if ((uint32_t) r != d || (uint32_t) r != L->seen) {
                        fprintf(stderr, "PXA_XCACHE_HANGDUMP:   il %d req %u ntok %u done %u seen %u n_req %llu err %llu\n", L->il, (unsigned) (uint32_t) r,
                                (unsigned) (r >> 32), (unsigned) d, (unsigned) L->seen, (unsigned long long) L->n_req, (unsigned long long) L->n_err);
                        ++shown;
                    }
                }
                last_req = tot;
            }
        });
    }
}

void pxa_xca_core::stop() {
    if (!started_) return;
    stop_.store(true, std::memory_order_release);
    if (th_.joinable()) th_.join();
    if (dump_th_.joinable()) dump_th_.join();
    started_ = false;
}

void pxa_xca_core::serve(pxa_xca_layer * L, uint64_t req) {
    const uint32_t q = (uint32_t) req;
    const int ntok = (int) (req >> 32);
    std::atomic_thread_fence(std::memory_order_acquire);   // the ids and the activation the flag publishes
    pxa_xca_sub * sb = (ntok >= 1 && ntok <= PXA_XCA_MAX_TOK) ? L->subs[ntok].load(std::memory_order_acquire) : nullptr;
    const int64_t t0 = pxa_xca_now_us();
    cur_t0_.store(t0); cur_il_.store(L->il);
    // The pool's first step can fail while it starts. Falling through to the
    // graph for that one request makes the first completion a different text
    // from every later one, which uses the pool. Retry the pool, then the graph.
    bool lean_ran = false;
    if (sb && sb->lean_ok) {
        for (int k = 0; k < 8 && !lean_ran; ++k) {
            lean_ran = lean_run(sb, L->slot, ntok);
            if (!lean_ran) std::this_thread::sleep_for(std::chrono::microseconds(250));
        }
        if (lean_ran) ++n_lean_;
    }
    bool graph_ran = false;
    if (!lean_ran && sb) {
        graph_ran = true;
        // N1: never publish zero rows. The card would merge them as this layer's cold-expert output and the text would change
        // silently, so a failed sub-graph or a width nobody registered stops the process, loudly, before "done" is written.
        const ggml_status st = ggml_graph_compute(sb->gf, &sb->plan);
        if (st != GGML_STATUS_SUCCESS)
            GGML_ABORT("PXA_XCACHE_ASYNC: layer %d: the cold experts' CPU sub-graph FAILED (width %d, ggml status %d); stopping instead of "
                       "giving the card zero rows (the output would be wrong)", L->il, ntok, (int) st);
    } else if (!sb) {
        GGML_ABORT("PXA_XCACHE_ASYNC: layer %d: no cold sub-graph registered for width %d (request %u); stopping instead of giving the card "
                   "zero rows (the output would be wrong)", L->il, ntok, (unsigned) q);
    }
    const uint64_t us = (uint64_t) (pxa_xca_now_us() - t0);
    if (us > pxa_xca_slow_us()) {
        const uint64_t k = pxa_xca_n_slow.fetch_add(1) + 1;
        if (k <= 3 || k % 100 == 0)
            LLAMA_LOG_WARN("PXA_XCACHE_ASYNC: layer %d: a cold-expert CPU step took %.1f s (> PXA_XCACHE_ASYNC_TIMEOUT_S, slow step #%llu): the card "
                           "waited for it, the output is unchanged; other jobs on the worker's CPUs slow decode down\n",
                           L->il, (double) us/1e6, (unsigned long long) k);
    }
    if (llama_pxa_tokmiss_on() && ntok >= 1 && ntok <= L->slot->n_tok_max) llama_pxa_tokmiss_note_request(L->slot->h_ids, L->slot->n_used, ntok, us);
    pxa_xca_pathlog_line(L, q, ntok, lean_ran, graph_ran);
    // Attempt 2. seq_cst orders ordinary stores. The matmul also uses streaming
    // stores, which an MFENCE does not order. Load both ends of the payload so
    // those lines have left the store buffer, SFENCE, then publish the flag.
    if (ntok >= 1 && ntok <= L->slot->n_tok_max && L->slot->h_out) {
        const size_t nfloat = (size_t) L->slot->n_embd * (size_t) L->slot->n_used * (size_t) ntok;
        if (nfloat > 0) {
            volatile float * out = L->slot->h_out;
            const float head = out[0];
            const float tail = out[nfloat - 1];
            (void) head; (void) tail;
        }
    }
#if defined(__x86_64__) || defined(__i386__)
    _mm_sfence();
#endif
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *L->slot->h_done = q;
#if defined(__x86_64__) || defined(__i386__)
    _mm_sfence();
#endif
    L->seen = q;
    cur_il_.store(-1);
    ++L->n_req; L->us_total += us; L->us_max = std::max(L->us_max, us);
    if (ntok >= 1 && ntok <= PXA_XCA_MAX_TOK) {
        pxa_xca_w_req[ntok].fetch_add(1, std::memory_order_relaxed); pxa_xca_w_us[ntok].fetch_add(us, std::memory_order_relaxed);
    }
}

void pxa_xca_core::run() {
#if defined(__linux__)
    prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);               // a sleeping worker wakes within a few microseconds of its timer
#endif
    int64_t last = pxa_xca_now_us();
    unsigned spins = 0;
    int placed_n = 0;                                     // PXA_THREAD_PLACE: the pool width this thread pinned itself for
    while (!stop_.load(std::memory_order_acquire)) {
        bool any = false;
        const int n = n_layers_.load(std::memory_order_acquire);
        for (int i = 0; i < n; ++i) {
            pxa_xca_layer * L = layers_[i];
            const uint64_t r = *L->slot->h_req;
            if ((uint32_t) r != L->seen) {
                serve(L, r);
                any = true;
            }
        }
        if (any) { last = pxa_xca_now_us(); spins = 0; continue; }
        PXA_XCA_RELAX();
        if ((++spins & 63) == 0) {
            const int pn = pool_n_.load(std::memory_order_acquire);
            if (pn != placed_n) { placed_n = pn; pxa_thread_place_leader(pn); }     // a no-op unless PXA_THREAD_PLACE=1
            const int64_t idle = pxa_xca_now_us() - last;
            if (idle > spin_us_) {
                const int64_t bo = pxa_xca_idle_backoff_us();
                std::this_thread::sleep_for(std::chrono::microseconds(bo > 0 && idle > bo ? 1000 : 20));
            }
        }
    }
}

int pxa_xca_core::serve_pending() {
    int n_served = 0;
    const int n = n_layers_.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        pxa_xca_layer * L = layers_[i];
        const uint64_t r = *L->slot->h_req;
        if ((uint32_t) r != L->seen) { serve(L, r); ++n_served; }
    }
    return n_served;
}

pxa_xca_totals pxa_xca_core::totals(bool with_gpu) {
    pxa_xca_totals t;
    const int n = n_layers_.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        pxa_xca_layer * L = layers_[i];
        t.requests += L->n_req; t.compute_us += L->us_total;
        if (i == 0) t.lean = n_lean_;
        t.max_us = std::max(t.max_us, L->us_max); t.errors += L->n_err;
        if (with_gpu && stats_fn_) {
            unsigned long long st[GGML_CUDA_COLD_STAT_N] = {};
            double khz = 1e6;
            if (stats_fn_(L->slot, st, &khz) && khz > 0) {
                t.gpu_waits += st[GGML_CUDA_COLD_STAT_WAITS];
                t.gpu_skipped += st[GGML_CUDA_COLD_STAT_SKIPPED];
                t.gpu_timeouts += st[GGML_CUDA_COLD_STAT_TIMEOUTS];
                t.gpu_wait_us += (double) st[GGML_CUDA_COLD_STAT_WAIT_CLK]*1000.0/khz;
                t.gpu_wait_max_us = std::max(t.gpu_wait_max_us, (double) st[GGML_CUDA_COLD_STAT_MAX_CLK]*1000.0/khz);
                t.chk_rows += st[GGML_CUDA_COLD_STAT_CHK_ROWS];
                t.chk_bad  += st[GGML_CUDA_COLD_STAT_CHK_BAD];
                { unsigned u = (unsigned) st[GGML_CUDA_COLD_STAT_CHK_MAXD]; float f; memcpy(&f, &u, 4); t.chk_maxd = std::max(t.chk_maxd, f); }
                if (st[GGML_CUDA_COLD_STAT_CHK_BAD]) {
                    ++t.chk_layers_bad; if (t.chk_first_bad_il < 0) t.chk_first_bad_il = L->il;
                    char b[64]; snprintf(b, sizeof(b), " %d:%llu/%llu", L->il, (unsigned long long) st[GGML_CUDA_COLD_STAT_CHK_BAD], (unsigned long long) st[GGML_CUDA_COLD_STAT_CHK_ROWS]);
                    t.chk_bad_layers += b;
                }
            }
        }
    }
    return t;
}

std::string pxa_xca_core::summary_line() {
    stop();
    const pxa_xca_totals t = totals(true);
    char buf[768];
    const double req = (double) std::max<uint64_t>(1, t.requests);
    const double wts = (double) std::max<uint64_t>(1, t.gpu_waits);
    snprintf(buf, sizeof(buf),
        "PXA_XCACHE_ASYNC: %llu cold-layer requests (%llu layer-steps had no cold expert); CPU sub-graph %.0f us avg / %llu us max; "
        "GPU waited %.0f us avg / %.0f us max per layer-step (%llu waits); errors %llu, slow waits (> PXA_XCACHE_ASYNC_TIMEOUT_S, waited out, output unchanged) %llu; lean %llu",
        (unsigned long long) t.requests, (unsigned long long) t.gpu_skipped, (double) t.compute_us/req, (unsigned long long) t.max_us,
        t.gpu_wait_us/wts, t.gpu_wait_max_us, (unsigned long long) t.gpu_waits, (unsigned long long) t.errors, (unsigned long long) t.gpu_timeouts,
        (unsigned long long) t.lean);
    std::string out = buf;
    out += "; CPU sub-graph by width (w:requests/avg us):";
    for (int w = 1; w <= PXA_XCA_MAX_TOK; ++w) {
        const uint64_t n = pxa_xca_w_req[w].load(), u = pxa_xca_w_us[w].load();
        if (n) { snprintf(buf, sizeof(buf), " %d:%llu/%.0f", w, (unsigned long long) n, (double) u/n); out += buf; }
    }
    if (t.chk_rows) {
        snprintf(buf, sizeof(buf), "; CHECK vs the scheduler split: %llu rows compared, %llu differ (%d layers), largest |difference| %.3g; layers with a difference (layer:bad/compared):",
                 (unsigned long long) t.chk_rows, (unsigned long long) t.chk_bad, t.chk_layers_bad, (double) t.chk_maxd);
        out += buf;
        out += t.chk_bad_layers;
    }
    return out;
}
