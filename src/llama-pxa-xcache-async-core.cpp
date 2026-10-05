// pxa / PXA expert-cache cold path without host round trips -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache-async-core.cpp -- see llama-pxa-xcache-async.h for the design and llama-pxa-xcache-async-core.h for the interface.
#include "llama-pxa-xcache-async-core.h"

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

// one (layer, width) CPU graph: the nodes the scheduler's cold split would have run, over the slot's own host buffers
struct pxa_xca_sub {
    ggml_context *        ctx = nullptr;      // a shared arena of the core (not owned)
    ggml_cgraph *         gf  = nullptr;
    std::vector<uint8_t>  scratch;     // the intermediates (up, gate, activated)
    std::vector<uint8_t>  work;        // the CPU plan's work buffer
    ggml_cplan            plan;
};

struct pxa_xca_layer {
    int                           il = -1, device = -1;
    struct ggml_cuda_cold_slot *  slot = nullptr;
    std::atomic<pxa_xca_sub *>    subs[PXA_XCA_MAX_TOK + 1];
    uint32_t                      seen = 0;                 // worker only
    uint64_t                      n_req = 0, us_total = 0, us_max = 0, n_err = 0;   // worker only (read after stop())
    pxa_xca_layer() { for (auto & s : subs) s.store(nullptr, std::memory_order_relaxed); }
};

static inline int64_t pxa_xca_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int pxa_xca_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
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
                             struct ggml_tensor * up_gate, struct ggml_tensor * down, int unary_op, int n_threads) {
    if (!slot || n_tok < 1 || n_tok > PXA_XCA_MAX_TOK || n_tok > slot->n_tok_max || !down || !(up_gate || (up && gate)) || n_threads < 1) return false;
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
    ggml_tensor * par = up_gate ? ggml_moe_up_gate(ctx, up_gate, nullptr, cur_h, ids_h, (enum ggml_unary_op) unary_op)
                                : ggml_moe_up_gate(ctx, up, gate, cur_h, ids_h, (enum ggml_unary_op) unary_op);
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
    sb->plan = ggml_graph_plan(sb->gf, n_threads);
    sb->work.assign(sb->plan.work_size + 64, 0);
    sb->plan.work_data = (uint8_t *) (((uintptr_t) sb->work.data() + 63) & ~(uintptr_t) 63);

    L->subs[n_tok].store(sb, std::memory_order_release);
    return true;
}

void pxa_xca_core::start() {
    if (started_) return;
    started_ = true;
    stop_.store(false);
    th_ = std::thread([this] { run(); });
}

void pxa_xca_core::stop() {
    if (!started_) return;
    stop_.store(true, std::memory_order_release);
    if (th_.joinable()) th_.join();
    started_ = false;
}

void pxa_xca_core::serve(pxa_xca_layer * L, uint64_t req) {
    const uint32_t q = (uint32_t) req;
    const int ntok = (int) (req >> 32);
    std::atomic_thread_fence(std::memory_order_acquire);   // the ids and the activation the flag publishes
    pxa_xca_sub * sb = (ntok >= 1 && ntok <= PXA_XCA_MAX_TOK) ? L->subs[ntok].load(std::memory_order_acquire) : nullptr;
    const int64_t t0 = pxa_xca_now_us();
    if (sb) {
        if (ggml_graph_compute(sb->gf, &sb->plan) != GGML_STATUS_SUCCESS) {
            ++L->n_err;
            memset(L->slot->h_out, 0, (size_t) L->slot->n_embd*L->slot->n_used*ntok*sizeof(float));
        }
    } else {
        ++L->n_err;                                        // a width nobody registered: rows of this layer are zero, the GPU is never left waiting
        const size_t n = (size_t) L->slot->n_embd*L->slot->n_used*(size_t) std::max(1, std::min(ntok, L->slot->n_tok_max));
        memset(L->slot->h_out, 0, n*sizeof(float));
    }
    const uint64_t us = (uint64_t) (pxa_xca_now_us() - t0);
    std::atomic_thread_fence(std::memory_order_release);   // the result is visible before the flag that announces it
    *L->slot->h_done = q;
    L->seen = q;
    ++L->n_req; L->us_total += us; L->us_max = std::max(L->us_max, us);
}

void pxa_xca_core::run() {
#if defined(__linux__)
    prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);               // a sleeping worker wakes within a few microseconds of its timer
#endif
    int64_t last = pxa_xca_now_us();
    unsigned spins = 0;
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
        if ((++spins & 63) == 0 && pxa_xca_now_us() - last > spin_us_) {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
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
        t.requests += L->n_req; t.compute_us += L->us_total; t.max_us = std::max(t.max_us, L->us_max); t.errors += L->n_err;
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
        "GPU waited %.0f us avg / %.0f us max per layer-step (%llu waits); errors %llu, timeouts %llu",
        (unsigned long long) t.requests, (unsigned long long) t.gpu_skipped, (double) t.compute_us/req, (unsigned long long) t.max_us,
        t.gpu_wait_us/wts, t.gpu_wait_max_us, (unsigned long long) t.gpu_waits, (unsigned long long) t.errors, (unsigned long long) t.gpu_timeouts);
    std::string out = buf;
    if (t.chk_rows) {
        snprintf(buf, sizeof(buf), "; CHECK vs the scheduler split: %llu rows compared, %llu differ (%d layers), largest |difference| %.3g; layers with a difference (layer:bad/compared):",
                 (unsigned long long) t.chk_rows, (unsigned long long) t.chk_bad, t.chk_layers_bad, (double) t.chk_maxd);
        out += buf;
        out += t.chk_bad_layers;
    }
    return out;
}
