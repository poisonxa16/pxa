// pxa / PXA expert-cache online adaptation -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache.cpp -- see llama-pxa-xcache.h for the design. The policy (what to swap, how many per step) is the closed
// library's (ggml_pxqn_xcache_policy); this file is the mechanism: counter read-back, the two-phase slot copy, the map flips.
#include "ggml-pxqn-tune.h"
#include "llama-pxa-counts.h"
#include "ggml-pxqn-ext.h"
#include "llama-pxa-tokmiss.h"
#include "llama-pxa-xcache.h"
#include "llama.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-pxqn-api.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-xcache.h"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <deque>
#include <memory>
#include <sys/stat.h>
#ifdef _WIN32
#include "pxa-win-compat.h"
#else
#include <unistd.h>
#endif
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef GGML_USE_CUDA

bool llama_pxa_xcache_residency_blocks(void) { return false; }
void llama_pxa_xcache_tick(llama_model &, int) {}
void llama_pxa_xcache_seq_start(llama_model &) {}
int llama_pxa_xcache_batch_start(llama_model &, int, bool, void (*)(void *), void *) { return 0; }
bool llama_pxa_xcache_companion_quiet(void) { return false; }
void llama_pxa_xcache_note_tokens(llama_model &, int) {}
void llama_pxa_xcache_adaptor_free(llama_model &) {}
bool llama_pxa_xcache_stats(const struct llama_model *, uint64_t *, uint64_t *, uint64_t *, uint64_t *, uint64_t *) { return false; }
bool llama_pxa_xcache_rtier_stats(const llama_model *, uint64_t *, uint64_t *) { return false; }
bool llama_pxa_xcache_evict_file_stats(const llama_model *, uint64_t *, uint64_t *) { return false; }
bool llama_pxa_xcache_role_stats(const llama_model *, llama_pxa_xc_role_stats *) { return false; }
int llama_pxa_xcache_check(const llama_model *) { return -1; }
bool llama_pxa_tokmiss_on() { return false; }
void llama_pxa_tokmiss_note_request(const int32_t *, int, int, uint64_t) {}
void llama_pxa_tokmiss_note_batch(const llama_token *, int) {}
int llama_xcache_draft_keep(const llama_token *, int n, int) { return n; }
float llama_xcache_token_slots(llama_token) { return -1.f; }
int llama_xcache_draft_miss_us(const llama_token *, int) { return 0; }   // no expert cache without CUDA: no draft miss bill

#else

// a residency group (hot swap) is active: its weights are mirrored to host RAM once and restored at unpark, so a cache that rewrites the
// hot stacks and the routing maps while the model runs would come back wrong. The active group can only be read through bind(): put
// back what was taken out (load time and the first tick are single-threaded in the thread that owns the model)
bool llama_pxa_xcache_residency_blocks(void) {
    void * prev = ggml_backend_cuda_residency_bind(nullptr);
    if (prev) ggml_backend_cuda_residency_bind(prev);
    return prev != nullptr;
}

namespace {

constexpr size_t PXA_XC_COPY_CHUNK = 512u << 10;   // a decode's own small transfers never wait behind more than one chunk (~0.16 ms at x4)

int env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

}

// the engine's view of the library's book of this session's learned counts (the "learn" role, ggml-pxqn-ext.h)
struct pxa_learn_handle {
    const ggml_pxqn_learn_api * t = nullptr;
    void * book = nullptr;
    std::string path;           // the learned csv this session writes; empty = it writes nothing
    double keep = 0;            // the share of the boot's learned counts each save carries
    double prior_tokens = 0;    // the tokens those counts were counted from
    bool armed() const { return !path.empty() && t && book; }
    bool counted() const { return armed() && t->counted && t->counted(book) != 0; }
    void stop() const { if (armed() && t->stop) t->stop(book); }
    // write_csv(path) writes the rows; the sessions on record after the save, -1 = nothing saved
    int save(const std::function<bool(const std::string &)> & write_csv, double tokens, bool shutdown) {
        if (!armed()) return -1;
        struct ud_t { const std::function<bool(const std::string &)> * w; } ud = { &write_csv };
        return t->save(book, [](void * u, const char * p) { return (*((ud_t *) u)->w)(std::string(p)) ? 1 : 0; }, &ud, tokens,
                       shutdown ? 1 : 0);
    }
    ~pxa_learn_handle() { if (t && book && t->release) t->release(book); }
};

// One-time probe of what the CPU can stream out of the pinned cold stack: the cold path's matvec is bound by this (GB/s of weight bytes).
// `threads` readers sum their slice of up to 256 MiB twice; the second pass is timed. Reported, and used for the miss price log only.
static double pxa_xc_probe_cpu_gbs(const void * base, size_t bytes, int threads) {
    const size_t n = std::min<size_t>(bytes, 256u << 20) & ~(size_t) 63;
    if (n < (8u << 20) || threads < 1) return 0.0;
    std::vector<uint64_t> sink((size_t) threads, 0);
    auto pass = [&]() {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) {
            th.emplace_back([&, t]() {
                const uint64_t * p = (const uint64_t *) base + ((n/8)*(size_t) t)/(size_t) threads;
                const uint64_t * e = (const uint64_t *) base + ((n/8)*(size_t) (t + 1))/(size_t) threads;
                uint64_t a = 0, b = 0, c = 0, d = 0;
                for (; p + 4 <= e; p += 4) { a += p[0]; b += p[1]; c += p[2]; d += p[3]; }
                sink[(size_t) t] = a + b + c + d;
            });
        }
        for (auto & x : th) x.join();
    };
    pass();                                        // first pass: page in / warm the TLB
    const int64_t t0 = ggml_time_us();
    pass();
    const double us = (double) (ggml_time_us() - t0);
    uint64_t keep = 0;
    for (uint64_t v : sink) keep ^= v;
    if (keep == 0x9e3779b97f4a7c15ull) fputc(0, stderr);   // the sums are used
    return us > 0 ? (double) n/(us*1e3) : 0.0;
}

// The directory we will write. A container whose cache is on its own disk loses the file on remove.
static void pxa_warn_learned_cache(const std::string & path) {
    if (path.empty()) return;
    std::string dir = path;
    const size_t sl = dir.find_last_of('/');
    if (sl == std::string::npos) dir = ".";
    else if (sl == 0) dir = "/";
    else dir = dir.substr(0, sl);
    if (access(dir.c_str(), W_OK) != 0) {
        LLAMA_LOG_WARN("PXA_XCACHE: learned counts cannot be saved. %s is not writable, so they are lost when this process ends.\n", dir.c_str());
        return;
    }
    if (access("/.dockerenv", F_OK) != 0) return;
    struct stat st_dir, st_root;
    if (stat(dir.c_str(), &st_dir) != 0 || stat("/", &st_root) != 0) return;
    if (st_dir.st_dev == st_root.st_dev) {
        LLAMA_LOG_WARN("PXA_XCACHE: learned counts in %s will be deleted when this container is removed. They are on the container's own disk. Mount a volume on %s and set PXA_CACHE_DIR to that path.\n", dir.c_str(), dir.c_str());
    }
}

// llama.cpp. Empty when the learned csv has nowhere writable. Never the curated path.
bool llama_pxa_xcache_learned_path(const llama_model * model, char * buf, size_t n);
// llama.cpp. The cache role's state file: the first existing one among the learned counts' places, named with suffix (false: none).
bool llama_pxa_xcache_role_path(const llama_model * model, const char * suffix, char * buf, size_t n);

struct llama_model::pxa_xc_adaptor {
    struct lay {
        int il = -1, dev = 0, n_expert = 0;
        ggml_tensor * mapT = nullptr;
        std::vector<ggml_tensor *> hotT, coldT;   // parallel: one per expert tensor kind (up / gate / down, or gate_up / down)
        size_t eb = 0;                            // bytes of one expert over all its tensors
        std::vector<int32_t> map;                 // host mirror of the device map: >= 0 hot slot, <= -2 cold slot
        std::vector<uint8_t> busy;                // an in-flight swap involves this expert
        std::vector<int32_t> spare;               // free COLD slots (pinned host RAM)
        size_t stage_off = 0;                     // offset (in u32) of this layer's counter block in the pinned staging buffer
        bool dirty = false;                       // map changed this tick, flush to the device
        // PXA_XCACHE_EVICT_FILE: where each tensor kind's expert 0 sits in the model file (the stacks hold the file's raw slices), so
        // an evicted expert's bytes are read back from the file instead of copied over the link; empty = this layer evicts by copy
        std::vector<size_t> foff;
        std::vector<int> ffd;
    };
    // one swap = evict e_out (hot slot s -> a free cold slot), then admit e_in (its cold slot -> the hot slot s that freed up)
    struct op {
        int li = 0, e_in = 0, e_out = 0;
        int s = -1;            // hot slot of e_out, then the slot e_in lands in
        int c_free = -1;       // free cold slot e_out is copied into
        int c_in = -1;         // cold slot e_in is copied from; free again once it is admitted
        int phase = 0;         // 0 evict copy in flight (hot -> cold), 1 admit copy in flight (cold -> hot)
        uint64_t due = 0;      // tick at which the current phase's flip is applied (earliest, in non-deterministic mode)
        void * ev = nullptr;
        std::shared_ptr<std::atomic<int>> fl;   // PXA_XCACHE_EVICT_FILE phase 0: the file read (0 pending, 1 done, -1 failed)
    };

    llama_model & model;
    const ggml_pxqn_xcache_policy * pol = nullptr;
    void * pst = nullptr;
    bool active = false;
    bool adapt = false;      // false = observe only (hit-rate statistics, no swaps)
    bool broken = false;     // a copy could not be enqueued: no new swaps
    bool det = true;
    int  obs_every = 4;
    int  delay = 2;
    int  log_every = 0;
    int  n_used = 0;
    FILE * trace = nullptr;    // PXA_XCACHE_TRACE=<file>: the cumulative routing counters of every decode (offline replay of policies, tests/replay-xcache-trace.cpp)
    bool   trace_hdr = false;
    std::string hold_flag;     // PXA_XCACHE_ADAPT_FLAG=<file>: while the file exists and starts with '0' no new swap is started (A/B inside one boot)
    int  chaos = 0;            // PXA_XCACHE_ADAPT_CHAOS=n: n random swaps per observation instead of the policy (reproducibility / numerics test)
    uint64_t chaos_rng = 0x9e3779b97f4a7c15ull;
    uint64_t tick = 0;
    uint32_t tokens_since_obs = 0;
    int  ticks_since_obs = 0;
    std::vector<lay> L;
    std::vector<op>  ops;
    std::vector<ggml_pxqn_xc_layer_cfg> lcfg;
    std::vector<ggml_pxqn_xc_obs>  obs;
    std::vector<ggml_pxqn_xc_view> views;
    std::vector<ggml_pxqn_xc_swap> plan_out;
    std::vector<std::vector<double>> prior;
    uint32_t * stage = nullptr;
    size_t stage_u32 = 0;
    std::vector<int> devs;
    double pcie_gbs = 0;
    uint64_t n_obs = 0, n_planned = 0, n_started = 0, n_admitted = 0, n_evicted = 0, n_waits = 0;
    int64_t  us_tick = 0, us_obs = 0, n_ticks = 0;     // host time the adaptation itself costs the decode thread
    uint64_t n_skip_ticks = 0;                          // companion decodes on the quiet path (llama_pxa_xcache_companion_quiet)
    std::vector<uint32_t> prev_hit, prev_miss;       // per layer: counters at the last observation
    std::vector<uint64_t> lay_hit, lay_miss;         // per layer: routings served by the hot stack / sent to the cold path since the start
    uint64_t tot_hit = 0, tot_miss = 0;               // routings served by the hot stack / that went to the cold stack since the start
    uint64_t win_hit = 0, win_miss = 0;               // the same since the last log line
    std::mutex mu;
    // PXA_XCACHE_ADAPT_RESET (default on): every sequence that starts at position 0 starts from the planned hot set, a fresh policy
    // state and zero routing counts, so the same request gives the same answer whatever was served before it (the hot GPU kernel and
    // the CPU cold path round differently, so WHICH experts are hot changes the floats). =0: the hot set carries across requests.
    bool reset_on = true;
    std::vector<op> applied;                          // completed swaps since the last reset, undone in reverse order at the next one
    std::vector<std::vector<int32_t>> init_map, init_spare, init_hot, init_cold;
    std::vector<std::vector<uint32_t>> base_cnt;      // per layer: device counters (n_expert + 8) at the last reset
    std::vector<std::vector<uint32_t>> rel_cnt;       // per layer: counts since the last reset, what the policy observes
    ggml_pxqn_xc_cfg cfg_saved = {};
    uint64_t n_resets = 0, n_undone = 0;
    int64_t  us_reset = 0;
    // PXA_XCACHE_RTIER (default off): the adaptation off the decode thread. The routing counters are copied
    // asynchronously at the observation tick and a planner thread runs the policy (observe + plan) while the card runs the next
    // decodes; its swaps start at a later tick. At most PXA_XCACHE_SWAP_CAP swaps per decode (default 2 here) and the copy-completion
    // schedule (no decode waits for a copy). Meant for RIX (rank-identical experts: the card and the CPU compute the same bits), where
    // WHEN an expert moves cannot change an answer; without RIX the answers depend on copy timing and a warning says so.
    bool rtier = false;
    int  swap_cap = 0;               // swaps per decode tick (0 = the policy's own budget only)
    std::thread rt_th;
    std::mutex rt_mu;
    std::condition_variable rt_cv;
    bool rt_stop = false, rt_job = false, rt_done = false;
    uint32_t * rt_stage[2] = { nullptr, nullptr };
    int  rt_buf = 0;
    void * rt_ev = nullptr;
    uint32_t rt_ntok = 0;
    std::vector<std::vector<int32_t>> rt_map;         // snapshot of the maps the plan is made against
    std::vector<std::vector<uint8_t>> rt_busy;
    std::vector<int32_t> rt_spare;
    std::vector<ggml_pxqn_xc_view> rt_views;
    std::vector<ggml_pxqn_xc_swap> rt_out;
    int32_t rt_nout = 0;
    uint64_t n_rt_jobs = 0, n_rt_skip = 0, n_rt_stale = 0;
    // PXA_XCACHE_SWAP_DUMMY (diagnostic, default off; od-b5 (e) swap-cost split, 2026-10-29): every planned swap moves its
    // bytes exactly like a real one (the evict: file read or hot -> free cold slot; the admit: that cold slot -> the hot slot over the
    // link) but the map never flips: the admit writes the evicted expert's own bytes back over themselves, so no routing, map upload
    // or hit rate changes and the answer stays bit-identical. Real-swap cost = copies (this arm vs SWAP_CAP small) + map flips/routing.
    // The plan keeps proposing the same swaps (its snapshot never moves), so this is the copy cost at the full cap.
    bool swap_dummy = env_int("PXA_XCACHE_SWAP_DUMMY", 0) != 0;
    uint64_t n_dummy = 0;
    // PXA_XCACHE_PREFILL_FREEZE (llama-pxa-xcache.h): after a batch wider than freeze.maxw no flip and no swap start
    llama_pxa_xc_freeze freeze;
    uint64_t n_frozen_ticks = 0, n_frozen_drop = 0;

    // ---- PXA_XCACHE_EVICT_FILE: evict = a host read of the expert's bytes from the model file into the free cold slot -------------
    // (an IO thread; the link carries only the admit). Every slot holds the file's raw bytes (pxa_xc_fill copies slices; swaps copy
    // whole slots), so the cold slot ends up byte-equal to what the D2H copy would have written.
    struct io_job { std::vector<int> fd; std::vector<size_t> off, n; std::vector<char *> dst; std::shared_ptr<std::atomic<int>> fl; };
    bool evict_file = false;
    std::thread io_th;
    std::mutex io_mu;
    std::condition_variable io_cv, io_done_cv;
    std::deque<io_job> io_q;
    bool io_stop = false;
    uint64_t n_ef = 0;
    std::atomic<uint64_t> n_ef_fail{0}, us_ef{0}, n_ef_done{0};
    int64_t  us_rt_worker = 0;
    // this session's learned counts: the library's "learn" role (ggml-pxqn-ext.h) keeps their book; armed = this session writes
    // <model>.expert-counts.learned.csv (and the role its record beside it); learn.keep / learn.prior_tokens: the share of the
    // boot's learned counts each save carries, and the tokens they were counted from
    pxa_learn_handle learn;
    bool learn_saved = false;       // one log line for the periodic save
    bool learn_wait_logged = false;

    // ---- the "cache" role (ggml-pxqn-ext.h): request-scoped state of the expert cache; the library decides, this adaptor reads
    // the counters and makes the moves. R-tier + RIX only (the hot set carried across requests; with RIX an expert's placement
    // never changes a bit). A move set is made in one synchronous burst (the reset_to_plan pattern), then the policy is
    // re-created at that point with the prior the role gives (role_reseed).
    const ggml_pxqn_cache_api * role = nullptr;
    void * role_st = nullptr;
    bool   role_on = false;
    bool   role_rebased = false;     // after a move set the policy observes the counts since base_cnt (it was re-created there)
    std::vector<std::vector<uint32_t>> role_sel;              // per layer: the counters at the running request's select
    std::vector<ggml_pxqn_xc_layer_cfg> role_lcfg;            // the re-created policy's layers and their prior
    std::vector<std::vector<double>>    role_prior;
    std::condition_variable rt_idle_cv;                       // rt_loop finished a job (a move set waits for it)
    uint64_t n_role_waits = 0, n_role_drop_plans = 0, n_role_drop = 0;
    int64_t  us_role = 0, us_role_moves = 0;
    bool     role_saved = false, role_save_warned = false;

    explicit pxa_xc_adaptor(llama_model & m) : model(m) {}

    ~pxa_xc_adaptor() {
        if (trace) fclose(trace);
        rt_shutdown();
        drain();
        io_shutdown();
        if (learn.armed()) persist_learned(true);
        if (pst && pol) pol->destroy(pst);
        pst = nullptr;
        if (role && role_st) role->destroy(role_st);
        role_st = nullptr;
        if (stage) ggml_backend_cuda_xcache_pinned_free(stage);
        for (auto * b : rt_stage) if (b) ggml_backend_cuda_xcache_pinned_free(b);
    }

    bool init() {
        // PXA_XCACHE_ADAPT: unset / 0 off (static plan) | observe (hit-rate statistics only) | 1 adapt when the library has the
        // policy. Off by default (main, 2026-10-04): adaptation moves experts between the CPU q8 path and the GPU kernel, so a
        // greedy answer depends on the requests before it, and the measured gain (+4% prose / +8% code on the 32 GB Flash-Next,
        // one P100) sits inside this box's load noise. The expert cache's ship rule is "beats the fallback AND stays exact".
        const char * am = getenv("PXA_XCACHE_ADAPT");
        const bool mode_obs = am && strcmp(am, "observe") == 0;
        const bool mode_off = !mode_obs && !(am && strcmp(am, "1") == 0);
        log_every = env_int("PXA_XCACHE_LOG", 0);
        if (mode_off) return false;
        if (llama_pxa_xcache_residency_blocks()) return false;      // hot swap restores the weights from a one-time mirror (see llama.cpp)
        pol = ggml_pxqn_xcache_policy_get();
        adapt = pol && !mode_obs;
        if (!adapt && !(mode_obs || log_every > 0 || getenv("PXA_XCACHE_SUMMARY"))) return false;   // nothing asked for
        det       = env_int("PXA_XCACHE_ADAPT_DET", 1) != 0;
        obs_every = std::max(1, env_int("PXA_XCACHE_ADAPT_EVERY", 4));
        delay     = std::max(1, env_int("PXA_XCACHE_ADAPT_DELAY", 1));
        chaos     = std::max(0, env_int("PXA_XCACHE_ADAPT_CHAOS", 0));
        if (const char * hf = getenv("PXA_XCACHE_ADAPT_FLAG"); hf && *hf) hold_flag = hf;
        if (const char * tf = getenv("PXA_XCACHE_TRACE"); tf && *tf) trace = fopen(tf, "wb");
        n_used    = (int) model.hparams.n_expert_used;
        size_t any_spare = 0;
        for (const auto & mp : model.pxa_xc_maps) {
            lay l;
            l.il = mp.second;
            l.mapT = mp.first;
            l.n_expert = (int) ((mp.first->ne[0] - 8)/2);
            for (const auto & f : model.pxa_xc_fills) {
                if (f.il != l.il) continue;
                l.hotT.push_back(f.hot);
                l.coldT.push_back(f.cold);
                l.dev = f.dev;
                l.eb += f.hot->nb[2];
                l.foff.push_back(f.file_off);
                l.ffd.push_back(f.file_fd);
            }
            if (std::find(l.ffd.begin(), l.ffd.end(), -1) != l.ffd.end()) { l.ffd.clear(); l.foff.clear(); }
            if (l.hotT.empty() || l.eb == 0 || l.n_expert <= 1) continue;
            // a layer whose hot stack stays in host memory (-ot to CPU: the split records dev -1) has no device copies to
            // schedule; it keeps its planned placement. Passing -1 to the CUDA helpers aborted with "invalid device ordinal".
            if (l.dev < 0) continue;
            const auto & xl = model.pxa_xc[l.il];
            l.map.assign(l.n_expert, -1);
            l.busy.assign(l.n_expert, 0);
            for (size_t k = 0; k < xl.hot.size(); ++k) if (xl.hot[k] >= 0) l.map[xl.hot[k]] = (int32_t) k;
            for (size_t k = 0; k < xl.cold.size(); ++k) {
                if (xl.cold[k] >= 0) l.map[xl.cold[k]] = -2 - (int32_t) k;
                else l.spare.push_back((int32_t) k);
            }
            any_spare += l.spare.size();
            L.push_back(std::move(l));
        }
        if (L.empty()) return false;
        if (adapt && any_spare == 0) adapt = false;       // the plan left no free cold slot: statistics only
        std::sort(L.begin(), L.end(), [](const lay & a, const lay & b) { return a.il < b.il; });
        for (auto & l : L) {
            l.stage_off = stage_u32;
            stage_u32 += (size_t) l.n_expert + 8;
            if (std::find(devs.begin(), devs.end(), l.dev) == devs.end()) devs.push_back(l.dev);
        }
        stage = (uint32_t *) ggml_backend_cuda_xcache_pinned_alloc(stage_u32*sizeof(uint32_t));
        if (!stage) return false;
        for (int d : devs) {    // what the plan left free once the weights, KV cache and compute buffers exist (slot-pool headroom)
            size_t fr = 0, tot = 0;
            ggml_backend_cuda_get_device_memory(d, &fr, &tot);
            LLAMA_LOG_INFO("PXA_XCACHE: device %d has %.0f MiB of %.0f MiB free after the load, the KV cache and the compute buffers\n", d, fr/1048576.0, tot/1048576.0);
        }
        prev_hit.assign(L.size(), 0);
        prev_miss.assign(L.size(), 0);
        lay_hit.assign(L.size(), 0);
        lay_miss.assign(L.size(), 0);
        obs.resize(L.size());
        views.resize(L.size());
        plan_out.resize(std::max<size_t>(8, 2*L.size()*8));
        if (!pol) {
            LLAMA_LOG_INFO("PXA_XCACHE: routing statistics ON (no adaptation policy in this build): %zu split layers\n", L.size());
            return true;
        }
        // measure the link once: the swap budget per step is a fraction of what the card's link carries
        for (int d : devs) {
            double h2d = 0, d2h = 0;
            ggml_backend_cuda_xcache_probe_link(d, &h2d, &d2h);
            if (h2d > 0 && (pcie_gbs == 0 || h2d < pcie_gbs)) pcie_gbs = h2d;
        }
        // prior: the static counts the plan was made from, as expected selections per token (sum over experts = n_used)
        prior.assign(L.size(), {});
        lcfg.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            auto & l = L[i];
            lcfg[i].n_expert     = l.n_expert;
            lcfg[i].n_slots      = (int32_t) model.pxa_xc[l.il].hot.size();
            lcfg[i].expert_bytes = l.eb;
            lcfg[i].prior        = nullptr;
            if (l.il < (int) model.pxa_xc_prior.size() && (int) model.pxa_xc_prior[l.il].size() == l.n_expert) {
                double tot = 0;
                for (double c : model.pxa_xc_prior[l.il]) tot += c;
                if (tot > 0) {
                    prior[i].resize(l.n_expert);
                    for (int e = 0; e < l.n_expert; ++e) prior[i][e] = model.pxa_xc_prior[l.il][e] * n_used / tot;
                    lcfg[i].prior = prior[i].data();
                }
            }
        }
        // Probe before the policy is built so a swap can be priced against a measured CPU miss.
        // Unknown rate (probe failed) leaves cpu_gbs at 0 and the policy keeps the ratio test only.
        double cpu_gbs = 0;
        if (!L.empty() && !L[0].coldT.empty() && L[0].coldT[0]->data) {
            const int nt = env_int("PXA_XCACHE_PROBE_THREADS", 16);
            cpu_gbs = pxa_xc_probe_cpu_gbs(L[0].coldT[0]->data, ggml_nbytes(L[0].coldT[0]), nt);
        }
        // Payback stays off unless asked. The same-binary ABBA (2026-10-10) cut swaps from
        // about 18k to about 70 and the hit rate from 87% to 83%, with no speed gain.
        double cpu_use = 0.0;
        if (const char * pb = getenv("PXA_XCACHE_SWAP_PAYBACK"); pb && pb[0] == '1' && pb[1] == 0 && cpu_gbs > 0.0) cpu_use = 0.65 * cpu_gbs;
        ggml_pxqn_xc_cfg cfg = {};
        cfg.n_layers = (int32_t) L.size();
        cfg.layers   = lcfg.data();
        cfg.n_used   = n_used;
        cfg.pcie_gbs = pcie_gbs;
        cfg.cpu_gbs  = cpu_use;
        cfg.flags    = det ? 1u : 0u;
        pst = pol->create(&cfg);
        if (!pst) return false;
        cfg_saved = cfg;
        reset_on = env_int("PXA_XCACHE_ADAPT_RESET", 1) != 0;
        init_map.resize(L.size()); init_spare.resize(L.size()); init_hot.resize(L.size()); init_cold.resize(L.size());
        base_cnt.resize(L.size()); rel_cnt.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            init_map[i]   = L[i].map;
            init_spare[i] = L[i].spare;
            init_hot[i]   = model.pxa_xc[L[i].il].hot;
            init_cold[i]  = model.pxa_xc[L[i].il].cold;
            base_cnt[i].assign((size_t) L[i].n_expert + 8, 0);
            rel_cnt[i].assign((size_t) L[i].n_expert, 0);
        }
        {   // what one routed expert costs when it misses: on the CPU (cold path) vs read over the measured link
            double cpu_us = 0, link_us = 0;
            size_t eb_sum = 0;
            for (const auto & l : L) eb_sum += l.eb;
            const double cpu_log = cpu_use > 0 ? cpu_use : 20.0;
            if (pol->miss_cost_us && pcie_gbs > 0 && pol->miss_cost_us(eb_sum/L.size(), pcie_gbs, cpu_log, &cpu_us, &link_us)) {
                LLAMA_LOG_INFO("PXA_XCACHE: a missed expert (%.2f MiB) costs ~%.0f us on the CPU (cold path, %.1f GB/s streamed from the pinned stack%s) against ~%.0f us over the %.2f GB/s link (zero-copy / DMA): %s\n",
                        eb_sum/(double)L.size()/1048576.0, cpu_us, cpu_log, cpu_gbs > 0 ? ", measured" : ", assumed", link_us, pcie_gbs,
                        link_us < cpu_us ? "the link is the cheaper way to serve a miss" : "the CPU serves misses, the link carries the background swaps");
            }
        }
        if (adapt && env_int("PXA_XCACHE_RTIER", 0) != 0) rt_start();
        if (adapt && env_int("PXA_XCACHE_EVICT_FILE", 0) != 0) io_start();
        freeze = llama_pxa_xc_freeze_parse(getenv("PXA_XCACHE_PREFILL_FREEZE"), getenv("PXA_XCACHE_DECODE_MAXW"), rtier);
        if (adapt && freeze.on) {
            LLAMA_LOG_INFO("PXA_XCACHE: prefill freeze on: only a batch of <= %d tokens (decode, MTP verify) moves experts; after a wider "
                    "one (a prefill ubatch) no map flips and no swap starts, its routing is only observed (PXA_XCACHE_PREFILL_FREEZE=0 off, "
                    "PXA_XCACHE_DECODE_MAXW sets the width)\n", freeze.maxw);
        }
        LLAMA_LOG_INFO("PXA_XCACHE: online adaptation %s: %zu split layers, %zu free cold slots, re-rank every %d decodes, link %.2f GB/s, %s schedule, swap payback %s\n",
                adapt ? "ON" : "observing only", L.size(), any_spare, obs_every, pcie_gbs, det ? "deterministic" : "copy-completion",
                cpu_use > 0 ? "on" : "off");
        if (adapt) arm_learn();
        if (adapt) role_init();
        return true;
    }

    // ---- copies -------------------------------------------------------------------------------------------------------------

    // to_hot: cold slot -> hot slot (H2D); else hot slot -> cold slot (D2H)
    bool copy_slot(lay & l, bool to_hot, int hot_slot, int cold_slot) {
        for (size_t k = 0; k < l.hotT.size(); ++k) {
            const size_t nb = l.hotT[k]->nb[2];
            char * hp = (char *) l.hotT[k]->data  + (size_t) hot_slot*nb;
            char * cp = (char *) l.coldT[k]->data + (size_t) cold_slot*nb;
            if (!ggml_backend_cuda_xcache_copy_async(l.dev, to_hot ? hp : cp, to_hot ? cp : hp, nb, PXA_XC_COPY_CHUNK)) return false;
        }
        return true;
    }

    void flush_maps() {
        for (auto & l : L) {
            if (!l.dirty) continue;
            ggml_backend_tensor_set(l.mapT, l.map.data(), 0, (size_t) l.n_expert*sizeof(int32_t));
            l.dirty = false;
        }
    }

    bool wait_ev(op & o) {
        if (o.fl) {                    // PXA_XCACHE_EVICT_FILE phase 0: the file read instead of a copy event
            if (o.fl->load() == 0) {
                std::unique_lock<std::mutex> lk(io_mu);
                io_done_cv.wait(lk, [&] { return o.fl->load() != 0; });
                ++n_waits;
            }
            const bool ok = o.fl->load() == 1;
            o.fl.reset();
            return ok;
        }
        if (!o.ev) return false;
        if (ggml_backend_cuda_xcache_event_done(o.ev) != 1) {
            ggml_backend_cuda_xcache_event_sync(o.ev);
            ++n_waits;
        }
        ggml_backend_cuda_xcache_event_free(o.ev);
        o.ev = nullptr;
        return true;
    }

    void apply_due() {
        std::vector<size_t> to_issue;      // swaps whose evict landed this tick: the admit copy goes out once the new map is on the device
        size_t w = 0;
        for (size_t i = 0; i < ops.size(); ++i) {
            op o = ops[i];
            bool keep = true;
            if (tick >= o.due && (det || (o.fl ? o.fl->load() != 0 : (o.ev && ggml_backend_cuda_xcache_event_done(o.ev) == 1)))) {
                lay & l = L[o.li];
                auto & xl = model.pxa_xc[l.il];
                if (!wait_ev(o)) {
                    // a copy could not be enqueued (an error): abandon this swap; no new swaps from here on. Phase 0 changed
                    // nothing yet; in phase 1 e_out already lives in its cold slot and the hot slot stays vacant, e_in keeps its
                    // cold slot, so every expert still has exactly one valid home
                    broken = true;
                    if (o.phase == 0 || swap_dummy) l.spare.push_back(o.c_free);   // a dummy never moved e_out out of its hot slot
                    l.busy[o.e_in] = l.busy[o.e_out] = 0;
                    keep = false;
                } else if (swap_dummy && o.phase == 0) {
                    o.phase = 1;                           // PXA_XCACHE_SWAP_DUMMY: no map change; the admit copy goes out below
                    o.due = tick + (det ? delay : 1);
                    to_issue.push_back(w);
                } else if (swap_dummy) {
                    l.spare.push_back(o.c_free);           // the copy of e_out landed on its own hot slot (same bytes): just free up
                    l.busy[o.e_in] = l.busy[o.e_out] = 0;
                    ++n_dummy;
                    keep = false;
                } else if (o.phase == 0) {
                    // the evicted expert's bytes are in the free cold slot: from now on it is served by the cold path and its hot
                    // slot is vacant
                    l.map[o.e_out] = -2 - o.c_free;
                    l.dirty = true;
                    xl.cold[o.c_free] = o.e_out;
                    xl.hot[o.s] = -1;
                    ++n_evicted;
                    o.phase = 1;
                    o.due = tick + (det ? delay : 1);
                    to_issue.push_back(w);
                } else {
                    // the admitted expert's bytes are in the vacated hot slot: route it there; its cold slot is free again
                    l.map[o.e_in] = o.s;
                    l.dirty = true;
                    xl.hot[o.s] = o.e_in;
                    xl.cold[o.c_in] = -1;
                    l.spare.push_back(o.c_in);
                    l.busy[o.e_in] = l.busy[o.e_out] = 0;
                    ++n_admitted;
                    if (reset_on) applied.push_back(o);
                    keep = false;
                }
            }
            if (keep) ops[w++] = o;
        }
        ops.resize(w);
        flush_maps();       // the vacated hot slots are unmapped on the device before anything is written into them
        for (size_t idx : to_issue) {
            op & o = ops[idx];
            lay & l = L[o.li];
            if (copy_slot(l, true, o.s, swap_dummy ? o.c_free : o.c_in)) o.ev = ggml_backend_cuda_xcache_event_record(l.dev);
            else o.ev = nullptr;     // picked up as a failed copy at its due tick
        }
    }

    // finish everything in flight (shutdown): complete both phases of every swap so the slot table is consistent
    void drain() {
        size_t guard = 0;
        while (!ops.empty() && guard++ < 8) {
            const uint64_t save_tick = tick;
            tick = ~0ull;
            bool d = det; det = true;
            apply_due();
            det = d;
            tick = save_tick;
        }
        for (auto & o : ops) if (o.ev) { ggml_backend_cuda_xcache_event_sync(o.ev); ggml_backend_cuda_xcache_event_free(o.ev); o.ev = nullptr; }
        ops.clear();
    }

    // ---- PXA_XCACHE_EVICT_FILE --------------------------------------------------------------------------------------------------
    void io_start() {
#ifdef _WIN32
        LLAMA_LOG_INFO("PXA_XCACHE_EVICT_FILE: not on Windows yet; evicts copy over the link\n");
#else
        size_t n_file = 0;
        for (const auto & l : L) if (!l.ffd.empty()) ++n_file;
        if (n_file == 0) {
            LLAMA_LOG_INFO("PXA_XCACHE_EVICT_FILE: no split layer has a readable model file (set at load); evicts copy over the link\n");
            return;
        }
        evict_file = true;
        io_th = std::thread([this] { io_loop(); });
        LLAMA_LOG_INFO("PXA_XCACHE_EVICT_FILE: on for %zu of %zu split layers: an evicted expert is read back from the model file into its free cold slot (IO thread), the link carries only the admit\n",
                n_file, L.size());
#endif
    }

    void io_loop() {
#ifndef _WIN32
        for (;;) {
            io_job j;
            {
                std::unique_lock<std::mutex> lk(io_mu);
                io_cv.wait(lk, [&] { return io_stop || !io_q.empty(); });
                if (io_q.empty()) return;          // stopping, nothing left
                j = std::move(io_q.front());
                io_q.pop_front();
            }
            const int64_t t0 = ggml_time_us();
            int ok = 1;
            for (size_t k = 0; k < j.dst.size() && ok == 1; ++k) {
                size_t got = 0;
                while (got < j.n[k]) {
                    const ssize_t r = pread(j.fd[k], j.dst[k] + got, j.n[k] - got, (off_t) (j.off[k] + got));
                    if (r < 0 && errno == EINTR) continue;
                    if (r <= 0) { ok = -1; break; }
                    got += (size_t) r;
                }
            }
            us_ef += (uint64_t) (ggml_time_us() - t0);
            ++n_ef_done;
            if (ok != 1) ++n_ef_fail;
            {
                std::lock_guard<std::mutex> lk(io_mu);
                j.fl->store(ok);
            }
            io_done_cv.notify_all();
        }
#endif
    }

    void io_shutdown() {
        if (!io_th.joinable()) return;
        { std::lock_guard<std::mutex> lk(io_mu); io_stop = true; }
        io_cv.notify_all();
        io_th.join();
    }

    // ---- PXA_XCACHE_RTIER: the policy on a planner thread ------------------------------------------------------------------------

    void rt_start() {
        for (auto & b : rt_stage) {
            b = (uint32_t *) ggml_backend_cuda_xcache_pinned_alloc(stage_u32*sizeof(uint32_t));
            if (!b) { LLAMA_LOG_WARN("PXA_XCACHE_RTIER: no pinned staging; the adaptation stays on the decode thread\n"); return; }
        }
        rtier    = true;
        det      = false;           // copy-completion schedule: a decode never waits for a swap's copy
        swap_cap = std::max(0, env_int("PXA_XCACHE_SWAP_CAP", 2));
        reset_on = false;           // the hot set carries across requests (RIX: placement does not change the bits)
        rt_map.resize(L.size()); rt_busy.resize(L.size()); rt_spare.resize(L.size()); rt_views.resize(L.size());
        rt_out.resize(plan_out.size());
        rt_th = std::thread([this]() { rt_loop(); });
        const bool rix = ggml_pxa_rix_on() != 0;
        LLAMA_LOG_INFO("PXA_XCACHE_RTIER: planner thread on, <= %d swaps per decode, copy-completion schedule, hot set carried across requests; "
                "RIX %s%s\n", swap_cap, rix ? "on" : "OFF",
                rix ? " (an expert's placement does not change the bits)" : ": answers depend on copy timing (PXA_PXQN_RIX=1 or PXA_XCACHE_RTIER=0)");
    }

    void rt_shutdown() {
        if (!rt_th.joinable()) return;
        { std::lock_guard<std::mutex> lk(rt_mu); rt_stop = true; }
        rt_cv.notify_all();
        rt_th.join();
        if (rt_ev) { ggml_backend_cuda_xcache_event_free(rt_ev); rt_ev = nullptr; }
    }

    // the planner thread: wait for the counters' copy, observe, plan against the snapshot; the decode thread starts the swaps
    void rt_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(rt_mu);
                rt_cv.wait(lk, [&]() { return rt_stop || rt_job; });
                if (rt_stop) return;
            }
            const int64_t t0 = ggml_time_us();
            void * ev = rt_ev;
            if (ev) ggml_backend_cuda_xcache_event_sync(ev);
            const uint32_t * st = rt_stage[rt_buf];
            size_t eb_sum = 0;
            for (size_t i = 0; i < L.size(); ++i) {
                const uint32_t * b = st + L[i].stage_off;
                obs[i].counts = b;
                obs[i].hits   = b[L[i].n_expert + 0];
                obs[i].misses = b[L[i].n_expert + 1];
                if (role_rebased) {
                    // the cache role: a move set re-created the policy at base_cnt (its counts start at 0 there), so it observes the counts
                    // since then; written by the decode thread while this thread was idle, before the job that reads them
                    const size_t n = (size_t) L[i].n_expert;
                    const auto & bc = base_cnt[i];
                    auto & rc = rel_cnt[i];
                    for (size_t e = 0; e < n; ++e) rc[e] = b[e] - bc[e];     // u32 wrap-safe
                    obs[i].counts = rc.data();
                    obs[i].hits   = b[n + 0] - bc[n + 0];
                    obs[i].misses = b[n + 1] - bc[n + 1];
                }
                const uint32_t dh = obs[i].hits - prev_hit[i], dm = obs[i].misses - prev_miss[i];   // u32 wrap-safe
                prev_hit[i] = obs[i].hits; prev_miss[i] = obs[i].misses;
                tot_hit += dh; tot_miss += dm; win_hit += dh; win_miss += dm;
                lay_hit[i] += dh; lay_miss[i] += dm;
                rt_views[i].map     = rt_map[i].data();
                rt_views[i].busy    = rt_busy[i].data();
                rt_views[i].n_spare = rt_spare[i];
                eb_sum += L[i].eb;
            }
            if (pst) pol->observe(pst, rt_ntok, obs.data());   // null only after a failed re-creation (the cache role), which also sets broken
            int32_t n = 0;
            if (!broken && pst) {
                int32_t n_budget = pol->swap_budget(pst, rt_ntok);
                if (swap_cap > 0) n_budget = std::min<int32_t>(n_budget, swap_cap*obs_every);
                if (n_budget > 0) {
                    const uint64_t budget_bytes = (uint64_t) n_budget * (eb_sum / L.size());
                    n = pol->plan(pst, rt_views.data(), budget_bytes, rt_out.data(), (int32_t) rt_out.size());
                    if (n < 0) n = 0;
                    if (swap_cap > 0) n = std::min<int32_t>(n, n_budget);
                }
            }
            {
                std::lock_guard<std::mutex> lk(rt_mu);
                rt_nout = n;
                rt_job  = false;
                rt_done = true;
                us_rt_worker += ggml_time_us() - t0;
            }
            rt_idle_cv.notify_all();    // a move set waiting for this job goes on
        }
    }

    // decode thread, every tick: start the swaps the planner finished, then hand it the next observation when one is due
    // frozen (PXA_XCACHE_PREFILL_FREEZE): the batch that just completed was a prefill ubatch: a plan the planner finished is dropped
    // (nothing starts on the prefill path; the prompt's own observation below re-plans, and the first decode tick starts that plan)
    void rt_tick(uint32_t n_tok_obs, bool due, bool frozen) {
        bool idle;
        {
            std::lock_guard<std::mutex> lk(rt_mu);
            if (rt_done) {
                rt_done = false;
                ++n_obs;
                if (frozen) {
                    n_frozen_drop += rt_nout > 0 ? rt_nout : 0;
                } else {
                    n_planned += rt_nout > 0 ? rt_nout : 0;
                    for (int32_t k = 0; k < rt_nout; ++k) {
                        const size_t before = ops.size();
                        start_swap(rt_out[k]);
                        if (ops.size() == before) ++n_rt_stale;       // the map moved since the snapshot: skipped
                    }
                }
                rt_nout = 0;
                // the planner is idle here (it only sets rt_done on its way back to waiting): safe to read the policy's stats
                if (log_every > 0 && n_obs % (uint64_t) log_every == 0) stats_line("");
            }
            idle = !rt_job;
        }
        if (!due) return;
        if (!idle) { ++n_rt_skip; return; }              // the planner is still on the last observation: count these tokens next time
        rt_buf ^= 1;
        uint32_t * st = rt_stage[rt_buf];
        bool ok = true;
        for (auto & l : L) {
            const size_t off = (size_t) l.n_expert*sizeof(int32_t);
            if (!ggml_backend_cuda_xcache_copy_async(l.dev, st + l.stage_off, (char *) l.mapT->data + off, ((size_t) l.n_expert + 8)*sizeof(uint32_t), 0)) { ok = false; break; }
        }
        if (rt_ev) { ggml_backend_cuda_xcache_event_free(rt_ev); rt_ev = nullptr; }
        if (ok) rt_ev = ggml_backend_cuda_xcache_event_record(L[0].dev);
        if (!ok || !rt_ev) { broken = true; return; }
        for (size_t i = 0; i < L.size(); ++i) {
            rt_map[i]   = L[i].map;
            rt_busy[i]  = L[i].busy;
            rt_spare[i] = (int32_t) L[i].spare.size();
        }
        {
            std::lock_guard<std::mutex> lk(rt_mu);
            rt_ntok = n_tok_obs;
            rt_job  = true;
            ++n_rt_jobs;
        }
        rt_cv.notify_one();
        tokens_since_obs = 0;
        ticks_since_obs  = 0;
    }

    // ---- observation + planning ---------------------------------------------------------------------------------------------

    bool read_counters() {
        for (auto & l : L) {
            const size_t off = (size_t) l.n_expert*sizeof(int32_t);
            if (!ggml_backend_cuda_xcache_copy_async(l.dev, stage + l.stage_off, (char *) l.mapT->data + off, ((size_t) l.n_expert + 8)*sizeof(uint32_t), 0)) return false;
        }
        for (int d : devs) {
            void * ev = ggml_backend_cuda_xcache_event_record(d);
            if (!ev) return false;
            ggml_backend_cuda_xcache_event_sync(ev);
            ggml_backend_cuda_xcache_event_free(ev);
        }
        return true;
    }

    bool held() const {
        if (hold_flag.empty()) return false;
        FILE * f = fopen(hold_flag.c_str(), "r");
        if (!f) return false;
        const int c = fgetc(f);
        fclose(f);
        return c == '0';
    }


    // Arms the learned counts through the library's "learn" role: a boot on learned counts (llama.cpp sets
    // model.pxa_xc_learned_tokens) carries a share of them in every save, a boot on the curated map saves this session's counts
    // only. No role (no library, an older one): nothing learned or written. The role off: not armed, nothing written.
    void arm_learn() {
        const ggml_pxqn_learn_api * t = ggml_pxqn_learn_api_get();
        if (!t) return;
        char msg[512];
        if (t->off(msg, sizeof(msg))) {
            LLAMA_LOG_INFO("PXA_XCACHE: %s\n", msg[0] ? msg : "learning off this session");
            return;
        }
        char buf[4096];
        const std::string where = llama_pxa_xcache_learned_path(&model, buf, sizeof(buf)) ? std::string(buf) : std::string();
        int rc = GGML_PXQN_LEARN_NOT_LEARNED;
        void * b = t->arm(where.c_str(), model.pxa_xc_learned_tokens, &rc, msg, sizeof(msg));
        if (rc == GGML_PXQN_LEARN_NO_PATH) {
            LLAMA_LOG_WARN("PXA_XCACHE: not saving learned counts. There is no writable directory beside the model or in the cache, so they are lost when this process ends.\n");
            return;
        }
        if (rc != GGML_PXQN_LEARN_ARMED || !b) return;
        learn.t = t;
        learn.book = b;
        learn.path = where;
        learn.keep = t->carry ? t->carry(b) : 0.0;
        learn.prior_tokens = t->carry_tokens ? t->carry_tokens(b) : 0.0;
        pxa_warn_learned_cache(learn.path);
        LLAMA_LOG_INFO("PXA_XCACHE: learning on; saving counts to %s. The curated file is left as it is.%s\n", learn.path.c_str(), msg);
    }

    void maybe_persist() {
        if (learn.armed() && n_obs > 0 && (n_obs % 32ull) == 0) persist_learned(false);
    }

    // The library builds the learned rows. This function packs the prior and the device counters, then saves them through learn
    // (the csv, then the role's record). The curated file is never opened for write. No library: nothing is saved. A boot on
    // learned counts carries learn.keep of them in the rows (the role's rows), and the record's tokens follow the rows.
    void persist_learned(bool shutdown) {
        if (!learn.armed() || !stage) return;
        if (!read_counters()) {
            if (shutdown) learn.stop();
            return;
        }
        role_save(shutdown);     // the cache role's state beside the learned csv (its own file, whatever the csv save does next)
        const int n_layer = (int) model.hparams.n_layer;
        const int n_expert = (int) model.hparams.n_expert;
        if (n_layer <= 0 || n_expert <= 0) return;
        int n_main = n_layer - (int) model.hparams.nextn_predict_layers;
        if (n_main < 0) n_main = 0;
        if (n_main > n_layer) n_main = n_layer;
        ggml_pxqn_xcache_learn_build_fn build = ggml_pxqn_xcache_learn_build_get();
        if (!build) {
            if (!learn_wait_logged) {
                LLAMA_LOG_WARN("PXA_XCACHE: not saving learned counts; the library that builds them is not loaded\n");
                learn_wait_logged = true;
            }
            if (shutdown) learn.stop();
            return;
        }
        const std::vector<double> prior = pxa_xc_prior_flat(model.pxa_xc_prior, n_layer, n_expert);   // llama-pxa-counts.h
        std::vector<int> obs_il;
        std::vector<uint32_t> obs;
        double routed_max = 0;   // the most routings one main-model layer counted: n_used per token (prompt or generated)
        for (size_t i = 0; i < L.size(); ++i) {
            const int il = L[i].il;
            if (il < 0 || il >= n_layer || L[i].n_expert != n_expert) continue;
            const uint32_t * b = stage + L[i].stage_off;
            obs_il.push_back(il);
            obs.insert(obs.end(), b, b + n_expert);
            if (il < n_main) {   // an MTP head's layer counts the draft's own tokens
                double s = 0;
                for (int e = 0; e < n_expert; ++e) s += (double) b[e];
                routed_max = std::max(routed_max, s);
            }
        }
        std::vector<uint8_t> routed((size_t) n_layer, 0);
        for (const auto & kv : model.tensors_by_name) {
            const std::string & name = kv.first;
            if (name.size() < 4 || name.compare(0, 4, "blk.") != 0) continue;
            if (name.find("ffn_down_exps.weight") == std::string::npos) continue;
            const int il = atoi(name.c_str() + 4);
            if (il >= 0 && il < n_layer) routed[(size_t) il] = 1;
        }
        const bool accum = learn.keep > 0 && learn.t && learn.t->rows;
        std::vector<uint64_t> counts((size_t) n_layer * (size_t) n_expert, 0);
        int wait_layer = -1;
        const int rc = accum ? learn.t->rows(n_layer, n_expert, n_main, prior.data(), learn.keep,
                                     obs_il.empty() ? nullptr : obs_il.data(), (int) obs_il.size(),
                                     obs.empty() ? nullptr : obs.data(), routed.data(), counts.data(), &wait_layer)
                             : build(n_layer, n_expert, n_main, prior.data(),
                                     obs_il.empty() ? nullptr : obs_il.data(), (int) obs_il.size(),
                                     obs.empty() ? nullptr : obs.data(), routed.data(), counts.data(), &wait_layer);
        if (rc == 1) {
            if (shutdown) learn.stop();
            return;
        }
        if (rc == 2) {
            if (shutdown) learn.stop();
            if (!learn_wait_logged) {
                LLAMA_LOG_WARN("PXA_XCACHE: not saving learned counts; layer %d has no counts yet\n", wait_layer);
                learn_wait_logged = true;
            }
            return;
        }
        if (rc != 0) {
            if (shutdown) learn.stop();
            return;
        }
        // every save rewrites the tokens with the rows (the role's record follows them)
        const double tokens = (n_used > 0 ? routed_max / n_used : 0.0) + (accum ? learn.keep * learn.prior_tokens : 0.0);
        const int sessions = learn.save([&](const std::string & p) {
            return llama_pxa_xcache_counts_write(&model, counts.data(), n_layer, n_expert, p.c_str());
        }, tokens, shutdown);
        if (sessions < 0) {
            LLAMA_LOG_WARN("PXA_XCACHE: could not save learned counts to %s\n", learn.path.c_str());
            return;
        }
        if (shutdown) {
            LLAMA_LOG_INFO("PXA_XCACHE: saved learned counts to %s (%.0f tokens over %d sessions)\n", learn.path.c_str(), tokens, sessions);
        } else if (!learn_saved) {
            learn_saved = true;
            LLAMA_LOG_INFO("PXA_XCACHE: saved learned counts to %s (%.0f tokens over %d sessions%s)\n", learn.path.c_str(), tokens,
                    sessions, learn.counted() ? ", this one included" : "");
        }
    }

    // plan_ok false (PXA_XCACHE_PREFILL_FREEZE, after a prefill ubatch): the policy observes the routing, nothing is planned or started
    void observe_and_plan(bool plan_ok = true) {
        if (!read_counters()) return;
        for (size_t i = 0; i < L.size(); ++i) {
            const uint32_t * b = stage + L[i].stage_off;
            obs[i].counts = b;
            obs[i].hits   = b[L[i].n_expert + 0];
            obs[i].misses = b[L[i].n_expert + 1];
            if (reset_on && !base_cnt.empty()) {        // counts since the last reset (u32 wrap-safe); all zero base = the raw counters
                const auto & bc = base_cnt[i];
                auto & rc = rel_cnt[i];
                for (int e = 0; e < L[i].n_expert; ++e) rc[(size_t) e] = b[e] - bc[(size_t) e];
                obs[i].counts = rc.data();
                obs[i].hits   = b[L[i].n_expert + 0] - bc[(size_t) L[i].n_expert + 0];
                obs[i].misses = b[L[i].n_expert + 1] - bc[(size_t) L[i].n_expert + 1];
            }
            const uint32_t dh = obs[i].hits - prev_hit[i], dm = obs[i].misses - prev_miss[i];   // u32 wrap-safe
            prev_hit[i] = obs[i].hits; prev_miss[i] = obs[i].misses;
            tot_hit += dh; tot_miss += dm; win_hit += dh; win_miss += dm;
            lay_hit[i] += dh; lay_miss[i] += dm;
        }
        ++n_obs;
        const uint32_t n_tok = tokens_since_obs;
        tokens_since_obs = 0;
        ticks_since_obs = 0;
        if (!pst) { maybe_persist(); return; }
        pol->observe(pst, n_tok, obs.data());
        if (!adapt || broken || !plan_ok || held()) { maybe_persist(); return; }
        size_t eb_sum = 0;
        for (size_t i = 0; i < L.size(); ++i) {
            views[i].map     = L[i].map.data();
            views[i].busy    = L[i].busy.data();
            views[i].n_spare = (int32_t) L[i].spare.size();
            eb_sum += L[i].eb;
        }
        if (chaos > 0 && !held()) {      // not the policy: a fixed pseudo-random schedule, the same on every run
            int32_t n = 0;
            for (int k = 0; k < chaos && n < (int32_t) plan_out.size(); ++k) {
                auto rnd = [&]() { chaos_rng = chaos_rng*6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (chaos_rng >> 33); };
                const int li = (int) (rnd() % L.size());
                const lay & l = L[li];
                if (l.spare.empty()) continue;
                int e_in = -1, e_out = -1;
                for (int tries = 0; tries < 64 && (e_in < 0 || e_out < 0); ++tries) {
                    const int e = (int) (rnd() % (uint32_t) l.n_expert);
                    if (l.busy[e]) continue;
                    if (l.map[e] <= -2 && e_in < 0) e_in = e;
                    else if (l.map[e] >= 0 && e_out < 0) e_out = e;
                }
                if (e_in >= 0 && e_out >= 0) plan_out[n++] = { li, e_in, e_out };
            }
            n_planned += n;
            for (int32_t k = 0; k < n; ++k) start_swap(plan_out[k]);
            maybe_persist();
            return;
        }
        const int32_t n_budget = pol->swap_budget(pst, n_tok);
        if (n_budget <= 0) { maybe_persist(); return; }
        const uint64_t budget_bytes = (uint64_t) n_budget * (eb_sum / L.size());
        const int32_t n = pol->plan(pst, views.data(), budget_bytes, plan_out.data(), (int32_t) plan_out.size());
        n_planned += n > 0 ? n : 0;
        for (int32_t k = 0; k < n; ++k) start_swap(plan_out[k]);
        maybe_persist();
    }

    void start_swap(const ggml_pxqn_xc_swap & sw) {
        if (sw.layer < 0 || sw.layer >= (int) L.size()) return;
        lay & l = L[sw.layer];
        if (sw.e_in < 0 || sw.e_in >= l.n_expert || sw.e_out < 0 || sw.e_out >= l.n_expert) return;
        if (l.map[sw.e_in] > -2 || l.map[sw.e_out] < 0 || l.busy[sw.e_in] || l.busy[sw.e_out] || l.spare.empty()) return;
        op o;
        o.li = sw.layer; o.e_in = sw.e_in; o.e_out = sw.e_out;
        o.c_in   = -2 - l.map[sw.e_in];
        o.s      = l.map[sw.e_out];
        o.c_free = l.spare.back();
        if (evict_file && !l.ffd.empty()) {
            // evict first: the expert's bytes from the model file -> free cold slot (IO thread; nothing on the link)
            io_job j;
            for (size_t k = 0; k < l.coldT.size(); ++k) {
                const size_t nb = l.coldT[k]->nb[2];
                j.fd.push_back(l.ffd[k]);
                j.off.push_back(l.foff[k] + (size_t) sw.e_out*nb);
                j.n.push_back(nb);
                j.dst.push_back((char *) l.coldT[k]->data + (size_t) o.c_free*nb);
            }
            o.fl = std::make_shared<std::atomic<int>>(0);
            j.fl = o.fl;
            { std::lock_guard<std::mutex> lk(io_mu); io_q.push_back(std::move(j)); }
            io_cv.notify_one();
            ++n_ef;
        } else {
            if (!copy_slot(l, false, o.s, o.c_free)) return;      // evict first: hot slot -> free cold slot
            o.ev = ggml_backend_cuda_xcache_event_record(l.dev);
            if (!o.ev) return;
        }
        l.spare.pop_back();
        l.busy[o.e_in] = l.busy[o.e_out] = 1;
        o.phase = 0;
        o.due = tick + (det ? delay : 1);
        ops.push_back(o);
        ++n_started;
    }

    // ---- self test (PXA_XCACHE_ADAPT_SELFTEST=1) ---------------------------------------------------------------------------------
    // Moves one hot and one cold expert of a few layers through the real two-phase machinery, checks that every byte of both
    // experts arrived where the map now says it is, then moves them back.
    bool slot_bytes(lay & l, bool hot, int slot, std::vector<std::vector<char>> & out) {
        out.clear();
        for (size_t k = 0; k < l.hotT.size(); ++k) {
            ggml_tensor * t = hot ? l.hotT[k] : l.coldT[k];
            const size_t nb = t->nb[2];
            std::vector<char> b(nb);
            ggml_backend_tensor_get(t, b.data(), (size_t) slot*nb, nb);
            out.push_back(std::move(b));
        }
        return true;
    }

    int selftest() {
        int tested = 0, bad = 0;
        const size_t stride = std::max<size_t>(1, L.size()/4);
        for (size_t li = 0; li < L.size() && tested < 4; li += stride) {
            lay & l = L[li];
            if (l.spare.empty()) continue;
            int e_in = -1, e_out = -1;
            for (int e = 0; e < l.n_expert && (e_in < 0 || e_out < 0); ++e) {
                if (l.map[e] <= -2 && e_in < 0) e_in = e;
                if (l.map[e] >= 0 && e_out < 0) e_out = e;
            }
            if (e_in < 0 || e_out < 0) continue;
            std::vector<std::vector<char>> in_bytes, out_bytes, got;
            slot_bytes(l, false, -2 - l.map[e_in], in_bytes);      // the cold expert, as the cold stack holds it
            slot_bytes(l, true,  l.map[e_out],     out_bytes);     // the hot expert, as the hot stack holds it
            auto run = [&](int a, int b) {
                ggml_pxqn_xc_swap sw = { (int32_t) li, a, b };
                const size_t n0 = ops.size();
                start_swap(sw);
                if (ops.size() == n0) return false;
                for (int t = 0; t < 4 && !ops.empty(); ++t) { ++tick; apply_due(); }
                return ops.empty();
            };
            bool ok = run(e_in, e_out);
            ok = ok && l.map[e_in] >= 0 && l.map[e_out] <= -2;
            bool same_in = false, same_out = false;
            if (ok) {
                slot_bytes(l, true, l.map[e_in], got);  same_in = got == in_bytes;
                slot_bytes(l, false, -2 - l.map[e_out], got); same_out = got == out_bytes;
            }
            ok = ok && same_in && same_out;
            // and back
            bool back = run(e_out, e_in) && l.map[e_out] >= 0 && l.map[e_in] <= -2;
            if (back) {
                slot_bytes(l, true, l.map[e_out], got);  back = got == out_bytes;
                slot_bytes(l, false, -2 - l.map[e_in], got); back = back && got == in_bytes;
            }
            LLAMA_LOG_INFO("PXA_XCACHE adapt selftest: layer %d expert %d in / %d out: %s, back: %s\n", l.il, e_in, e_out, ok ? "ok" : "FAILED", back ? "ok" : "FAILED");
            ++tested;
            if (!ok || !back) ++bad;
        }
        LLAMA_LOG_INFO("PXA_XCACHE adapt selftest: %s (%d layers)\n", tested > 0 && bad == 0 ? "PASS" : "FAIL", tested);
        return bad == 0 && tested > 0;
    }

    // ---- routing trace (PXA_XCACHE_TRACE) ------------------------------------------------------------------------------------
    // header: "XCTR1", n_layers, then per layer {il, n_expert, n_slots_hot, n_free_cold} + the initial map (int32 x n_expert) + {has_prior, float x n_expert};
    // records: {tick, n_tokens} + per layer the cumulative counters (u32 x n_expert, hits, misses)
    void trace_tick(int n_tokens) {
        if (!trace || !read_counters()) return;
        auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, trace); };
        if (!trace_hdr) {
            fwrite("XCTR1", 1, 5, trace);
            w32((uint32_t) L.size());
            for (const auto & l : L) {
                w32((uint32_t) l.il); w32((uint32_t) l.n_expert);
                w32((uint32_t) model.pxa_xc[l.il].hot.size()); w32((uint32_t) l.spare.size());
                fwrite(l.map.data(), 4, (size_t) l.n_expert, trace);
                const size_t li = (size_t) (&l - L.data());
                const bool hp = li < lcfg.size() && lcfg[li].prior != nullptr;
                w32(hp ? 1u : 0u);
                if (hp) for (int e = 0; e < l.n_expert; ++e) { const float v = (float) lcfg[li].prior[e]; fwrite(&v, 4, 1, trace); }
            }
            trace_hdr = true;
        }
        w32((uint32_t) tick); w32((uint32_t) n_tokens);
        for (size_t i = 0; i < L.size(); ++i) {
            const uint32_t * b = stage + L[i].stage_off;
            fwrite(b, 4, (size_t) L[i].n_expert, trace);
            w32(b[L[i].n_expert + 0]); w32(b[L[i].n_expert + 1]);
        }
        fflush(trace);
    }

    // ---- the slot tables' walk (llama_pxa_xcache_check) ---------------------------------------------------------------------
    // every expert has one home (map >= 0: the hot slot that names it; <= -2: the cold slot that names it), no slot holds two, the
    // model's slot lists name only experts whose map entry points back, a free cold slot is in the spare list or held by a swap
    // whose evict is in flight (exactly one of the two), and busy marks exactly the experts of the swaps in flight. A hot slot may
    // be vacant (a swap between its evict and its admit, or one abandoned after a failed copy).
    int check_tables() const {
        int bad = 0;
        for (size_t i = 0; i < L.size(); ++i) {
            const lay & l = L[i];
            const auto & xl = model.pxa_xc[l.il];
            const int nh = (int) xl.hot.size(), nc = (int) xl.cold.size();
            std::vector<int> seen_h((size_t) nh, 0), seen_c((size_t) nc, 0), free_c((size_t) nc, 0);
            if ((int) l.map.size() != l.n_expert || (int) l.busy.size() != l.n_expert) { ++bad; continue; }
            for (int e = 0; e < l.n_expert; ++e) {
                const int32_t v = l.map[(size_t) e];
                if (v >= 0) {
                    if (v >= nh || xl.hot[(size_t) v] != e) ++bad; else ++seen_h[(size_t) v];
                } else if (v <= -2) {
                    const int c = -2 - v;
                    if (c >= nc || xl.cold[(size_t) c] != e) ++bad; else ++seen_c[(size_t) c];
                } else {
                    ++bad;   // -1: no home
                }
            }
            for (int k : seen_h) if (k > 1) ++bad;
            for (int k : seen_c) if (k > 1) ++bad;
            for (int k = 0; k < nh; ++k) {
                const int e = xl.hot[(size_t) k];
                if (e >= 0 && (e >= l.n_expert || l.map[(size_t) e] != k)) ++bad;
            }
            for (int k = 0; k < nc; ++k) {
                const int e = xl.cold[(size_t) k];
                if (e >= 0 && (e >= l.n_expert || l.map[(size_t) e] != -2 - k)) ++bad;
            }
            for (int32_t c : l.spare) {
                if (c < 0 || c >= nc || xl.cold[(size_t) c] >= 0) ++bad; else ++free_c[(size_t) c];
            }
            std::vector<uint8_t> want((size_t) l.n_expert, 0);
            for (const op & o : ops) {
                if (o.li != (int) i) continue;
                if (o.e_in >= 0 && o.e_in < l.n_expert) want[(size_t) o.e_in] = 1;
                if (o.e_out >= 0 && o.e_out < l.n_expert) want[(size_t) o.e_out] = 1;
                // the free cold slot an evict is copying into: popped from the spare list, named at the evict's flip (a dummy swap
                // never names it)
                if ((o.phase == 0 || swap_dummy) && o.c_free >= 0 && o.c_free < nc) ++free_c[(size_t) o.c_free];
            }
            for (int k = 0; k < nc; ++k) if (xl.cold[(size_t) k] < 0 && free_c[(size_t) k] != 1) ++bad;
            for (int e = 0; e < l.n_expert; ++e) if ((l.busy[(size_t) e] != 0) != (want[(size_t) e] != 0)) ++bad;
        }
        return bad;
    }

    // ---- the cache role (ggml-pxqn-ext.h) ---------------------------------------------------------------------------------------

    void role_init() {
        const char * why = nullptr;
        if (!rtier)                                    why = "the hot set is not carried across requests here; PXA_XCACHE_RTIER=1 carries it";
        else if (!ggml_pxa_rix_on())                   why = "RIX is off, so a moved set would change the answers of the requests after it";
        else if (swap_dummy)                           why = "PXA_XCACHE_SWAP_DUMMY moves no expert";
        else if (!pol || !pst || base_cnt.size() != L.size() || rel_cnt.size() != L.size()) why = "no policy state";
        const ggml_pxqn_cache_api * t = why ? nullptr : ggml_pxqn_cache_api_get();
        if (!t) {
            if (why) LLAMA_LOG_INFO("PXA_XCACHE: cache role off (%s)\n", why);
            return;
        }
        std::vector<ggml_pxqn_cache_layer> ls(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            ls[i].il       = L[i].il;
            ls[i].n_expert = L[i].n_expert;
            ls[i].n_slots  = (int32_t) model.pxa_xc[L[i].il].hot.size();
        }
        ggml_pxqn_cache_args a = {};
        a.size       = (uint32_t) sizeof(a);
        a.n_layers   = (int32_t) L.size();
        a.layers     = ls.data();
        a.n_used     = n_used > 0 ? (double) n_used : 8.0;
        a.maxw       = freeze.maxw;
        a.model_path = model.pxa_model_path.c_str();
        struct stat st;
        if (!model.pxa_model_path.empty() && stat(model.pxa_model_path.c_str(), &st) == 0) a.model_size = (uint64_t) st.st_size;
        char msg[1024];
        void * s = t->create(&a, msg, sizeof(msg));
        if (!s) {
            if (msg[0]) LLAMA_LOG_INFO("PXA_XCACHE: cache role: %s\n", msg);
            return;
        }
        role = t;
        role_st = s;
        role_sel.assign(L.size(), {});
        role_on = true;
        role_load();
        if (msg[0]) LLAMA_LOG_INFO("PXA_XCACHE: cache role: %s\n", msg);
    }

    // ---- the role's state file: beside the learned csv, written where and when this session writes it (its save points, the exit
    // save closing the running request first), read at boot under the learned prior's switch and evidence rule. The learned prior
    // off neither reads nor writes it (the learned book does not arm). The role checks the bytes; a file it refuses is ignored
    // with one line.
    std::string role_file() const {
        static const char suf[] = ".expert-counts.learned.csv";
        const size_t ns = sizeof(suf) - 1;
        if (!role || !role->file_suffix || !role->file_suffix[0]) return "";
        if (!learn.armed() || learn.path.size() < ns || learn.path.compare(learn.path.size() - ns, ns, suf) != 0) return "";
        return learn.path.substr(0, learn.path.size() - ns) + role->file_suffix;
    }

    // the views of the counters in stage and of the resident sets the role reads
    struct role_view_buf {
        std::vector<const uint32_t *> cnt;
        std::vector<std::vector<uint8_t>> res;
        std::vector<const uint8_t *> resp;
        std::vector<const double *> pri;
        ggml_pxqn_cache_view v = {};
    };
    void role_view(role_view_buf & b) const {
        b.cnt.resize(L.size());
        b.pri.resize(L.size());
        b.res = role_resident();
        b.resp.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            b.cnt[i]  = stage + L[i].stage_off;
            b.resp[i] = b.res[i].data();
            b.pri[i]  = i < prior.size() && prior[i].size() == (size_t) L[i].n_expert ? prior[i].data() : nullptr;
        }
        b.v.size     = (uint32_t) sizeof(b.v);
        b.v.counters = b.cnt.data();
        b.v.resident = b.resp.data();
        b.v.prior    = b.pri.data();
    }

    // a learned save point; stage holds the counters persist_learned just read
    void role_save(bool shutdown) {
        if (!role_on) return;
        const std::string p = role_file();
        if (p.empty()) return;
        if (shutdown) role_close_request();
        const void * data = nullptr;
        const size_t n = role->save(role_st, &data);
        if (n == 0 || !data) return;
        if (!pxa_file_replace(p, std::string((const char *) data, n))) {
            if (!role_save_warned) LLAMA_LOG_WARN("PXA_XCACHE: could not save the cache role's state to %s\n", p.c_str());
            role_save_warned = true;
            return;
        }
        if (shutdown || !role_saved) LLAMA_LOG_INFO("PXA_XCACHE: saved the cache role's state to %s\n", p.c_str());
        role_saved = true;
    }

    // at boot (role_init)
    void role_load() {
        if (const ggml_pxqn_learn_api * lt = ggml_pxqn_learn_api_get(); lt && lt->off(nullptr, 0)) return;
        if (!role->file_suffix || !role->file_suffix[0] || !role->load) return;
        char buf[4096];
        if (!llama_pxa_xcache_role_path(&model, role->file_suffix, buf, sizeof(buf))) return;      // none yet
        if (model.pxa_xc_learned_tokens < 0) {
            LLAMA_LOG_INFO("PXA_XCACHE: not reading %s: this boot planned from the curated map (it is read with the learned counts "
                    "it was saved beside)\n", buf);
            return;
        }
        std::string bytes;
        if (FILE * f = fopen(buf, "rb")) {
            char tmp[65536];
            size_t n;
            while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) bytes.append(tmp, n);
            fclose(f);
        }
        char msg[512];
        const int rc = role->load(role_st, bytes.data(), bytes.size(), msg, sizeof(msg));
        if (rc < 0) LLAMA_LOG_WARN("PXA_XCACHE: %s: %s\n", buf, msg);
        else        LLAMA_LOG_INFO("PXA_XCACHE: %s: %s\n", buf, msg);
    }

    // the role stops for the rest of the session with one warning; fail = the adaptation stops starting swaps too (a copy or the
    // tables went wrong, the rule of any failed copy)
    void role_off(const char * why, bool fail) {
        if (role_on) {
            LLAMA_LOG_WARN("PXA_XCACHE: %s; the cache role is off for this session%s\n", why,
                    fail ? ", and no new swap starts (as after any failed copy)" : "");
        }
        role_on = false;
        if (role && role_st && role->halt) role->halt(role_st);
        if (fail) broken = true;
    }

    // the set each split layer serves once every swap in flight has landed (map-hot, minus each swap's e_out, plus its e_in):
    // n_slots experts per layer, what drain() leaves
    std::vector<std::vector<uint8_t>> role_resident() const {
        std::vector<std::vector<uint8_t>> r(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            r[i].assign((size_t) L[i].n_expert, 0);
            for (int e = 0; e < L[i].n_expert; ++e) if (L[i].map[(size_t) e] >= 0) r[i][(size_t) e] = 1;
        }
        for (const op & o : ops) {
            r[(size_t) o.li][(size_t) o.e_out] = 0;
            r[(size_t) o.li][(size_t) o.e_in]  = 1;
        }
        return r;
    }

    // the running request ended (stage holds the counters now)
    void role_close_request() {
        role_view_buf b;
        role_view(b);
        role->end(role_st, &b.v);
    }

    // a request start (llama_pxa_xcache_seq_start; prompt = llama_pxa_xcache_batch_start's PROMPT start): the role closes the
    // previous request and keeps the counters as the new request's base. true = the start ran
    bool role_seq_start(bool prompt = false) {
        if (!role_on || broken) return false;
        const int64_t t0 = ggml_time_us();
        if (!read_counters()) {
            role_off("the routing counters could not be read at a sequence start", false);
            return false;
        }
        role_view_buf b;
        role_view(b);
        role->start(role_st, &b.v, prompt ? 1 : 0);
        us_role += ggml_time_us() - t0;
        return true;
    }

    // llama_pxa_xcache_batch_start: the request-start decision for one main-model batch
    int role_batch(int n_tokens, bool strict) {
        if (!role || !role_st) return strict && n_tokens > 0 ? GGML_PXQN_CACHE_START_STRICT : GGML_PXQN_CACHE_START_NONE;
        return role->batch(role_st, role_on && !broken ? 1 : 0, strict ? 1 : 0, n_tokens);
    }

    // the tick the role asked for: it reads the counters and the resident set, and may ask for a move set. true = moved
    bool role_select() {
        if (!role_on || broken) return false;
        const int64_t t0 = ggml_time_us();
        if (!read_counters()) {
            role_off("the routing counters could not be read at a request's first decode", false);
            return false;
        }
        role_view_buf b;
        role_view(b);
        for (size_t i = 0; i < L.size(); ++i) role_sel[i].assign(b.cnt[i], b.cnt[i] + L[i].n_expert + 8);
        const int act = role->select(role_st, &b.v);
        if (act == GGML_PXQN_CACHE_ACT_NONE) {
            us_role += ggml_time_us() - t0;
            return false;
        }
        double ms = 0;
        int outcome = 0;
        if (act == GGML_PXQN_CACHE_ACT_MOVE) {
            const int64_t t1 = ggml_time_us();
            outcome = role_move() ? 1 : 2;
            const int64_t dt = ggml_time_us() - t1;
            us_role_moves += dt;
            ms = dt/1000.0;
        }
        char msg[1024];
        role->finish(role_st, role_on ? 1 : 0, outcome, ms, log_every > 0 ? msg : nullptr, log_every > 0 ? sizeof(msg) : 0);
        if (log_every > 0 && msg[0]) LLAMA_LOG_INFO("PXA_XCACHE: cache role: %s\n", msg);
        us_role += ggml_time_us() - t0;
        return outcome == 1;
    }

    // One synchronous burst, the reset_to_plan pattern. The planner stands still (a job in flight is waited for, its plan dropped:
    // it was made against the set this burst replaces) and every swap in flight lands (drain). Then per layer one chain on the
    // layer's copy stream, in order, so a single free cold slot carries it: evict e_out from its hot slot into the free cold slot
    // (D2H, also with PXA_XCACHE_EVICT_FILE: the stream orders it before the admit that reuses the slot), admit e_in from its cold
    // slot into that hot slot, and e_in's old cold slot is the next free one. No graph runs here (between two decodes), one event
    // per device is waited for before the maps move on the device, and the tables are walked after. A failure: one warning, the
    // role off and no new swap; an admit that cannot be enqueued leaves its hot slot vacant and every expert one home (apply_due's
    // rule). The moves are the role's plan on the settled tables.
    bool role_move() {
        if (rtier) {
            std::unique_lock<std::mutex> lk(rt_mu);
            if (rt_job) ++n_role_waits;
            rt_idle_cv.wait(lk, [&] { return !rt_job || rt_stop; });
            if (rt_done) {
                ++n_role_drop_plans;
                n_role_drop += rt_nout > 0 ? (uint64_t) rt_nout : 0;
                rt_nout = 0;
                rt_done = false;
                ++n_obs;
            }
        }
        drain();
        if (broken) {
            role_off("a swap in flight failed before a move set", false);
            return false;
        }
        std::vector<std::vector<uint8_t>> cur(L.size());
        std::vector<const uint8_t *> curp(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            cur[i].assign((size_t) L[i].n_expert, 0);
            for (int e = 0; e < L[i].n_expert; ++e) if (L[i].map[(size_t) e] >= 0) cur[i][(size_t) e] = 1;
            curp[i] = cur[i].data();
        }
        const ggml_pxqn_cache_move * mv = nullptr;
        const double * val = nullptr;
        const size_t np = role->plan(role_st, curp.data(), &mv, &val);
        size_t n_sw = 0;
        double g_sw = 0;
        bool ok = true;
        size_t p = 0;
        while (ok && p < np) {
            const int li = mv[p].layer;
            lay & l = L[(size_t) li];
            auto & xl = model.pxa_xc[l.il];
            if (l.spare.empty()) {      // no free cold slot in this layer (not a plan this adaptor makes): leave it as it is
                while (p < np && mv[p].layer == li) ++p;
                continue;
            }
            int32_t c_free = l.spare.back();
            l.spare.pop_back();
            for (; p < np && mv[p].layer == li; ++p) {
                const int e_in = mv[p].e_in, e_out = mv[p].e_out;
                const int32_t s = l.map[(size_t) e_out], c_in = -2 - l.map[(size_t) e_in];
                if (s < 0 || l.map[(size_t) e_in] > -2 || l.busy[(size_t) e_in] || l.busy[(size_t) e_out]) { ok = false; break; }
                if (!copy_slot(l, false, s, c_free)) { ok = false; break; }        // nothing moved: c_free is still free
                l.map[(size_t) e_out] = -2 - c_free;
                xl.cold[(size_t) c_free] = e_out;
                xl.hot[(size_t) s] = -1;
                l.dirty = true;
                if (!copy_slot(l, true, s, c_in)) { c_free = -1; ok = false; break; }   // e_out holds c_free now
                l.map[(size_t) e_in] = s;
                xl.hot[(size_t) s] = e_in;
                xl.cold[(size_t) c_in] = -1;
                c_free = c_in;
                ++n_sw;
                g_sw += val[p];
            }
            if (c_free >= 0) l.spare.push_back(c_free);
        }
        for (int d : devs) {
            void * ev = ggml_backend_cuda_xcache_event_record(d);
            if (!ev) { ok = false; continue; }
            ggml_backend_cuda_xcache_event_sync(ev);
            ggml_backend_cuda_xcache_event_free(ev);
        }
        flush_maps();
        ok = ok && check_tables() == 0;
        role->moved(role_st, ok ? 1 : 0, n_sw, g_sw);
        if (!ok) {
            role_off("a move set did not complete", true);
            return false;
        }
        role_reseed();
        return role_on;
    }

    // The policy starts over at the select's counters, with the prior the role gives. Its decayed counts described the set before;
    // kept, they would swap the moves straight back. The routings since its last observation go into the hit totals first; then
    // base_cnt = those counters and every later observation is relative to them (rt_loop), since a new policy state starts its
    // own counters at 0.
    void role_reseed() {
        for (size_t i = 0; i < L.size(); ++i) {
            const size_t n = (size_t) L[i].n_expert;
            const auto & c = role_sel[i];
            const uint32_t h = role_rebased ? c[n] - base_cnt[i][n] : c[n];
            const uint32_t m = role_rebased ? c[n + 1] - base_cnt[i][n + 1] : c[n + 1];
            const uint32_t dh = h - prev_hit[i], dm = m - prev_miss[i];      // u32 wrap-safe
            tot_hit += dh; tot_miss += dm; win_hit += dh; win_miss += dm; lay_hit[i] += dh; lay_miss[i] += dm;
            base_cnt[i] = c;
            prev_hit[i] = 0;
            prev_miss[i] = 0;
        }
        role_rebased = true;
        tokens_since_obs = 0;
        ticks_since_obs  = 0;
        role_lcfg = lcfg;
        role_prior.assign(L.size(), {});
        for (size_t i = 0; i < L.size(); ++i) {
            const size_t n = (size_t) L[i].n_expert;
            const double * mp = i < prior.size() && prior[i].size() == n ? prior[i].data() : nullptr;
            role_prior[i].assign(n, 0.0);
            if (!role->prior || !role->prior(role_st, (int32_t) i, mp, role_prior[i].data())) {
                role_prior[i].clear();
                role_lcfg[i].prior = nullptr;
                continue;
            }
            role_lcfg[i].prior = role_prior[i].data();
        }
        ggml_pxqn_xc_cfg c = cfg_saved;
        c.layers = role_lcfg.data();
        pol->destroy(pst);
        pst = pol->create(&c);
        if (!pst) role_off("the policy could not be re-created after a move set", true);
    }

    // ---- reporting ----------------------------------------------------------------------------------------------------------

    void stats_line(const char * tag) {
        ggml_pxqn_xc_stats st = {};
        if (pol && pst) pol->stats(pst, &st);
        const double all = tot_hit + tot_miss ? 100.0*tot_hit/(double)(tot_hit + tot_miss) : 0.0;
        const double win = win_hit + win_miss ? 100.0*win_hit/(double)(win_hit + win_miss) : 0.0;
        LLAMA_LOG_INFO("PXA_XCACHE%s: tick %llu, expert hit rate %.1f%% since start (%llu routings) / %.1f%% since the last line (%llu); "
                "swaps started %llu admitted %llu evicted %llu, in flight %zu, copy waits %llu, observations %llu, adaptation cost %.0f us per decode (%.0f us per observation)\n",
                tag, (unsigned long long) tick, all, (unsigned long long)(tot_hit + tot_miss), win, (unsigned long long)(win_hit + win_miss),
                (unsigned long long) n_started, (unsigned long long) n_admitted, (unsigned long long) n_evicted, ops.size(),
                (unsigned long long) n_waits, (unsigned long long) n_obs,
                n_ticks ? (double) us_tick/(double) n_ticks : 0.0, n_obs ? (double) us_obs/(double) n_obs : 0.0);
        if (n_resets) LLAMA_LOG_INFO("PXA_XCACHE%s: %llu sequence starts back to the planned hot set, %llu swaps undone, %.1f ms in total\n",
                tag, (unsigned long long) n_resets, (unsigned long long) n_undone, us_reset/1000.0);
        if (n_skip_ticks) LLAMA_LOG_INFO("PXA_XCACHE%s: skipped %llu companion ticks (their tokens still count toward the swap budget)\n",
                tag, (unsigned long long) n_skip_ticks);
        if (n_frozen_ticks) LLAMA_LOG_INFO("PXA_XCACHE%s: %llu prefill ubatches (> %d tokens) ran on frozen residency, %llu planned swaps dropped "
                "there (PXA_XCACHE_PREFILL_FREEZE)\n", tag, (unsigned long long) n_frozen_ticks, freeze.maxw, (unsigned long long) n_frozen_drop);
        if (rtier) LLAMA_LOG_INFO("PXA_XCACHE_RTIER%s: %llu planner jobs, %llu observations skipped (planner busy), %llu planned swaps stale at start, "
                "planner %.0f us per job (off the decode thread), cap %d swaps per decode\n", tag, (unsigned long long) n_rt_jobs,
                (unsigned long long) n_rt_skip, (unsigned long long) n_rt_stale, n_rt_jobs ? (double) us_rt_worker/(double) n_rt_jobs : 0.0, swap_cap);
        if (swap_dummy) LLAMA_LOG_INFO("PXA_XCACHE_SWAP_DUMMY%s: %llu dummy swaps (bytes moved, map never flipped)\n", tag, (unsigned long long) n_dummy);
        if (tag[0] != 0 && role && role_st && role->summary) {      // the summary only (tag is never null: "" or " summary")
            ggml_pxqn_cache_host h = {};
            h.size = (uint32_t) sizeof(h);
            h.on = role_on ? 1 : 0;
            h.waits = n_role_waits; h.dropped_jobs = n_role_drop_plans; h.dropped_moves = n_role_drop;
            h.ms_all = us_role/1000.0; h.ms_moves = us_role_moves/1000.0;
            char msg[2048];
            role->summary(role_st, &h, msg, sizeof(msg));
            if (msg[0]) LLAMA_LOG_INFO("PXA_XCACHE%s: cache role: %s\n", tag, msg);
        }
        if (evict_file) LLAMA_LOG_INFO("PXA_XCACHE_EVICT_FILE%s: %llu evicts read from the model file (%llu failed), %.0f us per read (IO thread), 0 evict bytes on the link\n",
                tag, (unsigned long long) n_ef, (unsigned long long) n_ef_fail.load(),
                n_ef_done.load() ? (double) us_ef.load()/(double) n_ef_done.load() : 0.0);
        win_hit = win_miss = 0;
        if (tag && *tag && env_int("PXA_XCACHE_SUMMARY_LAYERS", 1)) {      // the summary also names where the misses are
            std::string ln;
            char b[48];
            for (size_t i = 0; i < L.size(); ++i) {
                const uint64_t t = lay_hit[i] + lay_miss[i];
                snprintf(b, sizeof(b), "%s%d:%.0f", i ? " " : "", L[i].il, t ? 100.0*lay_hit[i]/(double) t : 0.0);
                ln += b;
            }
            LLAMA_LOG_INFO("PXA_XCACHE%s: hit %% per split layer (layer:hit): %s\n", tag, ln.c_str());
        }
    }

    // back to the state init() left: undo every completed swap in reverse order (each one exactly: e_in back to the cold slot it
    // came from, e_out back to the hot slot it left), the planned spare order, fresh policy state, counters re-based, tick 0.
    // Runs between two decodes (no graph reads a map or a slot), all copies on the layer's copy stream, waited before returning.
    void reset_to_plan() {
        if (!adapt || !reset_on || init_map.empty()) return;
        const int64_t t0 = ggml_time_us();
        drain();
        bool ok = !broken;
        const size_t n_undo = applied.size();
        for (size_t k = applied.size(); ok && k-- > 0; ) {
            const op & o = applied[k];
            lay & l = L[o.li];
            auto & xl = model.pxa_xc[l.il];
            auto it = std::find(l.spare.begin(), l.spare.end(), (int32_t) o.c_in);
            if (l.map[o.e_in] != o.s || l.map[o.e_out] != -2 - o.c_free || it == l.spare.end()) { ok = false; break; }
            // both copies on the layer's copy stream, in order; a copy that cannot be enqueued leaves every expert one valid home
            if (!copy_slot(l, false, o.s, o.c_in)) { ok = false; break; }      // e_in back into its old cold slot (free: a partial copy is harmless)
            l.spare.erase(it);
            l.map[o.e_in] = -2 - o.c_in;  xl.cold[o.c_in] = o.e_in;  xl.hot[o.s] = -1;   // e_in is served cold again, the hot slot is vacant
            l.dirty = true;
            if (!copy_slot(l, true, o.s, o.c_free)) { ok = false; break; }     // e_out back into the hot slot it left
            l.map[o.e_out] = o.s;  xl.hot[o.s] = o.e_out;  xl.cold[o.c_free] = -1;
            l.spare.push_back(o.c_free);
        }
        applied.clear();
        for (int d : devs) {
            void * ev = ggml_backend_cuda_xcache_event_record(d);
            if (!ev) { ok = false; continue; }
            ggml_backend_cuda_xcache_event_sync(ev);
            ggml_backend_cuda_xcache_event_free(ev);
        }
        for (size_t i = 0; ok && i < L.size(); ++i) {
            auto sp = L[i].spare, isp = init_spare[i];
            std::sort(sp.begin(), sp.end()); std::sort(isp.begin(), isp.end());
            if (L[i].map != init_map[i] || sp != isp || model.pxa_xc[L[i].il].hot != init_hot[i] || model.pxa_xc[L[i].il].cold != init_cold[i]) ok = false;
            else L[i].spare = init_spare[i];
        }
        flush_maps();
        if (!ok) {
            if (!broken) LLAMA_LOG_WARN("PXA_XCACHE: the reset to the planned hot set did not complete; the adaptation stops here (the static plan's answers no longer repeat exactly)\n");
            broken = true;
            return;
        }
        if (read_counters()) {
            for (size_t i = 0; i < L.size(); ++i) {
                const uint32_t * b = stage + L[i].stage_off;
                const size_t n = (size_t) L[i].n_expert;
                const uint32_t dh = (b[n] - base_cnt[i][n]) - prev_hit[i], dm = (b[n + 1] - base_cnt[i][n + 1]) - prev_miss[i];   // since the last observation
                tot_hit += dh; tot_miss += dm; win_hit += dh; win_miss += dm; lay_hit[i] += dh; lay_miss[i] += dm;
                base_cnt[i].assign(b, b + n + 8);
                prev_hit[i] = 0; prev_miss[i] = 0;
            }
        }
        if (pst && pol) { pol->destroy(pst); pst = pol->create(&cfg_saved); if (!pst) broken = true; }
        tick = 0; tokens_since_obs = 0; ticks_since_obs = 0;
        chaos_rng = 0x9e3779b97f4a7c15ull;
        ++n_resets; n_undone += n_undo;
        us_reset += ggml_time_us() - t0;
        if (n_resets == 2 || (log_every > 0 && n_undo > 0)) {
            LLAMA_LOG_INFO("PXA_XCACHE: sequence start: %zu swaps undone, the planned hot set is back (%.1f ms; PXA_XCACHE_ADAPT_RESET=0 carries it across requests)\n",
                    n_undo, (ggml_time_us() - t0)/1000.0);
        }
    }

    void on_tick(int n_tokens) {
        const int64_t t0 = ggml_time_us();
        ++tick;
        if (trace) trace_tick(n_tokens);
        // PXA_XCACHE_PREFILL_FREEZE: after a prefill ubatch the residency stays as it is (no flip, no swap start); a swap in flight
        // keeps its slots (nothing points at a slot being written) and is applied at the next decode tick
        const bool frozen = adapt && llama_pxa_xc_frozen(freeze, n_tokens);
        if (frozen) ++n_frozen_ticks;
        // the cache role: its select runs at the tick it asks for, before anything of this tick moves; after a move set nothing is
        // in flight and the policy starts over at this tick's counters, so the tick's own adaptation is skipped
        if (role && role_st && role->tick(role_st, n_tokens)) {
            if (role_select()) {
                us_tick += ggml_time_us() - t0;
                ++n_ticks;
                return;
            }
        }
        if (adapt && !frozen) apply_due();
        tokens_since_obs += (uint32_t) std::max(n_tokens, 0);
        ++ticks_since_obs;
        if (rtier) {
            const bool due = ticks_since_obs >= obs_every || n_tokens >= 32;
            rt_tick(tokens_since_obs, due, frozen);
            us_tick += ggml_time_us() - t0;
            ++n_ticks;
            return;
        }
        // a prefill-sized decode is an observation of its own: its routing is the prompt's, which is what the answer will use
        if (ticks_since_obs >= obs_every || n_tokens >= 32) {
            const int64_t t1 = ggml_time_us();
            observe_and_plan(!frozen);
            us_obs += ggml_time_us() - t1;
            if (log_every > 0 && n_obs % (uint64_t) log_every == 0) stats_line("");
        }
        us_tick += ggml_time_us() - t0;
        ++n_ticks;
    }
};


static llama_model * pxa_xc_miss_model = nullptr;

// PXA_XCACHE_MISS_TOKEN (default off): the token-aware bill (src/llama-pxa-tokmiss.h). The worker reports every served cold
// request, llama_decode reports every main-model batch; the bill of a draft is the measured cold slots of its tokens.
static pxa_tokmiss g_pxa_tokmiss;
bool llama_pxa_tokmiss_on() {
    static const bool on = [] { const char * e = getenv("PXA_XCACHE_MISS_TOKEN"); const bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_XCACHE_MISS_TOKEN: 1 (draft miss bill from measured per-token cold slots)\n");
        return v; }();
    return on;
}
static double pxa_tokmiss_fallback_us() {
    static const double v = [] { const char * e = getenv("PXA_XCACHE_MISS_SLOT_US"); return e && atof(e) > 0 ? atof(e) : 60.0; }();
    return v;
}
void llama_pxa_tokmiss_note_request(const int32_t * ids, int n_used, int ntok, uint64_t us) {
    g_pxa_tokmiss.note_request(ids, n_used, ntok, us);
}
void llama_pxa_tokmiss_note_batch(const llama_token * tok, int n) { g_pxa_tokmiss.note_batch(tok, n); }
int llama_xcache_draft_keep(const llama_token * tokens, int n, int budget_us) {
    if (!llama_pxa_tokmiss_on() || n <= 1 || !tokens) return n;
    return g_pxa_tokmiss.keep(tokens, n, budget_us, pxa_tokmiss_fallback_us());
}

float llama_xcache_token_slots(llama_token tok) { return llama_pxa_tokmiss_on() ? g_pxa_tokmiss.slots(tok) : -1.f; }

int llama_xcache_draft_miss_us(const llama_token * tokens, int n) {
    if (llama_pxa_tokmiss_on() && tokens && n > 0) return g_pxa_tokmiss.bill_us(tokens, n, pxa_tokmiss_fallback_us());
    llama_model * m = pxa_xc_miss_model;
    if (!m || n <= 0 || m->pxa_xc.empty()) return 0;
    int cold = 0;
    for (const auto & lay : m->pxa_xc) if (!lay.cold.empty()) cold++;
    // Measured 484 us per cold layer at width 4. Scale by draft length. Not a per-token router yet.
    return cold * 484 * n / 4;
}

static std::mutex & pxa_xc_tick_mu() { static std::mutex mu; return mu; }

bool llama_pxa_xcache_companion_quiet(void) {
    static const bool on = (ggml_pxqn_tune_flags() & GGML_PXQN_TUNE_COMPANION) != 0;
    return on;
}

void llama_pxa_xcache_note_tokens(llama_model & model, int n_tokens) {
    if (model.pxa_xc.empty()) return;
    std::lock_guard<std::mutex> lock(pxa_xc_tick_mu());
    auto * a = model.pxa_xc_ad;
    if (!a || !a->active) return;   // not initialised yet: the first target tick does that
    a->tokens_since_obs += (uint32_t) std::max(n_tokens, 0);
    ++a->n_skip_ticks;
}

void llama_pxa_xcache_tick(llama_model & model, int n_tokens) {
    // PXA_XCACHE_CPU_OVERLAP is not implemented. Do not bump n_tokens and call it overlap.
    pxa_xc_miss_model = &model;
    if (model.pxa_xc.empty()) return;
    std::lock_guard<std::mutex> lock(pxa_xc_tick_mu());
    if (!model.pxa_xc_ad) {
        auto * a = new llama_model::pxa_xc_adaptor(model);
        a->active = a->init();
        model.pxa_xc_ad = a;
        if (a->active && a->adapt && env_int("PXA_XCACHE_ADAPT_SELFTEST", 0) != 0) a->selftest();
    }
    if (!model.pxa_xc_ad->active) return;
    model.pxa_xc_ad->on_tick(n_tokens);
}

void llama_pxa_xcache_seq_start(llama_model & model) {
    if (model.pxa_xc.empty() || !model.pxa_xc_ad) return;     // nothing has run yet: nothing moved
    std::lock_guard<std::mutex> lock(pxa_xc_tick_mu());
    if (!model.pxa_xc_ad->active) return;
    model.pxa_xc_ad->reset_to_plan();
    model.pxa_xc_ad->role_seq_start();     // the cache role's request start
}

// llama_decode's request-start hook (llama-pxa-xcache.h): the decision and the adaptor's one bit under the tick lock; then, outside
// it (llama_synchronize ticks), the work in flight completes and the start runs. An adaptor that did not start (no policy, a hot-swap
// group) takes the strict path as before: synchronize, then a llama_pxa_xcache_seq_start that does nothing.
int llama_pxa_xcache_batch_start(llama_model & model, int n_tokens, bool strict, void (*sync)(void * ud), void * ud) {
    if (model.pxa_xc.empty() || !model.pxa_xc_ad) return GGML_PXQN_CACHE_START_NONE;
    int kind = GGML_PXQN_CACHE_START_NONE;
    {
        std::lock_guard<std::mutex> lock(pxa_xc_tick_mu());
        auto * a = model.pxa_xc_ad;
        kind = a->active ? a->role_batch(n_tokens, strict) : (strict && n_tokens > 0 ? GGML_PXQN_CACHE_START_STRICT : GGML_PXQN_CACHE_START_NONE);
    }
    if (kind == GGML_PXQN_CACHE_START_NONE) return kind;
    if (sync) sync(ud);         // llama_synchronize: the batch before this one has landed and ticked
    if (kind == GGML_PXQN_CACHE_START_STRICT) {
        llama_pxa_xcache_seq_start(model);
        return kind;
    }
    std::lock_guard<std::mutex> lock(pxa_xc_tick_mu());     // the cache role's own start, never reset_to_plan
    auto * a = model.pxa_xc_ad;
    return a && a->active && a->role_seq_start(true) ? kind : GGML_PXQN_CACHE_START_NONE;
}

bool llama_pxa_xcache_stats(const struct llama_model * model, uint64_t * hits, uint64_t * misses,
        uint64_t * swaps_started, uint64_t * swaps_admitted, uint64_t * swaps_evicted) {
    if (!model || !model->pxa_xc_ad || !model->pxa_xc_ad->active) return false;
    const auto * a = model->pxa_xc_ad;
    if (hits)           *hits           = a->tot_hit;
    if (misses)         *misses         = a->tot_miss;
    if (swaps_started)  *swaps_started  = a->n_started;
    if (swaps_admitted) *swaps_admitted = a->n_admitted;
    if (swaps_evicted)  *swaps_evicted  = a->n_evicted;
    return true;
}

bool llama_pxa_xcache_evict_file_stats(const llama_model * model, uint64_t * reads, uint64_t * failed) {
    if (!model || !model->pxa_xc_ad || !model->pxa_xc_ad->active || !model->pxa_xc_ad->evict_file) return false;
    auto * a = model->pxa_xc_ad;
    if (reads)  *reads  = a->n_ef;
    if (failed) *failed = a->n_ef_fail.load();
    return true;
}

bool llama_pxa_xcache_rtier_stats(const llama_model * model, uint64_t * jobs, uint64_t * stale) {
    if (!model || !model->pxa_xc_ad || !model->pxa_xc_ad->active || !model->pxa_xc_ad->rtier) return false;
    auto * a = model->pxa_xc_ad;
    std::lock_guard<std::mutex> lk(a->rt_mu);
    if (jobs)  *jobs  = a->n_rt_jobs;
    if (stale) *stale = a->n_rt_stale;
    return true;
}

bool llama_pxa_xcache_role_stats(const llama_model * model, llama_pxa_xc_role_stats * out) {
    if (!model || !model->pxa_xc_ad || !model->pxa_xc_ad->active) return false;
    auto * a = model->pxa_xc_ad;
    if (out) {
        out->on            = a->role_on;
        out->waits         = a->n_role_waits;
        out->dropped_plans = a->n_role_drop_plans;
        out->dropped_swaps = a->n_role_drop;
        out->ms            = a->us_role/1000.0;
        out->role          = {};
        out->role.size     = (uint32_t) sizeof(out->role);
        if (a->role && a->role_st && a->role->stats) a->role->stats(a->role_st, &out->role);
    }
    return true;
}

int llama_pxa_xcache_check(const llama_model * model) {
    if (!model || !model->pxa_xc_ad || !model->pxa_xc_ad->active) return -1;
    return model->pxa_xc_ad->check_tables();
}

void llama_pxa_xcache_adaptor_free(llama_model & model) {
    if (!model.pxa_xc_ad) return;
    auto * a = model.pxa_xc_ad;
    model.pxa_xc_ad = nullptr;
    if (a->active) {
        a->rt_shutdown();     // PXA_XCACHE_RTIER: the planner thread stops before the summary reads the policy
        a->drain();
        a->io_shutdown();     // PXA_XCACHE_EVICT_FILE: after drain (drain waits for the reads in flight)
        if (a->n_obs > 0 && (a->log_every > 0 || getenv("PXA_XCACHE_SUMMARY"))) a->stats_line(" summary");
    }
    delete a;
}

#endif
