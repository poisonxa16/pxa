// pxa / PXA expert cache: skip the cold tail of a decode / verify window -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache-skip.cpp -- PXA_XCACHE_SKIP_COLD_RANK: the lever, its step counter and its log (see llama-pxa-xcache-skip.h).
#include "llama-pxa-xcache-skip.h"

#include "llama-impl.h"
#include "llama-model.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda-xcache.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

int llama_pxa_xcache_skip_cold_maxtok(void) {
    static const int n = [] {
        const char * e = getenv("PXA_XCACHE_SKIP_COLD_MAXTOK");
        return e && *e && atoi(e) > 0 ? atoi(e) : 16;
    }();
    return n;
}

int llama_pxa_xcache_skip_cold_rank(void) {
    static const int r = [] {
        const char * e = getenv("PXA_XCACHE_SKIP_COLD_RANK");
        const int v = e && *e ? atoi(e) : 0;
        if (v > 0) {
            LLAMA_LOG_INFO("PXA_XCACHE_SKIP_COLD_RANK: %d (CHANGES OUTPUT: a cold expert that every token of a decode / verify window of <= %d "
                           "tokens ranks %d or worse by routing weight is not computed and adds 0, no renormalisation; resident experts "
                           "always run; not in the MTP draft)\n", v, llama_pxa_xcache_skip_cold_maxtok(), v);
        }
        return v > 0 ? v : 0;
    }();
    return r;
}

#ifndef GGML_USE_CUDA

void llama_pxa_xcache_skip_note_step(const llama_model &, int) {}
void llama_pxa_xcache_skip_summary(const llama_model &) {}

#else

namespace {

std::atomic<uint64_t> g_steps{0}, g_tokens{0};
std::mutex g_mu;
unsigned long long g_last[GGML_CUDA_SKIP_STAT_N] = {};
uint64_t g_last_steps = 0;

int log_every() {
    static const int n = [] { const char * e = getenv("PXA_XCACHE_SKIP_COLD_LOG"); return e && *e && atoi(e) > 0 ? atoi(e) : 0; }();
    return n;
}

// the counters summed over the devices that hold a split layer
bool read_stats(const llama_model & model, unsigned long long out[GGML_CUDA_SKIP_STAT_N]) {
    std::fill(out, out + GGML_CUDA_SKIP_STAT_N, 0ull);
    std::vector<int> devs;
    for (const auto & f : model.pxa_xc_fills) if (std::find(devs.begin(), devs.end(), f.dev) == devs.end()) devs.push_back(f.dev);
    bool any = false;
    for (int d : devs) {
        unsigned long long s[GGML_CUDA_SKIP_STAT_N];
        if (!ggml_backend_cuda_xcache_skip_stats(d, s)) continue;
        for (int i = 0; i < GGML_CUDA_SKIP_STAT_N; ++i) out[i] += s[i];
        any = true;
    }
    return any;
}

double ratio(unsigned long long a, unsigned long long b) { return b ? (double) a / (double) b : 0.0; }

}

void llama_pxa_xcache_skip_note_step(const llama_model & model, int n_tokens) {
    if (n_tokens < 1 || n_tokens > llama_pxa_xcache_skip_cold_maxtok() || llama_pxa_xcache_skip_cold_rank() == 0 || model.pxa_xc.empty()) return;
    const uint64_t steps = g_steps.fetch_add(1) + 1;
    g_tokens.fetch_add((uint64_t) n_tokens);
    const int every = log_every();
    if (every <= 0 || steps % (uint64_t) every != 0) return;
    std::lock_guard<std::mutex> lock(g_mu);
    unsigned long long s[GGML_CUDA_SKIP_STAT_N];
    if (!read_stats(model, s)) return;
    unsigned long long d[GGML_CUDA_SKIP_STAT_N];
    for (int i = 0; i < GGML_CUDA_SKIP_STAT_N; ++i) d[i] = s[i] - g_last[i];
    const uint64_t ds = steps - g_last_steps;
    LLAMA_LOG_INFO("PXA_XCACHE_SKIP_COLD: steps %llu..%llu (last width %d): skipped %llu of %llu cold experts (%.1f%%, %.2f per step), "
                   "%llu of %llu cold slots, in %llu of %llu layer windows\n",
                   (unsigned long long) (g_last_steps + 1), (unsigned long long) steps, n_tokens,
                   d[GGML_CUDA_SKIP_STAT_SKIP_EXPERTS], d[GGML_CUDA_SKIP_STAT_COLD_EXPERTS],
                   100.0*ratio(d[GGML_CUDA_SKIP_STAT_SKIP_EXPERTS], d[GGML_CUDA_SKIP_STAT_COLD_EXPERTS]),
                   ratio(d[GGML_CUDA_SKIP_STAT_SKIP_EXPERTS], ds),
                   d[GGML_CUDA_SKIP_STAT_SKIP_SLOTS], d[GGML_CUDA_SKIP_STAT_COLD_SLOTS],
                   d[GGML_CUDA_SKIP_STAT_WINDOWS_HIT], d[GGML_CUDA_SKIP_STAT_WINDOWS]);
    std::copy(s, s + GGML_CUDA_SKIP_STAT_N, g_last);
    g_last_steps = steps;
}

void llama_pxa_xcache_skip_summary(const llama_model & model) {
    if (llama_pxa_xcache_skip_cold_rank() == 0 || model.pxa_xc.empty()) return;
    std::lock_guard<std::mutex> lock(g_mu);
    unsigned long long s[GGML_CUDA_SKIP_STAT_N];
    if (!read_stats(model, s)) {
        LLAMA_LOG_INFO("PXA_XCACHE_SKIP_COLD summary: counters not readable\n");
        return;
    }
    const uint64_t steps = g_steps.load(), toks = g_tokens.load();
    const unsigned long long cx = s[GGML_CUDA_SKIP_STAT_COLD_EXPERTS], sx = s[GGML_CUDA_SKIP_STAT_SKIP_EXPERTS];
    LLAMA_LOG_INFO("PXA_XCACHE_SKIP_COLD summary: rank >= %d dropped on graphs of <= %d tokens; %llu decode/verify steps (%llu tokens), "
                   "%llu layer windows (%llu tokens), %llu with a drop; cold experts skipped %llu of %llu (%.1f%%; per step %.2f of %.2f, "
                   "per window %.3f of %.3f); cold slots skipped %llu of %llu (%.1f%%)\n",
                   llama_pxa_xcache_skip_cold_rank(), llama_pxa_xcache_skip_cold_maxtok(),
                   (unsigned long long) steps, (unsigned long long) toks,
                   s[GGML_CUDA_SKIP_STAT_WINDOWS], s[GGML_CUDA_SKIP_STAT_TOKENS], s[GGML_CUDA_SKIP_STAT_WINDOWS_HIT],
                   sx, cx, 100.0*ratio(sx, cx), ratio(sx, steps), ratio(cx, steps),
                   ratio(sx, s[GGML_CUDA_SKIP_STAT_WINDOWS]), ratio(cx, s[GGML_CUDA_SKIP_STAT_WINDOWS]),
                   s[GGML_CUDA_SKIP_STAT_SKIP_SLOTS], s[GGML_CUDA_SKIP_STAT_COLD_SLOTS],
                   100.0*ratio(s[GGML_CUDA_SKIP_STAT_SKIP_SLOTS], s[GGML_CUDA_SKIP_STAT_COLD_SLOTS]));
}

#endif
