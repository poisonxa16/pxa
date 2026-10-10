// pxa / PXA_XCACHE_MISS_TOKEN: a token-aware cold-expert miss bill for MTP drafts -- authored by PXA Network.
//
// The host worker sees, for every layer step with a cold expert, the cold ids of every row of the batch (h_ids). Each row of
// a decode / verify batch is one input token, so after the step the engine knows how many cold slots each input token paid
// over all split layers. pxa_tokmiss keeps an EMA of that count per token id (measured, not assumed). A draft token d will
// be an input row of the verify batch, so its expected miss bill is ema[d] cold slots times the live cost of one cold slot
// (worker wall us / cold slots served). Unseen tokens get the running mean. Header-only so the unit test needs no model.
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

struct pxa_tokmiss {
    static constexpr int MAXR = 16;
    std::atomic<uint32_t> row_cold[MAXR];       // cold slots per batch row of the step in flight (worker adds)
    std::atomic<uint64_t> us_total{0}, slots_total{0};
    std::mutex mu;
    std::unordered_map<int32_t, float> ema;     // token -> cold slots per step as an input row
    std::vector<int32_t> pending;               // the tokens of the step in flight
    double mean = 0.0; uint64_t n_rows = 0;
    float alpha = 0.3f;
    pxa_tokmiss() { for (auto & r : row_cold) r.store(0); }

    // worker: one served request (a layer step) with ids [n_used x ntok], ids >= 0 = cold slot; us = its CPU wall time
    void note_request(const int32_t * ids, int n_used, int ntok, uint64_t us) {
        uint64_t s = 0;
        for (int t = 0; t < ntok && t < MAXR; ++t) {
            uint32_t c = 0;
            for (int k = 0; k < n_used; ++k) c += ids[k + t*n_used] >= 0;
            if (c) row_cold[t].fetch_add(c, std::memory_order_relaxed);
            s += c;
        }
        slots_total.fetch_add(s, std::memory_order_relaxed);
        us_total.fetch_add(us, std::memory_order_relaxed);
    }
    // engine: a new main-model step starts with these input tokens; the previous step is complete, so fold it in
    void note_batch(const int32_t * tok, int n) {
        std::lock_guard<std::mutex> lk(mu);
        for (int t = 0; t < (int) pending.size() && t < MAXR; ++t) {
            const float c = (float) row_cold[t].exchange(0, std::memory_order_relaxed);
            auto it = ema.find(pending[t]);
            if (it == ema.end()) ema.emplace(pending[t], c); else it->second += alpha*(c - it->second);
            ++n_rows; mean += (c - mean)/(double) n_rows;
        }
        for (int t = (int) pending.size(); t < MAXR; ++t) row_cold[t].store(0, std::memory_order_relaxed);
        pending.assign(tok, tok + (n > 0 && tok ? n : 0));
        if (pending.size() > (size_t) MAXR) pending.clear();   // prefill: not a draft-shaped step
    }
    double us_per_slot(double fallback) const {
        const uint64_t s = slots_total.load(), u = us_total.load();
        return s >= 64 ? (double) u/(double) s : fallback;
    }
    float slots(int32_t tok) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = ema.find(tok);
        return it == ema.end() ? (float) mean : it->second;
    }
    // bill of the draft tokens in microseconds
    int bill_us(const int32_t * d, int n, double fallback_us) {
        const double ups = us_per_slot(fallback_us);
        double b = 0; for (int i = 0; i < n; ++i) b += slots(d[i])*ups;
        return (int) (b + 0.5);
    }
    // longest prefix (>= 1) whose cumulative bill stays within budget_us
    int keep(const int32_t * d, int n, int budget_us, double fallback_us) {
        if (n <= 1) return n;
        const double ups = us_per_slot(fallback_us);
        double b = 0; int k = 0;
        for (; k < n; ++k) { b += slots(d[k])*ups; if (k >= 1 && b > budget_us) break; }
        return k < 1 ? 1 : k;
    }
};
