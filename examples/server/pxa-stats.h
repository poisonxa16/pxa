#pragma once

// PXA_SPEED_STATS_v1 (2026-09-27): a built-in speed history for llama-server.
//
// One record per finished completion (prefill + decode t/s, prompt size, cache hits, draft
// acceptance, slot, split mode and card count), kept in a bounded in-memory ring and served by
//   GET /pxa/stats  -- JSON records (since=<unix s>, model=<name>, limit=<n>) + per-model medians
//   GET /pxa/speed  -- a self-contained page that charts them (public_pxa/speed.html)
// The numbers are the ones the server already computed for the response's `timings`
// (server_slot::get_timings()); nothing is re-timed. Everything stays on this machine.
//
// Levers:
//   PXA_STATS=0            the whole feature off: no ring, no file, both routes unregistered
//   PXA_STATS_MAX=<n>      ring capacity (default 10000 records)
//   PXA_STATS_FILE=<path>  also append each record as a JSONL line and reload the tail on start;
//                          "default" = $XDG_CACHE_HOME/pxa/speed-stats.jsonl (else ~/.cache/...).
//                          Unset = memory only (no file is ever written unless asked for).
//   PXA_STATS_FILE_MB=<n>  rotate the file at n MB (default 8): <path> -> <path>.1 (one generation)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct pxa_stat_rec {
    double      ts         = 0.0;   // unix seconds (wall clock, ms resolution)
    std::string model;              // model file basename
    int         slot       = -1;
    int         n_prompt   = 0;     // prompt tokens of the request
    int         n_cached   = 0;     // of those, reused from the prompt cache
    int         prompt_n   = 0;     // tokens actually prefilled (timings.prompt_n)
    double      prompt_ms  = 0.0;
    double      prefill_tps = 0.0;  // timings.prompt_per_second (0 when nothing was prefilled)
    int         n_gen      = 0;     // timings.predicted_n
    double      gen_ms     = 0.0;
    double      decode_tps = 0.0;   // timings.predicted_per_second (0 when nothing was generated)
    int         draft_n    = 0;     // speculative: proposed / accepted (0/0 when spec is off)
    int         draft_acc  = 0;
    std::string split;              // split mode at load: none/layer/attn/graph/tensor
    int         n_gpu      = -1;    // cards the model was loaded across (-1 = the load-time count)

    nlohmann::ordered_json to_json() const {
        nlohmann::ordered_json j;
        j["ts"] = ts; j["model"] = model; j["slot"] = slot;
        j["n_prompt"] = n_prompt; j["n_cached"] = n_cached; j["prompt_n"] = prompt_n;
        j["prompt_ms"] = prompt_ms; j["prefill_tps"] = prefill_tps;
        j["n_gen"] = n_gen; j["gen_ms"] = gen_ms; j["decode_tps"] = decode_tps;
        if (draft_n > 0) { j["draft_n"] = draft_n; j["draft_acc"] = draft_acc; }
        j["split"] = split; j["n_gpu"] = n_gpu;
        return j;
    }
    static bool from_json(const nlohmann::json & j, pxa_stat_rec & r) {
        try {
            r.ts = j.at("ts").get<double>();
            r.model = j.value("model", std::string());
            r.slot = j.value("slot", -1);
            r.n_prompt = j.value("n_prompt", 0); r.n_cached = j.value("n_cached", 0);
            r.prompt_n = j.value("prompt_n", 0);
            r.prompt_ms = j.value("prompt_ms", 0.0); r.prefill_tps = j.value("prefill_tps", 0.0);
            r.n_gen = j.value("n_gen", 0); r.gen_ms = j.value("gen_ms", 0.0);
            r.decode_tps = j.value("decode_tps", 0.0);
            r.draft_n = j.value("draft_n", 0); r.draft_acc = j.value("draft_acc", 0);
            r.split = j.value("split", std::string()); r.n_gpu = j.value("n_gpu", 0);
            return true;
        } catch (...) { return false; }
    }
};

inline const char * pxa_stats_split_name(int mode) {
    switch (mode) {
        case 0: return "none";  case 1: return "layer"; case 2: return "attn";
        case 3: return "graph"; case 4: return "tensor";
        default: return "?";
    }
}

// Prefill medians and the prefill chart skip requests that prefilled fewer tokens than this (a
// prompt-cache hit re-evaluates one token: that is latency, not prefill throughput). The record
// itself still carries the response's own number.
static constexpr int PXA_STATS_MIN_PREFILL = 64;

inline double pxa_stats_finite(double v) { return std::isfinite(v) && v > 0.0 ? v : 0.0; }

inline const char * pxa_stats_bucket(int n_prompt) {
    if (n_prompt < 1024)  return "<1k";
    if (n_prompt < 4096)  return "1-4k";
    if (n_prompt < 16384) return "4-16k";
    return ">16k";
}

inline double pxa_stats_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    double hi = v[m];
    if (v.size() % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + m);
    return 0.5 * (lo + hi);
}

class pxa_stats {
public:
    static pxa_stats & get() { static pxa_stats s; return s; }

    bool enabled() const { return enabled_; }
    size_t capacity() const { return cap_; }
    const std::string & file() const { return path_; }

    // Load-time facts (server.cpp, once after the model is up).
    void set_load_info(const std::string & split, int n_gpu) {
        std::lock_guard<std::mutex> lk(mu_);
        split_ = split; n_gpu_ = n_gpu;
    }

    // Called from the server loop when a completion's final result is sent. Cheap: one lock, one
    // push_back, and (only with PXA_STATS_FILE) one buffered line write.
    void record(pxa_stat_rec r) {
        if (!enabled_) return;
        r.prefill_tps = pxa_stats_finite(r.prefill_tps);
        r.decode_tps  = pxa_stats_finite(r.decode_tps);
        if (!std::isfinite(r.prompt_ms)) r.prompt_ms = 0.0;
        if (!std::isfinite(r.gen_ms))    r.gen_ms = 0.0;
        std::lock_guard<std::mutex> lk(mu_);
        if (r.split.empty()) r.split = split_;
        if (r.n_gpu < 0)     r.n_gpu = n_gpu_;
        if (!path_.empty()) append_line_locked(r);
        ring_.push_back(std::move(r));
        while (ring_.size() > cap_) ring_.pop_front();
    }

    // Filtered copy (since = unix s, model = exact name or empty, limit = newest n or 0 = all).
    std::vector<pxa_stat_rec> query(double since, const std::string & model, size_t limit) const {
        std::vector<pxa_stat_rec> out;
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto & r : ring_) {
            if (r.ts < since) continue;
            if (!model.empty() && r.model != model) continue;
            out.push_back(r);
        }
        if (limit > 0 && out.size() > limit) out.erase(out.begin(), out.end() - limit);
        return out;
    }

    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return ring_.size(); }

    // Per-model medians, overall and per prompt-size bucket.
    static nlohmann::ordered_json summarize(const std::vector<pxa_stat_rec> & recs) {
        struct acc { std::vector<double> dec, pre, acc; size_t n = 0; };
        std::map<std::string, acc> by_model;
        std::map<std::string, std::map<std::string, acc>> by_bucket;
        for (const auto & r : recs) {
            for (acc * a : { &by_model[r.model], &by_bucket[r.model][pxa_stats_bucket(r.n_prompt)] }) {
                a->n++;
                if (r.decode_tps  > 0.0) a->dec.push_back(r.decode_tps);
                if (r.prefill_tps > 0.0 && r.prompt_n >= PXA_STATS_MIN_PREFILL) a->pre.push_back(r.prefill_tps);
                if (r.draft_n > 0) a->acc.push_back((double) r.draft_acc / r.draft_n);
            }
        }
        auto row = [](const acc & a) {
            nlohmann::ordered_json j;
            j["requests"] = a.n;
            j["decode_tps_median"]  = pxa_stats_median(a.dec);
            j["prefill_tps_median"] = pxa_stats_median(a.pre);
            if (!a.acc.empty()) j["draft_accept_median"] = pxa_stats_median(a.acc);
            return j;
        };
        nlohmann::ordered_json out = nlohmann::ordered_json::object();
        for (const auto & [m, a] : by_model) {
            nlohmann::ordered_json j = row(a);
            nlohmann::ordered_json b = nlohmann::ordered_json::object();
            for (const char * k : { "<1k", "1-4k", "4-16k", ">16k" }) {
                auto it = by_bucket[m].find(k);
                if (it != by_bucket[m].end()) b[k] = row(it->second);
            }
            j["by_prompt_size"] = b;
            out[m] = j;
        }
        return out;
    }

private:
    pxa_stats() {
        const char * e = std::getenv("PXA_STATS");
        enabled_ = !(e && (std::string(e) == "0" || std::string(e) == "off" || std::string(e) == "false"));
        if (!enabled_) return;
        if (const char * m = std::getenv("PXA_STATS_MAX")) {
            const long v = std::strtol(m, nullptr, 10);
            if (v > 0) cap_ = (size_t) v;
        }
        if (const char * mb = std::getenv("PXA_STATS_FILE_MB")) {
            const double v = std::strtod(mb, nullptr);
            if (v > 0) rotate_bytes_ = (uintmax_t) (v * 1024.0 * 1024.0);
        }
        if (const char * f = std::getenv("PXA_STATS_FILE")) {
            std::string p = f;
            if (p == "default" || p == "1") p = default_path();
            if (!p.empty() && p != "0") {
                std::error_code ec;
                const auto parent = std::filesystem::path(p).parent_path();
                if (!parent.empty()) std::filesystem::create_directories(parent, ec);
                path_ = p;
                load_tail();
                out_.open(path_, std::ios::app);
                if (!out_) {
                    fprintf(stderr, "pxa-stats: cannot open %s for append; keeping memory only\n", path_.c_str());
                    path_.clear();
                }
            }
        }
    }

    static std::string default_path() {
        std::string base;
        if (const char * x = std::getenv("XDG_CACHE_HOME"); x && *x) base = x;
        else if (const char * h = std::getenv("HOME"); h && *h) base = std::string(h) + "/.cache";
        if (base.empty()) return "";
        return base + "/pxa/speed-stats.jsonl";
    }

    void load_tail() {
        for (const std::string & p : { path_ + ".1", path_ }) {
            std::ifstream in(p);
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                const auto j = nlohmann::json::parse(line, nullptr, false);
                if (j.is_discarded() || !j.is_object()) continue;
                pxa_stat_rec r;
                if (pxa_stat_rec::from_json(j, r)) {
                    ring_.push_back(std::move(r));
                    if (ring_.size() > cap_) ring_.pop_front();
                }
            }
        }
    }

    void append_line_locked(const pxa_stat_rec & r) {
        out_ << r.to_json().dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
        out_.flush();
        std::error_code ec;
        const uintmax_t sz = std::filesystem::file_size(path_, ec);
        if (!ec && sz >= rotate_bytes_) {
            out_.close();
            std::filesystem::rename(path_, path_ + ".1", ec);
            out_.open(path_, std::ios::trunc);
        }
    }

    bool enabled_ = true;
    size_t cap_ = 10000;
    uintmax_t rotate_bytes_ = 8ull * 1024 * 1024;
    std::string path_;
    std::ofstream out_;
    std::string split_ = "?";
    int n_gpu_ = 0;
    mutable std::mutex mu_;
    std::deque<pxa_stat_rec> ring_;
};
