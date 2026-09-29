#pragma once

// =============================================================================================
// PXA HOT SWAP -- the router (header-only; the engine half is ggml/src/pxa-residency.h).
//
// One server process serves several models. Every registered model keeps its weights in pinned
// host RAM; exactly one of them is on the cards. A request's `model` field picks the model: if it
// is not the one on the cards, the router waits until the active model has no request in flight,
// PARKS it (its KV cache and slots go to host RAM with it), UNPARKS the target (every card loads
// its own share over its own link at the same time), and only then lets the request run. Nothing
// is ever served by a model that is not fully back on the cards.
//
// Scheduling is FIFO by request with batching per model:
//   * a request for the active model runs at once, unless requests for other models are already
//     waiting -- then it queues behind them, so a busy model cannot starve the others;
//   * when a swap brings model M in, EVERY queued request for M is admitted together, so a burst
//     for one model costs one swap, not one per request;
//   * a swap starts only when the active model has drained (no request in flight).
// A failed swap fails the requests that wanted the target and puts the previous model back.
//
// The router knows nothing about llama: the server gives it park/unpark callbacks (swap_ops) that
// run on each model's own task loop. tests/test-pxa-hotswap.cpp drives it with fakes.
// =============================================================================================

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace pxa_hs {

// ---- registration ------------------------------------------------------------------------------
// --hot-model 'NAME=PATH [llama-server flags for this model]'
struct model_spec {
    std::string              name;
    std::string              path;
    std::vector<std::string> args;   // extra flags, applied on top of the shared ones
};

inline bool parse_model_spec(const std::string & arg, model_spec & out, std::string & err) {
    out = model_spec();
    // split on whitespace (no quoting: paths with spaces are not supported here)
    std::vector<std::string> tok;
    std::string cur;
    for (char c : arg) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!cur.empty()) { tok.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) tok.push_back(cur);
    if (tok.empty()) {
        err = "empty --hot-model value";
        return false;
    }
    const size_t eq = tok[0].find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= tok[0].size()) {
        err = "--hot-model wants NAME=PATH [flags], got '" + tok[0] + "'";
        return false;
    }
    out.name = tok[0].substr(0, eq);
    out.path = tok[0].substr(eq + 1);
    for (size_t i = 1; i < tok.size(); ++i) out.args.push_back(tok[i]);
    for (const char * bad : {"-m", "--model", "-a", "--alias", "--hot-model", "--port", "--host"}) {
        for (const auto & a : out.args) {
            if (a == bad) {
                err = "--hot-model '" + out.name + "': flag " + a + " cannot be set per model";
                return false;
            }
        }
    }
    return true;
}

// the id a registered model answers to, and the forms of it a client may send
inline std::string file_stem(const std::string & path) {
    size_t s = path.find_last_of("/\\");
    std::string f = s == std::string::npos ? path : path.substr(s + 1);
    const std::string ext = ".gguf";
    if (f.size() > ext.size() && f.compare(f.size() - ext.size(), ext.size(), ext) == 0) {
        f = f.substr(0, f.size() - ext.size());
    }
    return f;
}

// ---- swapping ----------------------------------------------------------------------------------
struct swap_ops {
    virtual ~swap_ops() = default;
    // Both block until done and run the work on the model's own task loop. `next` is the model that
    // will come in (the ops may keep part of idx on the cards if next fits beside it).
    virtual bool park  (int idx, int next, std::string & err) = 0;
    virtual bool unpark(int idx, std::string & err) = 0;
};

struct swap_record {
    int         from = -1;
    int         to   = -1;
    bool        ok   = false;
    double      ms_park   = 0.0;
    double      ms_unpark = 0.0;
    double      ms_total  = 0.0;
    std::string err;
    int64_t     t_end_ms  = 0;   // steady clock, ms
};

struct model_status {
    std::string name;
    bool        active   = false;
    int         inflight = 0;    // leases held (only the active model has any)
    int         waiting  = 0;    // requests queued for it
    uint64_t    n_requests = 0;  // leases granted
    uint64_t    n_swaps_in = 0;
    uint64_t    n_swap_fail = 0;
    double      ms_last_unpark = 0.0;
    double      ms_last_park   = 0.0;
};

inline int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class router {
public:
    router(std::vector<std::vector<std::string>> names, int active, swap_ops * ops)
        : names_(std::move(names)), ops_(ops), active_(active) {
        stats_.resize(names_.size());
        for (size_t i = 0; i < names_.size(); ++i) {
            stats_[i].name = names_[i].empty() ? std::string() : names_[i][0];
        }
    }

    size_t size() const { return names_.size(); }
    const std::string & name(int idx) const { return stats_[idx].name; }

    // -1: no such model; the caller decides what an empty field means
    int find(const std::string & name) const {
        if (name.empty()) return -1;
        for (size_t i = 0; i < names_.size(); ++i) {
            for (const auto & n : names_[i]) {
                if (n == name) return (int) i;
            }
        }
        return -1;
    }

    // Block until model idx is on the cards and pinned for the caller. On success the caller owns
    // one lease and MUST call release(idx) exactly once. `cancelled` is polled while waiting (the
    // client went away); `timeout_ms` < 0 waits forever.
    bool acquire(int idx, std::string & err, const std::function<bool()> & cancelled = nullptr,
                 int64_t timeout_ms = -1) {
        if (idx < 0 || idx >= (int) names_.size()) {
            err = "no such model";
            return false;
        }
        std::unique_lock<std::mutex> lk(mu_);
        if (!swapping_ && active_ == idx && queue_.empty()) {
            ++inflight_;
            ++stats_[idx].n_requests;
            return true;
        }
        ticket t;
        t.idx = idx;
        queue_.push_back(&t);
        const int64_t t0 = steady_ms();
        if (log_) {
            log_("request for '" + stats_[idx].name + "' waits: " + describe_locked());
        }
        for (;;) {
            if (t.granted) {
                return true;
            }
            if (t.failed) {
                err = t.err;
                return false;
            }
            if ((cancelled && cancelled()) || (timeout_ms >= 0 && steady_ms() - t0 > timeout_ms)) {
                const bool timed_out = !(cancelled && cancelled());
                drop_locked(&t);
                cv_.notify_all();
                err = timed_out ? "timed out waiting for model '" + stats_[idx].name + "'" : "client closed the connection";
                return false;
            }
            if (!swapping_ && !queue_.empty() && queue_.front() == &t) {
                if (active_ == idx) {
                    grant_all_locked(idx);
                    continue;
                }
                if (inflight_ == 0) {
                    swap_locked(lk, idx);
                    continue;
                }
            }
            cv_.wait_for(lk, std::chrono::milliseconds(50));
        }
    }

    // A request that names no model: served by whatever model is on the cards when its turn comes.
    // It takes its place in the FIFO like any other request (so a stream of unnamed requests cannot
    // starve a waiting swap) and is admitted with the batch of any swap that completes ahead of it.
    // Returns the model index (release(idx) later), or -1 with err.
    int acquire_any(std::string & err, const std::function<bool()> & cancelled = nullptr) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!swapping_ && active_ >= 0 && queue_.empty()) {
            ++inflight_;
            ++stats_[active_].n_requests;
            return active_;
        }
        ticket t;
        t.idx = -1;
        queue_.push_back(&t);
        for (;;) {
            if (t.granted) {
                return t.granted_idx;
            }
            if (t.failed) {
                err = t.err;
                return -1;
            }
            if (cancelled && cancelled()) {
                drop_locked(&t);
                cv_.notify_all();
                err = "client closed the connection";
                return -1;
            }
            if (!swapping_ && !queue_.empty() && queue_.front() == &t) {
                if (active_ < 0) {
                    drop_locked(&t);
                    cv_.notify_all();
                    err = "no model is on the cards (the last swap failed); name a model to load one";
                    return -1;
                }
                t.granted     = true;
                t.granted_idx = active_;
                ++inflight_;
                ++stats_[active_].n_requests;
                drop_locked(&t);
                cv_.notify_all();
                continue;
            }
            cv_.wait_for(lk, std::chrono::milliseconds(50));
        }
    }

    // Pin whatever model is on the cards now (monitoring endpoints). Waits out a swap in progress.
    // Returns the index, or -1 when no model is on the cards; release(idx) when >= 0.
    int acquire_active() {
        std::unique_lock<std::mutex> lk(mu_);
        while (swapping_) {
            cv_.wait(lk);
        }
        if (active_ >= 0) {
            ++inflight_;
        }
        return active_;
    }

    void release(int idx) {
        std::lock_guard<std::mutex> lk(mu_);
        if (idx != active_ || inflight_ <= 0) {
            // a lease for a model that is not active cannot exist; say so loudly, never go negative
            fprintf(stderr, "pxa hot swap: release of a lease on model %d (active %d, in flight %d)\n",
                    idx, active_, inflight_);
            return;
        }
        --inflight_;
        if (inflight_ == 0) {
            cv_.notify_all();
        }
    }

    int active() const {
        std::lock_guard<std::mutex> lk(mu_);
        return active_;
    }

    bool swapping() const {
        std::lock_guard<std::mutex> lk(mu_);
        return swapping_;
    }

    std::vector<model_status> status() const {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<model_status> out = stats_;
        for (size_t i = 0; i < out.size(); ++i) {
            out[i].active   = (int) i == active_;
            out[i].inflight = (int) i == active_ ? inflight_ : 0;
            out[i].waiting  = 0;
        }
        for (const auto * t : queue_) {
            if (t->idx >= 0) out[t->idx].waiting++;
        }
        return out;
    }

    swap_record last_swap() const {
        std::lock_guard<std::mutex> lk(mu_);
        return last_;
    }

    uint64_t n_swaps() const {
        std::lock_guard<std::mutex> lk(mu_);
        return n_swaps_;
    }

    void set_logger(std::function<void(const std::string &)> f) { log_ = std::move(f); }

private:
    struct ticket {
        int         idx = -1;          // -1: any model (acquire_any)
        int         granted_idx = -1;
        bool        granted = false;
        bool        failed  = false;
        std::string err;
    };

    std::string describe_locked() const {
        std::string s = "on the cards: " + (active_ >= 0 ? "'" + stats_[active_].name + "'" : std::string("none"));
        s += ", " + std::to_string(inflight_) + " in flight, " + std::to_string(queue_.size()) + " queued";
        if (swapping_) s += ", swapping to '" + stats_[swap_to_].name + "'";
        return s;
    }

    void drop_locked(ticket * t) {
        queue_.erase(std::remove(queue_.begin(), queue_.end(), t), queue_.end());
    }

    // admit every queued request for idx, and every "any model" request, onto idx
    void grant_all_locked(int idx) {
        for (auto it = queue_.begin(); it != queue_.end();) {
            if ((*it)->idx == idx || (*it)->idx < 0) {
                (*it)->granted     = true;
                (*it)->granted_idx = idx;
                ++inflight_;
                ++stats_[idx].n_requests;
                it = queue_.erase(it);
            } else {
                ++it;
            }
        }
        cv_.notify_all();
    }

    void fail_all_locked(int idx, const std::string & err) {
        for (auto it = queue_.begin(); it != queue_.end();) {
            if ((*it)->idx == idx) {
                (*it)->failed = true;
                (*it)->err    = err;
                it = queue_.erase(it);
            } else {
                ++it;
            }
        }
        cv_.notify_all();
    }

    // Called with the lock held, by the waiter at the head of the queue, when the active model has
    // drained. Runs park/unpark without the lock; nobody else can start a swap meanwhile.
    void swap_locked(std::unique_lock<std::mutex> & lk, int to) {
        swapping_ = true;
        swap_to_  = to;
        const int from = active_;
        if (log_) {
            log_("swap '" + (from >= 0 ? stats_[from].name : std::string("none")) + "' -> '" + stats_[to].name + "' begins");
        }
        lk.unlock();

        swap_record rec;
        rec.from = from;
        rec.to   = to;
        const auto t0 = std::chrono::steady_clock::now();
        std::string err;
        bool parked = true;
        if (from >= 0) {
            const auto tp = std::chrono::steady_clock::now();
            parked = ops_->park(from, to, err);
            rec.ms_park = ms_since(tp);
        }
        bool in = false;
        bool back = false;
        if (parked) {
            const auto tu = std::chrono::steady_clock::now();
            in = ops_->unpark(to, err);
            rec.ms_unpark = ms_since(tu);
            if (!in && from >= 0) {
                std::string err2;
                back = ops_->unpark(from, err2);
                if (!back) err += "; putting '" + stats_[from].name + "' back also failed: " + err2;
            }
        }
        rec.ms_total = ms_since(t0);
        rec.ok = in;
        rec.err = err;
        rec.t_end_ms = steady_ms();

        lk.lock();
        swapping_ = false;
        ++n_swaps_;
        last_ = rec;
        if (in) {
            active_ = to;
            stats_[to].n_swaps_in++;
            stats_[to].ms_last_unpark = rec.ms_unpark;
            if (from >= 0) stats_[from].ms_last_park = rec.ms_park;
            if (log_) {
                char buf[256];
                snprintf(buf, sizeof(buf), "swap done: park %.0f ms + unpark %.0f ms = %.0f ms", rec.ms_park, rec.ms_unpark, rec.ms_total);
                log_(buf);
            }
            grant_all_locked(to);
        } else {
            // the previous model stays (parked == false) or came back (back); otherwise nothing is on the cards
            active_ = (!parked || back) ? from : -1;
            stats_[to].n_swap_fail++;
            if (log_) log_("swap to '" + stats_[to].name + "' FAILED: " + err);
            fail_all_locked(to, "model '" + stats_[to].name + "' could not be loaded onto the cards: " + err);
        }
        cv_.notify_all();
    }

    static double ms_since(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    std::vector<std::vector<std::string>> names_;   // [model][accepted names], [0] is the id
    swap_ops *                  ops_;
    mutable std::mutex          mu_;
    std::condition_variable     cv_;
    int                         active_   = -1;
    int                         inflight_ = 0;
    bool                        swapping_ = false;
    int                         swap_to_  = -1;
    std::deque<ticket *>        queue_;
    std::vector<model_status>   stats_;
    swap_record                 last_;
    uint64_t                    n_swaps_ = 0;
    std::function<void(const std::string &)> log_;
};

// RAII lease; movable, releases once
class lease {
public:
    lease() = default;
    lease(router * r, int idx) : r_(r), idx_(idx) {}
    lease(const lease &) = delete;
    lease & operator=(const lease &) = delete;
    lease(lease && o) noexcept : r_(o.r_), idx_(o.idx_) { o.r_ = nullptr; }
    lease & operator=(lease && o) noexcept {
        if (this != &o) { reset(); r_ = o.r_; idx_ = o.idx_; o.r_ = nullptr; }
        return *this;
    }
    ~lease() { reset(); }
    void reset() {
        if (r_) { r_->release(idx_); r_ = nullptr; }
    }
    int  idx()  const { return idx_; }
    bool held() const { return r_ != nullptr; }
private:
    router * r_ = nullptr;
    int      idx_ = -1;
};

} // namespace pxa_hs
