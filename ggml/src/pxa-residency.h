#pragma once

// =============================================================================================
// PXA HOT SWAP -- residency groups (header-only core, no CUDA in here).
//
// WHAT IT IS. A residency group owns every large VRAM allocation one model runtime makes: its
// weights at load, the KV cache and compute buffers of every context created for it, their growth
// later, and the backend's scratch pools. Each allocation lives in its OWN reserved virtual address
// range, backed by physical memory in chunks (<= 256 MiB), so the physical VRAM under it can be
// released and re-created -- all of it or chunk by chunk -- while every device pointer the model, its
// contexts, their split tensors and their captured CUDA graphs hold stays valid.
//
//   park()    copies what must survive to pinned host RAM, then unmaps: the VRAM is free for
//             another model. Weights are immutable after load, so their host mirror is taken once
//             (prime() at boot, or the first park) and reused; the KV cache and anything else that
//             is not a compute buffer is copied every time it is unmapped; compute buffers and pools
//             are dropped (their contents are dead between graph evaluations).
//   unpark()  re-creates physical memory under the SAME addresses and copies the mirrors back,
//             one host thread per device so every card loads its own share over its own link at
//             the same time.
//
// PARTIAL RESIDENCY (lever, PXA_SWAP_KEEP=1 in the server). park(want_free) releases only as many
// chunks as the incoming model needs -- scratch first (free to drop), then weights (already
// mirrored, nothing to copy), then state (must be copied down) -- so a parked model can stay partly
// or wholly on the cards and its next unpark copies only what was actually released. When the model
// on the cards later needs more memory (a longer prompt grows its pool), reclaim() takes it from
// parked groups on demand. Default is a full park: the model on the cards then sees the same free
// VRAM it saw at boot, which matters because some routes (the Volta q8 attention staging, the
// checkpoint budget) are chosen from free VRAM.
//
// HOW AN ALLOCATION FINDS ITS GROUP. One group is ACTIVE for the whole process at a time (the
// model on the cards): the CUDA backend asks active() at its weight/KV/compute allocation sites, so
// every thread the model's runtime uses (its task loop, a speculative worker) lands in the right
// group. Exactly one model runs at a time, which is what makes a process-wide pointer the right
// scope. With no group active nothing changes, which is how single-model runs stay byte-identical.
// The llama layer marks the model-load window (PHASE_LOAD, per thread) so a weight allocation can
// be told from a runtime one: under -sm tensor/graph the KV cache is made of the same per-device
// split slices as the weights, and the buffer usage flag alone would call it a weight.
//
// The class of a whole buffer is decided at park time from the owner's usage flag, read through
// dev_ops::owner_class(): WEIGHTS usage allocated in PHASE_LOAD -> mirror once; COMPUTE usage ->
// scratch; anything else -> state (copied whenever it is unmapped, the conservative default).
//
// The core is written against dev_ops so the unit test (tests/test-pxa-residency.cpp) runs the whole
// bookkeeping -- classes, mirrors, remap-at-the-same-address, partial residency, reclaim, rollback on
// failure -- on a fake device made of host memory.
// =============================================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace pxa_res {

constexpr int MAX_DEV = 16;

enum { CLS_WEIGHTS = 0, CLS_STATE = 1, CLS_SCRATCH = 2, CLS_N = 3 };
enum { PHASE_RUNTIME = 0, PHASE_LOAD = 1 };

inline const char * cls_name(int c) {
    switch (c) {
        case CLS_WEIGHTS: return "weights";
        case CLS_STATE:   return "state";
        case CLS_SCRATCH: return "scratch";
    }
    return "?";
}

struct copy_op {
    uint64_t va;
    void *   host;
    size_t   size;
};

// Device primitives. Real: CUDA driver VMM + cudaMemcpyAsync from pinned memory. Test: host memory.
struct dev_ops {
    virtual ~dev_ops() = default;
    virtual size_t granularity(int dev) = 0;
    virtual bool   va_reserve(int dev, size_t size, uint64_t * va, std::string & err) = 0;
    virtual void   va_free(int dev, uint64_t va, size_t size) = 0;
    // create physical memory of `size` bytes on `dev`, map it at `va`, grant access; size and va
    // are multiples of granularity(dev)
    virtual bool   phys_map(int dev, uint64_t va, size_t size, std::string & err) = 0;
    // unmap [va, va+size) (exactly one earlier phys_map); the physical memory goes back to the device
    virtual void   phys_unmap(int dev, uint64_t va, size_t size) = 0;
    virtual void * host_alloc(size_t size) = 0;    // pinned; nullptr on failure
    virtual void   host_free(void * p, size_t size) = 0;
    // one batch of copies for one device; returns when all of them completed
    virtual bool   d2h(int dev, const std::vector<copy_op> & ops, std::string & err) = 0;
    virtual bool   h2d(int dev, const std::vector<copy_op> & ops, std::string & err) = 0;
    // wait for all work queued on the device (any stream) before its memory is copied or unmapped
    virtual bool   sync(int dev, std::string & err) = 0;
    // class of a whole buffer from its owner's usage: CLS_WEIGHTS / CLS_SCRATCH / CLS_STATE
    virtual int    owner_class(const void * owner) = 0;
    // physical memory the device can still give (bytes)
    virtual size_t mem_free(int dev) = 0;
};

// A scratch pool with its own reserved range that can drop and re-take its physical memory
// (the CUDA VMM pool). Its contents never survive a park; its ADDRESSES do (captured graphs).
struct pool_hook {
    virtual ~pool_hook() = default;
    virtual int    pool_dev() const = 0;
    virtual size_t pool_mapped() const = 0;
    virtual bool   pool_in_use() const = 0;
    virtual bool   pool_is_parked() const = 0;
    virtual bool   pool_park(std::string & err) = 0;
    virtual bool   pool_unpark(std::string & err) = 0;
};

struct op_stats {
    double   ms_total = 0.0;    // wall time of the whole operation
    double   ms_map   = 0.0;    // max over devices: physical create+map (unpark) or unmap (park)
    double   ms_copy  = 0.0;    // max over devices: copy time
    double   ms_dev[MAX_DEV] = {};
    uint64_t bytes_cls[CLS_N] = {};  // bytes per class held by the group (mapped for scratch)
    uint64_t bytes_copied = 0;       // bytes moved over PCIe by this operation
    uint64_t bytes_dev[MAX_DEV] = {};
    uint64_t bytes_mirror_new = 0;   // weight bytes mirrored for the first time
    uint64_t bytes_released = 0;     // park: physical bytes given back to the devices
    uint64_t bytes_resident = 0;     // bytes of this group still mapped after the call
    uint64_t verify_bad = 0;         // PXA_SWAP_VERIFY: weight bytes that no longer match the mirror
    int      n_dev = 0;
    char     err[256] = {};
};

// ---------------------------------------------------------------------------------------------
// the active group (process-wide) + load phase (per thread) + pinned budget
// ---------------------------------------------------------------------------------------------
class residency;

inline std::atomic<residency *> & active() {
    static std::atomic<residency *> r{nullptr};
    return r;
}

inline int & tl_phase() {
    static thread_local int p = PHASE_RUNTIME;
    return p;
}

inline std::atomic<uint64_t> & pinned_used() {
    static std::atomic<uint64_t> v{0};
    return v;
}
inline std::atomic<uint64_t> & pinned_limit() {   // 0 = no cap
    static std::atomic<uint64_t> v{0};
    return v;
}

inline size_t round_up(size_t x, size_t g) {
    return g ? ((x + g - 1) / g) * g : x;
}

inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

class residency {
public:
    // a whole buffer is mapped in chunks of chunk_max(); a slab grows by slab_grow() while it loads
    // (the unit test shrinks both to exercise many chunks with little memory)
    static size_t & chunk_max() { static size_t v = (size_t) 256 << 20; return v; }
    static size_t & slab_grow() { static size_t v = (size_t)  32 << 20; return v; }

    residency(std::string name, dev_ops * ops) : name_(std::move(name)), ops_(ops) {
        std::lock_guard<std::mutex> lk(reg_mu());
        registry().push_back(this);
    }

    ~residency() {
        {
            std::lock_guard<std::mutex> lk(reg_mu());
            auto & v = registry();
            v.erase(std::remove(v.begin(), v.end(), this), v.end());
        }
        std::lock_guard<std::mutex> lk(mu_);
        for (auto & r : recs_) {
            release_rec(r);
        }
        recs_.clear();
    }

    const std::string & name() const { return name_; }

    bool parked() const {
        std::lock_guard<std::mutex> lk(mu_);
        return parked_;
    }

    // ---- allocation ------------------------------------------------------------------------
    // A whole buffer (weights of one ctx on one device, a KV buffer, a compute buffer).
    bool alloc_buffer(int dev, size_t size, void ** ptr, std::string & err) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!check_alloc(dev, err)) return false;
        const size_t g = ops_->granularity(dev);
        rec r;
        r.dev   = dev;
        r.kind  = K_BUFFER;
        r.size  = std::max<size_t>(size, 1);
        r.vsize = round_up(r.size, g);
        r.phase = tl_phase();
        if (!ops_->va_reserve(dev, r.vsize, &r.va, err)) {
            return false;
        }
        for (size_t off = 0; off < r.vsize; off += chunk_max()) {
            const size_t len = std::min(chunk_max(), r.vsize - off);
            if (!map_with_reclaim(dev, r.va + off, len, err)) {
                for (auto & c : r.chunks) ops_->phys_unmap(dev, r.va + c.off, c.len);
                ops_->va_free(dev, r.va, r.vsize);
                return false;
            }
            r.chunks.push_back({off, len, true});
        }
        recs_.push_back(r);
        *ptr = (void *) (uintptr_t) r.va;
        return true;
    }

    // Name the ggml buffer that owns an allocation (its usage decides the class at park time).
    void set_owner(const void * ptr, const void * owner) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto & r : recs_) {
            if (r.kind == K_BUFFER && r.va == (uint64_t) (uintptr_t) ptr) {
                r.owner = owner;
                return;
            }
        }
    }

    // Returns true if ptr was one of our whole buffers (and is now released).
    bool free_buffer(const void * ptr) {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < recs_.size(); ++i) {
            rec & r = recs_[i];
            if (r.kind != K_BUFFER || r.va != (uint64_t) (uintptr_t) ptr) continue;
            release_rec(r);
            recs_.erase(recs_.begin() + i);
            return true;
        }
        return false;
    }

    // A packed sub-allocation (one per-device slice of a split tensor). Slices are never freed one
    // by one; they go away with the group. Load-window slices go to the weights slab, later ones
    // (the KV cache under a tensor split) to the state slab.
    bool alloc_slice(int dev, size_t size, void ** ptr, std::string & err) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!check_alloc(dev, err)) return false;
        const bool weights = tl_phase() == PHASE_LOAD;
        rec * s = nullptr;
        for (auto & r : recs_) {
            if (r.kind == K_SLAB && r.dev == dev && r.slab_weights == weights) { s = &r; break; }
        }
        const size_t g = ops_->granularity(dev);
        if (!s) {
            rec r;
            r.dev   = dev;
            r.kind  = K_SLAB;
            r.slab_weights = weights;
            r.phase = weights ? PHASE_LOAD : PHASE_RUNTIME;
            r.vsize = slab_va_bytes(g);
            if (!ops_->va_reserve(dev, r.vsize, &r.va, err)) return false;
            recs_.push_back(r);
            s = &recs_.back();
        }
        const size_t off = round_up(s->size, 256);
        const size_t end = off + std::max<size_t>(size, 1);
        if (end > s->vsize) {
            err = "slab virtual range exhausted";
            return false;
        }
        size_t top = s->mapped_end();
        while (end > top) {
            const size_t len = std::min(std::max(round_up(end - top, g), round_up(slab_grow(), g)), s->vsize - top);
            if (!map_with_reclaim(dev, s->va + top, len, err)) return false;
            s->chunks.push_back({top, len, true});
            top += len;
        }
        s->size = end;
        *ptr = (void *) (uintptr_t) (s->va + off);
        return true;
    }

    void add_pool(pool_hook * p) {
        std::lock_guard<std::mutex> lk(mu_);
        pools_.push_back(p);
    }
    void remove_pool(pool_hook * p) {
        std::lock_guard<std::mutex> lk(mu_);
        pools_.erase(std::remove(pools_.begin(), pools_.end(), p), pools_.end());
    }

    // bytes held per class (scratch = compute buffers + pools), pinned host bytes, and bytes mapped now
    void sizes(uint64_t out[CLS_N], uint64_t * pinned, uint64_t * resident = nullptr) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (int c = 0; c < CLS_N; ++c) out[c] = 0;
        uint64_t pin = 0, res = 0;
        for (const auto & r : recs_) {
            out[class_of(r)] += r.size;
            pin += r.mirror_cap;
            res += r.mapped_bytes();
        }
        for (const auto * p : pools_) {
            out[CLS_SCRATCH] += p->pool_mapped();
            if (!p->pool_is_parked()) res += p->pool_mapped();
        }
        if (pinned) *pinned = pin;
        if (resident) *resident = res;
    }

    // physical bytes unpark() would map now, per device
    void need(size_t out[MAX_DEV]) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (int d = 0; d < MAX_DEV; ++d) out[d] = 0;
        for (const auto & r : recs_) {
            for (const auto & c : r.chunks) if (!c.mapped) out[r.dev] += c.len;
        }
        for (const auto * p : pools_) {
            if (p->pool_is_parked()) out[p->pool_dev()] += p->pool_mapped();
        }
    }

    // ---- prime / park / unpark ----------------------------------------------------------------
    // Take the weight mirror now, releasing nothing (the model stays on the cards). A server does
    // this at boot for the model it opens with, so its first swap-out costs what every later one does.
    bool prime(op_stats * st_out) {
        op_stats st;
        const auto t0 = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mu_);
        if (parked_) {
            if (st_out) *st_out = st;
            return true;
        }
        std::vector<dev_plan> plan(MAX_DEV);
        std::string err;
        for (auto & r : recs_) {
            if (class_of(r) == CLS_WEIGHTS && r.mirror_top < r.size) {
                if (!plan_mirror_weights(r, plan[r.dev], st, err)) return fail(st, st_out, err);
            }
        }
        const bool ok = run_per_device(plan, [&](int d, dev_plan & dp, std::string & e) -> bool {
            const auto td = std::chrono::steady_clock::now();
            if (!ops_->sync(d, e)) return false;
            if (!dp.down.empty() && !ops_->d2h(d, dp.down, e)) return false;
            dp.ms_copy = ms_since(td);
            return true;
        }, err);
        if (!ok) return fail(st, st_out, "prime: copy to host failed: " + err);
        commit_down(plan, st);
        finish(st, plan, t0, st_out);
        return true;
    }

    // Park the group: nothing may run on it until unpark(). want_free == nullptr releases every
    // byte; otherwise, per device, only enough to leave want_free[dev] bytes free on it.
    bool park(op_stats * st_out, const size_t * want_free = nullptr) {
        op_stats st;
        const auto t0 = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mu_);
        if (parked_ && !want_free) {
            // already parked: a full park still releases whatever a partial one kept
            std::string err;
            if (!release_locked(nullptr, -1, (size_t) -1, st, err)) return fail(st, st_out, err);
            finish(st, {}, t0, st_out);
            return true;
        }
        for (const auto * p : pools_) {
            if (p->pool_in_use()) {
                return fail(st, st_out, "a scratch pool on device " + std::to_string(p->pool_dev()) +
                            " still has live allocations; park only between graph evaluations");
            }
        }
        std::string err;
        if (!release_locked(want_free, -1, 0, st, err)) return fail(st, st_out, err);
        parked_ = true;
        finish(st, {}, t0, st_out);
        return true;
    }

    bool unpark(op_stats * st_out) {
        op_stats st;
        const auto t0 = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mu_);
        if (!parked_) {
            if (st_out) *st_out = st;
            return true;
        }
        std::vector<dev_plan> plan(MAX_DEV);
        for (auto & r : recs_) {
            // a slab that went down whole comes back in chunk_max() pieces (fewer maps next time)
            if (r.kind == K_SLAB && !r.chunks.empty() && r.mapped_bytes() == 0) {
                recut_chunks(r);
            }
            for (size_t ci = 0; ci < r.chunks.size(); ++ci) {
                const chunk & c = r.chunks[ci];
                if (c.mapped) continue;
                dev_plan & dp = plan[r.dev];
                dp.used = true;
                dp.maps.push_back({&r, ci});
                if (class_of(r) != CLS_SCRATCH && c.off < r.size) {
                    const size_t n = std::min(c.off + c.len, r.size) - c.off;
                    dp.up.push_back({r.va + c.off, (char *) r.mirror + c.off, n});
                }
            }
        }
        for (auto * p : pools_) {
            if (p->pool_is_parked()) {
                plan[p->pool_dev()].used = true;
                plan[p->pool_dev()].pools.push_back(p);
            }
        }
        std::string err;
        bool ok = run_per_device(plan, [&](int d, dev_plan & dp, std::string & e) -> bool {
            const auto tm = std::chrono::steady_clock::now();
            for (auto & m : dp.maps) {
                chunk & c = m.r->chunks[m.ci];
                if (!map_with_reclaim(d, m.r->va + c.off, c.len, e)) return false;
                c.mapped = true;
                m.done = true;
            }
            for (auto * p : dp.pools) {
                if (!p->pool_unpark(e)) {
                    // make room from parked groups and try once more
                    reclaim(d, p->pool_mapped(), this);
                    e.clear();
                    if (!p->pool_unpark(e)) return false;
                }
                dp.pools_done.push_back(p);
            }
            dp.ms_map = ms_since(tm);
            const auto tc = std::chrono::steady_clock::now();
            if (!dp.up.empty() && !ops_->h2d(d, dp.up, e)) return false;
            dp.ms_copy = ms_since(tc);
            return true;
        }, err);

        if (!ok) {
            // roll back: release everything this attempt mapped, stay parked
            for (int d = 0; d < MAX_DEV; ++d) {
                for (auto & m : plan[d].maps) {
                    if (!m.done) continue;
                    chunk & c = m.r->chunks[m.ci];
                    ops_->phys_unmap(d, m.r->va + c.off, c.len);
                    c.mapped = false;
                }
                for (auto * p : plan[d].pools_done) {
                    std::string e2;
                    p->pool_park(e2);
                }
            }
            return fail(st, st_out, "unpark failed: " + err);
        }
        for (auto & dp : plan) {
            for (auto & op : dp.up) st.bytes_copied += op.size;
        }
        parked_ = false;
        finish(st, plan, t0, st_out);
        return true;
    }

    // Take `bytes` of physical memory on `dev` back from PARKED groups other than `except`
    // (scratch first, then mirrored weights, then state). Returns the bytes released. This is what
    // the model on the cards calls when it needs more memory than was left for it.
    static size_t reclaim(int dev, size_t bytes, residency * except) {
        std::vector<residency *> groups;
        {
            std::lock_guard<std::mutex> lk(reg_mu());
            groups = registry();
        }
        size_t got = 0;
        for (auto * g : groups) {
            if (g == except || got >= bytes) continue;
            std::lock_guard<std::mutex> lk(g->mu_);
            if (!g->parked_) continue;
            op_stats st;
            std::string err;
            if (g->release_locked(nullptr, dev, bytes - got, st, err)) {
                got += st.bytes_released;
                if (st.bytes_released) {
                    fprintf(stderr, "pxa-residency: reclaimed %.1f MiB on device %d from parked '%s'\n",
                            st.bytes_released / 1048576.0, dev, g->name_.c_str());
                }
            }
        }
        return got;
    }

private:
    enum { K_BUFFER = 0, K_SLAB = 1 };

    struct chunk {
        size_t off;
        size_t len;
        bool   mapped;
    };

    struct rec {
        int          dev   = 0;
        int          kind  = K_BUFFER;
        bool         slab_weights = false;
        uint64_t     va    = 0;
        size_t       size  = 0;       // bytes in use (a slab: its bump pointer)
        size_t       vsize = 0;       // reserved
        int          phase = PHASE_RUNTIME;
        const void * owner = nullptr;
        std::vector<chunk> chunks;    // physical backing, [0, mapped_end())
        void *       mirror = nullptr;
        size_t       mirror_cap = 0;
        size_t       mirror_top = 0;  // weights: bytes [0, mirror_top) mirrored

        size_t mapped_end() const {
            size_t e = 0;
            for (const auto & c : chunks) e = std::max(e, c.off + c.len);
            return e;
        }
        size_t mapped_bytes() const {
            size_t b = 0;
            for (const auto & c : chunks) if (c.mapped) b += c.len;
            return b;
        }
    };

    struct map_item {
        rec *  r;
        size_t ci;
        bool   done = false;
    };

    struct dev_plan {
        bool used = false;
        std::vector<copy_op> down, up, verify;
        std::vector<void *> verify_ref;
        std::vector<std::pair<size_t *, size_t>> after_top;   // mirror_top updates once the copies landed
        std::vector<map_item> unmaps;                         // chunks to release after the copies
        std::vector<map_item> maps;                           // unpark
        std::vector<pool_hook *> pools, pools_done;
        double ms_map = 0.0, ms_copy = 0.0;
    };

    static std::mutex & reg_mu() {
        static std::mutex m;
        return m;
    }
    static std::vector<residency *> & registry() {
        static std::vector<residency *> v;
        return v;
    }

    static bool env_flag(const char * name) {
        const char * e = getenv(name);
        return e && atoi(e) != 0;
    }

    static size_t slab_va_bytes(size_t g) {
        // per device per class; virtual only (32 GiB, like the CUDA VMM pool, is far above any card here)
        size_t gib = 32;
        if (const char * e = getenv("PXA_SWAP_SLAB_VA_GB")) gib = (size_t) std::max(1, atoi(e));
        return round_up(gib << 30, g);
    }

    bool check_alloc(int dev, std::string & err) const {
        if (parked_) {
            err = "residency '" + name_ + "' is parked: no allocation is allowed until it is unparked";
            return false;
        }
        if (dev < 0 || dev >= MAX_DEV) {
            err = "device out of range";
            return false;
        }
        return true;
    }

    // map, and if the device is full take the memory from parked groups once
    bool map_with_reclaim(int dev, uint64_t va, size_t len, std::string & err) {
        if (ops_->phys_map(dev, va, len, err)) return true;
        if (reclaim(dev, len, this) == 0) return false;
        err.clear();
        return ops_->phys_map(dev, va, len, err);
    }

    int class_of(const rec & r) const {
        if (r.kind == K_SLAB) return r.slab_weights ? CLS_WEIGHTS : CLS_STATE;
        const int oc = r.owner ? ops_->owner_class(r.owner) : CLS_STATE;
        if (oc == CLS_SCRATCH) return CLS_SCRATCH;
        if (oc == CLS_WEIGHTS && r.phase == PHASE_LOAD) return CLS_WEIGHTS;
        return CLS_STATE;
    }

    void release_rec(rec & r) {
        for (auto & c : r.chunks) {
            if (c.mapped) ops_->phys_unmap(r.dev, r.va + c.off, c.len);
        }
        r.chunks.clear();
        ops_->va_free(r.dev, r.va, r.vsize);
        drop_mirror(r.mirror, r.mirror_cap);
    }

    // a whole-down slab comes back in chunk_max() pieces instead of the SLAB_GROW steps it grew by
    void recut_chunks(rec & r) {
        const size_t end = r.mapped_end();
        r.chunks.clear();
        for (size_t off = 0; off < end; off += chunk_max()) {
            r.chunks.push_back({off, std::min(chunk_max(), end - off), false});
        }
    }

    bool ensure_mirror(rec & r, size_t need, bool keep, std::string & err) {
        if (r.mirror && r.mirror_cap >= need) {
            return true;
        }
        const uint64_t lim = pinned_limit().load();
        const uint64_t extra = need - (r.mirror ? r.mirror_cap : 0);
        if (lim && pinned_used().load() + extra > lim) {
            err = "pinned host budget exceeded (" + std::to_string((pinned_used().load() + extra) >> 20) +
                  " MiB needed, limit " + std::to_string(lim >> 20) +
                  " MiB; raise PXA_SWAP_PIN_BUDGET_MB or register fewer models)";
            return false;
        }
        void * n = ops_->host_alloc(need);
        if (!n) {
            err = "could not allocate " + std::to_string(need >> 20) + " MiB of pinned host memory";
            return false;
        }
        pinned_used() += need;
        if (r.mirror) {
            if (keep) std::memcpy(n, r.mirror, std::min(r.mirror_cap, need));
            ops_->host_free(r.mirror, r.mirror_cap);
            pinned_used() -= r.mirror_cap;
        }
        r.mirror = n;
        r.mirror_cap = need;
        return true;
    }

    void drop_mirror(void *& m, size_t & cap) {
        if (m) {
            ops_->host_free(m, cap);
            pinned_used() -= cap;
        }
        m = nullptr;
        cap = 0;
    }

    // plan the one-time copy of a weight record's not-yet-mirrored bytes
    bool plan_mirror_weights(rec & r, dev_plan & dp, op_stats & st, std::string & err) {
        if (!ensure_mirror(r, std::max(r.size, (size_t) 1), /*keep=*/true, err)) return false;
        dp.used = true;
        dp.down.push_back({r.va + r.mirror_top, (char *) r.mirror + r.mirror_top, r.size - r.mirror_top});
        dp.after_top.push_back({&r.mirror_top, r.size});
        st.bytes_mirror_new += r.size - r.mirror_top;
        return true;
    }

    // Release chunks. want_free == nullptr and only_dev < 0: everything (full park). want_free:
    // per device, until the device has that much free. only_dev >= 0: that device, until
    // `bytes` were released (reclaim). Order: pools, scratch, weights (mirror first if it is not
    // yet), state (copied down). Called with mu_ held.
    bool release_locked(const size_t * want_free, int only_dev, size_t bytes, op_stats & st, std::string & err) {
        const bool verify = env_flag("PXA_SWAP_VERIFY") && only_dev < 0;
        std::vector<dev_plan> plan(MAX_DEV);
        size_t budget[MAX_DEV];
        bool   all[MAX_DEV];
        for (int d = 0; d < MAX_DEV; ++d) {
            all[d] = false;
            budget[d] = 0;
            if (only_dev >= 0) {
                if (d == only_dev) budget[d] = bytes;
            } else if (!want_free) {
                all[d] = true;
            } else if (want_free[d]) {
                const size_t fr = ops_->mem_free(d);
                budget[d] = want_free[d] > fr ? want_free[d] - fr : 0;
            }
        }
        auto take = [&](int d, size_t len) -> bool {   // should this device give `len` more?
            if (all[d]) return true;
            if (budget[d] == 0) return false;
            budget[d] = budget[d] > len ? budget[d] - len : 0;
            return true;
        };
        // 1. pools
        for (auto * p : pools_) {
            const int d = p->pool_dev();
            if (p->pool_is_parked() || p->pool_mapped() == 0) continue;
            if (!take(d, p->pool_mapped())) continue;
            plan[d].used = true;
            plan[d].pools.push_back(p);
        }
        // 2-4. chunks by class
        for (int cls : {CLS_SCRATCH, CLS_WEIGHTS, CLS_STATE}) {
            for (auto & r : recs_) {
                if (class_of(r) != cls) continue;
                dev_plan & dp = plan[r.dev];
                bool any = false;
                for (size_t ci = r.chunks.size(); ci-- > 0;) {
                    const chunk & c = r.chunks[ci];
                    if (!c.mapped || !take(r.dev, c.len)) continue;
                    dp.used = true;
                    dp.unmaps.push_back({&r, ci});
                    any = true;
                    if (cls == CLS_STATE && c.off < r.size) {
                        if (!ensure_mirror(r, r.size, /*keep=*/true, err)) return false;
                        const size_t n = std::min(c.off + c.len, r.size) - c.off;
                        dp.down.push_back({r.va + c.off, (char *) r.mirror + c.off, n});
                    }
                }
                if (cls == CLS_WEIGHTS && any && r.mirror_top < r.size) {
                    if (!plan_mirror_weights(r, dp, st, err)) return false;
                }
                if (cls == CLS_WEIGHTS && verify && r.mirror_top > 0 && r.mirror_top == r.size) {
                    void * tmp = std::malloc(r.size);
                    if (!tmp) { err = "verify: out of host memory"; return false; }
                    dp.used = true;
                    dp.verify.push_back({r.va, tmp, r.size});
                    dp.verify_ref.push_back(r.mirror);
                }
            }
        }
        bool ok = run_per_device(plan, [&](int d, dev_plan & dp, std::string & e) -> bool {
            const auto td = std::chrono::steady_clock::now();
            if (!ops_->sync(d, e)) return false;
            if (!dp.verify.empty() && !ops_->d2h(d, dp.verify, e)) return false;
            if (!dp.down.empty() && !ops_->d2h(d, dp.down, e)) return false;
            dp.ms_copy = ms_since(td);
            return true;
        }, err);
        uint64_t bad = 0;
        for (auto & dp : plan) {
            for (size_t i = 0; i < dp.verify.size(); ++i) {
                if (ok) {
                    const unsigned char * a = (const unsigned char *) dp.verify[i].host;
                    const unsigned char * b = (const unsigned char *) dp.verify_ref[i];
                    for (size_t k = 0; k < dp.verify[i].size; ++k) bad += a[k] != b[k];
                }
                std::free(dp.verify[i].host);
            }
        }
        if (!ok) {
            err = "copy to host failed: " + err;
            return false;
        }
        st.verify_bad += bad;
        commit_down(plan, st);
        // release (addresses stay reserved)
        for (int d = 0; d < MAX_DEV; ++d) {
            dev_plan & dp = plan[d];
            if (!dp.used) continue;
            const auto tu = std::chrono::steady_clock::now();
            for (auto & u : dp.unmaps) {
                chunk & c = u.r->chunks[u.ci];
                ops_->phys_unmap(d, u.r->va + c.off, c.len);
                c.mapped = false;
                st.bytes_released += c.len;
            }
            for (auto * p : dp.pools) {
                std::string e;
                const size_t sz = p->pool_mapped();
                if (p->pool_park(e)) {
                    st.bytes_released += sz;
                } else {
                    fprintf(stderr, "pxa-residency '%s': pool on device %d kept its memory: %s\n",
                            name_.c_str(), d, e.c_str());
                }
            }
            dp.ms_map = ms_since(tu);
        }
        fill_plan_stats(st, plan);
        return true;
    }

    void commit_down(std::vector<dev_plan> & plan, op_stats & st) {
        for (auto & dp : plan) {
            for (auto & t : dp.after_top) *t.first = t.second;
            for (auto & op : dp.down) st.bytes_copied += op.size;
        }
    }

    template <typename F>
    bool run_per_device(std::vector<dev_plan> & plan, F && fn, std::string & err) {
        std::vector<std::thread> th;
        std::vector<std::string> errs(MAX_DEV);
        std::vector<int> okv(MAX_DEV, 1);
        for (int d = 0; d < MAX_DEV; ++d) {
            if (!plan[d].used) continue;
            th.emplace_back([&, d]() { okv[d] = fn(d, plan[d], errs[d]) ? 1 : 0; });
        }
        for (auto & t : th) t.join();
        bool ok = true;
        for (int d = 0; d < MAX_DEV; ++d) {
            if (!okv[d]) {
                ok = false;
                if (!err.empty()) err += "; ";
                err += "device " + std::to_string(d) + ": " + errs[d];
            }
        }
        return ok;
    }

    void fill_plan_stats(op_stats & st, const std::vector<dev_plan> & plan) const {
        for (int d = 0; d < MAX_DEV && d < (int) plan.size(); ++d) {
            const dev_plan & dp = plan[d];
            if (!dp.used) continue;
            uint64_t b = 0;
            for (auto & op : dp.down) b += op.size;
            for (auto & op : dp.up)   b += op.size;
            st.bytes_dev[d] += b;
            st.ms_dev[d]    += dp.ms_map + dp.ms_copy;
            st.ms_map  = std::max(st.ms_map, dp.ms_map);
            st.ms_copy = std::max(st.ms_copy, dp.ms_copy);
        }
    }

    void finish(op_stats & st, const std::vector<dev_plan> & plan, std::chrono::steady_clock::time_point t0, op_stats * out) const {
        if (!plan.empty()) fill_plan_stats(st, plan);
        for (int d = 0; d < MAX_DEV; ++d) if (st.bytes_dev[d] || st.ms_dev[d] > 0) st.n_dev++;
        for (const auto & r : recs_) {
            st.bytes_cls[class_of(r)] += r.size;
            st.bytes_resident += r.mapped_bytes();
        }
        for (const auto * p : pools_) {
            st.bytes_cls[CLS_SCRATCH] += p->pool_mapped();
            if (!p->pool_is_parked()) st.bytes_resident += p->pool_mapped();
        }
        st.ms_total = ms_since(t0);
        if (out) *out = st;
    }

    bool fail(op_stats & st, op_stats * out, const std::string & e) {
        snprintf(st.err, sizeof(st.err), "%s", e.c_str());
        if (out) *out = st;
        return false;
    }

    std::string        name_;
    dev_ops *          ops_;
    mutable std::mutex mu_;
    bool               parked_ = false;
    std::vector<rec>   recs_;
    std::vector<pool_hook *> pools_;
};

// RAII: mark the model-load window on this thread
struct load_phase_guard {
    int prev;
    load_phase_guard() : prev(tl_phase()) { tl_phase() = PHASE_LOAD; }
    ~load_phase_guard() { tl_phase() = prev; }
};

} // namespace pxa_res
