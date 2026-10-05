// PXA HOT SWAP -- residency group core (ggml/src/pxa-residency.h) on a fake device.
//
// The fake device is host memory with the same rules as the CUDA VMM path: an address range is
// reserved once and never moves; physical memory is created under part of it in granules (with
// garbage in it, like fresh VRAM) and destroyed on unmap (poisoned, then gone); an unmap must match
// exactly one earlier map; a device has a capacity, so a second model cannot be mapped while the
// first one still holds the card. Chunk sizes are shrunk so every buffer spans several chunks.
//
// What is checked:
//   1. park frees every byte of "VRAM" the group held and unpark maps it back at the SAME address;
//   2. weights come back bit-exact, and their mirror is taken once (the second park copies only
//      the KV/state bytes);
//   3. the KV/state class comes back with the contents it had at the LAST park, not the first;
//   4. compute buffers and pools are dropped and re-mapped without a copy;
//   5. split slices taken during the load window are weights, slices taken later are state (the KV
//      cache under -sm tensor) -- the buffer usage flag alone would call both weights;
//   6. an unpark that cannot get its memory (a failing device) rolls back completely: nothing stays
//      mapped, the group stays parked;
//   7. a park that cannot pin its mirrors (budget) changes nothing and reports why;
//   8. PXA_SWAP_VERIFY=1 counts weight bytes that changed behind the mirror's back;
//   9. no allocation is accepted while parked; a pool with live allocations refuses the park;
//  10. two groups alternate A -> B -> A on a card that holds only one of them;
//  11. prime() takes the weight mirror without touching the cards, so the first park copies KV only;
//  12. partial residency: a park that only has to make room for a small model releases scratch and
//      weights first, keeps the rest mapped (and its KV on the card, uncopied), and the next unpark
//      moves only what was released;
//  13. reclaim: the model on the cards takes memory from a partly-resident parked model on demand,
//      and the parked model still comes back intact;
//  14. an unpark on a full card makes its own room from parked groups instead of failing.
// CPU only, no model, no GPU. Exit 0 on pass.

#include "pxa-residency.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

struct fake_buffer {   // stands in for a ggml buffer: only the usage matters here
    int usage;         // 0 any, 1 weights, 2 compute
};

struct fake_dev : public pxa_res::dev_ops {
    static constexpr size_t G = 64 * 1024;   // granularity

    std::recursive_mutex mu;
    uint64_t next_va = 0x100000000ull;
    std::map<uint64_t, size_t> reserved;                                   // va -> size
    std::map<std::pair<int, uint64_t>, std::vector<unsigned char>> gran;   // (dev, granule va) -> bytes
    std::map<std::pair<int, uint64_t>, size_t> maps;                       // (dev, map va) -> size
    size_t capacity[pxa_res::MAX_DEV];
    size_t used[pxa_res::MAX_DEV] = {};
    int    fail_map_on_dev = -1;
    std::mt19937 rng{1234};

    fake_dev() { for (auto & c : capacity) c = (size_t) 1 << 40; }

    size_t granularity(int) override { return G; }

    bool va_reserve(int, size_t size, uint64_t * va, std::string & err) override {
        std::lock_guard<std::recursive_mutex> lk(mu);
        if (size % G) { err = "reserve size not a multiple of granularity"; return false; }
        *va = next_va;
        reserved[next_va] = size;
        next_va += size + G;   // a gap, so an overrun lands nowhere
        return true;
    }
    void va_free(int, uint64_t va, size_t) override {
        std::lock_guard<std::recursive_mutex> lk(mu);
        reserved.erase(va);
    }
    bool inside_reservation(uint64_t va, size_t size) {
        for (auto & r : reserved) {
            if (va >= r.first && va + size <= r.first + r.second) return true;
        }
        return false;
    }
    bool phys_map(int dev, uint64_t va, size_t size, std::string & err) override {
        std::lock_guard<std::recursive_mutex> lk(mu);
        if (dev == fail_map_on_dev) { err = "injected failure"; return false; }
        if (va % G || size % G) { err = "map not granular"; return false; }
        if (!inside_reservation(va, size)) { err = "map outside any reservation"; return false; }
        if (used[dev] + size > capacity[dev]) { err = "out of memory"; return false; }
        for (uint64_t a = va; a < va + size; a += G) {
            if (gran.count({dev, a})) { err = "map overlaps an existing mapping"; return false; }
        }
        for (uint64_t a = va; a < va + size; a += G) {
            auto & g = gran[{dev, a}];
            g.resize(G);
            for (auto & c : g) c = (unsigned char) rng();   // fresh VRAM holds garbage
        }
        maps[{dev, va}] = size;
        used[dev] += size;
        return true;
    }
    void phys_unmap(int dev, uint64_t va, size_t size) override {
        std::lock_guard<std::recursive_mutex> lk(mu);
        auto it = maps.find({dev, va});
        if (it == maps.end() || it->second != size) {
            fprintf(stderr, "fake: unmap of %llx/%zu does not match a mapping\n", (unsigned long long) va, size);
            ++g_fail;
            return;
        }
        for (uint64_t a = va; a < va + size; a += G) {
            gran.erase({dev, a});
        }
        maps.erase(it);
        used[dev] -= size;
    }
    void * host_alloc(size_t size) override { return std::malloc(size); }
    void   host_free(void * p, size_t) override { std::free(p); }

    // copy between [va, va+size) (possibly spanning mappings) and host; false if any byte is unmapped
    bool xfer(int dev, uint64_t va, void * host, size_t size, bool to_host) {
        unsigned char * h = (unsigned char *) host;
        size_t done = 0;
        while (done < size) {
            const uint64_t a  = va + done;
            const uint64_t ga = a - (a % G);
            auto it = gran.find({dev, ga});
            if (it == gran.end()) return false;
            const size_t o = a - ga;
            const size_t n = std::min(G - o, size - done);
            if (to_host) std::memcpy(h + done, it->second.data() + o, n);
            else         std::memcpy(it->second.data() + o, h + done, n);
            done += n;
        }
        return true;
    }
    bool d2h(int dev, const std::vector<pxa_res::copy_op> & ops, std::string & err) override {
        std::lock_guard<std::recursive_mutex> lk(mu);
        for (auto & op : ops) if (!xfer(dev, op.va, op.host, op.size, true))  { err = "d2h from unmapped memory"; return false; }
        return true;
    }
    bool h2d(int dev, const std::vector<pxa_res::copy_op> & ops, std::string & err) override {
        std::lock_guard<std::recursive_mutex> lk(mu);
        for (auto & op : ops) if (!xfer(dev, op.va, op.host, op.size, false)) { err = "h2d to unmapped memory"; return false; }
        return true;
    }
    bool sync(int, std::string &) override { return true; }
    int owner_class(const void * owner) override {
        const fake_buffer * b = (const fake_buffer *) owner;
        return b->usage == 1 ? pxa_res::CLS_WEIGHTS : b->usage == 2 ? pxa_res::CLS_SCRATCH : pxa_res::CLS_STATE;
    }
    size_t mem_free(int dev) override {
        std::lock_guard<std::recursive_mutex> lk(mu);
        return capacity[dev] > used[dev] ? capacity[dev] - used[dev] : 0;
    }

    // test helpers
    static unsigned char pat(unsigned seed, size_t i) { return (unsigned char) (seed * 131u + i * 7u + (i >> 9)); }
    void fill(int dev, void * ptr, size_t size, unsigned seed) {
        std::lock_guard<std::recursive_mutex> lk(mu);
        std::vector<unsigned char> buf(size);
        for (size_t i = 0; i < size; ++i) buf[i] = pat(seed, i);
        if (!xfer(dev, (uint64_t) (uintptr_t) ptr, buf.data(), size, false)) { fprintf(stderr, "fake: fill of unmapped memory\n"); ++g_fail; }
    }
    bool same(int dev, void * ptr, size_t size, unsigned seed) {
        std::lock_guard<std::recursive_mutex> lk(mu);
        std::vector<unsigned char> buf(size);
        if (!xfer(dev, (uint64_t) (uintptr_t) ptr, buf.data(), size, true)) return false;
        for (size_t i = 0; i < size; ++i) if (buf[i] != pat(seed, i)) return false;
        return true;
    }
    bool mapped_any(int dev, void * ptr, size_t size) {
        std::lock_guard<std::recursive_mutex> lk(mu);
        const uint64_t va = (uint64_t) (uintptr_t) ptr;
        for (uint64_t a = va - (va % G); a < va + size; a += G) if (gran.count({dev, a})) return true;
        return false;
    }
    bool mapped_all(int dev, void * ptr, size_t size) {
        std::lock_guard<std::recursive_mutex> lk(mu);
        const uint64_t va = (uint64_t) (uintptr_t) ptr;
        for (uint64_t a = va - (va % G); a < va + size; a += G) if (!gran.count({dev, a})) return false;
        return true;
    }
    void poke(int dev, void * ptr, size_t off, unsigned char x) {
        std::lock_guard<std::recursive_mutex> lk(mu);
        unsigned char b;
        const uint64_t va = (uint64_t) (uintptr_t) ptr + off;
        xfer(dev, va, &b, 1, true);
        b ^= x;
        xfer(dev, va, &b, 1, false);
    }
};

struct fake_pool : public pxa_res::pool_hook {
    fake_dev * fd; int dev; uint64_t va = 0; size_t size = 0; bool parked = false; int live = 0;
    fake_pool(fake_dev * f, int d, size_t sz) : fd(f), dev(d), size(sz) {
        std::string e;
        fd->va_reserve(dev, 1 << 24, &va, e);
        fd->phys_map(dev, va, size, e);
    }
    int    pool_dev() const override { return dev; }
    size_t pool_mapped() const override { return size; }
    bool   pool_in_use() const override { return live != 0; }
    bool   pool_is_parked() const override { return parked; }
    bool   pool_park(std::string &) override { if (!parked) { fd->phys_unmap(dev, va, size); parked = true; } return true; }
    bool   pool_unpark(std::string & e) override { if (parked) { if (!fd->phys_map(dev, va, size, e)) return false; parked = false; } return true; }
};

struct model_bufs {
    struct item { int dev; void * p; size_t size; unsigned seed; int cls; fake_buffer * owner; bool slice = false; };
    std::vector<item> items;
    std::vector<std::unique_ptr<fake_buffer>> owners;
};

// Build a "model" in group r on devices [0, ndev): per device one weight buffer and several weight
// slices (load window), then a KV buffer, KV slices and a compute buffer (runtime).
static void build_model(pxa_res::residency & r, fake_dev & fd, int ndev, unsigned seed, model_bufs & mb, size_t scale = 1) {
    pxa_res::active() = &r;
    std::string err;
    {
        pxa_res::load_phase_guard lp;
        for (int d = 0; d < ndev; ++d) {
            void * p = nullptr;
            const size_t sz = scale * (300 * 1024 + 17 * d);
            auto owner = std::make_unique<fake_buffer>(fake_buffer{0});
            CHECK(r.alloc_buffer(d, sz, &p, err), "weights alloc: %s", err.c_str());
            r.set_owner(p, owner.get());
            owner->usage = 1;   // llama marks weight buffers WEIGHTS after the load
            fd.fill(d, p, sz, seed + d);
            mb.items.push_back({d, p, sz, seed + d, pxa_res::CLS_WEIGHTS, owner.get()});
            mb.owners.push_back(std::move(owner));
            for (int k = 0; k < 5; ++k) {
                void * q = nullptr;
                const size_t ssz = scale * (40 * 1024 + 1000 * k + d);
                CHECK(r.alloc_slice(d, ssz, &q, err), "slice alloc: %s", err.c_str());
                fd.fill(d, q, ssz, seed + 100 + 10 * d + k);
                mb.items.push_back({d, q, ssz, seed + 100u + 10u * d + k, pxa_res::CLS_WEIGHTS, nullptr, true});
            }
        }
    }
    for (int d = 0; d < ndev; ++d) {
        void * p = nullptr;
        const size_t sz = scale * (70 * 1024 + d);
        auto kv = std::make_unique<fake_buffer>(fake_buffer{0});
        CHECK(r.alloc_buffer(d, sz, &p, err), "kv alloc: %s", err.c_str());
        r.set_owner(p, kv.get());
        fd.fill(d, p, sz, seed + 500 + d);
        mb.items.push_back({d, p, sz, seed + 500u + d, pxa_res::CLS_STATE, kv.get()});
        mb.owners.push_back(std::move(kv));
        for (int k = 0; k < 3; ++k) {   // KV slices under a tensor split: runtime -> state
            void * q = nullptr;
            const size_t ssz = scale * (9 * 1024 + 333 * k);
            CHECK(r.alloc_slice(d, ssz, &q, err), "kv slice alloc: %s", err.c_str());
            fd.fill(d, q, ssz, seed + 600 + 10 * d + k);
            mb.items.push_back({d, q, ssz, seed + 600u + 10u * d + k, pxa_res::CLS_STATE, nullptr, true});
        }
        auto cb = std::make_unique<fake_buffer>(fake_buffer{2});
        CHECK(r.alloc_buffer(d, scale * 50 * 1024, &p, err), "compute alloc: %s", err.c_str());
        r.set_owner(p, cb.get());
        fd.fill(d, p, scale * 50 * 1024, seed + 900 + d);
        mb.items.push_back({d, p, scale * 50 * 1024, seed + 900u + d, pxa_res::CLS_SCRATCH, cb.get()});
        mb.owners.push_back(std::move(cb));
    }
    pxa_res::active() = nullptr;
}

static bool all_intact(fake_dev & fd, const model_bufs & mb, bool with_scratch = false) {
    bool ok = true;
    for (auto & it : mb.items) {
        if (it.cls < 0) continue;
        if (it.cls == pxa_res::CLS_SCRATCH && !with_scratch) continue;
        if (!fd.same(it.dev, it.p, it.size, it.seed)) {
            fprintf(stderr, "  content lost: dev %d %p size %zu class %s\n", it.dev, it.p, it.size, pxa_res::cls_name(it.cls));
            ok = false;
        }
    }
    return ok;
}
static bool none_mapped(fake_dev & fd, const model_bufs & mb) {
    for (auto & it : mb.items) if (it.cls >= 0 && fd.mapped_any(it.dev, it.p, it.size)) return false;
    return true;
}
static bool all_mapped(fake_dev & fd, const model_bufs & mb) {
    for (auto & it : mb.items) if (it.cls >= 0 && !fd.mapped_all(it.dev, it.p, it.size)) return false;
    return true;
}

// what the group holds for a class: whole buffers at their size, slices as the packed slab they
// live in (each slice starts on a 256-byte boundary, so the slab carries the padding between them)
static uint64_t bytes_of(const model_bufs & mb, int cls) {
    uint64_t b = 0;
    std::map<int, uint64_t> top;
    for (auto & it : mb.items) {
        if (it.cls != cls) continue;
        if (!it.slice) { b += it.size; continue; }
        uint64_t & t = top[it.dev];
        t = pxa_res::round_up(t, 256) + it.size;
    }
    for (auto & t : top) b += t.second;
    return b;
}

static void test_single_model() {
    fake_dev fd;
    pxa_res::residency r("A", &fd);
    model_bufs mb;
    build_model(r, fd, 2, 7, mb);
    fake_pool pool(&fd, 0, 4 * fake_dev::G);
    r.add_pool(&pool);

    uint64_t sz[pxa_res::CLS_N];
    r.sizes(sz, nullptr);
    CHECK(sz[pxa_res::CLS_WEIGHTS] == bytes_of(mb, pxa_res::CLS_WEIGHTS), "weights bytes %llu vs %llu",
          (unsigned long long) sz[pxa_res::CLS_WEIGHTS], (unsigned long long) bytes_of(mb, pxa_res::CLS_WEIGHTS));
    CHECK(sz[pxa_res::CLS_STATE] == bytes_of(mb, pxa_res::CLS_STATE), "state bytes %llu vs %llu",
          (unsigned long long) sz[pxa_res::CLS_STATE], (unsigned long long) bytes_of(mb, pxa_res::CLS_STATE));

    const size_t used0 = fd.used[0] + fd.used[1];
    CHECK(used0 > 0, "nothing mapped");

    pxa_res::op_stats st;
    CHECK(r.park(&st), "park: %s", st.err);
    CHECK(r.parked(), "not parked");
    CHECK(fd.used[0] == 0 && fd.used[1] == 0, "VRAM still held after park: %zu %zu", fd.used[0], fd.used[1]);
    CHECK(none_mapped(fd, mb), "a range is still mapped after park");
    CHECK(pool.parked, "pool not parked");
    CHECK(st.bytes_released == used0, "released %llu of %zu", (unsigned long long) st.bytes_released, used0);
    CHECK(st.bytes_resident == 0, "resident after a full park: %llu", (unsigned long long) st.bytes_resident);
    CHECK(st.bytes_mirror_new == bytes_of(mb, pxa_res::CLS_WEIGHTS), "first park mirrors all weights");
    CHECK(st.bytes_copied == bytes_of(mb, pxa_res::CLS_WEIGHTS) + bytes_of(mb, pxa_res::CLS_STATE), "first park copy bytes %llu",
          (unsigned long long) st.bytes_copied);
    CHECK(st.n_dev == 2, "n_dev %d", st.n_dev);

    {   // a parked group takes no allocation
        pxa_res::active() = &r;
        void * p = nullptr;
        std::string e;
        CHECK(!r.alloc_buffer(0, 1024, &p, e), "allocation accepted while parked");
        CHECK(!r.alloc_slice(0, 1024, &p, e), "slice accepted while parked");
        pxa_res::active() = nullptr;
    }

    CHECK(r.unpark(&st), "unpark: %s", st.err);
    CHECK(!r.parked(), "still parked");
    CHECK(all_mapped(fd, mb), "a range is missing after unpark");
    CHECK(all_intact(fd, mb), "content not restored");
    CHECK(!pool.parked, "pool not unparked");
    CHECK(fd.used[0] + fd.used[1] == used0, "VRAM after unpark %zu != before %zu", fd.used[0] + fd.used[1], used0);
    CHECK(st.bytes_copied == bytes_of(mb, pxa_res::CLS_WEIGHTS) + bytes_of(mb, pxa_res::CLS_STATE), "unpark copy bytes");

    // the KV changes; the second park copies only state and brings back the NEW KV
    for (auto & it : mb.items) {
        if (it.cls == pxa_res::CLS_STATE) {
            it.seed += 1000;
            fd.fill(it.dev, it.p, it.size, it.seed);
        }
    }
    CHECK(r.park(&st), "park 2: %s", st.err);
    CHECK(st.bytes_mirror_new == 0, "weights mirrored twice");
    CHECK(st.bytes_copied == bytes_of(mb, pxa_res::CLS_STATE), "second park copied %llu, want state only %llu",
          (unsigned long long) st.bytes_copied, (unsigned long long) bytes_of(mb, pxa_res::CLS_STATE));
    CHECK(r.unpark(&st), "unpark 2: %s", st.err);
    CHECK(all_intact(fd, mb), "content after second round trip");

    // the pool refuses to be parked while it has live allocations
    pool.live = 1;
    CHECK(!r.park(&st), "park succeeded with a live pool allocation");
    CHECK(!r.parked() && all_mapped(fd, mb), "failed park changed something");
    pool.live = 0;
    r.remove_pool(&pool);
}

static void test_pin_budget() {
    fake_dev fd;
    pxa_res::residency r("B", &fd);
    model_bufs mb;
    build_model(r, fd, 1, 11, mb);
    pxa_res::pinned_limit() = 1024;   // far too small
    pxa_res::op_stats st;
    CHECK(!r.park(&st), "park should fail on the pinned budget");
    CHECK(std::string(st.err).find("budget") != std::string::npos, "budget error not reported: %s", st.err);
    CHECK(!r.parked() && all_mapped(fd, mb) && all_intact(fd, mb), "failed park changed the model");
    pxa_res::pinned_limit() = 0;
    CHECK(r.park(&st), "park after lifting the budget: %s", st.err);
    CHECK(r.unpark(&st) && all_intact(fd, mb), "round trip after budget");
}

static void test_unpark_rollback() {
    fake_dev fd;
    pxa_res::residency r("C", &fd);
    model_bufs mb;
    build_model(r, fd, 3, 21, mb);
    pxa_res::op_stats st;
    CHECK(r.park(&st), "park: %s", st.err);
    fd.fail_map_on_dev = 1;
    CHECK(!r.unpark(&st), "unpark should fail on device 1");
    CHECK(std::string(st.err).find("device 1") != std::string::npos, "failing device not named: %s", st.err);
    CHECK(r.parked(), "failed unpark left the group unparked");
    CHECK(fd.used[0] == 0 && fd.used[1] == 0 && fd.used[2] == 0, "failed unpark leaked mappings: %zu %zu %zu",
          fd.used[0], fd.used[1], fd.used[2]);
    fd.fail_map_on_dev = -1;
    CHECK(r.unpark(&st), "unpark after the failure cleared: %s", st.err);
    CHECK(all_intact(fd, mb), "content after a failed then good unpark");
}

static void test_verify() {
    fake_dev fd;
    pxa_res::residency r("D", &fd);
    model_bufs mb;
    build_model(r, fd, 1, 31, mb);
    pxa_res::op_stats st;
    CHECK(r.park(&st) && r.unpark(&st), "round trip");
    for (auto & it : mb.items) {   // something writes into a weight behind the mirror's back
        if (it.cls == pxa_res::CLS_WEIGHTS && it.owner) {
            fd.poke(it.dev, it.p, 5, 0xFF);
            fd.poke(it.dev, it.p, 77, 0x01);
            break;
        }
    }
    setenv("PXA_SWAP_VERIFY", "1", 1);
    CHECK(r.park(&st), "verify park: %s", st.err);
    unsetenv("PXA_SWAP_VERIFY");
    CHECK(st.verify_bad == 2, "verify counted %llu changed bytes, want 2", (unsigned long long) st.verify_bad);
}

static void test_two_models_one_card() {
    fake_dev fd;
    pxa_res::residency A("A", &fd), B("B", &fd);
    model_bufs ma, mbb;
    build_model(A, fd, 1, 41, ma, 8);
    pxa_res::op_stats st;
    CHECK(A.park(&st), "park A at boot: %s", st.err);
    build_model(B, fd, 1, 51, mbb, 8);
    const size_t one = fd.used[0];
    fd.capacity[0] = one + one / 4;   // a card that holds one of them (they are the same size)
    CHECK(B.park(&st), "park B at boot: %s", st.err);
    CHECK(A.unpark(&st), "A in: %s", st.err);
    for (int round = 0; round < 3; ++round) {
        for (auto & it : ma.items) if (it.cls == pxa_res::CLS_STATE) { it.seed += 7; fd.fill(it.dev, it.p, it.size, it.seed); }
        CHECK(A.park(&st), "A out (round %d): %s", round, st.err);
        CHECK(B.unpark(&st), "B in (round %d): %s", round, st.err);
        CHECK(all_intact(fd, mbb), "B content (round %d)", round);
        for (auto & it : mbb.items) if (it.cls == pxa_res::CLS_STATE) { it.seed += 3; fd.fill(it.dev, it.p, it.size, it.seed); }
        CHECK(B.park(&st), "B out (round %d): %s", round, st.err);
        CHECK(A.unpark(&st), "A in (round %d): %s", round, st.err);
        CHECK(all_intact(fd, ma), "A content (round %d)", round);
    }
    CHECK(fd.used[0] <= fd.capacity[0], "over capacity");
}

static void test_free_while_parked() {
    fake_dev fd;
    pxa_res::residency r("E", &fd);
    model_bufs mb;
    build_model(r, fd, 1, 61, mb);
    pxa_res::op_stats st;
    const uint64_t pin0 = pxa_res::pinned_used().load();
    CHECK(r.park(&st), "park: %s", st.err);
    CHECK(pxa_res::pinned_used().load() > pin0, "no pinned bytes accounted");
    for (auto & it : mb.items) {   // a context is freed while its model is parked
        if (it.cls == pxa_res::CLS_STATE && it.owner) {
            CHECK(r.free_buffer(it.p), "free of a parked buffer");
            it.cls = -1;
            break;
        }
    }
    CHECK(r.unpark(&st), "unpark after a free: %s", st.err);
    CHECK(all_intact(fd, mb), "content after freeing a parked buffer");
}

static void test_prime() {
    fake_dev fd;
    pxa_res::residency r("P", &fd);
    model_bufs mb;
    build_model(r, fd, 2, 71, mb);
    const size_t used0 = fd.used[0] + fd.used[1];
    pxa_res::op_stats st;
    CHECK(r.prime(&st), "prime: %s", st.err);
    CHECK(st.bytes_mirror_new == bytes_of(mb, pxa_res::CLS_WEIGHTS), "prime mirrored %llu, want all weights %llu",
          (unsigned long long) st.bytes_mirror_new, (unsigned long long) bytes_of(mb, pxa_res::CLS_WEIGHTS));
    CHECK(!r.parked() && fd.used[0] + fd.used[1] == used0 && all_intact(fd, mb, true), "prime changed the cards");
    CHECK(r.prime(&st) && st.bytes_copied == 0, "a second prime copied %llu bytes", (unsigned long long) st.bytes_copied);
    CHECK(r.park(&st), "park: %s", st.err);
    CHECK(st.bytes_mirror_new == 0 && st.bytes_copied == bytes_of(mb, pxa_res::CLS_STATE),
          "park after prime copied %llu (want KV only %llu)", (unsigned long long) st.bytes_copied,
          (unsigned long long) bytes_of(mb, pxa_res::CLS_STATE));
    CHECK(r.unpark(&st) && all_intact(fd, mb), "round trip after prime");
}

static void test_partial_and_reclaim() {
    // a big model A and a small model B
    fake_dev fd;
    pxa_res::residency A("A", &fd), B("B", &fd);
    model_bufs ma, mbb;
    build_model(A, fd, 1, 81, ma, 8);
    const size_t a_bytes = fd.used[0];
    pxa_res::op_stats st;
    CHECK(A.prime(&st), "prime A");
    CHECK(A.park(&st), "park A: %s", st.err);
    build_model(B, fd, 1, 91, mbb, 2);
    const size_t b_bytes = fd.used[0];
    CHECK(B.park(&st), "park B: %s", st.err);
    fd.capacity[0] = a_bytes + b_bytes + b_bytes / 2;   // A + B fit together
    CHECK(A.unpark(&st), "A in: %s", st.err);

    // A out, but only as much as B needs: B fits beside A, so nothing of A has to go
    size_t need[pxa_res::MAX_DEV];
    B.need(need);
    CHECK(need[0] == b_bytes, "B needs %zu, holds %zu", need[0], b_bytes);
    size_t want[pxa_res::MAX_DEV] = {};
    want[0] = need[0];
    CHECK(A.park(&st, want), "partial park: %s", st.err);
    CHECK(A.parked(), "A not marked parked");
    CHECK(st.bytes_released == 0 && st.bytes_copied == 0, "released %llu copied %llu with room to spare",
          (unsigned long long) st.bytes_released, (unsigned long long) st.bytes_copied);
    CHECK(B.unpark(&st), "B in beside A: %s", st.err);
    CHECK(all_intact(fd, ma) && all_intact(fd, mbb), "co-resident contents");

    // B out, A in: A never left, so the swap moves nothing
    CHECK(B.park(&st), "B out: %s", st.err);
    CHECK(A.unpark(&st), "A back: %s", st.err);
    CHECK(st.bytes_copied == 0, "A came back from a partial park by copying %llu bytes", (unsigned long long) st.bytes_copied);
    CHECK(all_intact(fd, ma), "A content after the round trip");

    // a smaller card: B needs more than is free; A releases scratch, then weights, and keeps its KV
    fd.capacity[0] = a_bytes + b_bytes / 2;
    B.need(need);
    want[0] = need[0];
    CHECK(A.park(&st, want), "partial park 2: %s", st.err);
    CHECK(st.bytes_released >= b_bytes - (fd.capacity[0] - a_bytes) && st.bytes_released < a_bytes,
          "released %llu (B needs %zu, card free before %zu)", (unsigned long long) st.bytes_released, b_bytes,
          fd.capacity[0] - a_bytes);
    bool kv_kept = true;
    for (auto & it : ma.items) if (it.cls == pxa_res::CLS_STATE && !fd.mapped_all(it.dev, it.p, it.size)) kv_kept = false;
    CHECK(kv_kept, "A's KV left the card although weights and scratch covered B's need");
    CHECK(B.unpark(&st), "B in after a partial park: %s", st.err);
    CHECK(all_intact(fd, mbb), "B content");

    // B grows (a longer prompt): the memory comes out of parked A on demand
    pxa_res::active() = &B;
    void * big = nullptr;
    std::string err;
    const size_t grow = fd.mem_free(0) + 3 * fake_dev::G;
    CHECK(B.alloc_buffer(0, grow, &big, err), "B growth with reclaim: %s", err.c_str());
    pxa_res::active() = nullptr;
    CHECK(B.free_buffer(big), "free the growth");

    // and A still comes back whole
    CHECK(B.park(&st), "B out: %s", st.err);
    CHECK(A.unpark(&st), "A back after reclaim: %s", st.err);
    CHECK(all_intact(fd, ma), "A content after being reclaimed from");
}

static void test_unpark_makes_room() {
    // A is parked with everything still on the card (a keep park with nothing to make room for); a
    // card that holds only one of them; B's unpark must take the room from parked A by itself
    fake_dev fd;
    pxa_res::residency A("A", &fd), B("B", &fd);
    model_bufs ma, mbb;
    build_model(A, fd, 1, 101, ma, 4);
    const size_t one = fd.used[0];
    pxa_res::op_stats st;
    CHECK(A.park(&st), "park A: %s", st.err);
    build_model(B, fd, 1, 111, mbb, 4);
    CHECK(B.park(&st), "park B: %s", st.err);
    fd.capacity[0] = one + one / 2;
    CHECK(A.unpark(&st), "A in: %s", st.err);
    size_t want[pxa_res::MAX_DEV] = {};
    want[0] = 1;   // "leave 1 byte free": nothing needs releasing
    CHECK(A.park(&st, want), "keep park: %s", st.err);
    CHECK(st.bytes_released == 0, "released %llu", (unsigned long long) st.bytes_released);
    CHECK(B.unpark(&st), "B in, taking the room from parked A: %s", st.err);
    CHECK(all_intact(fd, mbb), "B content");
    CHECK(B.park(&st), "B out: %s", st.err);
    CHECK(A.unpark(&st), "A back: %s", st.err);
    CHECK(all_intact(fd, ma), "A content after its room was taken");
}

int main() {
    // small chunks: every buffer spans several, so partial releases and re-cuts are exercised
    pxa_res::residency::chunk_max() = 3 * fake_dev::G;
    pxa_res::residency::slab_grow() = 2 * fake_dev::G;
    test_prime();
    test_single_model();
    test_pin_budget();
    test_unpark_rollback();
    test_verify();
    test_two_models_one_card();
    test_free_while_parked();
    test_partial_and_reclaim();
    test_unpark_makes_room();
    if (g_fail) {
        fprintf(stderr, "test-pxa-residency: %d FAILED\n", g_fail);
        return 1;
    }
    printf("test-pxa-residency: all checks passed\n");
    return 0;
}
