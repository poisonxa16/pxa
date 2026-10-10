// PXA HOT SWAP -- router (examples/server/pxa-hotswap.h): registry parsing, routing and the swap
// state machine, driven by fake park/unpark callbacks that take real time.
//
// Claims checked:
//   1. registration: NAME=PATH [flags] parses; bad forms and per-model flags that cannot be per
//      model (-m, --alias, --port...) are refused; a client can name a model by id, file name or stem;
//   2. a request for the active model runs at once, with no swap;
//   3. a request for another model waits until the active model has DRAINED, then exactly one
//      park(active) + unpark(target) runs, and the request is admitted only after unpark returned
//      (no token from a half-loaded model);
//   4. a burst for the target is admitted together after ONE swap;
//   5. fairness: once a request for another model is waiting, new requests for the active model
//      queue behind it instead of starving it;
//   6. a failed unpark fails the requests that wanted the target and puts the previous model back;
//      a failed park changes nothing and fails only those requests;
//   7. a waiter whose client goes away leaves the queue and causes no swap;
//   8. acquire_active() pins the model on the cards and waits out a swap in progress; an unnamed
//      request (acquire_any) queues like any other and runs on whatever model is on the cards then;
//   9. under 12 threads x 200 random requests over 3 models: at most one model is ever on the
//      cards, no lease is ever held on a model that is not, park never runs while the parked model
//      has a lease, and every request is served by the model it asked for.
//  10. a resumed context shift trusts the cache only when the kept prefix and the surviving
//      suffix still match. Qwen keeps nothing at the front (n_keep 0). Gemma keeps the BOS
//      token there, so a check that starts at cache[0] rejects a good resume.
// CPU only, no model, no GPU. Exit 0 on pass.

#include "pxa-hotswap.h"
#include "pxa-shift-keep.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

static std::atomic<int> g_fail{0};
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

using namespace pxa_hs;

struct fake_ops : public swap_ops {
    int n;
    std::mutex mu;
    std::vector<int> on_card;          // 1 if the model's weights are on the cards
    std::vector<int> leases;           // leases the TEST holds per model (maintained by the test)
    std::vector<std::string> log;
    int  park_ms = 20, unpark_ms = 40;
    int  fail_unpark = -1, fail_park = -1;
    std::atomic<int> n_park{0}, n_unpark{0};

    explicit fake_ops(int n_, int active) : n(n_), on_card(n_, 0), leases(n_, 0) {
        if (active >= 0) on_card[active] = 1;
    }
    int cards_used() {
        int c = 0;
        for (int v : on_card) c += v;
        return c;
    }
    bool park(int idx, int next, std::string & err) override {
        (void) next;
        {
            std::lock_guard<std::mutex> lk(mu);
            log.push_back("park " + std::to_string(idx));
            if (leases[idx] != 0) { ++g_fail; fprintf(stderr, "FAIL: park(%d) with %d leases held\n", idx, leases[idx]); }
            if (!on_card[idx])    { ++g_fail; fprintf(stderr, "FAIL: park(%d) of a model not on the cards\n", idx); }
            if (idx == fail_park) { err = "injected park failure"; return false; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(park_ms));
        std::lock_guard<std::mutex> lk(mu);
        on_card[idx] = 0;
        ++n_park;
        return true;
    }
    bool unpark(int idx, std::string & err) override {
        {
            std::lock_guard<std::mutex> lk(mu);
            log.push_back("unpark " + std::to_string(idx));
            if (cards_used() != 0) { ++g_fail; fprintf(stderr, "FAIL: unpark(%d) while another model is on the cards\n", idx); }
            if (idx == fail_unpark) { err = "injected out of memory"; return false; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(unpark_ms));
        std::lock_guard<std::mutex> lk(mu);
        on_card[idx] = 1;
        ++n_unpark;
        return true;
    }
    // what a request does once admitted: the model must be fully on the cards
    void use(int idx) {
        std::lock_guard<std::mutex> lk(mu);
        if (!on_card[idx]) { ++g_fail; fprintf(stderr, "FAIL: request served by model %d which is not on the cards\n", idx); }
        if (cards_used() != 1) { ++g_fail; fprintf(stderr, "FAIL: %d models on the cards\n", cards_used()); }
        leases[idx]++;
    }
    void done(int idx) {
        std::lock_guard<std::mutex> lk(mu);
        leases[idx]--;
    }
};

static std::vector<std::vector<std::string>> names3() {
    return {{"qwen", "Qwen3.8-27B.gguf", "Qwen3.8-27B"}, {"gemma", "g.gguf", "g"}, {"small", "s.gguf", "s"}};
}

static void test_parse() {
    model_spec s;
    std::string err;
    CHECK(parse_model_spec("gemma=/m/gemma-4-12b.gguf -c 8192 -sm layer", s, err), "parse: %s", err.c_str());
    CHECK(s.name == "gemma" && s.path == "/m/gemma-4-12b.gguf", "name/path %s %s", s.name.c_str(), s.path.c_str());
    CHECK(s.args.size() == 4 && s.args[0] == "-c" && s.args[3] == "layer", "args");
    CHECK(parse_model_spec("  a=/x.gguf  ", s, err) && s.args.empty(), "whitespace");
    CHECK(!parse_model_spec("", s, err), "empty accepted");
    CHECK(!parse_model_spec("/path/only.gguf", s, err), "missing name accepted");
    CHECK(!parse_model_spec("=x.gguf", s, err), "empty name accepted");
    CHECK(!parse_model_spec("a=", s, err), "empty path accepted");
    CHECK(!parse_model_spec("a=/x.gguf -m /y.gguf", s, err), "per-model -m accepted");
    CHECK(!parse_model_spec("a=/x.gguf --port 1", s, err), "per-model --port accepted");
    CHECK(file_stem("/a/b/Qwen3.8-27B-PXQ-mix27.gguf") == "Qwen3.8-27B-PXQ-mix27", "stem");
    CHECK(file_stem("plain") == "plain", "stem plain");

    fake_ops ops(3, 0);
    router r(names3(), 0, &ops);
    CHECK(r.find("qwen") == 0 && r.find("Qwen3.8-27B") == 0 && r.find("g.gguf") == 1 && r.find("s") == 2, "find");
    CHECK(r.find("") == -1 && r.find("gpt-4o") == -1, "unknown");
}

static void test_fast_path() {
    fake_ops ops(3, 0);
    router r(names3(), 0, &ops);
    std::string err;
    for (int i = 0; i < 5; ++i) {
        CHECK(r.acquire(0, err), "acquire: %s", err.c_str());
        ops.use(0); ops.done(0);
        r.release(0);
    }
    CHECK(ops.n_park == 0 && ops.n_unpark == 0, "a swap happened on the fast path");
    CHECK(r.status()[0].n_requests == 5, "n_requests %llu", (unsigned long long) r.status()[0].n_requests);
}

static void test_drain_then_swap() {
    fake_ops ops(3, 0);
    router r(names3(), 0, &ops);
    std::string err;
    CHECK(r.acquire(0, err), "A lease");
    ops.use(0);
    std::atomic<bool> b_in{false};
    std::thread tb([&] {
        std::string e;
        CHECK(r.acquire(1, e), "B: %s", e.c_str());
        ops.use(1);
        b_in = true;
        ops.done(1);
        r.release(1);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    CHECK(!b_in && ops.n_park == 0, "B ran or A was parked while A still had a request in flight");
    ops.done(0);
    r.release(0);
    tb.join();
    CHECK(b_in, "B never ran");
    CHECK(ops.n_park == 1 && ops.n_unpark == 1, "park %d unpark %d", ops.n_park.load(), ops.n_unpark.load());
    CHECK(r.active() == 1, "active %d", r.active());
    CHECK(ops.log.size() == 2 && ops.log[0] == "park 0" && ops.log[1] == "unpark 1", "order");
}

static void test_burst_one_swap() {
    fake_ops ops(3, 0);
    router r(names3(), 0, &ops);
    std::string err;
    CHECK(r.acquire(0, err), "A lease");
    std::vector<std::thread> th;
    std::atomic<int> served{0};
    for (int i = 0; i < 6; ++i) {
        th.emplace_back([&] {
            std::string e;
            CHECK(r.acquire(1, e), "B: %s", e.c_str());
            ops.use(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            ++served;
            ops.done(1);
            r.release(1);
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(r.status()[1].waiting == 6, "waiting %d", r.status()[1].waiting);
    r.release(0);
    for (auto & t : th) t.join();
    CHECK(served == 6, "served %d", served.load());
    CHECK(ops.n_unpark == 1, "a burst of 6 cost %d swaps", ops.n_unpark.load());
}

static void test_fairness() {
    fake_ops ops(3, 0);
    router r(names3(), 0, &ops);
    std::string err;
    CHECK(r.acquire(0, err), "A lease");
    std::vector<std::string> order;
    std::mutex om;
    std::thread tb([&] {
        std::string e;
        CHECK(r.acquire(1, e), "B");
        { std::lock_guard<std::mutex> lk(om); order.push_back("B"); }
        r.release(1);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    std::thread ta([&] {   // arrives after B is waiting: must not jump ahead of B
        std::string e;
        CHECK(r.acquire(0, e), "A2");
        { std::lock_guard<std::mutex> lk(om); order.push_back("A2"); }
        r.release(0);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    CHECK(order.empty(), "someone ran while A held the cards");
    r.release(0);
    tb.join();
    ta.join();
    CHECK(order.size() == 2 && order[0] == "B" && order[1] == "A2", "order %s,%s",
          order.size() > 0 ? order[0].c_str() : "-", order.size() > 1 ? order[1].c_str() : "-");
    CHECK(ops.n_unpark == 2 && r.active() == 0, "unparks %d active %d", ops.n_unpark.load(), r.active());
}

static void test_failures() {
    {
        fake_ops ops(3, 0);
        router r(names3(), 0, &ops);
        ops.fail_unpark = 1;
        std::string err;
        CHECK(!r.acquire(1, err), "acquire of a model that cannot load succeeded");
        CHECK(err.find("injected out of memory") != std::string::npos, "error text: %s", err.c_str());
        CHECK(r.active() == 0 && ops.on_card[0] == 1, "previous model not put back (active %d)", r.active());
        CHECK(r.status()[1].n_swap_fail == 1, "failure not counted");
        CHECK(r.acquire(0, err), "previous model serves after the failure: %s", err.c_str());
        ops.use(0); ops.done(0);
        r.release(0);
        ops.fail_unpark = -1;
        CHECK(r.acquire(1, err), "the model loads once the fault clears: %s", err.c_str());
        ops.use(1); ops.done(1);
        r.release(1);
    }
    {
        fake_ops ops(3, 0);
        router r(names3(), 0, &ops);
        ops.fail_park = 0;
        std::string err;
        CHECK(!r.acquire(2, err), "swap with a failing park succeeded");
        CHECK(r.active() == 0 && ops.on_card[0] == 1 && ops.n_unpark == 0, "failed park changed the cards");
        CHECK(r.acquire(0, err), "active model still serves");
        r.release(0);
    }
}

static void test_cancel() {
    fake_ops ops(3, 0);
    router r(names3(), 0, &ops);
    std::string err;
    CHECK(r.acquire(0, err), "A");
    std::atomic<bool> gone{false};
    std::thread tb([&] {
        std::string e;
        CHECK(!r.acquire(1, e, [&] { return gone.load(); }), "a cancelled waiter got in");
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    gone = true;
    tb.join();
    r.release(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    CHECK(ops.n_park == 0 && ops.n_unpark == 0 && r.active() == 0, "a cancelled request caused a swap");
    CHECK(r.status()[1].waiting == 0, "cancelled ticket left in the queue");
    // timeout
    CHECK(r.acquire(0, err), "A again");
    CHECK(!r.acquire(1, err, nullptr, 50) && err.find("timed out") != std::string::npos, "timeout: %s", err.c_str());
    r.release(0);
}

static void test_acquire_active() {
    fake_ops ops(3, 0);
    ops.unpark_ms = 200;
    router r(names3(), 0, &ops);
    std::thread tb([&] {
        std::string e;
        CHECK(r.acquire(2, e), "C");
        ops.use(2);
        ops.done(2);
        r.release(2);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));   // swap in progress
    CHECK(r.swapping(), "not swapping");
    const int a = r.acquire_active();
    CHECK(a == 2, "acquire_active during a swap returned %d", a);
    CHECK(ops.on_card[2] == 1, "acquire_active returned before the swap finished");
    r.release(a);
    tb.join();
}

static void test_any() {
    // an unnamed request runs on whatever is on the cards; behind a waiting swap it takes its turn
    fake_ops ops(3, 0);
    router r(names3(), 0, &ops);
    std::string err;
    int a = r.acquire_any(err);
    CHECK(a == 0, "fast path any -> %d", a);
    ops.use(0);
    std::thread tb([&] {
        std::string e;
        CHECK(r.acquire(1, e), "B");
        ops.use(1); ops.done(1);
        r.release(1);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    int got = -2;
    std::thread tany([&] {
        std::string e;
        got = r.acquire_any(e);
        if (got >= 0) { ops.use(got); ops.done(got); r.release(got); }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    CHECK(got == -2, "an unnamed request jumped the queue while a swap was waiting");
    ops.done(0);
    r.release(0);
    tb.join();
    tany.join();
    CHECK(got == 1, "the unnamed request behind B's swap ran on %d, want the model B brought in (1)", got);
    CHECK(ops.n_unpark == 1, "an unnamed request caused a swap (%d unparks)", ops.n_unpark.load());

    // nothing on the cards: an unnamed request is refused with a reason
    fake_ops ops2(3, 0);
    router r2(names3(), 0, &ops2);
    ops2.fail_unpark = 2;
    ops2.fail_park = -1;
    std::string e2;
    // make the put-back fail too: model 0 cannot come back either
    struct fail_all : public fake_ops { using fake_ops::fake_ops; bool unpark(int, std::string & e) override { e = "no memory"; return false; } };
    fail_all ops3(3, 0);
    router r3(names3(), 0, &ops3);
    CHECK(!r3.acquire(2, e2), "swap to a model that cannot load");
    CHECK(r3.active() == -1, "active %d after a failed swap and a failed put-back", r3.active());
    CHECK(r3.acquire_any(e2) == -1 && !e2.empty(), "unnamed request with nothing on the cards: %s", e2.c_str());
}

static void test_stress() {
    const int N = 3;
    fake_ops ops(N, 0);
    ops.park_ms = 2;
    ops.unpark_ms = 3;
    router r(names3(), 0, &ops);
    std::vector<std::thread> th;
    std::atomic<int> served{0};
    for (int t = 0; t < 12; ++t) {
        th.emplace_back([&, t] {
            std::mt19937 rng(1000 + t);
            for (int i = 0; i < 200; ++i) {
                int m = (rng() % 10) < 6 ? 0 : (int) (rng() % N);
                std::string e;
                if (rng() % 8 == 0) {   // an unnamed request now and then
                    m = r.acquire_any(e);
                    if (m < 0) { ++g_fail; fprintf(stderr, "FAIL: stress acquire_any: %s\n", e.c_str()); continue; }
                } else if (!r.acquire(m, e)) { ++g_fail; fprintf(stderr, "FAIL: stress acquire(%d): %s\n", m, e.c_str()); continue; }
                ops.use(m);
                if (r.active() != m) { ++g_fail; fprintf(stderr, "FAIL: lease on %d while %d is active\n", m, r.active()); }
                std::this_thread::sleep_for(std::chrono::microseconds(200 + rng() % 400));
                ops.done(m);
                r.release(m);
                ++served;
            }
        });
    }
    for (auto & x : th) x.join();
    CHECK(served == 12 * 200, "served %d", served.load());
    CHECK(ops.cards_used() == 1, "cards hold %d models", ops.cards_used());
    printf("  stress: %d requests, %llu swaps\n", served.load(), (unsigned long long) r.n_swaps());
}

static void test_sleep() {
    fake_ops ops(3, 0);
    ops.park_ms = 5;
    ops.unpark_ms = 5;
    router r(names3(), 0, &ops);
    std::string err;
    CHECK(r.try_sleep(err), "sleep: %s", err.c_str());
    CHECK(r.active() == -1, "active %d after sleep", r.active());
    CHECK(ops.cards_used() == 0, "cards hold %d models", ops.cards_used());
    CHECK(ops.n_park.load() == 1, "parks %d", ops.n_park.load());
    CHECK(!r.try_sleep(err), "a second sleep should refuse");
    CHECK(r.acquire(1, err), "wake by name: %s", err.c_str());
    CHECK(r.active() == 1, "active %d", r.active());
    CHECK(ops.cards_used() == 1 && ops.on_card[1] == 1, "model 1 not alone on the cards");
    r.release(1);
    CHECK(r.try_sleep(err), "sleep again: %s", err.c_str());
    CHECK(ops.cards_used() == 0, "cards not free");
    const int woke = r.acquire_any(err);
    CHECK(woke == 1, "unnamed request woke %d (%s)", woke, err.c_str());
    CHECK(!r.try_sleep(err), "sleep while a lease is held");
    CHECK(r.active() == 1, "lease lost, active %d", r.active());
    r.release(woke);
    printf("  sleep: park frees the cards, the next request loads one back\n");
}

static void test_shift_keep() {
    using V = std::vector<int32_t>;
    // Qwen: add_bos false, n_keep 0. The cache is the prompt with a prefix removed.
    V qwen_prompt = {10, 11, 12, 13, 14, 15, 16};
    V qwen_cache = {13, 14, 15, 16};
    CHECK(pxa_shift_keep_matches(qwen_prompt, (int32_t) qwen_prompt.size(),
                                 qwen_cache, (int32_t) qwen_cache.size(), 3, 0),
          "qwen keep should match");
    qwen_cache[0] = 99;
    CHECK(!pxa_shift_keep_matches(qwen_prompt, (int32_t) qwen_prompt.size(),
                                  qwen_cache, (int32_t) qwen_cache.size(), 3, 0),
          "qwen mismatch must not match");

    // Gemma: add_bos, n_keep 1. cache = [BOS] + prompt[n_keep + discarded :].
    // Comparing prompt[discarded + i] with cache[i] sees 13 against the BOS and fails.
    const int32_t bos = 2;
    V gemma_prompt = {bos, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    V gemma_cache = {bos, 14, 15, 16, 17, 18};
    CHECK(pxa_shift_keep_matches(gemma_prompt, (int32_t) gemma_prompt.size(),
                                 gemma_cache, (int32_t) gemma_cache.size(), 4, 1),
          "gemma keep should match");
    CHECK(gemma_prompt[4] != gemma_cache[0], "the old index pairs %d with %d",
          gemma_prompt[4], gemma_cache[0]);
    V gemma_bad_prefix = gemma_cache;
    gemma_bad_prefix[0] = 9;
    CHECK(!pxa_shift_keep_matches(gemma_prompt, (int32_t) gemma_prompt.size(),
                                  gemma_bad_prefix, (int32_t) gemma_bad_prefix.size(), 4, 1),
          "gemma wrong prefix must not match");
    V gemma_bad_suffix = gemma_cache;
    gemma_bad_suffix[2] = 99;
    CHECK(!pxa_shift_keep_matches(gemma_prompt, (int32_t) gemma_prompt.size(),
                                  gemma_bad_suffix, (int32_t) gemma_bad_suffix.size(), 4, 1),
          "gemma wrong suffix must not match");
    printf("  shift keep: qwen n_keep 0 and gemma n_keep 1\n");
}

int main() {
    test_shift_keep();
    test_parse();
    test_fast_path();
    test_drain_then_swap();
    test_burst_one_swap();
    test_fairness();
    test_failures();
    test_cancel();
    test_acquire_active();
    test_any();
    test_stress();
    test_sleep();
    if (g_fail) {
        fprintf(stderr, "test-pxa-hotswap: %d FAILED\n", g_fail.load());
        return 1;
    }
    printf("test-pxa-hotswap: all checks passed\n");
    return 0;
}
