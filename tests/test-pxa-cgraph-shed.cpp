// 2026-10-20: the PXA_CGRAPH_OOM_GUARD headroom shed rule (ggml-cuda/pxa/pxa-cgraph-shed.h).
// A model of the backend's per-key exec life cycle (ggml-cuda.cu: no instance -> capture + instantiate, the launch
// right after a capture is not a replay, the headroom shed runs before every instantiate while the card is short,
// four consecutive updates send the key eager) driven with two keys that alternate on one context. With the old
// rule (never replayed = stale) they destroy each other's exec forever and both keys go eager; with the recurring-key
// guard each key replays from its second appearance on. A one-off key is still shed.
#include "../ggml/src/ggml-cuda/pxa/pxa-cgraph-shed.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

struct key_state {
    bool instance = false;
    long n_captures = 0;
    long n_replays = 0;
    int  consec = 0;
    bool eager = false;
};

struct result { long replays = 0; long captures = 0; int eager_keys = 0; };

// headroom_short: every instantiate sees free < headroom (the shed runs each time)
static result run(const std::vector<std::string> & seq, bool recur_guard, bool headroom_short) {
    std::map<std::string, key_state> keys;
    for (const auto & k : seq) {
        key_state & g = keys[k];
        if (g.eager) continue;
        const bool update_required = !g.instance;
        if (update_required) { g.consec++; } else { g.consec = 0; }
        if (g.consec >= 4) { g.eager = true; g.instance = false; continue; }
        if (update_required) {
            g.n_captures++;
            if (headroom_short) {
                for (auto & kv : keys) {
                    key_state & o = kv.second;
                    if (&o == &g || !o.instance) continue;
                    if (pxa_cgraph_headroom_stale(o.n_replays, o.n_captures, recur_guard)) o.instance = false;
                }
            }
            g.instance = true;
        } else {
            g.n_replays++;
        }
    }
    result r;
    for (const auto & kv : keys) { r.replays += kv.second.n_replays; r.captures += kv.second.n_captures; r.eager_keys += kv.second.eager; }
    return r;
}

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++n_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

int main() {
    // the predicate itself
    CHECK(!pxa_cgraph_headroom_stale(1, 1, true), "a replayed exec is never stale");
    CHECK(!pxa_cgraph_headroom_stale(3, 9, false), "a replayed exec is never stale (old rule)");
    CHECK(pxa_cgraph_headroom_stale(0, 1, true), "a one-off exec (captured once, never replayed) is stale");
    CHECK(!pxa_cgraph_headroom_stale(0, 2, true), "a recurring key is not stale");
    CHECK(pxa_cgraph_headroom_stale(0, 2, false), "old rule: never replayed = stale");

    std::vector<std::string> alt;
    for (int i = 0; i < 40; ++i) { alt.push_back("decode"); alt.push_back("verify"); }

    // plenty of memory: both rules replay every appearance after the first
    const result ok_old = run(alt, false, false), ok_new = run(alt, true, false);
    CHECK(ok_old.replays == 78 && ok_new.replays == 78, "no shortfall: replays old %ld new %ld (want 78)", ok_old.replays, ok_new.replays);

    // shortfall, old rule: the two keys ping-pong until one of them hits four consecutive updates and goes eager
    const result short_old = run(alt, false, true);
    CHECK(short_old.eager_keys >= 1 && short_old.replays < 40, "old rule under shortfall: replays %ld eager keys %d (the ping-pong)",
          short_old.replays, short_old.eager_keys);

    // shortfall, recurring-key guard: one extra capture per key, then replays
    const result short_new = run(alt, true, true);
    CHECK(short_new.eager_keys == 0 && short_new.replays >= 75 && short_new.captures <= 4,
          "guard under shortfall: replays %ld captures %ld eager keys %d", short_new.replays, short_new.captures, short_new.eager_keys);

    // a one-off key captured once in between is still shed under the guard (the recurring keys keep theirs)
    std::vector<std::string> one = { "decode", "verify", "decode", "verify", "prefill-once", "decode", "verify", "decode" };
    const result r1 = run(one, true, true);
    CHECK(r1.eager_keys == 0, "one-off key: eager keys %d", r1.eager_keys);

    // a tensor split: ~130 segment keys per device per token, each its own exec (ggml-cuda.cu lru cap note)
    std::vector<std::string> ts;
    for (int t = 0; t < 20; ++t) for (int k = 0; k < 130; ++k) ts.push_back("seg" + std::to_string(k));
    const result ts_old = run(ts, false, true), ts_new = run(ts, true, true), ts_ok = run(ts, true, false);
    CHECK(ts_old.eager_keys >= 100, "old rule, tensor split under shortfall: eager keys %d of 130", ts_old.eager_keys);
    CHECK(ts_new.eager_keys == 0 && ts_new.replays == ts_ok.replays - 130, "guard, tensor split: eager %d replays %ld (no shortfall %ld)",
          ts_new.eager_keys, ts_new.replays, ts_ok.replays);

    printf("test-pxa-cgraph-shed: %s (2 keys: old replays=%ld eager=%d, guard replays=%ld captures=%ld; 130-key tensor split x 20 tokens: "
           "old replays=%ld eager=%d, guard replays=%ld eager=%d, no shortfall replays=%ld)\n",
           n_fail ? "FAIL" : "OK", short_old.replays, short_old.eager_keys, short_new.replays, short_new.captures,
           ts_old.replays, ts_old.eager_keys, ts_new.replays, ts_new.eager_keys, ts_ok.replays);
    return n_fail ? 1 : 0;
}
