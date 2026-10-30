// PXA_TSPLIT_EPI_PLAN_CACHE stash policy (pass 8e review R8): one entry per scheduler; stale duplicates never crowd out
// the still-valid plans of other schedulers. Drives the header the cache in ggml-cuda/reduce.cu uses, on the CPU.
#include "../ggml/src/ggml-cuda/pxa/pxa-epi-stash.h"
#include <cstdio>
#include <vector>

struct slot_t { bool valid = false; const void * owner = nullptr; uint64_t gen = 0; uint64_t used = 0; };
static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// a model of the cache: one active plan + 8 parked, the same transitions as ggml_cuda_pxa_epi_plan
struct cache_t {
    slot_t active; slot_t stash[8]; uint64_t tick = 0; int hits = 0, builds = 0;
    static void clear(slot_t & s) { s.gen = 0; }
    void plan(const void * owner, uint64_t gen) {
        if (active.valid && active.owner == owner && active.gen == gen) { ++hits; return; }
        for (auto & st : stash) {
            if (!(st.valid && st.owner == owner && st.gen == gen)) continue;
            std::swap(active, st); st.used = ++tick;
            if (st.valid) pxa_epi_stash_drop_owner(stash, st.owner, &st, clear);
            pxa_epi_stash_drop_owner(stash, owner, (const slot_t *) nullptr, clear);
            ++hits; return;
        }
        if (active.valid) {
            slot_t * s = pxa_epi_stash_pick(stash, active.owner);
            std::swap(active, *s); s->used = ++tick;
            pxa_epi_stash_drop_owner(stash, s->owner, s, clear);
        }
        pxa_epi_stash_drop_owner(stash, owner, (const slot_t *) nullptr, clear);
        active = slot_t{ true, owner, gen, 0 }; ++builds;
    }
    int count(const void * owner) const { int n = 0; for (auto & s : stash) n += s.valid && s.owner == owner; return n; }
};

int main() {
    int o[10];   // scheduler addresses: o[0] main, o[1] draft, o[2..6] verify widths 2..6
    // MTP round: main, draft, verify(w), main ... with the main scheduler re-splitting now and then
    {
        cache_t c; uint64_t gmain = 1;
        for (int round = 0; round < 200; ++round) {
            if (round % 37 == 0) ++gmain;   // main re-split: new generation
            c.plan(&o[0], gmain); c.plan(&o[1], 100); c.plan(&o[2 + round % 5], 200 + round % 5);
            for (int k = 0; k < 7; ++k) CHECK(c.count(&o[k]) <= 1);
        }
        // 7 schedulers fit (1 active + 8 parked): after warm-up every plan hits except the main re-splits
        CHECK(c.builds <= 7 + 200 / 37 + 1);
        printf("mtp model: builds %d hits %d\n", c.builds, c.hits);
    }
    // a stale same-owner entry is replaced in place, never duplicated
    {
        cache_t c;
        c.plan(&o[0], 1); c.plan(&o[1], 1); c.plan(&o[0], 2); c.plan(&o[1], 1); c.plan(&o[0], 3); c.plan(&o[1], 1);
        CHECK(c.count(&o[0]) == 1 && c.count(&o[1]) == 0 && c.active.owner == &o[1]);
        for (auto & s : c.stash) if (s.valid && s.owner == &o[0]) CHECK(s.gen == 3);
    }
    // pick: same owner first, then empty, then oldest
    {
        slot_t a[3];
        a[0] = { true, &o[1], 1, 5 }; a[1] = { true, &o[2], 1, 2 }; a[2] = { false, nullptr, 0, 9 };
        CHECK(pxa_epi_stash_pick(a, &o[2]) == &a[1]);
        CHECK(pxa_epi_stash_pick(a, &o[3]) == &a[2]);
        a[2] = { true, &o[4], 1, 9 };
        CHECK(pxa_epi_stash_pick(a, &o[3]) == &a[1]);
    }
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
