#pragma once
// PXA_TSPLIT_EPI_PLAN_CACHE stash policy (pass 8e review R8), header-only and host-only so a CPU unit test
// (tests/test-pxa-epi-stash.cpp) can drive it without a GPU. A slot type S needs: bool valid; const void * owner;
// uint64_t used. The cache in ggml-cuda/reduce.cu keeps one active plan plus N parked ones; a scheduler (owner) has
// one current split generation, so at most ONE of its plans can still match and every older one is stale for good.
#include <cstddef>
#include <cstdint>

// The slot that receives the plan being replaced (owned by `owner`): that owner's stale entry first, then an empty
// slot, then the least recently used one.
template <class S, size_t N>
S * pxa_epi_stash_pick(S (&a)[N], const void * owner) {
    for (auto & st : a) if (st.valid && st.owner == owner) return &st;
    for (auto & st : a) if (!st.valid) return &st;
    S * s = &a[0];
    for (auto & st : a) if (st.used < s->used) s = &st;
    return s;
}

// Drop every valid entry of `owner` except `keep` (nullptr: drop them all); `clear` releases the slot's containers.
template <class S, size_t N, class F>
int pxa_epi_stash_drop_owner(S (&a)[N], const void * owner, const S * keep, F clear) {
    int n = 0;
    for (auto & st : a) {
        if (&st == keep || !st.valid || st.owner != owner) continue;
        st.valid = false; st.owner = nullptr; clear(st); ++n;
    }
    return n;
}
