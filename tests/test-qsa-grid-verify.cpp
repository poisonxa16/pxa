// Regression test for PXA_QSA_GRID_VERIFY's comparison (2026-09-09).
//
// WHY IT EXISTS. The lever's entire purpose is to prove that the append-only pool grid equals a
// from-scratch rebuild, and until this test it could not do that: kpool_state_equal_or_die
// included n_new and is_new in the comparison, and both are PER-UBATCH DELTA fields -- the
// header defines n_new as "of those, the ones this ubatch must (re)pool" and is_new as "1 where
// THIS UBATCH (re)wrote a member of the pool". An incremental update that appended nothing
// reports 0; a from-scratch rebuild has no history and reports everything it just formed. On the
// real Flash-Next seat at 86,401 fill it aborted on exactly that -- "n_pool_real 256 vs 256,
// n_new 0 vs 128" -- with the grid itself in agreement. And because the old check folded its
// fields with && and guarded the per-sequence loop on the result, it returned before cells,
// pools or pos_min were ever looked at. So the instrument had never once compared the grid.
//
// The four cases below are the contract: delta-only differences are NOT differences; each real
// grid field IS one and is named; and identical states compare equal.
#include "llama-kv-cache-kpool.h"
#include <cstdio>
#include <cstring>

static int g_fail = 0;
static void ck(bool cond, const char * what) {
    printf("%s  %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_fail++;
}

static llama_kpool_state make_state() {
    llama_kpool_state st;
    st.seqs.resize(2);
    st.n_pool_real = 2;
    st.cache_safe  = true;
    st.n_new       = 0;
    for (int s = 0; s < 2; ++s) {
        auto & q = st.seqs[s];
        q.pos_min = 0;
        for (uint32_t i = 0; i < 8; ++i) {
            q.cells.emplace_back((llama_pos) i, i + 100u * (uint32_t) s);
        }
        q.pools  = {0, 4};
        q.is_new = {0, 0};
        q.scan_j = 8;
    }
    return st;
}

int main() {
    int bad = -1;

    // 1. identical states
    {
        auto a = make_state(); auto b = make_state();
        ck(llama_kpool_grid_diff(a, b, &bad) == nullptr, "identical states compare equal");
    }

    // 2. THE CASE THAT ABORTED ON REAL HARDWARE: only the per-ubatch delta fields differ.
    {
        auto a = make_state(); auto b = make_state();
        a.n_new = 0;  b.n_new = 128;                  // incremental knows; from-scratch does not
        a.seqs[0].is_new = {0, 0};
        b.seqs[0].is_new = {1, 1};
        const char * d = llama_kpool_grid_diff(a, b, &bad);
        ck(d == nullptr, "n_new / is_new differing is NOT a grid difference (the 86,401-fill abort)");
        if (d) printf("     reported '%s'\n", d);
    }

    // 3. each real grid field is caught AND named -- the old check could reach none of these
    {
        auto a = make_state(); auto b = make_state();
        b.seqs[1].cells[3].second += 1;
        const char * d = llama_kpool_grid_diff(a, b, &bad);
        ck(d && strcmp(d, "cells") == 0 && bad == 1, "a moved cell is caught and named ('cells', seq 1)");
    }
    {
        auto a = make_state(); auto b = make_state();
        b.seqs[0].pools = {0};
        const char * d = llama_kpool_grid_diff(a, b, &bad);
        ck(d && strcmp(d, "pools") == 0 && bad == 0, "a dropped pool is caught and named ('pools', seq 0)");
    }
    {
        auto a = make_state(); auto b = make_state();
        b.seqs[1].pos_min = 4;
        const char * d = llama_kpool_grid_diff(a, b, &bad);
        ck(d && strcmp(d, "pos_min") == 0 && bad == 1, "a moved pos_min is caught and named");
    }
    {
        auto a = make_state(); auto b = make_state();
        b.n_pool_real = 3;
        const char * d = llama_kpool_grid_diff(a, b, &bad);
        ck(d && strcmp(d, "n_pool_real") == 0, "n_pool_real is caught and named");
    }
    {
        auto a = make_state(); auto b = make_state();
        b.cache_safe = false;
        const char * d = llama_kpool_grid_diff(a, b, &bad);
        ck(d && strcmp(d, "cache_safe") == 0, "cache_safe is caught and named");
    }

    // 4. a grid difference BEHIND a delta difference is still found -- the old && ordering
    //    returned on n_new and never looked, which is the bug that hid the grid entirely.
    {
        auto a = make_state(); auto b = make_state();
        a.n_new = 0; b.n_new = 128;
        b.seqs[0].cells[0].second = 9999;
        const char * d = llama_kpool_grid_diff(a, b, &bad);
        ck(d && strcmp(d, "cells") == 0, "a grid difference behind a delta difference is still found");
    }

    printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
