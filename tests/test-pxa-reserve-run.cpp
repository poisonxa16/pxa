// PXA_RESERVE_GUARD run accounting (src/llama-pxa-reserve-run.h, issue #11). Pure, no model, no GPU.
#include "llama-pxa-reserve-run.h"

#include <cstdio>
#include <vector>

static int g_fail = 0;
#define CHECK_EQ(a, b) do { const long long _a = (long long) (a), _b = (long long) (b); if (_a != _b) { \
    fprintf(stderr, "FAIL %s:%d: %s = %lld, want %lld\n", __FILE__, __LINE__, #a, _a, _b); ++g_fail; } } while (0)

// Drive the guard the way llama_decode does: every plan in `served` serves that many
// ubatches, then the next plan is reserved. The reserve's own ubatch runs on the new plan, so it is
// the first ubatch that plan serves. Returns the 1-based reserve at which the guard latched, 0 = never;
// *max_run_out = the longest run seen.
static int latch_at(const std::vector<int> & served, int max_reserves, bool lifetime_mode, int * max_run_out = nullptr) {
    pxa_reserve_run_state st;
    int count = 0;
    for (int ub : served) {
        count++;
        const int run = pxa_reserve_run_note_reserve(st);
        if (max_run_out && run > *max_run_out) *max_run_out = run;
        if (pxa_reserve_run_should_latch(run, count, max_reserves, lifetime_mode)) {
            return count;
        }
        for (int i = 0; i < ub; ++i) {
            pxa_reserve_run_note_ubatch(st);
        }
    }
    return 0;
}

static std::vector<int> repeat(std::vector<int> pattern, int times) {
    std::vector<int> out;
    for (int t = 0; t < times; ++t) out.insert(out.end(), pattern.begin(), pattern.end());
    return out;
}

int main() {
    // the issue #11 server: per request one prefill plan (5 ubatches of a >2048-token prompt at
    // -ub 512) and one decode plan (4 tokens); 1000 requests = 2000 re-reserves, never a latch
    {
        int max_run = 0;
        CHECK_EQ(latch_at(repeat({5, 4}, 1000), 24, false, &max_run), 0);
        CHECK_EQ(max_run, 1);
    }
    // ... and the pre-fix lifetime rule (PXA_RESERVE_LIFETIME=1) latches it at reserve 25, request 13
    CHECK_EQ(latch_at(repeat({5, 4}, 1000), 24, true), 25);
    // short generations: the decode plan serves 1..3, but the prefill plan always amortizes, so
    // the run only ever alternates 1, 2
    for (int dec = 1; dec <= 3; ++dec) {
        int run = 0;
        CHECK_EQ(latch_at(repeat({5, dec}, 1000), 24, false, &run), 0);
        CHECK_EQ(run, 2);
    }
    // speculative decoding, short prompts, short answers (P56 measurement 2026-10-17): per request a
    // prefill plan of 2 ubatches and a decode plan; counting only main-plan ubatches the decode plan
    // serves 1-2 and the run climbs by one per reserve (latch at 25), counting the graph-slot
    // ubatches it serves 8+ (draft + verify steps of even a 10-token answer) and the run stays at 2
    {
        int run = 0;
        CHECK_EQ(latch_at(repeat({2, 1}, 1000), 24, false), 25);
        CHECK_EQ(latch_at(repeat({2, 8}, 1000), 24, false, &run), 0);
        CHECK_EQ(run, 2);
    }
    // rapid switching, 1..3 ubatches per plan: still latches, at reserve PXA_RESERVE_MAX + 1
    for (int ub = 1; ub < PXA_RESERVE_AMORTIZED_MIN; ++ub) {
        CHECK_EQ(latch_at(repeat({ub}, 1000), 24, false), 25);
        CHECK_EQ(latch_at(repeat({ub}, 1000), 8, false), 9);
    }
    // the speculative-decode case the guard exists for: two shapes taking turns, one ubatch each
    CHECK_EQ(latch_at(repeat({1, 1}, 1000), 24, false), 25);
    // 4+ ubatches per plan never latch, whatever the cap
    for (int ub = PXA_RESERVE_AMORTIZED_MIN; ub <= 8; ++ub) {
        CHECK_EQ(latch_at(repeat({ub}, 5000), 24, false), 0);
        CHECK_EQ(latch_at(repeat({ub}, 5000), 1, false), 0);
    }
    // the reserve right after an amortized plan restarts the run at 1, even deep into a run
    {
        pxa_reserve_run_state st;
        for (int i = 0; i < 20; ++i) {
            CHECK_EQ(pxa_reserve_run_note_reserve(st), i + 1);
            pxa_reserve_run_note_ubatch(st);   // 1 ubatch: unamortized
        }
        for (int i = 1; i < PXA_RESERVE_AMORTIZED_MIN; ++i) pxa_reserve_run_note_ubatch(st);
        CHECK_EQ(pxa_reserve_run_note_reserve(st), 1);
        CHECK_EQ(st.served, 0);
    }
    // a rapid burst shorter than the cap, then an amortized plan, then another burst: no latch
    {
        std::vector<int> s;
        for (int k = 0; k < 50; ++k) {
            for (int i = 0; i < 23; ++i) s.push_back(1);
            s.push_back(4);
        }
        CHECK_EQ(latch_at(s, 24, false), 0);
        // the same bursts one longer reach the cap
        s.clear();
        for (int i = 0; i < 25; ++i) s.push_back(1);
        CHECK_EQ(latch_at(s, 24, false), 25);
    }
    // the first reserve of a context has no measured predecessor work: run 1
    {
        pxa_reserve_run_state st;
        CHECK_EQ(pxa_reserve_run_note_reserve(st), 1);
    }
    // PXA_RESERVE_MAX=0 latches on the first re-reserve in both modes (unchanged meaning of 0)
    CHECK_EQ(latch_at(repeat({5, 4}, 10), 0, false), 1);
    CHECK_EQ(latch_at(repeat({5, 4}, 10), 0, true), 1);
    // the served counter saturates instead of wrapping
    {
        pxa_reserve_run_state st;
        st.served = INT_MAX - 1;
        pxa_reserve_run_note_ubatch(st);
        pxa_reserve_run_note_ubatch(st);
        CHECK_EQ(st.served, INT_MAX);
        CHECK_EQ(pxa_reserve_run_note_reserve(st), 1);
    }
    if (g_fail) {
        fprintf(stderr, "test-pxa-reserve-run: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-pxa-reserve-run: ok\n");
    return 0;
}
