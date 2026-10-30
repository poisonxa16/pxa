#pragma once
// PXA_CGRAPH_OOM_GUARD headroom shed: which of a context's execs count as STALE (safe to destroy before an
// instantiate when the card has less than PXA_CGRAPH_HEADROOM_MB free). Kept in a header so the CPU test
// tests/test-pxa-cgraph-shed.cpp checks the same rule the CUDA backend runs.
//
// 2026-10-20: the rule used to be "never replayed" (n_replays == 0) alone. A launch right
// after a capture is not counted as a replay, so under a persistent shortfall two keys that alternate on one
// context (decode / verify widths, the splits of one eval) destroyed each other's exec on every instantiate:
// each key recaptured on every eval, never replayed, and after four consecutive updates the key went eager
// (cooldown or permanent disable). A key captured more than once has come back, so it is not stale; a
// one-off key (captured once, never replayed) still is. The fail-soft path (an instantiate that really runs
// out of memory) still sheds by LRU whatever this says. PXA_CGRAPH_SHED_RECUR=0 restores the old rule.
static inline bool pxa_cgraph_headroom_stale(long n_replays, long n_captures, bool recur_guard) {
    if (n_replays > 0) {
        return false;
    }
    return !recur_guard || n_captures <= 1;
}
