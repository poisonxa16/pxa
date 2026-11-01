#pragma once
// PXA_MTP_STREAK=S[:F[:D]] (od-b8-gated-chain-stop, Overdrive 2.0 row B8, od2-lead 2026-10-29; default off, =0 off):
// the PXA_MTP_DEEP chain may run past its cap D (default 5) up to S (default 7) only on a cycle that FOLLOWS A FULL ONE,
// and stops past D as soon as the chain's cumulative draft probability falls under F (default 0.25).
//
// Why gated: the plain ceiling 7 (deep chain to n_max on every confident cycle) measured undetectable (
// +0.6 / -0.1 at n=4): the two extra verify columns are paid on every deep cycle, the extra tokens land only on the
// few where the whole chain was right. Full cycles come in runs (the "Updating N tokens" log lines):
// P(full | previous full) 0.58 code / 0.83 prose / 0.62 chat against 0.04-0.10 after a short cycle, so a streak is the
// cheap predictor of where the extension pays. Break-even per extension position: (cumulative probability the chain is
// still all-correct) x (conditional accept) > column price / token value = 3.5 / 14.0 ms = 0.25 on the code class -> F.
//
// "Full" = the verifier kept every drafted token of a chain that reached at least D. The cumulative probability is the
// product of the drafter's reported top-1 probabilities over the chain (under PXA_DRAFT_ARGMAX a device-decided pass
// reports the LOWER bound of its interval, so the product only ever stops earlier than the host's would; deterministic
// either way). The first D positions are never touched: below the cap the chain is exactly PXA_MTP_DEEP's.
//
// Greedy output is unchanged: only how many draft tokens the verifier sees changes, and the qwen4exp verify columns are
// bitwise one-token decode up to width 8. The configured n_max stays the ceiling: the Flash-Next V100 rule
// (examples/server/pxa-fn-v100-mtp.h) raises it to S when this lever is on, so the verify batch, the recurrent
// checkpoint rows and the expert-cache chain reserve are sized for width S + 1.
#include <cstdio>
#include <cstdlib>

struct pxa_mtp_streak_cfg {
    bool  on    = false;
    int   cap   = 7;       // S: the chain's ceiling on a cycle that follows a full one
    float floor = 0.25f;   // F: cumulative-probability floor for positions past D
    int   deep  = 5;       // D: today's deep cap (the chain below it is untouched)
};

// "7", "7:0.25", "7:0.25:5"; unset, empty, "0" or anything that does not parse -> off. S <= D is off (nothing to extend).
inline pxa_mtp_streak_cfg pxa_mtp_streak_parse(const char * e) {
    pxa_mtp_streak_cfg c;
    if (e == nullptr || *e == '\0' || std::atof(e) <= 0.0f) return c;
    float s = 7.0f, f = 0.25f, d = 5.0f;
    if (std::sscanf(e, "%f:%f:%f", &s, &f, &d) < 1) return c;
    if ((int) d < 1 || (int) s <= (int) d || f < 0.0f || f > 1.0f) return c;
    c.on = true; c.cap = (int) s; c.floor = f; c.deep = (int) d;
    return c;
}

inline const pxa_mtp_streak_cfg & pxa_mtp_streak() {
    static const pxa_mtp_streak_cfg c = [] {
        const pxa_mtp_streak_cfg r = pxa_mtp_streak_parse(std::getenv("PXA_MTP_STREAK"));
        if (r.on) {
            fprintf(stderr, "PXA_MTP_STREAK: on (past depth %d up to %d only after a full cycle, while the chain's cumulative "
                            "probability >= %.2f)\n", r.deep, r.cap, (double) r.floor);
        }
        return r;
    }();
    return c;
}

// the chain depth for this cycle: n_draft as PXA_MTP_DEEP left it (the configured ceiling when the EMA is high),
// held at D unless the previous cycle was full
inline int pxa_mtp_streak_depth(const pxa_mtp_streak_cfg & c, int n_draft, bool prev_full) {
    if (!c.on || n_draft <= c.deep) return n_draft;
    return prev_full ? (n_draft < c.cap ? n_draft : c.cap) : c.deep;
}

// after the verify: was this cycle full (a chain of at least D drafts, every one kept)?
inline bool pxa_mtp_streak_full(const pxa_mtp_streak_cfg & c, int n_drafted, int n_accepted) {
    return c.on && n_drafted >= c.deep && n_accepted >= n_drafted;
}

// inside the draft loop, with draft index i (0-based) just drawn and cum = the product of the chain's probabilities
// including row i: true = stop before keeping row i (only ever past the first D rows)
inline bool pxa_mtp_streak_stop(const pxa_mtp_streak_cfg & c, int i, float cum) {
    return c.on && i >= c.deep && cum < c.floor;
}
