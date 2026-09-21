#pragma once

// PXA_SPEC_SAMPLED - lossless speculative sampling at temp > 0.
//
// The exact-match acceptance rule keeps a drafted token only when the target's own draw happens to
// equal it, so per-position acceptance is p_target(draft) and a deep chain at temp 1 loses to plain
// decoding. The relaxed rule raises acceptance by keeping a draft token the target did not draw,
// which changes the emitted distribution. The rule below does neither: the drafter DRAWS its token
// from its own filtered distribution q, the verifier accepts it with probability min(1, p(x)/q(x))
// and on rejection emits a draw from the residual normalise(max(p - q, 0)).
//
//   P(emit y) = q(y) * min(1, p(y)/q(y)) + P(reject) * max(p(y)-q(y),0) / S
//             = min(p(y), q(y)) + max(p(y)-q(y), 0)            [ S = sum max(p-q,0) = P(reject) ]
//             = p(y)
//
// so the emitted token is distributed exactly as the target's own sampler would have emitted it,
// for ANY proposal q, and the expected acceptance is 1 - TV(p, q). q only has to be a proper
// distribution over token ids and to be the one actually drawn from - it does not have to be the
// same shape as p, which is why a truncated (top-k) q is legitimate here.
//
// Everything in this header is pure host arithmetic over candidate lists so that the rule can be
// proven on the CPU without a model: tests/test-spec-sampled.cpp drives it with synthetic p and q.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct pxa_spec_cand {
    int32_t id;
    float   logit;
    float   p;
};

// the samplers this rule knows how to apply identically to the draft distribution; anything else in
// a request's chain makes the rule refuse and fall back to exact matching.
enum pxa_spec_filter {
    PXA_SPEC_FILTER_TOP_P = 0,
    PXA_SPEC_FILTER_MIN_P,
    PXA_SPEC_FILTER_TEMP,
};

// descending by logit, ties broken by id so two runs order an exact tie the same way
inline void pxa_spec_sort(std::vector<pxa_spec_cand> & c) {
    std::sort(c.begin(), c.end(), [](const pxa_spec_cand & a, const pxa_spec_cand & b) {
        if (a.logit != b.logit) return a.logit > b.logit;
        return a.id < b.id;
    });
}

// softmax over the current logits into .p (normalised). Mirrors llama_sample_softmax_impl.
inline void pxa_spec_softmax(std::vector<pxa_spec_cand> & c) {
    if (c.empty()) return;
    const float max_l = c[0].logit;
    double sum = 0.0;
    for (auto & e : c) {
        e.p = expf(e.logit - max_l);
        sum += (double) e.p;
    }
    if (sum <= 0.0) {
        for (auto & e : c) e.p = 1.0f / (float) c.size();
        return;
    }
    for (auto & e : c) e.p = (float) ((double) e.p / sum);
}

// llama_sample_top_p_impl over a sorted window
inline void pxa_spec_apply_top_p(std::vector<pxa_spec_cand> & c, float top_p, size_t min_keep) {
    if (top_p >= 1.0f || c.empty()) return;
    pxa_spec_softmax(c);
    float  cum_sum  = 0.0f;
    size_t last_idx = c.size();
    for (size_t i = 0; i < c.size(); ++i) {
        cum_sum += c[i].p;
        if (cum_sum >= top_p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }
    c.resize(last_idx);
}

// llama_sample_min_p_impl, sorted branch
inline void pxa_spec_apply_min_p(std::vector<pxa_spec_cand> & c, float min_p, size_t min_keep) {
    if (min_p <= 0.0f || c.empty()) return;
    const float min_logit = c[0].logit + logf(min_p);
    size_t i = 1;
    for (; i < c.size(); ++i) {
        if (c[i].logit < min_logit && i >= min_keep) break;
    }
    c.resize(i);
}

inline void pxa_spec_apply_temp(std::vector<pxa_spec_cand> & c, float temp) {
    if (temp <= 0.0f) return;
    for (auto & e : c) e.logit /= temp;
}

// Apply {top_p, min_p, temp} in the request's own order to an already top-k-truncated, sorted
// window, then leave .p as the final normalised distribution the token is drawn from. This is the
// order llama.cpp's chain runs in - note that top_p and min_p cut on PRE-temperature logits and
// the temperature divide happens wherever the request puts it, which is why the order is passed in
// rather than assumed.
inline void pxa_spec_apply_chain(std::vector<pxa_spec_cand> & c,
                                 const std::vector<pxa_spec_filter> & order,
                                 float top_p, float min_p, float temp, size_t min_keep) {
    if (c.empty()) return;
    pxa_spec_sort(c);
    for (const auto f : order) {
        switch (f) {
            case PXA_SPEC_FILTER_TOP_P: pxa_spec_apply_top_p(c, top_p, min_keep); break;
            case PXA_SPEC_FILTER_MIN_P: pxa_spec_apply_min_p(c, min_p, min_keep); break;
            case PXA_SPEC_FILTER_TEMP:  pxa_spec_apply_temp (c, temp);            break;
        }
    }
    pxa_spec_softmax(c);
}

inline float pxa_spec_prob_of(const std::vector<pxa_spec_cand> & c, int32_t id) {
    for (const auto & e : c) {
        if (e.id == id) return e.p;
    }
    return 0.0f;
}

// draw from .p with u in [0,1). The list is normalised, so the final entry absorbs any rounding.
inline int32_t pxa_spec_draw(const std::vector<pxa_spec_cand> & c, float u) {
    if (c.empty()) return -1;
    double cum = 0.0;
    for (size_t i = 0; i + 1 < c.size(); ++i) {
        cum += (double) c[i].p;
        if ((double) u < cum) return c[i].id;
    }
    return c.back().id;
}

// The rule itself. `x` was drawn from q. Returns the token to emit and sets `accepted`.
// u_acc and u_res are independent uniforms in [0,1) supplied by the caller's RNG.
//
// A token with q(x) == 0 cannot have been drawn from q; treat it as a rejection rather than
// dividing by zero, so a caller that loses the q list degrades to "reject and resample from p"
// (still the target's distribution) instead of to an undefined branch.
inline int32_t pxa_spec_sampled_step(const std::vector<pxa_spec_cand> & p,
                                     const std::vector<pxa_spec_cand> & q,
                                     int32_t x, float u_acc, float u_res, bool & accepted) {
    accepted = false;
    if (p.empty()) return -1;

    const float px = pxa_spec_prob_of(p, x);
    const float qx = pxa_spec_prob_of(q, x);

    if (qx > 0.0f && (double) u_acc * (double) qx <= (double) px) {
        accepted = true;
        return x;
    }

    // residual normalise(max(p - q, 0)) - support is p's support, since max(0 - q, 0) = 0
    std::vector<pxa_spec_cand> r;
    r.reserve(p.size());
    double sum = 0.0;
    for (const auto & e : p) {
        const float d = e.p - pxa_spec_prob_of(q, e.id);
        if (d > 0.0f) {
            r.push_back({e.id, e.logit, d});
            sum += (double) d;
        }
    }
    if (r.empty() || sum <= 0.0) {
        // p and q agree to the last bit on p's support; a rejection here is float noise, and the
        // residual is then p itself.
        return pxa_spec_draw(p, u_res);
    }
    for (auto & e : r) e.p = (float) ((double) e.p / sum);
    return pxa_spec_draw(r, u_res);
}
