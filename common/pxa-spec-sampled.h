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

// Bug #213 (PXA_SPEC_RELAXED): the probability the target's own draw gave candidate j -- the softmax
// of the post-chain window's LOGITS, which is what llama_sample_token_with_rng draws from. The
// candidates' .p field cannot be used for this: the chain leaves it as exp(l - max) (unnormalised,
// so p_max == 1 and every p is inflated by 1/p_max) or stale from an earlier sampler in the queue.
// Works over any array of {logit} records (llama_token_data or pxa_spec_cand).
template <typename T>
inline double pxa_spec_cand_mass(const T * data, size_t n, size_t j) {
    if (n == 0 || j >= n) return 0.0;
    float max_l = data[0].logit;
    for (size_t k = 1; k < n; ++k) max_l = std::max(max_l, data[k].logit);
    double sum = 0.0;
    for (size_t k = 0; k < n; ++k) sum += (double) expf(data[k].logit - max_l);
    if (!(sum > 0.0)) return 1.0 / (double) n;
    return (double) expf(data[j].logit - max_l) / sum;
}

// PXA_SPEC_BLOCK_VERIFY - joint (block) verification of a sampled draft.
//
// The per-token rule above decides each drafted token on its own: position i survives with
// probability min(1, p_i(x_i)/q_i(x_i)) and the first rejection ends the step. That is exact, but it
// throws away a case the joint view can keep: a draft whose early token is "over-proposed" (q > p)
// and whose later tokens are "under-proposed" (q < p) carries more total mass under p than under q
// along the whole prefix, and deciding the prefix jointly lets that later surplus pay for the early
// deficit. The rule below keeps a running prefix weight
//
//     w_0 = 1,   w_i = min(1, w_{i-1} * p_{i-1}(x_{i-1}) / q_{i-1}(x_{i-1}))
//
// and, walking the whole block once, picks the LONGEST prefix length t whose stop test passes,
// where the stop test at i < n is a coin with probability
//
//     h_i = A_i / (A_i + 1 - w_i),   A_i = sum_y max(w_i * p_i(y) - q_i(y), 0)
//
// and at i == n is w_n. The emitted continuation is x_0..x_{t-1} followed by one more token: a draw
// from p_n when t == n (the free token), else a draw from the weighted residual
// normalise(max(w_t * p_t - q_t, 0)). With n == 1 this is exactly the per-token rule. The emitted
// token stream is distributed exactly as the target's own sampler would emit it (for any q, as long
// as q is the distribution the draft was drawn from), and the expected number of kept tokens is never
// below the per-token rule's. tests/test-spec-sampled.cpp checks both against an exact enumeration
// and by Monte Carlo, with a deliberately broken variant that the same checks must reject.
//
// ps.size() == n + 1 (target distributions at every verify position, the last one for the free
// token), qs.size() == xs.size() == n, eta.size() == n. Returns the number of drafted tokens kept
// (t) and writes the one extra token to `extra`.
inline double pxa_spec_block_residual_mass(const std::vector<pxa_spec_cand> & p,
                                           const std::vector<pxa_spec_cand> & q, double w) {
    double a = 0.0;
    for (const auto & e : p) {
        const double d = w * (double) e.p - (double) pxa_spec_prob_of(q, e.id);
        if (d > 0.0) a += d;
    }
    return a;
}

// normalise(max(w * p - q, 0)); empty when the residual carries no mass
inline std::vector<pxa_spec_cand> pxa_spec_block_residual(const std::vector<pxa_spec_cand> & p,
                                                          const std::vector<pxa_spec_cand> & q, double w) {
    std::vector<pxa_spec_cand> r;
    r.reserve(p.size());
    double sum = 0.0;
    for (const auto & e : p) {
        const double d = w * (double) e.p - (double) pxa_spec_prob_of(q, e.id);
        if (d > 0.0) {
            r.push_back({e.id, e.logit, (float) d});
            sum += d;
        }
    }
    if (!(sum > 0.0)) {
        r.clear();
        return r;
    }
    for (auto & e : r) e.p = (float) ((double) e.p / sum);
    return r;
}

// The prefix weights w[i] and stop probabilities h[i] for i = 1..n (index 0 holds w = h = 1: the
// empty prefix is always available).
inline void pxa_spec_block_weights(const std::vector<std::vector<pxa_spec_cand>> & ps,
                                   const std::vector<std::vector<pxa_spec_cand>> & qs,
                                   const std::vector<int32_t> & xs,
                                   std::vector<double> & w, std::vector<double> & h) {
    const size_t n = xs.size();
    w.assign(n + 1, 1.0);
    h.assign(n + 1, 1.0);
    for (size_t i = 1; i <= n; ++i) {
        const double px = (double) pxa_spec_prob_of(ps[i-1], xs[i-1]);
        const double qx = (double) pxa_spec_prob_of(qs[i-1], xs[i-1]);
        w[i] = qx > 0.0 ? std::min(1.0, w[i-1] * px / qx) : 0.0;
        if (i < n) {
            const double a = pxa_spec_block_residual_mass(ps[i], qs[i], w[i]);
            const double b = a + 1.0 - w[i];
            h[i] = (a > 0.0 && b > 0.0) ? std::min(1.0, a / b) : 0.0;
        } else {
            h[i] = w[i];
        }
    }
}

inline size_t pxa_spec_block_verify(const std::vector<std::vector<pxa_spec_cand>> & ps,
                                    const std::vector<std::vector<pxa_spec_cand>> & qs,
                                    const std::vector<int32_t> & xs,
                                    const std::vector<float> & eta, float u_res, int32_t & extra) {
    const size_t n = xs.size();
    extra = -1;
    if (n == 0 || ps.size() != n + 1 || qs.size() != n || eta.size() != n) {
        return 0;
    }
    std::vector<double> w, h;
    pxa_spec_block_weights(ps, qs, xs, w, h);
    size_t t = 0;
    for (size_t i = 1; i <= n; ++i) {
        if (h[i] > 0.0 && (double) eta[i-1] <= h[i]) {
            t = i;
        }
    }
    if (t == n) {
        extra = pxa_spec_draw(ps[n], u_res);
    } else {
        const auto r = pxa_spec_block_residual(ps[t], qs[t], w[t]);
        // an empty residual has stop probability 0, so only float noise lands here
        extra = r.empty() ? pxa_spec_draw(ps[t], u_res) : pxa_spec_draw(r, u_res);
    }
    return t;
}
