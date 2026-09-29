// Proof that PXA_SPEC_BLOCK_VERIFY (common/pxa-spec-sampled.h) emits the TARGET's distribution.
//
// A per-position check is not enough for a rule that decides a whole draft block at once: what has
// to hold is that the emitted token STREAM, across as many draft/verify steps as it takes, has the
// joint distribution the target alone would give it. So the test builds a small synthetic language
// model pair over a 3-token vocabulary whose next-token distributions depend on the last two tokens
// (the target p and a deliberately mismatched drafter q), and runs the speculative process from an
// empty context until K tokens have been emitted:
//
//   1. EXACT: every draft block, every stop outcome and every residual draw is enumerated with its
//      probability, giving the exact joint law of the first K emitted tokens. It must equal the
//      target's own joint law to float precision (total variation < 1e-6) for the block rule AND
//      for the per-token rule, at draft depths 1..4.
//   2. PLANTED BUG: the same enumeration with a known-wrong stop probability (h_i = w_i, the naive
//      "product of ratios" rule) and with an unweighted residual must land far away (TV > 1e-3), so
//      the check above is not blind.
//   3. ACCEPTANCE: the expected number of drafted tokens kept per step under the block rule is never
//      below the per-token rule's, and is strictly above it on at least one depth > 1.
//   4. MONTE CARLO: the real pxa_spec_block_verify, driven by a real RNG, many runs; chi-square of
//      the first-K-token frequencies against the exact target law (|z| < 4), and the planted bug
//      driven the same way must fail the same test.
//
// Build and run standalone:
//   g++ -O2 -std=c++17 -I common -o /tmp/tbv tests/test-spec-block-verify.cpp && /tmp/tbv

#include "pxa-spec-sampled.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
static const int V = 3;

using dist = std::vector<pxa_spec_cand>;
using seq  = std::vector<int32_t>;

// deterministic context-dependent distributions over {0,1,2}; `salt` picks target vs drafter
static dist model(const seq & ctx, int salt) {
    const int a = ctx.size() >= 1 ? ctx[ctx.size() - 1] : 3;
    const int b = ctx.size() >= 2 ? ctx[ctx.size() - 2] : 3;
    double w[V];
    double sum = 0.0;
    for (int y = 0; y < V; ++y) {
        const unsigned h = (unsigned) (a * 131 + b * 17 + y * 7 + salt * 1009 + 3);
        w[y] = 0.05 + (double) ((h * 2654435761u) >> 24) / 256.0;   // (0.05, 1.05)
        if (salt == 1 && y == (a + 1) % V) w[y] *= 3.0;            // drafter over-proposes one token
        sum += w[y];
    }
    dist d;
    for (int y = 0; y < V; ++y) {
        const float p = (float) (w[y] / sum);
        d.push_back({y, logf(p), p});
    }
    pxa_spec_sort(d);
    return d;
}
static dist P(const seq & c) { return model(c, 0); }
static dist Q(const seq & c) { return model(c, 1); }

static std::string key(const seq & s, size_t K) {
    std::string k;
    for (size_t i = 0; i < K && i < s.size(); ++i) k += (char) ('0' + s[i]);
    return k;
}

// exact target law of the first K tokens
static void target_law(const seq & ctx, double pr, size_t K, std::map<std::string, double> & out) {
    if (ctx.size() >= K) { out[key(ctx, K)] += pr; return; }
    for (const auto & e : P(ctx)) target_law([&] { seq c = ctx; c.push_back(e.id); return c; }(), pr * e.p, K, out);
}

enum rule { RULE_TOKEN, RULE_BLOCK, RULE_BUG_H, RULE_BUG_RES };

// the law of (tokens kept, extra token) for ONE verify step from ctx with draft xs
static void step_law(rule r, const seq & ctx, const seq & xs,
                     const std::function<void(size_t, int32_t, double)> & emit) {
    const size_t n = xs.size();
    std::vector<dist> ps(n + 1), qs(n);
    seq c = ctx;
    for (size_t i = 0; i <= n; ++i) {
        ps[i] = P(c);
        if (i < n) { qs[i] = Q(c); c.push_back(xs[i]); }
    }
    if (r == RULE_TOKEN) {
        double surv = 1.0;
        for (size_t i = 0; i < n; ++i) {
            const double px = pxa_spec_prob_of(ps[i], xs[i]), qx = pxa_spec_prob_of(qs[i], xs[i]);
            const double acc = qx > 0.0 ? std::min(1.0, px / qx) : 0.0;
            const auto res = pxa_spec_block_residual(ps[i], qs[i], 1.0);
            for (const auto & e : res) emit(i, e.id, surv * (1.0 - acc) * e.p);
            surv *= acc;
        }
        for (const auto & e : ps[n]) emit(n, e.id, surv * e.p);
        return;
    }
    std::vector<double> w, h;
    pxa_spec_block_weights(ps, qs, xs, w, h);
    if (r == RULE_BUG_H) {
        for (size_t i = 1; i <= n; ++i) h[i] = w[i];
    }
    for (size_t t = 0; t <= n; ++t) {
        // P(stop index == t) = h_t * prod_{j>t} (1 - h_j), h_0 = 1
        double pt = t == 0 ? 1.0 : h[t];
        for (size_t j = t + 1; j <= n; ++j) pt *= (1.0 - h[j]);
        if (pt <= 0.0) continue;
        if (t == n) {
            for (const auto & e : ps[n]) emit(t, e.id, pt * e.p);
        } else {
            const auto res = pxa_spec_block_residual(ps[t], qs[t], r == RULE_BUG_RES ? 1.0 : w[t]);
            for (const auto & e : res) emit(t, e.id, pt * e.p);
        }
    }
}

// exact law of the first K emitted tokens under rule r at draft depth n; also accumulates the
// expected kept tokens of the FIRST step (from the empty context)
static void spec_law(rule r, size_t n, const seq & ctx, double pr, size_t K,
                     std::map<std::string, double> & out, double * kept0) {
    if (ctx.size() >= K) { out[key(ctx, K)] += pr; return; }
    // enumerate every draft block drawn autoregressively from q
    std::function<void(seq, double)> drafts = [&](seq xs, double pq) {
        if (xs.size() == n) {
            step_law(r, ctx, xs, [&](size_t t, int32_t y, double p) {
                if (p <= 0.0) return;
                if (kept0 && ctx.empty()) *kept0 += pq * p * (double) t;
                seq c = ctx;
                c.insert(c.end(), xs.begin(), xs.begin() + t);
                c.push_back(y);
                spec_law(r, n, c, pr * pq * p, K, out, nullptr);
            });
            return;
        }
        seq c = ctx;
        c.insert(c.end(), xs.begin(), xs.end());
        for (const auto & e : Q(c)) {
            seq x2 = xs;
            x2.push_back(e.id);
            drafts(x2, pq * e.p);
        }
    };
    drafts({}, 1.0);
}

static double tv(const std::map<std::string, double> & a, const std::map<std::string, double> & b) {
    std::map<std::string, double> all;
    for (const auto & kv : a) all[kv.first] += 0.0;
    for (const auto & kv : b) all[kv.first] += 0.0;
    double s = 0.0;
    for (const auto & kv : all) {
        const auto ia = a.find(kv.first), ib = b.find(kv.first);
        s += std::fabs((ia == a.end() ? 0.0 : ia->second) - (ib == b.end() ? 0.0 : ib->second));
    }
    return 0.5 * s;
}

static double chi2_z(double chi2, int df) {
    const double a = 2.0 / (9.0 * (double) df);
    return (std::cbrt(chi2 / (double) df) - (1.0 - a)) / std::sqrt(a);
}

// Monte Carlo through the real pxa_spec_block_verify
static double mc_z(size_t n, size_t K, long N, uint32_t seed, bool bug, const std::map<std::string, double> & law,
                   double * kept_mean) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::map<std::string, long> obs;
    double kept = 0.0;
    long steps = 0;
    for (long run = 0; run < N; ++run) {
        seq ctx;
        while (ctx.size() < K) {
            std::vector<dist> ps(n + 1), qs(n);
            seq xs;
            seq c = ctx;
            for (size_t i = 0; i < n; ++i) {
                qs[i] = Q(c);
                const int32_t x = pxa_spec_draw(qs[i], u01(rng));
                xs.push_back(x);
                c.push_back(x);
            }
            c = ctx;
            for (size_t i = 0; i <= n; ++i) { ps[i] = P(c); if (i < n) c.push_back(xs[i]); }
            std::vector<float> eta(n);
            for (auto & e : eta) e = u01(rng);
            int32_t extra = -1;
            size_t t;
            if (!bug) {
                t = pxa_spec_block_verify(ps, qs, xs, eta, u01(rng), extra);
            } else {
                // planted bug: naive stop rule h_i = w_i, same draws otherwise
                std::vector<double> w, h;
                pxa_spec_block_weights(ps, qs, xs, w, h);
                t = 0;
                for (size_t i = 1; i <= n; ++i) if (w[i] > 0.0 && (double) eta[i-1] <= w[i]) t = i;
                const float ur = u01(rng);
                if (t == n) extra = pxa_spec_draw(ps[n], ur);
                else { auto r = pxa_spec_block_residual(ps[t], qs[t], w[t]); extra = pxa_spec_draw(r.empty() ? ps[t] : r, ur); }
            }
            if (ctx.empty()) { kept += (double) t; steps++; }
            ctx.insert(ctx.end(), xs.begin(), xs.begin() + t);
            ctx.push_back(extra);
        }
        obs[key(ctx, K)]++;
    }
    if (kept_mean) *kept_mean = steps ? kept / (double) steps : 0.0;
    double chi2 = 0.0;
    int df = -1;
    for (const auto & kv : law) {
        if (kv.second <= 0.0) continue;
        const double e = (double) N * kv.second;
        const auto it = obs.find(kv.first);
        const double o = it == obs.end() ? 0.0 : (double) it->second;
        chi2 += (o - e) * (o - e) / e;
        df++;
    }
    for (const auto & kv : obs) if (law.find(kv.first) == law.end()) chi2 += 1e9;   // impossible sequence
    return chi2_z(chi2, df);
}

int main() {
    const size_t K = 4;
    std::map<std::string, double> target;
    target_law({}, 1.0, K, target);

    printf("PXA_SPEC_BLOCK_VERIFY - exact enumeration, first %zu tokens, vocab %d (%zu sequences)\n", K, V, target.size());
    printf("%-6s %-14s %-14s %-14s %-14s %-10s %-10s\n", "depth", "TV token", "TV block", "TV bug-h", "TV bug-res",
           "kept tok", "kept blk");
    bool any_gain = false;
    for (size_t n = 1; n <= 4; ++n) {
        std::map<std::string, double> lt, lb, lh, lr;
        double kt = 0.0, kb = 0.0;
        spec_law(RULE_TOKEN,   n, {}, 1.0, K, lt, &kt);
        spec_law(RULE_BLOCK,   n, {}, 1.0, K, lb, &kb);
        spec_law(RULE_BUG_H,   n, {}, 1.0, K, lh, nullptr);
        spec_law(RULE_BUG_RES, n, {}, 1.0, K, lr, nullptr);
        const double t_t = tv(lt, target), t_b = tv(lb, target), t_h = tv(lh, target), t_r = tv(lr, target);
        // at depth 1 both planted bugs coincide with the correct rule, so they only have to be caught at n > 1
        const bool ok = t_t < 1e-6 && t_b < 1e-6 && (n == 1 || (t_h > 1e-3 && t_r > 1e-3)) && kb >= kt - 1e-9;
        if (n > 1 && kb > kt + 1e-6) any_gain = true;
        if (!ok) g_fail++;
        printf("%-6zu %-14.3e %-14.3e %-14.3e %-14.3e %-10.5f %-10.5f %s\n", n, t_t, t_b, t_h, t_r, kt, kb, ok ? "PASS" : "FAIL");
    }
    if (!any_gain) {
        g_fail++;
        printf("FAIL: the block rule never kept more drafted tokens than the per-token rule\n");
    }

    printf("\nMonte Carlo through pxa_spec_block_verify (chi-square of the first %zu tokens vs the exact law)\n", K);
    const long N = 200000;
    for (size_t n = 2; n <= 4; n += 2) {
        double km = 0.0, kmb = 0.0;
        const double z  = mc_z(n, K, N, 1234 + (uint32_t) n, false, target, &km);
        const double zb = mc_z(n, K, N, 4321 + (uint32_t) n, true,  target, &kmb);
        const bool ok = std::fabs(z) < 4.0 && zb > 8.0;
        if (!ok) g_fail++;
        printf("depth %zu  n=%ld  z(block)=%+7.2f  z(planted bug)=%+8.2f  kept/step %.4f  %s\n", n, N, z, zb, km,
               ok ? "PASS" : "FAIL");
    }

    printf("\n%s (%d failure(s))\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
