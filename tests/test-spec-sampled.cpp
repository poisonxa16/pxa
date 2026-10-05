// Proof that the PXA_SPEC_SAMPLED acceptance rule emits the TARGET's distribution.
//
// The rule is pure host arithmetic over candidate lists (common/pxa-spec-sampled.h), so it can be
// driven here with synthetic p and q and no model at all. For each shape we run the full
// draw-x-from-q / accept-with-min(1,p/q) / resample-the-residual step a few hundred thousand times
// and test the emitted-token frequencies against p with a chi-square goodness-of-fit test, plus the
// hard invariants: a token outside p's support is never emitted, and q == p never rejects.
//
// Build and run standalone:
//   g++ -O2 -std=c++17 -I common -o /tmp/test-spec-sampled tests/test-spec-sampled.cpp && /tmp/test-spec-sampled

#include "pxa-spec-sampled.h"

#include <cstdio>
#include <map>
#include <random>
#include <string>

static int g_fail = 0;

static std::vector<pxa_spec_cand> make_dist(const std::vector<std::pair<int32_t, float>> & w) {
    std::vector<pxa_spec_cand> c;
    double sum = 0.0;
    for (const auto & e : w) sum += (double) e.second;
    for (const auto & e : w) {
        const float p = (float) ((double) e.second / sum);
        c.push_back({e.first, logf(p), p});
    }
    pxa_spec_sort(c);
    return c;
}

// Wilson-Hilferty: a chi-square with df degrees of freedom, expressed as a standard normal deviate.
static double chi2_z(double chi2, int df) {
    const double a = 2.0 / (9.0 * (double) df);
    return (std::cbrt(chi2 / (double) df) - (1.0 - a)) / std::sqrt(a);
}

struct gof {
    double chi2;
    int    df;
    double z;
    int    outside;   // emissions of a token with p == 0 -- must be zero
};

static gof goodness(const std::map<int32_t, long> & obs, const std::vector<pxa_spec_cand> & p, long n) {
    gof g{0.0, 0, 0.0, 0};
    for (const auto & e : p) {
        if (e.p <= 0.0f) continue;
        const double exp_n = (double) n * (double) e.p;
        const auto   it    = obs.find(e.id);
        const double o     = it == obs.end() ? 0.0 : (double) it->second;
        g.chi2 += (o - exp_n) * (o - exp_n) / exp_n;
        g.df   += 1;
    }
    // one non-zero bin carries no degrees of freedom: every draw must land on that token, which
    // the `outside` count already checks, so there is nothing for the chi-square to say.
    g.df = g.df > 0 ? g.df - 1 : 0;
    g.z  = g.df > 0 ? chi2_z(g.chi2, g.df) : 0.0;
    for (const auto & kv : obs) {
        if (pxa_spec_prob_of(p, kv.first) <= 0.0f) g.outside += (int) kv.second;
    }
    return g;
}

// one shape: draw x ~ q, run the rule, tally what comes out
static void run_shape(const std::string & name,
                      const std::vector<pxa_spec_cand> & p,
                      const std::vector<pxa_spec_cand> & q,
                      long n, uint32_t seed, bool expect_all_accept = false) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    std::map<int32_t, long> obs;
    long n_acc = 0;
    for (long i = 0; i < n; ++i) {
        const int32_t x = pxa_spec_draw(q, u01(rng));
        bool accepted = false;
        const int32_t y = pxa_spec_sampled_step(p, q, x, u01(rng), u01(rng), accepted);
        obs[y]++;
        n_acc += accepted ? 1 : 0;
    }

    const gof g = goodness(obs, p, n);

    // expected acceptance = sum min(p, q) = 1 - TV(p, q)
    double tv_acc = 0.0;
    for (const auto & e : q) tv_acc += std::min((double) e.p, (double) pxa_spec_prob_of(p, e.id));
    const double acc = (double) n_acc / (double) n;

    const bool ok_fit   = std::abs(g.z) < 4.0;
    const bool ok_out   = g.outside == 0;
    const bool ok_acc   = std::abs(acc - tv_acc) < 0.01;
    const bool ok_all   = !expect_all_accept || n_acc == n;
    const bool ok       = ok_fit && ok_out && ok_acc && ok_all;
    if (!ok) g_fail++;

    printf("%-34s n=%-8ld chi2=%9.3f df=%3d z=%+6.2f  accept %.4f (expected %.4f)  outside=%d  %s\n",
           name.c_str(), n, g.chi2, g.df, g.z, acc, tv_acc, g.outside, ok ? "PASS" : "FAIL");
    if (!ok) {
        if (!ok_fit) printf("    FAIL: emitted frequencies do not match p (|z| >= 4)\n");
        if (!ok_out) printf("    FAIL: %d emissions of a token outside p's support\n", g.outside);
        if (!ok_acc) printf("    FAIL: acceptance %.4f != 1 - TV(p,q) = %.4f\n", acc, tv_acc);
        if (!ok_all) printf("    FAIL: q == p must never reject (%ld/%ld accepted)\n", n_acc, n);
    }
}

// the chain: three positions, each with its own p. Whatever the chain does after position 0, the
// FIRST emitted token must be distributed exactly as p0 - that is the guarantee a server gives a
// request. Rejection stops the chain, so this also exercises the stop path.
static void run_chain(const std::string & name,
                      const std::vector<std::vector<pxa_spec_cand>> & ps,
                      const std::vector<std::vector<pxa_spec_cand>> & qs,
                      long n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    std::map<int32_t, long> obs0;
    long n_tok = 0;
    for (long i = 0; i < n; ++i) {
        for (size_t k = 0; k < ps.size(); ++k) {
            const int32_t x = pxa_spec_draw(qs[k], u01(rng));
            bool accepted = false;
            const int32_t y = pxa_spec_sampled_step(ps[k], qs[k], x, u01(rng), u01(rng), accepted);
            if (k == 0) obs0[y]++;
            n_tok++;
            if (!accepted) break;   // the engine stops the chain here
        }
    }

    const gof g  = goodness(obs0, ps[0], n);
    const bool ok = std::abs(g.z) < 4.0 && g.outside == 0;
    if (!ok) g_fail++;
    printf("%-34s n=%-8ld chi2=%9.3f df=%3d z=%+6.2f  mean chain %.3f          outside=%d  %s\n",
           name.c_str(), n, g.chi2, g.df, g.z, (double) n_tok / (double) n, g.outside, ok ? "PASS" : "FAIL");
}

// the filter chain + the draw: build a window from raw logits, apply {top_k, top_p, min_p, temp},
// then check that pxa_spec_draw really samples the .p it left behind.
static void run_filter(const std::string & name, float temp, float top_p, float min_p, int top_k,
                       long n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 2.0f);

    std::vector<pxa_spec_cand> c;
    for (int32_t i = 0; i < 64; ++i) {
        const float l = gauss(rng);
        c.push_back({i, l, 0.0f});
    }
    pxa_spec_sort(c);
    if (top_k > 0 && (size_t) top_k < c.size()) c.resize(top_k);
    pxa_spec_apply_chain(c, {PXA_SPEC_FILTER_TOP_P, PXA_SPEC_FILTER_MIN_P, PXA_SPEC_FILTER_TEMP},
                         top_p, min_p, temp, 1);

    double sum = 0.0;
    for (const auto & e : c) sum += (double) e.p;

    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::map<int32_t, long> obs;
    for (long i = 0; i < n; ++i) obs[pxa_spec_draw(c, u01(rng))]++;

    const gof g = goodness(obs, c, n);
    const bool ok = std::abs(g.z) < 4.0 && g.outside == 0 && std::abs(sum - 1.0) < 1e-4;
    if (!ok) g_fail++;
    printf("%-34s n=%-8ld chi2=%9.3f df=%3d z=%+6.2f  kept %2zu of 64  sum=%.6f  %s\n",
           name.c_str(), n, g.chi2, g.df, g.z, c.size(), sum, ok ? "PASS" : "FAIL");
}

int main() {
    const long N = 250000;

    printf("PXA_SPEC_SAMPLED - lossless acceptance rule, CPU proof\n");
    printf("%-34s %-10s %-16s %-8s %s\n", "shape", "draws", "chi-square", "z", "");

    // 1. q == p: must never reject, and emits p trivially
    {
        auto p = make_dist({{1, 0.4f}, {2, 0.3f}, {3, 0.2f}, {4, 0.1f}});
        run_shape("q == p (never rejects)", p, p, N, 11, true);
    }

    // 2. disjoint support: every draft is rejected, every emission is a residual draw from p
    {
        auto p = make_dist({{1, 0.5f}, {2, 0.3f}, {3, 0.2f}});
        auto q = make_dist({{7, 0.6f}, {8, 0.4f}});
        run_shape("disjoint support", p, q, N, 12);
    }

    // 3. partial overlap, q sharper than p
    {
        auto p = make_dist({{1, 0.30f}, {2, 0.25f}, {3, 0.20f}, {4, 0.15f}, {5, 0.10f}});
        auto q = make_dist({{1, 0.70f}, {2, 0.20f}, {9, 0.10f}});
        run_shape("partial overlap, q sharper", p, q, N, 13);
    }

    // 4. q broader than p (q's tail is outside p: those draws must always be rejected)
    {
        auto p = make_dist({{1, 0.6f}, {2, 0.4f}});
        auto q = make_dist({{1, 0.25f}, {2, 0.25f}, {3, 0.25f}, {4, 0.25f}});
        run_shape("q broader than p", p, q, N, 14);
    }

    // 5. p near-deterministic, q wrong: the hard case for a drafter
    {
        auto p = make_dist({{1, 0.98f}, {2, 0.01f}, {3, 0.01f}});
        auto q = make_dist({{5, 0.90f}, {1, 0.10f}});
        run_shape("p near-deterministic, q wrong", p, q, N, 15);
    }

    // 6. p flat over a wide support, q a 20-token top-k of it (the shipped shape: top_k 20)
    {
        std::vector<std::pair<int32_t, float>> pw, qw;
        for (int32_t i = 0; i < 60; ++i) pw.push_back({i, 1.0f / (1.0f + (float) i)});
        for (int32_t i = 0; i < 20; ++i) qw.push_back({i, 1.0f / (1.0f + (float) i)});
        run_shape("wide p (60), top-k 20 q", make_dist(pw), make_dist(qw), N, 16);
    }

    // 7. a single-token p (the sampler window collapsed to one candidate)
    {
        auto p = make_dist({{3, 1.0f}});
        auto q = make_dist({{3, 0.5f}, {4, 0.5f}});
        run_shape("p is a point mass", p, q, N, 17);
    }

    // 8. a chain of three positions
    {
        std::vector<std::vector<pxa_spec_cand>> ps = {
            make_dist({{1, 0.40f}, {2, 0.30f}, {3, 0.20f}, {4, 0.10f}}),
            make_dist({{1, 0.20f}, {5, 0.50f}, {6, 0.30f}}),
            make_dist({{7, 0.60f}, {8, 0.40f}}),
        };
        std::vector<std::vector<pxa_spec_cand>> qs = {
            make_dist({{1, 0.50f}, {2, 0.20f}, {9, 0.30f}}),
            make_dist({{5, 0.60f}, {6, 0.40f}}),
            make_dist({{7, 0.30f}, {8, 0.30f}, {9, 0.40f}}),
        };
        run_chain("chain of 3, first token ~ p0", ps, qs, N, 18);
    }

    // 9-11. the filter chain itself
    run_filter("filters temp1 top_p .95 top_k 20", 1.0f, 0.95f, 0.0f, 20, N, 21);
    run_filter("filters temp .7 top_p .9 min_p .05", 0.7f, 0.90f, 0.05f, 40, N, 22);
    run_filter("filters temp 2 no cuts", 2.0f, 1.0f, 0.0f, 0, N, 23);

    printf("\n%s (%d failure(s))\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
