// PXA_SPEC_SAMPLED_GATE_v1 (review 2026-10-16): the sampled MTP draft chain's p_min floor and
// depth cutoff must not look at the token they decide on.
//
// mtp_speculative_gen_draft (common/speculative.cpp) draws x_i from the head's filtered q, asks for
// its probability, and DROPS x_i (ending the draft there) when that probability is under p_min or
// pushes the chain's cumulative probability under the depth-cutoff floor. The verifier then runs the
// exact rejection rule (pxa_spec_sampled_step) with the whole q. Whether that pair emits the
// target's distribution depends on which probability the drafter reports:
//   q(x_i)  (the old gate): x_i survives iff q(x_i) clears the floor, the surviving proposal is q
//           restricted to the passing set, and the output over-weights the head's top token;
//   max q   (pxa_spec_draft_gate_prob, the fix): the decision is a function of the row, made before
//           x_i exists, and the output is exactly the target's.
//
// The test transcribes the chain's gating order (cached free token always kept and collapsing the
// chain to one when its probability is under p_min; in-loop pre-push drop at i > 0; depth cutoff
// product; post-push stop) on a toy prefix-dependent target/draft pair, drives it with the shipped
// pxa_spec_apply_chain / pxa_spec_sampled_step / pxa_spec_draft_gate_prob, and checks
//   (1) exact enumeration of the first T emitted tokens: TV(emitted, target) < 1e-5 with the fix,
//       for every chain and every gated arm, and the planted old gate must show TV > 0.01 somewhere
//       (so the test can see the bug it guards);
//   (2) Monte Carlo, N = 1e6 per chain on the 4.0 preset arm (n_max 4, p_min .5, cutoff .5):
//       chi-square p > 0.05 with the fix, and p < 1e-6 for the old gate on the chains it biases.
//
// PXA_SPEC_EXACT_NMIN_v1 (2026-10-17): the same chain under an n_min floor (a draft
// shorter than n_min is dropped and the step decodes plainly). Dropping on the DRAWN length is biased
// (the length is set by rows computed from x_0, ...); deciding on the pre-draw bound
// (pxa_spec_nmin_keep: n_max, or 1 when the cached row's gate stops the chain) and verifying a short
// draft is exact. (3) exact enumeration on four n_min arms: TV < 1e-5 for the bound rule, TV > 0.01
// somewhere for the drawn-length drop; (4) MC N = 1e6 on n4 p.5 cut.5 n_min 2: chi-square p > 0.05
// for the bound rule, p < 1e-6 for the drawn-length drop on the chains it biases.
// CPU only: no model, no context.

#include "pxa-spec-sampled.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

static constexpr int V = 6;
static constexpr int T = 3;

using cands = std::vector<pxa_spec_cand>;
using seq   = std::vector<int32_t>;

struct chain_cfg { const char * name; int top_k; float top_p; float min_p; float temp; };
struct arm_cfg   { const char * name; int n_max; float p_min; float floor; int n_min; };

static uint64_t key_of(const seq & h) {
    uint64_t k = (uint64_t) h.size();
    for (int32_t t : h) k = k * 131u + (uint64_t) (t + 1);
    return k;
}

struct toy {
    chain_cfg cfg;
    uint32_t  seed;
    float     scale;
    float     noise;
    std::unordered_map<uint64_t, cands> pc, qc, rc;

    cands filtered(const std::vector<float> & l) const {
        cands c;
        for (int i = 0; i < V; ++i) c.push_back({ i, l[i], 0.0f });
        pxa_spec_sort(c);
        if (cfg.top_k > 0 && (int) c.size() > cfg.top_k) c.resize(cfg.top_k);   // the top-k window
        const std::vector<pxa_spec_filter> order = { PXA_SPEC_FILTER_TOP_P, PXA_SPEC_FILTER_MIN_P, PXA_SPEC_FILTER_TEMP };
        pxa_spec_apply_chain(c, order, cfg.top_p, cfg.min_p, cfg.temp, 1);
        return c;
    }
    void make(const seq & h) {
        const uint64_t k = key_of(h);
        if (pc.count(k)) return;
        std::mt19937 r(seed ^ (uint32_t) (k * 2654435761u) ^ (uint32_t) (k >> 32));
        std::normal_distribution<float> n01(0.0f, 1.0f);
        std::vector<float> lp(V), lq(V);
        for (int i = 0; i < V; ++i) { lp[i] = scale * n01(r); lq[i] = lp[i] + noise * n01(r); }
        cands p = filtered(lp), q = filtered(lq);
        cands res;
        double s = 0.0;
        for (const auto & e : p) {
            const double d = (double) e.p - (double) pxa_spec_prob_of(q, e.id);
            if (d > 0.0) { res.push_back({ e.id, e.logit, (float) d }); s += d; }
        }
        if (s > 0.0) { for (auto & e : res) e.p = (float) ((double) e.p / s); } else { res = p; }
        pc[k] = std::move(p); qc[k] = std::move(q); rc[k] = std::move(res);
    }
    const cands & p(const seq & h) { make(h); return pc[key_of(h)]; }
    const cands & q(const seq & h) { make(h); return qc[key_of(h)]; }
    const cands & r(const seq & h) { make(h); return rc[key_of(h)]; }
};

using dist = std::map<seq, double>;

static seq cat(const seq & a, int32_t t) { seq s = a; s.push_back(t); return s; }

// ---- exact enumeration of one step's emitted tokens (truncated to k) -----------------------------
struct drafted { int32_t x; const cands * q; };

static void step_outcomes(toy & m, const seq & h, const arm_cfg & a, bool gate_drawn, int k, dist & out,
                          bool nmin_exact = true) {
    auto add = [&](seq e, double w) { if ((int) e.size() > k) e.resize(k); out[e] += w; };

    std::function<void(const std::vector<drafted> &, double)> verify = [&](const std::vector<drafted> & d, double w) {
        std::function<void(size_t, const seq &, const seq &, double)> rec = [&](size_t i, const seq & hh, const seq & em, double w2) {
            if (w2 == 0.0) return;
            if ((int) em.size() >= k) { add(em, w2); return; }
            const cands & pd = m.p(hh);
            if (i == d.size()) {                                       // the free token
                for (const auto & e : pd) if (e.p > 0.0f) add(cat(em, e.id), w2 * e.p);
                return;
            }
            const int32_t x  = d[i].x;
            const double  px = pxa_spec_prob_of(pd, x);
            const double  qx = pxa_spec_prob_of(*d[i].q, x);
            const double  acc = qx > 0.0 ? std::min(1.0, px / qx) : 0.0;
            rec(i + 1, cat(hh, x), cat(em, x), w2 * acc);
            for (const auto & e : m.r(hh)) if (e.p > 0.0f) add(cat(em, e.id), w2 * (1.0 - acc) * e.p);
        };
        rec(0, h, seq(), w);
    };

    // the n_min floor: drop the draft (plain decode: one token from p) or verify it.
    // nmin_exact: decided on the pre-draw bound; otherwise on the drawn length (the old rule).
    int pre_len = a.n_max;
    auto finish = [&](const std::vector<drafted> & d, double w) {
        if (a.n_min > 0 && !pxa_spec_nmin_keep((int) d.size(), nmin_exact ? pre_len : -1, a.n_min)) {
            for (const auto & e : m.p(h)) if (e.p > 0.0f) add(seq{ e.id }, w * e.p);
            return;
        }
        verify(d, w);
    };

    std::function<void(int, const seq &, std::vector<drafted>, float, double)> draft =
        [&](int i, const seq & hh, std::vector<drafted> d, float cum, double w) {
        if (i >= a.n_max || i >= std::max(k, a.n_min)) { finish(d, w); return; }   // later positions cannot move the first k tokens nor the floor
        const cands & qd = m.q(hh);
        for (const auto & e : qd) {
            if (e.p <= 0.0f) continue;
            const float prob = pxa_spec_draft_gate_prob(qd, e.id, gate_drawn);
            if (prob < a.p_min) { finish(d, w * e.p); continue; }       // i > 0 pre-push drop
            float c2 = cum;
            if (a.floor > 0.0f) {
                c2 = cum * prob;
                if (c2 < a.floor) { finish(d, w * e.p); continue; }    // depth cutoff (draft non-empty)
            }
            std::vector<drafted> d2 = d;
            d2.push_back({ e.id, &qd });
            draft(i + 1, cat(hh, e.id), d2, c2, w * e.p);              // post-push stop unreachable at i > 0
        }
    };

    const cands & q0 = m.q(h);
    for (const auto & e : q0) {                                        // the cached free token: always kept
        if (e.p <= 0.0f) continue;
        const float prob0 = pxa_spec_draft_gate_prob(q0, e.id, gate_drawn);
        std::vector<drafted> d = { { e.id, &q0 } };
        pre_len = prob0 < a.p_min ? 1 : a.n_max;                       // the bound: known before x_1 is drawn
        if (prob0 < a.p_min) finish(d, e.p);                           // collapses the chain to one
        else draft(1, cat(h, e.id), d, std::max(0.0f, std::min(1.0f, prob0)), e.p);
    }
}

static dist emitted_exact(toy & m, const arm_cfg & a, bool gate_drawn, bool nmin_exact = true) {
    std::map<std::pair<seq, int>, dist> memo;
    std::function<const dist &(const seq &, int)> F = [&](const seq & h, int k) -> const dist & {
        auto key = std::make_pair(h, k);
        auto it = memo.find(key);
        if (it != memo.end()) return it->second;
        dist res;
        if (k == 0) {
            res[seq()] = 1.0;
        } else {
            dist one;
            step_outcomes(m, h, a, gate_drawn, k, one, nmin_exact);
            for (const auto & [e, w] : one) {
                seq h2 = h; h2.insert(h2.end(), e.begin(), e.end());
                for (const auto & [e2, w2] : F(h2, k - (int) e.size())) {
                    seq full = e; full.insert(full.end(), e2.begin(), e2.end());
                    res[full] += w * w2;
                }
            }
        }
        return memo.emplace(key, std::move(res)).first->second;
    };
    return F(seq{0}, T);
}

static dist target_joint(toy & m) {
    dist res;
    std::function<void(const seq &, const seq &, double)> rec = [&](const seq & h, const seq & em, double w) {
        if ((int) em.size() == T) { res[em] += w; return; }
        for (const auto & e : m.p(h)) if (e.p > 0.0f) rec(cat(h, e.id), cat(em, e.id), w * e.p);
    };
    rec(seq{0}, seq(), 1.0);
    return res;
}

static double tv(const dist & a, const dist & b) {
    double s = 0.0;
    for (const auto & [k, v] : a) { auto it = b.find(k); s += std::fabs(v - (it == b.end() ? 0.0 : it->second)); }
    for (const auto & [k, v] : b) if (!a.count(k)) s += v;
    return 0.5 * s;
}

// ---- Monte Carlo: the same chain, driven through pxa_spec_sampled_step -------------------------
static std::map<seq, long> emitted_mc(toy & m, const arm_cfg & a, bool gate_drawn, long n, uint32_t rs,
                                      bool nmin_exact = true) {
    std::mt19937 rng(rs);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::map<seq, long> counts;
    for (long s = 0; s < n; ++s) {
        seq h = { 0 }, em;
        while ((int) em.size() < T) {
            std::vector<drafted> d;
            const cands & q0 = m.q(h);
            const int32_t x0 = pxa_spec_draw(q0, u01(rng));
            float prob = pxa_spec_draft_gate_prob(q0, x0, gate_drawn);
            d.push_back({ x0, &q0 });
            seq hh = cat(h, x0);
            float cum = std::max(0.0f, std::min(1.0f, prob));
            const int pre_len = prob < a.p_min ? 1 : a.n_max;
            if (!(prob < a.p_min)) {
                for (int i = 1; i < a.n_max; ++i) {
                    const cands & qd = m.q(hh);
                    const int32_t x = pxa_spec_draw(qd, u01(rng));
                    prob = pxa_spec_draft_gate_prob(qd, x, gate_drawn);
                    if (prob < a.p_min) break;
                    if (a.floor > 0.0f) { cum *= prob; if (cum < a.floor) break; }
                    d.push_back({ x, &qd });
                    hh = cat(hh, x);
                    if (prob < a.p_min) break;
                }
            }
            seq hv = h, out;
            bool all = true;
            if (a.n_min > 0 && !pxa_spec_nmin_keep((int) d.size(), nmin_exact ? pre_len : -1, a.n_min)) {
                d.clear();                                     // dropped: the plain decode below
            }
            for (const auto & dr : d) {
                const cands & pd = m.p(hv);
                (void) pxa_spec_draw(pd, u01(rng));            // the target's own draw (discarded under the rule)
                const float ua = u01(rng), ur = u01(rng);
                bool acc = false;
                const int32_t y = pxa_spec_sampled_step(pd, *dr.q, dr.x, ua, ur, acc);
                out.push_back(y); hv = cat(hv, y);
                if (!acc) { all = false; break; }
            }
            if (all) out.push_back(pxa_spec_draw(m.p(hv), u01(rng)));
            em.insert(em.end(), out.begin(), out.end());
            h.insert(h.end(), out.begin(), out.end());
        }
        em.resize(T);
        counts[em] += 1;
    }
    return counts;
}

static double chi2_sf(double x, int k) {           // Wilson-Hilferty
    if (k <= 0) return 1.0;
    const double z = (std::cbrt(x / k) - (1.0 - 2.0 / (9.0 * k))) / std::sqrt(2.0 / (9.0 * k));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

static double chi2_p(const std::map<seq, long> & c, const dist & tgt, long n, double & stat, int & dof) {
    stat = 0.0; int cells = 0; double pe = 0.0, po = 0.0;
    for (const auto & [k, pr] : tgt) {
        const double e = pr * (double) n;
        auto it = c.find(k);
        const double o = it == c.end() ? 0.0 : (double) it->second;
        if (e < 5.0) { pe += e; po += o; continue; }
        stat += (o - e) * (o - e) / e; ++cells;
    }
    for (const auto & [k, o] : c) if (!tgt.count(k)) po += (double) o;   // impossible sequences
    if (pe > 0.0 || po > 0.0) { stat += (po - pe) * (po - pe) / std::max(pe, 1.0); ++cells; }
    dof = cells - 1;
    return chi2_sf(stat, dof);
}

int main() {
    const chain_cfg chains[] = {
        { "temp1.0_topk4",        4, 1.0f, 0.0f, 1.0f },
        { "temp0.7_topk4",        4, 1.0f, 0.0f, 0.7f },
        { "temp1.0_topk6_topp.9", 6, 0.9f, 0.0f, 1.0f },
        { "temp0.7_topk6_minp.1", 6, 1.0f, 0.1f, 0.7f },
    };
    const arm_cfg arms[] = {
        { "n4_p.5_cut.5 (4.0 preset)", 4, 0.50f, 0.5f, 0 },
        { "n4_p.5",                    4, 0.50f, 0.0f, 0 },
        { "n4_cut.5",                  4, 0.00f, 0.5f, 0 },
        { "n3_p.85 (moe auto row)",    3, 0.85f, 0.0f, 0 },
        { "n4_p.75 (stock p_min)",     4, 0.75f, 0.0f, 0 },
    };
    const float scale = 4.0f, noise = 0.7f;
    const uint32_t seed = 20261016u;

    double worst_fixed = 0.0, worst_old = 0.0;
    for (const auto & c : chains) {
        toy m{ c, seed, scale, noise, {}, {}, {} };
        const dist tgt = target_joint(m);
        for (const auto & a : arms) {
            const dist ef = emitted_exact(m, a, false);
            const dist eo = emitted_exact(m, a, true);
            double sf = 0.0; for (const auto & kv : ef) sf += kv.second;
            const double tf = tv(ef, tgt), to = tv(eo, tgt);
            worst_fixed = std::max(worst_fixed, tf);
            worst_old   = std::max(worst_old, to);
            printf("exact  %-22s %-28s TV fixed %.2e  TV old gate %.5f  (mass %.7f)\n", c.name, a.name, tf, to, sf);
            CHECK(tf < 1e-5, "%s %s: fixed gate TV %.3e", c.name, a.name, tf);
            CHECK(std::fabs(sf - 1.0) < 1e-5, "%s %s: emitted mass %.7f", c.name, a.name, sf);
        }
    }
    CHECK(worst_old > 0.01, "the planted old gate must be visibly biased (worst TV %.5f)", worst_old);
    printf("exact: worst TV fixed %.2e, worst TV old gate %.5f\n", worst_fixed, worst_old);

    const long N = 1000000;
    const arm_cfg & preset = arms[0];
    int n_old_biased = 0;
    for (size_t ci = 0; ci < sizeof(chains) / sizeof(chains[0]); ++ci) {
        const auto & c = chains[ci];
        toy m{ c, seed, scale, noise, {}, {}, {} };
        const dist tgt = target_joint(m);
        double st; int dof;
        const auto cf = emitted_mc(m, preset, false, N, 1000u + (uint32_t) ci);
        const double pf = chi2_p(cf, tgt, N, st, dof);
        printf("mc     %-22s fixed    N=%ld chi2 %.1f dof %d p %.3g\n", c.name, N, st, dof, pf);
        CHECK(pf > 0.05, "%s: fixed gate chi-square p %.3g", c.name, pf);
        const double to = tv(emitted_exact(m, preset, true), tgt);
        const auto co = emitted_mc(m, preset, true, N / 4, 2000u + (uint32_t) ci);
        double st2; int dof2;
        const double po = chi2_p(co, tgt, N / 4, st2, dof2);
        printf("mc     %-22s old gate N=%ld chi2 %.1f dof %d p %.3g (exact TV %.5f)\n", c.name, N / 4, st2, dof2, po, to);
        if (to > 0.005) { ++n_old_biased; CHECK(po < 1e-6, "%s: old gate not caught by chi-square (p %.3g)", c.name, po); }
    }
    CHECK(n_old_biased >= 1, "no chain exercised the old gate's bias at the preset");

    // ---- PXA_SPEC_EXACT_NMIN_v1: the n_min floor --------------------------------------------------
    const arm_cfg nmin_arms[] = {
        { "n4_p.5_cut.5_nmin2",        4, 0.50f, 0.5f, 2 },
        { "n4_p.5_cut.5_nmin3",        4, 0.50f, 0.5f, 3 },
        { "n3_p.85_nmin2 (moe auto)",  3, 0.85f, 0.0f, 2 },
        { "n4_p.75_nmin3",             4, 0.75f, 0.0f, 3 },
    };
    double worst_bound = 0.0, worst_drawn = 0.0;
    for (const auto & c : chains) {
        toy m{ c, seed, scale, noise, {}, {}, {} };
        const dist tgt = target_joint(m);
        for (const auto & a : nmin_arms) {
            const dist eb = emitted_exact(m, a, false, true);
            const dist ed = emitted_exact(m, a, false, false);
            const dist eo = emitted_exact(m, a, true,  false);   // PXA_SPEC_EXACT=0: both old rules
            double sb = 0.0; for (const auto & kv : eb) sb += kv.second;
            const double tb = tv(eb, tgt), td = tv(ed, tgt), to = tv(eo, tgt);
            worst_bound = std::max(worst_bound, tb);
            worst_drawn = std::max(worst_drawn, td);
            printf("nmin   %-22s %-28s TV bound %.2e  TV drawn-length %.5f  TV all-old %.5f  (mass %.7f)\n",
                   c.name, a.name, tb, td, to, sb);
            CHECK(tb < 1e-5, "%s %s: n_min bound rule TV %.3e", c.name, a.name, tb);
            CHECK(std::fabs(sb - 1.0) < 1e-5, "%s %s: n_min emitted mass %.7f", c.name, a.name, sb);
        }
    }
    CHECK(worst_drawn > 0.01, "the planted drawn-length n_min drop must be visibly biased (worst TV %.5f)", worst_drawn);
    printf("nmin: worst TV bound rule %.2e, worst TV drawn-length drop %.5f\n", worst_bound, worst_drawn);

    // The bound rule's Monte Carlo is ONE test pooled over the four chains (sum of the chi-square
    // statistics and of their degrees of freedom): four separate tests at .05 would fail about one
    // run in five on an exact rule. The per-chain p is printed for reading only.
    const arm_cfg & nmin_preset = nmin_arms[0];
    int n_drawn_biased = 0;
    double pool_stat = 0.0; int pool_dof = 0;
    for (size_t ci = 0; ci < sizeof(chains) / sizeof(chains[0]); ++ci) {
        const auto & c = chains[ci];
        toy m{ c, seed, scale, noise, {}, {}, {} };
        const dist tgt = target_joint(m);
        double st; int dof;
        const auto cb = emitted_mc(m, nmin_preset, false, N, 3000u + (uint32_t) ci, true);
        const double pb = chi2_p(cb, tgt, N, st, dof);
        printf("mc     %-22s n_min bound N=%ld chi2 %.1f dof %d p %.3g\n", c.name, N, st, dof, pb);
        pool_stat += st; pool_dof += dof;
        const double td = tv(emitted_exact(m, nmin_preset, false, false), tgt);
        const auto cd = emitted_mc(m, nmin_preset, false, N / 4, 4000u + (uint32_t) ci, false);
        double st2; int dof2;
        const double pd = chi2_p(cd, tgt, N / 4, st2, dof2);
        printf("mc     %-22s n_min drawn N=%ld chi2 %.1f dof %d p %.3g (exact TV %.5f)\n", c.name, N / 4, st2, dof2, pd, td);
        if (td > 0.005) { ++n_drawn_biased; CHECK(pd < 1e-6, "%s: drawn-length n_min drop not caught by chi-square (p %.3g)", c.name, pd); }
    }
    CHECK(n_drawn_biased >= 1, "no chain exercised the drawn-length n_min bias at the n_min preset");
    const double pool_p = chi2_sf(pool_stat, pool_dof);
    printf("mc     n_min bound pooled over %d chains: chi2 %.1f dof %d p %.3g\n",
           (int) (sizeof(chains) / sizeof(chains[0])), pool_stat, pool_dof, pool_p);
    CHECK(pool_p > 0.05, "n_min bound rule pooled chi-square p %.3g", pool_p);

    if (g_fail) { fprintf(stderr, "test-spec-sampled-gate: %d failure(s)\n", g_fail); return 1; }
    printf("test-spec-sampled-gate: OK\n");
    return 0;
}
