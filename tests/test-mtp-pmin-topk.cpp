// PXA_MTP_PMIN_TOPK_v1: the MTP confidence floor has to be compared against a probability that
// can actually reach it.
//
// The floor came from draft-MODEL speculation, where one proposal costs a whole second forward
// pass, and it was applied here to the FULL-VOCABULARY softmax of the head's argmax. On a
// 248320-token vocabulary that quantity was measured with mean 0.036, clearing 0.75 in 0.0 % of
// draft steps (measured 2026-09-08) -- so the stock 0.75 floor was not a confidence
// filter, it was an off switch, and the configured n_max was unreachable. The reference
// implementation of this drafter compares the same argmax renormalised over a top_k = 10 window
// instead. common_sampler_prob_topk_renorm() is that quantity.
//
// This test pins the arithmetic, not the policy: no model, no GPU, no sampler state.

#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_fail = 0;

static void fail(const char * fmt, ...) {
    fprintf(stderr, "  FAIL: ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    ++g_fail;
}

// The definition, written out the slow and obvious way: softmax over the k largest logits,
// evaluated at the largest one.
static double reference_topk(const std::vector<float> & logits, int k) {
    std::vector<float> v = logits;
    std::sort(v.begin(), v.end(), std::greater<float>());
    const int n = (int) std::min<size_t>(v.size(), (size_t) std::max(k, 1));
    const double max_val = v[0];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp((double) v[i] - max_val);
    }
    return 1.0 / sum;
}

static float argmax_logit(const std::vector<float> & logits) {
    return *std::max_element(logits.begin(), logits.end());
}

static void fill(std::vector<float> & v, uint32_t seed, float scale) {
    uint32_t s = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((float) ((s >> 9) & 0xFFFF) / 32768.0f - 1.0f) * scale;
    }
}

static void check_case(const char * name, const std::vector<float> & logits, int k, double tol) {
    const float  mx  = argmax_logit(logits);
    const double got = common_sampler_prob_topk_renorm((int) logits.size(), logits.data(), mx, k);
    const double ref = reference_topk(logits, k >= 2 && k < (int) logits.size() ? k : (int) logits.size());
    const double d   = std::fabs(got - ref);
    printf("  %-46s k=%-3d n=%-7zu got=%.9f ref=%.9f |d|=%.3e\n",
           name, k, logits.size(), got, ref, d);
    if (!(d <= tol)) {
        fail("%s: k=%d got %.9f, expected %.9f (|d| = %.3e > %.3e)", name, k, got, ref, d, tol);
    }
}

int main() {
    printf("test-mtp-pmin-topk: the MTP p_min compare quantity\n");

    // --- the arithmetic, against the definition ------------------------------------------------
    {
        std::vector<float> l(2048);
        fill(l, 1, 6.0f);
        for (int k : {2, 3, 10, 40, 64}) {
            check_case("random logits vs sorted-softmax", l, k, 1e-6);
        }
        // k outside the window falls back to the full-vocabulary softmax, by contract
        check_case("k = 0 -> full vocabulary",  l, 0,             1e-6);
        check_case("k = 1 -> full vocabulary",  l, 1,             1e-6);
        check_case("k = n -> full vocabulary",  l, (int) l.size(), 1e-6);
    }

    // --- degenerate inputs ---------------------------------------------------------------------
    {
        std::vector<float> flat(1000, 3.25f);           // every candidate identical
        const double got = common_sampler_prob_topk_renorm((int) flat.size(), flat.data(), 3.25f, 10);
        printf("  %-46s k=10 got=%.9f expected=%.9f\n", "all logits equal -> 1/k", got, 0.1);
        if (std::fabs(got - 0.1) > 1e-6) {
            fail("all-equal logits: expected 1/k = 0.1, got %.9f", got);
        }

        std::vector<float> spike(1000, -40.0f);         // one dominant candidate
        spike[517] = 20.0f;
        const double got2 = common_sampler_prob_topk_renorm((int) spike.size(), spike.data(), 20.0f, 10);
        printf("  %-46s k=10 got=%.9f expected~1\n", "one dominant logit -> ~1", got2);
        if (got2 < 1.0 - 1e-6) {
            fail("dominant logit: expected ~1.0, got %.9f", got2);
        }
    }

    // --- the point of the change, at the shipped vocabulary size -------------------------------
    // A 248320-wide head. The renormalised number must be >= the full-vocabulary one for the same
    // logits (it drops the mass outside the window), and on a realistic spread the two must land
    // on opposite sides of the stock 0.75 floor -- which is the whole reason the floor could never
    // fire on the full-vocabulary quantity.
    {
        const int n_vocab = 248320;
        std::vector<float> l(n_vocab);
        fill(l, 7, 4.0f);
        // give the top candidate a lead of the size the head actually produces
        const size_t best = 123457;
        l[best] = 12.0f;
        for (size_t i = 1; i <= 9; ++i) {
            l[best + i * 977] = 8.0f;    // nine rivals, a 4-logit lead: a CONFIDENT head step
        }
        const float  mx    = argmax_logit(l);
        const double full  = common_sampler_prob_topk_renorm(n_vocab, l.data(), mx, 0);
        const double top10 = common_sampler_prob_topk_renorm(n_vocab, l.data(), mx, 10);
        printf("  %-46s full=%.6f top10=%.6f\n", "248320-wide head: scale of the two quantities", full, top10);
        if (!(top10 >= full)) {
            fail("renormalising over a top-10 window must not lower the probability (full %.6f, top10 %.6f)",
                 full, top10);
        }
        // A head step this confident -- a 4-logit lead over its nine nearest rivals -- is exactly
        // what a floor is supposed to let through, and the stock 0.75 numeral lets it through on
        // one scale and blocks it on the other. That is the change, in one assertion.
        if (!(full < 0.75 && top10 > 0.75)) {
            fail("a confident step did not straddle the stock 0.75 floor (full %.6f, top10 %.6f) -- "
                 "the SAME numeral has to be a different knob on the two scales or this change is "
                 "pointless", full, top10);
        }
    }

    if (g_fail) {
        printf("test-mtp-pmin-topk: FAILED (%d)\n", g_fail);
        return 1;
    }
    printf("test-mtp-pmin-topk: OK\n");
    return 0;
}
