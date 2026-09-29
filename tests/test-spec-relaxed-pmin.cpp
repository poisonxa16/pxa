// Bug #213 (spec-relaxed-pmin-vs-unnormalised-p): PXA_SPEC_RELAXED keeps a rejected draft token when
// its post-chain probability mass clears PXA_SPEC_RELAXED_PMIN (0.05). The check read cur_p.p, which
// the sampler chain leaves as exp(l - max) -- unnormalised, so every token looked 1/p_max times more
// likely than it was. With the argmax at 0.30 a 0.02 draft token read 0.067 and was kept.
//
// The compare quantity is now pxa_spec_cand_mass() (common/pxa-spec-sampled.h): the softmax of the
// window's logits, which is exactly what the target's draw samples from. Pure arithmetic, no model.

#include "pxa-spec-sampled.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); ++g_fail; } } while (0)

int main() {
    const double pmin = 0.05;

    // a post-chain window: argmax 0.30, the draft token 0.02, the rest spread so the masses sum to 1
    const std::vector<float> mass = {0.30f, 0.25f, 0.20f, 0.13f, 0.10f, 0.02f};
    const size_t j_draft = 5;
    const float shift = 3.7f; // logits are only defined up to a constant

    std::vector<pxa_spec_cand> c;
    for (size_t k = 0; k < mass.size(); ++k) {
        const float l = logf(mass[k]) + shift;
        // .p as the chain leaves it (llama_sample_dist / softmax(normalize=false)): exp(l - max)
        c.push_back({(int32_t) (100 + k), l, expf(l - (logf(mass[0]) + shift))});
    }

    // the defect witness: the old compare quantity passes the floor
    CHECK(c[j_draft].p >= pmin, "expected the unnormalised p to pass the floor (defect witness)");

    // the fixed quantity is the real mass and fails it
    const double m = pxa_spec_cand_mass(c.data(), c.size(), j_draft);
    CHECK(std::fabs(m - 0.02) < 1e-6, "mass of the draft token must be 0.02");
    CHECK(m < pmin, "a 0.02 draft token must not clear a 0.05 floor");

    // masses sum to one, independent of order and shift
    double sum = 0.0;
    for (size_t k = 0; k < c.size(); ++k) sum += pxa_spec_cand_mass(c.data(), c.size(), k);
    CHECK(std::fabs(sum - 1.0) < 1e-6, "masses must sum to 1");

    // a token genuinely above the floor still passes
    CHECK(pxa_spec_cand_mass(c.data(), c.size(), 4) >= pmin, "a 0.10 token must clear a 0.05 floor");

    // edge cases
    CHECK(pxa_spec_cand_mass(c.data(), 0, 0) == 0.0, "empty window");
    CHECK(pxa_spec_cand_mass(c.data(), c.size(), c.size()) == 0.0, "index out of range");
    CHECK(std::fabs(pxa_spec_cand_mass(c.data(), 1, 0) - 1.0) < 1e-9, "single candidate has mass 1");

    if (g_fail) {
        fprintf(stderr, "test-spec-relaxed-pmin: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-spec-relaxed-pmin: OK\n");
    return 0;
}
