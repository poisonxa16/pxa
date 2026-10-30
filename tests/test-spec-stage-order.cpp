// PXA_SPEC_STAGE_ORDER_v1 (2026-10-19): `--spec-type mtp --spec-type ngram-mod` was refused by the chain
// validator while the reverse spelling ran. common_speculative_canonicalize_stage_order() puts the self-spec (table / suffix)
// stage in front of the model drafter whatever order it was typed in; PXA_SPEC_STAGE_ORDER=0 keeps the typed order (and the
// refusal). Pure parameter logic, no model, no GPU.

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); ++g_fail; } } while (0)

static common_params_speculative chain(std::initializer_list<const char *> specs) {
    common_params_speculative p;
    for (const char * s : specs) p.stages.push_back(common_speculative_stage_from_arg(s));
    p.type = p.stages.empty() ? COMMON_SPECULATIVE_TYPE_NONE : p.stages.front().type;
    return p;
}

int main() {
#ifdef _WIN32
    _putenv_s("PXA_SPEC_STAGE_ORDER", "");
#else
    unsetenv("PXA_SPEC_STAGE_ORDER");
#endif
    // the supporter's reverse spelling becomes the working chain, knobs travel with their stage
    {
        auto p = chain({"mtp:n_max=4,p_min=0.75", "ngram-mod:n_max=48,n_min=2,ngram_size_n=8"});
        std::string err;
        CHECK(!common_speculative_validate_chain(p, &err), "the typed reverse order is refused before canonicalizing");
        CHECK(common_speculative_canonicalize_stage_order(p), "reverse order is reordered");
        CHECK(p.stages[0].type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD && p.stages[1].type == COMMON_SPECULATIVE_TYPE_MTP, "ngram-mod -> mtp");
        CHECK(p.stages[0].n_max == 48 && p.stages[0].ngram_size_n == 8 && p.stages[1].n_max == 4 && p.stages[1].p_min == 0.75f,
              "each stage keeps its own knobs");
        CHECK(p.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD, "chain type follows the first stage");
        err.clear();
        CHECK(common_speculative_validate_chain(p, &err), err.c_str());
        CHECK(common_speculative_stage_chain_to_str(p) == "ngram_mod -> mtp", "chain string");
    }
    // the working order and single stages are left alone
    {
        auto p = chain({"ngram-mod", "mtp"});
        CHECK(!common_speculative_canonicalize_stage_order(p), "ngram-mod -> mtp unchanged");
        CHECK(p.stages[0].type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD, "still first");
        auto q = chain({"mtp"});
        CHECK(!common_speculative_canonicalize_stage_order(q), "a lone stage is unchanged");
        auto e = chain({});
        CHECK(!common_speculative_canonicalize_stage_order(e), "an empty chain is unchanged");
    }
    // a suffix stage typed after MTP moves too
    {
        auto p = chain({"mtp", "suffix"});
        CHECK(common_speculative_canonicalize_stage_order(p), "mtp, suffix reordered");
        CHECK(p.stages[0].type == COMMON_SPECULATIVE_TYPE_SUFFIX, "suffix first");
    }
    // the off switch keeps the typed order (and the validator's refusal)
    {
#ifdef _WIN32
        _putenv_s("PXA_SPEC_STAGE_ORDER", "0");
#else
        setenv("PXA_SPEC_STAGE_ORDER", "0", 1);
#endif
        auto p = chain({"mtp", "ngram-mod"});
        CHECK(!common_speculative_canonicalize_stage_order(p), "=0 keeps the order");
        CHECK(p.stages[0].type == COMMON_SPECULATIVE_TYPE_MTP, "mtp stays first under =0");
        CHECK(!common_speculative_validate_chain(p, nullptr), "and is still refused");
    }
    if (g_fail == 0) printf("test-spec-stage-order: all checks passed\n");
    // PXA_SPEC_MTP_NGRAM_SHALLOW_v1: the auto table in front of MTP is dropped when the budget holds it no deeper than the head
    {
        auto p = chain({"ngram:n_max=3,n_min=2,ngram_size_n=24", "mtp:n_max=3,p_min=0.5"});
        CHECK(!common_speculative_drop_shallow_ngram_front(p), "no auto flag: a typed chain is never touched");
        CHECK(p.stages.size() == 2, "typed chain kept");
        p.pxa_ngram_auto_front = true;
        int t = 0, h = 0;
        CHECK(common_speculative_drop_shallow_ngram_front(p, &t, &h), "table depth 3 <= head depth 3 is dropped");
        CHECK(t == 3 && h == 3, "depths reported");
        CHECK(p.stages.size() == 1 && p.stages[0].type == COMMON_SPECULATIVE_TYPE_MTP && p.type == COMMON_SPECULATIVE_TYPE_MTP, "MTP alone");
        CHECK(p.stages[0].n_max == 3 && p.stages[0].p_min == 0.5f, "the head keeps its knobs");
        CHECK(!p.pxa_ngram_auto_front, "flag cleared");
    }
    {
        auto p = chain({"ngram:n_max=7,n_min=2,ngram_size_n=24", "mtp:n_max=3,p_min=0.5"});
        p.pxa_ngram_auto_front = true;
        CHECK(!common_speculative_drop_shallow_ngram_front(p), "table depth 7 > head 3 is kept");
        CHECK(p.stages.size() == 2, "deep table kept");
    }
    {
        auto p = chain({"ngram:n_max=2,n_min=2,ngram_size_n=24", "mtp:n_max=3,p_min=0.5"});
        p.pxa_ngram_auto_front = true;
#ifdef _WIN32
        _putenv_s("PXA_SPEC_MTP_NGRAM_SHALLOW", "1");
#else
        setenv("PXA_SPEC_MTP_NGRAM_SHALLOW", "1", 1);
#endif
        CHECK(!common_speculative_drop_shallow_ngram_front(p), "PXA_SPEC_MTP_NGRAM_SHALLOW=1 keeps a shallow table");
        CHECK(p.stages.size() == 2, "off switch keeps the chain");
    }
    return g_fail == 0 ? 0 : 1;
}
