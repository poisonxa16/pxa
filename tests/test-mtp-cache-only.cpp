// PXA_MTP_DRAFT_CACHE_ONLY gate test.
//
// The reduced K/V-only graph drops everything downstream of the MTP head's cache write: the query
// projection, the attention, the output projection, the FFN and the LM head. That is free ONLY
// where the caller reads nothing back. Taking it anywhere else would hand the caller an
// uninitialised logit and change the generated text, so the whole safety of the feature is in this
// one predicate, and it is exhaustively enumerated here rather than argued about.
//
// The four required conditions (common/pxa-mtp-cache-only.h):
//   1. the lever is on (default off);
//   2. the caller wants ZERO outputs;
//   3. the MTP head is exactly one grafted layer;
//   4. there is at least one token to store.
//
// CPU only, no model, no server, no GPU. Exit 0 on pass.

#include "pxa-mtp-cache-only.h"

#include <cstdio>

static int fails = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

int main() {
    // exhaustive over the whole gate domain: every combination of the four conditions, with
    // n_layer_nextn and n_tokens swept past their boundaries in both directions
    long n = 0, n_ok = 0;
    for (int enabled = 0; enabled <= 1; ++enabled) {
        for (int wants = 0; wants <= 1; ++wants) {
            for (int nextn = -1; nextn <= 4; ++nextn) {
                for (int toks = -1; toks <= 4; ++toks) {
                    pxa_mtp_cache_only_state st;
                    st.enabled       = enabled != 0;
                    st.wants_output  = wants != 0;
                    st.n_layer_nextn = nextn;
                    st.n_tokens      = toks;

                    const bool got = pxa_mtp_cache_only_ok(st);
                    const bool expected = (enabled != 0) && (wants == 0) && (nextn == 1) && (toks >= 1);

                    if (got != expected) {
                        fprintf(stderr, "FAIL: enabled=%d wants_output=%d nextn=%d n_tokens=%d -> %d, expected %d\n",
                                enabled, wants, nextn, toks, (int) got, (int) expected);
                        fails++;
                    }
                    n++;
                    if (got) n_ok++;
                }
            }
        }
    }

    // the individually load-bearing refusals, named so a regression says which rule broke
    {
        pxa_mtp_cache_only_state good;
        good.enabled = true; good.wants_output = false; good.n_layer_nextn = 1; good.n_tokens = 64;
        check(pxa_mtp_cache_only_ok(good), "the fully satisfied gate opens");

        pxa_mtp_cache_only_state st = good; st.enabled = false;
        check(!pxa_mtp_cache_only_ok(st), "lever off refuses (this is the default)");

        st = good; st.wants_output = true;
        check(!pxa_mtp_cache_only_ok(st), "a caller that reads an output refuses");

        st = good; st.n_layer_nextn = 2;
        check(!pxa_mtp_cache_only_ok(st), "a multi-layer MTP head refuses");

        st = good; st.n_layer_nextn = 0;
        check(!pxa_mtp_cache_only_ok(st), "a model with no MTP head refuses");

        st = good; st.n_tokens = 0;
        check(!pxa_mtp_cache_only_ok(st), "an empty batch refuses");
    }

    // the default state of the world: with the lever unset nothing can open the gate, whatever the
    // rest of the inputs say
    for (int wants = 0; wants <= 1; ++wants) {
        for (int nextn = 0; nextn <= 3; ++nextn) {
            pxa_mtp_cache_only_state st;
            st.enabled = false;
            st.wants_output = wants != 0;
            st.n_layer_nextn = nextn;
            st.n_tokens = 1024;
            if (pxa_mtp_cache_only_ok(st)) { check(false, "gate opened with the lever off"); }
        }
    }

    printf("test-mtp-cache-only: %ld gate states enumerated, %ld open -> %s\n",
           n, n_ok, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
