// 2026-10-24: PXA_PROFILE_KEEP_TARGET, the rule in src/llama-pxa-profile-keep.h.
// A gemma4_mtp drafter loaded after its target must not replace the target's PXA model profile (the
// overwrite flipped ROUTER_FUSE off for the target: 49.40 vs 51.45 t/s on one P100). Truth table of the
// rule plus the load sequence against the real ggml profile registry (target, then companion).
#include "../src/llama-pxa-profile-keep.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static ggml_pxa_model_profile mk(const char * arch, int n_expert) {
    ggml_pxa_model_profile p;
    memset(&p, 0, sizeof(p));
    p.n_expert = n_expert;
    snprintf(p.arch_name, sizeof(p.arch_name), "%s", arch);
    return p;
}

// what src/llama.cpp does at each load
static void load(const char * arch, int n_expert, bool companion, const char * env, bool draft = false) {
    const ggml_pxa_model_profile p = mk(arch, n_expert);
    if (!llama_pxa_profile_keep_target(companion, draft, ggml_pxa_get_model_profile()->valid != 0, env)) {
        ggml_pxa_set_model_profile(&p);
    }
}

int main() {
    // truth table: (companion, draft load, registered, env)
    CHECK( llama_pxa_profile_keep_target(true,  false, true,  nullptr));
    CHECK( llama_pxa_profile_keep_target(true,  true,  true,  nullptr));
    CHECK(!llama_pxa_profile_keep_target(true,  false, false, nullptr));   // a lone companion still registers
    CHECK(!llama_pxa_profile_keep_target(false, false, true,  nullptr));   // a non-draft load replaces, as before
    CHECK(!llama_pxa_profile_keep_target(false, true,  true,  nullptr));   // a generic -md draft: only at =2
    CHECK(!llama_pxa_profile_keep_target(false, false, false, nullptr));
    CHECK(!llama_pxa_profile_keep_target(true,  false, true,  "0"));       // off switch
    CHECK(!llama_pxa_profile_keep_target(false, true,  true,  "0"));
    CHECK( llama_pxa_profile_keep_target(true,  false, true,  "1"));
    CHECK(!llama_pxa_profile_keep_target(false, true,  true,  "1"));
    CHECK( llama_pxa_profile_keep_target(true,  false, true,  ""));        // empty = default 1
    CHECK( llama_pxa_profile_keep_target(false, true,  true,  "2"));       // opt-in: every draft load
    CHECK( llama_pxa_profile_keep_target(true,  false, true,  "2"));
    CHECK(!llama_pxa_profile_keep_target(false, false, true,  "2"));       // the target itself still registers
    CHECK(!llama_pxa_profile_keep_target(false, true,  false, "2"));       // nothing registered yet

    // target then drafter: the target's profile survives, the generation does not move
    ggml_pxa_set_model_profile(nullptr);
    load("gemma4", 128, false, nullptr);
    const int gen = ggml_pxa_model_profile_generation();
    load("gemma4_mtp", 0, true, nullptr);
    CHECK(strcmp(ggml_pxa_get_model_profile()->arch_name, "gemma4") == 0);
    CHECK(ggml_pxa_get_model_profile()->n_expert == 128);
    CHECK(ggml_pxa_model_profile_generation() == gen);

    // =0: the old last-writer-wins
    load("gemma4_mtp", 0, true, "0");
    CHECK(strcmp(ggml_pxa_get_model_profile()->arch_name, "gemma4_mtp") == 0);
    CHECK(ggml_pxa_get_model_profile()->n_expert == 0);

    // a new target after a reset registers normally
    ggml_pxa_set_model_profile(nullptr);
    load("gemma4_mtp", 0, true, nullptr);
    CHECK(strcmp(ggml_pxa_get_model_profile()->arch_name, "gemma4_mtp") == 0);
    load("qwen35moe", 256, false, nullptr);
    CHECK(strcmp(ggml_pxa_get_model_profile()->arch_name, "qwen35moe") == 0);

    // generic dense draft for a MoE target: replaces at the default, kept at =2
    ggml_pxa_set_model_profile(nullptr);
    load("qwen3moe", 128, false, nullptr);
    load("qwen3", 0, false, nullptr, true);
    CHECK(strcmp(ggml_pxa_get_model_profile()->arch_name, "qwen3") == 0);
    ggml_pxa_set_model_profile(nullptr);
    load("qwen3moe", 128, false, "2");
    load("qwen3", 0, false, "2", true);
    CHECK(strcmp(ggml_pxa_get_model_profile()->arch_name, "qwen3moe") == 0);

    printf("test-pxa-profile-keep: %s (%d failures)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
