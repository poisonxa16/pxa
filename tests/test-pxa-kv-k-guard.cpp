// PXA_KV_K_GUARD (4.0.4) unit test: arch x requested K type -> the K type the context is created with.
//
// Qwen2-family models (qwen2, qwen2vl = Qwen2-VL / Qwen2.5-VL, qwen2moe) answer garbage with a 4-bit K cache, on the
// CPU and on the cards alike so the engine raises a K of 4 bits or fewer to q8_0 for
// them (src/llama-pxa-kv-guard.h; V stays as asked). This test checks
//   1. the pure rule over EVERY arch and EVERY ggml type: only the three Qwen2 archs change, only to q8_0, and only
//      for a quantized type of <= 5.0 bits per weight (the 4-bit family and below);
//   2. the KV-cache types the command line accepts, one by one, against a written-out table;
//   3. the exported llama_pxa_kv_k_guard_type() (what the server's /props reports) with PXA_KV_K_GUARD unset, =1, =0;
//   4. the V q8_0 hint the guard's line carries (#62282): the exact extra V bytes of -ctv q8_0, checked on the
//      supporter's dims (V q4_0 126 MiB -> q8_0 238 MiB = +112 MiB at -c 16384), and no hint for a V of q8_0 or wider.
// CPU only, no model, no backend. Exit 0 when every check holds.

#include "llama-pxa-kv-guard.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void set_guard_env(const char * v) {
#ifdef _WIN32
    _putenv_s("PXA_KV_K_GUARD", v ? v : "");
#else
    if (v) setenv("PXA_KV_K_GUARD", v, 1); else unsetenv("PXA_KV_K_GUARD");
#endif
}

static bool is_qwen2_family(llm_arch a) {
    return a == LLM_ARCH_QWEN2 || a == LLM_ARCH_QWEN2VL || a == LLM_ARCH_QWEN2MOE;
}

int main() {
    // ---- 1. every arch x every ggml type, guard on and off ----------------------------------------------------
    int raised_pairs = 0;
    for (int a = 0; a < (int) LLM_ARCH_UNKNOWN; ++a) {
        const llm_arch arch = (llm_arch) a;
        for (int t = 0; t < (int) GGML_TYPE_COUNT; ++t) {
            const ggml_type type = (ggml_type) t;
            const ggml_type on   = llama_pxa_kv_k_guard_pick(arch, type, true);
            const ggml_type off  = llama_pxa_kv_k_guard_pick(arch, type, false);
            CHECK(off == type, "guard off changed arch %d type %d", a, t);
            if (!is_qwen2_family(arch)) {
                CHECK(on == type, "arch %d (not Qwen2-family) type %d changed to %d", a, t, (int) on);
                continue;
            }
            const bool quant = ggml_is_quantized(type);
            const bool low   = quant && ggml_blck_size(type) > 0 &&
                               8.0*(double) ggml_type_size(type) <= 5.0*(double) ggml_blck_size(type);
            CHECK(on == (low ? GGML_TYPE_Q8_0 : type), "arch %d type %s -> %s, expected %s", a,
                  ggml_type_name(type), ggml_type_name(on), ggml_type_name(low ? GGML_TYPE_Q8_0 : type));
            if (on != type) ++raised_pairs;
        }
    }
    CHECK(raised_pairs > 0, "no (arch, type) pair was raised at all");

    // ---- 2. the KV cache types the command line takes (common.cpp kv_cache_type_from_str), written out ---------
    struct row { ggml_type asked; bool raised; };
    const row rows[] = {
        { GGML_TYPE_F32,    false }, { GGML_TYPE_F16,  false }, { GGML_TYPE_BF16, false },
        { GGML_TYPE_Q8_0,   false }, { GGML_TYPE_Q6_0, false }, { GGML_TYPE_Q5_1, false }, { GGML_TYPE_Q5_0, false },
        { GGML_TYPE_Q4_1,   true  }, { GGML_TYPE_Q4_0, true  }, { GGML_TYPE_IQ4_NL, true },
        // not cache types today, but "any type at 4 bits or fewer" covers them and the 5/6-bit K-quants stay
        { GGML_TYPE_Q2_K,   true  }, { GGML_TYPE_Q3_K, true  }, { GGML_TYPE_Q4_K, true }, { GGML_TYPE_IQ4_XS, true },
        { GGML_TYPE_Q5_K,   false }, { GGML_TYPE_Q6_K, false },
    };
    struct named { llm_arch arch; const char * name; };   // names written out: llama_model_arch_name is not LLAMA_API
    const named archs[] = { { LLM_ARCH_QWEN2, "qwen2" }, { LLM_ARCH_QWEN2VL, "qwen2vl" }, { LLM_ARCH_QWEN2MOE, "qwen2moe" },
                            { LLM_ARCH_QWEN3, "qwen3" }, { LLM_ARCH_QWEN3MOE, "qwen3moe" }, { LLM_ARCH_QWEN3VL, "qwen3vl" },
                            { LLM_ARCH_QWEN35, "qwen35" }, { LLM_ARCH_LLAMA, "llama" }, { LLM_ARCH_GEMMA3, "gemma3" },
                            { LLM_ARCH_DEEPSEEK2, "deepseek2" } };
    printf("arch       ");
    for (const row & r : rows) printf(" %7s", ggml_type_name(r.asked));
    printf("\n");
    for (const named & n : archs) {
        const llm_arch arch = n.arch;
        printf("%-10s ", n.name);
        for (const row & r : rows) {
            const ggml_type got  = llama_pxa_kv_k_guard_pick(arch, r.asked, true);
            const ggml_type want = is_qwen2_family(arch) && r.raised ? GGML_TYPE_Q8_0 : r.asked;
            CHECK(got == want, "%s %s -> %s, expected %s", n.name, ggml_type_name(r.asked),
                  ggml_type_name(got), ggml_type_name(want));
            printf(" %7s", got == r.asked ? "=" : ggml_type_name(got));
        }
        printf("\n");
    }

    // ---- 3. the exported function and the off switch -----------------------------------------------------------
    struct api { const char * arch; ggml_type asked; ggml_type on; };
    const api calls[] = {
        { "qwen2vl",  GGML_TYPE_Q4_0,   GGML_TYPE_Q8_0 }, { "qwen2",    GGML_TYPE_Q4_0,   GGML_TYPE_Q8_0 },
        { "qwen2moe", GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0 }, { "qwen2",    GGML_TYPE_Q4_1,   GGML_TYPE_Q8_0 },
        { "qwen2vl",  GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0 }, { "qwen2vl",  GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0 },
        { "qwen2vl",  GGML_TYPE_F16,    GGML_TYPE_F16  }, { "qwen3",    GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0 },
        { "qwen3vl",  GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0 }, { "llama",    GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0 },
        { "gemma3",   GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0 }, { "no-such",  GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0 },
        { nullptr,    GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0 },
    };
    const char * envs[] = { nullptr, "1", "0" };
    for (const char * e : envs) {
        set_guard_env(e);
        const bool off = e != nullptr && strcmp(e, "0") == 0;
        for (const api & c : calls) {
            const ggml_type got  = llama_pxa_kv_k_guard_type(c.arch, c.asked);
            const ggml_type want = off ? c.asked : c.on;
            CHECK(got == want, "PXA_KV_K_GUARD=%s: %s %s -> %s, expected %s", e ? e : "(unset)", c.arch ? c.arch : "(null)",
                  ggml_type_name(c.asked), ggml_type_name(got), ggml_type_name(want));
        }
    }
    set_guard_env(nullptr);

    // ---- 4. the V q8_0 hint (#62282) -----------------------------------------------------------------------
    // the supporter's file (#62054): Qwen2.5-VL-7B, 28 layers x 4 KV heads x 128 = 512 V values per cell, -c 16384
    const uint64_t MiB = 1048576;
    const std::vector<int64_t> qwen25vl7b(28, 512);
    const int64_t cells = 16384;
    CHECK(ggml_row_size(GGML_TYPE_Q4_0, 512)*28*cells == 126*MiB, "V q4_0 is not the 126 MiB the KV line printed");
    CHECK(ggml_row_size(GGML_TYPE_Q8_0, 512)*28*cells == 238*MiB, "V q8_0 is not the 238 MiB the K q8_0 line printed");
    const uint64_t extra = llama_pxa_kv_v_q8_extra_bytes(qwen25vl7b.data(), 28, cells, GGML_TYPE_Q4_0);
    CHECK(extra == 112*MiB, "supporter dims: +%llu bytes, expected +112 MiB", (unsigned long long) extra);
    const std::string hint = llama_pxa_kv_k_guard_v_hint(GGML_TYPE_Q4_0, extra);
    CHECK(hint == "for faster decode also set V to q8_0 (-ctv q8_0): +112 MiB at this context", "hint text: %s", hint.c_str());
    printf("hint (supporter, -c 16384, V q4_0): %s\n", hint.c_str());
    // q8_0 or wider: nothing to add, no hint
    for (ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32 }) {
        CHECK(!llama_pxa_kv_v_narrower_than_q8(t), "%s counted as narrower than q8_0", ggml_type_name(t));
        CHECK(llama_pxa_kv_v_q8_extra_bytes(qwen25vl7b.data(), 28, cells, t) == 0, "%s: extra bytes", ggml_type_name(t));
        CHECK(llama_pxa_kv_k_guard_v_hint(t, 112*MiB).empty(), "%s: a hint was printed", ggml_type_name(t));
    }
    // every narrower V cache type the command line takes: the row difference, layer by layer
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q6_0 }) {
        CHECK(llama_pxa_kv_v_narrower_than_q8(t), "%s not counted as narrower than q8_0", ggml_type_name(t));
        const uint64_t want = (uint64_t) (ggml_row_size(GGML_TYPE_Q8_0, 512) - ggml_row_size(t, 512))*28*cells;
        CHECK(want > 0 && llama_pxa_kv_v_q8_extra_bytes(qwen25vl7b.data(), 28, cells, t) == want, "%s: extra bytes", ggml_type_name(t));
        CHECK(!llama_pxa_kv_k_guard_v_hint(t, want).empty(), "%s: no hint", ggml_type_name(t));
    }
    // per-layer widths sum layer by layer; a layer with no V cache (width 0) adds nothing; no cells / no layers = 0
    const std::vector<int64_t> mixed = { 512, 0, 1024, 256 };
    CHECK(llama_pxa_kv_v_q8_extra_bytes(mixed.data(), 4, 4096, GGML_TYPE_Q4_0) ==
          (uint64_t) ((544 - 288) + (1088 - 576) + (272 - 144))*4096, "mixed widths");
    CHECK(llama_pxa_kv_v_q8_extra_bytes(mixed.data(), 4, 0, GGML_TYPE_Q4_0) == 0, "0 cells");
    CHECK(llama_pxa_kv_v_q8_extra_bytes(nullptr, 4, 4096, GGML_TYPE_Q4_0) == 0, "no layers");
    // under 10 MiB the hint keeps one decimal: 1 layer x 512 values x 4096 cells, q4_0 -> +1.0 MiB
    const uint64_t small = llama_pxa_kv_v_q8_extra_bytes(qwen25vl7b.data(), 1, 4096, GGML_TYPE_Q4_0);
    CHECK(small == MiB && llama_pxa_kv_k_guard_v_hint(GGML_TYPE_Q4_0, small) ==
          "for faster decode also set V to q8_0 (-ctv q8_0): +1.0 MiB at this context", "small hint");

    printf("test-pxa-kv-k-guard: %d checks, %d failed, %d (arch, type) pairs raised\n", g_checks, g_fail, raised_pairs);
    return g_fail == 0 ? 0 : 1;
}
