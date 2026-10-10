// PXA PXQ composition floor (src/pxa-pxq-composition.h), exercised on FAKE TENSOR LISTS. CPU only,
// no model, no weights, no GPU.
//
// Why this test exists (todo encoder-composition-floor-ple, 2026-09-30). The floor was PXQ-family
// bytes / EVERY output byte, so a model with a 51 GiB host-side per_layer_token_embd (Flash-Next)
// landed at 42.8%, was refused after 1h45m of encoding and the finished file was deleted. The floor
// now counts only RESIDENT weight tensors. This pins:
//   * the Flash-Next shape: old measure < 50% (would have failed), resident measure >= 50% (passes);
//   * a genuinely mislabelled file (91% MXFP4 / 0% PXQ, the case the floor exists for) still fails,
//     with or without host tables in the list;
//   * which tensors are host-side: per_layer_token_embd, an UNTIED token_embd, position / token-type /
//     relative-bias tables, any >= 1M-row gather table - and that a TIED token_embd (no output.weight)
//     and the output head are resident;
//   * the uniform-target "ZERO bytes of the named tier" rule is unchanged;
//   * the exact 50% boundary, an all-host-tables model (nothing to judge), and the override parse.
//
// The preflight's refusal-before-encoding is wired in llama-quantize.cpp; this file tests the
// accounting both passes share.

#include "pxa-pxq-composition.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("%-78s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_fail;
    }
}

// the ggml type ids llama-quantize.cpp passes as the family (PXQ1, PXQ4, PXQ4HQ, PXQ2, PXQ3, PXQ6) and a few others
static const std::set<int> FAMILY = { 248, 252, 253, 254, 255, 256 };
enum { PXQ4 = 252, PXQ3 = 255, PXQ6 = 256, MXFP4 = 39, Q8_0 = 8, Q6_K = 14, F32 = 0 };

static const size_t GiB = 1024ull * 1024ull * 1024ull;

static pxa_comp::tensor T(const char * name, int64_t ne1, int type, size_t bytes) {
    pxa_comp::tensor t;
    t.name = name; t.ne1 = ne1; t.type = type; t.bytes = bytes;
    return t;
}

int main() {
    using namespace pxa_comp;

    // ---- the host-side classifier --------------------------------------------------------------
    check(is_host_side_table("per_layer_token_embd.weight", 320001536, true),  "per_layer_token_embd is host-side");
    check(is_host_side_table("per_layer_token_embd.weight", 262144, false),    "per_layer_token_embd is host-side even when tied");
    check(is_host_side_table("token_embd.weight", 151936, true),               "token_embd WITH an output.weight is host-side");
    check(!is_host_side_table("token_embd.weight", 151936, false),             "token_embd WITHOUT output.weight (tied head) is resident");
    check(!is_host_side_table("output.weight", 151936, true),                  "output.weight is resident");
    check(!is_host_side_table("output.weight", 2000000, true),                 "output.weight is resident even with a 2M vocab");
    check(is_host_side_table("position_embd.weight", 512, true),               "position_embd is host-side");
    check(is_host_side_table("token_types.weight", 2, true),                   "token_types is host-side");
    check(is_host_side_table("enc.blk.0.attn_rel_b.weight", 32, true),         "T5 attn_rel_b is host-side");
    check(is_host_side_table("some_hash_table.weight", 1000000, true),         "a >= 1M-row gather table is host-side");
    check(!is_host_side_table("some_table.weight", 999999, true),              "a 999,999-row table is not caught structurally");
    check(!is_host_side_table("blk.3.ffn_down_exps.weight", 4096, true),       "an expert stack is resident");
    check(!is_host_side_table("blk.3.attn_q.weight", 8192, true),              "an attention weight is resident");
    check(!is_host_side_table("blk.0.ssm_out.weight", 5120, false),            "an ssm_out weight is resident");

    // ---- the Flash-Next shape (swift-fn48c, 2026-09-30): 42.8% by all bytes, passes by resident ----
    // PLE q8_0 51.15 GiB host-side, token_embd q6_K, output q8_0, PXQ experts + backbone.
    {
        std::vector<tensor> ts = {
            T("per_layer_token_embd.weight", 320001536, Q8_0, 51ull * GiB + GiB / 7),
            T("token_embd.weight",           151936,    Q6_K, 2 * GiB),
            T("output.weight",               151936,    Q8_0, 1 * GiB),
            T("blk.0.ffn_down_exps.weight",  2048,      PXQ3, 30 * GiB),
            T("blk.0.ffn_up_exps.weight",    2048,      PXQ3, 30 * GiB),
            T("blk.0.attn_q.weight",         8192,      MXFP4, 8 * GiB),
        };
        const result r = evaluate(ts, FAMILY, -1, /*has_output_weight*/ true);
        check(r.share_all < 0.50,       "Flash-Next shape: the OLD measure (all bytes) is under the floor");
        check(r.share > 0.50,           "Flash-Next shape: the resident measure clears the floor");
        check(!r.below_floor,           "Flash-Next shape: NOT refused");
        check(r.n_host == 2,            "Flash-Next shape: PLE + token_embd are the 2 excluded tables");
        check(r.host_names.size() == 2 && r.host_names[0] == "per_layer_token_embd.weight",
              "Flash-Next shape: the exclusions are named (in input order)");
        check(r.host_bytes == 51ull * GiB + GiB / 7 + 2 * GiB, "Flash-Next shape: host bytes = PLE + token_embd");
        check(r.total_bytes == r.host_bytes + r.resident_bytes, "Flash-Next shape: host + resident = total");
        check(r.resident_family_bytes == 60 * GiB, "Flash-Next shape: resident PXQ bytes = the two expert stacks");
    }

    // ---- the case the floor exists for still fails: 91% MXFP4 / 0% PXQ ---------------------------------
    {
        std::vector<tensor> ts = {
            T("token_embd.weight",          151936, Q6_K,  1 * GiB),
            T("output.weight",              151936, Q8_0,  1 * GiB),
            T("blk.0.ffn_down.weight",      4096,   MXFP4, 40 * GiB),
            T("blk.0.ffn_up.weight",        4096,   MXFP4, 40 * GiB),
            T("blk.0.attn_q.weight",        4096,   PXQ4,  6 * GiB),
        };
        const result r = evaluate(ts, FAMILY, PXQ4, true);
        check(r.below_floor && r.fails(), "mislabelled MXFP4 file: refused");
        check(!r.tier_absent,             "mislabelled MXFP4 file: its named tier is present (only the share fails)");
    }
    // ... and adding a huge host table does not rescue it, nor does it hurt a good file
    {
        std::vector<tensor> bad = {
            T("per_layer_token_embd.weight", 320001536, Q8_0, 200 * GiB),
            T("blk.0.ffn_down.weight",       4096,      MXFP4, 90 * GiB),
            T("blk.0.attn_q.weight",         4096,      PXQ4,  10 * GiB),
        };
        check(evaluate(bad, FAMILY, -1, true).below_floor, "a big host table does not rescue an MXFP4 majority");
        std::vector<tensor> good = {
            T("per_layer_token_embd.weight", 320001536, Q8_0, 200 * GiB),
            T("blk.0.ffn_down.weight",       4096,      PXQ4, 90 * GiB),
            T("blk.0.attn_q.weight",         4096,      MXFP4, 10 * GiB),
        };
        check(!evaluate(good, FAMILY, -1, true).fails(), "a PXQ majority passes with or without the host table");
    }

    // ---- tied embeddings: token_embd is the output head, so it is RESIDENT ---------------------------
    {
        std::vector<tensor> ts = {
            T("token_embd.weight",     151936, Q8_0,  30 * GiB),    // tied: also the head
            T("blk.0.attn_q.weight",   4096,   PXQ4,  25 * GiB),
        };
        const result tied   = evaluate(ts, FAMILY, -1, /*has_output_weight*/ false);
        const result untied = evaluate(ts, FAMILY, -1, /*has_output_weight*/ true);
        check(tied.n_host == 0 && tied.below_floor,  "tied token_embd counts as resident (25/55 -> refused)");
        check(untied.n_host == 1 && !untied.below_floor, "the same list with a separate head: token_embd excluded (25/25 -> passes)");
    }

    // ---- a uniform target that contributed none of its own tier fails regardless of the share ----------
    {
        std::vector<tensor> ts = {
            T("blk.0.ffn_down_exps.weight", 2048, PXQ3, 90 * GiB),   // a majority of PXQ, but PXQ3 not PXQ6
            T("blk.0.attn_q.weight",        4096, MXFP4, 10 * GiB),
        };
        const result r = evaluate(ts, FAMILY, PXQ6, true);
        check(!r.below_floor && r.tier_absent && r.fails(), "uniform PXQ6 target emitting PXQ3 only: ZERO named-tier bytes -> refused");
        check(!evaluate(ts, FAMILY, PXQ3, true).fails(),   "the same list under a PXQ3 target passes");
        check(!evaluate(ts, FAMILY, -1, true).fails(),     "a map-defined mix (no named tier) is judged on share only");
    }

    // ---- the boundary: exactly 50% passes, one byte under fails ----------------------------------------
    {
        std::vector<tensor> half = { T("blk.0.a.weight", 64, PXQ4, 500), T("blk.0.b.weight", 64, MXFP4, 500) };
        check(!evaluate(half, FAMILY, -1, true).below_floor, "exactly 50.0% passes");
        std::vector<tensor> under = { T("blk.0.a.weight", 64, PXQ4, 499), T("blk.0.b.weight", 64, MXFP4, 501) };
        check(evaluate(under, FAMILY, -1, true).below_floor, "49.9% fails");
    }

    // ---- nothing resident to judge, and the empty list ------------------------------------------------
    {
        std::vector<tensor> only_host = { T("per_layer_token_embd.weight", 320001536, Q8_0, 10 * GiB) };
        const result r = evaluate(only_host, FAMILY, -1, true);
        check(!r.below_floor && r.n_resident == 0, "a list of host tables only is not refused (nothing resident to judge)");
        check(!evaluate({}, FAMILY, -1, true).fails(), "an empty list is not refused");
    }

    // ---- the refusal names its next step (todo quantizer-composition-hint) -----------------------------------
    {
        // a ~0.6B model, untied: the q8_0 head is most of the resident non-PXQ bytes (token_embd itself is host-side)
        std::vector<tensor> small = {
            T("token_embd.weight",     151936, Q6_K, 120 * 1024 * 1024),
            T("output.weight",         151936, Q8_0, 160 * 1024 * 1024),
            T("blk.0.attn_k.weight",   1024,   Q8_0,  20 * 1024 * 1024),
            T("blk.0.ffn_down.weight", 1024,   PXQ4, 150 * 1024 * 1024),
        };
        const result r = evaluate(small, FAMILY, PXQ4, true);
        const std::string h = hint(r);
        check(r.below_floor,                                              "small model: refused (150 / 330 MiB resident)");
        check(h.find("--pxq-composition-override") != std::string::npos,  "small model: the hint names --pxq-composition-override");
        check(h.find("PXA_PXQ_COMPOSITION_OVERRIDE=1") != std::string::npos, "small model: ... and its env twin");
        check(h.find("small model") != std::string::npos,                 "small model: the hint says why (embedding/output tables)");
        check(h.find("89%") != std::string::npos,                         "small model: the table share is 160 of 180 MiB non-PXQ = 89%");
        // tied: token_embd is the head and resident, same diagnosis
        std::vector<tensor> tied = { T("token_embd.weight", 151936, Q8_0, 300 * 1024 * 1024), T("blk.0.ffn_up.weight", 1024, PXQ4, 200 * 1024 * 1024) };
        check(hint(evaluate(tied, FAMILY, PXQ4, false)).find("small model") != std::string::npos, "tied small model: same diagnosis");
        // a mislabelled big file (MXFP4 backbone majority) is NOT called a small model, but still names the flag
        std::vector<tensor> big = {
            T("output.weight",         151936, Q8_0,   1 * GiB),
            T("blk.0.ffn_down.weight", 4096,   MXFP4, 90 * GiB),
            T("blk.0.attn_q.weight",   4096,   PXQ4,  10 * GiB),
        };
        const std::string hb = hint(evaluate(big, FAMILY, -1, true));
        check(hb.find("small model") == std::string::npos,                "MXFP4 majority: not blamed on the tables");
        check(hb.find("tier map") != std::string::npos && hb.find("--pxq-composition-override") != std::string::npos,
              "MXFP4 majority: points at the tier map, and names the flag");
        // a uniform target with ZERO named-tier bytes but a passing share: the generic hint
        std::vector<tensor> absent = { T("blk.0.ffn_down_exps.weight", 2048, PXQ3, 90 * GiB), T("output.weight", 151936, Q8_0, 1 * GiB) };
        check(hint(evaluate(absent, FAMILY, PXQ6, true)).find("small model") == std::string::npos, "ZERO-named-tier refusal: generic hint");
    }

    // ---- the override env stays what it was: any non-zero integer ------------------------------------------
    check(override_requested("1"),       "PXA_PXQ_COMPOSITION_OVERRIDE=1 overrides");
    check(override_requested("2"),       "PXA_PXQ_COMPOSITION_OVERRIDE=2 overrides (any non-zero integer)");
    check(!override_requested("0"),      "PXA_PXQ_COMPOSITION_OVERRIDE=0 does not");
    check(!override_requested(""),       "an empty value does not");
    check(!override_requested("yes"),    "non-numeric text does not (atoi == 0, as before)");
    check(!override_requested(nullptr),  "unset does not");

    printf("\n%s (%d failed)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
