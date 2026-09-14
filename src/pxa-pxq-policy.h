#pragma once
//
// PXQ TIER POLICY (POLICY_REV 3) — a named MIX on top of the BACKBONE_REV 2 allocation table.
// ============================================================================================
//
// WHAT PROBLEM THIS SOLVES. Until now a PXQ level name meant two different things depending
// on whether the model had routed experts, and only one of them was any good:
//
//   * MoE (Flash-Next, the 122B): the level names the EXPERT tier and BACKBONE_REV 2 already
//     spends more on the always-resident backbone — PXQ2 puts attention at pxq4 and
//     attn_output at pxq4hq. That is a mix, and it is why our MoE files hold up at 2 bits.
//
//   * DENSE (Qwen3.8-27B and every other non-MoE model): the level names EVERYTHING. The two
//     dense branches in pxa_pxq_backbone_type() route the whole GEMM backbone — attention
//     projections included — to the named tier, so "PXQ2" really does mean attention at
//     2.27 bpw. They were written to satisfy the composition assertion (a PXQ3 target that
//     emits zero pxq3 bytes is refused, and the MoE table's promotion did exactly that on a
//     dense model), not because uniform was measured to be the right allocation.
//
// So the dense path had no policy at all. This header adds one. The rule is the same one the
// MoE table already encodes, stated once and applied to both shapes:
//
//   THE LEVEL NAMES THE TIER OF THE BULK — routed experts on a MoE model, the FFN on a dense
//   one — AND THE ATTENTION BLOCK IS BOUGHT UP FROM THERE.
//
// Attention is a small share of a dense transformer's bytes (Qwen3.8-27B: 21% of parameters
// against the FFN's 64%) and every one of those bytes is read on every token, so buying it up
// is cheap in file size and free in bandwidth terms relative to what it protects. attn_output
// gets one extra notch: it is this tree's worst-measured class (3.2x the Q4_K_M control on
// Laguna-S-2.1, the number that motivated BACKBONE_REV 2 in the first place).
//
// THE PXQ4 CAP. No policy promotes past pxq4, for two independent reasons and either one alone
// would be enough.
//
//   PORTABILITY. pxq4hq is llama.cpp-only in this release: the vLLM sidecar has no pxq4hq
//   decoder and no pxq4hq kernel, so a profile file containing one tensor of it would load on
//   one engine and not the other. A profile is a DEFAULT-SHAPED thing — it has to produce a
//   file that runs everywhere the level name runs, or the name means two different things
//   depending on where you point it. pxq4hq stays available as an explicit user choice
//   (--custom-q, a PXQU map, or the PXQ4-HQ level itself), which is where a type that only
//   one engine can read belongs.
//
//   SPEED. pxq4 and pxq4hq are the only two types pxa_pxq_mmvq_type() admits, so anything
//   above them drops the class off the fused single-kernel decode path onto the per-operand
//   divert — measured at -8..10% decode on sm_60 and -30..35% on sm_70 when the PXQ6 backbone
//   was retired on 2026-07-31. And of the two admitted types, pxq4 is the faster: the sm_70
//   campaign separated them at 33.861 t/s (pxq4 core) against 30.969 (pxq4hq), +9.3% on the
//   same path, while being SMALLER (4.2526 bpw against 4.52). So the cap costs nothing it
//   would have wanted anyway.
//
// THE PROFILES
//
//   uniform    Do nothing. BACKBONE_REV 2 exactly as it stands, which is what every file
//              published before 2026-09-09 was built with. It was the default until the KLD
//              ladder measured it: against a PXQ4 anchor at perplexity 6.27, uniform PXQ2
//              lands at 30.01 (4.8x, mean KLD 1.654, same-top-token 48.8% -- a coin flip),
//              while the same base under attn4 lands at 9.35 (1.49x, KLD 0.453, 72.8%) for
//              10.1% more bytes and FASTER decode on both engines. So the PXQ2 base now
//              defaults to attn4 and uniform is opt-in: --pxq-policy uniform (or
//              --pxq-uniform) still reproduces the old allocation exactly. PXQ3 and above are
//              unchanged; they already converge to that shape under balanced, and flipping a
//              default nobody measured for them would repeat the same mistake the other way.
//              POLICY_REV does not bump -- the profiles mean what they meant, only the
//              unspecified choice moved. The selection itself lives in
//              pxa_pxq_policy_for_tier() in src/llama-quantize.cpp.
//
//   balanced   Dense: FFN at the named tier; the attention block one notch above it;
//              attn_output two notches above it. Capped at pxq4, floored at the named tier.
//              MoE: the whole backbone at pxq4. (At PXQ2 that is rev 2 minus its pxq4hq
//              attn_output. At PXQ3 it makes the file SMALLER and the decode faster than rev
//              2's all-pxq4hq backbone: 4.2526 bpw against 4.52, on the type the sm_70
//              campaign measured 9% faster. At PXQ4 and above it is a no-op.)
//
//   attn4      Dense: FFN at the named tier; the whole attention block at pxq4 — i.e. the MoE
//              rev-2 rule applied verbatim to a dense model, with no dependence on how low the
//              named tier goes. On MoE it is the same rule as balanced. This is the aggressive
//              arm: at PXQ2 on the 27B it costs about twice what balanced does. Under the pxq4
//              cap it differs from balanced only at levels pxq1 and pxq2 — at pxq3 and above
//              the two converge, and building both would be two names for one file.
//
// ⚠ NOTHING HERE IS MEASURED ON HARDWARE YET. The direction is reasoned from this tree's own
// recorded per-class error numbers and from the MoE table's existing shape; the byte cost is
// exact; the quant-policy A/B arm list settles it.
//
// Kept in its own header so tests/test-pxq-policy.cpp can pin the tensor-name -> tier mapping
// for both architectures without linking the quantizer.

#include "ggml.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

enum pxa_pxq_policy_id {
    PXA_POLICY_UNIFORM  = 0,   // BACKBONE_REV 2 unchanged — today's bytes
    PXA_POLICY_BALANCED = 1,   // attention +1 notch, attn_output +2, cap pxq4hq
    PXA_POLICY_ATTN4    = 2,   // attention pxq4, attn_output pxq4hq (the MoE rule, verbatim)
};

enum pxa_pxq_policy_role {
    PXA_ROLE_OTHER    = 0,     // the policy layer does not touch it
    PXA_ROLE_FFN      = 1,     // dense FFN + shared expert: the bulk on a dense model
    PXA_ROLE_ATTN     = 2,     // q / qkv / q-LoRA pair / per-channel gate / DeltaNet out-proj
    PXA_ROLE_ATTN_OUT = 3,     // attn_output and its LoRA pair — the worst-measured class
};

static inline const char * pxa_pxq_policy_name(pxa_pxq_policy_id p) {
    switch (p) {
        case PXA_POLICY_BALANCED: return "balanced";
        case PXA_POLICY_ATTN4:    return "attn4";
        default:                  return "uniform";
    }
}

// Returns false (and leaves *out alone) on an unknown name, so the caller can complain loudly
// rather than silently building a file with a policy the operator did not ask for.
static inline bool pxa_pxq_policy_parse(const char * s, pxa_pxq_policy_id * out) {
    if (!s || !*s || !out) {
        return false;
    }
    std::string v(s);
    for (auto & c : v) c = (char) std::tolower((unsigned char) c);
    if (v == "uniform" || v == "off" || v == "0" || v == "rev2") { *out = PXA_POLICY_UNIFORM;  return true; }
    if (v == "balanced")                                         { *out = PXA_POLICY_BALANCED; return true; }
    if (v == "attn4" || v == "attn-4" || v == "attnhq")          { *out = PXA_POLICY_ATTN4;    return true; }
    return false;
}

// ---------------------------------------------------------------------------------------
// The promotion ladder. Only the five slab tiers that have a CUDA decode path AND a defined
// bit ordering are on it; q8_0/q6_k/f16 pins made elsewhere in the backbone table never reach
// this header. pxq6 is deliberately NOT on the ladder as a promotion TARGET (see the cap note
// above) but IS recognised as a floor, so a PXQ6 build is left alone rather than demoted.
// ---------------------------------------------------------------------------------------
// THE DEFAULT PROFILE FOR A NAMED BASE TYPE. It lives here, next to the profiles themselves,
// so the quantizer and tests/test-pxq-policy.cpp read the SAME rule -- a default duplicated in a
// test is a default that can silently drift from the one that ships.
//
// A tier moves only when it was MEASURED to be worth moving, one base at a time.
//
//   PXQ2 -> attn4     uniform PXQ2 is 4.8x the PXQ4 anchor's perplexity (30.01 vs 6.27, mean KLD
//                     1.654, same-top-token 48.8% -- a coin flip); attn4 is 1.49x (9.35, KLD
//                     0.453, 72.8%) for 10.1% more bytes and faster decode.
//   PXQ3 -> balanced  mean KLD 0.0758 against uniform's 0.1056 and same-top-token 88.833% against
//                     86.585%, for 4.2% more bytes and +15.8% / +14% decode on the two engines.
//                     Perplexity is 6.6151 vs 6.5884, i.e. INSIDE the +/-0.086 noise -- so this
//                     one is decided by the distribution agreement, not by perplexity, and it
//                     is worth knowing that the two disagree about which is better.
//   PXQ4 -> uniform   its attention is already at the cap, so every profile produces the SAME
//                     file. Nothing to choose.
//   PXQ4HQ, PXQ6      unmeasured, so unchanged. Not "safe" -- just not yet examined, which is
//                     exactly what uniform PXQ2 was until somebody measured it.
static inline pxa_pxq_policy_id pxa_pxq_policy_default_for(ggml_type base) {
    if (base == GGML_TYPE_PXQ2) return PXA_POLICY_ATTN4;
    if (base == GGML_TYPE_PXQ3) return PXA_POLICY_BALANCED;
    return PXA_POLICY_UNIFORM;
}

static inline int pxa_pxq_tier_rank(ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQ1:   return 1;
        case GGML_TYPE_PXQ2:   return 2;
        case GGML_TYPE_PXQ3:   return 3;
        case GGML_TYPE_PXQ4:   return 4;
        case GGML_TYPE_PXQ4HQ: return 5;
        case GGML_TYPE_PXQ6:   return 6;
        default:               return -1;   // not on the ladder
    }
}

static inline ggml_type pxa_pxq_tier_of_rank(int r) {
    switch (r) {
        case 1:  return GGML_TYPE_PXQ1;
        case 2:  return GGML_TYPE_PXQ2;
        case 3:  return GGML_TYPE_PXQ3;
        case 4:  return GGML_TYPE_PXQ4;
        case 5:  return GGML_TYPE_PXQ4HQ;
        case 6:  return GGML_TYPE_PXQ6;
        default: return GGML_TYPE_COUNT;
    }
}

// rank 4 == pxq4: MMVQ-admitted, the faster of the two admitted types, and the highest tier
// BOTH engines can decode. See the cap note at the top of this file.
#define PXA_PXQ_POLICY_CAP_RANK 4

// ---------------------------------------------------------------------------------------
// Role classification. Name matching only, plus ne[1] for the one case where the same name
// means two different things: attn_gate is a per-HEAD softplus vector on Laguna (ne[1] <= 256,
// pinned to f16 by the backbone table and never seen here) and a per-CHANNEL projection on
// qwen35 (5120 x 6144), which is an attention-block weight like any other.
// ---------------------------------------------------------------------------------------
static inline bool pxa_pxq_policy_name_is(const std::string & name, const char * leaf) {
    const size_t n = std::strlen(leaf);
    if (name.size() < n) {
        return false;
    }
    if (name.compare(name.size() - n, n, leaf) != 0) {
        return false;
    }
    return name.size() == n || name[name.size() - n - 1] == '.';
}

static inline pxa_pxq_policy_role pxa_pxq_policy_role_of(const std::string & name, int64_t ne1) {
    if (pxa_pxq_policy_name_is(name, "attn_output.weight")   ||
        pxa_pxq_policy_name_is(name, "attn_output_a.weight") ||
        pxa_pxq_policy_name_is(name, "attn_output_b.weight")) {
        return PXA_ROLE_ATTN_OUT;
    }
    // ssm_out is the DeltaNet block's output projection: same position in the residual stream
    // as attn_output on an attention layer, and on qwen35 the two alternate layer by layer.
    // It is ATTN rather than ATTN_OUT because the 3.2x number behind the extra notch was
    // measured on attn_output, not on this.
    if (pxa_pxq_policy_name_is(name, "attn_q.weight")   || pxa_pxq_policy_name_is(name, "attn_qkv.weight") ||
        pxa_pxq_policy_name_is(name, "attn_q_a.weight") || pxa_pxq_policy_name_is(name, "attn_q_b.weight") ||
        pxa_pxq_policy_name_is(name, "ssm_out.weight")) {
        return PXA_ROLE_ATTN;
    }
    if (pxa_pxq_policy_name_is(name, "attn_gate.weight")) {
        return ne1 > 256 ? PXA_ROLE_ATTN : PXA_ROLE_OTHER;
    }
    if (pxa_pxq_policy_name_is(name, "ffn_up.weight")   || pxa_pxq_policy_name_is(name, "ffn_gate.weight") ||
        pxa_pxq_policy_name_is(name, "ffn_down.weight") ||
        pxa_pxq_policy_name_is(name, "ffn_up_shexp.weight")   ||
        pxa_pxq_policy_name_is(name, "ffn_gate_shexp.weight") ||
        pxa_pxq_policy_name_is(name, "ffn_down_shexp.weight")) {
        return PXA_ROLE_FFN;
    }
    return PXA_ROLE_OTHER;
}

// ---------------------------------------------------------------------------------------
// THE ssm_out FLOOR (2026-09-10) — the DeltaNet output projection is never allocated below
// this type by a default recipe, on any PXQ level.
//
// MEASURED, Qwen3.8-27B (arch qwen35: 65 blocks, 48 of them DeltaNet), mean KL divergence
// against the file's own Q8_0 source, wikitext-2, -c 2048, 10 chunks, same base logits for
// every row. Everything in these files is the identical PXQ4 recipe; the ONLY thing that
// changes between rows is the type of the 48 blk.N.ssm_out.weight tensors, ~5% of the bytes:
//
//     ssm_out=pxq4    0.433     +0%        <- what every PXQ level resolved to before this
//     ssm_out=mxfp4   0.0596    -0.003%       (the pre-2026-08-25 flat landing)
//     ssm_out=pxq4hq  0.0493    +0.30%
//     ssm_out=pxq6    0.0439    +1.20%
//     ssm_out=q8_0    0.0425    +5.10%
//
// A 7x mean-KLD gap out of ONE class. ssm_out carries the whole DeltaNet block back into the
// residual stream on 48 of 65 layers, and unlike attn_output it reads a recurrent state
// rather than a softmax average, so there is no averaging in front of it to absorb the error.
// The class was claimed into the PXQ backbone on 2026-08-25 on a byte-parity argument with no
// fidelity arm behind it — the note in pxa_pxq_backbone_native_class() says so in as many
// words. This is that arm, and it says the class needs the bits.
//
// A FLOOR, NOT A PIN, and deliberately a separate layer from the profiles above: it lifts a
// PXQ-ladder landing that sits below the protected type and does nothing else. An explicit
// --custom-q rule or PXQU map entry, a geometry failure's q8_0, the MTP companion's q8_0 and
// a deliberately flat-MXFP4 backbone (PXA_PXQ_BACKBONE=legacy / =lite) all keep exactly what
// they resolved — the flat-MXFP4 backbone because that is a separately measured operating
// point, and the row above says what it costs (0.0596), not the 0.433 this floor exists for.
//
// WHY q8_0 AND NOT THE CHEAPER pxq6. The same portability rule that caps the profiles at
// pxq4: the vLLM path implements pxq2, pxq3 and pxq4 only (docs/VLLM.md, docs/LAUNCHER.md), so
// pxq6 and pxq4hq are llama.cpp-only. A PXQ4-level file is routed by its LEVEL, not by its
// contents, so 48 pxq6 tensors inside one would be handed to an engine that cannot decode
// them — a load failure with no warning in front of it. q8_0 is decodable by both, is already
// present in every PXQ file we ship (attn_k / attn_v are pinned to it), and measures the best
// of the five rows besides. It costs +5.1% instead of +1.2%; portability is worth 3.9%.
//
// ONE CONSTANT. The quantizer and tests/test-pxq-policy.cpp both read it here, so the
// shipping default and the test that pins it cannot drift apart.
#define PXA_PXQ_SSM_OUT_PROTECT GGML_TYPE_Q8_0

// Fidelity ordering for the floor ONLY: the tier ladder, plus q8_0 above all of it so the
// constant can name it. Everything else is -1, "not comparable", and the floor stands down
// rather than invent an ordering it was never given.
static inline int pxa_pxq_fidelity_rank(ggml_type t) {
    return t == GGML_TYPE_Q8_0 ? 8 : pxa_pxq_tier_rank(t);
}

// PXA_PXQ_SSM_OUT overrides the constant at run time, in the same shape PXA_PXQ_KV already
// uses for the K/V pin: a type name, or `off` to restore the pre-floor allocation for an A/B
// arm. Read once. An unknown name keeps the default and says so — silently building a file
// with an allocation nobody asked for is the failure mode this whole header exists to stop.
static inline ggml_type pxa_pxq_ssm_out_protect_type() {
    static const ggml_type t = [] {
        const char * e = getenv("PXA_PXQ_SSM_OUT");
        if (!e || !*e) {
            return (ggml_type) PXA_PXQ_SSM_OUT_PROTECT;
        }
        std::string s(e);
        for (auto & c : s) c = (char) std::tolower((unsigned char) c);
        if (s == "off" || s == "none" || s == "0") return GGML_TYPE_COUNT;   // no floor at all
        if (s == "q8_0")   return GGML_TYPE_Q8_0;
        if (s == "pxq6")   return GGML_TYPE_PXQ6;
        if (s == "pxq4hq") return GGML_TYPE_PXQ4HQ;
        if (s == "pxq4")   return GGML_TYPE_PXQ4;
        if (s == "pxq3")   return GGML_TYPE_PXQ3;
        if (s == "pxq2")   return GGML_TYPE_PXQ2;
        fprintf(stderr, "PXA_PXQ_SSM_OUT: unknown type '%s' — keeping the default floor\n", s.c_str());
        return (ggml_type) PXA_PXQ_SSM_OUT_PROTECT;
    }();
    return t;
}

static inline bool pxa_pxq_is_ssm_out(const std::string & name) {
    return pxa_pxq_policy_name_is(name, "ssm_out.weight");
}

// Applies the floor to whatever the rest of the pipeline resolved for this tensor.
//
//   resolved  the type the backbone table and the profile produced for it — or, on the write
//             loop's "an untouched MXFP4 default resolves to the whole-file tier" path, that
//             tier. Both routes reach a PXQ level for ssm_out and both are floored here.
//
// GGML_TYPE_COUNT ("leave it to the legacy pipeline"), mxfp4, q8_0 and f16 come back
// untouched: rank -1 is not a ladder landing, so there is nothing for a ladder floor to lift.
static inline ggml_type pxa_pxq_ssm_out_floor(const std::string & name, ggml_type resolved) {
    if (!pxa_pxq_is_ssm_out(name)) {
        return resolved;
    }
    const int rr = pxa_pxq_tier_rank(resolved);
    if (rr < 0) {
        return resolved;
    }
    const ggml_type prot = pxa_pxq_ssm_out_protect_type();
    return pxa_pxq_fidelity_rank(prot) > rr ? prot : resolved;
}

// ---------------------------------------------------------------------------------------
// THE RESOLVER.
//
//   base  the type BACKBONE_REV 2 resolved for this tensor (its dense branch returns the
//         named tier; its MoE branch returns the promoted backbone). Anything not on the
//         tier ladder — q8_0 K/V, q6_k token_embd, the f16 per-head gate, the q8_0 head — is
//         returned untouched: those pins are the backbone table's business, not the policy's.
//   moe   the model has routed experts, i.e. the named level refers to the expert stacks and
//         `base` is already a promotion rather than the level itself.
//
// Never demotes: the result is floored at `base` in every branch. That matters because a
// PXQ6 dense build arrives here with base == pxq6, above the cap, and must come back as pxq6.
// ---------------------------------------------------------------------------------------
static inline ggml_type pxa_pxq_policy_apply(const std::string & name, int64_t ne1,
                                             ggml_type base, bool moe, pxa_pxq_policy_id p) {
    if (p == PXA_POLICY_UNIFORM) {
        return base;
    }
    const int base_rank = pxa_pxq_tier_rank(base);
    if (base_rank < 0) {
        return base;   // an explicit non-ladder pin; not ours to move
    }
    const pxa_pxq_policy_role role = pxa_pxq_policy_role_of(name, ne1);
    if (role == PXA_ROLE_OTHER) {
        return base;
    }

    int want;
    if (moe) {
        // On a MoE model `base` is ALREADY the promoted always-resident backbone; the level
        // named the experts. One flat rule for every level: the whole backbone at pxq4, which
        // is MMVQ-admitted, byte-parity with the MXFP4 it replaced, and readable by both
        // engines. attn_output does not get its own notch here any more — the cap took it, and
        // the portability half of the cap is not negotiable for a default-shaped profile.
        want = 4;
    } else if (p == PXA_POLICY_ATTN4) {
        // The MoE rule, verbatim, on a dense model: attention lands at pxq4 no matter how low
        // the FFN goes.
        want = role == PXA_ROLE_FFN ? base_rank : 4;
    } else {
        // balanced: relative to the level, so the file keeps the shape of the tier it names.
        want = role == PXA_ROLE_FFN ? base_rank
             : role == PXA_ROLE_ATTN_OUT ? base_rank + 2 : base_rank + 1;
    }

    if (want > PXA_PXQ_POLICY_CAP_RANK) {
        want = PXA_PXQ_POLICY_CAP_RANK;
    }
    // NEVER DEMOTE — but only on a DENSE model, and the asymmetry is the whole difference
    // between the two shapes. On dense, `base` IS the level the file is named after, so a
    // result below it would produce a PXQ4 file whose attention is pxq3: the name would be a
    // lie, and the composition assertion exists precisely to catch that class of thing. On a
    // MoE model `base` is not the level at all — it is rev 2's own guess at a backbone, and
    // replacing that guess is the entire job. Pulling PXQ3's all-pxq4hq backbone down to pxq4
    // is a deliberate DEMOTION: 4.2526 bpw instead of 4.52, on the type the sm_70 campaign
    // measured ~9% faster, with the extra bits kept where the error actually is (attn_output).
    // The floor is not needed to protect a pxq6 MoE backbone either — that only exists under
    // PXA_PXQ_BACKBONE=pxq6, and pxa_pxq_policy_active() stands the profile down for it.
    if (!moe && want < base_rank) {
        want = base_rank;
    }
    const ggml_type out = pxa_pxq_tier_of_rank(want);
    return out == GGML_TYPE_COUNT ? base : out;
}
