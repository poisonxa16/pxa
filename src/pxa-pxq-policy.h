#pragma once
//
// PXQ POLICY NAMES AND THE ssm_out FLOOR. The profile NAMES (so a build that cannot write PXQ
// files can still parse and report them), the tier ladder rank, and the ssm_out fidelity floor.
// The allocation rule itself is the encoder's and lives in src/pxq-encoder/pxa-pxq-policy-mix.h.
//
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

