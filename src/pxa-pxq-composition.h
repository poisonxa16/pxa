#pragma once

// ------------------------------------------------------------------------------------------
// PXQ composition floor (llama-quantize.cpp, the PXQ-P5 assertion) - the accounting, as pure code.
//
// Header-only and free of every llama.cpp / ggml type on purpose: llama-quantize.cpp feeds it the
// tensors a run lands, and tests/test-pxa-composition-floor.cpp feeds it a fake tensor list with
// no model, no weights and no GPU.
//
// WHAT THE FLOOR IS FOR. A PXQ-family target (PXQ1/2/3/4/4HQ/6/UNIVERSAL) must not write a file
// whose NAME misrepresents its contents: a dense model quantized with the legacy backbone once
// emitted 91% MXFP4 / 0% PXQ, exit 0. So the run is refused when the PXQ-family tiers are less
// than half of the output's bytes, or when a UNIFORM target contributed none of its named tier.
//
// WHAT IT USED TO GET WRONG (todo encoder-composition-floor-ple, 2026-09-30). The share was taken
// over EVERY output byte. A model with host-side gather tables - Flash-Next's per_layer_token_embd
// is 51.15 GiB of q8_0 that can never be a panel codec - puts bytes in the denominator that no
// card ever holds: the swift-fn48c encode landed at 42.8% PXQ, was refused AFTER 1h45m of encoding
// and the finished file was deleted. The floor is a statement about the weights that live on the
// cards, so it now counts only RESIDENT weight tensors: the host-side tables are excluded from the
// share (and listed, with their bytes, in the log), and the check runs on the projected landing
// BEFORE the first tensor is encoded. A run that fails it has written nothing; a finished file is
// never deleted by it.
//
// WHICH TENSORS THE MODEL PUTS ON THE HOST. src/llama-load-tensors.cpp creates these in ctx_input
// (buft_input = the CPU buffer type, "very little benefit to offloading the input layer"):
//   token_embd.weight         the row-gather input table (a TIED model also reads it as the output
//                             head, which IS on the cards: with no output.weight it is resident)
//   per_layer_token_embd      the PLE / n-gram gather tables (gemma3n / gemma4 / qwen4exp)
//   position_embd.weight, token_types.weight, *attn_rel_b.weight   (BERT / T5 input tables)
// plus the structural rule llama-quantize.cpp already uses for its row-gather gate: a table with
// >= 1,000,000 rows is indexed by a hash or a token id, never multiplied.
// ------------------------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace pxa_comp {

// the floor: PXQ-family bytes must be at least this share of the resident bytes
static constexpr double FLOOR = 0.50;

// A tensor as the run lands it. type is the ggml type id of the OUTPUT tensor, bytes its size there.
struct tensor {
    std::string name;
    int64_t     ne1   = 0;     // rows (ne[1]); 0 for a 1-D tensor
    int         type  = -1;
    size_t      bytes = 0;
};

// PXA_PXQ_COMPOSITION_OVERRIDE / --pxq-composition-override: any non-zero integer downgrades the
// refusal to a warning. Unset, empty, "0" and non-numeric text leave the floor armed.
inline bool override_requested(const char * env) {
    return env != nullptr && std::atoi(env) != 0;
}

inline bool name_ends(const std::string & s, const char * suf) {
    const size_t n = std::char_traits<char>::length(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// True for a tensor the model keeps in host memory (see the header comment). `has_output_weight` is
// whether the source file carries a separate output.weight: without one the embedding table is TIED
// and also serves as the output head, which is resident.
inline bool is_host_side_table(const std::string & name, int64_t ne1, bool has_output_weight) {
    if (name.find("per_layer_token_embd") != std::string::npos) {
        return true;
    }
    if (name == "token_embd.weight") {
        return has_output_weight;
    }
    if (name == "position_embd.weight" || name == "token_types.weight" || name_ends(name, "attn_rel_b.weight")) {
        return true;
    }
    // structural: a row count this large only happens for a hash/id-indexed table (never the head)
    return name != "output.weight" && ne1 >= 1000000;
}

struct result {
    size_t total_bytes          = 0;   // every tensor
    size_t host_bytes           = 0;   // the excluded host-side tables
    size_t resident_bytes       = 0;   // the denominator of the floor
    size_t resident_family_bytes = 0;  // PXQ-family bytes among the resident tensors
    size_t named_tier_bytes     = 0;   // bytes of the uniform target's own tier (all tensors)
    int    n_host               = 0;
    int    n_resident           = 0;
    double share_all            = 0.0; // the OLD measure: family bytes / all bytes (reported, not judged)
    double share                = 0.0; // resident family bytes / resident bytes (judged)
    bool   below_floor          = false;
    bool   tier_absent          = false;
    bool   fails() const { return below_floor || tier_absent; }
    std::vector<std::string> host_names;   // the excluded tensors, in input order
};

// family: the ggml type ids that count as PXQ. spec_type: the one tier a UNIFORM PXQ target names
// (-1 = none, a map-defined mix). `floor` is exposed for tests; callers pass FLOOR.
inline result evaluate(const std::vector<tensor> & tensors, const std::set<int> & family, int spec_type,
                       bool has_output_weight, double floor = FLOOR) {
    result r;
    size_t family_all = 0;
    for (const tensor & t : tensors) {
        r.total_bytes += t.bytes;
        const bool fam = family.count(t.type) != 0;
        if (fam) {
            family_all += t.bytes;
        }
        if (spec_type >= 0 && t.type == spec_type) {
            r.named_tier_bytes += t.bytes;
        }
        if (is_host_side_table(t.name, t.ne1, has_output_weight)) {
            r.host_bytes += t.bytes;
            ++r.n_host;
            r.host_names.push_back(t.name);
            continue;
        }
        r.resident_bytes += t.bytes;
        ++r.n_resident;
        if (fam) {
            r.resident_family_bytes += t.bytes;
        }
    }
    r.share_all = r.total_bytes    ? (double) family_all / (double) r.total_bytes : 0.0;
    r.share     = r.resident_bytes ? (double) r.resident_family_bytes / (double) r.resident_bytes : 0.0;
    // nothing resident to judge (a model of host tables only) is not a mislabelled file
    r.below_floor = r.resident_bytes > 0 && r.share < floor;
    r.tier_absent = spec_type >= 0 && r.named_tier_bytes == 0;
    return r;
}

} // namespace pxa_comp
