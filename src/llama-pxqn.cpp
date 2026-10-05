#include "llama-pxqn.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static const char * const k_pxqn_site_names[PXQN_SITE_COUNT] = {
    "none", "attn_in", "ffn_in", "down_in", "out_in",
};

const char * llama_pxqn_site_name(int site) {
    return site >= 0 && site < PXQN_SITE_COUNT ? k_pxqn_site_names[site] : "?";
}

uint32_t llama_pxqn_parse_sites(const std::string & s) {
    uint32_t mask = 0;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) end = s.size();
        std::string tok = s.substr(pos, end - pos);
        // trim
        while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '\t')) tok.pop_back();
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(tok.begin());
        if (!tok.empty()) {
            int found = -1;
            for (int i = 1; i < PXQN_SITE_COUNT; ++i) {
                if (tok == k_pxqn_site_names[i]) { found = i; break; }
            }
            if (found < 0) {
                throw std::runtime_error("PXQN: unknown rotation site '" + tok + "' in " LLAMA_PXQN_KEY_ROT_SITES
                                         " (this build knows attn_in, ffn_in, down_in, out_in)");
            }
            mask |= PXQN_SITE_BIT(found);
        }
        pos = end + 1;
    }
    return mask;
}

std::string llama_pxqn_sites_str(uint32_t sites) {
    std::string out;
    for (int i = 1; i < PXQN_SITE_COUNT; ++i) {
        if (sites & PXQN_SITE_BIT(i)) {
            if (!out.empty()) out += ",";
            out += k_pxqn_site_names[i];
        }
    }
    return out;
}

void llama_pxqn_read_keys(const struct gguf_context * ctx, uint32_t & rev, uint64_t & seed, uint32_t & sites) {
    rev = 0; seed = 0; sites = 0;
    const int kr = gguf_find_key(ctx, LLAMA_PXQN_KEY_REV);
    if (kr < 0) {
        return;   // not a PXQN file
    }
    if (gguf_get_kv_type(ctx, kr) != GGUF_TYPE_UINT32) {
        throw std::runtime_error("PXQN: " LLAMA_PXQN_KEY_REV " is not a u32 -- malformed file");
    }
    rev = gguf_get_val_u32(ctx, kr);
    if (rev == 0) {
        throw std::runtime_error("PXQN: " LLAMA_PXQN_KEY_REV " = 0 -- malformed file (revisions start at 1)");
    }
    if (rev > PXQN_REV) {
        char buf[256];
        snprintf(buf, sizeof(buf), "PXQN: this file is PXQ-Next revision %u, and this build reads revision %u only. "
                 "Update the engine to a build that reads revision %u.", rev, PXQN_REV, rev);
        throw std::runtime_error(buf);
    }
    const int ks = gguf_find_key(ctx, LLAMA_PXQN_KEY_ROT_SEED);
    if (ks >= 0) {
        if (gguf_get_kv_type(ctx, ks) != GGUF_TYPE_UINT64) {
            throw std::runtime_error("PXQN: " LLAMA_PXQN_KEY_ROT_SEED " is not a u64 -- malformed file");
        }
        seed = gguf_get_val_u64(ctx, ks);
    }
    const int kt = gguf_find_key(ctx, LLAMA_PXQN_KEY_ROT_SITES);
    if (kt >= 0) {
        if (gguf_get_kv_type(ctx, kt) != GGUF_TYPE_STRING) {
            throw std::runtime_error("PXQN: " LLAMA_PXQN_KEY_ROT_SITES " is not a string -- malformed file");
        }
        sites = llama_pxqn_parse_sites(gguf_get_val_str(ctx, kt));
    }
    if (sites != 0 && ks < 0) {
        throw std::runtime_error("PXQN: " LLAMA_PXQN_KEY_ROT_SITES " names rotated sites but the file has no "
                                 LLAMA_PXQN_KEY_ROT_SEED " -- malformed file");
    }
}

// "blk.<N>.<rest>" -> N and <rest> ("" / -1 when not a block tensor)
static int pxqn_split_blk(const std::string & name, std::string & rest) {
    rest.clear();
    if (name.compare(0, 4, "blk.") != 0) return -1;
    size_t i = 4;
    int n = 0;
    bool any = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        n = n*10 + (name[i] - '0');
        ++i; any = true;
    }
    if (!any || i >= name.size() || name[i] != '.') return -1;
    rest = name.substr(i + 1);
    return n;
}

int llama_pxqn_tensor_layer(const std::string & tensor_name) {
    std::string rest;
    return pxqn_split_blk(tensor_name, rest);
}

int llama_pxqn_tensor_site(const std::string & tensor_name) {
    std::string rest;
    if (pxqn_split_blk(tensor_name, rest) < 0) return PXQN_SITE_NONE;
    // strip the ".weight" / ".bias" suffix: a bias is added AFTER the matmul and is never rotated
    const size_t dot = rest.rfind('.');
    if (dot == std::string::npos) return PXQN_SITE_NONE;
    const std::string suffix = rest.substr(dot + 1);
    const std::string base   = rest.substr(0, dot);
    if (suffix != "weight") return PXQN_SITE_NONE;

    struct { const char * name; int site; } static const k_map[] = {
        // attn_norm output
        { "attn_qkv",       PXQN_SITE_ATTN_IN },
        { "attn_gate",      PXQN_SITE_ATTN_IN },
        { "ssm_alpha",      PXQN_SITE_ATTN_IN },
        { "ssm_beta",       PXQN_SITE_ATTN_IN },
        { "ssm_ba",         PXQN_SITE_ATTN_IN },
        { "ssm_in",         PXQN_SITE_ATTN_IN },
        { "attn_q",         PXQN_SITE_ATTN_IN },
        { "attn_k",         PXQN_SITE_ATTN_IN },
        { "attn_v",         PXQN_SITE_ATTN_IN },
        // post_attention_norm output
        { "ffn_gate",       PXQN_SITE_FFN_IN  },
        { "ffn_up",         PXQN_SITE_FFN_IN  },
        { "ffn_gate_inp",   PXQN_SITE_FFN_IN  },
        { "ffn_gate_exps",  PXQN_SITE_FFN_IN  },
        { "ffn_up_exps",    PXQN_SITE_FFN_IN  },
        { "ffn_gate_shexp", PXQN_SITE_FFN_IN  },
        { "ffn_up_shexp",   PXQN_SITE_FFN_IN  },
        { "ffn_gate_inp_shexp", PXQN_SITE_FFN_IN },
        // SwiGLU output
        { "ffn_down",       PXQN_SITE_DOWN_IN },
        { "ffn_down_exps",  PXQN_SITE_DOWN_IN },
        { "ffn_down_shexp", PXQN_SITE_DOWN_IN },
        // DeltaNet gated-norm output / attention output
        { "ssm_out",        PXQN_SITE_OUT_IN  },
        { "attn_output",    PXQN_SITE_OUT_IN  },
    };
    for (const auto & e : k_map) {
        if (base == e.name) return e.site;
    }
    return PXQN_SITE_NONE;   // nextn.eh_proj, norms, conv, ... are not sites
}

// ---------------------------------------------------------------------------------------------------
// Which architectures this build rotates (see the header). The sites are wired in the shared graph
// builders: attn_in after the attention norm (build_std_attention, or pxqn_rot_in() in a builder that
// writes its own attention), out_in before the attention output matmul (llm_build_kqv / the std
// attention paths / the DeltaNet layer), ffn_in after the FFN norm and down_in before the down matmul
// (llm_build_ffn, llm_build_moe_ffn / llm_build_std_moe_ffn for routed experts). A family is listed only
// when every consumer of every site it names goes through those helpers.
// ---------------------------------------------------------------------------------------------------
uint32_t llama_pxqn_arch_rot_sites(llm_arch arch, bool moe) {
    constexpr uint32_t ATTN = PXQN_SITE_BIT(PXQN_SITE_ATTN_IN) | PXQN_SITE_BIT(PXQN_SITE_OUT_IN);
    constexpr uint32_t FFN  = PXQN_SITE_BIT(PXQN_SITE_FFN_IN)  | PXQN_SITE_BIT(PXQN_SITE_DOWN_IN);
    switch (arch) {
        // the Qwen3.5 / 3.8 family: DeltaNet + gated attention; its MoE variant rotates the attention sites only
        case LLM_ARCH_QWEN35:     return ATTN | FFN;
        case LLM_ARCH_QWEN35MOE:  return ATTN;
        // plain dense transformers (shared build_std_attention / llm_build_kqv / llm_build_ffn); a model with
        // routed experts (Mixtral-style llama, Qwen3 MoE, Gemma 4 MoE) takes the attention sites only
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_GEMMA4_MTP:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_MISTRAL3:   return moe ? ATTN : (ATTN | FFN);
        default:                  return 0;
    }
}
