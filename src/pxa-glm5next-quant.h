#pragma once

// ------------------------------------------------------------------------------------------
// GLM-5.3-Flash (LLM_ARCH_GLM5NEXT) quantiser keep-list.
//
// Header-only and free of every llama.cpp type on purpose: llama-quantize.cpp uses it inside
// the write loop, and tests/test-glm5next-quant-rules.cpp exercises it against the real
// tensor inventory of the unsloth UD-Q2_K_XL GGUF with no model, no GPU and no weights.
//
// WHY A KEEP-LIST AND NOT A TYPE PIN. Every tensor named here is either (a) something whose
// error is not a small numeric perturbation of an activation — the indexer CHOOSES which
// tokens are attended to, the mHC mixers rescale an entire residual stream, the KDA gates
// multiply a recurrent state that never resets — or (b) already smaller at its source type
// than anything we would requantise it to. Copying it through verbatim is both the safest and
// the cheapest answer, and it costs ~2.3 GiB of a ~86 GiB artifact.
//
// Sources: upstream PR #27773 src/llama-quant.cpp (the LLM_ARCH_GLM5_NEXT block), plus this
// tree's own DeepSeek-V4 backbone classes. Verified against the real tensor names taken from
// the GGUF header dump of shard 2.
// ------------------------------------------------------------------------------------------

#include <string>

// true when the tensor is a routed-expert stack (the part a PXQU .tiers map owns)
inline bool pxa_glm5next_is_expert(const std::string & name) {
    return name.size() >= 12 && name.compare(name.size() - 12, 12, "_exps.weight") == 0;
}

// n_trunk = n_layer - nextn_predict_layers, i.e. the first NextN/MTP block index.
// Pass a negative n_trunk to disable the NextN rule (e.g. when quantising WITH --mtp in mind).
inline bool pxa_glm5next_keep_at_source(const std::string & name, int n_trunk) {
    auto has = [&name](const char * s) { return name.find(s) != std::string::npos; };

    // ---- the NextN / MTP tail block ---------------------------------------------------
    // Our loader marks every tensor of a block >= n_trunk TENSOR_SKIP unless the context asks
    // for --mtp (src/llama-load-tensors.cpp, create_glm5next_tensors), so this block costs
    // disk and nothing else. Requantising it would spend hours and a second lossy pass on
    // weights the graph never reads, and would foreclose enabling MTP later without a redo.
    // Copy it through.
    if (n_trunk >= 0 && name.compare(0, 4, "blk.") == 0) {
        int il = 0; size_t p = 4;
        if (p < name.size() && name[p] >= '0' && name[p] <= '9') {
            while (p < name.size() && name[p] >= '0' && name[p] <= '9') { il = il*10 + (name[p]-'0'); ++p; }
            if (p < name.size() && name[p] == '.' && il >= n_trunk) {
                return true;
            }
        }
    }

    // ---- the head and the embeddings ---------------------------------------------------
    // Not a fidelity call: token_embd is Q5_K and output is Q4_K in the unsloth source, while
    // this tree's PXQ rules would put them at Q6_K and Q8_0. Requantising them here is BIGGER
    // (+0.6 GiB) *and* lossy. token_embd is a row gather and output is one GEMM per token, so
    // neither is on the Pascal k-quant-is-slow path that motivates moving the backbone to PXQ.
    if (name == "token_embd.weight" || name == "output.weight") return true;

    // ---- norms (also covered by the generic rule, kept here so the test sees one list) ----
    if (has("_norm.weight") || has("_norm.bias") || name == "output_norm.weight") return true;

    // ---- DSA lightning indexer: its output is a top-k SELECTION, not a number -----------
    if (has(".indexer."))              return true;   // attn_q_b, attn_k, proj, k_norm.{weight,bias}
    if (has("indexer_compressor_"))    return true;   // _gate, _ape

    // ---- manifold-constrained hyper-connections: each rescales a whole residual stream ---
    if (has("hc_attn_") || has("hc_ffn_") || has("output_hc_")) return true;

    // ---- KDA (Kimi Delta Attention) gates and the recurrent-state machinery --------------
    // ssm_a is [64] and carries no ".weight"; ssm_dt.bias is [8192]; the f/g low-rank pairs
    // and ssm_beta drive a state that persists across the whole sequence.
    if (has("ssm_a") || has("ssm_dt.") || has("ssm_norm") || has("ssm_conv1d") ||
        has("ssm_f_a") || has("ssm_f_b") || has("ssm_g_a") || has("ssm_g_b") ||
        has("ssm_beta")) return true;

    // ---- MLA low-rank absorb pair + the compressed KV projection -------------------------
    if (has("attn_k_b.weight") || has("attn_v_b.weight") || has("attn_kv_a_mqa.weight")) return true;

    // ---- the MLA q-LoRA pair -------------------------------------------------------------
    // Upstream #27773 refuses to take attn_q_a / attn_q_b below Q8_0 for this arch. In the
    // unsloth file attn_q_b is ALREADY Q8_0, so requantising it would be a second lossy pass
    // (and would need --i-know-this-is-double-lossy) to save 132 MiB of an ~86 GiB artifact;
    // attn_q_a is Q5_K and saves 9 MiB. Neither is worth the error. Copy both.
    // NOTE this also covers "indexer.attn_q_b", which the indexer rule above already caught.
    if (has("attn_q_a.weight") || has("attn_q_b.weight")) return true;

    // ---- MoE routing: a router error is the wrong expert, not a small one ----------------
    if (has("ffn_gate_inp.weight") || has("exp_probs_b")) return true;

    return false;
}
