// PXA_GLM5NEXT: the llama-quantize keep-list for GLM-5.3-Flash, exercised against the REAL
// tensor inventory of the unsloth UD-Q2_K_XL GGUF. CPU only, no model, no weights, no GPU.
//
// Why this test exists. The keep-list decides, per tensor, between "copy through at source
// precision" and "hand to the codec". Getting it wrong in the quiet direction — quantizing
// the lightning indexer, an mHC mixer, or a KDA gate — produces a file that loads, runs and
// is subtly wrong, and the only way to notice is a quality sweep on a 86 GiB artifact after a
// multi-hour job. The names below are transcribed from the header of the real file
// (from the shard-2 GGUF header dump), so a rename upstream shows up here.
//
// It also pins the two size-driven decisions that are easy to "tidy" later:
//   * token_embd (Q5_K) and output (Q4_K) are KEPT — the PXQ rules would put them at Q6_K and
//     Q8_0, which is bigger AND lossy.
//   * the NextN block (>= n_trunk) is KEPT whole — our loader marks it TENSOR_SKIP without
//     --mtp, so requantizing it spends hours on weights the graph never reads.

#include "pxa-glm5next-quant.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(const std::string & name, int n_trunk, bool want_keep, const char * why) {
    const bool got = pxa_glm5next_keep_at_source(name, n_trunk);
    if (got != want_keep) {
        printf("  FAIL %-46s expected %s, got %s   (%s)\n", name.c_str(),
               want_keep ? "KEEP" : "QUANT", got ? "KEEP" : "QUANT", why);
        ++g_fail;
    }
}

int main() {
    // GLM-5.3-Flash: 46 blocks, 1 NextN predict layer -> trunk is 0..44, block 45 is the MTP head.
    const int n_trunk = 46 - 1;

    printf("glm5next quantiser keep-list (n_trunk = %d)\n", n_trunk);

    // ---- KEEP: the precision-sensitive set (upstream #27773 leaves the same set alone) ----
    const char * keep[] = {
        // DSA lightning indexer — its output is a top-k SELECTION of tokens, not a number
        "blk.3.indexer.attn_k.weight",
        "blk.3.indexer.attn_q_b.weight",
        "blk.3.indexer.proj.weight",
        "blk.3.indexer.k_norm.weight",
        "blk.3.indexer.k_norm.bias",
        "blk.43.indexer_compressor_gate.weight",
        "blk.43.indexer_compressor_ape.weight",
        // manifold-constrained hyper-connections — each rescales a whole residual stream
        "blk.0.hc_attn_fn.weight", "blk.0.hc_attn_base.weight", "blk.0.hc_attn_scale.weight",
        "blk.0.hc_ffn_fn.weight",  "blk.0.hc_ffn_base.weight",  "blk.0.hc_ffn_scale.weight",
        // KDA gates + the recurrent-state machinery
        "blk.4.ssm_a", "blk.4.ssm_dt.bias", "blk.4.ssm_norm.weight",
        "blk.4.ssm_conv1d_q.weight", "blk.4.ssm_conv1d_k.weight", "blk.4.ssm_conv1d_v.weight",
        "blk.4.ssm_f_a.weight", "blk.4.ssm_f_b.weight",
        "blk.4.ssm_g_a.weight", "blk.4.ssm_g_b.weight",
        "blk.4.ssm_beta.weight",
        // MLA low-rank absorb pair + the compressed KV projection
        "blk.3.attn_k_b.weight", "blk.3.attn_v_b.weight", "blk.3.attn_kv_a_mqa.weight",
        // the MLA q-LoRA pair: upstream's >=Q8_0 floor, and attn_q_b is already Q8_0 in the
        // source, so requantising it would be a second lossy pass for 132 MiB
        "blk.3.attn_q_a.weight", "blk.3.attn_q_b.weight",
        // routing
        "blk.3.ffn_gate_inp.weight", "blk.3.exp_probs_b.bias",
        // norms
        "blk.3.attn_norm.weight", "blk.3.ffn_norm.weight",
        "blk.3.attn_q_a_norm.weight", "blk.3.attn_kv_a_norm.weight", "output_norm.weight",
        // head + embeddings: source Q5_K/Q4_K is both smaller and lossless vs our PXQ rules
        "token_embd.weight", "output.weight",
    };
    for (const char * n : keep) check(n, n_trunk, true, "precision-sensitive / cheaper at source");

    // ---- KEEP: the whole NextN block, whatever it is called ----
    const char * nextn[] = {
        "blk.45.attn_q_a.weight", "blk.45.ffn_gate_exps.weight", "blk.45.ffn_down_exps.weight",
        "blk.45.ffn_up_exps.weight", "blk.45.attn_output.weight",
        "blk.45.nextn.eh_proj.weight", "blk.45.nextn.shared_head_head.weight",
        "blk.45.nextn.embed_tokens.weight",
    };
    for (const char * n : nextn) check(n, n_trunk, true, "NextN block: TENSOR_SKIP without --mtp");

    // ---- QUANT: everything the tier map and the PXQ backbone own ----
    const char * quant[] = {
        // routed experts: the .tiers map owns these
        "blk.3.ffn_gate_exps.weight", "blk.3.ffn_up_exps.weight", "blk.3.ffn_down_exps.weight",
        "blk.44.ffn_down_exps.weight",
        // shared expert + the 3 dense blocks
        "blk.3.ffn_gate_shexp.weight", "blk.3.ffn_up_shexp.weight", "blk.3.ffn_down_shexp.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        // KDA attention projections and the MLA q pair
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
    };
    for (const char * n : quant) check(n, n_trunk, false, "owned by the tier map / PXQ backbone");

    // ---- the near-miss pairs the substring matcher must not confuse ----
    // attn_k_b / attn_v_b are KEPT, attn_k / attn_v are NOT; attn_q_a/attn_q_b are KEPT,
    // plain attn_q is NOT.
    check("blk.3.indexer.attn_q_b.weight", n_trunk, true,  "indexer q_b IS kept");
    check("blk.3.attn_q.weight",           n_trunk, false, "attn_q is not attn_q_a/attn_q_b");
    check("blk.0.attn_k.weight",           n_trunk, false, "attn_k is not attn_k_b");
    check("blk.3.attn_k_b.weight",         n_trunk, true,  "attn_k_b IS kept");

    // ---- the NextN rule must key on the block index, not on a substring ----
    check("blk.4.ffn_gate_exps.weight",  n_trunk, false, "block 4 is trunk");
    check("blk.45.ffn_gate_exps.weight", n_trunk, true,  "block 45 is NextN");
    // n_trunk < 0 disables the NextN rule (quantising a file we intend to run with --mtp)
    check("blk.45.ffn_gate_exps.weight", -1, false, "NextN rule disabled");

    // ---- expert detector ----
    struct { const char * n; bool e; } exp_cases[] = {
        {"blk.3.ffn_gate_exps.weight", true}, {"blk.3.ffn_down_exps.weight", true},
        {"blk.3.ffn_gate_shexp.weight", false}, {"blk.3.ffn_gate_inp.weight", false},
        {"exps.weight", false},   // 11 chars: not the "_exps.weight" suffix
        {"x_exps.weight", true}, {"weight", false},
    };
    for (const auto & c : exp_cases) {
        if (pxa_glm5next_is_expert(c.n) != c.e) {
            printf("  FAIL is_expert(%s) != %d\n", c.n, (int) c.e);
            ++g_fail;
        }
    }

    if (g_fail) { printf("FAILED: %d case(s)\n", g_fail); return 1; }
    printf("ALL OK\n");
    return 0;
}
