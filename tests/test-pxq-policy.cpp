// POLICY_REV 3 resolver: tensor name -> tier, pinned on the real tensor inventories of both
// campaign test models. CPU only, no model, no weights, no GPU, no llama link.
//
// WHY THIS TEST EXISTS. The policy layer is a name-matching rule that decides how many bits a
// class of weights gets. Every failure mode it has is silent: a leaf-name typo makes a class
// fall through to PXA_ROLE_OTHER and quietly keep the FFN's tier, an over-broad match sweeps a
// class it was never meant to touch, and either way the output file loads, runs, and is only
// wrong in the answers. The names below are transcribed from the actual GGUF headers
// (Qwen3.8-27B-Q8_0.gguf, 866 tensors; Qwen3.8-Flash-Next-Uncensored-PXQU-4xP100.gguf), so a
// rename in either converter shows up here rather than in a 10 GB artifact.
//
// It also pins the three decisions that are easy to "tidy" later and expensive to get wrong:
//   * the pxq4 CAP — no profile may EMIT pxq4hq, which only llama.cpp can decode;
//   * the NEVER-DEMOTE floor — a PXQ6 dense build must come back pxq6, not capped down;
//   * uniform is BYTE-IDENTICAL to BACKBONE_REV 2 — every published file stays reproducible.

#include "pxa-pxq-policy.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;

static void chk(const char * name, int64_t ne1, ggml_type base, bool moe,
                pxa_pxq_policy_id p, ggml_type want, const char * why) {
    ++g_checks;
    const ggml_type got = pxa_pxq_policy_apply(name, ne1, base, moe, p);
    if (got != want) {
        printf("  FAIL %-8s %-34s base=%-7s -> %-7s, expected %-7s   (%s)\n",
               pxa_pxq_policy_name(p), name, ggml_type_name(base),
               ggml_type_name(got), ggml_type_name(want), why);
        ++g_fail;
    }
}

int main() {
    printf("PXQ policy resolver (POLICY_REV 3)\n");

    // =====================================================================================
    // 1. Qwen3.8-27B — arch qwen35, DENSE, 65 blocks. 48 of them are DeltaNet layers
    //    (attn_qkv 5120x10240 in-proj, attn_gate 5120x6144 per-CHANNEL z-gate, ssm_out
    //    6144x5120 out-proj) and 17 are full attention (attn_q 5120x12288, attn_k/v
    //    5120x1024, attn_output 6144x5120). Under BACKBONE_REV 2 every one of these arrives
    //    with base == the named tier, which is the whole point of the exercise.
    // =====================================================================================
    printf("\n-- Qwen3.8-27B (qwen35, dense, hybrid DeltaNet/attention) --\n");

    // uniform: nothing moves, at any level. This is the reproducibility guarantee.
    for (const char * n : { "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
                            "blk.0.ssm_out.weight",  "blk.3.attn_q.weight",
                            "blk.3.attn_output.weight", "blk.0.ffn_up.weight",
                            "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight" }) {
        chk(n, 6144, GGML_TYPE_PXQ2, false, PXA_POLICY_UNIFORM, GGML_TYPE_PXQ2, "uniform never moves anything");
        chk(n, 6144, GGML_TYPE_PXQ3, false, PXA_POLICY_UNIFORM, GGML_TYPE_PXQ3, "uniform never moves anything");
        chk(n, 6144, GGML_TYPE_PXQ4, false, PXA_POLICY_UNIFORM, GGML_TYPE_PXQ4, "uniform never moves anything");
    }

    // balanced @ PXQ2: FFN stays pxq2, attention +1 (pxq3), attn_output +2 (pxq4).
    chk("blk.0.ffn_up.weight",       17408, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ2,   "FFN is the bulk; the level names it");
    chk("blk.0.ffn_gate.weight",     17408, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ2,   "FFN is the bulk");
    chk("blk.0.ffn_down.weight",      5120, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ2,   "FFN is the bulk");
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ3,   "DeltaNet in-proj is attention");
    chk("blk.3.attn_q.weight",       12288, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ3,   "attention +1");
    chk("blk.0.attn_gate.weight",     6144, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ3,   "per-CHANNEL gate is attention");
    chk("blk.0.ssm_out.weight",       5120, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ3,   "DeltaNet out-proj is attention");
    chk("blk.3.attn_output.weight",   5120, GGML_TYPE_PXQ2, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "+2 from pxq2 lands on the cap");

    // balanced @ PXQ3: attention pxq4, and the +2 on attn_output lands on the same cap.
    chk("blk.0.ffn_down.weight",      5120, GGML_TYPE_PXQ3, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ3,   "FFN at the level");
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ3, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "attention +1");
    chk("blk.3.attn_output.weight",   5120, GGML_TYPE_PXQ3, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "CAP: pxq4hq is llama.cpp-only, no vLLM decoder");

    // balanced @ PXQ4: the cap == the level, so the whole profile is a no-op here.
    chk("blk.0.ffn_down.weight",      5120, GGML_TYPE_PXQ4, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "FFN at the level");
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ4, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "CAP: no promotion above pxq4, so PXQ4 is a no-op");
    chk("blk.3.attn_output.weight",   5120, GGML_TYPE_PXQ4, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "CAP: pxq4 is the ceiling both engines read");

    // attn4 @ PXQ2 and @ PXQ3 give the SAME attention allocation — that is the profile's point.
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ2, false, PXA_POLICY_ATTN4, GGML_TYPE_PXQ4,   "attn4 is level-independent");
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ3, false, PXA_POLICY_ATTN4, GGML_TYPE_PXQ4,   "attn4 is level-independent");
    chk("blk.3.attn_output.weight",   5120, GGML_TYPE_PXQ2, false, PXA_POLICY_ATTN4, GGML_TYPE_PXQ4, "attn4 out-proj is pxq4");
    chk("blk.3.attn_output.weight",   5120, GGML_TYPE_PXQ3, false, PXA_POLICY_ATTN4, GGML_TYPE_PXQ4, "attn4 out-proj is pxq4");
    chk("blk.0.ffn_up.weight",       17408, GGML_TYPE_PXQ2, false, PXA_POLICY_ATTN4, GGML_TYPE_PXQ2,   "attn4 leaves the FFN alone");
    chk("blk.0.ffn_up.weight",       17408, GGML_TYPE_PXQ3, false, PXA_POLICY_ATTN4, GGML_TYPE_PXQ3,   "attn4 leaves the FFN alone");

    // NEVER DEMOTE. A PXQ6 dense build arrives above the cap and must come back untouched;
    // a pxq4hq FFN likewise. Capping these down would silently shrink a quality build.
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ6,   false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ6,   "floor: never demote below the level");
    chk("blk.3.attn_output.weight",   5120, GGML_TYPE_PXQ6,   false, PXA_POLICY_ATTN4,    GGML_TYPE_PXQ6,   "floor beats the attn4 target");
    chk("blk.0.ffn_up.weight",       17408, GGML_TYPE_PXQ6,   false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ6,   "FFN keeps the level");
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ4HQ, false, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4HQ, "floor: an explicit pxq4hq level is never pulled DOWN to the cap");

    // NON-LADDER PINS ARE NOT OURS. K/V (q8_0), token_embd (q6_k), the head (q8_0) and the
    // per-HEAD gate (f16) are decided by the backbone table; the policy must return them
    // verbatim or it would undo four separately-measured decisions.
    chk("blk.3.attn_k.weight",        1024, GGML_TYPE_Q8_0, false, PXA_POLICY_ATTN4,    GGML_TYPE_Q8_0, "K/V pin belongs to the backbone table");
    chk("blk.3.attn_v.weight",        1024, GGML_TYPE_Q8_0, false, PXA_POLICY_BALANCED, GGML_TYPE_Q8_0, "K/V pin belongs to the backbone table");
    chk("token_embd.weight",        248320, GGML_TYPE_Q6_K, false, PXA_POLICY_ATTN4,    GGML_TYPE_Q6_K, "row gather, not a GEMM");
    chk("output.weight",            248320, GGML_TYPE_Q8_0, false, PXA_POLICY_BALANCED, GGML_TYPE_Q8_0, "the head has its own resolver");
    chk("blk.0.attn_gate.weight",      256, GGML_TYPE_F16,  false, PXA_POLICY_ATTN4,    GGML_TYPE_F16,  "per-HEAD gate stays f16");

    // The per-head / per-channel split is decided on ne[1] alone. Pin BOTH sides: this is the
    // one place a single tensor name means two different things across our model zoo.
    if (pxa_pxq_policy_role_of("blk.0.attn_gate.weight", 6144) != PXA_ROLE_ATTN) {
        printf("  FAIL per-CHANNEL attn_gate (ne1=6144) must classify as ATTN\n"); ++g_fail;
    }
    if (pxa_pxq_policy_role_of("blk.0.attn_gate.weight", 72) != PXA_ROLE_OTHER) {
        printf("  FAIL per-HEAD attn_gate (ne1=72, the Laguna shape) must NOT be claimed\n"); ++g_fail;
    }
    g_checks += 2;

    // =====================================================================================
    // 2. Qwen3.8-Flash-Next — arch qwen4exp, MoE: 48 blocks, 512 experts, top-10, plus a
    //    shared expert and the same DeltaNet/attention hybrid backbone. Here the level names
    //    the EXPERT tier and `base` is already BACKBONE_REV 2's promoted backbone, so a
    //    profile flattens the ladder instead of climbing it.
    // =====================================================================================
    printf("-- Qwen3.8-Flash-Next (qwen4exp, MoE 512x top-10) --\n");

    // At PXQ2 rev 2 already produces exactly the profile's answer: this must be a no-op, or
    // the profile would be re-quantizing a recipe that is already correct.
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ4,   true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ2 MoE: rev2 already is this rule");
    chk("blk.3.attn_output.weight",   2560, GGML_TYPE_PXQ4HQ, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ2 MoE: the cap pulls rev2's pxq4hq attn_output to pxq4");

    // At PXQ3 rev 2 hands the WHOLE backbone pxq4hq. The profile pulls everything except
    // attn_output back to pxq4 — smaller (4.2526 bpw vs 4.52) and on the faster MMVQ type.
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ4HQ, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ3 MoE: pull the backbone to pxq4");
    chk("blk.0.attn_gate.weight",     6144, GGML_TYPE_PXQ4HQ, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ3 MoE: pull the backbone to pxq4");
    chk("blk.0.ssm_out.weight",       2560, GGML_TYPE_PXQ4HQ, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ3 MoE: pull the backbone to pxq4");
    chk("blk.0.ffn_up_shexp.weight",  2560, GGML_TYPE_PXQ4HQ, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "shared expert is always-resident bulk");
    chk("blk.3.attn_output.weight",   2560, GGML_TYPE_PXQ4HQ, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ3 MoE: attn_output joins the pxq4 backbone");

    // At PXQ4/PXQ6 rev 2 hands the whole backbone pxq4; the profile buys attn_output up one.
    chk("blk.3.attn_output.weight",   2560, GGML_TYPE_PXQ4, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ4 MoE: nothing to do, rev2 is already pxq4");
    chk("blk.0.attn_qkv.weight",     10240, GGML_TYPE_PXQ4, true, PXA_POLICY_BALANCED, GGML_TYPE_PXQ4,   "PXQ4 MoE: the rest stays pxq4");

    // On a MoE model balanced and attn4 are the SAME rule. If they ever diverge, the docs and
    // the provenance summary are lying about it.
    for (ggml_type b : { GGML_TYPE_PXQ4, GGML_TYPE_PXQ4HQ }) {
        for (const char * n : { "blk.0.attn_qkv.weight", "blk.3.attn_output.weight",
                                "blk.0.ffn_down_shexp.weight", "blk.0.ssm_out.weight" }) {
            ++g_checks;
            if (pxa_pxq_policy_apply(n, 2560, b, true, PXA_POLICY_BALANCED) !=
                pxa_pxq_policy_apply(n, 2560, b, true, PXA_POLICY_ATTN4)) {
                printf("  FAIL %s: balanced and attn4 must agree on a MoE model\n", n); ++g_fail;
            }
        }
    }

    // Routed expert stacks never reach the policy layer (the backbone table returns
    // GGML_TYPE_COUNT for them long before this), but if that guard is ever relaxed the
    // policy must still not claim them — the level IS their tier.
    if (pxa_pxq_policy_role_of("blk.0.ffn_down_exps.weight", 640) != PXA_ROLE_OTHER ||
        pxa_pxq_policy_role_of("blk.0.ffn_up_exps.weight",   640) != PXA_ROLE_OTHER ||
        pxa_pxq_policy_role_of("blk.0.ffn_gate_exps.weight", 640) != PXA_ROLE_OTHER) {
        printf("  FAIL routed expert stacks must never be claimed by the policy layer\n"); ++g_fail;
    }
    g_checks += 3;

    // =====================================================================================
    // 2b. THE PORTABILITY INVARIANT: no profile may ever PRODUCE pxq4hq. It is llama.cpp-only
    // in this release (no vLLM decoder, no vLLM kernel), so a profile file containing one
    // pxq4hq tensor would run on one engine and not the other. The only way pxq4hq comes out
    // of the resolver is the never-demote floor on a file the operator ASKED for at that
    // level. Swept over every role, every base and both profiles rather than spot-checked,
    // because this is the invariant that decides whether a profile file is portable at all.
    // =====================================================================================
    {
        const char * names[] = { "blk.0.attn_qkv.weight", "blk.3.attn_q.weight",
                                 "blk.3.attn_output.weight", "blk.0.attn_gate.weight",
                                 "blk.0.ssm_out.weight", "blk.0.ffn_up.weight",
                                 "blk.0.ffn_down.weight", "blk.0.ffn_up_shexp.weight" };
        const ggml_type bases[] = { GGML_TYPE_PXQ1, GGML_TYPE_PXQ2, GGML_TYPE_PXQ3, GGML_TYPE_PXQ4 };
        for (bool moe : { false, true }) {
            for (auto p2 : { PXA_POLICY_BALANCED, PXA_POLICY_ATTN4 }) {
                for (ggml_type b : bases) {
                    for (const char * n : names) {
                        ++g_checks;
                        const ggml_type got = pxa_pxq_policy_apply(n, 6144, b, moe, p2);
                        if (got == GGML_TYPE_PXQ4HQ) {
                            printf("  FAIL %s %s base=%s moe=%d -> pxq4hq; no profile may emit a "
                                   "type only one engine can read\n",
                                   pxa_pxq_policy_name(p2), n, ggml_type_name(b), (int) moe);
                            ++g_fail;
                        }
                    }
                }
            }
        }
    }

    // =====================================================================================
    // 2c. THE ssm_out FLOOR. The DeltaNet output projection measured mean KLD 0.433 against
    // its own Q8_0 source at pxq4 on Qwen3.8-27B and 0.042 at q8_0 — a 7x gap out of one
    // class, 48 tensors, ~5% of the file. Before the floor EVERY level put it on the ladder:
    // the dense branches hand it the named tier, and both profiles climb it to the pxq4 cap
    // because it is PXA_ROLE_ATTN. The floor runs on the resolver's OUTPUT, so it has to hold
    // for every (level x profile x shape) combination, not just the one that was measured.
    //
    // The composed helper below is the quantizer's own chain — profile, then floor — because
    // testing the floor on its own would pass while the real pipeline still shipped pxq4.
    // =====================================================================================
    printf("-- the ssm_out floor (DeltaNet out-proj) --\n");
    {
        const ggml_type prot = pxa_pxq_ssm_out_protect_type();

        // PORTABILITY. The floor's type ends up in EVERY default-built hybrid file, and a file
        // is routed to an engine by its LEVEL, not by its contents — so a floor type only one
        // engine can decode would hand the other engine a file it cannot load, with the level
        // name promising otherwise. pxq6 and pxq4hq are llama.cpp-only in this release (the
        // same reason the profiles are capped at pxq4). This is the check that stops the
        // "cheaper by 3.9% of file size" argument from quietly reintroducing that.
        ++g_checks;
        if (prot == GGML_TYPE_PXQ6 || prot == GGML_TYPE_PXQ4HQ) {
            printf("  FAIL the ssm_out floor is %s, which only llama.cpp can decode; a "
                   "default-built file must load on both engines\n", ggml_type_name(prot));
            ++g_fail;
        }

        // the chain the quantizer runs: BACKBONE_REV 2 hands `base` to the profile, the floor
        // runs on what comes back
        auto chain = [](const char * n, int64_t ne1, ggml_type base, bool moe, pxa_pxq_policy_id p) {
            return pxa_pxq_ssm_out_floor(n, pxa_pxq_policy_apply(n, ne1, base, moe, p));
        };

        // EVERY level, EVERY profile, dense AND MoE: ssm_out comes out at the protected type.
        // The dense rows are the 27B (ssm_out 6144x5120); the MoE rows are Flash-Next, where
        // rev 2 has already promoted the backbone before the profile sees it.
        for (auto p2 : { PXA_POLICY_UNIFORM, PXA_POLICY_BALANCED, PXA_POLICY_ATTN4 }) {
            for (ggml_type b : { GGML_TYPE_PXQ1, GGML_TYPE_PXQ2, GGML_TYPE_PXQ3,
                                 GGML_TYPE_PXQ4, GGML_TYPE_PXQ4HQ, GGML_TYPE_PXQ6 }) {
                ++g_checks;
                const ggml_type got = chain("blk.0.ssm_out.weight", 5120, b, false, p2);
                if (got != prot) {
                    printf("  FAIL %-8s dense base=%-7s ssm_out -> %-7s, expected %s\n",
                           pxa_pxq_policy_name(p2), ggml_type_name(b), ggml_type_name(got),
                           ggml_type_name(prot));
                    ++g_fail;
                }
                ++g_checks;
                const ggml_type gotm = chain("blk.0.ssm_out.weight", 2560, b, true, p2);
                if (gotm != prot) {
                    printf("  FAIL %-8s MoE   base=%-7s ssm_out -> %-7s, expected %s\n",
                           pxa_pxq_policy_name(p2), ggml_type_name(b), ggml_type_name(gotm),
                           ggml_type_name(prot));
                    ++g_fail;
                }
            }
        }

        // THE DEFAULT PROFILE PER LEVEL — the combination a no-flag run actually builds, read
        // from pxa_pxq_policy_default_for() rather than from a copy of it. PXQ2 -> attn4,
        // PXQ3 -> balanced, the rest -> uniform: three different code paths into the same
        // answer, which is the whole point of putting the floor after the profile.
        for (ggml_type b : { GGML_TYPE_PXQ2, GGML_TYPE_PXQ3, GGML_TYPE_PXQ4,
                             GGML_TYPE_PXQ4HQ, GGML_TYPE_PXQ6 }) {
            ++g_checks;
            const ggml_type got = chain("blk.0.ssm_out.weight", 5120, b, false,
                                        pxa_pxq_policy_default_for(b));
            if (got != prot) {
                printf("  FAIL no-flag %s target: ssm_out -> %s, expected %s (this is the file "
                       "users get)\n", ggml_type_name(b), ggml_type_name(got), ggml_type_name(prot));
                ++g_fail;
            }
        }

        // OVERRIDDEN. --custom-q, a PXQU map entry and the geometry demote all reach the write
        // loop as a resolved NON-LADDER type; the floor must hand every one of them back
        // untouched or it would overrule the operator (and undo the MTP companion's q8_0 pin,
        // and re-promote a deliberately flat-MXFP4 backbone that measured 0.060, not 0.433).
        for (ggml_type keep : { GGML_TYPE_MXFP4, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,
                                GGML_TYPE_F16, GGML_TYPE_COUNT }) {
            ++g_checks;
            const ggml_type got = pxa_pxq_ssm_out_floor("blk.0.ssm_out.weight", keep);
            if (got != keep) {
                printf("  FAIL the floor moved a non-ladder landing (type %d) to type %d; those "
                       "are not its to move\n", (int) keep, (int) got);
                ++g_fail;
            }
        }
        // A FLOOR, NOT A PIN: applying it to its own output must change nothing. Idempotence is
        // the property that holds whatever the constant is set to, so it survives the type
        // being switched — which is the one edit this rule is expected to take.
        for (ggml_type b : { GGML_TYPE_PXQ2, GGML_TYPE_PXQ3, GGML_TYPE_PXQ4,
                             GGML_TYPE_PXQ4HQ, GGML_TYPE_PXQ6 }) {
            ++g_checks;
            const ggml_type once = pxa_pxq_ssm_out_floor("blk.0.ssm_out.weight", b);
            if (pxa_pxq_ssm_out_floor("blk.0.ssm_out.weight", once) != once) {
                printf("  FAIL the floor is not idempotent from base %s\n", ggml_type_name(b));
                ++g_fail;
            }
        }

        // NAME SCOPE. The floor claims exactly one leaf. If it ever widened, every attention
        // and FFN class in the file would silently gain bits and the level name would stop
        // meaning anything — the same failure the role table's leaf-name checks exist for.
        for (const char * n : { "blk.0.attn_qkv.weight", "blk.3.attn_output.weight",
                                "blk.0.attn_gate.weight", "blk.0.ffn_down.weight",
                                "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_exps.weight",
                                "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
                                "blk.0.ssm_out_norm.weight", "blk.0.ssm_norm.weight" }) {
            ++g_checks;
            if (pxa_pxq_ssm_out_floor(n, GGML_TYPE_PXQ2) != GGML_TYPE_PXQ2) {
                printf("  FAIL the ssm_out floor claimed %s\n", n); ++g_fail;
            }
        }
        ++g_checks;
        if (!pxa_pxq_is_ssm_out("blk.47.ssm_out.weight")) {
            printf("  FAIL the floor must match the real GGUF leaf blk.N.ssm_out.weight\n"); ++g_fail;
        }
        // The MTP companion block reaches the backbone table's `.nextn.` branch and is pinned
        // to q8_0 there; a draft-path tensor is not the measured class and must not be lifted
        // by name alone. Non-ladder in, non-ladder out.
        ++g_checks;
        if (pxa_pxq_ssm_out_floor("blk.64.nextn.ssm_out.weight", GGML_TYPE_Q8_0) != GGML_TYPE_Q8_0) {
            printf("  FAIL the floor overrode the MTP companion's q8_0 pin\n"); ++g_fail;
        }
    }

    // =====================================================================================
    // 3. Profile-name parsing: a typo must FAIL, not silently pick something.
    // =====================================================================================
    pxa_pxq_policy_id p = PXA_POLICY_ATTN4;
    if (pxa_pxq_policy_parse("nonsense", &p) || p != PXA_POLICY_ATTN4) {
        printf("  FAIL unknown profile name must be rejected and leave the value alone\n"); ++g_fail;
    }
    if (!pxa_pxq_policy_parse("BALANCED", &p) || p != PXA_POLICY_BALANCED) {
        printf("  FAIL profile names must be case-insensitive\n"); ++g_fail;
    }
    // THE DEFAULT RULE (2026-09-09): PXQ2 defaults to attn4, everything else to uniform.
    // Asserted against the shipped function, not a copy of it.
    if (pxa_pxq_policy_default_for(GGML_TYPE_PXQ2) != PXA_POLICY_ATTN4) {
        printf("  FAIL the PXQ2 base must default to attn4 (uniform PXQ2 measured 4.8x the PXQ4 anchor)\n"); ++g_fail;
    }
    if (pxa_pxq_policy_default_for(GGML_TYPE_PXQ3) != PXA_POLICY_BALANCED) {
        printf("  FAIL the PXQ3 base must default to balanced (KLD 0.0758 vs uniform's 0.1056)\n"); ++g_fail;
    }
    for (ggml_type b : {GGML_TYPE_PXQ4, GGML_TYPE_PXQ4HQ, GGML_TYPE_PXQ6}) {
        if (pxa_pxq_policy_default_for(b) != PXA_POLICY_UNIFORM) {
            printf("  FAIL only the PXQ2 base was measured and moved; this one must stay uniform\n"); ++g_fail;
        }
    }

    if (!pxa_pxq_policy_parse("uniform", &p) || p != PXA_POLICY_UNIFORM) {
        printf("  FAIL 'uniform' must parse\n"); ++g_fail;
    }
    g_checks += 3;

    printf("\n%d checks, %d failure(s)\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
