// pxa / PXA expert cache: skip the cold tail of a decode / verify window -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache-skip.h -- PXA_XCACHE_SKIP_COLD_RANK (opt-in "fast" mode, default OFF, CHANGES OUTPUT).
//
// A split MoE layer (PXA_XCACHE) computes its cold (non-resident) experts on the CPU or reads them over the link; those are the slow
// part of an Overdrive decode. With PXA_XCACHE_SKIP_COLD_RANK=R, a COLD expert that every token of the window routes to with rank R or
// worse (rank 1 = the token's largest routing weight) is dropped: it is not computed, its slots contribute 0, and the other weights are
// NOT renormalised. The window is the graph's batch: one token for a decode, the draft + 1 tokens for an MTP / n-gram verify.
//   * resident (hot) experts are never dropped; a cold expert that any token of the window ranks better than R is computed for every
//     token that routes to it; R > n_expert_used drops nothing (exact)
//   * armed only on graphs of at most PXA_XCACHE_SKIP_COLD_MAXTOK tokens (default 16; prefill ubatches are exact), only for the main
//     model (never the MTP draft graph or the MTP head's layer), only on the n_expert_used routing the model was trained with
//   * the device op: ggml_moe_split_ids_set_skip_cold (ggml.h); counters: ggml_backend_cuda_xcache_skip_stats (ggml-cuda-xcache.h)
//   * log: PXA_XCACHE_SKIP_COLD_LOG=N prints the dropped experts of the last N decode / verify steps every N steps (0 = only the
//     summary when the model is freed)
// Lever off (unset / 0): no node carries the skip and every op runs the code it ran before (byte-identical output).
#pragma once

struct llama_model;

// R (>= 1) when PXA_XCACHE_SKIP_COLD_RANK asks for the skip, 0 otherwise. The first call with the lever on logs the banner.
int  llama_pxa_xcache_skip_cold_rank(void);
// widest graph (tokens) the skip is armed on (PXA_XCACHE_SKIP_COLD_MAXTOK, default 16)
int  llama_pxa_xcache_skip_cold_maxtok(void);
// a main-model decode of n_tokens completed (llama_synchronize): one step for the counters / the periodic log line
void llama_pxa_xcache_skip_note_step(const llama_model & model, int n_tokens);
// the totals (the model destructor; prints only when the lever was on)
void llama_pxa_xcache_skip_summary(const llama_model & model);
