#pragma once

// PXA_QSA_PROF: where a decoded token's QSA time actually goes.
//
// WHY THIS EXISTS
// ---------------
// Milestone 1 shipped a CORRECT gather (byte-identical to the architecture's own reference at
// 86k) that is nonetheless SLOWER than dense: measured on the P100 quad, 2026-09-09, decode
// t/s at 86,401 fill was dense 17.85, mask 13.22, gather 14.80. The gather beats the mask arm
// by 12% and that margin grows with depth, so the mechanism works; what does not pay is the
// SELECTION in front of it. "Selection" is six different things at once -- two indexer
// projections, the block pooling, the score, a full argsort over ~21.6k blocks, and the
// block->cell expansion -- plus a HOST-side grid rebuild that no device profiler can see, and
// a cost model cannot choose between them from the outside.
//
// So: measure first. This attributes every node of the decode graph to one QSA stage, and
// every microsecond of the host-side k-pool plan to one host bucket, and prints both.
//
//   PXA_QSA_PROF=1   the static cost model: per-stage NODE COUNTS and OUTPUT BYTES, per layer
//                    and in total, printed whenever the graph shape changes, plus the host
//                    bucket timings every PXA_QSA_PROF_EVERY decode steps. Zero distortion:
//                    node counts are read off the built graph and the host timers are four
//                    ggml_time_us() calls per ubatch.
//   PXA_QSA_PROF=2   also per-node device timing, bucketed into the same stages. This installs
//                    an eval callback, which makes ggml_backend_sched compute ONE NODE AT A
//                    TIME and synchronize after each: absolute microseconds are inflated and
//                    CUDA graph capture is off, so read the SHARES, not the totals.
//
// WHY NODE COUNTS ARE A COST MODEL HERE AND NOT A CURIOSITY
// --------------------------------------------------------
// The Flash-Next seat's decode profile on this hardware is ~1340 kernel launches per token at
// 98.4% host time (mj-deep, 2026-09-08): the step is bound by per-launch host work, not by
// device bandwidth. A stage's node count is therefore its first-order cost, and bytes are the
// second-order term. Both are printed.
//
// The stage boundaries are declared by the BUILDER (build_qwen4exp.cpp) with pxa_qsa_prof_tag()
// on each stage's root tensor; every other node inherits its stage from its sources, so the
// unnamed conts, reshapes and permutes land in the stage that produced them rather than in a
// catch-all. A stage's LAST tensor is tagged terminal, which stops the inheritance leaking into
// the rest of the model. Nodes reached from no tagged tensor are "other" -- the 36 DeltaNet
// layers, the FFN, the trunk -- and their count is the denominator that makes the QSA share
// readable.

#include <cstdint>

struct ggml_tensor;
struct ggml_cgraph;

enum pxa_qsa_stage {
    PXA_QSA_ST_IPROJ = 0,   // indexer q/k projections, their norms and ropes
    PXA_QSA_ST_POOL,        // block pooling, the write-back, and the pooled-key read
    PXA_QSA_ST_SCORE,       // the per-block score
    PXA_QSA_ST_TOPK,        // the top-k over blocks
    PXA_QSA_ST_EXPAND,      // block -> cell expansion (control arm: the n_kv-wide mask)
    PXA_QSA_ST_GATHER,      // the physical K/V row gather and its conts
    PXA_QSA_ST_ATTN,        // attention over the compact set (control arm: dense attention)
    PXA_QSA_ST_OTHER,       // every node reached from no tagged tensor
    PXA_QSA_ST_NB
};

enum pxa_qsa_host_bucket {
    PXA_QSA_H_SCAN = 0,     // llama_kpool_build_plan's scan of the cache's cells
    PXA_QSA_H_LAYOUT,       // llama_kpool_build_layout
    PXA_QSA_H_MARKNEW,      // llama_kpool_mark_new
    PXA_QSA_H_FILL,         // llama_kpool_fill (via llama_kpool_set_inputs)
    PXA_QSA_H_NB
};

// 0 = off. Read once from PXA_QSA_PROF.
int pxa_qsa_prof_level();

inline bool pxa_qsa_prof_on() { return pxa_qsa_prof_level() > 0; }

// Called by the builder. `terminal` marks the last tensor of a QSA region so that the nodes
// downstream of it (wo, the FFN, the next layer) do not inherit a QSA stage.
void pxa_qsa_prof_tag(const ggml_tensor * t, int stage, int il, bool terminal = false);

// Called by the builder at the start and at the end of the graph it builds.
void pxa_qsa_prof_begin_graph();
void pxa_qsa_prof_seal(const ggml_cgraph * gf, int n_tokens, int n_kv,
                       int n_pool, int n_top, int n_sel, bool gather);

// Host-side buckets, in microseconds.
void pxa_qsa_prof_host_add(int bucket, int64_t us);

// Level 2 only: the eval callback brackets each node with these. The scheduler computes and
// synchronizes exactly one node between the two calls, so the difference is that node's
// serialized wall time -- submit plus execute plus sync.
void pxa_qsa_prof_node_begin();
void pxa_qsa_prof_node_end(const ggml_tensor * t);

// One decode step finished. Reports every PXA_QSA_PROF_EVERY steps (default 32).
void pxa_qsa_prof_tick(int n_tokens);
