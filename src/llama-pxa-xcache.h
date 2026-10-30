// pxa / PXA expert-cache online adaptation -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache.h -- PXA_XCACHE: the mechanism that re-ranks the resident (hot) experts while a model decodes.
//
// The expert cache plans its hot set once, at load, from a routing profile of the model file. A prompt that is nothing like the
// profile (code, another language, a long document) routes to other experts, so the static plan misses more than it predicted.
// This module keeps the cache in step with what the model is actually routing:
//
//   * the side-0 split-ids op counts every routed expert per layer on the device (ggml/src/ggml-cuda/pxa/pxa-xcache.cuh);
//   * after a decode completes (llama_synchronize) the counters are read back and handed to the POLICY, which lives in the closed
//     PXQN library (ggml_pxqn_xcache_policy, ggml-pxqn-api.h) because ranking / hysteresis / budget are the part that is tuned;
//   * the policy answers with swaps (cold expert in, hot expert out). The engine runs each as two asynchronous copies on a copy
//     stream of its own and flips the layer's routing map between two decodes, EVICT FIRST:
//         evict : the displaced hot expert's slot -> a FREE slot of the layer's cold stack (pinned host RAM, a few per layer that the
//                 planner adds); when it has landed the map sends that expert to the cold path and its hot slot is vacant
//         admit : the admitted expert's cold slot -> the vacant hot slot; when it has landed the map points the expert at it and its
//                 old cold slot is the layer's free slot again
//     A slot is only ever written while no map entry points at it, and the map only changes while no graph is running, so a decode
//     never reads a slot that is being copied and never waits for a copy. The hot stacks keep their planned size (the free slots
//     cost host RAM, not VRAM) and an evicted expert is served at once by the cold path.
//   * the graph is untouched: same ops, same shapes, same ids -> no rebuild. Without the library (or PXA_XCACHE_ADAPT=0) none of
//     this runs and the static plan is exactly what it was.
//
// Reproducibility: by default (PXA_XCACHE_ADAPT_DET=1) every swap is applied at a fixed number of decodes after it was planned
// (waiting for its copy when it is not yet done, which at the sizes involved is rare), so the swap schedule is a function of the
// token stream, not of copy timing. Across requests: every sequence that starts at position 0 starts from the planned state
// (PXA_XCACHE_ADAPT_RESET, default on), because the hot set changes the floats (GPU kernel vs CPU cold path).
#pragma once

#include <cstdint>

struct llama_model;

// true while a hot-swap residency group is active: the adaptation stands down (its weights are mirrored once and restored at unpark)
bool llama_pxa_xcache_residency_blocks(void);

// a decode that queued n_tokens just completed (llama_synchronize): advance the adaptation by one tick
void llama_pxa_xcache_tick(llama_model & model, int n_tokens);

// a batch starts a sequence at position 0 (llama_decode, main model): with PXA_XCACHE_ADAPT_RESET (default on) the adaptation goes
// back to the planned hot set first, so a repeated request repeats its answer exactly; =0 keeps the hot set across requests
void llama_pxa_xcache_seq_start(llama_model & model);

// the speculation companion context takes the quiet path (a switch of the PXQN library, ggml-pxqn-tune.h; off without it):
// its decodes only note their tokens (llama_pxa_xcache_note_tokens) instead of ticking
bool llama_pxa_xcache_companion_quiet(void);
void llama_pxa_xcache_note_tokens(llama_model & model, int n_tokens);

// PXA_XCACHE_RTIER: planner-thread jobs run and planned swaps found stale at start (false = RTIER not active)
bool llama_pxa_xcache_rtier_stats(const llama_model * model, uint64_t * jobs, uint64_t * stale);
// PXA_XCACHE_EVICT_FILE: evicts read back from the model file (and how many failed); false when the mode is off
bool llama_pxa_xcache_evict_file_stats(const llama_model * model, uint64_t * reads, uint64_t * failed);

// finish every copy in flight, print the summary, free the adaptor (the model destructor)
void llama_pxa_xcache_adaptor_free(llama_model & model);
