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
// token stream, not of copy timing.
#pragma once

struct llama_model;

// true while a hot-swap residency group is active: the adaptation stands down (its weights are mirrored once and restored at unpark)
bool llama_pxa_xcache_residency_blocks(void);

// a decode that queued n_tokens just completed (llama_synchronize): advance the adaptation by one tick
void llama_pxa_xcache_tick(llama_model & model, int n_tokens);

// finish every copy in flight, print the summary, free the adaptor (the model destructor)
void llama_pxa_xcache_adaptor_free(llama_model & model);
