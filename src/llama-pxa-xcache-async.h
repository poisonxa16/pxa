// pxa / PXA expert-cache cold path without host round trips -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-xcache-async.h -- PXA_XCACHE_ASYNC: the cold half of a split expert layer computed by a host worker thread that the
// GPU's own stream hands work to and takes results from, instead of a scheduler split with two host synchronizations.
//
// Without the lever a decode through a cache layer is: GPU split (attention, router, hot experts) -> the scheduler synchronizes the
// stream and copies the cold ids and the activation to the host -> a CPU split computes the missed experts -> the result is copied
// back and the next GPU split starts. Two host syncs per cold layer (about 0.7 ms of fixed cost a layer, ~14 of 46 ms a token on
// Flash-Next 32GB on one P100) before any expert is computed, and the CPU never overlaps the GPU's hot experts.
//
// With it (graphs of 1 .. PXA_XCACHE_ASYNC_MAXTOK tokens) the layer's cold half is two nodes of the main graph and no split:
//
//   GPU stream:   ... router, ids -> [submit] -> hot experts of the layer -> [wait] -> merge -> ...
//                                       |                                      ^
//                                 ids + activation                         done flag + rows
//                                 into pinned host memory, request flag      from pinned host memory
//                                       v                                      |
//   host worker:  poll flags -> the layer's CPU sub-graph (the very ggml nodes the scheduler's cold split ran: up, gate, unary,
//                 down over the cold stack) -> results into pinned host memory -> done flag
//
// Exactness: the CPU kernels and their inputs are the ones the synchronous path uses (same nodes, same weights, same activation
// bytes), so the output is bit-identical to the split path; tests/test-pxa-xcache-async.cpp proves it. A graph wider than the cap
// (a prefill ubatch) keeps the scheduler path. CUDA-graph capture is off for a graph that carries the nodes (they wait on a host
// thread). Default OFF: PXA_XCACHE_ASYNC=1 turns it on.
#pragma once

struct llama_model;
struct ggml_tensor;

// PXA_XCACHE_ASYNC=1
bool llama_pxa_xcache_async_wanted(void);
// widest graph (tokens) that takes the async path: PXA_XCACHE_ASYNC_MAXTOK, default 8 (a decode, an MTP / n-gram verify batch)
int  llama_pxa_xcache_async_max_tokens(void);

// The rendezvous handle of layer `il` for a graph of n_tokens tokens (an opaque struct ggml_cuda_cold_slot *), creating the layer's
// slot, the (layer, width) CPU sub-graph and the worker thread on first use; nullptr when this layer cannot take the async path
// (the caller then builds the scheduler path). up / gate (or up_gate) / down are the layer's COLD stacks as the CPU sees them (the
// aliases of the pinned bytes); unary_op is a ggml_unary_op (SILU or GELU).
//
// fused_chain: the split path builds this layer's cold chain through ggml_moe_up_gate (merged up/gate stack, or a pair of one fusable type), which the
// scheduler keeps whole on the CPU. A layer built the other way (separate named up / gate / unary nodes, e.g. an up stack of one type and a gate stack
// of another) gets its activation node placed on the GPU by the scheduler, so its split-path result carries the GPU's rounding of silu: the async path
// (everything on the CPU) would differ from it in the last bits of the activation, amplified by the down matmul's q8 quantisation (measured on
// Flash-Next 32GB: 4 of 31 cold layers, |diff| up to 2.4e-4). Such a layer stays on the split path (bit-identical output) unless
// PXA_XCACHE_ASYNC_MIXED=1 asks for the all-CPU chain anyway.
void * llama_pxa_xcache_async_slot(const llama_model & model, int il, int n_tokens, int n_embd, int n_used,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * up_gate, ggml_tensor * down, int unary_op, int n_threads, bool fused_chain);

// stop the worker, print the summary, free the slots (the model destructor)
void llama_pxa_xcache_async_free(llama_model & model);

// PXA_XCACHE_ASYNC_CHECK (diagnostic): a node (side 4 of GGML_OP_MOE_SPLIT_IDS) that compares the cold rows of two results bit for bit
// on the GPU and adds the counts to the slot's counters; its I32 [2] result only carries the dependency
ggml_tensor * llama_pxa_xcache_async_check_node(struct ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, ggml_tensor * ids_cold, void * slot);
