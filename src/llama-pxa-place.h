// pxa / PXA placement planner -- authored by PXA Network (https://pxanetwork.com).
// llama-pxa-place.h -- PXA_STREAM_WEIGHTS=auto: decide, per device and BEFORE any weight is
// allocated, whether the model runs resident or which weights stream from pinned host RAM.
//
// One source of truth. The loader (src/llama.cpp, llm_load_tensors) gathers the inputs from GGUF
// metadata and the fit estimator it already has (weights + KV + compute per device, from
// get_layer_sizes), the planner decides, the loader routes the chosen tensors to the device's
// CUDA<d>_Stream buffer, and the decision is kept on the model for the boot banner and the
// PXA_EXPLAIN placement object (pxa_place_json).
//
// Decision (see stream-next plan, 2026-09-25):
//   R1 fits resident (weights + KV + compute + GEMM pool + the floor <= free, the same test the
//      context applies after the pool pre-grow): resident, nothing changes -- same buffers, same
//      reservations, same graphs, same bytes out. Streaming never engages for a model that fits.
//   R2 fits only with the worst-case logits reservation capped (a card the old binary refused
//      or crashed on): resident with the cap; llama_decode splits a ubatch past the cap.
//   R3 MoE that does not fit: stream the routed experts of just enough layers (spread evenly),
//      everything else resident. Wide graphs stream them through the ring, narrow graphs compute
//      them on the CPU from the same pinned bytes (the '-ot exps=CPU' path).
//   R4 dense that does not fit: spill FFN tensors (down first, then up+gate pairs), spread
//      evenly, exactly the deficit + 5%; attention / DeltaNet stay resident.
//   R5 the pin exceeds pin_ok = MemAvailable - max(16 GiB, 10% MemTotal) - PXA_STREAM_PIN_RESERVE_GB:
//      the same tensors go to mmap'd host RAM instead (today's CPU placement, no ring).
//   Auto stays off for -sm tensor/graph/attn, any user -ot / --n-cpu-moe / --fit, -ngl < all.
//   R3x (PXA_XCACHE, expert-cache 2026-09-25) MoE that does not fit AND per-expert routing counts
//      are known (<model>.expert-counts.csv or PXA_XCACHE_COUNTS): instead of whole layers, the
//      LEAST-ROUTED EXPERTS OF EVERY LAYER leave VRAM (global ranking by count, with a per-layer
//      cap chosen to minimise the routed mass that misses the resident set). Each split layer keeps
//      a resident hot stack and a pinned cold stack; GGML_OP_MOE_SPLIT_IDS / GGML_OP_MOE_MERGE
//      route every token's experts to the right stack, so a decode token pays the slow path only
//      for the few percent of routings that hit a cold expert, not for whole layers. The ring only
//      has to hold two layers' COLD bytes, which frees most of the layer-granular ring for experts.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum pxa_place_kind {
    PXA_PLACE_OTHER = 0,     // attention, DeltaNet, norms, router, ... (never spilled)
    PXA_PLACE_FFN_DOWN,      // dense ffn_down (spilled first)
    PXA_PLACE_FFN_UPGATE,    // dense ffn_up / ffn_gate (spilled as a pair, after the downs)
    PXA_PLACE_EXPERTS,       // routed experts (up/gate/down/gate_up _exps)
};

struct pxa_place_tensor {
    std::string    name;
    int            layer  = -1;
    int            device = -1;   // index into the planner's device list
    size_t         bytes  = 0;
    pxa_place_kind kind   = PXA_PLACE_OTHER;
};

struct pxa_place_device {
    int    id            = -1;  // CUDA device id
    size_t free_bytes    = 0;   // free VRAM at load
    size_t need_resident = 0;   // weights + KV + compute (+ output) on this device, fully resident
    size_t pool_need     = 0;   // bug #266 GEMM temporary pool worst case at the planned ubatch
    int    cc            = 0;   // compute capability x10 (60 P100, 70 V100; 0 = unknown) -> R4 cost table
    // bug #339 context guard (pxa_place_input::ctx_guard; llama.cpp sets these for an all-outputs load: perplexity, KLD, imatrix,
    // embeddings, which never take the logits cap): the worst-case logits reservation the planner took off need_resident for the cap
    // and the head GEMM's pool growth at the full ubatch width beyond pool_need (pxa_place_head_pool_need). Both on the head's device.
    size_t ctx_logits    = 0;
    size_t ctx_head      = 0;
};

struct pxa_place_input {
    std::vector<pxa_place_device> devices;
    std::vector<pxa_place_tensor> tensors;
    int    n_layer          = 0;
    int    mode_req         = 0;      // 0 off, 1 layers, 2 experts, 3 spill, 4 auto
    size_t pin_ok           = 0;      // bytes that may be pinned (0 = unknown: allow)
    size_t spill_extra      = 0;      // PXA_STREAM_EXTRA_MB: stream this much more than planned
    // PXA_XCACHE: per-layer routing count per expert (empty = unknown -> whole-layer streaming),
    // and the extra compute bytes a split layer's graph holds live at the planned ubatch
    int    n_expert         = 0;
    std::vector<std::vector<double>> xc_counts;   // [n_layer][n_expert], empty rows = no counts
    std::vector<int> xc_n_expert_l;               // [n_layer] experts in that layer's routed-expert tensors where layers differ
                                                  // (Nemotron 3.5 Bolt: 240 or 304 of 496); empty = n_expert in every layer
    size_t xc_extra_compute = 0;
    int    n_expert_used    = 0;      // routed experts per token (R4 miss cost = bytes moved per token)
    bool   xc_cold_zc       = false;  // cold path: true = zero-copy GPU read, false = CPU alias (pxa_cold_path_zc)
    size_t xc_slack_credit  = 0;      // bytes the expert cache may keep resident beyond the planner's margin (PXA_XCACHE_SLACK_MB): the slack a measured run left free after the KV cache and compute buffers
    // C042 small-tail rule (2026-10-18; llama.cpp sets these, a hand-built input keeps the old plan): on ONE sm_70 device
    // whose cold experts run on the CPU (not zero-copy) and whose deficit is at most half its free VRAM, the cold stacks' ring holds ONE
    // layer (the largest cold layer + the largest cold tensor + 32 MiB, not two layers: a narrow graph never reads the ring, a wide one
    // streams layer by layer) and xc_slack_auto of the 512 MiB + 5% margin goes back to hot experts. 35B-A3B PXQN4 on one V100:
    // 2618 -> ~2110 cold experts, decode 61.8 -> 67.9 t/s at -ub 2048 (v3.1 57.1). PXA_XCACHE_RING1=0 / PXA_XCACHE_SLACK_MB=0 turn them off.
    bool   xc_ring1_auto    = false;
    size_t xc_slack_auto    = 0;
    // R3a ring lending (PXA_XCACHE_RING_LEND, 2026-10-25; default OFF = the plan exactly as before): with the CPU cold path no graph
    // narrower than the CPU-alias width (1638 tokens on Flash-Next) reads the cold stacks' ring, so the plan does not charge it: its bytes
    // go to resident experts and the CUDA backend allocates the ring on demand when a wide graph streams (zero-copy reads when the VRAM
    // is not there; ggml/src/ggml-cuda/pxa/pxa-stream.cuh). The zero-copy cold path reads the ring from 65 tokens up: never lent there.
    bool   xc_ring_lend     = false;
    // C042 small tail on the async CPU-pool cold path (2026-10-19; llama.cpp sets it when PXA_XCACHE_ASYNC is on and
    // PXA_XCACHE_SMALLTAIL_COST is not =0): inside the small-tail scope the R4 per-layer price is PXA_XC_SMALLTAIL_ASYNC_LAYER_US
    bool   xc_async_cost    = false;
    double xc_miss_scale    = 1.0;    // online adaptation (src/llama-pxa-xcache.cpp) lowers the share of routings that really miss: the R4 miss price is scaled by this
    bool   xc_cold_pxqn     = false;  // cold path is the CPU alias AND the experts are PXQN read by the library's CPU matmul (own cost row)
    size_t floor_bytes      = 0;      // pxa_place_floor_bytes(): VRAM that must stay free after the pool
    // PXA_XCACHE_PLAN_V2 (2026-10-04; default OFF = the plan above byte for byte): the per-cold-layer cost refit to the
    // measured value and a verify-width term. A speculative round decodes xc_verify_width tokens in ONE target pass: the fixed cost of a
    // cold layer is paid once per round while every distinct cold expert the round routes to is read once, so the miss price per round
    // grows with the width (pxa_xc_width_factor) and the plan prefers fewer misses over fewer cold layers.
    bool   xc_plan_v2       = false;
    // PXA_XCACHE_MTP: VRAM per device that the speculative chain needs AFTER the weights (the per-step recurrent checkpoint rows of one
    // verify pass and the MTP companion context's buffers). Only the streaming branch of the planner pays it (a model that fits resident
    // keeps today's plan); without it a full card leaves the draft-length budget (llama_spec_ckpt_budget) nothing and the chain is clamped.
    std::vector<size_t> spec_reserve;
    double xc_verify_width  = 1.0;    // tokens per target decode of the planned run: 1 + the draft depth of an MTP chain, 1 without speculation
    // bug #339 (2026-10-21; llama.cpp sets it, PXA_PLACE_CTX_GUARD=0 = off; a hand-built input keeps the old plan): a streaming plan must
    // leave, by the planner's own estimate, the floor (+ spec_reserve + the device's ctx_logits + ctx_head) free after the pool, which is
    // the test the context applies (pxa_pool_pregrow) plus what an all-outputs context needs on top. The plan is chosen as before and
    // re-chosen with the guard as its target only when it falls short, so every plan that already met it is unchanged. The C042
    // small-tail auto slack (xc_slack_auto) gives back only the part of the 512 MiB margin the context does not need (minus ctx_*).
    // R1 (fits resident) adds ctx_logits only: an all-outputs context reserves the full logits, so it fits only if they fit; the
    // head term stays out of R1, so every resident plan the old planner made with the full logits is exactly as before.
    bool   ctx_guard        = false;
};
// bug #339: the head GEMM's pool need at an all-outputs width, from the library's estimate `lib_need`
// (ggml_backend_cuda_mul_mat_pool_need): a quantized head whose whole f16 copy (2*K*N) exceeds PXA_CUBLAS_SRC0_SLICE_MIB
// (`slice_bytes`, 0 = off) is converted in row slices (rows = slice/(2K) rounded down to 64, at least 64) plus the f16 copy
// of the activations (2*T*K), not as one copy (ggml-cuda.cu, PXA_CUBLAS_SRC0_SLICE).
size_t pxa_place_head_pool_need(size_t lib_need, int64_t K, int64_t N, int64_t T, bool quantized, size_t slice_bytes);

// VRAM that must stay free after the weights, KV, compute buffer and the pre-grown GEMM pool (bug
// #266) for lazily loaded kernels and cuBLAS: PXA_POOL_FLOOR_MB, default PXA_POOL_FLOOR_MB_DEFAULT.
// The planner's "fits resident" test and the context's refusal use the same number, so auto never
// plans a placement the context then refuses, and never refuses a config the old binary ran.
#define PXA_POOL_FLOOR_MB_DEFAULT 128
// the C042 small-tail rule's slack credit (pxa_place_input::xc_slack_auto): 384 of the 512 MiB + 5% margin go back to hot experts;
// the 35B-A3B run left ~660 MiB free after the load without it, so ~270 MiB stay free above the floor
#define PXA_XCACHE_SLACK_AUTO_MB 384
size_t pxa_place_floor_bytes();

struct pxa_place_result {
    int         mode = 0;             // 0 resident, 2 experts, 3 spill
    // PXA_XCACHE: per layer, the expert ids that leave VRAM (ascending); empty = layer not split.
    // A layer listed here is split expert-wise; its tensors are NOT in names (whole-tensor streaming).
    std::vector<std::vector<int32_t>> xc_cold;
    double      xc_miss = 0;          // routed mass that lands on cold experts (per the counts)
    int         xc_cap  = 0;          // per-layer cold cap the search settled on
    int         xc_layers = 0;        // layers holding a cold expert (R4: the L least-hot layers)
    double      xc_cost_layer_us = 0; // R4 cost constants used (last planned device), for the load log
    double      xc_cost_miss_us  = 0; //   us per cold layer / us per percent of routings missed
    bool        cpu_instead = false;  // R5: selected tensors go to mmap'd host RAM, no ring
    std::vector<std::string> names;   // tensors that leave VRAM
    std::vector<size_t> dev_stream;   // bytes streamed per device
    std::vector<size_t> dev_deficit;  // bytes that did not fit per device (0 = fits)
    std::vector<size_t> dev_ring;     // ring bytes per device
    bool        xc_ring_lent = false; // R3a: dev_ring is lent (not charged by the plan; the backend allocates it on demand)
    size_t      streamed = 0;
    std::string reason;
};

// the ring the loader allocates for a stream set: 2 x the largest per-layer group + the largest
// tensor + 32 MiB (src/llama-load-tensors.cpp)
size_t pxa_place_ring_bytes(const std::vector<pxa_place_tensor> & sel);

pxa_place_result pxa_place_plan(const pxa_place_input & in);

// one JSON object for PXA_EXPLAIN / the boot banner
std::string pxa_place_json(const pxa_place_result & r);

// PXA_XCACHE: choose the cold (layer, expert) set on one device. Candidates are the routed-expert
// tensors of each layer; one expert of layer l costs expert_bytes[l] (all its expert tensors) and
// max_tensor_bytes[l] of that in its largest single tensor. The ring the loader allocates for the
// cold stacks is ring_layers (2, or 1 under the C042 small-tail rule) x the largest per-layer cold group
// + the largest cold tensor + 32 MiB; 2 is exactly the whole-layer rule. Returns false when no selection frees `target` bytes net of that ring.
// ring_layers -1 / -2 (R3a, pxa_place_input::xc_ring_lend): the same ring size is reported in `ring` but not charged against `target`.
struct pxa_xc_choice {
    std::vector<std::vector<int32_t>> cold;   // per candidate layer (index into `layers`)
    size_t cold_bytes = 0, ring = 0;
    double miss = 0, total = 0;
    int    cap  = 0;
    int    n_layers = 0;                      // layers with a cold expert
    double cost_us  = 0;                      // R4: predicted decode cost per token of the cold path
};
// R4 decode-cost constants: c_layer_us per layer holding a cold expert, c_miss_us per percent of the
// device's routings that miss. See pxa_xc_cost() in llama-pxa-place.cpp for the table and its data.
struct pxa_xc_cost { double layer_us = 161.0, miss_us = 1420.0; };
// cc = compute capability x10 of the device, zc = zero-copy cold path, n_used = routed experts per
// token, expert_bytes = the device's candidate layers' per-expert bytes. PXA_XCACHE_COST_LAYER_US /
// PXA_XCACHE_COST_MISS_US override the result.
pxa_xc_cost pxa_xc_cost_for(int cc, bool zc, int n_used, const std::vector<size_t> & expert_bytes, bool pxqn_cpu = false);
// PXA_XCACHE_PLAN_V2: the miss price of a round that verifies `width` tokens relative to a one-token decode: distinct cold experts read per
// round / per token. Independent routing would give `width`; successive tokens of one answer route to overlapping experts, so the
// default keeps 75% of the extra width (PXA_XCACHE_VERIFY_KAPPA overrides): 1 + kappa x (width - 1).
double pxa_xc_width_factor(double width);
bool pxa_xc_choose(const std::vector<int> & layers, const std::vector<size_t> & expert_bytes,
        const std::vector<size_t> & max_tensor_bytes, const std::vector<std::vector<double>> & counts,
        int n_expert, size_t target, pxa_xc_choice & out, const pxa_xc_cost & cost = pxa_xc_cost(), int ring_layers = 2);
// the C042 small-tail scope (pxa_place_input::xc_ring1_auto): one device, sm_70 (sm_60 too with PXA_XCACHE_SMALLTAIL_SM60=1), CPU cold
// path, deficit <= free/2
bool pxa_xc_small_tail(size_t n_dev, int cc, bool cold_zc, size_t deficit, size_t free_bytes);

// Cold path for streamed routed experts (whole-layer or PXA_XCACHE cold stacks): does this ggml
// type have a CPU matmul fast enough to beat a zero-copy PCIe read? False for the PXQN family and
// PXQ6 (CPU route = generic panel dequant; Flash-Next PXQN on 2x P100: CPU cold 2.90 t/s vs
// zero-copy 12.43), true for everything else incl. PXQ1-PXQ4 (Ornith-35B PXQ4 on one V100: CPU
// cold 48.3 vs zero-copy 19.5 at L10).
bool pxa_type_cpu_fast(int ggml_type);

// PXA_XCACHE_MTP chain reserve, bytes per recurrent Gated-DeltaNet layer at `width` tokens per verify pass (nv value heads of head_v,
// key_dim = d_state x n_group, d_conv taps). Per-step: width-1 fp32 state rows + width conv rows (llama_kv_cache::per_step_alloc).
// replay (PXA_MTP_RESERVE_RECOMPUTE=1, R3d): one start state + width rows of the recurrence inputs (q, k, v, g, beta) + width conv
// rows, the PXA_SPEC_CKPT_REPLAY layout (pxa_replay_pack_elems). Flash-Next (nv 48, head_v 128, key_dim 2048, d_conv 4) at width 4:
// 9.47 MiB per-step vs 3.63 MiB replay, x 36 recurrent layers = 340.9 vs 130.5 MiB.
inline size_t pxa_spec_reserve_deltanet_layer(int64_t width, int64_t nv, int64_t head_v, int64_t key_dim, int64_t d_conv, bool replay) {
    const int64_t conv_row = (d_conv - 1)*(key_dim*2 + head_v*nv);
    const int64_t ssm_row  = head_v*head_v*nv;
    const int64_t in_row   = 2*key_dim + head_v*nv + 2*nv;
    return (size_t)(replay ? ssm_row + width*(in_row + conv_row) : (width - 1)*ssm_row + width*conv_row)*sizeof(float);
}
