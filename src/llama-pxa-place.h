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
    size_t xc_extra_compute = 0;
    int    n_expert_used    = 0;      // routed experts per token (R4 miss cost = bytes moved per token)
    bool   xc_cold_zc       = false;  // cold path: true = zero-copy GPU read, false = CPU alias (pxa_cold_path_zc)
    size_t floor_bytes      = 0;      // pxa_place_floor_bytes(): VRAM that must stay free after the pool
};

// VRAM that must stay free after the weights, KV, compute buffer and the pre-grown GEMM pool (bug
// #266) for lazily loaded kernels and cuBLAS: PXA_POOL_FLOOR_MB, default PXA_POOL_FLOOR_MB_DEFAULT.
// The planner's "fits resident" test and the context's refusal use the same number, so auto never
// plans a placement the context then refuses, and never refuses a config the old binary ran.
#define PXA_POOL_FLOOR_MB_DEFAULT 128
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
// cold stacks is 2 x the largest per-layer cold group + the largest cold tensor + 32 MiB, exactly
// the whole-layer rule. Returns false when no selection frees `target` bytes net of that ring.
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
pxa_xc_cost pxa_xc_cost_for(int cc, bool zc, int n_used, const std::vector<size_t> & expert_bytes);
bool pxa_xc_choose(const std::vector<int> & layers, const std::vector<size_t> & expert_bytes,
        const std::vector<size_t> & max_tensor_bytes, const std::vector<std::vector<double>> & counts,
        int n_expert, size_t target, pxa_xc_choice & out, const pxa_xc_cost & cost = pxa_xc_cost());

// Cold path for streamed routed experts (whole-layer or PXA_XCACHE cold stacks): does this ggml
// type have a CPU matmul fast enough to beat a zero-copy PCIe read? False for the PXQN family and
// PXQ6 (CPU route = generic panel dequant; Flash-Next PXQN on 2x P100: CPU cold 2.90 t/s vs
// zero-copy 12.43), true for everything else incl. PXQ1-PXQ4 (Ornith-35B PXQ4 on one V100: CPU
// cold 48.3 vs zero-copy 19.5 at L10).
bool pxa_type_cpu_fast(int ggml_type);
