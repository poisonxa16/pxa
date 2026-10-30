// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
#pragma once

// =============================================================================================
// PXA core step 3: the lever registry and the engine's own serve-flag defaults.
//
// ONE table declares every PXA_* / PXQ_* lever in the tree (generated into pxa-lever-catalog.inc
// by scripts/pxa-lever-catalog.py): its default at the shipping level, the scope it applies to
// (arch / tier / card class / split mode), its status and the ledger row that is its evidence.
//
// The serve flags a user usually has to know by heart (-sm, -b, -ub, -fa) are picked HERE, from
// rules keyed on the card set and the model file, when the command line does not set them. The
// launcher asks the engine for the same answer (PXA_EXPLAIN=1), so `llama-server -m file`, the
// docker image and pxa-launch get the same defaults on the same cards: the defaults live in the
// engine, not only in the launcher.
//
// Explicit flags always win. PXA_REFERENCE=1 stands every rule down; PXA_ENHANCE=0 stands down the
// batch table (as before) and the split rule.
// =============================================================================================

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct pxa_lever_decl {
    const char * name;      // the environment variable
    const char * deflt;     // default at the shipping (ENHANCE) level, as text
    const char * scope;     // what it applies to: arch / tier / card class / split mode, "any" if global
    const char * status;    // default-on | lever-off | rule | diagnostic | tool | site
    const char * evidence;  // ledger row id, bug id, or ""
    const char * rule;      // one line: when the default engages / where the lever is read
};

const pxa_lever_decl * pxa_lever_catalog(size_t * n);
const pxa_lever_decl * pxa_lever_find(const char * name);

// ---- the card set -----------------------------------------------------------------------------
struct pxa_topology {
    int                 n_dev = 0;
    std::vector<int>    cc;          // 600, 610, 700, ...
    std::vector<size_t> vram_mib;    // total per device, 0 when unknown
    std::vector<int>    pcie_width;  // PCIe link lanes per device, 0 / missing = unknown (2026-09-27)
    std::string         source;      // "cuda" | "PXA_TOPOLOGY" | "none"
    bool same_cc() const;
    int  narrow_link() const;        // index of the first device on a link below x4, -1 = none known
};

// With allow_override (explain runs only), PXA_TOPOLOGY ("2x600", "700,700", "600:16384,600:16384")
// wins and no CUDA device is touched; otherwise the CUDA backend is asked (device count and compute
// capability only). `devices` is gpt_params::devices (-dev), `max_gpu` --max-gpu.
pxa_topology pxa_topology_detect(const std::string & devices, int max_gpu, bool allow_override);
pxa_topology pxa_topology_parse(const char * spec);

// ---- the model file ---------------------------------------------------------------------------
struct pxa_model_info {
    bool        ok        = false;
    std::string path;
    std::string arch;
    std::string tier;        // PXQ1 | PXQ2 | PXQ3 | PXQ4 | PXQ4-HQ | PXQ6 | PXQ_UNIVERSAL | "" (not PXQ)
    int         n_expert  = -1;   // >0 experts, 0 dense, -1 unknown
    int         n_head_kv = -1;   // smallest per-layer KV head count, -1 unknown
    int         n_layer   = -1;   // <arch>.block_count, -1 unknown
    int         head_k    = -1;   // <arch>.attention.key_length, -1 unknown
    int         head_v    = -1;   // <arch>.attention.value_length
    int         full_attn_interval = -1;   // hybrid arches: every Nth layer holds KV, -1 unknown
    int         n_ctx_train = -1; // <arch>.context_length, -1 unknown
    int         n_nextn   = -1;   // <arch>.nextn_predict_layers (MTP head layers), 0 none, -1 unknown
    int         n_pxq     = 0;    // tensors carrying a PXQ codec
    uint32_t    pxq_mask  = 0;    // bit (ggml type id - 248) of every PXQ tensor type present (259 PXQN4 = bit 11, 257 PXQN3 = bit 9, 263 PXQN5 = bit 15)
    int         n_shards  = 1;
    uint64_t    bytes     = 0;    // over every shard
    std::vector<std::pair<std::string, std::string>> pxa_kv;   // the file's pxa.* keys, as text
};

// Metadata-only open of the GGUF header (and every shard of a split file). No tensor data read.
pxa_model_info pxa_model_probe(const std::string & path);

// ---- what the user already set ----------------------------------------------------------------
struct pxa_user_set {
    bool sm = false, b = false, ub = false, fa = false, ts = false;
    bool ngl         = false;   // -ngl given
    bool ctx         = false;   // -c given (flag or LLAMA_ARG_CTX_SIZE)
    int  n_parallel  = 1;       // -np in effect (the -c pick is np * 4096, the launcher's anchor)
    bool ngl_partial = false;   // -ngl given and smaller than "everything"
    bool fa_value    = true;    // the -fa value in effect (explicit or posture)
    int  n_ctx_value = 0;       // the -c value when given (0 = unset / trained window)
    bool kv_q8       = false;   // -ctk q8_0 and -ctv q8_0 both in effect
    bool kv          = false;   // -ctk or -ctv given (flag or LLAMA_ARG_CACHE_TYPE_K/V)
    int  sm_value    = -1;      // the -sm in effect when given (llama_split_mode: 1 layer, 4 tensor, ...)
};

// ---- one cost line ----------------------------------------------------------------------------
// An explicit flag or environment value that is HONORED but was measured slower than what ENHANCE
// would have picked. The banner prints each as ONE "PXA_REGISTRY: COST" line naming the cost and
// the better value; nothing about the user's value is changed.
struct pxa_cost {
    std::string what;       // "-sm layer", "PXA_TSPLIT_REDUCE=off", ...
    std::string evidence;   // ledger row id
    std::string text;       // what it costs and the better value
};

// ---- one decision -----------------------------------------------------------------------------
struct pxa_pick {
    std::string flag;       // "-sm" | "-b" | "-ub" | "-fa" | "-ngl"
    std::string value;      // "tensor", "8192", "adaptive", "on", ... ("" = left to the user's value)
    std::string status;     // MEASURED | INFERRED | RULE | FIT | USER | ADAPTIVE | OFF
    std::string evidence;   // ledger row id / bug id
    std::string why;
    bool        applied = false;   // the engine will set this flag (it was not given)
};

enum pxa_sm_pick { PXA_SM_KEEP = -1, PXA_SM_LAYER = 1, PXA_SM_TENSOR = 4 };

struct pxa_autoconfig {
    int  split       = PXA_SM_KEEP;   // PXA_SM_LAYER / PXA_SM_TENSOR, or KEEP (user set / stood down)
    int  n_batch     = 0;             // 0 = no measured cell (keep / adaptive)
    int  n_ubatch    = 0;
    bool ub_adaptive = false;         // no cell: the engine's VRAM ladder decides at load
    int  flash_attn  = -1;            // -1 keep, 0 off, 1 on
    int  n_gpu_layers = -1;           // -1 keep; 999 = every layer on the cards (-ngl unset, CUDA present)
    int  n_ctx       = 0;             // 0 keep; else the -c the engine picked (-c unset, ENHANCE)
    bool ts_even     = false;         // tensor split picked: set -ts 1,1,...
    const char * kv_type = nullptr;   // a KV cache type the engine picked for -ctk and -ctv (null = keep f16)
    const char * ub_rule = nullptr;   // a -ub RULE replaced the adaptive ladder (n_ubatch holds it)
    int  ub_fit_ceiling = 0;          // a FIT rule (pxa_registry_ub_fit_rule): -ub walks down from this at load to
                                      // n_ubatch, the measured cell (0 = none)
    std::string batch_cell;           // the -b/-ub cell's why (empty = none). OWNED copy: pxa_registry_batch_cell may hand back
                                      // a static buffer that a later resolve (pxa_autoconfig_costs' -sm layer re-resolve) rewrites
    int  batch_cell_ctx = 0;          // the cell's own -c (0 = none); see pxa_registry_batch_cell
    std::vector<std::pair<std::string, std::string>> env;   // set only if unset (setenv overwrite=0)
    std::vector<pxa_pick> picks;
    std::vector<pxa_cost> costs;      // explicit known-worse values (ENHANCE only), see pxa_cost
};

// PURE apart from reading PXA_AUTO_SM / PXA_AUTO_UB_LONG / PXA_AUTOCONFIG (=0 resolves as REFERENCE: the whole
// registry off in every tool): no file or device I/O. level = ggml_pxa_config_level() (0 REFERENCE, 1 DEFAULT, 2 ENHANCE),
// posture = 0 balance, 1 max. Unit-tested over a case table in tests/test-pxa-autoconfig.cpp.
pxa_autoconfig pxa_autoconfig_resolve(const pxa_topology & topo, const pxa_model_info & model,
                                      const pxa_user_set & user, int level, int posture);

// The measured batch cell alone (what the server's batch block applies). 0 = no cell: a launcher
// row marked INFERRED elsewhere is not a cell (the engine keeps its old table / VRAM ladder
// there). *status, when given, is "MEASURED", "INFERRED" (the gemma4 PXQ3/PXQ_UNIVERSAL pair and the
// expert PXQ4/PXQ6 flagship on a V100 pair, whose launcher rows are INFERRED - see the function body)
// or "RULE" (the qwen4exp off-topology rule). *n_ctx, when given and the
// return is non-zero, is the cell's own measured/planned -c (0 = the cell has none, and
// pxa_autoconfig_resolve falls back to np*4096) - the SAME number tools/pxa-launch.py's matching
// recipe row passes, so the engine and the launcher agree instead of running two independent
// formulas that can drift apart (#8382 problem 1).
int pxa_registry_batch_cell(const pxa_topology & topo, const pxa_model_info & model, int level,
                            int * n_batch, int * n_ubatch, const char ** why,
                            const char ** status = nullptr, int * n_ctx = nullptr);

// Bug #207 rule (PXA_AUTO_UB_VOLTA_Q8, default on at ENHANCE): one V100 (sm_70), q8_0 K and V,
// -c >= 49152 and no measured batch cell -> -ub 1024. At -ub 2048 the compute buffer leaves the
// card ~244 MiB free and the Volta MMA q8 attention route (fattn-volta-mma.cu, which keeps a
// 256 MiB margin for its f16 staging) declines, so attention falls back to the slow route:
// 387.5 t/s at pp65536 against 642.3 at -ub 1024 (27B PXQ3, b-stream arms p64-R, REPS 3).
// Returns the -ub (1024) or 0 when the rule does not apply. =0 stands it down.
int pxa_registry_ub_rule_volta_q8(const pxa_topology & topo, int level, int n_ctx, bool kv_q8,
                                  const char ** why);

// ---- an ENGINE-picked -ub has to fit (bug #207 follow-up, ws6-fix 2026-09-25) --------------------
// The rule above picked -ub 1024 on its own target -- one V100, the 12.46 GiB Qwen3.8-27B
// PXQ3-balanced file, -c 65536 q8_0 -- where the context then allocated with 27 MiB left on the card
// and the server aborted on its first decode (CUDA out of memory), while the ladder's -ub 2048 had
// refused cleanly at load. An allocation that succeeds is not a context that runs: the CUDA pool
// (the dequantised cuBLAS operand, up to PXA_CUBLAS_SRC0_SLICE_MIB = 256 MiB, plus the f16 K/V
// staging of a quantized cache), cuBLAS and lazily-loaded kernels all take VRAM AFTER load. So an
// -ub the ENGINE chose is verified at load, the way PXA_AUTO_CTX verifies a context size: by asking
// the allocator, then reading the cards' free VRAM. Where it came from decides how strictly:
enum pxa_ub_origin {
    PXA_UB_KEPT   = 0,   // the user's -ub, or a measured batch cell: never touched
    PXA_UB_RULE   = 1,   // a registry RULE (PXA_AUTO_UB_VOLTA_Q8): must allocate AND leave the reserve
    PXA_UB_LADDER = 2,   // the VRAM ladder's guess: a first pick that allocates is kept as before (it
                         // cannot change a boot that works today); a pick that does NOT allocate steps
                         // down, and every rung below it must leave the reserve
    PXA_UB_FIT    = 3,   // a registry FIT rule (pxa_registry_ub_fit_rule): -ub is a CEILING. The model is
                         // loaded per rung, largest first; a rung must plan a RESIDENT placement, allocate
                         // its context and leave the reserve free, else the next rung down is tried. The
                         // last rung is the measured cell (pxa_ub_fit_floor) and loads exactly as before.
};

// Gemma 4 expert file (26B-A4B) on ONE P100 or ONE V100 (2026-10-16): the
// measured cell's -ub (256 / 512) is the FLOOR of a fit walk whose ceiling is -ub 1024, instead of a fixed
// -ub. With the footprint set -ub 1024 boots a 16k prompt resident on a 16 GB card:
// server prefill at 16k, P100 322 -> 570 t/s (+77%), V100 2275 -> 2881 (+26.6%); at 32k the walk lands on
// 768 or 512, where 1024 does not leave the reserve. -ub 2048 is not a rung: slower than 1024 on a V100 at
// pp4096 (3007 vs 3401) and it does not fit a 16k prompt -- EXCEPT a PXQN4 file on 1x V100 with an f16 KV cache
// (2026-10-17, tensor-core expert GEMM): there the ceiling is 2048 (pp16384 +8.3%; PXA_AUTO_UB_GEMMA_2048=0 for 1024;
// kv_quant = the KV cache is not f16, which keeps 1024). Returns the ceiling (2048 / 1024), or 0 when the rule
// does not apply (another file or card set, a cell already at >= 1024, below ENHANCE, or PXA_AUTO_UB_FASTEST
// set by hand: =0 keeps the measured cell, =1/=N runs the generic walk). The server applies it only when
// -ub was not given; the walk itself is common.cpp pxa_fastest_ub_init.
int pxa_registry_ub_fit_rule(const pxa_topology & topo, const pxa_model_info & model, int level, int cell_ub,
                             const char ** why, bool kv_quant = false);

// PXA_AUTO_UB_DENSE_2048 (default ON, =0 off; 2026-10-18, ledger auto-ub-dense27-v100): the 27B dense PXQN4
// file (arch qwen35, no experts, 64 blocks or 65 with the nextn block) on ONE V100 or (2026-10-18) ONE P100 with an
// f16 KV cache, -ub AND -b both unset. -ub 2048 is a CEILING:
// the context is built at 2048, then 1024, 768, down to the stock -ub 512, and the first rung that allocates beside the
// weights and the KV cache at the requested -c AND leaves PXA_AUTO_UB_RESERVE_MB free is kept; -b is raised to it. The stock
// -ub 512 is the floor and is built exactly as without the rule. Measured (llama-bench, card 4, REPS 2): pp4096 986.5 ->
// 1127.8 t/s at -ub 2048 (+14.3%); at -c 16384 -ub 2048 does not allocate (2.0 GB compute buffer). Not bit-identical (another
// -ub is another accumulation split), correct: identity rule -> default ON with this off switch. 1x P100 (llama-bench, card 0,
// ABBA vs v3.1 same bracket, #23355): pp4096 248.0 -> 253.4 (v3.1 244.7, r 1.036), pp16384 224.1 -> 230.8 (-ub 2048 allocates
// at 16k on the P100: the PF16K long-prefill path keeps the compute buffer small; v3.1 210.9, r 1.094). Returns 2048, or 0 when the
// rule does not apply (another file, card set or KV type, below ENHANCE, PXA_AUTO_UB_FASTEST set by hand, or =0). The
// server applies it with no -ub/-b given and no measured cell (examples/server/server.cpp, walk: common.cpp
// pxa_fastest_ub_init); llama-bench per test instance (examples/llama-bench, pxa_registry_ub_walk with floor 512).
int pxa_registry_ub_fit_dense(const pxa_topology & topo, const pxa_model_info & model, int level, bool kv_quant,
                              const char ** why);

// PXA_AUTO_UB_STREAM_DECODE (default ON, =0 off; 2026-10-18, C042, ledger c042-stream-ub-decode): the server's
// PXA_AUTO_UB_STREAM picks -ub 4096 when one card streams a file's MoE experts (Flash-Next: most experts stream, prefill pays the
// PCIe re-cross per ubatch). A Qwen3.5/3.6 MoE PXQN4 file (qwen35moe) on ONE V100 that overflows the card by a small tail
// (file <= 1.5 x free VRAM) is the other case: only a few hundred MiB of experts stream, decode reads them on the CPU, and the
// 4096 compute buffer (+404 MiB) and split graph (+160 MiB) push ~350 more experts out of VRAM. There -ub 2048 is decode-sized:
// 35B-A3B 1x V100 decode 56.5 (-ub 4096) -> 61.8 t/s (-ub 2048; 67.9 with the xcache one-layer ring + slack, v3.1 57.1-60.1),
// prefill stays above v3.1 (1286 vs 1032 t/s at a 3.5k prompt). Returns 2048 or 0 (another file, card set, or =0); only asked
// when PXA_AUTO_UB_STREAM is unset or 1 (an explicit =N wins).
int pxa_registry_stream_ub_decode(const pxa_topology & topo, const pxa_model_info & model, uint64_t file_bytes,
                                  uint64_t free_bytes, const char ** why);

// PXA_STREAM_UB_CKPT (default OFF, =1 on; 2026-10-18, the 1x V100 Flash-Next default decode): the
// -ub PXA_AUTO_UB_STREAM picked (4096) is a CEILING for the context, checked against the speculative chain at load. On one
// 16 GB V100 the Flash-Next 4096 compute buffer is 1800 MiB; the per-step checkpoint budget keeps 256 MiB + 25% of it free
// (PXA_CKPT_BUDGET_COMPUTE_PCT) and so affords 2 of the 4 rows the MTP chain needs: n_max 3 -> 1, decode 40.9 -> 26.7 t/s
// prose. The walk rebuilds the CONTEXT (not the model: the placement stays as planned) at -ub - 512 ... down to floor_ub
// until the budget affords `target` rows (min(chain + 1, PXA_XCACHE_VERIFY_WIDTH = the width the planner reserved)), else
// keeps the rung with the most rows (the larger -ub on a tie). 3584 still runs an 8k prompt in two ubatches. An explicit
// -ub is never walked (the user's choice stands). OPT-IN: card 4 r3 (load 18-30, NOT-QUIET, N=2) walk 3584 vs 4096: decode prose
// 29.08/34.85 vs 27.04/34.48, code 37.54/41.06 vs 36.28/37.57 (+1..+9%, greedy sha equal), pp8k 398.5 vs 400.3; a prompt of
// (3584k, 4096k] tokens pays one more ubatch (28k: 7 instead of 6). Default ON needs a quiet A/B that clears the prefill cost.
//   ub0 / cap0: the -ub of the context that exists and the rows its budget affords. cap_at(ub) rebuilds the context at ub
//   and returns its rows, or 0 when it does not allocate (the walk stops there). trail: one "-ub N rows R;" per rung tried.
// Returns the -ub to keep (ub0 when nothing is better).
bool pxa_registry_stream_ub_ckpt_on();
int  pxa_registry_stream_ub_ckpt_walk(int ub0, int cap0, int target, int floor_ub,
                                      const std::function<int(int)> & cap_at, std::string * trail);

// PXA_AUTO_XCACHE_SMALLTAIL (default ON, =0 off; 2026-10-19, C042/C114/C115): the
// class of pxa_registry_stream_ub_decode (a qwen35moe PXQN4 file on ONE V100 that overflows it by a small tail, here measured against
// the card's total VRAM: VRAM < file <= 1.5 x VRAM). The resolver then arms the expert cache's async CPU-pool cold path
// (PXA_XCACHE_ASYNC, PXA_XCACHE_CPUPOOL, PXA_XCACHE_CPUPOOL_CPUS derived like the Overdrive cell, PXA_XCACHE_IDLE_BACKOFF_MS=200);
// the planner's small-tail async cost (src/llama-pxa-place.cpp) then spreads the cold experts over more, thinner layers.
bool pxa_registry_xcache_smalltail(const pxa_topology & topo, const pxa_model_info & model, const char ** why);

// PXA_AUTO_FLINT_1CARD (default ON, =0 off; 2026-10-23): a Flint PXQN4 file (arch flint) on ONE P100 with
// VRAM < file <= 1.5 x VRAM. The resolver sets PXA_XCACHE_SMALLTAIL_SM60=1: the expert-cache planner's C042 small-tail scope (one-layer
// ring + the auto slack credit, sm_70 only in generic code) admits that P100.
bool pxa_registry_flint_1card(const pxa_topology & topo, const pxa_model_info & model, const char ** why);

// PXA_AUTO_FLINT_ROUTER (default ON, =0 off; PXA_AUTO_FLINT_1CARD=0 also turns it off; 2026-10-23): the same Flint
// PXQN4 file on ONE P100 or V100 (VRAM < file <= 1.5 x VRAM). The resolver sets PXA_ROUTER_BF16=2: the BF16 router GEMV on an F32
// widening of the row (one kernel, F32 logits) instead of the cuBLAS BF16 GEMM and its two conversions. Not bit-identical.
bool pxa_registry_flint_1card_router(const pxa_topology & topo, const pxa_model_info & model, const char ** why);

// PXA_AUTO_FLINT_V100 (default ON, =0 off; PXA_AUTO_FLINT_1CARD=0 also turns it off; 2026-10-23): the same Flint
// PXQN4 file on ONE V100 (VRAM < file <= 1.5 x VRAM). The resolver arms the one-V100 plain-decode stack: PXA_CUDA_GRAPH_DECODE=1,
// PXA_CUDA_GRAPH_MOE=1, PXA_CUDA_GRAPH_LRU=256, PXA_SCHED_ASYNC_INPUTS=1, PXA_PLAIN_GREEDY_AMAX=1, PXA_ARGMAX_VAL_MB=1 (bit-identical).
bool pxa_registry_flint_1card_v100(const pxa_topology & topo, const pxa_model_info & model, const char ** why);

// PXA_AUTO_UB_RESERVE_MB (default 512; 0 turns the check off): VRAM that must still be free on every
// card the context uses once its weights, KV cache and compute buffers are allocated. 512 is the
// measured floor PXA_CKPT_BUDGET_FLOOR_MB already stands on (16 GB V100, Qwen3.8-27B: 256 MiB left
// died inside ggml_cuda_pool_vmm::alloc, 420 MiB survived) and it covers the pool's high-water on this
// shape (a 256 MiB cuBLAS operand slice or a <= 128 MiB chunked K/V staging, whichever op is running).
size_t pxa_registry_ub_reserve_bytes();

// The next rung below `ub` on the ladder 2048 > 1024 > 768 > 512 > 256; 0 when there is none.
int pxa_registry_ub_next_rung(int ub);

// PXA_SPEC_STREAM_MTP (default ON, =0 off): no AUTO speculation for Nemotron-H MoE (arch nemotron_h_moe, Nemotron 3.5 Super)
// when the file carries an MTP head and its routed experts STREAM from host memory -- the server boots as with
// --spec-type none. MEASURED 2026-10-17 (#22519): Super-VL PXQ4 on 2x P100, experts streamed
// whole-tensor (PXA_XCACHE=0), MTP n_max 1 vs 0 = prose 5.02 vs 5.61, code 5.19 vs 5.58 t/s (about -10%): a 2-token verify
// reads ~2x the routed expert bytes over PCIe. Streamed = PXA_STREAM_WEIGHTS forces it, or the file does not fit the cards
// and either PXA_XCACHE=0 or the expert cache declines because the head is loaded (it does unless PXA_XCACHE_MTP=1).
// Not in scope (unchanged): experts resident on the cards, -ot CPU expert placement, PXA_XCACHE_MTP=1 (cache + chain).
// Only the AUTO layer stands down: an explicit --spec-type / -md / PXA_AUTO_SPEC=1 still gets what it asked for.
// glm5next (2026-10-18): the same function also keeps AUTO speculation off for GLM-5.3-Flash whenever its routed
// experts are on the host (-ot CPU placement included, unlike Nemotron-H): PXA_GLM5NEXT_MTP=1 arms it there, =0 everywhere off.
struct pxa_spec_stream_in {
    std::string arch;
    bool has_head      = false;   // nextn_predict_layers > 0
    bool fits          = true;    // the file's device share fits the free VRAM of every card
    bool cpu_override  = false;   // -ot puts tensors on the CPU (experts computed there, not streamed)
    bool xcache_off    = false;   // PXA_XCACHE=0
    bool stream_forced = false;   // PXA_STREAM_WEIGHTS != 0
    bool xcache_mtp    = false;   // PXA_XCACHE_MTP != 0
};
// true: the auto speculation layer must not arm anything (why = the banner text)
bool pxa_registry_spec_stream_off(const pxa_spec_stream_in & in, std::string * why);
// has_head for the gate above: the trunk header carries a NextN head (nextn_in_header > 0) or one is attached at load
// (head_file = PXA_MTP_HEAD_FILE, non-empty): Nemotron 3.5 Super ships its head as a separate mtp-*.gguf (todo nemo-xcache-mtp).
bool pxa_registry_spec_head_present(int nextn_in_header, const char * head_file);

// One context build at one -ub, as the walk below sees it.
struct pxa_ub_attempt {
    bool   allocated     = false;   // the context was created
    bool   alloc_failure = false;   // it was not, and it was MEMORY that ran out (a smaller -ub may fit)
    size_t free_min      = 0;       // allocated: the least free VRAM over the cards the context uses
    int    worst_dev     = -1;      // the card that has it (-1: no CUDA card involved)
};

// The walk, PURE apart from the two callbacks, so tests/test-pxa-autoconfig.cpp drives it with a
// table of outcomes: build(ub) creates a context at `ub` and reports; discard() frees the context the
// last build created. Starts at `ub`; returns the -ub whose context is kept, or 0 = a clean refusal
// (nothing kept: every rung failed to allocate or left less than `reserve`, or a build failed for a
// reason a smaller -ub cannot fix). `trail` collects one clause per rejected rung for the boot line.
// floor_ub > 0 (a FIT walk): the walk never goes below floor_ub (a rung under it is replaced by it), and the floor
// is built exactly as without the walk: kept whenever it allocates (no reserve check), a refusal when it does not.
int pxa_registry_ub_walk(int ub, int origin, size_t reserve,
                         const std::function<pxa_ub_attempt(int)> & build,
                         const std::function<void()> & discard, std::string * trail, int floor_ub = 0);

// The least free VRAM over the CUDA devices a context built from these flags uses: the -dev list when
// given (gpt_params::devices), else every visible device; `only_dev` >= 0 narrows it to one card
// (-sm none: the main GPU). False when no CUDA device is involved (or no CUDA build).
bool pxa_registry_cuda_free_min(const std::string & devices, int only_dev, size_t * free_min, int * worst_dev);

// Boot banner: registry size, levers set in the environment (and any that no row declares), and
// every pick with its reason. Printed once.
void pxa_registry_banner(FILE * out, const pxa_topology & topo, const pxa_model_info & model,
                         const pxa_autoconfig & ac);

// One JSON object (single line) with the topology, the model, the picks and the env: what
// PXA_EXPLAIN=1 prints and what pxa-launch --doctor / --explain read.
std::string pxa_autoconfig_json(const pxa_topology & topo, const pxa_model_info & model,
                                const pxa_autoconfig & ac, int level);

// PXA_EXPLAIN=levers: the whole catalog as TSV.
void pxa_lever_catalog_dump(FILE * out);

// ---- the registry in every tool: one process, several models, several tool kinds -------------------------
// (2026-10-16; on top of 56b1b9cb5e)
//
// PXA_AUTOCONFIG=0: the whole registry off in every tool, the server's direct batch/ub choices included
// (measured batch cell, fit walk, stream -ub, the VRAM ladder): stock -b/-ub, no picks, no lever env.
bool pxa_autoconfig_off();

// F1: the lever env is PROCESS-WIDE and most engine reads latch it once. Only the FIRST model of a process
// (the -m model) writes it: `primary` = true sets each pick whose variable is unset or EMPTY (PXA_X= counts as
// unset, so a compose template's blank no longer turns a default-on pick off). A later model (a hot model, a
// draft) writes nothing unless PXA_HS_ENV_UNION=1 (the old first-wins union, empty = unset); each of its picks
// whose value differs from the process value is returned in *differs ("NAME=pick (process: value|unset)").
// Returns the number of variables set.
int pxa_registry_env_apply(const std::vector<std::pair<std::string, std::string>> & env, bool primary,
                           std::vector<std::string> * differs);

// The server log is user-facing (pasted into bug reports): a lever only the closed PXQN library reads is never
// printed by name (2026-10-18). pxa_lever_is_closed: one of the five Overdrive members (libllama) or a
// name the loaded closed library declares; pxa_lever_shown: the name, or "an Overdrive library lever".
bool        pxa_lever_is_closed(const std::string & name);
std::string pxa_lever_shown(const std::string & name);

// F3(a): the tool kind from argv[0]'s basename. GATE = llama-perplexity (PPL and KLD) and llama-imatrix: the
// quality instruments get the lever env only, never a -c, KV, batch, split or -ngl pick (the instrument must
// measure what the user typed). BENCH = llama-bench (its flags are the measurement axes: lever env only).
// SERVER = llama-server (its own block). CLI = every other common tool (llama-cli, speculative, ...).
enum pxa_tool_kind { PXA_TOOL_CLI = 0, PXA_TOOL_GATE = 1, PXA_TOOL_BENCH = 2, PXA_TOOL_SERVER = 3 };
int pxa_tool_kind_of(const char * argv0);

// The user set of a lever-only resolve (GATE, BENCH, a later model): every placement / batch / attention / KV /
// context axis counts as given, so the resolve picks nothing but env; split_mode is the one in effect.
pxa_user_set pxa_user_set_lever_only(int split_mode, bool ts_given, bool kv_q8);

// F3(a) for a CLI tool: the -c pick the tool may take. Default -c 0 means the trained window, so a pick
// applies only when the user gave no -c and it does not shrink the trained window (n_ctx_train unknown: no
// pick). Returns the -c to set, 0 = keep.
int pxa_tool_ctx_pick(int pick_n_ctx, bool user_ctx, int n_ctx_train);

// F3(c): llama-bench resolves the lever env for EVERY test instance (model x -sm x -ts x KV); the reads latch
// once per process, so only the picks every instance shares (same name, same value) are applied; the others
// are returned in *dropped ("NAME=value (instance i)") and named in a warning (run one -sm / model per
// invocation for its own picks). One instance (or all alike) = exactly its picks.
std::vector<std::pair<std::string, std::string>> pxa_env_common(
        const std::vector<std::vector<std::pair<std::string, std::string>>> & per_instance,
        std::vector<std::string> * dropped);
