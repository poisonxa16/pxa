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
    int         n_ctx_train = -1; // <arch>.context_length, -1 unknown
    int         n_pxq     = 0;    // tensors carrying a PXQ codec
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
    std::string status;     // MEASURED | INFERRED | RULE | USER | ADAPTIVE | OFF
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
    const char * ub_rule = nullptr;   // a -ub RULE replaced the adaptive ladder (n_ubatch holds it)
    const char * batch_cell = nullptr;
    int  batch_cell_ctx = 0;          // the cell's own -c (0 = none); see pxa_registry_batch_cell
    std::vector<std::pair<std::string, std::string>> env;   // set only if unset (setenv overwrite=0)
    std::vector<pxa_pick> picks;
    std::vector<pxa_cost> costs;      // explicit known-worse values (ENHANCE only), see pxa_cost
};

// PURE apart from reading PXA_AUTO_SM / PXA_AUTO_UB_LONG: no file or device I/O. level = ggml_pxa_config_level() (0 REFERENCE, 1 DEFAULT, 2 ENHANCE),
// posture = 0 balance, 1 max. Unit-tested over a case table in tests/test-pxa-autoconfig.cpp.
pxa_autoconfig pxa_autoconfig_resolve(const pxa_topology & topo, const pxa_model_info & model,
                                      const pxa_user_set & user, int level, int posture);

// The measured batch cell alone (what the server's batch block applies). 0 = no cell: a launcher
// row marked INFERRED elsewhere is not a cell (the engine keeps its old table / VRAM ladder
// there). *status, when given, is "MEASURED", "INFERRED" (the gemma4 PXQ3/PXQ_UNIVERSAL pair - see
// the function body) or "RULE" (the qwen4exp off-topology rule). *n_ctx, when given and the
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
};

// PXA_AUTO_UB_RESERVE_MB (default 512; 0 turns the check off): VRAM that must still be free on every
// card the context uses once its weights, KV cache and compute buffers are allocated. 512 is the
// measured floor PXA_CKPT_BUDGET_FLOOR_MB already stands on (16 GB V100, Qwen3.8-27B: 256 MiB left
// died inside ggml_cuda_pool_vmm::alloc, 420 MiB survived) and it covers the pool's high-water on this
// shape (a 256 MiB cuBLAS operand slice or a <= 128 MiB chunked K/V staging, whichever op is running).
size_t pxa_registry_ub_reserve_bytes();

// The next rung below `ub` on the ladder 2048 > 1024 > 768 > 512 > 256; 0 when there is none.
int pxa_registry_ub_next_rung(int ub);

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
int pxa_registry_ub_walk(int ub, int origin, size_t reserve,
                         const std::function<pxa_ub_attempt(int)> & build,
                         const std::function<void()> & discard, std::string * trail);

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
