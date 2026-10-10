#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""Generate common/pxa-lever-catalog.inc: ONE row for every PXA_* / PXQ_* lever in the tree.

PXA core step 3 (the lever registry). Every environment variable the engine, the launcher or the
container entrypoint reads is named here once, with its default at the shipping level, the scope
it applies to (arch / tier / card class / split mode), its status and the ledger row that is its
evidence. The engine embeds the generated table, prints the relevant rows in its boot banner and
names any PXA_* variable in the environment that no row declares (a misspelled lever used to do
nothing, quietly).

Sources, in order of authority:
  1. CURATED below: the levers whose default is a RULE (topology, model, split mode) and the serve
     flags the engine now picks itself. Hand-written, because a rule is not something a scan can
     read off a getenv.
  2. docs/LEVERS.md table rows (| `NAME` | default | ... | verdict |): default and verdict text.
  3. Everything else found as an exact string literal in the sources: status "site" (the lever is
     read at its own site with the default written there).

  scripts/pxa-lever-catalog.py           regenerate the .inc
  scripts/pxa-lever-catalog.py --check   exit 1 when a lever in the tree has no row (CI / ctest)
  scripts/pxa-lever-catalog.py --public-filter   drop the CLOSED_LEVERS rows from the existing .inc (public-tree export)
  scripts/pxa-lever-catalog.py --release-filter OUT [--closed-out INC]
        write the PUBLIC table to OUT without touching the tree (what a release build embeds and ships, even when
        it has the closed sources); INC = the dropped names, for the closed library (CMake runs this)
  scripts/pxa-lever-catalog.py --assert-clean INC FILE...   release guard: exit 1 if a FILE holds a name from INC
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "common", "pxa-lever-catalog.inc")
SCAN_DIRS = ["src", "common", "examples", "ggml/src", "tools/pxa-launch.py", "tools/pxa-entrypoint.sh"]
SCAN_EXT = (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".py", ".sh", ".inc")
NAME_RE = re.compile(r'"(PX[AQ]_[A-Z0-9_]+)"')
SH_RE = re.compile(r'\$\{?(PX[AQ]_[A-Z0-9_]+)')

# name: (default, scope, status, evidence, rule)
# status: default-on | lever-off | rule | diagnostic | tool | site
CURATED = {
    # ---- config level ------------------------------------------------------------------------
    "PXA_ENHANCE": ("level 2", "any", "default-on", "", "ENHANCE is the shipping level; =0 restores DEFAULT (level 1)"),
    "PXA_REFERENCE": ("off", "any", "lever-off", "", "=1: every PXA lever off, the reference path (A/B baseline); the PXQ codec kernels, VOLTA_F16_GEMM and MXFP4_DEQ_V2 stay on (bug #281)"),
    "PXA_MODE": ("balance", "any", "default-on", "", "balance = -fa on serving; max = -fa off ingest"),
    # ---- engine-side serve flags (this registry) -----------------------------------------------
    "PXA_AUTO_SM": ("rule", "2 identical cards, or 4 identical P100s; arch qwen35; tier PXQ4/PXQN4/PXQN4S8/PXQN5; -fa on",
                    "rule", "launcher-auto-tensor-split",
                    "-sm unset -> tensor on a pair the capability check admits and on 4x P100 (measured 2026-09-27), layer elsewhere"),
    "PXA_AUTO_SM_PXQN": ("on", "-sm unset; PXQN4/PXQN4S8/PXQN5 files", "default-on", "launcher-auto-tensor-split",
                         "=0: the auto tensor split admits PXQ4 files only (the pre-2026-09-27 rule)"),
    "PXA_AUTO_BATCH": ("rule", "measured cells (sm_60/sm_61/sm_70 x card count x file)", "rule", "prefill-auto-ubatch-model-aware",
                       "-b/-ub unset -> the measured cell, else the adaptive VRAM ladder"),
    "PXA_AUTO_UB_VOLTA_Q8": ("rule", "1x sm_70; q8_0 K+V; -c >= 49152; no batch cell", "rule", "bug-207",
                             "-ub 1024 instead of the VRAM ladder's 2048 so the Volta MMA q8 attention route keeps its staging margin (642.3 vs 387.5 t/s pp65536); =0 stands down"),
    "PXA_AUTO_UB_RESERVE_MB": ("512", "an -ub the engine picked (rule or VRAM ladder)", "rule", "auto-ub-fit-reserve",
                               "VRAM that must stay free on every card after load; an engine-picked -ub steps down 1024>768>512>256 until it does, else a clean refusal; 0 = no check"),
    "PXA_FA_DEEP_QKV_TILE": ("1 (AUTO)", "sm_70; quantized K/V; whole f16 staging > 128 MiB", "rule", "fa-deep-routes-memory-aware",
                             "deep quantized-KV attention goes to tile (chunked) / vec only when the whole-tensor f16 staging does not fit (+256 MiB margin); where it fits the release route runs (bit-identical to v2026.09.20); =2 the unconditional 128 MiB rule, =0 off"),
    "PXA_FA_F16_KV_CHUNK": ("AUTO", "quantized K/V; tile-f16 kernel", "rule", "fa-deep-routes-memory-aware",
                            "unset = chunk (8192 tokens) only past a 128 MiB whole-tensor scratch that does not fit; N > 0 forces N-token chunks; 0 = whole-tensor conversion"),
    "PXA_PPL_STREAM": ("off", "llama-perplexity", "tool", "",
                       "=1 scores each batch's rows as it is decoded: long-context PPL / KLD without holding n_ctx/2 x n_vocab logits (32 GB at 64k); same rows, order and base-file layout"),
    "PXA_DN_GM_FAULT": ("off", "any", "diagnostic", "auto-ub-fit-reserve",
                        "=1 test hook: the fused deltanet gather+mask launch behaves as refused for memory, proving the clean decline (tests/test-pxa-dn-gm-refused)"),
    # ---- server speed history (speed-graph, 2026-09-27; examples/server/pxa-stats.h) -----------
    "PXA_STATS": ("on", "llama-server", "default-on", "",
                  "=0: no speed history, GET /pxa/stats and /pxa/speed unregistered (zero overhead)"),
    "PXA_STATS_MAX": ("10000", "llama-server", "default-on", "", "records kept in the in-memory speed history ring"),
    "PXA_STATS_FILE": ("off", "llama-server", "lever-off", "",
                       "=<path> or =default (~/.cache/pxa/speed-stats.jsonl): append each record as JSONL and reload the tail at start"),
    "PXA_STATS_FILE_MB": ("8", "llama-server; PXA_STATS_FILE set", "default-on", "", "rotate the history file at this size (-> <path>.1)"),
    # ---- multi-GPU peer copies (rel-fixes, 2026-09-27) -----------------------------------------
    "PXA_P2P_SELFTEST": ("on", "2+ CUDA cards with a peer path", "default-on", "p2p-startup-selftest",
                         "startup self-test of every peer pair (4 MiB DMA + direct peer reads, compared on the host); a pair that corrupts keeps peer access off and copies through system memory; =0 skips the test"),
    "PXA_P2P_SELFTEST_CORRUPT": ("off", "test hook", "diagnostic", "p2p-startup-selftest",
                                 "=1 flips one received word so the fallback branch runs on a healthy system (tests/test-pxa-p2p-selftest)"),
    # ---- speculative verify (rel-fixes, 2026-09-27) --------------------------------------------
    "PXA_SPEC_FIXED_WIDTH": ("off", "llama-server speculation; recurrent models only with a per-step checkpoint", "lever-off", "spec-fixed-width-verify",
                             "=1 pads every verify batch to the slot's draft ceiling with never-accepted filler rows, so the verify width no longer follows the draft length (spec-verify-batch-invariance)"),
    "PXA_SPEC_DEPTH_CUTOFF": ("off", "serial MTP draft chain", "lever-off", "spec-fixed-width-verify",
                              "=P (0<P<1): stop the draft chain when its cumulative probability falls below P * min(1, n_past / PXA_SPEC_DEPTH_CUTOFF_CTX)"),
    "PXA_SPEC_DEPTH_CUTOFF_CTX": ("65536", "serial MTP draft chain", "lever-off", "spec-fixed-width-verify",
                                  "context depth at which the PXA_SPEC_DEPTH_CUTOFF floor reaches its full value"),
    # ---- MTP draft depth by context depth (spec-long, 2026-09-27) --------------------------------
    "PXA_SPEC_DEPTH_NMAX": ("rule", "MTP drafter; one CUDA card (sm_70 or sm_60)", "rule", "spec-depth-nmax-ramp",
                            "'D0:N0,D1:N1,...' caps the MTP chain at Ni from context depth Di on (a cap under n_max); AUTO 1x sm_70 '0:1,8192:2,16384:3', 1x sm_60 '0:1', else off; =0 off"),
    # ---- server prompt cache / slots (ws6-server, 2026-09-25) ----------------------------------
    "PXA_CACHE_HYBRID_USABLE": ("on", "recurrent/hybrid arch; --cache-ram > 0", "default-on", "cache-hybrid-usable-restore",
                                "RAM prompt cache restores a parked entry only where the recurrent state can be re-entered (extension or checkpoint below the divergence); =0 restores the similarity scan"),
    "PXA_SLOT_VICTIM": ("lru", "-np >= 2", "lever-off", "slot-victim-cheapest-cache",
                        "=cheap: a request that matches no slot overwrites the idle slot with the fewest cached tokens (LRU tie-break); unmeasured, off"),
    "PXA_AUTO_UB_LONG": ("rule", "4x sm_60", "rule", "prefill-auto-ubatch-model-aware",
                         "4x P100: dense file -ub 256, expert file -ub 2048; =0 flat 2048, =1 force 256"),
    "PXA_AUTO_TS": ("on", "1x sm_70 + 1x sm_60, -sm layer", "rule", "",
                    "mixed V100+P100 pair: -ts 1.4,0.6 when -ts is unset"),
    "PXA_AUTO_SAMPLERS": ("on", "arch family table", "default-on", "", "model-family sampler defaults for unset --temp/--top-k/--top-p/--min-p"),
    "PXA_AUTO_SPEC": ("rule", "arch x card class", "rule", "spec-auto-default-ngram-alone-volta",
                      "arms the measured speculation default per family; declines on tight VRAM"),
    "PXA_KV_UNIFIED_DEFAULT": ("off", "-np > 1", "lever-off", "", "=1: -np > 1 with no explicit choice -> --kv-unified"),
    "PXA_EXPLAIN": ("off", "any", "diagnostic", "",
                    "=1: print the settings the engine would pick (JSON line) and exit before loading; =levers dumps this table"),
    "PXA_TOPOLOGY": ("detect", "any", "diagnostic", "",
                     "explain-only card set, e.g. 2x600 or 700,700; no CUDA context is created"),
    # ---- hot swap (--hot-model; examples/server/pxa-hotswap.h, ggml/src/pxa-residency.h) ------
    "PXA_SWAP_KEEP": ("off", "--hot-model", "lever-off", "hotswap-residency-router",
                      "=1: a swap releases only what the incoming model needs (scratch, then mirrored weights, then KV); the rest stays on the cards and the next swap back copies only what left"),
    "PXA_SWAP_HEADROOM_MB": ("1024", "--hot-model + PXA_SWAP_KEEP=1", "lever-off", "hotswap-residency-router",
                             "free MiB left per card beside the incoming model under PXA_SWAP_KEEP; growth past it is reclaimed from parked models"),
    "PXA_SWAP_PIN_BUDGET_MB": ("MemAvailable - 8 GiB", "--hot-model", "rule", "hotswap-residency-router",
                               "cap on the pinned host RAM the weight/KV mirrors may take; a park past it fails and the model stays on the cards"),
    "PXA_SWAP_VERIFY": ("off", "--hot-model", "diagnostic", "",
                        "=1: every park re-reads the weights and counts the bytes that changed since they were mirrored (should be 0)"),
    "PXA_SWAP_PEER_ACCESS": ("on", "--hot-model, 2+ cards", "default-on", "",
                             "=0: residency mappings are private to their card (the fused tensor-split reduce reads peer memory, so leave it on)"),
    "PXA_SWAP_SLAB_VA_GB": ("32", "--hot-model, -sm tensor/graph", "diagnostic", "",
                            "virtual range reserved per card per class for split slices (address space only)"),
    "PXA_HOT_UNKNOWN": ("404", "--hot-model", "lever-off", "",
                        "=active: a request naming an unregistered model is served by the model on the cards instead of a 404"),
    # ---- Gemma 4 MTP verify / prefill / tsplit levers (2026-10-03; docs/GEMMA-MTP-VERIFY-LEVERS.md); all default off --
    "PXA_MOE_VERIFY2": ("off (=1 on)", "MoE id-GEMV, 2..4-token verify batches", "lever-off", "",
        "=1: one expert-grouped id-GEMV launch per MoE projection for a 2..4-token verify batch; bit-identical (tests/test-moe-verify2); stands aside under PXA_MMVQ_MOE_NWARPS=2/4 (docs/GEMMA-MTP-VERIFY-LEVERS.md)"),
    "PXA_MOE_VERIFY2_MAX_NY": ("4", "PXA_MOE_VERIFY2=1", "site", "",
        "widest verify batch (2..4) the verify2 MoE kernel serves"),
    "PXA_MOE_VERIFY2_TILE": ("off (0x0, untiled)", "PXA_MOE_VERIFY2=1", "lever-off", "",
        "=WxR (W warps in 1/2/4/8, R rows/warp in 1/2/4): multi-warp, multi-row blocks for the verify2 MoE GEMV; bit-identical; 4x4 measured best on V100"),
    "PXA_MOE_VERIFY2_TILE_UP": ("= PXA_MOE_VERIFY2_TILE", "PXA_MOE_VERIFY2=1", "lever-off", "",
        "=WxR: tile for the fused up/gate projection only"),
    "PXA_MOE_VERIFY2_TILE_DOWN": ("= PXA_MOE_VERIFY2_TILE", "PXA_MOE_VERIFY2=1", "lever-off", "",
        "=WxR: tile for the down projection only"),
    "PXA_MOE_VERIFY2_TILE_W1": ("off (=1 on)", "PXA_MOE_VERIFY2=1 + a tile set", "lever-off", "",
        "=1: one-token decode MoE also goes through the tiled verify2 kernel; bit-identical"),
    "PXA_MOE_VERIFY2_LOG": ("off", "PXA_MOE_VERIFY2=1", "diagnostic", "",
        "=1: log verify2 dispatches"),
    "PXA_VERIFY_NO_LOGITS_COPY": ("off (=1 on)", "Gemma 4 target + assistant drafter; PXA_VERIFY_ARGMAX on", "lever-off", "",
        "=1: greedy verify and drafter argmax draws read back the device argmax, not the full logit row; anything that needs logits keeps the copy"),
    "PXA_MMVQ_V100_ROWS_NY": ("off (unset/0)", "sm_70; mmvq at 2..3 columns, large nrows (lm_head)", "lever-off", "",
        "=2|4: multi-row-per-block mmvq; bit-identical"),
    "PXA_ROUTER_GEMV_NY": ("off (=1 on)", "MoE router, 2..4-token verify", "lever-off", "",
        "=1: router GEMV instead of cuBLAS for 2..4-token batches; output-identical, measured ~null"),
    "PXA_DENSE_FUG_NY": ("off (=1 on)", "dense up/gate, 2..4 columns, GELU/RELU", "lever-off", "",
        "=1: fused dense up/gate MMVQ at 2..4 columns (MTP verify); output-identical"),
    "PXA_FA_VEC_NY": ("off (=1 on)", "sm_70 FA, 2..4 columns, f16 K/V", "lever-off", "",
        "=1: 2..4-column FA through the 1-column f32 vector kernel; NOT output-identical (excluded from the recommended set)"),
    "PXA_MMQ_ID_TILES": ("3 (sm_70, ENHANCE)", "sm_70 routed-expert (mul_mat_id) quantized matmul", "default-on", "gemma-prefill-mmq-id-compact",
                         "=0: the stream-k launch (one persistent block per SM); =1 tile grid when it fills the SMs (>=90%); =2 always; =3 compact tile grid (only non-empty expert tiles launched) for the 64-column tile, else =1"),
    "PXA_MMQ_ID_OCC": ("auto (sm_70, ENHANCE)", "sm_70 routed-expert (mul_mat_id) quantized matmul, tiles <= 64 columns", "default-on", "gemma-prefill-mmq-id-occ2",
                       "unset: 2 resident blocks per SM for Q4_0/Q4_1/Q5_1/Q8_0 (measured wins), 1 for the rest (k-quants spill at 128 registers); =1 never, =2 every type"),
    "PXA_FA_SWA_KVMIN": ("on (ENHANCE; =0 off)", "sm_70 MMA FA, sliding-window layers", "default-on", "gemma-fa-swa-kvmin",
        "=0: do not skip the fully masked sliding-window head of the KV (identical outputs either way)"),
    "PXA_FA_D512_CHAIN_SPLIT": ("off (0)", "Gemma 4 head-512 attention chain", "lever-off", "",
        "=N: run the head-512 chain in N query-head groups when the score matrix exceeds PXA_FA_D512_CHAIN_SPLIT_MIB (ub2048 compute buffer); 2..8 identical, 16 not"),
    "PXA_FA_D512_CHAIN_SPLIT_MIB": ("512", "PXA_FA_D512_CHAIN_SPLIT > 1", "site", "",
        "split only past this whole-context score-matrix size (MiB)"),
    "PXA_FA_D512_Q8_CAST": ("off (=1 on)", "head-512 decode with q8_0 KV", "lever-off", "",
        "=1: cast K/V views to f16 and use the fused D512 FA kernel"),
    "PXA_WARMUP_NO_REUSE": ("off (=1 on)", "MoE warmup", "lever-off", "",
        "=1: never reuse the all-experts warmup graph for later 1-token ubatches"),
    "PXA_VTRANS_2D": ("off (=1 on)", "non-FA transposed V store", "lever-off", "",
        "=1: reshape a per-head V to 2D before the transposed store"),
    "PXA_GEMMA4_MTP_GATE_BACKOFF": ("off (0)", "Gemma 4 MTP gate", "lever-off", "",
        "=N (>8): exponential sit-out (8..N steps) after consecutive gate closes"),
    "PXA_TSPLIT_D512_MARK": ("off (=1 on)", "Gemma 4, -sm tensor", "lever-off", "",
        "=1: the split builder head-512 decode node takes the fused 512/512 route"),
    "PXA_TSPLIT_ISWA_GUARD": ("off (=1 on)", "Gemma 4, -sm tensor", "lever-off", "",
        "=1: keep a Gemma 4 split-attention load on the unified KV cache"),
    "PXA_AUTO_UB_FASTEST": ("off (unset/0)", "model load + context (common init)", "lever-off", "",
        "=1 walks -ub down from 2048 (=N from N) to the largest -ub that loads resident (no weight streaming) and allocates; result written back to -ub/-b"),
    "PXA_AUTO_UB_FASTEST_MIN": ("512", "PXA_AUTO_UB_FASTEST set", "site", "",
        "lowest rung of the PXA_AUTO_UB_FASTEST walk; it loads as without the lever (streaming allowed)"),
    "PXA_PXQN_CPU_MMV": ("on (=0 off)", "PXQN weights read on the CPU (PXA_XCACHE cold experts, CPU-resident layers)", "default-on", "pxqn-cpu-mmv",
        "the PXQN library's CPU matmul; =0 restores the previous CPU path and keeps PXQN cold experts on zero-copy"),
    "PXA_XCACHE_BOOTSTRAP": ("on (=0 off)", "llama-server, MoE file that does not fit, no routing counts found", "default-on", "xcache-counts-bootstrap-profile",
        "decode a built-in mixed text once at startup, count the routed experts, write <model>.expert-counts.csv (else $PXA_CACHE_DIR/$XDG_CACHE_HOME/pxa/$HOME/.cache/pxa) and restart so the expert cache is planned; =0: warn and stream whole layers"),
    "PXA_XCACHE_ADAPT": ("off (=1 adapt, =observe statistics only)", "PXA_XCACHE split layers; the PXQN library carries the policy", "lever-off", "xcache-online-adapt-and-missplit",
        "=1: re-ranks the resident experts while the model decodes (routing counted on the device, swaps run as async copies between decodes); +4% prose / +8% code on fn32 1x P100, inside load noise, and a greedy answer then depends on earlier requests (CPU q8 vs GPU rounding), so it is opt-in; =observe keeps the static plan and only reports the hit rate"),
    "PXA_XCACHE_SPARE": ("4", "PXA_XCACHE_ADAPT on", "site", "xcache-online-adapt-and-missplit",
        "free slots added to each split layer's cold stack (pinned host RAM) that a swap evicts a hot expert into; the hot stack keeps its planned size"),
    "PXA_XCACHE_ADAPT_EVERY": ("4", "PXA_XCACHE_ADAPT on", "site", "xcache-online-adapt-and-missplit",
        "decodes between two observations of the routing counters (a prefill-sized decode is an observation of its own)"),
    "PXA_XCACHE_ADAPT_DET": ("1", "PXA_XCACHE_ADAPT on", "default-on", "xcache-online-adapt-and-missplit",
        "=1 every swap is applied a fixed number of decodes after it was planned (waiting for its copy), so the swap schedule follows the token stream; =0 applies it when the copy lands"),
    "PXA_XCACHE_MTP": ("off (=1 arms)", "PXA_XCACHE active and a file with an MTP head", "lever-off", "xcache-mtp-chain",
        "=1: the expert cache runs together with an MTP speculative chain: the head's layer stays whole in VRAM and the planner keeps the chain's per-step recurrent checkpoint rows (and a small companion allowance) out of the experts' share, so the draft-length budget is not clamped to 1 on a full card; the auto spec row is armed instead of declining on the file-size headroom estimate"),
    "PXA_XCACHE_PLAN_V2": ("off (=1 arms)", "PXA_XCACHE active", "lever-off", "xcache-mtp-chain",
        "=1: the planner prices a cold layer at the measured 0.7 ms (not 1.3) and, with a speculative chain, a missed expert per round (verify width 1 + n_max): more cold layers, fewer misses"),
    "PXA_XCACHE_VERIFY_KAPPA": ("0.75", "PXA_XCACHE_PLAN_V2 on", "site", "xcache-mtp-chain",
        "share of a speculative round's extra verify width that reads distinct cold experts (1 = independent routing, 0 = the same experts every token)"),
    "PXA_XCACHE_VERIFY_WIDTH": ("server-set", "PXA_XCACHE_PLAN_V2 / PXA_XCACHE_MTP", "diagnostic", "xcache-mtp-chain",
        "tokens per target decode of the run (1 + the largest draft depth); the server exports it before the model is planned, an exported value wins"),
    "PXA_XCACHE_SPEC_COMPANION_MB": ("64", "PXA_XCACHE_MTP on", "site", "xcache-mtp-chain",
        "VRAM the planner keeps free for the MTP companion context on top of the per-step checkpoint rows"),
    "PXA_REP_GUARD_LAZY": ("off (=1 arms)", "a file where the repetition guard auto-arms (PXQ1 / PXQN1-heavy)", "lever-off", "rep-guard-lazy",
        "=1: the guard's repeat penalty + DRY wait for a span repeated back to back in the generated text (a loop's first lap), so the greedy verify stays argmax-only and the MTP draft matches; hot for PXA_REP_GUARD_LAZY_HOLD tokens after a repeat"),
    "PXA_REP_GUARD_LAZY_K": ("6", "PXA_REP_GUARD_LAZY on", "site", "rep-guard-lazy", "shortest span (repeated twice in a row, up to 256 tokens) that turns the guard hot"),
    "PXA_REP_GUARD_LAZY_HOLD": ("96", "PXA_REP_GUARD_LAZY on", "site", "rep-guard-lazy", "generated tokens the guard stays hot after the last repeat"),
    "PXA_NUMA_BIND": ("on (=0 off; off at PXA_REFERENCE)", "Linux; every visible card on one NUMA node", "default-on", "numa-bind-device-node",
        "=1: pin every thread of the process to the CPUs of the NUMA node the cards hang off (within the affinity it already had) and prefer that node for memory, before the first weight is allocated; the expert cache's CPU cold path and the per-layer-embedding gather then read local pages"),
    "PXA_XCACHE_ASYNC": ("off (=1 on)", "PXA_XCACHE cache layers on the CPU cold path (pinned stacks the CPU reads); graphs of 1..PXA_XCACHE_ASYNC_MAXTOK tokens; CUDA", "lever-off", "xcache-async-cold-op",
        "=1: the cold experts of a split layer are computed by a host worker thread that the GPU stream hands work to through flags in pinned memory (no scheduler split, no cudaStreamSynchronize and no ids / activation / result copy call per cold layer); the CPU sub-graph is the one the split path runs, so the output is bit-identical; wider graphs keep the split path; CUDA-graph capture is off for graphs that carry the nodes"),
    "PXA_XCACHE_CPUPOOL": ("off (=1 on)", "PXA_XCACHE_ASYNC on", "lever-off", "xcache-cpu-miss-compute",
        "off; needs the PXQN library; bit-identical"),
    "PXA_XCACHE_IDLE_BACKOFF_MS": ("0 (off)", "PXA_XCACHE_ASYNC on", "lever-off", "xcache-cpu-miss-compute",
        "N>0: an idle cold-path thread waits longer between checks. The result is unchanged."),
    "PXA_CUDA_GRAPH_REPLAN_KEEP": ("off (=1 on)", "CUDA graphs on (sm_60 needs PXA_CUDA_GRAPHS_PASCAL)", "lever-off", "xcache-next",
        "=1 (default): a scheduler re-plan no longer bumps the global CUDA-graph allocation generation (every captured graph of every scheduler was recaptured, so verify-width / MTP graphs hit too-many-updates and ran eager); the per-node property compare still forces a recapture when an address moved; reserve and cudaFree sites still bump"),
    "PXA_XCACHE_CPUPOOL_THREADS": ("the async worker's thread count", "PXA_XCACHE_CPUPOOL on", "site", "xcache-cpu-miss-compute",
        "how many threads the library uses"),
    "PXA_XCACHE_CPUPOOL_CPUS": ("unpinned", "PXA_XCACHE_CPUPOOL on", "site", "xcache-cpu-miss-compute",
        "host CPUs the library may pin threads to"),
    "PXA_XCACHE_CPUPOOL_ROWS_A": ("see the library", "PXA_XCACHE_CPUPOOL on", "site", "xcache-cpu-miss-compute",
        "chosen by the library"),
    "PXA_XCACHE_CPUPOOL_ROWS_C": ("see the library", "PXA_XCACHE_CPUPOOL on", "site", "xcache-cpu-miss-compute",
        "chosen by the library"),
    "PXA_XCACHE_ASYNC_GRAPH": ("off (=1 on)", "PXA_XCACHE_ASYNC on; CUDA graphs otherwise allowed (sm_60 needs PXA_CUDA_GRAPHS_PASCAL; MoE needs PXA_CUDA_GRAPH_MOE)", "lever-off", "xcache-next",
        "=1 (default): a graph carrying the cold submit / wait nodes stays capturable (the kernels keep their request number in device memory, the host worker only watches pinned flags), so the decode step replays as one CUDA graph"),
    "PXA_MTP_DRAFT_GUARD": ("off (=1 on)", "MTP drafting; temp-0 requests whose repetition guard / penalties / DRY are armed", "lever-off", "exact-mtp-default",
        "=1 (default): the draft token is the argmax of the draft head's top-W window after the request's own repeat penalty (prev tail + chain) and DRY (cloned, fed the chain); only the proposal changes, verify stays the exact match, output byte-identical"),
    "PXA_MTP_DRAFT_GUARD_W": ("16", "PXA_MTP_DRAFT_GUARD on", "site", "exact-mtp-default",
        "candidate window (2..128) the penalties are applied over"),
    "PXA_MTP_DRAFT_GUARD_STATS": ("off", "PXA_MTP_DRAFT_GUARD on", "diagnostic", "exact-mtp-default",
        "print draws / moved-off-argmax every 512 draws"),
    "PXA_XCACHE_ASYNC_MAXTOK": ("8", "PXA_XCACHE_ASYNC on", "site", "xcache-async-cold-op",
        "widest graph (tokens) that takes the async cold path: a decode and an MTP / n-gram verify batch; a prefill ubatch keeps the scheduler split"),
    "PXA_XCACHE_ASYNC_THREADS": ("the context's -t (-tb above one token)", "PXA_XCACHE_ASYNC on", "site", "xcache-async-cold-op",
        "CPU threads of the worker's sub-graph compute (OpenMP team led by the worker thread)"),
    "PXA_XCACHE_ASYNC_SPIN_US": ("3000", "PXA_XCACHE_ASYNC on", "site", "xcache-async-cold-op",
        "microseconds the worker keeps polling the request flags after its last request before it naps in 20 us steps (one core spins between layers of a token)"),
    "PXA_XCACHE_ASYNC_TIMEOUT_S": ("30", "PXA_XCACHE_ASYNC on", "site", "xcache-async-cold-op",
        "seconds a GPU wait gives up after (counted and reported at exit, the rows of that layer are zero) instead of hanging the card; must stay 0 timeouts"),
    "PXA_XCACHE_ASYNC_CHECK": ("0", "PXA_XCACHE_ASYNC on", "diagnostic", "xcache-async-cold-op",
        "=1: every async cold layer ALSO runs the scheduler's CPU split, the two results are compared bit for bit on the GPU and the counts are printed at exit; the merge takes the split's result (output = the baseline's); slow, for finding where the two paths differ"),
    "PXA_XCACHE_ASYNC_MIXED": ("0", "PXA_XCACHE_ASYNC on; cache layers whose split-path chain is built as separate up / gate / unary nodes (an up stack and a gate stack of different types)", "diagnostic", "xcache-async-cold-op",
        "=1: those layers take the async path too, computed entirely on the CPU; the split path places their activation node on the GPU, so the output then differs from the split path in the last bits of that activation (measured: 4 of 31 cold layers on Flash-Next 32GB); default: they stay on the split path and the output is bit-identical"),
    "PXA_XCACHE_ASYNC_OVERLAP": ("1", "PXA_XCACHE_ASYNC on; models with a plain shared expert", "diagnostic", "xcache-async-cold-op",
        "=0: the shared expert is queued after the routed experts as before instead of between the hot experts and the wait on the host worker (A/B for the overlap)"),
    "PXA_XCACHE_ASYNC_TICKETDEP": ("1", "PXA_XCACHE_ASYNC on", "diagnostic", "xcache-async-cold-op",
        "=0: the submit node is ordered by graph expansion order instead of a dependency edge on the hot half's id node (A/B)"),
    "PXA_XCACHE_ASYNC_BATCHONLY": ("1", "PXA_XCACHE_ASYNC on", "diagnostic", "xcache-async-cold-op",
        "=1: only graphs whose whole batch is at most PXA_XCACHE_ASYNC_MAXTOK tokens take the async path (the last layer of a prefill ubatch sees one row but stays on the split path); =0 per layer"),
    "PXA_PXQN_XCACHE": ("on (=0 off)", "the PXQN library carries the expert-cache policy", "default-on", "xcache-online-adapt-and-missplit",
        "=0 never resolves the library's online-adaptation policy: the expert cache keeps the static plan"),
    "PXA_XCACHE_BOOTSTRAPPED": ("unset", "internal", "diagnostic", "xcache-counts-bootstrap-profile",
        "set by the server on its own restart after a counts bootstrap; stops a second bootstrap"),
    "PXA_CACHE_DIR": ("unset ($XDG_CACHE_HOME/pxa, else $HOME/.cache/pxa)", "per-user engine caches (routing counts of a read-only model directory)", "site", "xcache-counts-bootstrap-profile",
        "where <model>.expert-counts.csv goes when the model directory is not writable"),
    "PXA_SPEC_SELECT": ("off (=1 arms; PXA_SPEC_DEFAULTS=1 turns it on for Flash-Next and Gemma 4 + assistant)", "libggml-pxqn present; chains of table and/or MTP stages; not np>1", "lever-off", "spec-selfmeasuring-selector",
        "=1: a per-slot self-measuring selector (cost model in libggml-pxqn) picks plain / table draft / MTP depth each round by measured tokens per ms; verification is unchanged, so only speed moves; =0 or no library = the static rows"),
    "PXA_SPEC_SELECT_TUNE": ("unset", "PXA_SPEC_SELECT armed", "diagnostic", "spec-selfmeasuring-selector",
        "key=value,... overrides of the selector's constants (lab use)"),
    "PXA_SPEC_SELECT_TRACE": ("off", "PXA_SPEC_SELECT armed", "diagnostic", "spec-selfmeasuring-selector",
        "=1 prints each switch / probe / change-of-text decision to stderr"),
    "PXA_ALLOC_FAIL_CLEAN": ("off (=1 on)", "any", "lever-off", "",
        "=1: a failed compute-graph allocation re-plans / fails with an error instead of placing tensors past a buffer end (SIGSEGV)"),
    # ---- tensor split ------------------------------------------------------------------------
    "PXA_TSPLIT_REDUCE": ("fused (engine default with -sm tensor)", "-sm tensor", "rule", "launcher-auto-tensor-split",
                          "the fused all-reduce every -sm tensor number was taken on"),
    "PXA_TSPLIT_REDUCE_PREFILL": ("0", "-sm tensor", "lever-off", "tsplit-prefill-pf-supersedes",
                                  "fused reduce on the prefill path; slower than PXA_TSPLIT_PF on 2 cards, no longer emitted"),
    "PXA_TSPLIT_FALLBACK": ("1 when -sm was auto-picked, else 0", "-sm tensor", "rule", "launcher-auto-tensor-split",
                            "a refused tensor split demotes to layer instead of stopping"),
    "PXA_TSPLIT_ALLOW_4WAY": ("on", "-sm tensor on >2 cards", "default-on", "tsplit-4card-default",
                              "=0: a >2-device tensor split warns and runs layer (the pre-2026-09-27 guard; bug #206 is fixed)"),
    "PXA_TSPLIT_FORCE_FA": ("on", "-sm tensor", "default-on", "", "the split attention builder needs flash attention"),
    "PXA_TSPLIT_UNPROVEN_ARCH": ("off", "-sm tensor", "lever-off", "", "=1 runs an arch the split has no evidence for"),
    "PXA_TSPLIT_GEMMA4": ("off", "-sm tensor, gemma4", "lever-off", "", "Gemma 4 tensor split is opt-in"),
    "PXA_DN_CONVFUSE": ("on", "delta-net decode (n_tok 1, one seq), row absorb live", "default-on", "dn-convfuse", "conv + next-step conv window + silu/q-k norm + beta-gate as one kernel; bit-exact; REFERENCE: off; =0 the separate kernels"),
    "PXA_DN_INPLACE": ("on", "delta-net decode with PXA_DN_CONVFUSE", "default-on", "dn-inplace-carry", "carried state read from the cache row (device-resolved index) times the reset mask, no gather copy; bit-exact; REFERENCE: off; =0 the gather+mask copy"),
    "PXA_DN_BA_KQMMV": ("on", "delta-net ssm_beta/ssm_alpha under PXA_KQMMV", "default-on", "dn-ba-kqmmv", "beta+alpha as one grouped f32-staged GEMV (no q8_1 quantize); changes output bits; =0 the incumbent quantize + two MMVQ"),
    "PXA_TSPLIT_LMHEAD": ("on", "-sm tensor", "default-on", "", "=0: full LM head on every card (vocab-parallel head changes output bits)"),
    # ---- ENHANCE-armed kernel levers (resolved in ggml-cuda/pxa/pxa-enhance.cuh) -------------
    "PXA_PXQ_INT8_PREFILL": ("1 on sm_61", "sm_61", "rule", "", "int8 prefill tile; +182% prefill on the 1080 Ti"),
    "PXA_ROUTER_FUSE": ("on sm_70", "sm_70 (not in a mixed rig)", "rule", "", "router GEMV fusion; a loss on sm_60"),
    "PXA_MTP_SHORTLIST": ("prefix:81920 on the 248320-id Qwen head (=0 off); Flash-Next / Gemma 4 heads only by an explicit value or PXA_SPEC_DEFAULTS", "MTP draft head (qwen35, qwen4exp, gemma4 assistant)", "default-on",
                          "mtp-shortlist-idprefix", "draft head scores token ids [0,N) only; the verify head is never windowed"),
    "PXA_SPEC_DEFAULTS_TUNE": ("v1_nmax=3,v1_pmin=0.85,v1_ub=512,v1_ramp=0,v1_cap=0,p1_on=1,p1_nmax=3,p1_pmin=0.8,p1_ub=512,p1_ramp=0,p1_cap=0",
                               "PXA_SPEC_DEFAULTS, qwen35 on one V100 (v1_) / one P100 (p1_)", "lever-off", "mtp-all-spec-defaults",
                               "the one-card rows' depth ceiling, confidence floor, physical batch, context-depth ramp (1 = keep the ramp) and _cap=1 (unclamp with the capped logits reservation PXA_LOGITS_CAP at the engine's own -ub instead of the smaller batch) from the environment, so a measurement window needs no rebuild"),
    "PXA_SPEC_SELECT_PLAIN": ("on where the per-model table armed the selector, else off (=1 / =0 decide)", "selector on a chain with a companion MTP head", "lever-off", "mtp-all-spec-defaults",
                              "a round without a draft is a legal choice for a chain with a companion head, so the selector can fall back to plain decode when the head loses"),
    "PXA_SPEC_DEFAULTS": ("on (=0 never arms the per-model / per-card table in libggml-pxqn; PXA_REFERENCE=1 off)", "speculation, any MTP / assistant model", "default-on",
                          "mtp-all-spec-defaults", "which chain / depth / floor / selector a model x these cards was measured best with; no row = the engine's static rows"),
    "PXA_SPEC_SAMPLED": ("on (=0 off)", "speculation", "default-on", "spec-lossless-rejection-sampling",
                         "lossless sampled acceptance at temp>0 (replaces relaxed)"),
    "PXA_SPEC_RELAXED": ("off (on only with PXA_SPEC_SAMPLED=0)", "speculation", "rule", "", "relaxed acceptance floor"),
    "PXA_FA_TILE_F32ACC": ("on (=0 f16 accumulator; off at REFERENCE)", "sm_60 tile-f16 flash attention", "default-on", "",
                           "fp32 running state (kqmax, kqsum, VKQ) in the tile-f16 FA kernel: the f16 accumulator's rounding depended on where a request's band sat in the KV ring (a correctness fix, not a speed lever); resolved in ggml-cuda/pxa/core/levers.cu"),
    "PXA_FA_TILE_V2": ("off (=1 on)", "sm_60 batched flash attention (tile-f16 shapes)", "lever-off", "",
                       "alternative schedule for the tile-f16 FA arithmetic (16-byte shared loads, no row pad), bitwise equal to the shipping kernel; no speed number yet; inert unless the tile-f16 kernel would have run"),
    "PXA_FA_GQA_PACK": ("4 on multi sm_60", "2+ sm_60", "rule", "", "deep-fill decode GQA packing"),
    "PXA_MOE_DEVICE_MAP": ("on multi sm_60", "2+ sm_60, expert files", "rule", "", "device-side expert routing table"),
    "PXA_PIPELINE_PP": ("on", "arch qwen35 / qwen35moe", "rule", "", "pipelined prompt processing (not qwen4exp: OOM)"),
    "PXA_KQ_MASK_PAD1": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    "PXA_KV_SEQ_SOA": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    "PXA_TOPK_RAW": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    "PXA_TOPK_MOE_MULTIROW": ("on", "any (ENHANCE)", "default-on", "", "house lever (router aliasing guard)"),
    "PXA_GETROWS_NARROW": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    "PXA_CPY_FASTDIV": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    "PXA_CONCAT_FLAT": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    "PXA_NORM_REGCACHE": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    "PXA_SCHED_RESET_LAZY": ("on", "any (ENHANCE)", "default-on", "", "house lever"),
    # ---- launcher / package / container (never read by the engine) ---------------------------
    "PXA_ENGINE_DIR": ("detect", "launcher", "tool", "", "the build dir holding bin/llama-server"),
    "PXA_MODELS_DIR": ("detect", "launcher", "tool", "", "model search roots"),
    "PXA_LAUNCH_STATE": ("~/.pxa", "launcher", "tool", "", "launcher state dir"),
    "PXA_SERVE_DIR": ("state/serve", "launcher", "tool", "", "--serve-name script dir"),
    "PXA_HOST_ROOT": ("cwd", "launcher", "tool", "", "host root for vLLM mounts"),
    "PXA_VLLM_IMAGE": ("unset", "launcher", "tool", "", "vLLM image override"),
    "PXA_PXQ4_LIB": ("unset", "launcher", "tool", "", "vLLM PXQ4 plugin path"),
    "PXA_NO_TUI": ("unset", "launcher", "tool", "", "plain prompts instead of the full-screen UI"),
    "PXA_HOME": ("/opt/pxa/dist", "container", "tool", "", "package root in the image"),
    "PXA_MODEL": ("unset", "container", "tool", "", "model the entrypoint serves when run with no arguments"),
    "PXA_GPUS": ("all visible", "container", "tool", "", "cards the entrypoint hands the launcher"),
    "PXA_ALLOW_BUSY": ("off", "container", "tool", "", "=1: let the entrypoint start on a card that is in use"),
}

# string literals the scan finds that are NOT levers (a tier name, an output marker, a log tag)
NOT_LEVERS = {"PXQ_UNIVERSAL", "PXA_EXPLAIN_JSON", "PXA_TSPLIT_P2P"}

CURATED.update({
    "PXA_LAUNCH_EXTRA": ("unset", "container", "tool", "", "more pxa-launch arguments for the no-argument entrypoint path (e.g. --explain)"),
    "PXA_LAUNCH_FAKE_GPUS": ("unset", "launcher", "diagnostic", "", "dry-run card table, e.g. 2x600 (parity harness); never for a real launch"),
    "PXA_LAUNCH_TABLES_ONLY": ("off", "launcher", "diagnostic", "", "=1: the launcher's own tables, the engine registry is not asked (parity harness)"),
    # ---- server prompt-batch admission / layer placement (pp-next, 2026-09-25) ------------------
    "PXA_PROMPT_RM_NOOP_SKIP": ("on", "server; chunk 2+ of a multi-chunk prompt admission (-b < prompt)", "default-on",
                                "pp-next-prompt-batch-drain",
                                "skips the llama_synchronize drain + no-op seq_rm when neither context holds a cell at or "
                                "beyond p0 (nothing to tear down); +21.8%/+12.1% quad 27B prefill at 20801/3121 (1.68 s/prefill "
                                "of drain removed on the V100 pair); decode and greedy sha unaffected (verified with NDEC held "
                                "equal, rel-integrate2 2026-09-25 #8823 - the earlier apparent regression was a harness NDEC "
                                "256-vs-512 mismatch between arms, not the lever); =0 restores the unconditional drain"),
    "PXA_LAYER_BALANCE": ("off", "-sm layer; every layer+head on the GPUs; same cc; no -ts; model fits", "lever-off",
                          "pp-next-layer-balance",
                          "re-places -sm layer's layers by prefill cost (weight elements + attention FLOPs) instead of bytes; "
                          "measured a quad loss (evened busy 0.57-0.59 but throughput fell) - the busiest card was not the "
                          "limiter, so evening cost throughput instead of buying it; stays off pending a length/serialization "
                          "root cause"),
    # ---- PXQ decode kernels (launch-defaults 2026-09-27, plan item S7a) -------------------------
    "PXA_PXQ_MMVQ_COLS": ("on at sm_70+ (ENHANCE), off below", "PXQ MMVQ launches at verify width 3-8", "default-on",
                          "s7a-mmvq-cols-sm70",
                          "wide multi-column tile (8 rows at 3-4 columns, 4 at 5-8) instead of the 2-row clamp; bit-identical; "
                          "V100 pair 27B PXQ4 MTP -sm layer +10.2/+14.9% decode (clean bracket), wash under -sm tensor, "
                          "inert at width 1; =0 restores the clamp, =1 arms every card the kernel runs on"),
    # ---- k-quant decode GEMV (hp-kquant-mmv 2026-09-27) ------------------------------------------
    "PXA_KQMMV": ("auto (on at sm_60)", "Q4_K/Q5_K/Q6_K/Q8_0 MUL_MAT and FUSED_UP_GATE whose activation fits the block (ny <= 2 at K 5120)", "rule",
                  "hp-kquant-mmv-decode-gemv",
                  "k-quant decode GEMV: 8 rows per block (one warp per row), f32 activation staged once per block (no "
                  "q8_1 launch), 16-byte weight loads, half2 dot on sm_60 / dp4a on sm_61+, src1-sharing matrices "
                  "grouped into one launch, gate/up fused; parity-tested vs the incumbent q8_1 MMVQ; P100 microbench "
                  "Q6_K 1.9-2.2x, Q8_0 1.5-1.7x, Q4_K/Q5_K 1.05-1.35x; REFERENCE: off; =0 restores the incumbent MMVQ, "
                  "=1 arms every card (the sm_61+ i8 path is slower than the dp4a MMVQ on V100)"),
    # ---- narrow PCIe links / peer-to-peer (launch-defaults 2026-09-27, bug #280) -----------------
    "PXA_P2P": ("on", "2+ cards", "default-on", "bug-280",
                "=0: CUDA peer access is never enabled (cross-card copies host-staged, the PXA fused/p2p reduces "
                "decline, NCCL_P2P_DISABLE=1): slower, for links whose peer path is not trustworthy (risers, IOMMU)"),
    "PXA_PCIE_LINK_WIDTH": ("sysfs", "any", "diagnostic", "bug-280",
                            "overrides the per-card PCIe link width the engine reads from sysfs (e.g. =1 to exercise the "
                            "narrow-link rules on a x4/x16 box); below x4: no pipeline parallelism, no auto -sm tensor"),
    "PXA_PIPELINE_PP": ("on for qwen35/qwen35moe multi-card -sm layer; off on a card below PCIe x4", "-sm layer, 2+ cards",
                        "rule", "bug-280",
                        "=1 forces pipeline parallelism (n_copies=2) on, =0 off; narrow links (x1/x2 risers) keep it off by default"),
    # ---- host memory for CPU-only gather tables (fn-next 2026-09-25, launch-defaults 2026-09-27) --
    "PXA_PLE_MMAP": ("1", "a CPU (host-buffer) per_layer_token_embd while the model-wide mmap is off (any -ot)", "default-on",
                     "qwen4exp-ple-table-pageable",
                     "the PLE gather table is served from a read-only file mapping (pageable, no copy) instead of a pinned "
                     "CUDA_Host copy - 51,880 MiB on Flash-Next; bit-identical; =0 restores the pinned copy, =2 adds MADV_WILLNEED"),
    "PXA_PLE_MMAP_TOK": ("1", "under PXA_PLE_MMAP: a host-buffer token_embd that is not also the output matrix", "default-on",
                         "qwen4exp-ple-table-pageable",
                         "the token embedding table (CPU GET_ROWS only) gets the same mapping - 497 MiB on Flash-Next; =0 keeps "
                         "it on the pinned copy"),
})

EXTRA_NAMES = set(CURATED)  # curated names are declared even when no literal names them yet


def clean(s, n):
    s = re.sub(r"\*\*|`|\[|\]|\(http[^)]*\)", "", s).strip()
    s = s.replace('\\', '/').replace('"', "'")
    return (s[: n - 1] + "~") if len(s) > n else s


SITES = {}   # name -> (relpath, line no, the site's text and the two lines after it)
GATE_RE = re.compile(r'PXA_PXQ6_GATE\(\s*\w+\s*,\s*"(PX[AQ]_[A-Z0-9_]+)"\s*,\s*(true|false|[01])\s*,\s*"([^"]*)"')
TABLE_RE = re.compile(r'\{\s*"(PX[AQ]_[A-Z0-9_]+)"\s*,\s*"[A-Z0-9_]+"\s*,\s*(-?\d+)\s*,\s*((?:"[^"]*"\s*)+)\}', re.S)
FROM_CODE = {}   # name -> (default, status, rule) read off a lever table row or a gate macro
EXCLUDE = None   # rel path -> True: not scanned (--release-filter: the closed files)


def scan():
    names = {}
    for d in SCAN_DIRS:
        p = os.path.join(ROOT, d)
        # sorted: the row for a lever read at several sites cites the FIRST one found, and os.walk's order is
        # the filesystem's, so two checkouts of one commit used to regenerate different "read at" lines
        files = [p] if os.path.isfile(p) else sorted(os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs)
        for f in files:
            if not f.endswith(SCAN_EXT) or f.endswith("pxa-lever-catalog.inc"):
                continue
            if EXCLUDE is not None and EXCLUDE(os.path.relpath(f, ROOT)):
                continue
            try:
                txt = open(f, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            found = set(NAME_RE.findall(txt))
            if f.endswith(".sh"):
                found |= set(SH_RE.findall(txt))
            rel = os.path.relpath(f, ROOT)
            for m in GATE_RE.finditer(txt):
                on = m.group(2) in ("true", "1")
                FROM_CODE.setdefault(m.group(1), ("on" if on else "off", "default-on" if on else "lever-off",
                                                  "gate " + rel + ": " + clean(m.group(3), 70)))
            for m in TABLE_RE.finditer(txt):
                desc = clean(re.sub(r'"\s*"', "", m.group(3)).strip('"'), 200)
                dl = desc.lower()
                st = ("default-on" if "default on" in dl else
                      "lever-off" if ("default off" in dl or "(default" in dl or "benchmarking" in dl
                                      or "unsound" in dl) else "site")
                FROM_CODE.setdefault(m.group(1), ("table " + m.group(2), st, "lever table " + rel + ": " + desc[:70]))
            lines = txt.splitlines()
            for n in found:
                names.setdefault(n, rel)
                if n not in SITES:
                    for i, ln in enumerate(lines):
                        if '"%s"' % n in ln and ("getenv" in ln or "environ" in ln or "_env" in ln):
                            SITES[n] = (rel, i + 1, " ".join(x.strip() for x in lines[i:i + 3]))
                            break
    for n in NOT_LEVERS:
        names.pop(n, None)
    return names


def levers_md():
    out = {}
    # The FULL lever table is docs/lab/LEVERS.md. docs/LEVERS.md used to be a symlink to it; for
    # v2026.10 it is a thinned public page (WS7), which would drop ~100 rows' defaults and verdicts
    # from the catalog if it were read here. Fall back to docs/LEVERS.md only when lab/ is absent.
    p = os.path.join(ROOT, "docs", "lab", "LEVERS.md")
    if not os.path.exists(p):
        p = os.path.join(ROOT, "docs", "LEVERS.md")
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"\|\s*`(PX[AQ]_[A-Z0-9_]+)[^`]*`\s*\|([^|]*)\|(.*)\|\s*([^|]*)\|\s*$", line)
        if m:
            out[m.group(1)] = (m.group(2), m.group(4))   # later rows (updates) supersede
    return out


def row_for(name, src, md):
    if name in CURATED:
        return CURATED[name]
    if name in FROM_CODE and name not in md:
        pass   # the code's own lever table / gate macro is the authority (handled below)
    if name in md:
        d, verdict = md[name]
        dl, vl = clean(d, 200).lower(), clean(verdict, 200).lower()
        scope = "any"
        for cc in ("sm_60", "sm_61", "sm_70"):
            if cc in dl:
                scope = cc if scope == "any" else scope + "/" + cc
        if "dead" in vl or "loss" in vl or "do not" in vl:
            status = "lever-off"
        elif re.search(r"\bon\b|\bdefault on\b", dl) or re.search(r"^\s*on\b", vl):
            status = "default-on"
        elif "off" in dl or dl.strip() in ("0", "unset", "-", ""):
            status = "lever-off"
        else:
            status = "site"
        return (clean(d, 40) or "see site", scope, status, "", "docs/LEVERS.md: " + clean(verdict, 60))
    diag = any(k in name for k in ("_DBG", "_DEBUG", "_TRACE", "_DUMP", "_PROFILE", "_STATS", "_LOG", "_CENSUS"))
    if name in FROM_CODE:
        d, st, rule = FROM_CODE[name]
        return (d, "any", "diagnostic" if diag else st, "", rule)
    site = SITES.get(name)
    if site:
        rel, ln, code = site
        d, st = site_default(name, code)
        return (d, "any", "diagnostic" if diag else st, "", "read at %s:%d" % (rel, ln))
    return ("see site", "any", "diagnostic" if diag else "site", "", "read at " + src)


def site_default(name, code):
    """(default, status) read off the getenv site itself; ('see site', 'site') when the site does not
    say it in one of the usual shapes"""
    q = code.replace(" ", "")
    m = re.search(r'environ\.get\("%s",("?)([^",)]*)\1\)' % name, q)
    if m:
        return (m.group(2) or "empty", "site")
    if re.search(r"\[0\]!='0'|!\w+\|\||on_unless_zero|:true;|\?true:|!=\"0\"", q):
        return ("on (=0 off)", "default-on")
    if re.search(r"\[0\]=='1'|on_if_one|==1\b|&&atoi\(\w+\)!=0|&&\*\w+=='1'|:false;|\?\w+:false", q):
        return ("off (=1 on)", "lever-off")
    m = re.search(r'\?(?:atoi|atol|atof|strtol|strtoll|strtod|std::stoi)\([^:]*\):(-?[0-9.]+)', q)
    if m:
        return (m.group(1), "site")
    return ("see site", "site")


def generate():
    names = scan()
    md = levers_md()
    rows = []
    for n in sorted(set(names) | EXTRA_NAMES):
        rows.append((n,) + tuple(row_for(n, names.get(n, "catalog"), md)))
    return names, rows


def emit(rows):
    lines = ["// GENERATED by scripts/pxa-lever-catalog.py -- do not edit by hand; edit the script's CURATED",
             "// table (or docs/LEVERS.md) and regenerate. Copyright (c) 2026 PXA Network.",
             "// { name, default, scope, status, evidence, rule }"]
    for r in rows:
        lines.append("{ " + ", ".join('"%s"' % c.replace('"', "'") for c in r) + " },")
    return "\n".join(lines) + "\n"


# Levers read only inside the closed library: their names come from scripts/pxa-closed-levers.txt (one name or
# NAME* prefix per line; the file ships with the closed sources, not in the public tree). --public-filter
# drops those rows from the .inc for the public export.
def closed_set():
    f = os.path.join(ROOT, "scripts", "pxa-closed-levers.txt")
    names, prefixes = set(), []
    if os.path.exists(f):
        for line in open(f, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line.endswith("*"):
                prefixes.append(line[:-1])
            elif line:
                names.add(line)
    return names, tuple(prefixes)


def public_filter():
    names, prefixes = closed_set()
    in_tree, _ = generate()          # the levers this (public) tree still reads, and where
    have = open(OUT, encoding="utf-8").read().splitlines(True)
    keep, dropped, moved = [], 0, 0
    for l in have:
        if not l.startswith('{ "'):
            keep.append(l)
            continue
        name = l[3:].split('"', 1)[0]
        if name in names or (prefixes and name.startswith(prefixes)):
            dropped += 1
            continue
        m = re.search(r'"read at ([^":]+)(:\d+)?"', l)
        if m and not os.path.exists(os.path.join(ROOT, m.group(1))):   # the read site is an omitted closed file
            if name not in in_tree:
                dropped += 1
                continue
            l = l[:m.start()] + '"read at %s"' % in_tree[name] + l[m.end():]   # still read in an open file: point there
            moved += 1
        keep.append(l)
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(keep)
    print("pxa-lever-catalog: public filter dropped %d closed row(s), re-pointed %d" % (dropped, moved))
    return 0


# Release builds (2026-10-08, owner: no closed rows in what ships): the engine's embedded table, Control's copy and
# the image's copy are the PUBLIC catalog even when the build has the closed sources. The closed files are the public
# tree's omit list (scripts/pxa-closed-files.txt, itself closed); a row is dropped when its lever is closed or is read
# only inside a closed file, re-pointed when an open file reads it too, and any closed path left in its text is
# replaced. The table ends with one "// built-in:" line of name hashes (no names) so PXA Control accepts the dropped
# levers a preset sets; INC (--closed-out) lists the dropped names for the closed library, which declares them at run
# time (ggml_pxqn_lever_known), so the engine does not call them typos.
def closed_files():
    f = os.path.join(ROOT, "scripts", "pxa-closed-files.txt")
    out = []
    if os.path.exists(f):
        for line in open(f, encoding="utf-8"):
            line = line.split("#", 1)[0].strip().rstrip("/")
            if line:
                out.append(line)
    return out


def lever_hash(name):
    """FNV-1a 64 of the name, 16 hex digits (PXA Control computes the same in Python and in the page)."""
    h = 0xcbf29ce484222325
    for b in name.encode("utf-8"):
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def release_filter(out_path, closed_out=None):
    global EXCLUDE
    paths = closed_files()

    def is_closed_path(rel):
        rel = rel.replace(os.sep, "/")
        return any(rel == p or rel.startswith(p + "/") for p in paths)

    EXCLUDE = is_closed_path
    in_tree = scan()                 # the levers the OPEN files read, and where
    names, prefixes = closed_set()
    path_res = [re.compile(r'(?<![\w./-])' + re.escape(p) + r'(/[^\s",;:)]*)?(:\d+)?(?![\w.-])') for p in paths]
    have = open(OUT, encoding="utf-8").read().splitlines(True)
    keep, dropped, moved, scrubbed = [], [], 0, 0
    for l in have:
        if not l.startswith('{ "'):
            keep.append(l)
            continue
        name = l[3:].split('"', 1)[0]
        if name in names or (prefixes and name.startswith(prefixes)):
            dropped.append(name)
            continue
        m = re.search(r'"read at ([^":]+)(:\d+)?"', l)
        if m and is_closed_path(m.group(1)):
            if name not in in_tree:
                dropped.append(name)
                continue
            l = l[:m.start()] + '"read at %s"' % in_tree[name] + l[m.end():]
            moved += 1
        for r in path_res:
            l2 = r.sub("the PXQN library", l)
            if l2 != l:
                scrubbed += 1
                l = l2
        keep.append(l)
    hashes = sorted({lever_hash(n) for n in dropped} | {lever_hash(p + "*") for p in prefixes})
    keep.append("// built-in: " + " ".join(hashes) + "\n")
    # the closed LEVERS (scripts/pxa-closed-levers.txt: the Overdrive set the Flash-Next preset turns on, PXA4, Tierpipe)
    keep.append("// built-in-preset: " + " ".join(sorted({lever_hash(n) for n in names} |
                                                          {lever_hash(p + "*") for p in prefixes})) + "\n")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(keep)
    os.replace(tmp, out_path)
    if closed_out:
        allc = sorted(set(dropped) | {p + "*" for p in prefixes})
        tmp = closed_out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("// GENERATED by scripts/pxa-lever-catalog.py --release-filter: the levers the closed library reads\n")
            f.writelines('"%s",\n' % n for n in allc)
        os.replace(tmp, closed_out)
    print("pxa-lever-catalog: release filter kept %d rows, dropped %d closed, re-pointed %d, scrubbed %d"
          % (sum(1 for l in keep if l.startswith('{ "')), len(dropped), moved, scrubbed))
    return 0


# --assert-clean INC FILE...: the release guard (make-release-tarball.sh, docker/Dockerfile). Exit 1 naming every FILE
# that holds closed content: a catalog copy (.inc/.txt) any name from INC (the --closed-out list) or a closed path; a
# binary any closed LEVER name (scripts/pxa-closed-levers.txt; the other dropped names belong to closed code that private
# builds compile in, e.g. the lock in libggml-pxqn).
def assert_clean(inc, files):
    every = [l.strip().rstrip(",").strip('"') for l in open(inc, encoding="utf-8") if l.startswith('"')]
    cn, cp = closed_set()
    levers = sorted(cn) + [p + "*" for p in cp]

    def pats(names):
        return [(n, re.compile(re.escape(n[:-1]).encode() if n.endswith("*")
                               else re.escape(n).encode() + rb"(?![A-Z0-9_])")) for n in names]
    p_every, p_levers = pats(every), pats(levers)
    paths = [p for p in closed_files() if "/" in p]
    names = every
    bad = 0
    for f in files:
        text = f.endswith(".inc") or f.endswith(".txt")
        try:
            data = open(f, "rb").read()
        except OSError as e:
            print("pxa-lever-catalog: cannot read %s: %s" % (f, e))
            bad += 1
            continue
        hits = sorted({n for n, p in (p_every if text else p_levers) if p.search(data)})
        if text:
            t = data.decode("utf-8", "replace")
            hits += sorted({p for p in paths if p in t})
        if hits:
            bad += 1
            print("pxa-lever-catalog: CLOSED content in %s: %s" % (f, " ".join(hits[:12]) + (" ..." if len(hits) > 12 else "")))
    if not bad:
        print("pxa-lever-catalog: clean: %d file(s), no closed lever name (%d checked)" % (len(files), len(names)))
    return 1 if bad else 0


def main():
    if "--public-filter" in sys.argv:
        return public_filter()
    if "--assert-clean" in sys.argv:
        a = sys.argv[sys.argv.index("--assert-clean") + 1:]
        return assert_clean(a[0], a[1:])
    if "--release-filter" in sys.argv:
        a = sys.argv[1:]
        out = a[a.index("--release-filter") + 1]
        co = a[a.index("--closed-out") + 1] if "--closed-out" in a else None
        return release_filter(out, co)
    names, rows = generate()
    text = emit(rows)
    if "--check" in sys.argv:
        have = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        declared = set(re.findall(r'^\{ "(PX[AQ]_[A-Z0-9_]+)"', have, re.M))
        missing = sorted(set(names) - declared)
        if missing:
            print("pxa-lever-catalog: %d lever(s) in the tree have no catalog row (run scripts/pxa-lever-catalog.py):"
                  % len(missing))
            for m in missing:
                print("   %s  (%s)" % (m, names[m]))
            return 1
        print("pxa-lever-catalog: OK, %d levers declared, every lever in the tree has a row" % len(declared))
        return 0
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote %s: %d rows (%d curated, %d from docs/LEVERS.md)" % (
        os.path.relpath(OUT, ROOT), len(rows), sum(1 for r in rows if r[0] in CURATED),
        sum(1 for r in rows if r[0] not in CURATED and r[5].startswith("docs/"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
