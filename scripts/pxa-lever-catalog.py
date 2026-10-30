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
                                                 and empty the evidence ids (scripts/pxa-public-scrub.py)
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
    "PXA_AUTOCONFIG": ("on (unset)", "every common tool (llama-server, llama-cli, llama-bench, ...)", "rule", "",
                       "=0: the whole autoconfig registry stands down in every tool (no split / -ngl / -c / KV picks, no lever env such as the P60 stack); kernels keep their own config level; the registry is applied after the GGUF header, before model load, with setenv no-overwrite (a lever named in the environment wins; an empty PXA_X= counts as unset); =0 also stops the server's direct batch/ub choices (measured cell, fit walk, stream -ub, -ub rule, VRAM ladder): stock -b/-ub. Gate tools (llama-perplexity, llama-imatrix) get the lever env only; a tool's -c pick never shrinks the trained window; llama-bench applies only the picks every test instance shares"),
    "PXA_HS_ENV_UNION": ("off", "a process with several models (llama-server --hot-model, a draft model in a common tool)", "lever-off", "",
                         "=1: every model's registry lever picks are applied (first-wins union, the pre-2026-10-16 behaviour); default: only the first (-m) model's picks are applied and a later model's differing picks are named in a WARNING line"),
    "PXA_REFERENCE": ("off", "any", "lever-off", "", "=1: every PXA lever off, the reference path (A/B baseline); the PXQ codec kernels, VOLTA_F16_GEMM and MXFP4_DEQ_V2 stay on (bug #281)"),
    "PXA_MODE": ("balance", "any", "default-on", "", "balance = -fa on serving; max = -fa off ingest"),
    # ---- engine-side serve flags (this registry) -----------------------------------------------
    "PXA_AUTO_SM": ("rule", "2 identical cards, or 4 identical P100s; arch qwen35; tier PXQ4/PXQN4/PXQN4S8/PXQN5; -fa on",
                    "rule", "",
                    "-sm unset -> tensor on a pair the capability check admits and on 4x P100 (measured 2026-09-27), layer elsewhere"),
    "PXA_AUTO_SM_PXQN": ("on", "-sm unset; PXQN4/PXQN4S8/PXQN5 files; the OneCard N4+N3+N5 mix on a P100 pair", "default-on", "",
                         "=0: the auto tensor split admits PXQ4 files only (the pre-2026-09-27 rule)"),
    "PXA_AUTO_BATCH": ("rule", "measured cells (sm_60/sm_61/sm_70 x card count x file)", "rule", "",
                       "-b/-ub unset -> the measured cell, else the adaptive VRAM ladder"),
    "PXA_AUTO_UB_STREAM": ("rule: 4096 (=0 off)", "1 card >= 15 GiB, file > free VRAM (streamed MoE experts), -ub unset", "rule", "",
        "=1: -ub/-b 4096 ahead of the measured batch cell (cold experts re-cross PCIe once per ubatch); Flash-Next 1x V100 28k 324 -> 597 t/s"),
    "PXA_NUMA_INTERLEAVE": ("rule: on for glm5next with host experts (=0 off, =1 any model with a host buffer; off at PXA_REFERENCE)",
        "Linux, >= 2 allowed memory nodes, no memory policy set yet (PXA_NUMA_BIND did not bind, no numactl)", "rule", "",
        "host pages (pinned expert buffers, loader threads, CPU pool) interleaved over every allowed node before the first weight is allocated; GLM-5.3-Flash 1x V100 + host experts, threads off the card's node: +9.3/+17.6/+9.0% decode, identical output"),
    "PXA_PLE_PIPELINE": ("rule: on with copy slots on exactly 4 sm_60 cards (=0 off, =1 any card set)", "qwen4exp PLE (Flash-Next) with scheduler copy slots (PXA_PIPELINE_PP), -sm layer", "rule", "",
        "on: no per-graph PLE barrier, the -sm layer prefill pipelines (4x P100 pp20801 +23.6%, pp3121 +14.3%, needle/greedy identical); never acts without copy slots; PXA_REFERENCE=1 off unless named"),
    "PXA_AUTO_UB_STREAM_DECODE": ("on (=0 off)", "server, 1x sm_70, qwen35moe PXQN4, free VRAM < file <= 1.5x free, -ub unset", "default-on",
        "",
        "on: the streamed-expert rule picks -ub 2048 (decode-sized compute buffer, more experts resident); =0: 4096 as PXA_AUTO_UB_STREAM"),
    "PXA_STREAM_UB_CKPT": ("off (=1 on)", "the -ub PXA_AUTO_UB_STREAM picked (never an explicit -ub), a recurrent model with a speculative chain", "lever-off",
        "",
        "on: the stream -ub is a ceiling; the context (not the placement) is rebuilt 512 narrower, floor 2048, until the per-step checkpoint budget affords min(chain+1, PXA_XCACHE_VERIFY_WIDTH) rows (Flash-Next 1x V100: 4096 gave 2-3 of 4 rows, MTP n_max 3 -> 1-2); unset/=0: the stream -ub as picked. Opt-in: decode +1..+9% at N=2 NOT-QUIET, one more ubatch for prompts in (3584k, 4096k]"),
    "PXA_AUTO_XCACHE_SMALLTAIL": ("on (=0 off)", "1x sm_70, qwen35moe PXQN4, VRAM < file <= 1.5x VRAM, config level >= 2", "default-on", "",
                                  "on: the expert cache's async CPU-pool cold path (PXA_XCACHE_ASYNC + PXA_XCACHE_CPUPOOL, pool CPUs derived like the Overdrive cell, PXA_XCACHE_IDLE_BACKOFF_MS=200; bit-identical to the split path at the same plan) and the planner's small-tail price (more, thinner cold layers); =0: the split path and the 1300 us price (pass-8f text); any one switch in the environment is kept"),
    "PXA_AUTO_FLINT_1CARD": ("on (=0 off)", "1x sm_60, flint PXQN4 (or the PXQN4+PXQN5 mix), VRAM < file <= 1.5x VRAM, config level >= 2", "default-on", "",
                             "on: PXA_XCACHE_SMALLTAIL_SM60=1 (the expert-cache small-tail plan on that P100: one-layer ring + 384 MiB slack credit, one more layer resident); =0: the generic sm_60 plan (two-layer ring, no credit)"),
    "PXA_AUTO_FLINT_ROUTER": ("on (=0 off; PXA_AUTO_FLINT_1CARD=0 too)", "1x sm_60 or sm_70, flint PXQN4 (or the PXQN4+PXQN5 mix), VRAM < file <= 1.5x VRAM, config level >= 2", "default-on", "",
                             "on: PXA_ROUTER_BF16=2 (one F32-widened BF16 router GEMV per layer instead of cuBLAS BF16 + two conversions; P100 0.97 -> 0.42 ms/token; not bit-identical, ships on the gates); =0: the cuBLAS path"),
    "PXA_AUTO_FLINT_V100": ("on (=0 off; PXA_AUTO_FLINT_1CARD=0 too)", "1x sm_70, flint PXQN4 (or the PXQN4+PXQN5 mix), VRAM < file <= 1.5x VRAM, config level >= 2", "default-on", "",
                            "on: PXA_CUDA_GRAPH_DECODE=1 + PXA_CUDA_GRAPH_MOE=1 + PXA_CUDA_GRAPH_LRU=256 + PXA_SCHED_ASYNC_INPUTS=1 + PXA_PLAIN_GREEDY_AMAX=1 + PXA_ARGMAX_VAL_MB=1 (graphs on the launch-bound decode segments between the cold layers; all bit-identical); =0: eager segments"),
    "PXA_XCACHE_SMALLTAIL_SM60": ("0 (=1 on)", "expert cache, 1x sm_60, CPU cold path, deficit <= free VRAM / 2", "lever-off", "",
                             "=1: one sm_60 card enters the C042 small-tail scope (one-layer ring for wide graphs + the PXA_XCACHE_SLACK_MB auto credit); the registry sets it for Flint PXQN4 on one P100 (PXA_AUTO_FLINT_1CARD); default 0 = sm_70 only"),
    "PXA_XCACHE_SMALLTAIL_COST": ("on (=0 off)", "expert cache, 1x sm_70, CPU cold path, deficit <= free VRAM / 2, PXA_XCACHE_ASYNC on", "default-on", "",
                                  "on: a cold layer is priced at the async path's 100 us (35B-A3B one V100: L 9 -> 17, miss 14.8% -> 5.6%); =0: the split path's 1300 us with the async pool kept; PXA_XCACHE_COST_LAYER_US overrides both"),
    "PXA_XCACHE_RING1": ("on (=0 off)", "expert cache, 1x sm_70, CPU cold path, deficit <= free VRAM / 2", "default-on", "",
        "on: the cold-stack ring holds one layer, the freed VRAM keeps hot experts; =0: two layers as before"),
    "PXA_XCACHE_SLACK_MB": ("auto (384 in the PXA_XCACHE_RING1 scope; =N sets, =0 off)", "expert cache planner", "default-on", "",
        "unset: 384 MiB of the planner's margin go to resident experts in the small-tail scope only; =N: N MiB on any plan; =0: none"),
    "PXA_XCACHE_RING_LEND": ("0 (=1 on)", "expert cache, CPU cold path (narrow graphs read no ring)", "lever-off", "",
        "=1: the planner does not charge the cold stacks' ring (its ~1 GiB keeps resident experts) and the CUDA backend allocates it on demand for a graph at or above the CPU-alias width (32 x n_expert / n_expert_used tokens; 1638 on Flash-Next), returning it at the next sync; short of VRAM that graph reads the pinned experts in place (zero-copy, as PXA_STREAM_ZC_NY does), never an abort; correct-not-identical vs =0 (more experts on the GPU); =0: the previous plan exactly"),
    "PXA_OVERDRIVE_V100_ADAPT": ("on (=0 off)", "the 1x V100 Overdrive Flash-Next class (qwen4exp PXQN + MTP head, cold experts in host RAM; Linux)", "default-on", "",
        "on: the Overdrive recipe on one V100 adapts the resident experts (PXA_XCACHE_ADAPT=1, _ADAPT_RESET=0, _RTIER=1, _SWAP_CAP=8, _LINK_FRAC=0.01, _EVICT_FILE=1, _RING_LEND=0, PXA_PXQN_RIX_MAXW=0, PXA_PLE_DIRECT=1; RIX keeps every bit placement-independent); =0: recipe E (frozen residency) on the V100 too; each row also yields to its own variable"),
    "PXA_XCACHE_RTIER": ("off (=1 on); on in the 1x V100 Overdrive Flash-Next class (qwen4exp PXQN + MTP head, cold experts in host RAM; Linux)", "PXA_XCACHE_ADAPT=1", "default-on", "",
        "=1: the residency policy runs on a planner thread (the decode thread only reads results and starts copies), at most PXA_XCACHE_SWAP_CAP swaps per decode, swaps applied on copy completion, the hot set carried across requests; =0: the policy on the decode thread"),
    "PXA_XCACHE_SWAP_CAP": ("2; 8 in the 1x V100 Overdrive Flash-Next class (qwen4exp PXQN + MTP head, cold experts in host RAM; Linux)", "PXA_XCACHE_RTIER=1", "site", "",
        "swaps one decode may start (the planner's budget is the smaller of this and the link budget)"),
    "PXA_XCACHE_EVICT_FILE": ("off (=1 on); on in the 1x V100 Overdrive Flash-Next class (qwen4exp PXQN + MTP head, cold experts in host RAM; Linux)", "PXA_XCACHE_ADAPT=1; Linux", "default-on", "",
        "=1: a swap's evict reads the expert's raw bytes back from the model file into the free cold slot on an IO thread (one descriptor per model file kept open) instead of a device-to-host copy, so the link carries only the admit; byte-identical slots (tests/test-pxa-xcache-adapt); =0: the D2H evict"),
    "PXA_PXQN_RIX_MAXW": ("8 (the async cold lane's width); 0 (every width) in the 1x V100 Overdrive Flash-Next class (qwen4exp PXQN + MTP head, cold experts in host RAM; Linux)", "PXA_PXQN_RIX on", "site", "",
        "the widest graph (tokens) the rank-identical expert kernels cover; 0 = every width (prefill included), so an expert's placement never changes a bit"),
    "PXA_XCACHE_LINK_FRAC": ("0.12; 0.01 in the 1x V100 Overdrive Flash-Next class (qwen4exp PXQN + MTP head, cold experts in host RAM; Linux)", "PXA_XCACHE_ADAPT=1", "site", "",
        "share of the link one observation step's swap copies may take (0.01..0.9)"),
    "PXA_XCACHE_RING_LEND_KEEP_MB": ("512", "PXA_XCACHE_RING_LEND=1", "lever-off", "",
        "free VRAM the on-demand (lent) ring never takes, for the CUDA pool that grows during the graph"),
    "PXA_XCACHE_RING_LEND_FAIL": ("0 (=1 test hook)", "PXA_XCACHE_RING_LEND=1, tests only", "diagnostic", "",
        "=1: every on-demand allocation of the lent ring fails, so a wide graph takes the zero-copy fallback (tests/run-xcache-ring-lend.sh)"),
    "PXA_MTP_RESERVE_RECOMPUTE": ("0 (=1 on)", "expert cache with MTP (PXA_XCACHE_MTP=1) on a Gated-DeltaNet model, one sequence", "lever-off", "",
        "=1: the planner prices the MTP chain reserve as replay rollback (one start state per recurrent layer + per-token recurrence inputs and conv rows) instead of width-1 per-step state rows, and PXA_SPEC_CKPT_REPLAY is forced on unless set; a rejected tail's state is recomputed by the same kernel (bit-identical to per-step at equal depth), the freed planes keep resident experts; =0: the previous reserve exactly"),
    "PXA_PLACE_CTX_GUARD": ("on (=0 off)", "PXA_STREAM_WEIGHTS=auto planner, any streaming plan; all-outputs loads (perplexity, KLD, imatrix, --embedding)", "default-on",
        "",
        "on: a streaming plan keeps the pool floor free after the GEMM pool for what the context really holds (an all-outputs load: full logits + the head's wide GEMM); capped contexts plan exactly as before, a plan that already met it is unchanged; =0: the previous plan (bug #339: Nemotron-3-Nano-30B PXQ4 1x V100 perplexity -c 2048 refused with 7.5 MiB free)"),
    "PXA_DN_PREFETCH": ("rule: on for sm_60 (unset), off elsewhere; =0 off, =1 on everywhere", "DeltaNet prefill (head 128, n_tokens > 8)", "rule", "",
        "next token's q/k/v/g/beta in flight in the recurrent kernel; bit-identical (greedy512 sha equal); 27B one P100 pp4096 +0.26% (2026-10-19)"),
    "PXA_PXQN_GU_DOWN": ("on (=0 off)", "sm_60 PXQN prefill FFN (FUSED_UP_GATE silu -> down_in RHT -> down MUL_MAT, k-scalar GEMM)", "default-on", "",
        "the GLU is computed in the down GEMM's x pre-pass (no GLU pass, no f32 GLU store/read-back); bit-identical (same SASS sequence, greedy512 sha equal); +0.34% C017 pp4096 P100; =0 -> separate GLU pass"),
    "PXA_DN_COLWARP": ("off (=1 on)", "DeltaNet prefill (head 128, scalar gate, n_tokens > 8)", "lever-off", "",
        "=1: column-warp prefill kernel, no block barrier per token; not bit-identical (sum order)"),
    "PXA_AUTO_UB_VOLTA_Q8": ("rule", "1x sm_70; q8_0 K+V; -c >= 49152; no batch cell", "rule", "",
                             "-ub 1024 instead of the VRAM ladder's 2048 so the Volta MMA q8 attention route keeps its staging margin (642.3 vs 387.5 t/s pp65536); =0 stands down"),
    "PXA_AUTO_UB_RESERVE_MB": ("512", "an -ub the engine picked (rule or VRAM ladder)", "rule", "",
                               "VRAM that must stay free on every card after load; an engine-picked -ub steps down 1024>768>512>256 until it does, else a clean refusal; 0 = no check"),
    "PXA_FA_DEEP_QKV_TILE": ("1 (AUTO)", "sm_70; quantized K/V; whole f16 staging > 128 MiB", "rule", "",
                             "deep quantized-KV attention goes to tile (chunked) / vec only when the whole-tensor f16 staging does not fit (+256 MiB margin); where it fits the release route runs (bit-identical to v2026.09.20); =2 the unconditional 128 MiB rule, =0 off"),
    "PXA_FA_F16_KV_CHUNK": ("AUTO", "quantized K/V; tile-f16 kernel", "rule", "",
                            "unset = chunk (8192 tokens) only past a 128 MiB whole-tensor scratch that does not fit; N > 0 forces N-token chunks; 0 = whole-tensor conversion"),
    "PXA_PPL_STREAM": ("off", "llama-perplexity", "tool", "",
                       "=1 scores each batch's rows as it is decoded: long-context PPL / KLD without holding n_ctx/2 x n_vocab logits (32 GB at 64k); same rows, order and base-file layout"),
    "PXA_DN_GM_FAULT": ("off", "any", "diagnostic", "",
                        "=1 test hook: the fused deltanet gather+mask launch behaves as refused for memory, proving the clean decline (tests/test-pxa-dn-gm-refused)"),
    # ---- server speed history (speed-graph, 2026-09-27; examples/server/pxa-stats.h) -----------
    "PXA_STATS": ("on", "llama-server", "default-on", "",
                  "=0: no speed history, GET /pxa/stats and /pxa/speed unregistered (zero overhead)"),
    "PXA_STATS_MAX": ("10000", "llama-server", "default-on", "", "records kept in the in-memory speed history ring"),
    "PXA_STATS_FILE": ("off", "llama-server", "lever-off", "",
                       "=<path> or =default (~/.cache/pxa/speed-stats.jsonl): append each record as JSONL and reload the tail at start"),
    "PXA_STATS_FILE_MB": ("8", "llama-server; PXA_STATS_FILE set", "default-on", "", "rotate the history file at this size (-> <path>.1)"),
    # ---- multi-GPU peer copies (rel-fixes, 2026-09-27) -----------------------------------------
    "PXA_P2P_SELFTEST": ("on", "2+ CUDA cards with a peer path", "default-on", "",
                         "startup self-test of every peer pair (4 MiB DMA + direct peer reads, compared on the host); a pair that corrupts keeps peer access off and copies through system memory; =0 skips the test"),
    "PXA_P2P_SELFTEST_CORRUPT": ("off", "test hook", "diagnostic", "",
                                 "=1 flips one received word so the fallback branch runs on a healthy system (tests/test-pxa-p2p-selftest)"),
    # ---- speculative verify (rel-fixes, 2026-09-27) --------------------------------------------
    "PXA_SPEC_FIXED_WIDTH": ("off", "llama-server speculation; recurrent models only with a per-step checkpoint", "lever-off", "",
                             "=1 pads every verify batch to the slot's draft ceiling with never-accepted filler rows, so the verify width no longer follows the draft length (spec-verify-batch-invariance)"),
    "PXA_SPEC_DEPTH_CUTOFF": ("off", "serial MTP draft chain", "lever-off", "",
                              "=P (0<P<1): stop the draft chain when its cumulative probability falls below P * min(1, n_past / PXA_SPEC_DEPTH_CUTOFF_CTX)"),
    "PXA_SPEC_DEPTH_CUTOFF_CTX": ("65536", "serial MTP draft chain", "lever-off", "",
                                  "context depth at which the PXA_SPEC_DEPTH_CUTOFF floor reaches its full value"),
    # ---- MTP draft depth by context depth (spec-long, 2026-09-27) --------------------------------
    "PXA_SPEC_DEPTH_NMAX": ("rule", "MTP drafter; one CUDA card (sm_70 or sm_60)", "rule", "",
                            "'D0:N0,D1:N1,...' caps the MTP chain at Ni from context depth Di on (a cap under n_max); AUTO 1x sm_70 '0:1,8192:2,16384:3', 1x sm_60 '0:1', else off; =0 off"),
    # ---- server prompt cache / slots (ws6-server, 2026-09-25) ----------------------------------
    "PXA_CACHE_HYBRID_USABLE": ("on", "recurrent/hybrid arch; --cache-ram > 0", "default-on", "",
                                "RAM prompt cache restores a parked entry only where the recurrent state can be re-entered (extension or checkpoint below the divergence); =0 restores the similarity scan"),
    "PXA_SLOT_VICTIM": ("lru", "-np >= 2", "lever-off", "",
                        "=cheap: a request that matches no slot overwrites the idle slot with the fewest cached tokens (LRU tie-break); unmeasured, off"),
    "PXA_AUTO_UB_LONG": ("rule", "4x sm_60", "rule", "",
                         "4x P100: dense file -ub 256, expert file -ub 2048; =0 flat 2048, =1 force 256"),
    "PXA_AUTO_TS": ("on", "1x sm_70 + 1x sm_60, -sm layer", "rule", "",
                    "mixed V100+P100 pair: -ts 1.4,0.6 when -ts is unset"),
    "PXA_AUTO_SAMPLERS": ("on", "arch family table", "default-on", "", "model-family sampler defaults for unset --temp/--top-k/--top-p/--min-p"),
    "PXA_AUTO_SPEC": ("rule", "arch x card class", "rule", "",
                      "arms the measured speculation default per family; declines on tight VRAM"),
    "PXA_KV_UNIFIED_DEFAULT": ("off", "-np > 1", "lever-off", "", "=1: -np > 1 with no explicit choice -> --kv-unified"),
    "PXA_EXPLAIN": ("off", "any", "diagnostic", "",
                    "=1: print the settings the engine would pick (JSON line) and exit before loading; =levers dumps this table"),
    "PXA_TOPOLOGY": ("detect", "any", "diagnostic", "",
                     "explain-only card set, e.g. 2x600 or 700,700; no CUDA context is created"),
    # ---- hot swap (--hot-model; examples/server/pxa-hotswap.h, ggml/src/pxa-residency.h) ------
    "PXA_SWAP_KEEP": ("off", "--hot-model", "lever-off", "",
                      "=1: a swap releases only what the incoming model needs (scratch, then mirrored weights, then KV); the rest stays on the cards and the next swap back copies only what left"),
    "PXA_SWAP_HEADROOM_MB": ("1024", "--hot-model + PXA_SWAP_KEEP=1", "lever-off", "",
                             "free MiB left per card beside the incoming model under PXA_SWAP_KEEP; growth past it is reclaimed from parked models"),
    "PXA_SWAP_PIN_BUDGET_MB": ("MemAvailable - 8 GiB", "--hot-model", "rule", "",
                               "cap on the pinned host RAM the weight/KV mirrors may take; a park past it fails and the model stays on the cards"),
    "PXA_SWAP_VERIFY": ("off", "--hot-model", "diagnostic", "",
                        "=1: every park re-reads the weights and counts the bytes that changed since they were mirrored (should be 0)"),
    "PXA_SWAP_PEER_ACCESS": ("on", "--hot-model, 2+ cards", "default-on", "",
                             "=0: residency mappings are private to their card (the fused tensor-split reduce reads peer memory, so leave it on)"),
    "PXA_SWAP_SLAB_VA_GB": ("32", "--hot-model, -sm tensor/graph", "diagnostic", "",
                            "virtual range reserved per card per class for split slices (address space only)"),
    "PXA_HOT_UNKNOWN": ("404", "--hot-model", "lever-off", "",
                        "=active: a request naming an unregistered model is served by the model on the cards instead of a 404"),
    "PXA_PROFILE_KEEP_TARGET": ("1 (=2 also any server -md draft, =0 off)", "a second model loaded after the target: the gemma4_mtp assistant drafter; with =2 also a separate-file -md draft", "default-on", "",
                                "the PXA model profile is one per process; a companion load keeps the target's profile, so the target's model-aware levers (ROUTER_FUSE, MOE_DEVICE_MAP, PXQ_MMVQ, PXQ_GEMM_2D) keep their values (test-pxa-profile-keep); =2 is unmeasured on dense/non-PXQ drafts, so opt-in; =0 = last load wins"),
    # ---- Gemma 4 MTP verify / prefill / tsplit levers (2026-10-03; docs/GEMMA-MTP-VERIFY-LEVERS.md); default off, except the
    # members of the all-V100 PXA_GEMMA4_V100_SET (src/llama.cpp: MOE_VERIFY2, _TILE 4x4, _TILE_W1, NO_LOGITS_COPY, ROWS_NY 4, DENSE_FUG_NY) --
    "PXA_MOE_VERIFY2": ("rule: on with the PXA_GEMMA4_V100_SET set (Gemma 4 sparse, all-V100 box, config level >= 2), else off (=1 on)", "MoE id-GEMV, 2..4-token verify batches", "rule", "",
        "=1: one expert-grouped id-GEMV launch per MoE projection for a 2..4-token verify batch; bit-identical (tests/test-moe-verify2); stands aside under PXA_MMVQ_MOE_NWARPS=2/4 (docs/GEMMA-MTP-VERIFY-LEVERS.md)"),
    "PXA_MOE_VERIFY2_MAX_NY": ("4", "PXA_MOE_VERIFY2=1", "site", "",
        "widest verify batch (2..4) the verify2 MoE kernel serves"),
    "PXA_MOE_VERIFY2_TILE": ("rule: on with the PXA_GEMMA4_V100_SET set (Gemma 4 sparse, all-V100 box, config level >= 2): 4x4; else off (0x0, untiled)", "PXA_MOE_VERIFY2=1", "rule", "",
        "=WxR (W warps in 1/2/4/8, R rows/warp in 1/2/4): multi-warp, multi-row blocks for the verify2 MoE GEMV; bit-identical; 4x4 measured best on V100"),
    "PXA_MOE_VERIFY2_TILE_UP": ("= PXA_MOE_VERIFY2_TILE", "PXA_MOE_VERIFY2=1", "lever-off", "",
        "=WxR: tile for the fused up/gate projection only"),
    "PXA_MOE_VERIFY2_TILE_DOWN": ("= PXA_MOE_VERIFY2_TILE", "PXA_MOE_VERIFY2=1", "lever-off", "",
        "=WxR: tile for the down projection only"),
    "PXA_MOE_VERIFY2_TILE_W1": ("rule: on with the PXA_GEMMA4_V100_SET set (Gemma 4 sparse, all-V100 box, config level >= 2), else off (=1 on)", "PXA_MOE_VERIFY2=1 + a tile set", "rule", "",
        "=1: one-token decode MoE also goes through the tiled verify2 kernel; bit-identical"),
    "PXA_MOE_VERIFY2_LOG": ("off", "PXA_MOE_VERIFY2=1", "diagnostic", "",
        "=1: log verify2 dispatches"),
    "PXA_VERIFY_NO_LOGITS_COPY": ("rule: on with the PXA_GEMMA4_V100_SET set (Gemma 4 sparse, all-V100 box, config level >= 2) and the PXA_FLASHNEXT_V100_SET set (qwen4exp, all-V100 box, config level >= 2), else off (=1 on, =0 off)", "Gemma 4 target + assistant drafter, qwen4exp verify + MTP draft; PXA_VERIFY_ARGMAX on", "rule", "",
        "=1: greedy verify and drafter argmax draws read back the device argmax, not the full logit row; anything that needs logits keeps the copy"),
    "PXA_FLASHNEXT_V100_SET": ("on (=0 off)", "qwen4exp (Flash-Next), all-V100 box, config level >= 2, GPU offload", "default-on", "",
        "arms PXA_VERIFY_NO_LOGITS_COPY=1 and PXA_PXQN_KSB=16 (setenv no-overwrite: each lever's own value wins) and, for an MTP row n_max 3/4 at p_min 0.75 at -np 1, PXA_MTP_DEEP=3:0.5:0.8 with the MTP ceiling raised to 5 (examples/server/pxa-fn-v100-mtp.h); greedy512 sha equal off vs on; =0 turns the set off"),
    "PXA_MTP_DEEP": ("rule: 3:0.5:0.8 with the PXA_FLASHNEXT_V100_SET set (qwen4exp, all-V100, ENHANCE, -np 1, MTP row n_max 3/4 at p_min 0.75; MTP ceiling raised to 5), else off (=B[:PD[:T]] on, =0 off)", "MTP self-speculation drafter (serial + batched slots); n_max is the ceiling", "rule", "",
        "acceptance-adaptive draft depth upward: base depth B below acceptance EMA T, n_max with p_min PD at or above it; greedy output unchanged; =0 off (the set then leaves the --spec-type row as written)"),
    "PXA_PXQN_KSB": ("rule: 16 with the PXA_FLASHNEXT_V100_SET set, else 0 (off)", "sm_70; int8 N1/N2/N5 decode GEMVs (ns=1) of <= N panels", "rule", "",
        "=N: one block per (panel, k-segment group) plus a last-block reduction for GEMVs of <= N panels; bit-identical to k_pxqn_mmv; =0 off"),
    "PXA_PXQN_KSB_KPB": ("1", "PXA_PXQN_KSB on", "site", "", "k-segments per block (1, 2, 4); bit-identical"),
    "PXA_PXQN_KSB_PF": ("2", "PXA_PXQN_KSB on", "site", "", "slabs in flight (2 or 4); bit-identical"),
    "PXA_MMVQ_V100_ROWS_NY": ("rule: on with the PXA_GEMMA4_V100_SET set (Gemma 4 sparse, all-V100 box, config level >= 2): 4; else off (unset/0)", "sm_70; mmvq at 2..3 columns, large nrows (lm_head)", "rule", "",
        "=2|4: multi-row-per-block mmvq; bit-identical"),
    "PXA_PXQN_MOE_HSCALE": ("per-model hint: 5 for arch flint, else 0 (=0 off, =k forces 2^-k on every model)", "PXQN routed experts, tiled prefill GEMM (> PXA_PXQN_MOE_DIRECT_MAX tokens) with the down in the same pass", "rule", "",
        "the half2-staged silu(gate)*up is stored as h*2^-k and the down output multiplied by 2^k: |h| up to 65504*2^k stays finite (Flint layer 46 reached 76.7k -> inf -> NaN); exact power-of-two scaling, 0 = unchanged bits"),
    "PXA_CB_AMAX": ("off (=1 on)", "llama-eval-callback with PXA_CB_DIGEST", "diagnostic", "",
        "=1: the digest line of an F32 node also carries the finite absmax and the non-finite count (first op leaving the f16 range / turning NaN); unset = the digest line unchanged"),
    "PXA_CB_SPECIAL": ("off (=1 on)", "llama-eval-callback prompt tokenization", "diagnostic", "",
        "=1: special tokens in the prompt text (chat-template markers such as <|user|>) tokenize as special tokens; unset = plain text as before"),
    "PXA_KQMMV_VERIFY_ID": ("4 (verify widths 2..4); 0 off; 2..8 the widest width served", "sm_60 KQMMV (h2) GEMVs whose one-column launch stages the whole activation", "default-on", "",
        "verify widths keep the KQMMV arithmetic (chunked k_kq_mmv form) instead of the q8_1 MMVQ once ny x K no longer fits the block (small grouped launches at widths 3..4: PXA_KQMMV_VERIFY_ID_SMALL); every verify column is bitwise the one-token decode GEMV (Nemotron Super ssm_out Q8_0 K 8192, attn k/v); plain decode untouched; =0 the previous rule"),
    "PXA_KQMMV_VERIFY_ID_SMALL": ("rule: on for nemotron_h_moe, qwen4exp and qwen35 with an sm_60 device (registry), else off (=1 on; a switch, not a width)", "sm_60 small grouped Q8_0 KQMMV launches (attention k/v, delta-net beta/alpha) at verify widths 3..4; with it on, launches of <= 1024 rows up to PXA_KQMMV_VERIFY_ID_SMALL_W", "rule", "",
        "=1: those launches keep the KQMMV arithmetic (chunked form) instead of the q8_1 MMVQ, bitwise the one-token decode (Nemotron Super K = 2/3; Hemlock K = 5/6/8 bi-probe 20/20, 24/24, 24/24, was 0/30; 27B W3 MTP 2x P100 qwen35 sm_60: K=3/4 verify 0/48 -> 48/48, MTP greedy == plain); scoped by the registry (27B W3 MTP: text changed, -1..-2%)"),
    "PXA_FA_VEC_NY1_SPLIT": ("rule: on for nemotron_h_moe and for qwen35 with an sm_60 device (registry), else off (=1 on)", "one-column-per-block FA vec kernel (NVIDIA f32 vec) at a 2..8-column verify batch", "rule", "",
        "the verify batch takes the KV split of the one-token decode step (occupancy heuristic for one column tile): every verify column is bitwise the width-1 attention at any n_kv (Nemotron Super n_kv ~1860: 0/32 -> fix); decode + prefill untouched; =0 the split over all column tiles"),
    "PXA_F32_VERIFY_COLSPLIT": ("on (=0 off)", "pre-Volta, f32 weight on the cuBLAS path (MoE router class), 2..8-column verify batch", "default-on", "",
        "one cuBLAS call per verify column, the exact call of the one-token decode step: router logits bitwise the plain decode's; plain decode untouched; =0 one GEMM over all columns"),
    "PXA_MMID_F16_GUARD": ("on (=0 off)", "CUDA MUL_MAT_ID per-expert loop and the fused MoE up/gate per-expert loop (non-q8_1 branch), f32 activations, at the cuBLAS f16 src1 conversion (dequant + cuBLAS: PXQ4 at > 8 rows/expert, MMQ types on Volta at the cuBLAS width)", "default-on", "",
        "a gathered row with finite absmax >= 65520 is converted to f16 as x*2^-k and its output multiplied by 2^k (was inf/NaN: Nemotron-3-Nano PXQ4 V100 KLD NaN, bug #338); in the fused loop the up and gate outputs are multiplied back before the bias and the GLU; rows below the threshold and every route that does not convert src1 to f16 (MMQ/MMVQ q8_1, DMMV, PXQ kernels, f32 cuBLAS) are bit-identical to =0; =0: plain copies, rows >= 65520 overflow on the f16 routes"),
    "PXA_F16_GUARD_DENSE": ("on when PXA_MMID_F16_GUARD is on (=0 off)", "CUDA dense MUL_MAT on the cuBLAS f16 route (f16 weights, dequantised PXQ/MMQ types at the cuBLAS width; the fused MoE down projection), f32 activations, outside a MUL_MAT_ID loop", "default-on", "",
        "a src1 column with finite absmax >= 65520 is converted to f16 as x*2^-k (k from its exponent, absmax*2^-k < 4096) and its output column multiplied by 2^k (was inf/NaN, review item); exact: the column equals 2^k times the stock result on the pre-scaled input; columns below the threshold and every non-f16 route are bit-identical to =0; PXA_MMID_F16_GUARD=0 turns it off too"),
    "PXA_F16_GUARD_OUT": ("on when PXA_MMID_F16_GUARD is on (=0 off)", "CUDA cuBLAS f16 route with an f16 OUTPUT (COMPUTE_16F: the P100 FP16_GEMM default, every cc >= 7.5 card, Volta with PXA_VOLTA_F16_GEMM=0), f32 activations that took the guarded conversion (MUL_MAT_ID / fused MoE loop armed, or PXA_F16_GUARD_DENSE)", "default-on", "",
        "a non-finite f16 GEMM output (a column below the 65520 input threshold times |w| > 1: Nemotron-3-Nano PXQ4 anchors reach 1.75) is recomputed in the f16->f32 copy from the same f16 operands with an f32 accumulator and replaced when finite (bug #338, test-mmid-f16-guard PXQ4 cases on P100); every finite output keeps its bits (bit-identical to =0 wherever the stock route did not overflow); PXA_MMID_F16_GUARD=0 turns it off too"),
    "PXA_F16_GEMV_NY": ("on (=0 the old dispatch)", "small-R f16 weight (R 2..512, the hc_inject class) at a 2..8-column batch", "default-on", "",
        "each column is bit-identical to the one-token k_pxa_gemv_f16 launch, so a verify column is the plain decode's own result (qwen4exp hc_inject: was PXA_SMALLN_FLOAT or the f16-rounding cuBLAS chain); prefill ubatches of 2..8 tokens take it too"),
    "PXA_KQMMV_VERIFY_ID_SMALL_W": ("8 when PXA_KQMMV_VERIFY_ID_SMALL is on (verify widths 2..8 for a launch of <= 1024 rows); the registry pins 4 for qwen35 on sm_60; 0 = PXA_KQMMV_VERIFY_ID's cap", "sm_60 KQMMV GEMVs of <= 1024 rows (qwen4exp / qwen35 DeltaNet alpha/beta Q8_0, small attention k/v) at 5..8-column verify batches", "rule", "",
        "the chunked k_kq_mmv form (per column bitwise the one-column launch) up to this width for small launches when the switch is on (registry: nemotron_h_moe, qwen4exp); wide launches keep the cap of 4; no effect with the switch off"),
    "PXA_SMALLN_FLOAT_YIELD": ("on (=0 off)", "PXA_SMALLN_FLOAT armed (Overdrive recipe) at a 2..8-column batch", "default-on", "",
        "PXA_SMALLN_FLOAT stands down where a column-exact twin of the decode path exists: 1-row weights (mul_mat_1row, PXA_SPEC_1ROW) and pre-Volta f32 (PXA_F32_VERIFY_COLSPLIT); everything else keeps the kernel"),
    "PXA_SPEC_QWEN4EXP_NGRAM": ("rule: 1 (the n-gram -> MTP row, selector on) when every device is sm_60, no -ot moves experts to the host and no expert cache runs with the head (PXA_XCACHE_MTP=1, Overdrive); else 0 (the MTP head alone, PXA_SPEC_SELECT=0 when unset)", "qwen4exp auto speculation with an MTP head (no --spec-type)", "rule", "",
        "the chain's draft schedule varies between identical requests (selector wall-clock timings); its text stays the plain text only where the verify is proven column-exact (P100: bi-probe K=2..8 bitwise, 12/12 at the plain sha, prose faster than the head alone); elsewhere the head alone keeps the schedule a function of the request. =1 / =0 forces either"),
    "PXA_MTP_HEAD_FILE": ("unset", "nemotron_h_moe trunk file + its separately shipped NextN head (mtp-*.gguf)", "tool", "",
        "path of the NextN head GGUF attached at load (blk.<n_trunk>.*); unset = byte-identical load"),
    "PXA_SPEC_CKPT_TEST_NO_RESTORE": ("off (=1 on)", "speculative decoding with a recurrent checkpoint (test only)", "diagnostic", "",
        "=1: skip the recurrent-state restore after a rejected draft (negative control of the MTP identity gate: output must differ); never for a real run"),
    "PXA_ROUTER_GEMV_NY": ("off (=1 on)", "MoE router, 2..4-token verify", "lever-off", "",
        "=1: router GEMV instead of cuBLAS for 2..4-token batches; output-identical, measured ~null"),
    "PXA_DENSE_FUG_NY": ("rule: on with the PXA_GEMMA4_V100_SET set (Gemma 4 sparse, all-V100 box, config level >= 2), else off (=1 on)", "dense up/gate, 2..4 columns, GELU/RELU", "rule", "",
        "=1: fused dense up/gate MMVQ at 2..4 columns (MTP verify); output-identical"),
    "PXA_FA_VEC_NY": ("off (=1 on)", "sm_70 FA, 2..4 columns, f16 K/V", "lever-off", "",
        "=1: 2..4-column FA through the 1-column f32 vector kernel; NOT output-identical (excluded from the recommended set)"),
    "PXA_MMQ_ID_TILES": ("3 (sm_70, ENHANCE)", "sm_70 routed-expert (mul_mat_id) quantized matmul", "default-on", "",
                         "=0: the stream-k launch (one persistent block per SM); =1 tile grid when it fills the SMs (>=90%); =2 always; =3 compact tile grid (only non-empty expert tiles launched) for the 64-column tile, else =1"),
    "PXA_MMQ_ID_OCC": ("auto (sm_70, ENHANCE)", "sm_70 routed-expert (mul_mat_id) quantized matmul, tiles <= 64 columns", "default-on", "",
                       "unset: 2 resident blocks per SM for Q4_0/Q4_1/Q5_1/Q8_0 (measured wins), 1 for the rest (k-quants spill at 128 registers); =1 never, =2 every type"),
    "PXA_FA_SWA_KVMIN": ("on (ENHANCE; =0 off)", "sm_70 MMA FA, sliding-window layers", "default-on", "",
        "=0: do not skip the fully masked sliding-window head of the KV (identical outputs either way)"),
    "PXA_FA_D512_CHAIN_SPLIT": ("off (0)", "Gemma 4 head-512 attention chain", "lever-off", "",
        "=N: run the head-512 chain in N query-head groups when the score matrix exceeds PXA_FA_D512_CHAIN_SPLIT_MIB (ub2048 compute buffer); 2..8 identical, 16 not"),
    "PXA_FA_D512_CHAIN_SPLIT_MIB": ("512", "PXA_FA_D512_CHAIN_SPLIT > 1", "site", "",
        "split only past this whole-context score-matrix size (MiB)"),
    "PXA_FA_D512_Q8_CAST": ("off (=1 on)", "head-512 decode with q8_0 KV", "lever-off", "",
        "=1: cast K/V views to f16 and use the fused D512 FA kernel"),
    "PXA_WARMUP_NO_REUSE": ("on (=0 off)", "MoE warmup", "default-on", "",
        "never reuse the all-experts warmup graph for later 1-token ubatches (fixes the Gemma 4 cross-request carry); =0 puts the old reuse back"),
    "PXA_VTRANS_2D": ("on (=0 off)", "non-FA transposed V store", "default-on", "",
        "reshape a per-head V to 2D before the transposed (-fa off) store (fixes Gemma 4 -fa off garbage, bug #332); =0 puts the scrambled store back"),
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
    "PXA_PXQN_CPU_MMV": ("on (=0 off)", "PXQN weights read on the CPU (PXA_XCACHE cold experts, CPU-resident layers)", "default-on", "",
        "the PXQN library's CPU matmul; =0 restores the previous CPU path and keeps PXQN cold experts on zero-copy"),
    "PXA_PXQN_CPU_MMID_Q": ("on (=0 off)", "PXQN routed experts in host RAM (-ot exps=CPU / --cpu-moe: the CPU MUL_MAT_ID)", "default-on", "",
        "one work queue over (expert, row chunk, column chunk) items and one shared activation quantisation per op; bit-identical; =0 restores the per-expert static row split"),
    "PXA_PXQN_CPU_MMID_RC": ("64", "PXA_PXQN_CPU_MMID_Q on", "site", "",
        "rows per queue item (a multiple of 8); tuning only, every value gives the same bytes"),
    "PXA_XCACHE_BOOTSTRAP": ("on (=0 off)", "llama-server, MoE file that does not fit, no routing counts found", "default-on", "",
        "decode a built-in mixed text once at startup, count the routed experts, write <model>.expert-counts.csv (else $PXA_CACHE_DIR/$XDG_CACHE_HOME/pxa/$HOME/.cache/pxa) and restart so the expert cache is planned; =0: warn and stream whole layers"),
    "PXA_XCACHE_ADAPT": ("off (=1 adapt, =observe statistics only)", "PXA_XCACHE split layers; the PXQN library carries the policy", "lever-off", "",
        "=1: re-ranks the resident experts while the model decodes (routing counted on the device, swaps run as async copies between decodes); +4% prose / +8% code on fn32 1x P100, inside load noise, and a greedy answer then depends on earlier requests (CPU q8 vs GPU rounding), so it is opt-in; =observe keeps the static plan and only reports the hit rate"),
    "PXA_XCACHE_SPARE": ("4", "PXA_XCACHE_ADAPT on", "site", "",
        "free slots added to each split layer's cold stack (pinned host RAM) that a swap evicts a hot expert into; the hot stack keeps its planned size"),
    "PXA_XCACHE_ADAPT_EVERY": ("4", "PXA_XCACHE_ADAPT on", "site", "",
        "decodes between two observations of the routing counters (a prefill-sized decode is an observation of its own)"),
    "PXA_XCACHE_ADAPT_RESET": ("1 (=0 the adapted hot set carries across requests)", "PXA_XCACHE_ADAPT on; a batch at position 0 on an empty KV cache, main model", "default-on", "",
        "identity fix: a repeated request repeats its greedy answer (the hot GPU kernel and the CPU cold path round differently)"),
    "PXA_XCACHE_ADAPT_DET": ("1", "PXA_XCACHE_ADAPT on", "default-on", "",
        "=1 every swap is applied a fixed number of decodes after it was planned (waiting for its copy), so the swap schedule follows the token stream; =0 applies it when the copy lands"),
    "PXA_XCACHE_MTP": ("off (=1 arms)", "PXA_XCACHE active and a file with an MTP head", "lever-off", "",
        "=1: the expert cache runs together with an MTP speculative chain: the head's layer stays whole in VRAM and the planner keeps the chain's per-step recurrent checkpoint rows (and a small companion allowance) out of the experts' share, so the draft-length budget is not clamped to 1 on a full card; the auto spec row is armed instead of declining on the file-size headroom estimate"),
    "PXA_XCACHE_PLAN_V2": ("off (=1 arms)", "PXA_XCACHE active", "lever-off", "",
        "=1: the planner prices a cold layer at the measured 0.7 ms (not 1.3) and, with a speculative chain, a missed expert per round (verify width 1 + n_max): more cold layers, fewer misses"),
    "PXA_XCACHE_NEMO_ZC": ("auto (=1 zero-copy, =0 cpu)", "arch nemotron_h_moe, non-PXQN routed experts, PXA_XCACHE split, cold path auto", "rule", "",
        "unset: the cold experts are computed on the CPU when the free CPUs of the process's set cover 0.9 x -t (load sampled at boot, own time excluded), else read zero-copy; =1 zero-copy (the previous default), =0 cpu"),
    "PXA_XCACHE_NEMO_GUARD": ("off (=1 on)", "PXA_XCACHE_NEMO_ZC auto chose cpu", "lever-off", "",
        "opt-in since 2026-10-19 (a move changes the rounding of the tokens after it, so greedy output followed the box load); off: a boot keeps the path it chose at load; =1: once a second the decode thread sums this process's run-queue wait and moves narrow graphs to zero-copy after 2 intervals above PXA_XCACHE_NEMO_GUARD_WAIT CPUs, back to the CPU (a trial) after 5 intervals with the free CPUs at 0.75 of the need; a trial that falls back within 5 intervals doubles the next wait (max 64)"),
    "PXA_XCACHE_NEMO_GUARD_WAIT": ("1.0", "PXA_XCACHE_NEMO_GUARD on", "site", "",
        "CPUs' worth of this process's run-queue wait per interval above which the CPU cold path moves to zero-copy"),
    "PXA_XCACHE_NEMO_GUARD_UP": ("0.75", "PXA_XCACHE_NEMO_GUARD on", "site", "",
        "share of the CPU path's need the free CPUs must reach (5 s in a row, doubling after a failed trial, max 64 s) before a trial move back to the CPU path"),
    "PXA_XCACHE_NEMO_GUARD_MS": ("1000", "PXA_XCACHE_NEMO_GUARD on", "site", "",
        "guard interval in ms (>= 100)"),
    "PXA_STREAM_ZC_NY": ("per model: 0, 32 experts mode, 64 expert-cache zero-copy cold path / guard", "any", "site", "",
        "streamed-weight graphs no wider than this read the pinned bytes in place; exported = authoritative for every stream buffer of the process; unset = each model's own width, set on its stream buffers by the loader (the engine never writes the variable)"),
    "PXA_XCACHE_COLD_THREADS": ("-t (passed by the front end with the model load)", "PXA_XCACHE_NEMO_ZC auto", "site", "",
        "CPU threads the cold path runs on, as the auto cold path choice counts them; exported = wins over the -t the front end passes with the load (the engine never writes it); neither = every CPU of the process's set"),
    "PXA_XCACHE_NEMO_SAMPLE_MS": ("300", "PXA_XCACHE_NEMO_ZC auto", "site", "",
        "length of the boot-time CPU load sample (ms, >= 20)"),
    "PXA_XCACHE_NEMO_CONFIRM": ("5 (=0 one window)", "PXA_XCACHE_NEMO_ZC auto, first window says zero-copy", "site", "",
        "windows the boot-time CPU load sample reads before it settles on zero-copy: a first window that says zero-copy is confirmed by the median free CPUs of this many windows (bug #347: one window caught in a burst locked Bolt-74B 2x P100 into zero-copy, ~10 vs 22-24 t/s); a first window that says cpu decides at once; =0 or 1: the single-window rule"),
    "PXA_XCACHE_NEMO_CONFIRM_GAP_MS": ("700", "PXA_XCACHE_NEMO_CONFIRM > 1", "site", "",
        "ms between the confirm windows of the boot-time CPU load sample (>= 0)"),
    "PXA_XCACHE_NEMO_NICE_W": ("1", "PXA_XCACHE_NEMO_ZC auto", "site", "",
        "share of a CPU that niced (nice > 0) work of other processes takes from the CPU cold path (0..1)"),
    "PXA_XCACHE_NEMO_CPU_NEED": ("0.9", "PXA_XCACHE_NEMO_ZC auto", "site", "",
        "free CPUs the CPU cold path needs per thread"),
    "PXA_XCACHE_VERIFY_KAPPA": ("0.75", "PXA_XCACHE_PLAN_V2 on", "site", "",
        "share of a speculative round's extra verify width that reads distinct cold experts (1 = independent routing, 0 = the same experts every token)"),
    "PXA_XCACHE_VERIFY_WIDTH": ("server-set", "PXA_XCACHE_PLAN_V2 / PXA_XCACHE_MTP", "diagnostic", "",
        "tokens per target decode of the run (1 + the largest draft depth); the server exports it before the model is planned, an exported value wins"),
    "PXA_XCACHE_SPEC_COMPANION_MB": ("64", "PXA_XCACHE_MTP on", "site", "",
        "VRAM the planner keeps free for the MTP companion context on top of the per-step checkpoint rows"),
    "PXA_REP_GUARD_LAZY": ("off (=1 arms)", "a file where the repetition guard auto-arms (PXQ1 / PXQN1-heavy)", "lever-off", "",
        "=1: the guard's repeat penalty + DRY wait for a span repeated back to back in the generated text (a loop's first lap), so the greedy verify stays argmax-only and the MTP draft matches; hot for PXA_REP_GUARD_LAZY_HOLD tokens after a repeat"),
    "PXA_REP_GUARD_LAZY_K": ("6", "PXA_REP_GUARD_LAZY on", "site", "", "shortest span (repeated twice in a row, up to 256 tokens) that turns the guard hot"),
    "PXA_REP_GUARD_LAZY_HOLD": ("96", "PXA_REP_GUARD_LAZY on", "site", "", "generated tokens the guard stays hot after the last repeat"),
    "PXA_NUMA_HOST_BIND": ("on (=0 off; off at PXA_REFERENCE)", "Linux; every visible card on one NUMA node that has no allowed CPU, all allowed CPUs on one other node (1x V100 Overdrive board cell)", "default-on",
        "",
        "the streamed experts' pinned pages are bound to the node the CPU threads sit on (MPOL_BIND for that one allocation, guarded by the node's free + file-backed RAM), so a node short of free RAM at load reclaims page cache instead of spilling the arena to the cards' node; placement only, bit-identical"),
    "PXA_NUMA_BIND": ("on (=0 off; off at PXA_REFERENCE)", "Linux; every visible card on one NUMA node", "default-on", "",
        "=1: pin every thread of the process to the CPUs of the NUMA node the cards hang off (within the affinity it already had) and prefer that node for memory, before the first weight is allocated; the expert cache's CPU cold path and the per-layer-embedding gather then read local pages"),
    "PXA_CUDA_GRAPH_PXQN_ID": ("off (=1 on)", "CUDA graphs on with batch capture (PXA_CUDA_GRAPH_BATCH); a PXQN file", "lever-off", "",
        "=1: a PXQN routed-expert MUL_MAT_ID at a verify width (2..8 tokens) no longer forces its graph eager (the PXQN id-GEMV reads the ids on the device, no sync); a node the library declines falls to the capture fail-soft. PXA_CUDA_GRAPH_LOG names the node that kept a graph eager"),
    "PXA_PXQ_GATEUP_H2SPLIT": ("on (=0 off)", "cc 6.0, PXQ2/PXQ3 up/gate at width 1 (the half2 mask covers the tier)", "default-on", "",
        "on: a width-1 PXQ2/PXQ3 dense up/gate on a P100 runs as two half2 decode GEMVs plus the fused GLU instead of the fused fp32 up/gate kernel; =0 puts the fused kernel back"),
    "PXA_CACHE_PARK_SPEC": ("on (=0 off)", "prompt cache on and a drafter armed (MTP or draft model)", "default-on", "",
        "on: a conversation parked in the RAM prompt cache also carries the drafter's sequence state and carry, so the first turn after a switch drafts at once; =0 parks the target state only"),
    "PXA_KV_ADMIT": ("off (presence-tested)", "any", "diagnostic", "",
        "set: every llama_kv_cache_seq_rm logs seq, p0, p1, head and used before/after and whether the sliding-window KV is active (hot-swap / Gemma cross-request diagnosis)"),
    "PXA_SWAP_ALLOW_PASCAL": ("off (presence-tested)", "--hot-model on a card below Volta", "diagnostic", "",
        "set: lifts the engine's refusal of --hot-model on Pascal (compute capability < 7.0); for a diagnostic run only, the refusal is the product default"),
    "PXA_SWAP_ALLOW_SWA": ("off (presence-tested)", "--hot-model with a sliding-window model", "diagnostic", "",
        "set: lifts the engine's refusal of --hot-model when the served or a registered model uses sliding-window attention (Gemma); for a diagnostic run only, the refusal is the product default"),
    "PXA_SWAP_TRACE": ("off (=1 on)", "hot swap (--hot-model)", "diagnostic", "",
        "=1: park and unpark print one FNV-1a hash line per buffer of the host mirror, so a weight or state change across a swap can be named"),
    "PXA_THREAD_PLACE": ("off (=1 on)", "Linux; PXA_XCACHE_CPUPOOL_CPUS set (the Overdrive CPU pool)", "lever-off", "",
        "=1: the cold-path host thread takes the pool list's spare entry (CPUS[n-1]), the thread that calls llama_decode takes its own core of the cards' node outside the pool, and the HTTP listener (and its pool) and the CUDA driver's helper threads get a mask of the remaining CPUs; affinity only, never leaves the CPU set the process was given; one boot line names what was placed"),
    "PXA_THREAD_PLACE_NODE": ("the node of the first PXA_XCACHE_CPUPOOL_CPUS entry", "PXA_THREAD_PLACE on", "site", "",
        "NUMA node whose CPUs the thread plan chooses from"),
    "PXA_XCACHE_ASYNC": ("off (=1 on)", "PXA_XCACHE cache layers on the CPU cold path (pinned stacks the CPU reads); graphs of 1..PXA_XCACHE_ASYNC_MAXTOK tokens; CUDA", "lever-off", "",
        "=1: the cold experts of a split layer are computed by a host worker thread that the GPU stream hands work to through flags in pinned memory (no scheduler split, no cudaStreamSynchronize and no ids / activation / result copy call per cold layer); the CPU sub-graph is the one the split path runs, so the output is bit-identical; wider graphs keep the split path; CUDA-graph capture is off for graphs that carry the nodes"),
    "PXA_XCACHE_CPUPOOL": ("off (=1 on)", "PXA_XCACHE_ASYNC on", "lever-off", "",
        "off; needs the PXQN library; bit-identical"),
    "PXA_XCACHE_IDLE_BACKOFF_MS": ("0 (off)", "PXA_XCACHE_ASYNC on", "lever-off", "",
        "N>0: an idle cold-path thread waits longer between checks. The result is unchanged."),
    "PXA_XCACHE_SWAP_PAYBACK": ("off (=1 on)", "xcache on", "lever-off", "",
        "1: a swap starts only when the hit-rate gain pays for both copies; off = the ratio test alone"),
    "PXA_XCACHE_PATHLOG": ("off (unset or 0)", "PXA_XCACHE_ASYNC on", "lever-off", "",
        "diagnostic: 1 = stderr, any other value = a file path; one line per cold layer step naming the experts the CPU computed. The result is unchanged."),
    "PXA_MAMBA2_FUSE": ("on (=0 off)", "Nemotron-H Mamba-2 layers whose state rows live on the GPU", "default-on", "",
        "the in-place Mamba-2 conv + scan ops; 0 = the unfused graph; CPU layers keep the unfused graph"),
    "PXA_RELU_SQR_FUSE": ("on (=0 off)", "relu^2 FFN (Nemotron) on CUDA", "default-on", "",
        "RELU -> SQR in one launch; bit-identical"),
    "PXA_CUDA_GRAPH_REPLAN_KEEP": ("off (=1 on)", "CUDA graphs on (sm_60 needs PXA_CUDA_GRAPHS_PASCAL)", "lever-off", "",
        "=1 (default): a scheduler re-plan no longer bumps the global CUDA-graph allocation generation (every captured graph of every scheduler was recaptured, so verify-width / MTP graphs hit too-many-updates and ran eager); the per-node property compare still forces a recapture when an address moved; reserve and cudaFree sites still bump"),
    "PXA_XCACHE_CPUPOOL_THREADS": ("the async worker's thread count", "PXA_XCACHE_CPUPOOL on", "site", "",
        "how many threads the library uses"),
    "PXA_XCACHE_CPUPOOL_CPUS": ("unpinned", "PXA_XCACHE_CPUPOOL on", "site", "",
        "host CPUs the library may pin threads to"),
    "PXA_XCACHE_CPUPOOL_ROWS_A": ("see the library", "PXA_XCACHE_CPUPOOL on", "site", "",
        "chosen by the library"),
    "PXA_XCACHE_CPUPOOL_ROWS_C": ("see the library", "PXA_XCACHE_CPUPOOL on", "site", "",
        "chosen by the library"),
    "PXA_XCACHE_CPUPOOL_TASK_BARRIER": ("on (=0 off)", "PXA_XCACHE_CPUPOOL on", "default-on", "",
        "on: a phase of a cold step ends when its last task has finished, so a pool thread the OS has preempted holds nobody up (first requests on a loaded box no longer stall); same tasks, bit-identical; =0: every pool thread must arrive at every phase barrier (the old pool)"),
    "PXA_XCACHE_ASYNC_GRAPH": ("off (=1 on)", "PXA_XCACHE_ASYNC on; CUDA graphs otherwise allowed (sm_60 needs PXA_CUDA_GRAPHS_PASCAL; MoE needs PXA_CUDA_GRAPH_MOE)", "lever-off", "",
        "=1 (default): a graph carrying the cold submit / wait nodes stays capturable (the kernels keep their request number in device memory, the host worker only watches pinned flags), so the decode step replays as one CUDA graph"),
    "PXA_MTP_DRAFT_GUARD": ("off (=1 on)", "MTP drafting; temp-0 requests whose repetition guard / penalties / DRY are armed", "lever-off", "",
        "=1 (default): the draft token is the argmax of the draft head's top-W window after the request's own repeat penalty (prev tail + chain) and DRY (cloned, fed the chain); only the proposal changes, verify stays the exact match, output byte-identical"),
    "PXA_MTP_DRAFT_GUARD_W": ("16", "PXA_MTP_DRAFT_GUARD on", "site", "",
        "candidate window (2..128) the penalties are applied over"),
    "PXA_MTP_DRAFT_GUARD_STATS": ("off", "PXA_MTP_DRAFT_GUARD on", "diagnostic", "",
        "print draws / moved-off-argmax every 512 draws"),
    "PXA_XCACHE_ASYNC_MAXTOK": ("8", "PXA_XCACHE_ASYNC on", "site", "",
        "widest graph (tokens) that takes the async cold path: a decode and an MTP / n-gram verify batch; a prefill ubatch keeps the scheduler split"),
    "PXA_XCACHE_ASYNC_THREADS": ("the context's -t (-tb above one token)", "PXA_XCACHE_ASYNC on", "site", "",
        "CPU threads of the worker's sub-graph compute (OpenMP team led by the worker thread)"),
    "PXA_XCACHE_ASYNC_SPIN_US": ("3000", "PXA_XCACHE_ASYNC on", "site", "",
        "microseconds the worker keeps polling the request flags after its last request before it naps in 20 us steps (one core spins between layers of a token)"),
    "PXA_XCACHE_ASYNC_TIMEOUT_S": ("30", "PXA_XCACHE_ASYNC on", "site", "",
        "the slow-wait line: a cold-layer wait longer than this is counted and logged and the card keeps waiting (output unchanged; until 2026-10-26 it gave up with zero rows); 20x as long (at least ~5 min) = a hung worker, the process stops"),
    "PXA_XCACHE_ASYNC_CHECK": ("0", "PXA_XCACHE_ASYNC on", "diagnostic", "",
        "=1: every async cold layer ALSO runs the scheduler's CPU split, the two results are compared bit for bit on the GPU and the counts are printed at exit; the merge takes the split's result (output = the baseline's); slow, for finding where the two paths differ"),
    "PXA_XCACHE_ASYNC_MIXED": ("0", "PXA_XCACHE_ASYNC on; cache layers whose split-path chain is built as separate up / gate / unary nodes (an up stack and a gate stack of different types)", "diagnostic", "",
        "=1: those layers take the async path too, computed entirely on the CPU; the split path places their activation node on the GPU, so the output then differs from the split path in the last bits of that activation (measured: 4 of 31 cold layers on Flash-Next 32GB); default: they stay on the split path and the output is bit-identical"),
    "PXA_XCACHE_ASYNC_OVERLAP": ("1", "PXA_XCACHE_ASYNC on; models with a plain shared expert", "diagnostic", "",
        "=0: the shared expert is queued after the routed experts as before instead of between the hot experts and the wait on the host worker (A/B for the overlap)"),
    "PXA_XCACHE_ASYNC_TICKETDEP": ("1", "PXA_XCACHE_ASYNC on", "diagnostic", "",
        "=0: the submit node is ordered by graph expansion order instead of a dependency edge on the hot half's id node (A/B)"),
    "PXA_XCACHE_SKIP_COLD_RANK": ("off (=R arms)", "PXA_XCACHE split layers; main-model graphs of 1..PXA_XCACHE_SKIP_COLD_MAXTOK tokens (decode, MTP / n-gram verify); not the MTP draft; CUDA", "lever-off", "",
        "=R (1-based): a COLD (non-resident) expert that every token of the window ranks R or worse by routing weight is not computed and adds 0 (no renormalisation); resident experts always run; R > n_expert_used drops nothing. CHANGES OUTPUT (KLD-gated); unset = byte-identical"),
    "PXA_XCACHE_SKIP_COLD_MAXTOK": ("16", "PXA_XCACHE_SKIP_COLD_RANK on", "site", "",
        "widest graph (tokens) the skip is armed on; a prefill ubatch wider than this is exact"),
    "PXA_XCACHE_SKIP_COLD_LOG": ("0", "PXA_XCACHE_SKIP_COLD_RANK on", "diagnostic", "",
        "=N: every N decode / verify steps log the cold experts skipped in those steps (reads the device counters after the step); 0 = only the summary when the model is freed"),
    "PXA_XCACHE_ASYNC_BATCHONLY": ("1", "PXA_XCACHE_ASYNC on", "diagnostic", "",
        "=1: only graphs whose whole batch is at most PXA_XCACHE_ASYNC_MAXTOK tokens take the async path (the last layer of a prefill ubatch sees one row but stays on the split path); =0 per layer"),
    "PXA_PXQN_MOE_GEMM": ("on (1): sm_60 and sm_80+ (sm_80+ off at REFERENCE); sm_70 takes PXA_PXQN_MOE_TC", "PXQN routed experts at prefill widths", "default-on", "",
                          "expert-tiled half2 GEMM instead of the dp4a id-GEMV tiles (sm_86 RTX 3080 Ti Hy-Memory 30B-A3B PXQN2 pp512 485 -> 3011 t/s); not bit-identical on sm_80+; =0 the id-GEMV on every card (previous route byte for byte), =2 any fp16-capable arch (A/B)"),
    "PXA_PXQN_MOE_TC_SM80": ("on (1) on sm_80+ (needs PXA_PXQN_MOE_TC, so off at REFERENCE)", "PXQN routed experts at prefill widths on sm_80+", "default-on", "",
                          "expert-tiled GEMM inner product on mma.m16n8k16 with fp32 accumulation instead of the half2 GEMM (sm_86 RTX 3080 Ti Hy-Memory 30B-A3B PXQN2 pp512 3035 -> 4697 t/s); not bit-identical to the half2 GEMM (det 12/12, KLD 0.0020 vs ub1 band 0.0042); =0 the half2 GEMM byte for byte"),
    "PXA_PXQN_MOE_GELU": ("on (=0 off); decode route auto on sm_60/61 only (=1 forces it on sm_70+)", "PXQN routed GELU experts (Gemma 4 MoE)", "default-on", "",
                          "prefill: expert-tiled GEMM with a GELU epilogue on every card (+198%/+32% V100 pp); decode: id-GEMV GELU route +12.4% on one P100, no gain on V100 (generic dp4a path kept)"),
    "PXA_PXQN_XCACHE": ("on (=0 off)", "the PXQN library carries the expert-cache policy", "default-on", "",
        "=0 never resolves the library's online-adaptation policy: the expert cache keeps the static plan"),
    "PXA_XCACHE_BOOTSTRAPPED": ("unset", "internal", "diagnostic", "",
        "set by the server on its own restart after a counts bootstrap; stops a second bootstrap"),
    "PXA_CACHE_DIR": ("unset ($XDG_CACHE_HOME/pxa, else $HOME/.cache/pxa)", "per-user engine caches (routing counts of a read-only model directory)", "site", "",
        "where <model>.expert-counts.csv goes when the model directory is not writable"),
    "PXA_SPEC_SELECT": ("off (=1 arms; PXA_SPEC_DEFAULTS=1 turns it on for Flash-Next and Gemma 4 + assistant)", "libggml-pxqn present; chains of table and/or MTP stages; not np>1", "lever-off", "",
        "=1: a per-slot self-measuring selector (cost model in libggml-pxqn) picks plain / table draft / MTP depth each round by measured tokens per ms; verification is unchanged, so only speed moves; =0 or no library = the static rows"),
    "PXA_SPEC_SELECT_TUNE": ("unset", "PXA_SPEC_SELECT armed", "diagnostic", "",
        "key=value,... overrides of the selector's constants (lab use)"),
    "PXA_SPEC_SELECT_TRACE": ("off", "PXA_SPEC_SELECT armed", "diagnostic", "",
        "=1 prints each switch / probe / change-of-text decision to stderr"),
    "PXA_MTP_OUTBUF_LAZY": ("on (=0 off)", "MTP contexts", "default-on", "",
        "=0: the old MTP output-buffer sizing (a whole-batch host buffer at init; host RAM grows with -b); A/B only"),
    "PXA_MTP_POOL_SCOPE": ("on (=0 off)", "MTP companion context", "default-on", "",
        "=0: the companion's GEMM pool pre-grow is priced on every layer (the old sizing) instead of the MTP block only"),
    "PXA_MTP_POOL_SHARE": ("on (=0 off); off with PXA_MTP_PREFETCH or PXA_TSPLIT_GRAPH", "MTP companion context, CUDA VMM builds", "default-on", "",
        "=0: the MTP context keeps its own GEMM temporary pool per card instead of sharing the target context's"),
    "PXA_MTP_FALLBACK_RELEASE": ("on (=0 off)", "recurrent/hybrid model, speculative chain that fails to build", "default-on", "",
        "=0: keep the recurrent checkpoint buffers allocated when the MTP chain fails to build and the server decodes plain"),
    "PXA_MTP_UB_STEP": ("on (=0 off)", "server, one card, auto-armed MTP chain, engine-picked -ub (VRAM ladder / Volta rule)", "default-on", "",
        "=0: keep the engine's -ub when the MTP chain would miss the pool floor at it (the chain then falls back to plain decode)"),
    "PXA_SPEC_STREAM_MTP": ("on (=0 off)", "server auto speculation; arch nemotron_h_moe with an MTP head, experts streamed from host (PXA_STREAM_WEIGHTS, or a non-fitting file with PXA_XCACHE=0 or without PXA_XCACHE_MTP)", "default-on", "",
        "=0: let the auto layer arm speculation on streamed Nemotron-H MoE experts again (MTP measured -10% there, #22519)"),
    "PXA_MTP_UB_STEP_EXTRA_MB": ("64", "with PXA_MTP_UB_STEP", "rule", "",
        "MiB the PXA_MTP_UB_STEP price adds for what the companion and the first verify allocate beyond their buffers"),
    "PXA_MTP_UB_STEP_MIN": ("512", "with PXA_MTP_UB_STEP", "rule", "",
        "lowest -ub PXA_MTP_UB_STEP steps to"),
    "PXA_ROUTER_BF16": ("off (=1 on where PXA_ROUTER_FUSE is on, =2 any card); rule: 2 for Flint PXQN4 on one P100/V100 (PXA_AUTO_FLINT_ROUTER)", "BF16 router weights (Flint ffn_gate_inp), single decode token", "rule", "",
        "=1/=2: the BF16 router takes the F32 router GEMV walk on a shared-memory F32 widening of the row instead of the cuBLAS BF16 GEMM + two conversions (keeps x and the logits in F32; output not identical to the default); =0 keeps cuBLAS"),
    "PXA_MOE_OFFLOAD_K": ("per-model hint: 4 for arch flint on sm_70; the old 32 tokens-per-expert bar for every other model and card", "host MoE experts (-ot exps=CPU / --n-cpu-moe), default offload batch", "rule", "",
        "tokens-per-expert bar for offloading host MoE expert matmuls to the card: offload when batch*n_active >= K*n_experts (32-token floor kept); =N sets the bar on every model and card, =0 restores the old bar"),
    "PXA_ALLOC_FAIL_CLEAN": ("off (=1 on)", "any", "lever-off", "",
        "=1: a failed compute-graph allocation re-plans / fails with an error instead of placing tensors past a buffer end (SIGSEGV)"),
    # ---- tensor split ------------------------------------------------------------------------
    "PXA_TSPLIT_REDUCE": ("fused (engine default with -sm tensor)", "-sm tensor", "rule", "",
                          "the fused all-reduce every -sm tensor number was taken on"),
    "PXA_TSPLIT_REDUCE_PREFILL": ("0", "-sm tensor", "lever-off", "",
                                  "fused reduce on the prefill path; slower than PXA_TSPLIT_PF on 2 cards, no longer emitted"),
    "PXA_TSPLIT_FALLBACK": ("1 when -sm was auto-picked, else 0", "-sm tensor", "rule", "",
                            "a refused tensor split demotes to layer instead of stopping"),
    "PXA_TSPLIT_ALLOW_4WAY": ("on", "-sm tensor on >2 cards", "default-on", "",
                              "=0: a >2-device tensor split warns and runs layer (the pre-2026-09-27 guard; bug #206 is fixed)"),
    "PXA_TSPLIT_WORKERS_ADAPT": ("on (=0 off)", "-sm tensor with PXA_TSPLIT_WORKERS active (auto on 4x sm_60)", "default-on", "",
        "per decode graph: the enqueue workers or the plain loop, picked by the measured token period (starts on the plain loop, 3-graph probes, >= 3% to switch, backoff 16..512 graphs, a probe that loses to a stall of more than 1.5x the preferred period waits >= 100x that stall in graphs (cap 8192), a probe graph slower than 2x the preferred median ends the probe at once, re-probe on drift); both modes put the same launches on every stream in the same order, so output is bit-identical; =0: always the workers"),
    "PXA_TSPLIT_WORKERS_ADAPT_LOG": ("off", "PXA_TSPLIT_WORKERS_ADAPT on", "diagnostic", "", "=1: one stderr line per probe decision"),
    "PXA_TSPLIT_WORKER_PIN": ("2", "PXA_TSPLIT_WORKERS active, PXA_TSPLIT_WORKER_CPUS unset", "default-on", "",
        "2 (default): every worker may run on any allowed cpu of the driver's node (no collapse when another job sits on the top cores); 1: one core per worker, counting down from the top of the driver's node cpus the process may use; 2: every worker may run on any allowed cpu of that node; 0: no pinning"),
    "PXA_TSPLIT_EPI_RSAG": ("on at 4 cards (=0 off, =1 also 3 cards)", "-sm tensor on 4 cards (=1: 3-4 cards), the epilogue push path", "default-on", "",
        "on (default, 4 cards): the epilogue exchange runs reduce-scatter + all-gather (each card sums one quarter of the row in the one rank order and sends it back to all) instead of every card pushing the whole row to every peer; same sums in the same order, so output is bit-identical; 4x P100 G12: reduce gap p50 61.5 -> 53.1 us, -1.1 ms per token; =0: the one-round push; =1: also at 3 cards (unmeasured)"),
    "PXA_TSPLIT_WORKER_INPUT_EVAL": ("on (=0 off)", "PXA_TSPLIT_WORKERS active, splits whose inputs all come from the host", "default-on", "",
        "on: the driver copies the host inputs of each device's first segment, then that device's worker enqueues the segment, so the four first segments are enqueued side by side; per-stream order unchanged (scheduling only, greedy output identical); did not fire before PXA_TSPLIT_INPUT_EVAL_OAE (only_active_experts gate); =0: the driver enqueues every split with inputs"),
    "PXA_TSPLIT_ASSIGN_CACHE": ("on (=0 off)", "PXA_TSPLIT_WORKERS active", "default-on", "",
        "on: a reused graph (same split generation) keeps the previous split-to-worker assignment instead of walking every node again; =0: assigned every graph"),
    "PXA_TSPLIT_INPUT_EVAL_OAE": ("on (=0 off)", "PXA_TSPLIT_WORKER_INPUT_EVAL on", "default-on", "",
        "on: PXA_TSPLIT_WORKER_INPUT_EVAL also fires while the scheduler's only_active_experts flag is set (it is set for every model by default, dense ones included, which kept the lever from ever firing); a split with an input in a WEIGHTS buffer (the partial expert copy, the MoE prefetch) stays on the driver; scheduling only, greedy output identical (quick test owed); =0: no split with inputs goes to a worker while the flag is set"),
    "PXA_TSPLIT_HEAD_LOCAL": ("auto: on for a 4-card head on a one-row graph (=0 off, =1 two cards, =2 any count)", "-sm tensor, vocab-parallel LM head", "default-on", "",
        "auto (unset): a 4-card head whose reduced row is a single row (every decode step) reads each card's own all-reduced row instead of copying the home card's (replicas are the same bits since the rank-order reduce, bug #247), which removes the per-card synchronous peer copy before the head (quick test owed: greedy sha vs RSAG build); =1: two cards (measured-wash on 2x P100); =2: any card count, any rows; =0: off"),
    "PXA_TSPLIT_WORKER_YIELD": ("0 (off)", "PXA_TSPLIT_WORKERS active", "lever-off", "",
        "=N: after N pause spins a waiting worker or driver calls sched_yield every 64th spin (hands a shared core back); 0: pause only"),
    "PXA_TSPLIT_FORCE_FA": ("on", "-sm tensor", "default-on", "", "the split attention builder needs flash attention"),
    "PXA_TSPLIT_UNPROVEN_ARCH": ("off", "-sm tensor", "lever-off", "", "=1 runs an arch the split has no evidence for"),
    "PXA_TSPLIT_GEMMA4": ("off", "-sm tensor, gemma4", "lever-off", "", "Gemma 4 tensor split is opt-in"),
    "PXA_DN_CONVFUSE": ("on", "delta-net decode (n_tok 1, one seq), row absorb live", "default-on", "", "conv + next-step conv window + silu/q-k norm + beta-gate as one kernel; bit-exact; REFERENCE: off; =0 the separate kernels"),
    "PXA_DN_INPLACE": ("on", "delta-net decode with PXA_DN_CONVFUSE", "default-on", "", "carried state read from the cache row (device-resolved index) times the reset mask, no gather copy; bit-exact; REFERENCE: off; =0 the gather+mask copy"),
    "PXA_DN_BA_KQMMV": ("on", "delta-net ssm_beta/ssm_alpha under PXA_KQMMV", "default-on", "", "beta+alpha as one grouped f32-staged GEMV (no q8_1 quantize); changes output bits; =0 the incumbent quantize + two MMVQ"),
    "PXA_TSPLIT_LMHEAD": ("on", "-sm tensor", "default-on", "", "=0: full LM head on every card (vocab-parallel head changes output bits)"),
    # ---- ENHANCE-armed kernel levers (resolved in ggml-cuda/pxa/pxa-enhance.cuh) -------------
    "PXA_PXQ_INT8_PREFILL": ("1 on sm_61", "sm_61", "rule", "", "int8 prefill tile; +182% prefill on the 1080 Ti"),
    "PXA_ROUTER_FUSE": ("on sm_70; mode 3 on one P100 with Gemma 4 MoE", "sm_70 (not in a mixed rig); 1x sm_60 + gemma4 MoE", "rule", "", "router GEMV fusion; mode 1 a loss on sm_60, mode 3 +3.4% Gemma 4 decode on one P100; =0 off"),
    "PXA_MTP_SHORTLIST": ("prefix:81920 on the 248320-id Qwen head (=0 off); Flash-Next / Gemma 4 heads only by an explicit value or PXA_SPEC_DEFAULTS", "MTP draft head (qwen35, qwen4exp, gemma4 assistant)", "default-on",
                          "", "draft head scores token ids [0,N) only; the verify head is never windowed"),
    "PXA_SPEC_DEFAULTS_TUNE": ("v1_nmax=3,v1_pmin=0.5 (sm_70; other cc>=700 cards 0.85 unless set),v1_ub=512,v1_ramp=0,v1_cap=1,p1_on=1,p1_nmax=3,p1_pmin=0.8,p1_ub=512,p1_ramp=0,p1_cap=0,v2t_on=1,v2t_nmax=3,v2t_pmin=0.75",
                               "PXA_SPEC_DEFAULTS, qwen35 on one V100 (v1_) / one P100 (p1_) / a V100 pair under -sm tensor (v2t_; v2t_on=0 restores the n-gram -> head cascade there)", "lever-off", "",
                               "the one-card rows' depth ceiling, confidence floor, physical batch, context-depth ramp (1 = keep the ramp) and _cap=1 (unclamp with the capped logits reservation PXA_LOGITS_CAP at the engine's own -ub instead of the smaller batch) from the environment, so a measurement window needs no rebuild"),
    "PXA_FA_QKV_NY1": ("-1 = auto: on at sm_70 (=0 the width-w kernel, =1 every arch)", "FA at verify widths 2..8, q4_0/q8_0 KV (PXA_FA_QKV_DIRECT)", "default-on", "",
                       "each verify column runs the width-1 kernel with its own keys and split, so a T-token verify equals T one-token decodes; not bit-identical to 3.1 but correct (27B one V100 KLD 0.00968 vs 0.00950 off, the engine's own width spread)"),
    "PXA_PXQN_XB_I8": ("-1 = auto: 20480 B on sm_70 (=0 off, N>0 N bytes on every arch)", "PXQN int8 GEMV x-staging at NY >= 2", "default-on", "",
                       "x-staging budget that fits 4 blocks/SM at verify widths (17408x5120 GU ny4 0.218 -> 0.152 ms); bit-identical"),
    "PXA_MMVQ_NW4": ("on (=0 off)", "q8_1 MMVQ at 5..8 columns", "default-on", "",
                     "nwarps stays 4 at every width so a column's sum is the 1-column association (Q8_0 head / ssm_beta / attn_k at MTP verify); =0 the stock drop to 2 warps"),
    "PXA_PXQN_VWPF": ("unset = on at sm_70 (=0 off)", "PXQN4 MMVQ at 3/6/7 columns, sm_70", "default-on", "",
                      "verify-width MMVQ on the 1-column pipelined body; same lane map, k order and fold -> bit-identical"),
    "PXA_SPEC_FIT": ("on (=0: the flat file-size headroom rule)", "PXA_AUTO_SPEC MTP row, one CUDA card", "default-on", "",
                     "the auto speculation gate measures free VRAM minus weights, KV, recurrent state, compute and the GEMM pools against the draft context, two checkpoint rows and a margin, instead of a flat 2 GiB file-size headroom"),
    "PXA_SPEC_FIT_POOL": ("off (=1: the fit counts the two GEMM pools)", "PXA_SPEC_FIT", "lever-off", "",
                          "=1: the fit subtracts the main context's GEMM pool (-ub) from available and adds the MTP context's pool to need; stricter: it also declines 27B PXQN4 one V100 -c 24576, where MTP runs (64.4 vs 41.1 t/s plain); the default leaves them out and relies on PXA_SPEC_AUTO_FALLBACK"),
    "PXA_SPEC_AUTO_FALLBACK": ("on (=0: a chain that cannot be built fails the boot)", "a chain the PXA_AUTO_SPEC block armed (no --spec-type / -md / PXA_AUTO_SPEC=1), MTP only, slot 0", "default-on", "",
                               "an auto-armed MTP chain whose context cannot be created drops the stage and decodes plain instead of exiting"),
    "PXA_MTP_INIT_FAULT": ("off", "debug", "diagnostic", "",
                           "=1 test hook: the MTP draft context is refused as if it did not fit, proving the plain-decode fallback (tests/mtp-init-fallback-tiny-test.sh)"),
    "PXA_MTP_INIT_FATAL": ("off (=1: an MTP context that cannot be built fails the boot)", "an MTP-only chain (--spec-type mtp, PXA_AUTO_SPEC=1 or auto-armed), slot 0", "default-on", "",
                           "bug #344: an explicit MTP-only chain whose draft context cannot be built (35B MTP PXQN4 one V100 -ngl 30 -c 4096: 99.5 MiB under the 128 MiB floor) warns once with the refusal's numbers and decodes plain instead of exiting; =1 restores the error"),
    "PXA_SPEC_EXACT": ("on (=0: the old rules - draft gate on the drawn token's probability, n_min on the drawn length, expert-cache cuts on sampled drafts; biased under sampling, for A/B only)", "sampled (temp > 0) MTP drafts with p_min > 0, a depth cutoff, n_min >= 2 or an expert-cache draft cut", "default-on", "",
                       "the sampled draft's confidence gate reads the head's row (max q), not the drawn token, and its n_min floor reads the pre-draw length bound, not the drawn length (expert-cache cuts are skipped on sampled drafts), so speculative sampling keeps the target distribution; greedy never reaches it (sha unchanged); 27B PXQN4 one V100 sampled -0.5..-3.4% vs the biased gate (undetectable at n6 in 5 of 6 cells)"),
    "PXA_SPEC_SELECT_PLAIN": ("on where the per-model table armed the selector, else off (=1 / =0 decide)", "selector on a chain with a companion MTP head", "lever-off", "",
                              "a round without a draft is a legal choice for a chain with a companion head, so the selector can fall back to plain decode when the head loses"),
    # ---- the P60 verify stack on the 35B-A3B P100 pair (registry pick PXA_AUTO_P60) ----------------------
    "PXA_AUTO_P60": ("rule", "arch qwen35moe, tier PXQN4, two or more cards all sm_60 (measured 2 and 4), layer split", "rule", "",
                     "the P60 verify stack (the 9 levers below plus PXA_DN_CONVFUSE=1 and PXA_FA_GQA_PACK=0) is a default on that cell: 35B-A3B MTP PXQN4 plain decode 72.7 -> 125 t/s (+72%), bare launch 74 -> 138 / 151 / 130 with MTP n_max 3 p_min 0.85; 4x P100 plain decode 73 -> 125 t/s (+71%); NOT bit-identical to OFF at decode / verify widths but correct (KLD 2.5e-3 vs the engine's own width noise 1.8e-3, same top-1 98.4%; prefill identical); =0 stands the pick down, any one lever named in the environment is kept"),
    "PXA_AUTO_PF16K": ("rule (on)", "arch qwen35moe (two or more cards all sm_60) or qwen35 dense (one or two sm_60 cards), tier PXQN4", "rule", "",
                       "long prefill on the 35B-A3B P100 pair: PXA_PF_STABLE_SHAPE=1, PXA_FA_TILE_H2CHUNK=1, PXA_FA_LONGPREFILL=3 by default (pp16384 +43.5%, pp4096 +22.1%; 4x P100 pp16384 +14.1%, pp4096 +3.8%); correct, not bit-identical (KLD 0.00148 / 0.00166 vs noise 0.0021); the dense 27B pair gets the two attention halves (pp16384 353.3 -> 372.7, pp4096 399.2 -> 403.6); =0 stands the pick down, any one of the three named in the environment is kept"),
    "PXA_FA_TILE_H2CHUNK": ("rule: on for the PXA_AUTO_PF16K cell, else off (=1 on)", "sm_60 tile-f16 prompt attention (f32acc route)", "rule", "",
                       "V*P products summed in half2 chunks; KLD vs f32acc 0.00148; pp4096 +4.4%, pp16384 +11.7%"),
    "PXA_FA_LONGPREFILL": ("rule: 3 for the PXA_AUTO_PF16K cell, else 0", "cc 6.0 head-256 prompt attention, GQA 2..16; =3 only at KV >= PXA_FA_LONGPREFILL_MINKV (8192)", "rule", "",
                       "=3 the lean head-256 kernel (+5.2% at pp16384 over H2CHUNK); =1 the packed kernel; =2/=4 also on sm_70 (correctness check); KLD 0.00166"),
    "PXA_SPEC_AUTO_MOE_MTP": ("on (=0 restores the n-gram row)", "arch qwen35moe with an MTP head, two sm_60 cards, layer split", "default-on", "",
                              "a bare launch arms the MTP head alone at depth 3 behind a 0.85 confidence floor instead of the n-gram stage"),
    "PXA_PF_STABLE_SHAPE": ("rule: on for qwen35moe on sm_60 cards, else off (=0 off, =1 on)", "multi-token graphs of a MoE with a shared-expert gate", "rule", "",
                            "the shared-expert gate stays unfused in a prompt graph so every ubatch has one shape (no scheduler re-plan per batch): 2x P100 35B pp16384 1150.6 -> 1569.7, sha identical"),
    "PXA_PF_SLICE": ("on for sm_60 (=0 off, =1 on any card)", "-sm tensor prefill on 2 cards: a PXQN ks-GEMM (ks split 1) feeding a 2-card prefill reduce", "default-on", "",
                     "the prefill reduce push is overlapped with its producing GEMM in token slices (same blocks, same order): 2x P100 27B PXQN4 pp4096 351.4 -> 399.2 (+13.6%), greedy512 sha identical; =0 restores the unsliced GEMM + reduce"),
    "PXA_PF_SLICE_TOK": ("auto (3 ks blocks per SM, <= 16 slices)", "PXA_PF_SLICE", "site", "", "token-slice width override (a multiple of 128)"),
    "PXA_PF_SLICE_LOG": ("off", "PXA_PF_SLICE", "diagnostic", "", "=1: print the overlap plan count every prompt graph; =2: also name each declined overlap"),
    "PXA_QWEN4EXP_HC_FUSED": ("on at ENHANCE (=0 off)", "arch qwen4exp (Flash-Next), decode and verify rows <= PXA_QWEN4EXP_HC_FUSED_MAXNT", "default-on", "",
                              "the hyper-connection mixer and combine run as two fused kernels; bit-identical (greedy sha on -sm layer and tensor), 2x V100 MTP n3 layer +7%/+6%, tensor +13%/+18%"),
    "PXA_P60_MEGA": ("rule: on for the PXA_AUTO_P60 cell, else off (=1 on)", "cc 6.0 (sm_60) only", "rule", "",
                     "persistent single-token layer halves (GDN + MoE); decline cleanly where the graph does not match"),
    "PXA_P60_MEGA_PXQN_T": ("rule: on for the PXA_AUTO_P60 cell, else off (=1 on)", "cc 6.0, needs PXA_P60_MEGA=1", "rule", "",
                            "the 2..6-token PXQN4 MoE half (speculative verify); the code default is OFF, the catalog used to say on"),
    "PXA_KQMMV_VW": ("rule: on for the PXA_AUTO_P60 cell, else off (=1 on)", "cc 6.0 Q8_0 verify-width GEMV (2..5 columns)", "rule", "",
                     "verify-width Q8_0 GEMV"),
    "PXA_DRAFT_ARGMAX": ("rule: on for qwen35moe MTP drafts (=0 off, =1 also qwen35)", "qwen35moe / qwen35 MTP draft head (DRAFT_GEN / UPDATE_ACCEPTED)", "rule", "",
                         "device argmax draft draw; the gate is decided on the device only when p clears p_min by the margin, else the exact host draw"),
    "PXA_HEAD_DUAL_LAYER": ("rule: on for qwen35moe on two cc 6.0 cards under -sm layer (ENHANCE); else off (=1 on, =0 off)", "qwen35 / qwen35moe under -sm layer on two cards, row-sliceable head type", "rule", "",
                            "LM head cut over both cards (first card holds a copy of rows [0,N/2), +258 MiB on the 35B); bit-identical logits; =0 keeps the head whole"),
    "PXA_HEAD_DUAL_PF_WHOLE": ("on with PXA_HEAD_DUAL_LAYER (=0 cuts the head on prompt ubatches too)", "prompt-sized ubatches (> 64 tokens) under PXA_HEAD_DUAL_LAYER", "rule", "",
                               "the whole head on the last layer's card for prompt ubatches keeps the -sm layer prefill pipeline (35B 2x P100 pp4096/pp16384 back to the RC2.1 level); identical logits"),
    "PXA_P60_MEGA_T_SPEC46": ("off (=1 on)", "cc 6.0 PXQN4 MoE verify half at 4..6 tokens", "lever-off", "",
                              "compile-time T instances for 4..6 tokens, bit-identical to the runtime-T instance; undetectable at N=4 (q11d), dropped from the default stack"),
    "PXA_HEAD_DUAL_LAYER_DRAFT": ("on with PXA_HEAD_DUAL_LAYER (=0 keeps the MTP draft head whole)", "the prefix-shortlisted MTP draft head under PXA_HEAD_DUAL_LAYER", "rule", "",
                                  "draft head cut at n/2 over both cards, no gather"),
    "PXA_P60_MEGA_ATTN_T": ("rule: 2 inside the P60 mega stack at the ENHANCE level (=0 off, =1 whole 2..6-token INV attention half)", "cc 6.0 PXQN4 attention at verify widths 2..6", "rule", "",
                            "=2 attn_fa_t + attn_post_t on the verify path's q (correct, not bit-identical to the graph's flash attention: KLD at the width band, 7% above it - ub4 0.004185 vs ub1 0.003913, about 1 SE; needle owed); =1 per-token identical to the single-token kernels (reference); =0 the graph's tail"),
    "PXA_PXQN_VW": ("rule: on for the PXA_AUTO_P60 cell, else off (=1 on)", "cc 6.0 PXQN4 dense GEMV at 2..6 columns", "rule", "",
                    "verify-width PXQN4 GEMV"),
    "PXA_PXQN_VW_ROT": ("on (=0 off)", "PXA_PXQN_VW + PXA_PXQN_VW_1P: an out_in PXQN_RHT feeding a 2..6-column PXQN4 GEMV (cc 6.0)", "default-on", "",
                        "the RHT is applied inside the one-pass verify GEMV's staging, one launch less; bit-identical (PXA_PXQN_VW_CHECK word check on the P100 pair); =0 keeps the RHT node"),
    "PXA_MTP_INIT_FATAL": ("off (=1 error)", "llama-server with an MTP-only speculation chain", "lever-off", "",
                           "=1: an MTP draft context that cannot be built exits with an error again instead of decoding plain (bug #344)"),
    "PXA_MTP_INIT_FAULT": ("off (=1 test hook)", "test only", "diagnostic", "",
                           "=1: refuse the MTP context as if it did not fit, to test the plain-decoding fallback (bug #344)"),
    "PXA_WEDGE_EXIT_MS": ("180000 (=0 off)", "llama-server, every card", "default-on", "",
                          "stall watchdog: one llama_decode stuck longer than this (3 checks, 5 s apart) exits 41/42 for a restart"),
    "PXA_WEDGE_JIT_MS": ("1800000 (=0 never stretch)", "llama-server on a card with no native SASS in the build (JIT)", "default-on", "",
                         "the wedge limit during the first PXA_WEDGE_JIT_DECODES decodes on a JIT card, so a first-start kernel compile is not taken for a hang"),
    "PXA_WEDGE_JIT_DECODES": ("256 (=0 no window)", "llama-server on a JIT card", "default-on", "",
                              "how many completed decodes the stretched PXA_WEDGE_JIT_MS limit lasts; after that PXA_WEDGE_EXIT_MS applies"),
    "PXA_P60_MEGA_ADDFOLD": ("on (=0 off)", "PXA_P60_MEGA + PXA_P60_MEGA_PXQN_T: the residual ADD before a 2..6-token PXQN4 MoE half (cc 6.0)", "default-on", "",
                             "the ADD is folded into the MoE half's kernel (staged sum), one launch less per verify layer; bit-identical (PXA_P60_MEGA_T_SPEC_CHECK checks the half and the ADD output word for word); =0 keeps the ADD node"),
    "PXA_P60_PRE_VW": ("rule: on for the PXA_AUTO_P60 cell, else off (=1 on)", "cc 6.0, 2..5-token batches", "rule", "",
                       "verify-width norm + RHT + input projections in one launch"),
    "PXA_PXQN_RBP": ("rule: on for the PXA_AUTO_P60 cell and one P100 + dense 27B PXQN4 (registry), else off (=1 on)", "sm_60 PXQN4 one-column GEMVs", "rule", "",
                     "persistent row-block GEMV with the trimmed decode; bit-identical"),
    "PXA_PXQN_RMS_RHT_GB": ("rule: on for the PXA_AUTO_P60 cell, one V100: 1, else off (=1 on)", "PXQN decode-width fused (add+)norm+RHT(+q8) sites", "rule", "",
                            "8 blocks x 128 threads with one in-kernel exchange instead of one 1024-thread block; bit-identical"),
    "PXA_ARGMAX_VAL_MB": ("rule: on for the PXA_AUTO_P60 cell, else off (=1 on)", "the per-forward argmax_val (PXA_VERIFY_ARGMAX)", "rule", "",
                          "argmax_val over many blocks instead of one 1024-thread block per row; bit-identical"),
    "PXA_PXQN_QPN": ("rule: on for the PXA_AUTO_P60 cell, else off (=1 exact, =2 fast)", "sm_70 PXQN4 verify-width GEMV", "rule", "",
                     "=1: HMMA m8n8k4 QPN form, bit-identical to the dp4a decode GEMV; =2: fp16-x fast form (opt-in, KLD 0.000066); inert on sm_60"),
    "PXA_SPEC_DEFAULTS": ("on (=0 never arms the per-model / per-card table in libggml-pxqn; PXA_REFERENCE=1 off)", "speculation, any MTP / assistant model", "default-on",
                          "", "which chain / depth / floor / selector a model x these cards was measured best with; no row = the engine's static rows"),
    "PXA_SPEC_MTP_NGRAM": ("rule: on for a lone MTP stage, arch qwen35, one card or -sm tensor on >= 2 cards, all cc 700 (=0 MTP alone; =1 any arch/split/card)",
                           "speculation, MTP files", "rule", "",
                           "the n-gram stage (n_max 7, n_min 2, key 24) in front of a lone MTP stage, bare launch or explicit --spec-type mtp (PXA Control); dropped again when the checkpoint budget holds it no deeper than the head"),
    "PXA_SPEC_BARE_MTP_AUTO": ("on (=0 a bare --spec-type mtp keeps the generic MTP knobs)", "speculation, arch qwen35 MTP files", "default-on",
                               "", "a bare --spec-type mtp (PXA Control's MTP box) gets the measured auto MTP chain of a bare launch"),
    "PXA_SPEC_MTP_NGRAM_NMAX": ("7 (PXA_FA_D256_VOLTA_TILE_MAXCOLS - 1)", "speculation, the n-gram stage PXA_SPEC_MTP_NGRAM adds", "lever-off",
                                "", "lab: a wider table draft; past the cap the verify leaves the Volta tile FA kernel and is not bit-identical"),
    "PXA_SPEC_MTP_NGRAM_SHALLOW": ("off (=1 keeps the auto n-gram stage even when the checkpoint budget holds it no deeper than the MTP head)",
                                   "speculation, the n-gram stage PXA_SPEC_MTP_NGRAM adds", "default-on", "",
                                   "a table that cannot draft deeper than the head only costs its per-step lookup (one V100, 27B W3 -c 8192: prose -1.2%)"),
    "PXA_SPEC_STAGE_ORDER": ("on (=0 keeps the typed stage order)", "speculation, two-stage chains", "default-on", "",
                             "a table/suffix stage typed after mtp/draft moves in front of it (the reverse spelling was refused)"),
    "PXA_SPEC_SAMPLED": ("on (=0 off)", "speculation", "default-on", "",
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
    "PXA_TURING_CUBLAS_NE11": ("64 (=256 opt-in)", "sm_75", "lever-off", "",
                               "docs/SM75.md: dense standard-quant (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K) MMQ -> fp16 cuBLAS crossover; mainline-Q-only, so 64 (the old crossover) stays the default at every level; 256 opt-in; 0 keeps MMQ"),
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
    "PXA_ENV_BY_PXA": ("set by PXA", "launcher / engine, internal", "tool", "", "names of the PXA_* variables a PXA launcher (or this engine before it re-runs itself) put in the environment, so the startup lab-knob warning lists only switches the user set; never set it by hand"),
    "PXA_HOME": ("/opt/pxa/dist", "container", "tool", "", "package root in the image"),
    "PXA_MODEL": ("unset", "container", "tool", "", "model the entrypoint serves when run with no arguments"),
    "PXA_GPUS": ("all visible", "container", "tool", "", "cards the entrypoint hands the launcher"),
    "PXA_ALLOW_BUSY": ("off", "container", "tool", "", "=1: let the entrypoint start on a card that is in use"),
    # ---- CUDA graph cache memory guard (2026-10-20) -------------------------------------------------
    "PXA_CGRAPH_OOM_GUARD": ("on (=0 off)", "any CUDA card with graph capture (keyed cache: V2, mainline admission, split replay)", "default-on", "",
        "under PXA_CGRAPH_HEADROOM_MB free, never-replayed execs are released before an instantiate; an instantiate out of memory sheds the LRU half, then all, then runs the graph eagerly; output-identical; =0: the old abort"),
    "PXA_CGRAPH_OPPARAMS": ("on (=0 off)", "any CUDA card with graph capture (sm_80+ by default; sm_60/70 with the graph levers)", "default-on", "",
        "bug #343: a captured graph is replayed only when every kernel node also has the same op_params (pad offsets, clamp bounds, norm eps, RHT k0, ...), not just SCALE; no-kernel nodes (VIEW, RESHAPE, PERMUTE, TRANSPOSE, NONE) keep the old rule; =0: the SCALE-only compare (stale kernel arguments replayed when only an op_param changes); a correctness guard, not a lever: stays on at every level, REFERENCE included (named in the banner)"),
    "PXA_CGRAPH_HEADROOM_MB": ("128", "with PXA_CGRAPH_OOM_GUARD", "default-on", "",
        "free-memory floor of the guard's headroom step; 0: skip that step (the out-of-memory fallback stays)"),
    "PXA_CGRAPH_OOM_TEST": ("0", "debug", "diagnostic", "",
        "=N: every Nth guarded instantiate acts as if the card were full (exercises the eager fallback; output must equal the unforced run)"),
}

# string literals the scan finds that are NOT levers (a tier name, an output marker, a log tag)
NOT_LEVERS = {"PXQ_UNIVERSAL", "PXA_EXPLAIN_JSON", "PXA_TSPLIT_P2P", "PXA_SPEC_AUTO_ARMED"}

CURATED.update({
    # ---- 2026-10-26: qwen4exp verify-width HC GEMVs + fused small-kernel chains, bit-identical, default off ----
    "PXA_HC_GEMV_2PH": ("rule: 1 with the PXA_FLASHNEXT_V100_SET set (qwen4exp, all-V100 box, config level >= 2), else off (=1 on, =0 off)", "sm_70; int8 N1/N2/N5 decode GEMVs (ns=1) of <= N panels (hc_*_down) + PXQN4S8 q8_1 MMVQ K <= 512 at 2..8 columns (hc_*_up)", "rule", "",
                        "=1 (or =N panel ceiling): the two-phase exact ladder GEMV (hc_*_down) and the full-warp short-K MMVQ (hc_*_up), bit-identical to the shipped kernels (tests/test-pxa-hc-gemv.cu, test-pxa-hc-mmvq.cu)"),
    "PXA_HC_GEMV_2PH_CH": ("0", "PXA_HC_GEMV_2PH on", "site", "",
                           "chain form: 0 = stepped across the slot threads, 1 = integer dots through shared memory, one chain thread per (row, column); bit-identical"),
    "PXA_HC_GEMV_2PH_GU": ("rule: 0x1f8 (widths 3..8) with the PXA_FLASHNEXT_V100_SET set, else 0 (=1 every width, =0 off)", "PXA_HC_GEMV_2PH on; sm_70 int8 PXQN4 K-split fused up/gate (ns > 1) at 2..8 columns", "rule", "",
                           "=1: the fused up/gate (Flash-Next shared-expert gate/up) also takes the two-phase form at every width, bit-identical to k_pxqn_mmv (tests/test-pxa-hc-gu.cu); a value > 1 is a width mask (bit n = n columns, e.g. 0x1e0 = 5..8), set from a per-card bench so it never runs where it loses"),
    "PXA_HC_GEMV_2PH_GURB": ("rule: 32 with the PXA_FLASHNEXT_V100_SET set, else 16", "PXA_HC_GEMV_2PH_GU on", "site", "", "rows per block of the fused up/gate form (16, 32); bit-identical"),
    "PXA_HC_GEMV_2PH_RB": ("32", "PXA_HC_GEMV_2PH on", "site", "", "rows per block (8, 16, 32); bit-identical"),
    "PXA_VFUSE": ("rule: 1 with the PXA_FLASHNEXT_V100_SET set (qwen4exp, all-V100 box, config level >= 2), else off (=1 on, =0 off)", "CUDA; qwen4exp (Flash-Next) verify widths 2..8", "rule", "",
                  "=1: the verify-width small-kernel chains in one launch each, bit-identical (every node still written; the q8_1 handed to the next GEMV through a per-device record): 1 HC lo scale+silu+q8_1, 2 out gate sigmoid+mul+q8_1, 4 shexp sigmoid+mul+add, 8 router f32 GEMV weights in flight, 16 hc_inject f16 GEMV steps in flight, 32 hc gate_mix+q8_1, 64 shexp 1-row gate GEMV folded into the tail, 128 DeltaNet gated norm+sigmoid+mul+q8_1, 256 hc combine_norm values in flight (tests/test-pxa-vfuse-graph.cpp, test-pxa-snf.cu, test-pxa-f16ny.cu, test-pxa-hccn.cu)"),
    "PXA_VFUSE_OFF": ("0", "PXA_VFUSE on", "site", "", "mask of PXA_VFUSE parts left unfused (the bit values of PXA_VFUSE); bit-identical either way"),
})
CURATED.update({
    "PXA_GLM_THINK_OFF": ("on", "jinja chat, GLM [gMASK] templates whose generation prompt opens <think> and ignore enable_thinking (GLM-5.3)", "default-on", "",
                          "a request with thinking off (chat_template_kwargs enable_thinking=false, or --reasoning off) gets the closed empty block <think></think> after <|assistant|> and the answer as content; thinking on/default and every other template render as before; =0: the template's own render (the think block stays open, the answer lands in reasoning_content)"),
    "PXA_BENCH_SHARED_STDOUT": ("off (=1 on)", "llama-bench with a stdout printer (-o json|csv|md|sql)", "tool", "",
                                "=1: the old shared stdout (status lines from the backends and the library can land inside the results); default: the results alone on stdout, every other line on stderr"),
    "PXA_PLE_PREFETCH_SYNC": ("off (=1 on)", "PXA_PLE_MMAP=2 (the Overdrive presets)", "lever-off", "",
                              "=1: the per-layer table's prefetch runs inside the load as before (the server waits for the whole table to be read); default: it runs on a background thread after the load, at idle I/O priority"),
    "PXA_PLE_POPULATE": ("off (=1 on)", "a model whose host tensors the model-wide mapping serves and that has a >= 1 GiB per-layer gather table (GGML_CUDA_NO_PINNED + -ot, the 1x V100 / P100 Overdrive presets)", "lever-off", "",
                         "=1: the model mapping is populated at load as before (the whole file is read before the server answers); default: the copied tensors are read as they are copied and the table per row"),
    "PXA_PLE_PREFETCH_PACE": ("on (=0 off)", "PXA_PLE_MMAP=2 background prefetch of the per-layer table", "default-on", "",
                              "=0: the background prefetch queues the whole table at once (a request's row reads wait behind it); default: two 8 MiB chunks in flight, paused while a decode runs"),
    "PXA_XC_FILL_SPANS": ("on (=0 off)", "the PXA_XCACHE fill when the model mapping is not populated (PXA_PLE_LAZY)", "default-on", "",
                          "=0: one read per run of consecutive expert ids in slot order; default: each expert tensor read front to back in spans of up to 64 MiB, one span announced ahead"),
    "PXA_LAUNCH_EXTRA": ("unset", "container", "tool", "", "more pxa-launch arguments for the no-argument entrypoint path (e.g. --explain)"),
    "PXA_LAUNCH_FAKE_TEMPS": ("unset", "launcher", "diagnostic", "", "test hook: card temperatures for the heat warning, e.g. 5:83,6:40; never for a real launch"),
    "PXA_LAUNCH_FAKE_TOPO": ("unset", "launcher", "diagnostic", "", "test hook: a directory of topo.txt / lscpu.txt / allowed.txt / env.txt read instead of nvidia-smi topo -m, lscpu and /sys; never for a real launch"),
    "PXA_OVERDRIVE_LIB": ("off (=1 on)", "any common tool; the launcher / PXA Control and the Flash-Next preset set it for the Overdrive class", "lever-off", "",
                          "=1: the engine turns on the five Overdrive switches that only the PXQN library reads, at startup; a value already in the environment for one of them is kept; no effect without the library"),
    "PXA_NO_OVERDRIVE": ("off (=1 on)", "launcher / PXA Control, a Flash-Next (qwen4exp) PXQN file", "tool", "", "=1: the launcher does not export the measured Overdrive switch set for the Flash-Next class (the plan says so); everything else is planned as before"),
    "PXA_LAUNCH_FAKE_STORAGE": ("unset", "launcher", "diagnostic", "", "test hook: hdd | ssd | nvme | unknown as the model file's storage kind for the PLE rotational-disk guard; never reads the machine; never for a real launch"),
    "PXA_STORAGE_KIND": ("unset (the kernel's rotational flag)", "launcher, a model with a per-layer (PLE) table", "tool", "", "ssd | nvme | hdd: the user's word for the model file's storage when the kernel's flag is wrong (virtual disks often say rotational); feeds the DECISION ple-storage guard"),
    "PXA_LAUNCH_FAKE_GPUS": ("unset", "launcher", "diagnostic", "", "dry-run card table, e.g. 2x600 (parity harness); never for a real launch"),
    "PXA_LAUNCH_TABLES_ONLY": ("off", "launcher", "diagnostic", "", "=1: the launcher's own tables, the engine registry is not asked (parity harness)"),
    # ---- server prompt-batch admission / layer placement (pp-next, 2026-09-25) ------------------
    "PXA_PROMPT_RM_NOOP_SKIP": ("on", "server; chunk 2+ of a multi-chunk prompt admission (-b < prompt)", "default-on",
                                "",
                                "skips the llama_synchronize drain + no-op seq_rm when neither context holds a cell at or "
                                "beyond p0 (nothing to tear down); +21.8%/+12.1% quad 27B prefill at 20801/3121 (1.68 s/prefill "
                                "of drain removed on the V100 pair); decode and greedy sha unaffected (verified with NDEC held "
                                "equal, rel-integrate2 2026-09-25 #8823 - the earlier apparent regression was a harness NDEC "
                                "256-vs-512 mismatch between arms, not the lever); =0 restores the unconditional drain"),
    "PXA_LAYER_BALANCE": ("off", "-sm layer; every layer+head on the GPUs; same cc; no -ts; model fits", "lever-off",
                          "",
                          "re-places -sm layer's layers by prefill cost (weight elements + attention FLOPs) instead of bytes; "
                          "measured a quad loss (evened busy 0.57-0.59 but throughput fell) - the busiest card was not the "
                          "limiter, so evening cost throughput instead of buying it; stays off pending a length/serialization "
                          "root cause"),
    # ---- PXQ decode kernels (launch-defaults 2026-09-27, plan item S7a) -------------------------
    "PXA_PXQ_MMVQ_COLS": ("on at sm_70+ (ENHANCE), off below", "PXQ MMVQ launches at verify width 3-8", "default-on",
                          "",
                          "wide multi-column tile (8 rows at 3-4 columns, 4 at 5-8) instead of the 2-row clamp; bit-identical; "
                          "V100 pair 27B PXQ4 MTP -sm layer +10.2/+14.9% decode (clean bracket), wash under -sm tensor, "
                          "inert at width 1; =0 restores the clamp, =1 arms every card the kernel runs on"),
    # ---- k-quant decode GEMV (hp-kquant-mmv 2026-09-27) ------------------------------------------
    "PXA_KQMMV": ("auto (on at sm_60)", "Q4_K/Q5_K/Q6_K/Q8_0 MUL_MAT and FUSED_UP_GATE whose activation fits the block (ny <= 2 at K 5120)", "rule",
                  "",
                  "k-quant decode GEMV: 8 rows per block (one warp per row), f32 activation staged once per block (no "
                  "q8_1 launch), 16-byte weight loads, half2 dot on sm_60 / dp4a on sm_61+, src1-sharing matrices "
                  "grouped into one launch, gate/up fused; parity-tested vs the incumbent q8_1 MMVQ; P100 microbench "
                  "Q6_K 1.9-2.2x, Q8_0 1.5-1.7x, Q4_K/Q5_K 1.05-1.35x; REFERENCE: off; =0 restores the incumbent MMVQ, "
                  "=1 arms every card (the sm_61+ i8 path is slower than the dp4a MMVQ on V100)"),
    # ---- narrow PCIe links / peer-to-peer (launch-defaults 2026-09-27, bug #280) -----------------
    "PXA_P2P": ("on", "2+ cards", "default-on", "",
                "=0: CUDA peer access is never enabled (cross-card copies host-staged, the PXA fused/p2p reduces "
                "decline, NCCL_P2P_DISABLE=1): slower, for links whose peer path is not trustworthy (risers, IOMMU)"),
    "PXA_PCIE_LINK_WIDTH": ("sysfs", "any", "diagnostic", "",
                            "overrides the per-card PCIe link width the engine reads from sysfs (e.g. =1 to exercise the "
                            "narrow-link rules on a x4/x16 box); below x4: no pipeline parallelism, no auto -sm tensor"),
    "PXA_PIPELINE_PP": ("on for qwen35/qwen35moe multi-card -sm layer; off on a card below PCIe x4", "-sm layer, 2+ cards",
                        "rule", "",
                        "=1 forces pipeline parallelism (n_copies=2) on, =0 off; narrow links (x1/x2 risers) keep it off by default"),
    # ---- host memory for CPU-only gather tables (fn-next 2026-09-25, launch-defaults 2026-09-27) --
    "PXA_PLE_MMAP": ("1", "a CPU (host-buffer) per_layer_token_embd while the model-wide mmap is off (any -ot)", "default-on",
                     "",
                     "the PLE gather table is served from a read-only file mapping (pageable, no copy) instead of a pinned "
                     "CUDA_Host copy - 51,880 MiB on Flash-Next; bit-identical; =0 restores the pinned copy, =2 adds MADV_WILLNEED"),
    "PXA_PLE_MMAP_TOK": ("1", "under PXA_PLE_MMAP: a host-buffer token_embd that is not also the output matrix", "default-on",
                         "",
                         "the token embedding table (CPU GET_ROWS only) gets the same mapping - 497 MiB on Flash-Next; =0 keeps "
                         "it on the pinned copy"),
    "PXA_PLE_DIRECT": ("on in the 1x V100 Overdrive Flash-Next class (qwen4exp PXQN + MTP head, cold experts in host RAM; Linux) (=0 off); elsewhere off (=1 on, =2 forced)", "qwen4exp PLE table served from a file mapping (PXA_PLE_MMAP or the model-wide mmap); Linux", "default-on",
                       "",
                       "=1: the PLE gather reads each ubatch's distinct rows from the GGUF with O_DIRECT preads (one 4 KiB extent per row) "
                       "behind a RAM row cache instead of faulting the table's pages into the page cache; bit-identical; refused (RAM path "
                       "kept) when O_DIRECT fails, the file is on a rotational/unidentified device, or a 256-row boot probe (best of 3) is under "
                       "PXA_PLE_DIRECT_MIN_ROWS_S; =2 skips those refusals"),
    "PXA_PLE_DIRECT_CACHE_MIB": ("256", "PXA_PLE_DIRECT on", "site", "", "RAM row cache in front of the O_DIRECT reads (0 = none)"),
    "PXA_PLE_DIRECT_DEPTH": ("32", "PXA_PLE_DIRECT on", "site", "", "O_DIRECT reads in flight"),
    "PXA_PLE_DIRECT_IO": ("aio", "PXA_PLE_DIRECT on; Linux", "site", "", "=threads: a pool of preading threads instead of Linux native AIO from the decode thread"),
    "PXA_PLE_DIRECT_MIN_ROWS_S": ("5000", "PXA_PLE_DIRECT=1", "site", "", "boot probe floor: below it the lever refuses and keeps the RAM path"),
    "PXA_PLE_DIRECT_STATS": ("0", "PXA_PLE_DIRECT on", "diagnostic", "", "=N: log the reader's counters every N gathers"),
    # 2026-10-14 (V100 K0/K2/K3)
    "PXA_PXQN_RMS_RHT_GB": ("rule: one V100: 1 (llama.cpp pick); one P100 + PXQN file: 1 (registry); the PXA_AUTO_P60 cell: 1; else off", "PXQN decode-width fused (add+)norm+RHT(+q8) sites", "rule", "", "=1: the norm+RHT site on 8 blocks x 128 threads with one in-kernel exchange instead of one 1024-thread block; bit-identical; 1x V100 27B tg128 +1.1%, 1x P100 27B PXQN4 +0.6..1.4% (with ARGMAX_VAL_MB); =0 puts the one-block site back"),
    "PXA_DN_GDEAD_INDEX": ("on (=0 off)", "PXA_FUSE_DELTANET gather-mask / carry sites", "site", "", "=0: the old per-site forward scans instead of the per-eval consumer index (same answers; 149 us host per DeltaNet layer on the 27B)"),
    "PXA_DN_GDEAD_CHECK": ("off (=1 on)", "PXA_DN_GDEAD_INDEX on", "diagnostic", "", "=1: run index and scan both and abort on any disagreement"),
    "PXA_HOST_PHASE": ("off (=1 on)", "CUDA backend", "diagnostic", "", "=1: print per-eval host time (prelude, first node, whole eval, consumer-index builds) at exit"),
    "PXA_ARGMAX_VAL_MB": ("rule: one P100 + PXQN file: 1 (registry); the PXA_AUTO_P60 and PXA_AUTO_V70 cells: 1; else off (=1 on)", "the per-forward argmax_val (PXA_VERIFY_ARGMAX)", "rule", "", "=1: argmax_val over many blocks instead of one 1024-thread block per row; bit-identical; 27B head 164.8 us -> 3 passes; 1x P100 27B PXQN4 +0.6..1.4% with RMS_RHT_GB; =0 puts the one-block kernel back"),
    # 2026-10-16 (the one-V100 27B plain-decode cell)
    "PXA_AUTO_V70": ("rule", "arch qwen35, exactly one sm_70 card, no tensor split, config level >= 2", "rule", "",
                     "the one-V100 plain-decode stack PXA_DN_CONVFUSE + PXA_DN_INPLACE + PXA_ARGMAX_VAL_MB + PXA_PLAIN_GREEDY_AMAX + PXA_DN_CF_DIRECT + PXA_SCHED_ASYNC_INPUTS = 1 is a default on that cell (server and llama.cpp rule): 27B PXQN4 server decode +3.2% (levers), +1.2% more (CF_DIRECT), +0.6% more (ASYNC_INPUTS), greedy512 sha identical; =0 stands the pick down, any one lever =0 in the environment is kept"),
    "PXA_SCHED_ASYNC_INPUTS": ("rule: on for the PXA_AUTO_V70 cell, else off (=1 on)", "scheduler split inputs, one device", "rule", "",
                               "=1: split-input copies through a pinned staging ring with per-device events instead of a host wait; one-V100 27B plain decode +0.6%, sha identical; =0 keeps the host wait (tensor split: measured loss, not armed there)"),
    "PXA_DN_CF_DIRECT": ("rule: on for the PXA_AUTO_V70 cell, else off (=1 on)", "PXA_DN_CONVFUSE decode cluster, ny = 1", "rule", "",
                         "=1: a conv-cluster output that only the recurrence reads stays in the cluster scratch (no scratch->dst DtoD copies, ~4 per DeltaNet layer); bit-identical; =0 restores the copies"),
    "PXA_PLAIN_GREEDY_AMAX": ("rule: on for the PXA_AUTO_V70 cell, else off (=1 on)", "server, plain greedy decode step", "rule", "",
                              "=1: the plain greedy step takes the device argmax (one id read back) instead of the full logits row; bit-identical"),
    "PXA_PXQN_QPN": ("off (=1 exact, =2 fast)", "sm_70 PXQN4 verify-width GEMV, ny >= PXA_PXQN_QPN_MINNY", "lever-off", "", "=1: HMMA m8n8k4 QPN form, bit-identical to the dp4a decode GEMV; =2: fp16-x fast form (opt-in, KLD 0.000066)"),
    # 2026-10-17 (the one-V100 Gemma 4 decode cell G50; these rows supersede the earlier ones for the same names)
    "PXA_AUTO_V70_G4": ("rule", "arch gemma4 with experts, exactly one sm_70 card, no tensor split, config level >= 2", "rule", "",
                        "the one-V100 Gemma 4 decode stack PXA_CUDA_GRAPH_MOE + PXA_ARGMAX_VAL_MB + PXA_PLAIN_GREEDY_AMAX + PXA_SCHED_ASYNC_INPUTS + PXA_G2_NORMFUSE + PXA_NORM_ADD_SCALE = 1 is a default on that cell (server and llama.cpp rule): 26B-A4B PXQN4 server decode 113.0 -> 120.1 tok/s, greedy512 sha identical; =0 stands the pick down, any one lever =0 in the environment is kept"),
    "PXA_CUDA_GRAPH_MOE": ("rule: on for the PXA_AUTO_V70_G4 cell, else off (=1 on)", "MoE MUL_MAT_ID, one token, quantized experts", "rule", "",
                           "=1: a one-token MoE step stays inside a CUDA graph (the expert ids are read on the device); bit-identical; a loss on 1x P100 Gemma 4 (eager is faster there)"),
    "PXA_ARGMAX_VAL_MB": ("rule: one P100 + PXQN file: 1 (registry); the PXA_AUTO_P60, PXA_AUTO_V70 and PXA_AUTO_V70_G4 cells: 1; else off (=1 on)", "the per-forward argmax_val (PXA_VERIFY_ARGMAX)", "rule", "", "=1: argmax_val over many blocks instead of one 1024-thread block per row; bit-identical; =0 puts the one-block kernel back"),
    "PXA_SCHED_ASYNC_INPUTS": ("rule: on for the PXA_AUTO_V70, PXA_AUTO_V70_G4 and PXA_AUTO_A80_NEMOH cells, else off (=1 on)", "scheduler split inputs, one device", "rule", "",
                               "=1: split-input copies through a pinned staging ring with per-device events instead of a host wait; sha identical; =0 keeps the host wait (tensor split: measured loss, not armed there)"),
    # 2026-10-21 (rig RTX 3080 Ti sm_86: Nemotron-H 4B decode 160.0 -> 200.0 tok/s, mainline 187.2)
    "PXA_AUTO_A80_NEMOH": ("rule", "arch nemotron_h, exactly one sm_80+ card, no tensor split, config level >= 2", "rule", "",
                           "PXA_SCHED_ASYNC_INPUTS = 1 is a default on that cell (llama.cpp and server rule): N4 tg128 REPS 3 two runs 160.0 -> 200.0 tok/s on sm_86, greedy512 sha identical on N4, 8B and Hy; P100/V100 unmeasured so not armed; =0 stands the pick down, PXA_SCHED_ASYNC_INPUTS=0 in the environment is kept"),
    "PXA_MAMBA2_SHARE_INPUTS": ("follows PXA_DN_SHARE_INPUTS (auto: on for n_seq_max <= 1; =0 off, =1 on)", "Nemotron-H Mamba-2 layers", "default-on", "",
                                "one set of routing-input views per graph instead of three per Mamba-2 layer: the decode graph no longer overflows the 32 split inputs and runs as one CUDA split (one graph launch per token); bit-identical; rig sm_86 N4 tg128 199.8 -> 240.6 tok/s on top of the async rule, server greedy512 sha identical"),
    # 2026-10-19/20 (beta-2 supporter report: one V100, 27B MTP after a 20.6k prompt)
    "PXA_SCHED_ASYNC_MAX_TOKENS": ("64 (=0 every graph takes the async path)", "PXA_SCHED_ASYNC_INPUTS, graphs of more than N tokens", "default-on", "",
                                   "a prompt batch wider than N tokens keeps the blocking input copy (the staging ring cost the long prefill 7-10%); decode and verify graphs keep the async path; one V100 27B 20.5k prefill 925 -> 1025 tok/s, greedy sha identical"),
    # 2026-10-23 (bug #350: 2x V100 -sm tensor, '!!!!' from c6000 up with the V70 pair's async inputs)
    "PXA_SCHED_ASYNC_WAR": ("1 (=0 the old guard: probe only, races)", "PXA_SCHED_ASYNC_INPUTS at n_copies == 1, multi-device splits", "default-on", "",
                            "before a split's input copies the destination's slot event is re-recorded on its own stream, so peer copies on the source stream and blocking fallbacks wait for ALL destination work, not only the last split with inputs (write-after-read race); copy ordering only, greedy sha identical; P01 tensor c7500 async on: guard PASS, =0 FAIL"),
    "PXA_FA_QKV_NY1_SHARE": ("0 (=1 two verify columns per block)", "PXA_FA_QKV_NY1 verify launch, GQA <= 6", "lever-off", "",
                             "=1: two verify columns share one block's K/V tile on the width-1 plan; byte-identical per column (QKVD_EXACT); ~5% faster at widths 2/4, slower at width 3"),
    "PXA_FA_QKV_NY1_OCC": ("1 (=2 two blocks per SM)", "PXA_FA_QKV_NY1 verify launch", "lever-off", "",
                           "=2: the per-column instance capped at 128 registers so two blocks share an SM; byte-identical; measured slower on sm_70 (spills)"),
    "PXA_FA_QKV_UNIF": ("0 (=1 on)", "PXA_FA_QKV_DIRECT decode + PXA_FA_QKV_NY1 verify, q4_0/q8_0 KV", "lever-off", "",
                        "=1: decode and verify columns on one uniform tile plan, so verify columns share the K/V tile and still equal the one-token decode bitwise (MTP == plain); changes plain decode bits vs the width-1 plan"),
    "PXA_PLAIN_GREEDY_AMAX": ("rule: on for the PXA_AUTO_V70 and PXA_AUTO_V70_G4 cells, else off (=1 on)", "server, plain greedy decode step", "rule", "",
                              "=1: the plain greedy step takes the device argmax (one id read back) instead of the full logits row; bit-identical"),
    "PXA_G2_NORMFUSE": ("rule: on for the PXA_AUTO_V70_G4 cell, else off (=1 on)", "a FUSED_RMS_NORM (one row) feeding q8_1 GEMVs or MoE expert up/gate", "rule", "",
                        "=1: the norm also writes its output's q8_1 (same bytes as quantize_q8_1) and the consumers skip their quantize launch; bit-identical"),
    "PXA_NORM_ADD_SCALE": ("rule: on for the PXA_AUTO_V70_G4 cell, else off (=1 on)", "FUSED_RMS_NORM -> ADD(residual) [-> MUL(one-element scalar)], f32, contiguous, ne0 >= 256", "rule", "",
                           "=1: the Gemma 4 layer-end norm, residual add and layer scalar run as one kernel (explicit round-to-nearest add/mul); bit-identical"),
    "PXA_G2_NORMFUSE_PXQN": ("on (=0 off)", "PXA_G2_NORMFUSE on sm_70, PXQN4 / PXQN4S8 consumers", "default-on", "",
                             "with PXA_G2_NORMFUSE on, sm_70 PXQN4 decode GEMVs (q8_1 MMVQ route) and PXQN4 expert up/gate read the norm's q8_1 sidecar; =0 keeps PXQN consumers on their own quantize"),
    "PXA_RBUDGET_FIT_CTX": ("on (=0 off)", "llama-server chat requests whose thinking budget (--reasoning-budget / thinking_budget_tokens) is larger than the slot's room after the prompt", "default-on", "",
                            "the budget is fitted to n_ctx_slot - prompt - min(1024, room/4), so the thought closes before the context end or a context shift drops the question (#361/#362: -c 4096 with budget 128000 gave 28,791 thinking tokens and no answer); budgets that fit and -1 (unlimited) are unchanged (test-reasoning-budget-fit)"),
    "PXA_ARGSORT_SMEM": ("on (=0 off)", "the bitonic argsort kernel (ARGSORT / ARGSORT_THRESH, rows <= 1024 columns), any card, index + key arrays fit the block's shared memory", "default-on", "",
                         "the same bitonic network with each thread holding its (index, key) in registers: stages below the warp size swap through warp shuffles, wider ones through shared memory, instead of two dependent global loads + a barrier per stage (Flint router sort, 256 experts, 47 calls/token); same network and ties, bit-identical (test-argsort-smem: host network equality, ON/OFF HASH-ALL equal); =0 keeps the global-key kernel"),
    "PXA_FLINT_GATE_FUSE": ("off (=1 on)", "the norm_w sigmoid router tail with no selection bias (Flint), any card, 32..1024 experts, top-k <= 32, PXA_ARGSORT_SMEM on", "lever-off", "",
                         "=1: SIGMOID -> ARGSORT DESC -> VIEW top-k -> GET_ROWS -> SUM_ROWS -> DIV as one launch per MoE layer (4 -> 1; 47 layers/token on Flint), one block per token row, bit-identical to the op chain (test-flint-gate-fuse: op chain vs fused HASH-ALL equal + hand-placed alias layouts). Default OFF since 2026-10-24: measured wash (undetectable at N=2) and it raced on the default path when the allocator shared its buffers (#354, P100 one-card non-det); the opt-in keeps the #354 alias guard (skips the dead probs write, declines to the op chain on any other shared buffer)"),
    "PXA_ROUTER_STAGE": ("on (=0 off)", "the ROUTER_FUSE mode 1/2 router GEMV (sm_70 auto), ne00 % 4 == 0, 2*ne00 floats <= 48 KB", "default-on", "",
                         "the one-warp router GEMV with its row and x staged in shared memory by 128 threads; same per-lane walk and fold, bit-identical (unit check 6 shapes); V100 2816x128 7.98 -> 5.95 us/call; =0 keeps the one-warp walk"),
    #  (2026-10-18, owner DEFAULTS rule): three registry cells
    "PXA_AUTO_P1": ("rule", "arch qwen35, a PXQN tier, exactly one sm_60 card, no tensor split, config level >= 2", "rule", "",
                    "PXA_DN_CONVFUSE + PXA_DN_CF_DIRECT + PXA_PLAIN_GREEDY_AMAX = 1 on that cell (on top of the one-P100 RMS_RHT_GB / ARGMAX_VAL_MB / RBP picks): 27B PXQN4 1x P100 conv cluster ~+2.5% server decode with the host g_dead index, every response sha identical; all three bit-identical; =0 stands the pick down, any one lever =0 in the environment is kept"),
    "PXA_AUTO_OVERDRIVE": ("rule", "Flash-Next class (arch qwen4exp, routed experts, MTP head, tier PXQN2 / PXQN4 / mixed), file larger than the cards' total VRAM, no tensor split, config level >= 2", "rule", "",
                           "the launcher's Overdrive set (expert cache + MTP chain, async cold worker, CUDA graphs, shared-expert shadow, MTP no-tick, depth confidence floor, PXQN-MT, no-pinned, PLE mmap 2) in the engine itself, with recipe E (PXA_XCACHE_ADAPT=0, PXA_SPEC_SELECT=0, PXA_SPEC_DEPTH_NMAX=0): preset 1x P100 +14% prose / +18% code, 2x P100 layer +55% prose; recipe E 1x V100 a further +17% / +27% and 12/12 bitwise; =0 (or PXA_NO_OVERDRIVE=1) stands the set down, any one switch in the environment is kept"),
    "PXA_XCACHE_CPUPOOL_CPUS": ("rule: the PXA_AUTO_OVERDRIVE cell derives it when unset (allowed CPUs after PXA_NUMA_BIND, one per core first, 16 wide); else unpinned", "PXA_XCACHE_CPUPOOL on", "rule", "",
                                "cold-expert CPU pool workers pinned in this order (worker i -> i-th CPU); the launcher's rule, now also the engine's on the Overdrive cell; Flash-Next 1x V100 pinned 35.64 vs unpinned 31.53 t/s, sha equal; an explicit list wins"),
    "PXA_DN_CF_DIRECT": ("rule: on for the PXA_AUTO_V70 and PXA_AUTO_P1 cells, else off (=1 on)", "PXA_DN_CONVFUSE decode cluster, ny = 1", "rule", "",
                         "=1: the conv cluster reads and writes its scratch directly (no scratch->dst copies); bit-identical; =0 the copies"),
    "PXA_PLAIN_GREEDY_AMAX": ("rule: on for the PXA_AUTO_V70, PXA_AUTO_V70_G4 and PXA_AUTO_P1 cells, else off (=1 on)", "server, plain greedy decode step", "rule", "",
                              "=1: a plain greedy step reads back one id from the device argmax instead of the logits row (lowest id on a tie); bit-identical; =0 the logits copy"),
    # 2026-10-19 (the one-V100 Nemotron-Nano-8B cells G65; these rows supersede the earlier ones for the same names)
    "PXA_AUTO_V70_LLAMA": ("rule", "arch llama, exactly one sm_70 card, no tensor split, config level >= 2", "rule", "",
                           "the one-V100 llama decode stack PXA_PLAIN_GREEDY_AMAX + PXA_ARGMAX_VAL_MB + PXA_SAMPLED_TOPW = 1 and PXA_ROPE_KV_FUSE = 2 is a default on that cell (server and llama.cpp rule): Nemotron-Nano-8B PXQN4, greedy512 shas identical; =0 stands the pick down, any one lever's own value in the environment is kept"),
    "PXA_SAMPLED_TOPW": ("rule: on for the PXA_AUTO_V70_LLAMA cell, else off (=1 on)", "arch llama head, server plain sampled step whose chain is a top-k window <= PXA_SAMPLED_TOPW_W", "rule", "",
                         "=1: the head also computes its top-W window (values desc, lower id first on ties) and a plain sampled step reads back 2W floats instead of the logits row; the sampler's raw top-k path draws from it; bit-identical; =0 the logits copy"),
    "PXA_SAMPLED_TOPW_W": ("64", "PXA_SAMPLED_TOPW on", "site", "", "the window width (a power of two <= 128); requests with top_k above it take the logits copy"),
    "PXA_ROPE_KV_FUSE": ("rule: 2 on the PXA_AUTO_V70_LLAMA cell, else 0", "one-token norm-mode f32 rope pair (Q, K) of a decode graph", "rule", "",
                         "=1: the Q / K rope pair also fuses across a RESHAPE; =2: + the two f16 KV-cache stores in the same launch (3 launches -> 1 per layer); bit-identical; declines on input/output overlap and under the CUDA-graph copy indirection"),
    "PXA_PXQN_GU_2SEG": ("auto (=0 off, =1 wherever it applies)", "sm_70 one-column PXQN4 dp4a fused up/gate", "default-on", "",
                         "auto: ks 8 at 161..240 panels runs as 256-thread blocks with two k-segments per thread (one wave instead of two on an 80-SM V100); Nemotron-Nano-8B 14336x4096 0.0886-0.0921 -> 0.0827 ms; bit-identical"),
    "PXA_OP_EVPROF": ("off (=N dump every N decode graphs)", "eager CUDA graphs (GGML_CUDA_DISABLE_GRAPHS=1)", "diagnostic", "", "per-op device time from cudaEvents between launches; a launch-sequence dump at decode graph 8"),
    "PXA_ARGMAX_VAL_MB": ("rule: one P100 + PXQN file: 1 (registry); the PXA_AUTO_P60, PXA_AUTO_V70, PXA_AUTO_V70_G4 and PXA_AUTO_V70_LLAMA cells: 1; else off (=1 on)", "the per-forward argmax_val (PXA_VERIFY_ARGMAX)", "rule", "", "=1: argmax_val over many blocks instead of one 1024-thread block per row; bit-identical; =0 puts the one-block kernel back"),
    "PXA_PLAIN_GREEDY_AMAX": ("rule: on for the PXA_AUTO_V70, PXA_AUTO_V70_G4, PXA_AUTO_P1 and PXA_AUTO_V70_LLAMA cells, else off (=1 on)", "server, plain greedy decode step (arch llama: also attaches the head's argmax node)", "rule", "",
                              "=1: a plain greedy step reads back one id from the device argmax instead of the logits row (lowest id on a tie); bit-identical; =0 the logits copy"),
    # 2026-10-21 (board cell G10): the one-P100 decode stack on the P100 LAYER pair
    "PXA_AUTO_P2L": ("rule", "arch qwen35, tier PXQN4, exactly two sm_60 cards, -sm layer, config level >= 2", "rule", "",
                     "PXA_PXQN_RBP + PXA_PXQN_RMS_RHT_GB + PXA_ARGMAX_VAL_MB + PXA_DN_CONVFUSE + PXA_DN_CF_DIRECT + PXA_PLAIN_GREEDY_AMAX = 1 on that cell: a layer pair runs every layer whole on one card (the one-card launch shapes) but no one-P100 pick reached it (all n_dev == 1); all six bit-identical; =0 stands the pick down, any one lever =0 in the environment is kept"),
})

#  (2026-10-18): the members of the PXA_AUTO_OVERDRIVE registry cell are a default on that cell
# The five closed members are named only by their lever_hash (no closed name in the open tree); their names
# come from scripts/pxa-closed-levers.txt, which is closed itself, so a public tree adds no row for them (it has none to add).
_OD_CLOSED_HASH = ("80ca2bd7e94d0095", "bea2dc0a20680249", "b6ad4b781e90bc1f", "c211474db703d3d7", "05dddbccc1207f4e")
_OD_MT_HASH = "05dddbccc1207f4e"   # the verify-width GEMV: its site default reads 0
def _lh(name):   # = lever_hash below (defined later in this file; this table is built at import)
    h = 0xcbf29ce484222325
    for b in name.encode("utf-8"):
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h
def _od_closed_members():
    if not os.path.exists(os.path.join(ROOT, "scripts", "pxa-closed-levers.txt")):
        return []
    names = set()
    for line in open(os.path.join(ROOT, "scripts", "pxa-closed-levers.txt"), encoding="utf-8"):
        n = line.split("#", 1)[0].strip()
        if n and not n.endswith("*"):
            names.add(n)
    by_hash = {_lh(n): n for n in names}
    return [by_hash[h] for h in _OD_CLOSED_HASH if h in by_hash]
for _n in ("PXA_XCACHE_PLAN_V2", "PXA_XCACHE_ASYNC", "PXA_XCACHE_MTP", "PXA_XCACHE_CPUPOOL", "PXA_XCACHE_IDLE_BACKOFF_MS",
           "PXA_XCACHE_ADAPT", "PXA_REP_GUARD", "PXA_XCACHE_ASYNC_GRAPH", "PXA_CUDA_GRAPHS_PASCAL", "PXA_XCACHE_ASYNC_TIMEOUT_S",
           "PXA_CUDA_GRAPH_REPLAN_KEEP", "PXA_F32_GEMV_NY", "PXA_SMALLN_FLOAT") + tuple(_od_closed_members()) + ("PXA_SPEC_SELECT",):
    # (PXA_CUDA_GRAPH_BATCH is in the cell too, but it is default on everywhere already)
    _site = {"PXA_REP_GUARD": "level default", "PXA_F32_GEMV_NY": "0"}.get(_n, "0" if _lh(_n) == _OD_MT_HASH else "off (=1 on)")
    _d, _sc, _st, _ev, _ru = CURATED.get(_n, (_site, "any", "lever-off", "", "the engine reads it at its site; see docs/LEVERS.md"))
    _val = {"PXA_REP_GUARD": "0", "PXA_XCACHE_IDLE_BACKOFF_MS": "200", "PXA_XCACHE_ADAPT": "0 (recipe E)", "PXA_SPEC_SELECT": "0 (recipe E)", "PXA_XCACHE_ASYNC_TIMEOUT_S": "30"}.get(_n, "1")
    CURATED[_n] = ("rule: " + _val + " on the PXA_AUTO_OVERDRIVE cell; elsewhere " + _d, _sc, "rule",
                   (_ev + ", " if _ev else "") + "", _ru)

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
    # first pass (the closed sources and their lists are still in the exported tree): the public tree's table is
    # the one every release build embeds (release_filter: closed rows dropped or re-pointed, closed paths and backend
    # source locations worded, evidence empty), so the source and the binaries cannot disagree; a closed file the
    # in-tree list names but the export's omit list does not yet carry is handled here too
    if os.path.exists(os.path.join(ROOT, "scripts", "pxa-closed-files.txt")):
        rc = release_filter(OUT)
        if rc:
            return rc
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
    # the exported source carries no internal evidence ids either (catalog, registry, its test): the same ids no
    # build ships (scripts/pxa-public-scrub.py; it refuses a git checkout, so this step runs on an export only)
    import subprocess
    return subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "pxa-public-scrub.py"), ROOT])


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


ROW_FIELDS = re.compile(r'^\{ ((?:"(?:[^"\\]|\\.)*", ){5})"((?:[^"\\]|\\.)*)" \},?\s*$')
FIELD = re.compile(r'"((?:[^"\\]|\\.)*)"')
BACKEND_PATH = re.compile(r'\bggml/src/[\w./+-]+(?::\d+)?')


def public_row(l):
    """The shipped (embedded) form of one catalog row: the evidence column is an internal record id, so it is left
    empty, and a backend source location in the rule column ("read at <file>:<line>", "gate <file>: ...") is worded
    as "the backend": the user-facing part of a row is its name, default, scope, status and what it does."""
    m = ROW_FIELDS.match(l)
    if not m:
        return l
    f = FIELD.findall(m.group(1))
    rule = m.group(2)
    rule = re.sub(r'^read at ' + BACKEND_PATH.pattern + r'$', "read in the backend", rule)
    rule = re.sub(r'^gate ' + BACKEND_PATH.pattern + r':\s*', "backend gate: ", rule)
    rule = BACKEND_PATH.sub("the backend", rule)
    f[4] = ""
    nl = "\n" if l.endswith("\n") else ""
    tail = "," if l.rstrip().endswith(",") else ""
    return '{ ' + ", ".join('"%s"' % x for x in f) + ', "%s" }' % rule + tail + nl


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
        l2 = public_row(l)
        if l2 != l:
            scrubbed += 1
            l = l2
        keep.append(l)
    # a kept (open) row may NAME a dropped closed lever in its text (e.g. a threshold knob the closed library reads):
    # word it generically, or --assert-clean refuses the package
    if dropped:
        dre = re.compile(r'(?<![A-Z0-9_])(' + "|".join(re.escape(n) for n in sorted(set(dropped), key=len, reverse=True)) + r')(?![A-Z0-9_])')
        for i, l in enumerate(keep):
            if l.startswith('{ "'):
                l2 = dre.sub("a library knob", l)
                if l2 != l:
                    keep[i] = l2
                    scrubbed += 1
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


# 10-24: the bit-identical GLM-5.x decode levers flip to default ON; =0 = the old path.
CURATED.update({
    "PXA_DSV4_SINKHORN_HC4": ("on (=0 off)", "hc_mult 4 (GLM-5.x glm5next, DeepSeek-V4) on CUDA", "default-on", "",
                              "on: the hc=4 Sinkhorn runs register-resident (same arithmetic, bit-identical); =0: the generic kernel"),
    "PXA_GLM_KDA_FUSE": ("on (=0 off)", "arch glm5next on CUDA (pattern-matched KDA glue)", "default-on", "",
                         "on (unset or 1 = mask 3): KDA gate chain + out-gate fused (bit-exact); 2 or other values = the bitmask; =0: unfused"),
    "PXA_GLM5NEXT_GRAPH_REUSE": ("on (=0 off)", "arch glm5next decode", "default-on", "",
                                 "on: decode reuses the previous graph when the shape allows (logits bit-identical); =0: rebuild every step"),
    "PXA_GLM_CONCAT_I32_CUDA": ("on (=0 off)", "I32 CONCAT on the CUDA backend (the glm5next indexer tail), copy-only layouts", "default-on", "",
                                "on: the I32 concat stays on the GPU (bit-exact copy); =0: the CPU backend round trip"),
    "PXA_KPOOL_FUSED_SCORE": ("on (=0 off)", "arch glm5next MLA indexer", "default-on", "",
                              "on: the indexer's relu/mul/sum_rows chain is one kpool_score node (bit-identical, test-kpool-score); =0: the unfused chain"),
})

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
