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
    "PXA_MTP_SHORTLIST": ("prefix:81920 on the 248320-id Qwen head (=0 off)", "qwen35 MTP", "default-on",
                          "mtp-shortlist-idprefix", "draft head scores token ids [0,N) only"),
    "PXA_SPEC_SAMPLED": ("on (=0 off)", "speculation", "default-on", "spec-lossless-rejection-sampling",
                         "lossless sampled acceptance at temp>0 (replaces relaxed)"),
    "PXA_SPEC_RELAXED": ("off (on only with PXA_SPEC_SAMPLED=0)", "speculation", "rule", "", "relaxed acceptance floor"),
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


def scan():
    names = {}
    for d in SCAN_DIRS:
        p = os.path.join(ROOT, d)
        files = [p] if os.path.isfile(p) else [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs]
        for f in files:
            if not f.endswith(SCAN_EXT) or f.endswith("pxa-lever-catalog.inc"):
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


def main():
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
