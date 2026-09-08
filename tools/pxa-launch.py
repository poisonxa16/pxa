#!/usr/bin/env python3
"""pxa-launch - pick your cards and your model; the launcher does the rest.

RUN IT WITH NO ARGUMENTS.
  1. it lists the NVIDIA cards in this box, with VRAM and whether another process
     is already resident on each one, and you tick the ones you want;
  2. it lists the model files it can find - name, size, family (dense / MoE /
     hybrid-MoE), PXQ tier, all read out of each file's own header - and whether
     each one fits the cards you ticked, and you pick one by number;
  3. it asks what the seat is for (chat / serve / long documents), which is the
     one performance question a human has to answer on these cards;
  4. it shows the settings people forget - chat template, --jinja, reasoning
     format, sampling defaults, API key - each with where the value came from;
  5. it picks the engine, the batch and micro-batch sizes, the tensor split, the
     flash-attention regime and the environment from the measured table below,
     prints the decision with the evidence line per choice, and starts the server.

  Full-screen (stdlib curses) when the terminal is at least 80x24; the identical
  questions as line prompts otherwise, or with --no-tui / PXA_NO_TUI=1. Every
  answer can be given on the command line instead, in which case nothing is asked:
      pxa-launch.py --gpus 2,4 --model /models/x.gguf --yes
      pxa-launch.py --gpus 2,4 --model /models/x.gguf --explain   # decide, run nothing

WHY
  Two runtimes serve one quant family. pxa runs on everything we own.
  vllm-pxq4 runs wherever its IMAGE has kernels for the card (see "sm_70 ONLY WAS
  A LIE", below) and brings real data parallelism that llama.cpp's `-sm layer`
  does not have. Choosing by hand means remembering which card is which, whether
  the model is dense or MoE, which PXQ tier is actually inside the file, and
  which of the two engines wins at the concurrency you actually serve at.

DESIGN RULE - NEVER MAGIC
  Prints the decision, the evidence for it, and the exact command before running.
  REFUSES rather than silently dropping a parameter that does not translate.
  Says UNMEASURED out loud instead of guessing quietly.
  A launcher that quietly picks differently turns every perf question into a
  debugging session about the launcher.

    --explain            decide and print, run nothing
    --engine             force llama|vllm (blockers are still reported AND STOP)
    --workload           chat | serve | longdoc  (default: asked, else from --np)
    --selftest           exercise the decision table AND print every topology row
    --accept-unmeasured  execute a branch this file labels [INFERRED]/UNMEASURED
    --allow-busy         select a card another process is already resident on
    --models-dir         where to look for models (also PXA_MODELS_DIR)
    --serve-name         also write a rerunnable restart script for this seat
    --no-tui             skip the full-screen UI, use the line prompts
    --list-chat-templates  the template names THIS engine accepts, and where the
                         list came from

TWO TABLES, TWO QUESTIONS, DELIBERATELY SEPARATE
  WHICH ENGINE wins is a llama.cpp-vs-vLLM comparison and lives in MOE_TABLE /
  DENSE_NUMBERS below. WHICH FLAGS to pass is a per-topology measurement on ONE
  engine and lives in RECIPES. They have different corpora and different dates, so
  they are resolved separately, labelled separately, and printed separately. A
  topology can have a MEASURED recipe and an UNMEASURED engine crossover at the
  same time - the sm_61 row is exactly that - and saying so is the whole point.

HOW TO READ EVERY CLAIM IN THIS FILE
  Exactly three tags, no fourth category. They appear in the comments AND in the
  printed output, so a reader can tell a measured branch from a guess without
  leaving the file:

    MEASURED     a number from a gated boot on this box, with its source doc:line.
    [INFERRED]   a branch taken from an ADJACENT measurement. Never a new number.
    UNMEASURED   nothing was measured. The launcher says the word and stops or asks.

  Source corpus: the 2026-08-24 measurement run (an internal baseline set that is
    not published; every number it produced is quoted inline below, with the boot
    conditions it was measured under, so nothing here depends on reading it), plus
    docs/lab/LEVERS.md ; src/llama-quantize.cpp ; src/llama-model-loader.cpp ;
    ggml/include/ggml.h
  Where this file differs from the spec it was written against, the comment says
  so and why - see "SPEC CORRECTIONS", below.

THE DECISION IS NOT JUST ABOUT THE HARDWARE

    DENSE 27B PXQ4, 2x P100 sm_60 -- vLLM wins everything measured
      single decode   vLLM 24.01  vs llama.cpp 13.7   (1.75x)   SCOREBOARD D3/D1
      agg decode @8   vLLM ~70    vs llama.cpp 12.4   (5.6x)    SCOREBOARD D3/D1
      prefill         vLLM ~225   vs llama.cpp 156.5  (1.44x)   SCOREBOARD D3/D1
      agg decode @4   vLLM UNMEASURED vs llama.cpp 12.0         SCOREBOARD D3
      ^ the llama.cpp side (D1) is ONE boot, below this corpus's own 2-boot bar,
        and the graphs-ON dense arm (D2) was never launched at all.

    MoE 35B PXQ4, 2x P100 sm_60 -- SPLIT SEAT, this is the important branch
      np1  llama.cpp 95.6  vs vLLM 30.4   llama.cpp 3.14x   SCOREBOARD M1/M7
      np4  llama.cpp 75.93 vs vLLM 64.82  llama.cpp +17.1%  MOE-CROSSOVER
      np5  llama.cpp 79.49 vs vLLM 64.32  llama.cpp +23.6%  <- llama.cpp PEAKS
      np6  llama.cpp 69.58 vs vLLM 75.60  vLLM +8.7%        <- crossover
      np7  llama.cpp 67.74 vs vLLM 87.03  vLLM +28.5%
      np8  llama.cpp 62.42 vs vLLM 95.81  vLLM +53.5%

  Root cause of that shape, one sentence: llama.cpp `-sm layer` is a SERIALIZED
  2-GPU PIPELINE, not data parallelism, so concurrent requests queue behind the
  same pipeline while vLLM's aggregate climbs.

SPEC CORRECTIONS - things this file does NOT do the way the spec says, with why
  (each was verified against the tree/box before being written here)

  C1. THE TIER IS READ FROM THE TENSOR DIRECTORY, NOT FROM `pxa.pxq.tier`.
      LAUNCHER-SPEC M3 says to read the tier from "custom pxa.pxq.* KV keys".
      There is no such key. `grep -n "gguf_set_val" src/llama-quantize.cpp` shows
      the only tier-bearing provenance KVs written are pxa.pxq6.{version,tier}
      (:1760-1767), pxa.pxq2.version (:1775), pxa.pxq3.version (:1780) and
      pxa.pxqu.version (:1790) - and PXQ1 writes NONE of them (":1506 comment:
      Fixed {-1,+1} book + the shared SUB16 -> no provenance KVs needed"). So the
      spec's own fix would still leave PXQ1 - the one tier it exists to refuse -
      invisible. The engine itself does not use KV either: it detects PXQ1 by
      TENSOR TYPE (src/llama-model-loader.cpp:527). This file does the same.
      Ground truth = the per-tensor ggml type histogram; the provenance KV is
      read too and any CONFLICT between the two is printed, never resolved
      silently. Verified on the box: 5/5 PXQ GGUFs yield a tier this way, 0/5
      through general.file_type (all report 38).

  C2. I-2's "22.3 -> 24.0 dense" IS A MoE NUMBER. Adversarial review caught this
      and it is right. SCOREBOARD rows M8/M9 sit in Table 1a "MoE 35B PXQ4", not
      the dense table; ENGINE-VERDICT.md:14-22 calls the same pair "the MoE seat".
      Dense TP=2 single is D3 = 24.01, a different measurement, and there is NO
      dense FAP-vs-FDO pair in the corpus at all (D2 = 0 boots). The OLD version
      of this file carried the same mislabel in its FULL_DECODE_ONLY comment.
      Fixed in both places.

  C3. "EVERY vLLM NUMBER IS sm_60" IS OVERSTATED. Also caught by review, also
      right. PORT-ASSESSMENT.md:82 records a vLLM shared-prefix win of 3.76x
      measured on the DGX (V100, sm_70). What is true, and what the eligibility
      fix actually rests on, is narrower: NO vLLM DECODE-OR-PREFILL THROUGHPUT
      NUMBER on sm_70 exists anywhere in this corpus. The comment at
      vllm_eligibility() states the narrow version.

  C4. THE FIT CHECK STILL CANNOT BLOCK ON A KV ESTIMATE - so it does not pretend
      to. LAUNCHER-SPEC lists R-17 as "ctx > n_ctx_train, OR fit-check impossible"
      while M8 forbids the KV formula from blocking anything until Q9 validates
      it. Review flagged the contradiction. Resolution here: R-17 blocks ONLY on
      facts that need no formula - ctx > n_ctx_train (a KV field), and weights
      alone > total VRAM of the selection under full offload (arithmetic on file
      bytes). Everything that needs the KV-per-token estimate warns and says
      [INFERRED]. A real 42.93 GB PXQ4 file on this box (Fusion-Coder-80) is
      refused by the weights-only clause on 2 cards, which was the case review
      raised.

  C5. THE `-sm graph` GUARD IS STRUCTURAL, NOT A NAME LIST. I-9 keys on three
      arch strings. The hazard is a property of the DeltaNet tensors, so this
      file refuses on `linear_attn.*` presence OR the arch allowlist, whichever
      fires. That closes the muse-glimmer/dflash hole review raised - and note
      the box's muse-glimmer PXQ4 has NO linear_attn tensors (verified), so it is
      still allowed, on evidence rather than on a name.

sm_70 ONLY WAS A LIE, AND IT MADE THE MoE BRANCH UNREACHABLE
  The old file set MIN_VLLM_CAP=70 and routed any selection containing a
  sub-sm_70 card to llama.cpp. Every vLLM decode/prefill number in the decision
  table below was produced on 2x P100 sm_60 (SCOREBOARD.md:6; MOE-CROSSOVER.md:3,
  image pxa-sm60-dev, libpxq4_sm60_v10.so, --attention-backend PASCAL_SDPA,
  MOE-CROSSOVER.md:77-82). So the old table could never reproduce a single one of
  its own vLLM cells, and the np>=6 vLLM branch was dead on the only hardware
  where the crossover was measured. Eligibility is a property of the resolved
  IMAGE, probed. See vllm_eligibility().
"""
import argparse, collections, json, os, re, shutil, struct, subprocess, sys

# Host directory bind-mounted at /c inside the serving containers.
# Override with PXA_HOST_ROOT; defaults to the current working directory.
_HOST_ROOT = os.environ.get("PXA_HOST_ROOT", os.getcwd())

SPEC_MD5 = "86115482841eaa371e6050bc58734ca8"       # baselines/LAUNCHER-SPEC.md
BYTES_PER_GIB = 1024 ** 3

# ---------------------------------------------------------------------------
# HARDWARE FACTS
# ---------------------------------------------------------------------------
# cc -> human card class. sm_61 is NOT folded into "Pascal" here: the corpus has
# no MoE crossover, no dense pair and no PXQ-tier throughput on sm_61, and the
# BALANCE-mode PXA_FA_MASK_SKIP_TILE win explicitly excludes all of sm_61
# (LEVERS.md:85). Folding it into a Pascal set is how it stops being visible.
CARD_CLASS = {60: "P100-class sm_60", 61: "GTX 1080 Ti-class sm_61", 70: "V100-class sm_70"}

# MEASURED, hardware-verified (LEVERS.md:99-103, ADAPTIVE-UB fallback table):
#   >=15 GiB card -> 2048 ; 11 GB 1080Ti class -> 768 ; else 512.
# ub2048/1024 compute buffers OOM next to a ~10 GB model on 11 GB; ub768 fits.
# The engine picks this itself at startup when -ub is UNSET, probing real free
# VRAM per device. So the correct launcher behaviour is to pass NO -ub and print
# the value adaptive-ub should land on. Emitting a single global -ub across a
# heterogeneous pool is the bug this replaces (old file: -ub 2048 for every card,
# including card 3's 11 GB).
def _is_multimodal(model_dir: str) -> bool:
    """Does this checkpoint declare a vision tower?

    Read from the checkpoint's own config.json rather than inferred from the architecture
    name: the PXQ conversions of text-only GGUFs borrow config and vision weights from a
    reference checkpoint, so the architecture string says multimodal even when nothing in the
    pipeline will ever send an image. Unreadable or absent config means "assume not" -- the
    flag this gates is a restriction, and restricting on a guess is worse than not.
    """
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    if any(k in cfg for k in ("vision_config", "visual_config", "image_token_id")):
        return True
    arch = " ".join(cfg.get("architectures") or [])
    return "ConditionalGeneration" in arch or "VL" in arch


def ub_for_card(mem_total_mib):
    if mem_total_mib >= 15 * 1024:
        return 2048
    if mem_total_mib >= 10 * 1024:
        return 768
    return 512

# ---------------------------------------------------------------------------
# THE MEASURED DECISION TABLE
# ---------------------------------------------------------------------------
# MEASURED 2026-08-24, 11 gated boots, cards 0+6, PXA-Coder-35B-v2 PXQ4,
# -c np*4096, raw /completion (NOT chat-templated), every boot gated on
# short-prompt correctness BEFORE its number was kept.
# Source: the 2026-08-24 MoE-crossover baseline run (internal; the numbers it
#   np : (llama.cpp agg tok/s, vLLM PP=2+FDO agg tok/s, winner, margin text)
# Neither curve is monotonic and the flip is SHARP: llama.cpp PEAKS at np5 ABOVE
# its own np4 value, then drops 12.5% in one step while vLLM climbs. The margin
# swings 32 points between np5 and np6. A straight line np4->np8 puts the
# threshold too early and misprices np5 by ~14%. THE TABLE IS STORED, NOT A SLOPE.
MOE_TABLE = {
    4: (75.93, 64.82, "llama", "+17.1%"),   # llama.cpp reused from SCOREBOARD M4
    5: (79.49, 64.32, "llama", "+23.6%"),   # llama.cpp PEAK
    6: (69.58, 75.60, "vllm",  "+8.7%"),    # crossover
    7: (67.74, 87.03, "vllm",  "+28.5%"),
    8: (62.42, 95.81, "vllm",  "+53.5%"),
}
MOE_NP1 = (95.6, 30.4, "llama", "3.14x")    # MEASURED SCOREBOARD M1 / M7
MOE_LLAMA_MAX_NP = 5        # MEASURED: llama.cpp wins at and below np=5
MOE_VLLM_MIN_NP  = 6        # MEASURED: vLLM wins at and from np=6
MOE_TABLE_MAX_NP = 8        # nothing above np=8 was run, on either engine

# MEASURED currency warning that rides every vLLM MoE decision:
# MOE-CROSSOVER section 6.4 measured 95.81 at np8 on tree 3e34872 where SCOREBOARD
# M7 has 88.7 on fdec4ae (+8.0%). The doc names the intervening commit as a
# HYPOTHESIS, not a measurement, and concludes every vLLM MoE number on the
# SCOREBOARD may now be stale in vLLM's favour. If it is stale the crossover could
# sit BELOW np=6 - i.e. this can change an engine decision, not just a number.
MOE_CURRENCY_NOTE = ("vLLM MoE currency: 95.81 (tree 3e34872) vs 88.7 (fdec4ae) = +8.0% on the "
                     "same cell, cause hypothesised and NOT tested. If the newer number is real "
                     "the crossover may sit BELOW np=6. [MOE-CROSSOVER 6.4]")

# MEASURED dense, 27B PXQ4, 2x P100 sm_60, cards 1,5 (SCOREBOARD 2a rows D1/D3).
DENSE_NUMBERS = {
    "single":  ("24.01", "13.7",  "1.75x"),
    "agg8":    ("~70",   "12.4",  "5.6x"),
    "prefill": ("~225",  "156.5", "1.44x"),
}
DENSE_AGG4_NOTE = ("dense agg@4 on vLLM is UNMEASURED (SCOREBOARD D3); llama.cpp side is 12.0. "
                   "No ratio is printed for np=4 because none was measured.")
DENSE_WEAKNESS = ("dense envelope: the llama.cpp side (SCOREBOARD D1) is ONE boot, below this "
                  "corpus's own 2-boot bar, and the graphs-ON dense arm (D2) was never launched. "
                  "Direction 1.75x-5.6x is not in doubt; the exact ratios are single-boot.")

# MEASURED MoE long-doc prefill, with the caveat printed EVERY time:
MOE_LONGDOC_NOTE = ("long-doc prefill: llama.cpp 1136 / ~1058 / ~1000 vs vLLM 567.6 / 595.8 / "
                    "594.4 tok/s (~1.7-1.9x). CAVEAT: CROSS-HARNESS, prompt lengths NOT matched "
                    "(2059 vs ~6.4k tok) [SCOREBOARD 0.2]. Directionally trusted, not controlled.")

# The anchor models. The crossover is a property of a model x hardware PAIR, not
# of the engines - other MoE arches on the box (qwen3next MoE-512, qwen3moe
# MoE-128, deepseek4 MoE-6) have NO engine-vs-engine data at any np.
ANCHOR_MOE_HINTS   = ("coder-35b", "coder35", "pxa-coder-35b")
ANCHOR_DENSE_HINTS = ("27b-unc", "qwen38-27b", "qwen3.8-27b")
ANCHOR_CTX_PER_SLOT = 4096      # MEASURED envelope: --ctx-size np*4096 (MOE-CROSSOVER section 3)

# ---------------------------------------------------------------------------
# PXQ TIERS - ggml TENSOR TYPE IDS (ground truth; see SPEC CORRECTION C1)
# ---------------------------------------------------------------------------
# ggml/include/ggml.h:478-511. These are what the loader dispatches on and what
# llama-model-loader.cpp:527 uses to detect PXQ1 content.
PXQ_GGML_TYPE = {
    248: "PXQ1",     # ggml.h:499
    252: "PXQ4",     # ggml.h:478
    253: "PXQ4-HQ",  # ggml.h:481
    254: "PXQ2",     # ggml.h:489
    255: "PXQ3",     # ggml.h:490
    256: "PXQ6",     # ggml.h:511
}
# Everything else a PXQ file legitimately contains (backbone carriers).
NON_PXQ_GGML_TYPE = {0: "f32", 1: "f16", 8: "q8_0", 14: "q6_K", 30: "bf16", 39: "MXFP4"}
# ggml type ids the CURRENT tree does not define at all. Any tensor carrying one
# cannot be dispatched: `grep -n "24[4-9]" ggml/include/ggml.h` yields only PXQ1's
# 248, and PXQ1C/PXQ2C appear nowhere in the tree. VERIFIED on the box: a real
# file, <models>/qwen3-coder-next-pxqu/Fusion-Coder-80-PXQU.gguf,
# carries 106 tensors of type 247 and 38 of type 246 plus pxa.pxq1c.* / pxa.pxq2c.*
# KVs - retired clustered variants this engine no longer implements. The old
# launcher emitted a full, confident command for it. R-23 refuses it.
KNOWN_GGML_TYPE_MAX = 256

# LLAMA_FTYPE ids (include/llama.h). KEPT FOR ONE PURPOSE ONLY: catching the two
# RETIRED ids. general.file_type CANNOT identify a PXQ tier - llama-quantize.cpp
# :1454-1509 rewrites EVERY PXQ tier to LLAMA_FTYPE_MOSTLY_MXFP4 (=38) before
# writing it at :1658. VERIFIED on the box: 138/138 GGUFs report 38 or a K-quant
# id; 0 yield a tier. The old file's whole PXQ_FTYPE table was dead code against
# reality, which is why the PXQ1 refusal could never fire.
RETIRED_FTYPE = {250: "PXQ4_LEGACY (MXFP4-repack)", 251: "PXQ5 (learned book + SE8)"}
FTYPE_MXFP4 = 38

# vLLM implements exactly one PXQ tier. PXQ-TYPE-MATRIX.md:69-70: "On the vLLM
# fork, only PXQ4 is supported; every other tier is refused cleanly at the
# conversion gate. No silent wrong output exists on the vLLM path."
VLLM_SUPPORTED_PXQ = {"PXQ4"}
# PXQ-TYPE-MATRIX.md:80-81 - these run on llama.cpp only.
LLAMA_ONLY_PXQ = {"PXQ4-HQ", "PXQ6", "PXQ3", "PXQ2", "PXQ_UNIVERSAL"}
# No CPU codec, GPU-only, open task #62 (PXQ-TYPE-MATRIX.md:67; RELEASE-GATE.md:177).
NO_CPU_CODEC = {"PXQ1", "PXQ6"}

# '-sm graph' is hard-guarded off for the DeltaNet hybrids: the cross-device
# all-reduce never reaches its consumers, so each device computes a different
# router top-8 -> degenerate output. Where graph split DOES work it is a phase
# trade, not a win: +64% prefill / -17% decode on 4x P100. Never for decode.
# SPEC CORRECTION C5: the arch names are one of TWO triggers; the structural one
# (linear_attn.* tensors) is the other, so an unnamed arch with the same tensors
# is caught too.
GRAPH_SPLIT_GUARDED_ARCHES = {"qwen35moe", "qwen3next", "qwen35", "qwen4exp"}

# ---------------------------------------------------------------------------
# vLLM IMAGES - eligibility is an IMAGE property (see docstring)
# ---------------------------------------------------------------------------
# caps  = compute capabilities this image has PXQ4 kernels for
# status: MEASURED | INFERRED | INELIGIBLE
VLLM_IMAGES = {
    "pxa-sm60-dev": {
        "caps": {60}, "status": "MEASURED",
        "why": "produced every MoE-crossover vLLM number (MOE-CROSSOVER.md:77-82; "
               "libpxq4_sm60_v10.so, --attention-backend PASCAL_SDPA)",
        # NOT A SELF-CONTAINED IMAGE. `pip list` inside it shows pip and nothing else:
        # it is a bare CUDA runtime, and torch, vllm and the PXQ4 plugin all live on the
        # HOST and are bind-mounted at run time. Every number attributed to this tag was
        # really produced by the image PLUS these paths. Recording that is not pedantry -
        # without it the launcher declares the image eligible on a box where the venv has
        # moved, and emits `vllm serve`, which is not even on PATH in this container.
        "host_env": {
            "mounts": {_HOST_ROOT: "/c"},
            # Presence is checked with lexists, not exists. venv/bin/python is a symlink
            # to /usr/bin/python3, which exists INSIDE the container and nowhere on this
            # host - so exists() calls a perfectly good venv missing and refuses a seat
            # that would have served. lexists asks the question we can actually answer
            # from here: is the entry there.
            # Rooted at $PXA_HOST_ROOT (the host dir mounted at /c above), so the
            # six probes stay six DISTINCT paths under a relocatable base.
            "requires": [os.path.join(_HOST_ROOT, p) for p in (
                         "pxq4-sm60/venv/pyvenv.cfg",
                         "pxq4-sm60/venv/bin/python",
                         "pxq4-sm60/venv/lib/python3.12/site-packages/torch",
                         "pxq4-sm60/1cat/vllm/__init__.py",
                         "moe-site/site",
                         "moe-site/libpxq4_sm60_v10.so")],
            "python": "/c/pxq4-sm60/venv/bin/python",
            "env": {"PYTHONPATH": "/c/moe-branch/site",
                    # v10, NOT v8 (stale: both shipping launchers moved to v10) and
                    # NOT v11. v11 exists and is FASTER on prefill with PXQ4_GEMM2D=1
                    # (300.1 tok/s, +37%) but FAILS raw-prompt correctness. A prefill
                    # win that changes what the model says is not a win.
                    "PXQ4_LIB": "/c/moe-branch/libpxq4_sm60_v10.so"},
            "why": "traced 2026-08-26 through the last boot on this tag "
                   "(xover-vllm-boot1) and confirmed by importing inside it: torch is "
                   "/c/pxq4-sm60/venv/lib/python3.12/site-packages/torch (2.7.1+cu126) "
                   "and vllm is /c/pxq4-sm60/1cat/vllm/__init__.py",
            # THE PART THAT MATTERS. vllm is an EDITABLE install
            # (__editable__.1cat_vllm-....pth) resolving to the 1cat WORKING TREE. The
            # seat does not run a built artifact; it imports whatever is checked out
            # there right now. Edit that repo and the running engine changes. Switch its
            # branch and the seat serves a different engine with no redeploy and no
            # version change to notice. Every sm_60 number in the corpus was taken this
            # way, which is why they cannot be reproduced from an image alone.
            "editable_source": os.environ.get("PXQ4_SM60_SRC", ""),
        }},
    # THE FAT IMAGE IS WITHDRAWN. One image spanning sm_60+sm_70 was tried and does
    # not work: 8 boot attempts on a Tesla V100, 8 failures (RELEASE-GATE.md 3.7). The
    # cause is structural, not configuration - VLLM_SKIP_C_STABLE=1 is required to build
    # against torch 2.7.1 (the last torch with sm_60 cubins) and it drops
    # csrc/libtorch_stable/, where an op the V100 serving path calls unconditionally
    # lives. You cannot have sm_60 cubins and that op in the same build. Hence two thin
    # images, each pinned to the torch its cards need. Do not reintroduce a fat tag.

    # THE SHIPPING SET: two thin images (scripts/build-images.sh), each built from our
    # own tree, each pinned to the torch its cards need. No third-party image is
    # eligible.
    "pxa-vllm:sm60": {
        "caps": set(), "caps_inferred": {60}, "status": "INFERRED",
        "why": "Pascal variant: torch 2.7.1 (last torch shipping sm_60 cubins), "
               "VLLM_SKIP_C_STABLE=1, arch 6.0;7.0. The sm_60 gate that produced the "
               "MEASURED numbers ran against the WITHDRAWN FAT IMAGE, not this tag - "
               "identical build arguments is an argument, not a boot. caps stays EMPTY "
               "until scripts/thin-image-gate.sh sm60 passes on a P100 pair against "
               "THIS tag"},
    "pxa-vllm:sm70": {
        "caps": set(), "caps_inferred": {70}, "status": "INFERRED",
        "why": "Volta variant: torch 2.10 (the tree's own pins), libtorch_stable BUILT, no "
               "compat shim - so the three torch-2.7.1 failures cannot occur by construction. "
               "NOT YET GATED. caps is deliberately EMPTY until a V100 smoke passes: this "
               "launcher does not route traffic to an image on the strength of an argument"},

    "kewaii/vllm:latest": {
        "caps": set(), "status": "INELIGIBLE",
        "why": "THIRD-PARTY image. Everything we ship must be buildable from our own tree, and "
               "pxa-vllm:sm70 replaces this. Technically it also silently overrode "
               "cudagraph_mode from its own SM70 compile policy, and the knob it honours "
               "crash-loops 3/3 at warmup (DGX-FDO-FAILURE.md)"},
}
# MEASURED per-class attention backend.
ATTN_BACKEND = {
    60: ("PASCAL_SDPA", "MEASURED - the arm that produced every MoE-crossover vLLM cell "
                        "(MOE-CROSSOVER.md:81)"),
    70: ("FLASH_ATTN_V100", "[INFERRED] - from launch-v100b.sh / alina-launch.sh recipes; no "
                            "engine-vs-engine number was ever taken on sm_70, and as of "
                            "2026-08-25 no vLLM image in this table has a GATED sm_70 seat "
                            "at all (RELEASE-GATE.md 3.7)"),
}


# ---------------------------------------------------------------------------
# THE TOPOLOGY RECIPE TABLE  (refreshed 2026-09-03)
# ---------------------------------------------------------------------------
# WHAT THIS IS. One row per (card topology x model class) that somebody actually
# BOOTED and TIMED. A row carries the launch flags, the env, the numbers those
# flags produced, and the doc:line the numbers were copied from. Nothing else may
# enter this table - if a cell was not run, there is no row and the launcher says
# so out loud.
#
# WHY IT EXISTS. The previous version of this file emitted NO -b/-ub at all and
# left everything to the engine's adaptive-ub. That is the right answer only where
# nothing was measured. Four topologies now have a measured best cell and adaptive
# -ub does not land on any of them: it picks 2048 on a 16 GiB card, while the
# measured best on a 2x P100 pair is -ub 256 and the measured best chunk on both
# pairs is -b 8192, which adaptive-ub never sets because -b is not its job.
#
# WHY THE FLAGS ARE STILL EMITTED EXPLICITLY EVEN THOUGH THE ENGINE IS GROWING ITS
# OWN PER-TOPOLOGY DEFAULTS. A launcher that relies on an engine default cannot be
# read: the same command line means two different things on two builds, and a user
# who copies the printed command onto an older binary gets a different seat with no
# warning. Every flag in this table is passed on the command line, so the printed
# command is the whole truth on any build that accepts the flags.
#
# THE TWO FA REGIMES (docs/COOKBOOK.md:41-63). On P100 / V100 / 1080 Ti,
# flash-attention is a DECODE win and a COLD-PREFILL loss, for this engine and for
# upstream ik alike. One server = one setting, so it is chosen by --workload:
#   chat, serve  -> -fa on   (the default, and what every row below was measured at
#                             except the 1080 Ti cold cell)
#   longdoc      -> -fa off  (ingest / summarize / embed passes that barely decode)
FA_BY_WORKLOAD = {"chat": "on", "serve": "on", "longdoc": "off"}
FA_REGIME_SRC = "docs/COOKBOOK.md:41-63 (Two FA regimes); bench/fair-battle.md:35-50"

# The eleven levers docs/COOKBOOK.md:181-223 names for the 4x P100 Flash-Next seat,
# in the doc's own order. Kept as DOCUMENTATION of what the engine now sets by
# itself, NOT emitted (2026-09-06, AUTO-DEFAULTS-AUDIT gap a, order #1400c): since
# ENHANCE became the default level (2026-09-03) every one of these is the engine's
# own answer for a 4-card non-mixed sm_60 topology -- the nine house levers follow
# pxa_house_lever_default() (ON at ENHANCE, any topology) and PXA_FA_GQA_PACK=4 /
# PXA_MOE_DEVICE_MAP=1 follow their multi-sm_60 gates (pxa-enhance.cuh:571-590),
# confirmed live in the Phase P alex-ref boot with no env set. Emitting them pinned
# the recipe against the engine's future defaults with no signal of drift; the
# engine is the single source of truth for kernel levers. NONE of the eleven differ
# from what ENHANCE sets, so the emitted env is empty. The twelfth lever the live
# seat exports (PXA_FN_MTP_HNORM_GROUPED=1) is not in the published set and has no
# engine default; it is neither emitted nor documented here (see the note below).
ENGINE_DEFAULTED_FLASHNEXT_LEVERS = {
    "PXA_FA_GQA_PACK": "4",         # deep-fill decode, the biggest single win
    "PXA_KQ_MASK_PAD1": "1",
    "PXA_KV_SEQ_SOA": "1",
    "PXA_TOPK_RAW": "1",            # host-overhead cuts
    "PXA_TOPK_MOE_MULTIROW": "1",   # router aliasing guard - correctness, keep on
    "PXA_GETROWS_NARROW": "1",
    "PXA_CPY_FASTDIV": "1",
    "PXA_CONCAT_FLAT": "1",
    "PXA_NORM_REGCACHE": "1",
    "PXA_SCHED_RESET_LAZY": "1",
    "PXA_MOE_DEVICE_MAP": "1",      # device-side expert-routing table
}
# The live seat script /usr/local/bin/pxa-seats-flashnext.sh exports a TWELFTH,
# PXA_FN_MTP_HNORM_GROUPED=1, which docs/COOKBOOK.md:181-223 does NOT list among
# the shipped set. It is not emitted here: the published recipe is the measured
# one, and adding a lever the recipe does not name would make this launcher's seat
# a different seat from the documented one. Named rather than silently omitted.
FLASHNEXT_SEAT_EXTRA_NOTE = (
    "/usr/local/bin/pxa-seats-flashnext.sh also exports PXA_FN_MTP_HNORM_GROUPED=1, which "
    "docs/COOKBOOK.md:181-223 does not list in the shipped set. NOT emitted here - this "
    "launcher serves the published recipe. Add it by hand if you are reproducing the live seat.")

# PXA_PIPELINE_PP: on by default for the qwen35 / qwen35moe families ONLY.
# qwen4exp (Flash-Next) OOMs at -c 150016 on four cards with n_copies=2 compute
# buffers, and the 22:27 arms on GPU 3 failed to boot for the same reason on an
# 11 GiB card. This launcher never turns it ON; it only records where the engine
# default applies, so the printed plan matches what the seat will do.
PIPELINE_PP_DEFAULT_ARCHES = {"qwen35", "qwen35moe"}
PIPELINE_PP_NOTE = (
    "PXA_PIPELINE_PP is an ENGINE default for the qwen35/qwen35moe families only. It is NOT "
    "on for qwen4exp: at -c 150016 across four cards the n_copies=2 compute buffers OOM "
    "(the measurement ledger, the 1080 Ti no-boot has the same cause). "
    "This launcher does not set it either way.")


class Recipe(object):
    """One row of the table. `numbers` is quoted ONLY when the row matched
    exactly; an adjacent match prints the flags and suppresses the numbers,
    because a number belongs to the cell it was taken on."""

    def __init__(self, key, title, ncards, caps, family, tiers, b, ub, ctx,
                 env, numbers, source, status="MEASURED", arch=None, notes=(),
                 sm="layer", ts=None, ot=None, np=None, threads=None, extra=()):
        self.key, self.title = key, title
        self.ncards, self.caps = ncards, frozenset(caps)
        self.family = frozenset([family] if isinstance(family, str) else family)
        self.tiers, self.arch = frozenset(tiers), arch
        self.b, self.ub, self.ctx = b, ub, ctx
        # sm/ts/ot/np/threads are named rather than dumped into `extra` so that a
        # user-supplied --ts / --np / --threads can OVERRIDE one without the recipe
        # emitting a second copy of the same flag. A duplicated flag on a
        # llama-server command line is a silent last-one-wins, which is exactly the
        # class of bug this file exists to make impossible.
        self.sm, self.ts, self.ot = sm, ts, ot
        self.np, self.threads = np, threads
        self.extra, self.env = list(extra), dict(env)
        self.numbers, self.source, self.status = numbers, source, status
        self.notes = list(notes)

    def hw_matches(self, sel):
        """SUBSET, not equality: a row may declare several card classes it applies
        to (the stock-GGUF row applies to any single card), and a selection matches
        when its classes are among them. Every 2+ card row in the table declares
        exactly ONE class, so a mixed-class pair still matches nothing - which is
        the intent: no mixed pair was ever measured."""
        return len(sel) == self.ncards and {g[2] for g in sel} <= set(self.caps)

    def model_matches(self, prof):
        if self.arch and (prof.get("arch") or "") != self.arch:
            return False
        if "any" not in self.family and model_family(prof) not in self.family:
            return False
        if self.tiers and (prof.get("tier") or "none") not in self.tiers:
            return False
        return True

    def family_str(self):
        return "any" if "any" in self.family else "|".join(sorted(self.family))

    def hardware_only(self):
        """The row reduced to what a DIFFERENT model on the SAME cards may borrow:
        the prefill chunk and the micro-batch, and nothing else.

        Everything dropped here is model-specific and would be a lie on another
        file: -ts 5079,12612,12612,11897 is Flash-Next's weight distribution over
        four particular cards, -ot names a tensor another model does not have,
        -c 150016 is that model's trained window, and the eleven PXA_* levers were
        measured on that architecture's shapes. Borrowing -b/-ub is defensible
        because they are properties of the SPLIT and the compute buffer, which is
        the thing the card topology actually fixes. Borrowing the rest is not."""
        c = Recipe.__new__(Recipe)
        c.__dict__.update(self.__dict__)
        c.ctx = c.ts = c.ot = c.np = c.threads = None
        c.extra, c.env, c.notes = [], {}, []
        c.status = "INFERRED"
        c.numbers = "no number: this row's numbers belong to its own model"
        return c

    def hw_str(self):
        return f"{self.ncards}x {'/'.join('sm_%d' % c for c in sorted(self.caps))}"


def model_family(prof):
    """dense | moe | hybrid-moe. 'hybrid-moe' means an MoE that also carries
    linear-attention (DeltaNet/SSM) layers - Qwen3.8 Flash-Next is the case that
    matters, because its recipe, its -ot and its pipeline default all differ."""
    if not prof:
        return "dense"
    if prof.get("is_moe"):
        return "hybrid-moe" if prof.get("deltanet") else "moe"
    return "hybrid" if prof.get("deltanet") else "dense"


# ---- the rows, most specific first ----------------------------------------
RECIPES = [
    # -- 2026-09-03 campaign, the four cells the release table is built on ----
    Recipe(
        "2xv100-dense-pxq4", "2x Tesla V100 (sm_70), dense 27B, PXQ4",
        2, {70}, ("dense", "hybrid"), {"PXQ4"},
        b=8192, ub=2048, ctx=32768, env={},
        numbers="prefill 1,369 t/s @3,121 tok  |  1,300 @20,801  |  decode 39.5 @fill 8 "
                "(release binary, quiet box, no env, auto -b/-ub picks these flags)",
        source="RELEASE-NOTES-2026-09-07.md:77 (headline table row); "
               "the measurement ledger FINAL V100 cells",
        notes=["-ub 2048 beats -ub 512 here ONLY at -b 8192: at the default -b 2048 the "
               "per-chunk llama_synchronize is what made small -ub look good, and removing it "
               "flips the order back (bench/fair-battle.md:169-170; "
               "the measurement ledger FINAL TABLE).",
               "-b 20480 buys +0.3% over -b 8192 (1,211/1,210 vs 1,207/1,204) - i.e. nothing "
               "(the measurement ledger chunk-size PROBE)."]),
    Recipe(
        "2xp100-dense-pxq4", "2x Tesla P100 (sm_60), dense 27B, PXQ4",
        2, {60}, ("dense", "hybrid"), {"PXQ4"},
        b=8192, ub=256, ctx=32768, env={},
        numbers="337.56 @3,121 | 315.29 @20,801 | decode 18.14 @fill 8 (release binary 2729f12060, "
                "GPUs 1;5, quiet box, no env, engine auto -b 8192 -ub 256, n=3/3/12; fold n=7 "
                "reference 340.16 | 316.84 | 17.83)",
        source="RELEASE-NOTES-2026-09-07.md:78 (P100 headline row); bench/fair-battle.md:303 "
               "(release row + measurement note); P100 final-binary capture, 2026-09-05; "
               "the measurement ledger FINAL TABLE",
        notes=["-ub does NOT transfer between pairs: 2,048 is best on the V100 pair and 256 on "
               "this one (-ub 256 gives 218.4 t/s @3,121 against 231 at -ub 2048 on the DEFAULT "
               "chunk; at -b 8192 the 256 cell is the 340.16 above). "
               "the measurement ledger (HONEST numbers, UNIFIED build "
               "#2/#3: P100 ub256 218.4 vs ub2048 231); docs/COOKBOOK.md:103-106.",
               "-ub 512 at the same -b 8192 measured 323.26 / 310.38 / 17.76 (n=7) - about 5% "
               "behind (the measurement ledger)."]),
    Recipe(
        "1x1080ti-pxq2", "1x GTX 1080 Ti 11 GB (sm_61), 35B MoE, PXQ2",
        1, {61}, ("moe", "hybrid-moe"), {"PXQ2"},
        b=2048, ub=768, ctx=8192,
        env={"PXA_AUTO_SPEC": "0"},
        extra=["--ctx-checkpoints", "0"],
        numbers="cold prefill 1,363.5 t/s @-fa off  |  chat prefill 746.6 @-fa on  |  "
                "decode 36.73 cold / 65.3 chat  (release binary, no env, auto -b/-ub)",
        source="RELEASE-NOTES-2026-09-07.md:79 (headline table row); "
               "the measurement ledger",
        notes=["BOTH FA cells are MEASURED on this card, which is why --workload longdoc is a "
               "real choice here and not an inference: 1,363.5 cold at -fa off against 746.6 "
               "chat at -fa on, decode 36.73 against 65.3.",
               "These numbers need the sm_61 int8 prefill tile. PXA_ENHANCE=1 arms it (it "
               "resolves PXA_PXQ_INT8_PREFILL to mode 1 on sm_61 silicon). Without it the SAME "
               "build does 573.4 / 423.9 t/s, less than half. Confirm from the server log's "
               "'PXA_PXQ_INT8_PREFILL: mode 1' and 'PXA_PXQ_I8_BLUT: ON' lines - a log without "
               "them is not this configuration (bench/fair-battle.md:255-268).",
               "-ub 2048 does NOT fit: a ~1.9 GiB compute buffer cannot allocate next to the "
               "resident model on 11 GiB (docs/COOKBOOK.md:144-147).",
               "PXA_PXQ2_MMQ=1 changes nothing on this build (1,301 / 728.6) - the int8 path "
               "already carries that work. Not emitted.",
               "--ctx-checkpoints 0 is part of the published protocol, not a launcher "
               "preference: 4/4 of the arm files that produced these numbers pass it "
               "(the campaign arm files arms-pub-n14, arms-pub-ship, arms-n14, arms-ship, "
               "both FA regimes).",
               "PXA_AUTO_SPEC=0 IS REQUIRED HERE, AND IT IS THE ONE THING THE PUBLISHED RECIPE "
               "DOES NOT SAY. examples/server/server.cpp's PXA AUTO-SPEC block arms "
               "'--spec-type ngram-mod:n_max=4,n_min=2' automatically for arch qwen35moe "
               "whenever PXA_ENHANCE=1 - and this file IS qwen35moe. The spec context then "
               "tries a 254.06 MiB per-step checkpoint buffer, falls back to a 62.81 MiB "
               "shadow, and the seat dies mid-prefill on a 5.8k-token prompt with "
               "'ggml_backend_cuda_buffer_type_alloc_buffer: allocating 254.06 MiB ... out of "
               "memory'. MEASURED here 2026-09-04 on GPU 3, twice. The published arms never hit "
               "it because they ran at PXA level=DEFAULT with the two int8 levers set BY HAND "
               "(see PUB_N14_chat.server.log:10 'PXA level=DEFAULT'), so AUTO_SPEC never "
               "engaged - while docs/COOKBOOK.md's recipe exports PXA_ENHANCE=1, which does "
               "engage it. The card has ~200 MiB spare after 9,907 MiB of weights + 160 MiB KV "
               "+ 733 MiB compute buffer on 11,002 MiB free; there is no room for a drafter. "
               "PXA_ENHANCE=1 is still exported, because it is what arms the int8 prefill tile "
               "the numbers depend on - only the auto-drafter is turned off.",
               "AMENDED 2026-09-04 (defaults lane): the engine now declines this auto-arm by "
               "itself - AUTO_SPEC refuses below 2 GiB of estimated post-weights headroom and "
               "refuses outright on a single-card sm_61 fleet, printing 'PXA_AUTO: spec "
               "DECLINED', and docs/COOKBOOK.md's 1080 Ti recipe now says so. Emitting "
               "PXA_AUTO_SPEC=0 explicitly stays correct and is what makes the printed command "
               "mean the same thing on an older binary; it is no longer the only thing standing "
               "between this card and the OOM."]),
    Recipe(
        "4xp100-flashnext", "4x Tesla P100 (sm_60), Qwen3.8 Flash-Next hybrid MoE, 150k ctx",
        4, {60}, ("hybrid-moe",), set(),
        b=2048, ub=2048, ctx=150016,
        ts="5079,12612,12612,11897", ot=r"per_layer_token_embd\.weight=CPU",
        np=2, threads=16, extra=["-wgt", "8", "--kv-unified", "--no-context-shift"],
        env={},   # the eleven PXA_* levers are ENHANCE defaults on this topology (ENGINE_DEFAULTED_FLASHNEXT_LEVERS)
        numbers="prefill 487.6 t/s @3,121 tok  |  376.7 @20,801  |  decode 24.57 t/s at low "
                "fill  (n=3, worst half-spread 1.55%, release binary on auto defaults, "
                "2026-09-06)  |  seat-measured deep fill, hand flags: ~230 @~86,000 prefill, "
                "~19.3 decode at ~86k  (n=7 median, 1 warmup discarded)",
        source="docs/COOKBOOK.md:181-223 (levers :189-194, command :195-200, table :211-215); "
               "flags cross-checked verbatim against /usr/local/bin/pxa-seats-flashnext.sh "
               "(start_alex); RELEASE-NOTES-2026-09-02.md for the arm-by-arm ladder",
        arch="qwen4exp",
        notes=["THIS IS THE PRODUCTION SEAT'S OWN RECIPE, reproduced flag for flag. -ts "
               "5079,12612,12612,11897 is DELIBERATELY uneven: card 0 shares this box with "
               "another resident server, so its slice is small on purpose. Widening it OOMs "
               "the seat. The automatic capacity split is NOT used on this row.",
               "-ot per_layer_token_embd -> CPU is mandatory, not an optimisation: the PLE "
               "gather table is ~51 GiB of a ~97 GiB file and offloading it dies during load "
               "with a single-card cudaMalloc of the whole tensor.",
               "-ub 2048 is the best of 2048/1024/512/256 on this four-card split "
               "(495.07/413.36/28.23 against 305.00/275.69/27.12 at -ub 256) - the OPPOSITE of "
               "the 2x P100 pair (the measurement ledger Alex "
               "4x P100 seat ubatch sweep).",
               FLASHNEXT_SEAT_EXTRA_NOTE,
               PIPELINE_PP_NOTE,
               "PXA_FA_KEYS_PER_SPLIT and PXA_GEMV_RPB were measured in the same run and are "
               "NEGATIVE at this fill depth - left off deliberately (docs/COOKBOOK.md:219-223 lab footnote)."]),

    # -- the standing per-card cookbook rows (older runs, still the published ones)
    Recipe(
        "1xp100-pxqu16", "1x Tesla P100 16 GB (sm_60), 35B MoE, PXQU-16 + q8_0 head",
        1, {60}, ("moe", "hybrid-moe"), {"PXQ_UNIVERSAL"},
        b=2048, ub=2048, ctx=8192, env={},
        numbers="decode 62.4 t/s (63.0 with ADDFUSE)  |  prefill 827-843 t/s @-ub 2048",
        source="docs/COOKBOOK.md:65-73",
        notes=["Decode is ub-insensitive on this row - drop to -b/-ub 512 if you want a smaller "
               "compute buffer (docs/COOKBOOK.md:72-73)."]),
    Recipe(
        "1xv100-pxqu16", "1x Tesla V100 16 GB (sm_70), 35B MoE, PXQU-16 + q8_0 head",
        1, {70}, ("moe", "hybrid-moe"), {"PXQ_UNIVERSAL"},
        b=2048, ub=2048, ctx=8192, env={},
        numbers="decode ~101-102 t/s (101.3 published)  |  prefill ~1,800-1,900 t/s @-ub 2048",
        source="docs/COOKBOOK.md:75-78"),
    Recipe(
        "2xpair-flagship-moe", "2x P100 or 2x V100, 35B MoE flagship PXQ4/PXQ6 (18.7 GB)",
        2, {60}, ("moe", "hybrid-moe"), {"PXQ4", "PXQ6", "PXQ4-HQ"},
        b=8192, ub=2048, ctx=8192, ts="1,1", env={},
        numbers="decode 55.7 t/s (2x P100)  |  prefill ~843 t/s",
        source="docs/COOKBOOK.md:80-89"),
    Recipe(
        "2xpair-flagship-moe-v100", "2x V100, 35B MoE flagship PXQ4/PXQ6 (18.7 GB)",
        2, {70}, ("moe", "hybrid-moe"), {"PXQ4", "PXQ6", "PXQ4-HQ"},
        b=8192, ub=2048, ctx=8192, ts="1,1", env={},
        numbers="the 2x-pair row was published on the P100 pair; the V100 pair runs the same "
                "command and its own number was NOT taken",
        source="docs/COOKBOOK.md:80-89", status="INFERRED"),
    Recipe(
        "1x-stock-gguf", "any single card, a STOCK (non-PXQ) GGUF on this engine",
        1, {60, 61, 70}, "any", {"none"},
        b=2048, ub=2048, ctx=8192, env={},
        numbers="engine-only same-quant decode control: V100 84.5 -> 87.2 (+3.2%, bit-identical), "
                "P100 44.0 -> 45.2 (+2.7%), 1080 Ti 52.2 -> 53.9 (+3.3%)",
        source="docs/COOKBOOK.md:149-169; bench/fair-battle.md:52-61", status="INFERRED",
        notes=["-ub 2048 is the cookbook's shape for this row, not a swept best. On an 11 GiB "
               "card the launcher drops it to the card-type value because a ub2048 compute "
               "buffer does not fit (LEVERS.md:99-103)."]),
]

# A dense 27B PXQ4 on a 2x V100 pair is the ONE cell where both engines have a
# 2026-09-03 number, and they SPLIT: llama.cpp owns prefill, the vLLM sm_70 line
# owns decode. Printed whenever that cell is selected, on either engine, because
# an operator choosing a seat there is choosing which half to win.
DENSE_V100_ENGINE_SPLIT = (
    "2x V100 + dense 27B PXQ4 is a SPLIT CELL, measured on both engines the same week: "
    "llama.cpp (this engine) 1,369 / 1,300 t/s prefill and 39.5 t/s single-stream decode "
    "[RELEASE-NOTES-2026-09-07.md:77]; the vLLM sm_70 serving line ~1,009 / ~984 t/s prefill, "
    "~50.4 t/s single decode, ~190 aggregate @8 streams and ~299 @16 "
    "[docs/COOKBOOK.md:225-263]. Prefill-heavy or single-stream-latency work -> llama.cpp; "
    "many concurrent streams -> the vLLM line. The two were NOT run head to head in one "
    "harness, so the ratios are cross-harness and directional.")


def recipe_for(sel, prof, workload):
    """-> (recipe, status, evidence[], notes[]).

    status is MEASURED (hardware AND model class match a row exactly),
    [INFERRED] (the hardware matches a row but the model class does not, or the
    row itself is inferred), or UNMEASURED (no row's hardware matches at all)."""
    ev, nts = [], []
    if not sel:
        return None, "UNMEASURED", ev, ["no cards selected - no topology row can apply"]
    for r in RECIPES:
        if not r.hw_matches(sel):
            continue
        if r.model_matches(prof):
            st = r.status
            ev.append(f"{'MEASURED' if st == 'MEASURED' else '[INFERRED]'} topology row "
                      f"'{r.key}' ({r.title}): {r.numbers}  [{r.source}]")
            nts.extend(r.notes)
            return r, st, ev, nts
    # hardware matches a row but nothing in it fits this model
    for r in RECIPES:
        if r.hw_matches(sel):
            d = r.hardware_only()
            ev.append(f"[INFERRED] topology row '{r.key}' ({r.hw_str()}), REDUCED: the CARD "
                      f"topology matches but the model does not - that row was measured on "
                      f"{r.family_str()}/{'|'.join(sorted(r.tiers)) or 'any tier'}"
                      + (f"/arch {r.arch}" if r.arch else "") +
                      f" and this is {model_family(prof)}/{prof.get('tier') or 'no PXQ tier'}"
                      f"/arch {prof.get('arch') or '?'}. Only -b {d.b} -ub {d.ub} are borrowed. "
                      f"Its numbers are NOT quoted.  [{r.source}]")
            nts.append(f"[INFERRED] -b/-ub only, from '{r.key}'. Its -ts, -ot, -c, -np, -t and "
                       f"PXA_* levers are all model-specific and were DROPPED rather than "
                       f"applied to a file they were not measured on. Nothing was measured for "
                       f"this model class on this topology - measure before you quote anything.")
            return d, "INFERRED", ev, nts
    caps = sorted({g[2] for g in sel})
    nts.append(f"UNMEASURED topology: {len(sel)} card(s), "
               f"{'/'.join('sm_%d' % c for c in caps)}. No row in the recipe table covers it. "
               f"Rows that exist: " + ", ".join(f"{r.ncards}x{'/'.join('sm_%d' % c for c in sorted(r.caps))}"
                                                for r in RECIPES) + ". "
               f"No -b and no -ub are emitted: the engine's adaptive-ub probes each device at "
               f"startup, which is the right answer where nothing was measured.")
    ev.append("UNMEASURED topology - no recipe row applies; adaptive-ub decides")
    return None, "UNMEASURED", ev, nts



def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HARDWARE INTROSPECTION (H1..H8)
# ---------------------------------------------------------------------------
def gpu_table():
    """[(index, name, cc_int, mem_total_MiB, mem_used_MiB, uuid)] or (None, error)."""
    if not shutil.which("nvidia-smi"):
        return None, "nvidia-smi not found - cannot detect GPUs. Use --engine to force."
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,memory.used,uuid",
                "--format=csv,noheader,nounits"])
    if out is None:
        return None, "nvidia-smi failed (driver not loaded?). Use --engine to force."
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        try:
            rows.append((int(p[0]), p[1], int(round(float(p[2]) * 10)),
                         int(p[3]), int(p[4]), p[5]))
        except ValueError:
            continue
    if not rows:
        return None, "nvidia-smi returned no usable rows"
    return rows, None


def resident_procs(gpus):
    """H4 -> {gpu_index: [(pid, name, mib), ...]}. This is a SHARED, LIVE box; the
    launcher must never hand a card to a second process by accident. Keyed by UUID
    because --query-compute-apps reports gpu_uuid, not index."""
    by_uuid = {g[5]: g[0] for g in gpus}
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits"])
    res = collections.defaultdict(list)
    if not out:
        return res, (out is not None)
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4:
            continue
        idx = by_uuid.get(p[3])
        if idx is None:
            continue
        try:
            res[idx].append((p[0], os.path.basename(p[1]), int(p[2])))
        except ValueError:
            continue
    return res, True


def peer_topology():
    """H3 -> (has_p2p, description). This box is all-PHB, no NVLink, no P2P
    (SCOREBOARD.md:6) - which is WHY custom all-reduce is off in every measured
    vLLM arm (--disable-custom-all-reduce, MOE-CROSSOVER.md:79). CAR costs ~18%
    vs NCCL on MoE (CAR-VERDICT.md) while the CAR KERNEL is exonerated. We READ
    the topology; we do not hardcode the answer."""
    out = _run(["nvidia-smi", "topo", "-m"], timeout=25)
    if not out:
        return None, "topology UNREADABLE (nvidia-smi topo -m failed) - treating as no-P2P"
    links = set(re.findall(r"\b(NV\d+|PIX|PXB|PHB|SYS|NODE)\b", out))
    nvlink = {l for l in links if l.startswith("NV")}
    if nvlink:
        return True, f"NVLink present ({','.join(sorted(nvlink))})"
    return False, f"no NVLink/P2P; interconnect {'/'.join(sorted(links)) or 'unknown'}"


# ---------------------------------------------------------------------------
# MODEL INTROSPECTION (M1..M11) - header-only reads, no tensor data, no GPU
# ---------------------------------------------------------------------------
GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_header(path, max_tensors=200000):
    """Read the GGUF KV block AND the tensor directory. Returns
    {kv, tensors:[(name,type)], n_tensors, ok, err}. Never raises: a launcher must
    not die on a file it was merely trying to describe. `ok` False means the file
    is not a usable GGUF - which is a REFUSAL (R-23), not a shrug: a 5.3 GB file
    with a zeroed header exists on this box (/tmp/TRUNCATED-pxq4.gguf, first 4
    bytes 00 00 00 00) and the old launcher built a full command for it."""
    out = {"kv": {}, "tensors": [], "n_tensors": 0, "ok": False, "err": None}
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                out["err"] = f"bad magic {magic!r} (expected b'GGUF')"
                return out
            ver, = struct.unpack("<I", f.read(4))
            nt, = struct.unpack("<Q", f.read(8))
            nkv, = struct.unpack("<Q", f.read(8))
            if nt > max_tensors or nkv > 100000:
                out["err"] = f"implausible header (n_tensors={nt}, n_kv={nkv})"
                return out
            out["n_tensors"] = nt

            def rstr():
                n, = struct.unpack("<Q", f.read(8))
                if n > (1 << 24):
                    raise ValueError("absurd string length")
                return f.read(n).decode("utf-8", "replace")

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:                       # array
                    et, = struct.unpack("<I", f.read(4))
                    ln, = struct.unpack("<Q", f.read(8))
                    vals = []
                    for _ in range(ln):
                        v = rval(et)
                        if len(vals) < 4096:
                            vals.append(v)
                    return vals
                b = f.read(GGUF_FIXED[t])
                if t == 4:
                    return struct.unpack("<I", b)[0]
                if t == 5:
                    return struct.unpack("<i", b)[0]
                if t == 6:
                    return struct.unpack("<f", b)[0]
                if t == 7:
                    return bool(b[0])
                if t == 10:
                    return struct.unpack("<Q", b)[0]
                if t == 11:
                    return struct.unpack("<q", b)[0]
                return None

            for _ in range(nkv):
                k = rstr()
                t, = struct.unpack("<I", f.read(4))
                out["kv"][k] = rval(t)
            for _ in range(nt):
                nm = rstr()
                nd, = struct.unpack("<I", f.read(4))
                ne = 1
                for _ in range(nd):
                    d, = struct.unpack("<Q", f.read(8))
                    ne *= d
                ty, = struct.unpack("<I", f.read(4))
                struct.unpack("<Q", f.read(8))       # offset
                out["tensors"].append((nm, ty, ne))
            out["ok"] = True
    except Exception as e:
        out["err"] = f"{e.__class__.__name__} while reading the header ({e})"
    return out


def tier_from_tensors(tensors):
    """M3, done the way the ENGINE does it (src/llama-model-loader.cpp:527 detects
    PXQ1 by tensor TYPE). -> (tier, hist, unknown_types)

    tier is one of: a PXQ name, 'PXQ_UNIVERSAL' (more than one PXQ type present),
    or None (no PXQ tensors at all - a K-quant / MXFP4 / f16 file).
    PXQ1 anywhere in the mix makes this file PXQ1-BEARING regardless of what else
    is in it: a curated UNIVERSAL map with PXQ1 experts has the same generation
    failure as a uniform PXQ1 file (PXQ-TYPE-MATRIX Findings 7 and 8)."""
    hist = collections.Counter(t for _, t, _ in tensors)
    pxq = {t: n for t, n in hist.items() if t in PXQ_GGML_TYPE}
    # Anything in the 240..259 band that is not a DEFINED ggml type is a retired or
    # foreign codec this engine cannot dispatch (see KNOWN_GGML_TYPE_MAX). Checked
    # against ggml/include/ggml.h, not against a hardcoded list of survivors.
    unknown = [t for t in hist if 240 <= t <= 259 and t not in PXQ_GGML_TYPE]
    if not pxq:
        return None, hist, sorted(unknown)
    if 248 in pxq:
        return "PXQ1", hist, sorted(unknown)
    if len(pxq) > 1:
        return "PXQ_UNIVERSAL", hist, sorted(unknown)
    return PXQ_GGML_TYPE[next(iter(pxq))], hist, sorted(unknown)


def tier_from_provenance_kv(kv):
    """The corroborating signal. Written by llama-quantize.cpp:
      pxa.pxq6.tier core|hq|lm32 -> PXQ4 | PXQ4-HQ | PXQ6   (:1760-1767)
      pxa.pxqu.version           -> PXQ_UNIVERSAL           (:1790, pxqu_out only)
      pxa.pxq2.version / pxa.pxq3.version alone -> PXQ2 / PXQ3 (:1775, :1780)
    PXQ1 writes NO provenance KV at all (:1506) - which is exactly why this is the
    corroborating signal and the tensor walk is the authoritative one."""
    if "pxa.pxqu.version" in kv:
        return "PXQ_UNIVERSAL"
    t = kv.get("pxa.pxq6.tier")
    if t == "core":
        return "PXQ4"
    if t == "hq":
        return "PXQ4-HQ"
    if t == "lm32":
        return "PXQ6"
    if "pxa.pxq2.version" in kv:
        return "PXQ2"
    if "pxa.pxq3.version" in kv:
        return "PXQ3"
    return None


def model_kind(path):
    """M10. gguf | gguf_broken | vllm_dir | hf_dir | lora_dir | weightless_dir |
    not_a_model_file | not_a_model | missing

    'not_a_model_file' exists because the OLD taxonomy had no slot for it: any
    existing file that did not end in .gguf fell through to 'missing' and the
    launcher told the operator a path that plainly exists does not exist. The box
    is full of tempting single files next to real models (one shard of six, a
    .tiers map, an .imatrix, a LoRA adapter)."""
    if not os.path.exists(path):
        return "missing"
    if os.path.isfile(path):
        if path.endswith(".gguf"):
            return "gguf"
        return "not_a_model_file"
    if not os.path.isdir(path):
        return "not_a_model_file"
    cfgp = os.path.join(path, "config.json")
    has_weights = False
    for root, _, files in os.walk(path):
        if any(f.endswith((".safetensors", ".bin", ".gguf")) for f in files):
            has_weights = True
            break
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return "lora_dir"
    if os.path.exists(cfgp):
        if not has_weights:
            return "weightless_dir"
        try:
            cfg = json.load(open(cfgp))
            q = (cfg.get("quantization_config") or {}).get("quant_method")
            if q == "pxq4":
                return "vllm_dir"
        except Exception:
            pass
        return "hf_dir"
    if has_weights:
        return "not_a_model"          # weights but no config.json - GGUF dir? name the file.
    return "not_a_model"


def _hf_tensor_names(path):
    """Cheap tensor-name list for a safetensors dir: the index json's weight_map.
    Returns [] when there is no index (single-shard dirs) - and the caller then
    says UNMEASURABLE rather than 'absent'."""
    for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(path, idx)
        if os.path.exists(p):
            try:
                return list(json.load(open(p)).get("weight_map", {}).keys())
            except Exception:
                return []
    return []


def model_profile(path, kind):
    """-> dict. Best effort, never raises. Every field says where it came from."""
    p = {"arch": None, "n_expert": 0, "is_moe": False, "tier": None, "tier_kv": None,
         "chat_template": None, "sampling": {},
         "tier_src": "not inspected", "ftype": None, "n_ctx_train": None,
         "mtp_tensors": 0, "mtp_kv": None, "mtp_src": "not inspected",
         "deltanet": False, "deltanet_src": "not inspected", "vision": False,
         "kv_bytes_tok": None, "kv_bytes_src": "UNMEASURABLE",
         "unknown_types": [], "hist": {}, "quant_method": None,
         "why": "not inspected", "hdr_err": None}

    if kind in ("gguf", "gguf_broken"):
        h = gguf_header(path)
        p["hdr_err"] = h["err"]
        if not h["ok"]:
            p["why"] = f"GGUF header UNREADABLE: {h['err']}"
            return p
        kv, tn = h["kv"], h["tensors"]
        p["arch"] = kv.get("general.architecture")
        p["ftype"] = kv.get("general.file_type")
        arch = p["arch"] or ""
        p["n_expert"] = int(kv.get(f"{arch}.expert_count") or 0)
        if not p["n_expert"]:
            for k, v in kv.items():
                if k.endswith(".expert_count") and isinstance(v, int):
                    p["n_expert"] = max(p["n_expert"], v)
        p["is_moe"] = p["n_expert"] > 0
        p["n_ctx_train"] = kv.get(f"{arch}.context_length")
        tier, hist, unknown = tier_from_tensors(tn)
        p["tier"], p["hist"], p["unknown_types"] = tier, dict(hist), unknown
        p["tier_kv"] = tier_from_provenance_kv(kv)
        p["tier_src"] = ("tensor-type histogram over %d tensors (the signal the loader "
                         "dispatches on, llama-model-loader.cpp:527)" % len(tn))
        # M5: MTP by TENSOR WALK, never by the KV flag. Two shipped f16 files declare
        # nextn_predict_layers=1 with ZERO nextn tensors (PXA-Agent-9B-f16.gguf,
        # PXA-Coder-35B-v2-f16.gguf) - the head was dropped in the recovery pipeline
        # and the flag survived. Arming MTP on those arms a drafter that does not exist.
        p["mtp_tensors"] = sum(1 for n, _, _ in tn if "nextn" in n or ".mtp" in n)
        # PER-LAYER EMBEDDING (PLE). qwen4exp and gemma3n carry a per_layer_token_embd
        # table that is a pure GET_ROWS gather - one lookup per token per head, no
        # GEMM - and it is ENORMOUS: on Flash-Next it is 160 x 320001536 = 51.2e9
        # elements, ~51 GiB of a 97 GiB file. It belongs in host RAM.
        # MEASURED 2026-08-28: leaving it on the GPU made a 5-card seat try to
        # cudaMalloc 16089.57 MiB for it on one card and die during load.
        for _n, _t, _ne in tn:
            if _n.startswith("per_layer_token_embd"):
                p["ple_tensor"] = _n
                try:
                    # gguf_header stores ne as the TOTAL ELEMENT COUNT (an int),
                    # not a dims list - iterating it raises and silently loses the
                    # size, which is how this subtraction failed the first time.
                    e = int(_ne)
                    p["ple_elems"] = e
                    # GGML block geometry is FIXED and documented (ggml.h), so this is
                    # arithmetic, not an estimate. Only the types a PLE table is ever
                    # stored in are listed; anything else leaves ple_bytes None and the
                    # caller then declines to subtract rather than guessing.
                    _BPE = {0: 4.0, 1: 2.0, 30: 2.0,      # f32, f16, bf16
                            8: 34.0 / 32.0,               # q8_0
                            14: 210.0 / 256.0,            # q6_K
                            12: 144.0 / 256.0,            # q4_K
                            20: 18.0 / 32.0}              # iq4_nl
                    if _t in _BPE:
                        p["ple_bytes"] = int(e * _BPE[_t])
                        p["ple_type"] = _t
                except Exception:
                    p["ple_elems"] = None
                break
        p["mtp_kv"] = kv.get(f"{arch}.nextn_predict_layers")
        p["mtp_src"] = "tensor walk for nextn/mtp names"
        # M6 / SPEC CORRECTION C5 and C6: structural DeltaNet detection.
        # C6 - VERIFIED ON THE BOX, and it matters: LAUNCHER-SPEC M6 says to detect
        # the hybrid by "presence of linear_attn.* KV/tensors". That is the
        # SAFETENSORS name (coder35-moe-pxq4-m1's quantization_config.pxq4_modules
        # carries linear_attn.in_proj_qkvz). In GGUF the SAME layers are named
        # ssm_*: Fusion-Coder-80-P100-PXQ4.gguf (arch qwen3next) has blk.N.ssm_a,
        # ssm_ba, ssm_conv1d, ssm_dt, ssm_norm, ssm_out and ZERO tensors matching
        # linear_attn. So the spec's structural detector is dead code on the only
        # format llama.cpp - the engine -sm graph applies to - actually serves.
        # Both namings are checked here.
        lin = [n for n, _, _ in tn if "linear_attn." in n]
        ssm = [n for n, _, _ in tn if ".ssm_" in n or n.startswith("ssm_")]
        p["deltanet"] = bool(lin or ssm) or any("linear_attn" in k for k in kv)
        if lin:
            p["deltanet_src"] = f"{len(lin)} linear_attn.* tensors (safetensors-style naming)"
        elif ssm:
            p["deltanet_src"] = (f"{len(ssm)} ssm_* tensors - the GGUF spelling of the same "
                                 f"linear-attention layers. Whether a PURE-SSM (non-hybrid) "
                                 f"model needs the -sm graph guard is UNMEASURED; guarding is "
                                 f"the safe direction")
        else:
            p["deltanet_src"] = "no linear_attn.* or ssm_* tensors"
        p["vision"] = any(n.startswith(("v.", "mm.")) for n, _, _ in tn) or \
            any(k.startswith("clip.") for k in kv)
        p["kv_bytes_tok"], p["kv_bytes_src"] = kv_bytes_per_token(kv, arch)
        # The correctness settings, read from the file rather than assumed.
        # Block count, used by the model picker to tell a MODEL from a fixture.
        # STRUCTURAL, not a name match: this repo ships ggml-vocab-*.gguf tokenizer
        # fixtures under models/, and a name filter would miss the next fixture
        # someone adds under a different name while wrongly hiding a real model
        # that happens to be called ggml-something.
        p["n_block"] = sum(1 for n, _t, _e in tn if n.startswith("blk."))
        p["chat_template"] = kv.get("tokenizer.chat_template")
        p["sampling"] = {k.rsplit(".", 1)[1]: v for k, v in kv.items()
                         if k.startswith("general.sampling.")}
        p["why"] = "read from GGUF header + tensor directory"
        return p

    if kind in ("hf_dir", "vllm_dir", "weightless_dir", "lora_dir"):
        try:
            cfg = json.load(open(os.path.join(path, "config.json")))
        except Exception as e:
            p["why"] = f"config.json unreadable ({e.__class__.__name__})"
            return p
        archs = cfg.get("architectures") or []
        p["arch"] = (archs[0] if archs else None) or cfg.get("model_type")
        for key in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
            v = cfg.get(key)
            if isinstance(v, int) and v > 0:
                p["n_expert"] = max(p["n_expert"], v)
        for sub in ("text_config", "llm_config"):
            s = cfg.get(sub) or {}
            for key in ("num_experts", "n_routed_experts", "num_local_experts"):
                v = s.get(key)
                if isinstance(v, int) and v > 0:
                    p["n_expert"] = max(p["n_expert"], v)
        p["is_moe"] = p["n_expert"] > 0
        p["n_ctx_train"] = cfg.get("max_position_embeddings") or \
            (cfg.get("text_config") or {}).get("max_position_embeddings")
        qc = cfg.get("quantization_config") or {}
        p["quant_method"] = qc.get("quant_method")
        if p["quant_method"] == "pxq4":
            p["tier"] = "PXQ4"
            p["tier_src"] = "config.json quantization_config.quant_method == 'pxq4'"
        elif p["quant_method"]:
            p["tier"] = None
            p["tier_src"] = f"config.json quant_method == {p['quant_method']!r} (not a PXQ tier)"
        else:
            p["tier_src"] = "config.json carries NO quantization_config - unquantized checkpoint"
        p["vision"] = bool(cfg.get("vision_config") or (cfg.get("text_config") or {}).get("vision_config"))
        names = _hf_tensor_names(path)
        if names:
            p["mtp_tensors"] = sum(1 for n in names if "nextn" in n or ".mtp" in n)
            p["mtp_src"] = "safetensors index weight_map walk"
            p["deltanet"] = any("linear_attn." in n for n in names)
            p["deltanet_src"] = "linear_attn.* in safetensors index" if p["deltanet"] else \
                "no linear_attn.* in safetensors index"
        else:
            mods = qc.get("pxq4_modules") or []
            p["deltanet"] = any("linear_attn" in str(m) for m in mods)
            p["deltanet_src"] = ("quantization_config.pxq4_modules names linear_attn"
                                 if p["deltanet"] else
                                 "NO safetensors index and no linear_attn in pxq4_modules - "
                                 "DeltaNet presence is UNMEASURABLE from this directory")
            p["mtp_src"] = "UNMEASURABLE (no safetensors index in this directory)"
            p["mtp_tensors"] = -1
        p["kv_bytes_tok"], p["kv_bytes_src"] = kv_bytes_per_token_hf(cfg)
        p["why"] = "read from config.json" + (" + safetensors index" if names else "")
        return p
    return p


def kv_bytes_per_token(kv, arch):
    """M8. [INFERRED] - this is ARITHMETIC, not a measurement.
      bytes/token = n_layer * n_head_kv * (d_k*b_k + d_v*b_v)
    It replaces the old file's flat `ctx * 64 * 1024`, a constant annotated
    "~64 KiB/token measured on the qwen35 hybrid" and then applied to every dense
    model, every other MoE arch and every GQA ratio on the box. Neither the old
    constant nor this formula has been validated against a real allocation
    (measurement queue Q9), so NEITHER MAY BLOCK ANYTHING. It warns."""
    try:
        n_layer = kv.get(f"{arch}.block_count")
        n_kv = kv.get(f"{arch}.attention.head_count_kv")
        if isinstance(n_kv, list):
            n_kv = max(int(x) for x in n_kv) if n_kv else None
        n_head = kv.get(f"{arch}.attention.head_count")
        if isinstance(n_head, list):
            n_head = max(int(x) for x in n_head) if n_head else None
        d_emb = kv.get(f"{arch}.embedding_length")
        d_k = kv.get(f"{arch}.attention.key_length")
        d_v = kv.get(f"{arch}.attention.value_length")
        if d_k is None and d_emb and n_head:
            d_k = d_emb // n_head
        if d_v is None:
            d_v = d_k
        if not (n_layer and n_kv and d_k and d_v):
            return None, ("UNMEASURABLE from this header (missing block_count / head_count_kv / "
                          "key_length) - no fit estimate is printed")
        # HYBRID ATTENTION: NOT EVERY LAYER HOLDS A KV CACHE.
        # MEASURED 2026-08-28 on qwen4exp (Qwen3.8-Flash-Next). The flat
        # n_layer form above overestimates this arch by EXACTLY 4x, because
        # full_attention_interval=4 means only 12 of its 48 layers keep a KV
        # cache; the other 36 are gated-delta-net linear-attention layers whose
        # recurrent state is FIXED per sequence and does not grow with context.
        # Left uncorrected the launcher prices 160k ctx at 15.0 GiB instead of
        # 3.75 GiB and refuses seats that fit comfortably.
        kv_layers = n_layer
        interval = kv.get(f"{arch}.full_attention_interval")
        ratios = kv.get(f"{arch}.attention.compress_ratios")
        how = f"{n_layer} layers"
        if isinstance(ratios, (list, tuple)) and len(ratios) == n_layer:
            # Most direct evidence: one entry per layer, nonzero => full attention.
            n_full = sum(1 for r in ratios if r)
            if 0 < n_full < n_layer:
                kv_layers = n_full
                how = f"{n_full} of {n_layer} layers (compress_ratios)"
        elif isinstance(interval, int) and interval > 1 and n_layer % interval == 0:
            kv_layers = n_layer // interval
            how = f"{kv_layers} of {n_layer} layers (full_attention_interval={interval})"
        b = kv_layers * n_kv * (d_k * 2 + d_v * 2)     # f16 K and V
        # THE MEASUREMENT IS PER-ARCH. It was taken on qwen4exp and it does NOT
        # transfer: another arch may count its full-attention layers differently,
        # or keep a per-token component in the linear layers this formula ignores.
        # Claiming MEASURED on an arch nobody booted is the exact failure this
        # file's three-tag rule exists to prevent.
        if arch == "qwen4exp":
            tag = ("MEASURED 2026-08-28: predicts 24576 B/token and the engine allocated "
                   "EXACTLY 3516.00 / 3840.00 / 6144.00 MiB at c=150016 / 163840 / 262144. "
                   "Three exact hits - Q9 is CLOSED for this arch only.")
        elif kv_layers != n_layer:
            tag = ("[INFERRED] hybrid-attention correction applied from the qwen4exp "
                   "measurement, but NOT validated on this arch. Q9 stays OPEN here - "
                   "warn only, never blocks.")
        else:
            tag = ("[INFERRED] UNVALIDATED against a real allocation (Q9) - warn only, "
                   "never blocks")
        return b, (f"arithmetic: {how} x {n_kv} kv-heads x ({d_k}+{d_v}) dims x 2 B (f16) "
                   f"= {b/1024:.1f} KiB/token. {tag}")
    except Exception:
        return None, "UNMEASURABLE (header arithmetic failed) - no fit estimate is printed"


def kv_bytes_per_token_hf(cfg):
    """M8 for a safetensors checkpoint. Same [INFERRED] status as the GGUF form:
    arithmetic on header fields, UNVALIDATED against a real allocation (Q9), so it
    warns and never blocks."""
    t = cfg.get("text_config") or cfg
    try:
        n_layer = t.get("num_hidden_layers")
        n_kv = t.get("num_key_value_heads") or t.get("num_attention_heads")
        d_h = t.get("head_dim")
        if d_h is None and t.get("hidden_size") and t.get("num_attention_heads"):
            d_h = t["hidden_size"] // t["num_attention_heads"]
        if not (n_layer and n_kv and d_h):
            return None, ("UNMEASURABLE from config.json (missing num_hidden_layers / "
                          "num_key_value_heads / head_dim) - no fit estimate is printed")
        b = n_layer * n_kv * d_h * 2 * 2
        return b, (f"[INFERRED] arithmetic: {n_layer} layers x {n_kv} kv-heads x {d_h} head-dim "
                   f"x 2 (K+V) x 2 B (f16) = {b/1024:.1f} KiB/token. UNVALIDATED against a real "
                   f"allocation (Q9) - warn only, never blocks")
    except Exception:
        return None, "UNMEASURABLE (config.json arithmetic failed) - no fit estimate is printed"


def model_bytes(path, kind):
    if kind in ("gguf", "gguf_broken"):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    if kind in ("hf_dir", "vllm_dir", "weightless_dir", "lora_dir", "not_a_model"):
        t = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".gguf")):
                    try:
                        t += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        return t
    return 0


def find_mmproj(model_path, kind):
    """Coverage hole review item 3: the spec asserted a paired projector is
    'discoverable next to the model' and never said how. On this box
    muse-glimmer-30b-abl/ holds TWO F16 projectors and
    muse-glimmer-30b-abl-mmproj/ holds an F16 AND a Q8_0 for the same base model.
    There is no rule that picks a winner, so this does not invent one:
    exactly one candidate -> use it and say so; more than one -> R-24, refuse to
    guess; none -> say none was found."""
    if kind not in ("gguf", "gguf_broken"):
        return [], "mmproj discovery only applies to GGUF seats"
    d = os.path.dirname(os.path.abspath(model_path))
    try:
        cands = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if f.startswith("mmproj") and f.endswith(".gguf"))
    except OSError:
        return [], "model directory unreadable"
    return cands, f"{len(cands)} mmproj*.gguf sibling(s) in {d}"


def docker_has_image(tag):
    """True iff docker actually holds this tag. Advisory only: a missing or broken
    docker degrades to "not present", never to a traceback, because this is called
    while EXPLAINING a decision and an explanation that crashes is worse than a
    vague one. _run() already swallows non-zero exit and timeout and returns None."""
    return _run(["docker", "image", "inspect", "-f", "ok", tag], timeout=10) == "ok"


def has_vllm_pxq4():
    try:
        import vllm_pxq4  # noqa: F401
        return True
    except Exception:
        return False


def infer_workload(np_, explicit):
    return explicit or ("chat" if np_ <= 1 else "serve")


# ---------------------------------------------------------------------------
# vLLM ELIGIBILITY IS AN IMAGE PROPERTY, NOT A COMPUTE CAPABILITY
# ---------------------------------------------------------------------------
def vllm_eligibility(sel, image_arg):
    """-> (eligible_caps:set, image:str|None, probe_trail:[str])

    The old file's `MIN_VLLM_CAP = 70` made the entire vLLM branch unreachable on
    the hardware every vLLM cell in the decision table was measured on. The narrow
    true statement (SPEC CORRECTION C3): NO vLLM decode-or-prefill THROUGHPUT
    number on sm_70 exists anywhere in this corpus - the only sm_70 vLLM figure on
    record is a 3.76x shared-prefix win on the DGX (PORT-ASSESSMENT.md:82), which
    is not a seat decision.

    Resolution order, and the probe that failed is always named:
      1. --vllm-image / PXA_VLLM_IMAGE
      2. the image's declared arch set (table above)
      3. bare-metal vllm_pxq4 importable, arch set from PXA_PXQ4_LIB's sm tag
    If no image resolves, vLLM is NOT eligible and the reason names the probe -
    never a bare 'vLLM is sm_70+ only', which is the false statement it replaces.
    """
    trail = []
    img = image_arg or os.environ.get("PXA_VLLM_IMAGE")
    if img:
        rec = VLLM_IMAGES.get(img)
        if rec is None:
            trail.append(f"probe 1: image {img!r} given but UNKNOWN to this launcher - its arch "
                         f"set is UNMEASURED, so no card is declared eligible on its say-so")
            return set(), img, trail
        if rec["status"] == "INELIGIBLE":
            trail.append(f"probe 1: image {img!r} is INELIGIBLE: {rec['why']}")
            return set(), img, trail
        missing = [q for q in (rec.get("host_env") or {}).get("requires", [])
                   if not os.path.lexists(q)]
        if missing:
            # An image whose runtime lives on the host is only as present as those paths.
            # Declaring it eligible when they are gone routes traffic to a container that
            # cannot import torch, and the failure surfaces minutes later as a boot crash
            # instead of here as a sentence.
            trail.append(f"probe 1: image {img!r} needs host paths that are NOT PRESENT, so "
                         f"no card is eligible under it: " + ", ".join(missing))
            return set(), img, trail
        caps = set(rec["caps"])
        inf = set(rec.get("caps_inferred") or ())
        trail.append(f"probe 1: image {img!r} -> MEASURED caps {sorted(caps) or '-'}"
                     + (f", [INFERRED] caps {sorted(inf)}" if inf else "")
                     + f" ({rec['why']})")
        return caps | inf, img, trail
    trail.append("probe 1: no --vllm-image and no PXA_VLLM_IMAGE")
    lib = os.environ.get("PXA_PXQ4_LIB", "")
    m = re.search(r"libpxq4_sm(\d+)_v\d+\.so", lib)
    if m and has_vllm_pxq4():
        cc = int(m.group(1))
        trail.append(f"probe 3: bare-metal vllm_pxq4 importable AND PXA_PXQ4_LIB names "
                     f"sm_{cc} ({os.path.basename(lib)}) -> eligible for sm_{cc} only")
        return {cc}, f"bare-metal:{os.path.basename(lib)}", trail
    if has_vllm_pxq4():
        trail.append("probe 3: vllm_pxq4 IS importable but PXA_PXQ4_LIB does not name an sm "
                     "tag - the arch set of this build is UNMEASURED, so no card is declared "
                     "eligible. Set PXA_VLLM_IMAGE or PXA_PXQ4_LIB=libpxq4_sm<cc>_v<n>.so")
    else:
        trail.append("probe 3: vllm_pxq4 is not importable in this interpreter")
    # "on this box" has to MEAN on this box. This line previously printed every
    # MEASURED row in the table, so an image that had never been built was advertised
    # as available and the operator was sent to build a command around a tag docker
    # does not have. The table says what an image WOULD be good for; only docker says
    # whether it exists. Print both, and never let a missing docker turn an advisory
    # line into a crash.
    meas = [k for k, v in VLLM_IMAGES.items() if v["status"] == "MEASURED"]
    present = [k for k in meas if docker_has_image(k)]
    absent = [k for k in meas if k not in present]
    trail.append("gated images PRESENT on this box: " + (", ".join(present) or "NONE"))
    if absent:
        trail.append("gated images NOT BUILT here (build them before use): "
                     + ", ".join(absent))
    return set(), None, trail


# ---------------------------------------------------------------------------
# REFUSALS - R-01..R-21 are the spec's; R-22..R-28 were added by adversarial
# review of the spec (each comment says which hole it closes).
# Exit codes, unchanged from the previous version of this file:
#   2 = no decision / unusable artifact / unusable card selection
#   3 = a parameter or engine request that does not translate
#   4 = the environment cannot run the command (no engine binary, no CUDA)
#   5 = --explain produced a plan that carries known-fatal blockers
# ---------------------------------------------------------------------------
class Refusal(Exception):
    def __init__(self, rid, text, code=2):
        super().__init__(text)
        self.rid, self.text, self.code = rid, text, code


R = {
 "R-01": ("REFUSING: PXQ1 content. A PXQ1 MoE file loads, clears the composition gate at 80.9% "
          "PXQ-family bytes, and generates INCOHERENT text - nothing downstream catches it "
          "(PXQ-TYPE-MATRIX.md:113, Finding 7). PXQ1 has no dense path and no CPU codec. "
          "Requantize to PXQ2 or above, or use a curated per-expert PXQ1 map WITH a coherence "
          "check that you ran and can point at."),
 "R-02": ("REFUSING: retired quant type {name}. Types 250/251 were removed 2026-07-21 "
          "(ggml.h:467-470, 'never reuse this id') and no engine we ship reads them. Requantize."),
 "R-03": ("REFUSING to guess the PXQ tier. general.file_type is {ftype} for every PXQ tier by "
          "design (llama-quantize.cpp:1454-1509 rewrites them all to MXFP4=38 before :1658 "
          "writes it) and the tensor directory shows no PXQ tensor types either. Without a tier "
          "I cannot enforce the PXQ1 refusal or the vLLM PXQ4-only gate. Re-quantize with a "
          "current quantizer, or pass --tier <T> to assert it yourself and own that assertion."),
 "R-05": ("REFUSING: {method} is not readable by EITHER engine. llama.cpp reads the PXQ family "
          "and GGUF k-quants; it does not read a {method} safetensors directory. (The message "
          "this replaces told the operator llama.cpp would read it - false, and it would have "
          "sent them to a load failure.)"),
 "R-06": ("REFUSING: vLLM needs a converted artifact and {conv} does not exist. Convert: "
          "tools/vllm-pxq4/tools/gguf_to_vllm.py {model} {conv}. Or --engine llama to serve the "
          "GGUF directly."),
 "R-07": ("REFUSING: forced --engine vllm but NO selected card is eligible under image {img} "
          "({cards}). Not proceeding: with no eligible card the parallel degree collapses to 1, "
          "CUDA_VISIBLE_DEVICES is never set, and the server inherits EVERY GPU on this box. "
          "This is the live bug this refusal exists for - the previous version of this file "
          "reported the blocker and then launched anyway."),
 "R-08": ("REFUSING: cudagraph_mode={mode}. FULL_AND_PIECEWISE captures PREFILL graphs and "
          "returns fluent garbage from character zero on short raw /v1/completions prompts. Its "
          "best aggregate (88.4, SCOREBOARD M8) is BELOW the correct config's (88.7, M7). There "
          "is no speed argument for it."),
 "R-10": ("REFUSING: -ts with vLLM. vLLM splits parallel work EVENLY; a per-card ratio has no "
          "equivalent and would be silently ignored."),
 "R-11": ("REFUSING: -sm {sm} with vLLM. vLLM's parallelism model has no -sm equivalent."),
 "R-12": ("REFUSING: -sm graph on a DeltaNet hybrid ({why}). It produces DEGENERATE output - the "
          "cross-device all-reduce never reaches its consumers and each device computes a "
          "different router top-8. Not fixable by an env: PXA_ALLOW_GRAPH_SPLIT_HYBRID only "
          "removes the guard. Use -sm layer. Even where graph split works it is a phase trade, "
          "not a win: +64% prefill / -17% decode on 4x P100."),
 "R-13L": ("REFUSING: -ctk {k} / -ctv {v} has no compiled FA vec kernel at head 128 on this "
           "build - it does not fall back, it HARD-ABORTS at request time (on_no_fattn_vec_case, "
           "'Unsupported KV type combination for head_size 128'). Compiled asymmetric pairs "
           "here: q8_0/q6_0, q8_0/iq4_nl, q6_0/q5_0 (LEVERS.md:263). NOTE: NO measurement "
           "exists for ANY asymmetric pair - this is an unbenched VRAM trade, not a free one."),
 "R-13V": ("REFUSING: --ctk/--ctv have no vLLM equivalent and would be silently dropped. (They "
           "WERE silently dropped by the previous version of this file: the vLLM branch never "
           "read them and the translation check inspected only -ts/-sm.)"),
 "R-14": ("REFUSING: --spec mtp with vLLM. The vLLM path has no MTP drafter and I will not "
          "substitute ngram for it - on this model class the two have OPPOSITE verdicts (ngram "
          "+23.0% code / +4.6% prose; MTP -8.6% at n_max=1 and -29.8% at n_max=2, LEVERS.md:746). "
          "Substituting a lever's meaning is worse than dropping it. Ask for ngram explicitly if "
          "that is what you want."),
 "R-15A": ("REFUSING: --spec mtp:n_max={n}. n_max>=2 is a MEASURED LOSS on both arches: P100 "
           "54.9 -> 47.4 (-14%, accept 0.42); V100 92.7 vs 94.1 (accept 0.960 -> 0.480) "
           "(LEVERS.md:300-301). Use n_max=1."),
 "R-15B": ("REFUSING: --spec mtp but this file has NO nextn/mtp tensors. Its "
           "nextn_predict_layers KV says {kvn}; the tensor directory says 0. Two shipped f16 "
           "files carry exactly this lie (PXA-Agent-9B-f16.gguf, PXA-Coder-35B-v2-f16.gguf) - "
           "the head was dropped in the recovery pipeline and the flag survived."),
 "R-16": ("REFUSING: {tier} has no CPU codec (GPU-only, open task #62; PXQ-TYPE-MATRIX.md:67, "
          "RELEASE-GATE.md:177). A CPU-only or partially-offloaded run ABORTS. You asked for "
          "-ngl {ngl}: offload every layer or pick another tier."),
 "R-17A": ("REFUSING: -c {ctx} exceeds this model's trained context {trained} "
           "({arch}.context_length)."),
 "R-17B": ("REFUSING: the WEIGHTS ALONE ({mb:.2f} GiB) exceed the total VRAM of the selected "
           "cards ({tb:.2f} GiB across {n}) with full offload requested. This is file-byte "
           "arithmetic, not the KV estimate - it needs no formula and it blocks. Add cards, or "
           "pick a smaller artifact."),
 "R-18": ("REFUSING: {dir} is an unquantized HF checkpoint (config.json has no "
          "quantization_config.quant_method). llama.cpp needs a GGUF; vLLM needs a PXQ4-converted "
          "directory. Convert first - I will not emit 'vllm serve --quantization pxq4' against "
          "fp16 weights."),
 "R-19": ("REFUSING: {dir} contains no .safetensors/.gguf weights (config.json only, {n} bytes "
          "of weight files). A config-only stub is not a model."),
 "R-20": ("REFUSING: card {i} has {mib} MiB resident ({procs}). This is a SHARED, LIVE box. Pass "
          "--gpus explicitly for free cards, or --allow-busy if you own that process."),
 "R-21": ("REFUSING: np={n} needs a cudagraph capture ladder above the measured [1,2,4,8]. I can "
          "widen it, but nothing above np=8 has been measured on either engine and a too-short "
          "ladder has cliffed before (task #78: a hardcoded [1,2] cliffed at 3+ concurrent). "
          "Re-run with --accept-unmeasured."),
 # ---- added by adversarial review of LAUNCHER-SPEC (not in R-01..R-21) ----
 "R-22": ("REFUSING: {path} exists but is not a servable artifact ({kind}). {detail} The "
          "taxonomy this replaces had no slot for it and reported 'path does not exist' for a "
          "path that plainly does."),
 "R-23": ("REFUSING: {path} is not a usable GGUF - {detail}. A 5.3 GB file with a zeroed header "
          "sits on this box today (/tmp/TRUNCATED-pxq4.gguf) and the previous version of this "
          "file built a full launch command for it."),
 "R-24": ("REFUSING to guess an mmproj. {n} candidates sit next to this model at different "
          "precisions and no measurement ranks them:\n      {list}\n    Pass --mmproj <path>, or "
          "--no-mmproj to serve text-only."),
 "R-25": ("REFUSING: --draft-model (external draft-model speculation) has ZERO coverage in this "
          "corpus - no decode, prefill, acceptance or quality number exists for it on any cell. "
          "The only measured speculation here is ngram-mod self-speculation (+23.0% code) and "
          "MTP n_max=1 (a LOSS on sparse MoE). Re-run with --accept-unmeasured to try it anyway. "
          "vLLM: no draft-model path is emitted by this launcher at all."),
 "R-26": ("REFUSING: --np {n}. Concurrency must be >= 1; -c would compute to {ctx} and "
          "--max-num-seqs {n} would be emitted verbatim."),
 "R-27": ("REFUSING: this is a multimodal/VL checkpoint (vision_config / vision tensors present) "
          "and the vLLM command this launcher emits has NO multimodal handling - no "
          "--limit-mm-per-prompt, no image-token accounting, no projector. Nothing in the corpus "
          "measured a VL seat on vLLM. Use --engine llama (which does have --mmproj), or "
          "--accept-unmeasured if you intend to serve it text-only."),
 "R-29": ("REFUSING: {dir} is a PXQ4-CONVERTED vLLM directory, but the engine that wins this "
          "seat is llama.cpp - which cannot read safetensors. Why llama.cpp: {why} "
          "Serve the GGUF this directory was converted FROM, or change the inputs so vLLM wins "
          "(a vLLM-eligible image + card selection, and np>=6 for a MoE)."),
 "R-28": ("REFUSING: this file carries {n} tensor(s) of ggml type id(s) {ids}, which the CURRENT "
          "tree does not define (checked against ggml/include/ggml.h; PXQ1C/PXQ2C appear nowhere "
          "in the tree). The engine cannot dispatch them. A real file on this box is in exactly "
          "this state: qwen3-coder-next-pxqu/Fusion-Coder-80-PXQU.gguf, 106 tensors of type 247 "
          "and 38 of type 246 with pxa.pxq1c.*/pxa.pxq2c.* provenance KVs. Requantize with the "
          "current quantizer."),
}


# ---------------------------------------------------------------------------
# THE DECISION
# ---------------------------------------------------------------------------
class Plan(object):
    def __init__(self):
        self.engine = None
        self.reason = ""
        self.evidence = []      # each line already carries MEASURED/[INFERRED]/UNMEASURED
        self.notes = []
        self.blockers = []
        self.refusals = []      # (rid, text, code)
        self.needs_ack = []     # reasons --accept-unmeasured is required
        self.elig = []          # eligible GPU rows for vLLM
        self.image = None
        self.recipe = None      # the matched row of RECIPES, or None
        self.rstatus = "UNMEASURED"   # MEASURED | INFERRED | UNMEASURED
        self.fa = "on"          # the flash-attention regime, set from --workload
        self.chat = None        # ChatOpts: template, jinja, reasoning, sampling, keys

    def refuse(self, rid, code=2, **kw):
        self.refusals.append((rid, R[rid].format(**kw), code))


def envelope_notes(plan, sel, prof, np_, per_slot_ctx, model_path):
    """Section 4.4. The whole decision table is keyed to TWO CARDS OF ONE CLASS.
    Outside that envelope the launcher labels the answer and, where the spec says
    so, requires --accept-unmeasured. SCOREBOARD.md:273 is the evidence that even
    2->2 does not transfer: 22.3 vs 24.6 tok/s on an IDENTICAL config between card
    pairs 1,5 and 0,6."""
    caps = sorted({g[2] for g in sel})
    n = len(sel)
    if n == 2 and caps == [60]:
        plan.evidence.append("MEASURED envelope: exactly 2 cards, both sm_60 - the table applies "
                             "as measured (SCOREBOARD.md:6, MOE-CROSSOVER.md:3)")
    elif n == 2 and caps == [70]:
        plan.notes.append("[INFERRED]: 2x sm_70. NO engine-vs-engine number exists on Volta on "
                          "this box, for either class. The table below is applied, not measured "
                          "here.")
    elif n == 2 and len(caps) > 1:
        plan.notes.append("mixed-class pair: PXA_AUTO_TS fills -ts 1.4,0.6 on an exactly-2-device "
                          "mixed sm_70+sm_60 pair with -ts unset (+9.78% decode) - MEASURED on "
                          "ONE cell only (LEVERS.md:154, 'PXQ4-35B split V100+P100'); it does not "
                          "generalize.")
    if n not in (1, 2) and plan.rstatus == "MEASURED":
        plan.notes.append(f"{n}-card selection: this is a MEASURED topology row "
                          f"('{plan.recipe.key}'), so it is NOT treated as off-envelope. The "
                          f"ENGINE-CHOICE table below is still 2-card only - what was measured "
                          f"on {n} cards is the llama.cpp recipe, not a llama-vs-vLLM "
                          f"comparison.")
    elif n not in (1, 2):
        plan.notes.append(f"UNMEASURED card count: the MoE crossover and the dense pair were both "
                          f"measured on exactly 2 cards. You selected {n}. The 2-card answer is "
                          f"printed and labelled [INFERRED]. SCOREBOARD.md:273 shows 22.3 vs 24.6 "
                          f"on an identical config between two P100 PAIRS - even 2->2 does not "
                          f"transfer cleanly.")
        plan.needs_ack.append(f"{n}-card selection (table is 2-card only)")
    if 61 in caps:
        plan.notes.append("selection contains an sm_61 card: the sm_61 LAUNCH RECIPE is measured "
                          "(bench/fair-battle.md:230-273, GTX 1080 Ti section), but the np5/np6 ENGINE-CHOICE "
                          "thresholds were never measured on sm_61, and the BALANCE-mode "
                          "PXA_FA_MASK_SKIP_TILE win explicitly excludes all of sm_61 "
                          "(LEVERS.md:85). Recipe MEASURED, engine crossover UNMEASURED.")
    if per_slot_ctx != ANCHOR_CTX_PER_SLOT and not (plan.recipe and plan.rstatus == "MEASURED"):
        plan.notes.append(f"ctx per slot is {per_slot_ctx}, not the measured {ANCHOR_CTX_PER_SLOT}: "
                          f"the anchors ran --ctx-size np*4096. A different per-slot context "
                          f"changes the KV footprint and was NOT measured.")
    base = os.path.basename(model_path).lower()
    if prof.get("is_moe"):
        if not any(h in base for h in ANCHOR_MOE_HINTS):
            plan.notes.append("model is not the MoE anchor (PXA-Coder-35B-v2): the crossover is a "
                              "property of a model x hardware PAIR, not of the engines. "
                              "qwen3next (MoE-512), qwen3moe (MoE-128) and deepseek4 (MoE-6) have "
                              "NO engine-vs-engine data at any np. [INFERRED]")
    elif prof.get("tier"):
        if not any(h in base for h in ANCHOR_DENSE_HINTS):
            plan.notes.append("model is not the dense anchor (Qwen3.8-27B-Unc): the dense ratios "
                              "below are that model's, on one boot of the losing side. [INFERRED]")
    # H5 heterogeneity - coverage review item 8.
    ubs = {ub_for_card(g[3]) for g in sel}
    if len(ubs) > 1:
        plan.notes.append(f"HETEROGENEOUS -ub expectation across this selection: the card-type "
                          f"table (LEVERS.md:99-103) wants {sorted(ubs)} on different cards while "
                          f"the CLI carries ONE global -ub. This launcher passes NO -ub so the "
                          f"engine's adaptive-ub probes each device itself - but whether "
                          f"adaptive-ub lands per-card correctly in a heterogeneous pool is "
                          f"UNMEASURED.")


def decide(sel, kind, model, forced, prof, np_, workload, elig_caps, image, probe_trail,
           per_slot_ctx=ANCHOR_CTX_PER_SLOT):
    """-> Plan. Refusals first, always (spec section 4.0 order of evaluation)."""
    p = Plan()
    p.image = image
    prof = prof or {}
    names = ", ".join(f"{g[0]}:{g[1].replace('NVIDIA ', '')} sm_{g[2]}" for g in sel)
    tier = prof.get("tier")
    p.elig = [g for g in sel if g[2] in elig_caps]

    # ---- 1. artifact resolvable? -------------------------------------------
    if kind == "missing":
        p.reason = f"model path does not exist: {model}"
        return p
    if kind == "not_a_model_file":
        b = os.path.basename(model)
        detail = "It is an existing file this launcher cannot serve."
        m = re.search(r"-(\d{5})-of-(\d{5})\.safetensors$", b)
        if m:
            detail = (f"It is shard {int(m.group(1))} of {int(m.group(2))} of a safetensors "
                      f"checkpoint - pass the DIRECTORY, not one shard.")
        elif b.endswith(".tiers"):
            detail = "It is a PXQ-UNIVERSAL tier map (--pxq-universal input), not a model."
        elif b.endswith(".imatrix"):
            detail = "It is an importance matrix (quantizer input), not a model."
        elif b.endswith(".safetensors"):
            detail = "It is a single safetensors file - pass the DIRECTORY that contains it."
        p.refuse("R-22", path=model, kind=kind, detail=detail)
        return p
    if kind == "lora_dir":
        p.refuse("R-22", path=model, kind="LoRA adapter directory",
                 detail="It has adapter_config.json + adapter weights and no base model. "
                        "Merge or subtract it first, then serve the resulting checkpoint.")
        return p
    if kind == "weightless_dir":
        p.refuse("R-19", dir=model, n=model_bytes(model, kind))
        return p
    if kind == "not_a_model":
        p.reason = f"path exists but has no config.json and is not a .gguf: {model}"
        return p
    if kind in ("gguf", "gguf_broken") and prof.get("hdr_err"):
        p.refuse("R-23", path=model, detail=prof["hdr_err"])
        return p
    if kind == "hf_dir":
        qm = prof.get("quant_method")
        if qm in ("compressed-tensors", "awq", "gptq", "fp8", "bitsandbytes"):
            p.refuse("R-05", method=qm)
        else:
            p.refuse("R-18", dir=model)
        return p

    # ---- 2/3. tier readable? tier servable? --------------------------------
    if prof.get("unknown_types"):
        u = prof["unknown_types"]
        cnt = sum(prof.get("hist", {}).get(t, 0) for t in u)
        p.refuse("R-28", n=cnt, ids=", ".join(str(t) for t in u))
        return p
    ft = prof.get("ftype")
    if isinstance(ft, int) and ft in RETIRED_FTYPE:
        p.refuse("R-02", name=RETIRED_FTYPE[ft])
        return p
    if tier == "PXQ1":
        # I-7. Positive detection by tensor type, so this fires on a uniform PXQ1
        # file AND on a curated UNIVERSAL map with PXQ1-mapped experts - the
        # HIGH-severity silent case the review found unguarded in the spec, where
        # R-01's condition was `tier in {PXQ1}` and a PXQ_UNIVERSAL tag never matched.
        p.refuse("R-01")
        return p
    if kind in ("gguf",) and tier is None:
        # Not a PXQ file at all. That is fine and common (K-quants, MXFP4, f16) -
        # it is only a refusal when the file ALSO claims MXFP4 ftype with no
        # readable composition, which is the ambiguous case R-03 exists for.
        if ft == FTYPE_MXFP4:
            p.refuse("R-03", ftype=ft)
            return p
    if tier and prof.get("tier_kv") and prof["tier_kv"] != tier:
        # Never resolve a conflict silently. Both signals get printed; the tensor
        # walk wins because it is what the loader dispatches on.
        p.notes.append(f"TIER SIGNAL CONFLICT: tensor directory says {tier}; provenance KV says "
                       f"{prof['tier_kv']}. Using {tier} (the loader dispatches on tensor type, "
                       f"llama-model-loader.cpp:527). Both are printed rather than reconciled - "
                       f"this file was built by a quantizer whose KV conditions differ from "
                       f"llama-quantize.cpp:1760-1790 as it stands today.")
    if tier == "PXQ_UNIVERSAL":
        p.notes.append("PXQ_UNIVERSAL: this is a MIXED per-tensor tier map. "
                       "PXQ-TYPE-MATRIX.md:119 Finding 8 records a UNIVERSAL MoE that loads PASS "
                       "and generates INCOHERENT - with the doc's own caution that the "
                       "incoherence is traceable to that build recipe (Q3_K_M source, no imatrix) "
                       "rather than to the codecs. Nothing verifies a coherence check ran on THIS "
                       "file. llama.cpp only, and you must acknowledge it.")
        p.needs_ack.append("PXQ_UNIVERSAL tier (no coherence check is verified for this file)")

    if not sel and not forced:
        p.reason = "no GPUs selected/visible (use --engine to force anyway)"
        return p

    # ---- 5. structural gates: what CAN run, before what is fastest ---------
    forced_v = forced == "vllm"
    if forced_v and sel and not p.elig:
        p.refuse("R-07", code=3, img=image or "<none resolved>",
                 cards=names or "no GPUs visible")
        return p
    if forced:
        p.engine = forced
        p.reason = f"forced by --engine ({names or 'no GPUs visible'})"
        if not sel:
            p.blockers.append("no GPUs visible - proceeding on your say-so; the engine may fail "
                              "to start, and CUDA_VISIBLE_DEVICES cannot be scoped (I-12)")
        if forced_v:
            if tier and tier not in VLLM_SUPPORTED_PXQ:
                p.blockers.append(f"FORCED vllm but this model is {tier} and the vLLM backend "
                                  f"implements PXQ4 only (PXQ-TYPE-MATRIX.md:69-70)")
            if kind == "gguf":
                p.blockers.append("FORCED vllm but the model is a raw .gguf - vLLM needs the "
                                  "converted form")
        if forced == "llama" and kind == "vllm_dir":
            p.blockers.append("FORCED llama but the model is a PXQ4-CONVERTED vLLM directory - "
                              "llama.cpp cannot read safetensors. Serve the GGUF it was "
                              "converted from.")
        for t in probe_trail:
            p.evidence.append("eligibility " + t)
        return p

    if tier and tier not in VLLM_SUPPORTED_PXQ and tier != "PXQ4":
        p.engine, p.reason = "llama", (
            f"tier is {tier}: the vLLM backend implements PXQ4 ONLY and refuses every other tier "
            f"cleanly at the conversion gate (PXQ-TYPE-MATRIX.md:69-70). llama.cpp reads "
            f"PXQ2/PXQ3/PXQ4/PXQ4-HQ/PXQ6/UNIVERSAL. Cards: {names}")
        p.evidence.append("MEASURED tier support: PXQ-TYPE-MATRIX.md:69-70, :80-81")
    elif tier is None and kind == "gguf":
        p.engine, p.reason = "llama", (
            f"no PXQ tensors in this file (composition: "
            f"{compose_str(prof.get('hist', {}))}); it is a stock GGUF quantization. "
            f"vLLM's PXQ4 backend has nothing to load. Cards: {names}")
    elif kind == "gguf":
        p.engine, p.reason = "llama", (
            f"model is a raw GGUF ({os.path.basename(model)}); vLLM needs a converted artifact "
            f"(tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: {names}")
    elif not p.elig:
        p.engine, p.reason = "llama", (
            f"no vLLM-eligible card in this selection ({names}). "
            f"{probe_trail[-1] if probe_trail else 'no image probe succeeded'}")
        for t in probe_trail:
            p.evidence.append("eligibility " + t)
    elif len(p.elig) < len(sel):
        p.engine, p.reason = "llama", (
            f"only {len(p.elig)}/{len(sel)} selected cards are eligible under image "
            f"{image}; vLLM cannot span a mixed-arch selection on a single-arch image, and "
            f"dropping the rest would silently change the parallel degree. Cards: {names}")
    elif len({g[2] for g in p.elig}) > 1:
        p.engine, p.reason = "llama", (
            f"selection spans more than one compute capability ({sorted({g[2] for g in p.elig})}) "
            f"and the vLLM command carries ONE --attention-backend. Cards: {names}")
    elif len(sel) < 2:
        p.engine, p.reason = "llama", (
            f"single GPU - no parallelism to gain and llama.cpp has lower single-stream "
            f"overhead. This is also the only measured single-card ground in the corpus "
            f"(QUANT-SPEED-AB.log, 1x P100 card 1). Cards: {names}")
    else:
        # ---- 6. both engines can run it. Pick on MEASURED performance. -----
        if not prof.get("is_moe"):
            p.engine = "vllm"
            s, a8, pf = DENSE_NUMBERS["single"], DENSE_NUMBERS["agg8"], DENSE_NUMBERS["prefill"]
            p.reason = (f"DENSE model on {len(sel)} eligible card(s) ({names}). vLLM wins dense at "
                        f"every workload MEASURED: {s[0]} vs {s[1]} tok/s single ({s[2]}), "
                        f"{a8[0]} vs {a8[1]} agg@8 ({a8[2]}), {pf[0]} vs {pf[1]} prefill ({pf[2]}).")
            p.evidence.append("MEASURED dense: SCOREBOARD rows D1 (llama.cpp) / D3 (vLLM), "
                              "27B PXQ4, 2x P100 sm_60, cards 1,5")
            p.notes.append(DENSE_WEAKNESS)
            if np_ == 4:
                p.notes.append(DENSE_AGG4_NOTE)
        elif workload == "longdoc":
            p.engine = "llama"
            p.reason = (f"MoE model, long-document workload ({names}). llama.cpp -sm layer holds "
                        f"the prefill record at every concurrency measured.")
            p.notes.append(MOE_LONGDOC_NOTE)
            p.evidence.append("MEASURED MoE longdoc: SCOREBOARD section 0.2 - cross-harness, "
                              "see the note")
        else:
            p.engine, p.reason, ev, nts = moe_seat(np_, names)
            p.evidence.extend(ev)
            p.notes.extend(nts)

    # A converted vLLM directory is not readable by llama.cpp. Routing one there
    # would emit a llama-server command against safetensors - a runnable-looking
    # command that cannot run. This gate did not exist before.
    if p.engine == "llama" and kind == "vllm_dir":
        p.refuse("R-29", dir=model, why=(p.reason or "no eligible card") + ".")
        return p

    # ---- 7. the LAUNCH RECIPE. Separate question from the ENGINE question:
    # which engine wins is a llama-vs-vLLM comparison; which flags to pass is a
    # per-topology measurement on ONE engine. They have different corpora and
    # different dates, so they are resolved separately and labelled separately.
    rec, rstatus, rev, rnotes = recipe_for(sel, prof, workload)
    p.recipe, p.rstatus = rec, rstatus
    p.fa = FA_BY_WORKLOAD.get(workload, "on")
    p.evidence.extend(rev)
    p.notes.extend(rnotes)
    if rec is None and sel:
        p.needs_ack.append(f"UNMEASURED topology ({len(sel)} card(s), "
                           f"{'/'.join('sm_%d' % c for c in sorted({g[2] for g in sel}))}): no "
                           f"recipe row covers it, so no -b/-ub is emitted")
    if p.engine == "llama":
        _fa_note(p, rec, rstatus, workload)
    # A MEASURED row that names this model's arch or its tier IS the coherence
    # evidence the PXQ_UNIVERSAL acknowledgement asks for: that row exists because
    # somebody booted THIS class of file on THIS topology and gated the output.
    # Leaving the ack in place there would make the launcher demand
    # --accept-unmeasured for the seat with the most measurement behind it, which
    # trains people to pass the flag by reflex - the opposite of what it is for.
    if rstatus == "MEASURED" and rec is not None and (rec.arch or "PXQ_UNIVERSAL" in rec.tiers):
        before = len(p.needs_ack)
        p.needs_ack = [x for x in p.needs_ack if not x.startswith("PXQ_UNIVERSAL tier")]
        if len(p.needs_ack) != before:
            p.notes.append(
                f"PXQ_UNIVERSAL acknowledgement WAIVED by recipe row '{rec.key}': that row was "
                f"measured on this tier"
                + (f" and this arch ({rec.arch})" if rec.arch else "") +
                f", with output-gated boots, which is the coherence evidence the general "
                f"PXQ_UNIVERSAL warning asks for. The warning above still stands for any OTHER "
                f"UNIVERSAL file - it is waived for this cell, not for the tier.")
    if (len(sel) == 2 and {g[2] for g in sel} == {70}
            and not prof.get("is_moe") and prof.get("tier") == "PXQ4"):
        p.notes.append(DENSE_V100_ENGINE_SPLIT)
    if (prof.get("arch") or "") in PIPELINE_PP_DEFAULT_ARCHES:
        p.notes.append("PXA_PIPELINE_PP is ON by ENGINE default for arch "
                       f"'{prof.get('arch')}' (qwen35/qwen35moe only). This launcher does not "
                       "set or unset it; the seat inherits the engine's default. "
                       + PIPELINE_PP_NOTE)

    envelope_notes(p, sel, prof, np_, per_slot_ctx, model)
    if p.engine == "vllm":
        if prof.get("is_moe"):
            p.notes.append(MOE_CURRENCY_NOTE)
        for t in probe_trail:
            p.evidence.append("eligibility " + t)
    return p


def _fa_note(p, rec, rstatus, workload):
    """The FA regime is the one launch choice a user makes with their own head, so
    it is always explained, and the label says whether THIS card's other regime was
    actually measured or only inferred from the shared regime table."""
    fa = p.fa
    if rec is not None and rec.key == "1x1080ti-pxq2":
        p.evidence.append(
            f"MEASURED FA regime on this exact card: -fa {fa} for workload '{workload}'. Both "
            f"cells were run: cold prefill 1,363.5 t/s at -fa off against chat prefill 746.6 at "
            f"-fa on; decode 36.73 cold against 65.3 chat "
            f"[RELEASE-NOTES-2026-09-07.md:79]")
        return
    if workload == "longdoc":
        p.notes.append(
            f"-fa off for --workload longdoc. [INFERRED] on this topology: the regime split is "
            f"MEASURED per CARD on single-card 35B runs (P100 817 -> 1,213 prefill / 56.7 -> "
            f"41.1 decode; V100 1,589 -> 1,700 / 94.1 -> 76.6; 1080 Ti 667 -> 1,001 / 65.4 -> "
            f"34.2) and the recipe row above was taken at -fa on, so the DIRECTION carries and "
            f"the magnitude does not. [{FA_REGIME_SRC}]")
        return
    p.evidence.append(
        f"MEASURED FA regime: -fa on for workload '{workload}' - the interactive/serving "
        f"setting, and what every recipe row above was measured at. Switch with "
        f"--workload longdoc if you are ingesting rather than chatting. [{FA_REGIME_SRC}]")


def moe_seat(np_, names):
    """The split seat. The TABLE is stored, never a slope (see MOE_TABLE)."""
    ev, nts = [], []
    if np_ <= 1:
        l, v, w, m = MOE_NP1
        ev.append(f"MEASURED np=1: llama.cpp {l} vs vLLM {v} tok/s ({m}) [SCOREBOARD M1/M7]")
        return "llama", (f"MoE at np=1 on {names}. llama.cpp -sm layer wins single-stream by "
                         f"{m}: {l} vs {v} tok/s."), ev, nts
    if np_ in MOE_TABLE:
        l, v, w, m = MOE_TABLE[np_]
        ev.append(f"MEASURED np={np_}: llama.cpp {l} vs vLLM {v} tok/s, {w} by {m} "
                  f"[MOE-CROSSOVER.md section 1, 11 gated boots, cards 0+6]")
        if np_ == 5:
            nts.append("np=5 is llama.cpp's PEAK (79.49, ABOVE its own np4 75.93) and it drops "
                       "12.5% in ONE step to np6. Do not read np5 as a point on a line from np4 "
                       "to np8 - that reading misprices it by ~14%.")
        eng = "llama" if w == "llama" else "vllm"
        return eng, (f"MoE at np={np_} on {names}. {'llama.cpp -sm layer' if eng == 'llama' else 'vLLM PP + FULL_DECODE_ONLY'} "
                     f"wins by {m}: {l} vs {v} tok/s aggregate."), ev, nts
    if np_ < 4:
        # np2/np3: bracketed by MEASURED np1 (+214%) and np4 (+17.1%), both
        # llama.cpp, so the WINNER is safe and the MARGIN is not. No cell exists.
        ev.append(f"[INFERRED] np={np_}: no np2/np3 cell exists. Bracketed by MEASURED np1 "
                  f"(llama.cpp 3.14x) and np4 (llama.cpp +17.1%), both llama.cpp.")
        nts.append(f"np={np_} is [INFERRED]: the winner is bracketed on both sides and is safe; "
                   f"the MARGIN is unknown and no number is printed for it.")
        return "llama", f"MoE at np={np_} on {names}. llama.cpp, by bracketing - see the note.", ev, nts
    # np > 8
    ev.append(f"[INFERRED] np={np_}: nothing above np={MOE_TABLE_MAX_NP} was run on either "
              f"engine. Both trends are established THROUGH np8 (llama.cpp 62.42 falling, vLLM "
              f"95.81 climbing).")
    nts.append(f"np={np_} is above the measured table. The direction is [INFERRED]; the capture "
               f"ladder is the real unknown and it is UNMEASURED (see R-21).")
    return "vllm", f"MoE at np={np_} on {names}. vLLM, by extrapolated direction - see the note.", ev, nts


def compose_str(hist):
    parts = []
    for t, n in sorted(hist.items(), key=lambda kv: -kv[1])[:6]:
        nm = PXQ_GGML_TYPE.get(t) or NON_PXQ_GGML_TYPE.get(t) or f"type{t}"
        parts.append(f"{nm}x{n}")
    return " ".join(parts) if parts else "empty"


# ---------------------------------------------------------------------------
# VRAM - see SPEC CORRECTION C4. Only formula-free facts may block.
# ---------------------------------------------------------------------------
def vram_check(plan, sel, mbytes, ctx, prof, ngl_all):
    notes = []
    if not sel:
        return notes
    total = sum(g[3] for g in sel) * 1024 * 1024
    free = sum((g[3] - g[4]) for g in sel) * 1024 * 1024
    # R-17B COMPARES GPU-RESIDENT BYTES, NOT FILE BYTES.
    # A per-layer-embedding table is pinned to host RAM by -ot and NEVER reaches
    # VRAM, so counting it here refuses seats that fit. MEASURED 2026-08-28:
    # Flash-Next PXQU is a 96.77 GiB file of which 51.15 GiB is per_layer_token_embd;
    # the GPU-resident remainder is 46.70 GiB and runs on five cards (75 GiB) with
    # room for a 3.75 GiB KV cache at 160k. Uncorrected, R-17B refused it outright.
    gpu_bytes = mbytes
    ple_note = None
    if mbytes and prof.get("ple_bytes"):
        gpu_bytes = mbytes - prof["ple_bytes"]
        ple_note = (f"PLE: {prof['ple_tensor']} is {prof['ple_bytes']/BYTES_PER_GIB:.2f} GiB and "
                    f"is pinned to host RAM by -ot, so the VRAM figures below use the "
                    f"GPU-resident remainder {gpu_bytes/BYTES_PER_GIB:.2f} GiB, not the "
                    f"{mbytes/BYTES_PER_GIB:.2f} GiB file.")
    elif mbytes and prof.get("ple_tensor"):
        ple_note = ("PLE present but its ggml type is not in the block-geometry table, so its "
                    "bytes were NOT subtracted. The VRAM figures below OVERSTATE what reaches "
                    "the cards - treat any refusal here as suspect and re-check by hand.")
    if ple_note:
        notes.append(ple_note)
    if gpu_bytes and ngl_all and gpu_bytes > total:
        plan.refuse("R-17B", mb=gpu_bytes / BYTES_PER_GIB, tb=total / BYTES_PER_GIB, n=len(sel))
        return notes
    if not mbytes:
        return notes
    kvb = prof.get("kv_bytes_tok")
    if kvb:
        kv = kvb * ctx
        need = gpu_bytes + kv
        notes.append(f"VRAM estimate [INFERRED, never blocks]: weights {gpu_bytes/BYTES_PER_GIB:.2f} "
                     f"GiB + KV {kv/BYTES_PER_GIB:.2f} GiB (= ctx {ctx} x "
                     f"{kvb/1024:.1f} KiB/tok) = {need/BYTES_PER_GIB:.2f} GiB vs "
                     f"{free/BYTES_PER_GIB:.2f} GiB free / {total/BYTES_PER_GIB:.2f} GiB total "
                     f"across {len(sel)} card(s). Compute buffers and fragmentation are NOT in "
                     f"this number.")
        notes.append("KV/token source: " + prof.get("kv_bytes_src", "?"))
        if need > free:
            notes.append("TIGHT FIT by that estimate. It is arithmetic, not a measurement (Q9 is "
                         "'measure KV bytes/token per arch family'), so it WARNS and does not "
                         "block. Reduce -c or add cards if the boot OOMs.")
        # MEASURED 2026-08-28: A CONFIG THAT LOADS IS NOT A CONFIG THAT RUNS.
        # A 5-card Flash-Next seat at c=262144 loaded cleanly, printed every buffer,
        # and then died on the FIRST TOKEN with "CUDA error: out of memory" in
        # llama_decode - the split had left card 0 with 51 MiB. Decode allocates
        # transient buffers beyond the compute buffer llama.cpp reports at init.
        # The same seat at c=163840 kept 807-1803 MiB free per card and ran.
        notes.append("HEADROOM RULE [MEASURED]: leave ~1200 MiB free per card AFTER load. "
                     "Decode allocates transient buffers that the init-time buffer report does "
                     "not include, so a seat can load and still OOM on its first token.")
    else:
        notes.append("VRAM estimate: " + prof.get("kv_bytes_src", "UNMEASURABLE") +
                     f" (weights alone {mbytes/BYTES_PER_GIB:.2f} GiB vs "
                     f"{free/BYTES_PER_GIB:.2f} GiB free)")
    return notes


# ---------------------------------------------------------------------------
# ENGINE RESOLUTION (llama.cpp build dirs)
# ---------------------------------------------------------------------------
ENGINE_DIR_CANDIDATES = [
    os.environ.get("PXA_ENGINE_DIR", ""),         # set PXA_ENGINE_DIR to your build dir
    "/mnt/models/pxa-sky-build/build70",          # DGX, sm_70
    "/mnt/models/PXA/build70",              # DGX, sm_70
    "./build-unified",                            # in-tree build
    "./build",                                    # in-tree build
]


def engine_ld_path(E):
    """These builds link libllama / libggml / libmtmd out of the build tree, not a
    system prefix. Without them the binary exists, is executable, and dies
    instantly on a missing .so.

    AND: any inherited /stubs directory is STRIPPED. Review found this - and
    PERPLEXITY-RESULTS.md:41-55 reproduces it on this box: a 66 KB stub
    libcuda.so.1 shadows the real 96 MB driver, ggml logs one line
    ('CUDA driver is a stub library'), offloads 0/33 layers, and the run is
    numerically CORRECT and ~50x SLOWER. Several existing helper scripts on this
    box still carry the stub dir. Returns (path, dropped[])."""
    parts = [f"{E}/bin", f"{E}/src", f"{E}/ggml/src", f"{E}/examples/mtmd",
             f"{E}/common", f"{E}/ggml/src/ggml-cuda"]
    existing = [p for p in parts if os.path.isdir(p)]
    prior = [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]
    dropped = [x for x in prior if "/stubs" in x]
    kept = [x for x in prior if "/stubs" not in x]
    return ":".join(existing + kept), dropped


def engine_runs(E):
    exe = f"{E}/bin/llama-server"
    if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        return False, "no executable bin/llama-server"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"], _ = engine_ld_path(E)
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=25, env=env)
    except Exception as e:
        return False, f"{e.__class__.__name__} invoking --version"
    blob = (r.stdout or "") + (r.stderr or "")
    if "error while loading shared libraries" in blob:
        missing = blob.split("error while loading shared libraries:")[-1].strip().split(":")[0]
        if missing.startswith(("libcudart", "libcuda.", "libcublas", "libnvrtc")):
            return False, f"NO CUDA RUNTIME ON THIS HOST ({missing})"
        return False, f"cannot load {missing} (even with the build's own lib dirs on LD_LIBRARY_PATH)"
    if r.returncode != 0 and not blob.strip():
        return False, f"--version exited {r.returncode} with no output"
    return True, "starts"


def resolve_engine_dir():
    tried = []
    env = os.environ.get("PXA_ENGINE_DIR")
    if env:
        ok, why = engine_runs(env)
        return env, ("PXA_ENGINE_DIR" if ok else f"PXA_ENGINE_DIR -- WILL NOT START: {why}")
    for d in ENGINE_DIR_CANDIDATES:
        if not os.path.isfile(f"{d}/bin/llama-server"):
            continue
        ok, why = engine_runs(d)
        if ok:
            note = "auto-detected" if not tried else f"auto-detected (skipped {len(tried)} broken)"
            return d, note
        tried.append(f"{d} ({why})")
    onpath = shutil.which("llama-server")
    if onpath:
        d = os.path.dirname(os.path.dirname(onpath))
        ok, why = engine_runs(d)
        if ok:
            return d, "found on PATH"
        tried.append(f"{d} on PATH ({why})")
    if tried:
        if all("NO CUDA RUNTIME ON THIS HOST" in t for t in tried):
            head = "this host has no CUDA runtime, so no build here can start:"
        else:
            head = "every candidate build is present but will not start:"
        return None, head + "\n      " + "\n      ".join(tried)
    return None, "no llama-server found"


# ---------------------------------------------------------------------------
# COMMAND CONSTRUCTION
# ---------------------------------------------------------------------------
COMPILED_CTKV_PAIRS = {("f16", "f16"), ("q8_0", "q8_0"), ("q6_0", "q6_0"), ("q5_0", "q5_0"),
                       ("q4_0", "q4_0"), ("q8_0", "q6_0"), ("q8_0", "iq4_nl"), ("q6_0", "q5_0")}


def parse_spec(spec):
    """-> (method, params dict) for 'mtp:n_max=1' / 'ngram-mod:n_max=4,n_min=2'."""
    if not spec:
        return None, {}
    method, _, rest = spec.partition(":")
    params = {}
    for kvp in rest.split(","):
        if "=" in kvp:
            k, _, v = kvp.partition("=")
            params[k.strip()] = v.strip()
    return method.strip(), params



# ---------------------------------------------------------------------------
# AUTOMATIC -ts FROM REAL FREE VRAM
# ---------------------------------------------------------------------------
# MEASURED 2026-08-28 on Flash-Next PXQU over cards 0,1,3,5,6.
#
# WHY THIS EXISTS: an even split is the default and it is WRONG on any pool where
# the cards are not equally free. Card 0 here carries a production seat (7865 of
# 16384 MiB free) and card 3 is an 11 GiB 1080 Ti also carrying production. An
# even five-way split puts ~9.2 GiB of weights on a card with 7.8 GiB free.
#
# -ts PARTITIONS BYTES, NOT LAYERS (llama.cpp:4071-4100) and llama.cpp folds a
# per-device compute allowance into the same walk - which is why CHANGING -ub
# REPACKS THE LAYERS and a -ts tuned at one ub can OOM at another (measured: PXQ4
# six-card at ub2048 pushed a 16140 MiB V100 over with the ub512-tuned split).
#
# THE HEADROOM TERM IS THE WHOLE POINT. A config that LOADS is not a config that
# RUNS: at c=262144 this model loaded, printed every buffer, then died on the
# first token with "CUDA error: out of memory" in llama_decode because the split
# left card 0 with 51 MiB. Decode allocates transient buffers that the init-time
# buffer report does not include.
#
# Compute-buffer figures are MEASURED at ub1024 and interpolated linearly in ctx:
#   ordinary card   282 MiB @ c8192   567 @ c150016   786 @ c262144
#   head card       980 MiB, FLAT - it did not grow with context across that range
TS_HEADROOM_MIB = 1200      # MEASURED: 807-1803 free per card ran; 51 free OOMd
TS_CUDA_CTX_MIB = 250       # per-device CUDA context, approximate
TS_HEAD_COMPUTE_MIB = 980   # MEASURED, flat in ctx


def _compute_buf_mib(ctx):
    """MEASURED at ub1024, linear interpolation in ctx between the two anchors."""
    lo_c, lo_v, hi_c, hi_v = 8192, 282.0, 262144, 786.0
    if ctx <= lo_c:
        return lo_v
    if ctx >= hi_c:
        return hi_v
    return lo_v + (hi_v - lo_v) * (ctx - lo_c) / (hi_c - lo_c)


def auto_tensor_split(sel, ctx):
    """Capacity-proportional -ts over the SELECTED cards, using free VRAM.

    Returns (ts_string, notes). The LAST device in the selection is treated as the
    head (llama.cpp places the output head on the last device in PCI order), so it
    is charged the larger head compute buffer.
    Returns (None, notes) if any card has no room at all - better to refuse loudly
    than to emit a split that cannot work."""
    notes, caps = [], []
    comp = _compute_buf_mib(ctx)
    for i, g in enumerate(sel):
        total_mib, used_mib = g[3], g[4]
        free_mib = total_mib - used_mib
        is_head = (i == len(sel) - 1)
        overhead = (TS_HEAD_COMPUTE_MIB if is_head else comp) + TS_CUDA_CTX_MIB + TS_HEADROOM_MIB
        caps.append(max(0.0, free_mib - overhead))
    if min(caps) <= 0:
        bad = [g[0] for g, c in zip(sel, caps) if c <= 0]
        notes.append(f"AUTO -ts DECLINED: card(s) {bad} have no capacity left after "
                     f"{TS_HEADROOM_MIB} MiB headroom + compute buffers at ctx={ctx}. "
                     f"Free a card, drop -c, or pass --ts by hand.")
        return None, notes
    tot = sum(caps)
    shares = [int(round(1000 * c / tot)) for c in caps]
    ts = ",".join(str(x) for x in shares)
    detail = "  ".join(f"{g[0]}:{c:.0f}MiB" for g, c in zip(sel, caps))
    notes.append(f"AUTO -ts {ts} [MEASURED method]: capacity-proportional over FREE VRAM after "
                 f"reserving {TS_HEADROOM_MIB} MiB decode headroom + {comp:.0f} MiB compute "
                 f"({TS_HEAD_COMPUTE_MIB} on the head card) + {TS_CUDA_CTX_MIB} MiB context "
                 f"per device. Capacities: {detail}. -ts partitions BYTES, not layers, and "
                 f"llama.cpp repacks when -ub changes - re-derive if you force a different -ub.")
    return ts, notes


def build_llama_cmd(plan, a, sel, prof, ctx, ub_expect, mmproj, explain=False):
    E, note = resolve_engine_dir()
    if E is None:
        print(f"  !! no WORKING llama-server found. {note}")
        if "NO CUDA RUNTIME ON THIS HOST" in note:
            print("     Every candidate build is FINE - this host just has no CUDA runtime.")
            print("     These builds are meant to run inside a CUDA container with the build")
            print("     bind-mounted. Run pxa-launch in there, or set PXA_ENGINE_DIR.")
        else:
            print("     Set PXA_ENGINE_DIR=/path/to/build (the dir containing bin/llama-server).")
        if not explain:
            sys.exit(4)
        # --explain still PRINTS the plan: hiding the command because the local host
        # lacks a CUDA runtime would hide the thing the operator asked to see. The
        # blocker rides the plan and --explain exits 5, so a CI caller can still tell
        # "clean plan" from "plan that will not start here".
        plan.blockers.append("no llama-server build starts on THIS host - the command below "
                             "names <ENGINE>/bin/llama-server, not a resolved binary")
        E = "<ENGINE>"
        note = "auto-detected"          # the failure is already printed above; do not repeat it
    if note != "auto-detected":
        print(f"  engine dir: {E}  [{note}]")

    ldp, dropped = engine_ld_path(E)
    if dropped:
        print(f"  LD_LIBRARY_PATH: DROPPED {dropped} - a CUDA stubs dir on the child's library "
              f"path makes ggml offload 0 layers and run ~50x slower while still producing "
              f"numerically correct output (PERPLEXITY-RESULTS.md:41-55).")

    R = plan.recipe
    cmd = [f"{E}/bin/llama-server", "-m", a.model, "--host", a.host, "--port", str(a.port),
           "-ngl", str(a.ngl), "-sm", a.sm, "-c", str(ctx),
           "-ctk", a.ctk, "-ctv", a.ctv, "-np", str(a.np),
           "-fa", plan.fa, "--cont-batching"]
    if getattr(a, "emit_threads", True):
        cmd += ["-t", str(a.threads)]
    else:
        print(f"  -t: NOT PASSED. The measured arms did not set it, so neither does this "
              f"launcher; llama.cpp picks its own (this host has {os.cpu_count()} cores). "
              f"Pass --threads N to force one.")
    print(f"  -fa {plan.fa}: the {'interactive/serving' if plan.fa == 'on' else 'cold-prefill'} "
          f"regime for --workload {a.workload or 'chat'}. One server carries ONE setting; "
          f"see the FA lines above for what the other one costs.")

    # ---- -b / -ub ---------------------------------------------------------
    # PRECEDENCE, and it is deliberate:
    #   1. --ub, if the operator forced one          (they own it, it is printed as forced)
    #   2. the matched RECIPE row's measured cell    (the whole point of the table)
    #   3. the qwen4exp ub1024 rule                  (measured, but off-topology)
    #   4. nothing at all -> the engine's adaptive-ub probes each device
    # The old file only ever did (4). That is correct where nothing was measured
    # and wrong everywhere a cell exists: adaptive-ub lands on 2048 for a 16 GiB
    # card, while the measured best on a 2x P100 pair is 256 - and it never sets
    # -b at all, because the prefill CHUNK is not its job. On both card pairs
    # -b 8192 is worth about +10% long-prompt prefill over the engine default
    # -b 2048 (the measurement ledger chunk-size PROBE).
    ub_auto = None
    if not a.ub and R is None and prof.get("arch") == "qwen4exp":
        ub_auto = 1024
    if a.ub:
        b_forced = a.b or a.ub
        cmd += ["-b", str(b_forced), "-ub", str(a.ub)]
        print(f"  -b {b_forced} -ub {a.ub} FORCED by --ub/--b. The card-type table expects "
              f"-ub {ub_expect} here"
              + (f" and the '{R.key}' row measured -b {R.b} -ub {R.ub}." if R else "."))
        print("  NOTE: -ub and -ts are coupled - llama.cpp folds a compute allowance into the "
              "-ts walk, so this -ub repacks the layers. If you also passed --ts, re-derive it.")
    elif R is not None:
        ub = R.ub
        if ub > ub_for_card(min(g[3] for g in sel)) and R.status != "MEASURED":
            # Only ever DOWNWARD, and only on an inferred row: an 11 GiB card
            # cannot allocate the ~1.9 GiB compute buffer a ub2048 cell needs, and
            # a row that was inferred rather than measured has no standing to
            # insist. A MEASURED row is emitted as measured, full stop.
            ub = ub_for_card(min(g[3] for g in sel))
            print(f"  -ub lowered from the row's {R.ub} to {ub}: the smallest card in this "
                  f"selection is {min(g[3] for g in sel)} MiB and the card-type table "
                  f"(LEVERS.md:99-103) caps it there. The row is [INFERRED] anyway, so it has "
                  f"no measured claim on {R.ub}.")
        cmd += ["-b", str(R.b), "-ub", str(ub)]
        tag = "MEASURED" if R.status == "MEASURED" else "[INFERRED]"
        print(f"  -b {R.b} -ub {ub} {tag} from recipe row '{R.key}' ({R.title}).")
        if R.status == "MEASURED":
            print(f"     {R.numbers}")
        print(f"     source: {R.source}")
    elif ub_auto:
        cmd += ["-b", str(ub_auto), "-ub", str(ub_auto)]
        print(f"  -ub {ub_auto} [MEASURED]: on this arch ub1024 is +20% prefill over the "
              f"adaptive 512 (337-366 -> 430-439 tok/s, six-card PXQ4, 2931-token prompt) "
              f"with decode UNCHANGED (30.6-30.9 vs 30.6-31.1, overlapping). The automatic "
              f"-ts below is derived from compute buffers measured at this same -ub.")
    else:
        print(f"  -b/-ub: NOT PASSED - no recipe row covers this topology, so the engine's "
              f"adaptive-ub probes each device at startup. Card-type table (LEVERS.md:99-103) "
              f"expects {ub_expect} on this selection. Verify against the server's own "
              f"'PXA posture: mode=... fa=... ub=...' line.")

    # ---- -ot (per-layer embedding to host RAM) ----------------------------
    # Not an optimisation: without it the gather table is offloaded with everything
    # else and the load dies with a cudaMalloc of the whole tensor on one card
    # (MEASURED 2026-08-28, 16089.57 MiB on device 2 of a 5-card Flash-Next seat).
    # The pattern is ANCHORED on the full name on purpose - a loose 'ple' regex also
    # matches blk.N.ple_key, ple_conv1d and the F32 ple_norm_* tensors, which are tiny
    # and MUST stay on the GPU.
    if R is not None and R.ot:
        cmd += ["-ot", R.ot]
        print(f"  -ot {R.ot} from recipe row '{R.key}' - the per-layer embedding table lives in "
              f"host RAM. It is GET_ROWS only (no GEMM); without this the load OOMs.")
    elif prof.get("ple_tensor"):
        cmd += ["-ot", r"per_layer_token_embd\.weight=CPU"]
        _e = prof.get("ple_elems")
        print("  -ot per_layer_token_embd -> CPU: this model carries a PLE gather table"
              + (f" of {_e/1e9:.1f}e9 elements" if _e else "") +
              ". It is GET_ROWS only (no GEMM), so host RAM costs a PCIe gather and frees "
              "a large fraction of the file from VRAM. Without this the load OOMs.")

    # ---- -ts --------------------------------------------------------------
    if a.ts:
        cmd += ["-ts", a.ts]
        print(f"  -ts {a.ts} FORCED by --ts; neither the recipe split nor the automatic "
              f"capacity split was used.")
    elif R is not None and R.ts:
        cmd += ["-ts", R.ts]
        print(f"  -ts {R.ts} from recipe row '{R.key}'. This is the split the row was MEASURED "
              f"with, so the automatic capacity split is NOT used - on the Flash-Next seat the "
              f"uneven slice is deliberate (card 0 shares this box with another server).")
    elif sel and len(sel) > 1:
        _ts, _tsnotes = auto_tensor_split(sel, ctx)
        for _n in _tsnotes:
            print("  " + _n)
        if _ts:
            cmd += ["-ts", _ts]

    # ---- the rest of the row's flags, then the operator's own -------------
    if R is not None and R.extra:
        cmd += list(R.extra)
        print(f"  extra flags from recipe row '{R.key}': {' '.join(R.extra)}")
    if a.spec:
        cmd += ["--spec-type", a.spec]
    if a.draft_model:
        cmd += ["-md", a.draft_model]
    if mmproj:
        cmd += ["--mmproj", mmproj]

    # ---- the correctness flags, from chat_defaults() ----------------------
    c = plan.chat
    if c is not None:
        if c.jinja:
            cmd += ["--jinja"]
        if c.template:
            cmd += ["--chat-template", c.template]
        if c.template_file:
            cmd += ["--chat-template-file", c.template_file]
        if c.reasoning_format:
            cmd += ["--reasoning-format", c.reasoning_format]
        if c.reasoning_budget is not None:
            cmd += ["--reasoning-budget", str(c.reasoning_budget)]
        for flag, val in (("--temp", c.temp), ("--top-p", c.top_p), ("--top-k", c.top_k),
                          ("--min-p", c.min_p), ("--repeat-penalty", c.repeat_penalty)):
            if val is not None:
                cmd += [flag, _fmtnum(val)]
        if c.slot_save_path:
            cmd += ["--slot-save-path", c.slot_save_path]
        if c.api_key:
            cmd += ["--api-key", c.api_key]

    # ---- ENV - every lever explicitly stated, on or off, with the reason ----
    # PXA_ENHANCE=1 is EXPORTED EXPLICITLY even where the engine is making it the
    # default level. A launcher that leans on an engine default cannot be read: the
    # same printed command would mean two different things on two builds, and a user
    # who copies it onto an older binary would get a different seat with no warning.
    # Exporting it is idempotent on a build that already defaults to it.
    env = {"PXA_ENHANCE": "1", "LD_LIBRARY_PATH": ldp}
    print("  PXA_ENHANCE=1: exported explicitly, not inherited. It auto-selects the "
          "measured-good kernel levers per device and prints its decision at startup "
          "(docs/COOKBOOK.md:24-39). On sm_61 it is what arms PXA_PXQ_INT8_PREFILL mode 1, "
          "without which the 1080 Ti numbers halve.")
    if R is not None and R.env:
        env.update(R.env)
        print(f"  {len(R.env)} PXA_* lever(s) from recipe row '{R.key}', copied verbatim from "
              f"{R.source.split(';')[0]}:")
        for k, v in R.env.items():
            print(f"      {k}={v}")
    if a.no_mmap:
        cmd += ["--no-mmap"]
        env["PXA_PARALLEL_LOAD"] = "1"   # -25..-46% cold load; INERT under mmap (one WARN)
    # GGML_CUDA_NO_PINNED is NOT emitted. It appears in NO measured recipe in this
    # corpus; a previous version of this file set it unconditionally and its effect
    # on the anchors is UNMEASURED (invariant I-11: never set a lever the anchor was
    # not measured with).
    return cmd, env


MEASURED_LADDER = [1, 2, 4, 8]      # MEASURED; the fix for the [1,2] cliff at 3+ (task #78)


def compilation_config(np_):
    """THE ONLY PLACE a vLLM --compilation-config is built. One construction site
    means the two correctness keys cannot be lost on some branch:

      custom_ops:["none"]  is MANDATORY wherever FULL_DECODE_ONLY is emitted on
        sm_60. Without it, PP=2+FDO is a HARD BOOT FAILURE - "CUDA error: an
        illegal memory access was encountered", Worker_PP1 ->
        determine_available_memory -> profile_run (MOE-CROSSOVER.md:271-292,
        container xover-vllm-boot1, ZERO tokens produced). Adding the key fixed it
        with no other change. Every working recipe on this box carries it
        (BUILD-RECIPE.md:95, graphs_arms.sh:6, live fat-smoke-588671). The previous
        version of this file carried it NOWHERE while emitting FDO - i.e. it
        emitted exactly the command proven twice not to boot.

      cudagraph_mode FULL_DECODE_ONLY is a CORRECTNESS requirement, not a knob.
        The vLLM default also captures PREFILL graphs; a raw /v1/completions
        prompt short enough to fit one prefills through a captured graph holding
        stale data and returns fluent garbage from character zero. Chat traffic
        hides it because the template pads past the captured sizes - which is why
        arithmetic gates stayed green while the bug was live. MEASURED: the broken
        config's best aggregate 88.4 (SCOREBOARD M8) is BELOW the correct config's
        88.7 (M7). There is no speed argument for it.
    """
    ladder = list(MEASURED_LADDER)
    while ladder[-1] < np_:
        ladder.append(ladder[-1] * 2)
    return {"custom_ops": ["none"],
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": ladder}


def build_vllm_cmd(plan, a, prof, ctx, used, image):
    cc = sorted({g[2] for g in used})[0]
    backend, backend_ev = ATTN_BACKEND.get(cc, (None, None))
    model = a.model
    deg = 1
    while deg * 2 <= len(used):
        deg *= 2
    if deg != len(used):
        # H6: vLLM rejects a non-power-of-two degree at startup. Truncating is a
        # REAL change to the run, so it is announced and it needs an ack.
        print(f"  parallel degree truncated to {deg} (power of two) from {len(used)} eligible "
              f"cards; card(s) {[g[0] for g in used[deg:]]} will NOT be used.")
        used = used[:deg]

    # `vllm serve` is correct for a self-contained image. It is WRONG for one whose
    # python lives on the host: vllm is not on PATH in pxa-sm60-dev, so the emitted
    # command could not start, and in non-explain mode this file exits 4 on the
    # shutil.which check - after printing a full, confident plan. An image that declares
    # a host python gets invoked through it, the same way the boot that produced the
    # measurements did.
    hostenv = (VLLM_IMAGES.get(image) or {}).get("host_env") or {}
    hpy = hostenv.get("python")
    if hpy:
        # The model path has to be translated through the same mount the interpreter
        # came through. This launcher runs INSIDE the container, where the host root does
        # not exist - only /c does. Emitting the host path produces a command that is
        # correct on paper and cannot find its weights, which surfaces as a load failure
        # minutes in rather than as a sentence here.
        for hsrc, hdst in sorted(hostenv.get("mounts", {}).items(),
                                 key=lambda kv: -len(kv[0])):
            if model.startswith(hsrc.rstrip("/") + "/"):
                mapped = hdst.rstrip("/") + model[len(hsrc.rstrip("/")):]
                print(f"  model path mapped for {image}: {model} -> {mapped} "
                      f"(via -v {hsrc}:{hdst})")
                model = mapped
                break
        else:
            if hostenv.get("mounts"):
                print(f"  ** {model} is OUTSIDE every mount {image} declares "
                      f"({', '.join(hostenv['mounts'])}). The command below names a path "
                      f"that will not exist in that container - mount it, or serve a model "
                      f"that is under one of those roots.")
        cmd = [hpy, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
               "--host", a.host, "--port", str(a.port)]
    else:
        cmd = ["vllm", "serve", model, "--host", a.host, "--port", str(a.port)]
    cmd += ["--quantization", "pxq4", "--dtype", "float16",
            "--max-model-len", str(ctx), "--max-num-seqs", str(a.np),
            "--gpu-memory-utilization", str(a.gmu)]
    # MULTIMODAL ENCODER BUDGET ON A SMALL CARD (pxq23, measured 2026-09-05).
    # A vision-capable architecture makes vLLM reserve an encoder cache (16384 tokens) and
    # profile the vision tower at its maximum feature size. On a 16 GiB card holding a 12.23
    # GiB model that budget is the difference between a working server and
    #   "0.22 GiB KV cache is needed, which is larger than the available KV cache memory
    #    (0.0 GiB)"
    # -- the engine refuses to start, and the message points at gpu_memory_utilization, which
    # is not where the memory went. Text-only serving of a multimodal checkpoint is the common
    # case here (the PXQ conversions borrow their vision tower from a reference checkpoint and
    # are gated text-only), so it is the DEFAULT, and it is announced rather than silent.
    # --vllm-mm restores image and video input.
    if _is_multimodal(model) and not getattr(a, "vllm_mm", False):
        cmd += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
        print("  --limit-mm-per-prompt {\"image\":0,\"video\":0}: this checkpoint declares a "
              "vision tower, and the encoder cache plus vision profiling it would reserve "
              "leaves ~0 GiB for KV on a 16 GiB card. Pass --vllm-mm if you actually need "
              "image or video input, and expect to lower --max-model-len to pay for it.")
    if backend:
        cmd += ["--attention-backend", backend]
        print(f"  --attention-backend {backend}: {backend_ev}")

    # KV BLOCK SIZE ON VOLTA FOR A GDN HYBRID (board A4/A6, measured 2026-09-06).
    # The v1.5.0 sm_70 image serves a DeltaNet/GDN hybrid at parity with the image it
    # replaces ONLY at --block-size 256: prefill 1002.85 @3k and 993.84 @20k against
    # 1010.0 / 987.3, decode 49.73 vs 49.71, agg@8 178.23 vs 178.33, and the same
    # byte-identical greedy continuation (sha c220beafa2d5) twelve times at np=1. The
    # live Volta seat runs with it. Gated on cc == 70 AND a GDN hybrid: on sm_60 this
    # was never measured and must not be assumed, and on a plain dense model the block
    # size is not what the measurement was about.
    bs = str(getattr(a, "vllm_block_size", "") or "")
    if bs == "0":
        print("  --block-size: suppressed by --vllm-block-size 0; vLLM picks its own.")
    elif bs:
        cmd += ["--block-size", bs]
        print(f"  --block-size {bs}: yours, via --vllm-block-size. The measured default on "
              f"this topology is 256 for a GDN hybrid on sm_70.")
    elif cc == 70 and prof.get("deltanet"):
        cmd += ["--block-size", "256"]
        print("  --block-size 256: MEASURED default for a GDN hybrid on sm_70. The rebased "
              "sm_70 image reaches parity with the image it replaces at this block size "
              "(prefill 1,002.85 / 993.84 t/s, decode 49.73, agg@8 178.23) and is "
              "byte-identical on the determinism gate. Pass --vllm-block-size to override.")
    # H3: no P2P on this box -> custom all-reduce OFF in every measured vLLM arm
    # (MOE-CROSSOVER.md:79). CAR costs ~18% vs NCCL on MoE while the CAR kernel
    # itself is exonerated. Read from topology, not hardcoded.
    if not plan_has_p2p():
        cmd += ["--disable-custom-all-reduce"]

    # MoE goes PIPELINE parallel; dense goes TENSOR parallel. PP=2 is the arm that
    # holds every MoE number in the table (MOE-CROSSOVER.md arm B).
    if prof.get("is_moe") and deg >= 2:
        cmd += ["--pipeline-parallel-size", str(deg), "--tensor-parallel-size", "1"]
        print(f"  MoE -> PIPELINE parallel (PP={deg}, TP=1). MEASURED: PP=2 + FULL_DECODE_ONLY "
              f"is the arm that produced every vLLM MoE cell in the table above.")
    else:
        cmd += ["--tensor-parallel-size", str(max(1, deg))]

    # THE COMPILATION CONFIG - three load-bearing keys.
    #
    # custom_ops:["none"] - I-3, and the single most dangerous omission in the
    #   previous version of this file. PP=2 + FDO WITHOUT this key is a HARD BOOT
    #   FAILURE: "RuntimeError: Worker failed with error 'CUDA error: an illegal
    #   memory access was encountered'", Worker_PP1 -> determine_available_memory
    #   -> profile_run (MOE-CROSSOVER.md:271-292, container xover-vllm-boot1,
    #   produced ZERO tokens). Adding the key fixed it with no other change. Every
    #   working recipe on this box carries it (BUILD-RECIPE.md:95, graphs_arms.sh:6,
    #   live container fat-smoke-588671). `grep -n custom_ops` on the old file
    #   returned 0 hits while it emitted FDO - i.e. it emitted exactly the command
    #   proven twice not to boot.
    #   Scope: MEASURED-REQUIRED on sm_60 PP>=2. MEASURED-present-and-healthy on
    #   sm_60 TP=2. On sm_70 its necessity is UNMEASURED - emitted anyway, which is
    #   the safe direction, and labelled [INFERRED] here.
    #
    # cudagraph_mode:FULL_DECODE_ONLY - I-1/I-2, a CORRECTNESS requirement, never a
    #   tuning knob. The default (FULL_AND_PIECEWISE) also captures PREFILL graphs
    #   at the ladder sizes; a raw /v1/completions prompt short enough to fit one
    #   prefills through a captured graph whose input buffer holds stale data and
    #   returns fluent garbage from character zero. Chat traffic never shows it
    #   because the chat template pads every prompt past the captured sizes - which
    #   is exactly why arithmetic gates stayed green while the bug was live.
    #   MEASURED: the broken config's BEST aggregate (88.4, SCOREBOARD M8) is BELOW
    #   the correct config's (88.7, M7). SPEC CORRECTION C2: the 22.3 -> 24.0
    #   single-stream pair that used to be quoted here as "dense" is a MoE pair
    #   (SCOREBOARD M8/M9, both in table 1a; ENGINE-VERDICT.md:14-22 calls it "the
    #   MoE seat"). Dense TP=2 single is D3 = 24.01 and has NO FAP arm at all.
    #
    # cudagraph_capture_sizes - powers of two covering --max-num-seqs. [1,2,4,8] is
    #   the MEASURED ladder and the fix for the earlier [1,2] cliff at 3+ concurrent
    #   (task #78). Widening past 8 is [INFERRED]: the precedent says a too-short
    #   ladder cliffs, but no np>8 ladder was ever measured.
    cc_obj = compilation_config(a.np)
    if cc_obj["cudagraph_capture_sizes"] != MEASURED_LADDER:
        print(f"  cudagraph_capture_sizes widened to {cc_obj['cudagraph_capture_sizes']} for "
              f"np={a.np}. [INFERRED] - the measured ladder is {MEASURED_LADDER}; nothing above "
              f"np=8 was measured.")
    cmd += ["--compilation-config", json.dumps(cc_obj)]
    if cc != 60:
        print("  custom_ops:[\"none\"] is emitted on sm_%d as well: MEASURED-required on sm_60 "
              "PP>=2, UNMEASURED on sm_70. Emitting it is the safe direction. [INFERRED]" % cc)

    # --speculative-config: ONLY for an explicitly requested ngram spec, with the
    # REQUESTED values. The previous version replaced ANY --spec (including
    # mtp:n_max=1) with a hardcoded {"method":"ngram","num_speculative_tokens":4} -
    # silently substituting a mechanism whose verdict is the OPPOSITE SIGN on this
    # model class (ngram +23.0% code vs MTP -8.6%/-29.8%, LEVERS.md:746).
    method, params = parse_spec(a.spec)
    if method:
        n = int(params.get("n_max", 4))
        cmd += ["--speculative-config",
                json.dumps({"method": "ngram", "num_speculative_tokens": n})]
        print(f"  --speculative-config: ngram, num_speculative_tokens={n} (YOUR values, not a "
              f"substitution).")

    env = {"TORCHDYNAMO_DISABLE": "1",           # MEASURED arm B env, MOE-CROSSOVER.md:80
           "VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}  # MEASURED arm B env, MOE-CROSSOVER.md:80
    # An image whose runtime lives on the host also carries the env that makes that
    # runtime importable. setdefault, not assignment: an explicit value already in the
    # measured arm above wins over the image's default.
    for _k, _v in hostenv.get("env", {}).items():
        env.setdefault(_k, _v)
    if cc == 70:
        # The sm_70 recipes carry this; the sm_60 arm does NOT - it carries
        # SITE=<site> LIB=libpxq4_sm60_v10.so PACKED=1 instead (MOE-CROSSOVER.md:78).
        # The old file emitted the Volta backend name unconditionally, i.e. into a
        # Pascal image - a config the corpus never ran.
        env["VLLM_SM70_QUANT_BACKEND"] = "turbomind"
    else:
        print("  sm_60 arm: the MEASURED recipe carries SITE=<site> LIB=libpxq4_sm60_v10.so "
              "PACKED=1 (MOE-CROSSOVER.md:78). Those are IMAGE-INTERNAL paths - this launcher "
              "will not fabricate them. Declare them in your container invocation or the run is "
              "off the measured envelope (I-11).")
    # VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE is NEVER set: it crash-loops
    # the container at warmup, 3/3 boots (TypeError: 'NoneType' object is not
    # subscriptable in compile_or_warm_up_model -> _dummy_run) and left a seat in a
    # --restart unless-stopped loop needing manual clearing (DGX-FDO-FAILURE.md).
    return cmd, env, used


_P2P = [None]


def plan_has_p2p():
    if _P2P[0] is None:
        ok, _ = peer_topology()
        _P2P[0] = bool(ok)
    return _P2P[0]


# ---------------------------------------------------------------------------
# POST-BOOT VERIFICATION CONTRACT (I-10 / R-09)
# ---------------------------------------------------------------------------
def print_post_boot_contract(engine, cv):
    """Every claim made at plan time needs a matching observation at run time, or
    it is not made. This process EXECs the server, so it cannot observe anything
    after the exec - therefore it makes NO health claim at all, and prints the
    checks whoever owns the seat must run. Passing a flag is not evidence the flag
    took effect: on kewaii/vllm:latest, FULL_DECODE_ONLY parses, boots healthy and
    is SILENTLY OVERRIDDEN back to FULL_AND_PIECEWISE by that image's own compile
    policy (DGX-FDO-FAILURE.md). That is the whole failure class this project keeps
    paying for."""
    print("  POST-BOOT CONTRACT - NOT PERFORMED BY THIS PROCESS (it execs the server):")
    print("    this launcher makes NO healthy/unhealthy claim about the resulting seat.")
    if engine == "vllm":
        print("    1. capture mode: grep the server log for the INSTALLED cudagraph mode.")
        print("       If it is not FULL_DECODE_ONLY, the seat is NOT healthy - shut it down")
        print("       (R-09). A derived image is the fix, not a flag.")
        print("    2. N-way split: per-device resident bytes. MEASURED PP=2 MoE = 10.71 GiB/rank")
        print("       (ENGINE-VERDICT.md section 4).")
    else:
        print("    1. posture: llama-server logs 'PXA posture: mode=... fa=... ub=...' at")
        print("       startup - compare ub against the card-type expectation printed above.")
        print("    2. offload: confirm 'offloaded N/N layers to GPU'. A stub libcuda on the")
        print("       library path yields 0/N, correct output and ~50x slower")
        print("       (PERPLEXITY-RESULTS.md:41-55).")
    print(f"    3. device scoping: echo the child's CUDA_VISIBLE_DEVICES back; expect {cv!r}.")
    print("    4. short-prompt correctness: a RAW, NON-chat-templated 1-token and 5-token")
    print("       completion BEFORE any number is trusted, exactly as all 11 crossover boots")
    print("       did (MOE-CROSSOVER.md section 4.3). Chat-templated traffic pads every prompt")
    print("       past the captured sizes, which is precisely why the FAP corruption survived")
    print("       arithmetic gating.")
    print("    5. speculation: if you armed one, the acceptance-rate line must be present and")
    print("       non-zero. If it is absent, DROP THE CLAIM and keep serving.")


# ---------------------------------------------------------------------------
# THE FOR-DUMMIES FRONT DOOR: pick cards by number, pick a model by number
# ---------------------------------------------------------------------------
# DESIGN RULE, unchanged: never magic. The picker does not decide anything. It
# only turns "which of these" into "--gpus 2,4 --model <path>", and every value it
# shows is read from the box or from the file, never assumed.
STATE_DIR = os.environ.get("PXA_LAUNCH_STATE",
                           os.path.join(os.path.expanduser("~"), ".cache", "pxa-launch"))
STATE_FILE = os.path.join(STATE_DIR, "state.json")
SCAN_CACHE = os.path.join(STATE_DIR, "modelscan.json")

# Roots searched when the user names none. Deliberately SHORT and portable: this
# file ships in a public repo and a box-specific path list would be a lie
# everywhere else. The launcher PRINTS the roots it searched, so "it found
# nothing" is always followed by "here is where to point it".
DEFAULT_MODEL_ROOTS = ["./models", "~/models", "/models"]
MODEL_SCAN_MAX_DEPTH = 4        # deep enough for <root>/<family>/<file>.gguf trees
MODEL_SCAN_MAX_FILES = 4000     # a guard, not a policy; the count is printed if hit


def _load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(d):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass            # a launcher must never fail because it could not remember


def human_bytes(n):
    if not n:
        return "?"
    if n >= BYTES_PER_GIB:
        return f"{n / BYTES_PER_GIB:.1f} GiB"
    return f"{n / (1024 * 1024):.0f} MiB"


def model_roots(cli_dirs):
    """-> (roots[], trail[]). Resolution order, each source named in the trail so a
    user who sees an empty list knows exactly which knob to turn."""
    roots, trail, seen = [], [], set()

    def add(p, why):
        p = os.path.abspath(os.path.expanduser(p))
        if p in seen:
            return
        seen.add(p)
        if os.path.isdir(p):
            roots.append(p)
            trail.append(f"{p}  [{why}]")
        else:
            trail.append(f"{p}  [{why}] - does not exist, skipped")

    for d in (cli_dirs or []):
        add(d, "--models-dir")
    env = os.environ.get("PXA_MODELS_DIR", "")
    for d in [x for x in env.split(os.pathsep) if x.strip()]:
        add(d, "PXA_MODELS_DIR")
    last = _load_state().get("last_model_dir")
    if last:
        add(last, "directory of your previous launch")
    for d in DEFAULT_MODEL_ROOTS:
        add(d, "built-in default")
    return roots, trail


def _scan_cache_load():
    try:
        with open(SCAN_CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def _scan_cache_save(c):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(SCAN_CACHE, "w") as f:
            json.dump(c, f)
    except Exception:
        pass


SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def describe_gguf(path, cache):
    """One line of facts about a GGUF, read from its own header. Cached on
    (size, mtime) because the interactive list reads every candidate and a header
    walk over a hundred files should not cost a hundred seconds twice."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = f"v3|{path}|{st.st_size}|{int(st.st_mtime)}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    prof = model_profile(path, "gguf")
    d = {"path": path, "size": st.st_size, "blocks": prof.get("n_block", 0),
         "family": model_family(prof), "tier": prof.get("tier"),
         "tier_kv": prof.get("tier_kv"), "arch": prof.get("arch"),
         "n_expert": prof.get("n_expert", 0), "n_ctx_train": prof.get("n_ctx_train"),
         "vision": bool(prof.get("vision")), "ple": bool(prof.get("ple_tensor")),
         "ple_bytes": prof.get("ple_bytes") or 0,
         "kv_bytes_tok": prof.get("kv_bytes_tok") or 0,
         "template": bool(prof.get("chat_template")),
         "err": prof.get("hdr_err")}
    cache[key] = d
    return d


def scan_models(roots):
    """-> (entries[], notes[]). Every .gguf under the roots, first shard only,
    each described from its own header."""
    cache, entries, notes, seen = _scan_cache_load(), [], [], set()
    n_files = 0
    skipped = 0
    for root in roots:
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count("/") - base_depth >= MODEL_SCAN_MAX_DEPTH:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in sorted(filenames):
                if not fn.endswith(".gguf"):
                    continue
                m = SHARD_RE.search(fn)
                if m and m.group(1) != "00001":
                    continue            # list the first shard only; -m takes that one
                if fn.startswith("mmproj"):
                    continue            # a projector is an accessory, not a model
                p = os.path.join(dirpath, fn)
                real = os.path.realpath(p)
                if real in seen:
                    continue
                seen.add(real)
                n_files += 1
                if n_files > MODEL_SCAN_MAX_FILES:
                    notes.append(f"stopped after {MODEL_SCAN_MAX_FILES} files - narrow the "
                                 f"search with --models-dir or PXA_MODELS_DIR")
                    break
                d = describe_gguf(p, cache)
                if d and not d.get("blocks"):
                    # No blk.* tensors: a tokenizer fixture or a projector-only
                    # file, not something you can serve. Counted and reported
                    # rather than silently dropped.
                    skipped += 1
                    continue
                if d:
                    entries.append(d)
            if n_files > MODEL_SCAN_MAX_FILES:
                break
    _scan_cache_save(cache)
    if skipped:
        notes.append(f"{skipped} .gguf file(s) with no transformer blocks were left out "
                     f"(tokenizer fixtures, projector-only files)")
    entries.sort(key=lambda e: (e["family"] != "hybrid-moe", -(e["size"] or 0), e["path"]))
    return entries, notes


def family_label(e):
    f = e.get("family", "dense")
    if f == "hybrid-moe":
        return f"hybrid-MoE ({e.get('n_expert') or '?'}e)"
    if f == "moe":
        return f"MoE ({e.get('n_expert') or '?'}e)"
    if f == "hybrid":
        return "hybrid (SSM)"
    return "dense"


def _ask(prompt, default=None):
    try:
        s = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(130)
    return s or (default or "")


def interactive_available():
    return sys.stdin.isatty() and sys.stdout.isatty()


def print_card_menu(gpus, procs):
    print("STEP 1 of 4 - which card(s) do you want to use?")
    print()
    print("   #   gpu  name                            VRAM        in use by another process?")
    for n, g in enumerate(gpus, 1):
        idx, name, cc, tot, used, _uuid = g
        busy = procs.get(idx) or []
        if busy:
            who = ", ".join(f"pid {p} {nm} ({mb} MiB)" for p, nm, mb in busy)
            st = f"YES - {who}"
        elif used > 512:
            st = f"YES - {used} MiB resident, no compute app listed"
        else:
            st = "free"
        print(f"  {n:>2}   {idx:>3}  {name.replace('NVIDIA ', ''):<30}  "
              f"{tot / 1024:>5.1f} GiB  sm_{cc}   {st}")
    print()


def pick_cards(gpus, procs):
    print_card_menu(gpus, procs)
    print("  Type the numbers from the # column, separated by commas.")
    print("  Two cards of the SAME type is the best-measured shape; four P100s is the")
    print("  Flash-Next seat. Press Enter for every card on the box.")
    while True:
        s = _ask("  cards> ")
        if not s:
            return [g[0] for g in gpus]
        try:
            nums = [int(x) for x in re.split(r"[,\s]+", s) if x]
        except ValueError:
            print("  ! numbers only, e.g.  3,5")
            continue
        if any(n < 1 or n > len(gpus) for n in nums):
            print(f"  ! pick between 1 and {len(gpus)}")
            continue
        picked = [gpus[n - 1][0] for n in nums]
        if len(set(picked)) != len(picked):
            print("  ! the same card twice")
            continue
        return picked


def print_model_menu(entries, trail, notes):
    print("STEP 2 of 4 - which model?")
    print()
    if not entries:
        print("  No .gguf files were found. Searched:")
        for t in trail:
            print(f"    {t}")
        for n in notes:
            print(f"    note: {n}")
        print()
        print("  Point the launcher at your models and run it again, either way round:")
        print("    tools/pxa-launch.py --models-dir /path/to/models")
        print("    export PXA_MODELS_DIR=/path/to/models:/another/path")
        return
    print("   #  file                                          size      family        PXQ tier")
    for n, e in enumerate(entries, 1):
        nm = os.path.basename(e["path"])
        if len(nm) > 44:
            nm = nm[:41] + "..."
        tier = e.get("tier") or ("-" if not e.get("err") else "UNREADABLE")
        extra = ""
        if e.get("ple"):
            extra += " PLE"
        if e.get("vision"):
            extra += " vision"
        print(f"  {n:>2}  {nm:<44}  {human_bytes(e['size']):>9}  {family_label(e):<13} "
              f"{tier}{extra}")
    print()
    print("  Searched: " + ", ".join(t.split("  [")[0] for t in trail if "skipped" not in t))
    for n in notes:
        print(f"  note: {n}")
    print()


def pick_model(entries, trail, notes):
    print_model_menu(entries, trail, notes)
    if not entries:
        sys.exit(2)
    print("  Type a number from the # column, or paste a full path to a file this list missed.")
    while True:
        s = _ask("  model> ")
        if not s:
            print("  ! pick one")
            continue
        if s.isdigit():
            n = int(s)
            if 1 <= n <= len(entries):
                return entries[n - 1]["path"]
            print(f"  ! pick between 1 and {len(entries)}")
            continue
        p = os.path.abspath(os.path.expanduser(s))
        if os.path.exists(p):
            return p
        print(f"  ! {p} does not exist")


def print_chat_summary(c, prof, indent="  "):
    """The settings that make a seat CORRECT, each with where it came from."""
    tmpl = prof.get("chat_template")
    print(f"{indent}chat template in the file: "
          + (f"{len(tmpl)} chars, looks like "
             f"'{prof.get('chat_template_family') or 'an unrecognised family'}'"
             if tmpl else "NONE"))
    for ln in template_preview(tmpl):
        print(f"{indent}  | {ln}")
    print()
    print(f"{indent}{'setting':<20} {'value':<44} {'source'}")
    for label, value, tag, _why in c.rows:
        v = value if len(str(value)) <= 44 else str(value)[:41] + "..."
        print(f"{indent}{label:<20} {v:<44} {tag}")
    for w in c.warnings:
        print(f"{indent}!! {w}")


def pick_chat_options(prof, a, engine_dir=None):
    """STEP 4: show what the launcher read out of the file, let the user change the
    handful of things that bite. Editing is opt-in; the default path is Enter."""
    print("STEP 4 of 4 - chat template, jinja, reasoning, sampling")
    print()
    c = chat_defaults(prof, a, engine_dir)
    print_chat_summary(c, prof)
    print()
    print("  These are the settings people forget. Press Enter to take them as they are,")
    print("  or type the letter of one to change it:")
    print("    t  chat template      j  jinja on/off        r  reasoning format")
    print("    b  reasoning budget   s  sampling            k  API key / host / port")
    while True:
        ch = _ask("  edit [Enter = accept]> ").lower()
        if not ch:
            return
        if ch == "t":
            names, prov = builtin_chat_templates(engine_dir)
            print(f"  built-in names ({prov}):")
            for i in range(0, len(names), 6):
                print("     " + "  ".join(f"{n:<18}" for n in names[i:i + 6]))
            v = _ask("  template name, a path to a .jinja file, or Enter to keep the "
                     "file's own> ")
            if v.endswith(".jinja") or os.path.sep in v:
                a.chat_template_file, a.chat_template = v, ""
            elif v:
                a.chat_template, a.chat_template_file = v, ""
            else:
                a.chat_template = a.chat_template_file = ""
        elif ch == "j":
            a.no_jinja = not a.no_jinja
            print(f"  --jinja is now {'OFF' if a.no_jinja else 'ON'}")
        elif ch == "r":
            v = _ask(f"  one of {'/'.join(REASONING_FORMATS)}, or Enter for the engine "
                     f"default> ")
            a.reasoning_format = v if v in REASONING_FORMATS else ""
        elif ch == "b":
            v = _ask("  budget in tokens (0 = no thinking, -1 = unlimited, Enter = unset)> ")
            a.reasoning_budget = int(v) if re.fullmatch(r"-?\d+", v or "") else None
        elif ch == "s":
            for name in ("temp", "top_p", "top_k", "min_p", "repeat_penalty"):
                cur = getattr(a, name)
                v = _ask(f"  {name} [{'unset' if cur is None else cur}]> ")
                if v:
                    try:
                        setattr(a, name, int(v) if name == "top_k" else float(v))
                    except ValueError:
                        print("  ! a number, or Enter to leave it")
        elif ch == "k":
            a.api_key = _ask("  API key (Enter = none, the port stays open)> ") or ""
            a.host = _ask(f"  host [{a.host}]> ", a.host)
            v = _ask(f"  port [{a.port}]> ", str(a.port))
            if v.isdigit():
                a.port = int(v)
        else:
            print("  ! t, j, r, b, s, k, or Enter")
            continue
        c = chat_defaults(prof, a, engine_dir)
        print()
        print_chat_summary(c, prof)
        print()


WORKLOAD_MENU = [
    ("chat", "chat / agent - you type, it answers", "-fa on: full decode speed"),
    ("serve", "serve - several people or agents at once", "-fa on, --np above 1"),
    ("longdoc", "long documents - ingest, summarize, embed in bulk",
     "-fa off: much faster cold prefill, much slower decode"),
]


def pick_workload():
    print("STEP 3 of 4 - what will you use it for?")
    print()
    for n, (key, what, why) in enumerate(WORKLOAD_MENU, 1):
        print(f"  {n:>2}  {what:<48} {why}")
    print()
    print("  This picks the flash-attention regime, and it is a real fork on these cards:")
    print("  FA is a decode win and a cold-prefill loss (docs/COOKBOOK.md 'Two FA regimes').")
    while True:
        s = _ask("  use [1]> ", "1")
        if s.isdigit() and 1 <= int(s) <= len(WORKLOAD_MENU):
            return WORKLOAD_MENU[int(s) - 1][0]
        print(f"  ! pick between 1 and {len(WORKLOAD_MENU)}")


def confirm(question):
    while True:
        s = _ask(f"  {question} [Y/n] ", "y").lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# --serve-name: a restart script, no systemd, no daemon, no hidden state
# ---------------------------------------------------------------------------
def write_serve_script(name, cmd, env, cv, serve_dir):
    """Writes an executable shell script that re-runs EXACTLY this seat. It is the
    printed command, frozen - not a call back into this launcher, which would be
    free to decide differently tomorrow on a box whose cards moved."""
    d = os.path.abspath(os.path.expanduser(
        serve_dir or os.environ.get("PXA_SERVE_DIR")
        or os.path.join(STATE_DIR, "serve")))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.sh")
    q = _shquote
    lines = ["#!/bin/sh",
             f"# pxa-launch seat '{name}' - written {_now()} by tools/pxa-launch.py.",
             "#",
             "# This is the exact command the launcher chose, frozen. Re-run it to bring the",
             "# seat back up. It does NOT call the launcher again: a launcher is free to",
             "# decide differently tomorrow (cards move, another process takes VRAM), and a",
             "# restart script that quietly changes the seat is worse than no script.",
             "# To get a NEW decision, run tools/pxa-launch.py again.",
             "set -eu",
             "",
             "# Refuse to start on top of a card that something else is already using -",
             "# the same check the launcher does (R-20).",
             f"CARDS={q(cv)}",
             'if command -v nvidia-smi >/dev/null 2>&1; then',
             '  for i in $(echo "$CARDS" | tr "," " "); do',
             '    used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits '
             '2>/dev/null || echo 0)',
             '    if [ "${used:-0}" -gt 512 ]; then',
             '      echo "pxa-serve: card $i already has ${used} MiB resident. Free it, or edit '
             'this script." >&2',
             '      exit 2',
             '    fi',
             '  done',
             'fi',
             "",
             f"export CUDA_VISIBLE_DEVICES={q(cv)}",
             f"export NVIDIA_VISIBLE_DEVICES={q(cv)}"]
    for k, v in env.items():
        lines.append(f"export {k}={q(str(v))}")
    # The API key is NEVER written into the script. A restart script is the file
    # most likely to be pasted into a chat, committed, or copied to another box.
    out = list(cmd)
    keyed = False
    if "--api-key" in out:
        i = out.index("--api-key")
        out[i + 1] = "${PXA_API_KEY:?set PXA_API_KEY before running this script}"
        keyed = True
    if keyed:
        lines += ["",
                  "# The API key is deliberately NOT in this file. Export PXA_API_KEY first.",
                  "#   PXA_API_KEY=... " + os.path.basename(path)]
    lines += ["", "exec " + " ".join(
        c if c.startswith("${PXA_API_KEY") else q(c) for c in out), ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)
    return path


def _shquote(s):
    s = str(s)
    if s and re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def _now():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# ---------------------------------------------------------------------------
# THE THINGS PEOPLE FORGET: chat template, jinja, reasoning, sampling, keys
# ---------------------------------------------------------------------------
# These are not performance settings. They are the settings that make a seat
# CORRECT, and they are the ones that get skipped because the server starts
# happily without them and only misbehaves later: a missing --jinja turns every
# request carrying `tools` into an HTTP 500 while plain chat keeps working, so the
# seat looks healthy while its main job is broken. That exact failure ran in
# production on this box until it was found by hand
# (/usr/local/bin/pxa-seats-flashnext.sh, the "WHY THIS EXISTS" header).
#
# Everything here is READ from the model file where the model file knows, and
# labelled with where it came from where it does not.

# The engine's own alias list. Resolution order, and the probe that answered is
# always named - the point is that this list cannot go stale against the build
# being launched.
_STATIC_CHAT_TEMPLATES = [
    "chatml", "llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip",
    "mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7", "phi3",
    "falcon3", "falcon_e", "zephyr", "monarch", "gemma", "orion", "openchat",
    "vicuna", "vicuna-orca", "deepseek", "deepseek2", "deepseek3", "command-r",
    "llama3", "chatglm3", "chatglm4", "minicpm", "exaone3", "rwkv-world",
    "granite", "gigachat", "megrez", "llama4", "hunyuan-moe", "kimi-k2",
    "gpt-oss", "bitnet", "grok-2", "bailing", "bailing-think", "bailing2",
    "seed_oss",
]
_TEMPLATE_MAP_RE = re.compile(r'\{\s*"([A-Za-z0-9_.-]+)"\s*,\s*LLM_CHAT_TEMPLATE_')


def builtin_chat_templates(engine_dir=None):
    """-> (names[], provenance). Never guesses silently: the returned provenance
    string says which of the three probes answered."""
    if engine_dir:
        exe = os.path.join(engine_dir, "bin", "llama-server")
        if os.path.isfile(exe):
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"], _ = engine_ld_path(engine_dir)
            try:
                r = subprocess.run([exe, "--help"], capture_output=True, text=True,
                                   timeout=30, env=env)
                blob = (r.stdout or "") + (r.stderr or "")
                m = re.search(r"built-?in templates?:?\s*(.+)", blob, re.I)
                if m:
                    names = [x.strip() for x in re.split(r"[,\s]+", m.group(1)) if x.strip()]
                    if len(names) > 5:
                        return names, "probe 1: the engine's own --help output"
            except Exception:
                pass
    # The source of truth for the map, in the SAME tree this launcher ships in, so
    # a fork that adds a template gets it here without editing this file.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(here, "src", "llama.cpp")
    try:
        with open(src, "r", errors="replace") as f:
            blob = f.read()
        i = blob.find("LLM_CHAT_TEMPLATES = {")
        if i >= 0:
            names = _TEMPLATE_MAP_RE.findall(blob[i:i + 8000])
            if names:
                return names, f"probe 2: the LLM_CHAT_TEMPLATES map in {src}"
    except Exception:
        pass
    return list(_STATIC_CHAT_TEMPLATES), ("probe 3: this file's static copy of the map - "
                                          "the engine was not runnable and src/llama.cpp was "
                                          "not readable, so this list may be STALE")


# Template fingerprints, keyed to the engine's OWN alias names so that what the
# launcher prints is what --chat-template would accept. Ordered: first match wins,
# most specific first. Mirrors src/llama.cpp's llama_chat_detect_template.
_TEMPLATE_FINGERPRINTS = [
    ("gpt-oss",  lambda t: "<|start|>" in t and "<|channel|>" in t),
    ("llama4",   lambda t: "<|header_start|>" in t and "<|header_end|>" in t),
    ("llama3",   lambda t: "<|start_header_id|>" in t and "<|end_header_id|>" in t),
    ("granite",  lambda t: "<|start_of_role|>" in t),
    ("gemma",    lambda t: "<start_of_turn>" in t),
    ("command-r", lambda t: "<|START_OF_TURN_TOKEN|>" in t and "<|USER_TOKEN|>" in t),
    ("chatglm4", lambda t: "[gMASK]<sop>" in t),
    ("chatglm3", lambda t: "[gMASK]sop" in t),
    ("kimi-k2",  lambda t: "<|im_middle|>" in t and "<|im_end|>" in t),
    ("chatml",   lambda t: "<|im_start|>" in t),
    ("mistral-v7", lambda t: "[SYSTEM_PROMPT]" in t),
    ("mistral-v1", lambda t: "[AVAILABLE_TOOLS]" in t),
    ("llama2",   lambda t: "[INST]" in t),
    ("phi3",     lambda t: "<|assistant|>" in t and "<|end|>" in t),
    ("zephyr",   lambda t: "<|user|>" in t and "<|endoftext|>" in t),
    ("vicuna",   lambda t: "USER: " in t and "ASSISTANT: " in t),
    ("deepseek", lambda t: "### Instruction:" in t and "<|EOT|>" in t),
]


def template_family(t):
    if not t:
        return None
    for name, test in _TEMPLATE_FINGERPRINTS:
        try:
            if test(t):
                return name
        except Exception:
            continue
    return None


# Features the NON-jinja template path in llama.cpp cannot render. Any of these in
# the embedded template means --jinja is not a preference, it is a requirement.
_JINJA_MARKERS = [
    ("{%- macro", "the template defines jinja macros"),
    ("{% macro", "the template defines jinja macros"),
    ("tool_calls", "the template renders tool calls"),
    ("tools", "the template takes a `tools` argument"),
    ("enable_thinking", "the template has a thinking/reasoning switch"),
    ("namespace(", "the template uses jinja namespaces"),
]

# What the engine does with reasoning tags when nothing is passed. Read from the
# source rather than assumed: common/common.h:510 initialises
# `reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK`.
REASONING_FORMATS = ["none", "auto", "deepseek", "deepseek-legacy"]
REASONING_DEFAULT_SRC = ("common/common.h:510 - the engine's own default is `deepseek` "
                         "(thinking-tag contents come back as message.reasoning_content). "
                         "The production Flash-Next seat passes NO --reasoning-format and so "
                         "inherits exactly this "
                         "[/usr/local/bin/pxa-seats-flashnext.sh, start_alex]")
PRODUCTION_CHAT_SRC = (
    "MEASURED against production: the live Flash-Next seat runs `--jinja --no-context-shift` "
    "with NO --chat-template (the file's embedded one) and NO --reasoning-format "
    "[/usr/local/bin/pxa-seats-flashnext.sh, start_alex]. The vLLM sm_70 seat runs "
    "`--enable-auto-tool-choice --tool-call-parser qwen3_coder` and no reasoning parser "
    "[same file, start_alina]. --jinja is there because without it every request carrying "
    "`tools` returns HTTP 500 while plain chat keeps working - that seat shipped broken for "
    "weeks on exactly this omission.")


class ChatOpts(object):
    """Every field carries (value, source, tag). tag is MEASURED / [INFERRED] /
    UNMEASURED / 'from the file' - a launcher that prints a sampling default
    without saying whether the model asked for it is guessing on the user's
    behalf."""

    def __init__(self):
        self.rows = []          # (label, value, tag, why)
        self.template = None    # None = use the file's embedded template
        self.template_file = None
        self.jinja = True
        self.reasoning_format = None
        self.reasoning_budget = None
        self.temp = self.top_p = self.top_k = self.min_p = self.repeat_penalty = None
        self.api_key = None
        self.slot_save_path = None
        self.warnings = []

    def add(self, label, value, tag, why):
        self.rows.append((label, value, tag, why))


def chat_defaults(prof, a, engine_dir=None):
    """Reads the model file for everything the model file knows, then fills the
    rest from production or from the engine's documented default. Returns ChatOpts
    with one row per decision, each tagged."""
    c = ChatOpts()
    tmpl = prof.get("chat_template")
    fam = template_family(tmpl)
    prof["chat_template_family"] = fam

    # ---- 1. the chat template ---------------------------------------------
    if a.chat_template_file:
        c.template_file = a.chat_template_file
        c.add("chat template", f"file {a.chat_template_file}", "YOURS",
              "--chat-template-file overrides whatever is inside the GGUF")
    elif a.chat_template:
        c.template = a.chat_template
        c.add("chat template", f"built-in '{a.chat_template}'", "YOURS",
              "--chat-template overrides the template inside the GGUF")
    elif tmpl:
        c.add("chat template", f"embedded in the GGUF ({len(tmpl)} chars"
              + (f", looks like '{fam}')" if fam else ", family not recognised)"), "from the file",
              "tokenizer.chat_template, shipped with the weights. This is what the model was "
              "trained to see; overriding it is how a model starts answering the wrong "
              "question in a slightly-wrong format.")
    else:
        c.add("chat template", "NONE IN THE FILE", "UNMEASURED",
              "this GGUF carries no tokenizer.chat_template, so llama-server will fall back to "
              "a generic one and chat quality is on you. Pass --chat-template <name> or "
              "--chat-template-file <path>.")
        c.warnings.append(
            "NO CHAT TEMPLATE IN THIS FILE. The server will still start and still answer, "
            "which is what makes this one expensive to find later. Published PXA files have "
            "been shipped in this state before - the Fusion4 PXQ2 tier needed a template fix "
            "in July 2026 after it went out without one. Pick a template explicitly.")
    if tmpl and "tool_calls" not in tmpl:
        c.warnings.append(
            "The embedded template has NO tool-call block (no `tool_calls` anywhere in it). "
            "Plain chat will work; anything that sends `tools` will not get a tool call back. "
            "If this seat is for an agent, use a template that renders tool calls.")

    # ---- 2. jinja ----------------------------------------------------------
    why_jinja = [w for m, w in _JINJA_MARKERS if tmpl and m in tmpl]
    if a.no_jinja:
        c.jinja = False
        c.add("--jinja", "OFF", "YOURS",
              "you passed --no-jinja. " + ("This template needs it (" + why_jinja[0] + ") - "
              "expect tool calls and thinking tags to break." if why_jinja else
              "Nothing in this template obviously needs it."))
    elif why_jinja:
        c.jinja = True
        c.add("--jinja", "ON", "MEASURED",
              "REQUIRED here: " + why_jinja[0] + ". Without --jinja llama-server renders the "
              "template on its non-jinja path, and a request carrying `tools` returns HTTP 500 "
              "while plain chat keeps working - the failure that ran unnoticed in production. "
              + PRODUCTION_CHAT_SRC.split(". ")[0] + ".")
    else:
        c.jinja = True
        c.add("--jinja", "ON", "[INFERRED]",
              "on by default because every recipe in docs/COOKBOOK.md passes it and it is what "
              "production runs. Nothing in this particular template demands it. "
              "--no-jinja turns it off.")

    # ---- 3. reasoning ------------------------------------------------------
    thinking = bool(tmpl and ("enable_thinking" in tmpl or "<think>" in tmpl))
    prof["thinking_template"] = thinking
    if a.reasoning_format:
        c.reasoning_format = a.reasoning_format
        c.add("--reasoning-format", a.reasoning_format, "YOURS", "you set it")
    elif thinking:
        c.add("--reasoning-format", "not passed -> engine default 'deepseek'", "MEASURED",
              "this template has a thinking switch, and the engine's default already extracts "
              "the tag contents into message.reasoning_content. " + REASONING_DEFAULT_SRC)
    else:
        c.add("--reasoning-format", "not passed -> engine default 'deepseek'", "[INFERRED]",
              "no thinking switch found in this template, so the setting should not matter. "
              + REASONING_DEFAULT_SRC)
    if a.reasoning_budget is not None:
        c.reasoning_budget = a.reasoning_budget
        c.add("--reasoning-budget", str(a.reasoning_budget), "YOURS",
              "caps thinking tokens, then forces the end sequence "
              "(common/reasoning-budget.cpp). 0 = think not at all, -1 = unlimited.")
    elif thinking:
        c.add("--reasoning-budget", "not passed (unlimited)", "[INFERRED]",
              "production passes no budget. Set --reasoning-budget 0 to switch thinking off "
              "for a latency-sensitive seat, or a token count to cap it "
              "(common/reasoning-budget.cpp).")

    # ---- 4. sampling -------------------------------------------------------
    s = prof.get("sampling") or {}
    for flag, key, cli in (("--temp", "temp", a.temp), ("--top-p", "top_p", a.top_p),
                           ("--top-k", "top_k", a.top_k)):
        if cli is not None:
            setattr(c, key, cli)
            c.add(flag, str(cli), "YOURS", "you set it")
        elif key in s:
            setattr(c, key, s[key])
            c.add(flag, _fmtnum(s[key]), "from the file",
                  f"general.sampling.{key} is written INTO this GGUF by the quantizer - it is "
                  f"the model author's own default, not a launcher preference.")
        else:
            c.add(flag, "not passed -> llama-server's built-in default", "[INFERRED]",
                  "this file carries no general.sampling.* key, so nothing here knows better "
                  "than the server does.")
    if a.min_p is not None:
        c.min_p = a.min_p
        c.add("--min-p", str(a.min_p), "YOURS", "you set it")
    if a.repeat_penalty is not None:
        c.repeat_penalty = a.repeat_penalty
        c.add("--repeat-penalty", str(a.repeat_penalty), "YOURS", "you set it")

    # ---- 5. the rest of the forgettable set --------------------------------
    if a.api_key:
        c.api_key = a.api_key
        c.add("--api-key", "set (not printed)", "YOURS",
              "the server refuses unauthenticated requests once this is on. NEVER echoed by "
              "this launcher, and NEVER written into a --serve-name script.")
    else:
        c.add("--api-key", "none - the port is OPEN", "UNMEASURED",
              f"anyone who can reach {a.host}:{a.port} can use this model. That is fine on a "
              f"loopback bind and not fine on 0.0.0.0. Set --api-key, or bind --host 127.0.0.1.")
    if a.slot_save_path:
        c.slot_save_path = a.slot_save_path
        c.add("--slot-save-path", a.slot_save_path, "YOURS",
              "lets /slots save and restore prompt caches - the release gate "
              "(bench/gate) uses it to replay a fill without re-prefilling.")
    else:
        c.add("--slot-save-path", "not set", "[INFERRED]",
              "only needed if you want /slots save+restore (the release gate does). "
              "Costs nothing to leave off.")
    c.add("--host / --port", f"{a.host}:{a.port}", "YOURS" if a.host != "0.0.0.0" else "default",
          "0.0.0.0 answers on every interface on this box. 127.0.0.1 answers only locally.")
    return c


def _fmtnum(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def template_preview(tmpl, nlines=6):
    if not tmpl:
        return ["(this file has no embedded chat template)"]
    out = [ln for ln in tmpl.splitlines()][:nlines]
    out = [(ln[:110] + "...") if len(ln) > 110 else ln for ln in out]
    total = len(tmpl.splitlines())
    if total > nlines:
        out.append(f"... {total - nlines} more lines ({len(tmpl)} chars total)")
    return out



# ---------------------------------------------------------------------------
# THE TERMINAL UI
# ---------------------------------------------------------------------------
# Six screens: CARDS, MODEL, CHAT, ENGINE, REVIEW, LAUNCH. stdlib `curses` only -
# this file has no dependencies and is not going to grow any.
#
# THE UI DECIDES NOTHING. Every screen either collects an answer the user already
# has (which cards, which file, chat or long documents) or DISPLAYS what the
# existing decision code returned. REVIEW runs plan_and_build() - the same
# function the command line runs - and shows its output verbatim, so the UI cannot
# drift from the plain-text path. If it ever disagrees with `--explain`, that is a
# bug in the display, never a second opinion about the seat.
#
# It degrades, it does not fail: no TTY, no curses, or a terminal under 80x24 and
# the line-prompt flow runs instead. Every non-interactive flag keeps working and
# never touches this code at all.
TUI_MIN_COLS, TUI_MIN_LINES = 80, 24


def tui_available():
    if os.environ.get("PXA_NO_TUI"):
        return False, "PXA_NO_TUI is set"
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False, "not a terminal"
    try:
        import curses  # noqa: F401
    except Exception as e:
        return False, f"curses unavailable ({e.__class__.__name__})"
    return True, ""


class _Capture(object):
    """Runs a function with stdout captured, and survives its SystemExit - a
    refusal is a result to display, not a crash of the UI."""

    def __init__(self):
        self.text, self.code, self.value = "", None, None

    def run(self, fn, *args, **kw):
        import io
        import contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                self.value = fn(*args, **kw)
            self.code = 0
        except SystemExit as e:
            self.code = e.code if isinstance(e.code, int) else 1
        except Exception as e:
            import traceback
            self.code = 99
            buf.write("\n" + traceback.format_exc())
            self.value = None
            del e
        self.text = buf.getvalue()
        return self


class LaunchTUI(object):
    def __init__(self, a, gpus, procs):
        self.a, self.gpus, self.procs = a, gpus, procs
        self.sel_idx = []            # GPU indexes ticked
        self.model = a.model or ""
        self.entries, self.mtrail, self.mnotes = [], [], []
        self.model_row = 0
        self.model_top = 0
        self.engine = a.engine        # None = take the recommendation
        self.rec_engine, self.rec_why, self.eng_block = None, "", {}
        self.workload = a.workload or "chat"
        self.status = ""
        self.built = None             # (plan, cmd, env, cv, prof, ctx)
        self.transcript = ""
        self.launched_ok = False
        self.return_after_model = "chat"

    # ---- chrome ---------------------------------------------------------
    def _frame(self, w, title, footer):
        import curses
        h, wd = w.getmaxyx()
        w.erase()
        bar = f" PXA launcher   {title} "
        w.attron(curses.A_REVERSE)
        w.addstr(0, 0, bar.ljust(wd - 1)[:wd - 1])
        w.attroff(curses.A_REVERSE)
        if self.status:
            w.addstr(h - 2, 1, self.status[:wd - 2], curses.A_BOLD)
        w.attron(curses.A_REVERSE)
        w.addstr(h - 1, 0, (" " + footer).ljust(wd - 1)[:wd - 1])
        w.attroff(curses.A_REVERSE)
        return h, wd

    def _too_small(self, w):
        h, wd = w.getmaxyx()
        if h < TUI_MIN_LINES or wd < TUI_MIN_COLS:
            w.erase()
            try:
                w.addstr(0, 0, f"Terminal is {wd}x{h}.")
                w.addstr(1, 0, f"This UI needs at least {TUI_MIN_COLS}x{TUI_MIN_LINES}.")
                w.addstr(2, 0, "Resize the window, or press q to drop to the plain prompts.")
            except Exception:
                pass
            w.refresh()
            return True
        return False

    @staticmethod
    def _put(w, y, x, text, attr=0, pad=False):
        """pad=True blanks the rest of the line first. Needed anywhere the content
        changes length between frames - the streamed server log above all, where a
        shorter line otherwise leaves the tail of the previous one behind and the
        pane reads like two logs interleaved."""
        h, wd = w.getmaxyx()
        if 0 <= y < h and x < wd:
            try:
                if pad:
                    w.addstr(y, x, " " * (wd - x - 1))
                w.addstr(y, x, str(text)[:wd - x - 1], attr)
            except Exception:
                pass

    # ---- screen 1: CARDS -------------------------------------------------
    def screen_cards(self, w):
        import curses
        row = 0
        while True:
            if self._too_small(w):
                if w.getch() in (ord("q"), 27):
                    return "abort"
                continue
            h, wd = self._frame(
                w, "1/5  CARDS - tick the cards this model should use",
                "up/down (or j/k) move   space tick   a all   n none   "
                "m model first   enter next   q quit")
            self._put(w, 2, 2, "  #  gpu  name                          VRAM used/total    "
                               "in use by another process?", curses.A_BOLD)
            for i, g in enumerate(self.gpus):
                idx, name, cc, tot, used, _u = g
                busy = self.procs.get(idx) or []
                if busy:
                    st = "busy: " + ", ".join(f"{n} (pid {p}, {mb} MiB)" for p, n, mb in busy)
                    at = curses.A_DIM
                elif used > 512:
                    st, at = f"busy: {used} MiB resident, no process named", curses.A_DIM
                else:
                    st, at = "free", 0
                mark = "x" if idx in self.sel_idx else " "
                line = (f" [{mark}] {idx:<3}  {name.replace('NVIDIA ', ''):<28} "
                        f"{used:>6}/{tot:<6} MiB sm_{cc}  {st}")
                self._put(w, 3 + i, 2, line,
                          (curses.A_REVERSE if i == row else at))
            base = 4 + len(self.gpus)
            self._put(w, base, 2, "MEASURED combinations on this class of box:", curses.A_BOLD)
            for j, r in enumerate(RECIPES):
                if r.status != "MEASURED":
                    continue
                self._put(w, base + 1 + j, 4,
                          f"{r.hw_str():<10}  {r.title}")
            # The hint is about the CARD SHAPE only - no model is chosen yet, so it
            # must not pretend to know which row will apply. It lists the rows whose
            # hardware matches and leaves the model half to the next screen.
            picked = [g for g in self.gpus if g[0] in self.sel_idx]
            hits = [r for r in RECIPES if picked and r.hw_matches(picked)
                    and r.status == "MEASURED"]
            if not picked:
                hint = "nothing ticked yet"
            elif hits:
                hint = "MEASURED for this shape: " + "; ".join(
                    f"{r.family_str()} {('|'.join(sorted(r.tiers)) or 'any tier')}"
                    for r in hits)
            else:
                hint = ("no measured row for this card shape - the launcher will say "
                        "UNMEASURED and ask you to confirm")
            self._put(w, h - 4, 2,
                      "selection: " + (", ".join(str(i) for i in self.sel_idx) or "none")
                      + "   -> " + hint, curses.A_BOLD)
            w.refresh()
            k = w.getch()
            if k == curses.KEY_RESIZE:
                continue
            if k in (ord("q"), 27):
                return "abort"
            if k in (curses.KEY_UP, ord("k")):
                row = (row - 1) % len(self.gpus)
            elif k in (curses.KEY_DOWN, ord("j")):
                row = (row + 1) % len(self.gpus)
            elif k == ord(" "):
                idx = self.gpus[row][0]
                if idx in self.sel_idx:
                    self.sel_idx.remove(idx)
                else:
                    self.sel_idx.append(idx)
                    self.sel_idx.sort()
            elif k == ord("a"):
                self.sel_idx = [g[0] for g in self.gpus]
            elif k == ord("n"):
                self.sel_idx = []
            elif k == ord("m"):
                self.return_after_model = "cards"
                return "model"
            elif k in (10, 13, curses.KEY_ENTER):
                if not self.sel_idx:
                    self.status = "tick at least one card with the space bar"
                    continue
                self.status = ""
                self.return_after_model = "chat"
                return "model" if not self.model else "chat"

    # ---- screen 2: MODEL -------------------------------------------------
    def _load_models(self):
        roots, self.mtrail = model_roots(self.a.models_dir)
        self.entries, self.mnotes = scan_models(roots)

    def _fits(self, e):
        """Weights-only fit against the ticked cards - the same formula-free check
        R-17B blocks on, minus the PLE table that never reaches VRAM."""
        picked = [g for g in self.gpus if g[0] in self.sel_idx]
        if not picked:
            return "-", 0
        total = sum(g[3] for g in picked) * 1024 * 1024
        gpu_bytes = (e.get("size") or 0) - (e.get("ple_bytes") or 0)
        return ("yes" if gpu_bytes < total else "NO"), gpu_bytes

    def screen_model(self, w):
        import curses
        if not self.entries:
            self.status = "scanning for models..."
            w.erase(); self._put(w, 1, 2, "scanning for models..."); w.refresh()
            self._load_models()
            self.status = ""
        while True:
            if self._too_small(w):
                if w.getch() in (ord("q"), 27):
                    return "abort"
                continue
            h, wd = self._frame(
                w, "2/5  MODEL - pick the file to serve",
                "up/down (or j/k) move   enter pick   d change directory   c back to cards   q quit")
            if not self.entries:
                self._put(w, 2, 2, "No .gguf files found. Searched:", curses.A_BOLD)
                for i, t in enumerate(self.mtrail[:8]):
                    self._put(w, 3 + i, 4, t)
                self._put(w, 12, 2, "Press d to add a directory, or quit and pass "
                                    "--models-dir / PXA_MODELS_DIR.")
            else:
                self._put(w, 2, 2,
                          "  file                                     size       family        "
                          "tier      fits", curses.A_BOLD)
                view = h - 8
                if self.model_row < self.model_top:
                    self.model_top = self.model_row
                if self.model_row >= self.model_top + view:
                    self.model_top = self.model_row - view + 1
                for i in range(self.model_top, min(len(self.entries), self.model_top + view)):
                    e = self.entries[i]
                    nm = os.path.basename(e["path"])
                    if len(nm) > 40:
                        nm = nm[:37] + "..."
                    fit, _ = self._fits(e)
                    line = (f"  {nm:<40} {human_bytes(e['size']):>9}  "
                            f"{family_label(e):<13} {(e.get('tier') or '-'):<9} {fit}")
                    self._put(w, 3 + i - self.model_top, 2, line,
                              curses.A_REVERSE if i == self.model_row else 0)
                e = self.entries[self.model_row]
                self._put(w, h - 5, 2, e["path"][:wd - 4], curses.A_DIM)
                self._put(w, h - 4, 2,
                          f"arch {e.get('arch') or '?'}   trained ctx "
                          f"{e.get('n_ctx_train') or '?'}   "
                          + ("per-layer-embedding table (stays in host RAM)   "
                             if e.get("ple") else "")
                          + ("vision projector expected" if e.get("vision") else ""),
                          curses.A_DIM)
            w.refresh()
            k = w.getch()
            if k == curses.KEY_RESIZE:
                continue
            if k in (ord("q"), 27):
                return "abort"
            if k == ord("c"):
                return "cards"
            if not self.entries:
                if k == ord("d"):
                    self._prompt_dir(w)
                continue
            if k in (curses.KEY_UP, ord("k")):
                self.model_row = (self.model_row - 1) % len(self.entries)
            elif k in (curses.KEY_DOWN, ord("j")):
                self.model_row = (self.model_row + 1) % len(self.entries)
            elif k == curses.KEY_NPAGE:
                self.model_row = min(len(self.entries) - 1, self.model_row + 10)
            elif k == curses.KEY_PPAGE:
                self.model_row = max(0, self.model_row - 10)
            elif k == ord("d"):
                self._prompt_dir(w)
            elif k in (10, 13, curses.KEY_ENTER):
                self.model = self.entries[self.model_row]["path"]
                self.a.model = self.model
                nxt, self.return_after_model = self.return_after_model, "chat"
                return "cards" if nxt == "cards" else "chat"

    def _prompt_dir(self, w):
        import curses
        curses.echo()
        curses.curs_set(1)
        h, wd = w.getmaxyx()
        self._put(w, h - 3, 2, "directory to search: " + " " * 40)
        w.move(h - 3, 23)
        try:
            d = w.getstr(h - 3, 23, 200).decode("utf-8", "replace").strip()
        except Exception:
            d = ""
        curses.noecho()
        curses.curs_set(0)
        if d:
            self.a.models_dir = [d] + list(self.a.models_dir or [])
            self.entries, self.model_row, self.model_top = [], 0, 0
            self._load_models()
            self.status = f"searched {d}: {len(self.entries)} file(s)"

    # ---- screen 3: CHAT --------------------------------------------------
    def screen_chat(self, w):
        import curses
        E, _ = resolve_engine_dir()
        prof = model_profile(self.model, model_kind(self.model))
        while True:
            c = chat_defaults(prof, self.a, E)
            if self._too_small(w):
                if w.getch() in (ord("q"), 27):
                    return "abort"
                continue
            h, wd = self._frame(
                w, "3/5  CHAT - template, jinja, reasoning, sampling",
                "t template   j jinja   r reasoning   b budget   s sampling   k key/host/port"
                "   enter next   c back   q quit")
            tmpl = prof.get("chat_template")
            self._put(w, 2, 2, "template inside this file: "
                      + (f"{len(tmpl)} chars, looks like "
                         f"'{prof.get('chat_template_family') or 'unrecognised'}'"
                         if tmpl else "NONE - the server will fall back to a generic one"),
                      curses.A_BOLD)
            for i, ln in enumerate(template_preview(tmpl, 5)):
                self._put(w, 3 + i, 4, "| " + ln, curses.A_DIM)
            y = 9
            self._put(w, y, 2, f"{'setting':<20} {'value':<40} source", curses.A_BOLD)
            for j, (label, value, tag, _why) in enumerate(c.rows):
                self._put(w, y + 1 + j, 2, f"{label:<20} {str(value)[:40]:<40} {tag}")
            y = y + 2 + len(c.rows)
            for j, warn in enumerate(c.warnings[:2]):
                self._put(w, y + j, 2, "!! " + warn, curses.A_BOLD)
            w.refresh()
            k = w.getch()
            if k == curses.KEY_RESIZE:
                continue
            if k in (ord("q"), 27):
                return "abort"
            if k == ord("c"):
                return "model"
            if k in (10, 13, curses.KEY_ENTER):
                return "engine"
            if k == ord("j"):
                self.a.no_jinja = not self.a.no_jinja
            elif k == ord("t"):
                names, prov = builtin_chat_templates(E)
                v = self._ask_line(w, f"template name ({len(names)} built in) or .jinja path, "
                                      f"empty = the file's own: ")
                if v.endswith(".jinja") or os.path.sep in v:
                    self.a.chat_template_file, self.a.chat_template = v, ""
                elif v in names:
                    self.a.chat_template, self.a.chat_template_file = v, ""
                elif v:
                    self.status = f"{v!r} is not one of the built-in names ({prov})"
                else:
                    self.a.chat_template = self.a.chat_template_file = ""
            elif k == ord("r"):
                v = self._ask_line(w, "reasoning format (%s), empty = engine default: "
                                      % "/".join(REASONING_FORMATS))
                self.a.reasoning_format = v if v in REASONING_FORMATS else ""
            elif k == ord("b"):
                v = self._ask_line(w, "reasoning budget in tokens (0 = none, -1 = unlimited, "
                                      "empty = unset): ")
                self.a.reasoning_budget = int(v) if re.fullmatch(r"-?\d+", v or "") else None
            elif k == ord("s"):
                for name, cast in (("temp", float), ("top_p", float), ("top_k", int),
                                   ("min_p", float), ("repeat_penalty", float)):
                    v = self._ask_line(w, f"{name} [{getattr(self.a, name)}]: ")
                    if v:
                        try:
                            setattr(self.a, name, cast(v))
                        except ValueError:
                            self.status = f"{name}: not a number"
            elif k == ord("k"):
                self.a.api_key = self._ask_line(w, "API key (empty = open port): ", hide=True)
                hv = self._ask_line(w, f"host [{self.a.host}]: ")
                if hv:
                    self.a.host = hv
                pv = self._ask_line(w, f"port [{self.a.port}]: ")
                if pv.isdigit():
                    self.a.port = int(pv)

    def _ask_line(self, w, prompt, hide=False):
        import curses
        h, wd = w.getmaxyx()
        self._put(w, h - 2, 1, " " * (wd - 2))
        self._put(w, h - 2, 1, prompt, curses.A_BOLD)
        w.refresh()
        if not hide:
            curses.echo()
        curses.curs_set(1)
        try:
            v = w.getstr(h - 2, min(wd - 2, 1 + len(prompt)), 200).decode("utf-8", "replace")
        except Exception:
            v = ""
        curses.noecho()
        curses.curs_set(0)
        return v.strip()

    # ---- screen 4: ENGINE ------------------------------------------------
    def _recommend(self):
        sel = [g for g in self.gpus if g[0] in self.sel_idx]
        kind = model_kind(self.model)
        prof = model_profile(self.model, kind)
        caps, img, trail = vllm_eligibility(sel, self.a.vllm_image)
        p = decide(sel, kind, self.model, None, prof, max(1, self.a.np or 1),
                   infer_workload(self.a.np or 1, self.workload), caps, img, trail)
        self.rec_engine = p.engine
        self.rec_why = p.reason
        self.eng_block = {}
        if p.engine != "vllm":
            self.eng_block["vllm"] = p.reason or "the decision table did not pick it"
        return p

    def screen_engine(self, w):
        import curses
        p = self._recommend()
        row = 0 if self.rec_engine == "llama" else 1
        opts = [("llama", "pxa - the llama.cpp engine in this tree"),
                ("vllm", "vllm-pxq4 - the sm_70 serving sidecar")]
        while True:
            if self._too_small(w):
                if w.getch() in (ord("q"), 27):
                    return "abort"
                continue
            h, wd = self._frame(w, "4/5  ENGINE - which runtime serves this seat",
                                "up/down (or j/k) move   enter accept   c back   q quit")
            y = 2
            for i, (key, label) in enumerate(opts):
                blocked = self.eng_block.get(key)
                tag = "  (recommended)" if key == self.rec_engine else ""
                at = curses.A_REVERSE if i == row else (curses.A_DIM if blocked else 0)
                self._put(w, y, 2, f" {'>' if i == row else ' '} {label}{tag}", at)
                if key == self.rec_engine:
                    for j, ln in enumerate(_wrap(self.rec_why, wd - 10)[:4]):
                        self._put(w, y + 1 + j, 8, ln, curses.A_DIM)
                    y += 1 + min(4, len(_wrap(self.rec_why, wd - 10)))
                elif blocked:
                    for j, ln in enumerate(_wrap("cannot be used here: " + blocked,
                                                 wd - 10)[:4]):
                        self._put(w, y + 1 + j, 8, ln, curses.A_DIM)
                    y += 1 + min(4, len(_wrap("cannot be used here: " + blocked, wd - 10)))
                y += 2
            self._put(w, y, 2, "evidence", curses.A_BOLD)
            for j, e in enumerate(p.evidence[:6]):
                for kk, ln in enumerate(_wrap(e, wd - 8)[:2]):
                    self._put(w, y + 1 + j * 2 + kk, 4, ln, curses.A_DIM)
            w.refresh()
            k = w.getch()
            if k == curses.KEY_RESIZE:
                continue
            if k in (ord("q"), 27):
                return "abort"
            if k == ord("c"):
                return "chat"
            if k in (curses.KEY_UP, ord("k")):
                row = (row - 1) % len(opts)
            elif k in (curses.KEY_DOWN, ord("j")):
                row = (row + 1) % len(opts)
            elif k in (10, 13, curses.KEY_ENTER):
                key = opts[row][0]
                if self.eng_block.get(key):
                    self.status = "that engine is blocked here - see the reason under it"
                    continue
                self.a.engine = None if key == self.rec_engine else key
                self.status = ""
                return "review"

    # ---- screen 5: REVIEW ------------------------------------------------
    def _build(self):
        self.a.gpus = ",".join(str(i) for i in self.sel_idx)
        self.a.model = self.model
        self.a.workload = self.workload
        cap = _Capture().run(plan_and_build, self.a, self.gpus)
        self.transcript = cap.text
        self.built = cap.value if cap.code == 0 else None
        return cap

    def screen_review(self, w):
        import curses
        cap = self._build()
        top = 0
        while True:
            if self._too_small(w):
                if w.getch() in (ord("q"), 27):
                    return "abort"
                continue
            h, wd = self._frame(
                w, "5/5  REVIEW - the decision, the evidence and the command",
                "up/down (or j/k) scroll   e edit ctx/np   s save script   enter LAUNCH   "
                "c back   q quit")
            lines = []
            for ln in self.transcript.splitlines():
                lines.extend(_wrap(ln, wd - 4) or [""])
            view = h - 5
            top = max(0, min(top, max(0, len(lines) - view)))
            for i in range(view):
                if top + i >= len(lines):
                    break
                ln = lines[top + i]
                at = 0
                if "MEASURED" in ln and "UNMEASURED" not in ln:
                    at = curses.A_BOLD
                elif "UNMEASURED" in ln or "REFUS" in ln or ln.strip().startswith("!!"):
                    at = curses.A_BOLD
                elif "[INFERRED]" in ln:
                    at = curses.A_DIM
                self._put(w, 2 + i, 2, ln, at)
            _cx = self.a.ctx or "(from the recipe)"
            _npv = self.a.np or "(from the recipe)"
            self._put(w, h - 3, 2,
                      f"ctx {_cx}   np {_npv}   scroll {top + 1}/{max(1, len(lines))}"
                      + ("   REFUSED - nothing will start" if self.built is None else ""),
                      curses.A_BOLD)
            w.refresh()
            k = w.getch()
            if k == curses.KEY_RESIZE:
                continue
            if k in (ord("q"), 27):
                return "abort"
            if k == ord("c"):
                return "engine"
            if k in (curses.KEY_DOWN, ord("j")):
                top += 1
            elif k in (curses.KEY_UP, ord("k")):
                top = max(0, top - 1)
            elif k == curses.KEY_NPAGE:
                top += 10
            elif k == curses.KEY_PPAGE:
                top = max(0, top - 10)
            elif k == ord("e"):
                v = self._ask_line(w, f"total ctx [{self.a.ctx or 'recipe default'}]: ")
                if v.isdigit():
                    self.a.ctx = int(v)
                v = self._ask_line(w, f"parallel slots -np [{self.a.np or 'recipe default'}]: ")
                if v.isdigit():
                    self.a.np = int(v)
                cap = self._build()
                top = 0
            elif k == ord("s"):
                if self.built is None:
                    self.status = "nothing to save: the plan was refused"
                    continue
                name = self._ask_line(w, "name for the restart script: ")
                if name:
                    _plan, cmd, env, cv, _pr, _cx = self.built
                    try:
                        sp = write_serve_script(name, cmd, env, cv, self.a.serve_dir)
                        self.status = f"wrote {sp}"
                    except OSError as ex:
                        self.status = f"could not write it: {ex}"
            elif k in (10, 13, curses.KEY_ENTER):
                if self.built is None:
                    self.status = "refused - read the reason above; nothing will be started"
                    continue
                return "launch"

    # ---- screen 6: LAUNCH ------------------------------------------------
    def screen_launch(self, w):
        import curses
        import threading
        import collections as _c
        plan, cmd, env, cv, _prof, _ctx = self.built
        e = dict(os.environ)
        e.update(env)
        e["CUDA_VISIBLE_DEVICES"] = cv
        e["NVIDIA_VISIBLE_DEVICES"] = cv
        log = _c.deque(maxlen=4000)
        state = {"phase": "starting the server process", "first_token": None}
        try:
            proc = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as ex:
            self.status = f"could not start: {ex}"
            return "review"

        def pump():
            try:
                for line in proc.stdout:
                    log.append(line.rstrip("\n"))
                    low = line.lower()
                    if "loading model" in low or "llama_model_loader" in low:
                        state["phase"] = "loading model"
                    elif "offloaded" in low and "layers to gpu" in low:
                        state["phase"] = "weights on the GPUs"
                    elif "listening" in low or "server is listening" in low:
                        state["phase"] = f"listening on {self.a.host}:{self.a.port}"
                        threading.Thread(target=probe, daemon=True).start()
            except Exception:
                pass

        def probe():
            """One tiny raw /completion. The post-boot contract this file has always
            printed says a seat is not trusted until a short RAW prompt comes back,
            so the UI performs exactly that check and reports only what it saw."""
            import json as _j
            import urllib.request
            host = "127.0.0.1" if self.a.host in ("0.0.0.0", "::") else self.a.host
            url = f"http://{host}:{self.a.port}/completion"
            body = _j.dumps({"prompt": "1 2 3", "n_predict": 4, "temperature": 0,
                             "cache_prompt": False}).encode()
            hdr = {"Content-Type": "application/json"}
            if self.a.api_key:
                hdr["Authorization"] = "Bearer " + self.a.api_key
            for _ in range(120):
                try:
                    req = urllib.request.Request(url, data=body, headers=hdr)
                    with urllib.request.urlopen(req, timeout=20) as r:
                        d = _j.loads(r.read().decode())
                    t = (d.get("timings") or {})
                    state["first_token"] = (
                        f"first token OK - {len((d.get('content') or ''))} chars back, "
                        f"prefill {t.get('prompt_per_second', 0):.1f} t/s, "
                        f"decode {t.get('predicted_per_second', 0):.1f} t/s")
                    state["phase"] = "serving"
                    return
                except Exception:
                    _sleep(1.0)
            state["first_token"] = "no answer from /completion after 120 s"

        threading.Thread(target=pump, daemon=True).start()
        w.nodelay(True)
        try:
            while True:
                if self._too_small(w):
                    if w.getch() in (ord("q"), 27):
                        break
                    continue
                h, wd = self._frame(w, "RUNNING - the server log is below",
                                    "q stop the server and come back   up/down scroll")
                self._put(w, 2, 2, f"state: {state['phase']}", curses.A_BOLD, pad=True)
                if state["first_token"]:
                    self._put(w, 3, 2, state["first_token"], curses.A_BOLD, pad=True)
                self._put(w, 4, 2, f"cards {cv}   pid {proc.pid}   "
                                   f"{'running' if proc.poll() is None else 'EXITED %s' % proc.returncode}")
                self._put(w, 5, 2, "-" * (wd - 4), curses.A_DIM)
                view = h - 8
                tail = list(log)[-view:]
                for i in range(view):
                    ln = tail[i] if i < len(tail) else ""
                    self._put(w, 6 + i, 2, ln.replace("\t", "    "), pad=True)
                w.refresh()
                k = w.getch()
                if k in (ord("q"), 27):
                    break
                if proc.poll() is not None and not log:
                    break
                _sleep(0.2)
        finally:
            w.nodelay(False)
            self._stop(proc, cmd)
        self.status = "server stopped"
        return "review"

    @staticmethod
    def _stop(proc, cmd):
        """SIGTERM, then a hard kill. If the seat is a container (the command IS a
        docker run), stop it BY NAME - never by image, which would take down every
        container built from it."""
        import signal
        try:
            if os.path.basename(cmd[0]) == "docker" and "--name" in cmd:
                name = cmd[cmd.index("--name") + 1]
                subprocess.run(["docker", "stop", name], capture_output=True, timeout=60)
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                for _ in range(100):
                    if proc.poll() is not None:
                        return
                    _sleep(0.1)
                proc.kill()
        except Exception:
            pass

    # ---- driver ----------------------------------------------------------
    def run(self):
        import curses

        def _main(w):
            curses.curs_set(0)
            w.keypad(True)
            screen = "cards"
            while True:
                if screen == "cards":
                    screen = self.screen_cards(w)
                elif screen == "model":
                    screen = self.screen_model(w)
                elif screen == "chat":
                    screen = self.screen_chat(w)
                elif screen == "engine":
                    screen = self.screen_engine(w)
                elif screen == "review":
                    screen = self.screen_review(w)
                elif screen == "launch":
                    screen = self.screen_launch(w)
                else:
                    return screen
        return curses.wrapper(_main)


def _wrap(text, width):
    out, line = [], ""
    for word in str(text).split():
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out or [""]


def _sleep(t):
    import time
    time.sleep(t)



# ---------------------------------------------------------------------------
# SELFTEST
# ---------------------------------------------------------------------------
def selftest(gpus):
    print("=== selftest: decision table against this machine ===")
    print(f"    spec: baselines/LAUNCHER-SPEC.md md5 {SPEC_MD5}")
    if gpus:
        for g in gpus:
            print(f"    card {g[0]}: {g[1]:<28} sm_{g[2]}  {g[3]} MiB total, {g[4]} MiB used, "
                  f"-ub table -> {ub_for_card(g[3])}")
    else:
        print("    no GPUs visible - the table below still exercises every branch")
    p2p, topo = peer_topology()
    print(f"    topology: {topo} -> custom all-reduce {'ON' if p2p else 'OFF'} "
          f"(MEASURED: CAR ~18% worse than NCCL on MoE without P2P)")

    idx = [g[0] for g in gpus]
    cases = [("all cards", idx), ("first card", idx[:1])]
    for cc in (60, 61, 70):
        sub = [g[0] for g in gpus if g[2] == cc]
        if sub:
            cases.append((f"all sm_{cc}", sub))
            if len(sub) >= 2:
                cases.append((f"2x sm_{cc}", sub[:2]))
    mixed = [g[0] for g in gpus if g[2] == 60][:1] + [g[0] for g in gpus if g[2] == 70][:1]
    if len(mixed) == 2:
        cases.append(("mixed sm_60+sm_70", mixed))
    if not cases[0][1]:
        cases = [("no GPUs", [])]

    # (label, artifact kind, profile). The kind matters: a converted vLLM directory
    # and a raw GGUF take different structural gates, and a PXQ3 file can only ever
    # BE a GGUF (the converter accepts PXQ4 only).
    profiles = [
        ("dense PXQ4", "vllm_dir", {"is_moe": False, "n_expert": 0, "tier": "PXQ4",
                                    "arch": "qwen35"}),
        ("MoE   PXQ4", "vllm_dir", {"is_moe": True, "n_expert": 256, "tier": "PXQ4",
                                    "arch": "qwen35moe"}),
        ("MoE   PXQ4", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ4",
                                "arch": "qwen35moe"}),
        ("MoE   PXQ3", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ3",
                                "arch": "qwen35moe"}),
        ("MoE   PXQ1", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ1",
                                "arch": "qwen35moe"}),
        ("MoE   UNIV", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ_UNIVERSAL",
                                "arch": "qwen35moe"}),
    ]
    # Run the table twice: once with eligibility as this box actually resolves it,
    # once with the MEASURED sm_60 image forced. On a box where no image resolves,
    # the first pass short-circuits every branch to llama.cpp and would hide the
    # decision table - which is the thing worth testing.
    live_caps, live_img, live_trail = vllm_eligibility(gpus, None)
    passes = [(live_caps, live_img, live_trail,
               f"eligibility as resolved here: image={live_img}, caps={sorted(live_caps) or 'none'}")]
    if not live_caps:
        rec = VLLM_IMAGES["pxa-sm60-dev"]
        passes.append((set(rec["caps"]), "pxa-sm60-dev",
                       ["probe 1: forced for selftest coverage"],
                       "hypothetical: image=pxa-sm60-dev (MEASURED sm_60 caps) - the arm every "
                       "vLLM cell in the table was produced on"))
    for caps, img, trail, banner in passes:
        print(f"  --- {banner} ---")
        for label, ids in cases:
            sel = [g for g in gpus if g[0] in set(ids)]
            for plabel, pkind, prof in profiles:
                for np_ in (1, 4, 5, 6, 8):
                    pl = decide(sel, pkind, "/x/coder35-moe-pxq4", None, prof, np_,
                                infer_workload(np_, None), caps, img, trail)
                    if pl.refusals:
                        out = f"REFUSE {pl.refusals[0][0]}"
                        rest = pl.refusals[0][1].split(".")[0]
                    else:
                        out = str(pl.engine)
                        rest = pl.reason
                    ack = "  <ACK-REQUIRED>" if pl.needs_ack else ""
                    print(f"  {label:18} {plabel:11} {pkind:9} np={np_:<2} -> {out:11} :: "
                          f"{rest[:48]}{ack}")
    print(f"  MoE table (MEASURED, cards 0+6): "
          + ", ".join(f"np{k}={v[2]}" for k, v in sorted(MOE_TABLE.items())))
    print(f"  llama.cpp through np={MOE_LLAMA_MAX_NP}; vLLM from np={MOE_VLLM_MIN_NP}; "
          f"nothing above np={MOE_TABLE_MAX_NP} measured on either engine")
    print(f"  vllm_pxq4 importable in this interpreter: {has_vllm_pxq4()}")

    # ---- THE TOPOLOGY RECIPE TABLE, every row, enumerated ------------------
    # Printed in full on every selftest run so that "what does this launcher do on
    # 2x V100" never needs the source read. Each row is EXERCISED, not just
    # printed: a synthetic selection matching its hardware and a synthetic profile
    # matching its model class are pushed through recipe_for(), and the row that
    # comes back must be the row we asked about. A table entry that cannot be
    # reached is worse than no entry.
    print("  --- topology recipe table (%d rows) ---" % len(RECIPES))
    row_fail = []
    for r in RECIPES:
        caps = sorted(r.caps)
        sel = []
        for i in range(r.ncards):
            cc = caps[i % len(caps)]
            sel.append((i, "synthetic", cc, 11264 if cc == 61 else 16384, 0, f"UUID-{i}"))
        tier = sorted(r.tiers)[0] if r.tiers else "PXQ_UNIVERSAL"
        prof = {"tier": None if tier == "none" else tier,
                "arch": r.arch or "qwen35moe",
                "is_moe": bool(r.family & {"moe", "hybrid-moe"}),
                "n_expert": 256 if r.family & {"moe", "hybrid-moe"} else 0,
                "deltanet": bool(r.family & {"hybrid-moe", "hybrid"})}
        got, st, _ev, _nt = recipe_for(sel, prof, "chat")
        ok = (got is r and st == r.status)
        if not ok:
            row_fail.append(f"{r.key} -> {got.key if got else None}/{st}")
        flags = ["-b %d" % r.b, "-ub %d" % r.ub, "-c %d" % r.ctx, "-sm %s" % r.sm]
        if r.np:
            flags.append("-np %d" % r.np)
        if r.threads:
            flags.append("-t %d" % r.threads)
        if r.ts:
            flags.append("-ts %s" % r.ts)
        if r.ot:
            flags.append("-ot %s" % r.ot)
        flags += list(r.extra)
        fa = "/".join(f"{w}:-fa {FA_BY_WORKLOAD[w]}" for w in ("chat", "serve", "longdoc"))
        print(f"  [{r.status:8}] {r.key:<24} {r.hw_str():<20} {r.family_str():<18} "
              f"{'|'.join(sorted(r.tiers)) or 'any':<22} {'REACHED' if ok else 'UNREACHABLE'}")
        print(f"             flags: {' '.join(flags)}")
        print(f"             fa:    {fa}")
        if r.env:
            print(f"             env:   PXA_ENHANCE=1 "
                  + " ".join(f"{k}={v}" for k, v in r.env.items()))
        else:
            print(f"             env:   PXA_ENHANCE=1")
        print(f"             src:   {r.source}")
        if r.status == "MEASURED":
            print(f"             meas:  {r.numbers}")
    # And the off-table case, exercised too: three cards of one class matches no
    # row, must come back UNMEASURED, and must NOT silently borrow a neighbour.
    off = [(i, "synthetic", 60, 16384, 0, f"U{i}") for i in range(3)]
    got_off, st_off, _e, _n = recipe_for(off, {"tier": "PXQ4", "arch": "qwen35",
                                               "is_moe": False, "deltanet": False}, "chat")
    off_ok = (got_off is None and st_off == "UNMEASURED")
    print(f"  [off-table] 3x sm_60 dense PXQ4 -> "
          f"{'UNMEASURED, no -b/-ub emitted' if off_ok else 'WRONG: ' + str(got_off)}")
    # STANDING ASSERTIONS, re-checked on every selftest run. These are the two
    # invariants that cost this project a live corruption bug and a dead boot, so
    # they are asserted mechanically rather than trusted to review.
    print("  --- standing assertions ---")
    ok_all = True
    # A1: every compilation-config this file can construct carries BOTH keys, at
    #     every np on the ladder, and never FULL_AND_PIECEWISE.
    bad = []
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33, 64):
        c = compilation_config(n)
        if c.get("cudagraph_mode") != "FULL_DECODE_ONLY":
            bad.append(f"np={n} mode={c.get('cudagraph_mode')}")
        if c.get("custom_ops") != ["none"]:
            bad.append(f"np={n} custom_ops={c.get('custom_ops')}")
        if max(c["cudagraph_capture_sizes"]) < n:
            bad.append(f"np={n} ladder {c['cudagraph_capture_sizes']} does not cover np")
    ok_all &= not bad
    print(f"  A1 every emitted compilation-config = FULL_DECODE_ONLY + custom_ops:[none], "
          f"ladder covers np: {'PASS' if not bad else 'FAIL ' + str(bad)}")
    # A2: FULL_AND_PIECEWISE never appears at an EMISSION SITE. The string is
    #     allowed in prose (the image table's 'why', the FDO comment block, R-08's
    #     refusal text) - what must never happen is it reaching a command, an env
    #     value or a JSON payload. So the scan is for emission markers on the same
    #     line, not for the bare string, which would flag its own documentation.
    EMIT = ("cmd +=", "cmd = [", "json.dumps", "env[", 'env = {', "execvpe")
    src = open(os.path.abspath(__file__)).read().splitlines()
    live = [i + 1 for i, ln in enumerate(src)
            if "FULL_AND_PIECEWISE" in ln and any(m in ln for m in EMIT)]
    ok_all &= not live
    print(f"  A2 FULL_AND_PIECEWISE never reachable as an emitted value: "
          f"{'PASS' if not live else 'FAIL at lines ' + str(live)}")
    # A3: --cudagraph-mode anything-but-FDO is refused (R-08) rather than honoured.
    a3 = "R-08" in "".join(src) and 'a.cudagraph_mode != "FULL_DECODE_ONLY"' in "".join(src)
    ok_all &= a3
    print(f"  A3 a non-FDO --cudagraph-mode request is refused, not honoured: "
          f"{'PASS' if a3 else 'FAIL'}")
    # A4: PXQ1 is detected by TENSOR TYPE, so a PXQ1-bearing UNIVERSAL map is caught
    #     too - the case the spec's `tier in {PXQ1}` condition could never match.
    t_uni, _, _ = tier_from_tensors([("blk.0.ffn_gate_exps.weight", 248, 1),
                                     ("blk.1.ffn_gate_exps.weight", 252, 1)])
    t_pure, _, _ = tier_from_tensors([("blk.0.ffn_gate_exps.weight", 248, 1)])
    a4 = (t_uni == "PXQ1" and t_pure == "PXQ1")
    ok_all &= a4
    print(f"  A4 PXQ1 tensors anywhere -> tier PXQ1 -> R-01 (uniform AND inside a UNIVERSAL "
          f"map): {'PASS' if a4 else 'FAIL ' + str((t_uni, t_pure))}")
    # A5: a bare 'mtp' spec can never become n_max>=2.
    a5 = parse_spec("mtp")[1].get("n_max") is None and "mtp:n_max=1" in "".join(src)
    ok_all &= a5
    print(f"  A5 bare --spec mtp expands to n_max=1 only: {'PASS' if a5 else 'FAIL'}")
    # A6: every row in the recipe table is REACHABLE through recipe_for, and an
    #     off-table topology returns no row at all rather than the nearest one.
    ok_all &= not row_fail and off_ok
    print(f"  A6 every recipe row reachable, off-table topology returns none: "
          f"{'PASS' if not row_fail and off_ok else 'FAIL ' + str(row_fail or 'off-table')}")
    # A7: the FA regime map is total over the workloads argparse accepts, and only
    #     longdoc turns FA off. This is the one setting a user picks with their own
    #     head, so a typo that silently served chat at -fa off must be caught here.
    a7 = (set(FA_BY_WORKLOAD) == {"chat", "serve", "longdoc"}
          and FA_BY_WORKLOAD["chat"] == "on" and FA_BY_WORKLOAD["serve"] == "on"
          and FA_BY_WORKLOAD["longdoc"] == "off")
    ok_all &= a7
    print(f"  A7 FA regime map total over chat/serve/longdoc, only longdoc is fa-off: "
          f"{'PASS' if a7 else 'FAIL ' + str(FA_BY_WORKLOAD)}")
    # A8: no recipe row may put a -ub the smallest card in its own topology cannot
    #     hold. The 11 GiB 1080 Ti is the case: a ub2048 compute buffer (~1.9 GiB)
    #     does not allocate next to a resident 10.7 GB model, which is why that row
    #     is 768 and why a row that drifted to 2048 must fail here, not at boot.
    # MEASURED rows only: an [INFERRED] row is allowed to name the cookbook's shape
    # and let build_llama_cmd clamp it downward on a card that cannot hold it. A
    # MEASURED row is emitted exactly as measured and so must be right here.
    a8bad = [r.key for r in RECIPES if r.status == "MEASURED"
             and r.ub > min(ub_for_card(11264 if c == 61 else 16384) for c in r.caps)]
    ok_all &= not a8bad
    print(f"  A8 no row exceeds the card-type -ub ceiling on its own smallest card: "
          f"{'PASS' if not a8bad else 'FAIL ' + str(a8bad)}")
    # A9: the Flash-Next row emits NO PXA_* env (2026-09-06): the eleven published
    #     levers are the engine's own ENHANCE defaults on 4x sm_60, and the launcher
    #     must not pin them. The documentation dict still names exactly eleven.
    fn = [r for r in RECIPES if r.key == "4xp100-flashnext"]
    a9 = bool(fn) and not fn[0].env and len(ENGINE_DEFAULTED_FLASHNEXT_LEVERS) == 11
    ok_all &= a9
    print(f"  A9 4xp100-flashnext emits no kernel-lever env (11 engine-defaulted levers documented): "
          f"{'PASS' if a9 else 'FAIL'}")
    print(f"  standing assertions: {'ALL PASS' if ok_all else 'FAILURES ABOVE'}")


# ---------------------------------------------------------------------------

def plan_and_build(a, gpus):
    """Everything between "which cards, which model" and "here is the command".

    Split out of main() so that the terminal UI can run the IDENTICAL code path and
    show its output, instead of reimplementing the decision and drifting from it.
    Prints the full never-magic report to stdout (the UI captures it), and returns
    (plan, cmd, env, cv, prof, ctx). Refusals still exit, as they always did."""
    # ---- card selection (I-12: never left to the ambient environment) -------
    cards = {int(x) for x in re.split(r"[,\s]+", a.gpus) if x.strip()} if a.gpus else set()
    sel = [g for g in (gpus or []) if g[0] in cards] if cards else (gpus or [])
    if cards and len(sel) != len(cards):
        missing = sorted(cards - {g[0] for g in sel})
        print(f"pxa-launch: --gpus asked for {missing} which are not visible", file=sys.stderr)
        sys.exit(2)

    kind = model_kind(a.model)
    prof = model_profile(a.model, kind)
    if a.tier:
        prof["tier"] = a.tier
        prof["tier_src"] = "ASSERTED by --tier (you own this assertion)"
    mbytes = model_bytes(a.model, kind)
    workload = infer_workload(a.np or 1, a.workload)
    a.workload = workload
    _engine_dir_for_help, _ = resolve_engine_dir()
    if getattr(a, "ask_chat", False) and kind in ("gguf", "gguf_broken") \
            and not prof.get("hdr_err"):
        pick_chat_options(prof, a, _engine_dir_for_help)
        print()

    # ---- recipe-supplied defaults ------------------------------------------
    # Resolved here as well as inside decide() so that -c / -np / -t can take the
    # row's measured values. An explicit flag always wins, and the substitution is
    # printed, so no value in the emitted command is ever unexplained.
    rec0, rstat0, _ev0, _nt0 = recipe_for(sel, prof, workload)
    row_defaults = []
    if a.np < 1:
        if rec0 is not None and rec0.np:
            a.np = rec0.np
            row_defaults.append(f"-np {a.np}")
        else:
            a.np = 1
    if not a.threads and rec0 is not None and rec0.threads:
        a.threads = rec0.threads
        row_defaults.append(f"-t {a.threads}")
    per_slot = ANCHOR_CTX_PER_SLOT
    if a.ctx:
        ctx = a.ctx
        per_slot = max(1, a.ctx // max(1, a.np))
    elif rec0 is not None and rec0.ctx:
        ctx = rec0.ctx
        per_slot = max(1, ctx // max(1, a.np))
        row_defaults.append(f"-c {ctx}")
    else:
        ctx = a.np * ANCHOR_CTX_PER_SLOT
    threads = a.threads or (os.cpu_count() or 16)
    a.emit_threads = bool(a.threads)   # only pass -t when asked for, or when a row sets it
    a.threads = threads

    if a.np < 1:
        print("=" * 78)
        print("pxa-launch: ENGINE = None")
        print(f"  REFUSING [R-26] " + R["R-26"].format(n=a.np, ctx=a.np * ANCHOR_CTX_PER_SLOT))
        print("=" * 78)
        sys.exit(2)

    elig_caps, image, probe_trail = vllm_eligibility(sel, a.vllm_image)

    plan = decide(sel, kind, a.model, a.engine, prof, a.np, workload,
                  elig_caps, image, probe_trail, per_slot)

    # ---- report ------------------------------------------------------------
    print("=" * 78)
    print(f"pxa-launch: ENGINE = {plan.engine}")
    size = f", {mbytes / BYTES_PER_GIB:.2f} GiB" if mbytes else ""
    print(f"  model:  {a.model}  [{kind}{size}]")
    if prof["arch"] or prof["n_expert"] or prof["tier"] or prof["hist"]:
        cls = "MoE" if prof["is_moe"] else "dense"
        line = f"  class:  {cls}"
        if prof["is_moe"]:
            line += f" ({prof['n_expert']} experts)"
        if prof["arch"]:
            line += f", arch={prof['arch']}"
        print(line + f"  [{prof['why']}]")
        print(f"  tier:   {prof['tier'] or 'not a PXQ file'}"
              + (f" (provenance KV says {prof['tier_kv']})" if prof.get("tier_kv") else "")
              + f"  [{prof['tier_src']}]")
        if prof["hist"]:
            print(f"  compose:{compose_str(prof['hist'])}")
        if prof["n_ctx_train"]:
            print(f"  trained ctx: {prof['n_ctx_train']}")
        if prof["mtp_tensors"] >= 0:
            print(f"  mtp:    {prof['mtp_tensors']} nextn/mtp tensors"
                  + (f"; KV nextn_predict_layers={prof['mtp_kv']}" if prof["mtp_kv"] is not None
                     else "")
                  + f"  [{prof['mtp_src']}]")
        print(f"  deltanet: {prof['deltanet']}  [{prof['deltanet_src']}]")
    else:
        print(f"  class:  UNKNOWN  [{prof['why']}]")
    print(f"  serve:  np={a.np}, workload={workload}, ctx={ctx} (={per_slot}/slot), "
          f"threads={threads}")
    print(f"  cards:  " + (", ".join(f"{g[0]}:{g[1].replace('NVIDIA ', '')} sm_{g[2]} "
                                     f"{g[3]/1024:.0f}GiB" for g in sel) or "none"))
    print(f"  recipe: {plan.recipe.key if plan.recipe else 'none'}"
          + (f" ({plan.recipe.title})" if plan.recipe else "")
          + f"  [{plan.rstatus}]")
    if row_defaults:
        print(f"  from the row (you did not set these): {', '.join(row_defaults)}")
    if plan.reason:
        print(f"  reason: {plan.reason}")
    for e in plan.evidence:
        print(f"  ev:     {e}")

    # ---- refusals gathered during the decision ------------------------------
    # H4 / R-20: a shared, live box. Do this AFTER the decision so the operator
    # still sees what the launcher would have chosen.
    if sel and not a.allow_busy:
        procs, ok = resident_procs(gpus or [])
        for g in sel:
            if procs.get(g[0]):
                plist = ", ".join(f"pid {p} {n} {m} MiB" for p, n, m in procs[g[0]])
                plan.refuse("R-20", i=g[0], mib=g[4], procs=plist)
            elif g[4] > 512:
                plan.refuse("R-20", i=g[0], mib=g[4],
                            procs="no compute app listed, but >512 MiB is resident")

    if plan.engine is None and not plan.refusals:
        print("=" * 78)
        sys.exit(2)

    for n in plan.notes:
        print(f"  ** {n}")
    for b in plan.blockers:
        print(f"  !! {b}")

    # ---- the correctness settings, always printed, never assumed -----------
    plan.chat = chat_defaults(prof, a, _engine_dir_for_help)
    print("  --- chat / serving settings (the ones people forget) ---")
    print_chat_summary(plan.chat, prof, indent="  ")

    # ---- parameter translation refusals (R-10..R-16, R-25, R-27) -----------
    if plan.engine == "vllm":
        if a.ts:
            plan.refuse("R-10", code=3)
        if a.sm and a.sm != "layer":
            plan.refuse("R-11", code=3, sm=a.sm)
        if (a.ctk, a.ctv) != ("f16", "f16"):
            plan.refuse("R-13V", code=3)
        m, _ = parse_spec(a.spec)
        if m == "mtp":
            plan.refuse("R-14", code=3)
        if a.draft_model:
            plan.refuse("R-25", code=3)
        if prof.get("vision") and not a.accept_unmeasured:
            plan.refuse("R-27", code=3)
        if a.np > MOE_TABLE_MAX_NP and not a.accept_unmeasured:
            plan.refuse("R-21", code=3, n=a.np)
    if plan.engine == "llama":
        if a.sm == "graph":
            why = None
            if prof.get("deltanet"):
                why = prof.get("deltanet_src", "linear-attention tensors present")
            elif prof.get("arch") in GRAPH_SPLIT_GUARDED_ARCHES:
                why = f"arch '{prof['arch']}' is on the guarded list (tools/pxa-launch.py)"
            if why:
                plan.refuse("R-12", code=3, why=why)
        if (a.ctk, a.ctv) not in COMPILED_CTKV_PAIRS:
            plan.refuse("R-13L", code=3, k=a.ctk, v=a.ctv)
        if a.draft_model and not a.accept_unmeasured:
            plan.refuse("R-25", code=3)
    # spec / MTP gates apply on both engines
    m, params = parse_spec(a.spec)
    if m == "mtp":
        nmax = int(params.get("n_max", 0) or 0)
        if nmax == 0:
            # I-8: a bare 'mtp' used to expand to n_max=4,n_min=2 - a MEASURED loss
            # on both arches, emitted by DEFAULT. It now expands to n_max=1 only.
            a.spec = "mtp:n_max=1"
            print("  --spec mtp expanded to mtp:n_max=1. MEASURED: n_max>=2 LOSES on both arches "
                  "(P100 54.9->47.4 accept 0.42; V100 92.7 vs 94.1 accept 0.960->0.480, "
                  "LEVERS.md:300-301). The previous version of this file expanded a bare 'mtp' to "
                  "n_max=4,n_min=2 - a measured loss, by default.")
            nmax = 1
        if nmax >= 2:
            plan.refuse("R-15A", code=3, n=nmax)
        if prof.get("mtp_tensors") == 0:
            plan.refuse("R-15B", code=3, kvn=prof.get("mtp_kv"))
        elif prof.get("mtp_tensors", 0) > 0:
            print("  MTP: PXA_MTP_LAZY_WARMUP is armed by PXA_ENHANCE and is MANDATORY whenever "
                  "MTP is active - without it MTP costs -33% prefill (LEVERS.md:152, :402). "
                  "PXA_MOE_FASTTG_MAX_NY is left at its shipped 8; =1 with MTP verify measured "
                  "48.1 -> 30.3 on P100 (LEVERS.md:409).")
            if prof.get("is_moe"):
                print("  MTP on a sparse MoE is a MEASURED LOSS even at n_max=1: -8.6% (n_max=1), "
                      "-29.8% (n_max=2) despite 0.800 acceptance (LEVERS.md:746). You asked for "
                      "it; it is emitted; the number is against you.")
    if prof.get("tier") in NO_CPU_CODEC and a.ngl < 99:
        plan.refuse("R-16", code=3, tier=prof["tier"], ngl=a.ngl)
    if plan.engine == "vllm" and a.cudagraph_mode != "FULL_DECODE_ONLY":
        plan.refuse("R-08", code=3, mode=a.cudagraph_mode)
    if prof.get("n_ctx_train") and ctx > int(prof["n_ctx_train"]):
        plan.refuse("R-17A", ctx=ctx, trained=prof["n_ctx_train"], arch=prof.get("arch"))

    # ---- VRAM (only formula-free facts may block; see SPEC CORRECTION C4) ---
    for n in vram_check(plan, sel, mbytes, ctx, prof, a.ngl >= 99):
        print(f"  ** {n}")

    # ---- mmproj resolution --------------------------------------------------
    mmproj = a.mmproj
    if plan.engine == "llama" and not mmproj and a.no_mmproj:
        # --no-mmproj SUPPRESSES resolution, so this branch emits no projector. It still
        # has to SAY that: "REFUSES rather than silently dropping" cuts both ways, and an
        # operator who suppresses a projector on a model that has three sitting beside it
        # deserves to see that the flag was read and what it turned off. Previously this
        # path printed nothing at all - the flag and the candidates both vanished.
        cands, _why = find_mmproj(a.model, kind)
        if cands:
            print(f"  mmproj: SUPPRESSED by --no-mmproj. {len(cands)} projector(s) are "
                  f"sitting next to this model and none will be attached:")
            for c in cands:
                print(f"            {c}")
            print("          This seat serves TEXT-ONLY. Drop --no-mmproj, or pass "
                  "--mmproj <path>, to take images.")
        elif prof.get("vision"):
            print("  mmproj: SUPPRESSED by --no-mmproj although this model carries vision "
                  "tensors. Serving TEXT-ONLY.")
        else:
            print("  mmproj: --no-mmproj accepted; nothing to suppress (no projector beside "
                  "this model, no vision tensors in it). Serving TEXT-ONLY either way.")
    if plan.engine == "llama" and not mmproj and not a.no_mmproj:
        cands, why = find_mmproj(a.model, kind)
        if prof.get("vision"):
            # The model itself carries vision tensors: a projector is part of the seat.
            if len(cands) == 1:
                mmproj = cands[0]
                print(f"  mmproj: {mmproj}  [exactly one candidate; {why}]")
            elif len(cands) > 1:
                plan.refuse("R-24", code=3, n=len(cands), list="\n      ".join(cands))
            else:
                print("  mmproj: NONE found next to this model although it carries vision "
                      "tensors. Serving TEXT-ONLY. Pass --mmproj to enable vision.")
        elif cands:
            # No vision tensors in the model file, but projectors sit beside it (the
            # muse-glimmer case: two F16 projectors in one dir, an F16 and a Q8_0 in
            # another, for the same base model). Nothing ranks them and nothing says
            # this model wants one, so NOTHING is attached - and the operator is told
            # what is there rather than left to wonder.
            print(f"  mmproj: NOT attached. {len(cands)} projector(s) sit next to this model "
                  f"but it carries no vision tensors and no measurement ranks them:")
            for c in cands:
                print(f"            {c}")
            print("          Pass --mmproj <path> if this seat is meant to take images.")

    if plan.refusals:
        print("  REFUSING:")
        for rid, text, _ in plan.refusals:
            print(f"    [{rid}] {text}")
        print("=" * 78)
        sys.exit(max(c for _, _, c in plan.refusals))

    if plan.needs_ack and not a.accept_unmeasured:
        print("  REFUSING to execute an UNMEASURED branch without acknowledgement:")
        for r in plan.needs_ack:
            print(f"    - {r}")
        print("    Re-run with --accept-unmeasured to proceed anyway. The plan above is the plan;")
        print("    the refusal is about executing it, not about printing it.")
        print("=" * 78)
        sys.exit(5 if a.explain else 3)

    # ---- build the command --------------------------------------------------
    ub_expect = sorted({ub_for_card(g[3]) for g in sel}) if sel else ["n/a"]
    if plan.engine == "llama":
        cmd, env = build_llama_cmd(plan, a, sel, prof, ctx, ub_expect, mmproj,
                                   explain=a.explain)
        used = sel
    else:
        used = plan.elig or sel
        if not a.gmu:
            cc = sorted({g[2] for g in used})[0] if used else 60
            smallest_mib = min((g[3] for g in used), default=16384)
            a.gmu = 0.90 if cc == 60 else 0.85
            why = ("0.90 = the MEASURED sm_60 arm" if cc == 60
                   else "0.85 = the healthy live sm_70 container")
            # 16 GiB CARDS RUNNING A LARGE RESIDENT MODEL (pxq23, measured 2026-09-05 on a
            # P100 serving Fusion2-35B PXQ2, 12.23 GiB resident). At 0.96 the allocator was
            # left 20 MiB of headroom and the PASCAL_SDPA prefill OOM'd at 6.5k tokens --
            # after the server had come up healthy and served short requests, which is the
            # worst shape of failure: it looks like a working seat until someone sends a long
            # prompt. 0.88 gave 1.87 GiB of KV cache (69,259 tokens) and a stable server on
            # the same card and model. The margin is what the prefill peak needs, not slack.
            if smallest_mib <= 17 * 1024 and a.gmu > 0.88:
                a.gmu = 0.88
                why = (f"0.88 = the 16 GiB ceiling (smallest card {smallest_mib} MiB); "
                       f"higher values leave too little for the prefill peak and OOM AFTER "
                       f"the server comes up healthy")
            print(f"  --gpu-memory-utilization {a.gmu}: recipe value for sm_{cc} "
                  f"({why}). NEVER SWEPT -> UNMEASURED as a tuning axis.")
        cmd, env, used = build_vllm_cmd(plan, a, prof, ctx, used, image)

    cv = ",".join(str(g[0]) for g in used)
    # I-12 / R-07: devices are NEVER left to the ambient environment. If we cannot
    # name the devices, we do not execute. The previous version turned an empty
    # device list into the string "all" and then skipped setting
    # CUDA_VISIBLE_DEVICES entirely, so the child inherited every GPU on a box with
    # six cards mid-measurement and a production VLM on card 3.
    if not cv:
        print("  REFUSING to execute with an unscoped device set: no card could be named for "
              "CUDA_VISIBLE_DEVICES. On this box that means inheriting every GPU, including "
              "cards other agents are measuring on. (I-12)")
        print("=" * 78)
        sys.exit(3)
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    print(f"  env:     CUDA_VISIBLE_DEVICES={cv} {envs}")
    if plan.engine == "vllm":
        # THE IMAGE IS A PREMISE, NOT AN INSTRUCTION. The emitted command is a bare
        # `vllm serve` - there is no `docker run` in it and the image name appears
        # nowhere. --vllm-image (and PXA_VLLM_IMAGE) decide only WHICH CARDS ARE
        # ELIGIBLE; the process then runs in whatever container this launcher is
        # already inside. Name that out loud, because the flag reads like it selects a
        # runtime, and a reader who believes it will attribute a measurement to an
        # image that was never involved.
        img = os.environ.get("PXA_VLLM_IMAGE") or getattr(a, "vllm_image", None)
        if img:
            print(f"  CONTAINER CONTRACT: eligibility was decided against {img!r}, and this "
                  f"command is NOT run in it.")
            print(f"    The command below execs HERE. Run this launcher inside {img} - or "
                  f"accept that the seat you get is whatever this container holds, which "
                  f"is not what the decision above was based on.")
            he = (VLLM_IMAGES.get(img) or {}).get("host_env") or {}
            if he:
                print(f"    {img} is NOT self-contained: its python, torch and vllm are on "
                      f"the HOST. It needs these mounts, or nothing starts:")
                for hsrc, hdst in he.get("mounts", {}).items():
                    print(f"      -v {hsrc}:{hdst}")
                print(f"      [{he.get('why', 'host dependency')}]")
                es = he.get("editable_source")
                if es:
                    import subprocess as _sp
                    try:
                        br = _sp.run(["git", "-C", es, "rev-parse", "--abbrev-ref", "HEAD"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                        sha = _sp.run(["git", "-C", es, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True, timeout=10).stdout.strip()
                        dirty = _sp.run(["git", "-C", es, "status", "--porcelain"],
                                        capture_output=True, text=True, timeout=15).stdout.strip()
                    except Exception:
                        br = sha = ""; dirty = ""
                    print(f"    vllm is an EDITABLE install of {es} - the seat imports that "
                          f"WORKING TREE, not a built artifact.")
                    if br or sha:
                        print(f"      right now that tree is {br} @ {sha}"
                              + ("  *** WITH UNCOMMITTED CHANGES ***" if dirty else ""))
                    print(f"      Editing or switching branches there changes what this seat "
                          f"serves, with no redeploy and no version to notice it by.")
        else:
            print("  CONTAINER CONTRACT: no image was named, so eligibility came from the "
                  "importable vllm_pxq4 in THIS interpreter. `vllm serve` execs here.")
    print(f"  command: {' '.join(cmd)}")
    print_post_boot_contract(plan.engine, cv)
    return plan, cmd, env, cv, prof, ctx


def main():
    ap = argparse.ArgumentParser(
        prog="pxa-launch",
        description="Pick your cards and your model by number; the launcher picks the engine, "
                    "the flags and the env, tells you why, and starts the server. "
                    "Run it with no arguments the first time.")
    # ---- the two questions a user actually has an answer to ----------------
    ap.add_argument("--model", default="",
                    help="path to a .gguf (or a converted vLLM directory). Omit to pick from a "
                         "numbered list of the models found under the search roots.")
    ap.add_argument("--gpus", "--cards", dest="gpus", default="", metavar="N,N",
                    help="GPU INDEXES as nvidia-smi reports them, e.g. --gpus 2,4. Omit to pick "
                         "from a numbered list. (--cards is the old name and still works.)")
    ap.add_argument("--models-dir", dest="models_dir", action="append", default=[],
                    metavar="DIR",
                    help="where to look for models. Repeatable. Also read from PXA_MODELS_DIR "
                         "(%s-separated), then the directory of your previous launch, then %s."
                         % (os.pathsep, ", ".join(DEFAULT_MODEL_ROOTS)))
    ap.add_argument("--workload", choices=["chat", "serve", "longdoc"], default=None,
                    help="chat/serve -> -fa on; longdoc -> -fa off. Default: asked "
                         "interactively, else chat when --np<=1 and serve above it.")
    # ---- scripted use: every one of these was here before and still works ---
    ap.add_argument("--engine", choices=["llama", "vllm"])
    ap.add_argument("-c", "--ctx", type=int, default=0,
                    help="TOTAL context. Default: the matched recipe row's ctx, else np * 4096.")
    ap.add_argument("--np", type=int, default=0,
                    help="concurrent slots. Default: the recipe row's value, else 1.")
    ap.add_argument("--ub", type=int, default=0,
                    help="force -ub (and -b, unless --b is given too). Default 0 = take the "
                         "measured recipe cell, or let adaptive-ub choose where none exists.")
    ap.add_argument("--b", type=int, default=0, dest="b",
                    help="force the prefill chunk -b. Only meaningful with --ub.")
    ap.add_argument("--threads", type=int, default=0, help="default: recipe row, else host cores")
    ap.add_argument("--ngl", type=int, default=999)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--spec", default="")
    ap.add_argument("--draft-model", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--sm", default="layer")
    ap.add_argument("--ctk", default="f16")
    ap.add_argument("--ctv", default="f16")
    ap.add_argument("--mmproj", default="")
    ap.add_argument("--no-mmproj", action="store_true")
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--gmu", type=float, default=0.0,
                    help="vLLM --gpu-memory-utilization. Default: 0.90 sm_60 / 0.85 sm_70 "
                         "(recipe values, NEVER swept -> UNMEASURED as a tuning axis)")
    ap.add_argument("--vllm-mm", action="store_true",
                    help="allow image/video input on a multimodal checkpoint. Off by default: "
                         "the encoder cache and vision profiling cost roughly a gigabyte, "
                         "which on a 16 GiB card is the whole KV budget. See _is_multimodal.")
    ap.add_argument("--vllm-block-size", default="", metavar="N",
                    help="vLLM --block-size. Default: unset, except a GDN hybrid on sm_70, "
                         "where 256 is the measured seat value (board A4). Pass a number to "
                         "override it, or 0 to emit no --block-size at all.")
    ap.add_argument("--vllm-image", default="")
    ap.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    ap.add_argument("--tier", default="", help="assert the PXQ tier yourself (R-03 escape)")
    # ---- the things people forget -----------------------------------------
    ap.add_argument("--chat-template", default="", metavar="NAME",
                    help="override the template inside the GGUF with a built-in one. "
                         "--list-chat-templates prints the names this engine accepts.")
    ap.add_argument("--chat-template-file", default="", metavar="PATH",
                    help="override it with a .jinja file")
    ap.add_argument("--list-chat-templates", action="store_true",
                    help="print the built-in chat template names and where the list came from")
    ap.add_argument("--no-jinja", action="store_true",
                    help="do NOT pass --jinja. Read the warning it prints first: without it, a "
                         "request carrying `tools` returns HTTP 500 while plain chat still works")
    ap.add_argument("--reasoning-format", default="", choices=[""] + REASONING_FORMATS,
                    help="how thinking tags come back. Unset = the engine's own default")
    ap.add_argument("--reasoning-budget", type=int, default=None, metavar="N",
                    help="cap thinking tokens (0 = no thinking, -1 = unlimited)")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--api-key", default="",
                    help="require this key on every request. Never printed and never written "
                         "into a --serve-name script.")
    ap.add_argument("--slot-save-path", default="", metavar="DIR",
                    help="enable /slots save+restore into this directory (the release gate "
                         "uses it)")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--accept-unmeasured", action="store_true")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    # ---- the rerunnable seat ------------------------------------------------
    ap.add_argument("--serve-name", default="", metavar="NAME",
                    help="also write an executable restart script for this exact seat. No "
                         "systemd, no daemon: a shell script you can rerun.")
    ap.add_argument("--serve-dir", default="",
                    help="where --serve-name writes. Default $PXA_SERVE_DIR, else %s."
                         % os.path.join(STATE_DIR, "serve"))
    ap.add_argument("--yes", "-y", action="store_true",
                    help="do not ask for confirmation before launching")
    ap.add_argument("--no-interactive", action="store_true",
                    help="never prompt; fail instead if --model or --gpus is missing")
    ap.add_argument("--no-tui", action="store_true",
                    help="skip the full-screen terminal UI and use the plain prompts "
                         "(same thing as setting PXA_NO_TUI=1)")
    a = ap.parse_args()

    if a.list_chat_templates:
        E, _n = resolve_engine_dir()
        names, prov = builtin_chat_templates(E)
        print(f"built-in chat template names ({len(names)}), {prov}:")
        for i in range(0, len(names), 6):
            print("   " + "  ".join(f"{n:<20}" for n in names[i:i + 6]))
        print("\nNone of these is needed when the GGUF carries its own template - which is the "
              "default, and the better answer. Use one only when the file has none, or when "
              "yours is broken.")
        return

    gpus, err = gpu_table()
    if err:
        print(f"pxa-launch: {err}", file=sys.stderr)
        if not a.engine and not a.selftest:
            sys.exit(2)
        gpus = []
    if a.selftest:
        selftest(gpus or [])
        return

    # ---- THE FRONT DOOR ----------------------------------------------------
    # Nothing here decides anything. It turns "which of these" into the same
    # --gpus/--model a scripted caller would have passed, and then the identical
    # code path runs. There is no interactive-only behaviour below this block.
    procs, _procs_ok = resident_procs(gpus or [])
    want_prompt = (not a.gpus or not a.model or a.workload is None)
    if want_prompt and not a.no_interactive and not a.no_tui and gpus:
        ok, why = tui_available()
        if ok:
            try:
                _tui_result = LaunchTUI(a, gpus, procs).run()
            except Exception as ex:
                # A UI failure must never cost the user their answer. Say what broke,
                # keep whatever was already chosen, and drop to the line prompts.
                print(f"pxa-launch: the terminal UI stopped ({ex.__class__.__name__}: {ex}). "
                      f"Falling back to the plain prompts.", file=sys.stderr)
            else:
                if _tui_result == "abort" and not (a.model and a.gpus):
                    print("pxa-launch: nothing chosen, nothing started.")
                    return
                if a.model and a.gpus:
                    # The UI already showed and (if asked) ran this seat. Print the
                    # same plan to the scrollback so the decision survives the
                    # full-screen session, then stop - it is not started twice.
                    a.explain = True
                    a.ask_chat = False
                    print()
                    print("The plan from that session, for your scrollback "
                          "(the server is not started again here):")
                    try:
                        plan_and_build(a, gpus)
                    except SystemExit:
                        pass
                    print("Rerun it any time with:")
                    print(f"  {sys.argv[0]} --gpus {a.gpus} --model {a.model} "
                          f"--workload {a.workload or 'chat'} --yes")
                return
        else:
            print(f"pxa-launch: full-screen UI not used ({why}); using the plain prompts.",
                  file=sys.stderr)
    if want_prompt and not a.no_interactive and interactive_available():
        print("=" * 78)
        print("pxa-launch - pick your cards and your model. The launcher does the rest.")
        print("            Everything it chooses is printed with the measurement behind it")
        print("            before anything starts. Ctrl-C is always safe here.")
        print("=" * 78)
        if not a.gpus:
            if not gpus:
                print("pxa-launch: no GPUs visible; nothing to pick from.", file=sys.stderr)
                sys.exit(2)
            a.gpus = ",".join(str(x) for x in pick_cards(gpus, procs))
            print()
        if not a.model:
            roots, trail = model_roots(a.models_dir)
            entries, mnotes = scan_models(roots)
            a.model = pick_model(entries, trail, mnotes)
            print()
        if a.workload is None:
            a.workload = pick_workload()
            print()
        a.ask_chat = True
    if not a.model:
        print("pxa-launch: no model. Pass --model /path/to/file.gguf, or run with no arguments "
              "in a terminal to pick one from a list.", file=sys.stderr)
        sys.exit(2)

    plan, cmd, env, cv, prof, ctx = plan_and_build(a, gpus)

    if a.serve_name:
        try:
            sp = write_serve_script(a.serve_name, cmd, env, cv, a.serve_dir)
            print(f"  restart script: {sp}")
            print( "     It is this command frozen, with the same busy-card check the launcher "
                   "does. It does NOT call the launcher again - re-run pxa-launch when you want "
                   "a fresh decision.")
        except OSError as ex:
            print(f"  !! could not write the restart script: {ex}")
    print("=" * 78)

    if a.explain:
        # exit 5 => a plan was produced but carries known-fatal blockers, so a CI
        # caller can tell "clean plan" from "plan that will not start".
        sys.exit(5 if plan.blockers else 0)
    if not shutil.which(cmd[0]) and not os.path.exists(cmd[0]):
        print(f"pxa-launch: {cmd[0]} not found", file=sys.stderr)
        sys.exit(4)
    if not a.yes and interactive_available():
        # Asked ONLY where a human is watching. A scripted caller never sees this
        # prompt, so adding it cannot break an existing pipeline.
        if not confirm("Start this server now?"):
            print("  not started. The command above is complete - copy it, or rerun with --yes.")
            sys.exit(0)
    st = _load_state()
    st["last_model_dir"] = os.path.dirname(os.path.abspath(a.model)) \
        if os.path.isfile(a.model) else os.path.abspath(a.model)
    st["last_command"] = cmd
    st["last_cards"] = cv
    _save_state(st)
    e = dict(os.environ)
    e.update(env)
    e["CUDA_VISIBLE_DEVICES"] = cv
    e["NVIDIA_VISIBLE_DEVICES"] = cv
    # FLUSH BEFORE EXEC, OR THE WHOLE POINT OF THIS FILE IS LOST.
    # execve replaces the process image and DISCARDS whatever is still sitting in
    # Python's stdout buffer. On a terminal that is invisible (line-buffered, so it
    # has already gone out); into a pipe or a file - a container log, a CI capture,
    # `pxa-launch ... | tee` - stdout is block-buffered at 8 KiB and the tail of the
    # plan simply never appears. MEASURED 2026-09-04: a real container log of a
    # successful boot ends mid-way through the notes, with the `env:` and `command:`
    # lines missing, while the same run's refusal path (which exits instead of
    # exec'ing, and so flushes) printed them fine. A launcher whose promise is "it
    # prints the decision and the command before it runs" cannot lose that text on
    # the one path where it actually runs.
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(cmd[0], cmd, e)


if __name__ == "__main__":
    main()
