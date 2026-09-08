#!/bin/bash
# pxa-profile.sh — reproducible per-card kernel-wall breakdown for PXQ MoE inference.
#
# Uses the fork's BUILT-IN op profiler (PXA_PROFILE=1) — no external nsys needed, so anyone
# with the fork binary + a public PXQ GGUF reproduces the same table. Emits two regimes:
#   DECODE  (short prompt, many gen tokens)  -> bandwidth/launch-bound op mix
#   PREFILL (long prompt, 1 gen token)       -> compute-bound op mix
#
# Reproduce (public models on HF poisonxa/PXA-Fusion2-35B-GGUF):
#   single 11-16 GB card:  MODEL=PXA-Fusion2-35B-PXQ2.gguf  (10.7 GB)  TS=""     NGL=99
#   2x16 GB (V100 pair):   MODEL=PXA-Fusion2-35B-PXQ4.gguf  (18.7 GB)  TS="1,1"  NGL=99
#
# Usage: GPUS=<uuid[,uuid]> MODEL=/path/to.gguf [TS=1,1] [BUILD=/path] ./pxa-profile.sh <tag>
#   BUILD defaults to <repo>/build (the tree this script lives in); OUT defaults to ./profile-<tag>.txt.
set -uo pipefail
TAG="${1:-run}"
GPUS="${GPUS:?set GPUS=GPU-uuid[,GPU-uuid]}"
MODEL="${MODEL:?set MODEL=/abs/path/to/pxq.gguf}"
TS="${TS:-}"
NGL="${NGL:-99}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || dirname "$HERE")"
BUILD="${BUILD:-$REPO/build}"
IMG="${IMG:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"
OUT="${OUT:-./profile-$TAG.txt}"
mkdir -p "$(dirname "$OUT")"
MDIR="$(dirname "$MODEL")"; MBASE="$(basename "$MODEL")"
tsflag=""; [ -n "$TS" ] && tsflag="-ts $TS"

run() { # $1 label  $2 prompt-tokens(approx via repeats)  $3 n_gen
  local label="$1" reps="$2" ngen="$3"
  local prompt; prompt=$(yes "the mixture of experts router dispatches tokens to specialist experts and the scheduler overlaps weight streaming with computation ." | head -n "$reps" | tr '\n' ' ')
  echo "======== PROFILE $TAG / $label (approx ${reps}0 prompt toks, gen $ngen) ========" | tee -a "$OUT"
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
    -e PXA_PROFILE=1 -e PXA_PROFILE_EVERY="${4:-4000}" \
    -e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd \
    -v "$BUILD":/build:ro -v "$MDIR":/models:ro "$IMG" \
    /build/bin/llama-cli -m "/models/$MBASE" -p "$prompt" -n "$ngen" --temp 0 \
    -c 8192 -ngl "$NGL" $tsflag -fa on -b 2048 -ub 2048 --no-display-prompt --simple-io \
    2>&1 | grep -E "PXA_PROFILE|top name buckets|us=|tokens per second|^  [A-Z_]" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

: > "$OUT"
echo "== pxa-profile $TAG  model=$MBASE  gpus=$GPUS  $(date -u +%FT%TZ) ==" | tee -a "$OUT"
# DECODE regime: tiny prompt, many gen -> decode ops dominate the accumulated profile
run "DECODE"  1  400  8000
# PREFILL regime: long prompt, 1 gen -> prefill ops dominate
run "PREFILL" 600 1   50000
echo "== done: top-op %% is the wall breakdown; MUL_MAT_ID / FUSED ops = expert GEMM, FLASH_ATTN = attention ==" | tee -a "$OUT"
