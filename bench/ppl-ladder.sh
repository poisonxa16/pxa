#!/bin/bash
# Perplexity ladder for the PXA Fusion2-35B PXQ tiers — the exact protocol behind the published table.
#   PXQ4 7.3563 ±0.0818 | PXQ3 7.4407 ±0.0830 (+1.1%) | PXQ2 8.3906 ±0.0961 (+14.1%)
# (PXQ4 = the 4-bit flagship, formerly PXQ6; the published file keeps the PXQ6 filename)
# Protocol: wikitext-2-raw TEST split, n_ctx=512, 200 chunks, fa on, f16 KV, b/ub 512.
# Usage: MODELS_DIR=/path/to/ggufs [BUILD=./build] [NGL=99] ./ppl-ladder.sh
set -eu
BUILD=${BUILD:-./build}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR to the directory holding PXA-Fusion2-35B-*.gguf}
NGL=${NGL:-99}

# corpus: wikitext-2-raw (test split). Fetched once, cached.
if [ ! -f wikitext-2-raw/wiki.test.raw ]; then
  curl -fL -o wikitext-2-raw-v1.zip \
    https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
  unzip -o wikitext-2-raw-v1.zip
fi

export LD_LIBRARY_PATH="$BUILD/bin:$BUILD/src:$BUILD/ggml/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# PXQ format families + the bit-exact fast kernels (all memcmp-gated, see determinism-gates.md)
export PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1
export PXA_PXQ6_KSPLIT=1 PXA_PXQ6_VECX=1 PXA_PXQ6_GUFUSE=1 PXA_PXQ6_SCATFUSE=1 PXA_PXQ6_RAGTAIL=1

for TIER in PXQ6 PXQ3 PXQ2; do
  M="$MODELS_DIR/PXA-Fusion2-35B-$TIER.gguf"
  [ -f "$M" ] || { echo "SKIP $TIER (no $M)"; continue; }
  echo "=== $TIER ==="
  "$BUILD/bin/llama-perplexity" -m "$M" -f wikitext-2-raw/wiki.test.raw \
    -c 512 --chunks 200 -ngl "$NGL" -fa on -ctk f16 -ctv f16 -b 512 -ub 512 \
    2>&1 | grep -E "Final estimate|perplexity:"
done
# Notes:
# - A single 16 GB card runs the 4-bit flagship only partially (-ngl < 99); its published number used
#   2 cards with -sm layer -ts 1,1. PXQ3/PXQ2 fit one 16 GB / 11 GB card at -ngl 99.
# - Expect Final estimates to match the table above to within the printed +/- error.
