#!/bin/bash
# KL-divergence of each PXQ tier vs the bf16 merge source — the strongest "quantization damage"
# measure (isolates quant loss from model quality; see llama.cpp discussion #4110).
#
# Two passes:
#   1) dump reference logits from the bf16 GGUF (needs ~140 GB RAM *or* any GPU rig that fits
#      bf16; CPU works — bf16 has full CPU kernels, PXQ tiers do NOT and need CUDA)
#   2) score each PXQ tier against the dump (GPU, same protocol as the ppl ladder)
#
# ⚠ Base-file size: ~2 bytes x n_vocab (248k) per token ≈ 0.5 MB/token.
#   200 chunks x 512 tok ≈ 50 GB; CHUNKS=100 halves it (KLD converges fast — 100 is plenty).
#
# Usage:
#   BF16=/path/pxa-35b-ornith-siq-bf16.gguf MODELS_DIR=/path/ggufs [CHUNKS=100] ./kld.sh
set -eu
BUILD=${BUILD:-./build}
BF16=${BF16:?path to the bf16 source gguf}
MODELS_DIR=${MODELS_DIR:?dir holding PXA-Fusion2-35B-*.gguf}
CHUNKS=${CHUNKS:-100}
BASE=${BASE:-kld-base-fusion2-bf16.bin}

[ -f wikitext-2-raw/wiki.test.raw ] || { curl -fL -o wikitext-2-raw-v1.zip \
  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip; unzip -o wikitext-2-raw-v1.zip; }

export LD_LIBRARY_PATH="$BUILD/bin:$BUILD/src:$BUILD/ggml/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1
export PXA_PXQ6_KSPLIT=1 PXA_PXQ6_VECX=1 PXA_PXQ6_GUFUSE=1 PXA_PXQ6_SCATFUSE=1 PXA_PXQ6_RAGTAIL=1

# Pass 1 — reference logits (CPU ok: add -ngl 0 -t $(nproc); on a CUDA build without a GPU,
# point LD_LIBRARY_PATH at the CUDA stubs dir so libcuda.so.1 resolves)
if [ ! -f "$BASE" ]; then
  "$BUILD/bin/llama-perplexity" -m "$BF16" -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base "$BASE" -c 512 --chunks "$CHUNKS" -ngl "${BF16_NGL:-0}" -t "$(nproc)"
fi

# Pass 2 — score each tier (GPU required: PXQ has no CPU codec)
for TIER in PXQ6 PXQ3 PXQ2; do
  M="$MODELS_DIR/PXA-Fusion2-35B-$TIER.gguf"
  [ -f "$M" ] || { echo "SKIP $TIER"; continue; }
  echo "=== KLD $TIER vs bf16 ==="
  "$BUILD/bin/llama-perplexity" -m "$M" --kl-divergence-base "$BASE" --kl-divergence \
    -c 512 -ngl 99 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 \
    2>&1 | grep -E "Mean|Median|KLD|Same top|Final"
done
# Report: Mean KLD, 99.9% KLD, and top-1 token agreement per tier — those three numbers are
# what quant reviewers (ubergarm/bartowski-style tables) compare.
