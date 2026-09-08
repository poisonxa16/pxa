#!/bin/bash
# PXA Network — Qwen3.8-Flash-Next (qwen4exp) PXQU seat: the 4 P100s + the 1080 Ti.
#
# MEASURED, 2026-08-28. Not a guess.
#
# WHY THESE CARDS: PXQU is 46702 MiB of GPU-resident weights (PXQ4 is 65032, which
# is why that one needed six cards). The four P100s alone CANNOT hold this at a
# large context - card 1 came up FORTY-EIGHT MiB short at c150016, and a PXQU layer
# is ~973 MiB, so there is nothing to shed. Two different -ts values produced
# byte-identical allocations, proving it is layer granularity and not tuning.
# Card 3 (the 1080 Ti) is the fifth card, owner-authorised.
#
# CARD 3 IS SHARED WITH PRODUCTION VLM + EMBEDDINGS. Budget ~5.0 GiB for us on top
# of their ~4.15 GiB, of 11264 total. Never evict them. sm_61 JITs from the
# compute_60 PTX in libggml.so, so no special build is needed. GP102 fp16 is 1/64
# rate, but k_pxa_gemv_f16_wide reads half2 and does its MATH in fp32, so the f16
# hc_* tensors placed there are not penalised. Keep the HEAD on a P100 (it is the
# last device in PCI order) - its compute buffer is 980 MiB vs ~570 elsewhere.
#
# -ts PARTITIONS BYTES, NOT LAYERS (llama.cpp:4071-4100), and llama.cpp folds a
# per-device compute allowance into the walk - so CHANGING -ub REPACKS THE LAYERS.
# These numbers are capacity-proportional for ub1024 at this context. Re-derive
# them if you change either.
#
# THE MODEL MUST LIVE ON NVMe. per_layer_token_embd is CPU-resident and mmap'd, so
# every decode token faults PLE pages off whatever disk holds the file. From the
# parity array that is ~17 MB/s and wedges the process in uninterruptible D state.
set -uo pipefail
# Set PXQ_HOST_GUARD to pin this seat to one host.
if [ -n "${PXQ_HOST_GUARD:-}" ] && [ "$(hostname)" != "$PXQ_HOST_GUARD" ]; then
  echo "WRONG HOST: $(hostname) (expected $PXQ_HOST_GUARD)"; exit 1
fi

NAME=${NAME:-pxa-flashnext}
PORT=${PORT:-8261}   # the hive expects Alex (id glimmer) here; see pxa-hive/brains.mjs
CARDS=${CARDS:-0,1,3,5,6}
MODEL=${MODEL:?set MODEL to the .gguf to serve}
ENGINE_TREE=${ENGINE_TREE:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}
TS=${TS:-110,268,94,268,261}
UB=${UB:-1024}
NCTX=${NCTX:-163840}          # 160k. 262144 LOADS but OOMs at decode: card 0 was left 51 MiB. Decode needs transient buffers beyond the reported compute buffer, so every card needs ~1200 MiB spare.
ALIAS=${ALIAS:-qwen3.8-flash-next}

[ -f "$MODEL" ] || { echo "model missing: $MODEL"; exit 1; }
case "$(stat -c %d "$MODEL")" in 46) : ;; *) echo "WARNING: model is NOT on the NVMe pool (dev $(stat -c %d "$MODEL")) - decode will fault PLE pages off a spindle" >&2;; esac

echo "PXA Network — Flash-Next PXQU"
echo "  model $MODEL"
echo "  cards $CARDS   ts $TS   ub $UB   ctx $NCTX   port $PORT"

# `docker rm -f` RETURNS BEFORE THE NAME IS RELEASED. Firing docker run straight
# after it loses the race with "container name is already in use", which then
# leaves NO seat running at all - the old one is already gone. Wait it out.
docker rm -f "$NAME" >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  docker ps -a --format '{{.Names}}' | grep -qx "$NAME" || break
  sleep 1
done
docker run -d --name "$NAME" --runtime=nvidia --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES="$CARDS" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e LD_LIBRARY_PATH=/src/build-rc/src:/src/build-rc/ggml/src \
  -v "$(dirname "$MODEL")":"$(dirname "$MODEL")":ro -v "$ENGINE_TREE":/src -w /src \
  -p 127.0.0.1:${PORT}:${PORT} \
  pxa-sm60-dev:latest \
  /src/build-rc/bin/llama-server \
    -m "$MODEL" -ngl 99 -ts "$TS" \
    -ot 'per_layer_token_embd\.weight=CPU' \
    -c "$NCTX" -ub "$UB" -t 16 \
    --host 0.0.0.0 --port "$PORT" --alias "$ALIAS" \
  >/dev/null || { echo "docker run failed" >&2; exit 1; }
echo "started $NAME; watch: docker logs -f $NAME"
