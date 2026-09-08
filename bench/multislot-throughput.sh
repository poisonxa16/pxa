#!/bin/bash
# multislot-throughput.sh — reproducible np>1 concurrency scaling for the PXQ server.
#
# Question it answers: when N clients hit the server at once, does aggregate t/s SCALE
# (batched MoE decode shares the weight load) or SERIALIZE? And is any np>1 loss a real
# batching bug or the known speculative-decode tax? It sweeps np x {MTP on/off} x
# {PXA_NP_SPEC_GATE on/off} so the cause is isolated, not guessed.
#
# Metric: launch K identical concurrent /completion streams, record each stream's
# server-reported predicted_per_second, report SINGLE (np1) vs AGGREGATE (sum over K) vs
# PER-STREAM. Aggregate/single = the concurrency scaling factor (ideal ~= K).
#
# Reproduce (public model, single 16 GB card holds PXQ2):
#   GPUS=<uuid> MODEL=/path/PXA-Fusion2-35B-PXQ2.gguf ./multislot-throughput.sh   (use a 16GB card; an 11GB 1080Ti OOMs a 10.7GB model once context is added)
#
# Usage: GPUS=<uuid[,uuid]> MODEL=/path.gguf [TS=1,1] [BUILD=/path] [PORT=8299] ./multislot-throughput.sh
#   BUILD defaults to <repo>/build (the tree this script lives in); OUT defaults to ./multislot.txt.
set -uo pipefail
GPUS="${GPUS:?set GPUS=GPU-uuid[,GPU-uuid]}"
MODEL="${MODEL:?set MODEL=/abs/path/to/pxq.gguf}"
TS="${TS:-}"; NGL="${NGL:-99}"; PORT="${PORT:-8299}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || dirname "$HERE")"
BUILD="${BUILD:-$REPO/build}"
IMG="${IMG:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"
OUT="${OUT:-./multislot.txt}"
NGEN="${NGEN:-160}"
mkdir -p "$(dirname "$OUT")"
MDIR="$(dirname "$MODEL")"; MBASE="$(basename "$MODEL")"
tsflag=""; [ -n "$TS" ] && tsflag="-ts $TS"
CN="pxg-multislot-$PORT"

serve() { # $1 np  $2 extra-env(space-separated KV)
  local np="$1" envs=""
  for kv in $2; do envs="$envs -e $kv"; done
  docker rm -f "$CN" >/dev/null 2>&1
  docker run -d --name "$CN" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$GPUS" $envs \
    -e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd \
    -v "$BUILD":/build:ro -v "$MDIR":/models:ro -p "$PORT:$PORT" "$IMG" \
    /build/bin/llama-server -m "/models/$MBASE" --host 0.0.0.0 --port "$PORT" \
    -c "${CTX:-8192}" -np "$np" -ngl "$NGL" $tsflag -fa on -b 512 -ub 512 >/dev/null 2>&1
  # wait for health
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    ! docker ps -q -f name="$CN" | grep -q . && { echo "  server died"; docker logs --tail 5 "$CN" 2>&1 | sed 's/^/    /'; return 1; }
    sleep 3
  done
  echo "  server never healthy"; return 1
}

fire() { # $1 concurrency K -> prints aggregate + per-stream
  local K="$1" tmp; tmp=$(mktemp -d)
  for k in $(seq 1 "$K"); do
    ( curl -sf "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"Write a detailed technical paragraph about GPU memory bandwidth. Stream $k.\",\"n_predict\":$NGEN,\"temperature\":0,\"cache_prompt\":false}" \
        > "$tmp/$k.json" 2>/dev/null ) &
  done
  wait
  local agg=0 n=0 line=""
  for k in $(seq 1 "$K"); do
    local tps; tps=$(grep -oE '"predicted_per_second":[0-9.]+' "$tmp/$k.json" 2>/dev/null | head -1 | cut -d: -f2)
    [ -z "$tps" ] && tps=0
    agg=$(awk "BEGIN{print $agg+$tps}"); n=$((n+1)); line="$line ${tps%.*}"
  done
  echo "K=$K  aggregate=$(printf %.1f "$agg") t/s  per-stream:$line"
  rm -rf "$tmp"
}

: > "$OUT"
echo "== multislot-throughput  model=$MBASE  gpus=$GPUS  $(date -u +%FT%TZ) ==" | tee -a "$OUT"
echo "-- np=1 baseline (single stream) --" | tee -a "$OUT"
serve 1 "" && fire 1 | tee -a "$OUT"
for cfg in "MTP-default:" "MTP-off:PXA_MTP_OFF=1" "spec-gate-on:PXA_NP_SPEC_GATE=1"; do
  label="${cfg%%:}"; env="${cfg#*:}"
  echo "-- np=4  [$label] --" | tee -a "$OUT"
  serve 4 "$env" && { fire 1 | tee -a "$OUT"; fire 2 | tee -a "$OUT"; fire 4 | tee -a "$OUT"; }
done
docker rm -f "$CN" >/dev/null 2>&1
echo "== done: aggregate(K)/aggregate(K=1) is the scaling factor; ideal ~ K. <1 = the np>1 tax ==" | tee -a "$OUT"
