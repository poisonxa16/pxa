#!/bin/bash
# BF16 GGUF -> PXQ4 -> six Teslas, n-gram table on CPU.
#
# Runs inside pxa-sm60-dev because the engine binaries link CUDA and the HOST
# has no CUDA runtime at all (no libcudart.so.12, no /usr/local/cuda). They also
# carry no RPATH, hence the explicit LD_LIBRARY_PATH.
#
# Card 3 (the production 1080 Ti) is excluded at the CONTAINER boundary via
# NVIDIA_VISIBLE_DEVICES -- it is not merely unselected, it is not present.
#
# Spindles: $PXQ_BF16D, $PXQ_PXQ4D and $PXQ_REFD must be three different
# devices; the device ids are verified below.
set -euo pipefail
# Set PXQ_HOST_GUARD to the hostname these artifacts live on to re-arm this check.
if [ -n "${PXQ_HOST_GUARD:-}" ] && [ "$(hostname)" != "$PXQ_HOST_GUARD" ]; then
  echo "WRONG HOST: $(hostname) (expected $PXQ_HOST_GUARD)"; exit 1
fi

REPO="${PXQ_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BF16D="${PXQ_BF16D:-./qwen4exp}"              # BF16 source dir
PXQ4D="${PXQ_PXQ4D:-./qwen4exp-pxq4}"         # PXQ4 output, DIFFERENT spindle
REFD="${PXQ_REFD:-./qwen4exp-testfile}"       # reference shards, third spindle
LOGD=$BF16D/logs
IMG=pxa-sm60-dev:latest
CARDS=0,1,2,4,5,6
BF16NAME=${BF16NAME:-Qwen3.8-Flash-Next-BF16-pleq8.gguf}
mkdir -p "$PXQ4D" "$LOGD"

[ "$(stat -c %d "$PXQ4D")" != "$(stat -c %d "$BF16D")" ] || { echo "PXQ4D/BF16D same device"; exit 1; }
case ",$CARDS," in *,3,*) echo "REFUSING: card 3 is the production 1080 Ti"; exit 1;; esac

step() { echo; echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

dock() {  # dock <name> <memlimit> <shell-command>
  local name=$1 mem=$2; shift 2
  docker run --name "$name" --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=$CARDS -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ${EXTRA_ENV:-} \
    --memory="$mem" --shm-size=1g \
    -v "$REPO":/src -v "$BF16D":/bf16 -v "$PXQ4D":/pxq4 -v "$REFD":/m \
    -w /src "$IMG" bash -c "
      export LD_LIBRARY_PATH=/src/build-cuda/src:/src/build-cuda/ggml/src:\$LD_LIBRARY_PATH
      $*"
}

step "quantize BF16 -> PXQ4"
if [ ! -s "$PXQ4D/Qwen3.8-Flash-Next-PXQ4.gguf" ]; then
  docker rm -f q4e-quant >/dev/null 2>&1 || true
  # no --rm: a build/quantize container that vanishes takes its log with it.
  EXTRA_ENV="-e BF16NAME=$BF16NAME" dock q4e-quant 64g \
    "./build-cuda/bin/llama-quantize /bf16/${BF16NAME} \
       /pxq4/Qwen3.8-Flash-Next-PXQ4.gguf PXQ4 12" \
    2>&1 | tee "$LOGD/quantize.log" | tail -40
fi
ls -l "$PXQ4D/Qwen3.8-Flash-Next-PXQ4.gguf"

step "confirm the row-gather table did NOT get a panel codec"
python3 - "$REPO/gguf-py" "$PXQ4D/Qwen3.8-Flash-Next-PXQ4.gguf" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); bad = 0
for t in r.tensors:
    if "per_layer_token_embd" in t.name or int(t.shape[-1]) >= 1_000_000:
        ty = t.tensor_type.name
        ok = not (ty.startswith("PXQ") or ty == "MXFP4")
        print(f"  {t.name} -> {ty}   [{'REFUSED-OK' if ok else 'PANEL CODEC ON A GATHER TABLE'}]")
        bad += 0 if ok else 1
print(f"  gather tensors carrying a panel codec: {bad}")
sys.exit(1 if bad else 0)
PY

GATE='The capital of France is Paris. The capital of Japan is'
GATEARGS="-ot per_layer_token_embd\\.weight=CPU -ngl 99 -ts 7,16,16,16,16,16 -c 2048 -n 80 -t 16 --no-warmup -fa off --temp 0"

step "behaviour gate: PXQ4 on six Teslas, n-gram table on CPU, PLE conv OFF"
docker rm -f q4e-gate-noconv >/dev/null 2>&1 || true
EXTRA_ENV="-e PXA_QWEN4EXP_NO_PLE_CONV=1" dock q4e-gate-noconv 80g \
  "./build-cuda/bin/llama-cli -m /pxq4/Qwen3.8-Flash-Next-PXQ4.gguf $GATEARGS -p '$GATE'" \
  2>&1 | tee "$LOGD/gate-pxq4-noconv.log" | tail -45

step "same, PLE conv ON (expected to degrade after ~30 tokens -- task #91)"
docker rm -f q4e-gate-conv >/dev/null 2>&1 || true
dock q4e-gate-conv 80g \
  "./build-cuda/bin/llama-cli -m /pxq4/Qwen3.8-Flash-Next-PXQ4.gguf $GATEARGS -p '$GATE'" \
  2>&1 | tee "$LOGD/gate-pxq4-conv.log" | tail -45

step "DONE"
