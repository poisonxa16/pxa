#!/usr/bin/env bash
# GPU 3 (GTX 1080 Ti) runner for pascal-ops unit work, per main #812/#414.
# flock -> pause file -> stop the three GPU-3 services -> run -> ALWAYS restore.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-pxa-vllm:sm60-v15}"
SCRIPT="${1:?usage: run-gpu3.sh <script.py> [args...]}"; shift || true

exec 9>/tmp/pxa-1080-bench.lock
flock 9
echo "[pascal-ops] holding /tmp/pxa-1080-bench.lock"

restore() {
  docker start pxa-embed-gpu pxa-llama-rerank pxa-ollama-embed >/dev/null 2>&1 || true
  rm -f /tmp/pxa-gpu3.paused
  echo "[pascal-ops] GPU 3 services restored, pause file removed"
}
trap restore EXIT INT TERM

touch /tmp/pxa-gpu3.paused
docker stop pxa-embed-gpu pxa-llama-rerank pxa-ollama-embed >/dev/null 2>&1 || true
sleep 2
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i 3

docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e TORCHDYNAMO_DISABLE=1 -e HOME=/tmp -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  -v "$ROOT":/work -w /work \
  --entrypoint python3 "$IMAGE" "/work/${SCRIPT}" "$@"
