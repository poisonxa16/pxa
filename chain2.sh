#!/bin/bash
# Fire the quantize+gate once the Q8_0-PLE conversion lands.
# Set PXQ_HOST_GUARD to the hostname these artifacts live on to re-arm this check.
if [ -n "${PXQ_HOST_GUARD:-}" ] && [ "$(hostname)" != "$PXQ_HOST_GUARD" ]; then
  echo "WRONG HOST: $(hostname) (expected $PXQ_HOST_GUARD)"; exit 1
fi
PXQ_WORK="${PXQ_WORK:-./qwen4exp}"          # BF16/PXQ4 working dir
BF16="${BF16:-$PXQ_WORK/Qwen3.8-Flash-Next-BF16-pleq8.gguf}"
while ps -eo args --no-headers | grep -q "[c]onvert_qwen4exp"; do sleep 60; done
echo "=== [$(date -u +%H:%M:%S)] converter exited ==="
sz=$(stat -c %s "$BF16" 2>/dev/null || echo 0)
echo "size: $(awk -v s=$sz 'BEGIN{printf "%.2f GiB", s/1073741824}')  (plan was 306 GB)"
if [ "$sz" -lt 280000000000 ]; then
  echo "LOOKS TRUNCATED -- refusing to quantize a partial model"
  tail -15 "$PXQ_WORK/logs/convert2.log" | tr '\r' '\n' | tail -8
  exit 1
fi
echo "=== [$(date -u +%H:%M:%S)] starting pipeline3.sh ==="
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline3.sh"
