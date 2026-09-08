#!/bin/bash
# usage: run_arm.sh <name> <awq|pxq4> <single|conc|all> [extra vllm args...]
# Boots a fresh OpenAI api_server on GPUs 4-7, waits healthy, runs bench_client
# inside the container, tears down. All output -> logs/<name>.log,
# results -> results/<name>.json, marker results/<name>.DONE (or .FAIL).
set -u
D=/mnt/models/pxa-hth
NAME=$1; ARM=$2; MODE=$3; shift 3
PORT=8420
LOG=$D/logs/$NAME.log
exec > "$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] arm=$ARM mode=$MODE extra: $*"

if docker ps --format '{{.Names}}' | grep -qE 'pxa-(mtp|step|awq-base|hth|tp4)'; then
  echo "REFUSING: experiment container already up"; docker ps; touch $D/results/$NAME.FAIL; exit 3
fi
docker rm -f pxa-hth >/dev/null 2>&1 || true

EXTRA_ENV=""
QARG=""
if [ "$ARM" = awq ]; then
  MODEL=/mnt/models/hf/philbert440/Qwen3.8-27B-W4A16-AWQ
  EXTRA_ENV="-e VLLM_SM70_QUANT_BACKEND=turbomind"
elif [ "$ARM" = awqmarlin ]; then
  MODEL=/mnt/models/hf/philbert440/Qwen3.8-27B-W4A16-AWQ
  EXTRA_ENV="-e VLLM_SM70_QUANT_BACKEND=marlin"
elif [ "$ARM" = pxq4 ]; then
  MODEL=/mnt/models/pxa-models/Qwen3.8-27B-PXQ4-vllm-p2a-nf
  EXTRA_ENV="-e VLLM_SM70_QUANT_BACKEND=marlin -e PYTHONPATH=/mnt/models/pxa-int-v5/site"
  QARG="--quantization pxq4"
elif [ "$ARM" = pxq4v6 ]; then
  MODEL=/mnt/models/pxa-models/Qwen3.8-27B-PXQ4-vllm-p2a-nf
  EXTRA_ENV="-e VLLM_SM70_QUANT_BACKEND=marlin -e PYTHONPATH=/mnt/models/pxa-int-v6/site"
  QARG="--quantization pxq4"
else
  echo bad arm; touch $D/results/$NAME.FAIL; exit 2
fi

docker run -d --name pxa-hth \
  --gpus '"device=4,5,6,7"' \
  --shm-size 16g \
  --tmpfs /root/home:exec,size=24g \
  -v /mnt/models:/mnt/models \
  -e HOME=/root/home -e TMPDIR=/root/home \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_SM70_FLASH_ATTN_V100=1 \
  -e VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0 \
  -e NCCL_P2P_DISABLE=0 \
  -e PYTHONUNBUFFERED=1 \
  $EXTRA_ENV \
  --entrypoint python kewaii/vllm:latest \
  -m vllm.entrypoints.openai.api_server \
  --model $MODEL $QARG \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 --port $PORT \
  --served-model-name m --trust-remote-code \
  "$@"

echo "[$(date -u +%FT%TZ)] container started, polling health (max 22 min)"
UP=0
for i in $(seq 1 132); do
  sleep 10
  if ! docker ps --format '{{.Names}}' | grep -q '^pxa-hth$'; then
    echo "[$(date -u +%FT%TZ)] CONTAINER DIED"; docker logs pxa-hth 2>&1 | tail -60
    docker rm -f pxa-hth >/dev/null 2>&1; touch $D/results/$NAME.FAIL; exit 1
  fi
  if docker exec pxa-hth python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=3); print('ok')" 2>/dev/null | grep -q ok; then
    UP=1; break
  fi
done
if [ "$UP" != 1 ]; then
  echo "[$(date -u +%FT%TZ)] TIMEOUT waiting for health"; docker logs pxa-hth 2>&1 | tail -80
  docker rm -f pxa-hth; touch $D/results/$NAME.FAIL; exit 1
fi
if [ "${LOCKGC:-0}" = 1 ]; then
  nvidia-smi -i 4,5,6,7 -lgc 1530,1530 && echo "[locked clocks 1530]" || echo "[lgc FAILED]"
fi
echo "[$(date -u +%FT%TZ)] HEALTHY, starting clock sampler + client"

( while docker ps --format '{{.Names}}' | grep -q '^pxa-hth$'; do
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -i 4,5,6,7 >> $D/logs/$NAME.clocks.csv
    sleep 5
  done ) &
CLKPID=$!

docker exec pxa-hth python /mnt/models/pxa-hth/bench_client.py \
  --port $PORT --model m --mode $MODE --tokens 512 --n 12 --conc "${CONC:-2,4,8,16}" \
  --out /mnt/models/pxa-hth/results/$NAME.json --tag $NAME
RC=$?
kill $CLKPID 2>/dev/null

echo "[$(date -u +%FT%TZ)] client rc=$RC; saving full engine log"
docker logs pxa-hth > $D/logs/$NAME.engine.log 2>&1
docker rm -f pxa-hth
if [ "${LOCKGC:-0}" = 1 ]; then nvidia-smi -i 4,5,6,7 -rgc && echo "[clocks unlocked]"; fi
if [ $RC = 0 ]; then touch $D/results/$NAME.DONE; else touch $D/results/$NAME.FAIL; fi
echo "[$(date -u +%FT%TZ)] arm $NAME finished rc=$RC"
