#!/bin/bash
# usage: boot_arm.sh <tag> <pxq4|awq> [extra env as K=V ...]
# Runs harness in a fresh container on GPUs 4-7. Serial: refuses if another
# mtp/step container is running.
set -e
TAG=$1; ARM=$2; shift 2
if docker ps --format '{{.Names}}' | grep -qE 'pxa-mtp-vllm|pxa-mtp2-vllm|pxa-step|pxa-hth|pxa-awq'; then
  echo "REFUSING: another experiment container is up"; docker ps --format '{{.Names}}' | grep -E 'mtp|step'; exit 3
fi
ENVS=""
for kv in "$@"; do ENVS="$ENVS -e $kv"; done
if [ "$ARM" = pxq4 ]; then
  MODEL=/mnt/models/pxa-models/Qwen3.8-27B-PXQ4-vllm-p2a-nf
  QUANT="--quant pxq4"
  ENVS="$ENVS -e PYTHONPATH=/mnt/models/pxa-int-v5/site -e VLLM_SM70_QUANT_BACKEND=marlin"
elif [ "$ARM" = awq ]; then
  MODEL=/mnt/models/hf/philbert440/Qwen3.8-27B-W4A16-AWQ
  QUANT=""
  ENVS="$ENVS -e VLLM_SM70_QUANT_BACKEND=turbomind"
else
  echo "bad arm"; exit 2
fi
mkdir -p /mnt/models/pxa-step/logs
docker run --rm --name pxa-step-$TAG \
  --gpus '"device=4,5,6,7"' \
  --shm-size 16g \
  --tmpfs /root/home:exec,size=24g \
  -v /mnt/models:/mnt/models \
  -e HOME=/root/home -e TMPDIR=/root/home \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_SM70_FLASH_ATTN_V100=1 \
  -e VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0 \
  -e PYTHONUNBUFFERED=1 \
  $ENVS \
  --entrypoint python \
  kewaii/vllm:latest \
  /mnt/models/pxa-step/harness.py --tag $TAG --model $MODEL $QUANT \
  > /mnt/models/pxa-step/logs/$TAG.log 2>&1
