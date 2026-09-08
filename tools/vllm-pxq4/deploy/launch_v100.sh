#!/usr/bin/env bash
# Launch a PXQ4-quantized vLLM server on V100-class (sm_70) cards.
#
#   launch_v100.sh <container-name> <port> [extra api_server args...]
#
# Configuration, all through the environment:
#   VLLM_IMAGE        sm_70-capable vLLM container image                      (REQUIRED)
#   MODELS            host directory holding converted checkpoints            (default ./models)
#   PLUGIN            host directory holding the PXQ4 plugin (its site/ tree) (default ./plugin)
#   MODEL             checkpoint path INSIDE the container                    (default /models/pxq4)
#   GPUS              value for NVIDIA_VISIBLE_DEVICES                        (default all)
#   TP                tensor-parallel size                                    (default 2)
#   GMU / MML / MNS   gpu-memory-utilization / max-model-len / max-num-seqs
#   EXTRA_ENV         extra "-e NAME=VALUE" pairs passed straight to docker
#   MTP_DEFAULTS_ENV  optional: name of your vLLM build's "disable the automatic sm_70 MTP
#                     defaults" environment variable. Those defaults force max_num_seqs=4,
#                     which contradicts max_num_seqs=1 for the dynamic GPU LRU path and makes
#                     the boot fail on its own assertion. If your build has such a knob, set
#                     MTP_DEFAULTS_ENV to its name and it is passed as <NAME>=0.
set -eu
NAME=$1; PORT=$2; shift 2

VLLM_IMAGE=${VLLM_IMAGE:?set VLLM_IMAGE to an sm_70-capable vLLM container image}
MODELS=${MODELS:-$PWD/models}
PLUGIN=${PLUGIN:-$PWD/plugin}
GPUS=${GPUS:-all}
TP=${TP:-2}
MTP_DEFAULTS_ENV=${MTP_DEFAULTS_ENV:-}

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  --shm-size 16g --tmpfs /root/home:exec,size=24g \
  -e HOME=/root/home -e TMPDIR=/root/home \
  -e PYTHONPATH=/plugin/site -e PYTHONUNBUFFERED=1 \
  -e VLLM_SM70_FLASH_ATTN_V100=1 -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  ${MTP_DEFAULTS_ENV:+-e ${MTP_DEFAULTS_ENV}=0} \
  -e PXQ4_MMV_SLICE_MAX=8 \
  ${EXTRA_ENV:-} \
  -v "$MODELS":/models -v "$PLUGIN":/plugin \
  -p 127.0.0.1:${PORT}:${PORT} --ipc=host \
  --entrypoint python "$VLLM_IMAGE" \
  -m vllm.entrypoints.openai.api_server \
    --model ${MODEL:-/models/pxq4} \
    --quantization pxq4 --attention-backend FLASH_ATTN_V100 \
    --tensor-parallel-size ${TP} --host 0.0.0.0 --port ${PORT} \
    --served-model-name m --trust-remote-code \
    --gpu-memory-utilization ${GMU:-0.94} --max-model-len ${MML:-32768} \
    --max-num-seqs ${MNS:-4} --enable-prefix-caching "$@"
