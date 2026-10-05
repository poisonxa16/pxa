#!/usr/bin/env bash
# PXQ4 on 4x V100 (sm_70), tensor-parallel 4.
#
# These flags are the measured-best configuration, not a guess. Every one of them
# was A/B'd against AWQ W4A16 on the same cards with the same client. Notes on the
# non-obvious ones are inline - please read them before changing anything, because
# three of these look like they should be improvements and are not.
#
# Configuration, all through the environment:
#   VLLM_IMAGE        sm_70-capable vLLM container image                       (REQUIRED)
#   MODEL             converted PXQ4 checkpoint directory                      (REQUIRED)
#   PXQ4_ROOT         install prefix of the PXQ4 plugin; its site/ goes on PYTHONPATH
#                                                                              (default /opt/pxq4)
#   GPUS              docker --gpus value, e.g. all or '"device=0,1,2,3"'      (default all)
#   PORT / NAME       listen port / served model name
#   MTP_DEFAULTS_ENV  optional: name of your vLLM build's "disable the automatic sm_70 MTP
#                     defaults" environment variable (see the note below).
set -euo pipefail

VLLM_IMAGE=${VLLM_IMAGE:?set VLLM_IMAGE to an sm_70-capable vLLM container image}
MODEL=${MODEL:?set MODEL to a converted PXQ4 checkpoint directory}
PXQ4_ROOT=${PXQ4_ROOT:-/opt/pxq4}
GPUS=${GPUS:-all}
PORT=${PORT:-8000}
NAME=${NAME:-pxq4-27b}
MTP_DEFAULTS_ENV=${MTP_DEFAULTS_ENV:-}

docker run -d --rm --name pxq4-serve \
  --gpus "$GPUS" \
  --shm-size 16g \
  --tmpfs /root/home:exec,size=24g \
  -v "$MODEL":"$MODEL":ro -v "$PXQ4_ROOT":"$PXQ4_ROOT":ro \
  -p ${PORT}:${PORT} \
  -e HOME=/root/home -e TMPDIR=/root/home \
  \
  `# --- the PXQ4 plugin. It needs no pip install into the image: the plugin is` \
  `# registered by PYTHONPATH plus a hand-written entry_points.txt, which also works` \
  `# when the image's root filesystem is read-only or full. ---` \
  -e PYTHONPATH="$PXQ4_ROOT/site" \
  \
  `# --- sm_70 attention. The stock backend does not support V100. ---` \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_SM70_FLASH_ATTN_V100=1 \
  \
  `# --- MTP defaults off, if your build applies them. The automatic sm_70 MTP defaults` \
  `# set max_num_seqs=4, which contradicts the engine's own requirement of max_num_seqs=1` \
  `# for the dynamic GPU LRU path, and the boot then fails on its own assertion. ---` \
  ${MTP_DEFAULTS_ENV:+-e ${MTP_DEFAULTS_ENV}=0} \
  \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python \
  "$VLLM_IMAGE" \
  -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization pxq4 \
    --attention-backend FLASH_ATTN_V100 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port ${PORT} \
    --served-model-name "$NAME" \
    --trust-remote-code \
    --gpu-memory-utilization 0.93 \
    --max-model-len 200000 \
    --enable-prefix-caching

echo "starting; first boot takes 13-18 min (torch.compile + graph capture)."
echo "  health:  curl -s localhost:${PORT}/health"
echo "  logs:    docker logs -f pxq4-serve"
