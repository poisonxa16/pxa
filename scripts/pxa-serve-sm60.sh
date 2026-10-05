#!/bin/bash
# PXA Network — PXQ4 sm_60 (Tesla P100) serving launcher.
#
# MEASURED shipping configuration. Every arm was correctness-gated AND
# byte-gated (20/20 greedy outputs identical to an NCCL reference boot) before
# its number was recorded. Measured on cards 1,6, 2026-08-27:
#
#   gate config (old site, v7 lib, CAR off, custom_ops none)  13.31 single
#   this config                                               24.0-26.4 single
#                                                             70-72 aggregate @8
#                                                             ~218 prefill
#
# WHY EACH VALUE, so none of them get "tidied" away later:
#
#  TORCHDYNAMO_DISABLE=1   LOAD-BEARING. Without it profile_run compiles the
#      LANGUAGE model through Inductor and dies with GPUTooOldForTriton on a
#      capability-6.0 card. This is the single flag that decides whether the
#      server starts at all. Do not remove it to "let dynamo try".
#  CAR left ON            The dominant speed lever: 13.3 -> 24.0, ~1.8x. It is
#      NOT passed --disable-custom-all-reduce. Byte-gated 20/20 against an NCCL
#      reference on this exact lib+site+config, so it is safe HERE. It remains
#      unsafe for MoE models on Pascal -- that is a different model class.
#  custom_ops default     Deliberately NOT "none". "none" is a correctness
#      workaround for a different config: with defaults at MNS=4/ladder[1,2,4]
#      a raw 1-token prompt returns "!!!!" while Paris and 391 still pass. It
#      clears at MNS=8/ladder[1,2,4,8], which is why this config is clean.
#  MNS=8, ladder[1,2,4,8] Both required for the above. MNS=16 is UNVERIFIED on
#      this arch -- its only test ran on a broken image.
#  GMU 0.90               0.94 reintroduces the "!!!!" raw-prompt failure.
#  SPLIT_MAX_BLOCKS=300   NOT 150. 150 is bimodal: four boots gave 24.58,
#      24.61, 19.75 and 9.27. 300 gave 24.01/23.97/23.97 across three. A single
#      good 150 sample looks like a win and is not.
#  PXQ4_MMV_SLICE_MAX=8   Confirmed a dead knob at serving level (14.43 vs
#      14.24 at 16). Kept only to pin it against re-litigation.
#  PXQ4_LIB explicit      The sidecar tree bundles an sm70-only .so built
#      against the torch 2.10 ABI. Unset, the loader can reach it on this
#      image and die on `undefined symbol: ...incref_pyobject...` mid-load.
#  PASCAL_SDPA            Pascal has no tensor cores; FLASH_ATTN_V100 is sm_70.
set -uo pipefail
# Set PXQ_HOST_GUARD to pin this seat to one host.
if [ -n "${PXQ_HOST_GUARD:-}" ] && [ "$(hostname)" != "$PXQ_HOST_GUARD" ]; then
  echo "WRONG HOST: $(hostname) (expected $PXQ_HOST_GUARD)"; exit 1
fi

NAME=${NAME:-pxa-pxq4-p100}
PORT=${PORT:-8199}
CARDS=${CARDS:-1,6}     # free P100s. NEVER 0 (production seat) or 3 (1080 Ti).
PKG=${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pxa/pxq4}
MODEL=${MODEL:?set MODEL to the model dir or repo id}
# host dir bind-mounted into the container so the model path resolves inside it
MODEL_MOUNT=${MODEL_MOUNT:-$( [ -e "$MODEL" ] && cd "$(dirname "$MODEL")" && pwd || echo "$PWD" )}
IMAGE=${IMAGE:-pxa-vllm:sm60}
SITE=${SITE:-$PKG/sidecar/site-sm60}
LIB=${LIB:-$PKG/kernels/libpxq4_sm60_v10.so}
GMU=${GMU:-0.90}; MML=${MML:-8192}; MNS=${MNS:-8}
# MML 8192 is the gated serving value. Benchmarks that send a ~20k-token prompt (the
# 20k prefill cell, the pxq2/pxq3 tier arms) must raise it explicitly -- an over-long
# prompt returns HTTP 400, which reads like a broken harness and cost two windows.
TP=$(awk -F, '{print NF}' <<<"$CARDS")

case "$CARDS" in *0*) echo "REFUSING: card 0 carries a production seat." >&2; exit 1;; esac
case "$CARDS" in *3*) echo "REFUSING: card 3 is the protected 1080 Ti." >&2; exit 1;; esac

echo "PXA Network — PXQ4 sm_60"
echo "  model $MODEL"
echo "  cards $CARDS (TP=$TP) port $PORT"
echo "  lib   $(basename "$LIB")   site $(basename "$SITE")"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --runtime=nvidia --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES="$CARDS" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e TORCHDYNAMO_DISABLE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
  -e VLLM_SM70_GDN_DECODE_FLASHQLA=0 -e VLLM_SM70_FUSED_SIGMOID_MIXED_QKV=0 \
  -e VLLM_SM70_GEMMA_LONG_PREFILL_FUSED=0 \
  -e PXQ4_MMV_SLICE_MAX=8 -e PXQ4_MMV_SPLIT_MAX_BLOCKS=300 \
  -e PYTHONPATH="$SITE" -e PXQ4_LIB="$LIB" \
  -e HOME=/tmp -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  -p 127.0.0.1:${PORT}:${PORT} \
  -v "$PKG":"$PKG" -v "$MODEL_MOUNT":"$MODEL_MOUNT" ${EXTRA_MOUNT:+-v "$EXTRA_MOUNT":"$EXTRA_MOUNT"} \
  --shm-size=16g --ipc=host \
  "$IMAGE" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --quantization pxq4 \
    --attention-backend PASCAL_SDPA --tensor-parallel-size "$TP" --dtype float16 \
    --host 0.0.0.0 --port "$PORT" --served-model-name qwen3.8-27b --trust-remote-code \
    --gpu-memory-utilization "$GMU" --max-model-len "$MML" --max-num-seqs "$MNS" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8]}' \
    >/dev/null || { echo "docker run failed" >&2; exit 1; }

echo -n "booting (~6 min: graph capture on Pascal is slow)"
for i in $(seq 1 100); do
  code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  [ "$code" = "200" ] && { echo; echo "HEALTHY after $((i*15))s"; break; }
  case "$(docker ps -a --filter "name=^/${NAME}$" --format '{{.Status}}')" in
    Exited*) echo; echo "DIED. First real error:" >&2
             docker logs "$NAME" 2>&1 | grep -aiE "GPUTooOldForTriton|next_power_of_2|undefined symbol|out of memory|ValueError" | head -5 >&2
             exit 1 ;;
  esac
  echo -n "."; sleep 15
done
[ "${code:-000}" = "200" ] || { echo "never came up; container kept as $NAME" >&2; exit 1; }

# The fp16 mmv hook is worth ~12% here and its absence is silent, so COUNT it.
# Note the string is "fp16", not "f16" -- grepping the wrong one reports a
# working hook as absent.
ARMED=$(docker logs "$NAME" 2>&1 | grep -ac 'fp16 mmv fast path armed')
echo "  fp16 mmv hook armed on $ARMED layers"
[ "$ARMED" -lt 50 ] && echo "  WARNING: hook did not arm; expect ~12% less decode. Check PYTHONPATH/PXQ4_LIB." >&2
echo "  Triton disable events (expect 0 with the dynamo flag set): $(docker logs "$NAME" 2>&1 | grep -ac 'Disabling Triton')"
