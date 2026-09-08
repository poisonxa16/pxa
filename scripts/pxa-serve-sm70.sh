#!/bin/bash
# PXA Network — PXQ4 sm_70 (Tesla V100) serving launcher.
#
# This is the MEASURED shipping configuration, not a guess. Every value below
# was A/B'd on cards 2,4 on 2026-08-27 and each arm was correctness-gated
# before its number was recorded. Results (single-stream decode / agg@4 / agg@8):
#
#   reproduce final-v10                    48.74 / 106.46 / 134.36
#   + PXQ4_MMV_SPLIT_MAX_BLOCKS=300        51.46 / 107.26 / 132.13   <- best single
#   + SPLIT=600                            51.31 / 106.60 / 134.47
#   + SPLIT=150                            49.51 / 105.07 / 134.66
#   + SPLIT=300, GMU 0.92                  51.32 / 104.85 / 133.76
#   + SPLIT=300, MNS=16, ladder[1..8,16]   50.98 / 107.02 / 135.78   <- SHIPPED
#
# MNS=16 is shipped: it clears the 50 tok/s bar on single-stream and takes both
# aggregate crowns. Set PXA_PROFILE=single to trade 0.5 tok/s of concurrency
# headroom for the best single-stream number instead.
#
# WHY THESE VALUES, so they are not "tidied" away later:
#  GMU 0.88     - measured ceiling for this model on a 16 GiB card at TP=2, and what
#                 the seat has run at since 2026-08-31. NOT 0.90+: 0.98 with a pinned
#                 12 GiB KV is a 4-card DGX setting and ABORTS before the model loads.
#  v12b lib     - the MMA decode kernel; PXQ4_MMV_MMA=1 arms it. v10 is the older
#                 scalar path and is slower on every decode cell.
#  MNBT 4096    - the seat value. 2048 halves the long-prefill chunk for no gain here.
#  block-size   - 256, v2026.09.05. On the GDN-hybrid path under this image the KV
#     256         block size is a measured parity/speed win and the compiled boot
#                 needs it; see RELEASE-NOTES "Volta" and docs/VLLM.md.
#  ladder       - must be passed EXPLICITLY. When cudagraph_capture_sizes is None
#                 the sm70 branch hard-codes [1,2] and every batch above 2 runs
#                 eager, ~4x slower. The launcher reads it back below and warns.
#  SPLIT=300    - routes gate_up from mono to split. Worth ~2.7 tok/s (48.74 ->
#                 51.46). 150 is worse, 600 is a wash on decode.
#  PXQ4_LIB     - pinned explicitly. The site tree bundles an sm70-only .so built
#                 against the torch 2.10 ABI; leaving this unset on the wrong
#                 image dies with an undefined-symbol crash mid-load.
#  ladder [..16]- the "16-token graph poisons the stack" finding is MoE-specific.
#                 Verified clean on this dense model; correctness held on all arms.
set -uo pipefail
# Set PXQ_HOST_GUARD to pin this seat to one host.
if [ -n "${PXQ_HOST_GUARD:-}" ] && [ "$(hostname)" != "$PXQ_HOST_GUARD" ]; then
  echo "WRONG HOST: $(hostname) (expected $PXQ_HOST_GUARD)"; exit 1
fi

NAME=${NAME:-pxa-pxq4-v100}
PORT=${PORT:-8262}
CARDS=${CARDS:-2,4}                 # V100 pair. NEVER card 3 (protected 1080 Ti).
MODEL=${MODEL:?set MODEL to the model dir or repo id}
# host dir bind-mounted into the container so the model path resolves inside it
MODEL_MOUNT=${MODEL_MOUNT:-$( [ -e "$MODEL" ] && cd "$(dirname "$MODEL")" && pwd || echo "$PWD" )}
IMAGE=${IMAGE:-pxa-vllm:sm70-v15c}
PKG=${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pxa/pxq4}
SITE=${SITE:-$PKG/sidecar/site-union}
LIB=${LIB:-$PKG/kernels/libpxq4_sm70_v12b.so}
GMU=${GMU:-0.88}
MML=${MML:-32768}

if [ "${PXA_PROFILE:-agg}" = "single" ]; then
  MNS=8;  LADDER="1,2,3,4,5,6,7,8"        # 51.46 single-stream
else
  MNS=16; LADDER="1,2,3,4,5,6,7,8,16"     # 50.98 single, best agg@4 and agg@8
fi
TP=$(awk -F, '{print NF}' <<<"$CARDS")

echo "PXA Network — PXQ4 sm_70"
echo "  model   : $MODEL"
echo "  cards   : $CARDS (TP=$TP)   port $PORT"
echo "  lib     : $LIB"
echo "  profile : ${PXA_PROFILE:-agg}  MNS=$MNS ladder=[$LADDER] GMU=$GMU"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --runtime=nvidia --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES="$CARDS" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH="$SITE" -e PXQ4_LIB="$LIB" \
  -e PXQ4_MMV_SPLIT_MAX_BLOCKS=300 -e PXQ4_MMV_MMA=1 \
  -e PYTHONUNBUFFERED=1 -e HOME=/tmp -e TMPDIR=/tmp \
  -p 127.0.0.1:${PORT}:${PORT} \
  -v "$PKG":"$PKG" -v "$MODEL_MOUNT":"$MODEL_MOUNT" ${EXTRA_MOUNT:+-v "$EXTRA_MOUNT":"$EXTRA_MOUNT"} \
  --shm-size=16g --ipc=host \
  "$IMAGE" /bin/bash -c "exec python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --served-model-name qwen3.8-27b --quantization pxq4 \
    --attention-backend FLASH_ATTN_V100 \
    --tensor-parallel-size $TP --dtype float16 \
    --enable-prefix-caching --trust-remote-code --block-size 256 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --gpu-memory-utilization $GMU --max-model-len $MML \
    --max-num-seqs $MNS --max-num-batched-tokens 4096 \
    --compilation-config '{\"cudagraph_capture_sizes\":[$LADDER]}' \
    --host 0.0.0.0 --port $PORT" >/dev/null || { echo "docker run failed" >&2; exit 1; }

echo -n "booting (graph capture is slow)"
for i in $(seq 1 160); do
  code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  [ "$code" = "200" ] && { echo; echo "HEALTHY after $((i*15))s"; break; }
  case "$(docker ps -a --filter "name=^/${NAME}$" --format '{{.Status}}')" in
    Exited*) echo; echo "DIED. First real error:" >&2
             docker logs "$NAME" 2>&1 | grep -aiE "Cuda error|out of memory|ValueError|Capability|undefined symbol" | head -5 >&2
             exit 1 ;;
  esac
  echo -n "."; sleep 15
done
[ "${code:-000}" = "200" ] || { echo "never came up; container kept as $NAME" >&2; exit 1; }

# A silent quoting slip here costs ~4x above 2 concurrent streams, so read it back.
TAKEN=$(docker logs "$NAME" 2>&1 | grep -ao "cudagraph_capture_sizes[^]]*]" | tail -1)
echo "  capture ladder taken: $TAKEN"
case "$TAKEN" in
  *"[1, 2]"*) echo "  WARNING: ladder collapsed to [1,2] -- the explicit list did NOT take." >&2 ;;
esac
echo "  Triton (expect enabled on sm_70): $(docker logs "$NAME" 2>&1 | grep -ac 'Disabling Triton') disable-events"
