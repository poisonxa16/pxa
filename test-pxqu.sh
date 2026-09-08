#!/bin/bash
# Test the PXQU build on the four P100s. Built is not tested.
#
# Every guard in here exists because the corresponding mistake was actually made:
#   --temp 0        : without it llama-cli samples, and arms return different token
#                     counts (59/127/69) even with --ignore-eos, so ms/token is noise.
#   decode grep     : `grep "eval time" | head -1` matches "prompt eval time" FIRST.
#                     That reported prefill as decode once already. Filter, do not head.
#   no rm of logs   : an earlier A/B deleted each arm's output. Keep every log.
#   nvidia-smi read : card 0 carries ~8.4 GiB of production granite. -ts must respect it.
#   card 3 refusal  : production VLM + embeddings. Opt-in only, with a ceiling.
set -u
# Set PXQ_HOST_GUARD to pin this script to one host.
[ -n "${PXQ_HOST_GUARD:-}" ] && [ "$(hostname)" != "$PXQ_HOST_GUARD" ] && \
  { echo "WRONG HOST: $(hostname) (expected $PXQ_HOST_GUARD)"; exit 1; }

REPO=${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
MODEL=${MODEL:?set MODEL to the .gguf to test}
ENGINE=${ENGINE:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)/build-cuda/bin/llama-cli}
CARDS=${CARDS:-0,1,5,6}
TS=${TS:-13,29,29,29}
UB=${UB:-1024}
NCTX=${NCTX:-150016}
NPRED=${NPRED:-127}
MODE=${MODE:-speed}
TAG=${TAG:-$(date +%H%M%S)}
LOG=${LOGDIR:-.}/pxqu-test-${MODE}-ub${UB}-${TAG}.log

[ -x "$ENGINE" ] || { echo "engine missing: $ENGINE"; exit 1; }
[ -f "$MODEL" ]  || { echo "model missing: $MODEL"; exit 1; }

# card 3 is production unless explicitly authorised, and even then it has a ceiling
for d in ${CARDS//,/ }; do
  if [ "$d" = "3" ] && [ "${ALLOW_CARD3:-0}" != "1" ]; then
    echo "REFUSING: card 3 is the production 1080 Ti. Set ALLOW_CARD3=1 deliberately."; exit 1
  fi
done

echo "=== pre-flight $(date -Is) ===" | tee "$LOG"
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader | tee -a "$LOG"
echo "model: $MODEL ($(stat -c %s "$MODEL") bytes, dev $(stat -c %d "$MODEL"))" | tee -a "$LOG"
echo "cards=$CARDS ts=$TS ub=$UB ctx=$NCTX npred=$NPRED mode=$MODE" | tee -a "$LOG"

if [ "$MODE" = "correct" ]; then
  PROMPT="Name the capital city of each country, one per line, nothing else: France, Japan, Brazil, Egypt, Canada, Australia, Kenya, Norway, Peru."
  EXTRA="--temp 0"
else
  PROMPT="Explain in two sentences what makes a mixture-of-experts model efficient."
  EXTRA="--temp 0 --ignore-eos"
fi

docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$CARDS" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e LD_LIBRARY_PATH=/src/build-cuda/src:/src/build-cuda/ggml/src \
  -v "$REPO":/src \
  -v "$(dirname "$MODEL")":"$(dirname "$MODEL")":ro \
  -v "$(cd "${LOGDIR:-.}" && pwd)":"$(cd "${LOGDIR:-.}" && pwd)" \
  -w /src --name "pxqu-test-$MODE-$TAG" \
  pxa-sm60-dev:latest \
  /src/build-cuda/bin/llama-cli \
    -m "$MODEL" -ngl 99 -ts "$TS" \
    -ot 'per_layer_token_embd\.weight=CPU' \
    -c "$NCTX" -ub "$UB" -n "$NPRED" -t 16 --no-warmup \
    $EXTRA -p "$PROMPT" >> "$LOG" 2>&1
RC=$?

echo "=== rc=$RC ===" | tee -a "$LOG"
echo "--- buffers ---"; grep -E "buffer size|KV self size|graph (nodes|splits)|offloaded" "$LOG"
echo "--- kernel engagement (must show pxq families, not zero) ---"
grep -oE "pxq[0-9]+:[0-9]+" "$LOG" | sort -u
echo "--- decode (NOT prompt eval) ---"
grep "eval time" "$LOG" | grep -v "prompt eval time"
echo "--- run count must be exactly $NPRED for a speed arm ---"
grep "eval time" "$LOG" | grep -v "prompt eval" | grep -oE "[0-9]+ runs"
echo "--- GENERATED TEXT (read this, do not just count it) ---"
sed -n '/^Explain in two sentences\|^Name the capital/,/llama_perf/p' "$LOG" | head -60
echo "full log: $LOG"
