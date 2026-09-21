#!/bin/bash
# CPU FIDELITY PROXY. PXA_PXQ_ERRBUDGET=1 dequantizes every written tensor straight back and
# reports relative RMS error vs the f32 source, per tensor class. Three arms are enough to fill
# the whole class x tier matrix, because a class's error at a tier does not depend on what any
# OTHER class got: PXQ2-uniform gives every class at pxq2, PXQ3-uniform at pxq3, PXQ2-attn4 the
# attention classes at pxq4/pxq4hq, PXQ4-uniform the FFN at pxq4. The output file is a
# by-product here and is deleted immediately; only the .errbudget.tsv is kept.
set -u
SRC=${SRC:?path to source .gguf}
WT=${WT:-$(pwd)}
LOGD=${LOGD:-./logs/quant-policy}
IMAGE=${IMAGE:-<your-build-image>}
SCRATCH=${SCRATCH:-/tmp/qp-scratch}
THREADS=${THREADS:-16}
for arm in "PXQ2 uniform" "PXQ3 uniform" "PXQ2 attn4" "PXQ4 uniform"; do
    set -- $arm; lvl=$1; pol=$2
    out="$SCRATCH/eb-${lvl}-${pol}.gguf"
    [ -s "$LOGD/eb-${lvl}-${pol}.tsv" ] && { echo "SKIP $lvl/$pol"; continue; }
    echo "=== $(date -u +%FT%TZ) errbudget $lvl/$pol"
    docker run --rm --name qp-eb-$lvl-$pol \
      -e PXA_PXQ_ERRBUDGET=1 -e PXA_PXQ_ERRBUDGET_MAXELEM=16777216 \
      -v "$(dirname "$SRC")":"$(dirname "$SRC")" -v "$SCRATCH":"$SCRATCH" \
      -w $WT "$IMAGE" \
      nice -n 19 ./build-cpu/bin/llama-quantize \
        --allow-requantize --i-know-this-is-double-lossy --pxq-policy "$pol" \
        "$SRC" "$out" "$lvl" "$THREADS" > "$LOGD/eb-${lvl}-${pol}.log" 2>&1
    cp -f "$out.errbudget.tsv" "$LOGD/eb-${lvl}-${pol}.tsv" 2>/dev/null
    rm -f "$out" "$out.errbudget.tsv"
    echo "=== $(date -u +%FT%TZ) errbudget $lvl/$pol done"
done
echo "ERRBUDGET ALL DONE $(date -u +%FT%TZ)"
