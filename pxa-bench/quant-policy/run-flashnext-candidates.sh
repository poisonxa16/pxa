#!/bin/bash
# Flash-Next (qwen4exp, MoE 512 experts top-10) POLICY_REV 3 arms.
# Source is the BF16-pleq8 GGUF, not a Q8_0 requantize: it is the file the shipped PXQU
# artifact was built from, so these arms are comparable to it rather than a generation behind.
# ONLY the PXQ3 pair is built: at PXQ2 the balanced profile IS BACKBONE_REV 2 on a MoE model
# (the resolver test pins this), so a PXQ2 arm would be a second name for the same bytes.
set -u
SRC=${SRC:-/path/to/model.gguf}
OUT=${OUT:-./out}
WT=${WT:-.}
LOGD=${LOGD:-./logs}
IMAGE=${IMAGE:-<your-build-image>}
THREADS=${THREADS:-20}
run() {
    local lvl=$1 pol=$2
    local out="$OUT/Qwen3.8-Flash-Next-${lvl}-${pol}.gguf"
    [ -s "$out" ] && { echo "SKIP (exists) $out"; return 0; }
    echo "=== $(date -u +%FT%TZ) FN $lvl/$pol -> $out"
    docker run --rm --name qp-fn-$lvl-$pol \
      -v "$(dirname "$SRC")":"$(dirname "$SRC")" -v "$(cd "$OUT" && pwd)":"$(cd "$OUT" && pwd)" \
      -w $WT "$IMAGE" \
      nice -n 19 ./build-cpu/bin/llama-quantize --pxq-policy "$pol" \
        "$SRC" "$out" "$lvl" "$THREADS" \
      > "$LOGD/fn-${lvl}-${pol}.log" 2>&1
    echo "=== $(date -u +%FT%TZ) FN $lvl/$pol rc=$? $(ls -l "$out" 2>/dev/null | awk '{print $5}')"
}
run PXQ3 balanced
run PXQ3 uniform
echo "FN ALL DONE $(date -u +%FT%TZ)"
