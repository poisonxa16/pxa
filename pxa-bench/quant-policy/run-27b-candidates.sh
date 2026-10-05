#!/bin/bash
# 27B (qwen35, dense) POLICY_REV 3 candidate files. CPU only, no GPU, nice 19.
# One file per arm; the uniform PXQ3 arm already exists on disk and is not re-made here, and the PXQ2 uniform arm is built first because nothing on disk has one.
set -u
SRC=${SRC:-/path/to/Qwen3.8-27B-Q8_0.gguf}
WT=${WT:-$(pwd)}
LOGD=${LOGD:-./logs/quant-policy}
OUT=${OUT:-./out}
IMAGE=${IMAGE:-<your-build-image>}
THREADS=${THREADS:-24}
run() {   # run <level> <policy> <outdir>
    local lvl=$1 pol=$2 dir=$3
    local out="$dir/Qwen3.8-27B-${lvl}-${pol}.gguf"
    mkdir -p "$dir"
    [ -s "$out" ] && { echo "SKIP (exists) $out"; return 0; }
    echo "=== $(date -u +%FT%TZ) $lvl/$pol -> $out"
    docker run --rm --name qp-27b-$lvl-$pol \
      -v "$(dirname "$SRC")":"$(dirname "$SRC")" -v "$(cd "$dir" && pwd)":"$(cd "$dir" && pwd)" \
      -w $WT "$IMAGE" \
      nice -n 19 ./build-cpu/bin/llama-quantize \
        --allow-requantize --i-know-this-is-double-lossy --pxq-policy "$pol" \
        "$SRC" "$out" "$lvl" "$THREADS" \
      > "$LOGD/27b-${lvl}-${pol}.log" 2>&1
    local rc=$?
    echo "=== $(date -u +%FT%TZ) $lvl/$pol rc=$rc  $(ls -l "$out" 2>/dev/null | awk '{print $5}')"
    return $rc
}
# PXQ2 uniform already exists: $OUT/qwen38-27b-pxq2/Qwen3.8-27B-PXQ2.gguf
# (8.921 GiB, backbone_rev 2, every GEMM class pxq2) — that is the control arm, not a rebuild.
run PXQ2 balanced "$OUT/qwen38-27b-pxq2-balanced"
run PXQ2 attn4    "$OUT/qwen38-27b-pxq2-attn4"
run PXQ3 balanced "$OUT/qwen38-27b-pxq3-balanced"
# PXQ3 attn4 is NOT built: at level pxq3 the two profiles CONVERGE. balanced's +1/+2 lands
# on pxq4/pxq4hq, which is exactly what attn4 pins there — byte-identical files, so building
# both would be two names for one arm. (They diverge at pxq2, where balanced gives pxq3/pxq4
# and attn4 gives pxq4/pxq4hq, and at pxq4, where balanced buys attention up to pxq4hq.)
echo "ALL DONE $(date -u +%FT%TZ)"
