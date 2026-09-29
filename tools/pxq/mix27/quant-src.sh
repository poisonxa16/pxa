#!/bin/bash
# quant-src.sh TAG FTYPE RULES NTHREADS -- one per-tensor-tier source file from the Q8_0 GGUF,
# with the RELEASE quantizer (pxa-rc-cut build-quant @153fa0d431, byte-identical to v1).
# The mix arms are then byte-merged out of these sources (mix-merge.py).
set -u
TAG=$1 FTYPE=$2 RULES=$3 NT=${4:-24}
QB=${QB:-/mnt/cacheone/pxa-rc-cut/build-quant}
OUT=/mnt/cachetwo/models/qwen38-27b-mix/src
docker run --rm --name b-mix27-q-$TAG -v $QB:/qb -v /mnt/user/models-cold:/models-cold -v $OUT:/out \
  pxa-sm60-dev:latest bash -c "LD_LIBRARY_PATH=/qb/src:/qb/ggml/src nice -n 19 /qb/bin/pxq-quantize --allow-requantize --i-know-this-is-double-lossy ${QEXTRA:-} --custom-q '$RULES' /models-cold/qwen38-27b-q8/Qwen3.8-27B-Q8_0.gguf /out/q38-27b-$TAG.gguf $FTYPE $NT"
echo "-- $TAG exit=$? size=$(stat -c %s $OUT/q38-27b-$TAG.gguf 2>/dev/null)"
