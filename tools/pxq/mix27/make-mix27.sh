#!/bin/bash
# make-mix27.sh [NTHREADS] -- THE RECIPE: Qwen3.8-27B PXQ mix27 (e20) straight from the Q8_0 source
# with the release quantizer (pxq-quantize). No merge step: the merged file used for the measurements
# is tensor-for-tensor identical to this output (verify-identity.py).
#   level PXQ3 balanced (FFN pxq3, attention pxq4, attn_k/v + ssm_alpha/beta + nextn.eh_proj q8_0)
#   + ssm_out            q8_0 -> pxq4   (-764 MiB, +0.0070 KLD)
#   + output.weight      q8_0 -> q6_K   (-293 MiB; must stay a k-quant for MTP, bug #204)
#   + ffn up/gate/down on blk 22-41 pxq3 -> pxq4 (+638 MiB; the middle layers carry the error)
set -u
NT=${1:-40}
QB=${QB:-/mnt/cacheone/pxa-rc-cut/build-quant}
OUT=${OUT:-/mnt/cachetwo/models/qwen38-27b-mix}
RULES='ssm_out\.weight=pxq4,^output\.weight=q6_K,^blk\.(2[2-9]|3[0-9]|4[01])\.ffn_(up|gate|down)\.weight=pxq4'
docker run --rm --name b-mix27-q-final -v $QB:/qb -v /mnt/user/models-cold:/models-cold -v $OUT:/out \
  pxa-sm60-dev:latest bash -c "LD_LIBRARY_PATH=/qb/src:/qb/ggml/src nice -n 19 /qb/bin/pxq-quantize --allow-requantize --i-know-this-is-double-lossy --custom-q '$RULES' /models-cold/qwen38-27b-q8/Qwen3.8-27B-Q8_0.gguf /out/Qwen3.8-27B-PXQ-mix27.gguf PXQ3 $NT"
echo "-- exit=$? size=$(stat -c %s $OUT/Qwen3.8-27B-PXQ-mix27.gguf 2>/dev/null)"
