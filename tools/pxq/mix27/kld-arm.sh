#!/bin/bash
# kld-arm.sh CARDS TAG GGUF [base]  -- in-distribution fidelity of one arm vs the Q8_0 source.
# Instrument (fixed for every arm): corpus qwen-chat-indist.txt (mkqwenchat.py, Qwen's own chat
# template, PXA_PPL_PARSE_SPECIAL=1 so turn markers are real tokens), -c 2048, CHUNKS chunks,
# -b/-ub 512, -fa on, -sm layer. 'base' writes the reference logits from the Q8_0 source.
# The caller holds the burst lock for CARDS.
set -u
CARDS=$1 TAG=$2 GGUF=$3 MODE=${4:-arm}
W=/mnt/cacheone/burst-b-mix27-wt
BLD=${BLD:-/mnt/cacheone/pxa-tsplit-V2/build-gpu}   # instrument build every arm was scored with (36e10ad954, sm 60;70)
K=/mnt/cachetwo/models/qwen38-27b-mix/kld
CHUNKS=${CHUNKS:-24}
BASE=$K/q8ref-c2048-n$CHUNKS.kldbase
LOG=/mnt/cacheone/burst-logs/b-mix27/kld; mkdir -p $LOG
if [ "$MODE" = base ]; then KARG="--kl-divergence-base /k/$(basename $BASE)"; else KARG="--kl-divergence-base /k/$(basename $BASE) --kl-divergence"; fi
t0=$SECONDS
docker run --rm --name b-mix27-kld-$TAG --runtime=nvidia --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=$CARDS -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PXA_PPL_PARSE_SPECIAL=1 -e LD_LIBRARY_PATH=/b/src:/b/ggml/src:/b/common \
  -v $BLD:/b:ro -v $K:/k -v "$(dirname $(readlink -f $GGUF))":/m:ro \
  pxa-sm60-dev:latest nice -n 10 /b/bin/llama-perplexity -m /m/$(basename $(readlink -f $GGUF)) -f /k/qwen-chat-indist.txt \
  -c 2048 --chunks $CHUNKS -b 512 -ub 512 -t 16 -ngl 99 -fa on -sm layer ${KEXTRA:-} $KARG > $LOG/$TAG.log 2>&1
rc=$?
echo "== $TAG rc=$rc $((SECONDS-t0))s $(stat -Lc %s $GGUF) B"
grep -aE "Final estimate|Mean    KLD|Mean KLD|Same top|Mean    PPL\(Q\)|Mean PPL|99.0%|Maximum KLD|RMS Δp" $LOG/$TAG.log | tail -n 10
