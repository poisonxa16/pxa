#!/bin/bash
# build-mix.sh NAME -- the candidate tier maps, merged from the per-tier source files.
# Sensitivity (in-dist KLD, one group per arm, see README): FFN layers 16-47 carry most of the error
# (pxq3->pxq4 per 16-layer band: b0 -0.0065, b1 -0.0339, b2 -0.0366, b3 -0.0092); head pxq4 costs
# +0.0010 for -644 MiB, ssm_out pxq4 +0.0070 for -764 MiB, attention pxq3 (all) +0.0256 for -660 MiB.
set -u
N=$1
B=/mnt/cacheone/models/quantaudit/Qwen3.8-27B-PXQ3-balanced-ssmout-q8_0.gguf
P4=/mnt/cachetwo/models/qwen38-27b-stock/Qwen3.8-27B-PXQ4.gguf
S=/mnt/cachetwo/models/qwen38-27b-mix/src; P6=$S/q38-27b-p6.gguf; P3A=$S/q38-27b-p3a.gguf; Q6H=$S/q38-27b-q6h.gguf
O=/mnt/cachetwo/models/qwen38-27b-mix/arms; mkdir -p $O
MID='^blk\.(1[6-9]|2[0-9]|3[0-9]|4[0-7])\.ffn_(up|gate|down)\.weight$'
BASE="--take $P3A=(^output|ssm_out)\.weight$ --take $P4=$MID --take $P4=^blk\.64\.ffn_(up|gate|down)\.weight$"
case $N in
  mix27-a) T="$BASE";;
  mix27-b) T="$BASE --take $P3A=^blk\.([0-9]|1[0-5]|4[89]|5[0-9]|6[0-3])\.attn_(qkv|gate|q|output)\.weight$ --take $P4=^blk\.(8|9|1[0-5]|4[89]|5[0-5])\.ffn_(up|gate)\.weight$";;
  mix27-c) T="$BASE --take $P4=^blk\.(1[2-5]|4[89]|5[01])\.ffn_(up|gate|down)\.weight$";;
  mix27-d) T="--take $P3A=(^output|ssm_out)\.weight$ --take $P4=^blk\.(2[0-9]|3[0-9]|4[0-3])\.ffn_(up|gate|down)\.weight$";;   # a minus L16-19,L44-47 and blk.64 FFN back to pxq3: -287 MiB for the MTP fit
  # e-series: MTP-capable. The head must stay a k-quant (bug #204: a PXQ head + MTP dequantizes the whole
  # head into the pool), so head = q6_K (-293 MiB vs q8_0, MMQ path); ssm_out pxq4; FFN pxq4 on a middle window.
  mix27-e32) T="--take $P3A=ssm_out\.weight$ --take $Q6H=^output\.weight$ --take $P4=^blk\.(1[6-9]|2[0-9]|3[0-9]|4[0-7])\.ffn_(up|gate|down)\.weight$";;
  mix27-e26) T="--take $P3A=ssm_out\.weight$ --take $Q6H=^output\.weight$ --take $P4=^blk\.(19|2[0-9]|3[0-9]|4[0-4])\.ffn_(up|gate|down)\.weight$";;
  mix27-e20) T="--take $P3A=ssm_out\.weight$ --take $Q6H=^output\.weight$ --take $P4=^blk\.(2[2-9]|3[0-9]|4[01])\.ffn_(up|gate|down)\.weight$";;
  *) echo unknown; exit 2;;
esac
python3 $(dirname $0)/mix-merge.py $O/$N.gguf $B $T --kv pxa.policy.name=$N --kv pxa.mix.recipe="$T"
