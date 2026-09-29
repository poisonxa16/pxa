#!/bin/bash
# arms.sh CARD ARM... -- merge each sensitivity arm from PXQ3-balanced (B) and score it (kld-arm.sh).
# Every arm changes ONE tensor group relative to B, so its dKLD and its byte delta are that group's.
set -u
CARD=$1; shift
H=/mnt/cacheone/burst-b-mix27-wt/tools/pxq/mix27
B=/mnt/cacheone/models/quantaudit/Qwen3.8-27B-PXQ3-balanced-ssmout-q8_0.gguf
P4=/mnt/cachetwo/models/qwen38-27b-stock/Qwen3.8-27B-PXQ4.gguf
S=/mnt/cachetwo/models/qwen38-27b-mix/src; P6=$S/q38-27b-p6.gguf; P3A=$S/q38-27b-p3a.gguf
A=/mnt/cachetwo/models/qwen38-27b-mix/arms; mkdir -p $A
# BASEF: the file every arm is merged onto (default PXQ3-balanced). b1 = B with head+ssm_out at pxq4 (-1.46 GB),
# used so that byte-ADDING arms still fit one 16 GB card at the fixed instrument (-ub 512).
B0=$B; B=${BASEF:-$B}; PFX=${PFX:-}
FFN='\.ffn_(up|gate|down)\.weight$'; ATT='\.attn_(qkv|gate|q|output)\.weight$'
band(){ case $1 in 0) echo '^blk\.([0-9]|1[0-5])';; 1) echo '^blk\.(1[6-9]|2[0-9]|3[01])';; 2) echo '^blk\.(3[2-9]|4[0-7])';; 3) echo '^blk\.(4[89]|5[0-9]|6[0-3])';; esac; }
for arm in "$@"; do
  case $arm in
    B) f=$B0;; P4) f=$P4;;
    b1) f=$A/b1.gguf; B=$B0; T="--take $P3A=(^output|ssm_out)\.weight$";;
    ffn4-b[0-3]) f=$A/$PFX$arm.gguf; T="--take $P4=$(band ${arm: -1})$FFN";;
    ffn6-b[0-3]) f=$A/$PFX$arm.gguf; T="--take $P6=$(band ${arm: -1})$FFN";;
    down4) f=$A/$PFX$arm.gguf; T="--take $P4=^blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_down\.weight$";;
    gu4) f=$A/$PFX$arm.gguf; T="--take $P4=^blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(up|gate)\.weight$";;
    ssm6) f=$A/$arm.gguf; T="--take $P6=ssm_out\.weight$";;
    ssm4) f=$A/$arm.gguf; T="--take $P3A=ssm_out\.weight$";;
    head6) f=$A/$arm.gguf; T="--take $P6=^output\.weight$";;
    head4) f=$A/$arm.gguf; T="--take $P3A=^output\.weight$";;
    attn3) f=$A/$arm.gguf; T="--take $P3A=^blk\.([0-9]|[1-5][0-9]|6[0-3])$ATT";;
    attn6) f=$A/$PFX$arm.gguf; T="--take $P6=^blk\.([0-9]|[1-5][0-9]|6[0-3])$ATT";;
    mix:*) f=$A/${arm#mix:}.gguf;;   # prebuilt candidate
    *) echo "unknown arm $arm"; continue;;
  esac
  if [ ! -e "$f" ]; then python3 $H/mix-merge.py $f $B $T | grep -v '^wrote' ; fi
  echo "arm $arm: $(python3 $H/mix-merge.py /dev/null $f --dry | tail -n 1)"
  $H/kld-arm.sh $CARD $(echo $PFX$arm | tr : -) $f
  case $arm in B|P4|b1|mix:*) ;; *) rm -f $f;; esac
done
