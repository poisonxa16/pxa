#!/bin/bash
# repair pass on ONE V100: v-repair.sh SET CARD FIXBUILD
#  1 release np1: det 12/12 + logit spread + needle ~95k tokens   2 release np2: det 12/12 (slot 1 busy) + spread
#  3 fixed build plain REPS 3                                       4 fixed build MTP REPS 3 + needle ~95k (deep MTP)
set -u
S=$1 C=$2
H=/mnt/cacheone/burst-b-mix27-wt/tools/pxq/mix27; L=/mnt/cacheone/burst-locks
E=/mnt/cachetwo/models/qwen38-27b-mix/Qwen3.8-27B-PXQ-mix27.gguf
echo "$(date '+%T') waiting for $S"
until $L/hold.sh $S b-mix27 50 >/dev/null 2>&1; do sleep 10; done
echo "$(date '+%T') HOLD $S"; pxa-mail windows hold v100 b-mix27 --pid $$ >/dev/null 2>&1
export PORT=18441
GATES="det:12 spread needle:400000" bash $H/speed-arm.sh $C vg-e20-np1 $E 0
NP=2 GATES="det:12 spread" bash $H/speed-arm.sh $C vg-e20-np2 $E 0
for i in $(seq 1 120); do FIX=$(cat /mnt/cacheone/burst-logs/b-mix27/repair/eng3.path 2>/dev/null); [ -n "$FIX" ] && [ -x "$FIX/bin/llama-server" ] && break; sleep 10; done
echo "$(date '+%T') fixed build: ${FIX:-NONE}"
[ -n "$FIX" ] && BLD=$FIX bash $H/speed-arm.sh $C vf-e20-mtp0 $E 0
[ -n "$FIX" ] && BLD=$FIX XARGS="--recurrent-ckpt-mode gpu-fallback" POSTGATES="needle:400000" bash $H/speed-arm.sh $C vf-e20-mtp1 $E 1
$L/release.sh $S b-mix27; pxa-mail windows unhold v100 b-mix27 >/dev/null 2>&1
echo "$(date '+%T') RELEASED $S"
