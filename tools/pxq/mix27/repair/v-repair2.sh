#!/bin/bash
# continuation of v-repair.sh on V2 (lock still held): MTP arm on eng3, then the np2 determinism controls.
set -u
H=/mnt/cacheone/burst-b-mix27-wt/tools/pxq/mix27; L=/mnt/cacheone/burst-locks; C=2; export PORT=18441
E=/mnt/cachetwo/models/qwen38-27b-mix/Qwen3.8-27B-PXQ-mix27.gguf; B=/mnt/cacheone/models/quantaudit/Qwen3.8-27B-PXQ3-balanced-ssmout-q8_0.gguf
FIX=/mnt/cacheone/burst-logs/b-mtpfit/eng3
while kill -0 3512667 2>/dev/null; do sleep 5; done
BLD=$FIX XARGS="--recurrent-ckpt-mode gpu-fallback" POSTGATES="needle:400000" bash $H/speed-arm.sh $C vf-e20-mtp1 $E 1
NP=2 GATES="detq:12" bash $H/speed-arm.sh $C vg-e20-np2q $E 0
NP=2 GATES="det:12" bash $H/speed-arm.sh $C vg-B-np2 $B 0
$L/release.sh V2 b-mix27; pxa-mail windows unhold v100 b-mix27 >/dev/null 2>&1
echo "$(date '+%T') RELEASED V2"
pxa-mail post --from b-mix27 --to b-stream --tag status --room burst "b-mix27: V2 RELEASED $(date '+%H:%M') EDT, drained. Yours." >/dev/null 2>&1
