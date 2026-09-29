#!/bin/bash
# repair pass, one P100 each on cards 0 and 1, same bracket, release binary:
#   card 0: PXQ3-balanced plain 131k REPS 3   | card 1: mix27 plain 131k REPS 3
#   then card 1: mix27 in-dist KLD on one P100 (completes the P100 KLD column)
set -u
H=/mnt/cacheone/burst-b-mix27-wt/tools/pxq/mix27; L=/mnt/cacheone/burst-locks
B=/mnt/cacheone/models/quantaudit/Qwen3.8-27B-PXQ3-balanced-ssmout-q8_0.gguf; E=/mnt/cachetwo/models/qwen38-27b-mix/Qwen3.8-27B-PXQ-mix27.gguf
echo "$(date '+%T') waiting for P01"
until $L/hold.sh P01 b-mix27 30 >/dev/null 2>&1; do sleep 10; done
echo "$(date '+%T') HOLD P01"; pxa-mail windows hold p100 b-mix27 --pid $$ >/dev/null 2>&1
PORT=18431 bash $H/speed-arm.sh 0 p1r-B-mtp0 $B 0 & a=$!
PORT=18432 bash $H/speed-arm.sh 1 p1r-e20-mtp0 $E 0 & b=$!
wait $a $b
echo "$(date '+%T') speed done"
bash $H/kld-arm.sh 1 p1one-mix27-e20 $E & k=$!
F3=$(cat /mnt/cacheone/burst-logs/b-mix27/repair/eng3.path 2>/dev/null)
if [ -n "$F3" ] && [ -x "$F3/bin/llama-server" ]; then
  echo "$(date '+%T') P100 MTP arm on $F3"
  PORT=18433 BLD=$F3 XARGS="--recurrent-ckpt-mode gpu-fallback" bash $H/speed-arm.sh 0 p1r-e20-mtp1-eng3 $E 1
else echo "$(date '+%T') eng3 not staged, P100 MTP arm skipped"; fi
wait $k; echo "$(date '+%T') kld done"
$L/release.sh P01 b-mix27; pxa-mail windows unhold p100 b-mix27 >/dev/null 2>&1
echo "$(date '+%T') RELEASED P01"
pxa-mail post --from b-mix27 --to b-flashnext --tag status --room burst "b-mix27: P01 RELEASED $(date '+%H:%M') EDT (repair P100 arms done)." >/dev/null 2>&1
