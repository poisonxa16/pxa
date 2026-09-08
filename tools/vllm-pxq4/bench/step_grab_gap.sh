#!/bin/bash
# usage: grab_gap.sh <tag> <arm>
TAG=${1:?}; ARM=${2:?}
end=$(( $(date +%s) + 7200 ))
while [ $(date +%s) -lt $end ]; do
  up=$(docker ps --format "{{.Names}}" | grep -cE "pxa-(hth|awq|mtp|step)")
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | tr -d " ")
  if [ "$up" = 0 ] && [ "${mem:-9999}" -lt 2000 ]; then
    echo "GAP-$TAG at $(date -u +%H:%M:%S)" >> /mnt/models/pxa-step/logs/grab.log
    cat >> /mnt/models/pxa-mtp/COORDINATION.md <<NOTE

## step-budget agent ($(date -u +%H:%M) UTC) — GRABBED gap (automated)
- pxa-step-$TAG launching NOW ($ARM bench+profile), auto-removes, ~20 min. DONE note follows.
NOTE
    bash /mnt/models/pxa-step/boot_arm.sh $TAG $ARM >> /mnt/models/pxa-step/logs/grab.log 2>&1
    rc=$?
    echo "- ($(date -u +%H:%M)) step-budget $TAG DONE (rc=$rc). Cards 4-7 free." >> /mnt/models/pxa-mtp/COORDINATION.md
    exit $rc
  fi
  sleep 8
done
echo "grab_gap $TAG timed out" >> /mnt/models/pxa-step/logs/grab.log
