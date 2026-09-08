#!/usr/bin/env bash
# Watch for new benchmark result JSONs and append a compact digest.
# Runs detached on the DGX: an ssh held open longer than ~4 minutes gets killed
# and takes the wait with it, so the waiting lives here and the coordinator just
# reads digest.txt.
D=/mnt/models/pxa-k3/digest.txt
SEEN=/mnt/models/pxa-k3/.seen
touch "$SEEN"
while true; do
  for f in /mnt/models/pxa-hth/results/*.json /mnt/models/pxa-step/results/*.json; do
    [ -e "$f" ] || continue
    grep -qxF "$f" "$SEEN" && continue
    echo "$f" >> "$SEEN"
    {
      echo "=== $(date -u +%FT%TZ)  $f"
      python3 /mnt/models/pxa-k3/digest.py "$f"
    } >> "$D" 2>&1
  done
  sleep 20
done
