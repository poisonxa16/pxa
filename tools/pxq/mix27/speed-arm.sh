#!/bin/bash
# speed-arm.sh CARDS TAG GGUF MTP(0|1) [SM(layer|tensor)] -- one-card fit + speed probe for one arm.
# Boot: -c 131072 -ctk q4_0 -ctv q4_0 -ub 256 -fa on -np 1 (the target serving line), optional
# --spec-type mtp. Records: boot OK/OOM, nvidia-smi MiB per card after load and after the long
# prefill, decode t/s (greedy 512, control + repetition class, REPS 3, sha of the 512 tokens),
# prefill t/s at ~25k tokens (REPS 3), draft_n/accepted when MTP. 60 s warm-up before brackets.
# GATES="det:12 spread needle:400000" runs gate-client.py modes instead of the speed probe; NP=2 sets -np;
# POSTGATES runs gate modes after the speed probe in the same boot.
# The caller holds the burst lock for CARDS.
set -u
CARDS=$1 TAG=$2 GGUF=$3 MTP=${4:-0} SM=${5:-layer}
W=/mnt/cacheone/burst-b-mix27-wt; BLD=${BLD:-/mnt/cacheone/pxa-seat-engine/v2026.09.20}; HERE=$W/tools/pxq/mix27
PORT=${PORT:-18427}; NAME=b-mix27-srv-$TAG; REPS=${REPS:-3}; CTX=${CTX:-131072}
OUT=/mnt/cacheone/burst-logs/b-mix27/speed; mkdir -p $OUT; R=$OUT/$TAG.txt; : > $R
say(){ echo "$*" | tee -a $R; }
EXTRA=""; SENV="-e PXA_AUTO_SPEC=0"   # MTP=0 is a PLAIN arm: the server auto-arms an n-gram drafter otherwise
[ "$MTP" = 1 ] && { EXTRA="--spec-type mtp"; SENV="${XENV:-}"; }
M=$(readlink -f $GGUF)
docker rm -f $NAME >/dev/null 2>&1
# drain-and-verify: docker rm -f returns before the CUDA context is gone
for c in ${CARDS//,/ }; do for i in $(seq 1 60); do u=$(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 200 ] && break; sleep 2; done; [ "$u" -lt 200 ] || { say "DRAIN FAIL card $c at $u MiB"; exit 3; }; done
docker run -d --name $NAME --runtime=nvidia --ipc=host -p 127.0.0.1:$PORT:8080 $SENV \
  -e NVIDIA_VISIBLE_DEVICES=$CARDS -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e LD_LIBRARY_PATH=/b/lib:/b/bin:/b/src:/b/ggml/src:/b/common:/b/examples/mtmd -v $BLD:/b:ro -v $(dirname $M):/m:ro \
  pxa-sm60-dev:latest /b/bin/llama-server -m /m/$(basename $M) -ngl 99 -c $CTX -np ${NP:-1} -t 16 -fa on \
  -ctk q4_0 -ctv q4_0 -ub 256 -b 2048 -sm $SM $EXTRA ${XARGS:-} --host 0.0.0.0 --port 8080 --no-context-shift >/dev/null
ok=0
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { ok=1; break; }
  docker ps --filter name=^$NAME$ --format '{{.Status}}' | grep -q Up || break
  sleep 5
done
docker logs $NAME > $OUT/$TAG.server.log 2>&1
smi(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
if [ $ok != 1 ]; then
  say "BOOT FAIL $TAG mtp=$MTP sm=$SM: $(grep -aiE 'out of memory|failed to allocate|error' $OUT/$TAG.server.log | head -n 3 | tr '\n' '|')"
  docker rm -f $NAME >/dev/null 2>&1; exit 2
fi
say "BOOT OK $TAG mtp=$MTP sm=$SM ctx=$CTX smi-after-load: $(smi)"
grep -aE "KV self size|CUDA0 KV buffer|compute buffer size|model buffer size|mtp|MTP" $OUT/$TAG.server.log | head -n 12 >> $R
PK=$OUT/$TAG.peak; ( while :; do nvidia-smi -i ${CARDS%%,*} --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > $PK 2>/dev/null & PKPID=$!
gates(){ for g in $1; do python3 $HERE/gate-client.py $PORT ${g%%:*} $TAG $( [ "${g#*:}" != "$g" ] && echo ${g#*:} ) 2>&1 | grep -aE "^GATE|Error|error" | tail -n 2 | tee -a $R; done; }
if [ -n "${GATES:-}" ]; then gates "$GATES"
elif [ -n "${DEEP:-}" ]; then python3 $HERE/deep-client.py $PORT $DEEP 2>&1 | tail -n 1 | tee -a $R
else python3 $HERE/speed-client.py $PORT $REPS $TAG 2>&1 | grep -v "^ \|^Traceback\|^  File" | tee -a $R; fi
[ -n "${POSTGATES:-}" ] && gates "$POSTGATES"
curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok || say "RUNTIME FAIL $TAG: $(docker logs $NAME 2>&1 | grep -aiE 'out of memory|cudaMalloc failed' | tail -n 2 | tr '\n' '|')"
kill $PKPID 2>/dev/null; say "peak MiB card ${CARDS%%,*} during run (1 s samples): $(sort -n $PK | tail -n 1)"
say "smi-after-run: $(smi)"
docker logs $NAME > $OUT/$TAG.server.log 2>&1
docker rm -f $NAME >/dev/null 2>&1
for c in ${CARDS//,/ }; do for i in $(seq 1 60); do [ $(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits) -lt 200 ] && break; sleep 2; done; done
