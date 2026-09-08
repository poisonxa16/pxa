#!/bin/bash
# Serial arm chain; waits for foreign experiment containers to clear, retries refusals.
D=/mnt/models/pxa-hth
wait_free() {
  while docker ps --format '{{.Names}}' | grep -qE 'pxa-(step|mtp|hth|awq-base)'; do sleep 30; done
}
run_one() {  # name conc lockgc mode arm extra...
  local NAME=$1 CONCL=$2 LGC=$3 MODE=$4 ARM=$5; shift 5
  for try in 1 2 3; do
    wait_free
    rm -f $D/results/$NAME.FAIL
    CONC=$CONCL LOCKGC=$LGC $D/run_arm2.sh $NAME $ARM $MODE "$@"
    [ -f $D/results/$NAME.DONE ] && return 0
    grep -q REFUSING $D/logs/$NAME.log && { sleep 60; continue; }
    return 1
  done
  return 1
}
OWNER="--gpu-memory-utilization 0.93 --max-model-len 200000 --enable-prefix-caching"
CGCFG='{"cudagraph_capture_sizes":[1,2,4,8,16]}'
run_one pxq4-cg 2,3,4,8,16 0 all pxq4 $OWNER --compilation-config "$CGCFG"   ; echo "CHAIN pxq4-cg rc=$?"
run_one awq-cg  2,3,4,8,16 0 all awq  $OWNER --compilation-config "$CGCFG"   ; echo "CHAIN awq-cg rc=$?"
run_one pxq4-lgc 2 1 singleprefill pxq4 $OWNER                                ; echo "CHAIN pxq4-lgc rc=$?"
run_one awq-ours 2 0 singleprefill awq --gpu-memory-utilization 0.85 --max-model-len 32768 --no-enable-prefix-caching ; echo "CHAIN awq-ours rc=$?"
echo CHAIN_ALL_DONE
