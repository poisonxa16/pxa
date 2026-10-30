#!/usr/bin/env bash
# PXA_XCACHE_OT CPU bit-identity gate (glm-mc-2-xcache-multidevice): tiny glm5next, static -ot expert placement vs the
# expert split under that same -ot (PXA_XCACHE_OT=n_hot), raw logits of prefill + greedy decode compared with cmp.
#   tests/run-xcache-ot-bitid.sh [build_dir=build-cpu]
set -euo pipefail
BD=${1:-build-cpu}
# a CPU gate: hide the GPUs of a CUDA build (with a card visible, -ngl 0 still offloads big-batch matmuls of host weights,
# which the split and the static placement do not do alike); the no-crash check at the end runs with the cards visible
GPU_ENV_CUDA=${CUDA_VISIBLE_DEVICES-__unset__}
export CUDA_VISIBLE_DEVICES=""
B=$BD/bin/test-pxa-xcache-ot-logits
T=$(mktemp -d)
python3 tests/gen_tiny_glm5next.py "$T/tiny.gguf" --seed 1234 >/dev/null
python3 - "$T/tiny.csv" <<'PY'
import random, sys
random.seed(7)
with open(sys.argv[1], "w") as f:
    f.write("tensor,expert,count\n")
    for il in range(3, 8):
        for e in range(8): f.write(f"blk.{il}.ffn_down_exps.weight,{e},{random.randint(0, 1000)}\n")
PY
fail=0
for mm in 0 1; do
  "$B" "$T/tiny.gguf" "$T/static.bin" 40 24 $mm >/dev/null 2>&1
  for h in 1 2 3 5 7; do
    PXA_XCACHE_OT=$h PXA_XCACHE_OT_COUNTS="$T/tiny.csv" "$B" "$T/tiny.gguf" "$T/ot.bin" 40 24 $mm > "$T/ot.log" 2>&1
    grep -q "split into hot + cold stacks" "$T/ot.log" || { echo "mmap=$mm n_hot=$h: split NOT engaged"; fail=1; continue; }
    if cmp -s "$T/static.bin" "$T/ot.bin"; then echo "mmap=$mm n_hot=$h BIT-IDENTICAL ($(stat -c %s "$T/ot.bin") bytes of logits)"
    else echo "mmap=$mm n_hot=$h DIFF"; fail=1; fi
  done
done
# PXA_XCACHE_ADAPT spares (the GLM-5.x work M1 step 2): free cold slots appended to every split layer must not change a single logit
for h in 2 5; do
  PXA_XCACHE_ADAPT=1 PXA_XCACHE_SPARE=3 PXA_XCACHE_OT=$h PXA_XCACHE_OT_COUNTS="$T/tiny.csv" "$B" "$T/tiny.gguf" "$T/ot.bin" 40 24 0 > "$T/ot.log" 2>&1
  sp=$(grep -c "free cold slots per layer" "$T/ot.log")
  if cmp -s "$T/static.bin" "$T/ot.bin"; then echo "adapt spares n_hot=$h BIT-IDENTICAL (spares planned: $([ "$sp" -gt 0 ] && echo yes || echo 'no: no adaptation policy in this build'))"
  else echo "adapt spares n_hot=$h DIFF"; fail=1; fi
done
# GPU-visible no-crash check (CUDA builds): the ADAPT adaptor must skip a host-resident split, not hand -1 to CUDA
if [ "$GPU_ENV_CUDA" = __unset__ ]; then unset CUDA_VISIBLE_DEVICES; else export CUDA_VISIBLE_DEVICES="$GPU_ENV_CUDA"; fi
if PXA_XCACHE_ADAPT=1 PXA_XCACHE_SPARE=3 PXA_XCACHE_OT=2 PXA_XCACHE_OT_COUNTS="$T/tiny.csv" "$B" "$T/tiny.gguf" "$T/ot.bin" 40 24 0 > "$T/ot.log" 2>&1; then
  echo "adapt spares n_hot=2 with the devices visible: runs (rc 0)"
else echo "adapt spares n_hot=2 with the devices visible: CRASH rc=$?"; grep -m3 -E 'error|abort' "$T/ot.log" || true; fail=1; fi
rm -rf "$T"
[ $fail = 0 ] && echo "run-xcache-ot-bitid: OK" || { echo "run-xcache-ot-bitid: FAIL"; exit 1; }
