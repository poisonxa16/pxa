#!/usr/bin/env bash
# test-sched-async-gate.sh -- PXA_SCHED_ASYNC_INPUTS / PXA_SCHED_ASYNC_MAX_TOKENS regression (beta-2 R2, 2026-10-19).
#
# The async split-input path drains at alloc_graph and stages inputs through a pinned ring; since
# PXA_SCHED_ASYNC_MAX_TOKENS a ubatch wider than the gate takes the blocking path on a scheduler that HAS the async
# slot events. Two crashes came out of that mix on one V100 (27B qwen35, 20.5k-token prompt, -ub 2048): a wide graph
# staging into an unopened ring generation, then the blocking path syncing on a slot event that marks only the last
# split with inputs (cuBLAS "internal operation failed" in the first prompt ubatch). This script runs one long prompt
# through back-to-back wide ubatches and decode steps with the gate on, the lever off and the gate off, and requires
# the same greedy text from all three (copy scheduling only: bit-identical) and no crash.
#
#   test-sched-async-gate.sh <llama-cli> <model.gguf> [prompt-file] [extra args...]
# Needs a GPU (exit 77 = skip without one). Default prompt: 6000 words of filler (~8k tokens), -ub 512 -b 512, -n 32.
set -u
CLI=${1:-}; MODEL=${2:-}; PF=${3:-}
shift $(( $# < 3 ? $# : 3 ))
[ -x "$CLI" ] && [ -f "$MODEL" ] || { echo "SKIP: usage: $0 <llama-cli> <model.gguf> [prompt-file] [args]"; exit 77; }
nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
if [ -z "$PF" ]; then
    PF=$TMP/prompt.txt
    for i in $(seq 1 400); do echo "Section $i. The river bends east of the old mill and the road follows it for a mile before the bridge."; done > "$PF"
    echo "Summarize the text above in one sentence." >> "$PF"
fi
run() { # label env...
    local label=$1; shift
    env "$@" "$CLI" -m "$MODEL" -ngl 999 -fa on -ub 512 -b 512 -c 16384 -n 32 --temp 0 -f "$PF" "${EXTRA[@]}" \
        > "$TMP/$label.out" 2> "$TMP/$label.err"
    echo $?
}
EXTRA=("$@")
rc_gate=$(run gate PXA_SCHED_ASYNC_INPUTS=1 PXA_SCHED_ASYNC_MAX_TOKENS=64)
rc_off=$(run off PXA_SCHED_ASYNC_INPUTS=0)
rc_all=$(run all PXA_SCHED_ASYNC_INPUTS=1 PXA_SCHED_ASYNC_MAX_TOKENS=0)
fail=0
for l in gate off all; do
    rc=$(eval echo \$rc_$l)
    if [ "$rc" != 0 ]; then echo "FAIL $l: exit $rc: $(grep -m2 -i 'error\|abort' "$TMP/$l.err" | tr '\n' ' ')"; fail=1; fi
done
if [ $fail = 0 ]; then
    for l in gate all; do
        if ! cmp -s "$TMP/off.out" "$TMP/$l.out"; then echo "FAIL $l: text differs from PXA_SCHED_ASYNC_INPUTS=0"; fail=1; fi
    done
fi
[ $fail = 0 ] && echo "PASS: gate / off / all-async: same greedy text, no crash ($(wc -c < "$TMP/off.out") bytes)"
exit $fail
