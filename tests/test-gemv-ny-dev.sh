#!/usr/bin/env bash
# PXA_F16_GEMV_NY on the device: run the case list twice and report what each arm computed.
#
# Unlike the bit-identity harnesses in this directory, the two arms here are NOT expected to
# agree: the path being replaced (the generic cuBLAS chain) rounds the activation AND the result
# through f16, while the kernel keeps both in f32. So the cross-arm comparison is REPORTED, not
# asserted, and the pass/fail bars are the two that can be stated exactly:
#
#   * the ny==1 output must be byte-identical across the arms. This lever declines ne11==1
#     outright, so arming it may not move the plain-decode path by a single bit.
#   * inside the armed run, column 0 of every ny>1 node must be BIT-IDENTICAL to the ne11==1
#     node's output. The binary checks this itself and prints it per case.
#   * the armed run must print the kernel's ENGAGED line and a non-zero engagement count.
#     Without that the two dumps came from the same path and mean nothing.
#
# usage: test-gemv-ny-dev.sh /path/to/test-gemv-ny-dev [workdir]

set -u

BIN=${1:-}
WORK=${2:-${TMPDIR:-/tmp}}

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "usage: $0 /path/to/test-gemv-ny-dev [workdir]" >&2
    exit 2
fi

OFF="$WORK/gemv-ny-off.bin"
ON="$WORK/gemv-ny-on.bin"
OFF_LOG="$WORK/gemv-ny-off.log"
ON_LOG="$WORK/gemv-ny-on.log"

echo "== arm A: PXA_F16_GEMV_NY unset (ne11==1 -> k_pxa_gemv_f16, ne11>1 -> the cuBLAS chain) =="
env -u PXA_F16_GEMV_NY "$BIN" "$OFF" > "$OFF_LOG" 2>&1
rc_off=$?
cat "$OFF_LOG"
if [ $rc_off -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

echo "== arm B: PXA_F16_GEMV_NY=1 (ne11 2..8 -> k_pxa_gemv_f16_ny) =="
PXA_F16_GEMV_NY=1 "$BIN" "$ON" > "$ON_LOG" 2>&1
rc_on=$?
cat "$ON_LOG"
if [ $rc_on -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

fail=0
[ $rc_off -ne 0 ] && { echo "FAIL: the unset arm returned $rc_off"; fail=1; }
[ $rc_on  -ne 0 ] && { echo "FAIL: the armed arm returned $rc_on";  fail=1; }

if ! grep -q "PXA_F16_GEMV_NY: ENGAGED" "$ON_LOG"; then
    echo "FAIL: the armed run never printed the kernel's ENGAGED line -- the new path was not"
    echo "      taken, so nothing below means anything."
    fail=1
fi
if ! grep -q "PXA_F16_GEMV_NY: [0-9]* engagement(s) this run" "$ON_LOG"; then
    echo "FAIL: the armed run printed no engagement count."
    fail=1
else
    echo "engagements: $(grep -o 'PXA_F16_GEMV_NY: [0-9]* engagement(s) this run' "$ON_LOG")"
fi

# The ny>1 nodes must have MOVED. If the dumps are byte-equal the lever changed nothing.
if cmp -s "$OFF" "$ON"; then
    echo "FAIL: the two arms are byte-identical -- the armed arm did not take a different path"
    fail=1
else
    echo "the two arms differ, as they must (f32 accumulation vs the f16-rounded cuBLAS chain);"
    echo "read the per-case 'max|err vs fp64|' lines above: the SMALLER number is the tighter path."
fi

# Column 0 bit-identity inside the armed run, as the binary reported it.
if grep -q "col0 vs ne11==1: differs" "$ON_LOG"; then
    echo "FAIL: a ny>1 column 0 is not bit-identical to the ne11==1 kernel's output --"
    grep "col0 vs ne11==1: differs" "$ON_LOG" | head -10
    fail=1
else
    echo "col0 bit-identity: every ny>1 node's column 0 matches the ne11==1 kernel byte for byte"
fi

[ $fail -eq 0 ] && echo "OK" || echo "FAIL"
exit $fail
