#!/usr/bin/env bash
# PXA_FA_TILE_V2 on the device: run the case list twice, once per schedule, and require the two
# dumps to be byte identical.
#
# The lever resolves once per process, so the comparison cannot happen inside one. This runs the
# harness with PXA_FA_TILE_V2=0 (the shipping tile-f16 schedule) and again with =1 (the swizzled
# schedule) and compares the raw fp32 output of every case with cmp.
#
# It also requires the v2 kernel's own "ENGAGED" line in the armed run. Without that check a card
# that never reaches the tile kernel at all -- anything but sm_60 -- would produce two identical
# dumps from the SAME kernel and report a green result that proves nothing.
#
# usage: test-fa-tile-v2-dev.sh /path/to/test-fa-tile-v2-dev [workdir]

set -u

BIN=${1:-}
WORK=${2:-${TMPDIR:-/tmp}}

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "usage: $0 /path/to/test-fa-tile-v2-dev [workdir]" >&2
    exit 2
fi

OFF="$WORK/fa-tile-v2-off.bin"
ON="$WORK/fa-tile-v2-on.bin"
OFF_LOG="$WORK/fa-tile-v2-off.log"
ON_LOG="$WORK/fa-tile-v2-on.log"

echo "== schedule A: PXA_FA_TILE_V2=0 (the shipping tile-f16 schedule) =="
PXA_FA_TILE_V2=0 "$BIN" "$OFF" > "$OFF_LOG" 2>&1
rc_off=$?
cat "$OFF_LOG"
if [ $rc_off -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

echo "== schedule B: PXA_FA_TILE_V2=1 (the swizzled schedule) =="
PXA_FA_TILE_V2=1 "$BIN" "$ON" > "$ON_LOG" 2>&1
rc_on=$?
cat "$ON_LOG"
if [ $rc_on -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

fail=0
[ $rc_off -ne 0 ] && { echo "FAIL: the shipping schedule run returned $rc_off"; fail=1; }
[ $rc_on  -ne 0 ] && { echo "FAIL: the swizzled schedule run returned $rc_on";  fail=1; }

if ! grep -q "PXA_FA_TILE_V2: ENGAGED" "$ON_LOG"; then
    echo "FAIL: the armed run never printed the v2 kernel's ENGAGED line -- the tile route was not"
    echo "      taken on this card, so byte equality here would prove nothing. This test is"
    echo "      meaningful only where fattn.cu sends a batch to the tile-f16 kernel (sm_60)."
    fail=1
fi

if cmp -s "$OFF" "$ON"; then
    echo "BITWISE EQUAL: $(stat -c%s "$OFF") bytes of fp32 output identical across both schedules"
else
    echo "FAIL: the two schedules disagree --"
    cmp -l "$OFF" "$ON" 2>/dev/null | head -20
    echo "      differing bytes: $(cmp -l "$OFF" "$ON" 2>/dev/null | wc -l) of $(stat -c%s "$OFF")"
    fail=1
fi

[ $fail -eq 0 ] && echo "OK" || echo "FAIL"
exit $fail
