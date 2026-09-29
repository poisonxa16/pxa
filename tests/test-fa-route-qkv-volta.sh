#!/usr/bin/env bash
# test-fa-route-qkv-volta.sh -- bug #236 route check, V100 (cc 700) only.
#
# On sm_70 a narrow flash-attention node over a quantized K/V cache went to WMMA, which converts the
# whole K and V view to f16 on every node of every token -- a scratch that grows with the context
# (376 MiB at n_kv 96k on Qwen3.8-27B q4_0) and OOMed one V100 at depth. PXA_FA_DEEP_QKV_TILE
# (default on) sends such a node to the vec kernel, reading the quantized rows in place, once the
# whole-cache conversion would exceed 128 MiB. Below that WMMA stays: on this base it measured
# faster than vec at width 1 at 1.5k-25k (impl mtpfit-fa-qkv-narrow-vec, REPS 3, V100).
# test-fa-route-qkv checks support and accuracy; this script adds the route, read from the
# PXA_CORE_ROUTES census the engine prints at exit.
#
#   deep   (D=256, 2 KV heads, n_kv 69632 -> 136 MiB), width 1 and 4, PXA_FA_DEEP_QKV_TILE=2 (the
#          unconditional 128 MiB rule): vec-f16, never wmma-f16 (on a tree without the fix: FAIL)
#   deep, width 1, default env (AUTO, memory-aware since ws6-fix, R9): the route the engine announces --
#          "AUTO keeps the release route" (the whole staging fits with a 256 MiB margin) -> wmma-f16,
#          "engaged ... does not fit" -> vec-f16 (rel-integrate3: the old rows assumed the fixed rule)
#   deep, PXA_FA_DEEP_QKV_TILE=0: wmma-f16                      (the revert switch works)
#   shallow (n_kv 1024), width 1, default env: wmma-f16         (the measured-faster route is kept)
#
# Usage: test-fa-route-qkv-volta.sh [path/to/test-fa-route-qkv]. Exit 0 pass, 1 fail, 77 skip.
set -u
BIN=${1:-$(dirname "$0")/test-fa-route-qkv}
[ -x "$BIN" ] || { echo "SKIP: $BIN not found"; exit 77; }
DEEP=69632
fail=0
check() { # label want forbid env... -- args
    local label=$1 want=$2 forbid=$3; shift 3
    local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local out rc routes ok=0
    out=$(env PXA_CORE_ROUTES=1 "${envs[@]}" "$BIN" "$@" 2>&1); rc=$?
    if ! grep -q "cc 700" <<<"$out"; then echo "SKIP: device 0 is not a V100 (cc 700)"; exit 77; fi
    routes=$(grep -E "PXA_CORE_ROUTES: +device 0 " <<<"$out" | grep -v TOTAL | awk '{print $4}' | tr '\n' ' ')
    if [ "$want" = auto ]; then                  # AUTO: the engine's own banner says which route it chose
        if grep -q "AUTO keeps the release route" <<<"$out"; then want=wmma-f16; forbid=vec-f16
        elif grep -q "PXA_FA_DEEP_QKV_TILE: engaged" <<<"$out"; then want=vec-f16; forbid=wmma-f16
        else want="(no AUTO banner)"; fi
    fi
    grep -qw -- "$want" <<<"$routes" && ok=1
    [ -n "$forbid" ] && grep -qw "$forbid" <<<"$routes" && ok=0
    [ $rc -eq 0 ] || ok=0
    printf '  %s %s: routes [%s] want %s%s, accuracy exit %d\n' "$([ $ok = 1 ] && echo 'ok  ' || echo FAIL)" \
        "$label" "${routes% }" "$want" "${forbid:+ not $forbid}" "$rc"
    [ $ok = 1 ] || fail=1
}
for t in q4_0 q8_0; do
    for w in 1 4; do
        check "D=256 $t width $w n_kv $DEEP PXA_FA_DEEP_QKV_TILE=2" vec-f16 wmma-f16 PXA_FA_DEEP_QKV_TILE=2 -- 256 "$t" "$w" "$DEEP"
    done
    check "D=256 $t width 1 n_kv $DEEP default (AUTO)" auto "" -- 256 "$t" 1 "$DEEP"
    check "D=256 $t width 1 n_kv $DEEP PXA_FA_DEEP_QKV_TILE=0" wmma-f16 "" PXA_FA_DEEP_QKV_TILE=0 -- 256 "$t" 1 "$DEEP"
    check "D=256 $t width 1 n_kv 1024 default" wmma-f16 "" -- 256 "$t" 1 1024
done
[ $fail = 0 ] && echo PASSED || echo FAILED
exit $fail
