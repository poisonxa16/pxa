#!/bin/bash
# sass-compare.sh <old_xelf_bdir> <new_xelf_bdir> [jobs] : per fatbin and arch sm_60/61/70/75: BYTES-SAME, or compare
# cuobjdump -sass text raw and with the nvcc arch-list tag normalised ("_860" in CCCL names, ",860" in the
# __CUDA_ARCH_LIST__ string, and the CCCL ABI namespace CUB_/THRUST_<ver>_<archs>_NS with its mangled length) -> SASS-SAME(RAW-SASS-SAME|RAW-SASS-DIFF) or SASS-DIFF.
one() { o=$(ls "$1/x/$3/"*.sm_$4.cubin 2>/dev/null); n=$(ls "$2/x/$3/"*.sm_$4.cubin 2>/dev/null)
  [ -n "$o" ] && [ -n "$n" ] || { echo "MISSING $3 sm_$4"; return; }
  cmp -s "$o" "$n" && { echo "BYTES-SAME $3 sm_$4"; return; }
  t=$(mktemp -d); cuobjdump -sass "$o" > $t/o; cuobjdump -sass "$n" > $t/n
  if cmp -s $t/o $t/n; then echo "SASS-SAME(RAW-SASS-SAME) $3 sm_$4"
  elif cmp -s <(norm < $t/o) <(norm < $t/n); then echo "SASS-SAME(RAW-SASS-DIFF) $3 sm_$4"
  else norm < $t/o | grep Function | sort > $t/fo; norm < $t/n | grep Function | sort > $t/fn
    echo "SASS-DIFF $3 sm_$4 lines=$(diff $t/o $t/n | grep -c '^[<>]') funcs=$(wc -l < $t/fo)/$(wc -l < $t/fn) only-old=$(comm -23 $t/fo $t/fn | wc -l) only-new=$(comm -13 $t/fo $t/fn | wc -l) eg-new=$(comm -13 $t/fo $t/fn | head -1 | awk '{print $3}' | cut -c1-60)"; fi; rm -rf $t; }
norm() { perl -pe 's/(\d+)([A-Z]+_\d+_600_610_700_750)_860_NS/($1-4).$2."_NS"/ge; s/,860//g; s/_860//g'; }
export -f one norm
for fb in $(ls "$1/x" | grep -v '\.err$'); do for a in 60 61 70 75; do echo "$1 $2 $fb $a"; done; done | xargs -P "${3:-8}" -L1 bash -c 'one "$@"' _
