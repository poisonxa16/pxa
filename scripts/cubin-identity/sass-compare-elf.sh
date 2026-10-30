#!/bin/bash
# sass-compare-elf.sh <old_x_dir> <new_x_dir> [jobs]: ELF (.so) variant of sass-compare.sh. cuobjdump -xelf numbers cubins
# globally (<lib>.<idx>.sm_XX.cubin), so adding an arch shifts the indices: pair the k-th cubin of each arch by index order.
norm() { perl -pe 's/(\d+)([A-Z]+_\d+_600_610_700_750)_860_NS/($1-4).$2."_NS"/ge; s/,860//g; s/_860//g'; }
one() { o=$1; n=$2; tag=$3
  cmp -s "$o" "$n" && { echo "BYTES-SAME $tag"; return; }
  t=$(mktemp -d); cuobjdump -sass "$o" > $t/o; cuobjdump -sass "$n" > $t/n
  if cmp -s $t/o $t/n; then echo "SASS-SAME(RAW-SASS-SAME) $tag"
  elif cmp -s <(norm < $t/o) <(norm < $t/n); then echo "SASS-SAME(RAW-SASS-DIFF) $tag"
  else norm < $t/o | grep Function | sort > $t/fo; norm < $t/n | grep Function | sort > $t/fn
    echo "SASS-DIFF $tag lines=$(diff $t/o $t/n | grep -c '^[<>]') funcs=$(wc -l < $t/fo)/$(wc -l < $t/fn) only-old=$(comm -23 $t/fo $t/fn | wc -l) only-new=$(comm -13 $t/fo $t/fn | wc -l)"; fi; rm -rf $t; }
export -f one norm
for a in 60 61 70 75; do
  paste -d' ' <(ls "$1" | grep "\.sm_$a\.cubin$" | sort -t. -k2,2n | sed "s|^|$1/|") <(ls "$2" | grep "\.sm_$a\.cubin$" | sort -t. -k2,2n | sed "s|^|$2/|") | awk -v a=$a '{printf "%s %s k%03d.sm_%s\n", $1, $2, NR, a}'
done | xargs -P "${3:-8}" -L1 bash -c 'one "$@"' _
