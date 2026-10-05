#!/usr/bin/env bash
# build_oracle_pxq4hq.sh -- extract pxa_deq_row_pxq6 (and the pair helper it calls) VERBATIM
# from the engine's ggml/src/pxq-cpu.c and compile them into the PXQ4HQ oracle binary. See
# oracle_pxq4hq.c for why the function is extracted rather than transcribed.
#
#   ENGINE=/path/to/engine-worktree ./build_oracle_pxq4hq.sh
# (needs a C compiler: run it inside a serving-image or dev-image container.)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ENGINE="${ENGINE:-$(cd "$HERE/../../../.." && pwd)}"
SRC="$ENGINE/ggml/src/pxq-cpu.c"
INC="$ENGINE/ggml/include"
OUT="${OUT:-$HERE/oracle_pxq4hq}"
[ -f "$SRC" ] || { echo "engine source not found: $SRC" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# awk: print each function from its signature line to the first line that is exactly "}".
# The order matters -- pxa_deq_pairs16 is called by pxa_deq_row_pxq6 and must be defined first.
for sig in "static inline void pxa_deq_pairs16(" "static void pxa_deq_row_pxq6("; do
  awk -v sig="$sig" '
    index($0, sig) == 1 { inside = 1 }
    inside { print }
    inside && /^\}$/ { inside = 0 }
  ' "$SRC" >> "$TMP/decode.c"
  echo >> "$TMP/decode.c"
done
for fn in pxa_deq_pairs16 pxa_deq_row_pxq6; do
  grep -q "$fn(" "$TMP/decode.c" || {
    echo "EXTRACTION FAILED: $fn not found in $SRC -- the engine's decode moved or was renamed." >&2
    exit 3; }
done
# The extracted body is C99 with <stdbool.h> semantics for its `bool hq` parameter.
sed -i '1i #include <stdbool.h>' "$TMP/decode.c"

python3 - "$HERE/oracle_pxq4hq.c" "$TMP/decode.c" "$TMP/oracle.c" <<'PY'
import sys
tpl, dec, out = sys.argv[1:4]
s = open(tpl).read()
marker = "/* @@PXA_ENGINE_DECODE_FUNCTIONS@@ */"
assert marker in s, "marker missing from oracle_pxq4hq.c"
open(out, "w").write(s.replace(marker, open(dec).read()))
PY

# -ffp-contract=off: the oracle must be a fixed sequence of separately rounded operations, not
# whatever the host compiler chose to fuse, or a "mismatch" means nothing.
"${CC:-cc}" -O2 -std=c11 -ffp-contract=off -I"$INC" "$TMP/oracle.c" -o "$OUT" -lm
echo "built $OUT"
