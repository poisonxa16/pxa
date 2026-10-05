#!/usr/bin/env bash
# EVERY TIER A PXQ FILE CONTAINS MUST CARRY ITS CODEBOOK.
#
# A PXQ tier is a codebook format: nothing can decode a pxq3 tensor without pxa.pxq3.book and
# pxa.pxq3.sub. Until 2026-09-09 llama-quantize chose those KVs from the REQUESTED FTYPE — the
# `if (pxq2_out)` branch wrote the pxq2 book — which was already wrong before the tier profiles
# existed: a PXQ2 target on a MoE model emits a pxq4 backbone (BACKBONE_REV 2) and stamped only
# pxa.pxq2.*, so the file was undecodable by any reader that took the KVs at their word. The
# vLLM converter is exactly such a reader, and it is the one that found this.
#
# The test quantizes the tiny qwen4exp fixture at PXQ2 (uniform AND balanced) and asserts that
# every PXQ type present among the output's tensors has its book and sub KVs. It is a MIXED-tier
# output by construction — routed experts at the level, backbone above it — which is the case the
# old code got wrong.
#
# CPU-only, seconds, no GPU, no real weights.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
QUANT="${QUANT:-$ROOT/build-cpu/bin/llama-quantize}"
PY="${PY:-python3}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

[ -x "$QUANT" ] || { echo "SKIP: no llama-quantize at $QUANT (set QUANT=)"; exit 0; }

# FIXTURE= lets a caller supply a pre-made f32 GGUF. That is not a convenience: the generator
# needs numpy and the CUDA dev image this tree builds in does not have it, so in that image the
# fixture is made on the host and the quantize runs in the container.
FIX="${FIXTURE:-}"
if [ -z "$FIX" ]; then
    echo "== generating the tiny qwen4exp fixture"
    FIX="$TMP/tiny-f32.gguf"
    PYTHONPATH="$ROOT/gguf-py" "$PY" "$HERE/gen_tiny_qwen4exp.py" "$FIX" >/dev/null 2>&1 \
      || { echo "SKIP: fixture generator failed (needs numpy + gguf-py)"; exit 0; }
fi
[ -s "$FIX" ] || { echo "SKIP: no fixture at $FIX"; exit 0; }

check_one() {   # check_one <level> <policy>
    local lvl="$1" pol="$2"
    local out="$TMP/tiny-$lvl-$pol.gguf"
    echo "== $lvl / $pol"
    # --pxq-composition-override is REQUIRED here and is not a workaround: the fixture is a
    # 12-block toy whose vocab tables are most of its bytes, so the PXQ family lands at ~15% and
    # the composition assertion (correctly) refuses it. That assertion is about the file being
    # honestly NAMED; this test is about the file being DECODABLE, and the two are independent.
    if ! "$QUANT" --pxq-composition-override --pxq-policy "$pol" "$FIX" "$out" "$lvl" 4 > "$TMP/q.log" 2>&1; then
        echo "  FAIL quantize returned non-zero"; tail -5 "$TMP/q.log"; fail=1; return
    fi
    # Stdlib-only GGUF header reader: gguf-py needs numpy, and the CUDA dev image this tree
    # builds in does not have it. Reading key NAMES and tensor TYPES needs neither.
    "$PY" - "$out" <<'PYEOF'
import struct, sys

SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
GGML  = {248:'PXQ1', 252:'PXQ4', 253:'PXQ4HQ', 254:'PXQ2', 255:'PXQ3', 256:'PXQ6'}

f = open(sys.argv[1], 'rb')
assert f.read(4) == b'GGUF', 'not a GGUF'
ver, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))

def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

def rd_str():
    (n,) = rd('<Q')
    return f.read(n).decode('utf-8', 'replace')

def skip_value(t):
    if t == 8:                       # string
        rd_str()
    elif t == 9:                     # array
        (et,) = rd('<I'); (n,) = rd('<Q')
        if et == 8:
            for _ in range(n): rd_str()
        elif et == 9:
            for _ in range(n): skip_value(9)
        else:
            f.seek(SIZES[et] * n, 1)
    else:
        f.seek(SIZES[t], 1)

keys = set()
for _ in range(n_kv):
    keys.add(rd_str())
    (t,) = rd('<I')
    skip_value(t)

present = set()
for _ in range(n_tensors):
    rd_str()
    (nd,) = rd('<I')
    f.seek(8 * nd, 1)
    (ty,) = rd('<I')
    f.seek(8, 1)                     # tensor data offset
    if ty in GGML:
        present.add(GGML[ty])

# which codebook a tier's tensors are decoded with. pxq4/pxq4hq/pxq6 all read the pxq6-family
# tables; pxq1's book is the fixed {-1,+1} pair and needs no KV of its own.
NEED = {'PXQ2': 'pxa.pxq2', 'PXQ3': 'pxa.pxq3',
        'PXQ4': 'pxa.pxq6', 'PXQ4HQ': 'pxa.pxq6', 'PXQ6': 'pxa.pxq6'}
present = sorted(present & set(NEED))
print('  tiers in file: ' + (', '.join(present) or '(none)'))
if not present:
    print('  FAIL no PXQ tensors were emitted at all -- the fixture or the recipe is wrong')
    sys.exit(1)
missing = [ty + ' needs ' + NEED[ty] + '.' + sfx
           for ty in present for sfx in ('book', 'sub') if NEED[ty] + '.' + sfx not in keys]
for m in missing:
    print('  FAIL ' + m)
if 'pxa.policy.summary' not in keys:
    print('  FAIL no pxa.policy.summary KV -- the file cannot say which rule made it')
    missing.append('policy')
sys.exit(1 if missing else 0)
PYEOF
    [ $? -eq 0 ] || fail=1
}

# uniform is the case that was ALREADY broken before any profile existed: PXQ2 on a MoE model
# emits pxq2 experts + a pxq4 backbone, and used to stamp only the pxq2 book.
check_one PXQ2 uniform
# balanced is the case the profiles add: same shape, and it must not regress it.
check_one PXQ2 balanced
check_one PXQ3 balanced

[ "$fail" -eq 0 ] && echo "OK — every emitted tier carries its codebook" || echo "FAILED"
exit "$fail"
