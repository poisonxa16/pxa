#!/usr/bin/env bash
# Build libpxq4_encode.so — the native PXQ4 encoder behind converter policies p2a/p2c.
#
# WHY THIS EXISTS: gguf_to_vllm/encoder.py binds a frozen C ABI by ctypes, but nothing in
# this repo built it, so every P2 policy raised SystemExit at planning time. P1 does not
# need it (P1 only moves bytes that are already PXQ4); P2 does, because it re-encodes
# tensors that are NOT PXQ4 in the artifact.
#
# The implementation is a file-static function in the ENGINE tree, not here. We #include the
# .inc.cpp so the static symbol is visible in our translation unit rather than vendoring a
# copy that would silently drift from the engine.
#
# usage: build_encoder.sh [MGV_SRC] [OUT]
set -euo pipefail
MGV="${1:-/path/to/engine-repo/src}"
OUT="${2:-libpxq4_encode.so}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for f in pxq6-quantize.inc.cpp ggml-pxq6-tables.h; do
  [ -f "$MGV/$f" ] || { echo "missing $MGV/$f — pass the mgv-wt src dir as \$1" >&2; exit 2; }
done

# -mf16c: the shim supplies ggml's fp16 conversions with RNE semantics.
# NO -ffast-math: the codec is parity-locked to exact fp32 rounding; relaxing it changes bytes.
g++ -O2 -std=c++17 -fPIC -shared -mf16c -pthread \
    -I"$MGV" -o "$OUT" "$HERE/../src/pxq4_encode_shim.cpp"
echo "built $OUT"
echo "VALIDATE BEFORE USE:  python3 src/pxq4_validate.py --so $OUT --gguf <a real pxq4 .gguf>"
echo "An encoder that produces plausible-but-wrong bytes poisons every artifact downstream."
