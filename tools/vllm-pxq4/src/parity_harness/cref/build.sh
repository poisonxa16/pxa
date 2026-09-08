#!/bin/sh
# Build the shipping-engine CPU dequant into a standalone CLI.
# No GPU, no CUDA, no ggml build system: pxq-cpu.c only needs ggml-impl.h and the three
# frozen table headers, all of which are vendored read-only under vendor/.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cc=${CC:-cc}
"$cc" -O2 -std=c11 -Wall \
    -I "$here/vendor/ggml/include" -I "$here/vendor/ggml/src" \
    -o "$here/pxq4_cref" \
    "$here/pxq4_cref.c" "$here/vendor/ggml/src/pxq-cpu.c" -lm
echo "built $here/pxq4_cref"
