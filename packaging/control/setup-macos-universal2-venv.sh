#!/bin/bash
# One-time (idempotent) setup on the macOS build Mac: a venv whose every native file is universal2 (x86_64 + arm64), so
# PyInstaller can build ONE bundle that runs on Intel and Apple silicon Macs. Needs the universal2 python.org Python 3.12
# (the .pkg python.org ships is universal2) and internet for pip. Nothing here is signed or notarized.
#   bash packaging/control/setup-macos-universal2-venv.sh [VENV]      (default ~/pxa/venv-u2)
# Pure-Python dependencies are arch-neutral. Native ones are fused: when PyPI has no universal2 wheel, the two single-arch
# wheels are downloaded and merged with delocate-merge (lipo per Mach-O file). Pillow is the only native dependency of the Control
# bundle (pystray/pyobjc are not bundled; the LGPL files ride as third_party.zip). The audit at the end must print OK.
set -euo pipefail
V="${1:-$HOME/pxa/venv-u2}"
PYBIN="${PYBIN:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12}"
PILLOW="${PILLOW:-12.3.0}"; PYI="${PYI:-6.22.3}"; DELOCATE="${DELOCATE:-0.13.0}"
lipo -archs "$PYBIN" | grep -q 'x86_64' && lipo -archs "$PYBIN" | grep -q 'arm64' || { echo "FAIL: $PYBIN is not universal2"; exit 1; }
[ -x "$V/bin/python" ] || "$PYBIN" -m venv "$V"
"$V/bin/pip" install -q --upgrade pip >/dev/null 2>&1 || true
"$V/bin/pip" install -q "pyinstaller==$PYI" "delocate==$DELOCATE"
WH="$V/wheels"; mkdir -p "$WH/in" "$WH/out"
if ! ls "$WH/out"/pillow-"$PILLOW"-*universal2*.whl >/dev/null 2>&1; then
  "$V/bin/pip" download -q --no-deps --only-binary=:all: --implementation cp --python-version 3.12 --dest "$WH/in" \
      --platform macosx_10_13_x86_64 "pillow==$PILLOW"
  "$V/bin/pip" download -q --no-deps --only-binary=:all: --implementation cp --python-version 3.12 --dest "$WH/in" \
      --platform macosx_11_0_arm64 "pillow==$PILLOW"
  "$V/bin/delocate-merge" -w "$WH/out" "$WH/in"/pillow-"$PILLOW"-cp312-cp312-macosx_10_13_x86_64.whl "$WH/in"/pillow-"$PILLOW"-cp312-cp312-macosx_11_0_arm64.whl
fi
"$V/bin/pip" install -q --force-reinstall --no-deps "$WH/out"/pillow-"$PILLOW"-*.whl
# audit: every Mach-O file that can end up in the bundle (the Pillow package, the PyInstaller bootloader, the venv interpreter)
# carries both architectures; build-only tools (delocate and its test data) never enter the bundle
bad=0; n=0
while IFS= read -r f; do
  file "$f" | grep -q 'Mach-O' || continue
  n=$((n + 1)); a="$(lipo -archs "$f" 2>/dev/null || true)"
  case "$a" in *x86_64*arm64*|*arm64*x86_64*) ;; *) echo "NOT universal2: $f ($a)"; bad=1;; esac
done < <(find "$V/lib/python3.12/site-packages/PIL" "$V/lib/python3.12/site-packages/PyInstaller/bootloader" -type f 2>/dev/null; ls "$V/bin/python3.12")
[ "$bad" = 0 ] && echo "OK: $n Mach-O files in $V, all universal2" || { echo "FAIL: single-arch native files above"; exit 1; }
