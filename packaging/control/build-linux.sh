#!/bin/bash
# Build the compiled PXA Control for Linux x86_64 inside packaging/control/Dockerfile.build-linux (BUILD-CONTROL.md).
#   docker build -t pxa-control-build:u2204 -f packaging/control/Dockerfile.build-linux packaging/control
#   docker run --rm --cpuset-cpus=18-35 -e TAG=v4.0.0 -v "$PWD":/src:ro -v /path/out:/out pxa-control-build:u2204 \
#       bash /src/packaging/control/build-linux.sh nuitka|pyinstaller|cython|all [ENGINE_DIR]
# TAG: the release tag VERSION names (the store checks a plugin's min_host against it); unset = the repo's VERSION.
# ENGINE_DIR (optional, mounted read-only): an unpacked release folder (bin/ lib/ lib-compat/); with it the onefile
# carries the engine too ("one download"). Outputs land in /out/<tool>/ with a build-<tool>.json of the numbers.
set -euo pipefail
TOOL="${1:-nuitka}"
ENGINE="${2:-}"
JOBS="${JOBS:-18}"
SRC=/src
W=/work/stage
mkdir -p /out
[ -n "${TAG:-}" ] || echo "!!! WARNING: no TAG: VERSION stays the repo's own, so plugins with a newer min_host (every bundled one needs 4.0) will not install. Set TAG (e.g. TAG=v4.0.0) for a release build. !!!" >&2
python3 "$SRC/packaging/control/stage.py" "$W" ${TAG:+--tag "$TAG"}      # the stage (same script on Windows and macOS)
# first token of the first line; a 'tag:' label is not part of the version (was 'tagv3.1': tr kept the label)
VER="$(head -1 "$W/VERSION" | sed -E 's/^[Tt][Aa][Gg]:[[:space:]]*//' | awk '{print $1}' | tr -cd 'A-Za-z0-9._-')"
DATA_NUITKA=(--include-data-dir="$W/pxa_control_ui=pxa_control_ui" --include-data-dir="$W/pxa_chat/ui=pxa_chat/ui"
             --include-data-files="$W/pxa_thinking_profiles.json=pxa_thinking_profiles.json"
             --include-data-files="$W/pxa_encode_cells.json=pxa_encode_cells.json"
             --include-data-files="$W/pxa-bench.py=pxa-bench.py" --include-data-files="$W/VERSION=VERSION"
             --include-data-files="$W/third_party.zip=third_party.zip"
             --include-data-files="$W/plugin-sdk.zip=plugin-sdk.zip")      # what plugins and speech run on (stage.py)
# the release's public store/plugin keys (store/pubkeys.json, public halves only): a package data file Nuitka and
# PyInstaller would otherwise leave out, and without it Control cannot verify the live store's catalogue or grants
PYDATA_KEYS=()
if [ -f "$W/pxa_control/store/pubkeys.json" ]; then
  DATA_NUITKA+=(--include-data-files="$W/pxa_control/store/pubkeys.json=pxa_control/store/pubkeys.json")
  PYDATA_KEYS=(--add-data "$W/pxa_control/store/pubkeys.json:pxa_control/store")
fi
if [ -n "$ENGINE" ]; then
  for d in bin lib lib-compat; do
    [ -d "$ENGINE/$d" ] && DATA_NUITKA+=(--include-data-dir="$ENGINE/$d=engine/$d")
  done
fi
cd "$W"
t0=$(date +%s)
case "$TOOL" in
nuitka)
  OUT=/out/nuitka${ENGINE:+-engine}
  python3 -m nuitka --standalone --onefile --assume-yes-for-downloads --jobs="$JOBS" --lto=no \
      --output-dir="$OUT" --output-filename=pxa-control \
      --include-package=pxa_control --include-package=pxa_chat --include-package=pxa_ctl --include-module=pxa_launch \
      --include-package=PIL --nofollow-import-to=pystray --nofollow-import-to=Xlib --nofollow-import-to=six \
      --nofollow-import-to='*.tests' \
      --onefile-tempdir-spec="{CACHE_DIR}/PXA/control-$VER" \
      "${DATA_NUITKA[@]}" pxa_control_entry.py
  BIN="$OUT/pxa-control" ;;
pyinstaller)
  OUT=/out/pyinstaller
  PYDATA=(--add-data "$W/pxa_control_ui:pxa_control_ui" --add-data "$W/pxa_chat/ui:pxa_chat/ui"
          --add-data "$W/pxa_thinking_profiles.json:." --add-data "$W/pxa_encode_cells.json:." --add-data "$W/pxa-bench.py:."
          --add-data "$W/VERSION:." --add-data "$W/plugin-sdk.zip:." "${PYDATA_KEYS[@]}")
  python3 -m PyInstaller --onefile --noconfirm --clean --name pxa-control --distpath "$OUT" --workpath /work/pyi-build \
      --specpath /work/pyi-build --paths "$W" --hidden-import pxa_launch --collect-submodules pxa_control \
      --collect-submodules pxa_chat --collect-submodules pxa_ctl --exclude-module pystray --exclude-module Xlib \
      --add-data "$W/third_party.zip:." "${PYDATA[@]}" pxa_control_entry.py
  BIN="$OUT/pxa-control" ;;
cython)
  OUT=/out/cython
  mkdir -p "$OUT"
  # compile every module of the package except __init__ (the re-export + attribute forwarding stays Python)
  python3 -m cython --version
  cythonize -3 -i -j "$JOBS" pxa_control/core.py pxa_control/auth.py pxa_control/bridge.py pxa_control/routes.py \
      pxa_control/server.py pxa_control/desktop.py > "$OUT/cythonize.log" 2>&1
  rm -f pxa_control/core.py pxa_control/auth.py pxa_control/bridge.py pxa_control/routes.py pxa_control/server.py \
      pxa_control/desktop.py pxa_control/*.c
  tar -C "$W" -czf "$OUT/pxa_control-cython.tar.gz" pxa_control
  BIN="" ;;
*) echo "usage: $0 nuitka|pyinstaller|cython [ENGINE_DIR]"; exit 2 ;;
esac
t1=$(date +%s)
SIZE=0
[ -n "$BIN" ] && [ -f "$BIN" ] && SIZE=$(stat -c %s "$BIN")
[ "$TOOL" = cython ] && SIZE=$(du -sb "$W/pxa_control" | cut -f1)
printf '{"tool": "%s", "engine": %s, "build_s": %d, "bytes": %d, "version": "%s"}\n' "$TOOL" "$([ -n "$ENGINE" ] && echo true || echo false)" \
    $((t1 - t0)) "$SIZE" "$VER" | tee "$OUT/build-$TOOL.json"
