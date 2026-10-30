#!/bin/bash
# Build the PXA Control macOS client as ONE universal2 (x86_64 + arm64) PyInstaller onefile. Control only: no engine on a Mac.
# Run on the macOS build Mac (Intel is fine: the bundle is assembled from universal2 parts, nothing is compiled):
#   PY=~/pxa/venv-u2/bin/python TAG=v4.0.0 bash packaging/control/build-macos-universal2.sh [OUT]   (TAG: the release tag)
# Prereq: packaging/control/setup-macos-universal2-venv.sh printed OK. Output: OUT/dist/pxa-control (onefile, universal2),
# OUT/onedir/ (audit copy: every Mach-O in it is checked with lipo), OUT/build-pyinstaller.json.
# PyInstaller ad-hoc signs each Mach-O (an arm64 binary must carry at least an ad-hoc signature to run); there is no Developer ID.
# PyInstaller keeps bytecode, not compiled machine code, so unlike the Nuitka Intel build it does not hide the Control logic
# (BUILD-CONTROL.md). Intel-only Nuitka path: build-macos.sh.
set -euo pipefail
PY="${PY:-$HOME/pxa/venv-u2/bin/python}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/out/macos-u2}"
W="$OUT/stage"
mkdir -p "$OUT"
[ -n "${TAG:-}" ] || echo "!!! WARNING: no TAG: VERSION stays the repo's own, so plugins with a newer min_host (every bundled one needs 4.0) will not install. Set TAG (e.g. TAG=v4.0.0) for a release build. !!!" >&2
"$PY" "$ROOT/packaging/control/stage.py" "$W" ${TAG:+--tag "$TAG"}
VER="$(head -1 "$W/VERSION" | sed -E 's/^[Tt][Aa][Gg]:[[:space:]]*//' | awk '{print $1}' | tr -cd 'A-Za-z0-9._-')"
PYDATA=(--add-data "$W/pxa_control_ui:pxa_control_ui" --add-data "$W/pxa_chat/ui:pxa_chat/ui"
        --add-data "$W/pxa_thinking_profiles.json:." --add-data "$W/pxa_encode_cells.json:." --add-data "$W/pxa-bench.py:."
        --add-data "$W/VERSION:." --add-data "$W/third_party.zip:." --add-data "$W/plugin-sdk.zip:.")
[ -f "$W/pxa_control/store/pubkeys.json" ] && PYDATA+=(--add-data "$W/pxa_control/store/pubkeys.json:pxa_control/store")
COMMON=(--noconfirm --clean --name pxa-control --paths "$W" --hidden-import pxa_launch --collect-submodules pxa_control
        --collect-submodules pxa_chat --collect-submodules pxa_ctl --exclude-module pystray --exclude-module Xlib
        --exclude-module tkinter --target-arch universal2 "${PYDATA[@]}")
cd "$W"
t0=$(date +%s)
# 1. onedir: the audit copy (what the onefile unpacks to), checked file by file
"$PY" -m PyInstaller "${COMMON[@]}" --onedir --distpath "$OUT/onedir" --workpath "$OUT/pyi-work-dir" --specpath "$OUT/pyi-work-dir" pxa_control_entry.py
bad=0; n=0
while IFS= read -r f; do
  file "$f" | grep -q 'Mach-O' || continue
  n=$((n + 1)); a="$(lipo -archs "$f" 2>/dev/null || true)"
  case "$a" in *x86_64*arm64*|*arm64*x86_64*) ;; *) echo "NOT universal2: $f ($a)"; bad=1;; esac
  sig="$(codesign -dv "$f" 2>&1 || true)"; case "$sig" in *Signature=adhoc*) ;; *) echo "NOT signed (ad-hoc): $f"; bad=1;; esac   # (-v on a framework's inner binary complains about bundle resources; a pipe to grep -q would SIGPIPE under pipefail)
done < <(find "$OUT/onedir" -type f)
[ "$bad" = 0 ] || { echo "AUDIT FAILED"; exit 1; }
echo "AUDIT OK: $n Mach-O files, all universal2 and signed"
# 2. onefile: what ships
"$PY" -m PyInstaller "${COMMON[@]}" --onefile --distpath "$OUT/dist" --workpath "$OUT/pyi-work-file" --specpath "$OUT/pyi-work-file" pxa_control_entry.py
t1=$(date +%s)
BIN="$OUT/dist/pxa-control"
lipo -archs "$BIN" | tee "$OUT/archs.txt"
codesign -v "$BIN"
printf '{"tool": "pyinstaller", "os": "macos", "arch": "universal2", "macho_audited": %d, "build_s": %d, "bytes": %d, "version": "%s"}\n' \
    "$n" $((t1 - t0)) "$(stat -f %z "$BIN")" "$VER" | tee "$OUT/build-pyinstaller.json"
