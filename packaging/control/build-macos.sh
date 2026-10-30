#!/bin/bash
# Build the compiled PXA Control for macOS (Intel x86_64 here; the same recipe on Apple silicon gives arm64) -- Control
# only: there is no CUDA engine on macOS, so this binary drives a rig on the network (--lan) or a remote engine.
# Needs Python 3.12 from python.org (not the CLT's 3.9) + the Xcode Command Line Tools (clang); no Homebrew:
#   python3.12 -m venv ~/pxa/venv && ~/pxa/venv/bin/pip install nuitka==4.2.2 zstandard==0.25.0 ordered-set==4.1.0 \
#       pillow==12.3.0 pystray==0.19.5
#   PY=~/pxa/venv/bin/python TAG=v4.0.0 bash packaging/control/build-macos.sh [OUT]   (TAG: the release tag VERSION names)
# Output: OUT/nuitka/pxa-control (onefile) + build-nuitka.json; the stage is the same stage.py as Linux and Windows.
set -euo pipefail
PY="${PY:-python3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/out/macos}"
JOBS="${JOBS:-4}"
W="$OUT/stage"
mkdir -p "$OUT"
[ -n "${TAG:-}" ] || echo "!!! WARNING: no TAG: VERSION stays the repo's own, so plugins with a newer min_host (every bundled one needs 4.0) will not install. Set TAG (e.g. TAG=v4.0.0) for a release build. !!!" >&2
"$PY" "$ROOT/packaging/control/stage.py" "$W" ${TAG:+--tag "$TAG"}
# first token of the first line; a 'tag:' label is not part of the version (was 'tagv3.1': tr kept the label)
VER="$(head -1 "$W/VERSION" | sed -E 's/^[Tt][Aa][Gg]:[[:space:]]*//' | awk '{print $1}' | tr -cd 'A-Za-z0-9._-')"
DATA=(--include-data-dir="$W/pxa_control_ui=pxa_control_ui" --include-data-dir="$W/pxa_chat/ui=pxa_chat/ui"
      --include-data-files="$W/pxa_thinking_profiles.json=pxa_thinking_profiles.json"
      --include-data-files="$W/pxa_encode_cells.json=pxa_encode_cells.json"
      --include-data-files="$W/pxa-bench.py=pxa-bench.py" --include-data-files="$W/VERSION=VERSION"
      --include-data-files="$W/third_party.zip=third_party.zip"
      --include-data-files="$W/plugin-sdk.zip=plugin-sdk.zip")      # what plugins and speech run on (stage.py)
[ -f "$W/pxa_control/store/pubkeys.json" ] && DATA+=(--include-data-files="$W/pxa_control/store/pubkeys.json=pxa_control/store/pubkeys.json")
cd "$W"
t0=$(date +%s)
"$PY" -m nuitka --standalone --onefile --assume-yes-for-downloads --jobs="$JOBS" --lto=no \
    --output-dir="$OUT/nuitka" --output-filename=pxa-control \
    --include-package=pxa_control --include-package=pxa_chat --include-package=pxa_ctl --include-module=pxa_launch \
    --include-package=PIL --nofollow-import-to=pystray --nofollow-import-to=Xlib --nofollow-import-to=six \
    --nofollow-import-to='*.tests' \
    --onefile-tempdir-spec="{CACHE_DIR}/PXA/control-$VER" \
    "${DATA[@]}" pxa_control_entry.py
t1=$(date +%s)
BIN="$OUT/nuitka/pxa-control"
SIZE=$(stat -f %z "$BIN")
printf '{"tool": "nuitka", "os": "macos", "arch": "%s", "build_s": %d, "bytes": %d, "version": "%s"}\n' "$(uname -m)" \
    $((t1 - t0)) "$SIZE" "$VER" | tee "$OUT/nuitka/build-nuitka.json"
