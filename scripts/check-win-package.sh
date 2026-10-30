#!/bin/bash
# scripts/check-win-package.sh — acceptance checks for the Windows release package. Read-only on the inputs.
#
#   scripts/check-win-package.sh <pxa-engine-win64-cuda12.zip | release-dir> [--ctest-log FILE] [--cuobjdump]
#        [--bundle-pxqn | --no-pxqn] [--expect-exports N] [--archs 60,61,70,75,80,86,89] [--ptx 89] [--json OUT] [--keep]
#
# Finds pxa-control-win64.exe, ggml-pxqn.dll, RELEASE-NOTES-windows.md and SHA256SUMS next to the zip.
# Needs the private block list (PXA_RELEASE_BLOCKLIST or --blocklist); without it the scan gate FAILS closed.
# Exit 0 = no FAIL; 1 = FAIL; 3 = could not run. Work files go under $CHECK_WIN_WORK (default $TMPDIR) and are removed at the end unless --keep. Runs niced on cores ${CHECK_WIN_CPUS:-36-53}.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Set CHECK_WIN_WORK to a disk with ~2 GB free when /tmp is small or in RAM (the zip is unpacked there).
WORK=${CHECK_WIN_WORK:-${TMPDIR:-/tmp}}
mkdir -p "$WORK"
CPUS=${CHECK_WIN_CPUS:-36-53}
export CHECK_WIN_CPUS="$CPUS"
RUN=(nice -n 15)
command -v taskset >/dev/null 2>&1 && RUN+=(taskset -c "$CPUS")
exec "${RUN[@]}" python3 -I "$ROOT/scripts/check-win-package.py" --work "$WORK" "$@"
