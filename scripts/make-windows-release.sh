#!/bin/bash
# scripts/make-windows-release.sh — assemble the Windows release. Does not publish.
#
#   scripts/make-windows-release.sh --dry-run
#   scripts/make-windows-release.sh --out <output-directory>
#   scripts/make-windows-release.sh --selftest
#
# Outputs (under --out):
#   pxa-control-win64.exe
#   pxa-engine-win64-cuda12.zip     engine bins + CUDA 12 runtime + MSVC runtime
#   ggml-pxqn.dll                   the licensed PXQN library, beside the zip
#   LICENSE-ggml-pxqn.txt           its terms (also inside the zip, next to LICENSE and NOTICE)
#   RELEASE-NOTES-windows.md        includes the SmartScreen note
#   SHA256SUMS
#
# The PXQN library goes inside the zip only with --bundle-pxqn. Without that flag the
# zip must not contain ggml-pxqn.dll (the same split as the staged Windows package).
# A DLL that still exports the old all-symbols table is refused. *.OLD* copies are ignored.
#
# Inputs are the staged host copies of the Windows VM build (C:\pxa\src on the VM).
# Default stage: the directory PXA_WIN_STAGE names (the latest ggml-pxqn.dll
# whose name is exactly that, newest mtime if several trees are searched).
#   --staged DIR          directory that holds the exe, the zip, and the dll
#   --engine-zip FILE     engine zip to rebuild (default: the one next to the dll)
#   --engine-bin DIR      loose bin\ from the VM instead of a zip
#   --control FILE        pxa-control-win64.exe
#   --pxqn FILE           ggml-pxqn.dll (default: the newest exact-name relink)
#   --cuda-bin DIR        extra CUDA runtime DLLs, if the engine bin does not have them
#   --redist DIR          extra MSVC runtime DLLs
#   --notes FILE          release-notes section (default: docs/windows-release-notes.md)
#   --no-pxqn             PXQ-only package, no library asset
#   --bundle-pxqn         also place the library in the zip (licence allows it for this package)
#   --closed-inc FILE     closed-name list for the guard (default: generate it)
#   --no-verify-hashes    do not check sibling .sha256 files during --dry-run
#
# The clean-package gate is the same one scripts/make-release-tarball.sh runs:
# closed lever names (scripts/pxa-lever-catalog.py --assert-clean, the library itself
# excluded), then scripts/release-scan.py on the stage and again on the zip
# (text, and the strings of every ELF and PE file). PXA_RELEASE_BLOCKLIST must be
# set. Unset or unreadable fails closed. Nothing here uploads or publishes.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
GUARDS="$ROOT/scripts/windows-release-guards.py"
SCAN="$ROOT/scripts/release-scan.py"
CATALOG="$ROOT/scripts/pxa-lever-catalog.py"

die() { echo "make-windows-release: $*" >&2; exit 1; }
info() { echo "make-windows-release: $*"; }

usage() {
  sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'
}

DRY=0
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --selftest)
      exec python3 "$ROOT/tests/test-make-windows-release.py" ;;
    -h|--help) usage; exit 0 ;;
    *) PASS+=("$1"); shift ;;
  esac
done

[[ -f "$GUARDS" && -f "$SCAN" ]] || die "packager scripts are missing"
PLAN=$(mktemp)
trap 'rm -f "$PLAN"' EXIT

python3 -I "$GUARDS" plan --json "$PLAN" "${PASS[@]}" || die "could not build the input plan"

if [[ "$DRY" -eq 1 ]]; then
  python3 -I "$GUARDS" check --plan "$PLAN"
  exit $?
fi

python3 -I "$GUARDS" assemble --plan "$PLAN" || die "assemble failed"

show() { python3 -I "$GUARDS" show --plan "$PLAN" "$1"; }
WORK=$(show work)
STAGE=$(show stage)
ZIP=$(show zip_path)
OUT=$(show out)
CLOSED=$(show closed_inc)

case "$WORK" in
  /|/mnt|/media|/usr|/boot|/root) die "refusing work directory" ;;
esac
case "$(dirname -- "$WORK")" in /mnt|/media) die "refusing work directory" ;; esac   # a whole disk or share root

if [[ -n "$CLOSED" ]]; then
  INC="$CLOSED"
elif [[ -f "$ROOT/scripts/pxa-closed-levers.txt" ]]; then
  info "closed-name list"
  python3 -I "$CATALOG" --release-filter "$WORK/pxa-lever-catalog.inc" \
    --closed-out "$WORK/pxqn-closed-levers.inc" \
    || die "lever catalog release filter failed (scripts/pxa-lever-catalog.py)"
  INC="$WORK/pxqn-closed-levers.inc"
else
  INC=""
fi

if [[ -n "$INC" ]]; then
  mapfile -t CLEAN < <(show clean_files)
  [[ ${#CLEAN[@]} -gt 0 ]] || die "no files to check for closed names"
  python3 -I "$CATALOG" --assert-clean "$INC" "${CLEAN[@]}" \
    || die "closed lever names in the Windows package (above): not packed"
fi

info "clean-package gate: the stage"
python3 -I "$SCAN" --jobs 4 --work "$WORK/release-scan" "$STAGE" \
  || die "clean-package gate FAILED on the stage (above): no zip"

python3 -I "$GUARDS" zip-engine --plan "$PLAN" || die "could not zip the engine"
info "clean-package gate: the zip"
if ! python3 -I "$SCAN" --jobs 4 --work "$WORK/release-scan" "$ZIP"; then
  mv -f "$ZIP" "$ZIP.rejected"
  die "clean-package gate FAILED on the zip (above): renamed to .rejected, not for upload"
fi

mkdir -p "$OUT"
cp -a "$STAGE/pxa-control-win64.exe" "$OUT/pxa-control-win64.exe"
cp -a "$ZIP" "$OUT/pxa-engine-win64-cuda12.zip"
if [[ -f "$STAGE/ggml-pxqn.dll" ]]; then
  cp -a "$STAGE/ggml-pxqn.dll" "$OUT/ggml-pxqn.dll"
  cp -a "$STAGE/LICENSE-ggml-pxqn.txt" "$OUT/LICENSE-ggml-pxqn.txt"
fi
cp -a "$STAGE/RELEASE-NOTES-windows.md" "$OUT/RELEASE-NOTES-windows.md"
(
  cd "$OUT"
  files=(pxa-control-win64.exe pxa-engine-win64-cuda12.zip)
  [[ -f ggml-pxqn.dll ]] && files+=(ggml-pxqn.dll LICENSE-ggml-pxqn.txt)
  files+=(RELEASE-NOTES-windows.md)
  sha256sum "${files[@]}" > SHA256SUMS
  sha256sum -c SHA256SUMS
) || die "SHA256SUMS did not verify"

info "DONE out=$OUT"
if [[ "$WORK" != "$OUT" && -d "$WORK" && -f "$WORK/.pxa-win-release-work" ]]; then
  rm -rf "$WORK"
fi
