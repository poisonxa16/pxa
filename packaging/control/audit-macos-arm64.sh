#!/bin/bash
# Static check of the arm64 slice of a universal2 PyInstaller onedir (run where it was built; no Apple silicon needed):
# every Mach-O file has an arm64 slice, the arm64 slice is ad-hoc signed (an arm64 binary will not run unsigned), its minimum macOS is
# <= 14, and every library it links (otool -arch arm64 -L) is a system library or a file that exists inside the bundle.
# NOT a substitute for running it on an Apple silicon Mac (mac-arm64-proof workflow / a real Mac).
#   bash packaging/control/audit-macos-arm64.sh <onedir dir>
set -uo pipefail
D="${1:?onedir}"; bad=0; n=0; sysdeps=0
while IFS= read -r f; do
  file "$f" | grep -q 'Mach-O' || continue
  n=$((n + 1))
  lipo -archs "$f" | grep -q arm64 || { echo "no arm64 slice: $f"; bad=1; continue; }
  sig="$(codesign -dv --arch arm64 "$f" 2>&1 || true)"; case "$sig" in *Signature=adhoc*) ;; *) echo "arm64 slice not signed: $f"; bad=1;; esac
  mv_="$(otool -arch arm64 -l "$f" 2>/dev/null | awk '/LC_BUILD_VERSION/{b=1} b&&/minos/{print $2; exit}')"
  [ -n "$mv_" ] && { maj="${mv_%%.*}"; [ "$maj" -le 14 ] || { echo "arm64 min macOS $mv_ > 14: $f"; bad=1; }; }
  while IFS= read -r dep; do
    case "$dep" in /usr/lib/*|/System/Library/*) sysdeps=$((sysdeps + 1));;
      @rpath/*|@loader_path/*|@executable_path/*) b="$(basename "$dep")"; [ -n "$(find "$D" -name "$b" -print -quit)" ] || { echo "missing in bundle: $dep (needed by $f)"; bad=1; };;
      *) echo "unexpected link path: $dep (in $f)"; bad=1;; esac
  done < <(otool -arch arm64 -L "$f" 2>/dev/null | tail -n +2 | awk '{print $1}')
done < <(find "$D" -type f)
[ "$bad" = 0 ] && echo "ARM64 STATIC AUDIT OK: $n Mach-O files, arm64 slices present + ad-hoc signed, links resolve ($sysdeps system-library links)" || { echo "ARM64 STATIC AUDIT FAILED"; exit 1; }
