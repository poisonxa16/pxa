#!/usr/bin/env bash
# release-image-archive.sh IMAGE OUT.tar.gz -- save a release docker image as an archive and run the clean-package
# gate on it (owner rule 2026-10-17: nothing is uploaded or handed out until the scan passes on the exact files).
#
#   PXA_RELEASE_BLOCKLIST=<block list> scripts/release-image-archive.sh pxa-cuda:4.0 /out/pxa-4.0-docker.tar.gz
#
# The archive is written as OUT.partial, scanned with scripts/release-scan.py (the same gate make-release-tarball.sh
# runs on the stage and the tarball), and only a PASS renames it to OUT. A FAIL leaves OUT.BLOCKED (kept for the
# fix-up, never shipped) and exits 2; the gate cannot run (no block list, no python) = exit 3. This script uploads
# nothing; the upload is a separate, owner-approved step that takes OUT only.
set -euo pipefail
IMAGE=${1:?usage: release-image-archive.sh IMAGE OUT.tar.gz}
OUT=${2:?usage: release-image-archive.sh IMAGE OUT.tar.gz}
HERE=$(dirname "$(readlink -f "$0")")
SCAN=$HERE/release-scan.py
die() { echo "release-image-archive: $*" >&2; exit "${RC:-1}"; }
[ -f "$SCAN" ] || RC=3 die "the clean-package gate is missing ($SCAN)"
[ -n "${PXA_RELEASE_BLOCKLIST:-}" ] && [ -r "$PXA_RELEASE_BLOCKLIST" ] || RC=3 die "PXA_RELEASE_BLOCKLIST unset or unreadable: refusing"
command -v docker >/dev/null || RC=3 die "docker not found"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "no such image: $IMAGE"
[ -e "$OUT" ] && die "$OUT exists (nothing is overwritten)"
WORK=${WORK:-$(dirname "$OUT")/.release-scan-$$}
mkdir -p "$WORK"
echo "saving $IMAGE -> $OUT.partial"
docker save "$IMAGE" | gzip -6 > "$OUT.partial"
echo "clean-package gate: $OUT.partial"
set +e
python3 "$SCAN" --work "$WORK" --json "$OUT.scan.json" "$OUT.partial"
rc=$?
set -e
case $rc in
  0) mv "$OUT.partial" "$OUT"; echo "PASS: $OUT ($(sha256sum "$OUT" | cut -c1-16))";;
  2) mv "$OUT.partial" "$OUT.BLOCKED"; RC=2 die "FAILED: the image holds blocked content ($OUT.scan.json); $OUT.BLOCKED must not be uploaded";;
  *) mv "$OUT.partial" "$OUT.BLOCKED"; RC=3 die "the gate could not run (rc=$rc); $OUT.BLOCKED must not be uploaded";;
esac
