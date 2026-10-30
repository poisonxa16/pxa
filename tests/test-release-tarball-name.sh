#!/usr/bin/env bash
# The release tarball's asset name (scripts/make-release-tarball.sh): with ARCH_TAG unset the "-sm..." part is derived from
# CUDA_ARCHS (the list the manifest writes as cuda_archs_built), and the legacy "-sm60_61_70" alias with the same bytes is
# written only when that list covers 60;61;70. An explicit ARCH_TAG is used as given. Hermetic: no git, no docker, no build.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MK="$HERE/../scripts/make-release-tarball.sh"
fail=0; n=0
ok()  { n=$((n+1)); echo "ok   $1"; }
bad() { n=$((n+1)); fail=$((fail+1)); echo "FAIL $1"; }
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1: got '$2' want '$3'"; fi; }
tagline() { env -u ARCH_TAG "$@" bash "$MK" --list-targets 2>&1 | sed -n 's/^asset arch tag: *//p'; }

out=$(bash "$MK" --selftest 2>&1); rc=$?
check "selftest (membership + name derivation + alias writer)" "$rc:$(echo "$out" | tail -1)" "0:make-release-tarball.sh: selftest ok"

# the default release list: the 7-arch name + the legacy alias
check "default CUDA_ARCHS" "$(tagline)" "sm60_61_70_75_80_86_89 + alias -sm60_61_70 (same bytes)"
# the name follows the list the build carries, never a hard-coded one
check "derived from CUDA_ARCHS" "$(tagline CUDA_ARCHS='60-real;61-real;70-real;75-real')" "sm60_61_70_75 + alias -sm60_61_70 (same bytes)"
check "PTX-only entry is not in the name" "$(tagline CUDA_ARCHS='60-real;61-real;70-real;75-real;89-virtual')" "sm60_61_70_75 + alias -sm60_61_70 (same bytes)"
check "legacy list: one file, no alias" "$(tagline CUDA_ARCHS='60-real;61-real;70-real')" "sm60_61_70"
check "no 60;61;70 floor: no alias" "$(tagline CUDA_ARCHS='75-real;86-real;89')" "sm75_86_89"
# an explicit ARCH_TAG keeps working, as given, no alias
got=$(ARCH_TAG=sm60_61_70-ubuntu22.04 bash "$MK" --list-targets 2>&1 | sed -n 's/^asset arch tag: *//p')
check "explicit ARCH_TAG" "$got" "sm60_61_70-ubuntu22.04 (explicit ARCH_TAG)"
got=$(ARCH_TAG=sm60_61_70 bash "$MK" --list-targets 2>&1 | sed -n 's/^asset arch tag: *//p')
check "explicit legacy ARCH_TAG" "$got" "sm60_61_70 (explicit ARCH_TAG)"
# a list pxa-update could not read refuses to name the asset (set ARCH_TAG)
env -u ARCH_TAG CUDA_ARCHS='100-real;60-real' bash "$MK" --list-targets >/dev/null 2>&1; rc=$?
check "three-digit sm refused without ARCH_TAG" "$rc" "1"

# the packaging tail: TARNAME carries ARCH_TAG, and the alias is written only after the clean-package gate passed on the
# tarball and its .sha256 was written (the alias is the gated bytes, cmp-checked inside write_legacy_alias)
grep -q '^TARNAME="pxa-${TAG}-linux-x86_64-cuda${CUDA_MAJOR_MINOR}-${ARCH_TAG}.tar.gz"$' "$MK" && ok "TARNAME uses ARCH_TAG" || bad "TARNAME uses ARCH_TAG"
gate=$(grep -n 'python3 "$RELEASE_SCAN" --work "$WORK/release-scan" "$TARPATH"' "$MK" | head -1 | cut -d: -f1)
sha=$(grep -n 'sha256sum "$TARNAME" > "$TARNAME.sha256"' "$MK" | head -1 | cut -d: -f1)
ali=$(grep -n 'write_legacy_alias "$OUT_DIR" "$TARNAME" "$ALIAS_NAME"' "$MK" | head -1 | cut -d: -f1)
if [ -n "$gate" ] && [ -n "$sha" ] && [ -n "$ali" ] && [ "$gate" -lt "$sha" ] && [ "$sha" -lt "$ali" ]; then
  ok "alias written after the gate and the .sha256 ($gate < $sha < $ali)"
else bad "alias order (gate=$gate sha=$sha alias=$ali)"; fi

echo "$((n-fail))/$n passed"
[ "$fail" = 0 ]
