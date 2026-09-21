#!/bin/bash
# scripts/gen-start-here.sh — render START-HERE.md from scripts/START-HERE.md.in.
#
# ONE renderer, two callers, so the tarball's copy and the repository's copy can never drift:
#
#   * scripts/make-release-tarball.sh calls it with the values it measured off the build it has
#     just packaged, and writes the result to the tarball root.
#   * a maintainer calls it with no arguments to refresh the START-HERE.md committed at the
#     repository root — the copy a reader sees on the web before downloading anything.
#
# START-HERE.md.in is the source. Never edit a rendered START-HERE.md; edit the .in and rerun
# this.
#
# Inputs (env, all optional):
#   TAG                 version tag (default: git describe of HEAD, else the release below)
#   CUDA_MAJOR_MINOR    e.g. 12.8            } these describe the BUILD, so make-release-tarball
#   GLIBC_FLOOR         e.g. GLIBC_2.34      } passes what it measured; the defaults describe
#   GLIBCXX_FLOOR       e.g. GLIBCXX_3.4.30  } the current published release and are copied from
#   BUILD_HOST_DRIVER   e.g. 580.142         } that tarball's own VERSION file, not guessed.
#   PATCHELF_OK         1 if the binaries got an RPATH to ./lib (default 1, as shipped)
#   PATCHELF_NOTE       override the sentence PATCHELF_OK selects
#   DRIVER_FLOOR        override the driver floor CUDA_MAJOR_MINOR selects
#   DISTRO_HINT         override the distro hint GLIBC_FLOOR selects
#   HEADER              1 (default) prepends the "generated file" comment; 0 omits it. The
#                       tarball copy omits it: there is no .in in a tarball to point at.
#   OUT                 output path (default: stdout)
set -eu

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IN=${IN:-$HERE/START-HERE.md.in}
[ -f "$IN" ] || { echo "gen-start-here: missing template: $IN" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
# defaults: the facts of the current published release (source: the VERSION file inside the
# v2026.09.07-rc1 tarball, which make-release-tarball.sh wrote from the build it packaged).
# ---------------------------------------------------------------------------------------------
TAG=${TAG:-$(git -C "$HERE" describe --tags --abbrev=0 2>/dev/null || echo v2026.09.07-rc1)}
CUDA_MAJOR_MINOR=${CUDA_MAJOR_MINOR:-12.8}
GLIBC_FLOOR=${GLIBC_FLOOR:-GLIBC_2.34}
GLIBCXX_FLOOR=${GLIBCXX_FLOOR:-GLIBCXX_3.4.30}
BUILD_HOST_DRIVER=${BUILD_HOST_DRIVER:-580.142}
PATCHELF_OK=${PATCHELF_OK:-1}
HEADER=${HEADER:-1}

# CUDA runtime -> documented minimum driver, from NVIDIA's CUDA toolkit release-notes table.
# Derived from CUDA_MAJOR_MINOR so a rerun on a different toolkit cannot ship a stale claim.
if [ -z "${DRIVER_FLOOR:-}" ]; then
  case "$CUDA_MAJOR_MINOR" in
    12.8*) DRIVER_FLOOR="570.00 (Linux)" ;;
    12.6*) DRIVER_FLOOR="560.28.03 (Linux)" ;;
    12.4*) DRIVER_FLOOR="550.54.14 (Linux)" ;;
    12.2*) DRIVER_FLOOR="535.86.09 (Linux)" ;;
    *)     DRIVER_FLOOR="see NVIDIA's CUDA $CUDA_MAJOR_MINOR toolkit release notes (minimum driver table)" ;;
  esac
fi

# glibc floor -> "which distro is new enough". Approximate on purpose (the glibc-to-release map
# is not exact across distro families); the authoritative check is always `ldd --version` on the
# target box against GLIBC_FLOOR itself. Ordered most-specific-first: a 2.34 floor must NOT fall
# through to the 2.3x arm, because Ubuntu 20.04 (glibc 2.31) is BELOW a 2.34 floor and naming it
# there is how a reader ends up with "version GLIBC_2.34 not found" after a 14 GB download.
if [ -z "${DISTRO_HINT:-}" ]; then
  case "$GLIBC_FLOOR" in
    GLIBC_2.4*|GLIBC_2.39*|GLIBC_2.38*)
      DISTRO_HINT="Ubuntu 24.04+, Debian 13+, Fedora 39+ (or any distro reporting glibc >= 2.38 via 'ldd --version')" ;;
    GLIBC_2.3[5-7]*)
      DISTRO_HINT="Ubuntu 22.04+, Debian 12+, Fedora 36+ (RHEL/Rocky/Alma 9 ship glibc 2.34 and are BELOW this floor; check with 'ldd --version')" ;;
    GLIBC_2.34*)
      DISTRO_HINT="Ubuntu 22.04+, Debian 12+, RHEL 9+ (or Rocky/Alma Linux 9+), Fedora 35+ (or any distro reporting glibc >= 2.34 via 'ldd --version')" ;;
    GLIBC_2.3[0-3]*)
      DISTRO_HINT="Ubuntu 20.04+, Debian 11+ (or any distro reporting glibc >= 2.31 via 'ldd --version')" ;;
    *)
      DISTRO_HINT="check with 'ldd --version' against this package's VERSION file (glibc_floor line)" ;;
  esac
fi

if [ -z "${PATCHELF_NOTE:-}" ]; then
  if [ "$PATCHELF_OK" = 1 ]; then
    PATCHELF_NOTE="Binaries carry an RPATH to ./lib; you do not need to set LD_LIBRARY_PATH by hand."
  else
    PATCHELF_NOTE="This build has no RPATH patch (patchelf was not available when it was packaged) -- always launch through ./pxa-launch or ./run-server.sh, which set LD_LIBRARY_PATH for you. Running bin/llama-server directly will fail to find libggml.so etc."
  fi
fi

render() {
  if [ "$HEADER" = 1 ]; then
    printf '%s\n' "<!-- generated from scripts/START-HERE.md.in by scripts/gen-start-here.sh; edit the .in -->"
  fi
  # "|" delimiter: the replacement text contains "/" (e.g. "./lib", "./pxa-launch"), which would
  # break a "/"-delimited sed substitution.
  sed -e "s|@@TAG@@|$TAG|g" \
      -e "s|@@CUDA_MAJOR_MINOR@@|$CUDA_MAJOR_MINOR|g" \
      -e "s|@@DRIVER_FLOOR@@|$DRIVER_FLOOR|g" \
      -e "s|@@GLIBC_FLOOR@@|${GLIBC_FLOOR#GLIBC_}|g" \
      -e "s|@@GLIBCXX_FLOOR@@|${GLIBCXX_FLOOR#GLIBCXX_}|g" \
      -e "s|@@DISTRO_HINT@@|$DISTRO_HINT|g" \
      -e "s|@@BUILD_HOST_DRIVER@@|$BUILD_HOST_DRIVER|g" \
      -e "s|@@PATCHELF_NOTE@@|$PATCHELF_NOTE|g" \
      "$IN"
}

if [ -n "${OUT:-}" ]; then
  render > "$OUT"
  # a leftover @@PLACEHOLDER@@ means a new one was added to the .in without being wired here
  if grep -q '@@[A-Z_]*@@' "$OUT"; then
    echo "gen-start-here: WARNING: unsubstituted placeholder(s) left in $OUT:" >&2
    grep -o '@@[A-Z_]*@@' "$OUT" | sort -u >&2
    exit 2
  fi
  echo "gen-start-here: wrote $OUT (tag $TAG, cuda $CUDA_MAJOR_MINOR, glibc floor ${GLIBC_FLOOR#GLIBC_})" >&2
else
  render
fi
