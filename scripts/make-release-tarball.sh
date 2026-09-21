#!/bin/bash
# scripts/make-release-tarball.sh — build a self-contained "for dummies" release tarball:
# untar, download a model, run ./pxa-launch. No build tools required on the target box.
#
# Inputs (env vars, all overridable; sane defaults for a same-repo run):
#   REF          git ref to package docs/scripts/tools/bench AND (for a self-build) source from
#                (default: current HEAD)
#   BUILD_DIR    a build directory. If it already has bin/llama-server, it is used as-is (no
#                build happens). If not, and BUILD_IMAGE is set, this script builds REF into it.
#                (default: ./build-spd)
#   BUILD_IMAGE  when BUILD_DIR is empty/missing bin/llama-server, the image to build REF in.
#                If this image does not exist locally, it is built from DOCKERFILE first. Unset
#                by default -- an empty BUILD_DIR with no BUILD_IMAGE is a hard error (this
#                script does not guess whether you wanted a build or forgot to point BUILD_DIR
#                somewhere). Set it to e.g. pxa-release-build:cu128-u2204 for a one-command
#                "build + package" run on an older-glibc base (see DOCKERFILE below).
#   DOCKERFILE   Dockerfile to build BUILD_IMAGE from if it does not exist locally (default:
#                docker/Dockerfile.release-build next to this script's repo root).
#   OUT_DIR      where the .tar.gz and .sha256 land (default: ./release-assets)
#   TAG          version tag baked into the filename and VERSION file (default: git describe of REF)
#   DEV_IMAGE    the container to run ldd/strip/objdump/patchelf inside for POST-PROCESSING the
#                already-built binaries (default: BUILD_IMAGE if this run just self-built with
#                one, else pxa-sm60-dev:latest). This should always be the SAME image the
#                binaries were actually linked in, or ldd/objdump report the wrong library paths
#                and the wrong glibc/libstdc++ floor.
#   CUDA_MAJOR_MINOR  override the detected CUDA runtime major.minor (e.g. 12.8)
#
# Packages BUILD_DIR (building it first via BUILD_IMAGE if it is not already built) plus
# docs/scripts/tools taken from REF via `git archive` (never the live working tree, so this is
# safe to run from a worktree that is itself the packaging target, and correct regardless of
# what branch BUILD_DIR happens to have been built from).
#
# What lands in the tarball's bin/ (the asset list; the lists themselves are the four *_BIN
# variables in the configuration block below, and `--list-targets` prints them):
#   llama-server llama-cli llama-bench llama-perplexity   required, always shipped
#   llama-pxq-export                                      required -- without it the "leaving
#                                                         PXQ" recipe in README.md and
#                                                         docs/COOKBOOK.md cannot be run from
#                                                         this package at all
#   llama-quantize llama-gguf-split                       shipped when they build
#   three self-test binaries                              required
#
# Dry run: `scripts/make-release-tarball.sh --list-targets` prints the build targets and the bin
# lists and exits, touching no git, no docker and no build directory.
#
# Idempotent: reruns overwrite the same-named output tar/sha in OUT_DIR, and skip the build step
# entirely once BUILD_DIR/bin/llama-server exists. Safe with NVIDIA_VISIBLE_DEVICES=none —
# nothing here, build included, touches a GPU.

set -u
set -o pipefail

# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HERE/.." && pwd)
DOCKERFILE=${DOCKERFILE:-$REPO_ROOT/docker/Dockerfile.release-build}

REF=${REF:-HEAD}
BUILD_DIR=${BUILD_DIR:-./build-spd}
BUILD_IMAGE=${BUILD_IMAGE:-}
OUT_DIR=${OUT_DIR:-./release-assets}
DEV_IMAGE=${DEV_IMAGE:-}
CUDA_MAJOR_MINOR=${CUDA_MAJOR_MINOR:-}
ARCH_TAG=${ARCH_TAG:-sm60_61_70}
CUDA_ARCHS=${CUDA_ARCHS:-60;61;70}
BUILD_TARGETS="llama-server llama-cli llama-bench llama-perplexity llama-quantize llama-imatrix llama-pxq-export llama-gguf-split test-pxq-cpu-dot test-kv-seq-shadow test-narrow-kernel-parity"

# The tarball's bin/ manifest. Kept here, next to BUILD_TARGETS, so the thing that gets BUILT and
# the thing that gets SHIPPED are read and changed together -- the v2026.09.07-rc1 tarball shipped
# without llama-pxq-export because these two lists lived 160 lines apart and only one of them was
# updated when the export tool landed.
REQUIRED_BIN="llama-server llama-cli llama-bench llama-perplexity"
# Required, but built on demand if an externally-supplied BUILD_DIR lacks it (a CPU-only link).
# A package without it cannot run the documented "leave PXQ" recipe, so a still-missing binary
# here is a hard error, not a warning.
# llama-imatrix joined this list for v2026.09.20: the package now carries the whole stock-quant
# path (convert_hf_to_gguf.py + gguf-py + llama-imatrix + llama-quantize + llama-perplexity), so a
# user can take a Hugging Face model all the way to a quantised GGUF without cloning the repo, and
# docs/tutorials/04 walks through exactly that sequence out of this package's own bin/.
REQUIRED_ONDEMAND_BIN="llama-pxq-export llama-imatrix"
TEST_BIN="test-pxq-cpu-dot test-kv-seq-shadow test-narrow-kernel-parity"
OPTIONAL_BIN="llama-quantize llama-gguf-split"

err()  { echo "make-release-tarball.sh: $*" >&2; }
die()  { err "$*"; exit 1; }
info() { echo "make-release-tarball.sh: $*"; }

# ---------------------------------------------------------------------------------------------
# arguments. This script is configured by environment variables (above); the only flags it takes
# are informational ones that run nothing.
# ---------------------------------------------------------------------------------------------
case "${1:-}" in
  --list-targets)
    echo "BUILD_TARGETS:          $BUILD_TARGETS"
    echo "bin/ required:          $REQUIRED_BIN"
    echo "bin/ required (on-demand build if BUILD_DIR lacks it): $REQUIRED_ONDEMAND_BIN"
    echo "bin/ self-tests:        $TEST_BIN"
    echo "bin/ optional:          $OPTIONAL_BIN"
    echo "CUDA archs:             $CUDA_ARCHS"
    exit 0
    ;;
  -h|--help)
    sed -n '2,/^set -u$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
    echo "flags: --list-targets   print the build targets and the bin/ manifest, then exit"
    exit 0
    ;;
  "") ;;
  *) die "unknown argument: $1 (this script is configured with the environment variables documented at the top of the file; the only flags are --list-targets and --help)" ;;
esac

# Run a bash script inside DEV_IMAGE, BYPASSING the image's nvidia_entrypoint.sh via
# --entrypoint. That entrypoint prints an NVIDIA/CUDA banner to STDOUT (not stderr) even with
# no GPU and no driver — confirmed by hand: it corrupts every `docker run ... bash -c "..."`
# invocation whose stdout is captured into a shell variable or redirected to a file (a real
# failure hit while writing this script: the banner text got embedded into a filename via
# command substitution, and separately into a copied .so file's bytes, both silently).
# Usage: dimg [docker-run args...] -- '<script for bash -c>'
dimg() {
  local args=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do args+=("$1"); shift; done
  shift || true  # drop the --
  docker run --rm --entrypoint bash "${args[@]}" "$DEV_IMAGE" -c "$1"
}

[ -d "$REPO_ROOT/.git" ] || [ -f "$REPO_ROOT/.git" ] || die "REPO_ROOT ($REPO_ROOT, derived from this script's location) is not a git checkout"
command -v docker >/dev/null || die "docker not found on PATH"
command -v git    >/dev/null || die "git not found on PATH"

# Resolve REF to a commit inside REPO_ROOT without touching the current worktree's checkout
# (this script may run from a worktree that is itself the packaging target).
COMMIT=$(git -C "$REPO_ROOT" rev-parse --verify "${REF}^{commit}" 2>/dev/null) || die "REF '$REF' does not resolve to a commit in $REPO_ROOT"
COMMIT_SHORT=${COMMIT:0:10}

if [ -z "${TAG:-}" ]; then
  TAG=$(git -C "$REPO_ROOT" describe --tags --exact-match "$COMMIT" 2>/dev/null) \
    || TAG=$(git -C "$REPO_ROOT" describe --tags --always "$COMMIT" 2>/dev/null) \
    || TAG="$COMMIT_SHORT"
fi
info "REF=$REF -> commit $COMMIT_SHORT, TAG=$TAG"

WORK=$(mktemp -d /tmp/pxq-release-XXXXXX)
# KEEP_WORK=1 keeps the staged tree. The trap deletes the stage on EVERY exit path, so a cut that
# dies in one of the guards below leaves nothing to look at -- and the guard's own one-line message
# is then the only evidence of what it refused. Inspecting a refused package has to be possible.
if [ -n "${KEEP_WORK:-}" ]; then
  trap 'echo "make-release-tarball.sh: KEEP_WORK set -- stage left at $WORK"' EXIT
else
  trap 'rm -rf "$WORK"' EXIT
fi

# A cmake --build against THIS BUILD_DIR still running would make the binaries we're about to
# copy a moving target. Best-effort guard: look at ACTUAL "cmake" processes (pgrep -x, exact
# binary name) rather than pgrep -f "cmake --build", which also matches any unrelated shell
# whose command-line text happens to quote/echo/monitor that phrase (a real false-positive seen
# in practice — a leftover `until ... grep -q "cmake --build"` polling loop matched itself).
#
# This box runs many worktrees that all name their build dir "build-spd" (or "build"), so a
# basename-only match on cmdline also false-positives across worktrees (seen in practice: a
# a different worktree's "cmake --build build-spd" matched ours by basename alone). Disambiguate by
# resolving the candidate process's own cwd through ITS OWN /proc/<pid>/mountinfo (namespace-
# aware — dockerized builds bind-mount BUILD_DIR's parent to something like /work, and the
# mountinfo "root" field names the real host-side subvolume path) and comparing that against
# BUILD_DIR_PARENT's basename.
BUILD_DIR_PARENT=$(dirname "$BUILD_DIR")
BUILD_DIR_BASE=$(basename "$BUILD_DIR")
BUILDING=0
for pid in $(pgrep -x cmake 2>/dev/null); do
  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
  case "$cmdline" in
    *--build*"$BUILD_DIR_BASE"*) ;;  # candidate: basename matches textually
    *) continue ;;
  esac
  # plain readlink, NOT -f: the target is a path inside the PROCESS's own mount namespace
  # (e.g. "/work/build-spd"), which usually does not exist/resolve in ours, so -f would just
  # come back empty.
  cwd_target=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
  mnt_point=$(dirname "$cwd_target")
  src=$(awk -v mp="$mnt_point" '$5==mp {print $4; exit}' "/proc/$pid/mountinfo" 2>/dev/null)
  [ -n "$src" ] || continue
  if [ "$(basename "$src")" = "$(basename "$BUILD_DIR_PARENT")" ]; then
    BUILDING=1
  fi
done
if [ "$BUILDING" = 1 ]; then
  die "a 'cmake' process building $BUILD_DIR appears to still be running — wait for it to finish"
fi

# ---------------------------------------------------------------------------------------------
# self-build: only if BUILD_DIR is not already built. Builds REF (via `git archive`, NOT this
# worktree's live checkout) into BUILD_DIR using BUILD_IMAGE, building BUILD_IMAGE itself from
# DOCKERFILE first if it doesn't exist locally yet. CPU-only (NVIDIA_VISIBLE_DEVICES=none) --
# compiling needs no GPU, only nvcc + the CUDA toolkit headers/stub libs BUILD_IMAGE carries.
# ---------------------------------------------------------------------------------------------
SELF_BUILT=0
if [ -x "$BUILD_DIR/bin/llama-server" ]; then
  info "BUILD_DIR=$BUILD_DIR already has bin/llama-server -- using it as-is, not building"
else
  [ -n "$BUILD_IMAGE" ] || die "BUILD_DIR=$BUILD_DIR has no bin/llama-server, and BUILD_IMAGE is not set -- either point BUILD_DIR at an existing build, or set BUILD_IMAGE (e.g. pxa-release-build:cu128-u2204) to have this script build REF into it."
  if ! docker image inspect "$BUILD_IMAGE" >/dev/null 2>&1; then
    [ -f "$DOCKERFILE" ] || die "BUILD_IMAGE=$BUILD_IMAGE does not exist locally, and DOCKERFILE=$DOCKERFILE was not found to build it"
    info "BUILD_IMAGE=$BUILD_IMAGE not found locally -- building it from $DOCKERFILE"
    docker build -f "$DOCKERFILE" -t "$BUILD_IMAGE" "$(dirname "$DOCKERFILE")" \
      || die "docker build of $BUILD_IMAGE from $DOCKERFILE failed"
  fi
  info "self-building REF=$REF (commit $COMMIT_SHORT) into BUILD_DIR=$BUILD_DIR with BUILD_IMAGE=$BUILD_IMAGE"
  BUILD_SRC="$WORK/build-src"
  mkdir -p "$BUILD_SRC" "$BUILD_DIR"
  git -C "$REPO_ROOT" archive "$COMMIT" | tar -x -C "$BUILD_SRC"
  # The archive has no .git, so cmake/build-info.cmake cannot discover the commit and the packaged
  # binary used to print "version: 0 (unknown)". Compute the same two values git would have and
  # inject them (see the PXA_BUILD_* block in cmake/build-info.cmake).
  BI_COMMIT=$(git -C "$REPO_ROOT" rev-parse --short "$COMMIT")
  BI_NUMBER=$(git -C "$REPO_ROOT" rev-list --count "$COMMIT")
  [ -n "$BI_COMMIT" ] && [ "${BI_NUMBER:-0}" -gt 0 ] || die "could not derive build-info for $COMMIT"
  info "build-info to inject: number=$BI_NUMBER commit=$BI_COMMIT"
  # BUILD_DIR is bind-mounted a second time, nested under the BUILD_SRC mount at the exact path
  # cmake will write its build tree to -- this keeps generated build artifacts landing directly
  # in the host's real BUILD_DIR without polluting (or needing to copy out of) the git-archive'd
  # source checkout, and without ever touching this worktree's own checkout.
  docker run --rm --name "pxq-pkg-selfbuild-$$" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none \
    -v "$BUILD_SRC:/work" -v "$BUILD_DIR:/work/build-out" -w /work "$BUILD_IMAGE" \
    bash -c "nice -n 19 cmake -B build-out -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\"$CUDA_ARCHS\" -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=2 -DPXA_BUILD_NUMBER=$BI_NUMBER -DPXA_BUILD_COMMIT=$BI_COMMIT && nice -n 19 cmake --build build-out -j6 --target $BUILD_TARGETS" \
    || die "self-build failed (see docker output above)"
  [ -x "$BUILD_DIR/bin/llama-server" ] || die "self-build finished but $BUILD_DIR/bin/llama-server is still missing"
  SELF_BUILT=1
  info "self-build complete: $BUILD_DIR/bin/llama-server"
fi

# Post-processing (ldd/objdump/patchelf/strip) MUST use the image the binaries were actually
# linked in, or the reported/bundled library paths and detected glibc floor are simply wrong for
# what got built. Default to BUILD_IMAGE when this run just built with one; otherwise fall back
# to the campaign's day-to-day dev image (which is what an externally-supplied BUILD_DIR, like
# fence's, was built with).
if [ -z "$DEV_IMAGE" ]; then
  if [ "$SELF_BUILT" = 1 ]; then DEV_IMAGE="$BUILD_IMAGE"; else DEV_IMAGE="pxa-sm60-dev:latest"; fi
fi
info "DEV_IMAGE (post-processing) = $DEV_IMAGE"

STAGE="$WORK/pxa-$TAG"
mkdir -p "$STAGE"/{bin,lib,docs,bench/fair,bench/gate}

# ---------------------------------------------------------------------------------------------
# export the docs/scripts/tools tree at REF (NOT the working tree — REF is the source of truth)
# ---------------------------------------------------------------------------------------------
EXPORT="$WORK/export"
mkdir -p "$EXPORT"
git -C "$REPO_ROOT" archive "$COMMIT" | tar -x -C "$EXPORT"

req() { [ -e "$EXPORT/$1" ] || die "REF $REF is missing required path: $1"; }
req tools/pxa-launch.py
req bench/fair/run.sh
req bench/fair/protocol.md
req bench/fair/weights/MANIFEST.sha256
req bench/gate
req docs/COOKBOOK.md
req docs/PXQ-EXPORT.md
req docs/KNOWN-ISSUES.md
req bench/fair-battle.md
req LICENSE
req LICENSING.md
req README.md

RELNOTES=$(ls "$EXPORT"/RELEASE-NOTES-*.md 2>/dev/null | sort | tail -1)
[ -n "$RELNOTES" ] || die "no RELEASE-NOTES-*.md found at REF $REF"
info "using release notes: $(basename "$RELNOTES")"

# ---------------------------------------------------------------------------------------------
# bin/ — copy the binaries this order requires (the four *_BIN lists are in the configuration
# block at the top of this script). llama-pxq-export, llama-quantize and llama-gguf-split are
# built into BUILD_DIR on demand (CPU-only link, no GPU needed) if BUILD_DIR doesn't already
# have them; of those three only llama-pxq-export is then REQUIRED to be present.
# ---------------------------------------------------------------------------------------------
MISSING_ONDEMAND=""
for b in $REQUIRED_ONDEMAND_BIN $OPTIONAL_BIN; do
  [ -x "$BUILD_DIR/bin/$b" ] || MISSING_ONDEMAND="$MISSING_ONDEMAND $b"
done
if [ -n "$MISSING_ONDEMAND" ]; then
  info "building missing target(s) into BUILD_DIR:$MISSING_ONDEMAND (CPU link only, NVIDIA_VISIBLE_DEVICES=none)"
  docker run --rm --name "pxq-pkg-extra-$$" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none \
    -v "$BUILD_DIR/..":/work -w /work "$DEV_IMAGE" \
    bash -c "nice -n 19 cmake --build $(basename "$BUILD_DIR") -j6 --target $MISSING_ONDEMAND" \
    || err "on-demand target build failed for:$MISSING_ONDEMAND — a required one is checked below"
fi

for b in $REQUIRED_BIN $REQUIRED_ONDEMAND_BIN $TEST_BIN; do
  [ -x "$BUILD_DIR/bin/$b" ] || die "BUILD_DIR is missing required binary: bin/$b"
  cp -a "$BUILD_DIR/bin/$b" "$STAGE/bin/"
done
for b in $OPTIONAL_BIN; do
  if [ -x "$BUILD_DIR/bin/$b" ]; then
    cp -a "$BUILD_DIR/bin/$b" "$STAGE/bin/"
  else
    info "optional binary not built, skipping: $b"
  fi
done
[ -x "$STAGE/bin/llama-quantize" ] || err "WARNING: llama-quantize could not be included (not built, and the on-demand build failed or was skipped)"
# Not a warning: the README's and docs/COOKBOOK.md's "leave PXQ" recipe is two commands, and this
# is the first one. A package that cannot run it is not shippable.
[ -x "$STAGE/bin/llama-pxq-export" ] || die "bin/llama-pxq-export is missing from the staged package — it is required (see the asset list at the top of this script). Build it with: cmake --build $BUILD_DIR --target llama-pxq-export"

# ---------------------------------------------------------------------------------------------
# lib/ — the engine's own shared libs, plus the CUDA/OpenMP runtime libs the binaries actually need.
# Found by ldd INSIDE the dev container (matches the runtime the binaries were linked against);
# copied as the exact versioned file (soname + real file), not the container's dev symlink tree.
# Driver libs (libcuda.so*, libnvidia-*) are intentionally excluded — those come from the host
# driver, never bundled.
# ---------------------------------------------------------------------------------------------
for f in src/libllama.so ggml/src/libggml.so examples/mtmd/libmtmd.so; do
  [ -f "$BUILD_DIR/$f" ] || die "BUILD_DIR is missing expected lib: $f"
  cp -a "$BUILD_DIR/$f" "$STAGE/lib/"
done
# any other libggml-*.so variants a different cmake config might produce (backend split builds)
find "$BUILD_DIR" -maxdepth 3 -name 'libggml*.so*' -exec cp -a {} "$STAGE/lib/" \; 2>/dev/null

LDD_SCRIPT='
set -e
export LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd
for b in /build/bin/llama-server /build/ggml/src/libggml.so /build/src/libllama.so /build/examples/mtmd/libmtmd.so; do
  ldd "$b" 2>/dev/null
done
'
LDD_OUT=$(dimg -v "$BUILD_DIR:/build:ro" -- "$LDD_SCRIPT") \
  || die "ldd-in-container step failed"

# Lines look like: "libcudart.so.12 => /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12 (0x...)"
# Keep cuda-toolkit runtime deps (cudart/cublas/cublasLt/nvrtc/nccl/etc under /usr/local/cuda* or
# a bare libnccl under /lib or /usr/lib) PLUS libgomp (GNU OpenMP runtime) PLUS libstdc++:
#  - libgomp: confirmed by an actual bare-container smoke test that a minimal Ubuntu base does
#    NOT ship libgomp.so.1 -- omitting it made llama-server fail to even start with "error while
#    loading shared libraries: libgomp.so.1: cannot open shared object file".
#  - libstdc++: bundled DELIBERATELY, even though most distros carry SOME libstdc++.so.6,
#    because the one on an older host may be older than what these binaries were linked
#    against. Bundling OUR copy (matching DEV_IMAGE's glibc floor) and having the wrapper
#    scripts put ./lib on LD_LIBRARY_PATH ahead of the system search path means the host's own
#    libstdc++ version stops mattering -- only its glibc does (see the GLIBC_FLOOR/GLIBCXX_FLOOR
#    detection below; GLIBCXX_FLOOR is reported for transparency but the actual constraint on
#    the host, once libstdc++ is bundled, is glibc alone).
# libc/libpthread/librt/libdl/libm/libgcc_s are NOT bundled -- those come from glibc itself,
# which this package cannot safely bundle (same reason as the driver: it has to match the
# kernel/loader already on the host). Explicitly excludes driver libs (libcuda.so*,
# libnvidia-*), which must come from the host driver, never bundled.
EXTRA_LIB_PATHS=$(echo "$LDD_OUT" \
  | awk '{print $3}' \
  | grep -E '^/' \
  | grep -viE 'libcuda\.so|libnvidia-' \
  | grep -iE 'libcudart|libcublas|libcublasLt|libnvrtc|libnccl|libcusparse|libcusolver|libcurand|libcufft|libgomp|libstdc\+\+' \
  | sort -u)
[ -n "$EXTRA_LIB_PATHS" ] || die "ldd found no CUDA/OpenMP/libstdc++ runtime libs to bundle — something is wrong with BUILD_DIR or DEV_IMAGE"

info "extra runtime libs to bundle (resolved inside $DEV_IMAGE):"
echo "$EXTRA_LIB_PATHS" | sed 's/^/  /'

# For each dependency, resolve it to its FULLY-versioned real file inside the container (e.g.
# libcudart.so.12 -> libcudart.so.12.8.90) and ship both: the fully-versioned file (the "exact
# versioned file" this order asks for) and the soname the binaries actually DT_NEEDED-reference.
#
# The soname is a HARD LINK to the versioned file, not a second copy and not a symlink. A second
# copy is what this script used to write, and for the 60;61;70 build it cost 1.25 GB of duplicated
# CUDA libraries and pushed the finished asset past the 2 GiB upload limit: gzip cannot see that
# two files 700 MB apart in the stream are identical, so every duplicate is paid for in full. A
# hard link is stored once by tar and comes out of tar as two ordinary files, which is exactly the
# property the copy was there to guarantee; a symlink would not be (it can arrive dangling if
# someone extracts only part of the archive).
while read -r libpath; do
  [ -n "$libpath" ] || continue
  base=$(basename "$libpath")
  realname=$(dimg -v "$BUILD_DIR:/build:ro" -- "readlink -f '$libpath' | xargs basename" 2>/dev/null)
  [ -n "$realname" ] || { err "WARNING: could not resolve real file for $libpath"; continue; }
  dimg -v "$BUILD_DIR:/build:ro" -- "cat \$(readlink -f '$libpath')" > "$STAGE/lib/$realname" 2>/dev/null \
    || { err "WARNING: could not copy $libpath"; continue; }
  [ -s "$STAGE/lib/$realname" ] || { err "WARNING: copy of $libpath produced an empty file"; rm -f "$STAGE/lib/$realname"; continue; }
  if [ "$base" != "$realname" ]; then
    ln -f "$STAGE/lib/$realname" "$STAGE/lib/$base" 2>/dev/null \
      || cp -a "$STAGE/lib/$realname" "$STAGE/lib/$base"
  fi
done <<< "$EXTRA_LIB_PATHS"

[ -n "$(ls -A "$STAGE/lib" 2>/dev/null)" ] || die "lib/ ended up empty"

# ---------------------------------------------------------------------------------------------
# RPATH: prefer patchelf (if the dev container has it); else document the wrapper-script
# fallback so binaries invoked directly still fail informatively, and the shipped wrappers
# (which set LD_LIBRARY_PATH) are the supported path.
# ---------------------------------------------------------------------------------------------
PATCHELF_OK=0
if dimg -- 'command -v patchelf' >/dev/null 2>&1; then
  PATCHELF_OK=1
  info "patchelf found in $DEV_IMAGE — setting RPATH=\$ORIGIN/../lib on bin/* and lib/*.so*"
  # NOTE: this image has no `file` binary — detect ELF-ness with `readelf -h` instead.
  # NOTE: "\$ORIGIN" (escaped) is deliberate: this literal string is passed through TWO shells
  # (this script's, then the container's `bash -c`) before reaching patchelf, and patchelf must
  # receive the literal 8 characters "$ORIGIN" — the dynamic LINKER expands that token at load
  # time, not any shell here; an unescaped $ORIGIN would have the inner bash substitute (empty,
  # unset) FIRST, silently writing a broken RPATH.
  dimg -v "$STAGE:/stage" -- '
    set -e
    for f in /stage/bin/*; do
      [ -x "$f" ] || continue
      readelf -h "$f" >/dev/null 2>&1 || continue
      patchelf --set-rpath "\$ORIGIN/../lib" "$f" || true
    done
    for f in /stage/lib/*.so*; do
      [ -f "$f" ] || continue
      readelf -h "$f" >/dev/null 2>&1 || continue
      patchelf --set-rpath "\$ORIGIN" "$f" 2>/dev/null || true
    done
  '
else
  info "patchelf NOT found in $DEV_IMAGE — binaries rely on the wrapper scripts' LD_LIBRARY_PATH (documented in START-HERE.md)"
fi

# ---------------------------------------------------------------------------------------------
# strip debug symbols if present and it actually shrinks things
# ---------------------------------------------------------------------------------------------
SIZE_BEFORE_STRIP=$(du -sb "$STAGE/bin" "$STAGE/lib" 2>/dev/null | awk '{s+=$1} END{print s}')
# NOTE: this image has no `file` binary — detect ELF-ness and debug sections with readelf.
HAS_DEBUG=$(dimg -v "$STAGE:/stage" -- '
  found=0
  for f in /stage/bin/* /stage/lib/*.so*; do
    [ -f "$f" ] || continue
    readelf -h "$f" >/dev/null 2>&1 || continue
    readelf -S "$f" 2>/dev/null | grep -qi "\.debug_info" && found=1
  done
  echo $found
')
if [ "$HAS_DEBUG" = "1" ]; then
  info "debug symbols present — stripping bin/* and lib/*.so*"
  dimg -v "$STAGE:/stage" -- '
    for f in /stage/bin/*; do [ -x "$f" ] && readelf -h "$f" >/dev/null 2>&1 && strip --strip-unneeded "$f" 2>/dev/null; done
    for f in /stage/lib/*.so*; do [ -f "$f" ] && strip --strip-unneeded "$f" 2>/dev/null; done
  '
else
  info "no debug symbols found in bin/lib — nothing to strip"
fi
SIZE_AFTER_STRIP=$(du -sb "$STAGE/bin" "$STAGE/lib" 2>/dev/null | awk '{s+=$1} END{print s}')
info "bin/+lib/ size before strip: $SIZE_BEFORE_STRIP bytes, after: $SIZE_AFTER_STRIP bytes"

# Re-link the soname to its versioned file. patchelf rewrites a file by replacing it, which breaks
# the hard link made when the libraries were staged, so the pair has to be joined again here --
# after every tool that edits these files has run and before anything is measured or archived.
# Identical content only: the check is cmp, not the file name.
for so in "$STAGE"/lib/*.so.*; do
  [ -f "$so" ] || continue
  case "$so" in *.so.[0-9]) ;; *) continue ;; esac      # a bare soname, e.g. libcublasLt.so.12
  real=$(ls -1 "$so".* 2>/dev/null | sort -V | tail -1)
  [ -n "$real" ] && [ -f "$real" ] || continue
  [ "$(stat -c%i "$so")" = "$(stat -c%i "$real")" ] && continue
  if cmp -s "$so" "$real"; then
    ln -f "$real" "$so" && info "lib/: $(basename "$so") stored once, as a link to $(basename "$real")"
  else
    err "WARNING: $(basename "$so") and $(basename "$real") differ -- both shipped in full"
  fi
done
SIZE_AFTER_LINK=$(du -sb --apparent-size "$STAGE/bin" "$STAGE/lib" 2>/dev/null | awk '{s+=$1} END{print s}')
DISK_AFTER_LINK=$(du -sb "$STAGE/bin" "$STAGE/lib" 2>/dev/null | awk '{s+=$1} END{print s}')
info "bin/+lib/ named bytes: $SIZE_AFTER_LINK, distinct bytes to archive: $DISK_AFTER_LINK"

# ---------------------------------------------------------------------------------------------
# wrappers
# ---------------------------------------------------------------------------------------------
cat > "$STAGE/pxa-launch" <<'WRAP'
#!/bin/bash
# pxa-launch — wraps tools/pxa-launch.py for this package's flat bin/+lib/ layout.
#
# tools/pxa-launch.py's own engine auto-detection (ENGINE_DIR_CANDIDATES) looks for a
# BUILD-TREE layout (E/bin, E/src, E/ggml/src, E/examples/mtmd) — this release tarball ships a
# flat bin/ + lib/ instead, so auto-detection alone would not find bin/llama-server here.
# PXA_ENGINE_DIR=$HERE tells it exactly where to look; LD_LIBRARY_PATH=$HERE/lib is then
# preserved by pxa-launch.py's own engine_ld_path() (which appends the build-tree subdirs it
# finds under PXA_ENGINE_DIR — none, in this flat layout — to whatever LD_LIBRARY_PATH it
# inherits, which is this line's $HERE/lib).
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PXA_ENGINE_DIR="$HERE"
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 "$HERE/tools/pxa-launch.py" "$@"
WRAP
chmod +x "$STAGE/pxa-launch"

cat > "$STAGE/run-server.sh" <<'WRAP'
#!/bin/bash
# run-server.sh — run llama-server directly (no auto-tuning) with this package's ./lib set up.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Card numbers should mean the same thing here as they do in nvidia-smi. CUDA's own
# default is CUDA_DEVICE_ORDER=FASTEST_FIRST, which sorts the cards by compute
# capability, while nvidia-smi numbers them by PCI bus id — so on a box with mixed
# cards, CUDA_VISIBLE_DEVICES=0 starts on a different GPU than the one `nvidia-smi -L`
# calls 0, silently. PCI_BUS_ID makes the two numberings one numbering. Yours wins if
# you set it: this only fills in a default.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
exec "$HERE/bin/llama-server" "$@"
WRAP
chmod +x "$STAGE/run-server.sh"

mkdir -p "$STAGE/tools"
cp -a "$EXPORT/tools/pxa-launch.py" "$STAGE/tools/pxa-launch.py"
chmod +x "$STAGE/tools/pxa-launch.py"

# The conversion half of the stock-quant path. bin/llama-quantize can only read a GGUF, so a
# package without the converter leaves a user who downloaded a Hugging Face model with nothing to
# run first, and docs/tutorials/04's first command is exactly this script. gguf-py is the library
# it imports, so the two travel together or neither works.
cp -a "$EXPORT/convert_hf_to_gguf.py" "$STAGE/tools/convert_hf_to_gguf.py"
chmod +x "$STAGE/tools/convert_hf_to_gguf.py"
[ -d "$EXPORT/gguf-py" ] || die "REF $REF is missing gguf-py (convert_hf_to_gguf.py imports it)"
cp -a "$EXPORT/gguf-py" "$STAGE/tools/gguf-py"
# convert_hf_to_gguf.py finds gguf-py by walking up from its own directory to the repo root; in
# this flat package the two sit side by side in tools/, so say so once rather than leave the
# import to luck.
cat > "$STAGE/tools/README-convert.md" <<'CONV'
# Converting a Hugging Face model in this package

`convert_hf_to_gguf.py` turns a downloaded Hugging Face model directory into a GGUF file.
It needs the `gguf` python package that sits next to it here:

```bash
cd tools
PYTHONPATH=$PWD/gguf-py python3 convert_hf_to_gguf.py /path/to/hf-model-dir --outfile model-f16.gguf --outtype f16
```

Then quantise it with the engine's own tool:

```bash
../bin/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

`../bin/llama-imatrix` builds an importance matrix first if you want the imatrix-guided types, and
`../bin/llama-perplexity` measures what a quantisation cost you. The PXQ tiers are not produced by
`llama-quantize` in this package — see the release notes for where that tool lives.
CONV

# ---------------------------------------------------------------------------------------------
# bench/fair, bench/gate, docs, license
# ---------------------------------------------------------------------------------------------
cp -a "$EXPORT/bench/fair/." "$STAGE/bench/fair/"

# bench/gate/ is copied BY NAME, not by wildcard, and LAST-RUN.md is deliberately NOT shipped.
#
# FOUND 2026-09-11, caught by tarball_scan.sh on the Sep-11 package: build-machine-paths hits=11,
# box-tooling hits=5, every one of them in this single file. LAST-RUN.md is the record of a gate
# run ON THE BUILD MACHINE -- it carries absolute /mnt paths and, worse, the internal model
# FILENAMES that run was pointed at. The release rules keep both out of shipping files, and
# neither is of any use to a user: what a user needs is the gate TOOLING.
#
# The wildcard was the defect. Every other doc in this script is copied by name (see the docs/
# block below); this was the one line that shipped "whatever is in the directory", so a run record
# written on Sep-9 evening was published in the Sep-11 package without anyone deciding to publish
# it -- and it was absent from the Sep-9 package only because that run had not happened yet. The
# scan result is the provenance: Sep-9 hits=0, Sep-11 hits=11, same script, different file.
#
# Sanitizing the file would be the wrong fix -- it is rewritten by every gate run, so the leak
# returns with the next one. Withholding it is durable. A package's gate result still reaches
# users through START-HERE.md and the manifest.
# keep() asserts the OUTCOME, not the command. As first written it checked that the file existed at
# REF and then ran `cp -a` without ever looking at the result -- and that copy failed for all three
# prompts, because the mkdir near the top of this script creates bench/gate/ but NOT
# bench/gate/prompts/. There is no `set -e` here, so the run continued and tarred a package whose own
# gate harness cannot start: run-gate.sh loops over those three files and dies "missing prompt file".
# The file was never lost by REF; it was lost in transit. A guard that checks the input and not the
# outcome reports success for a copy that did not happen -- the same shape as the LAST-RUN.md leak
# this function was written to fix, one level down.
keep() {
  [ -e "$EXPORT/$1" ] || die "REF $REF lost a required gate file: $1"
  mkdir -p "$STAGE/$(dirname "$1")"
  cp -a "$EXPORT/$1" "$STAGE/$1"
  [ -e "$STAGE/$1" ] || die "required gate file did not survive staging: $1"
}
keep bench/gate/README.md
keep bench/gate/run-gate.sh
keep bench/gate/prompts/coherence.txt
keep bench/gate/prompts/needle20801.txt
keep bench/gate/prompts/needle3121.txt

cp -a "$EXPORT/bench/fair-battle.md" "$STAGE/bench/fair-battle.md"

cp -a "$EXPORT/README.md" "$STAGE/docs/README.md"
cp -a "$RELNOTES" "$STAGE/docs/$(basename "$RELNOTES")"
cp -a "$EXPORT/docs/COOKBOOK.md" "$STAGE/docs/COOKBOOK.md"
[ -f "$EXPORT/docs/LAUNCHER.md" ] && cp -a "$EXPORT/docs/LAUNCHER.md" "$STAGE/docs/LAUNCHER.md"
# The reference for bin/llama-pxq-export, shipped with the binary rather than left online-only.
cp -a "$EXPORT/docs/PXQ-EXPORT.md" "$STAGE/docs/PXQ-EXPORT.md"
cp -a "$EXPORT/docs/KNOWN-ISSUES.md" "$STAGE/docs/KNOWN-ISSUES.md"
cp -a "$EXPORT/LICENSE" "$STAGE/LICENSE"
cp -a "$EXPORT/LICENSING.md" "$STAGE/LICENSING.md"
# top-level convenience copy of README/notes too, so a lister sees them at both levels
# (named after the actual REF's files, not hardcoded, so a rerun on a later tag doesn't ship a
# stale filename)
cp -a "$EXPORT/README.md" "$STAGE/README.md"
cp -a "$RELNOTES" "$STAGE/$(basename "$RELNOTES")"

# ---------------------------------------------------------------------------------------------
# THE CROSS-REFERENCED DOCS -- every link target the shipped docs name, not just the six above
# ---------------------------------------------------------------------------------------------
# ⚠ ADDED 2026-09-12 AFTER A LINK AUDIT OF THE SHIPPED rc3 PACKAGE: 13 .md files, every relative
# target resolved against the directory of the file that names it -> 25 DEAD LINKS. Every one of
# them pointed at a file that EXISTS IN THE TREE, so the defect was packaging, not content: the
# README names fourteen relative targets and this script staged six. The worst was `LEVERS.md` --
# the standing release rule names docs/LEVERS.md as THE lever doc, and the shipped README.md,
# RELEASE-NOTES-2026-09-11.md and docs/COOKBOOK.md all point at it, yet it was in no package at all.
# Ship each at the path the references already use, and FAIL if one is absent rather than stage a
# package whose own cross-references are dead -- a missing doc is the same class of defect as the
# gate-file copy this script already guards with keep(): the input was checked, the outcome was not.
keep_doc() { # path relative to EXPORT == path relative to STAGE
  [ -e "$EXPORT/$1" ] || die "a cross-referenced doc is absent from REF $REF: $1
       (a shipped README/notes file links to it; packaging without it ships a dead link)"
  mkdir -p "$STAGE/$(dirname "$1")"
  # -L: docs/LEVERS.md is a SYMLINK (-> lab/LEVERS.md). `cp -a` alone would carry the link into the
  # tarball, where its target is not present, and leave every reference dangling -- so dereference,
  # and ship the content at both paths, which is what the two reference styles in-tree expect.
  mkdir -p "$(dirname "$STAGE/$1")"
  cp -aL "$EXPORT/$1" "$STAGE/$1"
  [ -e "$STAGE/$1" ] || die "cross-referenced doc did not survive staging: $1"
}
# docs/LEVERS.md is a REAL FILE from v2026.09.20 on. It used to be a symlink to an engineering
# notebook (docs/lab/LEVERS.md) that was never meant for readers; the user-facing lever reference
# now lives at docs/LEVERS.md itself and the notebook is not part of the repository.
keep_doc docs/LEVERS.md
keep_doc docs/QUANTIZING.md
keep_doc docs/PXQU-CONVERT.md
keep_doc docs/PXA-SM70-SERVING.md
keep_doc BUILD-FROM-SOURCE.md
keep_doc bench/README.md
keep_doc RELEASE-NOTES-2026-09-02.md
keep_doc RELEASE-NOTES-2026-09-07.md
keep_doc RELEASE-NOTES-2026-09-09.md
keep_doc RELEASE-NOTES-2026-09-13.md

# The compose file and its plain-language guide, added for v2026.09.20: README.md tells the reader
# this release ships a Compose file for the engine and both sidecars, and names docker/COMPOSE.md
# as where it is explained. A package that makes that promise and then does not carry the two files
# sends the reader looking for something that is not there.
keep_doc docker/COMPOSE.md
keep_doc docker/docker-compose.yml

# Added 2026-09-14, again named by the dead-link guard: the 09-13 notes and docs/COOKBOOK.md cite the
# leaderboard for every cross-engine cell and every pending re-measure. The board has no outbound
# links of its own, so shipping it closes the class without opening another one, and the numbers in
# the notes stay traceable to the evidence inside the package rather than to a page on the web.
keep_doc bench/LEADERBOARD.md

# The step-by-step guides, added for v2026.09.20. They are the documentation a first-time reader is
# pointed at, they cross-reference each other, and they are written against THIS package's layout
# (./pxa-launch, bin/, tools/), so a package without them sends that reader to a web page for
# instructions about the files in their own directory. Named one by one, like every other doc here.
for t in "$EXPORT"/docs/tutorials/*.md; do
  [ -e "$t" ] || die "REF $REF is missing docs/tutorials/ (the step-by-step guides)"
  keep_doc "docs/tutorials/$(basename "$t")"
done

# Added 2026-09-14, named by the guard below once it learned to read <img> tags: every shipped page
# opens with the banner as an HTML <img>, the README embeds the fair-battle chart, and the 09-07
# notes embed their five dark chart variants. Images are copied whole, never merged.
keep_doc docs/assets/pxa-network-banner.png
keep_doc docs/assets/before-after-2026-09-07-dark.png
keep_doc docs/assets/context-vs-1cat-2026-09-07-dark.png
keep_doc docs/assets/home-spec-ladder-2026-09-07-dark.png
keep_doc docs/assets/us-vs-them-2026-09-03-dark.png
keep_doc docs/assets/vs-1cat-2026-09-07-dark.png
keep_doc banner.png
keep_doc bench/fair-battle.svg

# ⚠ ADDED 2026-09-12, NAMED BY THE DEAD-LINK GUARD BELOW, not by inspection. Staging the three
# historical notes and the docs they cite pulled in this second ring: QUANTIZING.md,
# PXA-SM70-SERVING.md and 09-09 all point at docs/VLLM.md (the serving line's own reference, 79 KB),
# and VLLM.md in turn points at the sm_60 recipe, the QWEN4EXP converter note, the vllm-pxq4 tool
# README and the serving stack's own README. The closure from the staged set is FIVE files and it
# terminates -- measured, not assumed -- which is why the answer is to ship them rather than de-link
# five sentences in docs that exist to be read. Nothing here is a doc we would withhold: the one
# file in pxa/pxq4/ that would not belong in a release (BOX-ENV.md, a box-environment record) is not
# linked from any shipped doc, so it stays out, and a lone README with its siblings absent is the
# smaller cost than a dead link in a shipped table.
keep_doc docs/VLLM.md
keep_doc docs/PXA-SM60-SERVING.md
keep_doc docs/QWEN4EXP-PXQ4.md
keep_doc tools/vllm-pxq4/README.md
keep_doc pxa/pxq4/README.md

# The two files copied INTO docs/ carry links written for the repo ROOT, so every one of them breaks
# by exactly one directory (docs/README.md alone accounted for 15 of the 25 dead links). Rewrite the
# targets for their new depth: docs/X.md -> X.md, and anything at the root gains a "../".
for f in "$STAGE/docs/README.md" "$STAGE/docs/$(basename "$RELNOTES")"; do
  [ -f "$f" ] || continue
  sed -i -E \
    -e 's#\]\(docs/#](#g' \
    -e 's#\]\(bench/#](../bench/#g' \
    -e 's#\]\(RELEASE-NOTES-#](../RELEASE-NOTES-#g' \
    -e 's#\]\(BUILD-FROM-SOURCE\.md\)#](../BUILD-FROM-SOURCE.md)#g' \
    -e 's#\]\(docker/#](../docker/#g' \
    -e 's#src="docs/#src="#g' \
    -e 's#src="bench/#src="../bench/#g' \
    -e 's#src="banner\.png"#src="../banner.png"#g' \
    "$f"
done

# The three HISTORICAL notes are cited by the shipped README as the evidence behind numbers the
# README itself quotes ("that is the only place the +40% figure comes from"), and the release rule
# requires every number in a shipped doc to be traceable to that release's bench evidence -- so they
# have to travel with the package. They were written for the repo, where they sit beside docs/ at
# the root and where a provenance comment may name a build-machine path. Two repairs, both narrow:
#
#  (1) LINKS. `](COOKBOOK.md)` written from the root means docs/COOKBOOK.md. The package's root
#      copies of these notes still sit at the root, so the bare name resolves to nothing, and
#      09-09's link to `../bench/gate/LAST-RUN.md` points above the package into nothing at all.
#  (2) PATHS. 09-07 names the absolute path of the profiler output behind the PASCAL-DECODE-GAP
#      numbers, and the leak guard below refuses to tar any stage carrying /mnt -- correctly.
#
# Sanitizing is the WRONG fix for a file a gate run rewrites (see the LAST-RUN.md note above): the
# leak returns with the next run. These three are frozen history that no run regenerates, so the
# prefix is the only thing that has to go and the reference keeps its meaning without it.
#
# LAST-RUN.md is deliberately NOT repointed at some other file: this script withholds it on purpose
# (build-machine paths + internal model filenames), so the sentence that names it is DE-LINKED and
# left as prose, which is what it is -- a pointer to an artifact that is not ours to publish.
for f in "$STAGE"/RELEASE-NOTES-*.md; do
  [ -f "$f" ] || continue
  grep -q -e '/mnt/' -e '/boot/' -e '](COOKBOOK.md)' -e '](KNOWN-ISSUES.md)' -e '](VLLM.md)' \
       -e 'LAST-RUN.md](' "$f" || continue
  sed -i -E \
    -e 's#\]\(COOKBOOK\.md\)#](docs/COOKBOOK.md)#g' \
    -e 's#\]\(KNOWN-ISSUES\.md\)#](docs/KNOWN-ISSUES.md)#g' \
    -e 's#\]\(VLLM\.md\)#](docs/VLLM.md)#g' \
    -e 's#\[`bench/gate/LAST-RUN\.md`\]\(\.\./bench/gate/LAST-RUN\.md\)#`bench/gate/LAST-RUN.md`#g' \
    -e 's#/mnt/[^ ,)]*#<build-machine path>#g' \
    -e 's#/boot/[^ ,)]*#<build-machine path>#g' \
    "$f"
done

# ---------------------------------------------------------------------------------------------
# VERSION
# ---------------------------------------------------------------------------------------------
# cmake's own stdout log does not echo back -D flags (only messages the project chooses to
# print); the flags actually in effect live in the build dir's own CMakeCache.txt instead.
BUILD_FLAGS=$(grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|GGML_CUDA|GGML_SCHED_MAX_COPIES)[:=]' \
  "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | grep -v -- '-STRINGS:' || true)
[ -n "$BUILD_FLAGS" ] || BUILD_FLAGS="(CMakeCache.txt not found or had none of the expected keys)"
CUDART_SONAME=$(find "$STAGE/lib" -maxdepth 1 -name 'libcudart.so.*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
if [ -z "$CUDA_MAJOR_MINOR" ] && [ -n "$CUDART_SONAME" ]; then
  # libcudart.so.12.8.90 -> 12.8
  CUDA_MAJOR_MINOR=$(echo "$CUDART_SONAME" | sed -E 's/^libcudart\.so\.([0-9]+)\.([0-9]+).*/\1.\2/')
fi
CUDA_MAJOR_MINOR=${CUDA_MAJOR_MINOR:-unknown}

CUOBJDUMP_ARCHES=$(dimg -v "$BUILD_DIR:/build:ro" -- \
  "cuobjdump --list-elf /build/ggml/src/libggml.so 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u -V" \
  2>/dev/null | tr '\n' ' ')
CUOBJDUMP_ARCHES=${CUOBJDUMP_ARCHES:-"(cuobjdump unavailable or reported nothing; built for sm_${CUDA_ARCHS//;/, sm_}) "}

# The real minimum-OS requirement is a glibc/libstdc++ floor, NOT "any Linux x86_64" -- found the
# hard way, packaging this very script: DEV_IMAGE's own base distro sets this floor (whatever
# glibc/libstdc++ it links against), and it can be NEWER than whatever "generic Linux" a reader
# assumes. Confirmed by an actual bare-container test: binaries built in a Ubuntu 24.04 (glibc
# 2.39) container refused to even start on Ubuntu 22.04 (glibc 2.35) with "version GLIBC_2.38
# not found" / "GLIBCXX_3.4.32 not found", despite every .so the binary needs being present in
# ./lib -- glibc/libstdc++ are the two libraries this package deliberately does NOT bundle (they
# come from the base OS, like the driver), so the base OS must be new enough on its own.
GLIBC_FLOOR=$(dimg -v "$STAGE:/stage" -- \
  "objdump -T /stage/bin/* /stage/lib/*.so* 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1" \
  2>/dev/null)
GLIBCXX_FLOOR=$(dimg -v "$STAGE:/stage" -- \
  "objdump -T /stage/bin/* /stage/lib/*.so* 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -1" \
  2>/dev/null)
GLIBC_FLOOR=${GLIBC_FLOOR:-"(objdump unavailable or reported nothing)"}
GLIBCXX_FLOOR=${GLIBCXX_FLOOR:-"(objdump unavailable or reported nothing)"}
info "minimum libc floor: $GLIBC_FLOOR / $GLIBCXX_FLOOR"

# VERSION ships to users, so it carries no path from the machine that built it: the build
# directory is NAMED, not located, plus whether this script built it or was handed it.
# Through rc1 this field was the absolute $BUILD_DIR, which told a reader nothing and named
# a directory on my box.
cat > "$STAGE/VERSION" <<EOF
tag:              $TAG
commit:           $COMMIT
ref:              $REF
build_dir:        $(basename "$BUILD_DIR")$([ "$SELF_BUILT" = 1 ] && echo " (self-built by this script)" || echo " (supplied)")
packaged:         $(date -u +%Y-%m-%dT%H:%M:%SZ)
cuda_runtime:     $CUDA_MAJOR_MINOR
cuda_archs_built: $CUDA_ARCHS
cuobjdump_arches: $CUOBJDUMP_ARCHES
glibc_floor:      $GLIBC_FLOOR
glibcxx_floor:    $GLIBCXX_FLOOR
patchelf_rpath:   $([ "$PATCHELF_OK" = 1 ] && echo yes || echo "no (wrapper-script LD_LIBRARY_PATH only)")
cmake_flags:
$BUILD_FLAGS
EOF

# ---------------------------------------------------------------------------------------------
# START-HERE.md — rendered by scripts/gen-start-here.sh, the SAME renderer a maintainer runs to
# refresh the START-HERE.md committed at the repo root. One template (scripts/START-HERE.md.in),
# one renderer, so the tarball's copy and the repository's copy cannot drift. The driver-floor
# and distro-hint tables live in that script; everything below is what only THIS build knows.
# ---------------------------------------------------------------------------------------------
NVIDIA_SMI_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
NVIDIA_SMI_DRIVER=${NVIDIA_SMI_DRIVER:-"(could not read nvidia-smi on the packaging host)"}

[ -x "$HERE/gen-start-here.sh" ] || die "missing renderer: $HERE/gen-start-here.sh"
TAG="$TAG" \
CUDA_MAJOR_MINOR="$CUDA_MAJOR_MINOR" \
GLIBC_FLOOR="$GLIBC_FLOOR" \
GLIBCXX_FLOOR="$GLIBCXX_FLOOR" \
BUILD_HOST_DRIVER="$NVIDIA_SMI_DRIVER" \
PATCHELF_OK="$PATCHELF_OK" \
HEADER=0 \
OUT="$STAGE/START-HERE.md" \
  "$HERE/gen-start-here.sh" || die "START-HERE.md render failed"

# ---------------------------------------------------------------------------------------------
# DEAD-LINK GUARD -- refuse to TAR a package whose own cross-references do not resolve
# ---------------------------------------------------------------------------------------------
# Same shape as the leak guard below, and the same lesson keep() already records one level up: a
# check that reads the INPUT (is the file there in REF?) reports success for a package that ships a
# broken reference. This one reads the STAGE.
#
# The defect class is not hypothetical. The shipped v2026.09.11-rc3 package had 25 dead links
# across its 13 .md files, every one of them a target that EXISTS in the tree -- so the fault was
# packaging, not content: the README names fourteen relative targets and this script staged six.
# Repointing those six would have fixed one package; this guard is what fixes the NEXT cut, because
# the same mistake arrives wearing a different filename (a note links a doc, and that doc links the
# next one). Targets are resolved against the directory of the file that NAMES them, which is the
# only reading a reader has.
#
# A target that is missing on purpose (LAST-RUN.md: build-machine paths + internal model
# filenames, withheld by the bench/gate block above) must be DE-LINKED in the sentence that names
# it, not tolerated here -- an allowlist would re-open the hole for the next file that lands in it.
DEADLINK=$(python3 - "$STAGE" <<'PY'
import os,re,sys
stage=sys.argv[1]
pat=re.compile(r'\[[^\]]*\]\(([^)\s]+)\)|<img[^>]*\ssrc="([^"]+)"')
dead=[]
for dp,_dn,fn in os.walk(stage):
    for f in fn:
        if not f.endswith('.md'):
            continue
        p=os.path.join(dp,f)
        try:
            txt=open(p,encoding='utf-8',errors='replace').read()
        except OSError:
            continue
        for m in pat.finditer(txt):
            t=(m.group(1) or m.group(2) or '').split('#')[0].strip()
            if not t or '://' in t or t.startswith('mailto:'):
                continue
            if not os.path.exists(os.path.normpath(os.path.join(dp,t))):
                dead.append('%s: %s' % (os.path.relpath(p,stage),t))
print('\n'.join(sorted(set(dead))))
PY
)
if [ -n "$DEADLINK" ]; then
  die "staged package has dead cross-references -- refusing to tar:
$DEADLINK
(each line is <file that names the link>: <target>; either ship the target with keep_doc, or
 de-link the sentence when the target is an artifact we deliberately do not publish)"
fi

# ---------------------------------------------------------------------------------------------
# leak guard -- refuse to TAR a package that carries build-machine paths
# ---------------------------------------------------------------------------------------------
# The staging rules in this script have now twice been the difference between a clean package and
# a published one (the bench/gate wildcard above; the empty-REF provenance bug in the wrapper).
# Both were silent: the package built, the tarball existed, and the fault was only visible to a
# reader who went looking. This guard is the one place the package build can FAIL on the thing it
# shipped rather than on the thing it printed.
#
# It checks the STAGE tree, not the repository, because the leak class lives in files this script
# WRITES (VERSION, START-HERE.md, run-server.sh, the manifest) that the repository never sees.
# `-I` skips binaries, so the .so/.elf payload is not read; two of the three build-machine roots
# are enough to catch the class; a fuller scan lives outside the repo and remains the full
# scan (attribution, host names, private branches) run against the finished artifact.
LEAK=$(grep -rIl -e '/mnt/' -e '/boot/' "$STAGE" 2>/dev/null || true)
if [ -n "$LEAK" ]; then
  die "staged package leaks build-machine paths -- refusing to tar:
$LEAK
$(grep -rIn -e '/mnt/' -e '/boot/' "$STAGE" 2>/dev/null | head -20)"
fi

# ---------------------------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
TARNAME="pxa-${TAG}-linux-x86_64-cuda${CUDA_MAJOR_MINOR}-${ARCH_TAG}.tar.gz"
TARPATH="$OUT_DIR/$TARNAME"

info "listing (pre-tar):"
find "$STAGE" -maxdepth 2 | sed "s#$STAGE#  pxa-$TAG#"

tar -C "$WORK" -czf "$TARPATH" "pxa-$TAG"
( cd "$OUT_DIR" && sha256sum "$TARNAME" > "$TARNAME.sha256" )

TARSIZE=$(du -h "$TARPATH" | awk '{print $1}')
TARSIZE_BYTES=$(du -b "$TARPATH" | awk '{print $1}')

info "DONE"
info "tarball:  $TARPATH ($TARSIZE, $TARSIZE_BYTES bytes)"
info "sha256:   $(cat "$OUT_DIR/$TARNAME.sha256")"
info "contents (full listing):"
tar -tzf "$TARPATH" | sed 's/^/  /'
