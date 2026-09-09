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
BUILD_TARGETS="llama-server llama-cli llama-bench llama-perplexity llama-quantize llama-pxq-export llama-gguf-split test-pxq-cpu-dot test-kv-seq-shadow test-narrow-kernel-parity"

# The tarball's bin/ manifest. Kept here, next to BUILD_TARGETS, so the thing that gets BUILT and
# the thing that gets SHIPPED are read and changed together -- the v2026.09.07-rc1 tarball shipped
# without llama-pxq-export because these two lists lived 160 lines apart and only one of them was
# updated when the export tool landed.
REQUIRED_BIN="llama-server llama-cli llama-bench llama-perplexity"
# Required, but built on demand if an externally-supplied BUILD_DIR lacks it (a CPU-only link).
# A package without it cannot run the documented "leave PXQ" recipe, so a still-missing binary
# here is a hard error, not a warning.
REQUIRED_ONDEMAND_BIN="llama-pxq-export"
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
trap 'rm -rf "$WORK"' EXIT

# A cmake --build against THIS BUILD_DIR still running would make the binaries we're about to
# copy a moving target. Best-effort guard: look at ACTUAL "cmake" processes (pgrep -x, exact
# binary name) rather than pgrep -f "cmake --build", which also matches any unrelated shell
# whose command-line text happens to quote/echo/monitor that phrase (a real false-positive seen
# in practice — a leftover `until ... grep -q "cmake --build"` polling loop matched itself).
#
# This box runs many worktrees that all name their build dir "build-spd" (or "build"), so a
# basename-only match on cmdline also false-positives across worktrees (seen in practice: a
# different lane's "cmake --build build-spd" matched ours by basename alone). Disambiguate by
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
# versioned file" this order asks for) and the soname the binaries actually DT_NEEDED-reference,
# the latter as a copy (not a symlink — tar/untar-by-hand on an unfamiliar box is more likely to
# preserve plain files correctly than symlinks).
while read -r libpath; do
  [ -n "$libpath" ] || continue
  base=$(basename "$libpath")
  realname=$(dimg -v "$BUILD_DIR:/build:ro" -- "readlink -f '$libpath' | xargs basename" 2>/dev/null)
  [ -n "$realname" ] || { err "WARNING: could not resolve real file for $libpath"; continue; }
  dimg -v "$BUILD_DIR:/build:ro" -- "cat \$(readlink -f '$libpath')" > "$STAGE/lib/$realname" 2>/dev/null \
    || { err "WARNING: could not copy $libpath"; continue; }
  [ -s "$STAGE/lib/$realname" ] || { err "WARNING: copy of $libpath produced an empty file"; rm -f "$STAGE/lib/$realname"; continue; }
  if [ "$base" != "$realname" ]; then
    cp -a "$STAGE/lib/$realname" "$STAGE/lib/$base"
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

# ---------------------------------------------------------------------------------------------
# bench/fair, bench/gate, docs, license
# ---------------------------------------------------------------------------------------------
cp -a "$EXPORT/bench/fair/." "$STAGE/bench/fair/"
cp -a "$EXPORT/bench/gate/." "$STAGE/bench/gate/"
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

cat > "$STAGE/VERSION" <<EOF
tag:              $TAG
commit:           $COMMIT
ref:              $REF
built_from_dir:   $BUILD_DIR
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
