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
#   COMPAT_BUILD_DIR  the build directory of the lib-compat library: the SAME recipe with -DPXA_X86_COMPAT=ON (no AVX / AVX2 /
#                everything past SSE2 in the CPU side of libggml). If it already has ggml/src/libggml.so it is used as-is; if not,
#                this script builds it from REF in DEV_IMAGE (target ggml only; the GPU code is the same, so the build is mostly
#                CUDA compiles). Default: <BUILD_DIR>-compat. The package carries it as lib-compat/ next to the fast lib/; the
#                launchers pick it from /proc/cpuinfo on a CPU without AVX2 (tools/pxa-cpu-check.sh). A package WITHOUT it is a
#                package that dies with "Illegal instruction" at load on such a CPU, so a missing lib-compat is a hard error:
#                SKIP_COMPAT=1 is the explicit way to build one anyway (a rehearsal; never a release).
#   CCACHE_HOST_DIR  a ccache directory on the host, mounted into the build containers (CCACHE_DIR=/ccache there). Unset = no cache.
#   CMAKE_EXTRA  extra cmake arguments for BOTH builds (e.g. the CUDA stubs link flags a no-driver build container needs)
#   JOBS         parallel compile jobs for the builds this script starts (default 6)
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
#   llama-imatrix                                         required -- PXA Control's Encode tab runs it (with the
#                                                         PXQN_DUMP hook it carries) to dump the activations a PXQN
#                                                         encode needs. The open dump tool only: the PXQN ENCODER is
#                                                         closed and comes from the licence server, never from here.
#   llama-quantize llama-gguf-split                       shipped when they build
#   three self-test binaries                              required
# What lands next to bin/ and lib/:
#   lib-compat/libggml.so   the compatibility build of libggml (see COMPAT_BUILD_DIR); lib-compat/libggml-pxqn.so is a hard link
#                           to lib/'s (that library is plain x86-64 with a run-time AVX2 choice, so one file serves both)
#   tools/convert_hf_to_gguf.py + tools/gguf-py/   the Hugging Face -> GGUF converter and the gguf python package of THIS
#                           engine commit (the converter puts it first on sys.path, so a pip "gguf" never shadows it), and
#                           tools/requirements-convert.txt, the pinned python packages the converter imports. The Encode tab's
#                           tool checks look for exactly these (tools/pxa_encode.py find_convert / find_tool).
#
# Dry run: `scripts/make-release-tarball.sh --list-targets` prints the build targets and the bin
# lists and exits, touching no git, no docker and no build directory.
# `scripts/make-release-tarball.sh --selftest` checks the finished-tarball membership test on a
# small archive. It does not compile, and it does not need docker.
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
COMPAT_BUILD_DIR=${COMPAT_BUILD_DIR:-${BUILD_DIR%/}-compat}
SKIP_COMPAT=${SKIP_COMPAT:-}
CCACHE_HOST_DIR=${CCACHE_HOST_DIR:-}
CMAKE_EXTRA=${CMAKE_EXTRA:-}
JOBS=${JOBS:-6}
# LLGuidance (faster JSON-schema / tool grammars) is added by the self-build below when the
# build image has cargo. docker/Dockerfile.release-build installs the Rust toolchain for that.
# An image without cargo keeps the previous grammar engine, so an old image still packages.
# THE RELEASE RECIPE, in one place: the fast build and the lib-compat build differ ONLY in the CPU flags. GGML_NATIVE=OFF is
# not optional: ggml's default is -march=native of the machine that compiles, which puts that machine's AVX-512 (or anything
# else) into a package other people run. The stubs link flags let a container without a driver link the CUDA driver symbols.
RECIPE_COMMON="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\"$CUDA_ARCHS\" -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=2 -DPXA_PXQN_ENCODER=OFF -DPXA_LOCK_PUBKEYS=pxl-2026-10:e0cd04be16b884fabac90f4843d4a0590f3962b0b1d13d96aa5df70b9a160cb1 -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DCMAKE_EXE_LINKER_FLAGS=\"-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs\" -DCMAKE_SHARED_LINKER_FLAGS=\"-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs\""
RECIPE_FAST="$RECIPE_COMMON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
RECIPE_COMPAT="$RECIPE_COMMON -DPXA_X86_COMPAT=ON"
BUILD_TARGETS="llama-server llama-cli llama-bench llama-perplexity llama-quantize llama-pxq-export llama-imatrix llama-gguf-split test-pxq-cpu-dot test-kv-seq-shadow test-narrow-kernel-parity"

# The tarball's bin/ manifest. Kept here, next to BUILD_TARGETS, so the thing that gets BUILT and
# the thing that gets SHIPPED are read and changed together -- the v2026.09.07-rc1 tarball shipped
# without llama-pxq-export because these two lists lived 160 lines apart and only one of them was
# updated when the export tool landed.
REQUIRED_BIN="llama-server llama-cli llama-bench llama-perplexity"
# Required, but built on demand if an externally-supplied BUILD_DIR lacks it (a CPU-only link).
# A package without it cannot run the documented "leave PXQ" recipe, so a still-missing binary
# here is a hard error, not a warning.
REQUIRED_ONDEMAND_BIN="llama-pxq-export llama-imatrix"
TEST_BIN="test-pxq-cpu-dot test-kv-seq-shadow test-narrow-kernel-parity"
OPTIONAL_BIN="llama-quantize llama-gguf-split"

err()  { echo "make-release-tarball.sh: $*" >&2; }
die()  { err "$*"; exit 1; }
info() { echo "make-release-tarball.sh: $*"; }

# True when MEMBER is an exact line of the tarball listing.
# Write the listing to a file, then match. A quiet grep on a pipe is wrong in
# this file: pipefail is on, the quiet grep exits at the first hit, and the
# producer is still writing. The producer dies on SIGPIPE and pipefail reports
# failure even though the member is present. A short listing fits in the pipe
# and hides it. A release listing does not. Holding the listing in a shell
# variable and then printing it into a quiet grep has the same shape.
tar_has_member() {
  local list rc
  list=$(mktemp)
  if ! tar -tzf "$1" > "$list"; then
    rm -f "$list"
    return 1
  fi
  grep -qx -- "$2" "$list"
  rc=$?
  rm -f "$list"
  return "$rc"
}

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
    echo "fast recipe:            $RECIPE_FAST"
    echo "lib-compat recipe:      $RECIPE_COMPAT (target ggml)"
    echo "lib-compat build dir:   $COMPAT_BUILD_DIR (built here if absent)"
    echo "also staged:            tools/convert_hf_to_gguf.py tools/gguf-py/ tools/requirements-convert.txt tools/pxa-cpu-check.sh"
    exit 0
    ;;
  --selftest)
    # No git, no docker, no compile. The fixture is long enough that the old
    # pipeline reports a present member as missing; the real check must not.
    st=$(mktemp -d /tmp/pxq-tar-member-XXXXXX)
    trap 'rm -rf "$st"' EXIT
    python3 - "$st" <<'PY'
import os, sys
base = os.path.join(sys.argv[1], "tree", "pkg", "bin")
os.makedirs(base)
for i in range(8000):
    with open(os.path.join(base, "a%d" % i), "w") as f:
        f.write("x")
with open(os.path.join(base, "pxa-update"), "w") as f:
    f.write("x")
for i in range(8000):
    with open(os.path.join(base, "b%d" % i), "w") as f:
        f.write("x")
PY
    [ $? -eq 0 ] || die "selftest fixture failed"
    tar -C "$st/tree" -czf "$st/pkg.tar.gz" pkg
    member="pkg/bin/pxa-update"
    # Build the old pipeline without a source line the scanner below would flag.
    listcmd=(tar -tzf "$st/pkg.tar.gz")
    if "${listcmd[@]}" | grep -qx -- "$member"; then
      die "selftest fixture did not trip the pipefail miss; the guard did not run"
    fi
    tar_has_member "$st/pkg.tar.gz" "$member" || die "selftest: present member reported missing"
    if tar_has_member "$st/pkg.tar.gz" "pkg/bin/no-such"; then
      die "selftest: absent member reported present"
    fi
    if grep -nE 'tar -tzf[^|#]*\|[[:space:]]*grep -q' "$0" >/dev/null; then
      die "selftest: a tar listing is still judged with grep -q under pipefail"
    fi
    info "selftest ok"
    exit 0
    ;;
  -h|--help)
    sed -n '2,/^set -u$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
    echo "flags: --list-targets   print the build targets and the bin/ manifest, then exit"
    echo "       --selftest       check tarball membership on a small archive, then exit"
    exit 0
    ;;
  "") ;;
  *) die "unknown argument: $1 (this script is configured with the environment variables documented at the top of the file; the only flags are --list-targets, --selftest and --help)" ;;
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
CCACHE_ARGS=()
[ -z "$CCACHE_HOST_DIR" ] || CCACHE_ARGS=(-e CCACHE_DIR=/ccache -v "$CCACHE_HOST_DIR:/ccache")
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
  docker run --rm --name "pxq-pkg-selfbuild-$$" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none "${CCACHE_ARGS[@]}" \
    -v "$BUILD_SRC:/work" -v "$BUILD_DIR:/work/build-out" -w /work "$BUILD_IMAGE" \
    bash -c "LLG=; command -v cargo >/dev/null 2>&1 && LLG=-DLLAMA_LLGUIDANCE=ON; nice -n 19 cmake -B build-out -S . $RECIPE_FAST -DPXA_BUILD_NUMBER=$BI_NUMBER -DPXA_BUILD_COMMIT=$BI_COMMIT \$LLG $CMAKE_EXTRA && nice -n 19 cmake --build build-out -j$JOBS --target $BUILD_TARGETS" \
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

# ---------------------------------------------------------------------------------------------
# lib-compat: the SAME recipe with -DPXA_X86_COMPAT=ON, target ggml only (libggml.so is the one library whose CPU side is
# built with AVX / AVX2 / FMA / F16C; libllama, libmtmd and the binaries are plain x86-64 with run-time-guarded AVX2 clones,
# and libggml-pxqn.so is plain x86-64 with a run-time choice, so one copy of it serves both). Found 2026-10-04 (lane
# pxqn-anycpu): the shipped libggml.so carried 424741 VEX instructions and AVX in four static initialisers = SIGILL while the
# library loads, on ANY CPU without AVX2, for every binary. Built from REF in DEV_IMAGE: the compat library has to be the
# same commit and the same compiler as the fast one, or the symbol-table check below refuses it.
#
# ccache + nvcc (impl ccache-nvcc-host-flags-key): ccache's key for an nvcc compile does not see the host flags that arrive
# through -Xcompiler, so a compat CUDA object would be served the fast build's AVX2 object out of a shared cache. The cmake
# option puts a marker header into every compat CUDA translation unit (ggml/src/pxa-x86-compat.h); the objdump scan below is
# the proof that it worked, and the reason a cache can never smuggle an AVX2 object into lib-compat/.
# ---------------------------------------------------------------------------------------------
COMPAT_BUILD_DIR=$(realpath -m "$COMPAT_BUILD_DIR")
if [ -n "$SKIP_COMPAT" ]; then
  err "SKIP_COMPAT is set -- this package will have NO lib-compat/ and dies with 'Illegal instruction' on a CPU without AVX2. A rehearsal, never a release."
elif [ -f "$COMPAT_BUILD_DIR/ggml/src/libggml.so" ]; then
  info "COMPAT_BUILD_DIR=$COMPAT_BUILD_DIR already has ggml/src/libggml.so -- using it as-is, not building"
else
  info "building lib-compat (-DPXA_X86_COMPAT=ON, target ggml) from REF=$REF into $COMPAT_BUILD_DIR with $DEV_IMAGE"
  COMPAT_SRC="${BUILD_SRC:-}"
  if [ -z "$COMPAT_SRC" ] || [ ! -d "$COMPAT_SRC" ]; then
    COMPAT_SRC="$WORK/compat-src"
    mkdir -p "$COMPAT_SRC"
    git -C "$REPO_ROOT" archive "$COMMIT" | tar -x -C "$COMPAT_SRC"
  fi
  CBI_COMMIT=$(git -C "$REPO_ROOT" rev-parse --short "$COMMIT")
  CBI_NUMBER=$(git -C "$REPO_ROOT" rev-list --count "$COMMIT")
  mkdir -p "$COMPAT_BUILD_DIR"
  docker run --rm --name "pxq-pkg-compat-$$" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none "${CCACHE_ARGS[@]}" \
    -v "$COMPAT_SRC:/work" -v "$COMPAT_BUILD_DIR:/work/build-compat" -w /work --entrypoint bash "$DEV_IMAGE" \
    -c "nice -n 19 cmake -B build-compat -S . $RECIPE_COMPAT -DPXA_BUILD_NUMBER=$CBI_NUMBER -DPXA_BUILD_COMMIT=$CBI_COMMIT $CMAKE_EXTRA && nice -n 19 cmake --build build-compat -j$JOBS --target ggml" \
    || die "lib-compat build failed (see docker output above)"
  [ -f "$COMPAT_BUILD_DIR/ggml/src/libggml.so" ] || die "lib-compat build finished but $COMPAT_BUILD_DIR/ggml/src/libggml.so is missing"
fi
if [ -z "$SKIP_COMPAT" ]; then
  grep -q '^PXA_X86_COMPAT:BOOL=ON' "$COMPAT_BUILD_DIR/CMakeCache.txt" 2>/dev/null \
    || die "COMPAT_BUILD_DIR=$COMPAT_BUILD_DIR was not configured with -DPXA_X86_COMPAT=ON -- it is not a compat build"
fi

STAGE="$WORK/pxa-$TAG"
mkdir -p "$STAGE"/{bin,lib,docs,bench/fair,bench/gate}
[ -n "$SKIP_COMPAT" ] || mkdir -p "$STAGE/lib-compat"

# ---------------------------------------------------------------------------------------------
# export the docs/scripts/tools tree at REF (NOT the working tree — REF is the source of truth)
# ---------------------------------------------------------------------------------------------
EXPORT="$WORK/export"
mkdir -p "$EXPORT"
git -C "$REPO_ROOT" archive "$COMMIT" | tar -x -C "$EXPORT"

req() { [ -e "$EXPORT/$1" ] || die "REF $REF is missing required path: $1"; }
req tools/pxa-launch.py
req tools/pxa-cpu-check.sh
req scripts/pxa-isa-scan.py
req scripts/make-convert-requirements.sh
req convert_hf_to_gguf.py
req gguf-py/gguf/__init__.py
req requirements/requirements-convert_hf_to_gguf.txt
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

if [ -f "$EXPORT/RELEASE-NOTES-v3.1.md" ]; then
  RELNOTES=$EXPORT/RELEASE-NOTES-v3.1.md
else
  RELNOTES=$(ls "$EXPORT"/RELEASE-NOTES-*.md 2>/dev/null | sort | tail -1)
fi
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

# pxa-update is a host tool, not a CUDA target. Built from the archived tree, same commit as the rest.
# This runs for every ARCH_TAG, so both the Ubuntu 24.04 tarball and the Ubuntu 22.04 tarball ship it.
# The binary is not committed; the source is tools/pxa-update.c.
[ -f "$EXPORT/tools/pxa-update.c" ] || die "tools/pxa-update.c is missing; bin/pxa-update must be in every release tarball"
info "building pxa-update"
docker run --rm --name "pxq-pkg-update-$$" -v "$EXPORT":/src -v "$STAGE":/stage "$DEV_IMAGE" \
  gcc -O2 -o /stage/bin/pxa-update /src/tools/pxa-update.c \
  || die "could not build pxa-update"
[ -x "$STAGE/bin/pxa-update" ] || die "bin/pxa-update was not staged"

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
# the compatibility libggml (any x86-64 CPU), next to the fast one; its libggml-pxqn.so is linked in AFTER patchelf and strip
# below, so the two names stay one file
if [ -z "$SKIP_COMPAT" ]; then
  cp -a "$COMPAT_BUILD_DIR/ggml/src/libggml.so" "$STAGE/lib-compat/libggml.so"
fi
# the closed PXQN library: a build configured with it MUST ship it, or PXQN files are refused
if grep -q '^PXA_PXQN_CLOSED_SRC:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
  [ -f "$STAGE/lib/libggml-pxqn.so" ] || die "PXA_PXQN_CLOSED_SRC=ON build but no ggml/src/libggml-pxqn.so to ship (build target ggml-pxqn)"
  info "shipping lib/libggml-pxqn.so (closed PXQN library, loaded by libggml from lib/)"
fi

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
    # HARD LINK, not `cp -a` (found 2026-09-27): `cp -a` on a single source file
    # makes an independent copy on every filesystem this box has (verified directly: ls -li
    # shows two different inodes) -- it does NOT reuse the source's data the way `--preserve=
    # links` does for a set of already-linked files. That silently doubled every large runtime
    # lib this loop touches inside the tarball: libcublasLt.so.12 (751 MB) shipped as two full
    # 751 MB entries instead of one, same for libnccl.so.2 (383 MB) and libcublas.so.12 (116 MB)
    # -- about 1.25 GB of pure duplication, most of why this release's tarball measured 2.1+ GB
    # against v2026.09.20's 1.31 GB for a comparable payload. The shipped v2026.09.20 artifact
    # itself (<assets>/pxa-v2026.09.20-...tar.gz) carries these same
    # pairs as genuine hard links (`tar -tzv` shows the "h...  0 bytes  link to ..." form) --
    # confirming a hard link, not a second copy, is the intended shape here. `ln -f` also means
    # patchelf/strip touching one name transparently updates the other, as long as they edit in
    # place rather than replace-via-rename (not exercised by this box's current DEV_IMAGE, which
    # has neither patchelf nor anything to strip -- re-check this note if that ever changes).
    ln -f "$STAGE/lib/$realname" "$STAGE/lib/$base"
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
    # lib-compat/libggml.so finds the CUDA / OpenMP runtime libs in ../lib, whichever way it is started
    for f in /stage/lib-compat/*.so*; do
      [ -f "$f" ] || continue
      readelf -h "$f" >/dev/null 2>&1 || continue
      patchelf --set-rpath "\$ORIGIN/../lib" "$f" 2>/dev/null || true
    done
  '
else
  info "patchelf NOT found in $DEV_IMAGE — binaries rely on the wrapper scripts' LD_LIBRARY_PATH (documented in START-HERE.md)"
fi

# ---------------------------------------------------------------------------------------------
# strip debug symbols if present and it actually shrinks things
# ---------------------------------------------------------------------------------------------
SIZE_BEFORE_STRIP=$(du -sb "$STAGE/bin" "$STAGE/lib" "$STAGE/lib-compat" 2>/dev/null | awk '{s+=$1} END{print s}')
# NOTE: this image has no `file` binary — detect ELF-ness and debug sections with readelf.
HAS_DEBUG=$(dimg -v "$STAGE:/stage" -- '
  found=0
  for f in /stage/bin/* /stage/lib/*.so* /stage/lib-compat/*.so*; do
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
    for f in /stage/lib/*.so* /stage/lib-compat/*.so*; do [ -f "$f" ] && strip --strip-unneeded "$f" 2>/dev/null; done
  '
else
  info "no debug symbols found in bin/lib — nothing to strip"
fi
SIZE_AFTER_STRIP=$(du -sb "$STAGE/bin" "$STAGE/lib" "$STAGE/lib-compat" 2>/dev/null | awk '{s+=$1} END{print s}')
info "bin/+lib/ size before strip: $SIZE_BEFORE_STRIP bytes, after: $SIZE_AFTER_STRIP bytes"

# ---------------------------------------------------------------------------------------------
# lib-compat guards. Run on the STAGED, patched, stripped files -- what the user gets -- not on the build tree.
#  1. lib-compat/libggml.so has no AVX-class instruction at all (VEX, FMA, F16C, ymm, zmm): scripts/pxa-isa-scan.py on its
#     objdump. This is the proof that neither the cmake flags nor a ccache hit put an AVX2 object into it.
#  2. its exported STRONG symbols (types T D B R: ggml's own ABI; weak ones are libstdc++ template instances the inliner kept
#     in one build and not in the other, e.g. std::operator+) and its soname are those of the fast lib/libggml.so, so libllama,
#     libmtmd, the binaries and libggml-pxqn.so bind to either one.
#  3. lib-compat/libggml-pxqn.so is lib/'s (a hard link): that library is plain x86-64 with a run-time AVX2 choice.
# ---------------------------------------------------------------------------------------------
COMPAT_SCAN="(no lib-compat in this package: SKIP_COMPAT was set)"
if [ -z "$SKIP_COMPAT" ]; then
  [ -f "$STAGE/lib-compat/libggml.so" ] || die "lib-compat/libggml.so is missing from the stage"
  COMPAT_SCAN=$(dimg -v "$STAGE:/stage:ro" -- 'objdump -d -w --no-show-raw-insn -M intel /stage/lib-compat/libggml.so' \
    | python3 "$EXPORT/scripts/pxa-isa-scan.py" --name lib-compat/libggml.so --require-baseline) \
    || die "lib-compat/libggml.so is not baseline x86-64 -- refusing to package:
$COMPAT_SCAN"
  info "lib-compat scan: $(echo "$COMPAT_SCAN" | sed -n 2p | sed 's/^ *//')"
  SYMDIFF=$(dimg -v "$STAGE:/stage:ro" -- '
    nm -D --defined-only /stage/lib/libggml.so | awk "\$2 ~ /^[TDBR]\$/ {print \$3}" | sort > /tmp/fast.syms
    nm -D --defined-only /stage/lib-compat/libggml.so | awk "\$2 ~ /^[TDBR]\$/ {print \$3}" | sort > /tmp/compat.syms
    diff /tmp/fast.syms /tmp/compat.syms | head -20
    echo "soname fast:   $(readelf -d /stage/lib/libggml.so | grep -o "SONAME.*")"
    echo "soname compat: $(readelf -d /stage/lib-compat/libggml.so | grep -o "SONAME.*")"
    echo "exported strong symbols fast=$(wc -l < /tmp/fast.syms) compat=$(wc -l < /tmp/compat.syms)"') \
    || die "could not compare the symbol tables of lib/ and lib-compat/"
  if echo "$SYMDIFF" | grep -q -E '^[<>]'; then
    die "lib-compat/libggml.so and lib/libggml.so export different symbols -- built from different commits or options; refusing to package:
$SYMDIFF"
  fi
  [ "$(echo "$SYMDIFF" | grep '^soname fast:' | sed 's/^soname fast: *//')" = "$(echo "$SYMDIFF" | grep '^soname compat:' | sed 's/^soname compat: *//')" ] \
    || die "lib-compat/libggml.so has a different soname than lib/libggml.so:
$SYMDIFF"
  info "lib-compat symbols: $(echo "$SYMDIFF" | grep '^exported strong symbols')"
  if [ -f "$STAGE/lib/libggml-pxqn.so" ]; then
    ln -f "$STAGE/lib/libggml-pxqn.so" "$STAGE/lib-compat/libggml-pxqn.so"
  fi
fi

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
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)
export PXA_ENGINE_DIR="$HERE"
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# A CPU without AVX2 / FMA / F16C gets lib-compat/ first on the path (one plain line says so): the fast lib/ would stop with
# "Illegal instruction" while it loads. tools/pxa-launch.py makes the same choice for the servers it starts itself.
if [ -r "$HERE/tools/pxa-cpu-check.sh" ]; then
  . "$HERE/tools/pxa-cpu-check.sh"
  pxa_cpu_pick "$HERE/lib-compat" || exit 1
fi
exec python3 "$HERE/tools/pxa-launch.py" "$@"
WRAP
chmod +x "$STAGE/pxa-launch"

cat > "$STAGE/run-server.sh" <<'WRAP'
#!/bin/bash
# run-server.sh — run llama-server directly (no auto-tuning) with this package's ./lib set up.
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# A CPU without AVX2 / FMA / F16C gets lib-compat/ first on the path (one plain line says so): the fast lib/ would stop with
# "Illegal instruction" while it loads. PXA_CPU_LIB=fast|compat forces a choice.
if [ -r "$HERE/tools/pxa-cpu-check.sh" ]; then
  . "$HERE/tools/pxa-cpu-check.sh"
  pxa_cpu_pick "$HERE/lib-compat" || exit 1
fi
# Card numbers should mean the same thing here as they do in nvidia-smi. CUDA's own
# default is CUDA_DEVICE_ORDER=FASTEST_FIRST, which sorts the cards by compute
# capability, while nvidia-smi numbers them by PCI bus id — so on a box with mixed
# cards, CUDA_VISIBLE_DEVICES=0 starts on a different GPU than the one `nvidia-smi -L`
# calls 0, silently. PCI_BUS_ID makes the two numberings one numbering. Yours wins if
# you set it: this only fills in a default.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# PXA Control next to the server: started from a terminal, this also opens a browser page on
# 127.0.0.1 (port 7777, or the next free one) that shows this server live and can stop it
# (docs/LAUNCHER.md, "PXA Control opens by itself"). --no-control (taken out here: the engine never
# sees it) or PXA_CONTROL=0 skips it; a service or a script (no terminal) gets it only with PXA_CONTROL=1.
pxa_control=1
for arg in "$@"; do [ "$arg" = "--no-control" ] && pxa_control=0; done
if [ "$pxa_control" = 0 ]; then
  args=()
  for arg in "$@"; do [ "$arg" = "--no-control" ] || args+=("$arg"); done
  set -- "${args[@]}"
fi
if [ "$pxa_control" = 1 ] && [ "${PXA_CONTROL:-}" != "0" ] && [ -f "$HERE/tools/pxa-launch.py" ] \
   && command -v python3 >/dev/null 2>&1; then
  PXA_ENGINE_DIR="${PXA_ENGINE_DIR:-$HERE}" python3 "$HERE/tools/pxa-launch.py" --control-for-pid $$ || true
fi
exec "$HERE/bin/llama-server" "$@"
WRAP
chmod +x "$STAGE/run-server.sh"
# `pxa` is the short name of pxa-launch: with no arguments at a terminal it opens PXA Control in the
# browser (the text menu is `pxa --tui`); with a model and cards it starts that server.
ln -sf pxa-launch "$STAGE/pxa"

mkdir -p "$STAGE/tools"
cp -a "$EXPORT/tools/pxa-launch.py" "$STAGE/tools/pxa-launch.py"
chmod +x "$STAGE/tools/pxa-launch.py"
# the CPU preflight the wrappers source (lib/ or lib-compat/); the Python twin is inside pxa-launch.py
cp -a "$EXPORT/tools/pxa-cpu-check.sh" "$STAGE/tools/pxa-cpu-check.sh"
# The source file is mode 0666. cp -a keeps that, and a package script the
# entrypoint executes must be executable.
chmod +x "$STAGE/tools/pxa-cpu-check.sh"
[ -s "$STAGE/tools/pxa-cpu-check.sh" ] || die "tools/pxa-cpu-check.sh did not survive staging"
[ -x "$STAGE/tools/pxa-cpu-check.sh" ] || die "tools/pxa-cpu-check.sh is not executable"

# ---------------------------------------------------------------------------------------------
# The Encode tab's open tools. PXA Control's Encode tab (tools/pxa_encode.py) looks for them in the install, in this order:
#   find_convert(): <engine dir>/convert_hf_to_gguf.py, <engine dir>/tools/convert_hf_to_gguf.py, <tools dir>/../convert_hf_to_gguf.py
#   find_tool():    <engine dir>/bin/<name>, <engine dir>/<name>, then PATH
# bin/llama-imatrix (the activation dump; it carries the PXQN_DUMP hook) is staged with the other binaries above.
# The converter is tools/convert_hf_to_gguf.py and the gguf python package it imports is tools/gguf-py/, taken from the SAME
# commit: the converter puts <its own dir>/gguf-py first on sys.path (unless NO_LOCAL_GGUF is set), so a "gguf" from pip -- whose
# tensor and type constants may be older or newer than this engine's -- never shadows it. That is the pin.
# tools/requirements-convert.txt lists the python packages the converter imports (scripts/make-convert-requirements.sh, the one
# renderer the container image uses too): the repo's own requirement lines, minus the "gguf" line (the bundled gguf-py replaces
# it), plus safetensors (named by the Encode tab's own check).
# NOT here, on purpose: the PXQN ENCODER. It is closed and comes from the licence server ("Get the encoder" in the Encode tab).
# Only the open dump and convert tools travel with the engine.
# ---------------------------------------------------------------------------------------------
cp -a "$EXPORT/convert_hf_to_gguf.py" "$STAGE/tools/convert_hf_to_gguf.py"
mkdir -p "$STAGE/tools/gguf-py"
cp -a "$EXPORT/gguf-py/gguf" "$STAGE/tools/gguf-py/gguf"
cp -a "$EXPORT/gguf-py/LICENSE" "$STAGE/tools/gguf-py/LICENSE" 2>/dev/null || true
find "$STAGE/tools/gguf-py" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
[ -f "$STAGE/tools/gguf-py/gguf/__init__.py" ] && [ -s "$STAGE/tools/convert_hf_to_gguf.py" ] || die "the converter did not survive staging"
sh "$EXPORT/scripts/make-convert-requirements.sh" "$EXPORT/requirements" > "$STAGE/tools/requirements-convert.txt" \
  || die "could not render tools/requirements-convert.txt (scripts/make-convert-requirements.sh)"
# PXA Control (pxa-launch --gui): the web front end, its page, the bench prompts it reuses and the
# lever catalog its Advanced panel validates against (found at tools/../common/ by pxa_control.py).
for f in pxa_control.py pxa_telemetry.py pxa_mqtt.py pxa-bench.py pxa_explain_footer.py pxa_thinking.py pxa_thinking_profiles.json pxa_encode.py pxa_encode_adapter.py pxa_encode_pkg.py pxa_encode_plan.py pxa_encode_cells.json; do
  if [ -f "$EXPORT/tools/$f" ]; then cp -a "$EXPORT/tools/$f" "$STAGE/tools/$f"; fi
done
# `pxa-update lib check|apply|rollback` hands over to tools/pxa_lib_update.py, which verifies the signed library release
# with tools/pxa_encode_pkg.py's Ed25519 verifier: BOTH must be in every release tarball. The loop above skips a missing
# file in silence, which here would ship a `lib` subcommand that cannot run - so these two are fatal.
#
# tools/pxa_expert_map.py is imported by PXA Control's Expert map routes (tools/pxa_control.py
# r_expert_map / r_expert_map_rebuild / r_expert_map_reset). The loop above skips a missing file in
# silence, so a package without it would build cleanly and answer every Expert map request with
# ModuleNotFoundError -- found 2026-10-09 by calling the route from a layout staged exactly as this
# script stages it. Fatal here, for the same reason as the two below.
#
# tools/pxa_mqtt.py is the Home Assistant publisher PXA Control imports (guarded, so the module may be
# absent, but every release carries it: the HA settings panel is part of the shipped page -- main's rule
# is that a new tools file is named in BOTH lists, this fatal one and the loop above).
for f in pxa_expert_map.py pxa_mqtt.py pxa_lib_update.py pxa_encode_pkg.py; do
  cp -a "$EXPORT/tools/$f" "$STAGE/tools/$f" || die "tools/$f is missing from the build tree; a shipped tool imports it"
done
[ -s "$STAGE/tools/pxa_lib_update.py" ] && [ -s "$STAGE/tools/pxa_encode_pkg.py" ] || die "the library updater did not survive staging"
if [ -d "$EXPORT/tools/pxa_control_ui" ]; then
  cp -a "$EXPORT/tools/pxa_control_ui" "$STAGE/tools/pxa_control_ui"
fi
if [ -d "$EXPORT/tools/pxa_ctl" ]; then        # the Profiles tab's backend package
  mkdir -p "$STAGE/tools/pxa_ctl" && cp -a "$EXPORT/tools/pxa_ctl/"*.py "$STAGE/tools/pxa_ctl/"
fi
# the Chat tab's in-house agent (tools/pxa_chat: stdlib package, optional import)
if [ -d "$EXPORT/tools/pxa_chat" ]; then
  cp -a "$EXPORT/tools/pxa_chat" "$STAGE/tools/pxa_chat"
  find "$STAGE/tools/pxa_chat" -name __pycache__ -prune -exec rm -rf {} +
fi
# The shipped catalog is the PUBLIC table (owner 2026-10-08: no closed rows in what ships), the one the engine embeds:
# a REF with the closed sources goes through the release filter, and then the catalog copy and every shipped binary but
# the closed library itself are checked for closed lever names (a BUILD_DIR configured with PXA_LEVER_CATALOG_FULL=ON,
# or built before the filter existed, stops here).
if [ -f "$EXPORT/common/pxa-lever-catalog.inc" ]; then
  mkdir -p "$STAGE/common"
  if [ -f "$EXPORT/scripts/pxa-closed-levers.txt" ]; then
    python3 "$EXPORT/scripts/pxa-lever-catalog.py" --release-filter "$STAGE/common/pxa-lever-catalog.inc" \
      --closed-out "$WORK/pxqn-closed-levers.inc" || die "lever catalog release filter failed (scripts/pxa-lever-catalog.py)"
    CLEAN_FILES=("$STAGE/common/pxa-lever-catalog.inc")
    while IFS= read -r -d '' f; do CLEAN_FILES+=("$f"); done < <(find "$STAGE/bin" "$STAGE/lib" "$STAGE/lib-compat" \
      -type f ! -name 'libggml-pxqn.so*' \( -perm -u+x -o -name '*.so*' \) -print0 2>/dev/null)
    python3 "$EXPORT/scripts/pxa-lever-catalog.py" --assert-clean "$WORK/pxqn-closed-levers.inc" "${CLEAN_FILES[@]}" \
      || die "closed lever names in the package (above): rebuild BUILD_DIR from REF without PXA_LEVER_CATALOG_FULL"
  else
    cp -a "$EXPORT/common/pxa-lever-catalog.inc" "$STAGE/common/"
  fi
fi

# pxa-entrypoint: the container ENTRYPOINT (no args -> pxa-launch; engine args -> straight through
# to run-server.sh; `doctor` -> pxa-launch --doctor). Harmless outside a container.
if [ -f "$EXPORT/tools/pxa-entrypoint.sh" ]; then
  cp -a "$EXPORT/tools/pxa-entrypoint.sh" "$STAGE/pxa-entrypoint"
  chmod +x "$STAGE/pxa-entrypoint"
fi
# The PXA names: pxa-server, pxa-bench, pxa-quantize, pxa-perplexity, pxa-cli are symlinks to the
# llama-* binaries, which keep working under their old names.
for t in server bench quantize perplexity cli; do
  [ -x "$STAGE/bin/llama-$t" ] && ln -sf "llama-$t" "$STAGE/bin/pxa-$t"
done

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

# The Overdrive preset (main #16694, 2026-10-09).  docs/COOKBOOK.md's "1x P100 -- Flash-Next Overdrive"
# recipe tells the user to load this file as the server environment, so a package without it ships a
# recipe whose first line names nothing.  Kept by EXACT PATH, not by a presets/* glob: the file names the
# closed PXQN levers verbatim, and the gates that sweep shipped text for those names
# (post-build-gates.sh gate 6, check-public-tree-v31.sh rule f) allow this one path and nothing else.
# The public TREE omits it (PUBLIC-OMIT-v31.txt); the package is where a user needs it.
keep presets/pxa-overdrive-flashnext-pxqn2-1xp100.env

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
keep_doc docs/LEVERS.md
keep_doc docs/lab/LEVERS.md
keep_doc docs/DEFAULTS.md
# Named by the dead-link guard (2026-10-08): docs/LAUNCHER.md points at docs/THINKING.md.
keep_doc docs/THINKING.md
# Named by the README (v3.1): the Home Assistant REST sensors and the MQTT page.
keep_doc docs/HOME-ASSISTANT.md
# PXQN ships compiled-only: the shipped lever docs are the public page, never the lab table
python3 "$EXPORT/scripts/pxa-thin-pxqn-docs.py" "$STAGE" --allow-missing || die "PXQN doc thinning failed (scripts/pxa-thin-pxqn-docs.py)"
keep_doc docs/QUANTIZING.md
keep_doc docs/PXQU-CONVERT.md
keep_doc docs/PXA-SM70-SERVING.md
keep_doc BUILD-FROM-SOURCE.md
keep_doc bench/README.md
keep_doc RELEASE-NOTES-2026-09-02.md
keep_doc RELEASE-NOTES-2026-09-07.md
keep_doc RELEASE-NOTES-2026-09-09.md
# RELEASE-NOTES-2026-09-13.md used to be staged only via the $RELNOTES auto-pick (it sorted
# last). Adding RELEASE-NOTES-2026-09-20.md below (2026-09-27) makes 09-20 sort
# last instead, so 09-13 needs its own explicit line now, same as 09-02/09-07/09-09 above --
# found by this script's own dead-link guard the moment 09-20 was added, exactly the "next ring"
# pattern this file's comments already describe.
keep_doc RELEASE-NOTES-2026-09-13.md

# Added 2026-09-14, again named by the dead-link guard: the 09-13 notes and docs/COOKBOOK.md cite the
# leaderboard for every cross-engine cell and every pending re-measure. The board has no outbound
# links of its own, so shipping it closes the class without opening another one, and the numbers in
# the notes stay traceable to the evidence inside the package rather than to a page on the web.
keep_doc bench/LEADERBOARD.md

# Added 2026-09-14, named by the guard below once it learned to read <img> tags: every shipped page
# opens with the banner as an HTML <img>, the README embeds the fair-battle chart, and the 09-07
# notes embed their two remaining dark chart variants. Images are copied whole, never merged.
keep_doc docs/assets/pxa-network-banner.png
keep_doc docs/assets/pxa-control-launch-dark.png
keep_doc docs/assets/pxa-control-chat-phone-light.png
# PXA v3 front page images (README.md and RELEASE-NOTES-v3.md embed them)
keep_doc docs/img/pxa-v3-onecard.png
keep_doc docs/img/pxa-v3-more.png
keep_doc docs/img/pxa-quality-vs-size.png
keep_doc docs/img/pxa-control-live.png
keep_doc docs/img/pxa-control-speed.png
keep_doc docs/assets/before-after-2026-09-07-dark.png
keep_doc docs/assets/home-spec-ladder-2026-09-07-dark.png
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
keep_doc docs/PXQ4-KERNELS.md   # was pxa/pxq4/README.md: a pxa/ directory in the tarball would collide with the ./pxa launcher

# Added 2026-09-27, named by the dead-link guard on
# the v2026.10 docs pass: README.md picked up five new cross-references (MODELS.md,
# RELEASE-NOTES-2026-09-20.md, docker/COMPOSE.md, docs/tutorials/00-start-here.md,
# docs/tutorials/README.md) plus one older one this script never staged (NOTICE), none of which
# were in REF before this pass. MODELS.md is new content (no historical version existed to port
# forward). RELEASE-NOTES-2026-09-20.md, docker/COMPOSE.md (+ its docker-compose.yml/.env.example
# companions) and the full docs/tutorials/ set (12 numbered guides + GLOSSARY.md + README.md) are
# ported forward from release/rc4/public-staging/ (the vetted public set actually used for
# v2026.09.20) with only the version-pinned image tag bumped (v2026.09.20-rc4 -> v2026.10) --
# not re-verified against v2026.10 line by line, since the last release already vetted them.
keep_doc NOTICE
keep_doc MODELS.md
keep_doc RELEASE-NOTES-2026-09-20.md
# Named by the dead-link guard (2026-10-08): README.md still links the v3 notes,
# and the v3.1 picker no longer auto-stages RELEASE-NOTES-v3.md.
keep_doc RELEASE-NOTES-v3.md
keep_doc docker/COMPOSE.md
keep_doc docker/docker-compose.yml
keep_doc docker/.env.example
keep_doc docs/tutorials/README.md
keep_doc docs/tutorials/GLOSSARY.md
keep_doc docs/tutorials/00-start-here.md
keep_doc docs/tutorials/01-run-your-first-model.md
keep_doc docs/tutorials/02-pick-settings-for-your-cards.md
keep_doc docs/tutorials/03-going-faster.md
keep_doc docs/tutorials/04-quantize-your-own-model.md
keep_doc docs/tutorials/05-quantize-for-vllm.md
keep_doc docs/tutorials/06-run-the-vllm-sidecar.md
keep_doc docs/tutorials/07-docker-compose.md
keep_doc docs/tutorials/08-long-chats.md
keep_doc docs/tutorials/09-measure-your-card.md
keep_doc docs/tutorials/10-when-something-goes-wrong.md
keep_doc docs/tutorials/11-switching-models-from-your-app.md
# Grafana import for the Control /metrics gauges. The package check looks
# for this file before it contacts Grafana.
keep_doc docs/grafana/pxa-control.json

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
    -e 's#\]\(MODELS\.md\)#](../MODELS.md)#g' \
    -e 's#\]\(NOTICE\)#](../NOTICE)#g' \
    -e 's#\]\(LICENSE\)#](../LICENSE)#g' \
    -e 's#\]\(LICENSING\.md\)#](../LICENSING.md)#g' \
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
BUILD_FLAGS=$(grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|GGML_CUDA|GGML_SCHED_MAX_COPIES|GGML_NATIVE|GGML_AVX|GGML_AVX2|GGML_FMA|GGML_F16C)[:=]' \
  "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | grep -v -- '-STRINGS:' || true)
[ -n "$BUILD_FLAGS" ] || BUILD_FLAGS="(CMakeCache.txt not found or had none of the expected keys)"
COMPAT_FLAGS="(no lib-compat in this package)"
if [ -z "$SKIP_COMPAT" ]; then
  COMPAT_FLAGS=$(grep -E '^(PXA_X86_COMPAT|GGML_NATIVE|GGML_AVX|GGML_AVX2|GGML_FMA|GGML_F16C)[:=]' "$COMPAT_BUILD_DIR/CMakeCache.txt" 2>/dev/null | grep -v -- '-STRINGS:' || true)
fi
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
  "objdump -T /stage/bin/* /stage/lib/*.so* /stage/lib-compat/*.so* 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1" \
  2>/dev/null)
GLIBCXX_FLOOR=$(dimg -v "$STAGE:/stage" -- \
  "objdump -T /stage/bin/* /stage/lib/*.so* /stage/lib-compat/*.so* 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -1" \
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
cpu_libs:         lib/ = fast (AVX, AVX2, FMA, F16C); lib-compat/ = plain x86-64 (picked by the launchers from /proc/cpuinfo)
lib_compat_scan:  $(echo "$COMPAT_SCAN" | sed -n 2,3p | sed 's/^ *//' | paste -sd';' -)
cmake_flags:
$BUILD_FLAGS
cmake_flags_lib_compat:
$COMPAT_FLAGS
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
# STAGED-IMPORTS GUARD -- refuse to TAR a package whose own python files import a module it does
# not carry
# ---------------------------------------------------------------------------------------------
# Same shape as the dead-link guard above, and the same lesson one level down: the tool lists in
# this script and the COPY lines in docker/Dockerfile are both explicit, and BOTH skip a file that
# was never named -- in silence. Found 2026-10-09: tools/pxa_expert_map.py is imported by the Expert
# map routes in tools/pxa_control.py (r_expert_map / r_expert_map_rebuild / r_expert_map_reset) and
# is named in neither list, so a package built from this tree would answer every Expert map request
# with ModuleNotFoundError -- a headline v3.1 feature, with measured numbers in the release notes.
# The check reads the TREE to learn what a module name means and the STAGE to see whether it
# travelled (a third-party or optional module resolves in neither and is left alone).
python3 "$EXPORT/scripts/pxa-staged-imports.py" "$STAGE" --tree "$EXPORT" \
  || die "staged package drops a module its own tools import (above) -- name it in the tool lists in
       this script AND on the docker/Dockerfile COPY line, or the image and the tarball will differ"

# The IMAGE is a second package with its own explicit COPY list, and it carries its own copy of
# PXA Control under /usr/local/bin. Check it too: the tarball and the image must not drift, and the
# tarball is the artifact whose build turns first.
if [ -f "$EXPORT/docker/Dockerfile" ]; then
  python3 "$EXPORT/scripts/pxa-staged-imports.py" --image "$EXPORT/docker/Dockerfile" --tree "$EXPORT" \
    || die "the container image COPY list drops a module its own tools import (above)"
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
# tar_has_member reads the whole listing first. A quiet grep on the listing pipe
# used to report bin/pxa-update missing when the member was in the archive.
tar_has_member "$TARPATH" "pxa-${TAG}/bin/pxa-update" \
  || die "tarball $TARNAME is missing bin/pxa-update"
tar_has_member "$TARPATH" "pxa-${TAG}/tools/pxa_lib_update.py" \
  || die "tarball $TARNAME is missing tools/pxa_lib_update.py (pxa-update lib)"
tar_has_member "$TARPATH" "pxa-${TAG}/docs/grafana/pxa-control.json" \
  || die "tarball $TARNAME is missing docs/grafana/pxa-control.json"
( cd "$OUT_DIR" && sha256sum "$TARNAME" > "$TARNAME.sha256" )

TARSIZE=$(du -h "$TARPATH" | awk '{print $1}')
TARSIZE_BYTES=$(du -b "$TARPATH" | awk '{print $1}')

info "DONE"
info "tarball:  $TARPATH ($TARSIZE, $TARSIZE_BYTES bytes)"
info "sha256:   $(cat "$OUT_DIR/$TARNAME.sha256")"
info "contents (full listing):"
tar -tzf "$TARPATH" | sed 's/^/  /'
