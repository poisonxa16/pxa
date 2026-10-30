#!/bin/bash
# The release image (docker/Dockerfile) builds LLGuidance (-DLLAMA_LLGUIDANCE=ON) and keeps Rust out of the runtime.
#
#   (default) static   docker/Dockerfile: the toolchain stage installs a PINNED Rust (rustup-init version + sha256 +
#                      toolchain version, minimal profile); the build stage passes -DLLAMA_LLGUIDANCE, default ON, and
#                      checks llama-server links it and --strip-debugs the executables; the runtime stage has no rustup/cargo/RUSTUP_HOME/CARGO_HOME and copies
#                      nothing from /opt/cargo or /opt/rustup. common/CMakeLists.txt: llguidance at a pinned commit, built
#                      with `cargo build --release --locked --package llguidance`. docker/Dockerfile.release-build: pinned.
#   --build [DIR]      CPU-only (no CUDA) cmake build with LLAMA_LLGUIDANCE=ON into DIR (default build-llg), then ctest
#                      test-grammar-llguidance (uses models/ggml-vocab-llama-bpe.gguf, a vocab-only file in the repo; no
#                      model weights), and llama-server must carry the llg_* symbols. Needs cargo on PATH and network
#                      access to github.com + crates.io for the first build (the crate is fetched by cmake).
#   --docker           docker build --target toolchain (no engine compile, no nvcc), with LLAMA_LLGUIDANCE=ON and OFF:
#                      cargo/rustc must run in the ON image and be absent in the OFF one; prints both image sizes.
#
# usage: tests/test-llguidance-image.sh [--build [DIR] | --docker]     exit 0 = PASS, 1 = FAIL, 77 = SKIP
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DF="$ROOT/docker/Dockerfile"
RB="$ROOT/docker/Dockerfile.release-build"
CM="$ROOT/common/CMakeLists.txt"
fails=0; passes=0
ok()  { echo "PASS  $*"; passes=$((passes+1)); }
bad() { echo "FAIL  $*"; fails=$((fails+1)); }
check() { local what="$1"; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }
finish() { echo "--- $passes passed, $fails failed"; [ "$fails" = 0 ]; exit $?; }

# stage NAME -> the lines of that stage (FROM ... AS NAME up to the next FROM)
stage() { awk -v n="$1" 'BEGIN{IGNORECASE=1} /^FROM /{on=(tolower($NF)==tolower(n))} on' "$DF"; }

static_checks() {
  local tc build rt
  tc="$(stage toolchain)"; build="$(stage build)"; rt="$(stage runtime)"
  check "Dockerfile has toolchain, build, runtime stages" test -n "$tc" -a -n "$build" -a -n "$rt"
  check "build stage is FROM toolchain" grep -qiE '^FROM toolchain AS build$' "$DF"
  check "toolchain: rustup-init version pinned"   grep -qE '^ARG RUSTUP_VERSION=[0-9]+\.[0-9]+\.[0-9]+$' <<<"$tc"
  check "toolchain: Rust toolchain pinned"        grep -qE '^ARG RUST_TOOLCHAIN=[0-9]+\.[0-9]+\.[0-9]+$' <<<"$tc"
  check "toolchain: rustup-init sha256 verified"  grep -q 'sha256sum -c' <<<"$tc"
  check "toolchain: x86_64 checksum present"      grep -qE 'x86_64\).*sum=[0-9a-f]{64}' <<<"$tc"
  check "toolchain: minimal profile"              grep -q -- '--profile minimal' <<<"$tc"
  check "toolchain: LLAMA_LLGUIDANCE defaults ON" grep -qE '^ARG LLAMA_LLGUIDANCE=ON$' <<<"$tc"
  check "build: LLAMA_LLGUIDANCE re-declared ON"  grep -qE '^ARG LLAMA_LLGUIDANCE=ON$' <<<"$build"
  check "build: cmake gets -DLLAMA_LLGUIDANCE"    grep -q -- '-DLLAMA_LLGUIDANCE="${LLAMA_LLGUIDANCE}"' <<<"$build"
  check "build: llama-server checked for llg_"    grep -q 'llg_new_constraint_any build/bin/llama-server' <<<"$build"
  check "build: executables --strip-debug (Rust std DWARF)" grep -q 'strip --strip-debug build/bin/llama-server' <<<"$build"
  check "runtime: no rustup/cargo/RUSTUP_HOME"    bash -c '! grep -qiE "rustup|cargo|RUSTUP_HOME|CARGO_HOME|rustc" <<<"$1"' _ "$rt"
  check "runtime: copies nothing from /opt"       bash -c '! grep -qE "COPY --from=[a-z]+ /opt" <<<"$1"' _ "$rt"
  check "cmake: llguidance pinned to a commit"    grep -qE 'GIT_TAG [0-9a-f]{40}' "$CM"
  check "cmake: cargo --locked --package llguidance" grep -q 'BUILD_COMMAND cargo build --release --locked --package llguidance' "$CM"
  check "release-build image: toolchain pinned"   grep -qE '^ARG RUST_TOOLCHAIN=[0-9]+\.[0-9]+\.[0-9]+$' "$RB"
  check "release-build image: no floating stable" bash -c '! grep -q -- "--default-toolchain stable" "$1"' _ "$RB"
}

build_checks() {
  local B="${1:-$ROOT/build-llg}"
  command -v cargo >/dev/null 2>&1 || { echo "SKIP: no cargo on PATH"; exit 77; }
  echo "cargo: $(cargo --version)   rustc: $(rustc --version)"
  local t0=$(date +%s)
  check "cmake configure (CPU, LLAMA_LLGUIDANCE=ON)" cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DLLAMA_LLGUIDANCE=ON \
      -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON
  check "build test-grammar-llguidance + llama-server" cmake --build "$B" -j "${JOBS:-6}" --target test-grammar-llguidance llama-server
  echo "build wall: $(( $(date +%s) - t0 )) s"
  check "CMakeCache has LLAMA_LLGUIDANCE=ON" grep -q 'LLAMA_LLGUIDANCE:BOOL=ON' "$B/CMakeCache.txt"
  check "libllguidance.a built" test -f "$B/llguidance/source/target/release/libllguidance.a"
  check "llama-server carries llg_* symbols" grep -q llg_new_constraint_any "$B/bin/llama-server"
  check "llama-server needs no libssl at run time" bash -c '! ldd "$1" | grep -qi libssl' _ "$B/bin/llama-server"
  local stripped="$B/bin/llama-server.strip-debug-check"
  check "llama-server --strip-debug (as the image does) keeps llg_* and runs --version" bash -c \
      'objcopy --strip-debug "$1" "$2" && grep -q llg_new_constraint_any "$2" && "$2" --version >/dev/null 2>&1' _ "$B/bin/llama-server" "$stripped"
  [ -f "$stripped" ] && echo "llama-server bytes: as built $(stat -c %s "$B/bin/llama-server"), --strip-debug $(stat -c %s "$stripped")"
  rm -f "$stripped"
  check "ctest test-grammar-llguidance" ctest --test-dir "$B" -R '^test-grammar-llguidance$' --output-on-failure
}

docker_checks() {
  command -v docker >/dev/null 2>&1 || { echo "SKIP: no docker"; exit 77; }
  local on=pxa-toolchain-check:llg-on off=pxa-toolchain-check:llg-off
  check "docker build --target toolchain (LLGUIDANCE=ON)"  docker build -q -f "$DF" --target toolchain --build-arg LLAMA_LLGUIDANCE=ON  -t "$on"  "$ROOT"
  check "docker build --target toolchain (LLGUIDANCE=OFF)" docker build -q -f "$DF" --target toolchain --build-arg LLAMA_LLGUIDANCE=OFF -t "$off" "$ROOT"
  check "ON image: cargo + rustc run" docker run --rm "$on" sh -c 'cargo --version && rustc --version'
  check "OFF image: no cargo"         docker run --rm "$off" sh -c '! command -v cargo'
  local s_on s_off
  s_on=$(docker image inspect -f '{{.Size}}' "$on" 2>/dev/null || echo 0); s_off=$(docker image inspect -f '{{.Size}}' "$off" 2>/dev/null || echo 0)
  echo "toolchain stage size: OFF=$s_off ON=$s_on delta=$((s_on - s_off)) bytes"
}

case "${1:-}" in
  ""|--static) static_checks ;;
  --build)  static_checks; build_checks "${2:-}" ;;
  --docker) static_checks; docker_checks ;;
  *) echo "usage: $0 [--static | --build [DIR] | --docker]"; exit 2 ;;
esac
finish
