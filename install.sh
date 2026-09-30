#!/usr/bin/env bash
# PXA one-command installer (Pascal 6.0/6.1 and Volta 7.0 GPUs, Linux x86_64).
#   curl -fsSL https://raw.githubusercontent.com/poisonxa16/pxa/main/install.sh | bash
#   ... | bash -s -- --docker          pull the container image instead
#   ... | bash -s -- --dir /opt/pxa    install somewhere else (default ~/.local/share/pxa)
#   ... | bash -s -- --version v2026.10.1
set -euo pipefail

REPO=poisonxa16/pxa
DIR="${HOME}/.local/share/pxa"
MODE=tarball
VERSION=latest
while [ $# -gt 0 ]; do
  case "$1" in
    --docker) MODE=docker ;;
    --dir) DIR="${2:?--dir needs a path}"; shift ;;
    --version) VERSION="${2:?--version needs a tag}"; shift ;;
    -h|--help) sed -n '2,6p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

die() { echo "$*" >&2; exit 1; }

# 1. platform checks
[ "$(uname -m)" = x86_64 ] || die "PXA ships Linux x86_64 binaries only."
command -v curl >/dev/null || die "curl is required."

glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
gl_major=${glibc%%.*}; gl_minor=${glibc#*.}; gl_minor=${gl_minor%%.*}
glibc_at_least() { [ "${gl_major:-0}" -gt "$1" ] || { [ "${gl_major:-0}" -eq "$1" ] && [ "${gl_minor:-0}" -ge "$2" ]; }; }
if [ "$MODE" = tarball ]; then
  glibc_at_least 2 35 || die "PXA needs glibc 2.35 or newer (Ubuntu 22.04+, Debian 12+, RHEL 9+); this host has ${glibc:-unknown}."
fi

# 2. GPU check: every card must be compute capability 6.0, 6.1 or 7.0
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first."
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:blank:]' | sort -u | paste -sd' ')
[ -n "$caps" ] || die "nvidia-smi reported no GPUs."
for c in $caps; do
  case "$c" in
    6.0|6.1|7.0) ;;
    *) die "PXA is built for Pascal (6.0, 6.1) and Volta (7.0); this host reports: $caps" ;;
  esac
done

# 3. resolve the release
if [ "$VERSION" = latest ]; then api="https://api.github.com/repos/$REPO/releases/latest"
else api="https://api.github.com/repos/$REPO/releases/tags/$VERSION"; fi
rel=$(curl -fsSL "$api") || die "Could not read the PXA release ($VERSION)."
tag=$(printf '%s\n' "$rel" | grep '"tag_name"' | cut -d'"' -f4 | sed -n 1p)
[ -n "$tag" ] || die "Could not resolve the PXA release tag."

if [ "$MODE" = docker ]; then
  command -v docker >/dev/null || die "docker not found."
  docker pull "ghcr.io/$REPO:$tag"
  echo "Pulled ghcr.io/$REPO:$tag"
  echo "Run: docker run --rm --gpus all -p 8080:8080 -v /path/to/models:/models ghcr.io/$REPO:$tag ..."
  echo "See https://github.com/$REPO#readme for the full command."
  exit 0
fi

# 4. pick the engine tarball (never the libggml-pxqn-* library assets)
urls=$(printf '%s\n' "$rel" | grep '"browser_download_url"' | cut -d'"' -f4 \
  | grep -E '/pxa-v[^/]*linux-x86_64[^/]*\.tar\.gz$' || true)
[ -n "$urls" ] || die "No engine tarball (pxa-v*-linux-x86_64*.tar.gz) in release $tag."
u22=$(printf '%s\n' "$urls" | grep -- '-ubuntu22\.04\.tar\.gz$' | sed -n 1p || true)
u24=$(printf '%s\n' "$urls" | grep -v -- '-ubuntu22\.04\.tar\.gz$' | sed -n 1p || true)
if glibc_at_least 2 38; then url=${u24:-$u22}; else url=${u22:-}; fi
[ -n "${url:-}" ] || die "No tarball in $tag matches glibc ${glibc}; the 24.04 build needs 2.38+."

# 5. download, verify, unpack
mkdir -p "$DIR" && cd "$DIR"
tgz=$(basename "$url")
curl -fL -o "$tgz" "$url"
curl -fL -o "$tgz.sha256" "$url.sha256"
sha256sum -c "$tgz.sha256" || die "Checksum mismatch for $tgz; not installing."
top=$(tar tzf "$tgz" | sed -n 1p | cut -d/ -f1)
tar xzf "$tgz"
ln -sfn "$top" current
echo "PXA $top installed in $DIR"
echo "Serve a GGUF with: $DIR/current/run-server.sh -m /path/to/model.gguf -ngl 99 -c 8192"
