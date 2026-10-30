#!/bin/bash
# Build sm70 and sm60 test images tagged :v32-field. Does not push.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK=${CPU_QUIET_LOCK:?set CPU_QUIET_LOCK to the quiet lock file}
build() {
    local arch="$1" base="$2" tag="$3"
    echo "build $tag from $base"
    flock -s "$LOCK" docker build -f "$ROOT/deploy/Dockerfile.field" \
        --build-arg BASE_IMAGE="$base" \
        -t "$tag" "$ROOT"
}
build sm70 ghcr.io/poisonxa16/pxa-vllm:sm70-v2026.10 pxa-vllm:sm70-v32-field
build sm60 ghcr.io/poisonxa16/pxa-vllm:sm60-v2026.10 pxa-vllm:sm60-v32-field
docker image inspect --format '{{.Id}} {{index .RepoTags 0}}' \
    pxa-vllm:sm70-v32-field pxa-vllm:sm60-v32-field
