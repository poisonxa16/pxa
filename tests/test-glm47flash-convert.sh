#!/usr/bin/env bash
# Conversion smoke test for GLM-4.7-Flash (glm4_moe_lite -> deepseek2/MLA).
#
# Runs the HF->GGUF converter against a tiny synthetic fixture that carries
# every tensor-name pattern the real 47-layer + NextN model has (shrunk
# dims, real tokenizer), and asserts the output is EXACTLY right: right
# arch, right tensor-name set (nothing missing, nothing extra, no NextN
# leak), right MoE/MLA/tokenizer metadata. CPU-only, no GPU, no 31 GB
# download -- safe to run in CI on every commit.
#
# Usage: tests/test-glm47flash-convert.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures/glm4-moe-lite-tiny"
CONVERT_PY="${REPO_ROOT}/convert_hf_to_gguf.py"

for f in config.json tokenizer.json tokenizer_config.json generation_config.json gen_fixture.py verify_fixture.py; do
  if [[ ! -f "${FIXTURE_DIR}/${f}" ]]; then
    echo "FAIL: fixture is missing ${f} (looked in ${FIXTURE_DIR})" >&2
    exit 1
  fi
done
if [[ ! -f "${CONVERT_PY}" ]]; then
  echo "FAIL: convert_hf_to_gguf.py not found at ${CONVERT_PY}" >&2
  exit 1
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/glm47-fixture-test.XXXXXX")"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

# Copy just the HF-repo-shaped inputs (config + real tokenizer) into a
# scratch dir; the generated model.safetensors is written there too, so
# the git-tracked fixture directory never gets a binary blob in it.
cp "${FIXTURE_DIR}/config.json" \
   "${FIXTURE_DIR}/tokenizer.json" \
   "${FIXTURE_DIR}/tokenizer_config.json" \
   "${FIXTURE_DIR}/generation_config.json" \
   "${WORKDIR}/"

echo "== generating synthetic weights =="
python3 "${FIXTURE_DIR}/gen_fixture.py" --dir "${WORKDIR}"

echo "== converting (convert_hf_to_gguf.py, bf16, CPU) =="
OUT_GGUF="${WORKDIR}/glm4-moe-lite-tiny.gguf"
if ! python3 "${CONVERT_PY}" "${WORKDIR}" --outfile "${OUT_GGUF}" --outtype bf16; then
  echo "FAIL: convert_hf_to_gguf.py exited non-zero on the fixture" >&2
  exit 1
fi
if [[ ! -f "${OUT_GGUF}" ]]; then
  echo "FAIL: converter reported success but ${OUT_GGUF} does not exist" >&2
  exit 1
fi

echo "== verifying architecture / tensor-name set / KVs / tokenizer =="
python3 "${FIXTURE_DIR}/verify_fixture.py" "${OUT_GGUF}" "${WORKDIR}/config.json"

echo "PASS: glm4_moe_lite (GLM-4.7-Flash) conversion smoke test"
