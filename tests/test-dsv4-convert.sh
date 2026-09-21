#!/usr/bin/env bash
# Conversion smoke + NUMERIC test for DeepSeek-V4 (deepseek_v4 -> deepseek4).
#
# Runs the HF->GGUF converter against a tiny synthetic fixture that carries
# every tensor-name pattern the real 43-layer DeepSeek-V4-Flash checkpoint has
# (fp8 e4m3 + ue8m0 128x128 block scales, MXFP4 routed experts, I64 hash
# routing tables, per-layer compressor/indexer tensors driven by
# compress_ratios, an MTP tail that must be dropped) and asserts the output is
# EXACTLY right: right arch, right tensor-name set, right metadata -- AND that
# the fp8 and MXFP4 dequant math is numerically correct against an
# independently computed reference.
#
# The numeric half is the point. A wrong nibble order or a wrong scale exponent
# produces a model that loads and emits fluent garbage with no loud signal; a
# name-set-only test cannot see it.
#
# CPU-only, no GPU, no 300 GB download -- safe to run in CI on every commit.
#
# Usage: tests/test-dsv4-convert.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures/dsv4-tiny"
TOKENIZER_DIR="${SCRIPT_DIR}/fixtures/glm4-moe-lite-tiny"
CONVERT_PY="${REPO_ROOT}/convert_hf_to_gguf.py"

for f in config.json gen_fixture.py verify_fixture.py; do
  if [[ ! -f "${FIXTURE_DIR}/${f}" ]]; then
    echo "FAIL: fixture is missing ${f} (looked in ${FIXTURE_DIR})" >&2
    exit 1
  fi
done
# the fixture borrows a real BPE tokenizer rather than committing a second 20 MB blob
for f in tokenizer.json tokenizer_config.json; do
  if [[ ! -f "${TOKENIZER_DIR}/${f}" ]]; then
    echo "FAIL: missing ${TOKENIZER_DIR}/${f} (borrowed by the dsv4 fixture)" >&2
    exit 1
  fi
done
if [[ ! -f "${CONVERT_PY}" ]]; then
  echo "FAIL: convert_hf_to_gguf.py not found at ${CONVERT_PY}" >&2
  exit 1
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/dsv4-fixture-test.XXXXXX")"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

cp "${FIXTURE_DIR}/config.json" "${WORKDIR}/"
cp "${TOKENIZER_DIR}/tokenizer.json" "${TOKENIZER_DIR}/tokenizer_config.json" "${WORKDIR}/"

echo "== generating synthetic fp8 / MXFP4 weights =="
python3 "${FIXTURE_DIR}/gen_fixture.py" --dir "${WORKDIR}"

echo "== converting (convert_hf_to_gguf.py, bf16, CPU) =="
OUT_GGUF="${WORKDIR}/dsv4-tiny.gguf"
if ! python3 "${CONVERT_PY}" "${WORKDIR}" --outfile "${OUT_GGUF}" --outtype bf16; then
  echo "FAIL: convert_hf_to_gguf.py exited non-zero on the fixture" >&2
  exit 1
fi
if [[ ! -f "${OUT_GGUF}" ]]; then
  echo "FAIL: converter reported success but ${OUT_GGUF} does not exist" >&2
  exit 1
fi

echo "== verifying arch / tensor-name set / KVs / dequant numerics =="
python3 "${FIXTURE_DIR}/verify_fixture.py" "${OUT_GGUF}" "${WORKDIR}/config.json" "${WORKDIR}/reference.npz"

echo "PASS: deepseek4 (DeepSeek-V4) conversion test"
