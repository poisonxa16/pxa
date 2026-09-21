#!/usr/bin/env bash
# test-pxq-export.sh — end-to-end correctness test for llama-pxq-export (PXQ -> F16 GGUF) and
# for the CPU panel dequant that backs it.
#
# What it proves, per PXQ tier, in order of how much it would hurt to get wrong:
#
#   1. THE TWO DECODERS AGREE. The CUDA kernels (ggml/src/ggml-cuda/pxa/pxq6.cuh, pxq23.cuh) and the
#      CPU panel dequant (ggml/src/pxq-cpu.c) are independent implementations of the same frozen
#      slab format. --verify decodes every PXQ tensor with BOTH and reports the worst ULP
#      difference per type. Expected: 0.
#
#   2. NON-PXQ TENSORS ARE UNTOUCHED. Every tensor whose type is not a PXQ slab type must be
#      byte-identical between the PXQ file and the export -- same type, same shape, same bytes.
#      A copy path that quietly re-encoded an MXFP4 backbone tensor would still produce a file
#      that loads.
#
#   3. STREAMING DOES NOT CHANGE THE ANSWER. --verify also re-decodes each PXQ tensor as ONE
#      call and requires the chunked, streamed result the tool actually wrote to be
#      bit-identical: the panel arithmetic, the file seeks and the chunk boundaries.
#
#   4. --cpu AND THE GPU PATH PRODUCE THE SAME FILE, byte for byte.
#
#   5. THE EXPORT LOADS AND GENERATES. Greedy (temp 0) continuations from the PXQ model and from
#      the export are compared and the number of agreeing leading tokens is reported. The two
#      hold the same weights, but the PXQ fused kernels and cuBLAS F16 sum in a different order,
#      so they drift eventually; the PXA_PXQ6=0 run (which puts the PXQ side on the same
#      dequant -> cuBLAS path) is the tight comparison.
#
#   6. llama-quantize REQUANTIZES FROM A PXQ SOURCE, and refuses to without the consent flag.
#
# Needs a CUDA build and one GPU. Small models only.
#
# Usage:
#   tests/test-pxq-export.sh <source.gguf> [build-dir]
#   PXQ_TIERS="PXQ4 PXQ2" BUILD_DIR=build-spd tests/test-pxq-export.sh <source.gguf>
#
# <source.gguf> is any small F32/F16/BF16/Q8_0 model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_MODEL="${1:-${PXQ_EXPORT_TEST_MODEL:-}}"
BUILD_DIR="${2:-${BUILD_DIR:-${REPO_ROOT}/build-spd}}"
TIERS="${PXQ_TIERS:-PXQ4 PXQ4HQ PXQ2 PXQ3 PXQ6 PXQ1}"
NPRED="${PXQ_EXPORT_NPRED:-24}"
PROMPT="${PXQ_EXPORT_PROMPT:-The capital of France is}"

if [[ -z "${SRC_MODEL}" || ! -f "${SRC_MODEL}" ]]; then
  echo "usage: $0 <source.gguf> [build-dir]   (or set PXQ_EXPORT_TEST_MODEL)" >&2
  exit 2
fi

QUANTIZE="${BUILD_DIR}/bin/llama-quantize"
EXPORT="${BUILD_DIR}/bin/llama-pxq-export"
CLI="${BUILD_DIR}/bin/llama-cli"
for b in "${QUANTIZE}" "${EXPORT}" "${CLI}"; do
  [[ -x "${b}" ]] || { echo "FAIL: missing ${b}" >&2; exit 1; }
done

WORK="${PXQ_EXPORT_WORKDIR:-$(mktemp -d)}"
mkdir -p "${WORK}"
echo "workdir: ${WORK}"

FIRST_TIER=""
for TIER in ${TIERS}; do
  echo
  echo "############ ${TIER} ############"
  PXQ_GGUF="${WORK}/model.${TIER}.gguf"
  F16_GGUF="${WORK}/model.${TIER}.f16.gguf"
  CPU_GGUF="${WORK}/model.${TIER}.f16cpu.gguf"

  echo "== 0. quantize -> ${TIER}"
  if [[ -s "${PXQ_GGUF}" ]]; then
    echo "   reusing ${PXQ_GGUF}"
  else
    # Name the whole body explicitly, for two reasons.
    #
    # (a) The quantizer asserts that a PXQ-named artifact is at least 50% PXQ bytes, and on a
    #     small model the vocab head dominates the budget: a default PXQ2 run lands at 36% and
    #     is refused. PXQ1 is worse -- it is a PXQ-UNIVERSAL EXPERT tier, so as a whole-file
    #     ftype it claims nothing at all on a dense model (9%).
    # (b) attn_k / attn_v default to q8_0, so without this they are never exercised as PXQ.
    #
    # NOT --token-embedding-type: token_embd is read by GET_ROWS one row at a time, and a PXQ
    # row is unreadable in isolation. The quantizer's row-gather guard catches
    # per_layer_token_embd by name and anything with ne1 >= 1e6 by size, but a 151936-row vocab
    # slips through both, and --token-embedding-type is an explicit override anyway. Forcing a
    # PXQ tier there produces a model that quantizes, loads, and then emits nothing -- measured.
    LC_TIER="$(echo "${TIER}" | tr 'A-Z' 'a-z')"
    BODY="attn_q.weight=${LC_TIER},attn_k.weight=${LC_TIER},attn_v.weight=${LC_TIER}"
    BODY="${BODY},attn_output.weight=${LC_TIER}"
    BODY="${BODY},ffn_up.weight=${LC_TIER},ffn_gate.weight=${LC_TIER},ffn_down.weight=${LC_TIER}"
    # token_embd -> a BLOCK codec, so it stays row-decodable for GET_ROWS, and small enough
    # that the PXQ body clears the 50% composition floor (with the default q6_K head, PXQ2
    # lands at 49.3% and is refused). PXQ1 stores the whole body in 66 MiB, less than a q4_K
    # head, so it needs the head smaller still.
    EMBD_TYPE=q4_K
    IMRULES=()
    if [[ "${TIER}" == "PXQ1" ]]; then
      EMBD_TYPE=q2_K
      IMRULES=(--ignore-imatrix-rules)   # q2_K without an imatrix; fine for a decoder test
    fi
    "${QUANTIZE}" --allow-requantize --i-know-this-is-double-lossy --custom-q "${BODY}" \
        --token-embedding-type "${EMBD_TYPE}" "${IMRULES[@]}" \
        "${SRC_MODEL}" "${PXQ_GGUF}" "${TIER}" 8 > "${WORK}/quantize.${TIER}.log" 2>&1 \
      || { tail -30 "${WORK}/quantize.${TIER}.log" >&2; echo "FAIL: quantize to ${TIER}" >&2; exit 1; }
  fi

  echo "== 1. export -> F16 on the GPU, with --verify (64 MiB chunks: multi-chunk on the head)"
  "${EXPORT}" "${PXQ_GGUF}" "${F16_GGUF}" --type f16 --device 0 --verify --chunk-mib 64 \
      > "${WORK}/export.${TIER}.log" 2>&1 \
    || { tail -30 "${WORK}/export.${TIER}.log" >&2; echo "FAIL: llama-pxq-export ${TIER}" >&2; exit 1; }
  grep -E "dequantized|verify:" "${WORK}/export.${TIER}.log" | sed 's/^main: /   /'

  N_DEQ=$(grep -oE "dequantized [0-9]+ PXQ" "${WORK}/export.${TIER}.log" | grep -oE "[0-9]+" | head -1)
  if [[ -z "${N_DEQ}" || "${N_DEQ}" -eq 0 ]]; then
    echo "FAIL: the export decoded 0 PXQ tensors for ${TIER} -- nothing was proven" >&2
    exit 1
  fi
  grep -q "verify: chunked == whole-tensor dequant" "${WORK}/export.${TIER}.log" \
    || { echo "FAIL: --verify did not confirm chunked == whole-tensor" >&2; exit 1; }

  # cross-engine ULP must be 0 on every type the run touched
  if ! grep -q "cross-check on" "${WORK}/export.${TIER}.log"; then
    echo "FAIL: --verify ran no cross-engine check (no CPU dequant for this tier?)" >&2
    exit 1
  fi
  while read -r line; do
    u=$(echo "${line}" | grep -oE "worst [0-9]+ ULP" | grep -oE "[0-9]+")
    if [[ "${u}" != "0" ]]; then
      echo "FAIL: CUDA and CPU decoders differ by ${u} ULP: ${line}" >&2
      exit 1
    fi
  done < <(grep "worst .* ULP" "${WORK}/export.${TIER}.log")

  echo "== 2. every non-PXQ tensor byte-identical"
  python3 "${SCRIPT_DIR}/pxq-export-diff.py" "${PXQ_GGUF}" "${F16_GGUF}" \
    || { echo "FAIL: tensor byte-identity check" >&2; exit 1; }

  echo "== 3. --cpu produces the same file"
  "${EXPORT}" "${PXQ_GGUF}" "${CPU_GGUF}" --type f16 --cpu > "${WORK}/export-cpu.${TIER}.log" 2>&1 \
    || { tail -30 "${WORK}/export-cpu.${TIER}.log" >&2; echo "FAIL: llama-pxq-export --cpu" >&2; exit 1; }
  cmp "${F16_GGUF}" "${CPU_GGUF}" \
    || { echo "FAIL: the --cpu export differs from the CUDA export" >&2; exit 1; }
  echo "   cmp: byte-identical to the CUDA export"

  echo "== 4. llama-quantize accepts the PXQ file as a source"
  if "${QUANTIZE}" --allow-requantize "${PXQ_GGUF}" "${WORK}/requant-out.gguf" Q4_K_M 8 \
       > "${WORK}/requant-noflag.${TIER}.log" 2>&1; then
    echo "FAIL: requantizing from ${TIER} succeeded WITHOUT --i-know-this-is-double-lossy" >&2
    exit 1
  fi
  grep -q "double-lossy" "${WORK}/requant-noflag.${TIER}.log" \
    || { tail -5 "${WORK}/requant-noflag.${TIER}.log" >&2
         echo "FAIL: the refusal did not name --i-know-this-is-double-lossy" >&2; exit 1; }
  echo "   refused without the consent flag (as designed)"
  "${QUANTIZE}" --allow-requantize --i-know-this-is-double-lossy \
      "${PXQ_GGUF}" "${WORK}/requant-out.gguf" Q4_K_M 8 \
      > "${WORK}/requant.${TIER}.log" 2>&1 \
    || { tail -30 "${WORK}/requant.${TIER}.log" >&2
         echo "FAIL: ${TIER} -> Q4_K_M direct requantize" >&2; exit 1; }
  echo "   ${TIER} -> Q4_K_M direct: $(ls -l "${WORK}/requant-out.gguf" | awk '{print $5}') bytes"
  rm -f "${WORK}/requant-out.gguf" "${CPU_GGUF}"

  [[ -z "${FIRST_TIER}" ]] && FIRST_TIER="${TIER}"
done

echo
echo "############ generation ############"
# Compare on the highest-fidelity tier that was run: an aggressive tier is a fine decoder
# test but a poor generation test, because its own output is degenerate.
for t in PXQ6 PXQ4HQ PXQ4 PXQ3 PXQ2 PXQ1; do
  if [[ -s "${WORK}/model.${t}.gguf" && -s "${WORK}/model.${t}.f16.gguf" ]]; then FIRST_TIER="${t}"; break; fi
done
PXQ_GGUF="${WORK}/model.${FIRST_TIER}.gguf"
F16_GGUF="${WORK}/model.${FIRST_TIER}.f16.gguf"

run_cli() {
  local model="$1"
  env ${2:-} "${CLI}" -m "${model}" -p "${PROMPT}" -n "${NPRED}" \
      --temp 0 --seed 1 -ngl 99 -c 512 --no-warmup 2> /dev/null
}

OUT_PXQ_FUSED="$(run_cli "${PXQ_GGUF}" "" || true)"
OUT_PXQ_DEQ="$(run_cli "${PXQ_GGUF}" "PXA_PXQ6=0 PXA_PXQ4=0" || true)"
OUT_F16="$(run_cli "${F16_GGUF}" "" || true)"

[[ -n "${OUT_F16}" ]] || { echo "FAIL: the exported F16 model produced no output" >&2; exit 1; }

# agreement measured on the GENERATED text only: llama-cli echoes the prompt, so a raw
# common prefix of exactly len(prompt) means ZERO generated tokens matched.
lcp_tokens() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys
a, b, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
ga = a[len(prompt):] if a.startswith(prompt) else a
gb = b[len(prompt):] if b.startswith(prompt) else b
n = 0
for x, y in zip(ga, gb):
    if x != y: break
    n += 1
print(len(ga[:n].split()), n, len(ga))
PY
}

read -r TOKS_FUSED CHARS_FUSED NCHARS <<< "$(lcp_tokens "${OUT_PXQ_FUSED}" "${OUT_F16}" "${PROMPT}")"
read -r TOKS_DEQ   CHARS_DEQ   _      <<< "$(lcp_tokens "${OUT_PXQ_DEQ}"   "${OUT_F16}" "${PROMPT}")"

echo "   export generated ${NCHARS} chars after the prompt"
echo "   ${FIRST_TIER} (fused kernels)      vs export: ${TOKS_FUSED} generated words (${CHARS_FUSED} chars) agree"
echo "   ${FIRST_TIER} (PXA_PXQ6=0 dequant) vs export: ${TOKS_DEQ} generated words (${CHARS_DEQ} chars) agree"

# Character counts, not word counts: an aggressive tier (PXQ1 on a 0.6B) generates a long
# unbroken repeat, which is degenerate text but a perfectly valid decoder check -- the two
# models still have to agree on it token for token.
if [[ "${NCHARS}" -lt 8 ]]; then
  echo "FAIL: the exported F16 model generated almost nothing (${NCHARS} chars)" >&2
  exit 1
fi
if [[ "${CHARS_DEQ}" -lt 4 ]]; then
  echo "FAIL: the export and the PXQ model disagree on the FIRST generated token, even against" >&2
  echo "      the dequant fallback. They hold the same weights; this is a bug." >&2
  exit 1
fi

echo
echo "PASS: tiers [${TIERS}] -- CUDA and CPU decoders agree to 0 ULP, non-PXQ tensors"
echo "      byte-identical, chunked == whole-tensor, --cpu == --device output,"
echo "      direct requantize gated and working, greedy agreement ${TOKS_DEQ} words."
