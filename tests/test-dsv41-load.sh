#!/usr/bin/env bash
# DeepSeek-V4.1 loader test: builds the tiny synthetic deepseek41 stubs (every
# variant) and checks test-dsv41-load accepts the good ones and rejects the bad
# ones with the right error. CPU only, seconds.
# Usage: tests/test-dsv41-load.sh /path/to/test-dsv41-load
set -euo pipefail
BIN=${1:?path to test-dsv41-load}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="${SCRIPT_DIR}/fixtures/dsv41-stub/gen_stub.py"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
run() { # variant expectation
  python3 -I "$GEN" "$TMP/$1.gguf" --variant "$1" >/dev/null
  if "$BIN" "$TMP/$1.gguf" "$2" >"$TMP/$1.out" 2>&1; then pass=$((pass+1)); tail -1 "$TMP/$1.out";
  else fail=$((fail+1)); echo "FAIL variant $1 (expect $2)"; tail -20 "$TMP/$1.out"; fi
}
run ok             ok
run noengram       ok
run ratio4         "fail:only supports compression ratios 0, 1 and 2"
run extra_hchead   "fail:wrong number of tensors"
run missing_gate   "fail:tensor 'blk.2.attn_compressor_gate.weight' not found"
run gate_on_ratio1 "fail:wrong number of tensors"
run kvsrc_ratio0   "fail:is a kv/index source but has compress ratio 0"
run hash           "fail:has no hash-routed layers"
run qnorm          "fail:wrong number of tensors"
run tid2eid        "fail:wrong number of tensors"
run cand_not_src   "fail:is not a kv source layer"
echo "test-dsv41-load: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
