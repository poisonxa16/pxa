#!/usr/bin/env bash
# Hostile-GGUF hparams (4.0.1 security review L4): a tiny good nemotron_h_moe trunk + NextN head (gen_tiny_nemotron_h_moe.py)
# loads; copies with a zero / non-dividing Mamba-2 size or a nextn_predict_layers past block_count are rejected with a
# message at load time (not a SIGFPE in the graph or a wrapped layer count). CPU only, seconds.
# Usage: tests/test-hparams-hostile.sh /path/to/test-dsv41-load
set -euo pipefail
BIN=${1:?path to test-dsv41-load}
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 -I "$D/gen_tiny_nemotron_h_moe.py" "$TMP/trunk.gguf" "$TMP/head.gguf" >/dev/null
A=nemotron_h_moe
pass=0; fail=0
run() { # name expectation base [KEY=VALUE...]
  local name=$1 want=$2 base="$TMP/$3.gguf" f="$TMP/$3.gguf"; shift 3
  if [ $# -gt 0 ]; then f="$TMP/$name.gguf"; python3 -I "$D/gguf-patch-u32.py" "$base" "$f" "$@"; fi
  if "$BIN" "$f" "$want" >"$TMP/$name.out" 2>&1; then pass=$((pass+1)); echo "ok   $name: $(tail -1 "$TMP/$name.out")";
  else fail=$((fail+1)); echo "FAIL $name (expect $want)"; tail -20 "$TMP/$name.out"; fi
}
M="fail:Mamba-2: ssm.time_step_rank"
run good          load trunk
# the trunk with the head attached (PXA_MTP_HEAD_FILE: block_count 6 + nextn 1) still loads
PXA_MTP_HEAD_FILE="$TMP/head.gguf" run head_attach load trunk
if ! grep -q "n_layer 7, nextn 1" "$TMP/head_attach.out"; then fail=$((fail+1)); echo "FAIL head_attach: the head was not attached"; fi
run dt_rank0      "$M" trunk $A.ssm.time_step_rank=0
run n_group0      "$M" trunk $A.ssm.group_count=0
run inner_nodiv   "$M" trunk $A.ssm.inner_size=129
run d_state0      "$M" trunk $A.ssm.state_size=0
run group_nodiv   "$M" trunk $A.ssm.group_count=16
# the NextN head file: block_count 7 (trunk + 1), nextn_predict_layers 1
run nextn_wrap    "fail:nextn_predict_layers 9 exceeds block_count 7" head $A.nextn_predict_layers=9
echo "test-hparams-hostile: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
