#!/usr/bin/env bash
# PXA_XCACHE_RING_LEND (R3a) GPU check: the lent ring is taken back on demand by a wide graph, and when it cannot be (the
# PXA_XCACHE_RING_LEND_FAIL=1 test hook = the card is full) the wide graph reads the pinned experts in place: no abort, and the
# answer equals an all-zero-copy run of the same plan (PXA_STREAM_ZC_NY=4096). The taken-back (copied) arm must answer; its bytes
# may differ from the in-place ones (copy vs zero-copy is correct-not-identical at some residencies; G30: f8df01b88110 vs 95427b95f060).
#   tests/run-xcache-ring-lend.sh <bin dir> <model.gguf> [llama-server args...]
# Needs a model that streams experts with the CPU cold path (PXA_XCACHE plan, e.g. Flash-Next on one 16 GB card). The CPU-alias
# width is lowered to 128 tokens (PXA_STREAM_NARROW_MOE) so every 256-token prompt ubatch is a "wide" graph that streams through
# the ring: the long-prefill path at a cheap prompt. Exit 0 = all three arms answered, the logs show the paths, fail == zc.
set -uo pipefail
BIN=$1; MODEL=$2; shift 2
PORT=${PORT:-18931}
OUT=${OUT:-$(mktemp -d)}
PROMPT=$(python3 -I -c 'print(" ".join(f"Item {i}: the {["red","green","blue","amber"][i%4]} box holds {i*7%23} stones." for i in range(80)) + " Question: how many stones does item 41 hold? Answer:")')
run_arm() {   # $1 = tag, rest = env assignments
  local tag=$1; shift
  env "$@" PXA_XCACHE_RING_LEND=1 PXA_STREAM_NARROW_MOE=128 PXA_STREAM_LOG=1 \
    "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port $PORT -ub 256 "${EXTRA[@]}" > "$OUT/srv-$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do
    curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  python3 -I - "$PORT" "$OUT/ans-$tag.json" <<PY
import json, sys, urllib.request
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/completion", method="POST",
      data=json.dumps({"prompt": """$PROMPT""", "n_predict": 32, "temperature": 0, "cache_prompt": False}).encode(),
      headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
json.dump({"content": r.get("content"), "tokens_evaluated": r.get("tokens_evaluated")}, open(sys.argv[2], "w"))
PY
  local rc=$?
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  return $rc
}
EXTRA=("$@")
fail=0
run_arm take  || { echo "FAIL: arm take did not answer"; fail=1; }
run_arm fail PXA_XCACHE_RING_LEND_FAIL=1 || { echo "FAIL: arm fail (zero-copy fallback) did not answer"; fail=1; }
run_arm zc PXA_STREAM_ZC_NY=4096 || { echo "FAIL: arm zc (all zero-copy reference) did not answer"; fail=1; }
grep -q "ring .* LENT to the expert cache" "$OUT/srv-take.log" || { echo "FAIL: the ring was not lent (plan not engaged?)"; fail=1; }
grep -q "lent ring taken back" "$OUT/srv-take.log" && echo "take: lent ring taken back ($(grep -o 'lent ring taken back[^(]*([^)]*)' "$OUT/srv-take.log" | head -1))" \
  || echo "take: ring NOT taken back (VRAM short: the arm ran the zero-copy fallback too)"
grep -q "could not be taken back (PXA_XCACHE_RING_LEND_FAIL)" "$OUT/srv-fail.log" && echo "fail: zero-copy fallback engaged" \
  || { echo "FAIL: the zero-copy fallback did not engage"; fail=1; }
echo "take answered: $(cat "$OUT/ans-take.json" 2>/dev/null)"
if [ -s "$OUT/ans-fail.json" ] && cmp -s "$OUT/ans-fail.json" "$OUT/ans-zc.json"; then echo "fallback == all-zero-copy: $(cat "$OUT/ans-fail.json")"
else echo "FAIL: fallback differs from the all-zero-copy run: $(cat "$OUT/ans-fail.json" 2>/dev/null) vs $(cat "$OUT/ans-zc.json" 2>/dev/null)"; fail=1; fi
echo "logs: $OUT"
[ $fail = 0 ] && echo "ALL PASS" || echo "FAILED"
exit $fail
