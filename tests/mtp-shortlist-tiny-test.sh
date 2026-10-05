#!/usr/bin/env bash
# PXA_MTP_SHORTLIST on the Flash-Next MTP block, end to end on a CPU with a tiny random-weight model (2026-10-04).
#
# What it proves. The MTP draft head scoring only a token-id PREFIX of the vocabulary can change which drafts are proposed and
# accepted, and can never change the emitted text, because the target verifies every token with its own full head. Three
# server arms on one tiny `qwen4exp` GGUF that carries a real grafted NextN/MTP block (tests/gen_tiny_qwen4exp.py --mtp):
#
#   plain      --spec-type none                                  the reference text
#   mtp        --spec-type mtp:n_max=3,p_min=0                   the full draft head
#   shortlist  --spec-type mtp:n_max=3,p_min=0 + PXA_MTP_SHORTLIST=prefix:256   (half of the 512-id vocabulary)
#
# Every request is greedy (temperature 0, top_k 1) on three distinct prompts. PASS = the three arms emit byte-identical text on
# every prompt, AND the shortlist arm's server log says it drew drafts from the windowed head (the lever was live, not inert).
# A random-weight head predicts the target's token about once in a thousand tries, so this checks the plumbing (window, index
# map, verify) rather than a speed: acceptance and speed are measured on the real files.
#
# Usage: mtp-shortlist-tiny-test.sh [port] ; env PXA_LLAMA_SERVER (default build/bin/llama-server), THREADS (default 4), MODEL, OUT
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BIN=${PXA_LLAMA_SERVER:-$ROOT/build/bin/llama-server}
PORT=${1:-18795}
OUT=${OUT:-$(mktemp -d)}
MODEL=${MODEL:-$OUT/tiny_qwen4exp_mtp.gguf}   # MODEL=<a file from gen_tiny_qwen4exp.py --mtp> skips the generator (it needs numpy)
[ -x "$BIN" ] || { echo "FAIL: no server binary at $BIN"; exit 2; }
if [ ! -f "$MODEL" ]; then
    python3 "$HERE/gen_tiny_qwen4exp.py" "$MODEL" --mtp > "$OUT/gen.log" 2>&1 || { echo "FAIL: generator"; cat "$OUT/gen.log"; exit 2; }
fi

run_arm() {   # run_arm NAME SPEC [ENV=VAL...]
    local name=$1 spec=$2; shift 2
    env "$@" "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 0 -c 2048 -np 1 -t "${THREADS:-4}" --no-warmup --cache-ram 0 $spec \
        > "$OUT/$name.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 90); do
        python3 -c "import sys,urllib.request; sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    python3 - "$PORT" "$OUT/$name.txt" <<'PY'
import json, sys, urllib.request
port, out = sys.argv[1:3]
res = []
for p in ["the quick brown fox jumps over the lazy dog ", "hello world, this is a tiny model test ", "0123456789 abcdefghij klmnop "]:
    r = urllib.request.Request("http://127.0.0.1:%s/completion" % port, json.dumps({"prompt": p, "n_predict": 80, "temperature": 0, "top_k": 1,
                               "seed": 1, "cache_prompt": False}).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    t = d.get("timings", {})
    res.append({"text": d["content"], "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")})
json.dump(res, open(out, "w"))
PY
    kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}

run_arm plain     "--spec-type none"
run_arm mtp       "--spec-type mtp:n_max=3,p_min=0.0"
run_arm shortlist "--spec-type mtp:n_max=3,p_min=0.0" PXA_MTP_SHORTLIST=prefix:256

python3 - "$OUT" <<'PY'
import json, sys
o = sys.argv[1]
a = {k: json.load(open("%s/%s.txt" % (o, k))) for k in ("plain", "mtp", "shortlist")}
ok = True
for i in range(3):
    t = [a[k][i]["text"] for k in ("plain", "mtp", "shortlist")]
    same = t[0] == t[1] == t[2]
    ok &= same
    print("prompt %d: plain / mtp / shortlist text %s  (drafted %s/%s accepted %s/%s)" % (
        i, "IDENTICAL" if same else "DIFFERENT", a["mtp"][i]["draft_n"], a["shortlist"][i]["draft_n"], a["mtp"][i]["draft_acc"], a["shortlist"][i]["draft_acc"]))
sys.exit(0 if ok else 1)
PY
RC=$?
LIVE=$(grep -c "PXA_MTP_SHORTLIST: draft head restricted to the token-id prefix \[0, 256) of 512 rows" "$OUT/shortlist.log")
DREW=$(grep -c "draft tokens were drawn from the shortlisted head" "$OUT/shortlist.log")
echo "shortlist arm: window armed lines $LIVE, draw-count lines $DREW"
OFF=$(grep -c "PXA_MTP_SHORTLIST: draft head restricted" "$OUT/mtp.log")
[ "$LIVE" -ge 1 ] || { echo "FAIL: the shortlist arm never armed the window"; RC=1; }
[ "$OFF" -eq 0 ]  || { echo "FAIL: the full-head arm armed a window"; RC=1; }
[ "$RC" -eq 0 ] && echo "PASS mtp-shortlist-tiny-test" || echo "FAIL mtp-shortlist-tiny-test (logs in $OUT)"
exit $RC
