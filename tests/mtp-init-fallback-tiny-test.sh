#!/usr/bin/env bash
# Bug #344: an MTP draft context that cannot be built must not stop the server (2026-10-21).
#
# What it proves, on a CPU with a tiny random-weight qwen4exp GGUF that carries a NextN/MTP block (tests/gen_tiny_qwen4exp.py --mtp):
#
#   plain     --spec-type none                                              the reference text
#   fallback  --spec-type mtp:n_max=3,p_min=0 + PXA_MTP_INIT_FAULT=1        the MTP context is refused (test hook): the server must
#             start, log the fallback warning exactly once, and emit text byte-identical to plain with no drafts
#   fatal     the same + PXA_MTP_INIT_FATAL=1                               the pre-#344 behaviour: the server exits ("server init failed")
#
# Every request is greedy (temperature 0, top_k 1) on three prompts. The real-card case (Qwen3.6-35B-A3B MTP PXQN4, one V100,
# -ngl 30 -c 4096: the companion left 99.5 MiB under the 128 MiB floor) is checked on the card; this checks the plumbing.
#
# Usage: mtp-init-fallback-tiny-test.sh [port] ; env PXA_LLAMA_SERVER (default build/bin/llama-server), THREADS (default 4), MODEL, OUT
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BIN=${PXA_LLAMA_SERVER:-$ROOT/build/bin/llama-server}
PORT=${1:-18796}
OUT=${OUT:-$(mktemp -d)}
MODEL=${MODEL:-$OUT/tiny_qwen4exp_mtp.gguf}   # MODEL=<a file from gen_tiny_qwen4exp.py --mtp> skips the generator (it needs numpy)
[ -x "$BIN" ] || { echo "FAIL: no server binary at $BIN"; exit 2; }
if [ ! -f "$MODEL" ]; then
    python3 "$HERE/gen_tiny_qwen4exp.py" "$MODEL" --mtp > "$OUT/gen.log" 2>&1 || { echo "FAIL: generator"; cat "$OUT/gen.log"; exit 2; }
fi
WARN='the MTP draft context could not be built'

run_arm() {   # run_arm NAME SPEC [ENV=VAL...] ; writes $OUT/NAME.txt (JSON) when the server came up, else no file
    local name=$1 spec=$2; shift 2
    rm -f "$OUT/$name.txt"
    env "$@" "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 0 -c 2048 -np 1 -t "${THREADS:-4}" --no-warmup --cache-ram 0 $spec \
        > "$OUT/$name.log" 2>&1 &
    local pid=$! up=0
    for _ in $(seq 1 90); do
        python3 -c "import sys,urllib.request; sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null && { up=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" = 1 ]; then
        python3 - "$PORT" "$OUT/$name.txt" <<'PY'
import json, sys, urllib.request
port, out = sys.argv[1:3]
res = []
for p in ["the quick brown fox jumps over the lazy dog ", "hello world, this is a tiny model test ", "0123456789 abcdefghij klmnop "]:
    r = urllib.request.Request("http://127.0.0.1:%s/completion" % port, json.dumps({"prompt": p, "n_predict": 64, "temperature": 0, "top_k": 1,
                               "seed": 1, "cache_prompt": False}).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    t = d.get("timings", {})
    res.append({"text": d["content"], "draft_n": t.get("draft_n")})
json.dump(res, open(out, "w"))
PY
    fi
    kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    return 0
}

RC=0
run_arm plain    "--spec-type none"
run_arm fallback "--spec-type mtp:n_max=3,p_min=0.0" PXA_MTP_INIT_FAULT=1
run_arm fatal    "--spec-type mtp:n_max=3,p_min=0.0" PXA_MTP_INIT_FAULT=1 PXA_MTP_INIT_FATAL=1

[ -f "$OUT/plain.txt" ]    || { echo "FAIL: the plain arm did not serve"; RC=1; }
[ -f "$OUT/fallback.txt" ] || { echo "FAIL: the fallback arm did not serve (the server exited on the refused MTP context)"; RC=1; }
[ -f "$OUT/fatal.txt" ]    && { echo "FAIL: PXA_MTP_INIT_FATAL=1 served instead of failing the boot"; RC=1; }
N=$(grep -c "$WARN" "$OUT/fallback.log")
[ "$N" -eq 1 ] || { echo "FAIL: the fallback arm logged the warning $N times (want 1)"; RC=1; }
grep -q "PXA_MTP_INIT_FAULT: the MTP context is refused" "$OUT/fallback.log" || { echo "FAIL: the test hook never fired"; RC=1; }
grep -q "server init failed" "$OUT/fatal.log" || { echo "FAIL: the fatal arm did not report 'server init failed'"; RC=1; }
[ "$(grep -c "$WARN" "$OUT/fatal.log")" -eq 0 ] || { echo "FAIL: the fatal arm logged the fallback warning"; RC=1; }
if [ -f "$OUT/plain.txt" ] && [ -f "$OUT/fallback.txt" ]; then
    python3 - "$OUT" <<'PY' || RC=1
import json, sys
o = sys.argv[1]
a, b = (json.load(open("%s/%s.txt" % (o, k))) for k in ("plain", "fallback"))
ok = True
for i in range(3):
    same = a[i]["text"] == b[i]["text"]
    nodraft = not b[i]["draft_n"]
    ok &= same and nodraft
    print("prompt %d: plain / fallback text %s, fallback drafted %s" % (i, "IDENTICAL" if same else "DIFFERENT", b[i]["draft_n"]))
sys.exit(0 if ok else 1)
PY
fi
[ "$RC" -eq 0 ] && echo "PASS mtp-init-fallback-tiny-test" || echo "FAIL mtp-init-fallback-tiny-test (logs in $OUT)"
exit $RC
