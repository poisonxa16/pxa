#!/usr/bin/env bash
# --dump-mtp-embd <dir> creates <dir> without a shell (4.0.1 security review L1: it ran system("mkdir -p '<dir>'"), so a quote in
# the value ran a command). A tiny random-weight qwen4exp GGUF (tests/gen_tiny_qwen4exp.py) boots on the CPU with a dump path
# that carries a quote + a command; the server must come up, create the path literally, and the command must not run.
# Usage: mtp-dump-dir-noshell-test.sh [port] ; env PXA_LLAMA_SERVER, THREADS (default 4)
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN=${PXA_LLAMA_SERVER:-$HERE/../build/bin/llama-server}
PORT=${1:-18797}
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
[ -x "$BIN" ] || { echo "FAIL: no server binary at $BIN"; exit 2; }
python3 "$HERE/gen_tiny_qwen4exp.py" "$OUT/tiny.gguf" > "$OUT/gen.log" 2>&1 || { echo "FAIL: generator"; cat "$OUT/gen.log"; exit 2; }
DUMP="$OUT/dump'; touch '$OUT/INJECTED'; echo '"
"$BIN" -m "$OUT/tiny.gguf" --host 127.0.0.1 --port "$PORT" -ngl 0 -c 512 -np 1 -t "${THREADS:-4}" --no-warmup --cache-ram 0 \
    --dump-mtp-embd "$DUMP" > "$OUT/server.log" 2>&1 &
pid=$! up=0
for _ in $(seq 1 90); do
    python3 -c "import sys,urllib.request; sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
RC=0
[ "$up" = 1 ] || { echo "FAIL: the server did not come up"; tail -20 "$OUT/server.log"; RC=1; }
if [ -e "$OUT/INJECTED" ]; then echo "FAIL: the --dump-mtp-embd value ran as a shell command"; RC=1; fi
[ -d "$DUMP" ] || { echo "FAIL: the dump directory was not created literally"; RC=1; }
[ "$RC" = 0 ] && echo "PASS: dump directory created without a shell; injected command did not run"
exit $RC
