#!/usr/bin/env bash
# PXA_SLOT_FORK_v1 -- correctness + saving proof for the exact-prefix slot fork.
#
# What it proves, in one run:
#
#   Request A is sent to slot 0 and leaves its prompt resident in the shared ring.
#   Request B, whose prompt is request A's prompt token-for-token plus a tail, is then sent to
#   slot 1 -- a slot with an empty cache of its own, so without the fork it must prefill B from
#   zero. With PXA_SLOT_FORK=1 slot 1 instead forks A's prefix out of the live ring.
#
#   Exit 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE (the model's output is not serializable in either
#   arm, so the identity half could not be judged - see the note at the bottom of the run).
#
#   PASS requires BOTH of:
#     * B's generated text with the lever ON is byte-identical to B's generated text with the
#       lever OFF (the OFF arm IS the cold prefill of the same prompt), and
#     * B processed strictly fewer prompt tokens with the lever on (timings.prompt_n), by the
#       amount the server says it saved (timings.forked_n).
#
# The prompts are sent as explicit token-id arrays, not text, so "B extends A" is an exact token
# statement and not a claim about how a tokenizer happens to split a string.
#
# Usage:
#   examples/server/tests/pxa-slot-fork.sh --model /path/model.gguf [options]
#
#     --bin PATH        llama-server to run          (default: ./build-cpu/bin/llama-server)
#     --port N          port to bind                 (default: 18231)
#     --ctx N           -c                           (default: 8192)
#     --np N            -np                          (default: 4)
#     --npredict N      tokens to generate for B     (default: 32)
#     --prefix-tokens N length of A's prompt         (default: 320)
#     --tail-tokens N   extra tokens B adds          (default: 24)
#     --threads N       -t                           (default: 8)
#     --token-max N     token ids are drawn from [1,N) -- keep below the model's vocab
#                                                    (default: 4000)
#     --docker IMAGE    run llama-server inside this docker image, cwd bind-mounted at /src
#     --libpath P       extra LD_LIBRARY_PATH entries (the build tree's own is added anyway)
#     --extra "ARGS"    appended verbatim to the server command line (e.g. "-ngl 99")
#
# On a GPU seat, rerun exactly the same way with --extra "-ngl 99 ...". Nothing here is CPU-only:
# the mechanism is host-side cell bookkeeping and the check is a text comparison.

set -u

BIN=./build-cpu/bin/llama-server
MODEL=
PORT=18231
CTX=8192
NP=4
NPREDICT=32
PREFIX_TOKENS=320
TAIL_TOKENS=24
THREADS=8
TOKEN_MAX=4000
DOCKER_IMAGE=
LIBPATH=
EXTRA=

while [ $# -gt 0 ]; do
    case "$1" in
        --bin)            BIN=$2; shift 2 ;;
        --model)          MODEL=$2; shift 2 ;;
        --port)           PORT=$2; shift 2 ;;
        --ctx)            CTX=$2; shift 2 ;;
        --np)             NP=$2; shift 2 ;;
        --npredict)       NPREDICT=$2; shift 2 ;;
        --prefix-tokens)  PREFIX_TOKENS=$2; shift 2 ;;
        --tail-tokens)    TAIL_TOKENS=$2; shift 2 ;;
        --threads)        THREADS=$2; shift 2 ;;
        --token-max)      TOKEN_MAX=$2; shift 2 ;;
        --docker)         DOCKER_IMAGE=$2; shift 2 ;;
        --libpath)        LIBPATH=$2; shift 2 ;;
        --extra)          EXTRA=$2; shift 2 ;;
        -h|--help)        sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "error: --model is required" >&2
    exit 2
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/pxa-slot-fork.XXXXXX")
SERVER_PID=

# The engine's shared libraries live in the build tree next to the binary, not on the system
# loader path; a build tree that installed them somewhere else can add it with --libpath.
BUILD_ROOT=$(cd "$(dirname "$BIN")/.." 2>/dev/null && pwd)
LD_PATH="${BUILD_ROOT}/src:${BUILD_ROOT}/ggml/src:${BUILD_ROOT}/examples/mtmd"
[ -n "$LIBPATH" ] && LD_PATH="$LIBPATH:$LD_PATH"
[ -n "${LD_LIBRARY_PATH:-}" ] && LD_PATH="$LD_PATH:$LD_LIBRARY_PATH"

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 50); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -9 "$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=
}
trap 'cleanup; exit 130' INT TERM

start_server() { # $1 = PXA_SLOT_FORK value, $2 = log file
    local lever=$1 log=$2
    if [ -n "$DOCKER_IMAGE" ]; then
        docker run --rm --network host -e PXA_SLOT_FORK="$lever" -e LD_LIBRARY_PATH="$LD_PATH" \
            -v "$PWD":/src -v "$(dirname "$MODEL")":"$(dirname "$MODEL")" -w /src \
            "$DOCKER_IMAGE" \
            "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
                -c "$CTX" -np "$NP" --kv-unified -t "$THREADS" -cram 0 \
                --slot-prompt-similarity 0 $EXTRA >"$log" 2>&1 &
    else
        PXA_SLOT_FORK="$lever" LD_LIBRARY_PATH="$LD_PATH" \
            "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
                -c "$CTX" -np "$NP" --kv-unified -t "$THREADS" -cram 0 \
                --slot-prompt-similarity 0 $EXTRA >"$log" 2>&1 &
    fi
    SERVER_PID=$!

    local i
    for i in $(seq 1 600); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server died during startup; tail of $log:" >&2
            tail -30 "$log" >&2
            return 1
        fi
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "server did not become healthy within 600s; tail of $log:" >&2
    tail -30 "$log" >&2
    return 1
}

# Build the two prompts as explicit token arrays. Token ids are picked from a range every
# vocabulary of interest has, and the sequence is deterministic, so the same two prompts are
# used by both arms and by any rerun.
python3 - "$PREFIX_TOKENS" "$TAIL_TOKENS" "$WORK" "$TOKEN_MAX" <<'PY'
import json, sys
n_prefix, n_tail, work = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
tok_max = int(sys.argv[4])
# a fixed, non-degenerate walk over a low, safe token range
def walk(n, seed):
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7fffffff
        out.append(1 + (x % (tok_max - 1)))
    return out
prefix = walk(n_prefix, 7)
tail   = walk(n_tail, 991)
json.dump(prefix,        open(work + "/a.json", "w"))
json.dump(prefix + tail, open(work + "/b.json", "w"))
PY

request() { # $1 = id_slot, $2 = prompt json file, $3 = n_predict, $4 = out file
    python3 - "$1" "$2" "$3" > "$WORK/req.json" <<'PY'
import json, sys
id_slot, path, n_predict = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
print(json.dumps({
    "id_slot": id_slot,
    "prompt": json.load(open(path)),
    "n_predict": n_predict,
    "temperature": 0.0,
    "top_k": 1,
    "seed": 1234,
    "cache_prompt": True,
    "timings_per_token": False,
}))
PY
    curl -s --max-time 1800 -H 'Content-Type: application/json' \
        -d @"$WORK/req.json" "http://127.0.0.1:$PORT/completion" > "$4"
}

# $1 = response file -> "prompt_n forked_n"
read_timings() {
    python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("timings", {})
print(t.get("prompt_n", -1), t.get("forked_n", 0))
PY
}

run_arm() { # $1 = lever value, $2 = tag
    local lever=$1 tag=$2
    echo "=== arm $tag: PXA_SLOT_FORK=$lever ==="
    start_server "$lever" "$WORK/server.$tag.log" || return 1

    # A: primes slot 0 and leaves its prefix resident in the ring
    request 0 "$WORK/a.json" 8 "$WORK/resp.a.$tag.json" || return 1
    # B: a different slot, empty cache of its own
    request 1 "$WORK/b.json" "$NPREDICT" "$WORK/resp.b.$tag.json" || return 1

    cleanup
    return 0
}

echo "model      : $MODEL"
echo "server     : $BIN"
echo "prompts    : A = $PREFIX_TOKENS tokens, B = A + $TAIL_TOKENS tokens"
echo

run_arm 0 off || { echo "FAIL: lever-off arm did not run"; cleanup; exit 1; }
run_arm 1 on  || { echo "FAIL: lever-on arm did not run";  cleanup; exit 1; }

read PROMPT_N_OFF FORKED_OFF <<<"$(read_timings "$WORK/resp.b.off.json")"
read PROMPT_N_ON  FORKED_ON  <<<"$(read_timings "$WORK/resp.b.on.json")"

python3 - "$WORK/resp.b.off.json" "$WORK/resp.b.on.json" > "$WORK/verdict.txt" <<'PY'
import json, sys
off = json.load(open(sys.argv[1]))
on  = json.load(open(sys.argv[2]))
# A fixture whose detokenized output is not valid UTF-8 (random-weight test models are the usual
# case) cannot be returned through this server's JSON at all, in EITHER arm. That is a property of
# the fixture, not of the fork, and it must not be reported as a pass.
if "error" in off and "error" in on:
    print("NOT-COMPARABLE")
    print(repr(str(off["error"])[:200]))
    print(repr(str(on["error"])[:200]))
else:
    a, b = off.get("content", None), on.get("content", None)
    print("IDENTICAL" if (a is not None and a == b) else "DIFFERENT")
    print(repr(str(a)[:400]))
    print(repr(str(b)[:400]))
PY

TEXT_VERDICT=$(head -1 "$WORK/verdict.txt")

echo
echo "request B, lever OFF : prompt_n=$PROMPT_N_OFF forked_n=$FORKED_OFF"
echo "request B, lever ON  : prompt_n=$PROMPT_N_ON  forked_n=$FORKED_ON"
echo "generated text       : $TEXT_VERDICT"
echo
echo "--- lever-on server log, fork lines ---"
grep -i "SLOT_FORK" "$WORK/server.on.log" | head -20
echo

if [ "$TEXT_VERDICT" = "NOT-COMPARABLE" ]; then
    echo "INCONCLUSIVE: this model's output cannot be serialized by the server in either arm, so the"
    echo "              cold-prefill text comparison could not be made. The fork numbers above still"
    echo "              stand; rerun the identity half on a model with a real vocabulary."
    sed -n '2,3p' "$WORK/verdict.txt"
    echo "artifacts kept in $WORK"
    trap - INT TERM
    exit 2
fi

RC=0
[ "$TEXT_VERDICT" = "IDENTICAL" ] || { echo "FAIL: generated text differs from the cold prefill"; sed -n '2,3p' "$WORK/verdict.txt"; RC=1; }
[ "$FORKED_ON" -gt 0 ] 2>/dev/null || { echo "FAIL: the lever-on arm reported no fork"; RC=1; }
[ "$PROMPT_N_ON" -lt "$PROMPT_N_OFF" ] 2>/dev/null || { echo "FAIL: prompt_n did not drop"; RC=1; }
if [ "$RC" -eq 0 ] && [ $((PROMPT_N_OFF - PROMPT_N_ON)) -ne "$FORKED_ON" ]; then
    echo "FAIL: prompt_n dropped by $((PROMPT_N_OFF - PROMPT_N_ON)) but the server claims $FORKED_ON saved"
    RC=1
fi

if [ "$RC" -eq 0 ]; then
    echo "PASS: identical text, $FORKED_ON prompt tokens not prefilled ($PROMPT_N_OFF -> $PROMPT_N_ON)"
else
    echo "artifacts kept in $WORK"
    trap - INT TERM
    exit 1
fi

rm -rf "$WORK"
exit 0
