#!/usr/bin/env bash
# Hybrid re-entry NaN test (PXA_HYBRID_REENTRY).
#
# Reproduces, on CPU in about six minutes, the failure a hybrid (recurrent + attention) seat shows
# when two slots share one unified ring, the prompt cache is on, and consecutive requests SHARE A
# LONG PREFIX. The shared prefix makes a slot re-enter its own cached prompt part-way
# ("Common part does not match fully"), which asks apply_checkpoint for a roll-back; when no
# checkpoint is old enough the slot takes the "forcing full prompt re-processing" branch. The
# server then reports non-finite logits (PXA_SAMPLE_SOFTFAIL / PXA_SOFTFAIL_BREAKER) and truncates
# the release.
#
# The model choice is what makes this possible on CPU: context checkpoints exist ONLY for
# recurrent/hybrid architectures (server-context.cpp only sets params_base.do_checkpoint inside
# "if (llama_model_has_recurrent(...))"), so a dense model never creates one and can never take
# this branch. Arch qwen35 IS hybrid (src/llama-arch.cpp llm_arch_is_hybrid), which makes the 0.8B
# Qwen3.5 draft model a complete, CPU-sized stand-in for the 27B seat.
#
# Usage:  hybrid-reentry-nan-test.sh [on|off] [port] [outdir]
#   the argument only sets PXA_KV_SEQ_RM_INDEX (default off == the shipped default path);
#   the defect under test is on the default path and does not need the index at all.
#
# The failure also needs the two slots to OVERLAP: the poisoned batch is one slot re-processing
# from position 0 while the other is mid-generation, so its cells are placed above the other
# sequence's live band instead of at index 0. Two rounds of 30 paired requests is enough to reach
# that interleaving on a quiet box; a shorter sweep (12 requests) drives the branch 11 times and
# stays clean, so do not trim N_PROMPTS/N_ROUNDS to make the test faster.
#
# Exits 0 only if the forced re-processing branch WAS exercised and the run produced no non-finite
# logit, no breaker and no failed/empty answer.

set -u

MODE=${1:-off}
PORT=${2:-18741}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${3:-$ROOT/build-cpu/hybrid-reentry-out}

BIN=${PXA_LLAMA_SERVER:-$ROOT/build-cpu/bin/llama-server}
MODEL=${PXA_HYBRID_MODEL:?set to a small hybrid GGUF, e.g. Qwen3.5-0.8B-Q8_0.gguf}

N_PROMPTS=${N_PROMPTS:-30}
N_ROUNDS=${N_ROUNDS:-2}
N_CHECK=${N_CHECK:-5}
N_PREDICT=${N_PREDICT:-32}
CRAM=${CRAM:-192}
NCTX=${NCTX:-8192}       # shrink this to put the shared unified ring under pressure
CRS=${CRS:-0.99}          # -crs: keep the prompt cache ACTIVE even for shared-prefix traffic
THREADS=${THREADS:-12}
# PREAMBLE_REPEAT is the prompt LENGTH knob, and the length is part of the trigger, not a taste:
# 1 gives ~165-token prompts, which is the shape that produced the non-finite logits (the slot's
# whole cached prompt fits under one 256-cell attention window, so the re-processed band lands
# outside it). Repeating the preamble 6 times gives ~560-token prompts, which drives the same
# "forcing full prompt re-processing" branch just as often and stays CLEAN -- two full runs, 30
# forced re-processings, 14 cache evictions, 0 softfail. Do not raise this default.
PREAMBLE_REPEAT=${PREAMBLE_REPEAT:-1}  # how many times the shared preamble is repeated (prompt length)

case "$MODE" in
    on)  export PXA_KV_SEQ_RM_INDEX=1 ;;
    off) export PXA_KV_SEQ_RM_INDEX=0 ;;
    *) echo "usage: $0 <on|off> [port] [outdir]" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
LOG=$OUT/$MODE-server.log
ANS=$OUT/$MODE-answers.txt
: > "$ANS"

if [ ! -x "$BIN" ];   then echo "missing server binary: $BIN"   >&2; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "missing model: $MODEL"         >&2; exit 2; fi

if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "port $PORT is already answering - pick another one" >&2; exit 2
fi

echo "== hybrid re-entry NaN test: index=$MODE port=$PORT model=$(basename "$MODEL")"

"$BIN" -m "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    -c "$NCTX" -np 2 --kv-unified \
    -ngl 0 -t "$THREADS" -b 512 -ub 512 \
    --ctx-checkpoints 32 \
    -cram "$CRAM" -crs "$CRS" \
    --no-warmup \
    > "$LOG" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null' EXIT

for _ in $(seq 1 180); do
    if ! kill -0 "$SRV" 2>/dev/null; then echo "server died during boot, see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "server never became healthy, see $LOG" >&2; tail -20 "$LOG" >&2; exit 1
fi

# ---------------------------------------------------------------------------------------------
# prompts: a LONG SHARED PREAMBLE followed by a short unique tail. That is the whole point: the
# slot's cached prompt and the next request agree for hundreds of tokens and then diverge, which
# is the re-entry the server sees from a chat client resending a conversation, a client replaying a
# system prompt, or an eval harness sweeping one template.
PREAMBLE="You are a careful technical assistant. Answer in one short sentence, with no preamble and no lists. \
Keep the wording plain, avoid adjectives, and never repeat the question back. The reader is an engineer who \
already knows the domain and wants the shortest correct statement of fact, so prefer a single clause over a \
paragraph and never pad the answer. Treat every question below as independent of the others."

PREAMBLE_ONE=$PREAMBLE
for _r in $(seq 2 "$PREAMBLE_REPEAT"); do PREAMBLE="$PREAMBLE $PREAMBLE_ONE"; done

ask() {
    local p=$1 n=$2
    python3 - "$p" "$n" "$PORT" <<'PY'
import json, sys, urllib.request
prompt, n, port = sys.argv[1], int(sys.argv[2]), sys.argv[3]
body = json.dumps({
    "prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    "seed": 1234, "cache_prompt": True, "stream": False,
}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    print(json.dumps(out.get("content", "")))
except Exception as e:
    print(json.dumps("<<REQUEST FAILED: %s>>" % e))
PY
}

# topic <i>: the shared preamble, then a tail in which the request number appears THREE times with
# identical wording in between. The prompts therefore agree for the whole preamble, diverge at the
# number, and RE-CONVERGE after it -- and because "1", "10" and "100" tokenize to different lengths,
# the slot's cached-token index and the prompt-token index of the common prefix need not be the same
# number. That is the shape that first produced non-finite logits.
topic() {
    local i=$1
    echo "${PREAMBLE} Question ${i}: in a transformer inference server, what does step number ${i} of the KV cache admission \
path do when a slot re-enters a cached prompt whose common prefix is ${i} tokens long and the ring already holds \
${i}00 cells for another sequence?"
}

echo "-- driving re-entry: $N_ROUNDS x $N_PROMPTS requests (2 concurrent)"
for round in $(seq 1 "$N_ROUNDS"); do
    i=1
    while [ "$i" -le "$N_PROMPTS" ]; do
        j=$((i + 1))
        ask "$(topic "$i")" "$N_PREDICT" > "$OUT/.warm.$i" &
        a=$!
        if [ "$j" -le "$N_PROMPTS" ]; then
            ask "$(topic "$j")" "$N_PREDICT" > "$OUT/.warm.$j" &
            b=$!
        else
            b=""
        fi
        wait "$a"; [ -n "$b" ] && wait "$b"
        i=$((i + 2))
    done
    echo "   round $round done (softfail so far: $(grep -c 'PXA_SAMPLE_SOFTFAIL' "$LOG" 2>/dev/null || echo 0))"
done
rm -f "$OUT"/.warm.*

echo "-- graded answers: $N_CHECK greedy completions"
for k in $(seq 1 "$N_CHECK"); do
    idx=$(( (k * 7) % N_PROMPTS + 1 ))
    ans=$(ask "$(topic "$idx")" "$N_PREDICT")
    printf 'check %d (prompt %d): %s\n' "$k" "$idx" "$ans" >> "$ANS"
done

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; trap - EXIT

# ---------------------------------------------------------------------------------------------
n_softfail=$(grep -c 'PXA_SAMPLE_SOFTFAIL' "$LOG" || true)
n_breaker=$(grep -c 'PXA_SOFTFAIL_BREAKER' "$LOG" || true)
n_trunc=$(grep -c 'truncated=true' "$LOG" || true)
n_force=$(grep -c 'forcing full prompt re-processing' "$LOG" || true)
n_common=$(grep -c 'Common part does not match fully' "$LOG" || true)
n_ckpt=$(grep -c 'created context checkpoint' "$LOG" || true)
n_restore=$(grep -c 'restored .*context checkpoint' "$LOG" || true)
n_evict=$(grep -c 'cache size limit reached' "$LOG" || true)
n_fail=$(grep -c 'REQUEST FAILED' "$ANS" || true)
n_empty=$(grep -c ': ""$' "$ANS" || true)

echo "== index=$MODE: softfail=$n_softfail breaker=$n_breaker truncated=$n_trunc"
echo "   exercised: common_part_mismatch=$n_common forced_reprocess=$n_force checkpoints_created=$n_ckpt restored=$n_restore evictions=$n_evict"
echo "   answers:   failed=$n_fail empty=$n_empty  ($ANS)"

rc=0
[ "$n_softfail" -eq 0 ] || rc=1
[ "$n_breaker"  -eq 0 ] || rc=1
[ "$n_trunc"    -eq 0 ] || rc=1
[ "$n_fail"     -eq 0 ] || rc=1
[ "$n_empty"    -eq 0 ] || rc=1
[ "$n_force"    -gt 0 ] || { echo "   NOT EXERCISED: the forced full re-processing branch never ran"; rc=2; }

if [ "$rc" -eq 0 ]; then echo "== index=$MODE: CLEAN"; else echo "== index=$MODE: NOT CLEAN (rc=$rc)"; fi
exit "$rc"
