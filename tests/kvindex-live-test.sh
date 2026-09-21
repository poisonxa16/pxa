#!/usr/bin/env bash
# PXA_KV_SEQ_RM_INDEX live-server test.
#
# The unit test (tests/test-kv-seq-rm-index.cpp) proves the indexed removal against the scan on a
# stand-in cache. This drives the REAL server through the sequence that broke on the V100 pair:
# a hybrid model (checkpoints only exist for recurrent/hybrid arches -- see server-context.cpp
# "if (llama_model_has_recurrent(...)) params_base.do_checkpoint = ..."), two slots on one unified
# ring, a prompt cache small enough to hit its ceiling and start evicting, and prompts that are
# distinct from their first token so every admission goes
#
#     prompt cache load -> apply_checkpoint -> kv cache rm [0, end) -> prefill
#       -> create_checkpoint (tolerance = 5 tokens before the end of the prompt)
#       -> kv cache rm [n_prompt - 5, end) -> decode
#
# which is exactly the sequence in the unify gate log (server-unify-def.log:2412-2419) that
# produced 153 PXA_SAMPLE_SOFTFAIL_v1 lines with the index on and 0 with it off.
#
# Usage:  live_kvindex_test.sh <on|off|check> [port] [outdir]
#   on    -> PXA_KV_SEQ_RM_INDEX=1                       (the arm under test)
#   off   -> PXA_KV_SEQ_RM_INDEX=0                       (the reference; its answers are the gate)
#   check -> PXA_KV_SEQ_RM_INDEX=1 PXA_KV_INDEX_CHECK=1  (verifies the index against a rebuild on
#                                                         every indexed removal; names any drift)
#
# Writes <outdir>/<mode>-server.log and <outdir>/<mode>-answers.txt, prints a one-line verdict, and
# exits non-zero if the run is not clean. CPU only, no GPU, no model download.

set -u

MODE=${1:-on}
PORT=${2:-18731}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${3:-$ROOT/build-cpu/kvindex-out}

BIN=${PXA_LLAMA_SERVER:-$ROOT/build-cpu/bin/llama-server}
# arch qwen35 == hybrid (src/llama-arch.cpp llm_arch_is_hybrid), which is what turns context
# checkpoints on at all. A dense model never creates one and cannot reproduce this.
MODEL=${PXA_KVINDEX_MODEL:?set to a small hybrid GGUF, e.g. Qwen3.5-0.8B-Q8_0.gguf}

N_PROMPTS=${N_PROMPTS:-30}
N_ROUNDS=${N_ROUNDS:-2}     # each round sends all N_PROMPTS; round 2 is where the cache is full
N_CHECK=${N_CHECK:-5}       # the graded answers at the end
N_PREDICT=${N_PREDICT:-32}
CRAM=${CRAM:-192}           # MiB. small on purpose: the ceiling has to be reached
THREADS=${THREADS:-12}
# 2 = both slots busy at once, which is the workload. 1 = one request at a time, which is the only
# way two arms execute the identical logical sequence: with two slots in flight the interleaving,
# and therefore the prompt-cache eviction order, is a matter of timing, and on a hybrid a different
# eviction order is a different re-entry point and can be a different token. Use CONC=1 for any
# comparison between arms; use the default for the stress.
CONC=${CONC:-2}

case "$MODE" in
    on)    export PXA_KV_SEQ_RM_INDEX=1; unset PXA_KV_INDEX_CHECK ;;
    off)   export PXA_KV_SEQ_RM_INDEX=0; unset PXA_KV_INDEX_CHECK ;;
    check) export PXA_KV_SEQ_RM_INDEX=1; export PXA_KV_INDEX_CHECK=1 ;;
    *) echo "usage: $0 <on|off|check> [port] [outdir]" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
LOG=$OUT/$MODE-server.log
ANS=$OUT/$MODE-answers.txt
COLD=$OUT/$MODE-cold-answers.txt
: > "$ANS"
: > "$COLD"

if [ ! -x "$BIN" ];   then echo "missing server binary: $BIN"   >&2; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "missing model: $MODEL"         >&2; exit 2; fi

# never talk to somebody else's server: the box runs seats on nearby ports and the engine's own
# PXA_PORT_GUARD refuses to boot on a busy one, which would leave this script grading a stranger.
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "port $PORT is already answering - pick another one" >&2; exit 2
fi

echo "== kvindex live test: mode=$MODE port=$PORT model=$(basename "$MODEL")"
echo "   PXA_KV_SEQ_RM_INDEX=${PXA_KV_SEQ_RM_INDEX} PXA_KV_INDEX_CHECK=${PXA_KV_INDEX_CHECK:-unset}"

"$BIN" -m "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    -c 8192 -np 2 --kv-unified \
    -ngl 0 -t "$THREADS" -b 512 -ub 512 \
    --ctx-checkpoints 32 \
    -cram "$CRAM" \
    --no-warmup \
    > "$LOG" 2>&1 &
SRV=$!
# only ever kill the pid we started (never pkill/pgrep -f)
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
# prompts: every prompt DIVERGES AT ITS FIRST WORD. That is deliberate. A shared preamble makes the
# server take the "Common part does not match fully" -> "forcing full prompt re-processing" branch,
# which is a different admission path with a defect of its own (tracked separately); the V100
# run this reproduces took neither branch (0 occurrences of both lines in the unify gate log) and
# admitted every request with kv cache rm p0=0.
# Distinct first tokens are what keep this test on that same path.
OPENERS=(alabaster brimstone cinnabar dulcimer effigy filament gossamer harrow ingot jubilee \
         kestrel lantern marzipan nocturne obelisk parapet quarry ravine sextant tarragon \
         umbra verdigris windlass xylem yarrow zephyr amber basalt clover dovetail)

BODY="Explain, in one short sentence and with no preamble or lists, what a transformer inference \
server must do to the key/value cache of one sequence when a request is admitted into a slot whose \
ring already holds cells belonging to another sequence, and why the order of that work matters for \
the positions the attention mask can still read."

ask() { # ask <prompt-text> <n_predict> [cache_prompt] -> prints the completion content on one line
    local p=$1 n=$2 c=${3:-1}
    python3 - "$p" "$n" "$PORT" "$c" <<'PY'
import json, sys, urllib.request
prompt, n, port, cache = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
body = json.dumps({
    "prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    "seed": 1234, "cache_prompt": cache, "stream": False,
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

# PXA_KVINDEX_SHARED_PREFIX=1 switches to the OTHER prompt shape: one long shared preamble followed
# by a per-request tail. That sends every admission down "Common part does not match fully" ->
# apply_checkp "forcing full prompt re-processing", which is a different defect and NOT this test's
# subject -- it produces non-finite logits with PXA_KV_SEQ_RM_INDEX=0 as well. Kept here, exactly as
# it was when it first did so, so that anyone can reproduce it from this same script.
SHARED_PREFIX=${PXA_KVINDEX_SHARED_PREFIX:-0}

PREAMBLE="You are a careful technical assistant. Answer in one short sentence, with no preamble and no lists. \
Keep the wording plain, avoid adjectives, and never repeat the question back. The reader is an engineer who \
already knows the domain and wants the shortest correct statement of fact, so prefer a single clause over a \
paragraph and never pad the answer. Treat every question below as independent of the others."

topic() { # topic <i> -> a whole prompt whose FIRST token is unique to i
    local i=$1
    if [ "$SHARED_PREFIX" = "1" ]; then
        echo "${PREAMBLE} Question ${i}: in a transformer inference server, what does step number ${i} of the KV cache admission \
path do when a slot re-enters a cached prompt whose common prefix is ${i} tokens long and the ring already holds \
${i}00 cells for another sequence?"
        return
    fi
    local w=${OPENERS[$(( (i - 1) % ${#OPENERS[@]} ))]}
    echo "${w}. ${BODY} Answer for case ${w}, numbered ${i}, and mention ${w} exactly once."
}

echo "-- warming the prompt cache: $N_ROUNDS x $N_PROMPTS requests ($CONC concurrent)"
for round in $(seq 1 "$N_ROUNDS"); do
    i=1
    while [ "$i" -le "$N_PROMPTS" ]; do
        if [ "$CONC" = "1" ]; then
            ask "$(topic "$i")" "$N_PREDICT" > "$OUT/.warm.$i"
            i=$((i + 1))
            continue
        fi
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
    echo "   round $round done ($(grep -c 'cache size limit reached' "$LOG" 2>/dev/null || echo 0) prompt-cache evictions so far)"
done
rm -f "$OUT"/.warm.*

echo "-- graded answers: $N_CHECK greedy completions on cached prompts"
for k in $(seq 1 "$N_CHECK"); do
    idx=$(( (k * 7) % N_PROMPTS + 1 ))
    ans=$(ask "$(topic "$idx")" "$N_PREDICT" 1)
    printf 'check %d (prompt %d): %s\n' "$k" "$idx" "$ans" >> "$ANS"
done

# The cached answers above prove the server still answers after all that churn, but they are NOT a
# cross-arm identity gate: they re-enter a cached prompt, and on a hybrid the re-entry point depends
# on the eviction order, which two concurrent slots do not reproduce exactly from run to run. A
# single-token difference between two clean arms is that, not a divergence.
#
# COLD is the gate. One request at a time, cache_prompt off, so the only thing that can change the
# answer is the state of the shared ring the run left behind. Byte-identical across arms or the run
# is not clean.
echo "-- cold answers: $N_CHECK greedy completions with the prompt cache bypassed"
for k in $(seq 1 "$N_CHECK"); do
    idx=$(( (k * 11) % N_PROMPTS + 1 ))
    ans=$(ask "$(topic "$idx")" "$N_PREDICT" 0)
    printf 'cold %d (prompt %d): %s\n' "$k" "$idx" "$ans" >> "$COLD"
done

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; trap - EXIT

# ---------------------------------------------------------------------------------------------
# verdict

n_softfail=$(grep -c 'PXA_SAMPLE_SOFTFAIL' "$LOG" || true)
n_breaker=$(grep -c 'PXA_SOFTFAIL_BREAKER' "$LOG" || true)
n_drift=$(grep -c 'KV seq index drift' "$LOG" || true)
n_trunc=$(grep -c 'truncated=true' "$LOG" || true)
n_evict=$(grep -c 'cache size limit reached' "$LOG" || true)
n_force=$(grep -c 'forcing full prompt re-processing' "$LOG" || true)
n_ckpt=$(grep -c 'created context checkpoint' "$LOG" || true)
n_restore=$(grep -c 'restored .*context checkpoint' "$LOG" || true)
n_rm0=$(grep -c 'kv cache rm' "$LOG" || true)
n_fail=$(grep -c 'REQUEST FAILED' "$ANS" || true)
n_empty=$(grep -c ': ""$' "$ANS" || true)
n_fail=$(( n_fail + $(grep -c 'REQUEST FAILED' "$COLD" || true) ))
n_empty=$(( n_empty + $(grep -c ': ""$' "$COLD" || true) ))
cold_sum=$(md5sum < "$COLD" | cut -d' ' -f1)

echo "== $MODE: softfail=$n_softfail breaker=$n_breaker index_drift=$n_drift truncated=$n_trunc"
echo "   exercised: kv_cache_rm=$n_rm0 checkpoints_created=$n_ckpt checkpoints_restored=$n_restore prompt_cache_evictions=$n_evict"
echo "   answers:   failed=$n_fail empty=$n_empty  ($ANS)"
echo "   COLD GATE: md5 $cold_sum  ($COLD)  -- must match the off arm's exactly"
echo "   off-path:  forcing_full_reprocess=$n_force (must be 0: that branch is a different defect)"

rc=0
[ "$n_softfail" -eq 0 ] || rc=1
[ "$n_breaker"  -eq 0 ] || rc=1
[ "$n_drift"    -eq 0 ] || rc=1
[ "$n_fail"     -eq 0 ] || rc=1
[ "$n_empty"    -eq 0 ] || rc=1
# the sequence under test has to have actually happened, or a clean run means nothing
[ "$n_ckpt"     -gt 0 ] || { echo "   NOT EXERCISED: no context checkpoint was ever created"; rc=2; }
[ "$n_evict"    -gt 0 ] || { echo "   NOT EXERCISED: the prompt cache never reached its ceiling"; rc=2; }
if [ "$SHARED_PREFIX" != "1" ]; then
    [ "$n_force" -eq 0 ] || { echo "   OFF PATH: the run hit 'forcing full prompt re-processing' -- prompts are sharing a prefix"; rc=2; }
fi

if [ "$rc" -eq 0 ]; then echo "== $MODE: CLEAN"; else echo "== $MODE: NOT CLEAN (rc=$rc)"; fi
exit "$rc"
