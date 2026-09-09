#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# PXA release gate. One command, no box-specific paths, no container assumptions.
#
#   MODEL=/path/to/model.gguf ./bench/gate/run-gate.sh
#
# Exits 0 only if every check passes. See bench/gate/README.md for what each check proves and
# for the two rules that make the multi-slot arm meaningful.
# ---------------------------------------------------------------------------------------------
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)

# ---- configuration (every one of these is overridable from the environment) -------------------
MODEL=${MODEL:-}                       # required: the .gguf under test
BIN=${BIN:-}                           # directory holding llama-server and the test binaries
GPUS=${GPUS:-}                         # e.g. "2,4"; empty = leave CUDA_VISIBLE_DEVICES alone
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18080}
NGL=${NGL:-99}
CTX=${CTX:-32768}
BATCH=${BATCH:-2048}
UBATCH=${UBATCH:-2048}
THREADS=${THREADS:-$( (nproc 2>/dev/null || echo 8) )}
FA=${FA:-on}                           # "" to omit the flag entirely
CHAT_JINJA=${CHAT_JINJA:-on}           # "" to omit --jinja (a server too old to have it needs this)
SERVER_ARGS=${SERVER_ARGS:-}           # anything extra (-ts, -ot, --kv-unified, ...)
REPS=${REPS:-12}                       # determinism repetitions per arm
NEEDLE_REPS=${NEEDLE_REPS:-4}
LOGIT_REPS=${LOGIT_REPS:-6}      # logit-reproducibility repetitions (0 disables)
CHAT_REPS=${CHAT_REPS:-4}              # /v1/chat/completions determinism repetitions
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
REQ_TIMEOUT=${REQ_TIMEOUT:-900}
N_PREDICT=${N_PREDICT:-256}   # thinking models spend the first ~100 tokens inside <think>; 32 truncated every needle answer on stock Qwen3.8
GATE_NP2=${GATE_NP2:-auto}             # auto | 1 | 0
NP2_CROSS_CHECK=${NP2_CROSS_CHECK:-1}  # compare the np=2 slot against the np=1 reference
GATE_TESTS=${GATE_TESTS:-1}
COHERENCE_EXPECT=${COHERENCE_EXPECT:-paris}
# GATE_STRICT (fail-closed, 2026-09-08): a release gate may SKIP an arm only when the hardware/
# build genuinely cannot run it (no -np flag, no slot-erase route, no completion_probabilities).
# It must never SKIP just because a knob quietly turned a required arm off. Under GATE_STRICT=1,
# every skc() call below -- the operator-choice disables (LOGIT_REPS=0, GATE_NP2=0, GATE_TESTS=0)
# -- becomes a FAIL instead of a SKIP; genuine hardware/build-absence (plain sk() calls) still
# SKIPs even in strict mode, because failing a CPU-only box for lacking a GPU proves nothing. The
# required-check CI workflow always runs with GATE_STRICT=1; leave it 0 for local/dev iteration.
GATE_STRICT=${GATE_STRICT:-0}
PROMPTS=${PROMPTS:-$HERE/prompts}
WORKDIR=${WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/pxa-gate.XXXXXX")}
# An externally supplied WORKDIR is not created by mktemp, and the FIRST thing the run does is
# redirect the np=1 server log into it -- a missing directory failed that redirect and reported
# "server would not boot at -np 1", i.e. a harness artefact wearing an engine failure's clothes
# (2026-09-06: the np=2 arm passed in the same run only because it mkdir -p's WORKDIR/slots).
mkdir -p "$WORKDIR" || { printf 'gate: cannot create WORKDIR %s\n' "$WORKDIR" >&2; exit 2; }   # die() is defined below

pass=0; fail=0; skip=0; SRV_PID=; SKIPPED=()
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
ok(){   printf '  PASS  %s\n' "$*"; pass=$((pass+1)); }
ko(){   printf '  FAIL  %s\n' "$*"; fail=$((fail+1)); }
# sk: a SKIP for a reason this gate stands behind even under GATE_STRICT -- the hardware or build
# genuinely cannot run the check (no -np flag, no slot-erase route, no completion_probabilities,
# no /v1/chat/completions route). Recorded by name for the final summary.
sk(){   printf '  SKIP  %s\n' "$*"; skip=$((skip+1)); SKIPPED+=("$*"); }
# skc: a SKIP for an *operator choice* (a knob turned a required arm off), never a hardware gap.
# Fine for local/dev iteration; a FAIL under GATE_STRICT=1, because a release gate is not allowed
# to quietly ship on a weaker bar than its own defaults -- see GATE_STRICT above.
skc(){
    if [ "$GATE_STRICT" = 1 ]; then
        ko "$* (GATE_STRICT=1: an operator-disabled arm is a FAIL, not a SKIP, in a release gate)"
    else
        sk "$*"
    fi
}
die(){  printf 'gate: %s\n' "$*" >&2; exit 2; }

# ---- resolve BIN -----------------------------------------------------------------------------
if [ -z "$BIN" ]; then
    for d in "$ROOT/build/bin" "$ROOT/build-spd/bin" "$ROOT/build-tok/bin" "$ROOT/build/Release/bin"; do
        [ -x "$d/llama-server" ] && { BIN=$d; break; }
    done
fi

# ---- --plan: dry-run preview, no server boot, no MODEL required -------------------------------
# Lists every arm this invocation would attempt and every SKIP this exact environment/build would
# produce, WITHOUT booting a server -- everything statically knowable (env config, GPU presence,
# llama-server --help, which test binaries exist) is checked for real; the handful of things only
# a live server can answer (does this build serve /v1/chat/completions, does it return
# completion_probabilities) are marked as runtime-only. Useful to prove the wiring is sound, and
# to see the fail-closed SKIP-vs-FAIL split GATE_STRICT=1 would apply, on a box with no GPU and no
# model handy.
GATE_PLAN=${GATE_PLAN:-0}
for a in "$@"; do [ "$a" = "--plan" ] && GATE_PLAN=1; done
if [ "$GATE_PLAN" = 1 ]; then
    printf 'gate --plan (dry run, no server boots) BIN=%s MODEL=%s GATE_STRICT=%s\n\n' \
        "${BIN:-<not found>}" "${MODEL:-<unset -- not required for --plan>}" "$GATE_STRICT"
    row(){ printf '  %-6s %-58s %s\n' "$1" "$2" "$3"; }  # verdict, arm, reason
    knob_row(){  # knob_row <arm> <disabled 0|1> <knob=val> -- the operator-choice knobs
        if [ "$2" = 1 ]; then
            [ "$GATE_STRICT" = 1 ] && row FAIL "$1" "$3 (GATE_STRICT=1: operator-disabled -> FAIL, not SKIP)" \
                                    || row SKIP "$1" "$3 (operator-disabled; would be a FAIL under GATE_STRICT=1)"
        else
            row RUN "$1" "attempted ($3)"
        fi
    }
    row  RUN  "1/6 greedy determinism, np=1 ($REPS runs)"            "outcome needs a live boot"
    row  RUN  "2/6 coherence"                                        "outcome needs a live boot"
    row  RUN  "3/6 chat completions (/v1/chat/completions, --jinja $([ -n "$CHAT_JINJA" ] && echo on || echo off), $CHAT_REPS runs)" \
              "SKIPs only if this build serves no /v1/chat/completions route -- runtime-only"
    row  RUN  "4/6 needle recall ($NEEDLE_REPS runs per prompt)"     "outcome needs a live boot"
    knob_row   "  logit reproducibility, np=1 + np=2" "$([ "$LOGIT_REPS" -gt 0 ] 2>/dev/null && echo 0 || echo 1)" "LOGIT_REPS=$LOGIT_REPS"
    [ "$LOGIT_REPS" -gt 0 ] 2>/dev/null && row RUN "  (same)" "also SKIPs at runtime if the server returns no completion_probabilities"
    case "$GATE_NP2" in
        0) knob_row "5/6 greedy determinism, np=2" 1 "GATE_NP2=0" ;;
        auto|1)
            if [ -n "$BIN" ] && [ -x "$BIN/llama-server" ]; then
                if "$BIN/llama-server" --help 2>&1 | grep -q -- '-np'; then
                    row RUN "5/6 greedy determinism, np=2 ($REPS runs)" "server has -np; SKIPs at runtime only if no slot-erase route"
                else
                    row SKIP "5/6 greedy determinism, np=2" "this server has no -np flag (checked for real via --help)"
                fi
            else
                row RUN "5/6 greedy determinism, np=2 ($REPS runs)" "cannot verify -np without BIN -- would check for real at boot"
            fi ;;
    esac
    row RUN "  token-0 logit match, np=1 vs np=2" "runtime-only: needs both logit-reproducibility runs above to be stable"
    knob_row "6/6 unit tests" "$([ "$GATE_TESTS" = 1 ] && echo 0 || echo 1)" "GATE_TESTS=$GATE_TESTS"
    if [ "$GATE_TESTS" = 1 ]; then
        gpu_ok=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q . && gpu_ok=1
        for t in test-pxq-cpu-dot test-kv-seq-shadow; do
            if [ -n "$BIN" ] && [ -x "$BIN/$t" ]; then row RUN "  $t" "binary present"
            else row FAIL "  $t" "binary missing from ${BIN:-<BIN unset>} -- not a hardware skip, this test is CPU-only"; fi
        done
        if [ -n "$BIN" ] && [ -x "$BIN/test-narrow-kernel-parity" ]; then row RUN "  test-narrow-kernel-parity" "binary present"
        elif [ "$gpu_ok" = 1 ]; then row FAIL "  test-narrow-kernel-parity" "binary missing but a GPU IS present on this box -- no excuse, would FAIL"
        else row SKIP "  test-narrow-kernel-parity" "GPU-only test, no GPU present on this box (checked for real via nvidia-smi)"; fi
    fi
    printf '\nThis is a preview only -- nothing above ran a request or booted a server. Run without\n--plan (and with MODEL set) for a real result.\n'
    exit 0
fi

[ -n "$BIN" ] && [ -x "$BIN/llama-server" ] || die "no llama-server found. Set BIN=<dir with llama-server> (looked in $ROOT/build/bin, build-spd/bin, build-tok/bin)."
[ -n "$MODEL" ] || die "set MODEL=<path to .gguf>"
[ -r "$MODEL" ]  || die "MODEL not readable: $MODEL"
for p in needle3121.txt needle20801.txt coherence.txt; do
    [ -r "$PROMPTS/$p" ] || die "missing prompt file $PROMPTS/$p"
done
command -v python3 >/dev/null || die "python3 is required (it is the only external dependency)"

[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
BASE="http://$HOST:$PORT"

say "gate starting"
say "  model    $MODEL"
say "  binaries $BIN"
say "  devices  ${CUDA_VISIBLE_DEVICES:-<all>}"
say "  logs     $WORKDIR"

# ---- HTTP, over python3 only (build containers often ship without curl) ----------------------
# http_ok <url>           -> exit 0 iff the URL answers 200 within 5 s
# http_post <url> <file>  -> POST the file as application/json, print the response body
# http_post_status <url>  -> print the status code of an empty POST
http_ok(){
    PXA_URL=$1 python3 - <<'PY' >/dev/null 2>&1
import os, sys, urllib.request
try:
    with urllib.request.urlopen(os.environ["PXA_URL"], timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}
http_post(){
    PXA_URL=$1 PXA_BODY=$2 PXA_T=$REQ_TIMEOUT python3 - <<'PY'
import os, sys, urllib.request
req = urllib.request.Request(os.environ["PXA_URL"],
                             data=open(os.environ["PXA_BODY"], "rb").read(),
                             headers={"Content-Type": "application/json"},
                             method="POST")
try:
    with urllib.request.urlopen(req, timeout=float(os.environ["PXA_T"])) as r:
        sys.stdout.write(r.read().decode("utf-8", "replace"))
except Exception as e:
    sys.stderr.write("http_post: %s\n" % e)
PY
}
http_post_status(){
    PXA_URL=$1 python3 - <<'PY'
import os, urllib.request, urllib.error
req = urllib.request.Request(os.environ["PXA_URL"], data=b"", method="POST")
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception:
    print(0)
PY
}

# ---- server lifecycle ------------------------------------------------------------------------
server_stop(){
    [ -n "$SRV_PID" ] || return 0
    kill "$SRV_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    SRV_PID=
}
trap 'server_stop' EXIT INT TERM

# server_start <n_parallel> <logname>
server_start(){
    local np=$1 name=$2 fa=() slots=() jinja=()
    [ -n "$FA" ] && fa=(-fa "$FA")
    # --jinja: use the model's own embedded chat template (via minja) for /v1/chat/completions
    # instead of the legacy built-in template matcher. This is how the check below can honestly
    # claim "production chat template" -- it is also how every real deployment on this engine is
    # actually run. Omit with CHAT_JINJA= on a server too old to have the flag.
    [ -n "$CHAT_JINJA" ] && jinja=(--jinja)
    # The POST /slots/<n>?action=erase route is only registered when --slot-save-path is set,
    # and erasing the other slot is what puts the tested slot at a known KV placement.
    if [ "$np" -gt 1 ]; then mkdir -p "$WORKDIR/slots"; slots=(--slot-save-path "$WORKDIR/slots/"); fi
    # shellcheck disable=SC2086
    "$BIN/llama-server" -m "$MODEL" -ngl "$NGL" -c "$CTX" -b "$BATCH" -ub "$UBATCH" \
        -t "$THREADS" ${fa[@]+"${fa[@]}"} ${jinja[@]+"${jinja[@]}"} ${slots[@]+"${slots[@]}"} -np "$np" --host "$HOST" --port "$PORT" \
        $SERVER_ARGS > "$WORKDIR/$name.log" 2>&1 &
    SRV_PID=$!
    local t0=$SECONDS
    until http_ok "$BASE/health"; do
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            say "server died during boot; last lines of $WORKDIR/$name.log:"; tail -20 "$WORKDIR/$name.log"; return 1
        fi
        (( SECONDS - t0 > BOOT_TIMEOUT )) && { say "boot timeout after ${BOOT_TIMEOUT}s"; tail -20 "$WORKDIR/$name.log"; return 1; }
        sleep 5
    done
    say "server up (-np $np) after $(( SECONDS - t0 ))s"
}

# complete <prompt-file> <n_predict> [id_slot]  -> prints the completion text on stdout
complete(){
    local pf=$1 npred=$2 slot=${3:-}
    PXA_PF=$pf PXA_NPRED=$npred PXA_SLOT=$slot python3 - <<'PY' > "$WORKDIR/req.json"
import json, os
body = {
    "prompt":       open(os.environ["PXA_PF"], encoding="utf-8", errors="replace").read(),
    "n_predict":    int(os.environ["PXA_NPRED"]),
    "temperature":  0,
    "top_k":        1,
    "seed":         0,
    "cache_prompt": False,
}
s = os.environ.get("PXA_SLOT", "")
if s != "":
    body["id_slot"] = int(s)
print(json.dumps(body))
PY
    http_post "$BASE/completion" "$WORKDIR/req.json" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("content",""), end="")
except Exception:
    pass'
}

sha(){ printf '%s' "$1" | sha256sum | cut -c1-12; }

# logits_sig <prompt-file> -> a canonical string of the top-2 probabilities of the FIRST generated
# token, at full float precision. The greedy sha only sees the ARGMAX; this sees the numbers the
# argmax was taken over, so it catches a nondeterministic forward pass whose wobble happens to be
# smaller than most top-2 margins. Empty output = the server returned no completion_probabilities
# (old server, or n_probs unsupported) -> the check SKIPs rather than failing.
# logit_arm <label> [id_slot] -- LOGIT_REPS identical prefill-only requests; PASS only if every
# returned probability is identical to the last digit.
#
# Also exposes, for the caller, LAST_LOGIT_SIG (the full-precision top-2 signature of the FIRST
# rep, valid only when LAST_LOGIT_ARM_OK=1) so a caller can cross-compare the actual token-0
# logits between two arms -- not just each arm's own internal reproducibility. That cross-compare
# is what ARM B (np=2) does against the np=1 reference below: this function alone only proves
# "this arm agrees with itself", never "np=1 and np=2 agree with each other".
LAST_LOGIT_SIG=""; LAST_LOGIT_ARM_OK=0
logit_arm(){
    local label=$1 slot=${2:-} r lsig lsigs="" lbad=0 n
    LAST_LOGIT_SIG=""; LAST_LOGIT_ARM_OK=0
    if [ "$LOGIT_REPS" -gt 0 ] 2>/dev/null; then :; else
        skc "logit reproducibility ($label) disabled (LOGIT_REPS=0)"; return
    fi
    say "=== logit reproducibility, $label, $LOGIT_REPS runs ==="
    for r in $(seq 1 "$LOGIT_REPS"); do
        lsig=$(logits_sig "$PROMPTS/needle3121.txt" "$slot")
        if [ -z "$lsig" ]; then lbad=1; break; fi
        [ "$r" = 1 ] && LAST_LOGIT_SIG=$lsig
        lsigs="$lsigs $(sha "$lsig")"
    done
    n=$(printf '%s\n' $lsigs | sort -u | grep -c .)
    if [ "$lbad" = 1 ]; then
        sk "logit reproducibility ($label): this server returned no completion_probabilities (n_probs unsupported), so the forward pass could not be compared below the argmax"
        LAST_LOGIT_SIG=""
    elif [ "$n" = 1 ]; then
        ok "logit reproducibility ($label) $LOGIT_REPS/$LOGIT_REPS identical ($(printf '%s\n' $lsigs | head -1))"
        LAST_LOGIT_ARM_OK=1
    else
        ko "logit reproducibility ($label): $n distinct probability sets across $LOGIT_REPS runs ->$lsigs -- the forward pass is nondeterministic even where the greedy sha is stable"
        LAST_LOGIT_SIG=""
    fi
}

logits_sig(){
    PXA_PF=$1 PXA_SLOT=${2:-} python3 - <<'PY' > "$WORKDIR/req.json"
import json, os
body = {
    "prompt":       open(os.environ["PXA_PF"], encoding="utf-8", errors="replace").read(),
    "n_predict":    1,
    "temperature":  0,
    "top_k":        1,
    "seed":         0,
    "cache_prompt": False,
    "n_probs":      2,
}
s = os.environ.get("PXA_SLOT", "")
if s != "":
    body["id_slot"] = int(s)
print(json.dumps(body))
PY
    http_post "$BASE/completion" "$WORKDIR/req.json" | python3 -c 'import json,sys
try:
    j = json.load(sys.stdin)
except Exception:
    sys.exit()
out = []
for e in (j.get("completion_probabilities") or []):
    for q in (e.get("probs") or []):
        out.append("%s=%.17g" % (q.get("tok_str", q.get("token", "")), q.get("prob", 0.0)))
print("|".join(out))'
}

# chat_complete <system> <user> [n_predict] -- POSTs to /v1/chat/completions (the OpenAI-compatible
# route every real client, and every tool-calling agent, actually talks to -- never /completion),
# letting the server render the model's OWN embedded chat template rather than a hand-built prompt
# string. Writes the raw response body to $WORKDIR/chat-resp.json and the HTTP status to
# $WORKDIR/chat-status (status 0 = connection/other error before a status line). Deliberately two
# files, not stdout: bash variables cannot hold the NUL bytes command substitution can silently
# truncate on, and this keeps the status distinguishable from a 200-with-an-empty-body without a
# second round trip. Callers read chat_content below for the parsed text.
chat_complete(){
    local sys=$1 usr=$2 npred=${3:-64}
    PXA_SYS=$sys PXA_USR=$usr PXA_NPRED=$npred python3 - <<'PY' > "$WORKDIR/chat-req.json"
import json, os
body = {
    "messages": [
        {"role": "system", "content": os.environ["PXA_SYS"]},
        {"role": "user",   "content": os.environ["PXA_USR"]},
    ],
    "temperature":  0,
    "top_k":        1,
    "seed":         0,
    "n_predict":    int(os.environ["PXA_NPRED"]),
    "cache_prompt": False,
}
print(json.dumps(body))
PY
    PXA_URL="$BASE/v1/chat/completions" PXA_BODY="$WORKDIR/chat-req.json" PXA_T="$REQ_TIMEOUT" \
      PXA_OUT="$WORKDIR/chat-resp.json" PXA_STATUS="$WORKDIR/chat-status" python3 - <<'PY'
import os, urllib.request, urllib.error
req = urllib.request.Request(os.environ["PXA_URL"],
                              data=open(os.environ["PXA_BODY"], "rb").read(),
                              headers={"Content-Type": "application/json"}, method="POST")
status, body = 0, b""
try:
    with urllib.request.urlopen(req, timeout=float(os.environ["PXA_T"])) as r:
        status, body = r.status, r.read()
except urllib.error.HTTPError as e:
    status, body = e.code, e.read()
except Exception:
    pass
open(os.environ["PXA_OUT"], "wb").write(body)
open(os.environ["PXA_STATUS"], "w").write(str(status))
PY
}
# chat_content -> the assistant message text from the last chat_complete response, or empty if the
# body did not parse (an absent or erroring route leaves nothing to print, by design).
chat_content(){
    python3 -c 'import json,sys
try:
    j = json.load(open(sys.argv[1], encoding="utf-8", errors="replace"))
    print(j["choices"][0]["message"]["content"], end="")
except Exception:
    pass' "$WORKDIR/chat-resp.json"
}

# erase_slots <n_parallel> -> returns non-zero if the endpoint is not available
erase_slots(){
    local np=$1 i code allok=0
    for i in $(seq 0 $((np-1))); do
        code=$(http_post_status "$BASE/slots/$i?action=erase")
        [ "$code" = 200 ] || allok=1
    done
    return $allok
}

# ---- ARM A: np=1 ------------------------------------------------------------------------------
say "=== 1/6  greedy determinism, np=1, $REPS runs ==="
REF_SHA=""; REF_LOGIT_SIG=""; REF_LOGIT_OK=0
if ! server_start 1 server-np1; then
    ko "server would not boot at -np 1 (see $WORKDIR/server-np1.log); every np=1 check below is unrunnable"
else
    shas=""
    for r in $(seq 1 "$REPS"); do
        out=$(complete "$PROMPTS/needle3121.txt" "$N_PREDICT")
        [ -n "$out" ] || { ko "np=1 run $r returned nothing"; break; }
        shas="$shas $(sha "$out")"
        [ "$r" = 1 ] && printf '%s' "$out" > "$WORKDIR/ref-np1.txt"
    done
    uniq_n=$(printf '%s\n' $shas | sort -u | grep -c .)
    REF_SHA=$(printf '%s\n' $shas | head -1)
    if [ "$uniq_n" = 1 ] && [ "$(printf '%s\n' $shas | grep -c .)" = "$REPS" ]; then
        ok "np=1 greedy determinism $REPS/$REPS byte-identical (sha $REF_SHA)"
    else
        ko "np=1 greedy determinism: $uniq_n distinct outputs across $REPS runs ->$shas"
    fi

    # ---- coherence -----------------------------------------------------------------------------
    say "=== 2/6  coherence ==="
    co=$(complete "$PROMPTS/coherence.txt" 16)
    if printf '%s' "$co" | grep -qi -- "$COHERENCE_EXPECT"; then
        ok "coherence: $(printf '%s' "$co" | head -c 60 | tr '\n' ' ')"
    else
        ko "coherence: expected /$COHERENCE_EXPECT/i, got '$(printf '%s' "$co" | head -c 120 | tr '\n' ' ')'"
    fi

    # ---- chat completions, the production template ----------------------------------------------
    # Every check above talks to /completion with a hand-built prompt string. No real client does
    # that: a real client, and every tool-calling agent, POSTs to /v1/chat/completions and lets the
    # server render the model's OWN chat template. A build can pass every /completion check above
    # and still be a failed build for that reason alone -- a broken/missing template, a stop-token
    # mismatch, a route that 500s -- none of which the checks above can see. Route absent (this
    # build has no OpenAI-compatible endpoint) -> SKIP; route present and wrong -> FAIL.
    say "=== 3/6  chat completions (/v1/chat/completions, production template) ==="
    chat_complete "You are a helpful, concise assistant." "$(cat "$PROMPTS/coherence.txt")" 32
    chat_status=$(cat "$WORKDIR/chat-status" 2>/dev/null || echo 0)
    if [ "$chat_status" = 404 ] || [ "$chat_status" = 501 ]; then
        sk "chat completions: /v1/chat/completions is not served by this build (status $chat_status) -- an OpenAI-compatible chat-template pass cannot be proved on it"
    elif [ "$chat_status" != 200 ]; then
        ko "chat completions: /v1/chat/completions returned status $chat_status, not 200 -- $(head -c 160 "$WORKDIR/chat-resp.json" | tr '\n' ' ')"
    else
        co1=$(chat_content)
        if [ -z "$co1" ]; then
            ko "chat completions: /v1/chat/completions returned 200 with no message content -- the production chat template may be rendering to nothing. Response: $(head -c 160 "$WORKDIR/chat-resp.json" | tr '\n' ' ')"
        elif ! printf '%s' "$co1" | grep -qi -- "$COHERENCE_EXPECT"; then
            ko "chat completions: expected /$COHERENCE_EXPECT/i through the production template, got '$(printf '%s' "$co1" | head -c 120 | tr '\n' ' ')'"
        else
            chat_shas=$(sha "$co1") chat_bad=0
            for r in $(seq 2 "$CHAT_REPS"); do
                chat_complete "You are a helpful, concise assistant." "$(cat "$PROMPTS/coherence.txt")" 32
                [ "$(cat "$WORKDIR/chat-status" 2>/dev/null)" = 200 ] || { chat_bad=1; break; }
                chat_shas="$chat_shas $(sha "$(chat_content)")"
            done
            chat_n=$(printf '%s\n' $chat_shas | sort -u | grep -c .)
            if [ "$chat_bad" = 1 ]; then
                ko "chat completions: a repeat request through /v1/chat/completions stopped returning 200 mid-run"
            elif [ "$chat_n" = 1 ]; then
                ok "chat completions: production template renders correctly, $CHAT_REPS/$CHAT_REPS byte-identical ($(printf '%s' "$co1" | head -c 60 | tr '\n' ' '))"
            else
                ko "chat completions: $chat_n distinct outputs across $CHAT_REPS runs at temperature 0 through the production template ->$chat_shas"
            fi
        fi
    fi

    # ---- needle recall -------------------------------------------------------------------------
    say "=== 4/6  needle recall, $NEEDLE_REPS runs per prompt ==="
    for n in 3121 20801; do
        shas=""; bad=0
        for r in $(seq 1 "$NEEDLE_REPS"); do
            out=$(complete "$PROMPTS/needle$n.txt" "$N_PREDICT")
            if printf '%s' "$out" | grep -q 'MAGENTA-7741' && printf '%s' "$out" | grep -q 'BLUE-HERON'; then :; else
                ko "needle$n run $r did not recall both identifiers: '$(printf '%s' "$out" | head -c 80 | tr '\n' ' ')'"; bad=1
            fi
            shas="$shas $(sha "$out")"
        done
        if [ "$(printf '%s\n' $shas | sort -u | grep -c .)" = 1 ]; then
            [ "$bad" = 0 ] && ok "needle$n recalled and sha-stable $NEEDLE_REPS/$NEEDLE_REPS ($(printf '%s\n' $shas | head -1))"
        else
            ko "needle$n sha unstable across $NEEDLE_REPS runs ->$shas"
        fi
    done

    # ---- logit reproducibility -------------------------------------------------------------------
    # The checks above compare the ARGMAX. This one compares the logits the argmax was taken over.
    # A forward pass can be nondeterministic and still produce a stable sha for as long as every
    # top-2 margin exceeds the wobble; the flips then look like rare flakes on long prompts and
    # under load, which is exactly how a write-after-read race in a fused kernel presents. On
    # 2026-09-03 the DeltaNet out-gate fusion (PXA_FUSE_DELTANET bit 1) moved the token-0
    # probability by up to 4e-2 on EVERY run of an identical prompt while 12/12 greedy shas
    # matched; the gate only saw it as a 1-in-4 flip at 20,801 tokens. This check sees it directly.
    logit_arm "np=1"
    REF_LOGIT_SIG=$LAST_LOGIT_SIG; REF_LOGIT_OK=$LAST_LOGIT_ARM_OK
fi
server_stop

# ---- ARM B: np=2 -------------------------------------------------------------------------------
say "=== 5/6  greedy determinism, np=2, other slot erased first ==="
run_np2=1
case "$GATE_NP2" in
    0) run_np2=0; skc "np=2 arm disabled (GATE_NP2=0)" ;;
    auto|1) "$BIN/llama-server" --help 2>&1 | grep -q -- '-np' || { run_np2=0; sk "this server has no -np flag"; } ;;
esac
if [ "$run_np2" = 1 ]; then
    if server_start 2 server-np2; then
        if erase_slots 2; then
            # Slot 1 now starts at the same KV placement slot 0 would: a byte comparison is valid.
            shas=""
            for r in $(seq 1 "$REPS"); do
                out=$(complete "$PROMPTS/needle3121.txt" "$N_PREDICT" 1)
                [ -n "$out" ] || { ko "np=2 run $r returned nothing"; break; }
                shas="$shas $(sha "$out")"
            done
            uniq_n=$(printf '%s\n' $shas | sort -u | grep -c .)
            got=$(printf '%s\n' $shas | head -1)
            if [ "$uniq_n" = 1 ] && [ "$(printf '%s\n' $shas | grep -c .)" = "$REPS" ]; then
                ok "np=2 slot 1 greedy determinism $REPS/$REPS byte-identical (sha $got)"
            else
                ko "np=2 slot 1 greedy determinism: $uniq_n distinct outputs across $REPS runs ->$shas"
            fi
            # Only meaningful if BOTH arms were themselves stable: comparing one sample of an
            # unstable arm against the reference reports agreement that does not exist.
            if [ "$NP2_CROSS_CHECK" = 1 ] && [ -n "$REF_SHA" ] && [ "$uniq_n" = 1 ]; then
                if [ "$got" = "$REF_SHA" ]; then
                    ok "np=2 slot 1 matches the np=1 reference at the same KV placement ($got)"
                else
                    ko "np=2 slot 1 ($got) differs from the np=1 reference ($REF_SHA) at the same KV placement -- see README, this is the check that catches a real multi-slot defect"
                fi
            elif [ "$NP2_CROSS_CHECK" = 1 ] && [ -n "$REF_SHA" ]; then
                sk "np=2 vs np=1 cross-check: not meaningful, the np=2 arm above was not stable"
            fi
            logit_arm "np=2 slot 1" 1
            # ---- token-0 logit match, np=1 vs np=2 --------------------------------------------
            # logit_arm above only proves each arm agrees WITH ITSELF (LOGIT_REPS runs at the same
            # np). It does not prove np=1 and np=2 agree with EACH OTHER -- and the greedy-sha
            # cross-check a few lines up only compares the argmax, which (per the 2026-09-03
            # DeltaNet incident documented in the README) can stay stable while the float logits
            # underneath it do not. This compares the actual top-2 probabilities of token 0,
            # full precision, at the same KV placement (slot 1 just erased, so the comparison is
            # valid per README rule 2) -- the strongest determinism claim this gate makes.
            if [ "$NP2_CROSS_CHECK" = 1 ] && [ "$REF_LOGIT_OK" = 1 ] && [ "$LAST_LOGIT_ARM_OK" = 1 ]; then
                if [ "$LAST_LOGIT_SIG" = "$REF_LOGIT_SIG" ]; then
                    ok "token-0 logit match: np=2 slot 1 == np=1 reference, identical to the last digit"
                else
                    ko "token-0 logit mismatch: np=2 slot 1 differs from the np=1 reference at the same KV placement -- np=1 [$REF_LOGIT_SIG] != np=2 [$LAST_LOGIT_SIG]"
                fi
            elif [ "$NP2_CROSS_CHECK" = 1 ]; then
                sk "token-0 logit-match np=1 vs np=2: not meaningful, one or both logit-reproducibility arms above were not stable/available"
            fi
        else
            sk "np=2: POST /slots/N?action=erase is unavailable on this build, so the tested slot cannot be put at a known KV placement -- see README"
        fi
    else
        ko "server would not boot at -np 2"
    fi
    server_stop
fi

# ---- unit tests ---------------------------------------------------------------------------------
if [ "$GATE_TESTS" = 1 ]; then
    say "=== 6/6  unit tests ==="
    gpu_present(){ command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q .; }
    # check_test <name> <want> [require] -- require=gpu (default cpu) marks a test that a CPU-only
    # build genuinely cannot produce (e.g. test-narrow-kernel-parity compares CUDA kernels against
    # a reference and is not even a ninja target when GGML_CUDA=OFF -- confirmed while building
    # this gate's own CPU arms). A missing binary is a legitimate hardware-absence SKIP only when
    # this box also has no GPU; a GPU box that forgot to build a GPU-only test still FAILs -- the
    # hardware is not absent there, so there is no excuse.
    check_test(){
        local name=$1 want=$2 require=${3:-cpu}
        if [ ! -x "$BIN/$name" ]; then
            if [ "$require" = gpu ] && ! gpu_present; then
                sk "$name is missing from $BIN and no GPU is present on this box -- a GPU-only test, expected absent on a CPU-only run"
            else
                ko "$name is missing from $BIN (build it, or set GATE_TESTS=0)"
            fi
            return
        fi
        if "$BIN/$name" > "$WORKDIR/$name.log" 2>&1 && tail -3 "$WORKDIR/$name.log" | grep -q -- "$want"; then
            ok "$name"
        else
            ko "$name: $(tail -1 "$WORKDIR/$name.log" | head -c 120)"
        fi
    }
    check_test test-pxq-cpu-dot           "OK"
    check_test test-kv-seq-shadow         "PASS"
    check_test test-narrow-kernel-parity  "ALL PASS" gpu
else
    skc "unit tests disabled (GATE_TESTS=0)"
fi

say "=== GATE RESULT: PASS=$pass FAIL=$fail SKIP=$skip  (logs in $WORKDIR) ==="
if [ "$skip" -gt 0 ]; then
    say "    skipped arms (a SKIP is not a FAIL, but a gate with skips has not proved what a clean gate proves -- read these before trusting the exit code):"
    for s in "${SKIPPED[@]}"; do printf '      - %s\n' "$s"; done
fi
[ "$fail" = 0 ]
