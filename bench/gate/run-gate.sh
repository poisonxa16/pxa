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
SERVER_ARGS=${SERVER_ARGS:-}           # anything extra (-ts, -ot, --kv-unified, ...)
REPS=${REPS:-12}                       # determinism repetitions per arm
NEEDLE_REPS=${NEEDLE_REPS:-4}
LOGIT_REPS=${LOGIT_REPS:-6}      # logit-reproducibility repetitions (0 disables)
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
REQ_TIMEOUT=${REQ_TIMEOUT:-900}
N_PREDICT=${N_PREDICT:-256}   # thinking models spend the first ~100 tokens inside <think>; 32 truncated every needle answer on stock Qwen3.8
GATE_NP2=${GATE_NP2:-auto}             # auto | 1 | 0
NP2_CROSS_CHECK=${NP2_CROSS_CHECK:-1}  # compare the np=2 slot against the np=1 reference
GATE_TESTS=${GATE_TESTS:-1}
COHERENCE_EXPECT=${COHERENCE_EXPECT:-paris}
PROMPTS=${PROMPTS:-$HERE/prompts}
WORKDIR=${WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/pxa-gate.XXXXXX")}
# An externally supplied WORKDIR is not created by mktemp, and the FIRST thing the run does is
# redirect the np=1 server log into it -- a missing directory failed that redirect and reported
# "server would not boot at -np 1", i.e. a harness artefact wearing an engine failure's clothes
# (2026-09-06: the np=2 arm passed in the same run only because it mkdir -p's WORKDIR/slots).
mkdir -p "$WORKDIR" || { printf 'gate: cannot create WORKDIR %s\n' "$WORKDIR" >&2; exit 2; }   # die() is defined below

pass=0; fail=0; skip=0; SRV_PID=
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
ok(){   printf '  PASS  %s\n' "$*"; pass=$((pass+1)); }
ko(){   printf '  FAIL  %s\n' "$*"; fail=$((fail+1)); }
sk(){   printf '  SKIP  %s\n' "$*"; skip=$((skip+1)); }
die(){  printf 'gate: %s\n' "$*" >&2; exit 2; }

# ---- resolve BIN -----------------------------------------------------------------------------
if [ -z "$BIN" ]; then
    for d in "$ROOT/build/bin" "$ROOT/build-spd/bin" "$ROOT/build-tok/bin" "$ROOT/build/Release/bin"; do
        [ -x "$d/llama-server" ] && { BIN=$d; break; }
    done
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
    local np=$1 name=$2 fa=() slots=()
    [ -n "$FA" ] && fa=(-fa "$FA")
    # The POST /slots/<n>?action=erase route is only registered when --slot-save-path is set,
    # and erasing the other slot is what puts the tested slot at a known KV placement.
    if [ "$np" -gt 1 ]; then mkdir -p "$WORKDIR/slots"; slots=(--slot-save-path "$WORKDIR/slots/"); fi
    # shellcheck disable=SC2086
    "$BIN/llama-server" -m "$MODEL" -ngl "$NGL" -c "$CTX" -b "$BATCH" -ub "$UBATCH" \
        -t "$THREADS" ${fa[@]+"${fa[@]}"} ${slots[@]+"${slots[@]}"} -np "$np" --host "$HOST" --port "$PORT" \
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
logit_arm(){
    local label=$1 slot=${2:-} r lsig lsigs="" lbad=0 n
    if [ "$LOGIT_REPS" -gt 0 ] 2>/dev/null; then :; else
        sk "logit reproducibility ($label) disabled (LOGIT_REPS=0)"; return
    fi
    say "=== logit reproducibility, $label, $LOGIT_REPS runs ==="
    for r in $(seq 1 "$LOGIT_REPS"); do
        lsig=$(logits_sig "$PROMPTS/needle3121.txt" "$slot")
        if [ -z "$lsig" ]; then lbad=1; break; fi
        lsigs="$lsigs $(sha "$lsig")"
    done
    n=$(printf '%s\n' $lsigs | sort -u | grep -c .)
    if [ "$lbad" = 1 ]; then
        sk "logit reproducibility ($label): this server returned no completion_probabilities (n_probs unsupported), so the forward pass could not be compared below the argmax"
    elif [ "$n" = 1 ]; then
        ok "logit reproducibility ($label) $LOGIT_REPS/$LOGIT_REPS identical ($(printf '%s\n' $lsigs | head -1))"
    else
        ko "logit reproducibility ($label): $n distinct probability sets across $LOGIT_REPS runs ->$lsigs -- the forward pass is nondeterministic even where the greedy sha is stable"
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
say "=== 1/4  greedy determinism, np=1, $REPS runs ==="
REF_SHA=""
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
    say "=== 2/4  coherence ==="
    co=$(complete "$PROMPTS/coherence.txt" 16)
    if printf '%s' "$co" | grep -qi -- "$COHERENCE_EXPECT"; then
        ok "coherence: $(printf '%s' "$co" | head -c 60 | tr '\n' ' ')"
    else
        ko "coherence: expected /$COHERENCE_EXPECT/i, got '$(printf '%s' "$co" | head -c 120 | tr '\n' ' ')'"
    fi

    # ---- needle recall -------------------------------------------------------------------------
    say "=== 3/4  needle recall, $NEEDLE_REPS runs per prompt ==="
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
fi
server_stop

# ---- ARM B: np=2 -------------------------------------------------------------------------------
say "=== 4/4  greedy determinism, np=2, other slot erased first ==="
run_np2=1
case "$GATE_NP2" in
    0) run_np2=0; sk "np=2 arm disabled (GATE_NP2=0)" ;;
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
    say "=== unit tests ==="
    check_test(){
        local name=$1 want=$2
        if [ ! -x "$BIN/$name" ]; then ko "$name is missing from $BIN (build it, or set GATE_TESTS=0)"; return; fi
        if "$BIN/$name" > "$WORKDIR/$name.log" 2>&1 && tail -3 "$WORKDIR/$name.log" | grep -q -- "$want"; then
            ok "$name"
        else
            ko "$name: $(tail -1 "$WORKDIR/$name.log" | head -c 120)"
        fi
    }
    check_test test-pxq-cpu-dot           "OK"
    check_test test-kv-seq-shadow         "PASS"
    check_test test-narrow-kernel-parity  "ALL PASS"
else
    sk "unit tests disabled (GATE_TESTS=0)"
fi

say "=== GATE RESULT: PASS=$pass FAIL=$fail SKIP=$skip  (logs in $WORKDIR) ==="
[ "$fail" = 0 ]
