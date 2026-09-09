#!/bin/bash
# bench/fair/run.sh — the three numbers, one command, on a documented rig.
#
#   ./run.sh --rig 4xp100 --plan     # print exactly what would run, touch nothing
#   ./run.sh --rig 4xp100            # run it and print the three blocks
#
# It prints THREE blocks and nothing else on stdout:
#
#   ENGINE-ONLY   same GGUF, two engines   — this engine vs the named upstream binary
#   CODEC-ONLY    same engine, two files   — PXQ4 vs MXFP4 at matched bytes
#   PRODUCT       best recipe per side     — what you would actually run
#
# Everything else — progress, warnings, why a cell is missing — goes to stderr, so
# `./run.sh --rig <rig> > blocks.txt` gives you three blocks a README can paste.
#
# It exits 2, not 0, when the run is not a fair comparison:
#   * a model in a printed arm has no sha256 in weights/MANIFEST.sha256 (or is
#     listed there as "sha: pending"),
#   * the arms of one block disagree about speculative decode / MTP,
#   * the upstream binary is not present, so the engine-only block cannot exist.
# A missing number is reported as missing. It is never backfilled from a looser
# protocol, and never quietly dropped so the remaining cells look complete.
#
# What is NOT in these three blocks, on purpose (bench/fair/protocol.md):
#   * an expert-codec comparison whose two sides are not the same base weights
#     (the MoE decode row against MXFP4 in bench/fair-battle.md is that: same
#     architecture and size class, a different model). Set
#     PRODUCT_SAME_BASE_WEIGHTS=no in a rig file and this refuses to print it as
#     a PRODUCT row;
#   * multi-box, NVLink and sidecar-speculation rows. Different hardware or a
#     different serving stack is a different table, not a fourth column here.
#
# The protocol every cell below obeys is bench/fair/protocol.md: llama-server
# /completion (not /v1/chat/completions), temperature 0, seed 42, n=7 with one
# warmup discarded, a unique prompt per repeat, cache_prompt off, and the prompt
# token count the server itself reports printed next to the number.
#
# Configuration is the rig file (rigs/<name>.env) plus these environment
# variables, all optional:
#   ENGINE_BIN     this engine's llama-server (default: the first of
#                  ../../build/bin, ../../build-spd/bin, ../bin, /usr/local/bin)
#   UPSTREAM_BIN   the comparison engine (default: /opt/pxa/bin/upstream-ik-server,
#                  which is where the container image puts it)
#   WEIGHTS_DIR    where the GGUFs are (default: ./weights)
#   FILL_TOKENS    prompt length for the prefill cell (default: 3121)
#   N_PREDICT      tokens generated for the decode cell (default: 192)
#   REPS           timed repeats after the discarded warmup (default: 7)
#   BOOT_TIMEOUT   seconds to wait for a server to answer /health (default: 900)
set -u

cd -- "$(dirname -- "$0")" || exit 1
HERE=$(pwd)

WEIGHTS_DIR=${WEIGHTS_DIR:-$HERE/weights}
MANIFEST=$WEIGHTS_DIR/MANIFEST.sha256
RIGS_DIR=$HERE/rigs
UPSTREAM_BIN=${UPSTREAM_BIN:-/opt/pxa/bin/upstream-ik-server}
FILL_TOKENS=${FILL_TOKENS:-3121}
N_PREDICT=${N_PREDICT:-192}
REPS=${REPS:-7}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
PORT_BASE=${PORT_BASE:-18190}

RIG=""
PLAN=0
EXIT_CODE=0
PORT_NEXT=$PORT_BASE

say()  { printf '%s\n' "$*" >&2; }
warn() { printf 'bench/fair: %s\n' "$*" >&2; }
die()  { printf 'bench/fair: %s\n' "$*" >&2; exit 1; }
out()  { printf '%s\n' "$*"; }
fail2(){ printf 'bench/fair: %s\n' "$*" >&2; EXIT_CODE=2; }

# The rig names, for the two places that list them. `ls` is fine here and `find`
# is not clearer: these are files this repo ships, with names it chose.
# shellcheck disable=SC2012
rig_names() { ls "$RIGS_DIR" 2>/dev/null | sed 's/\.env$//' | tr '\n' ' '; }

usage() {
    sed -n '2,/^set -u$/p' "$HERE/run.sh" | sed 's/^# \{0,1\}//; $d' >&2
    say "usage: ./run.sh --rig <name> [--plan]"
    say "rigs available: $(rig_names)"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --rig)   [ "$#" -ge 2 ] || die "--rig needs a name"; RIG=$2; shift 2 ;;
        --rig=*) RIG=${1#--rig=}; shift ;;
        --plan)  PLAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "unknown argument: $1" ;;
    esac
done

command -v python3 >/dev/null 2>&1 || die "python3 is required (it is the only external dependency)"

# ---------------------------------------------------------------------------
# the rig
# ---------------------------------------------------------------------------
[ -n "$RIG" ] || { usage; die "no rig given. A number without the machine it was measured on is not reproducible: pass --rig <name>."; }
RIG_FILE=$RIGS_DIR/$RIG.env
[ -f "$RIG_FILE" ] || die "no rig file $RIG_FILE. Available: $(rig_names)— copy the closest one and edit it; every value it cannot state honestly stays 'pending'."

# A rig file is data, not a plugin: KEY=VALUE and comments only, so sourcing it
# cannot run anything. Checked rather than trusted.
if grep -nvE '^[[:space:]]*(#|$|[A-Z_][A-Z0-9_]*=)' "$RIG_FILE" >/dev/null 2>&1; then
    grep -nvE '^[[:space:]]*(#|$|[A-Z_][A-Z0-9_]*=)' "$RIG_FILE" >&2
    die "$RIG_FILE has lines that are not comments or KEY=VALUE (shown above)"
fi

RIG_NAME=$RIG; RIG_DESC="(undescribed)"; RIG_DEVICES=""; RIG_SOURCE=""
ENGINE_ONLY_MODEL=pending; ENGINE_ONLY_ARGS=""; ENGINE_ONLY_NOTE=""
CODEC_PXQ_MODEL=pending; CODEC_PXQ_BYTES=""; CODEC_REF_MODEL=pending; CODEC_REF_BYTES=""
CODEC_ARGS=""; CODEC_NOTE=""
PRODUCT_PXA_MODEL=pending; PRODUCT_PXA_ARGS=""; PRODUCT_UP_MODEL=pending; PRODUCT_UP_ARGS=""
PRODUCT_SAME_BASE_WEIGHTS=pending; PRODUCT_NOTE=""
# shellcheck source=/dev/null
. "$RIG_FILE"

say "== bench/fair: rig $RIG_NAME — $RIG_DESC"
[ -n "$RIG_SOURCE" ] && say "   recipes from: $RIG_SOURCE"
[ "$PLAN" = 1 ] && say "   --plan: nothing is started, no GPU is touched"

# ---------------------------------------------------------------------------
# this engine's binary
# ---------------------------------------------------------------------------
if [ -z "${ENGINE_BIN:-}" ]; then
    for d in "$HERE/../../build/bin" "$HERE/../../build-spd/bin" "$HERE/../bin" "$HERE/../../bin" /usr/local/bin; do
        [ -x "$d/llama-server" ] && { ENGINE_BIN="$d/llama-server"; break; }
    done
fi
ENGINE_BIN=${ENGINE_BIN:-}
if [ -z "$ENGINE_BIN" ] || [ ! -x "$ENGINE_BIN" ]; then
    ENGINE_BIN_NOTE="engine binary not present: ${ENGINE_BIN:-<none found; set ENGINE_BIN>}"
    ENGINE_BIN_SHOW="<ENGINE_BIN>"
    ENGINE_BIN=pending
    warn "$ENGINE_BIN_NOTE (set ENGINE_BIN=/path/to/llama-server)"
else
    ENGINE_BIN_NOTE=""
    ENGINE_BIN_SHOW=$ENGINE_BIN
fi

# The comparison engine. The container image ships it at the default path; on a
# bare box, point UPSTREAM_BIN at your own build of the SAME commit — the one in
# the image's org.pxa.upstream.ik.sha label.
UPSTREAM_BIN_SHOW=$UPSTREAM_BIN
if [ -x "$UPSTREAM_BIN" ]; then
    UPSTREAM_NOTE=""
    UPSTREAM_REV_FILE=$UPSTREAM_BIN.rev
    if [ -r "$UPSTREAM_REV_FILE" ]; then
        UPSTREAM_LABEL="upstream $(head -1 "$UPSTREAM_REV_FILE")"
    else
        UPSTREAM_LABEL="upstream engine at $UPSTREAM_BIN (no .rev file next to it — revision unrecorded)"
    fi
else
    UPSTREAM_NOTE="upstream binary not present: $UPSTREAM_BIN"
    UPSTREAM_LABEL="$UPSTREAM_NOTE — the image ships it there, or set UPSTREAM_BIN to your own build of the same commit"
    UPSTREAM_BIN=pending
    warn "$UPSTREAM_NOTE"
fi

# ---------------------------------------------------------------------------
# weights: the existing MANIFEST verification, unchanged in what it checks.
# Its output moved to stderr (stdout is the three blocks), and under --plan a
# missing weights directory is reported rather than fatal — a plan describes a
# run on a rig that may not be this box.
# ---------------------------------------------------------------------------
say "== verifying weights against $(basename "$MANIFEST")"
[ -r "$MANIFEST" ] || die "no manifest at $MANIFEST"

# Only the real "hash  filename" lines are checked; "sha: pending" lines have no hash yet
# (bench/fair/protocol.md rule 7 — never invent one) and are listed, not checked.
MANIFEST_LINES=$(grep -E '^[0-9a-f]{64}  ' "$MANIFEST" || true)
[ -n "$MANIFEST_LINES" ] || die "MANIFEST.sha256 has no checkable entries — refusing to start."

WEIGHTS_OK=0
if [ -d "$WEIGHTS_DIR" ]; then
    cd "$WEIGHTS_DIR" || die "cannot enter $WEIGHTS_DIR"
    # Which of the manifest's files are actually here? Work this out FIRST: with none of them
    # present, `sha256sum -c --ignore-missing` exits non-zero with "no file was verified", which
    # reads as a checksum failure and is not one. An empty directory is a "fetch the weights"
    # message, not an integrity alarm.
    PRESENT=$(echo "$MANIFEST_LINES" | awk '{print $2}' | while read -r f; do [ -f "$f" ] && echo "$f"; done || true)
    if [ -z "$PRESENT" ]; then
        warn "none of the manifest's weight files are present in $WEIGHTS_DIR."
        warn "Download the artifacts named in weights/MANIFEST.sha256 — each entry there carries"
        warn "a provenance comment saying where it is published — or point WEIGHTS_DIR at your copies."
        [ "$PLAN" = 1 ] || die "nothing was checked and nothing was run."
    elif ! echo "$MANIFEST_LINES" | sha256sum -c - --ignore-missing >/dev/null 2>&1; then
        warn "sha256 verification FAILED on a file that IS present:"
        echo "$MANIFEST_LINES" | sha256sum -c - --ignore-missing 2>&1 | grep -v ': OK$' >&2
        die "refusing to start."
    else
        WEIGHTS_OK=1
        say "   verified: $(echo "$PRESENT" | tr '\n' ' ')"
    fi
    cd "$HERE" || exit 1
else
    warn "no weights directory at $WEIGHTS_DIR"
    [ "$PLAN" = 1 ] || die "refusing to start."
fi

# ---------------------------------------------------------------------------
# model resolution + sha state
#   model_path <name>   -> absolute path (or the literal "pending")
#   model_sha   <name>  -> "<64 hex>" | "pending"
# A name with no "/" is looked up in WEIGHTS_DIR, which is how the manifest
# names its files.
# ---------------------------------------------------------------------------
model_path() {
    case "$1" in
        pending|"") echo pending ;;
        */*)        echo "$1" ;;
        *)          echo "$WEIGHTS_DIR/$1" ;;
    esac
}
model_sha() {
    case "$1" in pending|"") echo pending; return ;; esac
    local base sha
    base=$(basename "$1")
    sha=$(awk -v f="$base" '$1 ~ /^[0-9a-f]{64}$/ && $2 == f {print $1; exit}' "$MANIFEST")
    [ -n "$sha" ] && echo "$sha" || echo pending
}
# One line naming a model, its sha state and whether the file is here.
model_line() {
    local m p sha present
    m=$1
    case "$m" in pending|"") echo "pending — no artifact named for this arm on this rig"; return ;; esac
    p=$(model_path "$m"); sha=$(model_sha "$m")
    if [ "$sha" = pending ]; then
        present="SHA PENDING — not in $(basename "$MANIFEST")"
    else
        present="sha256 ${sha%"${sha#????????}"}… (weights/MANIFEST.sha256)"
    fi
    if [ -f "$p" ]; then
        echo "$(basename "$m")  $present"
    else
        echo "$(basename "$m")  $present  [file not on this box: $p]"
    fi
}
# Does this arm block the run? (missing model, or a model with no recorded sha)
arm_pending() {
    case "$1" in pending|"") return 0 ;; esac
    [ "$(model_sha "$1")" = pending ] && return 0
    return 1
}

# ---------------------------------------------------------------------------
# the command a given arm runs, as one pasteable line
# ---------------------------------------------------------------------------
cmd_line() {   # cmd_line <bin> <model> <args> <port>
    local bin=$1 model=$2 args=$3 port=$4 pre=""
    [ -n "$RIG_DEVICES" ] && pre="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$RIG_DEVICES "
    local mp; mp=$(model_path "$model")
    [ "$mp" = pending ] && mp="<no artifact named for this arm>"
    echo "${pre}${bin} -m $mp $args --host 127.0.0.1 --port $port"
}

# ---------------------------------------------------------------------------
# speculative decode / MTP state of one arm.
# The server does not publish this on /props, so it is read from the arm's own
# startup log (the engine prints "PXA_AUTO: spec ..." there) and from the
# command line, which wins over any auto decision. Sets SPEC_STATE (compared
# between arms) and SPEC_EVIDENCE (the line printed when they disagree).
# ---------------------------------------------------------------------------
spec_state() {   # spec_state <arm label> <args> [logfile]
    local label=$1 args=$2 log=${3:-}
    local t line
    case " $args " in
        *" --spec-type "*)
            t=$(printf '%s' "$args" | sed -n 's/.*--spec-type[= ]\([^ ]*\).*/\1/p')
            SPEC_STATE="on:$t"; SPEC_EVIDENCE="$label: command line --spec-type $t"; return ;;
        *" -md "*|*" --model-draft "*)
            SPEC_STATE="on:draft-model"; SPEC_EVIDENCE="$label: command line draft model"; return ;;
    esac
    if [ -n "$log" ] && [ -r "$log" ]; then
        line=$(grep -m1 'PXA_AUTO: spec ' "$log" || true)
        if [ -n "$line" ]; then
            case "$line" in
                *"-> --spec-type "*)
                    t=$(printf '%s' "$line" | sed -n 's/.*-> --spec-type \([^ ]*\).*/\1/p')
                    SPEC_STATE="on:$t" ;;
                *) SPEC_STATE="off" ;;
            esac
            SPEC_EVIDENCE="$label: $line"; return
        fi
        SPEC_STATE="off"; SPEC_EVIDENCE="$label: no speculation line in the server log"; return
    fi
    SPEC_STATE="unknown"; SPEC_EVIDENCE="$label: not started (--plan)"
}

# ---------------------------------------------------------------------------
# run one arm: start the server, wait for it, measure, stop it.
# Sets ARM_PREFILL, ARM_DECODE, ARM_TOKENS, SPEC_STATE, SPEC_EVIDENCE.
# ---------------------------------------------------------------------------
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/pxa-fair.XXXXXX") || die "cannot create a work directory"
# shellcheck disable=SC2329  # invoked by the EXIT trap on the next line
cleanup() { [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null; rm -rf "$WORKDIR"; }
trap cleanup EXIT
SRV_PID=

run_arm() {   # run_arm <label> <bin> <model> <args>
    local label=$1 bin=$2 model=$3 args=$4
    local port=$PORT_NEXT log path
    PORT_NEXT=$((PORT_NEXT + 1))
    ARM_PREFILL="—"; ARM_DECODE="—"; ARM_TOKENS="—"
    path=$(model_path "$model")
    log=$WORKDIR/$(echo "$label" | tr -c 'A-Za-z0-9' '_').log

    say "-- $label: starting $bin on port $port"
    (
        [ -n "$RIG_DEVICES" ] && { export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$RIG_DEVICES"; }
        # shellcheck disable=SC2086
        exec "$bin" -m "$path" $args --host 127.0.0.1 --port "$port"
    ) >"$log" 2>&1 &
    SRV_PID=$!

    local waited=0
    until python3 -c 'import sys,urllib.request;urllib.request.urlopen("http://127.0.0.1:%s/health"%sys.argv[1],timeout=5)' "$port" >/dev/null 2>&1; do
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            warn "$label: server exited before answering /health — last 20 lines of its log:"
            tail -20 "$log" >&2
            SRV_PID=; return 1
        fi
        waited=$((waited + 2)); sleep 2
        [ "$waited" -ge "$BOOT_TIMEOUT" ] && { warn "$label: no /health after ${BOOT_TIMEOUT}s"; kill "$SRV_PID" 2>/dev/null; SRV_PID=; return 1; }
    done
    say "-- $label: up after ${waited}s, ${REPS} timed repeats (+1 discarded warmup)"

    local result
    result=$(python3 - "$port" "$FILL_TOKENS" "$N_PREDICT" "$REPS" <<'PY'
import json, statistics, sys, urllib.request

port, fill, npred, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
BASE = "http://127.0.0.1:%d" % port

# protocol.md rule 6: a unique prompt per repeat, including the discarded warmup. A repeated
# literal prompt lets a KV/prompt cache turn a decode benchmark into a cache-hit benchmark;
# cache_prompt stays false for the same reason. The filler is deterministic, so a rerun on the
# same rig sends byte-identical prompts for the same repeat index.
POOL = ("the quick sequence of tokens fills this context window with ordinary prose so that the "
        "prefill measurement has real work to do rather than a repeated phrase that any cache "
        "would recognise immediately and answer from memory instead of computing ").split()

def prompt(i, ntok):
    words, k = ["Repeat", "number", str(i), "begins", "here", "."], 0
    while len(words) < ntok:
        words.append(POOL[(k + i * 7) % len(POOL)]); k += 1
    return " ".join(words) + "\nSummarise the passage above in one sentence."

def once(i):
    body = json.dumps({"prompt": prompt(i, fill), "n_predict": npred, "temperature": 0,
                       "seed": 42, "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    t = j.get("timings", {})
    return (t.get("prompt_per_second"), t.get("predicted_per_second"), t.get("prompt_n"))

once(0)                                   # protocol.md rule 3: warmup, discarded
rows = [once(i) for i in range(1, reps + 1)]
pre = [r[0] for r in rows if r[0]]
dec = [r[1] for r in rows if r[1]]
tok = [r[2] for r in rows if r[2]]
print("%.1f %.2f %d" % (statistics.median(pre), statistics.median(dec),
                        statistics.median(tok) if tok else 0))
PY
    ) || { warn "$label: measurement failed"; kill "$SRV_PID" 2>/dev/null; SRV_PID=; return 1; }

    ARM_PREFILL=$(echo "$result" | awk '{print $1}')
    ARM_DECODE=$(echo "$result" | awk '{print $2}')
    ARM_TOKENS=$(echo "$result" | awk '{print $3}')
    spec_state "$label" "$args" "$log"

    kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=
    say "-- $label: prefill ${ARM_PREFILL} t/s @ ${ARM_TOKENS} tok, decode ${ARM_DECODE} t/s, spec ${SPEC_STATE}"
    return 0
}

# ---------------------------------------------------------------------------
# one block. Every block is: two arms, one table, the commands, and the reason
# for any cell that is not a number.
# ---------------------------------------------------------------------------
# NAME is the short label in the table column (kept short so the columns line up);
# LABEL is the full one-line description printed above it; BIN_SHOW is the path the
# printed command uses, which is the expected path even when the binary is missing.
BLOCK_A_NAME=""; BLOCK_A_BIN=""; BLOCK_A_MODEL=""; BLOCK_A_ARGS=""; BLOCK_A_LABEL=""; BLOCK_A_BIN_SHOW=""; BLOCK_A_BIN_NOTE=""
BLOCK_B_NAME=""; BLOCK_B_BIN=""; BLOCK_B_MODEL=""; BLOCK_B_ARGS=""; BLOCK_B_LABEL=""; BLOCK_B_BIN_SHOW=""; BLOCK_B_BIN_NOTE=""

emit_block() {   # emit_block <TITLE> <subtitle> <note>
    local title=$1 subtitle=$2 note=$3
    local blocked="" a_pre="—" a_dec="—" a_tok="—" b_pre="—" b_dec="—" b_tok="—"
    local a_spec="" b_spec="" a_ev="" b_ev=""
    local port_a=$PORT_NEXT port_b=$((PORT_NEXT + 1))

    # ---- can this block run at all?
    if arm_pending "$BLOCK_A_MODEL"; then blocked="${blocked}A"; fi
    if arm_pending "$BLOCK_B_MODEL"; then blocked="${blocked}B"; fi
    [ "$BLOCK_A_BIN" = pending ] && blocked="${blocked}a"
    [ "$BLOCK_B_BIN" = pending ] && blocked="${blocked}b"

    if [ "$PLAN" = 0 ] && [ -z "$blocked" ]; then
        if run_arm "$BLOCK_A_NAME" "$BLOCK_A_BIN" "$BLOCK_A_MODEL" "$BLOCK_A_ARGS"; then
            a_pre=$ARM_PREFILL; a_dec=$ARM_DECODE; a_tok=$ARM_TOKENS; a_spec=$SPEC_STATE; a_ev=$SPEC_EVIDENCE
        else
            a_spec=unknown; a_ev="$BLOCK_A_NAME: arm failed to run"
        fi
        if run_arm "$BLOCK_B_NAME" "$BLOCK_B_BIN" "$BLOCK_B_MODEL" "$BLOCK_B_ARGS"; then
            b_pre=$ARM_PREFILL; b_dec=$ARM_DECODE; b_tok=$ARM_TOKENS; b_spec=$SPEC_STATE; b_ev=$SPEC_EVIDENCE
        else
            b_spec=unknown; b_ev="$BLOCK_B_NAME: arm failed to run"
        fi
    else
        spec_state "$BLOCK_A_NAME" "$BLOCK_A_ARGS"; a_spec=$SPEC_STATE; a_ev=$SPEC_EVIDENCE
        spec_state "$BLOCK_B_NAME" "$BLOCK_B_ARGS"; b_spec=$SPEC_STATE; b_ev=$SPEC_EVIDENCE
    fi

    out '```text'
    out "$title — $subtitle"
    out "rig: $RIG_NAME — $RIG_DESC"
    out "protocol: bench/fair/protocol.md — /completion, temp 0, seed 42, n=$REPS median,"
    out "          1 warmup discarded, unique prompt per repeat, cache_prompt off"
    out "arm A: $BLOCK_A_LABEL"
    out "       binary $BLOCK_A_BIN_SHOW"
    out "       model  $(model_line "$BLOCK_A_MODEL")"
    out "arm B: $BLOCK_B_LABEL"
    out "       binary $BLOCK_B_BIN_SHOW"
    out "       model  $(model_line "$BLOCK_B_MODEL")"
    [ -n "$note" ] && out "note: $note"

    # ---- MTP/speculation agreement (protocol.md rule 5)
    if [ "$PLAN" = 1 ]; then
        out "MTP/speculation: checked at run time from each arm's startup log; the run exits 2"
        out "                 if the two arms do not report the same state"
    elif [ -n "$blocked" ]; then
        out "MTP/speculation: not checked — this block did not run"
    elif [ "$a_spec" != "$b_spec" ]; then
        out "MTP/speculation: ARMS DISAGREE — this comparison is void"
        out "  $a_ev"
        out "  $b_ev"
        fail2 "$title: speculative decode state differs between the arms ($a_spec vs $b_spec) — a codec or engine delta must not be a speculation delta in disguise (protocol.md rule 5)"
    else
        out "MTP/speculation: $a_spec on both arms"
        out "  $a_ev"
        out "  $b_ev"
    fi

    out ""
    printf '  %-34s %14s %14s\n' "arm" "prefill t/s" "decode t/s"
    printf '  %-34s %14s %14s\n' "A  $BLOCK_A_NAME" "$a_pre" "$a_dec"
    printf '  %-34s %14s %14s\n' "B  $BLOCK_B_NAME" "$b_pre" "$b_dec"
    out ""
    if [ "$PLAN" = 1 ]; then
        out "  prefill prompt: ~${FILL_TOKENS} tokens (the count printed here is the one the server"
        out "  itself reports having tokenized, per arm, not the target above)"
    else
        out "  prefill measured at the prompt length the server reported: A ${a_tok} tok, B ${b_tok} tok"
    fi
    out "  decode measured single stream over ${N_PREDICT} generated tokens"
    out ""
    out "commands:"
    out "  A: $(cmd_line "$BLOCK_A_BIN_SHOW" "$BLOCK_A_MODEL" "$BLOCK_A_ARGS" "$port_a")"
    out "  B: $(cmd_line "$BLOCK_B_BIN_SHOW" "$BLOCK_B_MODEL" "$BLOCK_B_ARGS" "$port_b")"

    if [ -n "$blocked" ]; then
        out ""
        out "NOT RUN:"
        case "$blocked" in *A*) out "  arm A: $(model_line "$BLOCK_A_MODEL")"; fail2 "$title: arm A has no verified artifact" ;; esac
        case "$blocked" in *B*) out "  arm B: $(model_line "$BLOCK_B_MODEL")"; fail2 "$title: arm B has no verified artifact" ;; esac
        case "$blocked" in *a*) out "  arm A: $BLOCK_A_BIN_NOTE"; fail2 "$title: $BLOCK_A_BIN_NOTE" ;; esac
        case "$blocked" in *b*) out "  arm B: $BLOCK_B_BIN_NOTE"; fail2 "$title: $BLOCK_B_BIN_NOTE" ;; esac
    elif [ "$PLAN" = 1 ]; then
        out ""
        out "NOT RUN: --plan (this is the plan, not a measurement)"
    fi
    out '```'
    out ""
}

# ---------------------------------------------------------------------------
# ENGINE-ONLY — same GGUF, two engines. The file must be one BOTH engines read:
# a PXQ file loads only in this one, which makes that a product comparison.
# ---------------------------------------------------------------------------
BLOCK_A_NAME="pxa"
BLOCK_A_LABEL="pxa — this engine"
BLOCK_A_BIN=$ENGINE_BIN
BLOCK_A_BIN_SHOW=$ENGINE_BIN_SHOW
BLOCK_A_BIN_NOTE=$ENGINE_BIN_NOTE
BLOCK_A_MODEL=$ENGINE_ONLY_MODEL
BLOCK_A_ARGS=$ENGINE_ONLY_ARGS
BLOCK_B_NAME="upstream"
BLOCK_B_LABEL="$UPSTREAM_LABEL"
BLOCK_B_BIN=$UPSTREAM_BIN
BLOCK_B_BIN_SHOW=$UPSTREAM_BIN_SHOW
BLOCK_B_BIN_NOTE=$UPSTREAM_NOTE
BLOCK_B_MODEL=$ENGINE_ONLY_MODEL
BLOCK_B_ARGS=$ENGINE_ONLY_ARGS
emit_block "ENGINE-ONLY" "same GGUF, two engines — isolates the kernel and scheduler work" "$ENGINE_ONLY_NOTE"

# ---------------------------------------------------------------------------
# CODEC-ONLY — same engine, PXQ4 vs MXFP4 at matched bytes.
# ---------------------------------------------------------------------------
CODEC_BYTES_NOTE=$CODEC_NOTE
if [ -n "$CODEC_PXQ_BYTES" ] && [ -n "$CODEC_REF_BYTES" ]; then
    CODEC_BYTES_NOTE="matched bytes: PXQ ${CODEC_PXQ_BYTES}, reference ${CODEC_REF_BYTES}${CODEC_NOTE:+ — $CODEC_NOTE}"
fi
BLOCK_A_NAME="pxa, PXQ4"
BLOCK_A_LABEL="pxa — PXQ4 file"
BLOCK_A_BIN=$ENGINE_BIN
BLOCK_A_BIN_SHOW=$ENGINE_BIN_SHOW
BLOCK_A_BIN_NOTE=$ENGINE_BIN_NOTE
BLOCK_A_MODEL=$CODEC_PXQ_MODEL
BLOCK_A_ARGS=$CODEC_ARGS
BLOCK_B_NAME="pxa, MXFP4"
BLOCK_B_LABEL="pxa — MXFP4 file, the same binary and the same flags"
BLOCK_B_BIN=$ENGINE_BIN
BLOCK_B_BIN_SHOW=$ENGINE_BIN_SHOW
BLOCK_B_BIN_NOTE=$ENGINE_BIN_NOTE
BLOCK_B_MODEL=$CODEC_REF_MODEL
BLOCK_B_ARGS=$CODEC_ARGS
emit_block "CODEC-ONLY" "same engine, two files at matched bytes — isolates the codec" "$CODEC_BYTES_NOTE"

# ---------------------------------------------------------------------------
# PRODUCT — best documented recipe per side. Not a controlled comparison and it
# does not pretend to be: different quant, different flags, the thing you would
# actually run. The one rule it does keep is the base weights (protocol: an
# expert-codec row whose two sides are different models is not a product row).
# ---------------------------------------------------------------------------
PRODUCT_BLOCK_NOTE=$PRODUCT_NOTE
case "$PRODUCT_SAME_BASE_WEIGHTS" in
    yes) : ;;
    no)
        PRODUCT_PXA_MODEL=pending; PRODUCT_UP_MODEL=pending
        PRODUCT_BLOCK_NOTE="refused: the rig file declares the two sides are not the same base weights. Same architecture and size class is not the same model; that comparison belongs in bench/fair-battle.md's expert-codec section, not in PRODUCT.${PRODUCT_NOTE:+ — $PRODUCT_NOTE}" ;;
    pending)
        PRODUCT_BLOCK_NOTE="the same-base-weights rule cannot be checked until this rig names both product artifacts.${PRODUCT_NOTE:+ — $PRODUCT_NOTE}" ;;
    *)
        PRODUCT_PXA_MODEL=pending; PRODUCT_UP_MODEL=pending
        PRODUCT_BLOCK_NOTE="refused: PRODUCT_SAME_BASE_WEIGHTS is '$PRODUCT_SAME_BASE_WEIGHTS' — say yes, no, or pending in the rig file.${PRODUCT_NOTE:+ — $PRODUCT_NOTE}" ;;
esac
BLOCK_A_NAME="pxa, best recipe"
BLOCK_A_LABEL="pxa — this engine"
BLOCK_A_BIN=$ENGINE_BIN
BLOCK_A_BIN_SHOW=$ENGINE_BIN_SHOW
BLOCK_A_BIN_NOTE=$ENGINE_BIN_NOTE
BLOCK_A_MODEL=$PRODUCT_PXA_MODEL
BLOCK_A_ARGS=$PRODUCT_PXA_ARGS
BLOCK_B_NAME="upstream, best recipe"
BLOCK_B_LABEL="$UPSTREAM_LABEL"
BLOCK_B_BIN=$UPSTREAM_BIN
BLOCK_B_BIN_SHOW=$UPSTREAM_BIN_SHOW
BLOCK_B_BIN_NOTE=$UPSTREAM_NOTE
BLOCK_B_MODEL=$PRODUCT_UP_MODEL
BLOCK_B_ARGS=$PRODUCT_UP_ARGS
emit_block "PRODUCT" "each side its own quant and its own flags — what you would actually run" "$PRODUCT_BLOCK_NOTE"

# ---------------------------------------------------------------------------
say "== done"
[ "$WEIGHTS_OK" = 1 ] || say "   (weights were not verified on this box)"
say "   Not in the three blocks above, deliberately: an expert-codec comparison whose two"
say "   sides are different base weights, and multi-box / NVLink / sidecar-speculation rows."
say "   Those live in bench/fair-battle.md with their own caveats."
if [ "$PLAN" = 1 ]; then
    say "== plan only: nothing was started and no GPU was touched. Drop --plan to measure."
    exit 0
fi
[ "$EXIT_CODE" = 0 ] || say "== exit 2: one or more blocks above is not a fair comparison (see the reasons)."
exit "$EXIT_CODE"
