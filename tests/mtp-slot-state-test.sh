#!/usr/bin/env bash
# Speculative slot state across erase and save/restore (2026-09-13).
#
# What it is for. A speculative slot's state lives in three places: the target's KV, the drafter's
# companion context KV, and a small per-sequence carry the drafter keeps outside any llama memory
# object. Erasing a slot used to empty only the first. Saving a slot used to write only the first.
# Both leave the next occupant of that sequence id drafting against the PREVIOUS occupant's tokens:
# a slot that reports itself erased or freshly restored, and a drafter that is quietly somewhere
# else. This test measures that directly, on the CPU, from the server's own draft counters.
#
# How it decides. Every request is greedy and byte-identical between arms, and every measurement is
# a pair: the SAME prompt is asked once on a clean sequence (the reference) and once after another
# prompt has been through that same slot and the slot has been erased. On a server that clears
# everything the two must agree in drafts proposed, drafts accepted and generated text. The arm
# argument sets the two levers (both default ON) so the pre-fix behaviour can be measured on the
# same binary - a fix whose "before" picture comes from a different build proves less.
#
#   on  : PXA_SLOT_ERASE_SPEC=1 PXA_SLOT_SPEC_STATE=1   (the shipped default)
#   off : PXA_SLOT_ERASE_SPEC=0 PXA_SLOT_SPEC_STATE=0   (the pre-fix conduct)
#
# The model must have an MTP head, or there is no companion context to get wrong and every arm is
# trivially equal. On CPU this is slow by construction (a 27B at well under one token a second);
# the prompts are short on purpose and the whole run is about ten minutes per arm.
#
# Usage: mtp-slot-state-test.sh [on|off] [port] [outdir]
set -u

ARM=${1:-on}
PORT=${2:-18772}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${3:-$ROOT/build-cpu/mtp-slot-state-out}

BIN=${PXA_LLAMA_SERVER:-$ROOT/build-cpu/bin/llama-server}
MODEL=${PXA_MTP_MODEL:?set to a GGUF with an MTP head, e.g. Qwen3.8-27B-PXQ2.gguf}
NPRED=${NPRED:-12}

case "$ARM" in
    on)  LEV_ERASE=1; LEV_STATE=1 ;;
    off) LEV_ERASE=0; LEV_STATE=0 ;;
    *)   echo "FAIL: arm must be on|off"; exit 2 ;;
esac

mkdir -p "$OUT" "$OUT/slots"
rm -f "$OUT"/slots/* "$OUT/$ARM.json" "$OUT/$ARM-server.log"

if [ ! -x "$BIN" ];   then echo "FAIL: no server binary at $BIN"; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "FAIL: no model at $MODEL";       exit 2; fi

echo "== arm $ARM (PXA_SLOT_ERASE_SPEC=$LEV_ERASE PXA_SLOT_SPEC_STATE=$LEV_STATE) on port $PORT"
PXA_SLOT_ERASE_SPEC=$LEV_ERASE PXA_SLOT_SPEC_STATE=$LEV_STATE \
"$BIN" -m "$MODEL" -c 4096 -np 2 --kv-unified --spec-type mtp \
       --slot-save-path "$OUT/slots/" --host 127.0.0.1 --port "$PORT" -t 48 --no-warmup \
       > "$OUT/$ARM-server.log" 2>&1 &
SRV=$!
trap 'kill '"$SRV"' 2>/dev/null' EXIT

if ! python3 - "$PORT" "$MODEL" <<'PY'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
for _ in range(600):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
        break
    except Exception:
        time.sleep(1)
else:
    print("FAIL: server did not come up"); sys.exit(2)
props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10).read())
if model not in json.dumps(props):
    print(f"FAIL: port {port} is serving something else"); sys.exit(2)
print("server up, identity confirmed")
PY
then tail -20 "$OUT/$ARM-server.log"; exit 2; fi

python3 - "$PORT" "$OUT" "$ARM" "$NPRED" <<'PY'
import hashlib, json, sys, urllib.request

port, out, arm, npred = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
base = f"http://127.0.0.1:{port}"

def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read().decode())

P_SUBJECT = ("A cache line holds a key and a value. State in one short paragraph what the key is for.")
P_POISON  = ("Rivers carve canyons over millions of years. Name two forces that do the carving, briefly.")
EXT       = " Add one more sentence."

def ask(text, label, slot=0, n=None):
    r = post("/completion", {"prompt": text, "n_predict": n or npred, "cache_prompt": True,
                             "id_slot": slot, "temperature": 0, "seed": 1})
    t = r["timings"]
    row = {
        "label": label,
        "prompt_n": t["prompt_n"], "predicted_n": t["predicted_n"],
        "draft_n": t.get("draft_n", 0), "draft_n_accepted": t.get("draft_n_accepted", 0),
        "sha": hashlib.sha256(r["content"].encode()).hexdigest()[:16],
        "n_bytes": len(r["content"]),
    }
    print(json.dumps(row))
    return row

def erase(slot=0):
    return post(f"/slots/{slot}?action=erase", {})

rows = {}
rows["ref"] = ask(P_SUBJECT, "1 reference: subject on a clean sequence")
print("save:", post("/slots/0?action=save", {"filename": "mtp.bin"}))
erase()
rows["poison1"] = ask(P_POISON, "2 poison: a different prompt through the same slot")
erase()
rows["after_erase"] = ask(P_SUBJECT, "3 subject again, after the erase")
erase()
rows["poison2"] = ask(P_POISON, "4 poison again, before the restore")
erase()
print("restore:", post("/slots/0?action=restore", {"filename": "mtp.bin"}))
rows["after_restore"] = ask(P_SUBJECT + EXT, "5 subject + extension, on the restored slot")

res = {"arm": arm, "rows": rows,
       "erase_matches_reference": (rows["after_erase"]["sha"] == rows["ref"]["sha"] and
                                   rows["after_erase"]["draft_n"] == rows["ref"]["draft_n"] and
                                   rows["after_erase"]["draft_n_accepted"] == rows["ref"]["draft_n_accepted"])}
print(json.dumps(res, indent=2))
open(f"{out}/{arm}.json", "w").write(json.dumps(res, indent=2))
PY
RC=$?
kill $SRV 2>/dev/null
echo "== arm $ARM rc=$RC (report: $OUT/$ARM.json)"
exit $RC
