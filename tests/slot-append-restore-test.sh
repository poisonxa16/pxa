#!/usr/bin/env bash
# Slot save/restore append-only test (2026-09-13).
#
# What it proves: after a slot's state is saved to disk, erased and restored, a request that
# STRICTLY EXTENDS the restored prompt must prefill only the new tokens. The failure mode this
# guards is a restore that is reported as successful and then silently thrown away, because the
# re-entry path asks for a rollback checkpoint that a restored slot does not have and falls into
# "forcing full prompt re-processing" - a 100k prefix recomputed from zero while the server logs
# a healthy restore.
#
# The instrument is the request's own timings.prompt_n (tokens actually prefilled), never a wall
# clock, and the run ends with a POSITIVE CONTROL: the same extended prompt on an erased slot,
# which must report the full count. A test whose "no recompute" reading is produced by an
# instrument that cannot see a recompute would pass on a broken server.
#
# The model must be hybrid (recurrent + attention): context checkpoints exist only for those
# architectures, so a dense model cannot reach the branch under test at all.
#
# Usage: slot-append-restore-test.sh [port] [outdir]
set -u

PORT=${1:-18771}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${2:-$ROOT/build-cpu/slot-append-out}

BIN=${PXA_LLAMA_SERVER:-$ROOT/build-cpu/bin/llama-server}
MODEL=${PXA_HYBRID_MODEL:?set to a small hybrid GGUF, e.g. Qwen3.5-0.8B-Q8_0.gguf}
NPRED=${NPRED:-1}

mkdir -p "$OUT" "$OUT/slots"
rm -f "$OUT"/slots/* "$OUT"/*.json "$OUT"/server.log

if [ ! -x "$BIN" ];   then echo "FAIL: no server binary at $BIN"; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "FAIL: no model at $MODEL";       exit 2; fi

echo "== booting server on port $PORT"
"$BIN" -m "$MODEL" -c 8192 -np 2 --kv-unified --slot-save-path "$OUT/slots/" \
       --host 127.0.0.1 --port "$PORT" -t 8 --no-warmup > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill '"$SRV"' 2>/dev/null' EXIT

# wait for health, then check IDENTITY: /props must name the model we launched, or some other
# server owns this port and every number below would be someone else's.
if ! python3 - "$PORT" "$MODEL" <<'PY'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
for _ in range(240):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
        break
    except Exception:
        time.sleep(1)
else:
    print("FAIL: server did not come up"); sys.exit(2)
props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10).read())
if model not in json.dumps(props):
    print(f"FAIL: port {port} is serving something else: {json.dumps(props)[:300]}"); sys.exit(2)
print("server up, identity confirmed")
PY
then tail -20 "$OUT/server.log"; exit 2; fi

python3 - "$PORT" "$OUT" <<'PY'
import json, sys, urllib.request

port, out = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"

def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode())

# a long, deterministic prompt and a strict extension of it
para = ("The cache holds one cell per token and every cell carries the sequence ids that own it. "
        "A restored slot owns cells it never computed in this process, which is the whole point. ")
prompt = "".join(f"[{i}] " + para for i in range(60))
ext    = "Question: how many cells does a restored slot own? Answer:"

n_prompt = len(post("/tokenize", {"content": prompt})["tokens"])
n_ext    = len(post("/tokenize", {"content": prompt + ext})["tokens"]) - n_prompt
print(f"prompt tokens = {n_prompt}, extension tokens = {n_ext}")

def complete(text, slot=0):
    r = post("/completion", {"prompt": text, "n_predict": 1, "cache_prompt": True,
                             "id_slot": slot, "temperature": 0, "seed": 1})
    return r["timings"]["prompt_n"], r

pn_cold, _ = complete(prompt)
print(f"1. cold prefill of the base prompt: prompt_n = {pn_cold}")

print("2. save:   ", post("/slots/0?action=save",    {"filename": "append.bin"}))
print("3. erase:  ", post("/slots/0?action=erase",   {}))
print("4. restore:", post("/slots/0?action=restore", {"filename": "append.bin"}))

pn_ext, _ = complete(prompt + ext)
print(f"5. extended request after restore: prompt_n = {pn_ext}")

post("/slots/0?action=erase", {})
pn_ctrl, _ = complete(prompt + ext)
print(f"6. POSITIVE CONTROL, same request on an erased slot: prompt_n = {pn_ctrl}")

verdict = {
    "n_prompt": n_prompt, "n_ext": n_ext,
    "pn_cold": pn_cold, "pn_after_restore": pn_ext, "pn_control": pn_ctrl,
}
# the control must show a full recompute, or the instrument proves nothing
ok_control = pn_ctrl >= n_prompt
# append-only: only the new tokens (allow the one-token logits rule and tokenizer slack)
ok_append  = pn_ext <= n_ext + 8
verdict["control_ok"] = ok_control
verdict["append_only"] = ok_append
verdict["verdict"] = "PASS" if (ok_control and ok_append) else ("REPRODUCED" if ok_control else "INSTRUMENT-DEAD")
print(json.dumps(verdict, indent=2))
open(out + "/append-restore.json", "w").write(json.dumps(verdict, indent=2))
sys.exit(0 if verdict["verdict"] == "PASS" else 1)
PY
RC=$?
kill $SRV 2>/dev/null
echo "== rc=$RC (report: $OUT/append-restore.json, server log: $OUT/server.log)"
exit $RC
