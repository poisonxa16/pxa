#!/usr/bin/env bash
# Prompt-cache park test (2026-09-20).
#
# What it proves, in three readings taken from the request's own timings.prompt_n (tokens the
# server actually prefilled) and never from a wall clock:
#
#   1. PARK      a conversation that loses its slot to another conversation is given back its
#                context on its next turn instead of being reprocessed.
#   2. PERSIST   with PXA_CACHE_DISK naming a file, the same is true ACROSS A RESTART: the parked
#                conversation is read back from disk at start-up.
#   3. CONTROL   with the cache off, the same third turn reports the FULL prompt. Without this
#                reading the first two prove nothing - an instrument that cannot see a
#                reprocess would report "no reprocess" on a server that reprocesses everything.
#
# The model must be hybrid (recurrent + attention): that is the case where the state being parked
# is more than attention cells, and it is the case this path gets wrong first.
#
# Runs on CPU (-ngl 0); no card is needed.
#
# Usage: prompt-cache-park-test.sh [port] [outdir]
set -u

PORT=${1:-18781}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${2:-$ROOT/build/prompt-cache-park-out}

BIN=${PXA_LLAMA_SERVER:-$ROOT/build/bin/llama-server}
MODEL=${PXA_HYBRID_MODEL:?set to a small hybrid GGUF, e.g. Qwen3.5-0.8B-Q8_0.gguf}

mkdir -p "$OUT"
rm -f "$OUT"/*.json "$OUT"/*.log "$OUT"/cache.bin

if [ ! -x "$BIN" ];   then echo "FAIL: no server binary at $BIN"; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "FAIL: no model at $MODEL";       exit 2; fi

SRV=""
boot() { # boot <tag> <extra server args...>
    local tag=$1; shift
    "$BIN" -m "$MODEL" -ngl 0 -c 32768 -np 1 --host 127.0.0.1 --port "$PORT" -t 8 --no-warmup \
           "$@" > "$OUT/server-$tag.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 240); do
        if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
            return 0
        fi
        kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server died during load ($tag)"; tail -20 "$OUT/server-$tag.log"; exit 2; }
        sleep 1
    done
    echo "FAIL: server did not come up ($tag)"; exit 2
}

stop() { # SIGINT, so the shutdown path that writes the cache file actually runs
    [ -n "$SRV" ] || return 0
    kill -INT "$SRV" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
    SRV=""
}
trap 'kill -9 "$SRV" 2>/dev/null' EXIT

ask() { # ask <who> <tag> -> prints "prompt_n sha"
    python3 - "$PORT" "$1" "$2" "$OUT" <<'PY'
import hashlib, json, sys, urllib.request
port, who, tag, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = "\n".join("%s line %d: the quick brown fox jumps over the lazy dog %d" % (who, i, i) for i in range(400))
req = urllib.request.Request(
    "http://127.0.0.1:%s/completion" % port,
    data=json.dumps({"prompt": body + "\n\nfinish the list.\n", "n_predict": 8,
                     "temperature": 0.0, "top_k": 1, "seed": 1234,
                     "cache_prompt": True, "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
open("%s/%s.json" % (out, tag), "w").write(json.dumps(r, indent=1))
t = r.get("timings", {})
n = r.get("prompt_n", t.get("prompt_n", r.get("tokens_evaluated", -1)))
print("%s %s" % (n, hashlib.sha256(r.get("content", "").encode()).hexdigest()[:12]))
PY
}

echo "== 1/4 park: boot with the RAM cache and a disk file"
export PXA_CACHE_DISK="$OUT/cache.bin"
boot park --cache-ram 4096 --cache-ram-similarity 0.5

read -r N_A1 SHA_A1 <<< "$(ask alpha a1)"     # alpha, cold
read -r N_B1 SHA_B1 <<< "$(ask bravo b1)"     # bravo takes the slot, alpha is parked
read -r N_A2 SHA_A2 <<< "$(ask alpha a2)"     # alpha comes back from the RAM cache
stop

echo "== 2/4 persist: restart against the same file"
boot persist --cache-ram 4096 --cache-ram-similarity 0.5
read -r N_A3 SHA_A3 <<< "$(ask alpha a3)"     # alpha comes back from DISK
stop

echo "== 3/4 control: restart with the cache off"
unset PXA_CACHE_DISK
boot control --cache-ram 0
read -r N_A4 SHA_A4 <<< "$(ask alpha a4)"     # must be the full prompt
stop

# The candidate scan has two branches, and only one of them was covered above. With reasoning
# tokens EXCLUDED (the default) the scan works on filtered copies; with them INCLUDED it works on
# the lists themselves, and that is the branch in which the scan used to empty what it measured -
# both the candidates and, one branch above, the caller's own prompt. So run the park reading again
# with `--reasoning-tokens none`, which is what selects that branch.
echo "== 4/4 park with reasoning tokens included (the other scan branch)"
boot parkri --cache-ram 4096 --cache-ram-similarity 0.5 --reasoning-tokens none
read -r N_C1 SHA_C1 <<< "$(ask alpha c1)"     # alpha, cold
read -r N_D1 SHA_D1 <<< "$(ask bravo d1)"     # bravo takes the slot, alpha is parked
read -r N_C2 SHA_C2 <<< "$(ask alpha c2)"     # alpha comes back
stop

echo
printf 'alpha cold        prompt_n=%s sha=%s\n' "$N_A1" "$SHA_A1"
printf 'bravo             prompt_n=%s sha=%s\n' "$N_B1" "$SHA_B1"
printf 'alpha from RAM    prompt_n=%s sha=%s\n' "$N_A2" "$SHA_A2"
printf 'alpha from DISK   prompt_n=%s sha=%s\n' "$N_A3" "$SHA_A3"
printf 'alpha control     prompt_n=%s sha=%s\n' "$N_A4" "$SHA_A4"
printf 'alpha from RAM, reasoning tokens included  prompt_n=%s sha=%s\n' "$N_C2" "$SHA_C2"
echo

rc=0
[ "$N_A4" -gt 0 ] 2>/dev/null || { echo "FAIL: the control did not report a prompt count; the instrument is blind"; exit 2; }
if [ "$N_A2" -ge "$N_A4" ]; then echo "FAIL: park - alpha's second turn reprocessed $N_A2 of $N_A4 tokens"; rc=1
else echo "PASS: park    - $N_A2 tokens reprocessed against the control's $N_A4"; fi
if [ "$N_A3" -ge "$N_A4" ]; then echo "FAIL: persist - after the restart alpha reprocessed $N_A3 of $N_A4 tokens"; rc=1
else echo "PASS: persist - $N_A3 tokens reprocessed after a restart, against the control's $N_A4"; fi
if [ "$N_C2" -ge "$N_A4" ]; then echo "FAIL: park (reasoning tokens included) - alpha reprocessed $N_C2 of $N_A4 tokens"; rc=1
else echo "PASS: park    - $N_C2 tokens reprocessed with reasoning tokens included, against the control's $N_A4"; fi

# The shas are REPORTED, not asserted. The prompt cache's replay path is already on record as
# changing temperature-0 output on this tree, so a difference here is a property of the cache and
# not of the park; asserting on it would turn a known issue into a red test that teaches nothing.
if [ "$SHA_A2" = "$SHA_A4" ] && [ "$SHA_A3" = "$SHA_A4" ]; then
    echo "NOTE: restored continuations match the uncached control ($SHA_A4)"
else
    echo "NOTE: restored continuations differ from the uncached control - RAM $SHA_A2, disk $SHA_A3, control $SHA_A4"
fi

exit $rc
