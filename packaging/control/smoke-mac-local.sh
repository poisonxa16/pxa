#!/bin/bash
# Smoke test of the macOS Control zip, run ON the Mac that tests it (an Intel Mac, an Apple silicon Mac, or the GitHub macos-14
# runner): no model, no GPU, no network beyond 127.0.0.1.
#   bash smoke-mac-local.sh <pxa-control-...-macos-*.zip> [workdir]      env: PORT (7878)  CHAT_SHOT=<png path> (also render the chat page headless)
# Checks: the zip's execute bits, `lipo -archs` has this Mac's architecture, the app starts in a fresh HOME (cold start time),
# /api/health 401 without the token, 200 with it, the page + chat assets + PWA files answer 200, https works (system CA bundle),
# and with CHAT_SHOT: a headless WebKit loads the chat tab and sees the chat page. Prints PASS/FAIL lines; exit 1 on any FAIL. The token is
# read from the file the app wrote and never printed.
set -uo pipefail
ZIP="${1:?zip}"; W="${2:-$(mktemp -d)}"; PORT="${PORT:-7878}"
fail=0; ck() { if [ "$2" = ok ]; then echo "PASS $1 $3"; else echo "FAIL $1 $3"; fail=1; fi; }
ARCH="$(uname -m)"; echo "host: $ARCH  macOS $(sw_vers -productVersion)  $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
case "$ARCH" in arm64) RUN=(arch -arm64);; *) RUN=(arch -x86_64);; esac
curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/api/health" && { echo "FAIL port $PORT is already in use (an older Control?): set PORT=..."; exit 1; }
rm -rf "$W/pkg" "$W/home"; mkdir -p "$W/pkg" "$W/home"
unzip -q "$ZIP" -d "$W/pkg"; D="$(echo "$W"/pkg/*/)"
[ -x "$D/pxa-control" ] && ck unzip-exec ok "(execute bit kept)" || ck unzip-exec fail ""
A="$(lipo -archs "$D/pxa-control" 2>&1)"; echo "archs: $A"
case " $A " in *" $ARCH "*) ck lipo-has-host-arch ok "($A)";; *) ck lipo-has-host-arch fail "($A)";; esac
( cd "$D" && HOME="$W/home" nohup "${RUN[@]}" ./pxa-control --lan --no-browser --no-tray --port "$PORT" > "$W/run.log" 2>&1 & )
t0=$(date +%s); up=fail
for i in $(seq 1 120); do curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/api/health" && { up=ok; break; }; sleep 2; done
ck cold-start $up "$(( $(date +%s) - t0 )) s to /api/health, native run via: ${RUN[*]}"
[ "$up" = ok ] || { tail -20 "$W/run.log"; exit 1; }
VER="$("${RUN[@]}" "$D/pxa-control" --version 2>&1 | head -1)"; echo "version: $VER"
T="$(cat "$W/home/.config/pxa/token" 2>/dev/null)"; B="http://127.0.0.1:$PORT"
c() { curl -s -o /dev/null -m 8 -w '%{http_code}' "$@"; }
[ "$(c $B/api/info)" = 401 ] && ck token-gate-401 ok "" || ck token-gate-401 fail ""
[ "$(c -H "X-PXA-Token: $T" $B/api/info)" = 200 ] && ck api-info-200 ok "" || ck api-info-200 fail ""
bad=""; for p in / /chat.css /chat-markdown.css /pwa.js /hosts.html /manifest.webmanifest /api/chat/servers; do
  [ "$(c -H "X-PXA-Token: $T" "$B$p")" = 200 ] || bad="$bad $p"; done
[ -z "$bad" ] && ck assets-200 ok "(page, chat css, pwa, hosts, chat api)" || ck assets-200 fail "($bad)"
n="$(curl -s -m 60 -H "X-PXA-Token: $T" "$B/api/models/available" | head -c 300)"
case "$n" in *'"repos": [{'*) ck https-ca ok "(Hugging Face listing came back)";; *) ck https-ca fail "(${n:0:100})";; esac
if [ -n "${CHAT_SHOT:-}" ]; then
  mkdir -p "$W/pw" && cp "$(dirname "$0")/smoke-chat.js" "$W/pw/smoke-chat.js"
  ( cd "$W/pw" && { [ -d node_modules/playwright ] || npm init -y >/dev/null 2>&1 && npm i --silent playwright@1.61.1 >/dev/null 2>&1; } \
      && npx playwright install webkit >/dev/null 2>&1 \
      && URL="$B/" TOKEN="$T" SHOT="$CHAT_SHOT" node smoke-chat.js ) && ck chat-renders-headless ok "(WebKit, shot $CHAT_SHOT)" || ck chat-renders-headless fail ""
fi
pkill -f "pxa-control --lan --no-browser --no-tray --port $PORT" 2>/dev/null
exit $fail
