#!/bin/bash
# Hermetic pxa-update test. Serves a tiny fake release. Does not touch GitHub.
set -euo pipefail
BIN=${BIN:?set BIN to the pxa-update binary}
ROOT=$(mktemp -d)
cleanup() {
  if [ -n "${PID:-}" ]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
  if [ -n "${PID2:-}" ]; then kill "$PID2" 2>/dev/null || true; wait "$PID2" 2>/dev/null || true; fi
  if [ -n "${SLEEP_PID:-}" ]; then kill "$SLEEP_PID" 2>/dev/null || true; wait "$SLEEP_PID" 2>/dev/null || true; fi
  rm -rf "$ROOT"
}
trap cleanup EXIT
PORT=$((18000 + RANDOM % 2000))
PORT2=$((PORT + 1))
STAGE="$ROOT/stage"
INS="$ROOT/install"
mkdir -p "$STAGE/pxa-v9.9.1" "$INS/pxa-v3" "$ROOT/www"
echo v9.9.1 > "$STAGE/pxa-v9.9.1/VERSION"
echo marker > "$STAGE/pxa-v9.9.1/pxa"
echo v3 > "$INS/pxa-v3/VERSION"
ln -s pxa-v3 "$INS/current"
# user data lives beside the version tree and must still be there after apply and rollback
mkdir -p "$INS/models"
echo keep-me > "$INS/models/weights"
echo "1,2,3" > "$INS/Qwen.expert-counts.csv"
tar -czf "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" -C "$STAGE" pxa-v9.9.1
HASH=$(sha256sum "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" | awk '{print $1}')
echo "$HASH  pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" > "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz.sha256"
cp "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" \
  "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz"
cp "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz.sha256" \
  "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz.sha256"
# a sidecar named so the old matcher would treat it as the tarball
echo "not-a-tarball" > "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz.sha256.fake"
TGZ_URL="http://127.0.0.1:$PORT/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz"
U22_URL="http://127.0.0.1:$PORT/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz"
SIDE_URL="http://127.0.0.1:$PORT/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz.sha256"
python3 - "$ROOT/www" "$PORT" "$TGZ_URL" "$U22_URL" "$SIDE_URL" << 'PY' &
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
www, port, url, u22, side = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/repos/poisonxa16/pxa/releases/latest"):
            body = json.dumps({"tag_name": "v9.9.1", "assets": [
                {"browser_download_url": side},
                {"browser_download_url": url},
                {"browser_download_url": u22},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        name = self.path.split("?")[0].lstrip("/")
        path = www + "/" + name
        try:
            data = open(path, "rb").read()
        except OSError:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a):
        pass
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
PY
PID=$!
python3 - "$PORT2" << 'PY' &
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[1])
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"message": "API rate limit exceeded"}).encode()
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
PY
PID2=$!
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -sf "http://127.0.0.1:$PORT/repos/poisonxa16/pxa/releases/latest" >/dev/null && break
  sleep 0.2
done
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT2/repos/poisonxa16/pxa/releases/latest" | grep -q 403 && break
  sleep 0.2
done
OUT=$("$BIN" check --dir "$INS" --base-url "http://127.0.0.1:$PORT")
echo "$OUT"
echo "$OUT" | grep -q "current=v3 latest=v9.9.1 update=yes"
printf 'tag:              v3\ncommit: abc\n' > "$INS/pxa-v3/VERSION"
OUT=$("$BIN" check --dir "$INS" --base-url "http://127.0.0.1:$PORT")
echo "$OUT" | grep -q "current=v3 latest=v9.9.1 update=yes"
printf 'tag:              v2026.10.3\ncommit: abc\n' > "$INS/pxa-v3/VERSION"
OUT=$("$BIN" check --dir "$INS" --base-url "http://127.0.0.1:$PORT")
echo "$OUT" | grep -q "current=v2026.10.3 latest=v9.9.1 update=yes"
echo v3 > "$INS/pxa-v3/VERSION"
"$BIN" apply --dir "$INS" --base-url "http://127.0.0.1:$PORT"
test "$(readlink "$INS/current")" = "pxa-v9.9.1"
test "$(readlink "$INS/previous")" = "pxa-v3"
grep -q v9.9.1 "$INS/current/VERSION"
test -f "$INS/pxa-v3/VERSION"
test "$(cat "$INS/models/weights")" = "keep-me"
test -f "$INS/Qwen.expert-counts.csv"
find "$INS" -name '*.tar.gz' | grep -q . && { echo "tarball left behind"; exit 1; }
"$BIN" rollback --dir "$INS"
test "$(readlink "$INS/current")" = "pxa-v3"
test -d "$INS/pxa-v9.9.1"
test "$(cat "$INS/models/weights")" = "keep-me"
"$BIN" rollback --dir "$INS"
test "$(readlink "$INS/current")" = "pxa-v9.9.1"
# checksum mismatch must not move current
echo "0000000000000000000000000000000000000000000000000000000000000000  pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" > "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz.sha256"
echo "0000000000000000000000000000000000000000000000000000000000000000  pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz" > "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz.sha256"
mkdir -p "$INS/pxa-v0"
echo v0 > "$INS/pxa-v0/VERSION"
ln -sfn pxa-v0 "$INS/current"
set +e
"$BIN" apply --dir "$INS" --base-url "http://127.0.0.1:$PORT" >"$ROOT/bad.out" 2>"$ROOT/bad.err"
RC=$?
set -e
test "$RC" != 0
test "$(readlink "$INS/current")" = "pxa-v0"
grep -q "checksum mismatch" "$ROOT/bad.err"
# a llama-server whose binary lives in this install blocks apply
echo "$HASH  pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" > "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz.sha256"
echo "$HASH  pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz" > "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70-ubuntu22.04.tar.gz.sha256"
cp "$(command -v sleep)" "$INS/pxa-v0/llama-server"
"$INS/pxa-v0/llama-server" 30 &
SLEEP_PID=$!
set +e
"$BIN" apply --dir "$INS" --base-url "http://127.0.0.1:$PORT" >"$ROOT/run.out" 2>"$ROOT/run.err"
RC=$?
set -e
test "$RC" = 2
test "$(readlink "$INS/current")" = "pxa-v0"
grep -q "server from this install is running" "$ROOT/run.err"
kill "$SLEEP_PID" 2>/dev/null || true
wait "$SLEEP_PID" 2>/dev/null || true
unset SLEEP_PID
PKG="$ROOT/running"
mkdir -p "$PKG/bin"
printf 'tag:              v9.9.1\ncommit: abc\n' > "$PKG/VERSION"
cp "$BIN" "$PKG/bin/pxa-update"
OUT=$("$PKG/bin/pxa-update" check --base-url "http://127.0.0.1:$PORT")
echo "$OUT" | grep -q "current=v9.9.1 latest=v9.9.1 update=no"
printf 'tag:              v2026.10.3\ncommit: old\n' > "$PKG/VERSION"
OUT=$("$PKG/bin/pxa-update" check --base-url "http://127.0.0.1:$PORT")
echo "$OUT" | grep -q "current=v2026.10.3 latest=v9.9.1 update=yes"
# GitHub 403 is a rate limit, not "release has no tag"
set +e
"$BIN" check --dir "$INS" --base-url "http://127.0.0.1:$PORT2" >"$ROOT/rate.out" 2>"$ROOT/rate.err"
RC=$?
set -e
test "$RC" != 0
grep -q "rate limit" "$ROOT/rate.err"
grep -q "release has no tag" "$ROOT/rate.err" && { echo "403 reported as a missing tag"; exit 1; }
# getconf failing must not pick a tarball
mkdir -p "$ROOT/fakebin"
printf '#!/bin/sh\nexit 1\n' > "$ROOT/fakebin/getconf"
chmod +x "$ROOT/fakebin/getconf"
ln -sfn pxa-v3 "$INS/current"
set +e
PATH="$ROOT/fakebin:$PATH" "$BIN" apply --dir "$INS" --base-url "http://127.0.0.1:$PORT" >"$ROOT/glibc.out" 2>"$ROOT/glibc.err"
RC=$?
set -e
test "$RC" != 0
test "$(readlink "$INS/current")" = "pxa-v3"
grep -q "C library" "$ROOT/glibc.err"
# an archive that carries models or expert counts is refused
BAD="$ROOT/badstage"
mkdir -p "$BAD/pxa-v9.9.1/models"
echo v9.9.1 > "$BAD/pxa-v9.9.1/VERSION"
echo secret > "$BAD/pxa-v9.9.1/models/weights"
echo 9 > "$BAD/pxa-v9.9.1/Model.expert-counts.csv"
tar -czf "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" -C "$BAD" pxa-v9.9.1
HASH2=$(sha256sum "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" | awk '{print $1}')
echo "$HASH2  pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz" > "$ROOT/www/pxa-v9.9.1-linux-x86_64-cuda12.8-sm60_61_70.tar.gz.sha256"
set +e
"$BIN" apply --dir "$INS" --base-url "http://127.0.0.1:$PORT" >"$ROOT/user.out" 2>"$ROOT/user.err"
RC=$?
set -e
test "$RC" != 0
test "$(readlink "$INS/current")" = "pxa-v3"
grep -q "expert counts" "$ROOT/user.err"
test "$(cat "$INS/models/weights")" = "keep-me"
echo PASS
