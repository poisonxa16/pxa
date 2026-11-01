#!/bin/sh
# Run a test that spawns or signals ONLY inside a fresh user + PID namespace with its own /proc (F295 incident).
# Box only, never pxanet. Refuses when it is not given a fresh namespace.
set -eu
command -v unshare >/dev/null || { echo "run-in-pidns: no unshare; refusing to run" >&2; exit 2; }
[ "$(hostname)" = pxanet ] && { echo "run-in-pidns: never on pxanet" >&2; exit 2; }
exec unshare -Urp --fork --mount-proc env PXA_FRESH_PIDNS=1 "$@"
