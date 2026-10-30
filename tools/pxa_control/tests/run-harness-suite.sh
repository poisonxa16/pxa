#!/bin/bash
# Run the PXA Control chat harness suite unattended (see README-HARNESS.md next to this file).
#   tools/pxa_control/tests/run-harness-suite.sh                                   # FAKE mode, every automatable case
#   tools/pxa_control/tests/run-harness-suite.sh --endpoint http://127.0.0.1:8263  # + the live cases on a real server
#   tools/pxa_control/tests/run-harness-suite.sh --only S06,T16 --out /tmp/h       # a few cases
# Any harness_suite.py flag passes through. CPU only, no GPU: it pins itself to PXA_HARNESS_CPUS (default 36-53, the
# PXA box's build cores) at nice PXA_HARNESS_NICE (default 15). Exit code = the suite's: 0 no FAIL, 1 FAIL, 2 setup error.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CPUS=${PXA_HARNESS_CPUS-36-53}
NICE=${PXA_HARNESS_NICE:-15}
PY=${PYTHON:-python3}
cd "$REPO" || exit 2
pre=()
if [ -n "$CPUS" ] && command -v taskset >/dev/null 2>&1; then pre+=(taskset -c "$CPUS"); fi
pre+=(nice -n "$NICE")
exec env -u DISPLAY -u WAYLAND_DISPLAY -u PXA_BUS_TOKEN -u PXA_CONTROL_TOKEN TERM=dumb "${pre[@]}" "$PY" "$HERE/harness_suite.py" "$@"
