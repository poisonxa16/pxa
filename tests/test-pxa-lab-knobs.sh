#!/bin/bash
# bug #324: "pxa: lab knobs set: ..." names only what the user set.
#   a) no PXA_*/PXQ* env, engine sets its own defaults at startup   -> no warning
#   b) one real lab knob in the env (+ the engine's own defaults)   -> the warning names exactly that one
#   c) the reporter's flags (-sm tensor -ts 1,1 --spec-type mtp), no env -> no warning
#   d) the warning text and the user-facing/tool exemptions are as before; a name the launcher listed in
#      PXA_ENV_BY_PXA is not the user's; a user switch the launcher did not inject still is
# usage: test-pxa-lab-knobs.sh <path to test-pxa-lab-knobs> ; exit 0 = PASS
set -u
BIN=${1:?usage: test-pxa-lab-knobs.sh <test-pxa-lab-knobs binary>}
fail=0
run() { # <expected> <label> <env assignments...> -- <args...>
    local want=$1 label=$2; shift 2
    local envs=() args=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    args=("$@")
    local got
    got=$(env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" "${envs[@]}" "$BIN" "${args[@]}" 2>/dev/null)
    if [ "$got" = "$want" ]; then echo "PASS $label"; else echo "FAIL $label"; echo "  want: $want"; echo "  got:  $got"; fail=1; fi
}
W='pxa: lab knobs set: %s (measured on one config; usually slower elsewhere — see docs/LEVERS.md)'
w() { printf "$W" "$1"; }

run "" "a) no env, engine defaults set at startup" -- engine-sets
run "$(w PXA_SOME_LAB_KNOB)" "b) one real lab knob" PXA_SOME_LAB_KNOB=1 -- engine-sets
run "$(w 'PXA_ONE PXQ_TWO')" "b2) two real lab knobs, own defaults not named" PXA_ONE=1 PXQ_TWO=x -- engine-sets
run "" "c) reporter flags, no env" -- engine-sets -sm tensor -ts 1,1 --spec-type mtp
run "$(w PXA_TSPLIT_REDUCE)" "c2) a user who typed the engine's default name is still named" PXA_TSPLIT_REDUCE=fused -- engine-sets
run "" "d1) user-facing + tool variables are not lab knobs" PXA_ENHANCE=1 PXA_MODE=x PXA_STATS=1 PXA_ENGINE_DIR=/x PXA_HOME=/y PXA_NO_OVERDRIVE=1 PXA_CONTROL=0 -- engine-sets
run "" "d2) launcher-injected names (PXA_ENV_BY_PXA) are not the user's" PXA_A=1 PXA_B=1 PXA_ENV_BY_PXA=PXA_A,PXA_B -- engine-sets
run "$(w PXA_C)" "d3) a user switch the launcher did not inject still is" PXA_A=1 PXA_C=1 PXA_ENV_BY_PXA=PXA_A -- engine-sets
run "" "d4) nothing at all: nothing printed" --
exit $fail
