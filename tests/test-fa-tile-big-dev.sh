#!/usr/bin/env bash
# PXA_FA_TILE_512 on the device: run the case list four times, because every lever and every
# reference choice resolves once per process and no two of them can be read from one run.
#
#   run A (unset)  -- the DEFAULT. Every case must be DECLINED by the CUDA backend. This is the
#                     positive control for the lever: it proves the default really does leave a
#                     512-wide flash-attention node to the scheduler (which places it on the CPU
#                     backend), and it proves that what run B switches is the support predicate.
#   run B (=1)     -- every case must be ENGAGED and must clear the error bar against the same
#                     graph computed on the CPU backend with an F32 V (an fp32 accumulator).
#   run C          -- the all-fp32 score tile, a reported diagnostic, never a gate.
#   run D          -- the same cases read against an F16 V reference, i.e. against ggml's fp16
#                     VKQ16 accumulator. A reported POSITIVE CONTROL that is EXPECTED to fail at
#                     depth; see the reference note at the top of test-fa-tile-big-dev.cpp.
#
# Run B is only meaningful on a cc 6.0 or cc 7.0 card: everywhere else the kernel declines by
# design (Turing+ has the MMA kernel, sm_61 has no fast fp16) and the run says DECLINED, which is
# correct behaviour and NOT a pass. That is why this driver checks for the ARMED line and for a
# non-zero engaged count rather than for a zero exit code alone.
#
# usage: test-fa-tile-big-dev.sh /path/to/test-fa-tile-big-dev [workdir]

set -u

BIN=${1:-}
WORK=${2:-${TMPDIR:-/tmp}}

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "usage: $0 /path/to/test-fa-tile-big-dev [workdir]" >&2
    exit 2
fi

OFF_LOG="$WORK/fa-tile-big-off.log"
ON_LOG="$WORK/fa-tile-big-on.log"
F32_LOG="$WORK/fa-tile-big-f32.log"
REF16_LOG="$WORK/fa-tile-big-ref16.log"

echo "== run A: PXA_FA_TILE_512 unset (the default) =="
env -u PXA_FA_TILE_512 "$BIN" > "$OFF_LOG" 2>&1
rc_off=$?
cat "$OFF_LOG"
if [ $rc_off -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

echo "== run B: PXA_FA_TILE_512=1 (armed) =="
PXA_FA_TILE_512=1 "$BIN" > "$ON_LOG" 2>&1
rc_on=$?
cat "$ON_LOG"
if [ $rc_on -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

# run C (=1 with PXA_FA_TILE_512_FP32=1) -- the SAME cases with the score tile, the probabilities
# and the QK chunk sum all in fp32. It is not a pass/fail gate: it is the two-way answer to
# "is the depth error precision or structure?". If run C clears the bar where run B does not, the
# fp16 carriers are the defect and the fix is a precision one. If run C fails the same cases by
# the same margin, no accumulator change will help and the defect is structural.
echo "== run C: PXA_FA_TILE_512=1 PXA_FA_TILE_512_FP32=1 (armed, all-fp32 score tile) =="
PXA_FA_TILE_512=1 PXA_FA_TILE_512_FP32=1 "$BIN" > "$F32_LOG" 2>&1
rc_f32=$?
cat "$F32_LOG"
if [ $rc_f32 -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

# run D (=1 with PXA_FA_TILE_512_REF_F16=1) -- the SAME cases, but the reference run gets an F16 V
# instead of an F32 one, which puts ggml's CPU flash-attention back on its fp16 VKQ16 accumulator.
# It is REPORTED, never gated, and it is a demonstration rather than a pass/fail: with signed
# operands the fp16 accumulator is a random walk, so run D's rms(d)/rms(v) should sit about TWO
# ORDERS OF MAGNITUDE above run B's and stay FLAT across the KV sweep. It may well still clear the
# bar. What it must not do is agree with run B -- if the two columns match, the reference V type is
# not reaching ggml's accumulator choice and the note at the top of test-fa-tile-big-dev.cpp is no
# longer describing this build.
echo "== run D: PXA_FA_TILE_512=1 PXA_FA_TILE_512_REF_F16=1 (armed, fp16-accumulator reference) =="
PXA_FA_TILE_512=1 PXA_FA_TILE_512_REF_F16=1 "$BIN" > "$REF16_LOG" 2>&1
rc_ref16=$?
cat "$REF16_LOG"
if [ $rc_ref16 -eq 2 ]; then
    echo "no CUDA device -- cannot run"; exit 2
fi

fail=0

# Run A: the default must decline every case. `engaged 0, declined N, of N cases`.
if ! grep -qE '^engaged 0, declined [0-9]+, of [0-9]+ cases$' "$OFF_LOG"; then
    echo "FAIL: with the lever unset the CUDA backend accepted a 512-wide node -- the default is"
    echo "      not OFF, or something other than PXA_FA_TILE_512 is gating the support predicate."
    fail=1
fi
[ $rc_off -ne 0 ] && { echo "FAIL: the default run returned $rc_off"; fail=1; }

# Run B: the kernel must have been armed, and it must have run something.
if ! grep -q "PXA_FA_TILE_512: ARMED" "$ON_LOG"; then
    echo "FAIL: the armed run never printed the kernel's ARMED line."
    fail=1
fi
if grep -qE '^engaged 0, ' "$ON_LOG"; then
    echo "SKIP-LEVEL RESULT: the armed run engaged ZERO cases. That is the correct answer on any"
    echo "      card but sm_60 / sm_70, and it means this run proves nothing about the kernel."
    echo "      Re-run it on a P100 or a V100 before reading the exit code as a pass."
    fail=1
fi
[ $rc_on -ne 0 ] && { echo "FAIL: the armed run returned $rc_on"; fail=1; }

# Run C is reported, never gated: a diagnostic that could fail the script would tempt a future
# reader to "fix" the diagnostic instead of the kernel.
if grep -q "PXA_FA_TILE_512_FP32:" "$F32_LOG"; then
    echo "== run C engaged the fp32 score tile. Side by side, armed fp16 vs armed fp32:"
    paste <(grep -E '^  ' "$ON_LOG") <(grep -E '^  ' "$F32_LOG" | sed 's/^.\{32\}//') | sed 's/^/  /'
else
    echo "NOTE: run C never printed the fp32 line -- the diagnostic did not engage and says nothing."
fi

# Run D reported, never gated, for the same reason as run C.
if grep -q "reference: .* F16 V" "$REF16_LOG"; then
    echo "== run D used the fp16-accumulator reference. Side by side, fp32 reference vs fp16 reference:"
    paste <(grep -E '^  ' "$ON_LOG") <(grep -E '^  ' "$REF16_LOG" | sed 's/^.\{32\}//') | sed 's/^/  /'
    echo "      Read the rms(d)/rms(v) columns against each other, not the OK/FAIL verdicts: run D"
    echo "      is expected to sit about 100x above run B and to stay flat with depth. Two columns"
    echo "      that MATCH mean the reference V type never reached ggml's accumulator choice."
else
    echo "NOTE: run D never printed its reference line -- the control did not engage and says nothing."
fi

[ $fail -eq 0 ] && echo "OK" || echo "FAIL"
exit $fail
