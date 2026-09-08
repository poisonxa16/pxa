#!/bin/bash
# bench/fair/run.sh — wires the existing bench harnesses to the bench/fair/protocol.md rules.
# Does NOT implement a new benchmark: it verifies weights, then calls ../speed-bench.sh (and,
# where applicable, ../measure.py) exactly as they exist today. Where those scripts cannot
# produce a protocol-compliant number, this script says "not automated yet" rather than
# reporting a number measured under a looser protocol as if it were one of the three.
#
# Usage:
#   cd bench/fair
#   PRODUCT_MODEL=/path/to/fusion2-35b-U16-q8head.gguf \
#   CODEC_A_MODEL=/path/to/PXA-Fusion2-35B-PXQ6.gguf \
#   CODEC_B_MODEL=/path/to/some-mxfp4-equivalent.gguf \
#   ./run.sh
#
# Every model env var above is optional; a number that has no model wired prints
# "not automated yet" instead of being skipped silently.
set -eu
cd "$(dirname "$0")"

echo "== bench/fair: verifying weights against weights/MANIFEST.sha256 =="

# Only the real "hash  filename" lines are checked; "sha: pending" lines have no hash yet
# (bench/fair/protocol.md rule 7 — never invent one) and are listed, not checked.
MANIFEST_LINES=$(grep -E '^[0-9a-f]{64}  ' weights/MANIFEST.sha256 || true)
if [ -z "$MANIFEST_LINES" ]; then
  echo "bench/fair: MANIFEST.sha256 has no checkable entries — refusing to start." >&2
  exit 1
fi

pushd weights >/dev/null
# Which of the manifest's files are actually here? Work this out FIRST: with none of them
# present, `sha256sum -c --ignore-missing` exits non-zero with "no file was verified", which
# reads as a checksum failure and is not one. An empty directory is a "fetch the weights"
# message, not an integrity alarm.
# `|| true`: under `set -e` a command substitution whose last command exits non-zero kills the
# script, and the loop below ends non-zero precisely when NO file matched — the case this is
# here to detect. Without it the script exits silently with status 1 and prints nothing at all.
PRESENT=$(echo "$MANIFEST_LINES" | awk '{print $2}' | while read -r f; do [ -f "$f" ] && echo "$f"; done || true)
if [ -z "$PRESENT" ]; then
  popd >/dev/null
  echo "bench/fair: none of the manifest's weight files are present in bench/fair/weights/." >&2
  echo "Nothing was checked and nothing was run. Download the artifacts named in" >&2
  echo "weights/MANIFEST.sha256 into that directory — each entry there carries a provenance" >&2
  echo "comment saying where it is published — or point PRODUCT_MODEL / CODEC_A_MODEL /" >&2
  echo "CODEC_B_MODEL at your own copies and add their sha256 to the manifest first." >&2
  exit 1
fi
if ! echo "$MANIFEST_LINES" | sha256sum -c - --ignore-missing 2>/tmp/pxq-fair-shacheck.$$ ; then
  echo "bench/fair: sha256 verification FAILED on a file that IS present. Refusing to start — see:" >&2
  cat /tmp/pxq-fair-shacheck.$$ >&2
  rm -f /tmp/pxq-fair-shacheck.$$
  popd >/dev/null
  exit 1
fi
rm -f /tmp/pxq-fair-shacheck.$$
popd >/dev/null

echo "bench/fair: verified: $PRESENT"
echo

# ---------------------------------------------------------------------------
# PRODUCT — best documented recipe, this engine. Wired to ../speed-bench.sh.
# ---------------------------------------------------------------------------
echo "== PRODUCT =="
if [ -n "${PRODUCT_MODEL:-}" ] && [ -f "$PRODUCT_MODEL" ]; then
  echo "Running ../speed-bench.sh MODEL=$PRODUCT_MODEL ..."
  PRODUCT_OUT=$(MODEL="$PRODUCT_MODEL" ../speed-bench.sh 2>&1) || {
    echo "PRODUCT: speed-bench.sh failed:"; echo "$PRODUCT_OUT"; PRODUCT_OUT=""; }
  echo "$PRODUCT_OUT"
  echo "PRODUCT: see gen_tps column above (median of 3 runs)."
  echo "CAVEAT: ../speed-bench.sh runs its own protocol (temp=1.0, /v1/chat/completions,"
  echo "        median-of-3, no discarded warmup) — not this directory's protocol.md"
  echo "        (temp=0, /completion, n=7 with 1 discarded warmup). Reported as-is, not"
  echo "        silently reconciled to the stricter protocol."
else
  echo "PRODUCT: not automated yet — set PRODUCT_MODEL to a gguf present in weights/."
fi
echo

# ---------------------------------------------------------------------------
# CODEC-ONLY — same engine, PXQ4 vs MXFP4 at matched bytes. Two speed-bench.sh runs, diffed.
# ---------------------------------------------------------------------------
echo "== CODEC-ONLY =="
if [ -n "${CODEC_A_MODEL:-}" ] && [ -f "${CODEC_A_MODEL:-/nonexistent}" ] \
   && [ -n "${CODEC_B_MODEL:-}" ] && [ -f "${CODEC_B_MODEL:-/nonexistent}" ]; then
  echo "Running ../speed-bench.sh against CODEC_A_MODEL=$CODEC_A_MODEL ..."
  A_OUT=$(MODEL="$CODEC_A_MODEL" ../speed-bench.sh 2>&1) || A_OUT="FAILED"
  echo "$A_OUT"
  echo "Running ../speed-bench.sh against CODEC_B_MODEL=$CODEC_B_MODEL ..."
  B_OUT=$(MODEL="$CODEC_B_MODEL" ../speed-bench.sh 2>&1) || B_OUT="FAILED"
  echo "$B_OUT"
  echo "CODEC-ONLY: compare the two gen_tps medians above by hand (this script does not"
  echo "            parse them — see bench/HEAD-TO-HEAD.md for the published PXQ4-vs-MXFP4"
  echo "            methodology this should match)."
  echo "CAVEAT: same protocol delta as PRODUCT above."
else
  echo "CODEC-ONLY: not automated yet — set CODEC_A_MODEL and CODEC_B_MODEL to two"
  echo "            matched-byte ggufs (e.g. a PXQ4 file and an MXFP4 file of the same"
  echo "            model) present in weights/."
fi
echo

# ---------------------------------------------------------------------------
# ENGINE-ONLY — same gguf, two engines (upstream ik_llama.cpp vs this engine). Needs a second
# engine binary; neither ../speed-bench.sh nor ../measure.py builds or drives one, and this
# script does not build a new benchmark to fill the gap (bench/fair/protocol.md, "the three
# numbers": this is the one most often missing).
# ---------------------------------------------------------------------------
echo "== ENGINE-ONLY =="
if [ -n "${UPSTREAM_SERVER_BIN:-}" ] && [ -x "${UPSTREAM_SERVER_BIN:-/nonexistent}" ] \
   && [ -n "${ENGINE_ONLY_MODEL:-}" ] && [ -f "${ENGINE_ONLY_MODEL:-/nonexistent}" ]; then
  echo "not automated yet — a second engine binary was provided (UPSTREAM_SERVER_BIN) but"
  echo "this script has no wiring to drive an upstream ik_llama.cpp server and diff it"
  echo "against ../speed-bench.sh's pxa run; see bench/fair-battle.md for the"
  echo "hand-run methodology this would need to automate."
else
  echo "not automated yet — needs a built upstream ik_llama.cpp server binary"
  echo "(UPSTREAM_SERVER_BIN) plus the same gguf (ENGINE_ONLY_MODEL) on both engines;"
  echo "see bench/fair-battle.md's 'Same-quant control' section for the numbers already"
  echo "measured this way by hand."
fi
