#!/usr/bin/env bash
# pxa / pxa-vllm installer.
# Reads the card, picks the supported path, and refuses to guess.
set -euo pipefail

say() { printf "%s\n" "$*"; }
die() { printf "error: %s\n" "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found - install the NVIDIA driver first."

mapfile -t CC < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "[:blank:]" | sort -u)
[ "${#CC[@]}" -gt 0 ] || die "no CUDA devices visible."

say "Detected compute capability: ${CC[*]}"
if [ "${#CC[@]}" -gt 1 ]; then
  say
  say "NOTE: mixed compute capabilities on this host. A vLLM wheel is built"
  say "against ONE capability and ONE torch. Mixed rigs should use the"
  say "llama.cpp engine, which handles a heterogeneous set in one process."
fi

PRIMARY="${CC[0]}"
case "$PRIMARY" in
  6.0)
    say
    say "Pascal (sm_60). Supported path: pxa-vllm sm60 variant."
    say "  torch 2.7.1+cu126 - the last torch shipping sm_60 cubins."
    say "  Install WITH the shipped constraints file, never on its own:"
    say "    pip install -c dist/constraints-sm60.txt torch==2.7.1+cu126 \\"
    say "        --index-url https://download.pytorch.org/whl/cu126"
    say "    pip install -c dist/constraints-sm60.txt dist/pxa_vllm-*-sm60-*.whl"
    ;;
  7.0)
    say
    say "Volta (sm_70). READ THIS BEFORE INSTALLING THE SERVING STACK."
    say
    say "  The vLLM path on sm_70 is NOT GATED. It has never passed a smoke"
    say "  test on a V100, so the launcher will not route traffic to it."
    say "  When forced to serve it managed ~4 tok/s, against ~117 tok/s from"
    say "  the llama.cpp engine on the same cards."
    say
    say "  Recommended on Volta: build the engine, not the serving stack."
    say "    see BUILD-FROM-SOURCE.md"
    ;;
  *)
    say
    say "Compute capability $PRIMARY is outside the two variants we build."
    say "The llama.cpp engine builds and runs on modern cards normally:"
    say "  see BUILD-FROM-SOURCE.md"
    ;;
esac

say
say "Engine (all cards): BUILD-FROM-SOURCE.md, or a prebuilt llama-* binary."
say "Adaptive launcher:  tools/pxa-launch.py - picks engine, image and params per card."
