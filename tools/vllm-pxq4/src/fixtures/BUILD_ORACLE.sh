#!/bin/sh
# Rebuild the gate-G1 oracle. It carries the production ggml C VERBATIM (see the header of
# gguf_to_vllm_oracle.c for exactly which line ranges), links against nothing, and prints
# float32 bits, so reference.py can be pinned against it by exact equality.
set -e
cc -O2 -std=c11 -o "$(dirname "$0")/oracle" "$(dirname "$0")/../gguf_to_vllm_oracle.c"
"$(dirname "$0")/oracle" --selfcheck
