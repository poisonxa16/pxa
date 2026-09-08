#!/bin/bash
# Qwen3.8-Flash-Next (qwen4exp) across the six Teslas, n-gram table on CPU.
#
# CARD MAP, in PCI_BUS_ID order (CUDA_VISIBLE_DEVICES orders FASTEST-FIRST otherwise, which
# would silently renumber everything):
#   0 P100  <- also carries a production granite seat, ~8.4 GiB already resident
#   1 P100
#   2 V100
#   3 GTX 1080 Ti  <- PRODUCTION VLM + embeddings. NEVER included.
#   4 V100
#   5 P100
#   6 P100
#
# Why per_layer_token_embd goes to the CPU: it is 160 x 320,001,536 in IQ4_NL, about 28.8 GiB,
# and it is a pure GET_ROWS table - a gather per token per head, no GEMM. Keeping it in host
# RAM costs a PCIe gather and frees nearly a third of the file from VRAM. The remaining ~38 GiB
# of weights sit comfortably in the ~87 GiB the six cards have free.
#
# The -ot pattern is ANCHORED. A loose 'ple' regex would also match blk.1's ple_key,
# ple_conv1d and three F32 ple_norm_* tensors, which must stay on the GPU.
set -u
# Set PXQ_HOST_GUARD to pin this script to one host.
[ -n "${PXQ_HOST_GUARD:-}" ] && [ "$(hostname)" != "$PXQ_HOST_GUARD" ] && \
  { echo "WRONG HOST: $(hostname) (expected $PXQ_HOST_GUARD)"; exit 1; }

ENGINE=${ENGINE:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)/build-cuda/bin/llama-cli}
MODEL=${MODEL:?set MODEL to the .gguf to run}
NCTX=${NCTX:-4096}
NPRED=${NPRED:-64}
PROMPT=${PROMPT:-"Explain in two sentences what makes a mixture-of-experts model efficient."}

[ -x "$ENGINE" ] || { echo "engine not built yet: $ENGINE"; exit 1; }
[ -f "$MODEL" ]  || { echo "model shard 1 missing: $MODEL"; exit 1; }

# refuse to run if the protected card would be visible
for d in ${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6}; do
  [ "$d" = "3" ] && { echo "REFUSING: card 3 is the production 1080 Ti"; exit 1; }
done

# the engine binaries carry no RPATH
ENGINE_BUILD=$(cd "$(dirname "$ENGINE")/.." && pwd)
export LD_LIBRARY_PATH="$ENGINE_BUILD/src:$ENGINE_BUILD/ggml/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6}

# Card 0 shares with a production seat, so it takes a smaller slice.
#
# -ts partitions BYTES, not layer counts (llama.cpp:4071-4100 walks layer_sizes[] against
# splits[id]*sum, with a max_compute allowance added per device). These six numbers are
# therefore a byte budget per card, and they land the 48 layers + output head as
#   4 / 7 / 11 / 11 / 7 / 8+head   over CUDA0..CUDA5 = P100,P100,V100,V100,P100,P100.
#
# That is four layers moved off the P100s onto the two V100s, which is the whole prize:
# post-load headroom is 2 layers per V100 (a layer is 1341 MiB; 11 layers leaves 1095 MiB
# free on a 16 GiB V100), so 12 would not load. The V100s run a layer 0.28 ms/token cheaper
# than the P100s, and because the six cards serialize with literally zero overlap (measured:
# summed kernel busy == union of busy intervals), every microsecond saved on any card comes
# straight off the critical path.
#
# Measured, 127 tokens per arm, three interleaved repeats each:
#   4/9/9/9/9/8+head  34.38, 34.20, 34.47 ms/token  (29.09, 29.24, 29.01 tok/s)
#   4/7/11/11/7/8     33.25, 32.94, 33.04 ms/token  (30.08, 30.35, 30.27 tok/s)
# -1.27 ms/token, -3.70%, no overlap between the two sets. nsys per-kernel capture predicted
# -1.114 ms/token from the same placement, so the wall clock and the kernel time agree.
TS=${TS:-20,34,52,52,34,40}

exec "$ENGINE" \
  -m "$MODEL" \
  -ngl 99 \
  -ts "$TS" \
  -ot 'per_layer_token_embd\.weight=CPU' \
  -c "$NCTX" \
  -n "$NPRED" \
  -t 16 \
  --no-warmup \
  -p "$PROMPT" \
  "$@"
