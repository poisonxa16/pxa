# PXQ4 on V100 — what this is and how to run it

PXQ4 is a 4.254 bpw weight format (ggml type id 252) with a CUDA kernel written for
sm_70. This directory has the vLLM integration, the kernel sources, and a launch
script with the measured-best configuration.

## Run it

    bash pxq4-serve.sh          # cards 4-7, TP4, port 8421

First boot is 13-18 minutes (torch.compile plus CUDA graph capture). That is normal
and is paid once per configuration.

## Where things are

    /path/to/models/pxa-int-v6/site/            the vLLM plugin (PYTHONPATH target)
    /path/to/models/pxa-int-v6/site/pxq4_vllm/_lib/libpxq4_sm70.so   the built kernel
    /path/to/models/pxa-int-v6/src/             kernel sources + host simulator + gates
    /path/to/models/pxa-models/Qwen3.8-27B-PXQ4-vllm-p2a-nf/         the model

## How it measures against AWQ W4A16

Same cards, same client, same params (`--gpu-memory-utilization 0.93
--max-model-len 200000 --enable-prefix-caching`, TP4, 512 tokens, medians):

| metric | AWQ | PXQ4 |
|---|---|---|
| prefill (2790-tok prompts, n=8) | 3159.8 tok/s | **3404.9** |
| single stream (n=12) | 62.18 | 60.9-63.2 (tie, see noise note) |
| 2 streams, aggregate | 110.76 | 105.82 |
| 4 streams | 36.47 | **37.30** |
| 8 streams | 73.14 | **74.09** |
| 16 streams | 144.13 | 131.1-133.9 |

Ahead on prefill and at 4 and 8 streams; behind at 2 and 16; single stream is a tie
inside clock noise.

## Three things that look like improvements and are not

1. **`cudagraph_capture_sizes=[1,2,4,8,16]`** pins decode at a batch-independent
   ~121 ms/step (8.25 tok/s) at 100% GPU utilisation. Reproduced on a clean boot.
   Cause not yet found. The default (1,2) is what the numbers above were measured on.
2. **`PYTORCH_ALLOC_CONF=expandable_segments:True`** hard-crashes any TP>1 run in
   `custom_all_reduce.cuh:976`.
3. **`VLLM_SM70_QUANT_BACKEND`** does nothing for PXQ4. It routes only AWQ, GPTQ and
   compressed-tensors layers (`envs.py:1488`).

## Known open issues

- **conc16 is 7-9% behind AWQ.** Not the kernel: above batch 2 decode runs eager
  because only sizes (1,2) are captured, and the linears are under 10 ms of a ~119 ms
  step. Fixing this means fixing item 1 above.
- **First-token latency spikes.** 3 of 12 short prompts deterministically pay ~0.9 s
  before the first token, on PXQ4 arms only, across every kernel version. The
  tokenizer is ruled out (0.2-0.4 ms offline) and it survives the prefill fix.

## Measurement caveat

SM clocks droop 1530 -> 1380-1425 MHz under sustained decode and recover in gappy
phases. This is autoboost, not throttling: no throttle reasons set, 210-240 W,
temperatures at or below 67 C. It is the source of the single-stream bimodality and
it affects AWQ equally. Lock clocks if you want tighter numbers.

## Correctness

The kernel is gated bit-exact against a host simulator: 17/17 legacy plus split plus
multi-token parity, including the fp32 partial accumulators; GPU parity across all
TP4 shapes for M in 1..8; a 400-launch stress run with zero failures.

Kill switch: `PXQ4_MMV_MT=0` reverts to the single-token mmv path.
