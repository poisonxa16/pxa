# Resident Q6_K on two V100s: keep the guards, enlarge the launch queues

For the configuration in [issue #15](https://github.com/poisonxa16/pxa/issues/15),
set the CUDA launch-queue size before starting the server:

```sh
CUDA_SCALE_LAUNCH_QUEUES=4x ./llama-server <the same server arguments>
```

The opt-in wrapper does this when the variable is unset and executes the command
without changing its arguments or PID:

```sh
sh tools/pxa-dual-volta.sh ./llama-server <the same server arguments>
```

An explicit `CUDA_SCALE_LAUNCH_QUEUES` value is preserved. To revert, unset the
variable and start the server directly. This profile does not select devices, change
batch/context/KV settings, or disable any FP16 guard, DQC, NY1 or MTP behavior.
The wrapper is optional; plain server startup keeps its existing defaults.

## Measurement

PXA 4.0.4, two V100-SXM2-32GB cards with NV2 NVLink, a Qwen3.6-35B-A3B-derived
Q6_K model, Q8 KV, resident weights, layer split 1:1, context capacity 262144,
batch/ubatch 4096/1024, MTP1. The same CPU-loaded Q8 projector was present.
Each setting used a fresh process. Two rounds ran default/2x/4x, then 4x/2x/default.
All numerical guards remained enabled. These are untraced measurements.

Median native prefill tokens/s:

| Input tokens | Default queues | 2x | 4x | 4x gain |
|---:|---:|---:|---:|---:|
| 4096 | 4649.8 | 4811.1 | 4827.8 | 3.83% |
| 16384 | 4833.2 | 5130.8 | 5140.9 | 6.37% |
| 65536 | 3471.6 | 3816.6 | 3822.3 | 10.10% |

At each input length all six 128-token continuations were byte-identical, and
MTP draft/acceptance counts matched. Arithmetic and schema-constrained JSON
checks passed. Full prompt/output counts and resident placement were verified.
The context capacity was not reduced. At 64K, decode medians were 83.60, 83.64 and
83.64 tokens/s: this addresses the prefill regression, not the separate NY1 decode
tradeoff. A 16K decode outlier occurred in the 2x arm; both 4x samples were stable.

The original issue's token recipe and end-token suppression were retained,
including the token-array SHA256
`fb1261d8cd22f33a85b8c434393e1105518bad698558d55f6c25480d95334f24`.
The server reported 262144 context and generated all 128 requested tokens.
This was not a full-262K-input, vision, concurrent-serving or long-soak test.
Prompt-only strict JSON formatting was already a baseline failure; this is a
performance profile with unchanged measured outputs, not a broad quality gate.

## Why this is a scheduling test

A separate instrumented guard-on/off pair had almost identical GPU kernel time
(28.539s versus 28.508s summed across devices), but first-content latency was
19.074s versus 17.373s. The guard added 19648 unscale launches; most of the wall
increase appeared as lost overlap between observed kernels on the two devices.
The guarded conversion itself was slightly faster than the unguarded converter.
Simply speeding up that conversion did not recover the regression.

These traces contain kernel and host API activity, not DMA/memset activity, and
exclude boundary-crossing events. They locate a scheduling problem; they are not
complete GPU-occupancy measurements or scored throughput results.

NVIDIA documents `CUDA_SCALE_LAUNCH_QUEUES` as the multiplier for pending-kernel/copy
command-buffer capacity, with supported values `0.25x`, `0.5x`, `2x`, and `4x`.
See the [CUDA 12.8 environment table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#cuda-environment-variables)
and [current execution-variable documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).
The untraced default/2x/4x intervention is the evidence that increasing the queue
capacity recovers this configuration's prefill loss. It does not establish a
universal queue default for other cards, models, splits or workloads.
