# PXQU recipe: Flash-Next on four P100s, ub2048, 150k+ KV

## What the constraint actually is

Not the KV cache. The model has 48 layers but `full_attention_interval=4`, so only
**12 layers hold a KV cache at all**; the other 36 are linear-attention (GDN) and carry a
fixed-size recurrent state instead. Each full-attention layer stores K(512) + V(512) +
indexer-K(128) = 1152 elements per token:

| context | KV f16 | KV q8_0 |
|---|---|---|
| 150k tokens | 3.86 GiB | 2.05 GiB |
| 200k tokens | 5.15 GiB | 2.74 GiB |

150k of context costs under 4 GiB. The constraint is entirely **weights**:

| bucket | params | share of GPU weights |
|---|---|---|
| routed experts (144 tensors) | 120.80B | **96.1%** |
| backbone (attn, shared expert, hyper-connections, PLE, norms) | 3.68B | 2.9% |
| token_embd + output head | 1.27B | 1.0% |
| `per_layer_token_embd` -> host RAM | 51.20B | (not on GPU) |

A straight PXQ4 of this model is **114.7 GiB** (`/path/to/models/qwen4exp/Qwen3.8-Flash-Next-PXQ4.gguf`).
It will never see a P100. The experts have to come down, and only the experts.

## Budget

Four P100s are cards 0, 1, 5, 6. **Card 0 carries another production model (~8.4 GiB)**,
which is not evictable, so the honest budget is 64.0 - 8.4 = **55.6 GiB**.

    55.6  physical
   - 1.20 CUDA context + cuBLAS handle, 4 cards
   - 3.46 compute buffer at n_ubatch=2048, 4 cards   (measured-derived, below)
   - 3.86 KV at 150k f16
   - 2.73 backbone + embed + head at PXQ4
   - 2.50 headroom held back deliberately
   = 41.85 GiB expert budget.

The compute-buffer figure is derived from a real six-card load, not guessed:
`llama_init_from_model` reported **127 MiB/card, and 490 MiB on the card holding the
output head**, at c=4096 / ub=512. The 490 MiB is exactly 512 x 248320 x 4 B, i.e. it
scales linearly with n_ubatch, so at ub=2048 it becomes ~1.98 GiB while the other cards
go to ~504 MiB: 3 x 504 MiB + 1.98 GiB = 3.46 GiB over four cards.

## The map

`pxa-bench/pxq-universal/recipes/pxqu56-177b-flashnext.tiers`

    experts      41.78 GiB   avg 2.971 bpw   42 tensors PXQ2 / 102 tensors PXQ3
    GPU weights  44.51 GiB
    host RAM     26.82 GiB   (per_layer_token_embd, IQ4_NL, gathered over PCIe)
    file         71.33 GiB

| target | need | have | headroom |
|---|---|---|---|
| card 0 shared, 150k KV f16 | 53.03 | 55.6 | **2.57 GiB** |
| card 0 shared, 200k KV f16 | 54.32 | 55.6 | 1.28 GiB |
| card 0 shared, 150k KV q8_0 | 51.22 | 55.6 | 4.38 GiB |
| card 0 shared, 200k KV q8_0 | 51.90 | 55.6 | 3.70 GiB |
| all four free, 150k KV f16 | 53.03 | 64.0 | 10.97 GiB |

A second map, `pxqu64-177b-flashnext.tiers` (50.37 GiB experts, 3.582 bpw, PXQ3/PXQ4),
exists for the case where card 0 is ever freed. Do not deploy it while that other model is resident.

## Why this beats the reference artifact

The public UD-IQ1_S we currently run puts `ffn_down_exps` at IQ4_NL and both
`ffn_gate_exps` and `ffn_up_exps` at **IQ1_S — 1.5625 bpw** — averaging 2.54 bpw over the
experts, 40.4 GiB of GPU weights, 68 GiB on disk.

This map is 71.33 GiB on disk, ~3 GiB more, with **nothing below PXQ2**.
Measured wrel (rng-42 lab protocol, from `ggml/include/ggml.h`): PXQ2 0.3020, PXQ3 0.1435.
Trading a 1.56-bpw tier for a 2.25/3.25-bpw tier at equal file size is the whole point.

## How it was solved

Lagrangian knapsack over the 144 expert tensors: start every tensor at PXQ2, repeatedly buy
the upgrade with the best error-reduction-per-byte until the budget is spent. This reproduces
the shape of the existing house recipes in `recipes/` (depth-graded, later layers richer).
Generator: `pxa-bench/pxq-universal/gen_pxqu_flashnext.py`.

## Two caveats, stated plainly

1. **The per-tensor sensitivity is a proxy, not a measurement.** The existing house recipes
   were solved against a measured `sens.json`. No sensitivity sweep exists for this
   architecture yet, so the weighting here is `depth x kind` (down-projection favoured 1.3x
   because it writes back into the residual; deeper layers favoured linearly). A real imatrix
   sweep will move assignments and should be run before this is called final.
2. **The compute buffer is measured at ub=512 and scaled, not measured at ub=2048.**
   The 127 MiB/card and 490 MiB output-card figures are real, read off a six-card load.
   The step to ub=2048 assumes linear scaling in n_ubatch, which is right for the output
   head (it is literally `n_ubatch x n_vocab x 4`) and approximately right for the rest.
   Confirm with one ub2048 load before building. The 2.57 GiB headroom at the 150k/f16
   target absorbs roughly a 0.6 GiB/card error; beyond that, q8_0 KV buys another 1.8 GiB.

## Build

    llama-quantize \
      --pxq-universal pxa-bench/pxq-universal/recipes/pxqu56-177b-flashnext.tiers \
      /path/to/models/qwen4exp/Qwen3.8-Flash-Next-BF16-pleq8.gguf \
      /path/to/models/qwen4exp/Qwen3.8-Flash-Next-PXQU56.gguf \
      PXQ_UNIVERSAL

Serve with `per_layer_token_embd` pinned to the host, as the six-card launcher already does:

    -ot 'per_layer_token_embd\.weight=CPU'  -c 153600  -ctk q8_0 -ctv q8_0

The `-ot` pattern is anchored on purpose: a loose `ple` regex also matches `ple_key`,
`ple_conv1d` and three F32 `ple_norm_*` tensors, which must stay on the GPU.
