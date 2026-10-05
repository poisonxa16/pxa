# Gemma-4 26B-A4B MTP verify levers: summary (ticket gemma-mtp-verify-kernel)

Grok Bot, 2026-10-03. Branches grokbot/gemma-verify-2row -> -logits -> -moe -> -dense (this one); see also
docs/GEMMA-MTP-VERIFY-GRAPH.md (graphs: measured GPU-bound, no lever). Every lever is default OFF.

Setup for all numbers: one V100 (GPU 4), gemma-4-26B-A4B-it QAT Q4_0 + its MTP assistant drafter, c16384, ub256,
greedy 256-token prose/code prompts, levers alternated, 3 rounds x 5 reps (15 runs per arm) unless noted.
w1/w2/w3 = target step ms (1/2/3-token forward) from PXA_HOST_TIMING; t/s = median decode.

## Recommended set (all output-identical; greedy shas + acceptance, and n_probs / temperature / grammar
## fallback requests, identical lever off vs on)

    PXA_MOE_VERIFY2=1 PXA_VERIFY_NO_LOGITS_COPY=1 PXA_MMVQ_V100_ROWS_NY=4 PXA_MOE_VERIFY2_TILE=4x4 PXA_DENSE_FUG_NY=1

| step added (cumulative)            | w2 ms | w3 ms | n1 prose/code t/s | n2 prose/code t/s | commit |
|---|---|---|---|---|---|
| none (baseline, 6/3 runs)          | 12.51 | 15.53 | 141.8 / 140.1 | 152.7 / 128.6 | - |
| PXA_MOE_VERIFY2                    | 12.22 | 14.83 | 144.9 / 143.2 | 159.4 / 132.3 | 6228ea7507 |
| + PXA_VERIFY_NO_LOGITS_COPY        | 11.50 | 13.75 | 154.2 / 152.3 | 172.9 / 143.8 | 27d8a7206f |
| + PXA_MMVQ_V100_ROWS_NY=4          | 11.19 | 13.42 | 158.2 / 155.7 | 175.2 / 145.3 | c662e82192 |
| + PXA_MOE_VERIFY2_TILE=4x4         |  9.97 | 11.54 | 175.1 / 172.5 | 199.8 / 164.3 | ecc42bf850 |
| + PXA_DENSE_FUG_NY                 |  9.74 | 11.40 | 179.0 / 176.1 | 202.7 / 166.7 | 5b34ed99c7 |

Net: n1 +26% prose / +26% code, n2 +33% / +30%, w2 -2.8 ms. Rows measured in different sessions; each step is its
own alternating A/B (baseline row: 6 runs n1, 3 runs n2).

What each does:
- PXA_MOE_VERIFY2: one expert-grouped id-GEMV launch per MoE projection for a 2..4-token verify batch instead of
  per-token loops / MMQ-id. Bit-identical (tests/test-moe-verify2 HASH-EXACT).
- PXA_VERIFY_NO_LOGITS_COPY: greedy verify and assistant-drafter draws read back only the GPU argmax, not the
  2 MB / 1 MB full logits; anything that needs logits (n_probs, grammar, temperature, biases) keeps the copy.
  Costs ~+0.13 ms on plain w1 steps (argmax node).
- PXA_MMVQ_V100_ROWS_NY=4: 4-row blocks for the Q6_K lm_head mmvq at 2..3 columns on sm_70. Bit-identical.
- PXA_MOE_VERIFY2_TILE=4x4 (WxR; _TILE_UP/_TILE_DOWN override): 4 warps x 4 rows per block in the verify2 MoE
  kernel (was 1 warp, 1 row: half occupancy, the K=704 down projection ran at ~170 GB/s). Bit-identical for all
  tiles 1x1..8x4. nvprof 2-token forward: MoE down 2.17 -> 0.90 ms. Splitting up/down tiles: within 0.1 ms.
- PXA_DENSE_FUG_NY: dense gate/up stays on the fused mmvq twin at 2..4 columns (was 2 plain mmvq + GELU kernel).
  Host-only; GELU/RELU only; outputs identical.

## Optional / not recommended
- PXA_MOE_VERIFY2_TILE_W1=1 (80232cc1d1): one-token decode MoE through the tiled kernel too (bit-identical).
  Target w1 8.65 -> 8.15 ms; llama-bench tg16 117.1 -> 125.2 t/s (+7%); MTP n1/n2 t/s unchanged (w1 steps are
  rare there). Good for plain decode; worth adding to the set once a plain-decode A/B is run on the server.
  (An apparent llama-bench pp1 regression was outlier reps in the average, not real.)
- PXA_ROUTER_GEMV_NY (1c9d498dbb): router GEMV at 2..4 tokens instead of cuBLAS. Output-identical, ~0.04 ms GPU,
  no measurable t/s. Not recommended.
- PXA_FA_VEC_NY (cd363af530): EXCLUDED. 2..4-column FA through the 1-column f32 vector kernel instead of the
  8-column WMMA f16 tile: w2 9.74 -> 9.28 ms, n1 179.0/176.1 -> 180.8/179.5, but outputs and acceptance change
  (prose accepted drafts 127 -> 114 of 128; n_probs/temp/grammar texts differ). Needs a quality review first.

## Remaining gap (recommended set, w2 9.74 ms vs ~8.4 ms target, ~1.3 ms)
2-token minus 1-token GPU time (nvprof, before DENSE_FUG_NY): FA +0.33 ms (bit-exact fix blocked; see FA_VEC_NY),
router +0.2 ms, dense ~+0.1 ms after DENSE_FUG_NY; MoE is now 0.25 ms BELOW the 1-token step. The rest is per-step
overhead outside these kernels (host submit ~5 ms overlaps the GPU; graphs measured no gain).
