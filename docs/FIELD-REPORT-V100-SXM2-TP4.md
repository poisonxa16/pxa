# Field report: PXQ4 + DFlash2 on 8x Tesla V100 SXM2 32 GB (TP4 per engine), 1Cat-vLLM 1.5.1

This is a measurement-backed report from running the PXQ4 vLLM backend as a customer-facing service, in a setup the existing
SM70 sweeps do not cover: 32 GB SXM2 cards, tensor parallel 4 (two engines per 8-GPU host), long agent-style prompts and
fine-tunes of Qwen3.8-27B. It is written to be useful to maintainers: what worked, what did not, three traps that cost us hours,
and a few small things that might be worth upstreaming. Everything below was measured between 2026-10-08 and 2026-10-11 on
the same host. Numbers are medians of 3 unless stated; "strict" means exact rejection sampling (no lenient acceptance).

## 1. Setup
| | |
|---|---|
| GPUs | 8x V100 SXM2 32 GB, two NVLink islands of 4; one engine per island (TP4) |
| engine | 1Cat-vLLM 1.5.1 (also 1.5.1.dev925 for a GLM build), our build of the PXQ4 SM70 kernel library ("v18b") |
| target | Qwen3.8-27B fine-tunes converted with `convert_hf_to_gguf` -> `llama-quantize PXQ4` -> `gguf_to_vllm` (policy `p2a`), vision tower kept |
| drafter | DFlash2, 7 draft tokens, drafter TP4 (BF16 or PXQ4) |
| KV | FP8 E4M3, `--mamba-cache-mode align`, prefix caching, 12 sequences, `--max-num-batched-tokens 2048`, `--long-prefill-token-threshold 1648` |
| context | 262,144; KV pool about 1.85-1.91M tokens per engine |
| workload | coding agents: 10-12 concurrent requests, 45-75k token prompts, about 88% prefix-cached, 150-550 output tokens |

## 2. Results that may be useful as reference points (strict acceptance, Qwen3.8-27B family, TP4)
| metric | value |
|---|---|
| single stream, code T0.6 / T0 | 215-255 / 225-317 tok/s |
| single stream, prose T0.6 / T0 | 123-148 / 120-180 tok/s |
| combined, 1 / 2 / 4 / 8 streams (code) | 220-282 / 327-341 / 437-450 / 496-552 tok/s |
| combined, 1 / 2 / 4 / 8 streams (prose) | 123-134 / 192-197 / 242 / 284-292 tok/s |
| cold prefill 32k / 140k | 3.3k / 2.2k tok/s (9.2 s / 61 s first token) |
| cold prefill 252k | 164 s (see section 5 for the +21% patch) |
| cached follow-up turn at 120k / 200k / 252k | 2.3 / 4.7 / 4.9 s, 98% of the prompt from cache |
| needle retrieval, 3 facts, 85k to 252k tokens | 3/3 at every length |
| accepted tokens per step under a 12-chat agent load | 2.6-2.8 (Swift fine-tune with the generic drafter) vs 4.0 (fine-tune with its own drafter) |
Quality gates run on the converted checkpoints (all strict): GSM8K-100 94-96%, HumanEval-164 91.5% greedy, IFEval-150 (thinking on) 88.7-90.0% prompt-level,
tool calls 24/24, vision 3/3 colours, up to 4 images per prompt.

## 3. Traps (each one cost us at least an hour)
### 3.1 The torch compile cache key ignores the drafter's quantization
Two runs on the same port with different drafters (BF16 vs PXQ4 version of the same DFlash2 drafter) load each other's compiled
graph. Symptoms: `RuntimeError: Worker failed with error ''weight''` (PXQ4 drafter after a BF16 run) or `''pxq4_slabs''` (BF16 drafter after a PXQ4 run) on the first
drafter forward, after weight loading has succeeded, with the log line "Directly load AOT compilation from path .../torch_aot_compile/<hash>/rank_N_0/model".
Weights and config of the drafter checkpoint were fine. Workaround: clear `<cache dir>/<port>/vllm/torch_compile_cache` when switching drafters, or use a cache directory per drafter.
Suggested fix: include the drafter's `quantization_config` (or its safetensors hash) in the AOT cache key.

### 3.2 `fp8_gemm_sm70: input must be float16` with an FP8 DFlash2 drafter
Community DFlash2 drafters are sometimes published as FP8 (E4M3 with 128x128 block scales, `weight_scale_inv`). The engine rejects them at the first forward.
Dequantizing to BF16 (multiply each block by its scale, drop the scale tensors and the `quantization_config`) boots and runs normally; a 4-line script is enough.

### 3.3 A drafter built from the stock model quantized with the "o/gate/up/down/fc only" recipe is not the shipped layout
The shipped Swift drafter (`drafter-pxq4c`: 36 U8 slabs, 36 F16 anchors, 45 BF16 tensors) also quantizes `q_proj`, `k_proj` and `v_proj`. An earlier quantizer
variant that keeps q/k/v in BF16 produces a different tensor set. When rebuilding a PXQ4 drafter for another fine-tune, match the 36-slab layout and verify names, dtypes and shapes
against the reference file before booting.

## 4. Things that did not help (so nobody else needs to retry them on this class of hardware)
| idea | result |
|---|---|
| PXQ4-quantized drafter instead of BF16 (TP4, 32 GB) | same quality, acceptance 4.04 -> 3.87, agent-load throughput equal within noise (15.6 vs 15.7 turns/min); no gain |
| extended CUDA graph capture sizes (72-96 tokens for 12 sequences x 8 rows) | no measurable gain on the agent workload (prefill-bound) |
| `--max-num-batched-tokens 8256` (to reach the Q8000 prefill path) | no extra speed over 3328 once solo-prefill chunking is on; worse tail latency under load |
| FP8 E5M2 KV instead of E4M3 | worse under load |
| `--num-speculative-tokens` other than 7 | disables the profile-gated verifier acceleration (`contract_mismatch`), boot loops at 5, 2.5x slower at 8 |
| lenient acceptance (`PXA_LENIENT_LOGR=-1.6`) | +6-7% agent-load throughput and +11-34% single-stream speed, but HumanEval 91.5% -> 84.1% (a real 7-point loss on code) |
| `thinking_token_budget` request parameter | rejected with HTTP 400 on this model runner ("Model Runner V2 does not yet support the thinking_token_budget request parameter") |
| async scheduling off | +8-15% at 4-8 streams, -10-20% long-context single decode; not adopted |

## 5. Solo-prefill chunking (small scheduler patch, lossless)
`--long-prefill-token-threshold 1648` keeps one huge prompt from monopolizing the engine, but it also forces a lone long prompt into 1648-token chunks even when nothing else is running.
A patch that ignores the threshold only while the request is the only one on the engine (`running + waiting + skipped_waiting <= 1`) plus `--max-num-batched-tokens 3328`:

| cold prompt | threshold 1648, MNBT 2048 | solo patch, MNBT 3328 |
|---|---|---|
| 32k | 9.2 s | 8.4 s |
| 140k | 61 s | 51 s |
| 252k | 164 s | 130 s |
Under real multi-request load the threshold still applies, so the protection is unchanged; the cost is about +1 s for the first token of a short chat that arrives behind a cold 138k prompt, and a 3% smaller KV pool.
Output is unchanged (chunking only). Patch: the scheduler's two `long_prefill_token_threshold` uses call a helper that returns 0 when the engine is idle.

## 6. Small items
- `--limit-mm-per-prompt '{"image":4}'` on a vision checkpoint cost nothing measurable in KV pool (1,853,986 -> 1,857,545 tokens); the engine default is 1 image per prompt.
- The converted fine-tune checkpoints keep `mtp-bf16.safetensors` next to two big shards; `convert_hf_to_gguf` and the pipeline's shard-count sanity check (written for 10+ shards) need adjusting.
- A half-written `*-BF16.gguf` left by a killed run makes the conversion step skip and `llama-quantize` fail with "tensor data is not within the file bounds"; delete it before rerunning.
- Test engines that load under the production served-model name are picked up by name-routing gateways as soon as they are healthy; give test engines their own name.

## 7. How this was measured
The harness (12-conversation agent load with think-time gaps, HumanEval/GSM8K/IFEval gates, long-context needle test, rolling restarts) is plain Python and shell;
the author can attach it to this PR or an issue if it is useful. All figures above are from the final builds, one test engine at a time on idle GPUs.

## 8. Scope: what should carry over to 16 GB / TP2 and what may not
Measured here: 32 GB SXM2, TP4, 1Cat-vLLM 1.5.1. The existing SM70 sweeps in this repo use 16 GB PCIe cards at TP2 on an older base, so please read the table before applying anything.
| finding | expected on 16 GB / TP2 |
|---|---|
| compile-cache key ignores the drafter's quantization (3.1) | applies unchanged (vLLM AOT cache behaviour) |
| FP8 DFlash2 drafter rejected, BF16 dequant works (3.2); drafter layout with q/k/v quantized (3.3) | applies unchanged |
| lenient acceptance costs code quality (HumanEval -7 points at LOGR -1.6 on one fine-tune) | applies in principle (algorithm and model property); size depends on the model |
| solo-prefill chunking (section 5) | the scheduler logic carries over; the gain depends on the chunk size the card can afford |
| PXQ4 drafter gives no speed gain (section 4) | **speed only, at TP4/32 GB.** On 16 GB the BF16 drafter (about 3.8 GiB, see `VLLM.md`) leaves almost no KV; the 1.3 GB PXQ4 drafter may be what makes DFlash fit at all. Do not read our result as "do not quantize the drafter" |
| `--max-num-batched-tokens 3328` and larger | needs activation memory that 16 GB cards may not have; keep 2048 if it does not fit |
| `--limit-mm-per-prompt '{"image":4}'` costs nothing (6) | measured at 32 GB with MNBT 3328; on 16 GB `VLLM.md` already documents a large encoder-cache reservation, so re-measure |
| extended capture ladder, E5M2 KV, draft counts (section 4) | negative results at TP4/32 GB under a prefill-bound workload; TP2 has cheaper all-reduce and different graph pressure, so re-test before copying the conclusion |
| `thinking_token_budget` rejected (HTTP 400) | specific to the model runner in 1.5.1; the 1.5.0 base may behave differently |
| absolute tok/s, prefill rates, KV pool sizes | not transferable (different TP degree and memory) |

## 9. Code
| file | purpose |
|---|---|
| `tools/vllm-pxq4/tools/dequant_fp8_drafter.py` | FP8 DFlash2 drafter to BF16 (3.2) |
| `tools/vllm-pxq4/tools/quantise_dflash2_drafter.py` | BF16 DFlash2 drafter to PXQ4; `--quantize-qkv` for forks that load a fused PXQ4 qkv |
| `tools/vllm-pxq4/patches/solo-prefill-scheduler.patch` | opt-in `PXA_LPT_SOLO=1` (section 5), with a logic test |
| `tools/vllm-pxq4/patches/swa-keep-hashed-block-pool.patch` | keeps hashed sliding-window blocks, restores partial prefix hits on 1.5.1 (section 10) |
| `tools/vllm-pxq4/patches/lenient-acceptance.patch` | opt-in lenient acceptance (`PXA_LENIENT_LOGR`), the lossy Speed profile (section 4); Quality = unset |
Both scripts reproduce the drafters used for the measurements here byte for byte (tensor payload, sha256 for the BF16 one).

## 10. Findings on 1.5.1 that touch the PXQ4 setup
All on Qwen3.8-27B PXQ4, TP4, 32 GB V100, DFlash2 k=7, unless noted. Single runs unless noted; short-throughput noise is +-5-10%.

### Prefix cache with the hybrid (Mamba) state
- 1.5.1 keeps Mamba checkpoints sparsely (`--prefix-cache-retention-interval`, default 0: only replay and shared-prefix boundaries). Partial hits were lost: on an 84.7k-token conversation, a cut at 30k or 50k cached 0 tokens (1.5.0: 26,368).
- Two changes restored them: `--prefix-cache-retention-interval 13184` and a `block_pool.py` change so hashed sliding-window blocks are not recycled first (`patches/swa-keep-hashed-block-pool.patch`, `PXA_SWA_KEEP_HASHED=1`). With both, cuts at 30k and 50k cache 26,368 tokens, the append case 82,400 of 84,668, the last-pair case 82,400.
- Retention 13184 vs 26368 gave identical hit counts and identical cold-prefill time. Retention costs about 4 s on a cold 84.7k prefill (32.8 s vs 28.6 s with 0).
- The block-pool patch was measured together with our other KV-cache patches (`kv_cache_utils`, `single_type_kv_cache_manager`), not on its own.
- `--long-prefill-token-threshold` cannot be below the block size (1648 here); lower values assert at init.
- `prompt_tokens_details` is null when nothing was cached and only shows cached tokens for prompts longer than one block.

### Scheduler and graph settings
| setting | result |
|---|---|
| `--long-prefill-token-threshold 1648` | new short chat behind a cold 138k prefill: 58 s -> 1.0 s with 4 decoders; a lone 138k prefill goes 62 s -> about 80 s (see section 5) |
| `--max-num-seqs` 8 -> 12 | pool -0.4%; 8 chats + a long prompt no longer queue (new chat 87 s queued -> 2.3 s); throughput and peak memory unchanged |
| `PXA_DECODE_SKIP_STEPS` 4 -> 0 | decoders nearly stall during a long prefill (about 15 vs 50 tok/s combined); keep a few decode-only steps after each prefill step |
| `--num-speculative-tokens` 5 | changes the cache alignment (block 3232), so retention 13184 is invalid and the boot fails; the kernels are built for 7+1 rows, k=8 is 2.5x slower |
| async scheduling off | +8-15% at 4-8 streams, -10-20% single long-context decode |
| kernel v20 (`in_proj_ba` GEMV, bit-exact) | GEMM time -0.8 ms, all-reduce +0.7 ms; no end-to-end gain (graph-launch skew between ranks) |
| FP8 KV: E5M2 vs E4M3 | 1.5.1's Q8000 prefill, E4M3 long-context decode and scalar-tail paths run only with E4M3; E5M2 disables them |

### The release prefill profile (`--max-num-batched-tokens 8192`)
- Faster long prefill (140k: 2.97k vs 2.55k tok/s) but about 4 GiB per GPU more reserved (weights 6.8 GiB, graph reserve 2.1 GiB estimated vs 0.9 actual, KV 15.3 GiB vs 19.4 GiB): pool 1.83M -> 1.33M tokens.
- With MNBT exactly 8192 and decoders running, the engine died: `RuntimeError: SM70 GQA architecture requires KV in [8192, 262144] with 32-token alignment, got 8160`. The mamba-aligned split schedules an unaligned 8160-token first chunk when decoders take 32 tokens of the budget; 8256 avoids it.
- 1.5.1 gave no extra decode speed for the PXQ4 format itself: the PXQ4 plugin and v18b kernel library work unchanged, and decode at 128-140k was 110-120 tok/s on both 1.5.0 and 1.5.1.
