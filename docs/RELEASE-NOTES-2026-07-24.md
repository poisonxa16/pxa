# Release notes — unified build (2026-07-24)

This release **unifies two lines of development into one binary**: the PXQ quant/speed
release line and the upstream-correctness/architecture-breadth integration line. Previously
these lived in separate build trees; this is the single canonical build with everything folded in.

## Headline

- **One build, everything in it.** PXQ codecs + all speed levers + the full upstream
  correctness/arch-breadth port set, in a single multi-arch binary (sm_60/61/70/86/89).
- **No behavioral regression on existing models** — the upstream ports that duplicated
  fixes already present were resolved in favor of the shipped (curated) versions; the new
  ports are additive (new architectures, correctness guards, and server robustness).

## New architecture support

- **Laguna (S-2.1 / XS-2.1)** — full `LLM_ARCH_LAGUNA` end-to-end support: loader, hparams, tokenizer, and a dedicated build-context graph.
- **Cohere2-MoE (North Mini Code)** — full arch support: loader, build-context graph,
  a graph-parallel expert path, chat parser + template, and correct thinking-block parsing
  under `--reasoning off`.
- **Gemma-4 E2B/E4B** — per-layer projection optimization and tolerance for models missing
  the shared-KV edge tensors. (Complements the existing Gemma-4 MoE fail-clean guard.)

## Correctness fixes (CUDA / kernels)

- dmmv out-of-bounds, fused-MoE up/gate expert-index correctness, norm in BF16.
- tile-f16 flash-attention softcap + Q6_0 copy-constant fixes.
- binbcast add/mul fast-path contiguity guards; MUL scalar fast-path contiguity guard;
  Q8_0 operand handling in `ggml_cuda_op_add`.
- DS4 fp16 NaN-clamp extended to the MoE-expert quantizer (beyond the upstream scope).
- Minor GGML discrepancies + a `set_rows` F32 defensive assert harvested from upstream.

## Correctness / robustness fixes (engine + server)

- MROPE/IMROPE uninitialized position sections in the legacy null-position decode path.
- CPU-only model load no longer crashes on a CUDA-enabled build.
- deepstack image-embedding stride out-of-bounds in the mtmd helper.
- Integer overflows in perplexity sizing at large context × vocab.
- Normalized the disabled-context-shift overflow error.
- Free raw multimedia data from `server_tokens` after encoding (memory).
- Infill double-submit + MTMD copy infinite-loop + multi-prompt use-after-move.
- Draft/target vocab-type compatibility check for speculative decode.

## MoE / MTP / prompt-cache

- `--prefetch-experts` MoE page-cache read-ahead.
- qwen35 / qwen35moe MTP warmup recurrent-conditioning shift.
- Prompt-cache: do not recover a state below the cache-RAM similarity threshold.
- Variance-based checkpoint eviction.

## Carried forward from the PXQ release line (already shipped, now in the same binary)

- **Speed levers** (all env-gated, `=0` rollback): `PXA_ENHANCE`, `PXA_SPEC_1ROW`
  (spec-verify 1-row GEMV, ~+6.6% MTP decode), `PXA_ROUTER_FUSE`, `PXA_SPEC_SMALLN`,
  `PXA_CUBLAS_EAGER_INIT` / `PXA_CUBLAS_EAGER_WS`, `PXA_VOLTA_CUBLAS_NE11`,
  `PXA_MTP_LAZY_WARMUP`, `PXA_FUSE_DELTANET`.
- **PXQ codecs**: PXQ1 / PXQ2 / PXQ4 / PXQ6 + the PXQU universal mixed-tier maps.
- **bradrlaw fixes**: multi-GPU `-sm layer` no-NVLink (PHB) warning; Gemma-4 MoE fail-clean
  guard; `libnccl.so.2` release packaging fix.

## Community bug-finders 🏅

- **Last-Guitar-5924** (r/LocalLLM) — deepseek2/MLA fa-off context-decay cliff on a Tesla P40.
- **[bradrlaw](https://github.com/bradrlaw)** — dual-GPU `-sm layer` PHB decode collapse +
  `libnccl.so.2` packaging.
