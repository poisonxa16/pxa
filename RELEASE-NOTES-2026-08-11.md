# Release notes — 2026-08-11

Everything since `v2026.08.09`. This release merges two lines that had diverged: the codec/kernel
line (130 commits) and the dense-arch line (34 commits). No work from either side was dropped.

## New architecture support

**Muse-Glimmer 30B (dense).** Full arch port: 3:1 sliding-window / NoPE global attention, full-width
sigmoid attention gate, sandwich norms, softcapped head. Quantizes on the PXQ ladder.

**Chat parsing for recipient-addressed harmony.** This family renders an assistant turn as
`<|start|>assistant to=<recipient><|message|>{content}{END}` — `to=self` for chain-of-thought
terminated by `<|eom|>`, `to=user` for the final answer terminated by `<|eot|>`. It is *recipient*
addressed, not *channel* addressed, and its template contains no `<|channel|>`, so the GPT-OSS
handler never matched it and no specialized parser was selected. The whole raw generation — markers,
recipients and the model's own deliberation — was returned as `content`, with `reasoning_content`
empty. Fixed: `content` is now the answer and `reasoning_content` the deliberation.

Worth knowing for this family: its template defaults to `Reasoning strength: high`, which makes long
generations exhaust the token budget mid-deliberation and return nothing at all. Pass
`chat_template_kwargs: {"reasoning_strength": "low"|"medium"}`. Measured on an identical prompt, the
returned answer was byte-identical at all three levels while `high` cost 3.4× the tokens and 3.4× the
latency.

## Performance

**`PXA_FA_TILE_VOLTA`** (new, default off) — routes sm_70 flash-attention to the tile kernel instead
of WMMA. **+6.9% prefill**, decode unchanged. Full measurement in `docs/lab/LEVERS.md`.

**Ubatch guidance.** On the dense 30B, prefill peaks at `-ub 1024` and is flat from 2048 upward,
while decode is ubatch-insensitive. The effect is large enough to dominate several kernel levers, so
sweep ubatch before treating any prefill figure as a configuration's number.

Best measured configuration for that model on 2×V100 — `-ub 1024` with the decode and attention
levers armed: **1319.74 t/s prefill, 40.79 t/s decode**.

## Also in this release

Carried in from the codec/kernel line: `PXA_PXQ6_SHFL` (+4.1%, bit-exact), `PXA_PXQ3_PAIRLUT`
(+3.7%, bit-exact), `PXQ_CANON_v2` chained-FFMA accumulation, sm_60 head-packed GQA flash-attention,
`PXA_AUTO_SPEC` self-speculation, and the CUDA-graph batch work.

Carried in from the dense-arch line: `PXA_PXQ_MMVQ_FUGSPLIT` (dense PXQ up/gate decode as two plain
MMVQ launches, **+14.7%** on sm_70, bit-identical), `PXA_F32PREC_F16GEMM`, and a fix to the WMMA
half-accumulate mask query-row stride (`ne11` → `nb31`) that produced garbage output on sm_70
sliding-window models at depth.

Every gated flag in this release has a row in `docs/lab/LEVERS.md` giving its mechanism, its default,
and -- where a throughput claim is made for it -- the measurement and the exact configuration that
produced it. Levers that ship without a throughput claim say so rather than implying one.
