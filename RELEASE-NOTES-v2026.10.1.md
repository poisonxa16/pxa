# PXA v2026.10.1

Fixes for everything found after v2026.10. Upgrade recommended.

## What was wrong in v2026.10

We found these after the release and fixed all of them. Some of them cost you real speed or broke real output, so here they are plainly.

1. **Single-card and layer-split decode was about 30% slower than it should have been.** The fused delta-net conv kernel was switched on for every setup, but it only pays off with tensor split. On one card it measured about 31% slower on a V100 and about 21% slower on a P100 (27B, decode). It is now on automatically only with `-sm tensor` (where it is about 4-5% faster on a pair). `PXA_DN_CONVFUSE=1` or `=0` in the environment still overrides it.
2. **Gemma 4 26B-A4B produced broken non-English output.** The router scaling was applied twice, which flattened the expert choice. It was there from 2026-09-20 on. English looked mostly fine, other languages did not. Fixed.
3. **4x P100 tensor split with a PXQN model could crash when free VRAM was uneven across the cards.** With uneven free memory the split fell on a row that cut through a PXQN rotation block. The split now rounds to whole rotation blocks, so the uneven case boots and generates.
4. **Automatic n-gram speculation could collapse decode on sampled chat (temperature above 0).** Drafts of up to 64 tokens looked good on repeated text but on fresh sampled chat they were mostly rejected (one user's 9B fell from 80 to 19 t/s). Drafts are now capped at 8 tokens: the collapse is gone, and sampled prose runs within a few percent of no speculation while edits still gain. `PXA_SPEC_NGRAM_NMAX` sets any other depth.
5. **One V100 lost 5-8% to a GEMV fusion.** The fused rotation-plus-GEMV path is now off on a single V100 (output identical). One V100 now decodes the 27B one-card file at about 35 t/s and PXQN4 at 34. The same switch costs about 2.5% of prompt processing (prefill) on a single V100, a trade we took for the decode gain.

## New in PXA Control

- **Report a problem** button: it builds a report with paths, host names, IPs and user names redacted, and shows you exactly what will be sent before anything leaves your machine. You review it first. The report goes to bugs.pxanetwork.com.
- **Community high-score board** in the #benchmarks channel on Discord: when you set a new record, PXA Control asks for a name. Nothing is sent without your click.
- **Discord and Support** links in the header.

## Numbers, v2026.10 vs v2026.10.1

Qwen3.8-27B, decode, tokens per second, same file and flags on both builds.

| Setup | v2026.10 | v2026.10.1 |
|---|---:|---:|
| 1x V100, PXQN4 | 22.7 | **34.0** |
| 1x V100, one-card 27B | 33-34 | **35.1** |
| 1x P100, PXQN4 | 19.9 | **24.0** |
| 2x V100, tensor split, PXQN4 | 56.4 | 56.4 |
| 2x P100, tensor split, PXQN4 | 37.8 | 37.8 |
| 4x P100, tensor split, PXQN4 | 30.8 | 30.7 |
| 4x P100, tensor split, PXQN4, uneven free memory | crash | runs |
| Ornith 35B-A3B, 2x P100, `-sm layer` (10-request chat run) | about 60 | about 67.6 |

`llama-bench tg128`, the same file and flags on both builds, the median of alternating runs. Output is byte-identical to v2026.10 on every row above, and 12/12 determinism, 30k needle recall, MTP speed and English plus Finnish output were rechecked on the final build.

Pairs are unchanged because the fused conv stays on for them. The single-card and layer-split rows are the ones that were held back.

## Docker

- Engine image: `ghcr.io/poisonxa16/pxa:v2026.10.1` (also `:latest`), built from this release's tarball.
- The vLLM sidecars have no changes in this release: keep using `ghcr.io/poisonxa16/pxa-vllm:sm60-v2026.10` and `:sm70-v2026.10`.

## Upgrade

Download the new tarball from the release page and unpack it over your old folder or into a new one. Same models, no re-quant needed.

## Community

- Discord: https://discord.gg/EqazvV9tf
- Support on Ko-fi: https://ko-fi.com/shatteredrealms1
