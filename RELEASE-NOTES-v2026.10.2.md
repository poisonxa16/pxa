# PXA v2026.10.2

A fix for large-context one-card decode, Gemma 4 speculation that works, and a lighter Gemma KV cache. Same models, no re-quant. Upgrade recommended.

## Fixed: a large context could silently push weights out to host RAM

On a single 16 GB card, an f16 KV cache at a large context (for example `-c 65536` on the Qwen3.8-27B one-card file) tipped the weights from "all on the card" to "part read from host RAM over PCIe on every step". Decode fell from about 24 to **1.7 t/s** and prefill from 50 to 14 t/s. It was not a kernel bug, it was a placement cliff.

- With no `-ctk` / `-ctv` given, the engine now picks the fastest KV type that keeps the weights on the card: f16 if it fits, else q8_0, else q4_0, and logs the arithmetic on the `PXA_REGISTRY` line.
- One P100, Qwen3.8-27B one-card file, no KV flags: `-c 8192` picks f16 (24.7 t/s), `-c 32768` picks q8_0 (24.4 t/s), `-c 65535` picks q4_0 (24.2 t/s). Everything stays resident. One V100 at `-c 65535` picks q4_0 and decodes at about 33 to 35 t/s.
- If a placement still has to spill, the server now says so in the log instead of just getting slow.
- Anything you pass yourself (`-ctk`, `-ctv`) is used as given.

## Gemma 4 26B-A4B: speculative decoding works, and the KV cache is much smaller

- **MTP with the upstream drafter.** The Gemma 4 assistant (drafter) file now loads as it is published: no conversion. Pass it with `-md` and the engine arms `--spec-type mtp:n_max=1` by itself. On one V100 (QAT q4_0, 16k context, greedy), long prose goes from 117 to about **135 t/s** and code to about 134 t/s; with `--spec-type mtp:n_max=2` prose reaches about 150 t/s (149 to 153 across runs). UD-Q4_K_XL: 118 to about 136 t/s prose, about 120 code.
- **Sliding-window KV is the default.** Gemma 4 keeps only the window it needs for its local layers: the KV cache at 16k context drops from 3.5 GB to **0.77 GB**. On a 16 GB card the old build could not even start the server at `-c 8192` and `-c 16384` with the QAT file; this one boots at 16k and recalls a needle placed in a 14k-token prompt. `PXA_GEMMA4_ISWA=0` restores the old layout.
- **V100 prefill on long Gemma prompts is about 3% faster** (pp4096 1986 to 2044 on the QAT file, 1997 to 2055 on UD-Q4_K_XL). pp512 and plain decode (about 115 to 120 t/s) are unchanged.

### Limits, plainly

- Non-English text is where the drafter agrees least. On a Finnish answer only about 40% of drafted tokens are accepted, so MTP does not help there. A gate stops drafting when acceptance drops under 0.5 and keeps decode near plain speed, but "near" means about 10% below it on Finnish (about 104 vs 116 t/s), and short English chat answers are a few percent below plain as well (111 vs 118). Long prose and code are where MTP pays. `PXA_GEMMA4_MTP_GATE=0` turns the gate off, `PXA_GEMMA4_MTP_AUTO=0` stops `-md` from arming MTP by itself.
- A Gemma 4 prompt of about 16k tokens still runs out of memory on a 16 GB card at the default batch size. 14k tokens is what we verified on one V100.
- The drafter file we tested is the q4_0 requant of the upstream assistant.

## One P40 item (unmeasured)

A single sm_61 card with 20 GiB or more (a 24 GB P40) was being treated like an 11 GB 1080 Ti: small batch, speculation declined. It now gets the 16 GB-class batch (`-b 2048 -ub 2048`), is allowed to speculate, and arms the MTP head at `n_max=1` when the model has one. **We do not own a P40: this path is unmeasured, its status in the registry is INFERRED.** The 11 GB 1080 Ti path is unchanged (checked on a 1080 Ti: `-ub 768` and speculation declined, as before). Thanks to thisistimow for the P40 report.

## PXA Control

- KV is **auto** by default in the launcher and in PXA Control (no `--ctk/--ctv` unless you choose one).
- GPU telemetry is included in the Report-a-problem bundle, with the same review-before-send rule as before.
- A heat warning on the Rig and Launch pages when a card runs hot.

## Checked on this build against v2026.10.1

Greedy 512-token output is byte-identical to v2026.10.1 on every Qwen row: 2x V100 (PXQN4), 2x P100 (PXQN4), 4x P100 (PXQN4), 1x V100 and 1x P100 (PXQN4 and the one-card file). Determinism 12/12 at np=1 and np=2 on the V100 pair and the P100 pair. 30k-token needle recalled on the V100 pair and on one P100. Decode is unchanged within noise on the rows this release does not touch: 1x V100 PXQN4 about 34 t/s, 2x P100 PXQN4 about 37.5 t/s, 1x P100 PXQN4 about 24 t/s.

## Downloads

Two tarballs (Ubuntu 24.04 and newer, and Ubuntu 22.04), plus the PXQN library for source builds (both OS variants), which matches the tag `v2026.10.2`.

## Upgrade

Unpack over your old folder or into a new one. Same models.

## Credits

mistrjirka, quenthalion, thisistimow (P40 report, first PXA-in-Odysseus confirmation).

## Community

- Discord: https://discord.gg/EqazvV9tf
- Support on Ko-fi: https://ko-fi.com/shatteredrealms1
