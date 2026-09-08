# 13 — MTP on 2xV100: blocked by the artifact, with the exact fix

MTP speculative decoding was retested on the two V100s (GPUs 2 and 4), TP=2,
against `qwen38-27b-unc-vllm-p2cf`. **It does not work, and cannot work with
any artifact currently on the box**, because the converter deliberately drops
the MTP block. This note records the evidence, the two independent blockers hit
along the way, and exactly what a fix costs.

## Measured

`qwen38-27b-unc-vllm-p2cf` (18 GB, gemma-norm fixed), 2xV100, TP=2,
FLASH_ATTN_V100, greedy, 512-token streams, n=8, matched params.

| config | e2e tok/s | decode tok/s | acceptance | mean accept len |
|---|---|---|---|---|
| no MTP, `--max-num-seqs 4` | **46.51** (sd 0.07) | 47.33 (sd 0.02) | — | — |
| MTP4, `--max-num-seqs 1` | **14.36** (sd 0.59) | 14.45 (sd 0.21) | **0.0103** | **1.041** |

Both configurations pass the `17x23 -> 391` check before and after the timed
runs. MTP is **3.24x slower** than not using it.

The 46.51 figure confirms the previously unverified 46.56 measured on the
broken-norm artifact — that number happened to be right, but it was measured on
a babbling model and could not have been trusted until now.

### Why the coherence check cannot catch this

With correct rejection sampling a speculative decoder produces *exactly the
target model's distribution* no matter how bad the draft is. A draft head with
uninitialised weights therefore yields perfectly coherent text — it just never
gets anything accepted, and you pay four extra forward passes per step for
nothing. `391` passes. The greedy output is fine. The throughput is a third.

**Acceptance rate is the only signal that says whether MTP is doing anything**,
which is why `bench/mtpmeasure.py` reports it and why it must be quoted
alongside any MTP tok/s number.

Note the definition: acceptance rate is accepted / drafted **tokens**. Dividing
by the number of draft *attempts* instead inflates it by the tokens-per-draft
(4 here) and is not comparable to anything. Cross-check against the DGX
reference: rate 0.5748 with mean accept length 3.30 implies
accepted/drafts = 2.30 and 2.30/4 = 0.575. Consistent.

Here: 162 accepted / 15704 drafted = 0.0103, mean accept length 1.041, and
every one of the 162 accepted tokens was at position 0 — nothing at positions
1, 2 or 3 was ever accepted. That is an untrained draft head, not a weak one.

## Blocker 1 (the real one): the converter does not emit the MTP block

The GGUF has it. `qwen35.block_count = 65`, `qwen35.nextn_predict_layers = 1`,
and block 64 carries all 15 tensors:

```
blk.64.attn_norm  attn_q  attn_k  attn_v  attn_output  attn_q_norm  attn_k_norm
blk.64.post_attention_norm  ffn_gate  ffn_up  ffn_down
blk.64.nextn.eh_proj  nextn.enorm  nextn.hnorm  nextn.shared_head_norm
```

`gguf_to_vllm/namemap.py::GGML_TO_HF` returns `None` for the MTP block range,
with the comment "plan §3: `mtp.*` is P3, not emitted in P1/P2". So:

| artifact | size | mtp.* tensors | layers |
|---|---|---|---|
| `qwen38-27b-unc-vllm-p2cf` | 18 GB | **0** | 0..63 |
| `qwen38-27b-unc-vllm-p1f` | 22 GB | **0** | 0..63 |
| `qwen38-27b-unc-vllm` (broken norms) | 18 GB | **0** | 0..63 |
| `Qwen3.8-27B-PXQ4-vllm-p2a-nf` | 14 GB | 15 | 0..63 |

`p2a-nf` is the only artifact with the MTP block — and it is **incomplete on
disk**: its index references `model-0000{1,2}-of-00006.safetensors`, neither of
which exists, and it has no `config.json`. It cannot load.

vLLM does not complain about any of this. It reads `mtp_num_hidden_layers: 1`
from `config.json`, logs

```
Detected MTP model. Sharing target model embedding weights with the draft model.
Detected MTP model. Sharing target model lm_head weights with the draft model.
```

— shares only the embedding and lm_head — and then runs the MTP block's own 15
weights uninitialised, silently. No error, no warning.

## Blocker 2 (config, already solved): draft sampler methods

Before reaching the weights, the boot dies with

```
ValueError: Static draft vocabulary phase one requires standard
probabilistic rejection sampling.
```

from `llm_base_proposer.py:2313`. The gate is

```python
self._enable_probabilistic_draft_probs = (
    self.speculative_config.rejection_sample_method == "standard"
    and self.speculative_config.draft_sample_method == "probabilistic")
```

Neither defaults to those values with `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0`,
so both must be passed explicitly. Working speculative-config:

```json
{"method": "qwen3_5_mtp", "num_speculative_tokens": 4,
 "rejection_sample_method": "standard", "draft_sample_method": "probabilistic"}
```

with `--max-num-seqs 1` and `VLLM_SM70_GEMMA_LONG_PREFILL_FUSED=0`.

## The fix

Two options.

**A. Restore p2a-nf's missing shards** — if `model-00001/00002-of-00006` and
`config.json` still exist somewhere (the DGX?), this is minutes rather than
hours. But p2a-nf predates the gemma-norm fix commit, so its *main stack* norms
need checking before trusting it, even though its MTP norms are correct (below).

**B. Teach the converter to emit `mtp.*`** — the P3 work, now fully specified.
The name map is 15:1 and unambiguous, confirmed against p2a-nf's own tensor
names:

```
blk.64.attn_norm             -> mtp.layers.0.input_layernorm
blk.64.post_attention_norm   -> mtp.layers.0.post_attention_layernorm
blk.64.attn_q / k / v        -> mtp.layers.0.self_attn.q_proj / k_proj / v_proj
blk.64.attn_output           -> mtp.layers.0.self_attn.o_proj
blk.64.attn_q_norm / k_norm  -> mtp.layers.0.self_attn.q_norm / k_norm
blk.64.ffn_gate / up / down  -> mtp.layers.0.mlp.gate_proj / up_proj / down_proj
blk.64.nextn.eh_proj         -> mtp.fc
blk.64.nextn.enorm           -> mtp.pre_fc_norm_embedding
blk.64.nextn.hnorm           -> mtp.pre_fc_norm_hidden
blk.64.nextn.shared_head_norm-> mtp.norm
```

**All seven MTP norms take the same gemma `-1` offset as the main stack.** This
is measured, not inferred — comparing the GGUF's blk.64 norm means against
p2a-nf's `mtp.*` norm means, which were produced while the transform was in
place:

| ggml tensor | gguf mean | p2a-nf mean | offset |
|---|---|---|---|
| `blk.64.attn_norm` | 1.03606 | 0.03606 | −1 |
| `blk.64.post_attention_norm` | 1.20627 | 0.20627 | −1 |
| `blk.64.attn_q_norm` | 1.79060 | 0.79060 | −1 |
| `blk.64.attn_k_norm` | 1.77951 | 0.77951 | −1 |
| `blk.64.nextn.enorm` | 0.53944 | −0.46056 | −1 |
| `blk.64.nextn.hnorm` | 0.84280 | −0.15720 | −1 |
| `blk.64.nextn.shared_head_norm` | 2.25205 | 1.25205 | −1 |

Consistent with the code: `qwen3_5_mtp.py` builds its block from
`Qwen3_5DecoderLayer` and all its norms from `Qwen3_5RMSNorm` — the same
gemma-convention classes as the main stack.

Reconversion also needs a policy decision the converter does not currently
have: which MTP modules are PXQ4 and which stay fp16. p2a-nf used the p2a
policy; p2cf is p2c. Grafting p2a-nf's MTP shard onto p2cf would mix policies
and is not recommended without checking the loader accepts it.

## Reproducing

```
# non-MTP baseline
/path/to/build-sm60/launch_v100.sh pxa-vllm-2v100 8420

# MTP4
EXTRA_ENV="-e VLLM_SM70_GEMMA_LONG_PREFILL_FUSED=0" MNS=1 \
  /path/to/build-sm60/launch_v100.sh pxa-vllm-2v100-mtp 8421 \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 4,
    "rejection_sample_method": "standard", "draft_sample_method": "probabilistic"}'

python3 /path/to/build-sm60/mtpmeasure.py --port 8421 --model m \
    --tokens 512 --n 8 --tag X --out /path/X.json
```

`mtpmeasure.py` gates on the 391 check either side and reports acceptance rate
and mean accept length from `/metrics`.
