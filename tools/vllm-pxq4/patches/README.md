# patches/

`solo-prefill-scheduler.patch`: with `PXA_LPT_SOLO=1`, `--long-prefill-token-threshold` is ignored while a request is the only one on the
engine (`running + waiting + skipped_waiting <= 1`); with any other request present the threshold applies as before. Output is unchanged.

Measured on Qwen3.8-27B PXQ4, TP4, 32 GB V100, `--max-num-batched-tokens 3328`, threshold 1648 (cold prompt first token):
32k 9.2 s -> 8.4 s, 140k 61 s -> 51 s, 252k 164 s -> 130 s. A short chat arriving behind a cold 138k prompt waits about 1 s longer.

On vLLM 1.5.1.dev23 the helper is inserted after the DFlash debug functions (this build has no `PXA_DECODE_SKIP_STEPS` lines) and every
`long_prefill_token_threshold` read in the scheduler, including the mamba-align cap, goes through `_pxa_lpt`. `python3 test_solo_prefill_helper.py` checks the helper logic.

## lenient-acceptance.patch (lossy, opt-in "Speed" profile)

Against stock vLLM 1.5.1 `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`. With `PXA_LENIENT_LOGR=<x>` a drafted token is also accepted
when its target logit is within `x` (a log ratio, for example -2.3 = at least 10% of the top token's probability) of the target's top logit; unset, the
kernels compile without the branch and match stock. It applies to greedy, sampled and the DFlash2 sparse top-k path, so greedy output changes too.

| profile | setting | single stream code T0.6 | 8 streams combined, code | agent load | HumanEval-164 |
|---|---|---|---|---|---|
| Quality | unset (stock, exact) | 205-225 tok/s | 496 | 15.7 turns/min | 91.5% |
| Speed | -1.6 / -2.3 | 249 / 255 tok/s | 552 (-2.3) | 16.8 turns/min (-1.6) | 84.1% (-1.6) |

The code-quality cost is measured only at -1.6 on one fine-tune (-7 points); -2.3 was run on small suites (about 30 items each) without a visible loss.
Keep exact acceptance for code and tool use, and run HumanEval/IFEval before offering a lenient tier.

`test_lenient_acceptance.py <stock file> <patched file>` needs one GPU. It passed (40 trials x 3 paths, 0 failures) on the 1.5.0 build of this change;
it was not rerun against the 1.5.1 file because all GPUs were serving.

`swa-keep-hashed-block-pool.patch`: ported to vLLM 1.5.1.dev23. That tree's `free_blocks` appends every freed block in caller order and has no `reuse_first` / `prepend_n` split (the 1.5.1 release does). With `PXA_SWA_KEEP_HASHED=1` hashed blocks are appended (LRU end, evicted last) and unhashed blocks are prepended; `reuse_first` does not pull hashed blocks forward. Unset or `0` (the default here; the field-report patch defaulted the variable to `1`) keeps dev23's single append. The patch also adds `FreeKVCacheBlockQueue.prepend_n`. `python3 test_swa_keep_hashed.py` checks the split. Measured together with the other KV-cache patches in section 10 of the field report.

The image build `deploy/Dockerfile.field` applies all three patches. None of the levers are set in the image, so a normal boot matches stock dev23.

`dflash-pxqn-load.patch`: a DFlash draft layer i is constructed with vLLM prefix `layers.{i + target_depth}`. PXQN checkpoints key `pxqn_modules` as `layers.{i}`. The patch shifts those keys by the target depth before the layers are built, so `pxq4_slabs` / `pxq4_anchor` are registered and `load_weights` finds them. The entry's `layer` field is left at the checkpoint index, which is the rotation the weights were stored with.
