# PXA release gate -- volta family, v2026.09.20 package

```
package    : <package>
VERSION    : v2026.09.20  commit 032c280d6269a9bcb7705b94cbabc041947b68cd
binary     : build 5732 (6b3c637d94bd744027f1d027f7b3ddd7c96e2490), cuda_archs_built 60;61;70, cuobjdump sm_60 sm_61 sm_70
model      : /path/to/Qwen3.8-27B-PXQ4.gguf
cards      : 2,4 (host indices, CUDA_DEVICE_ORDER=PCI_BUS_ID)
shape      : -c 32768, SERVER_ARGS='--spec-type none -sm layer'
gate flags : GATE_STRICT=1, REPS/NEEDLE_REPS/LOGIT_REPS at the gate's defaults (12/4/6)
started    : 2026-09-21T17:06:40Z
```

```
[17:06:40] gate starting
[17:06:40]   model    /path/to/Qwen3.8-27B-PXQ4.gguf
[17:06:40]   binaries <package>/bin
[17:06:40]   devices  2,4
[17:06:40]   logs     ./gate/logs/workdir-qwen-v100-layer
[17:06:40] === 1/6  greedy determinism, np=1, 12 runs ===
[17:06:55] server up (-np 1) after 15s
  PASS  np=1 greedy determinism 12/12 byte-identical (sha 2e9a5fc373d1)
[17:08:27] === 2/6  coherence ===
  PASS  coherence:  Paris. The capital city of Germany is Berlin. The capital c
[17:08:29] === 3/6  chat completions (/v1/chat/completions, production template) ===
  PASS  chat completions: production template renders correctly, 4/4 byte-identical (Paris)
[17:08:31] === 4/6  needle recall, 4 runs per prompt ===
  PASS  needle3121 recalled and sha-stable 4/4 (2e9a5fc373d1)
  PASS  needle20801 recalled and sha-stable 4/4 (75e139bac02a)
[17:10:53] === logit reproducibility, np=1, 6 runs ===
  PASS  logit reproducibility (np=1) 6/6 identical (9d0f565b072b)
[17:11:16] === 5/6  greedy determinism, np=2, other slot erased first ===
[17:11:31] server up (-np 2) after 15s
  PASS  np=2 slot 1 greedy determinism 12/12 byte-identical (sha 2e9a5fc373d1)
  PASS  np=2 slot 1 matches the np=1 reference at the same KV placement (2e9a5fc373d1)
[17:13:09] === logit reproducibility, np=2 slot 1, 6 runs ===
  PASS  logit reproducibility (np=2 slot 1) 6/6 identical (9d0f565b072b)
  PASS  token-0 logit match: np=2 slot 1 == np=1 reference, identical to the last digit
[17:13:31] === 6/6  unit tests ===
  PASS  test-pxq-cpu-dot
  PASS  test-kv-seq-shadow
  PASS  test-narrow-kernel-parity
[17:13:35] === GATE RESULT: PASS=13 FAIL=0 SKIP=0  (logs in ./gate/logs/workdir-qwen-v100-layer) ===
```

Note: this cut's greedy determinism shas (`2e9a5fc373d1`, `75e139bac02a`) are unchanged from the
previous cut's gate on this same prompt shape — the new attention kernels below the 1,280-token KV
floor and the new q8_0 prefill path do not touch this gate's prompt lengths or code path.

**Two new levers exercised by this gate, both on by default on V100s:** `PXA_FA_D256_VOLTA_TILE`
(engages once KV reaches 1,280 tokens — both needle checks and the np=2 checks at 20,801 tokens
cross that floor) and `PXA_FA_MMA_VOLTA_Q8` (the prefill path for both needle lengths). Both are
armed by their shipped defaults in the run above; nothing here was set by hand.
