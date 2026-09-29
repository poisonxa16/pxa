# PXA release gate -- volta family, RC3 artifact

```
package    : <package>
VERSION    : v2026.09.09-rc3  commit 6caeb014db133a6173fb92d25dd729bc33c9208c
binary     : version: 5412 (6caeb014db)
model      : /path/to/Qwen38-27B-Unc-PXQ4.gguf
cards      : 2,4 (host indices, CUDA_DEVICE_ORDER=PCI_BUS_ID)
shape      : -c 65536 -t 16, SERVER_ARGS='-sm layer --kv-unified'
gate flags : GATE_STRICT=1, REPS/NEEDLE_REPS/LOGIT_REPS at the gate's defaults (12/4/6)
started    : 2026-09-09T20:05:46Z
```

```
[20:05:46] gate starting
[20:05:46]   model    /path/to/Qwen38-27B-Unc-PXQ4.gguf
[20:05:46]   binaries <package>/bin
[20:05:46]   devices  2,4
[20:05:46]   logs     ./gate/logs/work-volta-20260909T200546Z
[20:05:46] === 1/6  greedy determinism, np=1, 12 runs ===
[20:06:01] server up (-np 1) after 15s
  PASS  np=1 greedy determinism 12/12 byte-identical (sha 86aab8e1de46)
[20:07:49] === 2/6  coherence ===
  PASS  coherence:  Paris. The capital city of Germany is Berlin. The capital c
[20:07:50] === 3/6  chat completions (/v1/chat/completions, production template) ===
  PASS  chat completions: production template renders correctly, 4/4 byte-identical (Paris)
[20:07:55] === 4/6  needle recall, 4 runs per prompt ===
  PASS  needle3121 recalled and sha-stable 4/4 (86aab8e1de46)
  PASS  needle20801 recalled and sha-stable 4/4 (495c1783cad8)
[20:10:13] === logit reproducibility, np=1, 6 runs ===
  PASS  logit reproducibility (np=1) 6/6 identical (a7b0de08a607)
[20:10:47] === 5/6  greedy determinism, np=2, other slot erased first ===
[20:11:03] server up (-np 2) after 16s
  PASS  np=2 slot 1 greedy determinism 12/12 byte-identical (sha 86aab8e1de46)
  PASS  np=2 slot 1 matches the np=1 reference at the same KV placement (86aab8e1de46)
[20:12:52] === logit reproducibility, np=2 slot 1, 6 runs ===
  PASS  logit reproducibility (np=2 slot 1) 6/6 identical (a7b0de08a607)
  PASS  token-0 logit match: np=2 slot 1 == np=1 reference, identical to the last digit
[20:13:14] === 6/6  unit tests ===
  PASS  test-pxq-cpu-dot
  PASS  test-kv-seq-shadow
  PASS  test-narrow-kernel-parity
[20:13:20] === GATE RESULT: PASS=13 FAIL=0 SKIP=0  (logs in ./gate/logs/work-volta-20260909T200546Z) ===
```

gate exit code: **0**  (0 = every check passed)
finished: 2026-09-09T20:13:20Z   workdir: `./gate/logs/work-volta-20260909T200546Z`

---

[21:37:26] gate starting
[21:37:26]   model    /path/to/Qwen3.8-Flash-Next-Uncensored-PXQU-4xP100.gguf
[21:37:26]   binaries <package>/bin
[21:37:26]   devices  0,1,5,6
[21:37:26]   logs     ./gate/logs/work-pascal-chatfix-20260909T213726Z
[21:37:26] === 1/6  greedy determinism, np=1, 12 runs ===
[21:39:12] server up (-np 1) after 106s
  PASS  np=1 greedy determinism 12/12 byte-identical (sha 82474843116e)
[21:40:49] === 2/6  coherence ===
  PASS  coherence:  Paris, which is situated in the north-central part of the c
[21:40:51] === 3/6  chat completions (/v1/chat/completions, production template) ===
  PASS  chat completions: production template renders correctly, 4/4 byte-identical (Paris)
[21:40:55] === 4/6  needle recall, 4 runs per prompt ===
  PASS  needle3121 recalled and sha-stable 4/4 (82474843116e)
  PASS  needle20801 recalled and sha-stable 4/4 (0371d396f910)
[21:45:24] === logit reproducibility, np=1, 6 runs ===
  PASS  logit reproducibility (np=1) 6/6 identical (17295c99f04d)
[21:46:21] === 5/6  greedy determinism, np=2, other slot erased first ===
[21:47:51] server up (-np 2) after 90s
  PASS  np=2 slot 1 greedy determinism 12/12 byte-identical (sha 82474843116e)
  PASS  np=2 slot 1 matches the np=1 reference at the same KV placement (82474843116e)
[21:49:30] === logit reproducibility, np=2 slot 1, 6 runs ===
  PASS  logit reproducibility (np=2 slot 1) 6/6 identical (17295c99f04d)
  PASS  token-0 logit match: np=2 slot 1 == np=1 reference, identical to the last digit
[21:50:25] === 6/6  unit tests ===
  PASS  test-pxq-cpu-dot
  PASS  test-kv-seq-shadow
  PASS  test-narrow-kernel-parity
[21:50:30] === GATE RESULT: PASS=13 FAIL=0 SKIP=0  (logs in ./gate/logs/work-pascal-chatfix-20260909T213726Z) ===

