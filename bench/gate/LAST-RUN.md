[19:07:00] gate starting
[19:07:00]   model    /mnt/cachetwo/models/qwen38-27b-stock/Qwen3.8-27B-PXQ4.gguf
[19:07:00]   binaries /mnt/cacheone/rel-gate/pxa-v2026.09.13-rc3/bin
[19:07:00]   devices  2,4
[19:07:00]   logs     /mnt/user/PXACLAW/speed-campaign/release/logs/workdir-volta
[19:07:00] === 1/6  greedy determinism, np=1, 12 runs ===
[19:07:19] server up (-np 1) after 19s
  PASS  np=1 greedy determinism 12/12 byte-identical (sha 2e9a5fc373d1)
[19:08:54] === 2/6  coherence ===
  PASS  coherence:  Paris. The capital city of Germany is Berlin. The capital c
[19:08:56] === 3/6  chat completions (/v1/chat/completions, production template) ===
  PASS  chat completions: production template renders correctly, 4/4 byte-identical (Paris)
[19:08:58] === 4/6  needle recall, 4 runs per prompt ===
  PASS  needle3121 recalled and sha-stable 4/4 (2e9a5fc373d1)
  PASS  needle20801 recalled and sha-stable 4/4 (75e139bac02a)
[19:11:24] === logit reproducibility, np=1, 6 runs ===
  PASS  logit reproducibility (np=1) 6/6 identical (ceff2af3d586)
[19:11:48] === 5/6  greedy determinism, np=2, other slot erased first ===
[19:12:04] server up (-np 2) after 15s
  PASS  np=2 slot 1 greedy determinism 12/12 byte-identical (sha 2e9a5fc373d1)
  PASS  np=2 slot 1 matches the np=1 reference at the same KV placement (2e9a5fc373d1)
[19:13:45] === logit reproducibility, np=2 slot 1, 6 runs ===
  PASS  logit reproducibility (np=2 slot 1) 6/6 identical (ceff2af3d586)
  PASS  token-0 logit match: np=2 slot 1 == np=1 reference, identical to the last digit
[19:14:07] === 6/6  unit tests ===
  PASS  test-pxq-cpu-dot
  PASS  test-kv-seq-shadow
  PASS  test-narrow-kernel-parity
[19:14:12] === GATE RESULT: PASS=13 FAIL=0 SKIP=0  (logs in /mnt/user/PXACLAW/speed-campaign/release/logs/workdir-volta) ===
