[19:16:58] gate starting
[19:16:58]   model    /mnt/cachetwo/models/qwen38-27b-stock/Qwen3.8-27B-PXQ4.gguf
[19:16:58]   binaries /mnt/cacheone/rel-gate/pxa-v2026.09.13-rc3/bin
[19:16:58]   devices  0,1,5,6
[19:16:58]   logs     /mnt/user/PXACLAW/speed-campaign/release/logs/workdir-pascal
[19:16:58] === 1/6  greedy determinism, np=1, 12 runs ===
[19:17:13] server up (-np 1) after 15s
  PASS  np=1 greedy determinism 12/12 byte-identical (sha 8b3a29d52309)
[19:21:55] === 2/6  coherence ===
  PASS  coherence:  Paris. The capital city of Germany is Berlin. The capital c
[19:21:58] === 3/6  chat completions (/v1/chat/completions, production template) ===
  PASS  chat completions: production template renders correctly, 4/4 byte-identical (Paris)
[19:22:02] === 4/6  needle recall, 4 runs per prompt ===
  PASS  needle3121 recalled and sha-stable 4/4 (8b3a29d52309)
  PASS  needle20801 recalled and sha-stable 4/4 (425279ff8911)
[19:31:23] === logit reproducibility, np=1, 6 runs ===
  PASS  logit reproducibility (np=1) 6/6 identical (5dabd06c2a90)
[19:32:50] === 5/6  greedy determinism, np=2, other slot erased first ===
[19:33:06] server up (-np 2) after 15s
  PASS  np=2 slot 1 greedy determinism 12/12 byte-identical (sha 8b3a29d52309)
  PASS  np=2 slot 1 matches the np=1 reference at the same KV placement (8b3a29d52309)
[19:37:52] === logit reproducibility, np=2 slot 1, 6 runs ===
  PASS  logit reproducibility (np=2 slot 1) 6/6 identical (5dabd06c2a90)
  PASS  token-0 logit match: np=2 slot 1 == np=1 reference, identical to the last digit
[19:39:16] === 6/6  unit tests ===
  PASS  test-pxq-cpu-dot
  PASS  test-kv-seq-shadow
  PASS  test-narrow-kernel-parity
[19:39:22] === GATE RESULT: PASS=13 FAIL=0 SKIP=0  (logs in /mnt/user/PXACLAW/speed-campaign/release/logs/workdir-pascal) ===
