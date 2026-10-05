# Gemma-4 26B-A4B MTP verify step: CUDA-graph capture/replay (ticket gemma-mtp-verify-kernel)

Grok Bot, 2026-10-02, branch grokbot/gemma-verify-graph (from grokbot/gemma-verify-2row 6228ea7507).
This branch has no code change. It records why a new PXA_MTP_VERIFY_GRAPH lever was not added.

## Why graphs never replay for the verify graph by default (single V100)
1. Arch gate in ggml_backend_cuda_graph_compute: min cc is CC_AMPERE unless PXA_CUDA_GRAPH_DECODE (sm_70)
   or PXA_CUDA_GRAPH_PREFILL (sm_60) is set, so every graph on V100 runs eager.
2. check_node_graph_compatibility: at ny>=2, MOE_FUSED_UP_GATE needs PXA_CUDA_GRAPH_BATCH, and MUL_MAT_ID
   needs PXA_CUDA_GRAPH_MOE.
With the existing levers PXA_CUDA_GRAPH_DECODE=1 PXA_CUDA_GRAPH_BATCH=1 PXA_CUDA_GRAPH_MOE=1 (+PXA_MOE_VERIFY2=1),
the w2/w3 verify graph (1437 nodes), the w1 graph and the drafter graph (83 nodes) are all captured and replayed
(about 2000 replays per run, capture_fail=0). There was no crash over 8 requests (the alloc-generation guard
handles the re-reserve). Output sha and acceptance were identical for every request rep.

## Measurement (GPU 4 V100, Q4_0 QAT, c16384 ub256, PXA_HOST_TIMING medians)
| step | eager (verify2 on) | graph replay (verify2 on) |
|---|---|---|
| w2 | 12.25 ms = submit 5.0 + sync 6.4 | 12.37 ms = submit 0.56 + sync 11.45 |
| w3 | 14.85 ms = submit 5.0 + sync 9.4 | 15.02 ms = submit 0.66 + sync 13.86 |
| w1 | 8.55 ms | 8.6-9.1 ms (+ recapture stalls) |
Replay removes about 4.5 ms of host submit, but sync grows by the same amount. The roughly 5 ms submit always
overlapped GPU execution, so the step is GPU-bound and graphs cannot shorten it. Capture churn (a new key per
request, alloc-generation recaptures, LRU evictions) costs extra on top. A new lever would only add risk, so none was added.

## Where the w2 GPU time goes (nvprof gpu-trace of llama-server, verify2 on, per target-w2 + drafter cycle)
- dense Q4_0 mmvq ncols=2: 2.39 ms
- MoE pxa_mv2 (gate_up + down): 3.63 ms; the w1 MoE path is about 2.55 ms
- lm_head Q6_K mmvq ncols=2: 1.146 ms; ncols=1 is 0.729 ms, so ncols=2 runs at 1.57x and is not bandwidth-bound
- logits DtoH, pageable at about 3.1 GB/s: 2.0 MB per w2 step = 0.635 ms, plus 1.0 MB per drafter step = 0.32 ms.
  This happens although PXA_VERIFY_ARGMAX already reduces on the GPU.
- FA ext_f16 at 2 cols: 0.67 ms, versus vec at 1 col 0.43 ms
- router through cuBLAS sgemm + splitK: 0.40 ms, versus k_pxa_router_gemv_f32 at 1 col 0.22 ms

## Next targets, ranked by ms per cycle and by risk
1. Skip or shrink the full-logits DtoH when the greedy verify only needs the GPU argmax ids, or use a pinned
   staging buffer. Saves up to about 0.95 ms per cycle (0.64 ms on the target step).
2. Re-tune the Q6_K lm_head mmvq at ncols=2/3 (rows per block / nwarps). The bandwidth floor is about 0.70 ms,
   so about 0.4 ms per w2 step is available.
3. Router gemv for 2-4 cols in place of cuBLAS: about 0.2 ms. FA vec kernel for 2 cols: about 0.25 ms.
