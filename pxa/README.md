# pxa/ — engine developer notes

Short, hard-won rules that are not obvious from the code and that cost a measurement window
each time they were relearned. Add to this file when a bug turns out to be a rule.

---

## A graph node you read back host-side must be a graph OUTPUT — and must not be a view

ggml-alloc's contract (`ggml/include/ggml-alloc.h`):

    ggml_set_input():  all input tensors are allocated at the beginning of the graph in
                       non-overlapping addresses
    ggml_set_output(): output tensors are never freed and never overwritten

Any other node is an ordinary intermediate. The moment its last consumer has run, its block
goes back on the allocator's free list and a later node of the *same* graph can take it. The
graph is still correct — everything downstream was computed before the reuse — so the only
thing that breaks is a host-side read of that node *after* `ggml_backend_graph_compute()`
returns. Which is exactly what `llama_decode()` does for logits and embeddings.

Two rules follow, and both have been paid for:

1. **Flag it.** If any code outside the graph reads a node back — `llama_get_logits_ith()`,
   `llama_get_embeddings_ith()`, a hand-rolled `ggml_backend_tensor_get()` — that node needs
   `ggml_set_output()`. Naming it (`cb(cur, "result_norm", -1)`) is *not* enough: the name is
   how `llama_decode()` finds the node, the flag is what keeps its data alive.

2. **Materialise it first.** The flag protects the *tensor*, not the memory a view points into.
   A `ggml_reshape_*` / `ggml_view_*` node flagged as an output still reads through to a
   source block the allocator is free to reuse across scheduler splits. `ggml_cont()` first,
   then flag the result. `build_qwen4exp.cpp` (`build_qwen4exp_ple`, `build_qwen4exp_mtp`)
   carries the original note; that is where this was first learned.

### The case that made it a rule

Every grafted NextN (MTP) head in this tree ends:

    cur = <the head block's FFN output>
    cb(cur, "result_norm", -1);                       // <-- read back as the NEXT draft step's
    cur = build_output(..., shared_head_norm, ...);   //     conditioning hidden state
    cb(cur, "result_output", -1);

`common/speculative.cpp`'s `mtp_accept_batch()` reads that row back after the accepted-token
commit decode and hands it to the following `MTP_OP_DRAFT_GEN` decode as `prev_embeddings`.
It was never flagged. On a multi-row commit batch the allocator handed its block to the
`ggml_mul` inside the head's own output norm, so the row read back was a different node's data.

The symptom is what made it expensive to find: **the logits were perfect and only the hidden
was wrong**, because the logits are computed before the reuse. On the 27B that read as a free
carried draft token accepted 95.4 % of the time whose conditioning hidden was orthogonal to
the true row — so depth 1 worked and every deeper chain collapsed to noise (0.045 top-1 at
depth 2), and two measurement windows went looking for a position bug, a row-map bug and a
pre-norm/post-norm mismatch that were all fine. One `ggml_set_output()` per head graph fixed
it, bit-for-bit.

Regression test: `tests/test-mtp-head-output.cpp` (no model, no GPU — it builds the head's
tail at its real shape and allocates it through `ggml_gallocr` the way `llama_decode()` does).
It asserts the flagged case is bit-identical AND that the unflagged case still reproduces the
reuse, so the test cannot quietly lose its teeth. `PXA_MTP_HEAD_OUTPUT=0` reproduces the
defect at runtime for an A/B on one binary; it is never a shipping value.

### How to check a node you are adding

* does anything call `llama_get_*_ith()` or `ggml_backend_tensor_get()` on it after the
  decode? → `ggml_set_output()`
* is it a reshape/view/permute of another node? → `ggml_cont()` first
* is it only consumed inside the graph? → leave it alone; flagging it costs the allocator a
  block it could have reused
