# SWE-bench Lite — 19.6% (11/56), graded by the OFFICIAL docker harness

The claim, precisely: on the first 56 instances of **SWE-bench Lite**, the PXQ4 flagship (formerly PXQ6)
driving the minimal ReAct harness in `run_swe.py` produced patches that the **official**
`swebench.harness.run_evaluation` (prebuilt per-instance docker images, FAIL_TO_PASS +
PASS_TO_PASS) graded as **11 resolved = 19.6%**. Of the 38 instances where a non-empty patch
was submitted, 28.9% resolved; 18 hit the turn limit with no patch. Zero harness errors.

Why this is stated so carefully: self-graded agent benchmarks are the community's #1 fraud
tell. Ours is not self-graded — the grader is SWE-bench's own docker evaluation, unmodified.

## The harness (deliberately minimal)

`run_swe.py` is a single-file ReAct tool loop: `list_dir` / `read_file` / `grep` /
`submit_patch`, over a **real shallow git checkout** at each issue's base commit. No RAG, no
repo map, no multi-agent scaffold, no SWE fine-tuning. The point is to measure the model, not
the scaffold.

## Reproduce

```bash
# 1) serve the flagship (the PXQ6/PXQ6-MTP files = the PXQ4 tier; see ../speed-bench.sh for flags)
# 2) generate predictions (resumable; appends to results.jsonl):
python3 run_swe.py --limit 56          # our published run; --limit 300 for the full Lite set
# 3) grade with the OFFICIAL harness (this is the number that counts):
pip install swebench
python3 -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path predictions.jsonl \
  --max_workers 4 --run_id pxq-repro
```

The first resolved instance in our run was `django__django-10914` — a good smoke target:
`python3 run_swe.py --instance django__django-10914`.

Scope note: ~3B active params, a bare-bones loop, salvaged single-GPU hardware. Compare
against other ~3B-active or similar-scaffold numbers, not against frontier agent stacks.
