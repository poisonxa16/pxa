# Quality gauntlets — factual 11/12 · agentic-hard 9/10 · coding ~19/24

The three capability sets behind the model-card scorecard, plus the runner. All were run
against the PXQ4 flagship (formerly PXQ6) via the OpenAI-compatible server endpoint, temp 0.6 / top_p 0.95 /
top_k 20, NO-THINK (enable_thinking=false, the mode these models are actually served in), 20k max tokens. Set GAUNTLET_THINKING=1 for a thinking run; never compare across modes.

| file | items | published score | what it probes |
|---|---|---|---|
| `factual_knowledge_gauntlet.json` | 12 | **11/12** | myth-busting factual recall (traps included) |
| `agentic_hard_gauntlet.json` | 10 | **9/10** | multi-step reasoning: bug analysis, ambiguity, conditional logic |
| `coding_challenges.json` | 24 | **~19/24 coherent solutions** | hard coding: subtle-bug hunts, language-lawyer traps, algorithmic tasks |

## Reproduce

```bash
# serve the model first (see ../speed-bench.sh), then:
python3 runner.py 8080 factual_knowledge_gauntlet.json results_factual.jsonl
python3 runner.py 8080 agentic_hard_gauntlet.json     results_agentic.jsonl
python3 runner.py 8080 coding_challenges.json         results_coding.jsonl
```

`runner.py` records the full transcript (content, reasoning, tool_calls, finish reason,
timings) per item into the JSONL.

## Grading honesty

Each item carries `gold` (the expected answer/behavior) and `rubric` (what counts) — grading
is done against those fields by a human reading the transcript. The coding score is written
"~19/24 **coherent solutions**" deliberately: it counts solutions that are correct or
correct-modulo-minor-defect per the rubric; it is not an executed pass@1 harness (SWE-bench in
`../swe/` is the executed, officially-graded coding number — trust that one first).
