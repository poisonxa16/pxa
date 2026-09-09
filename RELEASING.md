# Releasing

A tag on this repository is a claim: this build is deterministic, coherent, and correct at the
API surface a real client actually uses. I don't tag anything I haven't run the release gate
against. This is the checklist I follow every time.

The gate itself is [`bench/gate/run-gate.sh`](bench/gate/README.md) — one script, no container
assumptions, exits 0 only if every check passed. Read the gate's own README for what each check
proves; this file is only about *when* I run it and what I have to show for it before a tag goes
out.

## 1. The CPU arms run themselves

Every push to `main`, and every tag, the CPU-runnable arms run automatically
(`.github/workflows/gate.yml`): greedy determinism at np=1 and np=2, the token-0 logit-match
between them, coherence, the chat-completions arm through the production template, needle recall,
and the two CPU unit tests. They run `GATE_STRICT=1`, so a skip that isn't genuine hardware
absence fails the run instead of quietly passing it. If that workflow is red, I don't move on to
step 2 — there is no GPU arm that fixes a build that fails on CPU.

## 2. The GPU arms are mine to run, by hand

GitHub's own runners have no GPU, so the arms that need one — the full `REPS=12`/`NEEDLE_REPS=4`
pass at real context length, on the real release model, on the actual cards I ship the binary
for — are self-hosted. I run them myself before every tag:

```
MODEL=/path/to/the/release/model.gguf \
GPUS=<the card indices this build targets> \
GATE_STRICT=1 \
./bench/gate/run-gate.sh 2>&1 | tee bench/gate/LAST-RUN.md
```

`GATE_STRICT=1` here too — the same fail-closed rule applies whether the box is a CI runner or
mine. If I need a faster pass while iterating on a fix, I drop `GATE_STRICT` and lower `REPS` or
`LOGIT_REPS` locally, but the run I tag against always uses the defaults, strict, in full.

I run this once per card family the tag claims support for (a Pascal box and a Volta box are two
separate runs, two separate logs, if the tag claims both) and keep every log — `LAST-RUN.md` is
whichever one I'm about to tag against; the rest live wherever I keep prior release evidence, not
in this file.

## 3. The log is part of the tag, not a side note

`bench/gate/LAST-RUN.md` — the captured output of the run in step 2, `GATE RESULT` line and all —
gets committed alongside the tag. **My tag message must cite it**: name the file, quote the
`GATE RESULT: PASS=… FAIL=… SKIP=…` line, and if `SKIP` is anything but 0, say what skipped and
why in the same message. A tag message that doesn't cite the gate log is not a finished tag,
whatever the CI checkmark says — the required CI check only ever proves the CPU arms; the GPU
arms exist solely in that log and in my word that I ran them.

```
git tag -a v2026.MM.DD -m "$(cat <<'EOF'
<one line: what this tag is>

Gate: bench/gate/LAST-RUN.md, run on <card family>, GATE RESULT: PASS=N FAIL=0 SKIP=0.
<if SKIP>0: name each skipped arm and why here.>
EOF
)"
```

## 4. What each side actually proves

The CPU workflow proves the engine is deterministic and correct on the host CPU path, and that
the OpenAI-compatible chat route works end to end through the model's own template — a real bar,
and the one every automated check on this repository can hold a tag to. It does **not** prove the
CUDA kernels are correct, and it cannot: GitHub's runners have no GPU, so every GPU-path check
(`NGL>0`, the fused kernels, real card topology) is a `SKIP` there by design, not a `FAIL` — see
`bench/gate/README.md`'s SKIP-is-fail-closed rule for why a skip for a genuinely absent GPU is the
correct, honest outcome on that runner. The GPU pass in step 2 is what actually proves the thing a
release is claiming to ship. Neither one is optional; they prove different halves of the same
claim.
