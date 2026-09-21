# Benchmark your card

**We can only auto-tune the cards somebody has measured.** The engine fills in `-b`/`-ub` and
arms its levers from a per-topology table; every row in it came from a real run. If your card set
is not in that table, the engine falls back to an adaptive-VRAM ladder — safe, but leaving
performance on the floor. A ten-minute report from you fixes that for everyone with your card.

**Cards we do not have and would like reports from:** Tesla **P40**, **P4**, **GP100**,
**Titan V**, **V100 32 GB** (SXM2 or PCIe), **Titan Xp**, GTX **1070** / **1080**, and the Quadro
P-series. Anything Pascal (`sm_60` / `sm_61`) or Volta (`sm_70`) is in scope. Reports from card
sets already in the table are still useful — a second data point on different silicon of the same
model is how a row stops being one person's box.

---

## Three commands

### 1. Untar

Grab the tarball from [the latest release](https://github.com/poisonxa16/pxa/releases) — no
Docker, no toolchain, no `pip install`. It needs glibc ≥ 2.34 (the floor printed in the tarball's `VERSION`) and an NVIDIA driver new enough for
CUDA 12.8.

```bash
tar xzf pxa-*-linux-x86_64-cuda12.8-sm60_61_70.tar.gz
cd pxa-*/
cat START-HERE.md          # requirements, the model file and its sha256, three steps
```

### 2. Serve, with no flags

This is the important part: **pass nothing you do not have to.** The whole point of the report is
what the engine picks *on its own* for your card.

```bash
./run-server.sh -m /path/to/your-model.gguf -ngl 99 -c 8192
```

Copy the startup banner — the `version:` line, the `PXA config level:` line and the whole
`PXA_AUTO:` block. That block is the engine telling you what it decided and why, and it is the
half of the report that matters most.

### 3. Measure

Either of these is fine; the gate is better if you have the time.

**The release gate** (determinism + coherence + needle recall + unit tests, exits 0 only if
everything passed):

```bash
MODEL=/path/to/your-model.gguf ./bench/gate/run-gate.sh
```

**Or a plain throughput loop**, if you just want the numbers — three runs, take the median:

```bash
for i in 1 2 3; do
  curl -s http://localhost:8080/completion \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a detailed technical summary of how a GPU executes a matrix multiply.",
         "n_predict":192,"temperature":0,"top_k":1,"cache_prompt":false}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin)["timings"]; print("prefill %.1f t/s   decode %.1f t/s" % (d["prompt_per_second"], d["predicted_per_second"]))'
done
```

---

## What to paste

Post this in [Discord](https://discord.gg/EqazvV9tf) or open a **benchmark report** issue. Please
include all five parts — a number without its banner and its card line cannot go into the table.

````
### Card
<paste: nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv>

### Banner
<paste: the `version:`, `PXA config level:` and full `PXA_AUTO:` block from the server log>

### Model
<file name, quant tier, and its sha256 if you have it>

### Command
<the exact command line you ran, including any flags you did add>

### Numbers
prefill: X t/s   decode: Y t/s   (median of 3)
gate:    <PASS / FAIL / the summary line from run-gate.sh>
````

**Please report failures too.** A card that will not boot, a gate that fails determinism, or a
number that is *worse* than the fallback is more useful than another green row — the ladder only
gets fixed where somebody shows it is wrong.

## A filled-in example

A first-time reader walked `START-HERE.md` on my own box on 2026-09-09 — card 0, a Tesla P100
16 GB, driver and CUDA as printed below — and this is the report that came back. It is here as a
worked example of the five parts, not as a published number: the published floors live in
`START-HERE.md` and `docs/COOKBOOK.md`, and this row is one box on a build newer than the one
those floors were taken on.

````
### Card
Tesla P100-PCIE-16GB, 16384 MiB, 580.142

### Banner
version: build=5204 commit="d8b0def5"
PXA config level: ENHANCE (default; PXA_ENHANCE=0 for DEFAULT, PXA_REFERENCE=1 for REFERENCE)
PXA level=ENHANCE | dev0 Tesla P100-PCIE-16GB(sm_60): FP16_GEMM ON [2:1 hgemm] MASK_SKIP_TILE ON [bit-exact] | mode=balance [fa-on serving] | spec: SPEC_RELAXED ON [G3, spec lanes]
PXA posture: mode=balance [fa-on serving, decode-first] fa=on (explicit) ub=2048 (adaptive: dev0 free-min 16006 MiB, min total 16269 MiB, model share 13492 MiB (uniform split) -> headroom 2513 MiB)
PXA_AUTO: samplers arch=qwen35moe -> temp=0.70 top_k=20 top_p=0.80 min_p=0.00 (qwen no-think agent defaults; override: the CLI flag or PXA_AUTO_SAMPLERS=0)
PXA_AUTO: spec arch=qwen35moe -> --spec-type ngram-mod:n_max=4,n_min=2 (measured +23.0% first-request decode on code traffic ...)
PXA_SPEC_RELAXED: ON (pmin=0.050) -- relaxed draft acceptance is live for temp>0 sampling

### Model
fusion2-35b-U16-q8head.gguf — PXQU-16 with the q8_0 output head,
sha256 643e55b07faf8b9b359fcb050494ec9c56e51c2db8d81751b4120b97782c5779

### Command
./run-server.sh -m fusion2-35b-U16-q8head.gguf -ngl 99 -c 8192 -fa on
(the card was pinned by UUID — CUDA_VISIBLE_DEVICES=GPU-... — because at the time of this run the
scripts did not pin the device order; they do now, so the nvidia-smi index works. --port was moved
off 8080 only because this shared box already had that port bound.)

### Numbers
prefill: 1427.7 t/s @ prompt_n 5792, ub 2048   decode: 71.9 t/s   (median of 7, one warmup discarded, unique prompt and seed per repeat, temp 0, cache_prompt false)
first token: 4054 ms for a 5792-token prompt
gate: not run — this was a throughput pass only. Both numbers clear their published floors (>= 62
t/s decode, >= 817 t/s prefill) by a wide margin, with no drafting detected (no draft_n in the
timings) to account for the decode figure.
````

Two things worth copying from it: the prefill number is quoted **with** its `prompt_n` and `ub`,
which is what makes it comparable to anything, and the report says plainly which parts were not
run instead of leaving them blank.

## What happens to your report

A confirmed card set gets its own row in the per-topology table, so the next person with your card
gets the right batch geometry with no flags. Rows carry the source they came from, so your report
is credited in the table rather than absorbed. If a number cannot be reproduced, it is left out
rather than averaged in — the honesty rules that apply to our own numbers apply to reports too.
