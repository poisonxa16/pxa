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

## What happens to your report

A confirmed card set gets its own row in the per-topology table, so the next person with your card
gets the right batch geometry with no flags. Rows carry the source they came from, so your report
is credited in the table rather than absorbed. If a number cannot be reproduced, it is left out
rather than averaged in — the honesty rules that apply to our own numbers apply to reports too.
