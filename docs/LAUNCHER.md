# `pxa-launch` — start here

`tools/pxa-launch.py` is the front door to this engine. **You pick the cards and the
model. It picks everything else** — which runtime, the batch and micro-batch sizes,
the tensor split, the flash-attention regime, the chat template, the environment —
and it shows you the measurement behind every choice before anything starts.

Run it with no arguments the first time:

```bash
python3 tools/pxa-launch.py
```

## Three steps

1. **Pick your cards.** It lists every NVIDIA GPU in the box with its VRAM and
   whether another process is already sitting on it. Tick the ones you want.
2. **Pick your model.** It lists the `.gguf` files it can find, with size, family
   (dense / MoE / hybrid-MoE) and PXQ tier read out of each file's own header, and
   whether it fits the cards you ticked.
3. **Say what it is for** — chat, serving several people at once, or long
   documents. That is the only performance question you are asked, and it is asked
   because on these cards it genuinely changes the answer.

Then it shows you the plan, the evidence and the exact command, and asks before
starting. Nothing is chosen silently and nothing is started without your say-so.

If your models are not where it looked, tell it once:

```bash
python3 tools/pxa-launch.py --models-dir /path/to/models
# or, permanently:
export PXA_MODELS_DIR=/path/to/models:/another/path
```

## The full-screen version, and the plain one

With a terminal at least 80x24 you get a full-screen UI: arrow keys (or `j`/`k`)
to move, space to tick a card, enter to go on, `q` to quit. Screens: **CARDS →
MODEL → CHAT → ENGINE → REVIEW → RUNNING**. From CARDS you can press `m` to pick
the model first and come back — the order is yours.

Anywhere the UI cannot run — no terminal, a terminal under 80x24, `curses`
missing, `--no-tui`, or `PXA_NO_TUI=1` — the same questions are asked as plain
line prompts. Nothing is only available in one of the two. A real session, taken
from this box:

```text
==============================================================================
pxa-launch - pick your cards and your model. The launcher does the rest.
            Everything it chooses is printed with the measurement behind it
            before anything starts. Ctrl-C is always safe here.
==============================================================================
STEP 1 of 4 - which card(s) do you want to use?

   #   gpu  name                            VRAM        in use by another process?
   1     0  Tesla P100-PCIE-16GB             16.0 GiB  sm_60   YES - pid 27968 llama-server (8398 MiB)
   2     1  Tesla P100-PCIE-16GB             16.0 GiB  sm_60   free
   3     2  Tesla V100-PCIE-16GB             16.0 GiB  sm_70   free
   4     3  GeForce GTX 1080 Ti              11.0 GiB  sm_61   YES - pid 2751094 llama-server (478 MiB)
   5     4  Tesla V100-PCIE-16GB             16.0 GiB  sm_70   free
   6     5  Tesla P100-PCIE-16GB             16.0 GiB  sm_60   free
   7     6  Tesla P100-PCIE-16GB             16.0 GiB  sm_60   free

  Type the numbers from the # column, separated by commas.
  Two cards of the SAME type is the best-measured shape; four P100s is the
  Flash-Next seat. Press Enter for every card on the box.
  cards> 3,5

STEP 2 of 4 - which model?

   #  file                                          size      family        PXQ tier
   1  Qwable-27B-PXQ4HQ.gguf                         15.3 GiB  hybrid (SSM)  PXQ4-HQ
   2  Qwable-27B-PXQ4core.gguf                       14.6 GiB  hybrid (SSM)  PXQ4
   3  Qwable-27B-MXFP4-lite.gguf                     14.6 GiB  hybrid (SSM)  -

  Searched: /models

  Type a number from the # column, or paste a full path to a file this list missed.
  model> 2

STEP 3 of 4 - what will you use it for?

   1  chat / agent - you type, it answers              -fa on: full decode speed
   2  serve - several people or agents at once         -fa on, --np above 1
   3  long documents - ingest, summarize, embed in bulk -fa off: much faster cold prefill, much slower decode

  This picks the flash-attention regime, and it is a real fork on these cards:
  FA is a decode win and a cold-prefill loss (docs/COOKBOOK.md 'Two FA regimes').
  use [1]> 1

STEP 4 of 4 - chat template, jinja, reasoning, sampling

  chat template in the file: 8057 chars, looks like 'chatml'
    | {%- set image_count = namespace(value=0) %}
    | {%- set video_count = namespace(value=0) %}
    | {%- macro render_content(content, do_vision_count, is_system_content=false) %}
    |     {%- if content is string %}
    |         {{- content }}
    |     {%- elif content is iterable and content is not mapping %}
    | ... 152 more lines (8057 chars total)

  setting              value                                        source
  chat template        embedded in the GGUF (8057 chars, looks l... from the file
  --jinja              ON                                           MEASURED
  --reasoning-format   not passed -> engine default 'deepseek'      MEASURED
  --reasoning-budget   not passed (unlimited)                       [INFERRED]
  --temp               1                                            from the file
  --top-p              0.95                                         from the file
  --top-k              20                                           from the file
  --api-key            none - the port is OPEN                      UNMEASURED
  --slot-save-path     not set                                      [INFERRED]
  --host / --port      0.0.0.0:8080                                 default

  These are the settings people forget. Press Enter to take them as they are,
  or type the letter of one to change it:
    t  chat template      j  jinja on/off        r  reasoning format
    b  reasoning budget   s  sampling            k  API key / host / port
  edit [Enter = accept]>
```

...and then the same plan, evidence and command the full-screen REVIEW screen
shows, printed to the terminal.

For scripts, answer everything on the command line and no question is asked at
all:

```bash
python3 tools/pxa-launch.py --gpus 2,4 --model /models/Qwable-27B-PXQ4core.gguf --yes
python3 tools/pxa-launch.py --gpus 2,4 --model /models/Qwable-27B-PXQ4core.gguf --explain   # decide, print, run nothing
```

## Design rule: never magic

The launcher is built around four commitments, and they explain most of its
behaviour:

1. **It prints the decision, the evidence, and the command** before executing.
2. **It refuses rather than silently dropping** a parameter that does not
   translate between engines.
3. **It says `UNMEASURED` out loud** instead of guessing quietly.
4. **It makes no claim it cannot back**, including no health claim about the instance
   it starts — it `exec`s the server, so it cannot observe anything afterwards.

A launcher that quietly picks differently turns every performance question into a
debugging exercise about the launcher.

Two runtimes serve one quant family. pxa runs on every card the PXQ family
supports. vllm-pxq4 runs wherever its image has PXQ4 kernels for the card, and
brings real data parallelism that llama.cpp's `-sm layer` does not have. Choosing
by hand means remembering which card is which, whether the model is dense or MoE,
which PXQ tier is actually inside the file, and which engine wins at the
concurrency you actually serve at. That is what this script is for.

---

## What it picks, per topology

This is the whole table. Every row was booted and timed; the source column is the
document and line the numbers were copied from. A card shape that is not in this
table gets `[INFERRED]` flags from the nearest row (its `-b`/`-ub` only, never its
model-specific flags) or the word `UNMEASURED` and a request to confirm — it never
gets a number that was not measured for it.

`--selftest` prints this same table from the code, so it cannot drift from what
the launcher actually does.

| cards | model | flags the launcher passes | measured result | source |
|---|---|---|---|---|
| 2× V100 (sm_70) | dense 27B, PXQ4 | `-b 8192 -ub 2048 -fa on -c 32768 -sm layer` | prefill **1,369** t/s @3,121 · **1,300** @20,801 · decode **39.5** @fill 8 (release binary, quiet box, no env) | `RELEASE-NOTES-2026-09-07.md:77`; the measurement ledger |
| 2× P100 (sm_60) | dense 27B, PXQ4 | `-b 8192 -ub 256 -fa on -c 32768 -sm layer` | prefill **337.6** t/s @3,121 · **315.3** @20,801 · decode **18.1** @fill 8 (release binary, quiet box, no env; fold n=7 reference 340.16 / 316.84 / 17.83) | `RELEASE-NOTES-2026-09-07.md:78`; bench/fair-battle.md:303; P100 final-binary capture, 2026-09-05 |
| 1× GTX 1080 Ti (sm_61, 11 GB) | 35B MoE, PXQ2 | `-b 2048 -ub 768 -c 8192 --ctx-checkpoints 0`, `-fa on` for chat / `-fa off` for long documents, `PXA_AUTO_SPEC=0` | cold prefill **1,363.5** t/s (fa off) · chat prefill **746.6** (fa on) · decode **36.73** cold / **65.3** chat (release binary, no env) | `RELEASE-NOTES-2026-09-07.md:79`; the measurement ledger |
| 4× P100 (sm_60) | Qwen3.8 Flash-Next hybrid MoE @150k | `-c 150016 -np 2 --kv-unified -t 16 -wgt 8 -ts 5079,12612,12612,11897 -ot per_layer_token_embd\.weight=CPU --no-context-shift` — **no `PXA_*` levers and no `-b`/`-ub`**: the engine's `4x sm_60` row picks `-b 2048 -ub 2048` and ENHANCE sets all eleven published levers itself | prefill **487.6** t/s @3,121 · **376.7** @20,801 · decode **24.57** at low fill (release binary, auto, no env; n=3, worst half-spread 1.55%) | `RELEASE-NOTES-2026-09-07.md`, "Config default"; four-card automatic-defaults probe, 2026-09-06 |
| 1× P100 (sm_60) | 35B MoE, PXQU-16 + q8_0 head | `-b 2048 -ub 2048 -c 8192 -fa on` | decode 62.4 t/s · prefill 827-843 t/s | `docs/COOKBOOK.md:65-73` |
| 1× V100 (sm_70) | 35B MoE, PXQU-16 + q8_0 head | `-b 2048 -ub 2048 -c 8192 -fa on` | decode ~101-102 t/s · prefill ~1,800-1,900 t/s | `docs/COOKBOOK.md:75-78` |
| 2× P100 (sm_60) | 35B MoE flagship PXQ4/PXQ6 | `-b 8192 -ub 2048 -c 8192 -ts 1,1 -fa on` | decode 55.7 t/s · prefill ~843 t/s | `docs/COOKBOOK.md:80-89` |
| 2× V100 (sm_70) | 35B MoE flagship PXQ4/PXQ6 | same command | `[INFERRED]` — the published pair row is the P100 one; the V100 pair's own number was never taken | `docs/COOKBOOK.md:80-89` |
| any single card | a stock (non-PXQ) GGUF | `-b 2048 -ub 2048 -c 8192 -fa on`, `-ub` lowered on an 11 GB card | `[INFERRED]` — engine-only same-quant decode control: V100 +3.2% (bit-identical), P100 +2.7%, 1080 Ti +3.3% | `docs/COOKBOOK.md:149-169` |

`PXA_ENHANCE=1` is exported on every row, explicitly, even though the engine is
making it the default level. A launcher that leans on an engine default cannot be
read: the same printed command would mean two different things on two builds, and
someone who copies it onto an older binary would get a different seat with no
warning.

### Two things the cookbook recipe does not say, and the 1080 Ti needs

Both were found by booting the published recipe as written, on the card it was
written for, and both stop the seat rather than slow it down:

- **`PXA_AUTO_SPEC=0`.** `PXA_ENHANCE=1` makes the server auto-arm
  `--spec-type ngram-mod:n_max=4,n_min=2` for the `qwen35moe` family (the
  PXA AUTO-SPEC block in `examples/server/server.cpp`). That drafter's context
  asks for a 254 MiB per-step checkpoint buffer and falls back to a 62.8 MiB
  shadow. On 11 GiB, with 9,907 MiB of weights, 160 MiB of KV and a 733 MiB
  compute buffer, there is no room: the seat loads, answers short prompts, and
  then dies mid-prefill on a 5.8k-token prompt with `out of memory`. Reproduced
  twice, 2026-09-04. `PXA_ENHANCE=1` still has to be exported — it is what arms
  the sm_61 int8 prefill tile the published numbers depend on — so the launcher
  keeps `ENHANCE` and turns off only the auto-drafter.
- **`--ctx-checkpoints 0`.** Every published 1080 Ti arm passes it (4 of 4 arm
  files, both FA regimes). The cookbook recipe omits it.

### The two FA regimes

On P100, V100 and 1080 Ti, flash-attention is a **decode win and a cold-prefill
loss** — for this engine and for upstream ik alike. One server carries one
setting, so the launcher asks which one you want:

| workload | `-fa` | what you get |
|---|---|---|
| chat, serve | `on` | full decode speed and a solid prefill in the same server |
| longdoc | `off` | 26-56% more cold prefill, 16-48% less decode |

Measured per card in `docs/COOKBOOK.md:41-63` and `bench/fair-battle.md:35-50`.
On the 1080 Ti both cells were measured on the card itself, so that row's regime
switch is `MEASURED` rather than inferred.

---

## Things people forget

The launcher asks about these because the server starts happily without them and
only misbehaves later. Each row shows a value **and where the value came from**;
none of them is a launcher preference invented on your behalf.

| setting | default | why it matters |
|---|---|---|
| **chat template** | the one embedded in the GGUF (`tokenizer.chat_template`), shown to you with its detected family and first lines | it is what the model was trained to see. Overriding it is how a model starts answering the wrong question in a slightly wrong format. If the file has none, the launcher says so loudly — published PXA files have shipped in that state before. |
| **tool-call block** | checked, warned about | a template with no `tool_calls` block serves chat fine and never returns a tool call. If the seat is for an agent, that is the whole job missing. |
| **`--jinja`** | ON, and the launcher says whether this template *requires* it | without it, a request carrying `tools` returns HTTP 500 while plain chat keeps working. A production seat on this box ran that way for weeks and looked healthy. |
| **`--reasoning-format`** | unset → the engine's own `deepseek` (`common/common.h:510`), which is what the live seat inherits | decides whether thinking-tag contents come back as `message.reasoning_content` or inline. |
| **`--reasoning-budget`** | unset (unlimited) | `0` switches thinking off for a latency-sensitive seat; a token count caps it (`common/reasoning-budget.cpp`). |
| **context size `-c`** | the recipe row's value | the launcher prints the KV arithmetic and warns when the fit is tight. It blocks only on facts that need no formula. |
| **slots `-np`** | the recipe row's value (2 on the Flash-Next seat, 1 elsewhere) | more slots means more KV, and on the hybrid it also means `--kv-unified`. |
| **`-fa` regime** | from `--workload` | see above; the one performance question you answer yourself. |
| **sampling** (`--temp`, `--top-p`, `--top-k`) | the model's own `general.sampling.*` keys when the GGUF carries them, otherwise not passed at all | these are written into the file by the quantizer. They are the model author's defaults, not ours. |
| **`--api-key` / `--host` / `--port`** | no key, `0.0.0.0:8080` | with no key on `0.0.0.0`, anyone who can reach the port can use the model. The launcher says so every time. A key is never echoed and never written into a saved script. |
| **`--mmproj`** | auto-detected beside the model; refuses to guess between several | a vision model without a projector serves text only and never says why. |
| **`--slot-save-path`** | off | needed for `/slots` save+restore, which the release gate uses to replay a fill without re-prefilling. |

---

## Saving a seat you can bring back

```bash
python3 tools/pxa-launch.py --gpus 2,4 --model /models/Qwable-27B-PXQ4core.gguf \
    --serve-name v100-chat --yes
```

writes `~/.cache/pxa-launch/serve/v100-chat.sh` (override with `--serve-dir` or
`PXA_SERVE_DIR`): an executable shell script holding the exact command, the exact
environment, and the same busy-card check the launcher does. No systemd, no
daemon, no hidden state.

It deliberately does **not** call the launcher again — a launcher is free to decide
differently tomorrow, and a restart script that quietly changes the seat is worse
than no script. Run `pxa-launch` again when you want a fresh decision. An API key
is never written into it; the script reads `PXA_API_KEY` from the environment
instead.

---

## The screens

A real session, captured from a terminal, on this box. Two P100s, a 27B PXQ4
model, chat workload.

**1. CARDS.** Space ticks a card. The busy column is read from `nvidia-smi --query-compute-apps`, so a card another process is sitting on says so by name. The line at the bottom tells you, as you tick, whether the shape you are building has a measured row.

```text
 PXA launcher   1/5  CARDS - tick the cards this model should use

    #  gpu  name                          VRAM used/total    in use by another process?
   [x] 0    Tesla P100-PCIE-16GB              5/16384  MiB sm_60  free
   [x] 1    Tesla P100-PCIE-16GB              5/16384  MiB sm_60  free

  MEASURED combinations on this class of box:
    2x sm_70    2x Tesla V100 (sm_70), dense 27B, PXQ4
    2x sm_60    2x Tesla P100 (sm_60), dense 27B, PXQ4
    1x sm_61    1x GTX 1080 Ti 11 GB (sm_61), 35B MoE, PXQ2
    4x sm_60    4x Tesla P100 (sm_60), Qwen3.8 Flash-Next hybrid MoE, 150k ctx
    1x sm_60    1x Tesla P100 16 GB (sm_60), 35B MoE, PXQU-16 + q8_0 head
    1x sm_70    1x Tesla V100 16 GB (sm_70), 35B MoE, PXQU-16 + q8_0 head
    2x sm_60    2x P100 or 2x V100, 35B MoE flagship PXQ4/PXQ6 (18.7 GB)

  selection: 0, 1   -> MEASURED for this shape: dense|hybrid PXQ4; hybrid-moe|moe PXQ4|PXQ4-HQ|PXQ6

 up/down (or j/k) move   space tick   a all   n none   m model first   enter next   q quit
```

**2. MODEL.** Everything in this list was read from the file's own GGUF header: family, PXQ tier, trained context, whether it carries a per-layer-embedding table or vision tensors. `fits` is the weights-only check against the cards you ticked. `d` points the search somewhere else.

```text
 PXA launcher   2/5  MODEL - pick the file to serve

    file                                     size       family        tier      fits
    Qwable-27B-PXQ4HQ.gguf                    15.3 GiB  hybrid (SSM)  PXQ4-HQ   yes
    Qwable-27B-PXQ4core.gguf                  14.6 GiB  hybrid (SSM)  PXQ4      yes
    Qwable-27B-MXFP4-lite.gguf                14.6 GiB  hybrid (SSM)  -         yes

  /models/Qwable-27B-PXQ4HQ.gguf
  arch qwen35   trained ctx 262144

 up/down (or j/k) move   enter pick   d change directory   c back to cards   q quit
```

**3. CHAT.** The template that is actually inside the file, its detected family, and its first lines - then every setting people forget, each with where the value came from. `from the file` means the model author put it there. `MEASURED` means production runs it. Nothing here was invented by the launcher.

```text
 PXA launcher   3/5  CHAT - template, jinja, reasoning, sampling

  template inside this file: 8057 chars, looks like 'chatml'
    | {%- set image_count = namespace(value=0) %}
    | {%- set video_count = namespace(value=0) %}
    | {%- macro render_content(content, do_vision_count, is_system_content=false) %}
    |     {%- if content is string %}
    |         {{- content }}
    | ... 153 more lines (8057 chars total)
  setting              value                                    source
  chat template        embedded in the GGUF (8057 chars, looks  from the file
  --jinja              ON                                       MEASURED
  --reasoning-format   not passed -> engine default 'deepseek'  MEASURED
  --reasoning-budget   not passed (unlimited)                   [INFERRED]
  --temp               1                                        from the file
  --top-p              0.95                                     from the file
  --top-k              20                                       from the file
  --api-key            none - the port is OPEN                  UNMEASURED
  --slot-save-path     not set                                  [INFERRED]
  --host / --port      127.0.0.1:8405                           YOURS

 t template   j jinja   r reasoning   b budget   s sampling   k key/host/port   enter next   c back   q quit
```

**4. ENGINE.** The recommended runtime carries the reason from the decision table; the other one is greyed with the blocker that rules it out. You can still pick the other one when nothing blocks it.

```text
 PXA launcher   4/5  ENGINE - which runtime serves this seat

   > pxa - the llama.cpp engine in this tree  (recommended)
        model is a raw GGUF (Qwable-27B-PXQ4core.gguf); vLLM needs a converted artifact
        (tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: 0:Tesla P100-PCIE-16GB sm_60, 1:Tesla P100-PCIE-16GB sm_60

     vllm-pxq4 - the sm_70 serving sidecar
        cannot be used here: model is a raw GGUF (Qwable-27B-PXQ4core.gguf); vLLM needs a converted artifact
        (tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: 0:Tesla P100-PCIE-16GB sm_60, 1:Tesla P100-PCIE-16GB sm_60

  evidence
    MEASURED topology row '2xp100-dense-pxq4' (2x Tesla P100 (sm_60), dense 27B, PXQ4): prefill 340.16 t/s @3,121
    tok | 316.84 @20,801 | decode 17.83 @fill 8 (n=7, two passes, GPUs 1;5, fold 24ebec4096)
    MEASURED FA regime: -fa on for workload 'chat' - the interactive/serving setting, and what every recipe row
    above was measured at. Switch with --workload longdoc if you are ingesting rather than chatting.
    MEASURED envelope: exactly 2 cards, both sm_60 - the table applies as measured (SCOREBOARD.md:6,
    MOE-CROSSOVER.md:3)

 up/down (or j/k) move   enter accept   c back   q quit
```

**5. REVIEW.** This is the output of the same `plan_and_build()` the command line runs - the UI cannot drift from `--explain`, because it is not a second copy of the logic. Scroll it, edit ctx and slots in place with `e`, save a restart script with `s`, or press enter to start.

```text
 PXA launcher   5/5  REVIEW - the decision, the evidence and the command

  ==============================================================================
  pxa-launch: ENGINE = llama
  model: /models/Qwable-27B-PXQ4core.gguf [gguf, 14.60 GiB]
  class: dense, arch=qwen35 [read from GGUF header + tensor directory]
  tier: PXQ4 (provenance KV says PXQ4) [tensor-type histogram over 866 tensors (the signal the loader dispatches on,
  llama-model-loader.cpp:527)]
  compose:f32x360 PXQ4x325 MXFP4x145 q8_0x35 q6_Kx1
  trained ctx: 262144
  mtp: 4 nextn/mtp tensors; KV nextn_predict_layers=1 [tensor walk for nextn/mtp names]
  deltanet: True [336 ssm_* tensors - the GGUF spelling of the same linear-attention layers. Whether a PURE-SSM
  (non-hybrid) model needs the -sm graph guard is UNMEASURED; guarding is the safe direction]
  serve: np=1, workload=chat, ctx=32768 (=32768/slot), threads=72
  cards: 0:Tesla P100-PCIE-16GB sm_60 16GiB, 1:Tesla P100-PCIE-16GB sm_60 16GiB
  recipe: 2xp100-dense-pxq4 (2x Tesla P100 (sm_60), dense 27B, PXQ4) [MEASURED]
  from the row (you did not set these): -c 32768
  reason: model is a raw GGUF (Qwable-27B-PXQ4core.gguf); vLLM needs a converted artifact
  (tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: 0:Tesla P100-PCIE-16GB sm_60, 1:Tesla P100-PCIE-16GB sm_60
  ev: MEASURED topology row '2xp100-dense-pxq4' (2x Tesla P100 (sm_60), dense 27B, PXQ4): prefill 340.16 t/s @3,121
  tok | 316.84 @20,801 | decode 17.83 @fill 8 (n=7, two passes, GPUs 1;5, fold 24ebec4096)
  [RELEASE-NOTES-2026-09-07.md:78 (P100 headline row, carries this fold reference);
  bench/fair-battle.md:217 (full fold n=7 table); the measurement ledger
  FINAL TABLE]
  ev: MEASURED FA regime: -fa on for workload 'chat' - the interactive/serving setting, and what every recipe row
  above was measured at. Switch with --workload longdoc if you are ingesting rather than chatting.
  [docs/COOKBOOK.md:41-63 (Two FA regimes); bench/fair-battle.md:35-50]
  ev: MEASURED envelope: exactly 2 cards, both sm_60 - the table applies as measured (SCOREBOARD.md:6,
  MOE-CROSSOVER.md:3)
  ** -ub does NOT transfer between pairs: 2,048 is best on the V100 pair and 256 on this one (-ub 256 gives 218.4
  t/s @3,121 against 231 at -ub 2048 on the DEFAULT chunk; at -b 8192 the 256 cell is the 340.16 above).
  the measurement ledger (HONEST numbers, UNIFIED build #2/#3:
  P100 ub256 218.4 vs ub2048 231); docs/COOKBOOK.md:103-106.
  ** -ub 512 at the same -b 8192 measured 323.26 / 310.38 / 17.76 (n=7) - about 5% behind
  (the measurement ledger).
  ** PXA_PIPELINE_PP is ON by ENGINE default for arch 'qwen35' (qwen35/qwen35moe only). This launcher does not set
  or unset it; the seat inherits the engine's default. PXA_PIPELINE_PP is an ENGINE default for the qwen35/qwen35moe
  ctx (from the recipe)   np 1   scroll 1/101

 up/down (or j/k) scroll   e edit ctx/np   s save script   enter LAUNCH   c back   q quit
```

**6. RUNNING.** The server's own log streams in the lower pane. The state line at the top is the post-boot contract this launcher has always printed, performed instead of described: it waits for the listening line, then sends one small raw `/completion` and reports what came back.

```text
 PXA launcher   RUNNING - the server log is below

  state: serving
  first token OK - 4 chars back, prefill 18.5 t/s, decode 23.0 t/s
  cards 0,1   pid 85   running
  ------------------------------------------------------------------------------------------------------------------
  llama_iGQA_PACK: NH=0 (OFF — stock one-block-per-head vec kernel)
  PXA_FA_GQA_QSMEM: OFF (Q staged in shared for NH=4/8)
  PXA_FA_VEC_ILP: ON (D=256 decode V-pass 4-way ILP; PXA_FA_VEC_ILP=0 reverts)
  PXA_PXQ4_2D_SPLIT dev1: FIRING (S=4 panels=160 ny=1 R=10240 K=5120, target 448 blocks)
  PXA_PXQ_MMVQ_FUGSPLIT dev1: OFF (dense PXQ up/gate decode = fused twin) [auto: sm_70+]
  PXA_PXQ_DENSE_GATEUP dev1: FIRING (S=2 panels=272 ny=1 R=17408 K=5120)
  =============================== NCCL main communicator initialized
  INFO [                    init] initializing slots | tid="22502421118976" timestamp=1788493431 n_slots=1 kv_unified
  srv          init: Exclude reasoning tokens when selecting slot based on similarity: start: <think>, end: </think>
  use `--reasoning-tokens none` to disable.
  INFO [                    init] new slot | tid="22502421118976" timestamp=1788493431 id_slot=0 n_ctx_slot=32768
  pxa_reserve_real_graph: reserved the real-path graph at n_tokens = 256, n_kv = 32768 (nodes = 3080)
  prompt cache is enabled, size limit: 8192 MiB
  use `--cache-ram 0` to disable the prompt cache
  init: chat template, example_format: '<|im_start|>system
  You are a helpful assistant<|im_end|>
  <|im_start|>user
  Hello<|im_end|>
  <|im_start|>assistant
  Hi there<|im_end|>
  <|im_start|>user
  How are you?<|im_end|>
  <|im_start|>assistant
  slot print_timing: id  0 | task 0 |
  prompt eval time =     270.83 ms /     5 tokens (   54.17 ms per token,    18.46 tokens per second)
  INFO [ieval time =     174.02 ms /     4 tokens (   43.51 ms per token,    22.99 tokens per second)
  INFO [total time =     444.85 ms /     9 tokens: codec=PXQ4 bpw=4.59 backbone_rev=2 tier=core source=n/a | tid="225
  INFO [== Prolog_server_request] request | tid="22499730591744" timestamp=1788493432 remote_addr="127.0.0.1" remote_
  INFO [   launch_srelease_slots] slot released | tid="22502421118976" timestamp=1788493432 id_slot=0 id_task=0 n_ctx
  INFO [    batch_pendslots_idle] all slots are idle | tid="22502421118976" timestamp=1788493432431 id_slot=0 id_task

 q stop the server and come back   up/down scroll
```

The same screen a few minutes later, serving.

```text
 PXA launcher   RUNNING - the server log is below

  state: serving
  first token OK - 4 chars back, prefill 18.5 t/s, decode 23.0 t/s
  cards 0,1   pid 85   running
  ------------------------------------------------------------------------------------------------------------------
  llama_iGQA_PACK: NH=0 (OFF — stock one-block-per-head vec kernel)
  PXA_FA_GQA_QSMEM: OFF (Q staged in shared for NH=4/8)
  PXA_FA_VEC_ILP: ON (D=256 decode V-pass 4-way ILP; PXA_FA_VEC_ILP=0 reverts)
  PXA_PXQ4_2D_SPLIT dev1: FIRING (S=4 panels=160 ny=1 R=10240 K=5120, target 448 blocks)
  PXA_PXQ_MMVQ_FUGSPLIT dev1: OFF (dense PXQ up/gate decode = fused twin) [auto: sm_70+]
  PXA_PXQ_DENSE_GATEUP dev1: FIRING (S=2 panels=272 ny=1 R=17408 K=5120)
  =============================== NCCL main communicator initialized
  INFO [                    init] initializing slots | tid="22502421118976" timestamp=1788493431 n_slots=1 kv_unified
  srv          init: Exclude reasoning tokens when selecting slot based on similarity: start: <think>, end: </think>
  use `--reasoning-tokens none` to disable.
  INFO [                    init] new slot | tid="22502421118976" timestamp=1788493431 id_slot=0 n_ctx_slot=32768
  pxa_reserve_real_graph: reserved the real-path graph at n_tokens = 256, n_kv = 32768 (nodes = 3080)
  prompt cache is enabled, size limit: 8192 MiB
  use `--cache-ram 0` to disable the prompt cache
  init: chat template, example_format: '<|im_start|>system
  You are a helpful assistant<|im_end|>
  <|im_start|>user
  Hello<|im_end|>
  <|im_start|>assistant
  Hi there<|im_end|>
  <|im_start|>user
  How are you?<|im_end|>
  <|im_start|>assistant
  slot print_timing: id  0 | task 0 |
  prompt eval time =     270.83 ms /     5 tokens (   54.17 ms per token,    18.46 tokens per second)
  INFO [ieval time =     174.02 ms /     4 tokens (   43.51 ms per token,    22.99 tokens per second)
  INFO [total time =     444.85 ms /     9 tokens: codec=PXQ4 bpw=4.59 backbone_rev=2 tier=core source=n/a | tid="225
  INFO [== Prolog_server_request] request | tid="22499730591744" timestamp=1788493432 remote_addr="127.0.0.1" remote_
  INFO [   launch_srelease_slots] slot released | tid="22502421118976" timestamp=1788493432 id_slot=0 id_task=0 n_ctx
  INFO [    batch_pendslots_idle] all slots are idle | tid="22502421118976" timestamp=1788493432431 id_slot=0 id_task

 q stop the server and come back   up/down scroll
```

`q` sends SIGTERM, waits for the process, and returns you to REVIEW with the plan still on screen. Nothing is left running behind your back.

```text
 PXA launcher   5/5  REVIEW - the decision, the evidence and the command

  ==============================================================================
  pxa-launch: ENGINE = llama
  model: /models/Qwable-27B-PXQ4core.gguf [gguf, 14.60 GiB]
  class: dense, arch=qwen35 [read from GGUF header + tensor directory]
  tier: PXQ4 (provenance KV says PXQ4) [tensor-type histogram over 866 tensors (the signal the loader dispatches on,
  llama-model-loader.cpp:527)]
  compose:f32x360 PXQ4x325 MXFP4x145 q8_0x35 q6_Kx1; PXA_FA_VEC_ILP=0 reverts)
  trained ctx: 262144
  mtp: 4 nextn/mtp tensors; KV nextn_predict_layers=1 [tensor walk for nextn/mtp names]]
  deltanet: True [336 ssm_* tensors - the GGUF spelling of the same linear-attention layers. Whether a PURE-SSM
  (non-hybrid) model needs the -sm graph guard is UNMEASURED; guarding is the safe direction]
  serve: np=1, workload=chat, ctx=32768 (=32768/slot), threads=72421118976" timestamp=1788493431 n_slots=1 kv_unified
  cards: 0:Tesla P100-PCIE-16GB sm_60 16GiB, 1:Tesla P100-PCIE-16GB sm_60 16GiBlarity: start: <think>, end: </think>
  recipe: 2xp100-dense-pxq4 (2x Tesla P100 (sm_60), dense 27B, PXQ4) [MEASURED]
  from the row (you did not set these): -c 32768="22502421118976" timestamp=1788493431 id_slot=0 n_ctx_slot=32768
  reason: model is a raw GGUF (Qwable-27B-PXQ4core.gguf); vLLM needs a converted artifactodes = 3080)
  (tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: 0:Tesla P100-PCIE-16GB sm_60, 1:Tesla P100-PCIE-16GB sm_60
  ev: MEASURED topology row '2xp100-dense-pxq4' (2x Tesla P100 (sm_60), dense 27B, PXQ4): prefill 340.16 t/s @3,121
  tok | 316.84 @20,801 | decode 17.83 @fill 8 (n=7, two passes, GPUs 1;5, fold 24ebec4096)
  [RELEASE-NOTES-2026-09-07.md:78 (P100 headline row, carries this fold reference);
  bench/fair-battle.md:217 (full fold n=7 table); the measurement ledger
  FINAL TABLE]
  ev: MEASURED FA regime: -fa on for workload 'chat' - the interactive/serving setting, and what every recipe row
  above was measured at. Switch with --workload longdoc if you are ingesting rather than chatting.
  [docs/COOKBOOK.md:41-63 (Two FA regimes); bench/fair-battle.md:35-50]
  ev: MEASURED envelope: exactly 2 cards, both sm_60 - the table applies as measured (SCOREBOARD.md:6,
  MOE-CROSSOVER.md:3)
  ** -ub does NOT transfer between pairs: 2,048 is best on the V100 pair and 256 on this one (-ub 256 gives 218.4
  t/s @3,121 against 231 at -ub 2048 on the DEFAULT chunk; at -b 8192 the 256 cell is the 340.16 above).
  the measurement ledger (HONEST numbers, UNIFIED build #2/#3:
  P100 ub256 218.4 vs ub2048 231); docs/COOKBOOK.md:103-106.
  ** -ub 512 at the same -b 8192 measured 323.26 / 310.38 / 17.76 (n=7) - about 5% behind
  (the measurement ledger).odec=PXQ4 bpw=4.59 backbone_rev=2 tier=core source=n/a | tid="225
  ** PXA_PIPELINE_PP is ON by ENGINE default for arch 'qwen35' (qwen35/qwen35moe only). This launcher does not set
  or unset it; the seat inherits the engine's default. PXA_PIPELINE_PP is an ENGINE default for the qwen35/qwen35moe
  ctx (from the recipe)   np 1   scroll 1/100
 server stopped
 up/down (or j/k) scroll   e edit ctx/np   s save script   enter LAUNCH   c back   q quit
```

> The blocks above were captured through a minimal terminal emulator so they could be
> pasted here as text. The screens themselves are what a real terminal draws; a few
> lines inside the streamed log pane show that emulator's own overwrite artifacts,
> not the server's output.

### Keys

| key | where | what |
|---|---|---|
| `up`/`down`, `j`/`k` | everywhere | move |
| `space` | CARDS | tick / untick a card |
| `a` / `n` | CARDS | tick all / none |
| `m` | CARDS | pick the model first, then come back |
| `d` | MODEL | search a different directory |
| `c` | anywhere | back one screen |
| `t` `j` `r` `b` `s` `k` | CHAT | template, jinja, reasoning format, budget, sampling, key/host/port |
| `e` | REVIEW | edit context size and slots in place |
| `s` | REVIEW | write a rerunnable restart script |
| `enter` | everywhere | accept and go on; on REVIEW, start the server |
| `q` | everywhere | back out; on RUNNING, stop the server |

The window must be at least 80x24. Below that the UI says so and waits for you to
resize, and `q` drops you to the line prompts.

---

## How a claim is tagged

Exactly three tags, no fourth category. They appear in the source comments *and*
in the printed output, so you can tell a measured branch from a guess without
leaving the file.

| Tag | Meaning |
|---|---|
| `MEASURED` | A number from a correctness-gated boot on the PXA reference bench, carrying the id of the bench row that produced it. |
| `[INFERRED]` | A branch taken from an **adjacent** measurement. Never a new number. |
| `UNMEASURED` | Nothing was measured. The launcher says the word, and then either stops or asks for `--accept-unmeasured`. |

**Bench row ids** are stable labels for individual gated boots, quoted inline
beside every number they back:

- `D1` / `D3` — dense 27B PXQ4, 2× P100 sm_60 (D1 llama.cpp, D3 vLLM)
- `M1` / `M4` / `M7`–`M9` — MoE 35B PXQ4, 2× P100 sm_60
- `np1`, `np4`..`np8` — the MoE crossover sweep, 11 gated boots on one 2× P100 pair

Every boot behind a row was gated on short-prompt correctness **before** its
number was kept. An ungated boot has no row and appears nowhere.

---

## The decision pipeline

The launcher runs seven stages in a fixed order. **What cannot run is settled
before what is fastest** — refusals come first, always.

### 1. Resolve the artifact

`model_kind()` classifies the path before anything else reads it. The taxonomy is
deliberately fine-grained so the error names the real problem:

| kind | meaning |
|---|---|
| `gguf` | a readable GGUF file |
| `gguf_broken` | a GGUF whose header will not parse |
| `vllm_dir` | a PXQ4-converted directory (safetensors + `quantization_config.quant_method`) |
| `hf_dir` | an unquantized or foreign-quantized HF checkpoint |
| `lora_dir` | an adapter directory with no base model |
| `weightless_dir` | `config.json` and no weights |
| `not_a_model_file` | an existing file that is not servable — one safetensors shard, a `.tiers` map, an `.imatrix` |
| `missing` / `not_a_model` | nothing there, or nothing recognisable |

A single shard (`model-00003-of-00014.safetensors`) is detected by name and the
message tells you to pass the directory instead.

### 2. Inspect the GGUF — header only

`gguf_header()` reads the KV block and the tensor directory. **No tensor data is
read and no GPU is touched.** From that walk the launcher derives:

- `general.architecture`, `<arch>.expert_count` (→ dense vs MoE),
  `<arch>.context_length` (the trained context)
- the **tensor-type histogram** — printed as the `compose:` line
- MTP/nextn head presence, by tensor walk
- DeltaNet/linear-attention presence
- vision tensors
- the per-layer embedding table, if present
- KV bytes/token, by arithmetic over header fields

### 3. Detect the PXQ tier — from tensors, not from a KV

**This is the single most important design decision in the file.**

There is no generic `pxa.pxq.tier` key to read. `llama-quantize.cpp` rewrites
*every* PXQ tier to `LLAMA_FTYPE_MOSTLY_MXFP4` (=38) before writing it, so
`general.file_type` cannot identify a tier — verified over a 138-file library:
138/138 report 38 or a K-quant id, **0 yield a tier**. And PXQ1, the one tier this
launcher exists to refuse, writes no provenance KV at all.

So the ground truth is the **per-tensor ggml type histogram**:

| ggml type id | tier |
|---|---|
| 248 | PXQ1 |
| 252 | PXQ4 |
| 253 | PXQ4-HQ |
| 254 | PXQ2 |
| 255 | PXQ3 |
| 256 | PXQ6 |

A file carrying more than one PXQ type is `PXQ_UNIVERSAL` — a mixed per-tensor
tier map. Non-PXQ types that legitimately appear alongside (f32, f16, bf16, q8_0,
q6_K, MXFP4) are backbone carriers and are counted separately.

This mirrors the engine, which detects PXQ1 by tensor type in
`src/llama-model-loader.cpp:527`. The provenance KV **is** read as well, and any
conflict between the two signals is printed and never resolved silently — the
tensor walk wins, because that is what the loader dispatches on.

Verified against real artifacts: 5/5 PXQ GGUFs yield a tier this way; 0/5 through
`general.file_type`.

### 4. Choose the engine

**Structural gates first** — what each engine can physically read:

| Condition | Engine | Why |
|---|---|---|
| tier is PXQ1 | **refuse** | R-01 — loads, passes composition gates, generates incoherent text |
| tier is PXQ4-HQ/PXQ6/UNIVERSAL | llama | vLLM implements PXQ2, PXQ3 and PXQ4 only |
| tier is PXQ2/PXQ3, model is a raw `.gguf` | llama | servable by vLLM, but only after conversion (`--policy m2`); see `docs/VLLM.md` §2 |
| no PXQ tensors at all (stock K-quant, MXFP4, f16) | llama | vLLM's PXQ4 backend has nothing to load |
| model is a raw `.gguf` | llama | vLLM needs a converted artifact |
| no vLLM-eligible card | llama | the failed probe is named |
| only *some* selected cards eligible | llama | vLLM cannot span a mixed-arch selection on a single-arch image, and dropping cards would silently change the parallel degree |
| selection spans >1 compute capability | llama | the vLLM command carries one `--attention-backend` |
| single GPU | llama | no parallelism to gain, and lower single-stream overhead |
| converted vLLM dir but llama.cpp wins | **refuse** | R-29 — llama.cpp cannot read safetensors |

**Only if both engines can genuinely run it** does the launcher pick on measured
performance.

#### Dense models → vLLM

vLLM wins dense at every workload measured (27B PXQ4, 2× P100 sm_60, rows D1/D3):

| | vLLM | llama.cpp | ratio |
|---|---|---|---|
| single-stream decode | 24.01 | 13.7 | 1.75× |
| aggregate decode @8 | ~70 | 12.4 | 5.6× |
| prefill | ~225 | 156.5 | 1.44× |
| aggregate decode @4 | *UNMEASURED* | 12.0 | — |

The launcher prints a standing caveat with this: the llama.cpp side (D1) is **one
boot**, below this bench's own two-boot bar, and the graphs-on dense arm (D2) was
never launched. The *direction* is not in doubt; the exact ratios are single-boot.

#### MoE models → a split instance, decided by concurrency

This is the important branch. MoE 35B PXQ4, 2× P100 sm_60:

| `--np` | llama.cpp | vLLM | winner | margin |
|---|---|---|---|---|
| 1 | 95.6 | 30.4 | llama.cpp | 3.14× |
| 4 | 75.93 | 64.82 | llama.cpp | +17.1% |
| 5 | **79.49** | 64.32 | llama.cpp | +23.6% ← llama.cpp peaks |
| 6 | 69.58 | 75.60 | **vLLM** | +8.7% ← crossover |
| 7 | 67.74 | 87.03 | vLLM | +28.5% |
| 8 | 62.42 | 95.81 | vLLM | +53.5% |

**The table is stored, not a slope.** Neither curve is monotonic and the flip is
sharp: llama.cpp *peaks* at np=5 above its own np=4 value, then drops 12.5% in one
step while vLLM climbs. The margin swings 32 points between np5 and np6. A
straight line from np4 to np8 would put the threshold too early and misprice np5
by ~14%.

Root cause of that shape, one sentence: llama.cpp `-sm layer` is a **serialized
two-GPU pipeline, not data parallelism**, so concurrent requests queue behind the
same pipeline while vLLM's aggregate climbs.

A currency warning rides every vLLM MoE decision: the crossover sweep measured
95.81 at np8 on a newer engine revision where row M7 has 88.7 on an older one
(+8.0%) for the same cell. The cause is a hypothesis, not a measurement — so if
that gap is real, the crossover may sit **below** np=6. That can change an engine
decision, not just a number.

#### Long-document workload → llama.cpp

`--workload longdoc` routes MoE to llama.cpp, which holds the prefill record at
every concurrency measured (1136 / ~1058 / ~1000 vs vLLM 567.6 / 595.8 / 594.4
tok/s, ~1.7–1.9×). The caveat is printed every time: that comparison is
**cross-harness and the prompt lengths were not matched** (2059 vs ~6.4k tokens).
Directionally trusted, not controlled.

### 5. vLLM eligibility is an *image* property

Not a compute capability. An earlier version gated on `MIN_VLLM_CAP = 70` and
routed every sub-sm_70 card to llama.cpp — which made the entire vLLM branch
unreachable on the very hardware every vLLM cell in the table was measured on
(2× P100 sm_60).

The narrow true statement is: **no vLLM decode-or-prefill throughput number on
sm_70 exists at all.** The only sm_70 vLLM figure on record is a 3.76×
shared-prefix win, which is not an instance decision.

So eligibility is *probed*, in this order, and the probe that failed is always
named:

1. `--vllm-image` / `PXA_VLLM_IMAGE` → look the tag up in the launcher's table
2. the image's declared arch set
3. bare-metal: `vllm_pxq4` importable **and** `PXA_PXQ4_LIB` naming an sm tag

If no image resolves, vLLM is not eligible and the reason names the probe — never
a bare "vLLM is sm_70+ only".

**Known image tags:**

| tag | caps | status | notes |
|---|---|---|---|
| `pxa-sm60-dev` | {60} | MEASURED | produced every MoE-crossover vLLM number. **Not self-contained** — see below. |
| `pxa-vllm:sm60` | — | INFERRED | Pascal thin image, torch 2.7.1. `caps` stays empty until a gated boot on a P100 pair passes against *this* tag. |
| `pxa-vllm:sm70` | — | INFERRED | Volta thin image, torch 2.10. Not yet gated. |

An image the launcher does not know has an UNMEASURED arch set and gets no card,
whoever built it.

> **Why there is no single fat image.** One image spanning sm_60+sm_70 was tried:
> 8 boot attempts on a V100, 8 failures. The cause is structural, not
> configuration — `VLLM_SKIP_C_STABLE=1` is required to build against torch 2.7.1
> (the last torch with sm_60 cubins) and it drops `csrc/libtorch_stable/`, where
> an op the V100 serving path calls unconditionally lives. You cannot have sm_60
> cubins and that op in the same build. Hence two thin images.

#### Images whose runtime lives on the host

An image can be a bare CUDA runtime whose python, torch and vllm all live on the
host and are bind-mounted at run time — `pip list` inside it shows pip and nothing
else. Every number attributed to such a tag was really produced by the image
**plus** those host paths.

Those paths are site-local, so **none are hardcoded**. Declare them in a JSON
descriptor and point `PXA_VLLM_HOST_ENV` at it:

```json
{"pxa-sm60-dev": {
   "mounts":   {"/srv/pxa": "/c"},
   "requires": ["/srv/pxa/venv/pyvenv.cfg",
                "/srv/pxa/venv/bin/python",
                "/srv/pxa/venv/lib/python3.12/site-packages/torch",
                "/srv/pxa/vllm-src/vllm/__init__.py",
                "/srv/pxa/kernels/libpxq4_sm60_v10.so"],
   "python":   "/c/venv/bin/python",
   "env":      {"PYTHONPATH": "/c/site",
                "PXQ4_LIB":  "/c/kernels/libpxq4_sm60_v10.so"},
   "editable_source": "/srv/pxa/vllm-src",
   "why": "how you traced it, so the next reader does not have to"}}
```

| key | meaning |
|---|---|
| `mounts` | host dir → container dir. The **model path is translated** through the longest matching prefix before the command is emitted; a model outside every mount is called out by name. |
| `requires` | host entries that must exist or the image is declared ineligible. Checked with `lexists`, not `exists` — `venv/bin/python` is typically a symlink to an interpreter that exists only *inside* the container. |
| `python` | the interpreter to invoke inside the container. `vllm serve` is wrong for this class of image; vllm is not on PATH there. |
| `env` | env the host runtime needs to be importable. Merged with `setdefault`, so a value from a measured arm always wins. |
| `editable_source` | if vllm is an editable install, the tree it resolves to. The launcher prints that tree's branch, sha and dirtiness, so a measurement can be attributed to something. |

Without a declared host environment, a tag marked `needs_host_env` yields **no
eligible card** — because the measurements it carries were produced by the image
plus a runtime the tag alone does not describe.

> **The container contract.** `--vllm-image` decides only *which cards are
> eligible*. The emitted command is a bare `vllm serve` — there is no `docker run`
> in it and the image name appears nowhere. The command **execs where the launcher
> is running.** The launcher says this out loud, because the flag reads like it
> selects a runtime, and a reader who believes that will attribute a measurement
> to an image that was never involved.

### 6. Derive the tensor split from free VRAM

An even split is llama.cpp's default and it is **wrong on any pool where the cards
are not equally free**. On a real five-card pool, one 16 GiB card was already
carrying another instance (7865 of 16384 MiB free) and one was an 11 GiB 1080 Ti also
in use — an even five-way split puts ~9.2 GiB of weights on a card with 7.8 GiB
free.

So `auto_tensor_split()` computes a **capacity-proportional** split over *free*
VRAM. Per card:

```
capacity = free_MiB − compute_buffer − cuda_context(250 MiB) − headroom(1200 MiB)
```

then shares are `round(1000 × capacity / Σcapacity)`.

Two details carry all the weight:

- **The head card is charged more.** llama.cpp places the output head on the last
  device in PCI order, so the last card in the selection is charged a **980 MiB**
  head compute buffer — measured, and flat in context — instead of the ordinary
  card's buffer, which is interpolated linearly in ctx between measured anchors
  (282 MiB @ c8192 → 786 MiB @ c262144, at ub1024).

- **The headroom term is the whole point.** *A config that loads is not a config
  that runs.* At c=262144 a five-card instance loaded, printed every buffer, then died
  on the **first token** with `CUDA error: out of memory` in `llama_decode` —
  the split had left one card with 51 MiB. Decode allocates transient buffers that
  the init-time buffer report does not include. The same instance at c=163840 kept
  807–1803 MiB free per card and ran. Hence the 1200 MiB reserve.

If any card has no capacity left, the launcher **declines to emit a split** rather
than emitting one that cannot work:

```
AUTO -ts DECLINED: card(s) [0, 1] have no capacity left after 1200 MiB headroom
+ compute buffers at ctx=16384. Free a card, drop -c, or pass --ts by hand.
```

> **`-ts` and `-ub` are coupled.** `-ts` partitions **bytes, not layers**, and
> llama.cpp folds a per-device compute allowance into the same walk — so changing
> `-ub` *repacks the layers*, and a split tuned at one `-ub` can OOM at another.
> This was measured: a six-card PXQ4 instance at ub2048 pushed a 16140 MiB V100 over
> with the ub512-tuned split. If you force `--ub`, re-derive `--ts`.

### 7. Pick the prefill chunk and the micro-batch

Precedence, in order, and the reason for each step:

1. **`--ub` / `--b`, if you forced one.** You own it; it is printed as forced, and
   so is the value the matched row measured, so the difference is visible.
2. **The matched recipe row's measured cell.** This is the point of the table.
   Adaptive-ub lands on 2048 on a 16 GiB card, while the measured best on a
   2× P100 pair is **256** — and it never sets `-b` at all, because the prefill
   *chunk* is not its job. On both card pairs `-b 8192` is worth about **+10%**
   long-prompt prefill over the engine default `-b 2048`
   (the measurement ledger chunk-size PROBE). On an `[INFERRED]` row only, a `-ub`
   the smallest selected card cannot hold is lowered to the card-type value and
   the lowering is printed; a `MEASURED` row is emitted exactly as measured.
3. **The `qwen4exp` `-ub 1024` rule**, where no row covers the topology: ub1024 is
   **+20% prefill** over the adaptive 512 (337–366 → 430–439 tok/s, six-card PXQ4,
   2931-token prompt) with **decode unchanged** (30.6–30.9 vs 30.6–31.1,
   overlapping). Not a trade; free. Not generalised to other architectures, which
   have different activation shapes.
4. **Nothing at all**, where no row covers the topology. The engine's adaptive-ub
   then probes real free VRAM per device and picks per card, which is the right
   answer where nothing was measured — and one global `-ub` across a heterogeneous
   pool would be wrong by construction.

In case 4 the launcher *prints* the value adaptive-ub should land on, from the
measured card-type table, so you can check it against the server's own
`PXA posture: mode=... fa=... ub=...` line:

| card | expected `-ub` |
|---|---|
| ≥15 GiB | 2048 |
| ≥10 GiB (11 GB 1080 Ti class) | 768 |
| smaller | 512 |

ub2048/1024 compute buffers are measured to OOM next to a ~10 GB model on an
11 GB card; ub768 fits.

`-ub` and `-ts` are **coupled**: llama.cpp folds a per-device compute allowance
into the `-ts` walk, so changing `-ub` repacks the layers and a split tuned at one
`-ub` can overflow a card at another. Force one and the launcher says so and tells
you to re-derive the other.

### 8. Offload the per-layer embedding table to CPU

Some architectures (`qwen4exp`, `gemma3n`) carry a **per-layer token embedding
(PLE)** table: `per_layer_token_embd.weight`. It is enormous — on one 97 GiB model
it is 160 × 320001536 = 51.2e9 elements, about **51 GiB of the file** — and it is
a pure `GET_ROWS` gather: one lookup per token per head, **no GEMM**.

So it belongs in host RAM. When the launcher sees it, it emits:

```
-ot per_layer_token_embd\.weight=CPU
```

**This is not an optimisation.** Without it the gather table is offloaded with
everything else and the load dies on a `cudaMalloc` of the whole tensor on one
card — measured at 16089.57 MiB on device 2 of a five-card instance.

Two subtleties:

- **The pattern is anchored on the full name deliberately.** A loose `ple` regex
  also matches `blk.N.ple_key`, `ple_conv1d` and the F32 `ple_norm_*` tensors,
  which are tiny and **must stay on the GPU**.

- **The VRAM check subtracts it.** R-17B compares *GPU-resident* bytes, not file
  bytes. A PLE table pinned to host RAM never reaches VRAM, so counting it would
  refuse instances that fit. Measured: a 96.77 GiB file of which 51.15 GiB is PLE has
  a GPU-resident remainder of 46.70 GiB and runs on five cards (75 GiB) with room
  for a 3.75 GiB KV cache at 160k context. Uncorrected, R-17B refused it outright.
  If the PLE's ggml type is not in the block-geometry table, the bytes are **not**
  subtracted and the launcher says so — the figures then overstate what reaches
  the cards, and any refusal should be treated as suspect.

---

## What the fit check may and may not block on

The KV-per-token figure is arithmetic over header fields, unvalidated against a
real allocation on every architecture but one. **So it is never allowed to
block.** It warns, labelled `[INFERRED]`.

Only two facts need no formula, and only those two block:

- **R-17A** — `-c` exceeds `<arch>.context_length` (a KV field, read directly).
- **R-17B** — GPU-resident weights *alone* exceed the total VRAM of the selection
  under full offload. File-byte arithmetic.

Everything else warns:

```
** VRAM estimate [INFERRED, never blocks]: weights 14.64 GiB + KV 4.06 GiB
   (= ctx 16384 x 260.0 KiB/tok) = 18.70 GiB vs 2.26 GiB free / 32.00 GiB total
   across 2 card(s). Compute buffers and fragmentation are NOT in this number.
** HEADROOM RULE [MEASURED]: leave ~1200 MiB free per card AFTER load.
```

---

## Flags

### Model and hardware

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | asked for | GGUF file, or a PXQ4-converted directory. Omit it and you pick from a numbered list. |
| `--gpus 0,1` | asked for | Card selection, by the index `nvidia-smi` reports. **Never left to the ambient environment** — the launcher always sets `CUDA_VISIBLE_DEVICES` explicitly and refuses to execute if it cannot name the devices. `--cards` is the old spelling and still works. |
| `--models-dir DIR` | see below | Where to look for models. Repeatable. Search order: `--models-dir`, then `PXA_MODELS_DIR` (`:`-separated), then the directory of your previous launch, then `./models`, `~/models`, `/models`. The roots actually searched are always printed. |
| `--engine llama\|vllm` | auto | Force an engine. Blockers are still reported **and still stop the run**. |
| `--tier T` | — | Assert the PXQ tier yourself; the R-03 escape hatch. You own the assertion. |

### Workload shape

| Flag | Default | Meaning |
|---|---|---|
| `--np N` | 1 | Concurrent slots. Drives the MoE engine choice, `-np`, and `--max-num-seqs`. Must be ≥1. |
| `--workload chat\|serve\|longdoc` | `chat` if `--np`≤1, else `serve` | `longdoc` routes MoE to llama.cpp for prefill. |
| `-c`, `--ctx N` | `np × 4096` | **Total** context, not per slot. 4096/slot is the measured envelope. |
| `--threads N` | **not passed** unless a recipe row sets it | `-t`. The measured arms did not set it, so neither does the launcher; llama.cpp picks its own. |

### Placement and memory

| Flag | Default | Meaning |
|---|---|---|
| `--ngl N` | 999 | Layers to offload. Below 99 on a GPU-only tier triggers R-16. |
| `--ts A,B` | auto | Forces the tensor split; the automatic capacity split is then **not** used. Refused on vLLM (R-10). |
| `--sm layer\|graph\|row` | `layer` | Split mode. `graph` on a DeltaNet hybrid is refused (R-12). Refused outright on vLLM (R-11). |
| `--ub N` | 0 = the matched recipe row's measured cell | Forces `-ub` (and `-b`, unless `--b` is given too). Where no row covers the topology nothing is passed and adaptive-ub probes each device. |
| `--b N` | 0 | Forces the prefill chunk `-b`. Only meaningful together with `--ub`. |
| `--ctk`, `--ctv` | `f16` | KV cache types. Checked against the compiled FA kernel pairs (R-13L). Refused on vLLM (R-13V). |
| `--no-mmap` | off | Adds `--no-mmap` and sets `PXA_PARALLEL_LOAD=1` (−25..−46% cold load; inert under mmap). |
| `--gmu F` | 0.90 sm_60 / 0.85 sm_70 | vLLM `--gpu-memory-utilization`. These are recipe values, **never swept** — UNMEASURED as a tuning axis. |

### Chat, template and sampling — the things people forget

| Flag | Default | Meaning |
|---|---|---|
| `--chat-template NAME` | the file's own embedded template | Override with a built-in. `--list-chat-templates` prints the names **this** engine accepts, read from the engine or from `src/llama.cpp`'s own map so the list cannot go stale. |
| `--chat-template-file PATH` | — | Override with a `.jinja` file. |
| `--no-jinja` | off (jinja is ON) | Turn `--jinja` off. Read the warning first: without it a request carrying `tools` returns HTTP 500 while plain chat keeps working. |
| `--reasoning-format` | unset → the engine's `deepseek` | `none` / `auto` / `deepseek` / `deepseek-legacy`. |
| `--reasoning-budget N` | unset | Cap thinking tokens; `0` = no thinking, `-1` = unlimited. |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty` | the model's own `general.sampling.*` keys where the GGUF has them, else not passed | The launcher never invents a sampling default. |
| `--api-key KEY` | none — **the port is open** | Never echoed, never written into a `--serve-name` script. |
| `--slot-save-path DIR` | off | Enables `/slots` save+restore. |

### Multimodal

| Flag | Default | Meaning |
|---|---|---|
| `--mmproj PATH` | auto | Vision projector. Auto-resolves **only** if the model carries vision tensors and exactly one candidate sits beside it; two or more is R-24. |
| `--no-mmproj` | off | Suppress projector resolution. The launcher still lists what it *would* have found and states the instance is text-only. |

### Speculation

| Flag | Default | Meaning |
|---|---|---|
| `--spec METHOD[:k=v,...]` | — | e.g. `mtp:n_max=1`, `ngram-mod:n_max=4`. A bare `mtp` expands to `mtp:n_max=1` — **not** the old `n_max=4,n_min=2`, which was a measured loss emitted by default. |
| `--draft-model PATH` | — | External draft-model speculation. Zero coverage in this bench → R-25. |

### vLLM specifics

| Flag | Default | Meaning |
|---|---|---|
| `--vllm-image TAG` | — | Decides **card eligibility only**, not where the command runs. |
| `--cudagraph-mode MODE` | `FULL_DECODE_ONLY` | Anything else is refused (R-08). This is a correctness requirement, not a knob. |

### Serving and control

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `8080` | |
| `--explain` | off | Decide and print; run nothing. Exits 5 if the plan carries known-fatal blockers, 0 if clean. |
| `--selftest` | off | Run the decision table against this machine's real cards. |
| `--accept-unmeasured` | off | Execute a branch the launcher labels `[INFERRED]`/`UNMEASURED`. |
| `--allow-busy` | off | Select a card another process is already resident on. |
| `--serve-name NAME` | — | Also write an executable restart script for this exact seat. |
| `--serve-dir DIR` | `$PXA_SERVE_DIR`, else `~/.cache/pxa-launch/serve` | Where `--serve-name` writes. |
| `--yes`, `-y` | off | Do not ask for confirmation before launching. A scripted caller is never asked anyway. |
| `--no-tui` | off | Skip the full-screen UI and use the plain prompts. Same as `PXA_NO_TUI=1`. |
| `--no-interactive` | off | Never prompt at all; fail instead if `--model` or `--gpus` is missing. |
| `--list-chat-templates` | off | Print the built-in chat template names and where the list came from, then exit. |

---

## Refusals

Refusals are the point of the tool, not an inconvenience. Each has a stable id, so
you can grep for it. Numbering has gaps: **R-04 and R-09 are not implemented as
refusals** — R-09 is the post-boot verification contract, which is printed rather
than enforced, because this process `exec`s the server and cannot observe it.

### Artifact and tier

| id | Refuses |
|---|---|
| **R-01** | **PXQ1 content.** A PXQ1 MoE file loads, clears the composition gate at 80.9% PXQ-family bytes, and generates **incoherent text** — nothing downstream catches it. No dense path, no CPU codec. Detected by tensor type, so it fires on a uniform PXQ1 file *and* on a UNIVERSAL map with PXQ1-mapped experts. |
| **R-02** | Retired quant types 250/251, removed 2026-07-21. No shipped engine reads them. |
| **R-03** | Guessing the tier. `general.file_type` is 38 for every PXQ tier by design and the tensor directory shows no PXQ types either. Without a tier the PXQ1 refusal and the vLLM PXQ4-only gate cannot be enforced. Escape: `--tier`. |
| **R-05** | A foreign-quantized safetensors directory (`compressed-tensors`, `awq`, `gptq`, `fp8`, `bitsandbytes`) — readable by **neither** engine. |
| **R-06** | vLLM requested but the converted artifact does not exist. Names the convert command. |
| **R-18** | An unquantized HF checkpoint. The launcher will not emit `vllm serve --quantization pxq4` against fp16 weights. |
| **R-19** | A config-only stub with no weight files. |
| **R-22** | An existing path that is not a servable artifact — one safetensors shard, a `.tiers` map, an `.imatrix`, a LoRA adapter directory. |
| **R-23** | A file that is not a usable GGUF. *A truncated 5.3 GB file with a zeroed header is a shape that really occurs.* |
| **R-28** | Tensors carrying ggml type ids the current tree does not define (e.g. 246/247, retired clustered PXQ1C/PXQ2C variants). The engine cannot dispatch them. |

### Engine and card selection

| id | Refuses |
|---|---|
| **R-07** | Forced `--engine vllm` with **no eligible card**. Not a warning: with no eligible card the parallel degree collapses to 1, `CUDA_VISIBLE_DEVICES` is never set, and the server inherits **every GPU on the host**. |
| **R-20** | A card with a resident process, or >512 MiB resident with no compute app listed. This may be a shared, live box. Escape: `--allow-busy`. |
| **R-29** | A PXQ4-converted vLLM directory when llama.cpp wins the instance — llama.cpp cannot read safetensors. Names *why* llama.cpp won. |

### Parameters that do not translate

| id | Refuses |
|---|---|
| **R-10** | `-ts` with vLLM. vLLM splits work evenly; a per-card ratio has no equivalent and would be silently ignored. |
| **R-11** | `-sm` with vLLM. No equivalent in its parallelism model. |
| **R-12** | `-sm graph` on a DeltaNet hybrid. Produces **degenerate output** — the cross-device all-reduce never reaches its consumers and each device computes a different router top-8. Not fixable by an env var. |
| **R-13L** | A `-ctk`/`-ctv` pair with no compiled FA vec kernel at head 128. It does **not** fall back — it hard-aborts at request time. Compiled asymmetric pairs: `q8_0/q6_0`, `q8_0/iq4_nl`, `q6_0/q5_0`. |
| **R-13V** | `--ctk`/`--ctv` with vLLM — no equivalent, would be silently dropped. |
| **R-14** | `--spec mtp` with vLLM. No MTP drafter there, and ngram will **not** be substituted: on this model class the two have opposite verdicts (ngram +23.0% code; MTP −8.6%). Substituting a lever's meaning is worse than dropping it. |
| **R-15A** | `mtp:n_max≥2` — a measured loss on both architectures. |
| **R-15B** | `--spec mtp` on a file with **no** nextn/mtp tensors. Two shipped f16 files declare `nextn_predict_layers=1` with zero such tensors — the head was dropped in the pipeline and the flag survived. |
| **R-25** | `--draft-model`. Zero coverage in this bench on any cell. Escape: `--accept-unmeasured` (llama.cpp only; no vLLM draft path is emitted at all). |

### Configuration and envelope

| id | Refuses |
|---|---|
| **R-08** | `cudagraph_mode` other than `FULL_DECODE_ONLY`. `FULL_AND_PIECEWISE` captures **prefill** graphs and returns fluent garbage from character zero on short raw completions. Its best aggregate (88.4) is *below* the correct config's (88.7) — there is no speed argument for it. |
| **R-16** | A partial offload (`--ngl` < 99) on PXQ1 or PXQ6. The launcher still refuses this pending validation of the CPU dequant path — the codec itself now covers all six tiers (`docs/PXQ-CPU-DOT.md`); this is a launcher policy hold, not a technical "would abort." |
| **R-17A** | `-c` beyond the model's trained context. |
| **R-17B** | GPU-resident weights alone exceeding total VRAM under full offload. |
| **R-21** | `--np` above the measured cudagraph capture ladder `[1,2,4,8]`. A too-short ladder has cliffed before — a hardcoded `[1,2]` ladder cliffed at 3+ concurrent. Escape: `--accept-unmeasured`. |
| **R-24** | Guessing among multiple mmproj candidates when nothing ranks them. |
| **R-26** | `--np` < 1. |
| **R-27** | A multimodal/VL checkpoint on vLLM — the emitted command has no multimodal handling at all. Escape: `--engine llama`, or `--accept-unmeasured` to serve text-only. |

### Exit codes

| code | meaning |
|---|---|
| 0 | `--explain` produced a clean plan |
| 2 | no decision / unusable artifact / unusable card selection |
| 3 | a parameter or engine request that does not translate |
| 4 | the environment cannot run the command (no engine binary, no CUDA runtime) |
| 5 | `--explain` produced a plan carrying known-fatal blockers, **or** an unacknowledged UNMEASURED branch |

Exit 5 exists so a CI caller can distinguish "clean plan" from "plan that will not
start here".

---

## The measurement envelope

The entire decision table is keyed to **two cards of one class**. Outside that
envelope the launcher labels the answer and, in several cases, requires
`--accept-unmeasured`.

The evidence that this caution is warranted: **22.3 vs 24.6 tok/s on an identical
config between two different P100 pairs.** Even 2→2 does not transfer cleanly.

Triggers you will see:

- **card count ≠ 2** — the 2-card answer is printed and labelled `[INFERRED]`;
  executing needs an ack.
- **an sm_61 card in the selection** — the np5/np6 thresholds were never measured
  on sm_61, and the BALANCE-mode `PXA_FA_MASK_SKIP_TILE` win explicitly excludes
  all of sm_61.
- **heterogeneous `-ub` expectation** — the card-type table wants different values
  on different cards while the CLI carries one global `-ub`. The launcher passes
  none so adaptive-ub probes per device, but *whether adaptive-ub lands per-card
  correctly in a heterogeneous pool is UNMEASURED*.
- **`PXQ_UNIVERSAL` tier** — a UNIVERSAL MoE is on record loading PASS and
  generating incoherent output. Nothing verifies a coherence check ran on *your*
  file.

Note that sm_61 is deliberately **not** folded into a generic "Pascal" class with
sm_60. The bench has no MoE crossover, no dense pair and no PXQ-tier throughput on
sm_61 — folding it in is how that gap stops being visible.

---

## Worked examples

### A. Single card

A real run on this box, captured verbatim. One P100, a PXQU-16 file, nothing but
`--explain` and the two things a user knows.

```
$ python3 tools/pxa-launch.py --gpus 6 \
    --model /models/fusion35bv2/fusion2-35b-U16.gguf --explain
```

```text
==============================================================================
pxa-launch: ENGINE = llama
  model:  /models/fusion2-35b-U16.gguf  [gguf, 13.06 GiB]
  class:  MoE (256 experts), arch=qwen35moe  [read from GGUF header + tensor directory]
  tier:   PXQ_UNIVERSAL (provenance KV says PXQ_UNIVERSAL)  [tensor-type histogram over 733 tensors (the signal the loader dispatches on, llama-model-loader.cpp:527)]
  compose:f32x301 MXFP4x291 PXQ3x58 PXQ2x41 PXQ4x21 q8_0x20
  trained ctx: 262144
  mtp:    0 nextn/mtp tensors  [tensor walk for nextn/mtp names]
  deltanet: True  [210 ssm_* tensors - the GGUF spelling of the same linear-attention layers. Whether a PURE-SSM (non-hybrid) model needs the -sm graph guard is UNMEASURED; guarding is the safe direction]
  serve:  np=1, workload=chat, ctx=8192 (=8192/slot), threads=72
  cards:  6:Tesla P100-PCIE-16GB sm_60 16GiB
  recipe: 1xp100-pxqu16 (1x Tesla P100 16 GB (sm_60), 35B MoE, PXQU-16 + q8_0 head)  [MEASURED]
  from the row (you did not set these): -c 8192
  reason: tier is PXQ_UNIVERSAL: the vLLM backend implements PXQ4 ONLY and refuses every other tier cleanly at the conversion gate (PXQ-TYPE-MATRIX.md:69-70). llama.cpp reads PXQ2/PXQ3/PXQ4/PXQ4-HQ/PXQ6/UNIVERSAL. Cards: 6:Tesla P100-PCIE-16GB sm_60
  ev:     MEASURED tier support: PXQ-TYPE-MATRIX.md:69-70, :80-81
  ev:     MEASURED topology row '1xp100-pxqu16' (1x Tesla P100 16 GB (sm_60), 35B MoE, PXQU-16 + q8_0 head): decode 62.4 t/s (63.0 with ADDFUSE)  |  prefill 827-843 t/s @-ub 2048  [docs/COOKBOOK.md:65-73]
  ev:     MEASURED FA regime: -fa on for workload 'chat' - the interactive/serving setting, and what every recipe row above was measured at. Switch with --workload longdoc if you are ingesting rather than chatting. [docs/COOKBOOK.md:41-63 (Two FA regimes); bench/fair-battle.md:35-50]
  ** PXQ_UNIVERSAL: this is a MIXED per-tensor tier map. PXQ-TYPE-MATRIX.md:119 Finding 8 records a UNIVERSAL MoE that loads PASS and generates INCOHERENT - with the doc's own caution that the incoherence is traceable to that build recipe (Q3_K_M source, no imatrix) rather than to the codecs. Nothing verifies a coherence check ran on THIS file. llama.cpp only, and you must acknowledge it.
  ** Decode is ub-insensitive on this row - drop to -b/-ub 512 if you want a smaller compute buffer (docs/COOKBOOK.md:72-73).
  ** PXQ_UNIVERSAL acknowledgement WAIVED by recipe row '1xp100-pxqu16': that row was measured on this tier, with output-gated boots, which is the coherence evidence the general PXQ_UNIVERSAL warning asks for. The warning above still stands for any OTHER UNIVERSAL file - it is waived for this cell, not for the tier.
  ** PXA_PIPELINE_PP is ON by ENGINE default for arch 'qwen35moe' (qwen35/qwen35moe only). This launcher does not set or unset it; the seat inherits the engine's default. PXA_PIPELINE_PP is an ENGINE default for the qwen35/qwen35moe families only. It is NOT on for qwen4exp: at -c 150016 across four cards the n_copies=2 compute buffers OOM (the measurement ledger, the 1080 Ti no-boot has the same cause). This launcher does not set it either way.
  ** model is not the MoE anchor (PXA-Coder-35B-v2): the crossover is a property of a model x hardware PAIR, not of the engines. qwen3next (MoE-512), qwen3moe (MoE-128) and deepseek4 (MoE-6) have NO engine-vs-engine data at any np. [INFERRED]
  --- chat / serving settings (the ones people forget) ---
  chat template in the file: 7764 chars, looks like 'chatml'
    | {%- set image_count = namespace(value=0) %}
    | {%- set video_count = namespace(value=0) %}
    | {%- macro render_content(content, do_vision_count, is_system_content=false) %}
    |     {%- if content is string %}
    |         {{- content }}
    |     {%- elif content is iterable and content is not mapping %}
    | ... 148 more lines (7764 chars total)

  setting              value                                        source
  chat template        embedded in the GGUF (7764 chars, looks l... from the file
  --jinja              ON                                           MEASURED
  --reasoning-format   not passed -> engine default 'deepseek'      MEASURED
  --reasoning-budget   not passed (unlimited)                       [INFERRED]
  --temp               1                                            from the file
  --top-p              0.95                                         from the file
  --top-k              20                                           from the file
  --api-key            none - the port is OPEN                      UNMEASURED
  --slot-save-path     not set                                      [INFERRED]
  --host / --port      0.0.0.0:8080                                 default
  ** VRAM estimate [INFERRED, never blocks]: weights 13.06 GiB + KV 0.16 GiB (= ctx 8192 x 20.0 KiB/tok) = 13.22 GiB vs 16.00 GiB free / 16.00 GiB total across 1 card(s). Compute buffers and fragmentation are NOT in this number.
  ** KV/token source: arithmetic: 10 of 40 layers (full_attention_interval=4) x 2 kv-heads x (256+256) dims x 2 B (f16) = 20.0 KiB/token. [INFERRED] hybrid-attention correction applied from the qwen4exp measurement, but NOT validated on this arch. Q9 stays OPEN here - warn only, never blocks.
  ** HEADROOM RULE [MEASURED]: leave ~1200 MiB free per card AFTER load. Decode allocates transient buffers that the init-time buffer report does not include, so a seat can load and still OOM on its first token.
  mmproj: NOT attached. 1 projector(s) sit next to this model but it carries no vision tensors and no measurement ranks them:
            /models/mmproj-ornith-f16.gguf
          Pass --mmproj <path> if this seat is meant to take images.
  engine dir: <ENGINE>  [PXA_ENGINE_DIR -- WILL NOT START: NO CUDA RUNTIME ON THIS HOST (libcudart.so.12)]
  -t: NOT PASSED. The measured arms did not set it, so neither does this launcher; llama.cpp picks its own (this host has 72 cores). Pass --threads N to force one.
  -fa on: the interactive/serving regime for --workload chat. One server carries ONE setting; see the FA lines above for what the other one costs.
  -b 2048 -ub 2048 MEASURED from recipe row '1xp100-pxqu16' (1x Tesla P100 16 GB (sm_60), 35B MoE, PXQU-16 + q8_0 head).
     decode 62.4 t/s (63.0 with ADDFUSE)  |  prefill 827-843 t/s @-ub 2048
     source: docs/COOKBOOK.md:65-73
  PXA_ENHANCE=1: exported explicitly, not inherited. It auto-selects the measured-good kernel levers per device and prints its decision at startup (docs/COOKBOOK.md:24-39). On sm_61 it is what arms PXA_PXQ_INT8_PREFILL mode 1, without which the 1080 Ti numbers halve.
  env:     CUDA_VISIBLE_DEVICES=6 PXA_ENHANCE=1 LD_LIBRARY_PATH=<ENGINE>/bin:<ENGINE>/src:<ENGINE>/ggml/src:<ENGINE>/examples/mtmd:<ENGINE>/common
  command: <ENGINE>/bin/llama-server -m /models/fusion2-35b-U16.gguf --host 0.0.0.0 --port 8080 -ngl 999 -sm layer -c 8192 -ctk f16 -ctv f16 -np 1 -fa on --cont-batching -b 2048 -ub 2048 --jinja --temp 1 --top-p 0.95 --top-k 20
  POST-BOOT CONTRACT - NOT PERFORMED BY THIS PROCESS (it execs the server):
    this launcher makes NO healthy/unhealthy claim about the resulting seat.
    1. posture: llama-server logs 'PXA posture: mode=... fa=... ub=...' at
       startup - compare ub against the card-type expectation printed above.
    2. offload: confirm 'offloaded N/N layers to GPU'. A stub libcuda on the
       library path yields 0/N, correct output and ~50x slower
       (PERPLEXITY-RESULTS.md:41-55).
    3. device scoping: echo the child's CUDA_VISIBLE_DEVICES back; expect '6'.
    4. short-prompt correctness: a RAW, NON-chat-templated 1-token and 5-token
       completion BEFORE any number is trusted, exactly as all 11 crossover boots
       did (MOE-CROSSOVER.md section 4.3). Chat-templated traffic pads every prompt
       past the captured sizes, which is precisely why the FAP corruption survived
       arithmetic gating.
    5. speculation: if you armed one, the acceptance-rate line must be present and
       non-zero. If it is abse
```

Read it as: PXQ_UNIVERSAL tier → vLLM implements PXQ4 only → llama.cpp. One card →
no split, no `-ts`. The `1xp100-pxqu16` row matched on both hardware and model
class, so `-b 2048 -ub 2048 -c 8192` came from a measurement with its source line
attached, and the general PXQ_UNIVERSAL acknowledgement is waived **for this cell
only**, with the reason printed. The projector sitting beside the model is named
and deliberately not attached. The sampling values came out of the file's own
`general.sampling.*` keys, not out of this launcher.

### B. Two identical cards

```
$ python3 tools/pxa-launch.py --model ...-PXQ4.gguf --gpus 2,4 -c 8192 --explain
```

The envelope line now confirms the table applies as measured:

```
  ev:     MEASURED envelope: exactly 2 cards, both sm_60 - the table applies
          as measured (bench D1/D3 and the crossover sweep)
```

and the automatic split appears:

```
  AUTO -ts 697,303 [MEASURED method]: capacity-proportional over FREE VRAM after
  reserving 1200 MiB decode headroom + 282 MiB compute (980 on the head card)
  + 250 MiB context per device. Capacities: 2:1237MiB  4:539MiB. -ts partitions
  BYTES, not layers, and llama.cpp repacks when -ub changes - re-derive if you
  force a different -ub.

  command: ... -c 8192 ... -ts 697,303
```

Both cards are physically identical 16 GiB V100s with identical free memory — yet
the split is **697/303, not 500/500**, because the *head* card (the last in the
selection) is charged the 980 MiB output-head buffer against the ordinary card's
282 MiB. That asymmetry is the difference between an instance that runs and an instance that
OOMs on its first token.

If the cards are too full, the launcher declines rather than emitting a bad split:

```
  AUTO -ts DECLINED: card(s) [0, 1] have no capacity left after 1200 MiB headroom
  + compute buffers at ctx=16384. Free a card, drop -c, or pass --ts by hand.
```

### C. A mixed pool

```
$ python3 tools/pxa-launch.py --model ...-PXQ4.gguf --gpus 0,2,3 --np 4 --explain
```

Cards 0 (P100 sm_60), 2 (V100 sm_70), 3 (1080 Ti sm_61) — three classes, three
warnings, then a refusal to *execute*:

```
  ** UNMEASURED card count: the MoE crossover and the dense pair were both
     measured on exactly 2 cards. You selected 3. The 2-card answer is printed
     and labelled [INFERRED]. MEASURED: 22.3 vs 24.6 on an identical config
     between two different P100 PAIRS - even 2->2 does not transfer cleanly.
  ** selection contains an sm_61 card: the np5/np6 thresholds were NEVER measured
     on sm_61, and the BALANCE-mode PXA_FA_MASK_SKIP_TILE win explicitly excludes
     all of sm_61. UNMEASURED.
  ** HETEROGENEOUS -ub expectation across this selection: the card-type table
     wants [768, 2048] on different cards while the CLI carries ONE global -ub.
     This launcher passes NO -ub so the engine's adaptive-ub probes each device
     itself (a mixed pool matches no recipe row, so nothing is emitted) - but
     whether adaptive-ub lands per-card correctly in a heterogeneous
     pool is UNMEASURED.

  REFUSING to execute an UNMEASURED branch without acknowledgement:
    - 3-card selection (table is 2-card only)
    Re-run with --accept-unmeasured to proceed anyway. The plan above is the plan;
    the refusal is about executing it, not about printing it.
```

That last sentence is the model for every refusal in the tool: **the plan is still
printed in full.** The refusal is about *executing* it, never about hiding it.

Note also that vLLM would be ineligible here regardless — the selection spans
three compute capabilities and the vLLM command carries a single
`--attention-backend`.

---

## What gets emitted

### llama.cpp

```
<ENGINE>/bin/llama-server -m MODEL --host H --port P
  -ngl 999 -sm layer -c CTX -ctk f16 -ctv f16 -np N
  -fa {on|off} --cont-batching                     regime from --workload
  [-t THREADS]                       only if a row sets it, or you asked
  [-b B -ub UB]                      from the matched recipe row; forced by --ub/--b;
                                     absent only where no row covers the topology
  [-ot per_layer_token_embd\.weight=CPU]   from the row, or because a PLE table is present
  [-ts A,B]                          from the row, automatic from free VRAM, or forced
  [row extras]                       e.g. -wgt 8 --kv-unified --no-context-shift
                                     --ctx-checkpoints 0
  [--spec-type ...] [-md ...] [--mmproj ...]
  [--jinja] [--chat-template ...|--chat-template-file ...]
  [--reasoning-format ...] [--reasoning-budget N]
  [--temp T] [--top-p P] [--top-k K] [--min-p M] [--repeat-penalty R]
  [--slot-save-path DIR] [--api-key KEY]
```

with `PXA_ENHANCE=1`, the matched row's own `PXA_*` levers, and an
`LD_LIBRARY_PATH` pointing into the build tree.

> **The stubs trap.** Any inherited `/stubs` directory is **stripped** from
> `LD_LIBRARY_PATH` and the removal is announced. A 66 KB stub `libcuda.so.1`
> shadows the real driver, ggml logs one line, offloads 0/N layers, and the run is
> numerically **correct** and ~50× slower. A CUDA toolkit install puts that
> directory on the path routinely, so this is not hypothetical.

`GGML_CUDA_NO_PINNED` is deliberately **not** emitted: it appears in no measured
recipe here, and its effect on the anchors is unmeasured. The rule is never to set
a lever the anchor was not measured with.

### vLLM

```
vllm serve MODEL --host H --port P                 (self-contained image)
HOSTPY -m vllm.entrypoints.openai.api_server ...   (host-runtime image)
  --quantization pxq4 --dtype float16
  --max-model-len CTX --max-num-seqs N
  --gpu-memory-utilization 0.90|0.85
  --attention-backend PASCAL_SDPA|FLASH_ATTN_V100
  [--disable-custom-all-reduce]      when topology shows no P2P
  --pipeline-parallel-size D --tensor-parallel-size 1     (MoE)
  --tensor-parallel-size D                                (dense)
  --compilation-config {"custom_ops":["none"],
                        "cudagraph_mode":"FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes":[1,2,4,8]}
```

Three load-bearing details:

- **`custom_ops:["none"]` is mandatory** wherever `FULL_DECODE_ONLY` is emitted on
  sm_60. Without it, PP=2 + FDO is a **hard boot failure** — an illegal memory
  access in `determine_available_memory → profile_run`, on a boot that produced
  zero tokens. Adding the key fixed it with no other change. Its necessity on
  sm_70 is unmeasured; it is emitted anyway, which is the safe direction.

- **MoE goes pipeline-parallel, dense goes tensor-parallel.** PP=2 is the arm that
  holds every MoE number in the table.

- **Custom all-reduce is disabled from *read* topology**, not hardcoded. Measured:
  CAR costs ~18% vs NCCL on MoE without P2P, while the CAR kernel itself is
  exonerated.

- **The parallel degree is truncated to a power of two** if the eligible card count
  is not one — vLLM rejects a non-power-of-two degree at startup — and the dropped
  cards are named.

`VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE` is **never** set: it crash-looped
the container at warmup 3/3 boots and left an instance in a restart loop.

---

## The post-boot contract

The launcher `exec`s the server, so it can observe nothing afterwards — and
therefore **makes no health claim at all**. Instead it prints the checks whoever
owns the instance must run:

1. **Capture mode** (vLLM): grep the log for the *installed* cudagraph mode. If it
   is not `FULL_DECODE_ONLY`, the instance is not healthy — shut it down. *Passing a
   flag is not evidence the flag took effect:* an image can parse
   `FULL_DECODE_ONLY`, boot healthy, and silently override it back from its own
   compile policy. A derived image is the fix, not a flag.
   **Posture** (llama.cpp): the server logs `PXA posture: mode=... fa=... ub=...`
   at startup — compare `ub` against the expectation printed above.
2. **Split / offload**: per-device resident bytes for vLLM; `offloaded N/N layers
   to GPU` for llama.cpp. A stub libcuda yields 0/N, correct output, ~50× slower.
3. **Device scoping**: echo the child's `CUDA_VISIBLE_DEVICES` back.
4. **Short-prompt correctness**: a **raw, non-chat-templated** 1-token and 5-token
   completion before any number is trusted — exactly as all 11 boots of the
   crossover sweep did. Chat-templated traffic pads every prompt past the captured
   sizes, which is precisely why the prefill-graph corruption survived arithmetic
   gating.
5. **Speculation**: if you armed one, the acceptance-rate line must be present and
   non-zero. If it is absent, **drop the claim** and keep serving.

---

## Environment variables

### Read by the launcher

| Variable | Effect |
|---|---|
| `PXA_ENGINE_DIR` (alias `PXQ_ENGINE_DIR`) | Directory containing `bin/llama-server`. Wins over auto-detection — and if that build will not start, that is **reported**, never quietly stepped over. |
| `PXA_VLLM_IMAGE` | Same as `--vllm-image`. |
| `PXA_VLLM_HOST_ENV` | Path to the JSON descriptor of site-local host runtimes. Unreadable or non-object content is reported and treated as **absent**, which refuses rather than proceeding on a half-read file. |
| `PXA_PXQ4_LIB` | Bare-metal fallback probe; the `libpxq4_sm<cc>_v<n>.so` name supplies the arch set. |
| `PXA_MODELS_DIR` | `:`-separated list of directories to search for models. Same as repeating `--models-dir`. |
| `PXA_LAUNCH_STATE` | Where the launcher remembers the directory of your last launch and the last command. Default `~/.cache/pxa-launch`. Losing it costs you nothing but a re-typed `--models-dir`. |
| `PXA_SERVE_DIR` | Where `--serve-name` writes restart scripts. Default `$PXA_LAUNCH_STATE/serve`. |
| `PXA_NO_TUI` | Any non-empty value skips the full-screen UI and uses the line prompts. |
| `PXA_API_KEY` | Read by a saved restart script, never by the launcher itself. The key is deliberately not stored in the script. |

Engine auto-detection, when `PXA_ENGINE_DIR` is unset, looks in this order: build
directories inside the repo (`build`, `build-cuda`, `build-unified`,
`build-release`, `build-sm60`, `build-sm70`, `build-all`), the same names as
siblings of the checkout, then install prefixes (`/usr/local`, `/usr`, `/opt/pxa`,
`/opt/pxa`, `~/.local`), then `llama-server` on `PATH`. **Every candidate
that exists but will not start is printed with its reason** — never skipped in
silence. There is no site-specific build list in the file: a hardcoded absolute
path is a machine's private detail.

### Emitted into the child

| Variable | Engine | Why |
|---|---|---|
| `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` | both | Always set explicitly. |
| `PXA_ENHANCE=1` | llama.cpp | Every measured row's env, exported **explicitly** rather than inherited from an engine default, so the printed command means the same thing on any build. |
| the matched row's `PXA_*` levers | llama.cpp | Copied verbatim from the row's source document — 11 on the 4x P100 Flash-Next seat, `PXA_AUTO_SPEC=0` on the 1080 Ti, none elsewhere. |
| `LD_LIBRARY_PATH` | llama.cpp | Build-tree libs, with any `/stubs` directory stripped. |
| `PXA_PARALLEL_LOAD=1` | llama.cpp | Only with `--no-mmap`. |
| `TORCHDYNAMO_DISABLE=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | vLLM | Measured crossover-sweep arm B env. |
| `VLLM_SM70_QUANT_BACKEND=turbomind` | vLLM, sm_70 | From the sm_70 serving recipe. |

`PXA_ALLOW_GRAPH_SPLIT_HYBRID` exists but only **removes the R-12 guard** — it
does not make graph split correct on a DeltaNet hybrid.

---

## `--selftest`

Runs the decision table against this machine's real cards, with no model file
involved. It prints the detected cards, their `-ub` table expectation, the peer
topology, the resolved vLLM eligibility, and then the engine choice for every
combination of card-set × model class × tier × `--np`:

```
=== selftest: decision table against this machine ===
    card 0: Tesla P100-PCIE-16GB  sm_60  16384 MiB total, 14989 MiB used, -ub table -> 2048
    card 3: NVIDIA GeForce GTX 1080 Ti  sm_61  11264 MiB total, 9552 MiB used, -ub table -> 768
    topology: no NVLink/P2P; interconnect NODE/PHB/PIX/PXB/SYS -> custom all-reduce OFF
  --- eligibility as resolved here: image=None, caps=none ---
  all cards   MoE PXQ4 gguf np=6 -> llama   :: model is a raw GGUF; vLLM needs...  <ACK-REQUIRED>
  all cards   MoE PXQ1 gguf np=6 -> REFUSE R-01 :: REFUSING: PXQ1 content
  first card  MoE PXQ3 gguf np=5 -> llama   :: tier is PXQ3: the vLLM backend...
```

It then prints **every row of the topology recipe table** — the flags, the FA
regime per workload, the env and the source line for each — and exercises each
one: a synthetic card selection and model profile matching the row are pushed
through the same matcher the real run uses, and the row that comes back must be
the row being described. A table entry that cannot be reached is worse than no
entry, so `REACHED` / `UNREACHABLE` is printed per row and an unreachable one
fails assertion A6. The off-table case (three cards of one class) is exercised
too and must come back with no row at all rather than the nearest one.

```
  --- topology recipe table (9 rows) ---
  [MEASURED] 2xp100-dense-pxq4        2x sm_60   dense|hybrid  PXQ4     REACHED
             flags: -b 8192 -ub 256 -c 32768 -sm layer
             fa:    chat:-fa on/serve:-fa on/longdoc:-fa off
             env:   PXA_ENHANCE=1
             src:   RELEASE-NOTES-2026-09-07.md:78 (P100 headline row); bench/fair-battle.md:217; ...
             meas:  prefill 340.16 t/s @3,121 tok | 316.84 @20,801 | decode 17.83 @fill 8
  [off-table] 3x sm_60 dense PXQ4 -> UNMEASURED, no -b/-ub emitted
```

Finally it re-checks nine standing assertions, each of which exists because
ignoring it once cost this project a live bug:

| | what it asserts |
|---|---|
| A1 | every emitted vLLM compilation-config is `FULL_DECODE_ONLY` + `custom_ops:["none"]`, ladder covers `np` |
| A2 | `FULL_AND_PIECEWISE` is never reachable as an emitted value |
| A3 | a non-FDO `--cudagraph-mode` request is refused, not honoured |
| A4 | PXQ1 tensors anywhere yield tier PXQ1, so R-01 fires inside a UNIVERSAL map too |
| A5 | a bare `--spec mtp` expands to `n_max=1` only |
| A6 | every recipe row is reachable; an off-table topology returns none |
| A7 | the FA regime map is total over chat/serve/longdoc and only longdoc is fa-off |
| A8 | no MEASURED row exceeds the card-type `-ub` ceiling on its own smallest card |
| A9 | the 4x P100 Flash-Next row carries exactly the eleven published levers |

Use it after changing images, drivers or card layout to see what the launcher
*would* do before a model is involved.

---

## See also

| Path | What it holds |
|---|---|
| `docs/lab/LEVERS.md` | The supported `PXA_*` levers, defaults and measurements |
| `docs/PXA-SM60-SERVING.md` | Reproduced sm_60 instance numbers and the shipping recipe |
| `docs/PXA-SM70-SERVING.md` | The same for sm_70 |
| `scripts/pxa-serve-sm60.sh`, `scripts/pxa-serve-sm70.sh` | The recipes themselves |
| `tools/vllm-pxq4/` | The vLLM PXQ4 backend and `gguf_to_vllm.convert` |
| `src/llama-quantize.cpp` | Which provenance KVs each PXQ tier writes |
| `src/llama-model-loader.cpp` | How the loader detects PXQ1, by tensor type |
| `ggml/include/ggml.h` | The ggml type ids this launcher dispatches on |
| `docs/RENAME-MAP.md` | The PXQ tier display-name ladder and retired ids |
