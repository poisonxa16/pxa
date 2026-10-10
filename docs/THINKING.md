# Thinking on/off in PXA Control

Every model that can think gets a **Thinking** switch (auto / on / off) and a **thinking budget**
(tokens of thinking per reply) in PXA Control: on the Launch tab (the server's default), on the
Chat tab (per request) and as a column on the Models tab. The switch is the same everywhere;
what it does underneath depends on the model, and PXA Control works that out from the GGUF.

- **Budgets are suggestions.** The numbers below are starting points for interactive chat, not
  measurements. A model of 10B parameters or fewer gets the smaller one. Every budget field shows
  the suggestion as its placeholder; type a number to use your own, `-1` for no cap.
- **auto** sends nothing: the model's own default (its template, or the engine's `PXA_AUTO`) stays.

## How the model is recognised

`tools/pxa_thinking.py` reads `general.architecture`, `general.name` / `basename` / size, and
`tokenizer.chat_template` from the GGUF header (or the running server's `/props`), then:

1. **Template signature.** The chat template is matched against the table
   (`tools/pxa_thinking_profiles.json`). This is the strongest evidence: a fine-tune named
   "Qwen3.8" whose template has no switch gets no switch.
2. **Architecture + name.** No template, or one the table does not know: the architecture list and
   a name pattern pick the family (e.g. `phi3` + "reasoning" = Phi-4-reasoning, `qwen3moe` +
   "Thinking-2507" = always thinks, `qwen3moe` + "Instruct-2507" = none). A family that needs a
   template feature (e.g. `enable_thinking`) is only chosen if the template has it.
3. **Template inspection (unknown models).** The template is searched for `enable_thinking`,
   `thinking`, `reasoning_effort` (with its allowed values) and a generation prompt that opens
   `<think>`; the matching `generic-*` profile is used.
4. **Unknown.** Nothing found: the toggle is hidden (a safe default; nothing is sent).

With jinja2 installed, PXA Control also renders the template with thinking on and off and shows
whether the two prompts differ (`verify` in `/api/thinking`). Without jinja2 it still works.

## How on and off are done (methods)

Each family lists its methods in the table (`methods.on` / `methods.off`), most preferred first.
PXA Control uses the first one the model's template allows. The Chat tab's **fallback method**
box uses the next one.

| method | what is sent | used for |
|---|---|---|
| `template-kwarg` | `chat_template_kwargs` (`enable_thinking`, `thinking`, `reasoning`, `reasoning_effort`, `thinking_budget`) | every hybrid model; always the first choice |
| `soft-tag` | a tag the model honours: ` /no_think` / ` /think` at the end of the last user message (Qwen3, `/nothink` for GLM), or a system-prompt line (`<\|think_off\|>` PXA Coder/Agent, `/no_think` SmolLM3 and Nemotron v1.5, `detailed thinking off` Nemotron v1) | the fallback for kwarg families; the only switch for Nemotron |
| `empty-think-prefill` | the request gets a final assistant message holding an empty, closed thought: `<think>\n\n</think>\n\n`, or the family's own tokens (`[THINK][/THINK]`, `<think></think>`, the harmony analysis→final channels for gpt-oss, Apriel's `[BEGIN FINAL RESPONSE]`). If the generation prompt already opened the thought, only the close is sent. The engine continues a final assistant message as a prefill (`examples/server/server-common.cpp`), and refuses one while `enable_thinking` is on, so `enable_thinking: false` goes with it. The reply comes back as plain content (no `reasoning_content`). | **off** for models that always think: R1 and distills, QwQ-style, Kimi K2 Thinking, Magistral, Phi-4-reasoning, MiniMax-M2, Step-3.5, Apriel, GLM 5.x, gpt-oss |
| `reasoning-budget` | `thinking_budget_tokens: 0` / `--reasoning-budget 0`: the engine closes the thought at once | the fallback after the prefill, and the only server-wide off for always-think models. **Not yet measured live**: check it when a model seat is up |
| `effort-level` | the lowest effort level | Muse-Glimmer; the gpt-oss / GLM 5.x fallback |

The prefill string is measured on the model's own template (rendered with `enable_thinking`
false, as the engine renders a prefilled request) when jinja2 is installed. Without jinja2 it is
read from the template source. For a kwarg family whose generation prompt already closes the
thought, no prefill is offered. The soft tag and the prefill only work per request (the Control
chat, or any client that sends them). Server-wide, PXA Control uses the first method a launch
flag can carry, and the plan says so.

## Where the switch is applied

| place | what is sent | scope |
|---|---|---|
| Launch (server profile, or the per-model setting) | `--reasoning on/off` (= `enable_thinking`), `--chat-template-kwargs` for other keys (`thinking`, `reasoning_effort`, ...), `--reasoning-budget N` (engine-enforced families), `--reasoning-budget 0` (off for always-think models) | every client of that server |
| Chat tab / engine proxy (`pxa_thinking` `{mode, budget, level, fallback}` in the body of `POST /api/engine/v1/chat/completions`) | `chat_template_kwargs`, a soft tag (system line or `/no_think` suffix), an empty-think assistant prefill, `thinking_budget_tokens`, `max_tokens` | one request |

Precedence at launch: the server's own setting, then the per-model setting (saved in
`control.json` → `thinking_models`), then auto. The operator's **extra args** come after the
thinking flags, so they still win. Families switched by a prompt line (Nemotron) can only be
switched per request; the plan says so.

The engine proxy removes `pxa_thinking` before forwarding; a body without it is forwarded
unchanged. The reply carries `X-PXA-Thinking` (family, notes) so you can see what was done.

### Budget enforcement

- **engine**: the engine's reasoning-budget sampler. It closes the thought after N tokens and the
  answer follows. Per request: `thinking_budget_tokens`; per server: `--reasoning-budget`. It only
  arms when the engine's chat handler knows the thinking tags of that template (the generic
  `<think>` parser, Gemma 4, Ministral/Magistral, Kimi, DeepSeek V3.2, Cohere2 MoE, LFM2).
- **template**: the model's own budget kwarg (Seed-OSS `thinking_budget`).
- **guard**: nothing can stop the thinking early (gpt-oss and Muse-Glimmer: the engine's handler
  has no thinking tags; Granite, Hermes, Nemotron v1). PXA Control caps `max_tokens` at
  budget + 1024 so a runaway thought cannot run forever.

For every family, when thinking is on and `max_tokens` is smaller than budget + 1024, it is raised
so the answer has room after the thought.

## Families

| family | models | mechanism | on | off | on via | off via (preferred first) | default | levels | budget (>10B / ≤10B) | enforced by |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen38-effort` | Qwen3.8 family (Swift, Flash-Next, Hemlock, Victoria, ThinkingCap) | kwarg+effort | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | on | reasoning_effort: low/medium/xhigh (default xhigh) | 8192 / 4096 | engine |
| `pxa-froggeric` | PXA Coder / PXA Agent (Qwen3.6 froggeric template) | kwarg | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false}; fallback system line `<\|think_on\|>`/system line `<\|think_off\|>` | template-kwarg → soft-tag | template-kwarg → soft-tag → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | on |  | 4096 / 2048 | engine |
| `gemma4` | Gemma 4 (12B, 26B-A4B, 31B; incl. MTP assistant) | kwarg | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<\|channel>thought\n<channel\|>`) | off |  | 4096 / 2048 | engine |
| `gpt-oss` | gpt-oss (20B, 120B; harmony format) | effort | reasoning_effort (level) | reasoning_effort low (lowest effort) | effort-level | empty-think-prefill → effort-level (prefill `<\|channel\|>analysis<\|message\|><\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>`) | always | reasoning_effort: low/medium/high (default medium) | 4096 / 2048 | guard |
| `muse-glimmer` | Muse Glimmer (recipient harmony) | effort | reasoning_strength (level) | reasoning_strength low (lowest effort) | effort-level | effort-level | always | reasoning_strength: low/medium/high (default high) | 4096 / 2048 | guard |
| `hunyuan-v3` | Hunyuan V3 (hy_v3) | effort | kwargs {"reasoning_effort":"high"} | kwargs {"reasoning_effort":"no_think"} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<think:opensource></think:opensource>`) | off | reasoning_effort: low/high (default high) | 4096 / 2048 | engine |
| `glm5-effort` | GLM 5.x Flash (glm5next) | effort | reasoning_effort (level) | reasoning_effort low (lowest effort) | effort-level | empty-think-prefill → effort-level (prefill `<think></think>`) | always | reasoning_effort: low/high/max (default max) | 8192 / 4096 | engine |
| `cohere-reasoning` | Cohere Command A Reasoning / Cohere2 MoE | kwarg | kwargs {"reasoning":true} | kwargs {"reasoning":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<\|START_THINKING\|><\|END_THINKING\|>`) | on |  | 4096 / 2048 | engine |
| `command-r7b` | Command R7B (cohere2) | kwarg | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<\|START_THINKING\|><\|END_THINKING\|>`) | on |  | 2048 / 2048 | engine |
| `seed-oss` | ByteDance Seed-OSS (native thinking budget) | budget-kwarg | kwargs {"thinking_budget":-1} | kwargs {"thinking_budget":0} | template-kwarg | template-kwarg | on |  | 4096 / 2048 | template |
| `smollm3` | SmolLM3 | kwarg | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false}; fallback system line `/think`/system line `/no_think` | template-kwarg → soft-tag | template-kwarg → soft-tag → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | on |  | 2048 / 2048 | engine |
| `deepseek-hybrid` | DeepSeek V3.1 / V3.2 / V4 (hybrid thinking) | kwarg | kwargs {"thinking":true,"enable_thinking":true} | kwargs {"thinking":false,"enable_thinking":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<think></think>`) | template |  | 8192 / 4096 | engine |
| `kimi-thinking` | Kimi K2 Thinking (always thinks) | always | (default) | engine force-close (budget 0) | default | empty-think-prefill → reasoning-budget (prefill `<think></think>`) | always |  | 8192 / 8192 | engine |
| `apriel-thinker` | ServiceNow Apriel Thinker (always thinks, untagged) | always | (default) | — | default | empty-think-prefill (prefill `Here are my reasoning steps:\n\n[BEGIN FINAL RESPONSE]\n`) | always |  | 8192 / 4096 | guard |
| `magistral` | Magistral / Ministral Reasoning ([THINK]) | always | (default) | engine force-close (budget 0) | default | empty-think-prefill → reasoning-budget (prefill `[THINK][/THINK]`) | always |  | 8192 / 4096 | engine |
| `granite-thinking` | IBM Granite 3.2 / 3.3 (thinking kwarg) | kwarg | kwargs {"thinking":true} | kwargs {"thinking":false} | template-kwarg | template-kwarg | off |  | 2048 / 2048 | guard |
| `hermes4` | Hermes 4 (thinking kwarg) | kwarg | kwargs {"thinking":true} | kwargs {"thinking":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | off |  | 4096 / 2048 | guard |
| `nemotron-detailed` | Llama Nemotron v1 (Nano 4B/8B, Super 49B, Ultra 253B) | system | system line `detailed thinking on` | system line `detailed thinking off` | soft-tag | soft-tag | off |  | 4096 / 2048 | guard |
| `nemotron-soft` | Nemotron v1.5 / Nano v2 (/think, /no_think in the system prompt) | system | system line `/think` | system line `/no_think` | soft-tag | soft-tag → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | on |  | 4096 / 2048 | engine |
| `phi4-reasoning` | Phi-4 reasoning / mini-reasoning (always thinks) | always | (default) | engine force-close (budget 0) | default | empty-think-prefill → reasoning-budget (prefill `<think>\n\n</think>\n\n`) | always |  | 8192 / 4096 | engine |
| `minimax-m2` | MiniMax M2 (interleaved thinking, always on) | always | (default) | engine force-close (budget 0) | default | empty-think-prefill → reasoning-budget (prefill `<think>\n\n</think>\n\n`) | always |  | 8192 / 8192 | engine |
| `glm-hybrid` | GLM-4.5 / 4.6 / 4.7 (enable_thinking, /nothink) | kwarg | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false}; fallback user suffix `/nothink` | template-kwarg | template-kwarg → soft-tag → empty-think-prefill (prefill `<think></think>`) | on |  | 8192 / 4096 | engine |
| `qwen-hybrid` | Qwen3 hybrid / Qwen3.5 / Qwen3.6 and derivatives (Ornith, Fusion, GRM, Laguna, ...) | kwarg | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false}; fallback user suffix `/think`/user suffix `/no_think` | template-kwarg → soft-tag | template-kwarg → soft-tag → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | on |  | 4096 / 2048 | engine |
| `deepseek-r1` | DeepSeek R1 / R1 distills / QwQ / always-think <think> models | always | (default) | engine force-close (budget 0) | default | empty-think-prefill → reasoning-budget (prefill `<think>\n\n</think>\n\n`) | always |  | 8192 / 4096 | engine |
| `generic-kwarg` | unknown model: template reads enable_thinking | kwarg | kwargs {"enable_thinking":true} | kwargs {"enable_thinking":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | on |  | 4096 / 2048 | engine |
| `generic-thinking-kwarg` | unknown model: template reads 'thinking' | kwarg | kwargs {"thinking":true} | kwargs {"thinking":false} | template-kwarg | template-kwarg → empty-think-prefill (prefill `<think>\n\n</think>\n\n`) | off |  | 4096 / 2048 | guard |
| `generic-effort` | unknown model: template reads reasoning_effort | effort | reasoning_effort (level) | reasoning_effort low (lowest effort) | effort-level | effort-level | always | reasoning_effort: low/medium/high (default medium) | 4096 / 2048 | guard |
| `generic-always` | unknown model: the generation prompt always opens <think> | always | (default) | engine force-close (budget 0) | default | empty-think-prefill → reasoning-budget (prefill `<think>\n\n</think>\n\n`) | always |  | 8192 / 4096 | engine |
| `none` | no thinking mode | none | — | — | — | — | none |  | — | none |
| `unknown` | unknown model (no template, no rule): toggle hidden | unknown | — | — | — | — | none |  | — | none |

Notes:

- **Always-think models** (R1, QwQ-style, Kimi K2 Thinking, Magistral, MiniMax M2, Phi-4-reasoning,
  Step 3.5, Apriel, GLM 5.x): their template cannot turn thinking off. "Off" pre-fills an empty,
  closed thought per request. Server-wide (Launch), "off" is `--reasoning-budget 0`, which is not
  yet measured live.
- **gpt-oss**: "off" pre-fills the harmony analysis channel empty and jumps to the final channel.
  The fallback is the lowest effort. **Muse-Glimmer**: effort levels only; "off" is the lowest level.
- **Gemma 4**: the engine's `PXA_AUTO` turns thinking off by default for Gemma 4, so auto = off.
- **Qwen3.8 family**: `reasoning_effort` accepts only `low`, `medium`, `xhigh` (the template raises
  on anything else); PXA Control only offers those.
- **Heuristic-only families** (no template on PXANET to test against): Seed-OSS, Phi-4-reasoning,
  MiniMax M2, Hermes 4, Nemotron v1.5 / Nano v2, Magistral by name. Their rules follow the
  model cards; the info line says "architecture + name".

## Models on PXANET (scan of 2026-10-06)

| on-disk models (PXANET) | family | mechanism | suggested budget | enforced by |
|---|---|---|---|---|
| DSPARK-KVPROBE-STUB, DSPARK-LIVE-STUB, DeepSeek-V4-Flash-0731-DSPARK-TARGET-STUB, DeepSeek-V4-Flash-0731-DSpark-support | `deepseek-hybrid` | kwarg | 8192 | engine |
| gemma-4-12b-it-PXQ2-attn4, gemma-4-12b-it-PXQ3-balanced, gemma-4-12b-it-PXQ4-tmpl, gemma-4-12b-it-PXQ4, gemma-4-12b-it-Q8_0, gemma-4-12b-it-qat-q4_0 … (+5) | `gemma4` | kwarg | 4096 | engine |
| gemma-4-12B-it-assistant-F16 | `gemma4` | kwarg | 2048 | engine |
| GLM-5.3-Flash-PXQU-pxq2 | `glm5-effort` | effort | 8192 | engine |
| Muse-Glimmer-30B-Abliterated-PXQ4, Muse-Glimmer-30B-Abliterated-Q8_0, Muse-Glimmer-30B-PXQ3, Muse-Glimmer-30B-PXQ4, Muse-Glimmer-30B-UD-Q4_K_XL | `muse-glimmer` | effort | 4096 | guard |
| dflash-kquant | `muse-glimmer` | effort | 2048 | guard |
| Llama-3.1-Nemotron-Nano-8B-v1-BF16, Llama-3.1-Nemotron-Nano-8B-v1-Q8_0, Llama-3.1-Nemotron-Nano-8B-v1-pxqn4-ladder-ldlq, Llama-3.1-Nemotron-Nano-8B-v1-pxqn4-ladder.skel | `nemotron-detailed` | system | 2048 | guard |
| Fusion-Coder-80-PXQ6-P100-262k, Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated.i1-Q3_K_M, Huihui-Qwen3-Coder-Next-abliterated.i1-Q5_K_M, Qwen2.5-VL-7B-Instruct-Q4_K_M, Qwen3-30B-A3B-Instruct-2507-abliterated-Q3_K_M, Qwen3-4B-Instruct-2507-Q4_K_M … (+12) | `none` | none | — | none |
| OrionLLM.GRM-3.2-Sky.Q8_0.mtp, PXA-Coder-35B-recovered-f16, PXA-Coder-35B-v2-PXQ4, PXA-Coder-35B-v2-f16 | `pxa-froggeric` | kwarg | 4096 | engine |
| PXA-Agent-9B-PXQ4, PXA-Agent-9B-f16, cliff9b-recovered-q8 | `pxa-froggeric` | kwarg | 2048 | engine |
| A3B-35B-PXQ4-fromQ3KS, AgentWorld-35B-PXQ4, Huihui-Qwable-3.6-27b-abliterated_F16-MTP, Huihui-Qwen3.6-27B-abliterated-MTP-Q3_K, Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q2_K, Huihui-Qwen3.6-35B-A3B-abliterated.i1-Q3_K_S … (+31) | `qwen-hybrid` | kwarg | 4096 | engine |
| Ornith-1.0-9B-heretic-MTP-Q8_0, Ornith-9B-Heretic-PXQ4, Ornith-9B-Unc-PXQ4-armA, Ornith-9B-Unc-PXQ4-armC, Ornith-9B-Unc-PXQ4-armF, Ornith-9B-Unc-PXQ4-imx … (+12) | `qwen-hybrid` | kwarg | 2048 | engine |
| PXA-Hemlock-124B-A5B-32GB, PXA-Hemlock-124B-A5B-PXQN4, PXA-Hemlock-124B-A5B-PXQN5, Qwen3.8-27B-PXQ-mix27, Qwen3.8-27B-PXQ2-GGUF, Qwen3.8-27B-PXQ2-attn4 … (+62) | `qwen38-effort` | kwarg+effort | 8192 | engine |
| Qwen3.8-27B-DFlash2-Q4_K_M | `qwen38-effort` | kwarg+effort | 4096 | engine |
| ggml-vocab-refact | `unknown` | unknown | — | none |

The DFlash drafter files carry their target's template; they are drafters, so the switch does
not matter for them.

## API

- `GET /api/thinking?model=PATH` | `?sid=SERVER` | `?port=N` → profile, saved settings, source, launch flags.
- `POST /api/thinking/override` `{"model": PATH, "thinking": {"mode": "on", "budget": 4096, "level": null}}`
  saves the per-model setting (`"thinking": null` deletes it).
- `GET /api/thinking/table` → the whole table.
- Launch/plan bodies accept `"thinking": {"mode": "auto|on|off", "budget": -1..262144 | null, "level": "..."}`.

## Changing the table

Edit `tools/gen_thinking_profiles.py`, run it (it rewrites `tools/pxa_thinking_profiles.json`), run
`python3 tests/test-pxa-thinking.py`, and commit both files.
