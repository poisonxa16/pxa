# Generates tools/pxa_thinking_profiles.json (the table PXA Control reads). Edit here, run, commit both.
import json
ON_ET = {"kwargs": {"enable_thinking": True}}
OFF_ET = {"kwargs": {"enable_thinking": False}}
FORCE_CLOSE = {"budget_tokens": 0}
F = []
def fam(**k): F.append(k)

# ---------- template-signature families (most specific first) ----------
fam(id="qwen38-effort", label="Qwen3.8 family (Swift, Flash-Next, Hemlock, Victoria, ThinkingCap)",
    mechanism="kwarg+effort", default="on",
    on=ON_ET, off=OFF_ET,
    levels={"key": "reasoning_effort", "values": ["low", "medium", "xhigh"], "default": "xhigh"},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine"},
    match={"template": {"all": ["enable_thinking", "Reasoning effort is set to", "<think>"]},
           "arch": ["qwen35", "qwen35moe", "qwen4exp", "dflash"],
           "name": r"(?i)qwen3\.8|swift|flash.?next|hemlock|victoria|thinkingcap",
           "requires": ["enable_thinking"]},
    source="disk template 08a763ee5e33 (Swift/Flash-Next/Hemlock/Victoria/Qwen3.8-27B) and a7e79f8fe37f; "
           "reasoning_effort accepts only xhigh|medium|low (raise_exception otherwise)")
fam(id="pxa-froggeric", label="PXA Coder / PXA Agent (Qwen3.6 froggeric template)",
    mechanism="kwarg", default="on", on=ON_ET, off=OFF_ET,
    fallback={"on": {"system_prefix": "<|think_on|>"}, "off": {"system_prefix": "<|think_off|>"}},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "engine"},
    match={"template": {"all": ["enable_thinking", "<|think_off|>"]}},
    source="disk templates 804515ec77a1, 5a988eae8c64: enable_thinking default true; <|think_off|>/<|think_on|> in the "
           "system prompt also switch it; thinking auto-closes after 2 consecutive tool failures")
fam(id="gemma4", label="Gemma 4 (12B, 26B-A4B, 31B; incl. MTP assistant)",
    mechanism="kwarg", default="off", on=ON_ET, off=OFF_ET,
    tags=["<|channel>thought", "<channel|>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "engine"},
    match={"template": {"all": ["<|channel>", "<|think|>", "enable_thinking"]},
           "arch": ["gemma4", "gemma4_mtp", "gemma4-assistant"], "requires": ["enable_thinking"]},
    source="disk templates ad0bceafd1fe, 5a538c54b0fe; engine models/templates/google-gemma-4-31B-it.jinja. "
           "Thinking on = <|think|> in the system turn and an open thought channel; off = the prompt ends with an "
           "already-closed empty <|channel>thought<channel|>. The engine's PXA_AUTO turns it OFF by default "
           "(common/chat.cpp common_chat_auto_reasoning)")
fam(id="gpt-oss", label="gpt-oss (20B, 120B; harmony format)",
    mechanism="effort", default="always", on={}, off=None,
    levels={"key": "reasoning_effort", "values": ["low", "medium", "high"], "default": "medium", "off_value": "low"},
    tags=["<|channel|>analysis", "<|end|>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "guard"},
    match={"template": {"all": ["<|channel|>", "reasoning_effort"]}, "arch": ["gpt-oss"]},
    source="engine models/templates/openai-gpt-oss-120b.jinja: 'Reasoning: medium' in the system message; "
           "the analysis channel cannot be switched off, low is the minimum. The engine's gpt-oss handler sets no "
           "thinking tags, so --reasoning-budget does not arm: the budget is a max_tokens guard")
fam(id="muse-glimmer", label="Muse Glimmer (recipient harmony)",
    mechanism="effort", default="always", on={}, off=None,
    levels={"key": "reasoning_strength", "values": ["low", "medium", "high"], "default": "high", "off_value": "low"},
    tags=[], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "guard"},
    match={"template": {"all": ["<atem:function_calls>", "reasoning_strength"]}, "arch": ["muse-glimmer"]},
    source="disk templates a508d59766d1, d6e113e32b08: 'Reasoning strength: <x>.' (free text, default high); "
           "reasoning goes to the 'self' recipient; no off switch")
fam(id="hunyuan-v3", label="Hunyuan V3 (hy_v3)",
    mechanism="effort", default="off", on={"kwargs": {"reasoning_effort": "high"}}, off={"kwargs": {"reasoning_effort": "no_think"}},
    levels={"key": "reasoning_effort", "values": ["low", "high"], "default": "high", "off_value": "no_think"},
    tags=["<think:opensource>", "</think:opensource>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "engine"},
    match={"template": {"all": ["reasoning_effort", "no_think", "opensource"]}, "arch": ["hy_v3"]},
    source="engine models/templates/tencent-Hunyuan-V3.jinja: reasoning_effort no_think (default) | low | high")
fam(id="glm5-effort", label="GLM 5.x Flash (glm5next)",
    mechanism="effort", default="always", on={}, off=None,
    levels={"key": "reasoning_effort", "values": ["low", "high", "max"], "default": "max", "off_value": "low"},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine"},
    match={"template": {"all": ["Reasoning Effort:", "reasoning_effort", "<think>"]}, "arch": ["glm5next"]},
    source="disk template 049a2a7f0788 (GLM 5.3 Flash): 'Reasoning Effort: Max' unless low|high; the generation "
           "prompt always opens <think>")
fam(id="cohere-reasoning", label="Cohere Command A Reasoning / Cohere2 MoE",
    mechanism="kwarg", default="on", on={"kwargs": {"reasoning": True}}, off={"kwargs": {"reasoning": False}},
    tags=["<|START_THINKING|>", "<|END_THINKING|>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "engine"},
    match={"template": {"all": ["<|START_THINKING|>", "reasoning_effort"]}, "arch": ["cohere2_moe"],
           "requires": ["reasoning"]},
    source="engine models/templates/Cohere2MoE.jinja: reasoning (bool), or reasoning_effort 'none' = off")
fam(id="command-r7b", label="Command R7B (cohere2)",
    mechanism="kwarg", default="on", on=ON_ET, off=OFF_ET,
    tags=["<|START_THINKING|>", "<|END_THINKING|>"], launch=True,
    budget={"suggested": 2048, "small": 2048, "enforce": "engine"},
    match={"template": {"all": ["<|START_THINKING|>", "enable_thinking"]},
           "arch": ["cohere2"], "name": r"(?i)r7b", "requires": ["enable_thinking"]},
    source="engine models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja: off = an empty "
           "<|START_THINKING|><|END_THINKING|> prefill")
fam(id="seed-oss", label="ByteDance Seed-OSS (native thinking budget)",
    mechanism="budget-kwarg", default="on", on={"kwargs": {"thinking_budget": -1}}, off={"kwargs": {"thinking_budget": 0}},
    budget_kwarg="thinking_budget",
    tags=["<seed:think>", "</seed:think>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "template",
            "note": "the model is trained to respect thinking_budget (multiples of 512 recommended); 0 = answer directly"},
    match={"template": {"any": ["<seed:think>", "thinking_budget"]}, "arch": ["seed_oss"]},
    source="Seed-OSS model card (chat_template_kwargs thinking_budget); not on this box: detection by arch, "
           "template signature from the published template")
fam(id="smollm3", label="SmolLM3",
    mechanism="kwarg", default="on", on=ON_ET, off=OFF_ET,
    fallback={"on": {"system_prefix": "/think"}, "off": {"system_prefix": "/no_think"}},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 2048, "small": 2048, "enforce": "engine"},
    match={"template": {"all": ["enable_thinking", "Reasoning Mode", "/no_think"]}, "arch": ["smollm3"]},
    source="engine models/templates/HuggingFaceTB-SmolLM3-3B.jinja: 'Reasoning Mode: /think|/no_think'")
fam(id="deepseek-hybrid", label="DeepSeek V3.1 / V3.2 / V4 (hybrid thinking)",
    mechanism="kwarg", default="template", on={"kwargs": {"thinking": True, "enable_thinking": True}},
    off={"kwargs": {"thinking": False, "enable_thinking": False}},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine"},
    match={"template": {"all": ["<\uff5cAssistant\uff5c>", "<think>"], "any": ["thinking_mode", "thinking is defined", "set thinking"]},
           "arch": ["deepseek2", "deepseek4", "deepseek4-dspark"], "name": r"(?i)v3\.[12]|terminus|v4",
           "requires": ["thinking"]},
    source="engine models/templates/deepseek-ai-DeepSeek-V3.1/V3.2/V4.jinja, disk da05440f0f0a (V4 Flash stub): "
           "the official templates read 'thinking' (default off); llama.cpp's copies also read enable_thinking, so "
           "both keys are sent")
fam(id="kimi-thinking", label="Kimi K2 Thinking (always thinks)",
    mechanism="always", default="always", on={}, off=FORCE_CLOSE,
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 8192, "enforce": "engine"},
    match={"template": {"all": ["<|im_assistant|>", "<think>", "<|tool_calls_section_begin|>"]},
           "arch": ["deepseek2", "kimi-k2"], "name": r"(?i)kimi.*think"},
    source="engine models/templates/Kimi-K2-Thinking.jinja; the engine's kimi_k2 handler sets the think tags, "
           "so 'off' = thinking_budget_tokens 0 (the sampler closes the block at once)")
fam(id="apriel-thinker", label="ServiceNow Apriel Thinker (always thinks, untagged)",
    mechanism="always", default="always", on={}, off=None,
    tags=["Here are my reasoning steps:", "[BEGIN FINAL RESPONSE]"], launch=False,
    budget={"suggested": 8192, "small": 4096, "enforce": "guard"},
    match={"template": {"any": ["Here are my reasoning steps", "[BEGIN FINAL RESPONSE]"]}},
    source="engine models/templates/Apriel-1.6-15b-Thinker-fixed.jinja, unsloth-Apriel-1.5.jinja")
fam(id="magistral", label="Magistral / Ministral Reasoning ([THINK])",
    mechanism="always", default="always", on={}, off=FORCE_CLOSE,
    tags=["[THINK]", "[/THINK]"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine",
            "note": "the default reasoning system prompt is injected only when the chat has no system message"},
    match={"template": {"all": ["[THINK]", "[SYSTEM_PROMPT]"]},
           "arch": ["mistral3", "mistral4", "llama"], "name": r"(?i)magistral|ministral.*reason"},
    source="engine models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja and the ministral_3 handler "
           "([THINK]/[/THINK] tags known to the engine)")
fam(id="granite-thinking", label="IBM Granite 3.2 / 3.3 (thinking kwarg)",
    mechanism="kwarg", default="off", on={"kwargs": {"thinking": True}}, off={"kwargs": {"thinking": False}},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 2048, "small": 2048, "enforce": "guard",
            "note": "the switch only changes the system prompt; the engine sees no think tags in the template"},
    match={"template": {"all": ["<|start_of_role|>", "elif thinking"]},
           "arch": ["granite", "granitemoe"], "name": r"(?i)3\.[23]", "requires": ["thinking"]},
    source="engine models/templates/ibm-granite-granite-3.3-2B-Instruct.jinja: 'thinking' adds the reasoning "
           "system prompt when the chat has no system message of its own")
fam(id="hermes4", label="Hermes 4 (thinking kwarg)",
    mechanism="kwarg", default="off", on={"kwargs": {"thinking": True}}, off={"kwargs": {"thinking": False}},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "guard"},
    match={"arch": ["llama", "qwen3", "qwen3moe"], "name": r"(?i)hermes.?4", "requires": ["thinking"]},
    source="Hermes 4 model card (chat_template_kwargs thinking=true); not on this box, heuristic only, "
           "cross-checked against the file's template")
fam(id="nemotron-detailed", label="Llama Nemotron v1 (Nano 4B/8B, Super 49B, Ultra 253B)",
    mechanism="system", default="off",
    on={"system_prefix": "detailed thinking on"}, off={"system_prefix": "detailed thinking off"},
    tags=["<think>", "</think>"], launch=False,
    budget={"suggested": 4096, "small": 2048, "enforce": "guard"},
    match={"arch": ["llama", "deci"], "name": r"(?i)nemotron.*(nano.?(4b|8b)|super.?49b|ultra.?253b)(?!.*v1\.5)"},
    source="model card (system prompt 'detailed thinking on|off'); disk 4b588fb15154 (Llama-3.1-Nemotron-Nano-8B-v1) "
           "has no switch in its template")
fam(id="nemotron-soft", label="Nemotron v1.5 / Nano v2 (/think, /no_think in the system prompt)",
    mechanism="system", default="on", on={"system_prefix": "/think"}, off={"system_prefix": "/no_think"},
    tags=["<think>", "</think>"], launch=False,
    budget={"suggested": 4096, "small": 2048, "enforce": "engine"},
    match={"arch": ["llama", "deci", "nemotron_h", "nemotron-h"], "name": r"(?i)nemotron.*(v1\.5|nano.?(9b|12b).?v2)"},
    source="model cards; not on this box (heuristic)")
fam(id="phi4-reasoning", label="Phi-4 reasoning / mini-reasoning (always thinks)",
    mechanism="always", default="always", on={}, off=FORCE_CLOSE,
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine",
            "note": "enforcement depends on the engine detecting <think> in this template; the max_tokens guard is the fallback"},
    match={"arch": ["phi3"], "name": r"(?i)reason"},
    source="model card; not on this box (heuristic)")
fam(id="minimax-m2", label="MiniMax M2 (interleaved thinking, always on)",
    mechanism="always", default="always", on={}, off=FORCE_CLOSE,
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 8192, "enforce": "engine"},
    match={"arch": ["minimax-m2"]},
    source="model card; arch minimax-m2 (template not on this box)")
fam(id="glm-hybrid", label="GLM-4.5 / 4.6 / 4.7 (enable_thinking, /nothink)",
    mechanism="kwarg", default="on", on=ON_ET, off=OFF_ET,
    fallback={"on": {}, "off": {"user_suffix": " /nothink"}},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine"},
    match={"template": {"all": ["[gMASK]", "enable_thinking", "<think>"]}, "arch": ["glm4moe", "glm-dsa"],
           "requires": ["enable_thinking"]},
    source="engine models/templates/GLM-4.6.jinja, GLM-4.7-Flash.jinja")
fam(id="qwen-hybrid", label="Qwen3 hybrid / Qwen3.5 / Qwen3.6 and derivatives (Ornith, Fusion, GRM, Laguna, ...)",
    mechanism="kwarg", default="on", on=ON_ET, off=OFF_ET,
    fallback={"on": {"user_suffix": " /think"}, "off": {"user_suffix": " /no_think"}},
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "engine"},
    match={"template": {"all": ["enable_thinking", "<think>"]},
           "arch": ["qwen3", "qwen3moe", "qwen3next", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen4exp",
                    "laguna", "hunyuan-moe", "mimo2", "internlm2"],
           "name_not": r"(?i)instruct-2507|thinking|coder|qwen3-(vl|next)-[0-9.]+b(-a[0-9]+b)?-instruct",
           "requires": ["enable_thinking"]},
    source="engine models/templates/Qwen-Qwen3-0.6B.jinja, Qwen3.5-4B.jinja; disk 2006cdc006d3, 2eb1912776fc, "
           "71dc5117fbb5, 78c9fe670a12, 783cead953c5, 0ef7932a1deb, 4bad019e10ea, a692e2293a8a, b8ffbabb7a71, "
           "f54c154d3de0, f0ae8663a876. /think and /no_think are honoured only by Qwen3 (2504) templates")
fam(id="deepseek-r1", label="DeepSeek R1 / R1 distills / QwQ / always-think <think> models",
    mechanism="always", default="always", on={}, off=FORCE_CLOSE,
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine"},
    match={"template": {"all": ["<\uff5cAssistant\uff5c>", "<think>"]},
           "arch": ["deepseek2", "qwen2", "llama", "qwen3", "qwen3moe", "qwen3next", "qwen3vl", "qwen3vlmoe",
                    "step35", "ernie4_5", "ernie4_5-moe", "bailingmoe2", "glm4", "olmo2", "olmo3"],
           "name": r"(?i)deepseek.?r1|r1.?distill|qwq|thinking|step.?3\.5|glm.?z1|ring"},
    source="engine models/templates/deepseek-ai-DeepSeek-R1-Distill-*.jinja, llama-cpp-deepseek-r1.jinja, "
           "Qwen-QwQ-32B.jinja, stepfun-ai-Step-3.5-Flash.jinja. 'Off' cannot remove the thought: the engine closes "
           "it at once (thinking_budget_tokens 0)")
# ---------- generic families (template inspection, step 3) ----------
fam(id="generic-kwarg", label="unknown model: template reads enable_thinking",
    mechanism="kwarg", default="on", on=ON_ET, off=OFF_ET, generic=True,
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "engine"},
    source="runtime template inspection")
fam(id="generic-thinking-kwarg", label="unknown model: template reads 'thinking'",
    mechanism="kwarg", default="off", on={"kwargs": {"thinking": True}}, off={"kwargs": {"thinking": False}}, generic=True,
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "guard"},
    source="runtime template inspection")
fam(id="generic-effort", label="unknown model: template reads reasoning_effort",
    mechanism="effort", default="always", on={}, off=None, generic=True,
    levels={"key": "reasoning_effort", "values": ["low", "medium", "high"], "default": "medium", "off_value": "low"},
    tags=[], launch=True,
    budget={"suggested": 4096, "small": 2048, "enforce": "guard"},
    source="runtime template inspection")
fam(id="generic-always", label="unknown model: the generation prompt always opens <think>",
    mechanism="always", default="always", on={}, off=FORCE_CLOSE, generic=True,
    tags=["<think>", "</think>"], launch=True,
    budget={"suggested": 8192, "small": 4096, "enforce": "engine"},
    source="runtime template inspection")
# ---------- no thinking (toggle hidden) ----------
fam(id="none", label="no thinking mode", mechanism="none", default="none", on=None, off=None,
    tags=[], launch=False, budget={"suggested": 0, "small": 0, "enforce": "none"},
    match={"arch": ["llama", "llama4", "gemma", "gemma2", "gemma3", "gemma3n", "qwen", "qwen2", "qwen2moe", "qwen2vl",
                    "qwen3", "qwen3moe", "qwen3next", "qwen3vl", "qwen3vlmoe", "mistral3", "command-r", "cohere2",
                    "phi2", "phi3", "deepseek2", "glm4", "chatglm", "granite", "granitemoe", "ernie4_5", "ernie4_5-moe",
                    "bailingmoe2", "dots1", "falcon", "mpt", "baichuan", "starcoder", "starcoder2", "bloom", "stablelm",
                    "gpt2", "gptj", "gptneox", "internlm2", "minicpm", "olmo", "olmo2", "dbrx", "arctic", "xverse",
                    "orion", "plamo", "codeshell", "openelm", "jais", "bitnet", "mellum", "grok", "deci", "mamba", "lfm2"],
           "name": r".*"},
    source="Llama 3.x/4, Qwen2.5, Qwen3-2507 Instruct, Qwen3-Coder, Gemma 2/3, Mistral/Devstral, Command-R, Kimi K2 "
           "Instruct, Phi-3.5/4, LFM2, GigaChat ... (engine models/templates/*, disk 4b588fb15154, 5f150cdf2e4a, "
           "7fd4a3534ed6, fa3f1e3beb64, b5b5ba98ad50, 6c510c876bda, 899134038234, f64773bd0298)")
fam(id="unknown", label="unknown model (no template, no rule): toggle hidden", mechanism="unknown", default="none",
    on=None, off=None, tags=[], launch=False, budget={"suggested": 0, "small": 0, "enforce": "none"},
    source="safe default")

# ---------- how on / off is done, in order of preference ----------
# template-kwarg      chat_template_kwargs (enable_thinking, thinking, reasoning, reasoning_effort, thinking_budget)
# soft-tag            a tag the model honours, in the last user message (/think, /no_think, /nothink) or the
#                     system prompt (<|think_off|>, "detailed thinking off", Reasoning Mode /no_think)
# empty-think-prefill the assistant turn starts with an empty, closed thought (<think>\n\n</think>\n\n or the
#                     family's own tokens; only the close if the generation prompt already opened it). The engine
#                     continues a final assistant message (examples/server/server-common.cpp, prefill_assistant)
# reasoning-budget    the engine's budget sampler closes the thought at once (thinking_budget_tokens 0 /
#                     --reasoning-budget 0); NOT yet measured live, so it comes after the prefill
# effort-level        the lowest effort level (the model still thinks a little)
# default             nothing to send: the model thinks by itself
# Preference: template kwarg, then soft tag, then the prefill. "Fallback" in the UI = the next method.
P_THINK = {"open": "<think>\n", "close": "\n</think>\n\n"}
P_THINK_TIGHT = {"open": "<think>", "close": "</think>"}
K, S, PF, RB, EF, DF = "template-kwarg", "soft-tag", "empty-think-prefill", "reasoning-budget", "effort-level", "default"
HOW = {
    "qwen38-effort": (P_THINK, [K], [K, PF]),
    "pxa-froggeric": (P_THINK, [K, S], [K, S, PF]),
    "gemma4": ({"open": "<|channel>thought\n", "close": "<channel|>"}, [K], [K, PF]),
    "gpt-oss": ({"open": "", "close": "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"},
                [EF], [PF, EF]),
    "muse-glimmer": (None, [EF], [EF]),
    "hunyuan-v3": ({"open": "<think:opensource>", "close": "</think:opensource>"}, [K], [K, PF]),
    "glm5-effort": (P_THINK_TIGHT, [EF], [PF, EF]),
    "cohere-reasoning": ({"open": "<|START_THINKING|>", "close": "<|END_THINKING|>"}, [K], [K, PF]),
    "command-r7b": ({"open": "<|START_THINKING|>", "close": "<|END_THINKING|>"}, [K], [K, PF]),
    "seed-oss": (None, [K], [K]),
    "smollm3": (P_THINK, [K, S], [K, S, PF]),
    "deepseek-hybrid": (P_THINK_TIGHT, [K], [K, PF]),
    "kimi-thinking": (P_THINK_TIGHT, [DF], [PF, RB]),
    "apriel-thinker": ({"open": "Here are my reasoning steps:\n", "close": "\n[BEGIN FINAL RESPONSE]\n"}, [DF], [PF]),
    "magistral": ({"open": "[THINK]", "close": "[/THINK]"}, [DF], [PF, RB]),
    "granite-thinking": (None, [K], [K]),
    "hermes4": (P_THINK, [K], [K, PF]),
    "nemotron-detailed": (None, [S], [S]),
    "nemotron-soft": (P_THINK, [S], [S, PF]),
    "phi4-reasoning": (P_THINK, [DF], [PF, RB]),
    "minimax-m2": (P_THINK, [DF], [PF, RB]),
    "glm-hybrid": (P_THINK_TIGHT, [K], [K, S, PF]),
    "qwen-hybrid": (P_THINK, [K, S], [K, S, PF]),
    "deepseek-r1": (P_THINK, [DF], [PF, RB]),
    "generic-kwarg": (P_THINK, [K], [K, PF]),
    "generic-thinking-kwarg": (P_THINK, [K], [K, PF]),
    "generic-effort": (None, [EF], [EF]),
    "generic-always": (P_THINK, [DF], [PF, RB]),
    "none": (None, [], []),
    "unknown": (None, [], []),
}
assert set(HOW) == {f["id"] for f in F}, set(HOW) ^ {f["id"] for f in F}
for f in F:
    pf, on_m, off_m = HOW[f["id"]]
    f["prefill"] = pf
    f["methods"] = {"on": on_m, "off": off_m}

T = {"version": 2,
     "note": "Per-model thinking profiles for PXA Control. Budgets are SUGGESTIONS (tokens of thinking per reply) for "
             "interactive chat, not measurements; 'small' applies to models of 10B parameters or fewer. enforce: "
             "engine = the engine's reasoning-budget sampler (per request thinking_budget_tokens, per server "
             "--reasoning-budget), template = the model's own budget kwarg, guard = only max_tokens can cap it.",
     "answer_reserve": 1024,
     "small_params_b": 10,
     "methods_note": "methods.on / methods.off: how the switch is done, most preferred first (template-kwarg, "
                     "soft-tag, empty-think-prefill, reasoning-budget, effort-level, default). The first one the "
                     "model's template allows is used; the UI's fallback option takes the next.",
     "families": F}
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pxa_thinking_profiles.json")
with open(OUT, "w") as f:
    json.dump(T, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(len(F), "families")
