#!/usr/bin/env python3
# Writes <vocab.gguf>.hfinp / .hfout for test-tokenizer-hf-parity from the model's HF tokenizer.json + tokenizer_config.json
# (tokenizers lib + the chat template rendered the way transformers' apply_chat_template does: jinja2 sandbox, trim/lstrip blocks).
# usage: gen-tokenizer-hf-parity.py <hf_model_dir> <vocab.gguf>
import json, os, sys
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

hf_dir, gguf = sys.argv[1], sys.argv[2]
tk = Tokenizer.from_file(os.path.join(hf_dir, 'tokenizer.json'))
cfg = json.load(open(os.path.join(hf_dir, 'tokenizer_config.json'), encoding='utf-8'))

def raise_exception(m):
    raise Exception(m)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals['raise_exception'] = raise_exception
env.filters['tojson'] = lambda x, **k: json.dumps(x, ensure_ascii=False, **k)
tmpl = env.from_string(cfg['chat_template'])

raw = [
    "Hello world",
    " Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "你好，世界！南北阁是一个大语言模型。",
    "Mixed 中文 and English, numbers 1234567890 and 3.14159.",
    "  leading spaces and    internal   runs\n\n\ttabs\t\tend  ",
    "emoji test 😀🚀 and accents: café naïve Zürich",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nreasoning here\n</think>\n\nanswer",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "URL https://example.com/a?b=c&d=e#frag and path /usr/local/bin",
    "\n\nStarts with newlines",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n Leading space after a special<|im_end|>\n",
]
special = ["Hello world", " Hello", "<|im_start|>system\nhi"]
chat_user = ["What is 2+2?", "Write a haiku about autumn.", "你好，请介绍一下你自己。", " starts with a space"]

cases = []
for t in raw:
    cases.append(('N', t, tk.encode(t, add_special_tokens=False).ids))
bos_id = tk.token_to_id(cfg['bos_token'])
for t in special:
    ids = tk.encode(t, add_special_tokens=True).ids
    # deliberate deviation: HF's post-processor doubles the BOS when the text already starts with it; the engine keeps one
    # (llama.cpp warns about that double as a likely mistake, and the server chat path relies on it)
    if len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
        ids = ids[1:]
    cases.append(('S', t, ids))
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]
for mode in ('C', 'D', 'T'):
    for u in (chat_user if mode != 'T' else ["What's the weather in Paris right now?", "今天北京天气怎么样？"]):
        msgs = ([{"role": "system", "content": "You are a helpful assistant."}] if mode == 'D' else []) + [{"role": "user", "content": u}]
        kw = {'tools': tools} if mode == 'T' else {}
        txt = tmpl.render(messages=msgs, add_generation_prompt=True, bos_token=cfg['bos_token'], eos_token=cfg['eos_token'], **kw)
        cases.append((mode, u, tk.encode(txt, add_special_tokens=False).ids))

with open(gguf + '.hfinp', 'w', encoding='utf-8', newline='\n') as f:
    for _, t, _ in cases:
        f.write(t + '\n__ggml_vocab_test__\n')
with open(gguf + '.hfout', 'w', encoding='utf-8', newline='\n') as f:
    for m, _, ids in cases:
        f.write(m + ' ' + ' '.join(map(str, ids)) + '\n')
print(f'{len(cases)} cases -> {gguf}.hfinp/.hfout')
