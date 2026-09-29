#!/usr/bin/env python3
"""mkqwenchat.py OUT [--chars N] [--seed S] -- in-distribution KLD/PPL corpus for Qwen3.8-27B.

Every document is rendered through the checkpoint's OWN chat template (chat_template.jinja from
the stock HF dir), so the scored tokens include the turn markers and the assistant-turn
positions a served model actually lives in. Mix (by chars): multi-turn chat (oasst1), instruction
following (dolly), math reasoning (gsm8k), code (github-code-clean: python/js/c++/go/rust), tool
calling (glaive-fc-v2, tools in the system turn), encyclopedic continuation (wikitext-2 TEST).
Docs are shuffled with a fixed seed so every -c 2048 window mixes styles. Deterministic.
Score with PXA_PPL_PARSE_SPECIAL=1 (else <|im_start|> etc. are scored as plain text).
"""
import sys, json, gzip, random, re
import pyarrow.parquet as pq
from jinja2 import Environment

OUT = sys.argv[1]
CHARS = int(sys.argv[sys.argv.index('--chars') + 1]) if '--chars' in sys.argv else 320000
SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 27
R = '/mnt/cacheone/datasets/pxq4-imatrix-clean/raw'
TPL = '/mnt/cachetwo/models/qwen38-27b-stock-vllm/chat_template.jinja'
env = Environment(); env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(Exception(m))
tpl = env.from_string(open(TPL).read())
rng = random.Random(SEED)

def render(msgs):
    return tpl.render(messages=msgs, add_generation_prompt=False, enable_thinking=False, bos_token='')

SHARE = {'oasst1': .25, 'dolly': .15, 'gsm8k': .15, 'code': .20, 'glaive': .10, 'wiki': .15}
docs = {k: [] for k in SHARE}

# oasst1: root prompt + top-ranked reply chain (up to 4 turns), English only
for line in gzip.open(f'{R}/oasst1.jsonl.gz', 'rt'):
    t = json.loads(line); p = t.get('prompt') or {}
    if p.get('lang') != 'en': continue
    msgs, node = [], p
    while node and len(msgs) < 4:
        msgs.append({'role': 'user' if node['role'] == 'prompter' else 'assistant', 'content': node['text']})
        reps = [r for r in node.get('replies', []) if not r.get('deleted')]
        if not reps: break
        reps.sort(key=lambda r: (r.get('rank') is None, r.get('rank') or 0))
        node = reps[0]
    if len(msgs) >= 2 and msgs[-1]['role'] == 'assistant':
        docs['oasst1'].append(render(msgs))
    if len(docs['oasst1']) > 3000: break

for line in open(f'{R}/dolly-15k.jsonl'):
    d = json.loads(line)
    u = d['instruction'] + (('\n\n' + d['context']) if d.get('context') else '')
    docs['dolly'].append(render([{'role': 'user', 'content': u}, {'role': 'assistant', 'content': d['response']}]))

g = pq.read_table(f'{R}/gsm8k-train.parquet').to_pylist()
for d in g[:4000]:
    a = re.sub(r'<<[^>]*>>', '', d['answer']).replace('####', 'The answer is')
    docs['gsm8k'].append(render([{'role': 'user', 'content': d['question']}, {'role': 'assistant', 'content': a}]))

pf = pq.ParquetFile(f'{R}/github-code-clean-shard0.parquet')
LANGS = {'Python', 'JavaScript', 'C++', 'Go', 'Rust', 'TypeScript', 'Java'}
for rb in pf.iter_batches(batch_size=2000, columns=['code', 'language', 'path']):
    for d in rb.to_pylist():
        c = d['code']
        if d['language'] not in LANGS or not (1500 < len(c) < 6000): continue
        cut = c.find('\n', len(c) // 3)
        u = f"Here is the start of `{d['path'].split('/')[-1]}` ({d['language']}). Finish the file.\n\n```\n{c[:cut]}\n```"
        docs['code'].append(render([{'role': 'user', 'content': u}, {'role': 'assistant', 'content': '```\n' + c[cut + 1:] + '\n```'}]))
    if len(docs['code']) > 1500: break

gl = json.load(open(f'{R}/glaive-fc-v2.json'))
for d in gl[:6000]:
    parts = re.split(r'\n*(USER|ASSISTANT|FUNCTION RESPONSE): ', d['chat'])
    msgs = [{'role': 'system', 'content': d['system'].replace('SYSTEM: ', '', 1)}]
    for i in range(1, len(parts) - 1, 2):
        role = {'USER': 'user', 'ASSISTANT': 'assistant', 'FUNCTION RESPONSE': 'user'}[parts[i]]
        txt = parts[i + 1].replace('<|endoftext|>', '').strip()
        if parts[i] == 'FUNCTION RESPONSE': txt = 'Tool result: ' + txt
        if msgs and msgs[-1]['role'] == role: msgs[-1]['content'] += '\n' + txt
        else: msgs.append({'role': role, 'content': txt})
    if len(msgs) >= 3 and msgs[-1]['role'] == 'assistant' and '<functioncall>' in d['chat']:
        docs['glaive'].append(render(msgs))

text = open('/mnt/user/models/ppl-corpus/wikitext-2-raw/wiki.test.raw').read()
cur = ''
for p in [p for p in re.split(r'\n\s*\n', text) if p.strip()]:
    cur += p.strip() + '\n\n'
    if len(cur) >= 2400:
        cut = cur.find(' ', len(cur) // 4)
        docs['wiki'].append(render([{'role': 'user', 'content': 'Continue this encyclopedia article in the same register.\n\n' + cur[:cut].strip()},
                                    {'role': 'assistant', 'content': cur[cut:].strip()}]))
        cur = ''

out, stats = [], {}
for k, share in SHARE.items():
    pool = docs[k][:]; rng.shuffle(pool); n = 0; budget = CHARS * share
    for d in pool:
        if n >= budget: break
        out.append(d); n += len(d)
    stats[k] = (len(docs[k]), n)
rng.shuffle(out)
open(OUT, 'w', encoding='utf-8').write(''.join(out))
print('docs', len(out), 'chars', sum(map(len, out)), {k: v for k, v in stats.items()}, '->', OUT)
