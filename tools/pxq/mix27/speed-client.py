#!/usr/bin/env python3
"""speed-client.py PORT REPS TAG -- warm-up 60 s, then decode (control, repetition) and 25k prefill brackets."""
import sys, json, time, hashlib, urllib.request, statistics
port, reps, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
U = f'http://127.0.0.1:{port}/completion'
def post(d):
    r = urllib.request.Request(U, json.dumps(d).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=1800))
def chat(u): return f'<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
CTRL = chat('Write a detailed, well-structured essay on the history of the printing press and its effect on European science, politics and religion.')
REP = chat('Write a Python module that defines 40 small functions named f1..f40, each returning its own index times two, with a docstring on every function and a test for each.')
base = dict(temperature=0, top_k=1, n_predict=512, cache_prompt=False, seed=1)
t0 = time.time()
while time.time() - t0 < 60:
    post(dict(base, prompt=CTRL, n_predict=128))
long = open('/mnt/cachetwo/models/qwen38-27b-mix/kld/qwen-chat-indist.txt').read()[:95000]
for cls, p in (('control', CTRL), ('repetition', REP)):
    rs, shas, dr = [], set(), []
    for i in range(reps):
        o = post(dict(base, prompt=p)); t = o['timings']
        rs.append(t['predicted_per_second']); shas.add(hashlib.sha256(o['content'].encode()).hexdigest()[:16])
        dr.append((t.get('draft_n'), t.get('draft_n_accepted')))
    print(f'{tag} decode {cls}: t/s {" ".join("%.2f" % x for x in rs)} median {statistics.median(rs):.2f} n={t["predicted_n"]} sha512 {",".join(sorted(shas))} draft {dr}', flush=True)
rs = []
for i in range(reps):
    o = post(dict(base, prompt=long, n_predict=1)); t = o['timings']
    rs.append(t['prompt_per_second']); n = t['prompt_n']
print(f'{tag} prefill {n} tok: t/s {" ".join("%.1f" % x for x in rs)} median {statistics.median(rs):.1f}', flush=True)
