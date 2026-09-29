#!/usr/bin/env python3
"""gate-client.py PORT MODE TAG [ARG] -- the once-only ship gates for the mix27 file on its serving line.

MODES
  detq N       as det, with slot 1 idle (the established slot-pinned np2 gate).
  det N        N greedy-512 requests on the control prompt, pinned to slot 0; PASS iff one sha.
               With a server at -np 2, a background thread keeps slot 1 busy with a different prompt,
               so slot 0 decodes batched (the np2 half of the 12/12 rule).
  spread       10 identical prefill-only requests (n_predict 1, n_probs 2) on a ~3k prompt, slot 0;
               PASS iff ONE distinct token-0 top-1 probability (memory pxq-logit-spread-gate).
  needle CH    one needle at ~50% depth of a CH-character haystack (the in-dist corpus, repeated);
               PASS iff the passphrase appears in the greedy answer. Prints prompt_n, prefill and decode t/s.
"""
import sys, json, time, hashlib, threading, urllib.request
port, mode, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
arg = sys.argv[4] if len(sys.argv) > 4 else None
U = f'http://127.0.0.1:{port}/completion'
def post(d, timeout=3600):
    r = urllib.request.Request(U, json.dumps(d).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def chat(u): return f'<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
CTRL = chat('Write a detailed, well-structured essay on the history of the printing press and its effect on European science, politics and religion.')
BG = chat('Explain, step by step, how a hash table handles collisions, with examples in C.')
CORPUS = open('/mnt/cachetwo/models/qwen38-27b-mix/kld/qwen-chat-indist.txt').read()
base = dict(temperature=0, top_k=1, cache_prompt=False, seed=1)

def props():
    try: return json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/props', timeout=10))
    except Exception: return {}

if mode in ('det', 'detq'):
    n = int(arg or 12); nslots = props().get('total_slots', 1)
    stop = threading.Event()
    def bg():
        while not stop.is_set():
            try: post(dict(base, prompt=BG, n_predict=256, id_slot=1))
            except Exception: time.sleep(1)
    th = None
    if nslots >= 2 and mode == 'det':
        th = threading.Thread(target=bg, daemon=True); th.start(); time.sleep(2)
    shas = []
    for i in range(n):
        o = post(dict(base, prompt=CTRL, n_predict=512, id_slot=0))
        shas.append(hashlib.sha256(o['content'].encode()).hexdigest()[:16])
    stop.set()
    top = max(set(shas), key=shas.count)
    print(f'GATE {mode} {tag} np={nslots}{" slot1-busy" if (nslots >= 2 and mode == "det") else ""}: {shas.count(top)}/{n} identical, distinct={len(set(shas))} sha512 {sorted(set(shas))} '
          f'{"PASS" if len(set(shas)) == 1 else "FAIL"}', flush=True)
elif mode == 'spread':
    p = chat('Summarise the following text in three sentences.\n\n' + CORPUS[:12000])
    probs, toks = [], []
    for i in range(10):
        o = post(dict(base, prompt=p, n_predict=1, n_probs=2, id_slot=0))
        cp = o.get('completion_probabilities') or [{}]
        top = cp[0].get('probs') or cp[0].get('top_probs') or []
        if not top: print(f'GATE spread {tag}: no probs in response {json.dumps(cp[0])[:200]} FAIL'); sys.exit(0)
        probs.append(float(top[0].get('prob', top[0].get('probability')))); toks.append(top[0].get('tok_str', top[0].get('token')))
    d = sorted(set(f'{x:.9f}' for x in probs))
    print(f'GATE spread {tag}: n=10 top1={toks[0]!r} spread={max(probs)-min(probs):.9f} distinct={len(d)} '
          f'{"PASS" if len(d) == 1 and len(set(toks)) == 1 else "FAIL"}', flush=True)
elif mode == 'needle':
    ch = int(arg or 400000)
    hay = (CORPUS * (ch // len(CORPUS) + 1))[:ch]
    cut = hay.rfind('\n', 0, ch // 2) + 1
    NEEDLE = '\nIMPORTANT NOTE: the secret passphrase for the vault is "cobalt-heron-7291". Remember it.\n'
    p = chat('Read the following document carefully.\n\n' + hay[:cut] + NEEDLE + hay[cut:] +
             '\n\nQuestion: what is the secret passphrase for the vault mentioned in the document? Answer with the passphrase only.')
    o = post(dict(base, prompt=p, n_predict=24), timeout=7200); t = o['timings']
    ok = 'cobalt-heron-7291' in o['content']
    print(f'GATE needle {tag}: prompt_n={t["prompt_n"]} prefill {t["prompt_per_second"]:.1f} t/s decode {t["predicted_per_second"]:.2f} t/s '
          f'draft {t.get("draft_n")}/{t.get("draft_n_accepted")} answer={o["content"].strip()[:60]!r} {"PASS" if ok else "FAIL"}', flush=True)
