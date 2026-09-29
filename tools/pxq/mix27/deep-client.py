#!/usr/bin/env python3
"""deep-client.py PORT NCHARS -- one deep request (prompt ~NCHARS chars of the corpus, repeated if needed), n_predict 32 greedy.
Proves the 131k context is usable at depth, not just bootable; prints prompt tokens, prefill t/s, decode t/s at depth."""
import sys, json, urllib.request
port, n = int(sys.argv[1]), int(sys.argv[2])
t = open('/mnt/cachetwo/models/qwen38-27b-mix/kld/qwen-chat-indist.txt').read()
p = (t * (n // len(t) + 1))[:n]
r = urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(dict(prompt=p, n_predict=32, temperature=0, top_k=1, cache_prompt=False)).encode(), {'Content-Type': 'application/json'})
o = json.load(urllib.request.urlopen(r, timeout=3600)); tm = o['timings']
print(f"deep prompt_n={tm['prompt_n']} prefill {tm['prompt_per_second']:.1f} t/s decode@depth {tm['predicted_per_second']:.2f} t/s draft {tm.get('draft_n')}/{tm.get('draft_n_accepted')}")
