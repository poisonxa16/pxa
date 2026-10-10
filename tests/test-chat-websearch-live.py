#!/usr/bin/env python3
"""live check of the chat web tools: built-in provider (no local services), SearXNG via settings, one error path."""
import json, os, sys
os.environ.pop("PXA_CHAT_SEARCH_URL", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from pxa_chat import search, tools

# 1. built-in: default settings, no searxng, no reader
ctx = tools.Ctx(None, "t", search_cfg=search.config(None))
assert ctx.search_cfg["provider"] == "builtin" and not ctx.search_cfg["reader_url"]
ok, out = tools.execute("web_search", {"query": "llama.cpp pascal gpu", "n": 3}, ctx)
res = json.loads(out)["results"] if ok else []
print("builtin search", ok, len(res), res[0]["url"] if res else out[:150])
assert ok and res and all(r["url"].startswith("http") for r in res)
ok2, txt = tools.execute("web_fetch", {"url": "https://example.com/"}, ctx)
print("builtin fetch", ok2, len(txt), txt[:60].replace("\n", " "))
assert ok2 and "documentation examples" in txt

# 2. SearXNG through the saved setting
cfg = search.config({"provider": "searxng", "searxng_url": "http://127.0.0.1:8080"})
ok, out = tools.execute("web_search", {"query": "pxa llama.cpp pascal"}, tools.Ctx(None, "t", search_cfg=cfg))
j = json.loads(out)
print("searxng", ok, len(j["results"]), "note:", j.get("note", "")); assert ok and j["results"] and not j.get("note")

# 3. error path: dead SearXNG falls back to built-in with a plain message; dead + no net would raise a short error
cfg = search.config({"provider": "searxng", "searxng_url": "http://127.0.0.1:9"})
ok, out = tools.execute("web_search", {"query": "pxa llama.cpp pascal"}, tools.Ctx(None, "t", search_cfg=cfg))
j = json.loads(out); print("fallback", ok, j.get("note", "")[:90]); assert ok and "failed" in j["note"] and j["results"]
try:
    search.brave("x", 3, "badkey"); assert 0
except Exception as e:
    print("bad key ->", search._explain(e))
print("PASS")
