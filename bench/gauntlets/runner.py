#!/usr/bin/env python3
# Generalized gauntlet runner: PORT SET_JSON OUT_JSONL. Delivery-guard, records timings+MTP.
# NO-THINK BY DEFAULT: every model here is evaluated in the mode it is actually served in
# (agent mode, enable_thinking=false). Set GAUNTLET_THINKING=1 to score a thinking run instead;
# never mix the two in one comparison, and never compare a thinking run to a no-think one.
import json, os, time, urllib.request, sys
PORT=sys.argv[1]; SET=sys.argv[2]; OUT=sys.argv[3]
LOG=OUT.replace("results_","run_").replace(".jsonl",".log")
URL=f"http://127.0.0.1:{PORT}/v1/chat/completions"
SAMP={"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,"repeat_penalty":1.0}
def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"; open(LOG,"a").write(line+"\n"); print(line,flush=True)
def call(messages, max_tokens, think=THINK, tools=None):
    body={"messages":messages,"max_tokens":max_tokens,"chat_template_kwargs":{"enable_thinking":bool(think)}, **SAMP}
    if tools: body["tools"]=tools
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"content-type":"application/json"})
    with urllib.request.urlopen(req,timeout=900) as r: d=json.load(r)
    msg=d["choices"][0]["message"]; tim=d.get("timings",{}) or {}
    return (msg.get("content") or ""), (msg.get("reasoning_content") or ""), (msg.get("tool_calls") or []), \
           d["choices"][0].get("finish_reason"), tim
raw=json.load(open(SET)); chs = raw if isinstance(raw,list) else raw.get("challenges",raw.get("items",raw.get("probes",[])))
open(OUT,"w").close(); open(LOG,"w").close()
log(f"{SET.split('/')[-1]}: {len(chs)} items -> {OUT} (model, {'THINKING' if THINK else 'NO-THINK'}, temp0.6)")
for i,ch in enumerate(chs):
    pid=ch.get("id",f"item-{i}"); prompt=ch.get("prompt") or ch.get("content") or ""
    tools=None
    if ch.get("needs_tools") and ch.get("tools_spec"):
        try: tools=json.loads(ch["tools_spec"])
        except Exception: pass
    rec={"id":pid,"category":ch.get("category"),"prompt":prompt,"gold":ch.get("gold")}
    t0=time.time()
    try:
        c_raw, reasoning, tcalls, finish, tim = call([{"role":"user","content":prompt}], 20000, True, tools)
        content=c_raw; guard=None
        if (not c_raw.strip()) and (not tcalls):
            try:
                c2,_,_,_,_ = call([{"role":"user","content":prompt+"\n\nYou have reasoned enough. Output ONLY your final answer now."}],4000,False)
                if c2.strip(): content=c2; guard="reprompt"
                elif reasoning.strip(): content=reasoning[-3000:]; guard="harvest"
            except Exception:
                if reasoning.strip(): content=reasoning[-3000:]; guard="harvest_err"
        rec.update({"content":content,"reasoning_len":len(reasoning),"tool_calls":tcalls,"finish":finish,"guard":guard,
                    "decode_tps":round(tim.get("predicted_per_second",0),1),"pred_tok":tim.get("predicted_n"),
                    "content_head":content[:400],"wall_s":round(time.time()-t0,1)})
        da=tim.get("draft_n_accepted"); dn=tim.get("draft_n"); rec["mtp_accept"]=round(da/dn,2) if (da is not None and dn) else None
    except Exception as e:
        rec["error"]=str(e)[:200]; rec["wall_s"]=round(time.time()-t0,1)
    open(OUT,"a").write(json.dumps(rec)+"\n")
    log(f"  [{i+1}/{len(chs)}] {pid} dec={rec.get('decode_tps')} mtp={rec.get('mtp_accept')} final_len={len(rec.get('content','') or '')} tc={len(rec.get('tool_calls') or [])} guard={rec.get('guard')} err={rec.get('error','-')}")
log("SET DONE")
