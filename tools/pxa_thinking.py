"""Per-model thinking (reasoning) switch and thinking-token budget for PXA Control.

Every model family turns thinking on and off differently: a chat-template kwarg (enable_thinking,
thinking, reasoning), an effort level (reasoning_effort, reasoning_strength), a line in the system
prompt ("detailed thinking on", /no_think), or not at all (DeepSeek R1 always thinks; Llama 3 never
does). This module maps ONE switch (on / off / auto) and ONE budget (tokens of thinking) onto the
right mechanism for the model at hand, from tools/pxa_thinking_profiles.json.

Detection order (detect()):
  1. an explicit template signature (substrings of tokenizer.chat_template) in the table;
  2. the GGUF general.architecture + general.name heuristic in the table, cross-checked against the
     template when the file has one (a rule that needs enable_thinking is skipped if the template
     never reads it);
  3. runtime inspection of an unknown template (which kwargs it reads, whether the generation
     prompt opens <think>), giving a generic profile;
  4. the safe default: 'unknown', toggle hidden.

How on/off is done (methods, per family in the table, most preferred first): template-kwarg, then
soft-tag (/think, /no_think, a system tag), then empty-think-prefill (the answer starts with an empty,
closed thought), then reasoning-budget (the engine closes the thought at once: not yet measured live),
then effort-level (the lowest effort). The first one the model's template allows is used; the fallback
option takes the next one.

Where it is applied:
  * per request (the Control chat): apply_request() rewrites an OpenAI chat body: chat_template_kwargs,
    a system-prompt line, a /no_think suffix, an assistant prefill, thinking_budget_tokens (the engine's
    per-request budget, examples/server/server-common.cpp), and a max_tokens guard;
  * per server (Launch): launch_args() gives llama-server flags: --reasoning on|off,
    --chat-template-kwargs JSON, --reasoning-budget N.

stdlib only. verify_render() uses jinja2 when it is installed (it is optional) to prove a profile
against the model's own template: it renders the generation prompt with the switch on and off.
"""

import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.environ.get("PXA_THINKING_PROFILES") or os.path.join(HERE, "pxa_thinking_profiles.json")
MODES = ("auto", "on", "off")
BUDGET_MAX = 262144
LEVEL_RE = re.compile(r"^[A-Za-z_]{1,16}$")
# effort selector (default off: effort None = unchanged behaviour). Each level -> a thinking-token
# budget; "off" switches thinking off, "max" = unlimited (-1).
EFFORTS = ("off", "low", "medium", "high", "max")
EFFORT_BUDGET = {"off": 0, "low": 1024, "medium": 4096, "high": 16384, "max": -1}
# what the client may say about thinking on its own (any of these = the request has its own setting)
_CLIENT_KW = ("enable_thinking", "thinking", "reasoning", "reasoning_effort", "reasoning_strength",
              "thinking_budget", "thinking_mode")
_CLIENT_TOP = ("reasoning_effort", "thinking_budget_tokens", "reasoning", "thinking", "reasoning_budget")
_SOFT_RE = re.compile(r"(^|\s)/(no_?think|think)\b")
_table_cache = {}
K, S, PF, RB, EF, DF = "template-kwarg", "soft-tag", "empty-think-prefill", "reasoning-budget", "effort-level", "default"
METHOD_LABEL = {K: "template kwarg", S: "soft tag", PF: "empty-think prefill", RB: "reasoning budget 0",
                EF: "lowest effort", DF: "model default"}


def load_table(path=None):
    p = path or TABLE_PATH
    st = os.stat(p)
    key = (p, st.st_mtime_ns, st.st_size)
    if key not in _table_cache:
        with open(p, encoding="utf-8") as f:
            t = json.load(f)
        t["_by_id"] = {x["id"]: x for x in t["families"]}
        _table_cache.clear()
        _table_cache[key] = t
    return _table_cache[key]


# ---------------------------------------------------------------------------------------------
# template inspection (static: no jinja needed)
# ---------------------------------------------------------------------------------------------
_STMT_RE = re.compile(r"\{%-?(.*?)-?%\}", re.S)
_EXPR_RE = re.compile(r"\{\{-?(.*?)-?\}\}", re.S)
_VAR_COND = {
    "enable_thinking": re.compile(r"\benable_thinking\b"),
    "thinking": re.compile(r"(?<![\w.'\"])thinking\b(?!['\"])(?!\s*=)"),
    "reasoning": re.compile(r"(?<![\w.'\"])reasoning\b(?!['\"])(?!\s*=)"),
    "reasoning_effort": re.compile(r"\breasoning_effort\b"),
    "reasoning_strength": re.compile(r"\breasoning_strength\b"),
    "thinking_budget": re.compile(r"\bthinking_budget\b"),
}


def template_facts(tmpl):
    """What a chat template reads and emits, from its source text."""
    if not tmpl:
        return {"has_template": False}
    stmts = " ".join(_STMT_RE.findall(tmpl) + _EXPR_RE.findall(tmpl))   # {% if x %} and {{ 'a' if x }}
    reads = sorted(k for k, rx in _VAR_COND.items() if rx.search(stmts))
    # 'thinking' as a variable, not a message field (message.thinking / msg['thinking'] / ns.thinking)
    if "thinking" in reads and not re.search(r"(\bif|\belif|\band|\bor|\bnot|set\s+\w+\s*=)\s+thinking\b|\bthinking\s+is\s+(not\s+)?(defined|undefined|true|false)", stmts):
        reads.remove("thinking")
    if "reasoning" in reads and not re.search(r"(\bif|\belif|\bnot)\s+reasoning\b|\breasoning\s+is\s+(not\s+)?(defined|undefined)", stmts):
        reads.remove("reasoning")
    gp = tmpl.rfind("add_generation_prompt")
    tail = tmpl[gp:] if gp >= 0 else ""
    effort_values = []
    m = re.search(r"reasoning_effort[^\n]{0,80}?\bin\s*[\[(]([^\])]{1,120})[\])]", tmpl)
    if m:
        effort_values = re.findall(r"['\"]([A-Za-z_]{1,16})['\"]", m.group(1))
    return {
        "has_template": True,
        "sha1": hashlib.sha1(tmpl.encode("utf-8", "replace")).hexdigest()[:12],
        "reads": reads,
        "think_tag": "<think>" in tmpl,
        "gen_opens_think": "<think>" in tail and "</think>" not in tail.split("<think>")[-1][:40],
        "gen_has_think": "<think>" in tail,
        "soft_switch": "/no_think" in tmpl or "/nothink" in tmpl,
        "effort_values": effort_values,
    }


def _sig_ok(sig, tmpl):
    if not tmpl or not sig:
        return False
    if any(s not in tmpl for s in sig.get("all", [])):
        return False
    if sig.get("any") and not any(s in tmpl for s in sig["any"]):
        return False
    if any(s in tmpl for s in sig.get("none", [])):
        return False
    return True


def _name_blob(name, basename=None, path=None):
    return " ".join(x for x in (name, basename, os.path.basename(path) if path else None) if x)


def inspect_generic(tmpl, table=None):
    """Step 3: a generic family id for a template no rule knows."""
    f = template_facts(tmpl)
    if not f.get("has_template"):
        return None, f
    r = f["reads"]
    if "enable_thinking" in r:
        return "generic-kwarg", f
    if "thinking" in r:
        return "generic-thinking-kwarg", f
    if "reasoning_effort" in r or "reasoning_strength" in r:
        return "generic-effort", f
    if f["gen_opens_think"]:
        return "generic-always", f
    return "none", f


def params_b(params=None, size_label=None):
    """Total parameters in billions, from the tensor count or general.size_label ('27B', '124B-A5B', '8x7B')."""
    if params:
        return params / 1e9
    if size_label:
        m = re.match(r"^\s*(?:(\d+)x)?([\d.]+)\s*([BMKT])", str(size_label), re.I)
        if m:
            n = float(m.group(2)) * (int(m.group(1)) if m.group(1) else 1)
            return n * {"B": 1, "M": 1e-3, "K": 1e-6, "T": 1e3}[m.group(3).upper()]
    return None


def suggested_budget(fam, params=None, size_label=None, table=None):
    t = table or load_table()
    b = fam.get("budget") or {}
    pb = params_b(params, size_label)
    if pb is not None and pb <= t.get("small_params_b", 10) and b.get("small") is not None:
        return int(b["small"])
    return int(b.get("suggested") or 0)


def detect(arch=None, name=None, template=None, basename=None, path=None, params=None, size_label=None,
           table=None):
    """-> the resolved profile for one model (a dict the UI and apply_* use)."""
    t = table or load_table()
    fams = t["families"]
    by = t["_by_id"]
    nb = _name_blob(name, basename, path)
    chosen, how, conf, notes = None, None, None, []
    facts = template_facts(template)
    # 1. template signature
    if template:
        for fm in fams:
            sig = (fm.get("match") or {}).get("template")
            if sig and _sig_ok(sig, template):
                chosen, how, conf = fm, "template signature", "high"
                break
    # 2. architecture + name
    if chosen is None and arch:
        for fm in fams:
            m = fm.get("match") or {}
            if arch not in (m.get("arch") or []):
                continue
            if m.get("name") and not re.search(m["name"], nb or ""):
                continue
            if m.get("name_not") and re.search(m["name_not"], nb or ""):
                continue
            if template and any(s not in template for s in m.get("requires", [])):
                notes.append(f"rule '{fm['id']}' fits {arch} but this file's template does not read "
                             f"{', '.join(m['requires'])}: skipped")
                continue
            if fm["id"] == "none" and template:
                gid, _ = inspect_generic(template, t)
                if gid not in (None, "none"):
                    notes.append(f"{arch} usually has no thinking mode, but this file's template has one")
                    continue
            chosen, how, conf = fm, "architecture + name", "medium" if template is None else "high"
            break
    # 3. runtime inspection of the template
    if chosen is None and template:
        gid, _ = inspect_generic(template, t)
        chosen, how, conf = by[gid], "template inspection", "low" if gid != "none" else "medium"
    # 4. safe default
    if chosen is None:
        chosen, how, conf = by["unknown"], "default", "none"
    return resolve(chosen, how, conf, facts, notes, params=params, size_label=size_label, table=t,
                   arch=arch, name=name or basename, template=template)


def _have_jinja():
    try:
        import jinja2  # noqa: F401
        return True
    except ImportError:
        return False


def prefill_text(fm, template=None, facts=None):
    """The text the assistant turn must start with so the model skips its thought, or None (no prefill
    for this family) or "" (not needed: the generation prompt already closes the thought). Measured on
    the model's own template when jinja2 is there (rendered with enable_thinking false, as the engine
    does for a prefilled request), else read from the template source."""
    pf = fm.get("prefill")
    if not pf:
        return None
    o, c = pf.get("open") or "", pf["close"]
    if not template:
        return o + c
    if _have_jinja():
        try:
            r = render_prompt(template, {"messages": [{"role": "user", "content": "Hi"}],
                                         "chat_template_kwargs": {"enable_thinking": False}}).rstrip()
            if c.strip() and r.endswith(c.strip()):
                return ""
            if o.strip() and r.endswith(o.strip()):
                return c
            return o + c
        except Exception:       # noqa: BLE001 - fall back to reading the source
            pass
    facts = facts or template_facts(template)
    if "enable_thinking" in (facts.get("reads") or []) and fm.get("off") and (fm["off"].get("kwargs") or {}):
        return ""               # the kwarg closes it already; without a render we cannot prove otherwise
    gp = template.rfind("add_generation_prompt")
    tail = template[gp:] if gp >= 0 else ""
    if o.strip() and o.strip() in tail and c.strip() not in tail.split(o.strip())[-1][:80]:
        return c
    return o + c


def _soft(fm, mode):
    fb = (fm.get("fallback") or {}).get(mode)
    if fb:
        return fb
    if fm.get("mechanism") == "system":
        return fm.get(mode)
    return None


def available_methods(fm, mode, facts, pre):
    """The family's methods for on|off that this file allows, in order of preference."""
    out, why = [], []
    lv = fm.get("levels") or {}
    act = fm.get(mode) or {}
    for m in (fm.get("methods") or {}).get(mode) or []:
        if m == K:
            keys = set((act.get("kwargs") or {}).keys())
            if not keys:
                continue
            reads = set(facts.get("reads") or []) if facts.get("has_template") else None
            if reads is not None and not (keys & reads):
                why.append(f"the template does not read {', '.join(sorted(keys))}: {METHOD_LABEL[K]} skipped")
                continue
        elif m == S:
            if not _soft(fm, mode):
                continue
        elif m == PF:
            if mode != "off" or not pre:
                if pre == "" and mode == "off":
                    why.append("the generation prompt already closes the thought: no prefill needed")
                continue
        elif m == RB:
            if mode != "off" or (fm.get("off") or {}).get("budget_tokens") is None:
                continue
        elif m == EF:
            if not lv or (mode == "off" and not lv.get("off_value")):
                continue
        out.append(m)
    return out, why


def resolve(fm, how, conf, facts, notes, params=None, size_label=None, table=None, arch=None, name=None,
            template=None):
    t = table or load_table()
    mech = fm["mechanism"]
    levels = json.loads(json.dumps(fm.get("levels"))) if fm.get("levels") else None
    if levels and fm.get("generic") and facts.get("effort_values"):
        levels["values"] = facts["effort_values"]
        levels["default"] = levels["values"][-1]
    if levels and fm.get("generic") and "reasoning_strength" in (facts.get("reads") or []) \
            and "reasoning_effort" not in (facts.get("reads") or []):
        levels["key"] = "reasoning_strength"
    supported = mech not in ("none", "unknown")
    pre = prefill_text(fm, template, facts) if supported else None
    m_on, w_on = available_methods(fm, "on", facts, pre)
    m_off, w_off = available_methods(fm, "off", facts, pre)
    notes = list(notes) + [w for w in w_on + w_off if w not in notes]
    can_off = any(m in (K, S, PF, RB) for m in m_off)
    budget = suggested_budget(fm, params, size_label, t)
    enforce = (fm.get("budget") or {}).get("enforce", "none")
    p = {
        "family": fm["id"], "label": fm["label"], "mechanism": mech, "supported": supported,
        "default": fm.get("default"), "can_disable": can_off, "levels": levels,
        "partial_off": bool(m_off) and m_off[0] in (PF, RB),
        "methods": {"on": m_on, "off": m_off}, "on_method": m_on[0] if m_on else None,
        "off_method": m_off[0] if m_off else None, "prefill": pre or None,
        "prefill_tags": [(fm.get("prefill") or {}).get("open") or "", (fm.get("prefill") or {}).get("close") or ""]
        if fm.get("prefill") else None,
        "budget": {"suggested": budget, "enforce": enforce, "note": (fm.get("budget") or {}).get("note"),
                   "is_suggestion": True},
        "tags": fm.get("tags") or [], "launch": bool(fm.get("launch")),
        "on": fm.get("on"), "off": fm.get("off"), "fallback": fm.get("fallback"),
        "budget_kwarg": fm.get("budget_kwarg"),
        "detected_by": how, "confidence": conf, "notes": list(notes), "source": fm.get("source"),
        "template": {k: v for k, v in facts.items() if k in ("has_template", "sha1", "reads", "gen_opens_think")},
        "arch": arch, "name": name,
    }
    p["summary"] = summarize(p)
    return p


def summarize(p):
    m = p["mechanism"]
    if not p["supported"]:
        return "no thinking switch for this model: the toggle is hidden"
    lv = p.get("levels")
    if m == "kwarg+effort":
        s = f"chat_template_kwargs {json.dumps(p['on'].get('kwargs'))} / {json.dumps(p['off'].get('kwargs'))}, effort {lv['key']} = {'|'.join(lv['values'])}"
    elif m == "kwarg":
        s = f"chat_template_kwargs {json.dumps((p['on'] or {}).get('kwargs'))} / {json.dumps((p['off'] or {}).get('kwargs'))}"
    elif m == "effort":
        s = f"{lv['key']} = {'|'.join(lv['values'])}" + (f", off = {lv['off_value']}" if lv.get("off_value") else "")
    elif m == "system":
        s = f"system prompt line '{p['on'].get('system_prefix')}' / '{p['off'].get('system_prefix')}'"
    elif m == "budget-kwarg":
        s = f"chat_template_kwargs {p['budget_kwarg']} = N (0 = off, -1 = unlimited)"
    elif m == "always":
        s = "always thinks"
    else:
        s = m
    d = {"on": "on by default", "off": "off by default", "always": "always on", "template": "template default"}.get(p["default"], "")
    if p["family"] in ("gemma4",):
        d = "off by default (engine PXA_AUTO)"
    mo = p.get("methods") or {}
    how = []
    if mo.get("off"):
        how.append("off via " + ", then ".join(METHOD_LABEL[x] for x in mo["off"]))
    else:
        how.append("cannot be switched off")
    if p.get("prefill"):
        how.append("prefill " + json.dumps(p["prefill"]))
    return s + (f"; {d}" if d else "") + "; " + "; ".join(how)


# ---------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------
class Invalid(ValueError):
    pass


def clean_settings(x):
    """{"mode": auto|on|off, "budget": int|None, "level": str|None} or Invalid."""
    if x in (None, "", {}):
        return {"mode": "auto", "budget": None, "level": None}
    if not isinstance(x, dict):
        raise Invalid("thinking must be an object {mode, budget, level}")
    mode = x.get("mode", "auto") or "auto"
    if mode not in MODES:
        raise Invalid("thinking mode must be auto, on or off")
    b = x.get("budget")
    if b in ("", None):
        b = None
    else:
        if isinstance(b, bool):
            raise Invalid("thinking budget must be a whole number")
        try:
            b = int(b)
        except (TypeError, ValueError):
            raise Invalid("thinking budget must be a whole number")
        if not -1 <= b <= BUDGET_MAX:
            raise Invalid(f"thinking budget must be between -1 (unlimited) and {BUDGET_MAX}")
    lv = x.get("level")
    if lv in ("", None):
        lv = None
    elif not isinstance(lv, str) or not LEVEL_RE.match(lv):
        raise Invalid("thinking effort level: 1-16 of A-Z a-z _")
    eff = x.get("effort")
    if eff in ("", None):
        eff = None
    elif eff not in EFFORTS:
        raise Invalid("thinking effort must be off, low, medium, high or max")
    lock = x.get("lock", False)
    if not isinstance(lock, bool):
        raise Invalid("thinking lock must be true or false")
    out = {"mode": mode, "budget": b, "level": lv}
    if eff is not None:
        out["effort"] = eff
    if lock:
        out["lock"] = True
    return out


def client_thinking(body):
    """-> the list of thinking settings the request itself carries (empty = none)."""
    found = []
    if not isinstance(body, dict):
        return found
    kw = body.get("chat_template_kwargs")
    if isinstance(kw, dict):
        found += ["chat_template_kwargs." + k for k in _CLIENT_KW if k in kw]
    found += [k for k in _CLIENT_TOP if k in body]
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") in ("system", "developer", "user"):
            c = m.get("content")
            texts = [c] if isinstance(c, str) else [x.get("text") for x in c if isinstance(x, dict)] \
                if isinstance(c, list) else []
            if any(isinstance(t, str) and _SOFT_RE.search(t) for t in texts):
                found.append("soft tag in messages")
                break
    return found


def _strip_client(body):
    body = dict(body)
    kw = body.get("chat_template_kwargs")
    if isinstance(kw, dict):
        kw = {k: v for k, v in kw.items() if k not in _CLIENT_KW}
        if kw:
            body["chat_template_kwargs"] = kw
        else:
            body.pop("chat_template_kwargs")
    for k in _CLIENT_TOP:
        body.pop(k, None)
    msgs = body.get("messages")
    if isinstance(msgs, list):
        out = []
        for m in msgs:
            if isinstance(m, dict) and m.get("role") in ("system", "developer", "user") \
                    and isinstance(m.get("content"), str) and _SOFT_RE.search(m["content"]):
                m = dict(m, content=_SOFT_RE.sub(lambda g: g.group(1), m["content"]).strip())
            out.append(m)
        body["messages"] = out
    return body


def apply_effort(body, p, settings, reserve=None):
    """The admin's effort selector on a request that has no pxa_thinking. -> (body, notes).
    No effort set: unchanged. The request's own thinking setting wins unless settings.lock."""
    s = settings or {}
    eff = s.get("effort")
    if not eff:
        return body, []
    own = client_thinking(body)
    if own and not s.get("lock"):
        return body, [f"effort {eff} not applied: the request sets its own thinking ({', '.join(own)})"]
    notes = []
    if own:
        body = _strip_client(body)
        notes.append(f"effort {eff} locked by the admin: removed the request's own setting ({', '.join(own)})")
    lv = (p or {}).get("levels") or {}
    level = eff if eff in (lv.get("values") or ()) else None
    if eff == "off":
        body, n = apply_request(body, p, "off", budget=0, reserve=reserve)
    else:
        body, n = apply_request(body, p, "on", budget=EFFORT_BUDGET[eff], level=level, reserve=reserve)
    return body, notes + [f"effort {eff}: thinking budget {EFFORT_BUDGET[eff]}"] + n


# a top-level OpenAI reasoning_effort: the engine ignores it (no reader in examples/server or common),
# so the Control proxy maps it onto the budget (and the family's own effort kwarg when it has one).
CLIENT_EFFORT = {"none": "off", "off": "off", "minimal": "low", "low": "low", "medium": "medium",
                 "high": "high", "xhigh": "max", "max": "max"}


def map_client_effort(body, p, reserve=None):
    """-> (body, notes). Only when the body has a top-level reasoning_effort; else unchanged."""
    if not isinstance(body, dict) or "reasoning_effort" not in body:
        return body, []
    raw = body.get("reasoning_effort")
    eff = CLIENT_EFFORT.get(str(raw).lower()) if isinstance(raw, str) else None
    if eff is None or not p or not p.get("supported"):
        return body, [f"reasoning_effort {raw!r}: not mapped"]
    body = dict(body)
    body.pop("reasoning_effort")
    lv = p.get("levels") or {}
    kw = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
    level = raw if (raw in (lv.get("values") or ()) and lv.get("key") not in kw) else None
    budget = body.pop("thinking_budget_tokens", None)
    b = EFFORT_BUDGET[eff] if budget is None else int(budget)
    if eff == "off":
        body, n = apply_request(body, p, "off", budget=0, reserve=reserve)
    else:
        body, n = apply_request(body, p, "on", budget=b, level=level, reserve=reserve)
    return body, [f"reasoning_effort {raw} -> thinking budget {b}"] + n


def effective_budget(p, settings):
    b = (settings or {}).get("budget")
    return p["budget"]["suggested"] if b is None else b


# ---------------------------------------------------------------------------------------------
# apply: one request
# ---------------------------------------------------------------------------------------------
def _sys_prefix(msgs, line):
    if msgs and msgs[0].get("role") in ("system", "developer"):
        c = msgs[0].get("content")
        if isinstance(c, str):
            if not c.startswith(line):
                msgs[0] = dict(msgs[0], content=line + ("\n\n" + c if c else ""))
        elif isinstance(c, list):
            msgs[0] = dict(msgs[0], content=[{"type": "text", "text": line}] + list(c))
    else:
        msgs.insert(0, {"role": "system", "content": line})


def _user_suffix(msgs, suffix):
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            c = msgs[i].get("content")
            if isinstance(c, str):
                if not c.rstrip().endswith(suffix.strip()):
                    msgs[i] = dict(msgs[i], content=c + suffix)
            elif isinstance(c, list):
                msgs[i] = dict(msgs[i], content=list(c) + [{"type": "text", "text": suffix}])
            return


def _apply_action(body, act, notes):
    if not act:
        return
    if act.get("kwargs"):
        kw = dict(body.get("chat_template_kwargs") or {})
        kw.update(act["kwargs"])
        body["chat_template_kwargs"] = kw
    msgs = body.get("messages")
    if isinstance(msgs, list):
        msgs = [dict(m) if isinstance(m, dict) else m for m in msgs]
        if act.get("system_prefix"):
            _sys_prefix(msgs, act["system_prefix"])
        if act.get("user_suffix"):
            _user_suffix(msgs, act["user_suffix"])
        body["messages"] = msgs
    if act.get("budget_tokens") is not None:
        body["thinking_budget_tokens"] = int(act["budget_tokens"])


def _apply_method(body, p, mode, use_fallback, notes):
    """switch thinking on|off with the family's preferred method (the next one with use_fallback);
    -> the method used, or None."""
    ms = list((p.get("methods") or {}).get(mode) or [])
    if use_fallback and len(ms) > 1:
        notes.append(f"fallback: {METHOD_LABEL[ms[1]]} instead of {METHOD_LABEL[ms[0]]}")
        ms = ms[1:]
    for m in ms:
        if m == K:
            _apply_action(body, {"kwargs": (p.get(mode) or {}).get("kwargs")}, notes)
            notes.append(f"thinking {mode}: chat_template_kwargs {json.dumps((p.get(mode) or {}).get('kwargs'))}")
        elif m == S:
            act = (p.get("fallback") or {}).get(mode) or (p.get(mode) if p.get("mechanism") == "system" else None)
            _apply_action(body, act, notes)
            tag = (act.get("user_suffix") or act.get("system_prefix") or "").strip()
            notes.append(f"thinking {mode}: soft tag '{tag}' in the "
                         + ("last user message" if act.get("user_suffix") else "system prompt"))
        elif m == PF:
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs or (isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant"):
                notes.append("the request already ends with an assistant message: no prefill")
                continue
            body["messages"] = list(msgs) + [{"role": "assistant", "content": p["prefill"]}]
            kw = dict(body.get("chat_template_kwargs") or {})
            kw["enable_thinking"] = False   # the engine refuses a prefill while enable_thinking is on
            body["chat_template_kwargs"] = kw
            notes.append(f"thinking off: the answer starts with an empty thought {json.dumps(p['prefill'])} "
                         "(assistant prefill; the reply comes back as plain content)")
        elif m == RB:
            body["thinking_budget_tokens"] = 0
            notes.append("thinking off: the engine closes the thought at once (thinking_budget_tokens 0; "
                         "not yet measured live)")
        elif m == EF:
            notes.append("this model cannot stop thinking: using the lowest effort" if mode == "off"
                         else "thinking on: effort level")
        elif m == DF:
            pass
        return m
    if mode == "off":
        notes.append("this model cannot stop thinking: unchanged")
    return None


def apply_request(body, p, mode="auto", budget=None, level=None, use_fallback=False, reserve=None):
    """An OpenAI chat body -> (new body, notes). mode on|off|auto; budget None = the suggestion,
    -1 = unlimited. use_fallback: the family's next method (e.g. the soft tag instead of the kwarg)."""
    t = load_table()
    reserve = t.get("answer_reserve", 1024) if reserve is None else reserve
    body = dict(body or {})
    notes = []
    if not p or not p.get("supported"):
        return body, ["this model has no thinking switch: nothing changed"]
    on = None if mode == "auto" else (mode == "on")
    lv = p.get("levels")
    used = None
    if on is not None:
        used = _apply_method(body, p, "on" if on else "off", use_fallback, notes)
        if on is False:
            level = lv["off_value"] if (lv and lv.get("off_value") and used in (EF, PF)) else None
    if lv:
        want = level or (lv["default"] if on is True and used == EF else None)
        if level is not None and level not in lv["values"] and level != lv.get("off_value"):
            notes.append(f"effort '{level}' is not one of {', '.join(lv['values'])}: ignored")
            want = None
        if want:
            kw = dict(body.get("chat_template_kwargs") or {})
            kw[lv["key"]] = want
            body["chat_template_kwargs"] = kw
    thinking_now = on is True or (on is None and p["default"] in ("on", "always", "template")) \
        or (on is False and used in (EF, None))
    b = p["budget"]["suggested"] if budget is None else int(budget)
    enf = p["budget"]["enforce"]
    if thinking_now and b is not None and b >= 0:
        if p["mechanism"] == "budget-kwarg" and p.get("budget_kwarg"):
            kw = dict(body.get("chat_template_kwargs") or {})
            kw[p["budget_kwarg"]] = b
            body["chat_template_kwargs"] = kw
            notes.append(f"thinking budget {b} via chat_template_kwargs.{p['budget_kwarg']}")
        if enf == "engine" and (budget is not None or mode != "auto"):
            body["thinking_budget_tokens"] = b
            notes.append(f"thinking budget {b} tokens (engine-enforced)")
        if b > 0:
            need = b + reserve
            mt = body.get("max_tokens")
            if enf == "guard":
                if mt is None or (isinstance(mt, int) and mt > need):
                    body["max_tokens"] = need
                    notes.append(f"max_tokens capped at {need} (budget {b} + {reserve} for the answer; this model's "
                                 "thinking cannot be cut by the engine)")
            elif isinstance(mt, int) and 0 < mt < need:
                body["max_tokens"] = need
                notes.append(f"max_tokens raised {mt} -> {need} so the answer still fits after {b} tokens of thinking")
    return body, notes


# ---------------------------------------------------------------------------------------------
# apply: one server launch
# ---------------------------------------------------------------------------------------------
def launch_args(p, settings):
    """-> (argv to append to llama-server, notes). Only what the engine can set server-wide."""
    s = clean_settings(settings)
    out, notes = [], []
    if not p or not p.get("supported"):
        if s["mode"] != "auto":
            notes.append("thinking: this model has no thinking switch (ignored)")
        return out, notes
    mode, lv = s["mode"], p.get("levels")
    kwargs = {}
    used = None
    if mode in ("on", "off"):
        for m in (p.get("methods") or {}).get(mode) or []:
            if m == K:
                kwargs.update((p.get(mode) or {}).get("kwargs") or {})
            elif m == RB:
                out += ["--reasoning-budget", "0"]
                notes.append("thinking off: this model always thinks; server-wide, --reasoning-budget 0 closes the "
                             "thought at once (not yet measured live; the Control chat uses the empty-think prefill "
                             "per request)")
            elif m == EF:
                if mode == "off":
                    kwargs[lv["key"]] = lv["off_value"]
                    notes.append(f"thinking off: this model cannot stop thinking; lowest effort {lv['off_value']}")
            elif m in (S, PF):
                notes.append("thinking %s: %s only works per request (the Control chat does it; other clients "
                             "must add it themselves)" % (mode, "the soft tag" if m == S else "the empty-think prefill"))
                continue
            used = m
            break
        if used is None and mode == "off" and not notes:
            notes.append("thinking off: this model cannot switch it")
    if lv and s["level"] and mode != "off":
        if s["level"] in lv["values"] or s["level"] == lv.get("off_value"):
            kwargs[lv["key"]] = s["level"]
        else:
            notes.append(f"effort '{s['level']}' is not one of {', '.join(lv['values'])}: ignored")
    et = kwargs.pop("enable_thinking", None)
    if et is not None:
        out = ["--reasoning", "on" if et else "off"] + out
    b = s["budget"]
    thinking = mode == "on" or (mode == "auto" and p["default"] in ("on", "always", "template")) \
        or (mode == "off" and used in (EF, None))
    if p["mechanism"] == "budget-kwarg" and p.get("budget_kwarg") and mode != "off" and b is not None:
        kwargs[p["budget_kwarg"]] = b
    if kwargs:
        out += ["--chat-template-kwargs", json.dumps(kwargs, separators=(",", ":"), sort_keys=True)]
    if thinking and b is not None and b >= 0 and "--reasoning-budget" not in out:
        if p["budget"]["enforce"] == "engine":
            out += ["--reasoning-budget", str(b)]
        elif p["budget"]["enforce"] == "guard":
            notes.append(f"thinking budget {b}: the engine cannot cut this model's thinking; the Control chat caps "
                         "max_tokens instead (server-wide there is no cap)")
    return out, notes


# ---------------------------------------------------------------------------------------------
# optional proof by rendering (jinja2)
# ---------------------------------------------------------------------------------------------
def _jinja_env():
    import datetime
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from jinja2 import nodes
    from jinja2.ext import Extension

    class _Generation(Extension):          # HF's {% generation %} ... {% endgeneration %} marker
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.Scope(body, lineno=lineno)

    def _raise(m):
        raise ValueError(m)

    e = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                      extensions=["jinja2.ext.loopcontrols", _Generation])
    e.globals["raise_exception"] = _raise
    e.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
    e.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False, separators=None: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys, separators=separators)
    return e


def render_prompt(tmpl, body, engine_enable_thinking=True, bos="", eos=""):
    """The prompt the template makes for a chat body, the way the engine calls it: the request's
    chat_template_kwargs on top of enable_thinking (common/chat.cpp passes it to every template)."""
    env = _jinja_env()
    msgs = list(body.get("messages") or [])
    prefill = None
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant":
        prefill = msgs.pop()     # the engine continues a final assistant message (prefill_assistant)
    ctx = {"messages": msgs, "add_generation_prompt": True, "bos_token": bos,
           "eos_token": eos, "tools": body.get("tools")}
    kw = dict(body.get("chat_template_kwargs") or {})
    ctx["enable_thinking"] = kw.pop("enable_thinking", engine_enable_thinking)
    ctx.update(kw)
    out = env.from_string(tmpl).render(**ctx)
    if prefill is not None:
        c = prefill.get("content")
        out += c if isinstance(c, str) else "".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
    return out


def verify_render(tmpl, p, sample=None):
    """Render with the switch on and off; {ok, differs, on_tail, off_tail, error}. Needs jinja2."""
    try:
        import jinja2  # noqa: F401
    except ImportError:
        return {"ok": False, "error": "jinja2 is not installed: not verified by rendering"}
    if not tmpl or not p or not p.get("supported"):
        return {"ok": False, "error": "nothing to verify"}
    base = sample or {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    try:
        on_b, _ = apply_request(base, p, "on", budget=-1)
        off_b, _ = apply_request(base, p, "off", budget=-1)
        a = render_prompt(tmpl, on_b)
        b = render_prompt(tmpl, off_b)
    except Exception as e:      # noqa: BLE001 - a template that cannot render here is reported, not fatal
        return {"ok": False, "error": f"{e.__class__.__name__}: {str(e)[:200]}"}
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    closes = None
    if p.get("off_method") == PF and p.get("prefill") and p.get("prefill_tags"):
        o, c = (x.strip() for x in p["prefill_tags"])
        t = b.rstrip()
        closes = t.endswith(c) and (not o or o in t[-(len(o) + len(c) + 400):])
    return {"ok": True, "differs": a != b, "on_tail": a[max(0, i - 40):][-160:], "off_tail": b[max(0, i - 40):][-160:],
            "off_method": p.get("off_method"), "off_closes_thought": closes}
