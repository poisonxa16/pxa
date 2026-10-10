"""Text tool-call reader for models without native tool calls.

Python port of the owner's hive harness reader (PXACLAW tools/local-lane/harness/protocol.mjs, 2026-09-22):
brace counting that understands string literals, one lenient JSON repair (raw newlines inside strings),
<think> stripping, the Qwen <function=NAME><parameter=K>V</parameter></function> dialect, the
<tool_call>/<invoke name=..>/<function name=..> tag dialects, and the JSON {"tool": ...} dialect.
Same rules: a cut-off <parameter> is refused, `content` keeps its indentation, and the only heredoc repair
is for a COMPLETE `run` call.
"""
import json
import re

FN_RE = re.compile(r"<function=([A-Za-z_][\w-]*)>")
OPEN_RE = re.compile(r"""<(?:invoke|function|tool)\s+name\s*=\s*["']([A-Za-z_][\w-]*)["']\s*>""")
PARAM_RE = re.compile(r"""<parameter(?:\s+name\s*=\s*["']([A-Za-z_][\w-]*)["']|=([A-Za-z_][\w-]*))\s*>([\s\S]*?)</parameter>""")
QWEN_PARAM_RE = re.compile(r"<parameter=([A-Za-z_][\w-]*)>([\s\S]*?)</parameter>")
NAME_RE = re.compile(r"^[A-Za-z_][\w-]*$")


def strip_thinking(text):
    """drop <think>...</think> pairs; an unclosed <think> swallows everything after it."""
    s = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    m = re.search(r"<think>", s, flags=re.I)
    return s[:m.start()] if m else s


def match_brace(text, start):
    """index of the brace closing the {...} at `start`, ignoring braces in JSON strings; -1 if none."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_lenient(src):
    """json.loads, then one repair: raw \\n \\r \\t inside a string literal become escapes."""
    try:
        return json.loads(src)
    except ValueError:
        pass
    out, in_str, esc = [], False, False
    for ch in src:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
                continue
            if ch == "\\":
                esc = True
                out.append(ch)
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, ch))
            continue
        if ch == '"':
            in_str = True
        out.append(ch)
    try:
        return json.loads("".join(out))
    except ValueError:
        return None


def json_calls(text, stop_after=None):
    """every JSON object in text that parses AND has a string `tool`, in order."""
    out, i, n = [], 0, len(text)
    while i < n and (stop_after is None or len(out) < stop_after):
        if text[i] == "{":
            end = match_brace(text, i)
            if end >= 0:
                o = parse_lenient(text[i:end + 1])
                if isinstance(o, dict) and isinstance(o.get("tool"), str):
                    out.append(o)
                    i = end
        i += 1
    return out


def _content_trim(key, v):
    # `content` is a file's bytes: strip exactly one newline each side, keep the indentation
    if key == "content":
        v = re.sub(r"^\r?\n", "", v, count=1)
        return re.sub(r"\r?\n$", "", v, count=1)
    return v.strip()


def parse_tag_call(s):
    opener = OPEN_RE.search(s)
    block = re.search(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", s)
    tool, multi = None, False
    if opener:
        tool = opener.group(1)
        rest = s[opener.end():]
        close = re.search(r"</(?:invoke|function|tool)>", rest)
        body = rest[:close.start()] if close else rest
        multi = OPEN_RE.search(s, opener.end()) is not None
    elif block:
        body = block.group(1)
        multi = s.count("<tool_call>") > 1
    else:
        return None
    a = {}
    for m in PARAM_RE.finditer(body):
        key = m.group(1) or m.group(2)
        a[key] = _content_trim(key, m.group(3))
    # a <tool_call>{"name":..,"arguments":{..}}</tool_call> body (Hermes/Qwen JSON-in-tag)
    if not opener and not a:
        st = body.find("{")
        if st >= 0:
            end = match_brace(body, st)
            o = parse_lenient(body[st:end + 1]) if end >= 0 else None
            if isinstance(o, dict) and isinstance(o.get("name"), str):
                args = o.get("arguments")
                if isinstance(args, str):
                    args = parse_lenient(args)
                if isinstance(args, dict) and NAME_RE.match(o["name"]):
                    return dict(args, tool=o["name"], _multi=multi)
            if isinstance(o, dict) and isinstance(o.get("tool"), str):
                return dict(o, _multi=multi)
    if not tool and isinstance(a.get("tool"), str):
        tool = a["tool"].strip()
    if not tool:
        t = re.search(r"<tool_name>\s*([A-Za-z_][\w-]*)\s*</tool_name>", body)
        if t:
            tool = t.group(1)
    if not tool or not NAME_RE.match(tool):
        return None
    # an opened <parameter that never closed means the reply was cut: refuse rather than guess
    if len(re.findall(r"<parameter[\s=]", body)) > body.count("</parameter>"):
        return None
    a.pop("tool", None)
    a["tool"] = tool
    a["_multi"] = multi
    return a


def parse_action(text):
    """-> {"tool": name, **args, "_multi": bool} or None when the reply carries no call."""
    if not isinstance(text, str) or not text:
        return None
    s = strip_thinking(text)
    first = FN_RE.search(s)
    if first:                                           # Qwen native dialect wins when both appear
        multi = FN_RE.search(s, first.end()) is not None
        close = s.find("</function>", first.end())
        body = s[first.end():close if close >= 0 else len(s)]
        a = {"tool": first.group(1)}
        for m in QWEN_PARAM_RE.finditer(body):
            a[m.group(1)] = _content_trim(m.group(1), m.group(2))
        a["_multi"] = multi
        return a
    tag = parse_tag_call(s)
    if tag:
        return tag
    calls = json_calls(s, 2)
    if not calls:
        m = re.search(r'\{\s*"tool"\s*:\s*"run"', s)
        if m:
            tail = s[m.start():].rstrip()
            o = parse_lenient(tail + '"}')
            hd = isinstance(o, dict) and isinstance(o.get("cmd"), str) and re.search(r"<<-?\s*['\"]?([A-Za-z_]+)['\"]?", o["cmd"])
            if hd and re.search(r"\n" + re.escape(hd.group(1)) + r"\s*$", o["cmd"]):
                return dict(o, _multi=False, _repaired=True)
        return None
    return dict(calls[0], _multi=len(calls) > 1)


def split_call(text):
    """(prose before the call, call or None): the prose is what the user sees for a text-protocol step."""
    call = parse_action(text)
    if not call:
        return text, None
    s = strip_thinking(text)
    cut = len(s)
    for pat in (r"<function=", r"<tool_call>", r"<(?:invoke|function|tool)\s+name", r'\{\s*"tool"\s*:'):
        m = re.search(pat, s)
        if m:
            cut = min(cut, m.start())
    prose = re.sub(r"```(?:json|xml)?\s*$", "", s[:cut]).rstrip()
    return prose, call


def call_to_openai(call):
    """protocol call -> (name, arguments dict)."""
    args = {k: v for k, v in call.items() if k not in ("tool", "_multi", "_repaired")}
    return call["tool"], args
