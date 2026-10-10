"""What a long chat SENDS to the model. The saved transcript is never rewritten.

Counting uses the server's /tokenize and /props n_ctx when they answer. chars/4 is only the fallback, and the
marker says so. The system prompt, pinned messages, the recalled memory block (it lives in the system prompt)
and the last few turns stay verbatim. Older turns become one structured summary from the same model, and the
next compaction summarizes that summary plus the newly aged turns. A tool call is never split from its result.
Big tool outputs are trimmed head and tail before any turn is dropped. If the summary call fails, the oldest
turns are dropped and the marker says so.
"""
import json
import re

THRESHOLD = 0.75
TARGET = 0.45
KEEP_TURNS = 4
SUMMARY_ALLOWANCE = 400          # tokens reserved for the summary while deciding what still fits

SUMMARY_SYS = (
    "You compact an older part of a chat into a structured summary. The assistant will see your summary "
    "instead of those turns. Use exactly these sections:\n"
    "Goals:\nDecisions:\nFacts:\nOpen tasks:\nFiles:\n"
    "Keep every file name and every number exactly as written. Do not invent. If a section has nothing, write none."
)

MARK_OPEN = "<compacted-history>"
MARK_CLOSE = "</compacted-history>"

CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){1,}\b")
FILE_RE = re.compile(r"\b[\w./+-]+\.(?:py|js|mjs|c|cc|cpp|h|hpp|md|txt|json|css|html|gguf|ya?ml|sh|toml)\b", re.I)
NUM_RE = re.compile(r"\b\d[\d,]{2,}(?:\.\d+)?\b")


def n_ctx_of(props):
    """context length from a llama-server /props body, or None."""
    if not isinstance(props, dict):
        return None
    for src in (props, props.get("default_generation_settings") or {}):
        if isinstance(src, dict):
            for k in ("n_ctx", "n_ctx_train"):
                v = src.get(k)
                if isinstance(v, int) and v > 0:
                    return v
    return None


def message_text(m):
    c = m.get("content")
    if isinstance(c, list):
        bits = []
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                bits.append(str(p.get("text") or ""))
            elif p.get("type") == "image_url":
                bits.append("[image]")
        c = "\n".join(bits)
    text = str(c or "")
    if m.get("tool_calls"):
        text += "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)[:2000]
    return text


def estimate(msgs):
    """chars/4 plus a few tokens per message. The fallback counter, also the planner."""
    n = 0
    for m in msgs or []:
        n += (len(message_text(m)) + 3) // 4 + 4
    return n


def is_tool_result(m):
    if m.get("role") == "tool":
        return True
    return m.get("role") == "user" and str(m.get("content") or "").startswith(("[result of", "[tool result]"))


def is_real_user(m):
    return m.get("role") == "user" and not is_tool_result(m)


def is_summary(m):
    return m.get("role") == "user" and str(m.get("content") or "").startswith(MARK_OPEN)


def atomic_groups(msgs):
    """groups that must stay together: an assistant tool call plus the results that follow it."""
    groups, i = [], 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            g = [m]
            i += 1
            while i < len(msgs) and is_tool_result(msgs[i]):
                g.append(msgs[i])
                i += 1
            groups.append(g)
        else:
            groups.append([m])
            i += 1
    return groups


def align_upto(msgs, upto):
    """the largest group boundary at or before upto, so a call and its result are not split."""
    upto = max(0, min(int(upto or 0), len(msgs)))
    n = last = 0
    for g in atomic_groups(msgs):
        if n + len(g) > upto:
            return last
        n += len(g)
        last = n
    return last


def split_turns(msgs):
    turns, cur = [], []
    for g in atomic_groups(msgs):
        if cur and is_real_user(g[0]):
            turns.append(cur)
            cur = []
        cur.append(g)
    if cur:
        turns.append(cur)
    return turns


def turn_index(turns, i):
    n = 0
    for t in turns[:i]:
        for g in t:
            n += len(g)
    return n


def count_turns(msgs):
    return sum(1 for m in msgs if is_real_user(m))


def head_tail(text, limit):
    if len(text) <= limit:
        return text
    mark = "\n...[trimmed]...\n"
    room = max(0, limit - len(mark))
    head = max(0, room * 2 // 3)
    tail = max(0, room - head)
    return text[:head] + mark + (text[-tail:] if tail else "")


def trim_copy(msgs, limit):
    """a send-copy with big tool outputs cut to head + tail. Stored messages are not touched."""
    out, changed = [], False
    for m in msgs:
        c = m.get("content")
        if is_tool_result(m) and isinstance(c, str) and len(c) > limit:
            out.append(dict(m, content=head_tail(c, limit)))
            changed = True
        else:
            out.append(m)
    return out, changed


def summary_message(text):
    body = (text or "").strip() or "(earlier turns were removed)"
    return {"role": "user", "content": (
        MARK_OPEN + "\nSummary of earlier turns. The full transcript is saved; this is what you can see of them. "
        "File names and numbers are verbatim.\n" + body + "\n" + MARK_CLOSE)}


def project(history, state):
    """the send list for a stored compact state: pinned messages, the summary, then the uncovered tail."""
    if not state or not state.get("upto"):
        return list(history)
    upto = align_upto(history, state.get("upto"))
    head, tail = history[:upto], history[upto:]
    pinned = [m for m in head if m.get("pinned")]
    mid = [summary_message(state.get("summary"))] if state.get("summary") else []
    return pinned + mid + list(tail)


def render_dropped(msgs, limit=20000):
    lines = []
    for m in msgs:
        if m.get("pinned"):
            continue
        lines.append(f"{m.get('role')}: {message_text(m)[:3000]}")
    return "\n".join(lines)[:limit]


def verbatim_bits(text):
    found = []

    def add(s):
        s = " ".join(str(s).split())
        if s and s not in found and len(s) <= 200:
            found.append(s)
    for rx in (CODE_RE, FILE_RE, NUM_RE):
        for m in rx.finditer(text or ""):
            add(m.group(0))
            if len(found) >= 40:
                return found
    return found


def ensure_verbatim(summary, source):
    """file names, codenames and numbers the model dropped go back on the summary, exactly as written."""
    missing = [b for b in verbatim_bits(source) if b not in (summary or "")]
    if not missing:
        return (summary or "").strip()
    # Leading line so a later trim still keeps the file names and numbers.
    return ("Verbatim: " + "; ".join(missing) + "\n" + (summary or "").strip()).strip()


def is_overflow(err):
    msg = str(err or "").lower()
    if any(k in msg for k in ("context size", "context length", "context window", "n_ctx", "max context",
                              "exceeds the available", "context_length_exceeded", "too many tokens",
                              "prompt is too long", "context overflow")):
        return True
    return "context" in msg and any(k in msg for k in ("exceed", "overflow", "too long", "too large"))


def limits(n_ctx, max_tokens, threshold, harder=False):
    """(high, target) token counts. high is the compact trigger, target is where compaction stops."""
    n_ctx = max(1, int(n_ctx or 8192))
    reserve = int(max_tokens) if max_tokens else min(1024, max(128, n_ctx // 8))
    reserve = min(reserve, max(1, n_ctx // 2))
    usable = max(1, n_ctx - reserve)
    th = float(threshold if threshold is not None else THRESHOLD)
    th = min(0.95, max(0.02, th))
    ratio = 0.35 if harder else TARGET
    return max(1, int(usable * th)), max(1, int(usable * ratio))


def _cost(msgs, overhead, tokens):
    return overhead + tokens(msgs)


def decide(history, prior, overhead, high, target, tokens, summarize, keep_turns=KEEP_TURNS,
           force=False, harder=False, estimated=False):
    """-> (send_messages, state or None, event or None).

    `tokens(msgs)` counts messages only. `overhead` is the system prompt. `summarize(prior_summary, dropped_text)`
    returns the new summary or raises. The history list is not mutated.
    """
    history = list(history or [])
    prior = dict(prior) if isinstance(prior, dict) and prior.get("upto") else None
    keep = 1 if harder else max(1, min(int(keep_turns or KEEP_TURNS), 12))
    base = align_upto(history, (prior or {}).get("upto") or 0)

    def over(msgs, line):
        return _cost(msgs, overhead, tokens) > line

    projected = project(history, prior)
    if not force and not harder and not over(projected, high):
        return projected, prior, None

    turns = split_turns(history)
    # fold older turns until the kept tail, plus a summary allowance, fits the target
    new_upto = base
    if len(turns) > keep:
        new_upto = max(base, align_upto(history, turn_index(turns, len(turns) - keep)))
    guard = 0
    while guard < 8:
        guard += 1
        tail = history[new_upto:]
        pinned = [m for m in history[:new_upto] if m.get("pinned")]
        probe = pinned + [{"role": "user", "content": "x" * (SUMMARY_ALLOWANCE * 4)}] + tail
        if not over(probe, target) or new_upto >= len(history):
            break
        # the kept tail itself is too big: give up one more turn, but never the last real user turn
        nxt = align_upto(history, new_upto + 1)
        # jump to the next turn boundary if we are mid-turn
        later = [turn_index(turns, i) for i in range(len(turns)) if turn_index(turns, i) > new_upto]
        nxt = later[0] if later else nxt
        if nxt <= new_upto:
            break
        if count_turns(history[nxt:]) < 1:
            break
        new_upto = nxt
    new_upto = align_upto(history, new_upto)
    if new_upto <= base:
        trimmed, _ = trim_copy(projected, 500 if harder else 1200)
        return trimmed, prior, None

    dropped_msgs = history[base:new_upto]
    source = render_dropped(dropped_msgs)
    prior_summary = (prior or {}).get("summary") or ""
    method, note = "summary", ""
    try:
        text = summarize(prior_summary, source) if source.strip() else prior_summary
        if not str(text or "").strip():
            raise ValueError("empty summary")
        method = "summary"
    except Exception:           # noqa: BLE001 — a failed summary drops the turns; the turn itself must not die
        method = "dropped"
        text = prior_summary
        note = "the summary could not be made, so those turns were dropped"
    text = ensure_verbatim(text, source)
    if method == "dropped" and not text.strip():
        text = "(older turns were dropped because the summary could not be made)"
    folded = count_turns(dropped_msgs)
    state = {
        "summary": text,
        "upto": new_upto,
        "turns": int((prior or {}).get("turns") or 0) + folded,
        "method": method,
        "estimated": bool(estimated or (prior or {}).get("estimated")),
        "note": note,
    }
    send = project(history, state)
    send, trimmed = trim_copy(send, 500 if harder else 1200)
    event = {"turns": folded, "covered_turns": state["turns"], "method": method, "summary": text,
             "estimated": state["estimated"], "note": note, "upto": new_upto, "trimmed": trimmed}
    return send, state, event
