"""The chat agent's tools: OpenAI tool schemas (descriptions adapted from the Mythos agent's tool table),
plain-words lines for beginners, the approval policy, and the executors. Files are fenced to the chat's sandbox
folder (fence.py), web fetch refuses private addresses unless approved, web search exists only when an in-house
SearXNG endpoint is configured, shell is Advanced-only with an approval for every call."""
import ast
import html
import json
import operator
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import fence, guards, oai, picker, search, sessions

MAX_OUT = 12000


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


SPECS = {
    "list_files": _fn("list_files", "List the files in the sandbox folder (or a sub-folder of it).",
                      {"path": {"type": "string", "description": "folder inside the sandbox, default ."}}, []),
    "read_file": _fn("read_file", "Read a file in the sandbox folder; returns LINE-NUMBERED content. Pass offset "
                     "(1-based start line) + limit (line count) to page a big file.",
                     {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
    "write_file": _fn("write_file", "Create a file in the sandbox folder or fully replace one. Pass the COMPLETE file "
                      "content, never a placeholder like '// rest unchanged'. Set append:true to add to the end instead.",
                      {"path": {"type": "string"}, "content": {"type": "string"}, "append": {"type": "boolean"}},
                      ["path", "content"]),
    "search_files": _fn("search_files", "Regex-search file contents in the sandbox folder (case-insensitive); "
                        "returns file:line: text.",
                        {"pattern": {"type": "string"}, "path": {"type": "string", "description": "optional file"}},
                        ["pattern"]),
    "web_fetch": _fn("web_fetch", "Fetch a web page and return its readable text (HTML stripped). Use it to READ a "
                     "source instead of guessing.", {"url": {"type": "string"}, "max_chars": {"type": "integer"}}, ["url"]),
    "web_search": _fn("web_search", "Search the web; returns the top results (title, url, snippet). Cite the urls you use.",
                      {"query": {"type": "string"}, "n": {"type": "integer", "description": "results, default 5"}}, ["query"]),
    "calculate": _fn("calculate", "Evaluate an arithmetic expression exactly (+ - * / // % ** and parentheses, "
                     "round, abs, min, max).", {"expression": {"type": "string"}}, ["expression"]),
    "ask_server": _fn("ask_server", "Delegate a self-contained task to ANOTHER model running on this machine and get "
                      "its answer back. It starts from a blank conversation and sees ONLY task and context, so include "
                      "every fact it needs. It has no tools. Use it for a second opinion or a parallel draft.",
                      {"server": {"type": "string", "description": "the other server's name or key, from the list"},
                       "task": {"type": "string"}, "context": {"type": "string"}}, ["server", "task"]),
    "spawn_agent": _fn("spawn_agent", "Hand one self-contained task to a sub-agent with a fresh context. It may use "
                       "only the tools you name, and only tools you already have. It cannot spawn further agents. "
                       "Include note_write and note_read so it can use this turn's shared scratchpad. "
                       "You get its answer and a short trace. Leave server empty to use this chat's server.",
                       {"task": {"type": "string"},
                        "tools": {"type": "array", "items": {"type": "string"},
                                  "description": "subset of your own tools; omit for no tools"},
                        "max_steps": {"type": "integer"}, "max_tokens": {"type": "integer"},
                        "server": {"type": "string", "description": "optional other local server name or key"}},
                       ["task"]),
    "note_write": _fn("note_write", "Write one note on this turn's shared scratchpad. You and every sub-agent of this "
                      "turn see the same notes. mode replace (the default) keeps the latest text for that key. "
                      "mode append adds to the end. The pad is size-capped and lasts only for this turn. To keep a "
                      "note for later chats, copy it with remember.",
                      {"key": {"type": "string"}, "text": {"type": "string"},
                       "mode": {"type": "string", "description": "replace (default) or append"}},
                      ["key", "text"]),
    "note_read": _fn("note_read", "Read this turn's shared scratchpad. Pass a key, or omit it (or pass all) to read "
                     "every note. Empty when this turn has not written one. Notes from earlier turns are not here.",
                     {"key": {"type": "string", "description": "a key, or all"}},
                     []),
    "run_command": _fn("run_command", "Run ONE program in the sandbox folder (no shell: no pipes, redirects or &&). "
                       "Returns the exit code, stdout and stderr. The user approves every call.",
                       {"command": {"type": "string", "description": "e.g. python3 script.py"},
                        "timeout": {"type": "integer", "description": "seconds, default 60, max 300"}}, ["command"]),
    "host_run": _fn("host_run", "Run ONE program on the host machine, outside the sandbox, and return its exit code "
                    "and output. Only works while the user has turned host access on for this session, and only for a "
                    "command on their allowlist; the user approves every call. The command is a LIST of arguments "
                    "(no shell, so no pipes, redirects, ; or &&).",
                    {"command": {"type": "array", "items": {"type": "string"},
                                 "description": "the program and each argument as its own item, e.g. [\"nvidia-smi\",\"-L\"]"},
                     "timeout": {"type": "integer", "description": "seconds, default 30, max 120"}}, ["command"]),
    "host_read": _fn("host_read", "Read a file on the host machine, outside the sandbox. Only works while the user has "
                     "turned host access on for this session, and only for a path on their allowlist; the user approves "
                     "every call. Read-only, capped at 64 KiB.",
                     {"path": {"type": "string"}}, ["path"]),
    "remember": _fn("remember", "Save one lasting fact about the user to your memory across chats, as one short "
                    "sentence about the user (e.g. \"The user prefers metric units.\"). Never secrets.",
                    {"fact": {"type": "string"}}, ["fact"]),
    "forget": _fn("forget", "Remove a fact from your memory across chats: pass its [id] or the fact itself.",
                  {"fact": {"type": "string", "description": "the [id] like m3, or the fact's words"}}, ["fact"]),
    "memory_search": _fn("memory_search", "Search saved facts about the user. Returns only facts whose words match "
                         "the query. It does not dump unrelated recent facts.",
                         {"query": {"type": "string"}, "k": {"type": "integer", "description": "max facts, default 8"}},
                         ["query"]),
    "memory_save": _fn("memory_save", "Save one lasting fact about the user, as one short sentence about the user "
                       "(e.g. \"The user prefers metric units.\"). Same checks as remember: no secrets.",
                       {"text": {"type": "string"}}, ["text"]),
    "chat_search": _fn("chat_search", "Search past chats on this machine by title, summary and messages. Incognito "
                       "chats are never included. Returns ids you can pass to chat_read.",
                       {"query": {"type": "string"}, "k": {"type": "integer", "description": "max chats, default 5"}},
                       ["query"]),
    "chat_read": _fn("chat_read", "Read one past chat by id. range is summary (the compact summary), all, last:N, "
                     "or A-B (1-based message numbers). An incognito chat cannot be read.",
                     {"session_id": {"type": "string"},
                      "range": {"type": "string", "description": "summary, all, last:N, or A-B"}},
                     ["session_id"]),
}
ADVANCED_ONLY = {"run_command", "host_run", "host_read"}
HOST_ONLY = {"host_run", "host_read"}
MEMORY_TOOLS = ["remember", "forget", "memory_search", "memory_save", "chat_search", "chat_read"]
MEMORY_WRITES = ("remember", "forget", "memory_save")
ORDER = ["list_files", "read_file", "write_file", "search_files", "web_fetch", "web_search", "calculate",
         "ask_server", "spawn_agent", "note_write", "note_read", "run_command", "host_run", "host_read"]
LABELS = {
    "list_files": ("See files", "Look at what's in its own sandbox folder"),
    "read_file": ("Read files", "Open files in its sandbox folder"),
    "write_file": ("Write files", "Save files in its sandbox folder (asks you first)"),
    "search_files": ("Search files", "Find text inside its sandbox files"),
    "web_fetch": ("Read web pages", "Open a web page and read it"),
    "web_search": ("Search the web", "Look things up with your search engine"),
    "calculate": ("Calculator", "Do exact math"),
    "ask_server": ("Ask another server", "Get a second opinion from another model on this machine"),
    "spawn_agent": ("Sub-agent", "Hand one task to a fresh sub-agent that can use a subset of your tools"),
    "note_write": ("Shared notes", "Leave a note this turn's sub-agents can read"),
    "note_read": ("Shared notes", "Read a note left during this turn"),
    "run_command": ("Run commands", "Run a program in the sandbox (asks every time)"),
    "host_run": ("Run commands on this machine", "Run an allowlisted program on the host, outside the sandbox "
                 "(off unless you turn host access on; asks every time)"),
    "host_read": ("Read files on this machine", "Read an allowlisted file on the host, outside the sandbox "
                  "(off unless you turn host access on; asks every time)"),
    "remember": ("Remember", "Save a fact about you for later chats"),
    "forget": ("Forget", "Remove a fact from memory"),
    "memory_search": ("Search memory", "Look up a saved fact about you"),
    "memory_save": ("Remember", "Save a fact about you for later chats"),
    "chat_search": ("Search chats", "Find an older chat on this machine"),
    "chat_read": ("Read a chat", "Open an older chat, or its summary"),
}


class Ctx(object):
    def __init__(self, sandbox, chat_id="", run_id="", advanced=False, app=None, current_key=None,
                 search_url=None, cancel_ev=None, approved_private=False, memory=None, sessions=None,
                 host_ok=False, host_why="", host_allow=None, host_audit=None, search_cfg=None):
        self.search_cfg = search_cfg or search.config(None)
        self.sandbox, self.chat_id, self.run_id, self.advanced = sandbox, chat_id, run_id, advanced
        self.app, self.current_key, self.search_url, self.cancel_ev = app, current_key, search_url, cancel_ev
        self.approved_private = approved_private
        self.memory = memory            # memory.Store, or None when memory is off for this chat
        self.sessions = sessions        # session store, so chat_search / chat_read can read past chats
        self.memory_event = None        # what the last remember/forget did, for the UI's "Memory updated" row
        self.host_ok = bool(host_ok)    # the session switch: off unless the user turned it on for this chat
        self.host_why = host_why or ""  # why it is off, shown verbatim on a refusal
        self.host_allow = host_allow or {"commands": [], "read_paths": []}
        self.host_audit = host_audit    # callable(record) or None; every host call writes exactly one record
        self.host_note = None           # what the last host call did, for the UI's activity row


def available(search_url=None, advanced=False, host=False):
    """the tool names the model is offered. A host tool is offered only while the session switch is on:
    a tool the model cannot use is a tool it will waste a turn on."""
    out = []
    for n in ORDER:
        if n in ADVANCED_ONLY and not advanced:
            continue
        if n in HOST_ONLY and not host:
            continue
        out.append(n)
    return out


def _host(url):
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return ""


def _short(s, n=70):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n - 1] + "..."


def _expr(e):
    return _short(str(e or "").replace("*", "\u00d7"), 50)


def _argv_text(v):
    """a host command is a list; show it the way a person would read it, without pretending it is a shell line."""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v or "")


def say(name, a):
    """the short line shown while a tool runs ('Calculating 2450x18/100', 'Reading notes.txt')."""
    a = a if isinstance(a, dict) else {}
    if name == "list_files":
        return "Listing " + ("files" if a.get("path") in (None, "", ".") else _short(a["path"]))
    if name == "read_file":
        return "Reading " + _short(a.get("path"))
    if name == "write_file":
        return ("Adding to " if a.get("append") else "Saving ") + _short(a.get("path"))
    if name == "search_files":
        return f'Searching files for "{_short(a.get("pattern"), 40)}"'
    if name == "web_fetch":
        return "Fetching " + _short(re.sub(r"^https?://", "", str(a.get("url") or "")).rstrip("/"), 60)
    if name == "web_search":
        return f'Searching the web for "{_short(a.get("query"), 50)}"'
    if name == "calculate":
        return "Calculating " + _expr(a.get("expression"))
    if name == "ask_server":
        return f"Asking {_short(a.get('server'), 40)}"
    if name == "spawn_agent":
        return "Sub-agent: " + _short(a.get("task"), 60)
    if name == "note_write":
        return ("Adding to note " if _append_mode(a) else "Writing note ") + _short(a.get("key"), 40)
    if name == "note_read":
        k = str(a.get("key") or "").strip()
        return "Reading shared notes" if not k or k.lower() in ("all", "*") else "Reading note " + _short(k, 40)
    if name == "run_command":
        return "Running " + _short(a.get("command"), 60)
    if name == "host_run":
        return "Running on this machine " + _short(_argv_text(a.get("command")), 60)
    if name == "host_read":
        return "Reading on this machine " + _short(a.get("path"))
    if name in ("remember", "forget", "memory_save"):
        return "Updating memory"
    if name == "memory_search":
        return f'Searching memory for "{_short(a.get("query"), 50)}"'
    if name == "chat_search":
        return f'Searching past chats for "{_short(a.get("query"), 50)}"'
    if name == "chat_read":
        return "Reading " + _short(a.get("session_id") or "a past chat", 40)
    return f"Using {name}"


def said(name, a):
    """the same line once it is done ('Calculated 2450x18/100', 'Read notes.txt', 'Fetched example.com')."""
    a = a if isinstance(a, dict) else {}
    if name == "list_files":
        return "Listed " + ("files" if a.get("path") in (None, "", ".") else _short(a["path"]))
    if name == "read_file":
        return "Read " + _short(a.get("path"))
    if name == "write_file":
        return ("Added to " if a.get("append") else "Saved ") + _short(a.get("path"))
    if name == "search_files":
        return f'Searched files for "{_short(a.get("pattern"), 40)}"'
    if name == "web_fetch":
        return "Fetched " + _short(re.sub(r"^https?://", "", str(a.get("url") or "")).rstrip("/"), 60)
    if name == "web_search":
        return f'Searched the web for "{_short(a.get("query"), 50)}"'
    if name == "calculate":
        return "Calculated " + _expr(a.get("expression"))
    if name == "ask_server":
        return f"Asked {_short(a.get('server'), 40)}"
    if name == "spawn_agent":
        return "Sub-agent finished: " + _short(a.get("task"), 50)
    if name == "note_write":
        return ("Added to note " if _append_mode(a) else "Wrote note ") + _short(a.get("key"), 40)
    if name == "note_read":
        k = str(a.get("key") or "").strip()
        return "Read shared notes" if not k or k.lower() in ("all", "*") else "Read note " + _short(k, 40)
    if name == "run_command":
        return "Ran " + _short(a.get("command"), 60)
    if name == "host_run":
        return "Ran on this machine " + _short(_argv_text(a.get("command")), 60)
    if name == "host_read":
        return "Read on this machine " + _short(a.get("path"))
    if name in ("remember", "forget", "memory_save"):
        return "Memory updated"
    if name == "memory_search":
        return f'Searched memory for "{_short(a.get("query"), 50)}"'
    if name == "chat_search":
        return f'Searched past chats for "{_short(a.get("query"), 50)}"'
    if name == "chat_read":
        return "Read " + _short(a.get("session_id") or "a past chat", 40)
    return f"Used {name}"


def fallback_line(name, a, ok, out, decision=None):
    """only for an EMPTY final model turn: one short line built from the last tool result (never a template
    wrapped around a real answer)."""
    a = a if isinstance(a, dict) else {}
    out = str(out or "").strip()
    doing = say(name, a)
    doing = doing[:1].lower() + doing[1:]
    if decision == "deny":
        return f"Skipped {doing} because you said no."
    if decision == "timeout":
        return f"Skipped {doing} because nobody approved it in time."
    if not ok:
        why = re.sub(r"^error:\s*", "", out)
        return f"{say(name, a)} didn't work: {_short(why, 160)}"
    first = next((l.strip() for l in out.splitlines() if l.strip()), "")
    if name == "calculate":
        return f"**{first.split(' = ')[-1]}** ({first.replace('*', chr(215))})" if " = " in first else first
    if name == "write_file":
        return f"{said(name, a)} ({_short(first, 80)})." if first else said(name, a) + "."
    if name == "read_file":
        body = "\n".join(re.sub(r"^\s*\d+[|:]\s?", "", l) for l in out.splitlines()[:40])
        return f"{_short(a.get('path'))}:\n\n```\n{body[:2000]}\n```"
    if name == "list_files":
        names = [l.strip() for l in out.splitlines() if l.strip()][:20]
        return "In the sandbox folder: " + ", ".join(names) if names else "The sandbox folder is empty."
    if name == "web_fetch":
        return f"From {_host(str(a.get('url') or ''))}: {_short(' '.join(out.split()), 240)}"
    if name == "web_search":
        try:
            res = json.loads(out).get("results") or []
        except ValueError:
            res = []
        return ("Top results:\n" + "\n".join(f"- [{x.get('title') or x.get('url')}]({x.get('url')})" for x in res[:5])
                if res else "The search found nothing.")
    if name in ("ask_server", "search_files", "spawn_agent"):
        return _short(out, 600)
    if name in ("remember", "forget", "memory_save", "memory_search", "chat_search", "chat_read",
                "note_write", "note_read"):
        return _short(out, 600)
    if name == "run_command":
        return f"`{_short(a.get('command'), 80)}` printed:\n\n```\n{out[:2000]}\n```"
    if name == "host_run":
        return f"On this machine, `{_short(_argv_text(a.get('command')), 80)}` printed:\n\n```\n{out[:2000]}\n```"
    if name == "host_read":
        return f"{_short(a.get('path'))}:\n\n```\n{out[:2000]}\n```"
    return _short(first, 240)


def approval_needed(name, a, ctx):
    """-> None, or {"scope", "question", "detail", "always_ok"} when the user must say yes first."""
    a = a if isinstance(a, dict) else {}
    if name == "write_file":
        return {"scope": "write_file", "always_ok": True,
                "question": f"Let the assistant {'add to' if a.get('append') else 'save'} the file {_short(a.get('path'), 60)}?",
                "detail": f"It stays inside this chat's sandbox folder. {len(str(a.get('content') or ''))} characters."}
    if name == "web_fetch":
        why = guards.private_host(_host(str(a.get("url") or "")))
        if why:
            return {"scope": "web_fetch_private", "always_ok": True,
                    "question": "Let the assistant open a page on your own network?",
                    "detail": f"{why}. Pages there can be routers, admin panels or other apps on this machine."}
        return None
    if name == "run_command":
        return {"scope": "run_command", "always_ok": False,
                "question": "Let the assistant run this command?",
                "detail": f"{a.get('command')}\nIt runs in the sandbox folder with a time limit. You'll be asked every time."}
    if name in HOST_ONLY:
        # A host call gets a card only when it is ALLOWED. Anything else is refused in the executor with
        # plain text -- a card the user can approve and still be refused is worse than no card at all.
        if not ctx.host_ok:
            return None
        if name == "host_run":
            argv = a.get("command")
            kind, val, why = guards.host_allowed(argv=argv if isinstance(argv, list) else [], allow=ctx.host_allow)
            if not kind:
                return None
            return {"scope": "host:" + " ".join(val), "always_ok": True,
                    "question": "Let the assistant run this on this machine, outside the sandbox?",
                    "detail": f"{' '.join(val)}\nIt runs on the host itself, not in the chat's folder. "
                              "You can allow this exact command for the rest of the session."}
        rp, why = guards.host_path_allowed(a.get("path"), ctx.host_allow)
        if not rp:
            return None
        return {"scope": "host:read:" + rp, "always_ok": True,
                "question": "Let the assistant read this file on this machine, outside the sandbox?",
                "detail": f"{rp}\nRead-only, capped at 64 KiB. You can allow this path for the rest of the session."}
    return None


# ---- shared scratchpad ------------------------------------------------------------------------
# One pad per parent turn. Sub-agents share it. The next turn starts empty. A note is not a memory fact.
NOTE_KEY_MAX = 40
NOTE_TEXT_MAX = 2000
NOTE_PAD_MAX = 8000
NOTE_COUNT_MAX = 24
_NOTE_CTRL = re.compile(r"\[\[[^\]]*\]\]")
_PAD_LOCK = threading.Lock()
_NOTE_STUB = "Shared notes stay on this turn. They are not carried forward."


def _append_mode(a):
    a = a if isinstance(a, dict) else {}
    return str(a.get("mode") or "").strip().lower() == "append" or a.get("append") is True


class Pad(object):
    """the notes for one turn. last write wins, unless the write says append."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items = {}
        self.order = []

    def snapshot(self):
        with self.lock:
            return [{"key": k, "text": self.items[k]} for k in self.order if k in self.items]


def pad_of(ctx):
    """the pad on this context. A child is given the parent's pad, so this does not start a second one."""
    with _PAD_LOCK:
        p = getattr(ctx, "pad", None)
        if not isinstance(p, Pad):
            p = Pad()
            ctx.pad = p
        return p


def _note_key(raw):
    k = _NOTE_CTRL.sub(" ", str(raw or ""))
    k = " ".join(k.split()).strip()
    if not k or k.lower() in ("all", "*"):
        return None
    return k[:NOTE_KEY_MAX].rstrip()


def _note_text(raw):
    return _NOTE_CTRL.sub(" ", str(raw or "")).strip()


def _emit_notes(ctx):
    emit = getattr(ctx, "emit", None)
    if not emit:
        return
    emit("notes", {"notes": pad_of(ctx).snapshot()})


def note_write(a, ctx):
    """(ok, text). replace overwrites the key. append adds a line. A full pad is refused, not silently dropped."""
    a = a if isinstance(a, dict) else {}
    key = _note_key(a.get("key"))
    if str(a.get("key") or "").strip().lower() in ("all", "*"):
        return False, "error: key cannot be all"
    if not key:
        return False, "error: key is required"
    text = _note_text(a.get("text"))
    if not text:
        return False, "error: text is required"
    append = _append_mode(a)
    pad = pad_of(ctx)
    with pad.lock:
        prev = pad.items.get(key, "")
        nxt = (prev + "\n" + text) if append and prev else text
        others = sum(len(pad.items[k]) for k in pad.order if k != key and k in pad.items)
        if key not in pad.items and len(pad.items) >= NOTE_COUNT_MAX:
            return False, "error: the shared notes are full"
        room = NOTE_PAD_MAX - others
        if room <= 0:
            return False, "error: the shared notes are full"
        cut = False
        if len(nxt) > NOTE_TEXT_MAX:
            nxt = nxt[:NOTE_TEXT_MAX].rstrip()
            cut = True
        if len(nxt) > room:
            nxt = nxt[:room].rstrip()
            cut = True
        if not nxt:
            return False, "error: the shared notes are full"
        if key not in pad.items:
            pad.order.append(key)
        pad.items[key] = nxt
    _emit_notes(ctx)
    verb = "Added to" if append and prev else "Wrote"
    return True, verb + " " + key + (" (cut to fit)" if cut else "") + "."


def note_read(a, ctx):
    """(ok, text). A missing key is an empty note, not a failure, so a sibling can try again."""
    a = a if isinstance(a, dict) else {}
    key = _note_key(a.get("key"))
    snap = pad_of(ctx).snapshot()
    if key is None:
        if not snap:
            return True, "No shared notes yet."
        return True, _clip("\n".join(n["key"] + ": " + n["text"] for n in snap), 4000)
    for n in snap:
        if n["key"] == key:
            return True, n["text"]
    return True, "No note named %s." % key


def _args_key(raw):
    if isinstance(raw, dict):
        return str(raw.get("key") or "")[:NOTE_KEY_MAX]
    try:
        o = json.loads(raw or "{}")
    except ValueError:
        return ""
    if isinstance(o, dict):
        return str(o.get("key") or "")[:NOTE_KEY_MAX]
    return ""


def redact_note_messages(messages):
    """a copy of the turn's messages whose note text is gone. The pad snapshot stays on the turn for search."""
    from . import protocol
    names = {}
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        if role == "assistant":
            m2 = dict(m)
            calls, changed = [], False
            for c in m.get("tool_calls") or []:
                if not isinstance(c, dict):
                    calls.append(c)
                    continue
                fn = dict(c.get("function") or {})
                name = fn.get("name")
                names[c.get("id")] = name
                if name in ("note_write", "note_read"):
                    key = _args_key(fn.get("arguments"))
                    fn["arguments"] = json.dumps({"key": key} if key else {})
                    calls.append(dict(c, function=fn))
                    changed = True
                else:
                    calls.append(c)
            if changed:
                m2["tool_calls"] = calls
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                prose, call = protocol.split_call(content)
                if isinstance(call, dict) and call.get("tool") in ("note_write", "note_read"):
                    key = str(call.get("key") or "")[:NOTE_KEY_MAX]
                    kept = json.dumps({"tool": call["tool"], "key": key} if key else {"tool": call["tool"]})
                    m2["content"] = (prose + "\n" + kept).strip() if prose else kept
            out.append(m2)
            continue
        if role == "tool" and names.get(m.get("tool_call_id")) in ("note_write", "note_read"):
            out.append(dict(m, content=_NOTE_STUB))
            continue
        content = m.get("content")
        if role == "user" and isinstance(content, str) and (
                content.startswith("[result of note_write]") or content.startswith("[result of note_read]")):
            out.append(dict(m, content=content.split("\n", 1)[0] + "\n" + _NOTE_STUB))
            continue
        out.append(m)
    return out


# ---- executors ---------------------------------------------------------------------------------
def _clip(s, n=MAX_OUT):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n[cut at {n} characters]"


class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private):
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith(("http://", "https://")):
            raise urllib.error.URLError("redirect to a non-web address refused")
        if not self.allow_private and guards.private_host(_host(newurl)):
            raise urllib.error.URLError("the page redirected to your own network; refused (approve it to allow)")
        return urllib.request.HTTPRedirectHandler.redirect_request(self, req, fp, code, msg, headers, newurl)


def html_text(raw):
    s = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", raw)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|article)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def web_fetch(a, ctx):
    url = str(a.get("url") or "").strip()
    if not re.match(r"^https?://", url, re.I):
        raise ValueError("only http:// and https:// addresses can be opened")
    mx = max(500, min(int(a.get("max_chars") or 8000), 40000))
    if guards.private_host(_host(url)) and not ctx.approved_private:
        raise ValueError("that address is on your own network; it needs your approval")
    if not ctx.approved_private:
        md = search.reader_markdown(url, ctx.search_cfg)
        if md:
            return guards.screen_injected(_clip("Source: " + url + "\n\n" + md, mx))
    op = urllib.request.build_opener(_Redirects(ctx.approved_private))
    req = urllib.request.Request(url, headers={"User-Agent": "PXA-Control-Chat/1", "Accept": "text/html,text/plain,*/*"})
    with op.open(req, timeout=20) as r:
        ctype = r.headers.get("Content-Type") or ""
        raw = r.read(2 << 20)
    text = raw.decode("utf-8", "replace")
    if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        text = html_text(text)
    return guards.screen_injected(_clip(text, mx))


def web_search(a, ctx):
    q = str(a.get("query") or "").strip()
    if not q:
        raise ValueError("empty query")
    n = max(1, min(int(a.get("n") or 5), 15))
    res, note = search.search(q, n, ctx.search_cfg)
    for x in res:
        x["snippet"] = guards.screen_injected(str(x.get("snippet") or "")[:300])
    d = {"query": q, "results": res}
    if note:
        d["note"] = note
    return json.dumps(d, indent=1)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}


def calculate(a, ctx=None):
    expr = str(a.get("expression") or "")
    if len(expr) > 300:
        raise ValueError("expression too long")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(r) > 1000 or abs(l) > 1e6):
                raise ValueError("number too large")
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(x) for x in n.args])
        raise ValueError("only numbers, + - * / // % ** ( ) and round/abs/min/max are allowed")
    return f"{expr} = {ev(ast.parse(expr, mode='eval'))}"


SUB_SYSTEM = ("You are {name}, acting as a SUB-AGENT. Another AI assistant delegated the task below to you. "
              "Answer it directly and completely in one reply: no questions back, no preamble about being a sub-agent. "
              "You have NO tools and cannot delegate further. If something cannot be known or done, say so plainly.")
_ask_locks = {}
_ask_guard = threading.Lock()


def ask_server(a, ctx, get=None, post=None):
    """port of the owner's Open WebUI 'ask alex / ask alina' handoff: one level deep, no tools, refuse fast."""
    want = str(a.get("server") or "").strip().lower()
    task = str(a.get("task") or "").strip()
    if not task:
        raise ValueError("task is required")
    servers = picker.list_servers(ctx.app, get)
    cand = [s for s in servers if want and (s["key"].lower() == want or (s["name"] or "").lower() == want
                                            or want in (s["name"] or "").lower() or str(s["port"]) == want)]
    if not cand:
        names = ", ".join(s["name"] for s in servers if s["status"] in ("ready", "busy") and s["key"] != ctx.current_key)
        return f"error: no server called {a.get('server')!r}. Other servers that are up: {names or 'none'}."
    t = cand[0]
    if t["key"] == ctx.current_key:
        return "error: that is the server you are running on; answer yourself."
    if t["status"] != "ready":
        return f"error: {t['name']} is {t['status_words'].lower()}; answer yourself or try later."
    with _ask_guard:
        lk = _ask_locks.setdefault(t["key"], threading.Lock())
    if not lk.acquire(blocking=False):
        return f"error: {t['name']} is already working on another delegated task; answer yourself or retry later."
    try:
        g = get or oai.get_json
        try:
            slots = g(t["base_url"], "/slots", 3.0)
            if isinstance(slots, list) and any(isinstance(s, dict) and s.get("is_processing") for s in slots):
                return f"error: {t['name']} is busy serving another request; answer yourself or retry later."
        except oai.OAIError:
            pass                               # /slots can be disabled: the model check below still fails loud
        m = g(t["base_url"], "/v1/models", 3.0)
        data = (m or {}).get("data") or []
        model = data[0].get("id") if data else None
        if not model:
            return f"error: {t['name']} did not report a model at /v1/models; not delegating."
        user = task if not a.get("context") else f"{task}\n\n--- context ---\n{a.get('context')}"
        body = {"model": model, "temperature": 0.4, "max_tokens": 4096, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": SUB_SYSTEM.format(name=t["name"])},
                             {"role": "user", "content": user}]}
        t0 = time.time()
        r = (post or oai.post_json)(t["base_url"], "/v1/chat/completions", body, 300.0)
        msg = ((r.get("choices") or [{}])[0]).get("message") or {}
        ans = (msg.get("content") or "").strip()
        if not ans:
            return f"error: {t['name']} returned an empty answer."
        return _clip(f"[{t['name']} answered in {time.time() - t0:.1f} s]\n{ans}")
    finally:
        lk.release()


SAFE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp"}


def run_command(a, ctx):
    if not ctx.advanced:
        raise ValueError("running commands is only available in Advanced mode")
    cmd = str(a.get("command") or "").strip()
    if not cmd:
        raise ValueError("empty command")
    hit = guards.danger_hit(cmd)
    if hit:
        return f"refused: this command matches a blocked pattern ({hit}). It was not run."
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"could not split the command: {e}")
    if any(t in ("|", "||", "&&", ";", ">", ">>", "<", "&") for t in argv):
        return "refused: no shell here, so pipes, redirects, ; and && do not work. Run one program at a time."
    to = max(1, min(int(a.get("timeout") or 60), 300))
    os.makedirs(os.path.dirname(ctx.sandbox), mode=0o700, exist_ok=True)   # the chat's files: owner-only, like the sessions
    os.makedirs(ctx.sandbox, mode=0o700, exist_ok=True)
    try:
        p = subprocess.run(argv, cwd=ctx.sandbox, env=dict(SAFE_ENV, HOME=ctx.sandbox), capture_output=True,
                           timeout=to, stdin=subprocess.DEVNULL, shell=False)
    except FileNotFoundError:
        return f"error: program {argv[0]!r} not found"
    except subprocess.TimeoutExpired:
        return f"error: the command ran longer than {to} s and was stopped"
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    return _clip(f"exit code {p.returncode}\n--- stdout ---\n{out[-8000:]}\n--- stderr ---\n{err[-3000:]}")


# ---- host access (opt-in per session; every call is allowlist-checked and audited) -----------------
HOST_TIMEOUT_MAX = 120


def _host_rec(tool, ctx, **kw):
    """one audit record per host call. The sink (pxa_control's append_jsonl) redacts before it writes:
    nothing here ever carries the environment or the output, only what was asked and what happened."""
    rec = {"ts": round(time.time(), 3), "tool": tool, "chat_id": ctx.chat_id or "", "session": ctx.run_id or "",
           "sandbox": ctx.sandbox or ""}
    rec.update({k: v for k, v in kw.items() if v is not None})
    return rec


def _host_audit(ctx, rec):
    """an audit sink that fails must not become a call that succeeds unlogged -- but it must not take the
    tool down either, so the failure is recorded on the context for the caller to surface."""
    if not ctx.host_audit:
        return
    try:
        ctx.host_audit(rec)
    except Exception as e:               # noqa: BLE001
        ctx.host_note = {"audit_error": f"{e.__class__.__name__}: {e}"}


def host_run(a, ctx):
    argv = a.get("command")
    # a model sometimes writes a number for a numeric flag (--port 8080 arrives as int). Coercing is safe
    # because the allowlist decides what may run either way; the shape is what must stay a list, not a shell string.
    argv = [str(x) for x in argv] if isinstance(argv, (list, tuple)) else None
    if not ctx.host_ok:
        why = ctx.host_why or "host access is off"
        _host_audit(ctx, _host_rec("host_run", ctx, decision="off", argv=argv, why=why))
        return f"refused: {why}"
    kind, val, why = guards.host_allowed(argv=argv or [], allow=ctx.host_allow)
    if not kind:
        _host_audit(ctx, _host_rec("host_run", ctx, decision="denied", argv=argv, why=why))
        return f"refused: {why}"
    try:
        to = max(1, min(int(a.get("timeout") or 30), HOST_TIMEOUT_MAX))
    except (TypeError, ValueError):
        to = 30
    t0 = time.time()
    try:
        p = subprocess.run(val, cwd="/", env=guards.host_env(), capture_output=True, timeout=to,
                           stdin=subprocess.DEVNULL, shell=False)
    except FileNotFoundError:
        _host_audit(ctx, _host_rec("host_run", ctx, decision="error", argv=val, why="not found", secs=time.time() - t0))
        return f"error: program {val[0]!r} not found on this machine"
    except subprocess.TimeoutExpired:
        _host_audit(ctx, _host_rec("host_run", ctx, decision="timeout", argv=val, secs=round(time.time() - t0, 2)))
        return f"error: the command ran longer than {to} s and was stopped"
    except OSError as e:
        _host_audit(ctx, _host_rec("host_run", ctx, decision="error", argv=val, why=f"{e.__class__.__name__}: {e}"))
        return f"error: could not start {val[0]!r} ({e})"
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    ctx.host_note = {"tool": "host_run", "argv": val, "exit": p.returncode, "bytes": len(p.stdout) + len(p.stderr)}
    _host_audit(ctx, _host_rec("host_run", ctx, decision="allowed", argv=val, exit_code=p.returncode,
                               bytes=len(p.stdout) + len(p.stderr), secs=round(time.time() - t0, 2)))
    return _clip(f"exit code {p.returncode}\n--- stdout ---\n{out[-8000:]}\n--- stderr ---\n{err[-3000:]}")


def host_read(a, ctx):
    if not ctx.host_ok:
        why = ctx.host_why or "host access is off"
        _host_audit(ctx, _host_rec("host_read", ctx, decision="off", path=str(a.get("path") or ""), why=why))
        return f"refused: {why}"
    rp, why = guards.host_path_allowed(a.get("path"), ctx.host_allow)
    if not rp:
        _host_audit(ctx, _host_rec("host_read", ctx, decision="denied", path=str(a.get("path") or ""), why=why))
        return f"refused: {why}"
    if os.path.isdir(rp):
        return f"error: {rp} is a folder; host access reads files, not folders"
    try:
        with open(rp, "rb") as f:
            raw = f.read(guards.READ_CAP)
            cut = bool(f.read(1))
    except OSError as e:
        _host_audit(ctx, _host_rec("host_read", ctx, decision="error", path=rp, why=f"{e.__class__.__name__}: {e}"))
        return f"error: cannot read {rp} ({e})"
    if b"\x00" in raw:
        _host_audit(ctx, _host_rec("host_read", ctx, decision="binary", path=rp, bytes=len(raw)))
        return f"refused: {rp} looks like a binary file; host access reads text"
    ctx.host_note = {"tool": "host_read", "path": rp, "bytes": len(raw), "cut": cut}
    _host_audit(ctx, _host_rec("host_read", ctx, decision="allowed", path=rp, bytes=len(raw), truncated=cut))
    tail = f"\n[read stopped at {guards.READ_CAP} bytes]" if cut else ""
    return _clip(f"{rp} ({len(raw)} bytes){tail}\n{raw.decode('utf-8', 'replace')}")


def memory_tool(name, a, ctx):
    from . import memory as M
    if ctx.memory is None:
        return False, "error: memory is turned off for this chat"
    if name == "memory_search":
        q = str(a.get("query") or "").strip()
        if not q:
            return False, "error: query is required"
        try:
            k = int(a.get("k") or 8)
        except (TypeError, ValueError):
            k = 8
        facts = ctx.memory.search(q, k=k)
        if not facts:
            return True, "No memories match that."
        lines = []
        for f in facts:
            when = time.strftime("%Y-%m-%d", time.localtime(f.get("updated") or 0))
            pin = ", pinned" if f.get("pinned") else ""
            lines.append(f"[{f['id']}] {f['text']} (saved {when}{pin})")
        return True, _clip("\n".join(lines))
    if name == "memory_save":
        fact = str(a.get("text") or a.get("fact") or "").strip()
    else:
        fact = str(a.get("fact") or a.get("fact_or_id") or a.get("id") or "").strip()
    if name in ("remember", "memory_save"):
        try:
            r = ctx.memory.add(fact, source=ctx.chat_id or None)
        except M.MemoryError_ as e:
            ctx.memory_event = {"op": "refused", "text": M.clean(fact)[:120] if "secret" not in str(e) else "",
                                "why": str(e)}
            return False, f"error: not saved: {e}"
        ctx.memory_event = {"op": name, "status": r["status"], "fact": r["fact"], "evicted": r["evicted"]}
        verb = {"added": "Saved", "updated": "Updated", "same": "Already remembered"}[r["status"]]
        return True, f"{verb} [{r['fact']['id']}]: {r['fact']['text']}"
    gone = ctx.memory.forget(fact)
    if gone is None:
        return False, f"error: nothing in memory matches {fact!r}"
    ctx.memory_event = {"op": "forget", "fact": gone}
    return True, f"Forgot [{gone['id']}]: {gone['text']}"


def _chat_tools(name, a, ctx):
    if ctx.sessions is None:
        return False, "error: past chats are not available"
    if name == "chat_search":
        q = str(a.get("query") or "").strip()
        if not q:
            return False, "error: query is required"
        hits = sessions.search_chats(ctx.sessions, q, skip_id=ctx.chat_id or None, k=a.get("k") or 5)
        if not hits:
            return True, "No past chats match that."
        lines = []
        for h in hits:
            when = time.strftime("%Y-%m-%d", time.localtime(h.get("updated") or 0))
            lines.append(f"[{h['id']}] {h['title']} ({when})\n{h.get('snippet') or ''}")
        return True, _clip("\n\n".join(lines))
    sid = str(a.get("session_id") or a.get("id") or "").strip()
    ok, text = sessions.read_chat(ctx.sessions, sid, a.get("range") or a.get("span") or "summary")
    return ok, _clip(text)


def execute(name, a, ctx):
    """-> (ok, text). Never raises."""
    try:
        if not isinstance(a, dict):
            raise ValueError("arguments must be an object")
        if name == "list_files":
            return True, fence.ls(ctx.sandbox, a.get("path") or ".")
        if name == "read_file":
            return True, fence.read_range(ctx.sandbox, a.get("path"), a.get("offset") or 1, a.get("limit") or 200)
        if name == "write_file":
            return True, fence.write(ctx.sandbox, a.get("path"), a.get("content"), bool(a.get("append")))
        if name == "search_files":
            return True, fence.grep(ctx.sandbox, str(a.get("pattern") or ""), a.get("path") or None)
        if name == "web_fetch":
            return True, web_fetch(a, ctx)
        if name == "web_search":
            return True, web_search(a, ctx)
        if name == "calculate":
            return True, calculate(a)
        if name == "ask_server":
            r = ask_server(a, ctx)
            return not r.startswith("error:"), r
        if name == "spawn_agent":
            from . import subagent
            return subagent.run(a, ctx)
        if name == "note_write":
            return note_write(a, ctx)
        if name == "note_read":
            return note_read(a, ctx)
        if name in ("remember", "forget", "memory_save", "memory_search"):
            return memory_tool(name, a, ctx)
        if name in ("chat_search", "chat_read"):
            return _chat_tools(name, a, ctx)
        if name == "run_command":
            r = run_command(a, ctx)
            return not r.startswith(("error:", "refused:")), r
        if name == "host_run":
            r = host_run(a, ctx)
            return not r.startswith(("error:", "refused:")), r
        if name == "host_read":
            r = host_read(a, ctx)
            return not r.startswith(("error:", "refused:")), r
        return False, f"error: there is no tool called {name!r}"
    except fence.Outside as e:
        return False, f"error: {e}"
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, SyntaxError) as e:
        return False, f"error: {e}"
    except urllib.error.HTTPError as e:
        return False, f"error: the page answered HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"error: {getattr(e, 'reason', None) or e}"
    except oai.OAIError as e:
        return False, f"error: {e}"
    except Exception as e:           # noqa: BLE001
        return False, f"error: {e.__class__.__name__}: {e}"


def text_protocol_prompt(names):
    """the system-prompt block for a model without native tool calls (the local-lane JSON dialect)."""
    lines = ["You can use tools. To use one, reply with ONLY a JSON object on its own, like:",
             '{"tool": "read_file", "path": "notes.txt"}',
             "One tool per reply. After you send it you get the result back, then continue. "
             "When you have the final answer, reply normally with no JSON.", "", "Tools:"]
    for n in names:
        f = SPECS[n]["function"]
        params = ", ".join(f'{k}{"" if k in f["parameters"]["required"] else "?"}' for k in f["parameters"]["properties"])
        lines.append(f"- {n}({params}): {f['description']}")
    return "\n".join(lines)
