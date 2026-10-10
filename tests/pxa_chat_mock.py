#!/usr/bin/env python3
"""Stdlib mock OpenAI-compatible server for the PXA Control chat agent tests and UI smoke test.

  python3 pxa_chat_mock.py --port 18901 --model mock-qwen-27b --tools 1 --ctx 65536

Scenarios, picked from the last user message (lowercase):
  native tools (--tools 1, request has `tools`): "save"/"file" -> write_file; "read" -> read_file;
  "calc"/"%" -> calculate; "fetch"/"open" -> web_fetch of a loopback URL (needs approval); "ask" -> ask_server;
  "loop" -> calls calculate forever; "list" -> list_files; "command" -> run_command;
  "remember ..." -> remember ("vegetarian" -> "The user is vegetarian.", "password" -> a secret it must refuse);
  "forget ..." -> forget; a non-stream memory-extraction request -> "- The user is vegetarian." or NONE; "dinner" -> a reply that is vegetarian only when the memory block says so
  text protocol (--tools 0, or request without `tools`): the same calls written as {"tool": ...} JSON text
  "slow" -> streams one token every 0.3 s for 60 s (cancel / step-timeout tests); "markdown"/"table" -> a reply with
  a heading, list, table and code block; "empty" -> after the tool result the model sends an EMPTY final turn
  (the harness's fallback line); anything else -> plain chat.
After a tool result the mock answers like a model would: in its own words, about the user's question, with the
result in context (no fixed wrapper). Non-stream requests (the ask_server handoff) get a JSON answer.
"""
import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer



MARKDOWN = ("## GPU power, in short\n\nLower power limits trade a little speed for a lot less heat:\n\n"
            "- **Quiet**: about 70% power\n- **Balanced**: the card's default\n- **Max**: the highest limit\n\n"
            "| Card | Default | Quiet |\n|---|---:|---:|\n| P100 | 250 W | 175 W |\n| V100 | 300 W | 210 W |\n\n"
            "Set it from a script with `nvidia-smi`:\n\n```bash\nsudo nvidia-smi -i 0 -pl 175\n```\n")


def answer_after(name, res, question):
    """what a decent model says once it has the tool result: about the question, in context."""
    first = res.splitlines()[0] if res else ""
    if name == "calculate" and " = " in first:
        val = first.split(" = ")[-1]
        val = val[:-2] if val.endswith(".0") else val
        return f"18% of 2,450 is **{val}**." if "18%" in question else f"That comes to **{val}**."
    if name == "write_file":
        return "Saved your pasta shopping list as `shopping-list.md`: spaghetti, tomatoes, basil and parmesan."
    if name == "read_file":
        return "Your shopping list has four things on it: spaghetti, tomatoes, basil and parmesan."
    if name == "list_files":
        return f"Your sandbox folder has: {first or 'nothing yet'}."
    if name == "web_fetch":
        return ("I couldn't open that page, so I can't summarize it." if res.startswith("error")
                else f"The page says: {' '.join(res.split())[:120]}")
    if name == "ask_server":
        return f"The other model agrees: {first[:80]}"
    if name in ("remember", "memory_save"):
        return ("I can't keep that one: " + first.split("not saved:")[-1].strip() if res.startswith("error")
                else "Got it, I'll keep that in mind for later chats.")
    if name == "forget":
        return "Done, that's forgotten." if not res.startswith("error") else "I couldn't find that in my memory."
    if name in ("memory_search", "chat_search", "chat_read"):
        return res[:500] if res else "Nothing came back."
    if name == "spawn_agent":
        return ("Result: " + res[:500]) if res and not str(res).startswith("error") else ("The sub-agent did not finish: " + first[:160])
    return first[:80]


def plan(text, cfg):
    t = text.lower()
    if "memory_search" in t:
        q = text.split("memory_search", 1)[-1].strip(" :.") or "vegetarian"
        return "memory_search", {"query": q}
    if "memory_save" in t:
        body = text.split("memory_save", 1)[-1].strip(" :.")
        if "password" in t:
            return "memory_save", {"text": "The user's password is hunter22."}
        return "memory_save", {"text": body or "The user prefers metric units."}
    if "chat_search" in t:
        q = text.split("chat_search", 1)[-1].strip(" :.") or "docker"
        return "chat_search", {"query": q}
    if "chat_read" in t:
        parts = text.split("chat_read", 1)[-1].strip().split()
        return "chat_read", {"session_id": parts[0] if parts else "", "range": parts[1] if len(parts) > 1 else "summary"}
    if "spawn_agent" in t:
        task = text.split("spawn_agent", 1)[-1].strip()
        return "spawn_agent", {"task": task or "answer the question", "tools": ["calculate"] if "calc" in t else [],
                               "max_steps": 3}
    if "loop" in t:
        return "calculate", {"expression": "1+1"}
    if "forget" in t:
        return "forget", {"fact": "vegetarian" if "vegetarian" in t else t.split("forget", 1)[1].strip(" .")}
    if "remember" in t:
        if "password" in t:
            return "remember", {"fact": "The user's password is hunter22."}
        return "remember", {"fact": "The user is vegetarian." if "vegetarian" in t else
                            text.split("remember", 1)[1].strip(" :.").capitalize() + "."}
    if "command" in t:
        return "run_command", {"command": "echo hello from the sandbox"}
    if "save" in t or "file" in t and "read" not in t:
        return "write_file", {"path": "shopping-list.md", "content": "# Pasta dinner\n- spaghetti\n- tomatoes\n- basil\n- parmesan\n"}
    if "read" in t:
        return "read_file", {"path": "shopping-list.md"}
    if "list" in t:
        return "list_files", {}
    if "calc" in t or "%" in t:
        return "calculate", {"expression": "2450*18/100"}
    if "fetch" in t or "open" in t:
        return "web_fetch", {"url": "http://127.0.0.1:9/admin"}
    if "ask" in t:
        return "ask_server", {"server": cfg.ask_target or "other", "task": "Give a one-line second opinion: is 2+2=4?"}
    return None


class Quiet(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass                     # a client that hangs up mid-stream (cancel tests) is expected


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            return self._json(200, {"status": "ok"})
        if p == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": self.server.cfg.model, "object": "model"}] if self.server.cfg.model else []})
        if p == "/props":
            tpl = "{% if tools %}<tool_call>{% endif %}{{ messages }}" if self.server.cfg.tools else "{{ messages }}"
            if self.server.cfg.think:
                tpl += "<think>"
            return self._json(200, {"default_generation_settings": {"n_ctx": self.server.cfg.ctx}, "model_path": f"/models/{self.server.cfg.model}.gguf",
                                    "chat_template": tpl,
                                    "chat_template_caps": {"supports_tools": bool(self.server.cfg.tools), "supports_tool_calls": bool(self.server.cfg.tools),
                                                           "supports_system_role": True}})
        if p == "/slots":
            return self._json(200, [{"id": 0, "is_processing": bool(self.server.cfg.busy)}])
        return self._json(404, {"error": "not found"})

    def _chunk(self, obj):
        data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _plain(self, text):
        t = text.lower()
        if "markdown" in t or "table" in t:
            return MARKDOWN
        if "who are you" in t or "hello" in t or t.strip() in ("hi", "hey"):
            return f"Hi! I'm {self.server.cfg.model}, running on your own machine. What can I help with?"
        return f"You asked about \"{text[:60]}\". Here's a short answer from {self.server.cfg.model} (a mock server)."

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.split("?")[0] == "/tokenize":
            text = str(body.get("content") if body.get("content") is not None else body.get("prompt") or "")
            return self._json(200, {"tokens": [1] * ((len(text) + 3) // 4)})
        self.server.last_body = body
        if body.get("stream") and "OVERFLOWCTX" in json.dumps(body.get("messages") or []):
            if not getattr(self.server, "overflow_done", False):
                self.server.overflow_done = True
                return self._json(400, {"error": {"message": "request exceeds the available context size (n_ctx)", "code": 400}})
        msgs = body.get("messages") or []
        last = msgs[-1] if msgs else {}
        native = bool(body.get("tools")) and self.server.cfg.tools
        textual = not native and any("You can use tools" in (m.get("content") or "") for m in msgs if m.get("role") == "system")
        after_tool = last.get("role") == "tool" or (last.get("role") == "user" and str(last.get("content") or "").startswith(("[result of", "[tool result]")))
        user_text = str(last.get("content") or "") if last.get("role") == "user" else ""
        call = None
        question = next((str(m.get("content") or "") for m in reversed(msgs) if m.get("role") == "user"
                         and not str(m.get("content") or "").startswith(("[result of", "[tool result]"))), "")
        if after_tool:
            if "loop" in json.dumps([m.get("content") for m in msgs if m.get("role") == "user"][:1]).lower():
                call = ("calculate", {"expression": "1+1"})
            res = "\n".join(l for l in str(last.get("content") or "").splitlines() if not l.startswith(("[result of", "[tool result]")))
            tool = next((tc["function"]["name"] for m in reversed(msgs) if m.get("role") == "assistant"
                         for tc in (m.get("tool_calls") or [])), None)
            if not tool:
                tool = next((n for n in ("calculate", "write_file", "read_file", "list_files", "web_fetch", "ask_server", "remember", "forget", "spawn_agent")
                             for m in reversed(msgs) if m.get("role") == "assistant" and f'"{n}"' in str(m.get("content") or "")), None)
            reply = "" if call or "empty" in question.lower() else answer_after(tool, res, question)
        elif native or textual:
            call = plan(user_text, self.server.cfg)
            reply = "" if call else self._plain(user_text)
            if not call and "dinner" in user_text.lower():
                sysm = " ".join(str(m.get("content") or "") for m in msgs if m.get("role") == "system")
                block = sysm.split("<memory>", 1)[1].split("</memory>", 1)[0] if "<memory>" in sysm else ""
                reply = ("How about a chickpea and spinach curry? It's meat-free, so it fits your diet."
                         if "vegetarian" in block.lower() else "How about a chicken stir-fry with rice?")
        else:
            reply = self._plain(user_text)
        sys_blob = " ".join(str(m.get("content") or "") for m in msgs if m.get("role") == "system")
        if not body.get("stream") and "You compact an older part" in sys_blob:
            src = str(last.get("content") or "")
            lines = []
            for line in src.splitlines():
                t = line.strip().lstrip("-* ").strip()
                t = re.sub(r"^(?:user|assistant|system|tool)\s*:\s*", "", t, count=1, flags=re.I)
                if t.lower().startswith("the user ") and t not in lines:
                    lines.append(t[:300])
            for rx in (r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){1,}\b", r"\b[\w./+-]+\.(?:py|txt|md|json)\b", r"\b\d[\d,]{2,}\b"):
                for bit in re.findall(rx, src):
                    if bit not in lines:
                        lines.append(bit)
            summary = "Facts:\n" + "\n".join("- " + x for x in lines[:30])
            if not lines:
                summary = "Facts:\n- (none)"
            return self._json(200, {"model": self.server.cfg.model, "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": summary}}]})
        if not body.get("stream") and any("You extract lasting facts" in str(m.get("content") or "") for m in msgs):
            ask = str(last.get("content") or "")
            facts = []
            for line in ask.splitlines():
                t = line.strip().lstrip("-* ").strip()
                if t.lower().startswith("the user ") and t not in facts:
                    facts.append(t if t.endswith((".", "!", "?")) else t + ".")
                if len(facts) >= 3:
                    break
            low = ask.lower()
            if "vegetarian" in low and not any("vegetarian" in f.lower() for f in facts):
                facts.insert(0, "The user is vegetarian.")
            if "password" in low:
                facts.append("The user's password is hunter22.")
            return self._json(200, {"model": self.server.cfg.model, "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": "\n".join(f"- {f}" for f in facts) or "NONE"}}]})
        if not body.get("stream") and "You summarize one earlier chat" in sys_blob:
            self.server.ref_n = getattr(self.server, "ref_n", 0) + 1
            src = str(last.get("content") or "")
            bits = re.findall(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){1,}\b", src)
            text = " ".join(bits) or src[:400]
            return self._json(200, {"model": self.server.cfg.model, "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": text}}]})
        if not body.get("stream"):
            return self._json(200, {"model": self.server.cfg.model, "choices": [{"index": 0, "finish_reason": "stop",
                                                                       "message": {"role": "assistant", "content": reply or "4, yes."}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if "slow" in user_text.lower():
                for i in range(200):
                    self._chunk({"choices": [{"index": 0, "delta": {"content": f"tick{i} "}}]})
                    time.sleep(0.3)
            if self.server.cfg.think and not after_tool and (body.get("chat_template_kwargs") or {}).get("enable_thinking", True):
                for w in f"The user asked: {user_text[:50]}. Work out what they need and answer it directly.".split(" "):
                    self._chunk({"choices": [{"index": 0, "delta": {"reasoning_content": w + " "}}]})
                    time.sleep(self.server.cfg.delay)
            if call and native:
                name, a = call
                raw = json.dumps(a)
                cid = f"call_{int(time.time() * 1000) % 100000}"
                self._chunk({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": cid, "type": "function",
                                                                                 "function": {"name": name, "arguments": raw[:len(raw) // 2]}}]}}]})
                self._chunk({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[len(raw) // 2:]}}]}}]})
                self._chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            elif call:
                name, a = call
                text = ("Sure.\n" if "please" in user_text.lower() else "") + json.dumps(dict({"tool": name}, **a))
                for i in range(0, len(text), 12):
                    self._chunk({"choices": [{"index": 0, "delta": {"content": text[i:i + 12]}}]})
                self._chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            else:
                for w in reply.split(" "):
                    self._chunk({"choices": [{"index": 0, "delta": {"content": w + " "}}]})
                    time.sleep(self.server.cfg.delay)
                self._chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                             "timings": {"predicted_per_second": 31.4, "prompt_per_second": 812.0, "predicted_n": len(reply.split())},
                             "usage": {"completion_tokens": len(reply.split()), "prompt_tokens": 40}})
            self._chunk(b"[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(port=0, model="mock-model", tools=1, ctx=32768, busy=0, ask_target=None, think=0, delay=0.0):
    s = Quiet(("127.0.0.1", port), H)
    s.cfg = argparse.Namespace(port=port, model=model, tools=tools, ctx=ctx, busy=busy, ask_target=ask_target, think=think, delay=delay)
    s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18901)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--tools", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--busy", type=int, default=0)
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.03)
    ap.add_argument("--ask-target", default=None)
    a = ap.parse_args()
    s = Quiet(("127.0.0.1", a.port), H)
    s.cfg = a
    s.daemon_threads = True
    print(f"mock OpenAI server on 127.0.0.1:{a.port} model={a.model} tools={a.tools}", flush=True)
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
