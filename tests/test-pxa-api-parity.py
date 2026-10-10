#!/usr/bin/env python3
"""Live check of /v1/responses and /v1/messages.

Tools, streaming, and a thinking-block history. No GPU of its own: point it at a
server that is already up.

    PXA_API_URL=http://127.0.0.1:18131 python3 tests/test-pxa-api-parity.py

Skipped when PXA_API_URL is unset.
"""
import json
import os
import unittest
import urllib.error
import urllib.request

URL = os.environ.get("PXA_API_URL", "").rstrip("/")


def post(path, body, timeout=180):
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.headers.get("Content-Type", ""), res.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read().decode()


def sse_events(raw):
    events = []
    name = None
    for line in raw.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                events.append((name, None))
            else:
                events.append((name, json.loads(payload)))
            name = None
    return events


@unittest.skipUnless(URL, "set PXA_API_URL to a running llama-server")
class ApiParity(unittest.TestCase):
    def test_responses_sync_text(self):
        code, ctype, raw = post("/v1/responses", {
            "model": "llama",
            "instructions": "Reply in a few words.",
            "input": "Say hello.",
            "max_output_tokens": 24,
            "temperature": 0,
        })
        self.assertEqual(code, 200, raw[:400])
        self.assertIn("application/json", ctype)
        body = json.loads(raw)
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        texts = []
        for item in body["output"]:
            if item.get("type") == "message":
                for part in item["content"]:
                    if part.get("type") == "output_text":
                        texts.append(part["text"])
        self.assertTrue(any(t.strip() for t in texts), body["output"])
        self.assertIn("output_tokens", body["usage"])

    def test_responses_stream_events(self):
        code, ctype, raw = post("/v1/responses", {
            "model": "llama",
            "input": "Say hello.",
            "max_output_tokens": 16,
            "temperature": 0,
            "stream": True,
        })
        self.assertEqual(code, 200, raw[:400])
        self.assertIn("text/event-stream", ctype)
        names = [n for n, _ in sse_events(raw)]
        for need in ("response.created", "response.output_text.delta", "response.completed"):
            self.assertIn(need, names)

    def test_responses_function_call(self):
        code, _, raw = post("/v1/responses", {
            "model": "llama",
            "input": "What is the weather in Paris? Use the tool.",
            "max_output_tokens": 64,
            "temperature": 0,
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
        })
        self.assertEqual(code, 200, raw[:400])
        body = json.loads(raw)
        calls = [o for o in body["output"] if o.get("type") == "function_call"]
        self.assertTrue(calls, body["output"])
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["status"], "completed")
        args = json.loads(calls[0]["arguments"])
        self.assertIn("city", args)
        self.assertTrue(str(calls[0]["call_id"]).startswith("fc_"))

    def test_messages_sync_text(self):
        code, ctype, raw = post("/v1/messages", {
            "model": "llama",
            "max_tokens": 24,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Say hello."}],
        })
        self.assertEqual(code, 200, raw[:400])
        self.assertIn("application/json", ctype)
        body = json.loads(raw)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["stop_reason"], "end_turn")
        texts = [b["text"] for b in body["content"] if b.get("type") == "text"]
        self.assertTrue(any(t.strip() for t in texts))
        self.assertIn("output_tokens", body["usage"])

    def test_messages_stream_events(self):
        code, ctype, raw = post("/v1/messages", {
            "model": "llama",
            "max_tokens": 16,
            "temperature": 0,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hello."}],
        })
        self.assertEqual(code, 200, raw[:400])
        self.assertIn("text/event-stream", ctype)
        names = [n for n, _ in sse_events(raw)]
        for need in ("message_start", "content_block_start", "content_block_delta", "message_stop"):
            self.assertIn(need, names)

    def test_messages_tool_use(self):
        code, _, raw = post("/v1/messages", {
            "model": "llama",
            "max_tokens": 64,
            "temperature": 0,
            "tool_choice": {"type": "any"},
            "tools": [{
                "name": "get_weather",
                "description": "Get the weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        })
        self.assertEqual(code, 200, raw[:400])
        body = json.loads(raw)
        self.assertEqual(body["stop_reason"], "tool_use")
        uses = [b for b in body["content"] if b.get("type") == "tool_use"]
        self.assertEqual(len(uses), 1)
        self.assertEqual(uses[0]["name"], "get_weather")
        self.assertIn("city", uses[0]["input"])
        self.assertTrue(uses[0]["id"])

    def test_messages_thinking_request_and_history(self):
        # A model with no thinking template still has to accept the switch and a prior thinking block.
        code, _, raw = post("/v1/messages", {
            "model": "llama",
            "max_tokens": 32,
            "temperature": 0,
            "thinking": {"type": "enabled", "budget_tokens": 16},
            "messages": [{"role": "user", "content": "Say hello."}],
        })
        self.assertEqual(code, 200, raw[:400])
        body = json.loads(raw)
        self.assertEqual(body["type"], "message")
        for block in body["content"]:
            self.assertIn(block["type"], ("text", "thinking", "tool_use"))
            if block["type"] == "thinking":
                self.assertIn("thinking", block)

        code, _, raw = post("/v1/messages", {
            "model": "llama",
            "max_tokens": 16,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "The user said hi."},
                    {"type": "text", "text": "Hello."},
                ]},
                {"role": "user", "content": "Say hi again."},
            ],
        })
        self.assertEqual(code, 200, raw[:400])
        body = json.loads(raw)
        texts = [b.get("text", "") for b in body["content"] if b.get("type") == "text"]
        self.assertTrue(any(t.strip() for t in texts))


if __name__ == "__main__":
    unittest.main()
