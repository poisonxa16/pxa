"""Minimal stdlib OpenAI-compatible client: streamed chat completions over SSE, with cancel and a per-step
deadline (both shut the socket down, which wakes a blocked read), plus small JSON GETs for /v1/models, /props,
/slots. No third-party code."""
import http.client
import json
import socket
import threading
import time
import urllib.parse


class OAIError(Exception):
    def __init__(self, msg, code="server_error"):
        Exception.__init__(self, msg)
        self.code = code


def _split(base_url):
    u = urllib.parse.urlparse(base_url)
    if u.scheme != "http" or not u.hostname:
        raise OAIError(f"bad server address {base_url!r}", "bad_server")
    root = u.path.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return u.hostname, u.port or 80, root


def get_json(base_url, path, timeout=3.0):
    """GET {server root}{path} -> parsed JSON. path is root-relative ('/props', '/v1/models')."""
    host, port, root = _split(base_url)
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        c.request("GET", root + path, headers={"Accept": "application/json"})
        r = c.getresponse()
        raw = r.read(4 << 20)
        if r.status != 200:
            raise OAIError(f"HTTP {r.status} from {path}", "http_%d" % r.status)
        return json.loads(raw.decode("utf-8", "replace") or "null")
    except (OSError, http.client.HTTPException) as e:
        raise OAIError(f"{path}: {e.__class__.__name__}", "unreachable")
    except ValueError:
        raise OAIError(f"{path} did not return JSON", "bad_reply")
    finally:
        c.close()


def post_json(base_url, path, body, timeout=600.0):
    host, port, root = _split(base_url)
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        c.request("POST", root + path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read(16 << 20)
        if r.status != 200:
            raise OAIError(f"HTTP {r.status}: {raw[:300].decode('utf-8', 'replace')}", "http_%d" % r.status)
        return json.loads(raw.decode("utf-8", "replace"))
    except (OSError, http.client.HTTPException) as e:
        raise OAIError(f"{path}: {e.__class__.__name__}", "unreachable")
    finally:
        c.close()


class Stream(object):
    """one streamed /v1/chat/completions call. cancel() and the deadline both shut the socket down."""

    def __init__(self, base_url, body, step_timeout=180.0, idle_timeout=120.0):
        self.base_url, self.body = base_url, dict(body, stream=True)
        self.step_timeout, self.idle_timeout = float(step_timeout), float(idle_timeout)
        self._conn = None
        self._lock = threading.Lock()
        self.cancelled = self.timed_out = False

    def _kill(self):
        with self._lock:
            c = self._conn
        if c is not None and c.sock is not None:
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def cancel(self):
        self.cancelled = True
        self._kill()

    def _deadline(self):
        self.timed_out = True
        self._kill()

    def run(self, on_text=None, on_think=None):
        """-> {"content", "reasoning", "tool_calls": [{id, name, arguments}], "finish_reason", "usage", "timings"}"""
        host, port, root = _split(self.base_url)
        out = {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": None, "usage": None, "timings": None}
        calls = {}
        timer = threading.Timer(self.step_timeout, self._deadline)
        timer.daemon = True
        c = http.client.HTTPConnection(host, port, timeout=self.idle_timeout)
        with self._lock:
            self._conn = c
        timer.start()
        try:
            if self.cancelled:
                raise OAIError("cancelled", "cancelled")
            c.request("POST", root + "/v1/chat/completions", body=json.dumps(self.body).encode(),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            r = c.getresponse()
            if r.status != 200:
                raw = r.read(64 << 10).decode("utf-8", "replace")
                try:
                    j = json.loads(raw)
                    msg = j.get("error") if isinstance(j, dict) else None
                    msg = msg.get("message") if isinstance(msg, dict) else msg
                except ValueError:
                    msg = None
                raise OAIError(f"the server answered HTTP {r.status}: {(msg or raw)[:300]}", "http_%d" % r.status)
            ctype = r.getheader("Content-Type") or ""
            if "event-stream" not in ctype:          # a server that ignores stream=true
                j = json.loads(r.read(16 << 20).decode("utf-8", "replace"))
                msg = ((j.get("choices") or [{}])[0]).get("message") or {}
                out["content"] = msg.get("content") or ""
                out["reasoning"] = msg.get("reasoning_content") or ""
                if out["content"] and on_text:
                    on_text(out["content"])
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    calls[i] = {"id": tc.get("id") or f"call_{i}", "name": fn.get("name") or "",
                                "arguments": fn.get("arguments") or ""}
                out["finish_reason"] = ((j.get("choices") or [{}])[0]).get("finish_reason")
                out["usage"], out["timings"] = j.get("usage"), j.get("timings")
            else:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except ValueError:
                        continue
                    if j.get("error"):
                        e = j["error"]
                        raise OAIError(str(e.get("message") if isinstance(e, dict) else e)[:300], "server_error")
                    if j.get("timings"):
                        out["timings"] = j["timings"]
                    if j.get("usage"):
                        out["usage"] = j["usage"]
                    ch = (j.get("choices") or [None])[0]
                    if not ch:
                        continue
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        out["reasoning"] += d["reasoning_content"]
                        if on_think:
                            on_think(d["reasoning_content"])
                    if d.get("content"):
                        out["content"] += d["content"]
                        if on_text:
                            on_text(d["content"])
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", len(calls))
                        cur = calls.setdefault(i, {"id": None, "name": "", "arguments": ""})
                        if tc.get("id"):
                            cur["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            cur["name"] += fn["name"]
                        if fn.get("arguments"):
                            cur["arguments"] += fn["arguments"]
        except (OSError, http.client.HTTPException, ValueError) as e:
            if self.cancelled:
                raise OAIError("cancelled", "cancelled")
            if self.timed_out:
                raise OAIError(f"the server took longer than {int(self.step_timeout)} s for one step", "step_timeout")
            if isinstance(e, socket.timeout):
                raise OAIError(f"the server went quiet for {int(self.idle_timeout)} s", "step_timeout")
            raise OAIError(f"lost the connection to the server ({e.__class__.__name__})", "unreachable")
        finally:
            timer.cancel()
            with self._lock:
                self._conn = None
            c.close()
        if self.cancelled:
            raise OAIError("cancelled", "cancelled")
        if self.timed_out:
            raise OAIError(f"the server took longer than {int(self.step_timeout)} s for one step", "step_timeout")
        for i in sorted(calls):
            tc = calls[i]
            out["tool_calls"].append({"id": tc["id"] or f"call_{int(time.time() * 1000)}_{i}", "name": tc["name"],
                                      "arguments": tc["arguments"]})
        return out
