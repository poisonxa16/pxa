"""Minimal stdlib OpenAI-compatible client: streamed chat completions over SSE, with cancel and a per-step
deadline (both shut the socket down, which wakes a blocked read), plus small JSON GETs for /v1/models, /props,
/slots. No third-party code.

Stream rules (edge cases in the harness catalogue, ids in brackets):
- a llama-server sends NOTHING, not even headers, until the first token: the wait for the first byte has its own
  allowance (first_byte_timeout) with a progress callback, and the per-step deadline starts at the first byte [S12 K04];
- an SSE event may span several data: lines (joined per the SSE spec) [S04];
- a stream that ends without [DONE] and without a finish_reason was cut (server crash, proxy): it is an error
  ("incomplete"), never a finished answer [S06]; a stream with no readable event at all is "bad_stream" [S10];
- tool-call fragments without an index continue the open call; a name repeated in every chunk is kept once [T03 T04];
- an HTTP error keeps its status, Retry-After, and a plain-text message (an HTML error page is reduced to text) [K08 E04]."""
import http.client
import json
import re
import select
import socket
import threading
import time
import urllib.parse


class OAIError(Exception):
    def __init__(self, msg, code="server_error", status=None, retry_after=None):
        Exception.__init__(self, msg)
        self.code = code
        self.status = status
        self.retry_after = retry_after


def plain_error(raw, limit=300):
    """an error body as one short line: the JSON error message, else the text with any HTML markup removed."""
    raw = raw if isinstance(raw, str) else (raw or b"").decode("utf-8", "replace")
    msg = None
    try:
        j = json.loads(raw)
        msg = j.get("error") if isinstance(j, dict) else None
        if isinstance(msg, dict):
            msg = msg.get("message") or json.dumps(msg)[:limit]
    except ValueError:
        msg = None
    if not msg:
        msg = raw
        if re.search(r"(?i)<(html|body|head|title|h1|p|div)\b", msg):
            t = re.search(r"(?is)<title[^>]*>(.*?)</title>", msg)
            body = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", msg)
            body = re.sub(r"(?s)<[^>]+>", " ", body)
            msg = ((t.group(1).strip() + ": ") if t and t.group(1).strip() else "") + body
    return " ".join(str(msg).split())[:limit]


def _retry_after(r):
    try:
        v = float((r.getheader("Retry-After") or "").strip())
        return v if 0 <= v <= 3600 else None
    except (TypeError, ValueError):
        return None


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


def post_json(base_url, path, body, timeout=600.0, on_conn=None):
    """POST JSON, read JSON. on_conn(conn, True) is called once the request is sent and on_conn(conn, False) at the
    end, so a caller can shut the socket down from another thread to cancel the call (shutdown_conn)."""
    host, port, root = _split(base_url)
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        c.request("POST", root + path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        if on_conn is not None:
            on_conn(c, True)
        r = c.getresponse()
        raw = r.read(16 << 20)
        if r.status != 200:
            raise OAIError(f"HTTP {r.status}: {plain_error(raw)}", "http_%d" % r.status, r.status, _retry_after(r))
        return json.loads(raw.decode("utf-8", "replace"))
    except (OSError, http.client.HTTPException) as e:
        raise OAIError(f"{path}: {e.__class__.__name__}", "unreachable")
    finally:
        if on_conn is not None:
            on_conn(c, False)
        c.close()


def shutdown_conn(c):
    """wake a thread blocked on this connection (cancel): shut the socket down, both ways."""
    sock = getattr(c, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class Stream(object):
    """one streamed /v1/chat/completions call. cancel() and the deadlines all shut the socket down.

    step_timeout counts from the FIRST byte (generation); first_byte_timeout bounds the wait before it (prefill,
    a queue behind busy slots); idle_timeout is the longest silence between two chunks once the reply started."""

    def __init__(self, base_url, body, step_timeout=180.0, idle_timeout=120.0, first_byte_timeout=None,
                 on_wait=None, wait_every=15.0):
        self.base_url, self.body = base_url, dict(body, stream=True)
        self.step_timeout, self.idle_timeout = float(step_timeout), float(idle_timeout)
        # None keeps the old contract (the step deadline covers the wait too); a number gives the wait its own budget
        self.first_byte_timeout = float(first_byte_timeout) if first_byte_timeout else None
        self.on_wait, self.wait_every = on_wait, float(wait_every)
        self._conn = None
        self._lock = threading.Lock()
        self.cancelled = self.timed_out = False
        self.first_byte_s = None          # seconds from the request to the first byte
        self.events = 0                   # readable SSE events received

    def _kill(self):
        with self._lock:
            c = self._conn
        if c is not None:
            shutdown_conn(c)

    def cancel(self):
        self.cancelled = True
        self._kill()

    def _deadline(self):
        self.timed_out = True
        self._kill()

    def _wait_first_byte(self, c, t0):
        """block until the server starts answering (readable socket), the allowance runs out, or a cancel."""
        sock = c.sock
        if sock is None or not self.first_byte_timeout:
            return
        last = t0
        while True:
            try:
                ready, _, _ = select.select([sock], [], [], 0.5)
            except (OSError, ValueError):
                return                      # socket gone: getresponse() reports it
            if ready or self.cancelled:
                return
            now = time.time()
            if now - t0 >= self.first_byte_timeout:
                self.timed_out = True
                raise OAIError(f"the server did not start answering within {int(self.first_byte_timeout)} s "
                               "(it was still reading the prompt or waiting for a free slot)", "first_byte_timeout")
            if self.on_wait is not None and now - last >= self.wait_every:
                last = now
                try:
                    self.on_wait(now - t0)
                except Exception:           # noqa: BLE001 - a progress note must never break the call
                    pass

    def run(self, on_text=None, on_think=None):
        """-> {"content", "reasoning", "tool_calls": [{id, name, arguments}], "finish_reason", "usage", "timings"}"""
        host, port, root = _split(self.base_url)
        out = {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": None, "usage": None, "timings": None}
        calls = {}
        last_idx = [None]
        timer = threading.Timer(self.step_timeout, self._deadline)
        timer.daemon = True
        c = http.client.HTTPConnection(host, port, timeout=self.idle_timeout)
        with self._lock:
            self._conn = c
        saw_done = [False]
        t0 = time.time()
        try:
            if self.cancelled:
                raise OAIError("cancelled", "cancelled")
            if not self.first_byte_timeout:
                timer.start()
            c.request("POST", root + "/v1/chat/completions", body=json.dumps(self.body).encode(),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            self._wait_first_byte(c, t0)
            if self.cancelled:
                raise OAIError("cancelled", "cancelled")
            if self.first_byte_timeout:
                timer.start()
            r = c.getresponse()
            self.first_byte_s = round(time.time() - t0, 3)
            if r.status != 200:
                raw = r.read(64 << 10)
                raise OAIError(f"the server answered HTTP {r.status}: {plain_error(raw)}", "http_%d" % r.status,
                               r.status, _retry_after(r))
            ctype = r.getheader("Content-Type") or ""
            if "event-stream" not in ctype:          # a server that ignores stream=true
                j = json.loads(r.read(16 << 20).decode("utf-8", "replace"))
                if isinstance(j, dict) and j.get("error"):
                    e = j["error"]
                    raise OAIError(str(e.get("message") if isinstance(e, dict) else e)[:300], "server_error")
                msg = ((j.get("choices") or [{}])[0]).get("message") or {}
                out["content"] = msg.get("content") or ""
                out["reasoning"] = msg.get("reasoning_content") or ""
                if out["content"] and on_text:
                    on_text(out["content"])
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    calls[i] = {"id": tc.get("id") or f"call_{i}", "name": fn.get("name") or "",
                                "arguments": args if isinstance(args, str) else json.dumps(args or {})}
                out["finish_reason"] = ((j.get("choices") or [{}])[0]).get("finish_reason")
                out["usage"], out["timings"] = j.get("usage"), j.get("timings")
                saw_done[0] = True
                self.events = 1
            else:
                data = []

                def dispatch():
                    if not data:
                        return False
                    payload = "\n".join(data)
                    del data[:]
                    if payload.strip() == "[DONE]":
                        saw_done[0] = True
                        return True
                    try:
                        j = json.loads(payload)
                    except ValueError:
                        return False
                    if not isinstance(j, dict):
                        return False
                    self.events += 1
                    if j.get("error"):
                        e = j["error"]
                        raise OAIError(str(e.get("message") if isinstance(e, dict) else e)[:300], "server_error")
                    if j.get("timings"):
                        out["timings"] = j["timings"]
                    if j.get("usage"):
                        out["usage"] = j["usage"]
                    ch = (j.get("choices") or [None])[0]
                    if not isinstance(ch, dict):
                        return False
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
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") or {}
                        i = tc.get("index")
                        if not isinstance(i, int):
                            # no index [T03]: a known id continues its call; an id or a name starts a new one;
                            # a bare arguments fragment continues the open call
                            hit = [k for k, v in calls.items() if tc.get("id") and v["id"] == tc.get("id")]
                            if hit:
                                i = hit[0]
                            elif tc.get("id") or (fn.get("name") and last_idx[0] is None):
                                i = len(calls)
                            elif fn.get("name") and last_idx[0] is not None and calls[last_idx[0]]["name"] \
                                    and fn.get("name") != calls[last_idx[0]]["name"]:
                                i = len(calls)
                            else:
                                i = last_idx[0] if last_idx[0] is not None else len(calls)
                        last_idx[0] = i
                        cur = calls.setdefault(i, {"id": None, "name": "", "arguments": ""})
                        if tc.get("id") and not cur["id"]:
                            cur["id"] = tc["id"]
                        name = fn.get("name")
                        if name and name != cur["name"]:          # a name repeated in every chunk is kept once [T04]
                            cur["name"] += name
                        if fn.get("arguments"):
                            a = fn["arguments"]
                            cur["arguments"] += a if isinstance(a, str) else json.dumps(a)
                    return False

                while True:
                    line = r.readline(4 << 20)
                    if not line:
                        dispatch()
                        break
                    line = line.decode("utf-8", "replace").rstrip("\r\n").lstrip(" \t")
                    if line == "":
                        if dispatch():
                            break
                        continue
                    if line.startswith(":"):
                        continue                         # comment / keep-alive
                    field, _, value = line.partition(":")
                    if field != "data":
                        continue                         # event:, id:, retry:
                    if value.startswith(" "):
                        value = value[1:]
                    if data and value.strip() == "[DONE]":
                        dispatch()
                    data.append(value)
                    # one-line events (every server we know) dispatch at once, without waiting for the blank line
                    if len(data) == 1 and (value.strip() == "[DONE]" or _complete_json(value)):
                        if dispatch():
                            break
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
        if not saw_done[0] and not out["finish_reason"]:
            if not self.events:
                raise OAIError("the server closed the stream without sending anything readable", "bad_stream")
            raise OAIError("the server closed the stream before the reply finished", "incomplete")
        if not self.events and not out["content"] and not out["reasoning"]:
            raise OAIError("the server's stream had no readable events", "bad_stream")
        for i in sorted(calls):
            tc = calls[i]
            out["tool_calls"].append({"id": tc["id"] or f"call_{int(time.time() * 1000)}_{i}", "name": tc["name"],
                                      "arguments": tc["arguments"]})
        return out


def _complete_json(v):
    """True when this data line is one whole JSON value (the common case: dispatch it without the blank line)."""
    v = v.strip()
    if not v or v[0] not in "{[":
        return False
    try:
        json.loads(v)
        return True
    except ValueError:
        return False
