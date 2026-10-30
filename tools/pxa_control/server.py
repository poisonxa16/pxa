"""PXA Control: the HTTP server -- the request handler (guards, auth, dispatch, the engine proxy, the log stream),
the bounded threading server, and serve(), the entry point `pxa` / `pxa-launch --gui` calls."""
from .core import *  # noqa: F401,F403
from .auth import *  # noqa: F401,F403
from .bridge import *  # noqa: F401,F403
from .routes import *  # noqa: F401,F403
import http.client
import select


def _tracking_opener(socks):
    """a urllib opener whose upstream connection hands its socket to `socks` once connected, so another thread
    can shut it down (the engine proxy's client watch). Everything else is urllib's default opener."""
    def connect(conn):
        http.client.HTTPConnection.connect(conn)
        socks.append(conn.sock)
    conn_cls = type("PXATrackedConnection", (http.client.HTTPConnection,), {"connect": connect})

    class Handler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(conn_cls, req)
    return urllib.request.build_opener(Handler)


def _shut(socks):
    for s in list(socks):
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "PXAControl/1"
    app = None          # set on the subclass made by make_handler
    timeout = REQUEST_TIMEOUT_S       # H7: StreamRequestHandler sets it on the socket; a stalled client frees its thread

    def handle(self):
        # a client that drops the connection mid-reply (health checkers, a closed tab) is not an
        # error worth a traceback in the log
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        if os.environ.get("PXA_CONTROL_ACCESS_LOG"):          # H8: the request line carries ?token= on the first visit
            sys.stderr.write("pxa-control: " + scrub_secrets(fmt % args, (self.app.token,) if self.app else ()) + "\n")

    def end_headers(self):
        xfo = getattr(self, "_xfo", None)                     # a plugin's own page (S3) may be framed by this same origin, nothing else
        for k, v in SECURITY_HEADERS:                         # H9: on every answer, streams and redirects included
            self.send_header(k, xfo if (k == "X-Frame-Options" and xfo) else v)
        super().end_headers()

    # ---- guards ---------------------------------------------------------------------------
    def _host_ok(self):
        host = (self.headers.get("Host") or "").strip().lower()
        if not host:
            return False
        if self.app.lan:
            return True            # the token is the guard on a LAN bind
        h = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        return h in ("127.0.0.1", "localhost", "[::1]")

    def _origin_ok(self):
        return origin_matches_host(self.headers)

    def _cookie_tokens(self):
        """the values of this Control's cookie, then of the legacy shared name (H3)."""
        mine, legacy = [], []
        c = self.headers.get("Cookie") or ""
        for part in c.split(";"):
            k, _, v = part.strip().partition("=")
            if k == getattr(self.app, "cookie_name", None):
                mine.append(v)
            elif k == TOKEN_COOKIE:
                legacy.append(v)
        return mine + legacy

    def _cookie_token(self):
        t = self._cookie_tokens()
        return t[0] if t else None

    def _auth(self, query, method="GET"):
        """-> (ok, set_cookie). Token from cookie, X-PXA-Token header, Authorization: Bearer, or ?token= on a GET
        (the link the launcher opens; H10: a POST/DELETE never takes it from the query string)."""
        if not self.app.token:
            return True, False
        auth = self.headers.get("Authorization") or ""
        bearer = auth[7:].strip() if auth[:7].lower() == "bearer " else None     # a Prometheus scrape job
        cands = [(c, False) for c in self._cookie_tokens()]
        cands += [(self.headers.get("X-PXA-Token"), False), (bearer, False)]
        if method in ("GET", "HEAD"):
            cands.append(((query.get("token") or [None])[0], True))
        for cand, from_query in cands:
            if cand and hmac.compare_digest(str(cand).encode("utf-8", "replace"), self.app.token.encode()):
                return True, from_query
        return False, False

    def _needs_token(self, method, path):
        """H1: False only for what a Control on this machine shows without the token."""
        if not self.app.token:
            return False
        if self.app.lan:
            return True
        if method in ("GET", "HEAD") and path in OPEN_ON_LOOPBACK:
            return False
        if method in ("GET", "HEAD") and path in STATIC and path not in PAGE_PATHS:
            return False                     # js / css / icons: public code, nothing of the user's
        return True

    # ---- responses ------------------------------------------------------------------------
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        extra = dict(extra or {})
        self._xfo = extra.pop("X-Frame-Options", None)
        if ctype.startswith("text/html") and "Content-Security-Policy" not in extra:
            self.send_header("Content-Security-Policy", page_csp(body))      # H9: hashes, not 'unsafe-inline'
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json_body(self, limit=MAX_BODY):
        n = content_length(self.headers, limit)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            raise Invalid("request body is not JSON")
        if not isinstance(body, dict):              # every route reads fields: a list or a number was a 500 before
            raise Invalid("request body must be a JSON object")
        return body

    def _login_page(self):
        why = ("This rig's GUI is on the network; enter the token printed where it was started." if self.app.lan else
               "Open PXA Control from the terminal that started it (its address ends in ?token=...), or run "
               "<code>pxa</code> again: it opens this page signed in. Or paste the token here.")
        return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                "<title>PXA Control</title><body style='font-family:system-ui;background:#111214;color:#ECEDEF;"
                "display:grid;place-items:center;height:100vh;margin:0'><form method=get "
                "style='display:grid;gap:10px;min-width:280px;max-width:420px'><b>PXA Control</b><span style='color:#A0A4AC'>"
                + why + "</span>"
                "<input name=token autocomplete=off style='padding:8px;border-radius:8px;border:1px solid #33363C;"
                "background:#1D1F23;color:inherit'><button style='padding:8px;border-radius:8px;border:0;"
                "background:#E08A3C;color:#111214;font-weight:700'>Open</button></form>")

    # ---- routing --------------------------------------------------------------------------
    def do_HEAD(self):
        p = urllib.parse.urlparse(self.path).path
        if p not in STATIC:            # a HEAD on /api/log/stream used to hold a thread forever
            return self._send(405, {"error": "HEAD only on the page"})
        self.do_GET()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        self._xfo = None
        u = urllib.parse.urlparse(self.path)
        path, query = u.path, urllib.parse.parse_qs(u.query)
        if not self._host_ok():
            return self._send(421, err_body("unexpected Host header", "bad_host", 421))
        if path.startswith("/plugins/"):             # S4: a plugin's OBS overlay; its own key, loopback peers only (plugins/overlay.py)
            from .plugins import host as _ph, overlay as _pov
            try:
                return _pov.serve(self, _ph.host_of(self.app), method, path, query)
            except (BrokenPipeError, ConnectionResetError):
                return
        api = path.startswith("/api/") or path == "/metrics"
        # H2: the API (and every state change) answers this page and the user's own tools only: not a page on
        # another site, not one on another port of this machine, not a sandboxed or file: page (Origin: null)
        if (api or method not in ("GET", "HEAD")) and not (fetch_site_ok(self.headers) and self._origin_ok()):
            return self._send(403, err_body("cross-origin request refused", "cross_origin", 403))
        ok, set_cookie = True, False
        if self._needs_token(method, path):
            ok, set_cookie = self._auth(query, method)
            if not ok:
                # H7: past AUTH_FAIL_MAX wrong/missing tokens a minute, an address gets a bare 429 (no login page,
                # no JSON to build). A right token always passes: on loopback every client is 127.0.0.1, and a
                # flood from another local process must not lock the owner out. Brute force is answered by the
                # token's size (192 bits per session, 144 on --lan), not by this counter.
                who = self.client_address[0] if self.client_address else "?"
                if self.app.auth_throttle.fail(who) > self.app.auth_throttle.limit:
                    return self._send(429, err_body("too many wrong tokens from this address: wait a minute",
                                                    "rate_limited", 429), extra={"Retry-After": str(int(AUTH_FAIL_WINDOW_S))})
                if path == "/metrics":
                    return self._send(401, "token required (Authorization: Bearer <token>)\n", "text/plain; charset=utf-8")
                if method == "GET" and not path.startswith("/api/"):
                    return self._send(401, self._login_page(), "text/html; charset=utf-8")
                return self._send(401, err_body("token required", "unauthorized", 401))
        self.app.last_request = time.time()
        _REQ.who = f"page {self.client_address[0]}" if self.client_address else "api"     # a page is looking: a background Control stays up
        if set_cookie and method == "GET":
            # drop the token from the address bar once it is in a cookie
            q = {k: v for k, v in query.items() if k != "token"}
            loc = path + (("?" + urllib.parse.urlencode(q, doseq=True)) if q else "")
            self.send_response(303)
            self.send_header("Location", loc)
            self.send_header("Set-Cookie", f"{self.app.cookie_name}={self.app.token}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Strict")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        try:
            if method == "GET" and path in STATIC:
                fn, ctype = STATIC[path]
                try:
                    with open(os.path.join(UI_DIR, fn), "rb") as f:
                        data = f.read()
                except OSError:
                    return self._send(404, {"error": f"{fn} missing from {UI_DIR}"})
                return self._send(200, data, ctype)
            if path.startswith("/api/engine/"):
                return self._engine_proxy(method, path[len("/api/engine/"):], u.query)
            if method == "GET" and path == "/api/log/stream":
                return self._log_stream(query)
            if path.startswith("/api/plugins/") and not any(p == path for (_m, p) in ROUTES):
                return self._plugin_forward(method, path, u.query)
            route = ROUTES.get((method, path))
            if route is None:
                if any(p == path for (_m, p) in ROUTES):
                    return self._send(405, err_body(f"{method} is not allowed on {path}", "method_not_allowed", 405))
                return self._send(404, err_body("not found", "not_found", 404))
            body = self._json_body(BODY_LIMITS.get(path, MAX_BODY)) if method in ("POST", "DELETE") else None
            res = route(self.app, body, query)
            if hasattr(res, "stream_to"):          # a live event stream (the chat agent's /api/chat/events)
                return res.stream_to(self)
            if isinstance(res, Reply):
                return self._send(res.code, res.body, res.ctype, res.extra)
            return self._send(200, res)
        except Invalid as e:
            return self._send(getattr(e, "status", 400), err_body(e, getattr(e, "code", "invalid"), getattr(e, "status", 400)))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            if HF is not None and isinstance(e, HF.HFError):  # a download/list failure, with the status HF gave us
                return self._send(e.status, err_body(e, e.code, e.status))
            if CTL_ERRORS and isinstance(e, CTL_ERRORS):      # the Profiles backend's refusals: locked 423, forbidden 403, ...
                return self._send(e.status, err_body(e, e.code, e.status, getattr(e, "detail", None)))
            import traceback
            traceback.print_exc()
            msg = scrub_secrets(f"{e.__class__.__name__}: {e}", (self.app.token,))      # H8
            return self._send(500, err_body(msg, "internal", 500))

    # ---- plugin forward (S2) -----------------------------------------------------------------
    def _plugin_forward(self, method, path, query):
        """/api/plugins/<id>/* -> the plugin's process. Reached only after the Host, Origin/fetch-metadata and token checks
        above; only method, path, query, Content-Type and the body cross; the answer is never HTML and is sandboxed."""
        from .plugins import host as _ph, PluginError as _PE
        pid = path.split("/")[3] if path.count("/") >= 4 else ""
        h = _ph.host_of(self.app)
        if not h.knows(pid):
            return self._send(404, err_body("not found", "not_found", 404))
        body = b""
        if method in ("POST", "DELETE"):
            n = content_length(self.headers, BODY_LIMITS.get(path, MAX_BODY))
            body = self.rfile.read(n) if n else b""
        try:
            status, ctype, data = h.forward(pid, method, path, query, self.headers.get("Content-Type"), body)
        except _PE as e:
            return self._send(getattr(e, "status", 400), err_body(e, e.code, getattr(e, "status", 400)))
        return self._send(status, data, ctype, {"Content-Security-Policy": _ph.PLUGIN_CSP})

    # ---- SSE log --------------------------------------------------------------------------
    def _log_stream(self, query):
        try:
            since = int((query.get("since") or ["0"])[0])
        except ValueError:
            since = 0
        seat = self.app.get_seat(_q(query, "sid") or "main", create=not _q(query, "sid"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_ping = time.time()
        try:
            while True:
                items = seat.lines_since(since, 500)
                if items:
                    since = items[-1][0]
                    payload = json.dumps({"seq": since, "lines": [x[1] for x in items]})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                elif time.time() - last_ping > 15:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_ping = time.time()
                with seat.cond:
                    seat.cond.wait(timeout=1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    # ---- engine proxy ---------------------------------------------------------------------
    def _watch_client(self, socks, gone, done):
        """until done: if the client hangs up, mark gone and shut the upstream socket(s)."""
        conn = self.connection
        while not done.is_set():
            try:
                ready, _, _ = select.select([conn], [], [], 0.5)
            except (OSError, ValueError):
                return
            if not ready or done.is_set():
                continue
            try:
                peek = conn.recv(1, socket.MSG_PEEK)
            except BlockingIOError:
                continue
            except OSError:
                peek = b""
            if peek == b"":
                gone.set()
                _shut(socks)
                return
            done.wait(0.5)                      # pipelined bytes, not a hang-up: look again later

    def _engine_proxy(self, method, sub, qs):
        allowed = ENGINE_GET if method == "GET" else (ENGINE_POST if method == "POST" else set())
        if sub not in allowed:
            return self._send(404, {"error": f"{method} /{sub} is not proxied"})
        query = urllib.parse.parse_qs(qs or "")
        port = self.app.target_port(query)       # Invalid (400) for a port this GUI does not know
        if not port:
            return self._send(503, {"error": "no server running"})
        fwd = urllib.parse.urlencode({k: v for k, v in query.items() if k not in ("sid", "port", "token")}, doseq=True)
        url = f"http://127.0.0.1:{port}/{sub}" + (("?" + fwd) if fwd and method == "GET" else "")
        data = None
        think_info = None
        hdr = {"Accept": self.headers.get("Accept") or "*/*"}
        if method == "POST":
            try:
                n = content_length(self.headers, MAX_BODY * 8)
            except Invalid as e:
                return self._send(413 if "large" in str(e) else 400, {"error": str(e)})
            data = self.rfile.read(n)
            hdr["Content-Type"] = "application/json"
            if sub == "v1/chat/completions":
                data, think_info = self.app.thinking_rewrite(data, query)
        req = urllib.request.Request(url, data=data, headers=hdr, method=method)
        # Watch the client while the engine works: llama-server sends nothing until the first token, so a closed tab
        # (or Stop) during a long prefill used to go unnoticed until then. On EOF from the client the upstream socket
        # is shut, which the engine sees (is_connection_closed) and frees the slot [S16 X10].
        socks, gone, done = [], threading.Event(), threading.Event()
        watcher = threading.Thread(target=self._watch_client, args=(socks, gone, done), daemon=True,
                                   name="pxa-proxy-watch")
        watcher.start()
        try:
            r = _tracking_opener(socks).open(req, timeout=900)
        except urllib.error.HTTPError as e:
            done.set()
            body = e.read()
            return self._send(e.code, body, e.headers.get("Content-Type") or "application/json",
                              extra={"Content-Security-Policy": SANDBOX_CSP})
        except Exception as e:
            done.set()
            if gone.is_set():
                self.close_connection = True
                return                         # the client left first: nobody to answer
            return self._send(502, {"error": f"server unreachable: {e.__class__.__name__}"})
        ctype = r.headers.get("Content-Type") or "application/octet-stream"
        streaming = "event-stream" in ctype
        if not streaming:
            try:
                body = r.read()
            except OSError:
                done.set()
                if gone.is_set():
                    self.close_connection = True
                    return
                raise
            done.set()
            if sub in ("v1/chat/completions", "completion"):
                try:
                    j = json.loads(body)
                    self.app.record_timings(j.get("timings"), j.get("model"))
                except Exception:
                    pass
            if sub == "pxa/speed":
                ctype = "text/html; charset=utf-8"
            return self._send(r.status, body, ctype,
                              extra=dict(_think_header(think_info), **{"Content-Security-Policy": SANDBOX_CSP}))
        self.send_response(r.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", SANDBOX_CSP)
        for k, v in _think_header(think_info).items():
            self.send_header(k, v)
        self.end_headers()
        tail = b""
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                tail = (tail + chunk)[-65536:]
        except (BrokenPipeError, ConnectionResetError, OSError, http.client.HTTPException):
            # the client left, or stopped reading for REQUEST_TIMEOUT_S (a socket.timeout on the write) [S14]:
            # close the upstream so the engine stops generating for nobody, and drop this connection
            _shut(socks)
            r.close()
            done.set()
            self.close_connection = True
            return
        finally:
            done.set()
        for line in reversed(tail.split(b"\n")):
            if line.startswith(b"data:") and b'"timings"' in line:
                try:
                    j = json.loads(line[5:].strip())
                    self.app.record_timings(j.get("timings"), j.get("model"))
                except Exception:
                    pass
                break


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    max_connections = MAX_CONNECTIONS
    _BUSY_BODY = b'{"error": "too many connections", "code": "busy", "status": 503}\n'
    _BUSY = (b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\n"
             b"Retry-After: 5\r\nContent-Length: " + str(len(_BUSY_BODY)).encode() + b"\r\n\r\n" + _BUSY_BODY)

    def __init__(self, *a, **k):
        self._slots = threading.BoundedSemaphore(self.max_connections)     # H7: a thread per connection, bounded
        super().__init__(*a, **k)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(self._BUSY)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def make_server(app, host, port):
    H = type("PXAControlHandler", (Handler,), {"app": app})
    return _Server((host, port), H)


def _quiet(fn):
    try:
        fn()
    except Exception:
        pass


def _lan_addresses():
    out = []
    try:
        r = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
        out = [x for x in r.stdout.split() if ":" not in x]
    except Exception:
        pass
    return out or ["<this-machine>"]


SERVING = {}


def _mins(sec):
    return f"{sec / 60:.0f} min" if sec >= 60 else f"{sec:.0f} s"


def has_desktop(environ=None):
    env = environ if environ is not None else os.environ
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")) or sys.platform in ("darwin", "win32")


def headless_hint(port, lan=False, environ=None, user=None):
    """-> [lines]: the "no desktop here" help, or [] when there is a display or Control is already on
    the LAN. Headless = neither DISPLAY nor WAYLAND_DISPLAY (an SSH session without X forwarding is
    that case). The ssh line uses the server address SSH_CONNECTION carries (3rd field) and the
    login name; where they are unknown it keeps <user> / <this-machine> placeholders."""
    env = environ if environ is not None else os.environ
    if lan or has_desktop(env):
        return []
    parts = (env.get("SSH_CONNECTION") or "").split()
    host = parts[2] if len(parts) >= 4 else "<this-machine>"
    if user is None:
        try:
            import getpass
            user = getpass.getuser()
        except Exception:       # noqa: BLE001
            user = "<user>"
    port = int(port)
    return [
        "  No desktop here, and PXA Control listens on this machine only (127.0.0.1), so another computer cannot reach it as it is.",
        f"  A) rerun with `pxa --gui --lan`: it listens on every interface (0.0.0.0), port {DEFAULT_PORT} by default, plain http,",
        "     behind a random access token. Open the address it prints (it ends in ?token=...); a cookie remembers it after that.",
        "     No other approval step. Use it on a network you trust.",
        f"  B) or from your own computer: ssh -L {port}:127.0.0.1:{port} {user}@{host}   then open the PXA Control address above",
        "     The text menu instead: pxa --tui",
    ]


def open_in_browser(url, wait=False):
    """open url in this desktop's browser (in the background unless wait); False when there is no
    desktop here."""
    if not has_desktop():
        return False

    def _open():
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:       # noqa: BLE001
            pass
    if wait:
        _open()
    else:
        threading.Thread(target=_open, daemon=True).start()
    return True


def serve(L, port=DEFAULT_PORT, lan=False, open_browser=True, models_dirs=None, companion=False, idle_s=None):
    """Entry point for `pxa-launch --gui` (and a bare `pxa`). Returns an exit code. companion=True: a
    Control the launcher started in the background next to a server; it closes itself after idle_s
    (CONTROL_IDLE_S) with no server running and no page asking."""
    app = App(L, port=port, lan=lan, models_dirs=models_dirs)
    app.companion = bool(companion)
    if idle_s is None:
        try:                                     # PXA_CONTROL_IDLE_S: the 10-minute rule, in seconds
            idle_s = float(os.environ.get("PXA_CONTROL_IDLE_S") or CONTROL_IDLE_S)
        except ValueError:
            idle_s = CONTROL_IDLE_S
    app.idle_s = idle_s
    host = "0.0.0.0" if lan else "127.0.0.1"
    try:
        srv = make_server(app, host, port)
    except OSError as e:
        print(f"pxa-launch --gui: cannot listen on {host}:{port} ({e}). Pick another with --port.",
              file=sys.stderr)
        return 2
    try:
        run_rec = write_run_record(port, lan, companion, app.token)
    except OSError:
        run_rec = None
    SERVING[port] = srv                          # who is listening where (tests stop one through it)
    url = control_url(port, lan, app.token)
    # H8: the sign-in link goes to a terminal, never into a log file (a background Control writes control.log).
    # A --lan Control in a container prints its token on purpose (docker logs is how you find it) unless you
    # gave it one yourself in PXA_CONTROL_TOKEN.
    try:
        tty = sys.stdout.isatty()
    except (AttributeError, ValueError):
        tty = False
    own_token = bool(os.environ.get("PXA_CONTROL_TOKEN"))
    show_secret = tty or (lan and not own_token)
    print("=" * 78)
    if show_secret:
        print(f"PXA Control: {url}")
    else:
        print(f"PXA Control: {control_url(port, lan)}")
        print("  sign in: run `pxa` again (it opens this Control signed in)" if not lan else
              "  sign in with the token in PXA_CONTROL_TOKEN")
    if companion:
        print(f"  started next to a server (pid {os.getpid()}); it closes itself {_mins(idle_s)} after the last "
              "server stops with no page open.")
    else:
        print("  the launcher in a browser. Ctrl-C stops it (and the servers it started).")
    if lan:
        shown_tok = app.token if show_secret else "$PXA_CONTROL_TOKEN"
        print(f"  listening on every interface, port {port}. Access token (keep it private):")
        print(f"    {shown_tok}")
        ips = _lan_addresses()
        for ip in ips[:3]:
            print(f"  open  http://{ip}:{port}/?token={shown_tok}")
        if len(ips) > 3:
            print(f"        (or any of this machine's other {len(ips) - 3} addresses, same port and token)")
    else:
        print("  this machine only (--lan to share it)")
        if not companion:
            for _ln in headless_hint(port, lan):
                print(_ln)
    print(f"  config: {config_path()}   levers: {len(app.catalog)} from {app.catalog_src or 'NOT FOUND'}")
    tel = app.start_telemetry()                   # the Live tab's history on disk (PXA_CONTROL_TELEMETRY=0: off)
    if tel is not None:
        print(f"  history: {tel.path} (a sample every {app.tel_cfg['sample_s']:g} s, kept {app.tel_cfg['raw_days']:g} days, "
              f"minutes {app.tel_cfg['rollup_days']:g} days)" + ("" if tel.is_writer else
              f"; another PXA Control (pid {tel.writer_pid()}) records it, this one reads"))
    elif app.tel_error:
        print(f"  history: off ({app.tel_error})")
    mq = app.start_mqtt()                         # Home Assistant over MQTT (PXA_CONTROL_MQTT=1: on)
    if mq is not None:
        print(f"  home assistant: mqtt -> {app.mqtt_cfg['host']}:{app.mqtt_cfg['port']} "
              f"every {app.mqtt_cfg['interval']:g} s under '{app.mqtt_cfg['base']}'")
    elif app.mqtt_error:
        print(f"  home assistant: off ({app.mqtt_error})")
    if app.gpuctl is not None:                    # Profiles: background only when a switch needs it (default: none)
        g = app.gpuctl
        print(f"  GPU profiles: adapter {g.adapter.kind}; changes {'ALLOWED' if g.settings['allow_gpu_control'] else 'off'}"
              f", auto-start {'on' if g.settings['autostart'] else 'off'}, schedules {'on' if g.settings['schedules'] else 'off'}"
              + (f", {len(g.guard.locks())} lock(s) in force" if g.guard.locks() else ""))
        if g.needs_background():
            g.start_background()
            try:
                b = g.boot()
                if b["reapplied"] or b["queued"] or b["errors"]:
                    print(f"  GPU profiles at start: re-applied {len(b['reapplied'])}, queued {len(b['queued'])} auto-start(s)"
                          + (f", errors: {'; '.join(b['errors'])}" if b["errors"] else ""))
            except Exception as e:      # noqa: BLE001
                print(f"  GPU profiles at start: {e.__class__.__name__}: {e}")
    elif app.gpuctl_error:
        print(f"  GPU profiles: off ({app.gpuctl_error})")
    try:                                          # grants refresh on their own: 12 h +- 1 h, on wake, after a redeem (PXA_STORE_SYNC=0: off)
        from . import store as _store
        if getattr(app, "store_cache", None) is None:
            app.store_cache = _store.EntitlementCache(config_dir())
        app.store_sync = _store.start_background(config_dir(), app.store_cache)
    except Exception as e:      # noqa: BLE001
        app.store_sync = None
        print(f"  store sync: off ({e.__class__.__name__})")
    try:                                          # installed plugins with consent (and a grant) start with Control (PXA_PLUGIN_AUTOSTART=0: off)
        from .plugins import pxp as _pxp, host as _ph
        if os.environ.get("PXA_PLUGIN_AUTOSTART", "1").strip() != "0" and _pxp.installed(config_dir()):
            started = _ph.host_of(app).autostart()
            if started:
                print("  plugins: started " + ", ".join(started))
    except Exception as e:      # noqa: BLE001
        print(f"  plugins: not started ({e.__class__.__name__})")
    mdns_adv = None
    if lan:                                       # mDNS: _pxa-control._tcp on the home network, no secret in it (PXA_CONTROL_MDNS=0: off)
        try:
            from . import mdns as _mdns
            mdns_adv, _line = _mdns.start_for(port, CONTROL_VERSION, lan)
            if _line:
                print(f"  mdns: {_line}")
        except Exception as e:      # noqa: BLE001
            print(f"  mdns: off ({e.__class__.__name__})")
    print("=" * 78)
    sys.stdout.flush()
    # read the model headers once in the background: the first scan of a spinning-disk library takes
    # tens of seconds, and the page should not wait on it at the first visit
    threading.Thread(target=lambda: _quiet(app.models), daemon=True).start()
    if open_browser and not lan:
        open_in_browser(url)                      # signed in: the session token rides in the address once
    stop_ev = threading.Event()
    if companion:
        def _idle_watch():
            while not stop_ev.wait(min(CONTROL_IDLE_POLL_S, max(0.2, idle_s / 4))):
                try:
                    idle = app.idle_seconds()
                except Exception:       # noqa: BLE001
                    continue
                if idle >= idle_s:
                    print(f"PXA Control: no server running and no page open for {_mins(idle)}: closing.")
                    sys.stdout.flush()
                    srv.shutdown()
                    return
        threading.Thread(target=_idle_watch, daemon=True, name="pxa-control-idle").start()
    try:
        import signal

        def _term(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)      # `docker stop` / kill <pid>: stop the seat too
        # a GUI started in the background (`... &`, nohup) inherits SIGINT=ignored: take it back,
        # so kill -INT stops it (and its servers) the same way Ctrl-C does
        signal.signal(signal.SIGINT, _term)
    except (ValueError, OSError):
        pass
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\npxa-launch --gui: stopping.")
    finally:
        stop_ev.set()
        SERVING.pop(port, None)
        if mdns_adv is not None:                  # goodbye packets: the rig leaves other routers' lists at once
            _quiet(mdns_adv.stop)
        if app.gpuctl is not None:                # first: no auto-start may restart a server we are about to stop
            _quiet(app.gpuctl.stop)
        if run_rec:
            try:
                os.unlink(run_rec)
            except OSError:
                pass
        for seat in list(app.seats.values()):
            if seat.running():
                print("pxa-launch --gui: stopping the server it started (%s, pid %s)." % (seat.sid, seat.proc.pid))
                seat.stop()
        if getattr(app, "store_sync", None) is not None:
            _quiet(app.store_sync.stop)
        if getattr(app, "plugin_host", None) is not None:      # plugins stop with Control
            _quiet(app.plugin_host.close)
        app.encode_shutdown()
        app.live.close()
        _quiet(app.stop_telemetry)
        _quiet(app.stop_mqtt)
        srv.server_close()
    return 0


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]   # star-import carries the _private names too
