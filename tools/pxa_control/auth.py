"""PXA Control: the security policy (threat model THREAT-MODEL.md, fixes H1-H10): fetch-metadata and Origin
checks, the hashed-script CSP and the headers every answer carries, the wrong-token throttle, the secret scrubber.
Nothing here imports the rest of Control at module level except core."""
from .core import *  # noqa: F401,F403


# ---------------------------------------------------------------------------------------------
# security policy (threat model THREAT-MODEL.md: H1-H10). One block on purpose: the
# package split moves it whole into pxa_control/auth.py.
# ---------------------------------------------------------------------------------------------
# H1: on a Control bound to this machine these answer without the token (monitoring; read-only; no secrets).
# Everything else under /api/ and the engine proxy need it. The page shell, its scripts, styles and icons are
# public code (the same files every package ships) and open on loopback and on --lan alike: a browser navigation
# cannot carry the X-PXA-Token header, so the shell loads first and signs the tab in from sessionStorage
# (the token is never a cookie: browsers send a host's cookies to every
# port of it). On --lan nothing under /api/ is open, /api/health and /metrics included.
OPEN_ON_LOOPBACK = {"/api/health", "/metrics"}
PAGE_PATHS = ("/", "/index.html", "/share.html")     # every HTML page: a ?token= visit redirects to <page>#token=...
LOGIN_PATH = "/login"                                # the sign-in form the shell sends a tab without a token to
PUBLIC_GET = set()                                   # API paths that serve public code like STATIC: the Vigil page and HUD
                                                     # shells and their fixed asset list (pxa_chat.public_code); they load
                                                     # /auth.js first, every data route behind them still needs the token
TICKET_TTL_S = 60.0                                  # a navigation ticket lives this long, and is good for one GET
TICKET_MAX = 256                                     # unredeemed tickets kept; the oldest go first

# H9: headers on every response (end_headers), and the two content security policies.
SECURITY_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Permissions-Policy", "camera=(), geolocation=(), usb=(), serial=(), hid=(), bluetooth=(), payment=(), "
                           "display-capture=(), microphone=(self)"),     # the chat's voice input uses the microphone
)
# what the engine's own HTML (/pxa/speed, an engine error page) may do when it comes through the proxy: run its
# inline scripts in an opaque origin (no cookie, no same-origin API), load nothing from anywhere.
SANDBOX_CSP = ("sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
               "img-src data:; frame-ancestors 'none'")
_INLINE_SCRIPT_RE = re.compile(rb"<script\b([^>]*)>(.*?)</script\s*>", re.I | re.S)
_SRC_ATTR_RE = re.compile(rb"\bsrc\s*=", re.I)
_CSP_CACHE = {}


def inline_script_hashes(html):
    """the CSP source expressions ('sha256-...') of every inline <script> block in an HTML document (bytes).
    Browsers hash the element's text after newline normalisation (CR LF and CR become LF)."""
    import base64
    import hashlib
    out = []
    for attrs, body in _INLINE_SCRIPT_RE.findall(html or b""):
        if _SRC_ATTR_RE.search(attrs):
            continue
        body = body.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        h = "'sha256-" + base64.b64encode(hashlib.sha256(body).digest()).decode() + "'"
        if h not in out:
            out.append(h)
    return out


def page_csp(html):
    """the page's CSP: scripts from this origin or one of the page's own inline blocks (hashed now, so the page
    may change without a server change), never 'unsafe-inline' or eval; inline styles stay allowed."""
    import hashlib
    key = hashlib.sha256(html or b"").digest()
    v = _CSP_CACHE.get(key)
    if v is None:
        scripts = " ".join(["'self'"] + inline_script_hashes(html))
        v = (f"default-src 'self'; script-src {scripts}; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
             "font-src 'self' data:; connect-src 'self'; worker-src 'self'; manifest-src 'self'; media-src 'self' blob:; "
             "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
        if len(_CSP_CACHE) > 32:
            _CSP_CACHE.clear()
        _CSP_CACHE[key] = v
    return v


_TOKEN_QS_RE = re.compile(r"(?i)([?&](?:token|access_token|ticket|key|k)=)[^&\s\"']+")   # k= : a plugin overlay key


def scrub_secrets(text, secrets_=()):
    """a log line or an error message with the Control token, the query-string tokens and anything the bug-report
    redactor calls a secret replaced (H8). Paths, hosts and names are left alone: this is not the report redactor."""
    if not isinstance(text, str) or not text:
        return text
    for v in secrets_:
        if v and len(str(v)) >= 6:
            text = text.replace(str(v), "<redacted>")
    text = _TOKEN_QS_RE.sub(r"\1<redacted>", text)
    for rx, rep in _SECRET_PATTERNS:
        text = rx.sub(rep, text)
    return text


class AuthThrottle(object):
    """H7: wrong or missing tokens per client address. Past AUTH_FAIL_MAX inside AUTH_FAIL_WINDOW_S, that address's
    wrong tokens get a bare 429 until the window has passed. A right token never counts, never resets the count,
    and always passes."""

    def __init__(self, limit=None, window=None):
        self.limit = AUTH_FAIL_MAX if limit is None else limit
        self.window = AUTH_FAIL_WINDOW_S if window is None else window
        self.lock = threading.Lock()
        self.fails = {}

    def _recent(self, who, now):
        q = self.fails.get(who)
        if not q:
            return None
        while q and now - q[0] > self.window:
            q.popleft()
        if not q:
            self.fails.pop(who, None)
            return None
        return q

    def blocked(self, who, now=None):
        now = time.time() if now is None else now
        with self.lock:
            q = self._recent(who, now)
            return bool(q) and len(q) >= self.limit

    def fail(self, who, now=None):
        now = time.time() if now is None else now
        with self.lock:
            if len(self.fails) > 4096:                 # a scan from many addresses must not grow this for ever
                self.fails.clear()
            q = self._recent(who, now)
            if q is None:
                q = self.fails[who] = deque()
            q.append(now)
            return len(q)


class Tickets(object):
    """Single-use, path-bound, short-lived stand-ins for the token, for the navigations that cannot carry the
    X-PXA-Token header: a plugin's sealed frame, a <script src>, a download link, a page opened in a new tab.
    The page mints one through the header (POST /api/auth/ticket {path}) and puts ?ticket=<t> on that GET.
    A ticket opens exactly one GET of exactly that path and query (the same name=value pairs, any order; ?ticket=
    itself aside) within TICKET_TTL_S; any try, right or wrong, spends it. A plugin frame's ticket opens that plugin."""

    def __init__(self, ttl=TICKET_TTL_S, cap=TICKET_MAX):
        self.ttl, self.cap = float(ttl), int(cap)
        self._lock = threading.Lock()
        self._t = {}                    # ticket -> (path, monotonic deadline), oldest first

    @staticmethod
    def path_ok(path):
        """a ticket may name one API path: absolute, no query or fragment, nothing that could walk elsewhere."""
        return (isinstance(path, str) and 0 < len(path) <= 512 and (path.startswith("/api/") or path == "/metrics")
                and not any(c in path for c in "?#\\%") and "//" not in path and "/../" not in path + "/"
                and "/./" not in path + "/" and all(32 < ord(c) < 127 for c in path))

    @staticmethod
    def query_ok(query):
        return isinstance(query, str) and len(query) <= 1024 and "#" not in query and all(32 < ord(c) < 127 for c in query)

    @staticmethod
    def _pairs(query):
        """a query string or parse_qs dict -> its sorted (name, value) pairs, ?ticket= left out."""
        if isinstance(query, dict):
            items = [(k, v) for k, vs in query.items() for v in vs]
        else:
            items = urllib.parse.parse_qsl(query or "", keep_blank_values=True)
        return tuple(sorted((k, v) for k, v in items if k != "ticket"))

    def mint(self, path, query=""):
        t = secrets.token_urlsafe(24)
        now = time.monotonic()
        with self._lock:
            for k in [k for k, (_p, end, _q) in self._t.items() if end <= now]:
                del self._t[k]
            while len(self._t) >= self.cap:
                self._t.pop(next(iter(self._t)))
            self._t[t] = (path, now + self.ttl, self._pairs(query))
        return t

    def redeem(self, t, path, query=None):
        if not isinstance(t, str) or not 0 < len(t) <= 64:
            return False
        with self._lock:
            ent = self._t.pop(t, None)
        return (ent is not None and hmac.compare_digest(ent[0].encode(), str(path).encode()) and time.monotonic() < ent[1]
                and ent[2] == self._pairs(query or {}))

    def age(self, t, seconds):
        """tests: make one ticket `seconds` older."""
        with self._lock:
            if t in self._t:
                p, end, q = self._t[t]
                self._t[t] = (p, end - seconds, q)


def fetch_site_ok(headers):
    """H2: a browser says where a request comes from. Only this page (same-origin) and the user's own
    navigation (none: typed, bookmarked, opened by the launcher) may use the API; a page on another site, or on
    another port of this machine (same-site: cookies ride along), may not."""
    v = (headers.get("Sec-Fetch-Site") or "").strip().lower()
    return v in ("", "same-origin", "none")


def origin_matches_host(headers):
    """H2: no Origin (curl, scripts, older browsers on a same-origin GET) or exactly http(s)://<Host>.
    'null' (sandboxed frames, file: pages, privacy redirects) is refused."""
    o = headers.get("Origin")
    if o is None:
        return True
    o = o.strip()
    try:
        u = urllib.parse.urlparse(o)
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not u.netloc or u.path not in ("", "/") or u.query or u.params:
        return False
    return u.netloc.lower() == (headers.get("Host") or "").strip().lower()


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]   # star-import carries the _private names too
