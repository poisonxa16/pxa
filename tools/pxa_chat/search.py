"""Web search providers for the chat agent. Stdlib only.

Built-in (default, no keys, no setup): DuckDuckGo's HTML page, then Bing's, parsed here. Optional: your own SearXNG, a Brave or
Tavily key. Page text: a plain fetch with HTML stripped, or your own crawl4ai / reader service. Queries go only to the
provider the user picked; a failing provider falls back to the built-in one and says so.
"""
import json
import os
import re
import time
import html as _html
import urllib.error
import urllib.parse
import urllib.request

PROVIDERS = ("builtin", "searxng", "brave", "tavily")
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0 PXA-Control"
DEFAULTS = {"provider": "builtin", "searxng_url": "", "brave_key": "", "tavily_key": "", "reader_url": "", "reader_token": ""}
SECRETS = ("brave_key", "tavily_key", "reader_token")


def config(saved):
    """saved settings dict (or None) -> a full, validated settings dict. PXA_CHAT_SEARCH_URL still works as a default."""
    c = dict(DEFAULTS)
    env = os.environ.get("PXA_CHAT_SEARCH_URL", "")
    if re.match(r"^https?://", env):
        c.update(provider="searxng", searxng_url=env)
    for k in DEFAULTS:
        v = (saved or {}).get(k)
        if isinstance(v, str):
            c[k] = v.strip()
    if c["provider"] not in PROVIDERS:
        c["provider"] = "builtin"
    for k in ("searxng_url", "reader_url"):
        if c[k] and not re.match(r"^https?://", c[k]):
            c[k] = ""
    return c


def public(c):
    """settings for the browser: secrets become booleans."""
    d = {k: v for k, v in c.items() if k not in SECRETS}
    d.update({k + "_set": bool(c.get(k)) for k in SECRETS})
    return d


def _get(url, headers=None, data=None, timeout=15, limit=4 << 20):
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(limit).decode("utf-8", "replace")


def _explain(e):
    if isinstance(e, urllib.error.HTTPError):
        return "HTTP %s" % e.code + (" (bad or missing key)" if e.code in (401, 403) else "")
    return str(getattr(e, "reason", None) or e)


def _clean(s):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"(?s)<[^>]+>", "", s or ""))).strip()


def ddg(q, n):
    page = _get("https://html.duckduckgo.com/html/", {"Content-Type": "application/x-www-form-urlencoded"},
                urllib.parse.urlencode({"q": q}).encode())
    out = []
    for m in re.finditer(r'(?s)<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]*class="result__a"|\Z)', page):
        href = _html.unescape(m.group(1))
        if href.startswith("//"):
            href = "https:" + href
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        url = qs["uddg"][0] if "uddg" in qs else href
        if not re.match(r"^https?://", url) or "duckduckgo.com/y.js" in url:
            continue
        sn = re.search(r'(?s)class="result__snippet"[^>]*>(.*?)</a>', m.group(3))
        out.append({"title": _clean(m.group(2)), "url": url, "snippet": _clean(sn.group(1)) if sn else ""})
        if len(out) >= n:
            break
    if not out:
        raise ValueError("DuckDuckGo returned no results (it may be rate-limiting this address)")
    return out


def bing(q, n):
    import base64
    page = _get("https://www.bing.com/search?" + urllib.parse.urlencode({"q": q, "setlang": "en-US", "cc": "US", "mkt": "en-US"}))
    out = []
    for blk in re.split(r'<li class="b_algo"', page)[1:]:
        m = re.search(r'(?s)<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', blk)
        if not m:
            continue
        url = _html.unescape(m.group(1))
        u = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("u")
        if u and u[0].startswith("a1"):                 # Bing wraps results in a redirect; the target is base64 after "a1"
            b = u[0][2:]
            try:
                url = base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)).decode("utf-8", "replace")
            except ValueError:
                continue
        if not re.match(r"^https?://", url):
            continue
        sn = re.search(r'(?s)<p[^>]*class="b_lineclamp[^>]*>(.*?)</p>', blk)
        out.append({"title": _clean(m.group(2)), "url": url, "snippet": _clean(sn.group(1)) if sn else ""})
        if len(out) >= n:
            break
    if not out:
        raise ValueError("Bing returned no results")
    return out


def builtin(q, n):
    """no keys, no setup: DuckDuckGo's HTML page, then Bing's if DuckDuckGo refuses (it sometimes shows a bot check)."""
    errs = []
    for fn in (ddg, bing):
        try:
            return fn(q, n)
        except Exception as e:  # noqa: BLE001
            errs.append(_explain(e))
    raise ValueError("; ".join(errs))


def searxng(q, n, base):
    j = json.loads(_get(base.rstrip("/") + "/search?" + urllib.parse.urlencode({"q": q, "format": "json"})))
    out = [{"title": x.get("title"), "url": x.get("url"), "snippet": str(x.get("content") or "")}
           for x in (j.get("results") or [])[:n]]
    if not out:
        raise ValueError("SearXNG returned no results")
    return out


def brave(q, n, key):
    j = json.loads(_get("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": n}),
                        {"X-Subscription-Token": key, "Accept": "application/json"}))
    out = [{"title": x.get("title"), "url": x.get("url"), "snippet": _clean(x.get("description"))}
           for x in ((j.get("web") or {}).get("results") or [])[:n]]
    if not out:
        raise ValueError("Brave returned no results")
    return out


def tavily(q, n, key):
    j = json.loads(_get("https://api.tavily.com/search", {"Content-Type": "application/json", "Authorization": "Bearer " + key},
                        json.dumps({"query": q, "max_results": n}).encode()))
    out = [{"title": x.get("title"), "url": x.get("url"), "snippet": str(x.get("content") or "")}
           for x in (j.get("results") or [])[:n]]
    if not out:
        raise ValueError("Tavily returned no results")
    return out


def search(q, n, cfg):
    """-> (results, note). note is '' or a plain sentence about a fallback. Raises ValueError when even the built-in fails."""
    p = cfg.get("provider")
    note = ""
    try:
        if p == "searxng" and cfg.get("searxng_url"):
            return searxng(q, n, cfg["searxng_url"]), ""
        if p == "brave" and cfg.get("brave_key"):
            return brave(q, n, cfg["brave_key"]), ""
        if p == "tavily" and cfg.get("tavily_key"):
            return tavily(q, n, cfg["tavily_key"]), ""
        if p != "builtin":
            note = "The %s search is not fully set up (missing URL or key); used the built-in search instead." % p
    except Exception as e:  # noqa: BLE001
        note = "The %s search failed (%s); used the built-in search instead." % (p, _explain(e))
    try:
        return builtin(q, n), note
    except Exception as e:  # noqa: BLE001
        raise ValueError((note + " " if note else "") + "The built-in search failed: " + _explain(e))


def reader_markdown(url, cfg):
    """readable markdown from the user's crawl4ai / reader service, or None (the caller then does a plain fetch)."""
    base = cfg.get("reader_url")
    if not base:
        return None
    try:
        h = {"Content-Type": "application/json"}
        if cfg.get("reader_token"):
            h["Authorization"] = "Bearer " + cfg["reader_token"]
        j = json.loads(_get(base.rstrip("/") + "/crawl", h, json.dumps({"urls": [url]}).encode(), 40, 8 << 20))
        res = (j.get("results") or [j])[0] or {}
        md = res.get("markdown")
        if isinstance(md, dict):
            md = md.get("fit_markdown") or md.get("raw_markdown") or ""
        md = (md or "").strip()
        return md if len(md) > 80 else None
    except Exception:  # noqa: BLE001
        return None


_found = {"t": 0, "v": None}


def detect(force=False):
    """local services worth offering (never switched on silently). Cached for a minute."""
    if not force and _found["v"] is not None and time.time() - _found["t"] < 60:
        return _found["v"]
    f = {}
    for port in (8080, 8888):
        base = "http://127.0.0.1:%d" % port
        try:
            j = json.loads(_get(base + "/search?q=test&format=json", timeout=1.5, limit=1 << 20))
            if isinstance(j, dict) and "results" in j:
                f["searxng_url"] = base
                break
        except Exception:  # noqa: BLE001
            pass
    for port in (11235, 8082):
        base = "http://127.0.0.1:%d" % port
        try:
            _get(base + "/health", timeout=1.5, limit=1 << 16)
            f["reader_url"] = base
            break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):          # crawl4ai with auth on
                f["reader_url"] = base
                break
        except Exception:  # noqa: BLE001
            pass
    _found.update(t=time.time(), v=f)
    return f
