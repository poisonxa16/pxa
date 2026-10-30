"""PXA Control: the FREE shareable bench card (todo control-share-report).

One click turns a benchmark from this rig's history into an image: the rig's cards, the model, the measured speed, a
"made with PXA" mark, the Discord badge and the Store link along the bottom. It is the funnel, never paid: nothing here
asks the Store, an entitlement, a donation tier or the network. It only READS the bench history (bench.jsonl, written by
the Benchmarks tab) and the card names; it lives on its own page (/share.html) so the Benchmarks tab stays untouched.

Public text rules: what + how much, never how. The card names the cards, the model and the speeds;
it never carries flags, levers, split mode, paths, ports, hashes or the command line, and never a rival engine's name.

Routes (token-guarded like every /api route):
    GET /api/share/runs              the bench runs a card can be made from, newest first
    GET /api/share/card?id=&name=1   one card as JSON (id = the run's ts; default the newest; name=1 adds the board name)
    GET /api/share/card.svg?id=...   the same card as a 1200x630 SVG (the page turns it into a PNG)
"""
import datetime
import os
import re

from .core import Invalid, Reply, QUANT_RE, clean_board_name, load_config, read_jsonl

FREE = True                                        # the rule, not a setting: the card is never gated
W, H = 1200, 630                                   # the og:image size Discord and X preview without cropping
SITE_URL = "https://pxanetwork.com"
DISCORD_URL = "https://discord.pxanetwork.com"
BOARD_URL = "https://benchmarks.pxanetwork.com"
STORE_URL = SITE_URL                               # the Store has no public page of its own yet: the site carries it
MAX_RUNS = 200

# never on a public card (the public text rules): other engines are not named, by name or by binary
RIVAL_RE = re.compile(r"(?i)\b(?:ik[_-]?)?llama[._ -]?cpp\b|\bllama[-_]server\b|\bollama\b|\bvllm\b|\bexllama\w*|"
                      r"\blm[ -]?studio\b|\bkobold\w*|\bsglang\b|\btabby[ -]?api\b|\btext-generation-webui\b|"
                      r"\btensorrt[-_ ]?llm\b|\bmlc[-_ ]?llm\b|\blocalai\b|\bjan\.ai\b|\bgpt4all\b|\baphrodite\b")
_VENDOR_RE = re.compile(r"(?i)^\s*(nvidia|amd|intel)\s+")


def scrub(s, limit=80):
    """one line of public text: no control characters, no rival engine names, bounded."""
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(s or ""))
    s = RIVAL_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -_.")
    return s[:limit]


def _model_label(model):
    """the file's name without its folder, extension, split suffix or quant: the WHAT, never where it lives."""
    base = os.path.basename(str(model or "").replace("\\", "/"))
    base = re.sub(r"(?i)\.gguf$", "", base)
    base = re.sub(r"(?i)-\d{5}-of-\d{5}$", "", base)
    qm = QUANT_RE.search(base)
    quant = qm.group(1).upper() if qm else ""
    name = base[:qm.start()] + base[qm.end():] if qm else base
    name = re.sub(r"[-_.]{2,}", "-", name).strip("-_. ")
    return scrub(name, 64) or "a local model", quant


def _gpu_name(n):
    return scrub(_VENDOR_RE.sub("", str(n or "")).replace("GeForce ", ""), 40) or "GPU"


def rig_label(names):
    """['GTX 1080 Ti', 'GTX 1080 Ti', 'RTX 3090'] -> '2x GTX 1080 Ti + RTX 3090' (order of first appearance)."""
    seen = []
    for n in names:
        n = _gpu_name(n)
        for row in seen:
            if row[0] == n:
                row[1] += 1
                break
        else:
            seen.append([n, 1])
    if not seen:
        return "this rig"
    return " + ".join(("%dx %s" % (c, n)) if c > 1 else n for n, c in seen)


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v and 0 < v < 1e7 else 0.0


def _ts(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v and 0 <= v < 1e11 else 0.0


def _speeds(rec):
    by = {c.get("class"): c for c in (rec.get("classes") or []) if isinstance(c, dict)}
    dec = _num((by.get("prose") or {}).get("decode_tps"))
    if not dec:
        dec = max([_num(c.get("decode_tps")) for c in by.values()] or [0.0])
    pre = _num((by.get("long") or {}).get("prefill_tps"))
    if not pre:
        pre = max([_num(c.get("prefill_tps")) for c in by.values()] or [0.0])
    return round(dec, 1), round(pre, 0)


def _runs():
    return [r for r in read_jsonl("bench.jsonl", limit=MAX_RUNS) if isinstance(r, dict) and _speeds(r)[0] > 0]


def _gpu_rows(app):
    try:
        rows, _err = app.gpus()
    except Exception:            # noqa: BLE001 - a card list we cannot read is "this rig", never a 500
        rows = []
    return [r for r in (rows or []) if isinstance(r, (list, tuple)) and len(r) >= 2]


def card(app, rec, with_name=False):
    """bench record -> the public card. Only fields that pass the public text rules leave this function."""
    rows = _gpu_rows(app)
    want = [c for c in (rec.get("cards") or []) if isinstance(c, int)]
    sel = [g for g in rows if g[0] in want] if want else rows
    if want and not sel:
        sel = rows[:len(want)]
    names = [g[1] for g in sel]
    vram = 0.0
    for g in sel:
        try:
            vram += float(g[3]) / 1024.0 if len(g) > 3 else 0.0
        except (TypeError, ValueError):
            pass
    model, quant = _model_label(rec.get("model"))
    dec, pre = _speeds(rec)
    rig = rig_label(names)
    try:
        when = datetime.datetime.fromtimestamp(float(rec.get("ts") or 0)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError, OSError):
        when = ""
    name = clean_board_name(load_config().get("user_name", "")) if with_name else ""
    what = model + ((" " + quant) if quant else "")
    headline = "%s runs %s at %s tokens/s" % (rig, what, _fmt(dec))
    lines = [headline]
    if pre:
        lines.append("Prompt reading: %s tokens/s" % _fmt(pre))
    lines.append("Measured on my own rig with PXA (free) - %s" % SITE_URL)
    lines.append("Discord: %s" % DISCORD_URL)
    return {"free": FREE, "id": rec.get("ts"), "when": when, "rig": rig, "gpu_count": len(names), "vram_gb": round(vram, 1),
            "model": model, "quant": quant, "decode_tps": dec, "prefill_tps": pre or None, "name": name,
            "headline": headline, "text": "\n".join(lines), "mark": "Made with PXA",
            "links": {"site": SITE_URL, "discord": DISCORD_URL, "board": BOARD_URL, "store": STORE_URL}}


def _fmt(v):
    return ("%.0f" % v) if v >= 100 else ("%.1f" % v)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def svg(c):
    """the card as a self-contained SVG (no script, no external resource): the PNG the page saves is drawn from it."""
    t = lambda x, y, s, size, fill="#ECEDEF", weight=400, anchor="start", extra="": (   # noqa: E731
        '<text x="%d" y="%d" font-size="%d" font-weight="%d" fill="%s" text-anchor="%s"%s>%s</text>'
        % (x, y, size, weight, fill, anchor, extra, _esc(s)))
    rig = c["rig"]
    rsize = 64 if len(rig) <= 22 else 52 if len(rig) <= 30 else 40
    model = c["model"] + ((" \u00b7 " + c["quant"]) if c["quant"] else "")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
        'font-family="Inter, Segoe UI, system-ui, -apple-system, Helvetica, Arial, sans-serif">' % (W, H, W, H),
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#16181C"/>'
        '<stop offset="1" stop-color="#0E0F11"/></linearGradient></defs>',
        '<rect width="%d" height="%d" rx="28" fill="url(#bg)"/>' % (W, H),
        '<rect x="0" y="0" width="%d" height="10" fill="#E08A3C"/>' % W,
        # the PXA mark: a vector wordmark (no image file, so the SVG stays self-contained)
        '<rect x="60" y="54" width="92" height="44" rx="10" fill="#E08A3C"/>',
        t(106, 86, "PXA", 28, "#111214", 800, "middle"),
        t(168, 86, "Made with PXA", 26, "#A0A4AC", 600),
        t(W - 60, 86, c["when"], 24, "#A0A4AC", 400, "end"),
        t(60, 196, rig, rsize, "#ECEDEF", 800),
        t(60, 252, model, 34, "#C9CCD2", 500),
        t(60, 420, _fmt(c["decode_tps"]), 150, "#E08A3C", 800),
        t(60, 470, "tokens / second, writing", 30, "#A0A4AC", 500),
        t(60, 510, "Measured on my own rig \u00b7 PXA is free", 20, "#7D818A", 500),
    ]
    if c.get("prefill_tps"):
        parts.append(t(W - 60, 420, _fmt(c["prefill_tps"]), 72, "#ECEDEF", 700, "end"))
        parts.append(t(W - 60, 470, "tokens / second, reading the prompt", 26, "#A0A4AC", 500, "end"))
    if c.get("name"):
        parts.append(t(W - 60, 196, "by " + c["name"], 30, "#C9CCD2", 600, "end"))
    parts += [
        '<rect x="0" y="530" width="%d" height="100" fill="#1D1F23"/>' % W,
        # the Discord badge
        '<rect x="60" y="555" width="330" height="50" rx="25" fill="#5865F2"/>',
        t(225, 588, "discord.pxanetwork.com", 22, "#FFFFFF", 700, "middle"),
        # the Store link at the bottom
        t(W - 60, 588, "PXA + plugins in the Store: pxanetwork.com", 22, "#E08A3C", 700, "end"),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def _pick(query):
    runs = _runs()
    if not runs:
        raise Invalid("run a benchmark first: the card is made from a run in this rig's bench history", 404, "no_runs")
    rid = (query.get("id") or [None])[0]
    if rid in (None, "", "latest"):
        return runs[-1]
    try:
        want = float(rid)
    except (TypeError, ValueError):
        raise Invalid("id must be a run's timestamp", 400, "bad_id")
    for r in runs:
        if abs(_ts(r.get("ts")) - want) < 1e-3:
            return r
    raise Invalid("no bench run with that id", 404, "not_found")


def _with_name(query):
    return str((query.get("name") or [""])[0]).lower() in ("1", "true", "yes", "on")


def register(ROUTES, STATIC):
    """the share routes and the share page (its own page: Benchmarks/Models/Live and chat are not touched)."""

    def r_share_runs(app, body, query):
        out = []
        for r in reversed(_runs()):
            model, quant = _model_label(r.get("model"))
            dec, _pre = _speeds(r)
            out.append({"id": r.get("ts"), "model": model, "quant": quant, "decode_tps": dec,
                        "when": datetime.datetime.fromtimestamp(_ts(r.get("ts"))).strftime("%Y-%m-%d %H:%M")})
        return {"free": FREE, "runs": out}

    def r_share_card(app, body, query):
        return card(app, _pick(query), _with_name(query))

    def r_share_card_svg(app, body, query):
        c = card(app, _pick(query), _with_name(query))
        return Reply(svg(c).encode("utf-8"), "image/svg+xml; charset=utf-8",
                     extra={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:"})

    ROUTES[("GET", "/api/share/runs")] = r_share_runs
    ROUTES[("GET", "/api/share/card")] = r_share_card
    ROUTES[("GET", "/api/share/card.svg")] = r_share_card_svg
    STATIC["/share.html"] = ("share.html", "text/html; charset=utf-8")
    STATIC["/share.js"] = ("share.js", "application/javascript; charset=utf-8")
    STATIC["/share.css"] = ("share.css", "text/css; charset=utf-8")
