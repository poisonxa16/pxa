"""Server picker: the running PXA servers Control knows, with friendly names, resolved context, capabilities,
and the auto-select rule (last used if it is up, otherwise the healthiest).

The hive seat rule (PXACLAW hive/lib/seat.mjs): a server is a URL and nothing else; the model id is whatever
the URL reports at /v1/models, and if it reports none the call fails loudly instead of inventing a name.
"""
import os
import threading
import time

from . import oai

PROBE_TTL = 8.0
_cache = {}
_cache_lock = threading.Lock()


def stable_key(x):
    """the key a chat remembers: survives a restart (managed sid / docker name / port), not a pid."""
    k = x.get("key") or ""
    if k.startswith("m:") or k.startswith("d:"):
        return k
    return f"port:{x.get('port')}" if x.get("port") else k


def _probe(port, get=None):
    get = get or oai.get_json
    now = time.time()
    with _cache_lock:
        hit = _cache.get(port)
        if hit and now - hit[0] < PROBE_TTL:
            return hit[1]
    base = f"http://127.0.0.1:{port}/v1"
    info = {"model": None, "n_ctx": None, "caps": {}, "template": "", "error": None, "slots_busy": None}
    try:
        m = get(base, "/v1/models", 2.0)
        data = (m or {}).get("data") or (m or {}).get("models") or []
        if data:
            info["model"] = data[0].get("id") or data[0].get("name") or data[0].get("model")
    except oai.OAIError as e:
        info["error"] = str(e)
    try:
        p = get(base, "/props", 2.0) or {}
        dg = p.get("default_generation_settings") or {}
        info["n_ctx"] = dg.get("n_ctx") or p.get("n_ctx")
        info["caps"] = p.get("chat_template_caps") or {}
        info["template"] = (p.get("chat_template") or "")[:200000]
        info["model_path"] = p.get("model_path")
    except oai.OAIError:
        pass
    with _cache_lock:
        _cache[port] = (now, info)
    return info


def capabilities(info):
    caps, tpl = info.get("caps") or {}, info.get("template") or ""
    if caps:
        tools = bool(caps.get("supports_tools") or caps.get("supports_tool_calls"))
        src = "chat_template_caps"
    else:
        tools = ("tool_call" in tpl or "tools" in tpl and "{%" in tpl)
        src = "template" if tpl else "unknown"
    thinking = bool(caps.get("supports_thinking") or caps.get("supports_reasoning")
                    or "<think>" in tpl or "enable_thinking" in tpl or "reasoning_content" in tpl)
    return {"tools": tools, "thinking": thinking, "source": src,
            "tools_words": "Can use tools directly" if tools else "Uses tools through text (works, a bit slower)"}


def _int(v):
    try:
        v = int(str(v).strip())
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def ctx_info(configured, resolved):
    auto = not configured
    if resolved and auto:
        label = f"Auto (picked {resolved:,} tokens to fit your card)"
    elif resolved:
        label = f"{resolved:,} tokens"
    else:
        label = "Auto" if auto else f"{configured:,} tokens"
    return {"configured": configured, "resolved": resolved, "auto": auto, "label": label}


def _short(model):
    if not model:
        return None
    m = os.path.basename(str(model))
    for ext in (".gguf", ".pxq", ".pxqn"):
        if m.lower().endswith(ext):
            m = m[:-len(ext)]
    return m


def quant_notice_for(app, names):
    """The standard-GGUF sentence for these names, or empty.

    One PXA quant among the names means the serving file is ours, so the page stays quiet.
    Otherwise the first name the library already calls a standard quant supplies the line.
    The configured path comes first (an inspected file in the model list), then the path and
    the id the running server reports.
    """
    names = [n for n in names if isinstance(n, str) and n.strip()]
    launch = getattr(app, "L", None)
    if launch is not None and any(launch.is_pxa_quant(n) for n in names):
        return ""
    fn = getattr(app, "_quant_notice", None)
    if fn is None and launch is not None:
        fn = launch.standard_gguf_notice
    if fn is None:
        return ""
    for n in names:
        hit = fn(n) or ""
        if hit:
            return hit
    return ""


def list_servers(app, get=None):
    fl = app.fleet()
    profiles = {}
    try:
        profiles = app.profiles()
    except Exception:            # noqa: BLE001
        pass
    out = []
    for x in fl.get("instances") or []:
        up = bool(x.get("running") or x.get("attached"))
        port = x.get("port")
        if not port or (not up and x.get("kind") != "managed"):
            continue
        health = x.get("health")
        info = {}
        if up:
            if health is None and x.get("attached"):
                from_health = _probe(port, get)
                health = "ok" if from_health.get("model") else "down"
            info = _probe(port, get) if health == "ok" else {}
        cfg = None
        if x.get("kind") == "managed":
            cfg = _int(((profiles.get(x.get("sid")) or {}).get("settings") or {}).get("ctx"))
        else:
            cfg = _int(x.get("ctx"))
        model = info.get("model")
        if not up:
            status = "down"
        elif health == "ok":
            status = "busy" if (x.get("slots_idle") == 0 and (x.get("slots_processing") or 0) > 0) else "ready"
        elif health == "loading" or x.get("phase") in ("starting", "loading"):
            status = "starting"
        else:
            status = "down"
        if status in ("ready", "busy") and not model:
            status, words = "error", "This server did not say which model it runs, so chat cannot use it"
        else:
            words = {"ready": "Ready", "busy": "Busy with another request", "starting": "Starting up...",
                     "down": "Not running"}[status]
        gpus = x.get("gpus") or []
        cards = "GPU " + "+".join(str(g) for g in gpus) if gpus else "CPU / unknown cards"
        name = _short(model) or _short(x.get("model")) or x.get("label") or x.get("name") or f"port {port}"
        out.append({"key": stable_key(x), "fleet_key": x.get("key"), "kind": x.get("kind"), "sid": x.get("sid"),
                    "name": name, "server_name": x.get("label") or x.get("name"), "model": model,
                    "quant_notice": quant_notice_for(app, [x.get("model"), info.get("model_path"), model]),
                    "cards": gpus, "cards_label": cards, "port": port,
                    "base_url": f"http://127.0.0.1:{port}/v1", "status": status, "status_words": words,
                    "ctx": ctx_info(cfg, _int(info.get("n_ctx"))), "caps": capabilities(info) if info else
                    {"tools": False, "thinking": False, "source": "unknown", "tools_words": "-"},
                    "slots_idle": x.get("slots_idle"), "slots_processing": x.get("slots_processing")})
    return out


RANK = {"ready": 0, "busy": 1, "starting": 2, "error": 3, "down": 4}


def choose(servers, last_key=None):
    """-> (key or None, reason). Last used when it is up; else the healthiest (ready > busy, then tool
    support, then the larger context, then the lower port)."""
    by = {s["key"]: s for s in servers}
    last = by.get(last_key) if last_key else None
    if last and last["status"] in ("ready", "busy"):
        return last["key"], "last_used"
    up = [s for s in servers if s["status"] in ("ready", "busy")]
    if up:
        up.sort(key=lambda s: (RANK[s["status"]], not s["caps"].get("tools"), -((s["ctx"] or {}).get("resolved") or 0),
                               s["port"] or 0))
        return up[0]["key"], "healthiest"
    if last:
        return last["key"], "last_used_waiting"      # keep the user's choice: it re-attaches when it comes back
    return None, "none_running"


def resolve(app, key, get=None):
    """the current entry for a remembered key (a restarted server keeps its key), or None."""
    for s in list_servers(app, get):
        if s["key"] == key:
            return s
    return None


def forget_cache():
    with _cache_lock:
        _cache.clear()
