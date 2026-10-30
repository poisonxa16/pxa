"""What the Plugins tab draws and serves (PLUGINS-PROTOTYPE-PLAN S3; PLUGINS-STORE-DESIGN s3, s5).

* GET  /api/plugins/host/cards   every installed plugin as one card: name, version, who signed it (and whether the signature and
  every file still check out), what the person allowed, the consent card's lines in plain words (every permission, each port it
  listens on, its overlay), and where it stands with the one gate (gate.panel: active / missing / revoked ...), with the coins
  that unlock it and the wallet's balance when the signed catalogue and statement say so.
* GET  /api/plugins/host/frame?plugin_id=<id>   the plugin's own page (manifest ui.tab.page, a hash-verified file of the signed
  package) as an HTML document that runs in an OPAQUE origin: the answer's CSP says `sandbox allow-scripts` (no allow-same-origin),
  `connect-src 'none'`, nothing loads from anywhere. The Plugins tab shows it in <iframe sandbox="allow-scripts">; the frame has no
  cookie, no token and no way to call Control. Its only way out is the postMessage bridge below.
* The bridge (plugin-host.js on the Control page side, `pxa` object injected here on the frame side): the page asks
  `pxa.fetch("/api/plugins/<its id>/...", {method, body, headers})`; the Control page checks the message came from THAT frame,
  that the path is under /api/plugins/<that plugin's id>/ (never another plugin, never /api/plugins/host or any other Control
  route), and makes the call with its own cookie; the answer comes back as {status, ctype, text | b64}. `pxa.theme` / `pxa.onTheme`
  carry the page's colours; `pxa.resize(px)` asks for a height. Nothing else crosses.
v1 limit: the page is ONE html file (inline css and js); its other files are not served to the frame.
"""
import os

from . import PluginError, PERMISSIONS, NET_SCOPED, NAV_GLYPHS, consent_lines, scoped_host
from . import gate, pxp

MAX_PAGE = 2 << 20
FRAME_CSP = ("sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
             "img-src data: blob:; font-src data:; media-src data: blob:; connect-src 'none'; form-action 'none'; "
             "base-uri 'none'; frame-ancestors 'self'")

# permission -> (group, icon, reach). reach is what the card's tag says, from what the permission REALLY lets the plugin do:
#   local  "stays on this computer"   works inside this computer's PXA (speech runs here, nothing leaves it), no tag of concern
#   sense  "uses a sensor"            hears or sees what is around you or on this computer (mic, what it plays, camera)
#   out    "reaches beyond PXA"       other programs and windows, your files, the network (home network, internet), Home Assistant
GROUPS = ("Hears and speaks", "Your devices", "Network", "Files and programs", "PXA")
REACH_WORDS = {"local": "stays on this computer", "sense": "uses a sensor", "out": "reaches beyond PXA"}
PERM_UI = {
    "speech.listen": ("Hears and speaks", "mic", "local"),
    "speech.speak": ("Hears and speaks", "bell", "local"),
    "device.mic": ("Your devices", "mic", "sense"),
    "device.audio-capture": ("Your devices", "mic", "sense"),
    "device.camera": ("Your devices", "live", "sense"),
    "device.speaker": ("Your devices", "bell", "local"),
    "input.type": ("Your devices", "encode", "out"),            # types into another program's window
    "net.loopback": ("Network", "globe", "local"),
    "net.lan": ("Network", "globe", "out"),
    "net.internet": ("Network", "globe", "out"),
    "net.http": ("Network", "globe", "out"),
    "home.assistant": ("Network", "plug", "out"),
    "fs.config": ("Files and programs", "folder", "local"),
    "fs.read": ("Files and programs", "folder", "out"),
    "fs.write": ("Files and programs", "folder", "out"),
    "fs.workspace.read": ("Files and programs", "folder", "local"),
    "fs.workspace.write": ("Files and programs", "folder", "local"),
    "proc.spawn": ("Files and programs", "play", "out"),
    "host.shell": ("Files and programs", "alert", "out"),
    "model.chat": ("PXA", "chat", "local"),
    "plugin.tools": ("PXA", "chat", "local"),
    "gpu.read": ("PXA", "gpu", "local"),
    "engine.control": ("PXA", "servers", "out"),                 # starts and stops servers (and what they open)
}
# a signature that did not check out, in words (the error codes of load_manifest)
PROBLEMS = {
    "bad_signature": "The signature does not match, so this plugin will not run.",
    "unknown_key": "It is signed with a key this PXA Control does not know, so it will not run.",
    "bad_file": "A file changed after it was signed, so this plugin will not run. Install it again from the Store.",
    "invalid_manifest": "Its description is not valid, so this plugin will not run.",
}


def lines(m, rec=None):
    """the consent card's rows for a manifest: [{key, words, group, icon, reach, tag, care, ask}]; ask = a tick box (a permission),
    otherwise an "also" line (a port it listens on, its overlay) that follows from the permissions."""
    out = []
    old = set((rec or {}).get("asked") or [])
    for key, words in consent_lines(m):
        h = scoped_host(key)
        if h is not None:
            g, ic, rc = "Network", "globe", "out"
        elif key in PERM_UI:
            g, ic, rc = PERM_UI[key]
        elif key.startswith("listen:"):
            g, ic, rc = "Network", "globe", ("out" if " your home network " in " %s " % words else "local")
        elif key == "overlay":
            g, ic, rc = "Your devices", "live", "local"
        else:
            g, ic, rc = "PXA", "shield", "local"
        out.append({"key": key, "words": words, "group": g, "icon": ic, "reach": rc, "tag": REACH_WORDS[rc],
                    "care": rc != "local",
                    "ask": key in PERMISSIONS or h is not None,
                    "new": bool(rec) and (key in PERMISSIONS or h is not None) and key not in old})
    return out


def _catalog(cfg_dir):
    """the signed catalogue as the Store tab last saw it (cached file), else the fixture. Never a network call."""
    import json
    from .. import store as S
    try:
        with open(os.path.join(cfg_dir, "store", "catalog.json")) as f:
            v = json.load(f)
        if isinstance(v, dict) and v.get("items"):
            return v
    except (OSError, ValueError):
        pass
    return S.catalog_view()


def _entitlement(cfg_dir, cache, m, catalog):
    ent = m.get("entitlement")
    if m["edition"] != "supporter" or not ent:
        return None
    p = gate.panel(cfg_dir, ent, "plugin", cache, name=m["name"])
    if p["state"] == "missing":
        ri = gate._cache(cfg_dir, cache).record().get("revoked_items")
        gone = isinstance(ri, dict) and ent in ri           # {item_id: ent_id}, the shape Core / the plugin host share
        if gone:                      # a signed revocation took its grant (store.apply_revocations keeps the item): say so, not "locked"
            p = dict(p, state="revoked", active=False, how=None, message=gate.MESSAGES["revoked"].format(name=m["name"]))
    cost = gate.cost(gate.catalog_row(catalog, ent))
    st = p["state"]
    have = p.get("have")
    if p["active"] and p["how"] == "grant":
        action = "none"
    elif st == "missing":
        action = "unlock" if cost else "store"
    else:
        action = "sync"
    p = dict(p, cost=cost, action=action, enough=(have is not None and cost is not None and have >= cost))
    return p


def _store_row(catalog, m, off):
    """what the Store's signed catalogue (as last seen) says about an installed plugin: its item, the newest listed version,
    whether that is an Update for this one, and whether the user turned it off (Deactivate). Catalogue data only."""
    item = m.get("entitlement") or m["plugin_id"]
    row = gate.catalog_row(catalog, item) or {}
    latest = ((row.get("bundle") or {}).get("version")) if isinstance(row.get("bundle"), dict) else None
    return {"item_id": item, "latest": latest, "update": bool(latest and pxp.ver_key(latest) > pxp.ver_key(m["version"])),
            "off": bool(off)}


def refresh_statement(cfg_dir):
    """one rate-limited read of the wallet's signed statement (store.keep_statement: at most one per STMT_GAP_S, best effort),
    so a card says the coins the wallet holds NOW, not the balance of the last time the Store tab was open."""
    from .. import store as S
    try:
        url = S.store_url()
        return bool(url) and S.keep_statement(S.LicdClient(url, cfg_dir, None, timeout=5), cfg_dir)
    except Exception:  # noqa: BLE001 - the card shows the last signed balance
        return False


def _speech_on_disk(cfg_dir):
    from ..speech import setup as SS
    return SS.on_disk(cfg_dir)


def cards(h, app, catalog=None, fresh=False):
    """the Plugins tab's list: one card per installed plugin (see the module doc). fresh: read the wallet's statement first
    (the tab does that when it opens, never on its 5 s refresh)."""
    cfg_dir, cache = h.cfg_dir, h.cache
    if fresh and any(i.get("edition") == "supporter" and i.get("gate") not in gate.RUNS for i in pxp.installed(cfg_dir, h.pubkeys, cache)):
        refresh_statement(cfg_dir)
    cat = catalog if catalog is not None else _catalog(cfg_dir)
    rows, speech_by = [], []
    off = set(h.disabled()) if hasattr(h, "disabled") else set()
    consent_raw = h.consent._load()
    for it in pxp.installed(cfg_dir, h.pubkeys, cache):
        pid = it["plugin_id"]
        row = {"plugin_id": pid, "name": it.get("name") or pid, "version": it.get("version"), "edition": it.get("edition"),
               "publisher": "PXA", "ok": bool(it.get("ok"))}
        row.update({k: v for k, v in h.status_of(pid).items() if k in ("state", "why", "restarts")})
        if not it.get("ok"):
            code = it.get("error") or "invalid_manifest"
            row["signature"] = {"ok": False, "code": code, "text": PROBLEMS.get(code, PROBLEMS["invalid_manifest"])}
            row["lines"], row["consent"], row["entitlement"], row["page"] = [], {"status": "none", "permissions": [], "missing": []}, None, None
            rows.append(row)
            continue
        try:
            m, _d = h.manifest(pid)
        except PluginError as e:
            row["ok"] = False
            row["signature"] = {"ok": False, "code": e.code, "text": PROBLEMS.get(e.code, PROBLEMS["invalid_manifest"])}
            row["lines"], row["consent"], row["entitlement"], row["page"] = [], {"status": "none", "permissions": [], "missing": []}, None, None
            rows.append(row)
            continue
        rec = consent_raw.get(pid)
        cs = h.consent.state(m)
        row["signature"] = {"ok": True, "kid": m["signature"]["kid"], "text": "Every file checked just now."}
        row["consent"] = cs
        row["lines"] = lines(m, rec if rec and not rec.get("revoked_at") else None)
        row["entitlement"] = _entitlement(cfg_dir, cache, m, cat)
        tab = ((m.get("ui") or {}).get("tab")) or {}
        page = tab.get("page")
        row["page"] = {"title": tab.get("title") or m["name"]} if page and page in (m.get("files") or {}) else None
        if row["page"] and tab.get("nav") is True:                       # its own tab in Control's bar (plugin-host.js navSync)
            row["page"].update(nav=True, glyph=tab.get("glyph") if tab.get("glyph") in NAV_GLYPHS else "plug")
        row["store"] = _store_row(cat, m, pid in off)
        row["listens"] = bool(m.get("listen"))
        row["keys"] = len(h.keys.names(pid)) if getattr(h, "keys", None) else 0     # S4: how many keys it keeps (a count, never a name or value)
        if {"speech.listen", "speech.speak"} & set(m.get("permissions") or []):
            speech_by.append(row["name"])
        rows.append(row)
    # "Speech on this computer": shown when a plugin needs speech or speech is set up (to update it), else None. The status
    # can be a `setup --status` process: asked only when a plugin needs speech or a setup left its files here
    speech = None
    if hasattr(h, "speech_setup_status") and (speech_by or _speech_on_disk(cfg_dir)):
        st = h.speech_setup_status()
        if speech_by or st.get("engines") != "missing":
            speech = dict(st, needed_by=speech_by)
    return {"plugins": rows, "notices": list(h.notices), "lan": bool(getattr(app, "lan", False)), "speech": speech}


# ---- the sandboxed page -------------------------------------------------------------------------------------------------
BRIDGE_JS = r"""<script>
(function(){
  var pending = {}, seq = 0, listeners = [];
  var api = {theme: {}, id: null};
  window.pxa = api;
  api.fetch = function (path, opts) {
    opts = opts || {};
    return new Promise(function (resolve, reject) {
      var id = ++seq;
      pending[id] = {resolve: resolve, reject: reject};
      parent.postMessage({pxa: "plugin/1", op: "fetch", id: id, path: String(path), method: opts.method || "GET",
                          body: opts.body == null ? null : String(opts.body), headers: opts.headers || {}}, "*");
      setTimeout(function () { if (pending[id]) { delete pending[id]; reject(new Error("no answer from PXA Control")); } }, 70000);
    }).then(function (r) {
      return {ok: r.status >= 200 && r.status < 300, status: r.status, ctype: r.ctype, text: function () { return Promise.resolve(r.text || ""); },
              json: function () { return Promise.resolve(JSON.parse(r.text || "null")); }, b64: r.b64 || null};
    });
  };
  api.resize = function (px) { parent.postMessage({pxa: "plugin/1", op: "resize", height: Math.round(Number(px) || 0)}, "*"); };
  api.onTheme = function (fn) { listeners.push(fn); if (api.theme && api.theme.name) fn(api.theme); };
  window.addEventListener("message", function (ev) {
    if (ev.source !== parent) return;
    var d = ev.data;
    if (!d || d.pxa !== "plugin/1") return;
    if (d.op === "result" && pending[d.id]) { var p = pending[d.id]; delete pending[d.id]; p.resolve(d); }
    else if (d.op === "error" && pending[d.id]) { var q = pending[d.id]; delete pending[d.id]; q.reject(new Error(d.error || "refused")); }
    else if (d.op === "hello" || d.op === "theme") {
      api.theme = d.theme || {}; api.id = d.plugin_id || api.id;
      var r = document.documentElement;
      Object.keys(api.theme.vars || {}).forEach(function (k) { r.style.setProperty(k, api.theme.vars[k]); });
      r.dataset.theme = api.theme.name || "";
      listeners.forEach(function (fn) { try { fn(api.theme); } catch (e) {} });
    }
  });
  parent.postMessage({pxa: "plugin/1", op: "ready"}, "*");
})();
</script>"""


def frame_html(h, pid):
    """-> bytes: the plugin's page with the bridge client in front. 404 when it has none, 400 when it does not verify."""
    from .host import HostError
    m, d = h.manifest(pid)
    tab = ((m.get("ui") or {}).get("tab")) or {}
    page = tab.get("page")
    if not page or page not in (m.get("files") or {}):
        raise HostError("%s has no page" % m["name"], "no_page", 404)
    p = os.path.realpath(os.path.join(d, *page.split("/")))
    root = os.path.realpath(d)
    if not p.startswith(root + os.sep):
        raise HostError("the page is outside the plugin", "bad_file", 400)
    try:
        with open(p, "rb") as f:
            raw = f.read(MAX_PAGE + 1)
    except OSError:
        raise HostError("the page is missing", "bad_file", 400)
    if len(raw) > MAX_PAGE:
        raise HostError("the page is too large", "bad_file", 400)
    html = raw.decode("utf-8", "replace")
    low = html.lower()
    shim = BRIDGE_JS
    i = low.find("<head")
    if i >= 0:
        j = low.find(">", i)
        html = html[:j + 1] + shim + html[j + 1:]
    else:
        html = "<!doctype html><meta charset=utf-8>" + shim + html
    return html.encode("utf-8")


def register(ROUTES, wrap_plain):
    """wrap_plain(fn) -> a route (app, body, query) that gives fn the plugin host, translating PluginError."""
    from ..core import Reply

    def r_cards(h, app, body, query):
        return cards(h, app, fresh=bool((query.get("fresh") or [""])[0]))

    def r_frame(h, app, body, query):
        pid = str((query.get("plugin_id") or [""])[0])
        html = frame_html(h, pid)
        return Reply(html, "text/html; charset=utf-8", 200, {"Content-Security-Policy": FRAME_CSP, "X-Frame-Options": "SAMEORIGIN"})

    ROUTES[("GET", "/api/plugins/host/cards")] = wrap_plain(r_cards)
    ROUTES[("GET", "/api/plugins/host/frame")] = wrap_plain(r_frame)
    return ROUTES
