"""PXA Control chat agent (v3.1): an in-house, stdlib-only agent harness for Control's Chat tab.

Optional: PXA Control imports this package if it is there and loses only the new chat bits if it is not.
    import pxa_chat; pxa_chat.register(ROUTES, STATIC, Reply, config_dir, load_config)
Built from PXA's own earlier code (ported, not imported): an in-house agent harness (protocol, sandbox, tools),
its seat rule, an in-house agent and its guards (tool table, DANGER denylist, injection screen, approvals), and an
in-house sub-agent server.
"""
import os
import threading

from . import routes as _routes
from .routes import HostPolicy      # so a host can build the policy without reaching into .routes

__version__ = "1.0"
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
AGENT_PLUGINS = ("markdown", "attach", "actions", "params", "context", "ux", "stream", "memory")
# routes whose request body may be larger than Control's 1 MiB default: a run carries up to 4 images of <=1.4 MB
# of base64 each (routes.take_images), so 1 MiB refused any image over ~750 KB [C13]
BODY_LIMITS = {"/api/chat/run": 8 << 20}
_hubs = {}
_lock = threading.Lock()


def register(ROUTES, STATIC, Reply, config_dir, load_config=None, host=None, who=None):
    """add the chat routes, agent.js/agent.css and the agent-<name> plugins to Control's tables.

    `host` is a routes.HostPolicy, or a callable taking the app and returning one: it says whether host
    access is permitted on this Control, where the allowlist file lives, and where an audit record goes.
    A callable is the usual form, because the answer depends on how that Control was started (--lan).
    Without one the hub refuses every host call, so a caller that forgets it gets the safe behaviour.
    `who` is a callable returning the asking client's address (Control's req_who); it keys the per-client run cap."""
    def hub_for(app):
        with _lock:
            h = _hubs.get(id(app))
            if h is None:
                h = _routes.Hub(os.path.join(config_dir(), "chat"), load_config,
                                host(app) if callable(host) else host, who=who)
                _hubs[id(app)] = h
            return h
    ROUTES.update(_routes.make_routes(hub_for, Reply))
    # absolute paths: Control's static handler joins them onto its UI dir, and os.path.join keeps an absolute one
    STATIC.update({"/agent.js": (os.path.join(UI_DIR, "agent.js"), "application/javascript; charset=utf-8"),
                   "/agent.css": (os.path.join(UI_DIR, "agent.css"), "text/css; charset=utf-8")})
    for n in AGENT_PLUGINS:              # the Assistant view's plugins (agent-<name>.js/.css; CHAT-PLUGINS.md)
        STATIC.update({f"/agent-{n}.js": (os.path.join(UI_DIR, f"agent-{n}.js"), "application/javascript; charset=utf-8"),
                       f"/agent-{n}.css": (os.path.join(UI_DIR, f"agent-{n}.css"), "text/css; charset=utf-8")})
    STATIC["/agent-manifest.json"] = (os.path.join(UI_DIR, "agent-manifest.json"), "application/manifest+json; charset=utf-8")
    STATIC["/agent-sw.js"] = (os.path.join(UI_DIR, "agent-sw.js"), "application/javascript; charset=utf-8")
    STATIC["/agent-icon-192.png"] = (os.path.join(UI_DIR, "agent-icon-192.png"), "image/png")
    STATIC["/agent-icon-512.png"] = (os.path.join(UI_DIR, "agent-icon-512.png"), "image/png")
    _vendor_mimes = {".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                     ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
                     "": "text/plain; charset=utf-8"}
    vendor = os.path.join(UI_DIR, "vendor")
    for dirpath, _, names in os.walk(vendor):
        for name in names:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, UI_DIR).replace(os.sep, "/")
            ext = os.path.splitext(name)[1].lower()
            STATIC["/" + rel] = (full, _vendor_mimes.get(ext, "application/octet-stream"))
    return True
