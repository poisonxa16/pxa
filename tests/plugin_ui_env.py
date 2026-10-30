"""Shared fixture for the Plugins tab checks (tests/test-pxa-plugin-ui.py and tests/plugin-ui-gui-check.py): signed stub plugins
built with a throwaway key, installed into a config folder, each with a page that talks through the postMessage bridge.

make_plugin(cfg, pid, ...) builds, signs (build-pxp.py's own builder) and installs one; nothing here is a real key or a real plugin.
"""
import importlib.util
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

PLUG_SEED = bytes(range(81, 113))      # throwaway plugin publisher key (never a real key)
PLUG_KID = "plug-ui-test"

STUB = r'''#!/usr/bin/env python3
import json, os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))
from pxa_plugin_child import Plugin


def handle(method, path, query, ctype, body):
    if path.endswith("/echo"):
        return 200, "application/json", {"method": method, "path": path, "query": query, "body": body.decode("utf-8", "replace")}
    if path.endswith("/bin"):
        return 200, "audio/wav", b"RIFF\x00\x01\x02\x03"
    return 404, "application/json", {"error": "no such stub route"}


sys.exit(Plugin(handle).run())
'''

# the plugin's own page: ONE html file. It reports what the bridge lets it do, and what it must not be able to do.
PAGE = r'''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>%(name)s page</title>
<style>
  body{margin:0;padding:14px;font:14px/1.45 var(--font,system-ui,sans-serif);background:var(--surface,#fff);color:var(--text,#111)}
  h2{margin:0 0 8px;font-size:16px} .row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
  button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line,#ccc);background:var(--surface-2,#eee);color:inherit}
  pre{margin:8px 0;padding:8px;border:1px solid var(--line,#ccc);border-radius:8px;overflow:auto;white-space:pre-wrap}
  .ok{color:var(--ok,green)} .bad{color:var(--bad,red)}
</style>
<h2 id="hd">%(name)s</h2>
<p id="hi" class="ok">waiting for the bridge</p>
<div class="row">
  <button id="b-echo">Call my route</button>
  <button id="b-other">Call another plugin</button>
  <button id="b-ctl">Call Control</button>
  <button id="b-same">Read Control's page</button>
  <button id="b-bin">Binary answer</button>
</div>
<pre id="out">-</pre>
<script>
var out = document.getElementById("out");
function show(t) { out.textContent = t; }
pxa.onTheme(function (t) { document.getElementById("hi").textContent = "bridge ready, theme " + t.name; document.documentElement.dataset.pluginTheme = t.name; });
document.getElementById("b-echo").onclick = function () {
  pxa.fetch("/api/plugins/%(pid)s/echo?x=1", {method: "POST", body: JSON.stringify({hello: "page"}), headers: {"Content-Type": "application/json"}})
    .then(function (r) { return r.json(); }).then(function (j) { show("echo ok " + JSON.stringify(j)); }, function (e) { show("echo failed " + e.message); });
};
document.getElementById("b-other").onclick = function () {
  pxa.fetch("/api/plugins/someone-else/echo").then(function (r) { show("OTHER ALLOWED " + r.status); }, function (e) { show("other refused: " + e.message); });
};
document.getElementById("b-ctl").onclick = function () {
  pxa.fetch("/api/plugins/host").then(function (r) { show("CONTROL ALLOWED " + r.status); }, function (e) { show("control refused: " + e.message); });
};
document.getElementById("b-same").onclick = function () {
  var r = [];
  try { r.push("cookie=" + JSON.stringify(document.cookie)); } catch (e) { r.push("cookie blocked"); }
  try { r.push("storage=" + typeof localStorage.length); } catch (e) { r.push("storage blocked"); }
  try { r.push("parent=" + typeof parent.document.title); } catch (e) { r.push("parent blocked"); }
  fetch("/api/info").then(function (x) { r.push("direct fetch " + x.status); show(r.join(" ; ")); }, function () { r.push("direct fetch blocked"); show(r.join(" ; ")); });
};
document.getElementById("b-bin").onclick = function () {
  pxa.fetch("/api/plugins/%(pid)s/bin").then(function (r) { show("bin " + r.status + " " + r.ctype + " b64=" + r.b64); }, function (e) { show("bin failed " + e.message); });
};
</script>
'''


def _build_mod():
    spec = importlib.util.spec_from_file_location("build_pxp", os.path.join(TOOLS, "pxa-private", "plugins", "build-pxp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def plugin_keys():
    from pxa_control import store as S
    return {PLUG_KID: S.ed25519_public(PLUG_SEED).hex()}


def make_plugin(cfg, work, pid, name, edition="free", perms=None, listen=None, entitlement=None, overlay=False, version="0.1.0"):
    """build + sign + install one stub plugin into cfg (config dir); -> the install row."""
    from pxa_control.plugins import pxp
    src = os.path.join(work, "src", pid)
    for sub in ("bin", "ui", "lib"):
        os.makedirs(os.path.join(src, sub), exist_ok=True)
    with open(os.path.join(src, "bin", "stub"), "w") as f:
        f.write(STUB)
    with open(os.path.join(src, "ui", "index.html"), "w") as f:
        f.write(PAGE % {"pid": pid, "name": name})
    if overlay:
        with open(os.path.join(src, "ui", "overlay.html"), "w") as f:
            f.write("<p>overlay</p>\n")
    shutil.copy(os.path.join(TOOLS, "pxa_control", "plugins", "child.py"), os.path.join(src, "lib", "pxa_plugin_child.py"))
    spec = {"schema": 1, "plugin_id": pid, "name": name, "version": version, "publisher": "pxa", "edition": edition,
            "min_host": "4.1", "entry": {"kind": "process", "argv": ["bin/stub", "serve"], "protocol": "pxa-plugin/1"},
            "routes": ["/api/plugins/%s/" % pid], "ui": {"tab": {"title": name, "page": "ui/index.html"}},
            "permissions": list(perms or []), "entitlement": (entitlement or pid) if edition == "supporter" else None,
            "pack": ["bin", "ui", "lib"]}
    if listen:
        spec["listen"] = listen
    if overlay:
        spec["overlay"] = {"page": "ui/overlay.html", "scope": "loopback"}
    with open(os.path.join(src, "plugin.src.json"), "w") as f:
        json.dump(spec, f)
    path, _body = _build_mod().build(src, PLUG_SEED, PLUG_KID, os.path.join(work, "out"))
    return pxp.install(path, cfg, plugin_keys())


def install_all(cfg, work):
    """the three cards of the check: a free voice plugin (speech.*, net.lan, device.audio-capture, a scoped host, a port), a supporter
    plugin whose unlock item is `dictate`, and a free plugin with no permissions."""
    make_plugin(cfg, work, "voice-demo", "Voice demo", "free",
                ["speech.listen", "speech.speak", "device.audio-capture", "net.lan", "net.internet:api.example.com", "model.chat"],
                [{"port": 15310, "proto": "wyoming", "scope": "lan"}], overlay=True)
    make_plugin(cfg, work, "dictate-stub", "Dictate (test stub)", "supporter", ["device.mic", "input.type", "speech.listen"], entitlement="dictate")
    make_plugin(cfg, work, "quiet-demo", "Quiet demo", "free", [])
