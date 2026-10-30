#!/usr/bin/env python3
"""Plugin manifest additions (PLUGINS-PROTOTYPE-PLAN S4): device.audio-capture, net.lan, scoped net.internet:<host>,
`listen` [{port, proto, scope}] and `overlay` {page, scope}. Pure unit: no network, no GPU, no server.

    python3 tests/test-pxa-plugin-manifest.py

The three example manifests from the design notes (SATELLITE/SCRIBE/COHOST-DESIGN.md s4) and the real plugin.src.json of
Satellite (6ec7b5a843) and Scribe (71de0d68ff) are copied here as literals, so the
test runs without those branches; every plugin.src.json in the private tree is checked too (the packager's own path).
"""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)
os.environ.setdefault("PXA_CONTROL_CONFIG_DIR", tempfile.mkdtemp(prefix="pxa-plugin-manifest-test-"))

from pxa_control import plugins as P  # noqa: E402

SHA = hashlib.sha256(b"x").hexdigest()
SIG = {"kid": "plug-2026-10", "sig": "<b64url>"}

SATELLITE_DESIGN = {
    "schema": 1, "plugin_id": "satellite", "name": "Satellite", "version": "0.1.0", "publisher": "pxa", "edition": "supporter",
    "min_host": "4.1", "entry": {"kind": "process", "argv": ["bin/satellite"], "protocol": "pxa-plugin/1"},
    "routes": ["/api/plugins/satellite/"], "ui": {"tab": {"title": "Satellite", "page": "ui/index.html"}},
    "permissions": ["net.lan", "model.chat"],
    "listen": [{"port": 10310, "proto": "wyoming", "scope": "lan"}, {"port": 11510, "proto": "http", "scope": "lan"}],
    "entitlement": "satellite", "files": {"bin/satellite": "<sha256>", "ui/index.html": "<sha256>"},
    "signature": SIG}
SCRIBE_DESIGN = {
    "schema": 1, "plugin_id": "scribe", "name": "Scribe", "version": "0.1.0", "publisher": "pxa", "edition": "supporter",
    "min_host": "4.1", "entry": {"kind": "process", "argv": ["bin/scribe"], "protocol": "pxa-plugin/1"},
    "routes": ["/api/plugins/scribe/"], "ui": {"tab": {"title": "Scribe", "page": "ui/index.html"}},
    "permissions": ["device.mic", "device.audio-capture", "model.chat", "fs.workspace.write", "fs.write"],
    "entitlement": "scribe", "files": {"bin/scribe": "<sha256>", "bin/capture-win.exe": "<sha256>", "bin/capture-mac": "<sha256>",
                                       "ui/index.html": "<sha256>"}, "signature": SIG}
COHOST_DESIGN = {
    "schema": 1, "plugin_id": "cohost", "name": "Co-host", "version": "0.1.0", "publisher": "pxa", "edition": "supporter",
    "min_host": "4.1", "entry": {"kind": "process", "argv": ["bin/cohost"], "protocol": "pxa-plugin/1"},
    "routes": ["/api/plugins/cohost/"], "ui": {"tab": {"title": "Co-host", "page": "ui/index.html"}},
    "overlay": {"page": "overlay/index.html", "scope": "loopback"},
    "permissions": ["net.internet:irc.chat.twitch.tv", "net.internet:youtube.googleapis.com", "net.internet:oauth2.googleapis.com",
                    "device.speaker", "model.chat", "fs.workspace.write"],
    "entitlement": "cohost", "files": {"bin/cohost": "<sha256>", "ui/index.html": "<sha256>", "overlay/index.html": "<sha256>"},
    "signature": SIG}
# the real sources (plugin.src.json: no files/signature; pack/dev_only are the packager's and never signed)
SATELLITE_SRC = {
    "schema": 1, "plugin_id": "satellite", "name": "Satellite", "version": "0.1.0",
    "publisher": "pxa", "edition": "supporter", "min_host": "4.1",
    "entry": {"kind": "process", "argv": ["bin/satellite", "serve"], "protocol": "pxa-plugin/1"},
    "routes": ["/api/plugins/satellite/"],
    "ui": {"tab": {"title": "Satellite", "page": "ui/index.html"}},
    "permissions": ["net.lan", "model.chat"],
    "listen": [{"port": 10310, "proto": "wyoming", "scope": "lan"}, {"port": 11510, "proto": "http", "scope": "lan"}],
    "entitlement": "satellite",
    "python": {"package": "satellite", "path": "lib"},
    "pack": ["bin", "ui", "lib"], "dev_only": ["lib/satellite/fakes.py"]}
SCRIBE_SRC = {
    "schema": 1, "plugin_id": "scribe", "name": "Scribe", "version": "0.1.0",
    "publisher": "pxa", "edition": "supporter", "min_host": "4.0",
    "entry": {"kind": "process", "argv": ["bin/scribe"], "protocol": "pxa-plugin/1"},
    "routes": ["/api/plugins/scribe/"],
    "ui": {"tab": {"title": "Scribe", "page": "ui/index.html"}},
    "permissions": ["device.mic", "device.audio-capture", "model.chat", "fs.workspace.write", "fs.write"],
    "entitlement": "scribe",
    "python": {"package": "scribe", "path": "lib"},
    "pack": ["bin", "ui", "lib"], "dev_only": ["lib/scribe/devdemo.py", "lib/scribe/fakes.py"]}


def filled(m):
    """a design example with real hashes (the notes write <sha256>)."""
    m = copy.deepcopy(m)
    m["files"] = {k: SHA for k in m["files"]}
    return m


def from_src(spec):
    """what build-pxp signs: the source minus pack/dev_only, plus files (+ the python package's __init__)."""
    body = {k: v for k, v in spec.items() if k not in ("pack", "dev_only", "exclude", "files", "signature")}
    files = {body["entry"]["argv"][0]: SHA, body["ui"]["tab"]["page"]: SHA}
    if body.get("python"):
        files["%s/%s/__init__.py" % (body["python"]["path"], body["python"]["package"])] = SHA
    if body.get("overlay"):                                      # an overlay page ships in the package (Co-host)
        files[body["overlay"]["page"]] = SHA
    body["files"] = files
    body["signature"] = SIG
    return body


class Accepts(unittest.TestCase):
    def test_the_three_design_examples(self):
        for m in (SATELLITE_DESIGN, SCRIBE_DESIGN, COHOST_DESIGN):
            self.assertEqual(P.validate_manifest(filled(m)), [], m["plugin_id"])

    def test_the_real_satellite_and_scribe_sources(self):
        for spec in (SATELLITE_SRC, SCRIBE_SRC):
            self.assertEqual(P.validate_manifest(from_src(spec)), [], spec["plugin_id"])

    def test_every_source_in_the_private_tree(self):
        base = os.path.join(TOOLS, "pxa-private", "plugins")
        seen = 0
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            for n in sorted(os.listdir(os.path.join(base, d))) if os.path.isdir(os.path.join(base, d)) else []:
                if not n.endswith(".src.json"):                         # plugin.src.json, plugin.plus.src.json, ...
                    continue
                with open(os.path.join(base, d, n)) as fh:
                    self.assertEqual(P.validate_manifest(from_src(json.load(fh))), [], d + "/" + n)
                seen += 1
        self.assertGreaterEqual(seen, 1)

    def test_ui_tab_may_ask_for_its_own_nav_tab(self):
        for glyph in P.NAV_GLYPHS:
            m = filled(SATELLITE_DESIGN)
            m["ui"]["tab"].update(nav=True, glyph=glyph)
            self.assertEqual(P.validate_manifest(m), [], glyph)
        m = filled(SATELLITE_DESIGN)
        m["ui"]["tab"]["nav"] = False                                   # glyph optional, nav optional
        self.assertEqual(P.validate_manifest(m), [])

    def test_new_permissions_and_their_words(self):
        for k in ("device.audio-capture", "net.lan", "speech.listen", "speech.speak"):
            self.assertIn(k, P.PERMISSIONS)
        self.assertEqual(P.PERMISSIONS["device.audio-capture"], "hear what this computer plays (a call, a video)")
        self.assertEqual(P.permission_words("net.internet:irc.chat.twitch.tv"), "reach irc.chat.twitch.tv on the internet (nothing else)")
        lines = dict(P.consent_lines(filled(COHOST_DESIGN)))
        self.assertIn("net.internet:youtube.googleapis.com", lines)
        self.assertIn("overlay", lines)
        lines = dict(P.consent_lines(filled(SATELLITE_DESIGN)))
        self.assertEqual(lines["listen:10310"], "accept wyoming connections from your home network on port 10310")

    def test_consent_covers_scoped_hosts_and_new_hosts_need_reconsent(self):
        cfg = tempfile.mkdtemp()
        cs = P.ConsentStore(cfg)
        m = filled(COHOST_DESIGN)
        cs.grant(m, m["permissions"])
        self.assertTrue(cs.allowed(m, "net.internet:irc.chat.twitch.tv"))
        self.assertFalse(cs.allowed(m, "net.internet:api.twitch.tv"))
        m2 = copy.deepcopy(m)
        m2["permissions"].append("net.internet:api.twitch.tv")
        self.assertEqual(P.validate_manifest(m2), [])
        self.assertEqual(cs.state(m2)["status"], "needs_reconsent")


class Refuses(unittest.TestCase):
    def bad(self, m, word):
        probs = P.validate_manifest(m)
        self.assertTrue(probs and any(word in x for x in probs), probs)

    def test_internet_wildcards_addresses_and_home_names(self):
        for host in ("*", "*.twitch.tv", "twitch.*", "", "192.168.1.5", "10.0.0.1", "[::1]", "localhost", "nas.local",
                     "router.lan", "printer.home.arpa", "IRC.CHAT.TWITCH.TV", "irc.chat.twitch.tv:6697", "-a.example.com",
                     "a..example.com", "example", "http://example.com", "example.com/path"):
            m = filled(COHOST_DESIGN)
            m["permissions"] = ["net.internet:" + host, "model.chat"]
            self.bad(m, "permissions")
        self.assertIsNone(P.scoped_host("net.internet:*"))

    def test_ui_tab_nav_and_glyph(self):
        for nav, glyph in (("yes", "plug"), (1, "plug"), (True, "rocket"), (True, ""), (True, None)):
            m = filled(SATELLITE_DESIGN)
            m["ui"]["tab"].update(nav=nav, glyph=glyph)
            self.bad(m, "ui.tab")

    def test_too_many_hosts(self):
        m = filled(COHOST_DESIGN)
        m["permissions"] = ["net.internet:h%d.example.com" % i for i in range(P.MAX_SCOPED_HOSTS + 1)]
        self.bad(m, "scoped internet hosts")

    def test_public_bind_and_listen_shapes(self):
        for ln in ({"port": 10310, "proto": "wyoming", "scope": "public"},
                   {"port": 10310, "proto": "wyoming", "scope": "internet"},
                   {"port": 10310, "proto": "wyoming", "scope": "any"},
                   {"port": 10310, "proto": "wyoming", "scope": "lan", "host": "0.0.0.0"},
                   {"port": 10310, "proto": "wyoming", "scope": "lan", "bind": "0.0.0.0"},
                   {"port": 80, "proto": "http", "scope": "lan"},
                   {"port": 70000, "proto": "http", "scope": "lan"},
                   {"port": True, "proto": "http", "scope": "lan"},
                   {"port": "10310", "proto": "http", "scope": "lan"},
                   {"port": 10310, "proto": "smtp", "scope": "lan"},
                   {"port": 10310, "proto": "http"}):
            m = filled(SATELLITE_DESIGN)
            m["listen"] = [ln]
            self.bad(m, "listen")
        m = filled(SATELLITE_DESIGN)
        m["permissions"] = ["model.chat"]                       # lan without the net.lan permission
        self.bad(m, "net.lan")
        m = filled(SATELLITE_DESIGN)
        m["listen"] = [{"port": 10310, "proto": "http", "scope": "lan"}, {"port": 10310, "proto": "wyoming", "scope": "lan"}]
        self.bad(m, "twice")
        m = filled(SATELLITE_DESIGN)
        m["listen"] = [{"port": 10310 + i, "proto": "http", "scope": "lan"} for i in range(P.MAX_LISTEN + 1)]
        self.bad(m, "listen")
        m = filled(SATELLITE_DESIGN)
        m["listen"] = []
        self.bad(m, "listen")
        m = filled(SATELLITE_DESIGN)                            # loopback needs no net.lan
        m["permissions"], m["listen"] = ["model.chat"], [{"port": 10310, "proto": "http", "scope": "loopback"}]
        self.assertEqual(P.validate_manifest(m), [])

    def test_free_loopback_port(self):
        """Review item 4: port 0 (the system picks a free port) only on loopback; the Anibal wrapper serves on 127.0.0.1:0."""
        m = filled(SATELLITE_DESIGN)
        m["permissions"], m["listen"] = ["model.chat"], [{"port": 0, "proto": "http", "scope": "loopback"}]
        self.assertEqual(P.validate_manifest(m), [])
        lines = dict(P.consent_lines(m))
        self.assertEqual(lines["listen:0"], "accept http connections from programs on this computer, on a free port")
        from pxa_control.plugins import ui as PUI
        row = [r for r in PUI.lines(m) if r["key"] == "listen:0"][0]
        self.assertEqual((row["reach"], row["ask"]), ("local", False))
        m["listen"] = [{"port": 0, "proto": "http", "scope": "loopback"}, {"port": 0, "proto": "ws", "scope": "loopback"}]
        self.bad(m, "twice")                                    # one consent line per port
        m = filled(SATELLITE_DESIGN)                            # never a free port off this computer, even with net.lan
        self.assertIn("net.lan", m["permissions"])
        m["listen"] = [{"port": 0, "proto": "http", "scope": "lan"}]
        self.bad(m, "port 0 only with scope loopback")
        for ln in ({"port": 0, "proto": "http", "scope": "public"}, {"port": -1, "proto": "http", "scope": "loopback"},
                   {"port": 1023, "proto": "http", "scope": "loopback"}, {"port": False, "proto": "http", "scope": "loopback"}):
            m = filled(SATELLITE_DESIGN)
            m["listen"] = [ln]
            self.bad(m, "listen")
        m = filled(SATELLITE_DESIGN)                            # fixed ports keep their words
        self.assertEqual(dict(P.consent_lines(m))["listen:10310"], "accept wyoming connections from your home network on port 10310")

    def test_overlay_off_loopback(self):
        for ov in ({"page": "overlay/index.html", "scope": "lan"},
                   {"page": "overlay/index.html", "scope": "public"},
                   {"page": "overlay/index.html"},
                   {"page": "overlay/index.html", "scope": "loopback", "host": "0.0.0.0"},
                   {"page": "../overlay.html", "scope": "loopback"},
                   {"page": "overlay/missing.html", "scope": "loopback"},
                   "overlay/index.html"):
            m = filled(COHOST_DESIGN)
            m["overlay"] = ov
            self.bad(m, "overlay")

    def test_unknown_permission_still_refused(self):
        m = filled(SCRIBE_DESIGN)
        m["permissions"] = ["device.audio-capture", "device.everything"]
        self.bad(m, "permissions")
        m["permissions"] = ["net.lan", "net.lan"]
        self.bad(m, "permissions")


if __name__ == "__main__":
    unittest.main(verbosity=1)
