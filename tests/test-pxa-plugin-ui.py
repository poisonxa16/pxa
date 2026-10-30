#!/usr/bin/env python3
"""The Plugins tab's server side (PLUGINS-PROTOTYPE-PLAN S3): GET /api/plugins/host/cards (signature, consent, entitlement per
plugin, the consent card's words), the sandboxed plugin page GET /api/plugins/host/frame, the framing headers, and static checks
of plugin-host.js (the bridge's path guard, no innerHTML from data, store wording). The browser side is tests/plugin-ui-gui-check.py.

    python3 tests/test-pxa-plugin-ui.py

Loopback only, throwaway keys, no GPU, no internet. POSIX.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
TMP = tempfile.mkdtemp(prefix="pxa-plugin-ui-test-")
CFG = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_CONFIG_DIR"] = CFG
os.environ["PXA_STORE_URL"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import plugin_ui_env as E  # noqa: E402
import pxa_control as C  # noqa: E402
from pxa_control import store as S  # noqa: E402
from pxa_control import plugins as P  # noqa: E402
from pxa_control.plugins import gate as G, host as H, ui as U  # noqa: E402

UI_DIR = os.path.join(ROOT, "tools", "pxa_control_ui")
ENT_SEED = bytes(range(1, 33))
ENT_KID = "ent-ui-test"
ENT_KEYS = {ENT_KID: S.ed25519_public(ENT_SEED).hex()}


def grant(item, ent="0123456789abcdef0123456789abcdef"):
    _seed, pub = S.install_key(CFG)
    import time
    now = int(time.time())
    st = {"v": 1, "iss": "licd", "kid": ENT_KID, "ent_id": ent, "kind": "plugin", "item_id": item, "level": None, "sub": "a" * 32,
          "install_pub": pub, "issued": now, "refresh_after": now + 86400, "not_after": now + 180 * 86400}
    return {"statement": st, "signature": S.b64u(S.ed25519_sign(ENT_SEED, S.ENT_PREFIX + S.canonical(st)))}


class Tab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        E.install_all(CFG, TMP)
        cls.cache = S.EntitlementCache(CFG, ENT_KEYS)
        cls.host = H.PluginHost(CFG, cache=cls.cache, pubkeys=E.plugin_keys(), health_s=3600, backoff=(0, 0, 0, 0, 0), tick_s=3600)
        cls.app = C.App(C.desktop.load_launcher(), port=7779, models_dirs=[os.path.join(TMP, "models")])
        cls.app.store_cache = cls.cache
        cls.app.plugin_host = cls.host
        cls.srv = C.make_server(cls.app, "127.0.0.1", 0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        for pid in ("voice-demo", "dictate-stub", "quiet-demo"):
            self.host.stop(pid, "test")
        with self.cache.lock:
            for p in [self.cache.path] + self.cache._journal_paths():   # grants.json AND both lock journals: the
                try:                                                   # journal keeps a lock across a grants.json
                    os.unlink(p)                                       # rewrite by design (RC3.6 item 4), so a clean
                except FileNotFoundError:                              # store for the next test drops all three
                    pass
        with self.host.consent.lock:
            self.host.consent._save({})
        self.app.lan = False

    def ctl(self, method, path, body=None, token=True, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Content-Type": "application/json"}
        if token:
            h["X-PXA-Token"] = self.app.token
        h.update(headers or {})
        r = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                b = resp.read()
                return resp.status, b, resp.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def cards(self):
        st, b, _h = self.ctl("GET", "/api/plugins/host/cards")
        self.assertEqual(st, 200, b)
        return {p["plugin_id"]: p for p in json.loads(b)["plugins"]}

    # ---- the list ------------------------------------------------------------------------------------------
    def test_every_installed_plugin_is_a_card_with_signature_and_no_consent_yet(self):
        c = self.cards()
        self.assertEqual(sorted(c), ["dictate-stub", "quiet-demo", "voice-demo"])
        for p in c.values():
            self.assertTrue(p["ok"])
            self.assertEqual(p["signature"]["ok"], True)
            self.assertEqual(p["signature"]["kid"], E.PLUG_KID)
            self.assertEqual(p["publisher"], "PXA")
            self.assertEqual(p["consent"]["status"], "none")
            self.assertEqual(p["state"], "stopped")
        self.assertEqual(c["voice-demo"]["version"], "0.1.0")
        self.assertIsNone(c["voice-demo"]["entitlement"])           # free
        self.assertEqual(c["voice-demo"]["page"], {"title": "Voice demo"})

    def test_consent_card_has_plain_words_for_every_permission(self):
        words = {l["key"]: l for l in self.cards()["voice-demo"]["lines"]}
        self.assertIn("turn speech into text on this computer", words["speech.listen"]["words"])
        self.assertEqual(words["speech.speak"]["words"], "turn text into speech on this computer")
        self.assertEqual(words["net.lan"]["words"], "accept connections from your home network")
        self.assertEqual(words["device.audio-capture"]["words"], "hear what this computer plays (a call, a video)")
        self.assertEqual(words["net.internet:api.example.com"]["words"], "reach api.example.com on the internet (nothing else)")
        self.assertIn("listen:15310", words)
        self.assertIn("your home network", words["listen:15310"]["words"])
        self.assertIn("overlay", words)
        # every permission is a tick box; the port and the overlay are "also" lines that follow from them
        for k in ("speech.listen", "speech.speak", "net.lan", "device.audio-capture", "model.chat", "net.internet:api.example.com"):
            self.assertTrue(words[k]["ask"], k)
        self.assertFalse(words["listen:15310"]["ask"])
        self.assertFalse(words["overlay"]["ask"])
        # the tag says what the permission really reaches: local speech stays here, the network and other programs do not
        self.assertEqual(words["speech.listen"]["tag"], "stays on this computer")
        self.assertEqual(words["speech.speak"]["tag"], "stays on this computer")
        self.assertEqual(words["model.chat"]["tag"], "stays on this computer")
        self.assertEqual(words["device.audio-capture"]["tag"], "uses a sensor")
        for k in ("net.lan", "net.internet:api.example.com", "listen:15310"):
            self.assertEqual(words[k]["tag"], "reaches beyond PXA", k)
        self.assertEqual(words["overlay"]["reach"], "local")
        self.assertEqual([k for k, l in words.items() if l["care"]], ["device.audio-capture", "net.lan", "net.internet:api.example.com", "listen:15310"])
        # no permission shows its raw key as the only words
        for k, l in words.items():
            self.assertNotEqual(l["words"], k)
        # every permission Control knows has a group and an icon of the page's table
        for k in P.PERMISSIONS:
            self.assertIn(k, U.PERM_UI, k)
        self.assertEqual(self.cards()["quiet-demo"]["lines"], [])

    def test_consent_states_flow_through_the_card(self):
        m, _d = self.host.manifest("voice-demo")
        self.assertEqual(self.ctl("POST", "/api/plugins/host/consent", {"plugin_id": "voice-demo", "permissions": ["speech.listen", "net.lan"]})[0], 200)
        c = self.cards()["voice-demo"]
        self.assertEqual(c["consent"]["status"], "partial")
        self.assertEqual(sorted(c["consent"]["permissions"]), ["net.lan", "speech.listen"])
        self.ctl("POST", "/api/plugins/host/consent", {"plugin_id": "voice-demo", "permissions": m["permissions"]})
        self.assertEqual(self.cards()["voice-demo"]["consent"]["status"], "granted")
        self.assertEqual(self.ctl("POST", "/api/plugins/host/consent/revoke", {"plugin_id": "voice-demo"})[0], 200)
        self.assertEqual(self.cards()["voice-demo"]["consent"]["status"], "revoked")

    def test_a_new_version_asking_for_more_marks_the_new_rows(self):
        m, d = self.host.manifest("quiet-demo")
        self.host.consent.grant(m, [])
        E.make_plugin(CFG, TMP, "quiet-demo", "Quiet demo", "free", ["speech.speak"], version="0.2.0")
        c = self.cards()["quiet-demo"]
        self.assertEqual(c["version"], "0.2.0")
        self.assertEqual(c["consent"]["status"], "needs_reconsent")
        self.assertEqual([(l["key"], l["new"]) for l in c["lines"]], [("speech.speak", True)])
        # put it back for the other tests
        E.make_plugin(CFG, TMP, "quiet-demo", "Quiet demo", "free", [])

    # ---- the gate: missing / active / revoked ---------------------------------------------------------------
    def test_supporter_plugin_states_missing_active_revoked(self):
        e = self.cards()["dictate-stub"]["entitlement"]
        self.assertEqual((e["state"], e["active"], e["item_id"], e["action"]), ("missing", False, "dictate", "unlock"))
        self.assertTrue(e["cost"] and e["cost"] > 0, "the catalogue row's coins")
        self.assertEqual(e["message"], "Dictate (test stub) unlocks with PXA coins in the Store.")
        self.assertNotRegex(json.dumps(e), r"(?i)\b(paid|pay|buy|sell|price)\b")
        self.cache.add(grant("dictate"))
        e = self.cards()["dictate-stub"]["entitlement"]
        self.assertEqual((e["state"], e["active"], e["how"], e["action"]), ("active", True, "grant", "none"))
        import time
        self.cache.apply_revocations(S.sign_revocations(ENT_SEED, ENT_KID, ["0123456789abcdef0123456789abcdef"], int(time.time())))
        e = self.cards()["dictate-stub"]["entitlement"]
        self.assertEqual((e["state"], e["active"], e["action"]), ("revoked", False, "sync"))     # a signed revocation, not just "locked"
        self.assertIn("no longer unlocked", e["message"])
        self.cache.add(grant("dictate", ent="fedcba9876543210fedcba9876543210"))                 # unlocked again: a new grant clears it
        e = self.cards()["dictate-stub"]["entitlement"]
        self.assertEqual((e["state"], e["active"]), ("active", True))
        self.assertNotIn("revoked_items", {k: v for k, v in self.cache._load().items() if v})

    # ---- consent only at the computer ------------------------------------------------------------------------
    def test_consent_is_refused_from_another_device_on_lan(self):
        self.app.lan = True
        orig = H._who_local
        H._who_local = lambda: False
        try:
            st, b, _h = self.ctl("POST", "/api/plugins/host/consent", {"plugin_id": "quiet-demo", "permissions": []})
        finally:
            H._who_local = orig
        self.assertEqual(st, 403)
        self.assertEqual(json.loads(b)["code"], "consent_local_only")
        self.assertTrue(json.loads(self.ctl("GET", "/api/plugins/host/cards")[1])["lan"])

    # ---- speech on this computer ------------------------------------------------------------------------------
    def test_speech_setup_is_started_at_the_computer_itself(self):
        self.app.lan = True
        orig = H._who_local
        H._who_local = lambda: False
        try:
            st, b, _h = self.ctl("POST", "/api/plugins/host/speech/setup", {"languages": True})
        finally:
            H._who_local = orig
        self.assertEqual(st, 403)
        self.assertEqual(json.loads(b)["code"], "setup_local_only")
        st, b, _h = self.ctl("GET", "/api/plugins/host/speech")
        self.assertEqual(st, 200, b)
        self.assertIn(json.loads(b)["engines"], ("missing", "system", "ready", "update"))

    def test_speech_panel_lists_the_plugins_that_need_it(self):
        st, b, _h = self.ctl("GET", "/api/plugins/host/cards")
        sp = json.loads(b)["speech"]
        self.assertEqual(sorted(sp["needed_by"]), ["Dictate (test stub)", "Voice demo"])
        self.assertIn(sp["engines"], ("missing", "system", "ready", "update"))
        self.assertIn("english", sp["download_mb"])

    # ---- a plugin that does not verify ------------------------------------------------------------------------
    def test_tampered_plugin_is_listed_as_will_not_run(self):
        d = P.pxp.folder(CFG, "quiet-demo")
        page = os.path.join(d, "ui", "index.html")
        os.chmod(page, 0o600)
        with open(page, "a") as f:
            f.write("<!-- changed -->")
        try:
            p = self.cards()["quiet-demo"]
            self.assertFalse(p["ok"])
            self.assertEqual(p["signature"]["ok"], False)
            self.assertEqual(p["signature"]["code"], "bad_file")
            self.assertIn("will not run", p["signature"]["text"])
            self.assertEqual(p["lines"], [])
            st, b, _h = self.ctl("GET", "/api/plugins/host/frame?plugin_id=quiet-demo")
            self.assertEqual(st, 400)                                   # the page is only served from a verified folder
        finally:
            os.chmod(page, 0o600)
            E.make_plugin(CFG, TMP, "quiet-demo", "Quiet demo", "free", [])
        self.assertTrue(self.cards()["quiet-demo"]["ok"])

    # ---- the sandboxed page -------------------------------------------------------------------------------------
    def test_frame_is_an_opaque_origin_page_the_page_itself_may_frame(self):
        st, b, h = self.ctl("GET", "/api/plugins/host/frame?plugin_id=voice-demo")
        self.assertEqual(st, 200)
        self.assertTrue(h["Content-Type"].startswith("text/html"))
        csp = h["Content-Security-Policy"]
        self.assertTrue(csp.startswith("sandbox allow-scripts;"), csp)
        self.assertNotIn("allow-same-origin", csp)
        for want in ("default-src 'none'", "connect-src 'none'", "form-action 'none'", "base-uri 'none'", "frame-ancestors 'self'"):
            self.assertIn(want, csp)
        self.assertEqual(h["X-Frame-Options"], "SAMEORIGIN")
        self.assertEqual(h.get_all("X-Frame-Options"), ["SAMEORIGIN"])
        html = b.decode()
        self.assertIn("window.pxa = api", html)                          # the bridge client comes first
        self.assertLess(html.index("window.pxa = api"), html.index("pxa.onTheme"))
        self.assertIn("/api/plugins/voice-demo/echo", html)              # and the plugin's own page follows, unchanged
        # nothing else is framable: the API and the page keep DENY
        for path in ("/api/plugins/host/cards", "/api/info", "/"):
            st, _b, hh = self.ctl("GET", path)
            self.assertEqual(hh.get_all("X-Frame-Options"), ["DENY"], path)
        # a following request on the same connection style never inherits SAMEORIGIN
        st, _b, hh = self.ctl("GET", "/api/plugins/host/cards")
        self.assertEqual(hh["X-Frame-Options"], "DENY")

    def test_frame_needs_the_token_a_known_plugin_and_never_serves_other_files(self):
        self.assertEqual(self.ctl("GET", "/api/plugins/host/frame?plugin_id=voice-demo", token=False)[0], 401)
        self.assertEqual(self.ctl("GET", "/api/plugins/host/frame?plugin_id=nope-nope")[0], 404)
        self.assertEqual(self.ctl("GET", "/api/plugins/host/frame?plugin_id=host")[0], 404)
        self.assertEqual(self.ctl("GET", "/api/plugins/host/frame?plugin_id=../etc")[0], 404)
        self.assertEqual(self.ctl("GET", "/api/plugins/host/frame")[0], 404)
        self.assertEqual(self.ctl("GET", "/api/plugins/host/frame?plugin_id=voice-demo", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.ctl("GET", "/api/plugins/host/frame?plugin_id=voice-demo", headers={"Origin": "null"})[0], 403)

    def test_frame_without_a_page_is_404(self):
        m, d = self.host.manifest("quiet-demo")
        orig = U.frame_html
        # a manifest with no ui: ask the function directly with a host whose manifest has none
        class _H(object):
            def manifest(_s, pid):
                mm = dict(m)
                mm.pop("ui")
                return mm, d
        with self.assertRaises(P.PluginError) as cm:
            orig(_H(), "quiet-demo")
        self.assertEqual(cm.exception.code, "no_page")

    # ---- the page's code -------------------------------------------------------------------------------------
    def test_static_files_and_page_wiring(self):
        for path, ctype in (("/plugin-host.js", "application/javascript"), ("/plugin-host.css", "text/css")):
            st, b, h = self.ctl("GET", path)
            self.assertEqual(st, 200, path)
            self.assertTrue(h["Content-Type"].startswith(ctype))
        st, b, _h = self.ctl("GET", "/")
        page = b.decode()
        self.assertIn('src="/plugin-host.js"', page)
        self.assertIn('href="/plugin-host.css"', page)
        self.assertIn('id="ph-root"', page)
        self.assertIn('id="ph-page"', page)
        with open(os.path.join(UI_DIR, "sw.js")) as f:
            sw = f.read()
        self.assertIn('"/plugin-host.js"', sw)

    def test_bridge_code_guards(self):
        with open(os.path.join(UI_DIR, "plugin-host.js"), encoding="utf-8") as f:
            js = f.read()
        self.assertIn("ev.source !== frame.el.contentWindow", js)        # only the frame it opened
        self.assertIn('startsWith("/api/plugins/" + id + "/")', js)      # only that plugin's routes
        self.assertIn('includes("..")', js)
        self.assertIn('sandbox: "allow-scripts"', js)                    # no allow-same-origin: an opaque origin
        self.assertNotIn("allow-same-origin", js)
        self.assertNotIn("innerHTML", js)                                # data is never parsed as markup
        self.assertNotIn("eval(", js)
        for w in (r"\bpaid\b", r"\bpay\b", r"\bbuy\b", r"\bsell\b", r"\bprice\b", r"\bpurchase\b"):
            self.assertIsNone(re.search(w, js, re.I), w)                  # store wording: coins and supporter rewards
        self.assertIn("Unlock with ", js)
        with open(os.path.join(UI_DIR, "index.html"), encoding="utf-8") as f:
            self.assertIn('<dialog class="dlg" id="sd"', f.read())

    def test_plugin_tab_code(self):
        """a plugin with ui.tab.nav gets its own button + section (navSync); only from fixed glyphs, text via textContent."""
        with open(os.path.join(UI_DIR, "plugin-host.js"), encoding="utf-8") as f:
            js = f.read()
        self.assertIn("function navSync(", js)
        self.assertIn('"plug-" + ', js)
        self.assertIn("frameOf(ev.source)", js)
        self.assertNotIn("outerHTML", js)
        self.assertNotIn("insertAdjacentHTML", js)
        with open(os.path.join(UI_DIR, "index.html"), encoding="utf-8") as f:
            html = f.read()
        self.assertIn("pluginTabOpen(", html)


NODE = shutil.which("node")
PANEL_RUN = r"""
const src = require("fs").readFileSync(process.argv[1], "utf8");
const a = src.indexOf("function speechPanel(sp) {"), b = src.indexOf("\n  }\n", a);
let sent;
const el = (tag, attrs, ...kids) => ({tag, attrs: attrs || {}, kids}), chip = () => null, icon = () => null;
const speechSetup = (btn, body) => { sent = body; };
const speechPanel = eval("(" + src.slice(a, b + 4) + ")");
const out = [], walk = n => { if (!n || typeof n !== "object") return; if (n.tag === "button") { sent = undefined;
  n.attrs.onclick({currentTarget: null}); out.push([n.attrs["data-act"], n.kids.join(""), sent]); } (n.kids || []).forEach(walk); };
walk(speechPanel(JSON.parse(process.argv[2])));
console.log(JSON.stringify(out));
"""


def speech_buttons(sp):
    """plugin-host.js's speechPanel(sp) run in node: [[data-act, label, the body its click posts], ...]."""
    import subprocess
    out = subprocess.run([NODE, "-e", PANEL_RUN, os.path.join(UI_DIR, "plugin-host.js"), json.dumps(sp)], capture_output=True,
                         text=True, timeout=60)
    if out.returncode:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


@unittest.skipUnless(NODE, "node is not installed")
class SpeechButtons(unittest.TestCase):
    def sp(self, **kw):
        d = {"supported": True, "ready": True, "engines": "system", "models": {"english": True, "languages": False},
             "download_mb": {"english": 120, "languages": 500, "engines": 120, "add_languages": 620}, "job": None}
        d.update(kw)
        return d

    def test_add_languages_says_what_the_run_downloads(self):
        b = dict((x[0], x) for x in speech_buttons(self.sp()))
        self.assertEqual(b["speech-languages"][1:], ["Add other languages (about 620 MB)", {"languages": True}])


    def test_try_again_retries_the_same_kind_of_run(self):
        for langs in (True, False):
            job = {"state": "failed", "step": "models", "message": "Downloading the voice models failed.", "languages": langs}
            b = dict((x[0], x) for x in speech_buttons(self.sp(ready=False, engines="missing", job=job)))
            self.assertEqual(b["speech-setup"][1:], ["Try again", {"languages": True} if langs else {}])

class SpeechAsk(unittest.TestCase):
    """the 5 s refresh asks the speech Python for its status (a `setup --status` process in a compiled Control) only when an
    installed plugin needs speech or speech was set up here (a marker or a setup job on disk): never for nothing."""

    def setUp(self):
        from unittest import mock
        self.cfg = tempfile.mkdtemp(prefix="speech-ask-", dir=TMP)
        self.host = H.PluginHost(self.cfg, pubkeys=E.plugin_keys(), health_s=3600, tick_s=3600)
        self.asked = []
        st = {"supported": True, "engines": "ready", "ready": True, "download_mb": {"english": 0, "languages": 0}, "job": None}
        p = mock.patch.object(self.host, "speech_setup_status", lambda: self.asked.append(1) or dict(st))
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.host.close)

    def test_no_speech_plugin_and_nothing_set_up_asks_nothing(self):
        self.assertIsNone(U.cards(self.host, object())["speech"])
        self.assertEqual(self.asked, [])

    def test_a_finished_setup_is_still_shown_to_update_it(self):
        from pxa_control.speech import setup as SS
        site = SS.site_dir(self.cfg, (3, 99))
        os.makedirs(site)
        with open(os.path.join(site, SS.MARKER), "w") as f:
            json.dump({"pins": "x"}, f)
        sp = U.cards(self.host, object())["speech"]
        self.assertEqual((sp["engines"], sp["needed_by"], len(self.asked)), ("ready", [], 1))

    def test_a_setup_job_is_shown(self):
        from pxa_control.speech import setup as SS
        SS.write_job(self.cfg, {"state": "failed", "step": "models", "message": "x", "updated": 1})
        self.assertIsNotNone(U.cards(self.host, object())["speech"])
        self.assertEqual(len(self.asked), 1)


class Words(unittest.TestCase):
    # every permission Control knows, and the tag its card row must carry (from what it REALLY lets a plugin do)
    TAGS = {
        "speech.listen": "local", "speech.speak": "local", "model.chat": "local", "plugin.tools": "local", "gpu.read": "local",
        "net.loopback": "local", "device.speaker": "local", "fs.config": "local", "fs.workspace.read": "local", "fs.workspace.write": "local",
        "device.mic": "sense", "device.audio-capture": "sense", "device.camera": "sense",
        "net.lan": "out", "net.internet": "out", "net.http": "out", "home.assistant": "out", "input.type": "out",
        "fs.read": "out", "fs.write": "out", "proc.spawn": "out", "host.shell": "out", "engine.control": "out",
    }

    def test_tag_table_per_permission(self):
        self.assertEqual(sorted(self.TAGS), sorted(P.PERMISSIONS), "every permission is in the table")
        for perm, reach in self.TAGS.items():
            row = U.lines({"permissions": [perm]})[0]
            self.assertEqual(row["reach"], reach, perm)
            self.assertEqual(row["tag"], U.REACH_WORDS[reach], perm)
            self.assertEqual(row["care"], reach != "local", perm)
        # a tag never contradicts its words: "nothing leaves it" / "on this computer" rows are never tagged as reaching out
        for perm in P.PERMISSIONS:
            row = U.lines({"permissions": [perm]})[0]
            if "nothing leaves it" in row["words"] or perm.startswith("speech."):
                self.assertEqual(row["tag"], "stays on this computer", perm)
            if row["tag"] == "reaches beyond PXA":
                self.assertNotIn("nothing leaves", row["words"], perm)
        # a scoped host, a port on the home network and a port on this computer
        self.assertEqual(U.lines({"permissions": ["net.internet:example.org"]})[0]["reach"], "out")
        ls = U.lines({"permissions": ["net.lan"], "listen": [{"port": 2000, "proto": "tcp", "scope": "lan"}, {"port": 2001, "proto": "tcp", "scope": "loopback"}]})
        self.assertEqual([l["reach"] for l in ls], ["out", "out", "local"])

    def test_page_draws_the_server_tag_not_its_own(self):
        with open(os.path.join(UI_DIR, "plugin-host.js"), encoding="utf-8") as f:
            js = f.read()
        self.assertIn("l.tag", js)
        self.assertNotIn('"reaches beyond PXA"', js)          # the words live in ui.py only

    def test_toast_stacks_above_modals_and_backdrop_covers_the_viewport(self):
        with open(os.path.join(UI_DIR, "index.html"), encoding="utf-8") as f:
            page = f.read()
        self.assertIn('id="toast" role="status" aria-live="polite" popover="manual"', page)
        self.assertIn("showPopover", page)
        self.assertIn(".dlg::backdrop{position:fixed;inset:0;", page)

    def test_lines_without_a_record(self):
        m = {"permissions": ["speech.listen", "net.internet:example.org"], "listen": [{"port": 1234, "proto": "tcp", "scope": "loopback"}]}
        ls = U.lines(m)
        self.assertEqual([l["key"] for l in ls], ["speech.listen", "net.internet:example.org", "listen:1234"])
        self.assertEqual([l["ask"] for l in ls], [True, True, False])
        self.assertFalse(any(l["new"] for l in ls))

    def test_every_problem_code_has_words(self):
        for code in ("bad_signature", "unknown_key", "bad_file", "invalid_manifest"):
            self.assertIn("will not run", U.PROBLEMS[code])


UPD_RUN = r"""
const src = require("fs").readFileSync(process.argv[1], "utf8");
const grab = (head, end) => { const a = src.indexOf(head); if (a < 0) throw new Error(head + " is missing"); return src.slice(a, src.indexOf(end, a) + end.length); };
const gateRun = eval("(" + grab("function gateRun(p) {", "}\n").trim() + ")");
const canUpdate = eval("(" + grab("function canUpdate(p) {", "}\n").trim() + ")");
const updatesReady = eval("(" + grab("function updatesReady(plugins) {", "\n  }\n") + ")");
const badge = {hidden: true, textContent: "", title: ""}, sr = {textContent: ""};
const $ = sel => (sel === "#nav-plug-upd" ? badge : sel === "#nav-plug-upd-sr" ? sr : null);
const navUpdates = eval("(" + grab("function navUpdates(n) {", "\n  }\n") + ")");
const out = [];
for (const plugins of JSON.parse(process.argv[2])) {
  const n = updatesReady(plugins); navUpdates(n); out.push([n, badge.hidden, badge.textContent, badge.title, sr.textContent, plugins.map(canUpdate)]);
}
console.log(JSON.stringify(out));
"""

# reads of the cards in flight (the page opens: load(true) twice). The cards shown always come from the newest read that
# answered OK: a late answer older than them is dropped, a late OK answer newer than them is drawn even when a still newer read
# failed, and a plain load() joins the newest read while it is in flight
LOAD_RUN = r"""
const src = require("fs").readFileSync(process.argv[1], "utf8");
const grab = (head, end) => { const a = src.indexOf(head); if (a < 0) throw new Error(head + " is missing"); return src.slice(a, src.indexOf(end, a) + end.length); };
let data = null, last = "", err = null, loading = null, loadSeq = 0, doneOk = 0;
const badges = [], render = () => {}, updatesReady = ps => ps.length, navUpdates = n => badges.push(n);
const pending = [], fetch = () => new Promise(res => pending.push(res));
const answer = (i, j) => pending[i](j ? {ok: true, status: 200, json: async () => j} : {ok: false, status: 503, json: async () => ({})});
const load = eval("(" + grab("async function load(force, fresh) {", "\n  }\n") + ")");
(async () => {
  const a = load(true), b = load(true);
  answer(1, {plugins: [{plugin_id: "new"}]}); await b;
  answer(0, {plugins: [{plugin_id: "old"}, {plugin_id: "older"}]}); await a;
  const r1 = {ids: data.plugins.map(p => p.plugin_id), badges: badges.slice(), idle: loading === null};
  const c = load(true), d = load(true);                                // the older one fails late: no error over a good answer
  answer(3, {plugins: []}); await d;
  answer(2, null); await c;
  const r2 = {ids: data.plugins.map(p => p.plugin_id), err, idle: loading === null};
  const e = load(true), f = load(true);                                // the newer one fails first, the older one answers OK late:
  answer(5, null); await f;                                           //   its cards are newer than the ones shown, so they are drawn
  const failed = {ids: data.plugins.map(p => p.plugin_id), err};
  answer(4, {plugins: [{plugin_id: "late-ok"}]}); await e;
  const r3 = {failed, ids: data.plugins.map(p => p.plugin_id), err, idle: loading === null};
  const g = load(true), h = load(true);                                // the older one answers first: drawn, and the newer read is
  answer(6, {plugins: [{plugin_id: "first"}]}); await g;              //   still in flight, so a plain load() joins it
  const mid = {ids: data.plugins.map(p => p.plugin_id), busy: loading !== null};
  load(); mid.fetches = pending.length;                               // joined: no new read was started
  answer(7, {plugins: [{plugin_id: "second"}, {plugin_id: "2b"}]}); await h;
  const r4 = {mid, ids: data.plugins.map(p => p.plugin_id), badges: badges.slice(), idle: loading === null};
  console.log(JSON.stringify([r1, r2, r3, r4]));
})();
"""

# plugins.js: a page left open reads the signed catalogue again once it is 6 h old, at the next quarter-hour tick or the moment
# the page is shown again (a background tab never ran the old visible-only 6 h tick). A read that fails is tried again at the next
# quarter-hour tick (the Store keeps the old catalogue object when a read fails), and the preview's sample Store is never re-read
# (that would reset its simulated unlocks)
CAT_RUN = r"""
const src = require("fs").readFileSync(process.argv[1], "utf8");
const grab = (head, end) => { const a = src.indexOf(head); if (a < 0) throw new Error(head + " is missing"); return src.slice(a, src.indexOf(end, a) + end.length); };
const CATALOG_EVERY_MS = 6 * 3600 * 1000, CATALOG_CHECK_MS = 15 * 60 * 1000;
let catalogAt = 0, seenCatalog = null, now = 0;
const loads = [], document = {visibilityState: "visible"}, P = {source: "server", load: f => { loads.push(f); }};
Date.now = () => now;
const catalogIfOld = eval("(" + grab("function catalogIfOld() {", "\n  }\n") + ")");
const catalogSeen = eval("(" + grab("function catalogSeen(s) {", "\n  }\n") + ")");
const H = 3600 * 1000, c1 = {v: 1}, c2 = {v: 2}, c3 = {v: 3}, out = [];
const step = (label, t, vis, f) => { now = t; document.visibilityState = vis; f(); out.push([label, loads.length, catalogAt / H]); };
step("first catalogue", 0, "visible", () => catalogSeen({catalog: c1}));
step("5 h, shown", 5 * H, "visible", catalogIfOld);
step("7 h, hidden", 7 * H, "hidden", catalogIfOld);
step("30 h, shown again", 30 * H, "visible", catalogIfOld);
step("the read fails", 30.125 * H, "visible", () => catalogSeen({catalog: c1}));
step("a quarter hour later", 30.25 * H, "visible", catalogIfOld);
step("the read answers", 30.375 * H, "visible", () => catalogSeen({catalog: c2}));
step("30.5 h, shown", 30.5 * H, "visible", catalogIfOld);
step("same catalogue", 31 * H, "visible", () => catalogSeen({catalog: c2}));
step("new catalogue", 32 * H, "visible", () => catalogSeen({catalog: c3}));
step("failed read", 33 * H, "visible", () => catalogSeen({catalog: null}));
P.source = "preview";
step("preview, 40 h", 40 * H, "visible", catalogIfOld);
P.source = "server";
step("server, 40 h", 40 * H, "visible", catalogIfOld);
console.log(JSON.stringify({out, forced: loads.every(f => f === true)}));
"""


@unittest.skipUnless(NODE, "node is not installed")
class UpdateBadge(unittest.TestCase):
    """Review item 2: a new version of an installed plugin reaches the user without a release. Control reads the signed catalogue
    each time its page opens; the plugin's card then offers "Update to v...", and the Plugins tab says how many updates wait."""

    def run_js(self, prog, path, *args):
        import subprocess
        out = subprocess.run([NODE, "-e", prog, os.path.join(UI_DIR, path)] + list(args), capture_output=True, text=True, timeout=60)
        if out.returncode:
            raise AssertionError(out.stderr)
        return json.loads(out.stdout)

    def test_the_badge_counts_what_the_card_offers_to_update(self):
        upd = {"update": True, "latest": "1.1.0"}
        got = self.run_js(UPD_RUN, "plugin-host.js", json.dumps([
            [],
            [{"ok": True, "store": upd}],
            [{"ok": True, "store": upd, "entitlement": {"active": False, "how": None}}],     # locked: no Update button
            [{"ok": True, "store": {"update": False}}, {"ok": True}],
            [{"ok": False, "store": upd}],                                                   # will not run: no buttons at all
            [{"ok": True, "store": upd}, {"ok": True, "store": upd, "entitlement": {"active": True, "how": "grant"}}],
            [],
        ]))
        self.assertEqual(got, [[0, True, "", "", "", []],
                               [1, False, "1", "1 plugin update ready", ", 1 plugin update ready", [True]],
                               [0, True, "", "", "", [False]],
                               [0, True, "", "", "", [False, False]],
                               [0, True, "", "", "", [False]],
                               [2, False, "2", "2 plugin updates ready", ", 2 plugin updates ready", [True, True]],
                               [0, True, "", "", "", []]])

    def test_a_late_older_answer_does_not_overwrite_the_cards(self):
        got = self.run_js(LOAD_RUN, "plugin-host.js")
        self.assertEqual(got, [{"ids": ["new"], "badges": [1], "idle": True},
                               {"ids": [], "err": None, "idle": True},
                               {"failed": {"ids": [], "err": "HTTP 503"}, "ids": ["late-ok"], "err": None, "idle": True},
                               {"mid": {"ids": ["first"], "busy": True, "fetches": 8},
                                "ids": ["second", "2b"], "badges": [1, 0, 1, 1, 2], "idle": True}])

    def test_an_open_page_reads_the_catalogue_again_after_6_h(self):
        got = self.run_js(CAT_RUN, "plugins.js")
        self.assertEqual(got, {"out": [["first catalogue", 0, 0], ["5 h, shown", 0, 0], ["7 h, hidden", 0, 0],
                                       ["30 h, shown again", 1, 24.25], ["the read fails", 1, 24.25], ["a quarter hour later", 2, 24.5],
                                       ["the read answers", 2, 30.375], ["30.5 h, shown", 2, 30.375], ["same catalogue", 2, 30.375],
                                       ["new catalogue", 2, 32], ["failed read", 2, 32], ["preview, 40 h", 2, 32],
                                       ["server, 40 h", 3, 34.25]], "forced": True})

    def test_wiring(self):
        with open(os.path.join(UI_DIR, "index.html"), encoding="utf-8") as f:
            nav = [l for l in f.read().splitlines() if 'data-tab="plugins"' in l and "<button" in l]
        self.assertEqual(len(nav), 1, nav)
        # the number is for the eye; a screen reader hears "Plugins, 2 plugin updates ready" (a title on a span is not read)
        self.assertIn('<span class="nbadge" id="nav-plug-upd" aria-hidden="true" hidden></span><span class="sr" id="nav-plug-upd-sr"></span></button>',
                      nav[0])
        with open(os.path.join(UI_DIR, "plugin-host.js"), encoding="utf-8") as f:
            js = f.read()
        self.assertIn("navUpdates(updatesReady(data.plugins))", js)       # every cards answer sets the badge
        self.assertIn("if (canUpdate(p)) out.push(", js)                   # the card's button and the badge use one rule
        self.assertIn(".filter(canUpdate)", js)
        self.assertNotIn("st.update && gateRun(p)", js)
        self.assertRegex(js, r"PXAStore\.on\(")                            # a fresh catalogue -> the cards are asked again
        with open(os.path.join(UI_DIR, "plugins.js"), encoding="utf-8") as f:
            pj = f.read()
        self.assertIn("P.on(catalogSeen);", pj)
        self.assertIn("setInterval(catalogIfOld, CATALOG_CHECK_MS);", pj)
        self.assertIn('document.addEventListener("visibilitychange", catalogIfOld);', pj)


if __name__ == "__main__":
    unittest.main(verbosity=1)
