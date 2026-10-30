#!/usr/bin/env python3
"""PXA Control's ONE supporter gate (tools/pxa_control/plugins/gate.py) and plugin packages (plugins/pxp.py).
No network, no GPU, no loopback server: pure unit.

    python3 tests/test-pxa-plugin-gate.py

Covers: the grant states (active, refresh_due, missing, expired, revoked, wrong_machine, clock, invalid), the closed engine
library's reader cases replayed (tests/test-pxqn-feature.py, Booster P7: one gate, two readers, same answers), the wallet
statement and a wallet view never running anything, cost from the catalogue row only, unlock = redeem -> statement -> sync,
.pxp install/load (signature, listed files only, hashes on every load, the grant for supporter plugins), the install route,
and the layout rule: no supporter plugin source in the open Control tree or the package (plan S9).
"""
import json
import os
import re
import sys
import tempfile
import time
import unittest
import zipfile
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-plugin-gate-test-")
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_STORE_URL"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)

from pxa_control import plugins as P  # noqa: E402
from pxa_control import store as S  # noqa: E402
from pxa_control.plugins import gate as G  # noqa: E402
from pxa_control.plugins import pxp  # noqa: E402

BANNED = re.compile(r"\b(paid|pay|pays|sell|sold|selling|buy|bought|purchase|purchased|price|prices|priced|pricing)\b", re.I)
SEED = bytes(range(1, 33))              # the test's licd entitlement key (never a real key)
INSTALL = bytes(range(101, 133))        # this machine's install key
OTHER = bytes(range(201, 233))
NOW = 1_800_000_000
KEYS = {"t1": S.ed25519_public(SEED).hex()}


def grant(item="session-persist", kind="feature", install=INSTALL, issued=NOW - 3600, ref=NOW + 86400,
          until=NOW + 180 * 86400, kid="t1", ent="0123456789abcdef0123456789abcdef", tamper=False):
    """signed exactly as licd signs (the same builder as tests/test-pxqn-feature.py)."""
    st = {"v": 1, "iss": "licd", "kid": kid, "ent_id": ent, "kind": kind, "item_id": item, "level": None,
          "sub": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "install_pub": S.ed25519_public(install).hex(),
          "issued": issued, "refresh_after": ref, "not_after": until}
    sig = S.ed25519_sign(SEED, S.ENT_PREFIX + S.canonical(st))
    if tamper:
        st["not_after"] += 1
    return {"statement": st, "signature": S.b64u(sig)}


def world(grants, revoked=(), seen=0, install=INSTALL, now=NOW):
    """a config dir holding grants.json the way EntitlementCache writes it -> (cfg, cache with a fixed clock)."""
    cfg = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(cfg, "plugins"))
    os.makedirs(os.path.join(cfg, "store"))
    if install is not None:
        with open(os.path.join(cfg, "plugins", "install.key"), "wb") as f:
            f.write(install)
    d = {"grants": {g["statement"]["ent_id"] + str(i): g for i, g in enumerate(grants)}, "revoked": list(revoked),
         "newest": {}, "seen": seen}
    with open(os.path.join(cfg, "store", "grants.json"), "w") as f:
        json.dump(d, f)
    return cfg, S.EntitlementCache(cfg, pubkeys=KEYS, clock=lambda: now)


class ReaderCases(unittest.TestCase):
    """the closed library's reader cases (tests/test-pxqn-feature.py) on Control's gate: same grants file, same answers."""

    CASES = [
        ("valid grant", dict(grants=[grant()]), True, "active"),
        ("another feature", dict(grants=[grant(item="rig-autotune")]), False, "missing"),
        ("kind plugin", dict(grants=[grant(kind="plugin")]), False, "missing"),
        ("tampered statement", dict(grants=[grant(tamper=True)]), False, "invalid"),
        ("another machine", dict(grants=[grant(install=OTHER)]), False, "wrong_machine"),
        ("expired", dict(grants=[grant(issued=NOW - 200 * 86400, ref=NOW - 199 * 86400, until=NOW - 3600)]), False, "expired"),
        ("not yet valid", dict(grants=[grant(issued=NOW + 7200, ref=NOW + 90000, until=NOW + 100000)]), False, "invalid"),
        ("revoked", dict(grants=[grant()], revoked=["0123456789abcdef0123456789abcdef"]), False, "revoked"),
        ("clock gone back", dict(grants=[grant()], seen=NOW + 3 * 86400), False, "clock"),
        ("unknown key id", dict(grants=[grant(kid="t2")]), False, "invalid"),
        ("no install key", dict(grants=[grant()], install=None), False, "wrong_machine"),   # Control makes a new key: not this grant's
        ("valid among others", dict(grants=[grant(item="rig-autotune"), grant(tamper=True),
                                            grant(ent="fedcba9876543210fedcba9876543210")]), True, "active"),
    ]

    def test_same_answers_as_the_library_reader(self):
        for name, kw, runs, want in self.CASES:
            with self.subTest(name):
                cfg, cache = world(**kw)
                s = G.state(cfg, "session-persist", "feature", cache)
                self.assertEqual((G.run_allowed(cfg, "session-persist", "feature", cache), s["state"]), (runs, want), name)

    def test_refresh_due_still_runs_and_plugin_kind_is_separate(self):
        cfg, cache = world([grant(item="dictate", kind="plugin", issued=NOW - 2 * 86400, ref=NOW - 86400)])
        self.assertEqual(G.state(cfg, "dictate", "plugin", cache)["state"], "refresh_due")
        self.assertTrue(G.run_allowed(cfg, "dictate", "plugin", cache))
        self.assertFalse(G.run_allowed(cfg, "dictate", "feature", cache))       # a plugin grant never switches a feature on

    def test_a_hand_edited_key_never_beats_a_revoke(self):
        cfg, cache = world([grant()], revoked=["0123456789abcdef0123456789abcdef"])     # file key != the signed ent_id
        self.assertFalse(cache.has("session-persist", "feature"))

    def test_the_gate_answer_is_the_entitlement_cache_answer(self):
        for name, kw, _runs, _want in self.CASES:
            cfg, cache = world(**kw)
            self.assertEqual(G.run_allowed(cfg, "session-persist", "feature", cache), cache.has("session-persist", "feature"), name)


class PanelAndUnlock(unittest.TestCase):
    def statement(self, cfg, owned, coins):
        S.GRANT_PUBKEYS["stmt-t"] = S.ed25519_public(SEED).hex()
        st = {"v": 1, "iss": "licd", "kid": "stmt-t", "type": "wallet-statement", "wallet": "PXW-GATE-TEST", "issued": NOW,
              "coins": coins, "from_donations": coins, "earned": 0, "redeemed": 0, "reversed": 0, "adjusted": 0, "last_id": 1,
              "last_hash": "ab" * 32, "owned": list(owned)}
        S.save_statement(cfg, {"statement": st, "signature": S.b64u(S.ed25519_sign(SEED, S.STMT_PREFIX + S.canonical_ascii(st)))})

    def test_statement_and_view_inform_the_panel_and_never_run(self):
        cfg, cache = world([])
        self.statement(cfg, ["dictate"], 7)
        p = G.panel(cfg, "dictate", "plugin", cache, name="Dictate")
        self.assertEqual((p["active"], p["how"], p["have"], p["state"]), (True, "statement", 7, "missing"))
        self.assertFalse(G.run_allowed(cfg, "dictate", "plugin", cache))
        with self.assertRaises(G.Locked) as e:
            G.require(cfg, "dictate", "plugin", cache, name="Dictate")
        self.assertEqual((e.exception.status, e.exception.code), (403, "locked"))
        cfg2, cache2 = world([])                                                 # a statement copied to a second computer
        with open(S._statement_path(cfg)) as f, open(S._statement_path(cfg2), "w") as g:
            g.write(f.read())
        self.assertFalse(G.run_allowed(cfg2, "dictate", "plugin", cache2))
        view = {"coins": 0, "owned": [{"item_id": "dictate", "status": "active"}]}
        cfg3, cache3 = world([])
        self.assertEqual(G.panel(cfg3, "dictate", "plugin", cache3, view=view)["how"], "wallet")
        self.assertFalse(G.run_allowed(cfg3, "dictate", "plugin", cache3))
        view["owned"][0]["status"] = "revoked"
        self.assertFalse(G.panel(cfg3, "dictate", "plugin", cache3, view=view)["active"])

    def test_words_are_coins_and_supporter_rewards(self):
        for k, msg in G.MESSAGES.items():
            self.assertIsNone(BANNED.search(msg), k)
            self.assertNotRegex(msg, r"\d")                                     # no amounts in Control's words
        cfg, cache = world([])
        p = G.panel(cfg, "dictate", "plugin", cache, name="Dictate")
        self.assertIn("coins", p["message"])
        self.assertFalse(p["active"])

    def test_cost_comes_from_the_catalogue_row_only(self):
        row = G.catalog_row(S.catalog_view(), "dictate")
        self.assertEqual((row["kind"], G.cost(row)), ("plugin", 100))
        self.assertEqual(G.cost({"id": "x", "coins": 250}), 250)
        self.assertEqual(G.cost({"id": "x", S.CATALOG_COST_KEY: 50}), 50)
        self.assertIsNone(G.cost({"id": "x", "coins": True}))
        self.assertIsNone(G.cost(None))
        self.assertIsNone(G.catalog_row({"items": []}, "dictate"))
        with open(os.path.join(TOOLS, "pxa_control", "plugins", "gate.py"), encoding="utf-8") as f:
            self.assertNotRegex(f.read(), r"\b\d+ coins\b")

    def test_unlock_is_redeem_then_statement_then_sync(self):
        cfg, cache = world([])
        calls = []

        class Client(object):
            def wallet(self, action, args=None):
                calls.append(("wallet", action, dict(args or {})))
                if action == "redeem":
                    return {"item_id": args["item_id"], "redeemed": True, "coins_used": 100}
                raise RuntimeError("statement read failed")                   # the wallet answer stands

            def sync(self, c):
                calls.append(("sync",))
                return {"grants": 1, "refused": []}
        out = G.unlock(Client(), cfg, "dictate", cache)
        self.assertEqual([c[:2] for c in calls], [("wallet", "redeem"), ("wallet", "statement"), ("sync",)])
        self.assertEqual((out["redeemed"], out["synced"]["grants"]), (True, 1))

        class Short(Client):
            def wallet(self, action, args=None):
                e = RuntimeError("not enough coins")
                e.code = "not_enough_coins"
                raise e
        calls.clear()
        with self.assertRaises(RuntimeError) as e:
            G.unlock(Short(), cfg, "dictate", cache)
        self.assertEqual((e.exception.code, calls), ("not_enough_coins", []))


# ---- a synthetic supporter plugin (the real ones live in the private tree; this test must not need them) ------------
PLUG_SEED = bytes(range(41, 73))
PLUG_KEYS = {"plug-t": S.ed25519_public(PLUG_SEED).hex()}
FILES = {
    "bin/echo": b"#!/usr/bin/env python3\nprint('echo')\n",
    "lib/echoplug/__init__.py": b"from .core import hello\nNAME = 'echo'\n",
    "lib/echoplug/core.py": b"def hello():\n    return 'hello from the package'\n",
    "ui/index.html": b"<!doctype html><title>Echo</title><p>A supporter reward.</p>\n",
}


def build(out, edition="supporter", entitlement="echoplug", files=None, extra=None, seed=PLUG_SEED, kid="plug-t", mutate=None):
    files = dict(FILES if files is None else files)
    body = {"schema": 1, "plugin_id": "echoplug", "name": "Echo", "version": "0.1.0", "publisher": "pxa", "edition": edition,
            "min_host": "4.0", "entry": {"kind": "process", "argv": ["bin/echo"], "protocol": "pxa-plugin/1"},
            "routes": ["/api/plugins/echoplug/"], "ui": {"tab": {"title": "Echo", "page": "ui/index.html"}},
            "permissions": ["model.chat"], "entitlement": entitlement, "python": {"package": "echoplug", "path": "lib"},
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    body["signature"] = {"kid": kid, "sig": S.b64u(S.ed25519_sign(seed, P.MANIFEST_PREFIX + P.canonical(body)))}
    if mutate:
        mutate(body, files)
    path = os.path.join(out, "echoplug-%d.pxp" % len(os.listdir(out)))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("plugin.json", json.dumps(body))
        for k, v in files.items():
            z.writestr(k, v)
        for k, v in (extra or {}).items():
            if isinstance(v, zipfile.ZipInfo):
                z.writestr(v, b"x")
            else:
                z.writestr(k, v)
    return path


class CoinsWording(unittest.TestCase):
    """Review item 2 (10-24): every Store item unlocks with PXA coins (the 4.0 beta is the one exception, with the
    Valued Supporter role); a one-off donation gives coins and no role, so a Store item is never called a supporter reward."""
    BANNED = r"(?i)\b(paid|pay|sell|sells|sold|buy|bought|price|prices|priced|purchase\w*)\b"

    def test_missing_says_coins(self):
        m = G.MESSAGES["missing"].format(name="Dictate")
        self.assertEqual(m, "Dictate unlocks with PXA coins in the Store.")
        for v in G.MESSAGES.values():
            self.assertNotRegex(v, self.BANNED)
            self.assertNotIn("\u2014", v)
            self.assertNotIn("supporter reward", v.lower())

    def test_catalogue_fixture_says_coins(self):
        for row in S.CATALOG_FIXTURE:
            words = " ".join(str(row.get(k, "")) for k in ("name", "tagline", "description"))
            self.assertNotIn("supporter reward", words.lower(), row["id"])
            self.assertNotRegex(words, self.BANNED, row["id"])
            self.assertNotIn("\u2014", words, row["id"])
        d = [r for r in S.CATALOG_FIXTURE if r["id"] == "dictate"][0]["description"]
        self.assertTrue(d.startswith("Unlocks with PXA coins."), d)

    def test_host_ui_says_coins(self):
        ui = os.path.join(TOOLS, "pxa_control_ui")
        for f in ("plugin-host.js", "plugins.js"):
            with open(os.path.join(ui, f), encoding="utf-8") as fh:
                text = fh.read()
            code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))
            self.assertNotIn("supporter reward", code.lower(), f)

    def test_wallet_empty_redeem_grid_says_store_items(self):
        with open(os.path.join(TOOLS, "pxa_control_ui", "wallet.js"), encoding="utf-8") as fh:
            code = fh.read()
        self.assertIn("Store items appear here once the Store catalogue has loaded.", code)
        self.assertNotIn("Supporter rewards appear here", code)


class Packages(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(dir=TMP)

    def test_install_and_load_on_the_grant(self):
        path = build(self.out)
        cfg, cache = world([])
        with self.assertRaises(P.PluginError) as e:
            pxp.install(path, cfg)                                              # not a release key
        self.assertEqual(e.exception.code, "unknown_key")
        info = pxp.install(path, cfg, PLUG_KEYS)
        self.assertEqual((info["plugin_id"], info["edition"], info["entitlement"]), ("echoplug", "supporter", "echoplug"))
        self.assertEqual(oct(os.stat(os.path.join(info["dir"], "bin", "echo")).st_mode & 0o777), "0o700")
        with self.assertRaises(G.Locked):
            pxp.load(cfg, "echoplug", PLUG_KEYS, cache)
        self.assertEqual(pxp.installed(cfg, PLUG_KEYS, cache)[0]["gate"], "missing")
        cfg2, cache2 = world([grant(item="echoplug", kind="plugin")])
        pxp.install(path, cfg2, PLUG_KEYS)
        mod = pxp.load(cfg2, "echoplug", PLUG_KEYS, cache2)
        self.assertEqual((mod.hello(), mod.NAME, mod.PXA_PLUGIN_MANIFEST["version"]), ("hello from the package", "echo", "0.1.0"))
        with open(os.path.join(pxp.folder(cfg2, "echoplug"), "lib", "echoplug", "added.py"), "w") as f:
            f.write("x = 1\n")
        with self.assertRaises(P.PluginError) as e:
            pxp.load(cfg2, "echoplug", PLUG_KEYS, cache2)                       # an unlisted file in the package never loads
        self.assertEqual(e.exception.code, "bad_file")
        cur = os.path.join(cfg2, "plugins", "installed", "echoplug", "current")
        with open(cur, "w") as f:
            f.write("..\n")
        self.assertIsNone(pxp.folder(cfg2, "echoplug"))                         # a version, never a path
        pxp.install(path, cfg2, PLUG_KEYS)                                      # reinstall the same version: clean again
        self.assertEqual(pxp.load(cfg2, "echoplug", PLUG_KEYS, cache2).hello(), "hello from the package")

    def test_free_plugin_needs_no_grant(self):
        cfg, cache = world([])
        pxp.install(build(self.out, edition="free", entitlement=None), cfg, PLUG_KEYS)
        self.assertEqual(pxp.load(cfg, "echoplug", PLUG_KEYS, cache).hello(), "hello from the package")

    def test_refused_packages_leave_nothing_installed(self):
        link = zipfile.ZipInfo("lib/echoplug/link.py")
        link.external_attr = (0o120777 << 16)
        bad = {
            "extra file": build(self.out, extra={"lib/echoplug/evil.py": b"x = 1\n"}),
            "parent path": build(self.out, extra={"../evil.py": b"x = 1\n"}),
            "a link": build(self.out, extra={"lib/echoplug/link.py": link}),
            "changed file": build(self.out, mutate=lambda b, f: f.update({"lib/echoplug/core.py": b"def hello():\n    return 1\n"})),
            "missing file": build(self.out, mutate=lambda b, f: f.pop("ui/index.html")),
            "re-signed by another key": build(self.out, seed=OTHER),
            "supporter without an item": build(self.out, entitlement=None),
        }
        for name, path in bad.items():
            with self.subTest(name):
                cfg, _cache = world([])
                with self.assertRaises(P.PluginError):
                    pxp.install(path, cfg, PLUG_KEYS)
                self.assertIsNone(pxp.folder(cfg, "echoplug"), name)
                left = os.listdir(os.path.join(cfg, "plugins", "installed", "echoplug")) \
                    if os.path.isdir(os.path.join(cfg, "plugins", "installed", "echoplug")) else []
                self.assertEqual(left, [], name)                               # no staging folder left behind
        with self.assertRaises(P.PluginError):
            pxp.read_manifest(os.path.join(TOOLS, "pxa_control", "plugins", "gate.py"))

    def test_manifest_python_field(self):
        m = json.loads(zipfile.ZipFile(build(self.out)).read("plugin.json"))
        self.assertEqual(P.validate_manifest(m), [])
        m["python"] = {"package": "Echo-Plug", "path": "lib"}
        self.assertTrue(P.validate_manifest(m))
        m["python"] = {"package": "other", "path": "lib"}
        self.assertTrue(any("__init__" in x for x in P.validate_manifest(m)))


class InstallRoute(unittest.TestCase):
    def test_route_needs_the_grant_and_installs_the_pxp(self):
        out = tempfile.mkdtemp(dir=TMP)
        path = build(out)
        cfg, cache = world([])
        routes = {}
        old_url, old_cl, old_keys = S.store_url, S.LicdClient, dict(P.PLUGIN_PUBKEYS)
        got = []

        class FakeClient(S.LicdClient):
            def __init__(self, *a, **k):
                pass

            def catalog(self):                     # a licd from before the bundle block: the manifest path below
                return {"items": [{"id": "echoplug", "kind": "plugin"}]}

            def manifest(self, item):
                return {"files": [{"file_id": "other.gguf", "name": "other.gguf"}, {"file_id": "echoplug-0.1.0.pxp", "name": "echoplug-0.1.0.pxp"}]}

            def download(self, item, file_id, dest_dir, progress=None):
                got.append((item, file_id))
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, file_id)
                with open(path, "rb") as a, open(dest, "wb") as b:
                    b.write(a.read())
                return {"path": dest}
        try:
            S.store_url = lambda: "http://store.invalid"
            S.LicdClient = FakeClient
            P.PLUGIN_PUBKEYS.update(PLUG_KEYS)
            S.register(routes, lambda: cfg, KEYS)

            class App(object):
                pass
            app = App()
            app.store_cache = cache
            with self.assertRaises(Exception) as e:
                routes[("POST", "/api/plugins/install")](app, {"item_id": "echoplug"}, {})
            self.assertEqual((getattr(e.exception, "status", 0), getattr(e.exception, "code", "")), (403, "locked"))
            self.assertEqual(got, [])                                           # no grant: not even a download
            cfg2, cache2 = world([grant(item="echoplug", kind="plugin")])
            routes2 = {}
            S.register(routes2, lambda: cfg2, KEYS)
            app.store_cache = cache2
            r = routes2[("POST", "/api/plugins/install")](app, {"item_id": "echoplug"}, {})
            self.assertEqual((r["installed"]["plugin_id"], got), ("echoplug", [("echoplug", "echoplug-0.1.0.pxp")]))
            self.assertEqual(os.listdir(os.path.join(cfg2, "plugins", "incoming")), [])   # the download is not kept
            rows = routes2[("GET", "/api/plugins/installed")](app, None, {})["plugins"]
            self.assertEqual([(x["plugin_id"], x["gate"]) for x in rows], [("echoplug", "active")])
        finally:
            S.store_url, S.LicdClient = old_url, old_cl
            P.PLUGIN_PUBKEYS.clear()
            P.PLUGIN_PUBKEYS.update(old_keys)


class Layout(unittest.TestCase):
    """plan S9: supporter plugin sources stay out of the open Control tree and the package."""

    def test_no_supporter_plugin_source_in_the_open_control_tree(self):
        base = os.path.join(TOOLS, "pxa_control")
        for dirpath, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for n in names:
                p = os.path.join(dirpath, n)
                self.assertFalse(n == "plugin.src.json" or n.endswith(".pxp"), p)
                if n == "plugin.json":
                    with open(p, encoding="utf-8") as f:
                        self.assertNotEqual(json.load(f).get("edition"), "supporter", p)
        sub = [d for d in os.listdir(os.path.join(base, "plugins")) if os.path.isdir(os.path.join(base, "plugins", d)) and d != "__pycache__"]
        self.assertEqual(sub, [], "plugin code lives in the private tree, not under tools/pxa_control/plugins")

    def test_private_tree_is_closed_and_the_packager_refuses_leaks(self):
        with open(os.path.join(ROOT, "scripts", "pxa-closed-files.txt")) as f:
            self.assertIn("tools/pxa-private", [x.strip() for x in f])
        with open(os.path.join(ROOT, "scripts", "make-release-tarball.sh")) as f:
            sh = f.read()
        self.assertIn('plugin_leak_guard "$STAGE"', sh)
        for word in ("plugin.src.json", "plugin.json", "*.pxp", "tools/pxa-private", "tools/anibal", "tools/vigil", "tools/pxa_plugins"):
            self.assertIn(word, sh)
        body = sh[sh.index("plugin_leak_guard() {"):]
        body = body[:body.index("\n}\n") + 3]
        import subprocess
        import tempfile
        import zipfile
        for name, rel in (("clean", None), ("pxp", "share/a.pxp"), ("manifest", "plugins/x/plugin.json"),
                          ("anibal", "tools/anibal/core.py"), ("vigil", "tools/vigil/a.py"), ("framework", "tools/pxa_plugins/a.py"),
                          ("package", "lib/py/anibal/__init__.py"), ("renamed", "bin/data.bin"),
                          # any case (review F9): a stray manifest or an Anibal folder never slips by its spelling
                          ("manifest-case", "plugins/x/Plugin.json"), ("pxp-case", "share/ANIBAL.PXP"), ("anibal-case", "tools/Anibal/core.py"),
                          ("package-case", "lib/py/Anibal/__init__.py"), ("renamed-case", "bin/data2.bin")):
            with self.subTest(name), tempfile.TemporaryDirectory() as st:
                os.makedirs(os.path.join(st, "tools", "pxa_control"))
                with open(os.path.join(st, "tools", "pxa_control", "core.py"), "w") as f:
                    f.write("x = 'anibal/voice'\n")
                if rel:
                    os.makedirs(os.path.dirname(os.path.join(st, rel)), exist_ok=True)
                    if name.startswith("renamed"):
                        with zipfile.ZipFile(os.path.join(st, rel), "w") as z:
                            z.writestr("Plugin.JSON" if name.endswith("case") else "plugin.json", "{}")
                    else:
                        open(os.path.join(st, rel), "w").write("x\n")
                r = subprocess.run(["bash", "-c", 'die() { echo "DIE: $*"; exit 7; }\n' + body + '\nplugin_leak_guard "$1"', "g", st],
                                   capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0 if rel is None else 7, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=1)
