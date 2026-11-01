#!/usr/bin/env python3
"""4.0.1 key rotation 2026-10b: every client-side verifier trusts the OLD and the NEW kid, and for the two keys that sign code a
client LOADS (package key: encoder packages, GPU runtime, engine/library updates, beta file lists; plugin-publisher key: plugins)
the OLD key is trusted ONLY for the artifacts already published under it (a pinned sha256 list); anything new needs the new kid.

  (a) a blob signed with the new kid verifies        (b) old-kid data / pinned artifacts still verify
  (c) an old-kid signature over anything NOT pinned is REJECTED (code retired_key)       (d) an unknown kid is rejected
plus the shipped lists themselves: the exact public keys of the rotation, the old kid's pins bound to the old PUBLIC key.
Fake keys from fixed seeds; no network. Optional: PXA_TEST_PXP_DIR=<folder of the published .pxp files> checks every one of them
still installs under the shipped pubkeys.json (the pins match the real artifacts)."""
import copy
import glob
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
os.environ.pop("PXA_PACKAGE_PUBKEY", None)

import encode_fakes as F                     # noqa: E402  (pure-python Ed25519 signer the package tests use)
import pxa_encode_pkg as PK                  # noqa: E402
import pxa_engine_update as EU               # noqa: E402
import pxa_lib_update as LU                  # noqa: E402
from pxa_control import plugins as P         # noqa: E402
from pxa_control import store as S           # noqa: E402

# the rotation's public keys (public halves only)
ROT = {
    "pkg-2026-10": "de9a21ae8508a373452f422fe110ea08e68ac72569019cb2214442facfdf7af4",
    "pkg-2026-10b": "3e0216325217b7cf25f642ac68af6531af61f046c17b9cda4b456272fe9acbea",
    "ent-2026-10": "58061ebc1e9ef1c68c4021df250eb1297dc9c5082a2cc04752a246e917f0a15f",
    "ent-2026-10b": "121e0e8bee095cbf25a48a6edac9fa285fca0bd84c3d1748158cf24c8a4f1d41",
    "plug-2026-10": "b2c3d31d1d635e63170891469ff336fc89928349232271d4b49c1d683f248f71",
    "plug-2026-10b": "bc59b747eea79b5f43c0147902057c92c9e048c12994748f5fc133093629e97b",
}
OLD_SEED, NEW_SEED, STRANGER_SEED = (hashlib.sha256(b"dual-trust-" + n).digest() for n in (b"old", b"new", b"stranger"))
OLD_HEX, NEW_HEX = F.ed_pub(OLD_SEED).hex(), F.ed_pub(NEW_SEED).hex()
PINNED_SHA = hashlib.sha256(b"an artifact published under the old kid").hexdigest()
FRESH_SHA = hashlib.sha256(b"an artifact made after the rotation").hexdigest()
b64u = F._b64u


class _Keys(object):
    """swap the shipped package keys for the test pair: pkg-old (retired, pins PINNED_SHA) + pkg-new"""

    def setUp(self):
        self.saved = (PK.PACKAGE_PUBKEYS, PK.PACKAGE_PINNED, os.environ.pop("PXA_PACKAGE_PUBKEY", None))
        PK.PACKAGE_PUBKEYS = {"pkg-old": OLD_HEX, "pkg-new": NEW_HEX}
        PK.PACKAGE_PINNED = {OLD_HEX: frozenset((PINNED_SHA,))}

    def tearDown(self):
        PK.PACKAGE_PUBKEYS, PK.PACKAGE_PINNED, env = self.saved
        os.environ.pop("PXA_PACKAGE_PUBKEY", None)
        if env is not None:
            os.environ["PXA_PACKAGE_PUBKEY"] = env

    def code(self, fn, *a):
        try:
            fn(*a)
        except PK.PackageError as e:
            return getattr(e, "code", "?")
        return "ok"


def pkg_answer(seed, kid, sha, edition="free", build="fre20261030a"):
    st = json.dumps({"build_id": build, "cuda_major": 12, "edition": edition, "kid": kid, "pkg": "", "platform": "linux-x86_64",
                     "sha256": sha, "size": 1234}, sort_keys=True, separators=(",", ":"))
    return {"build_id": build, "edition": edition, "statement": st, "signature": b64u(F.ed_sign(seed, PK.SIGN_PREFIX + st.encode())),
            "kid": kid, "url": "https://lic.pxanetwork.com/v1/free/download/" + build}


class PackageStatements(_Keys, unittest.TestCase):
    def check(self, seed, kid, sha):
        return self.code(PK.check_statement, pkg_answer(seed, kid, sha), "free", "linux-x86_64", 12)

    def test_a_new_kid_verifies_anything_it_signs(self):
        self.assertEqual(self.check(NEW_SEED, "pkg-new", FRESH_SHA), "ok")
        self.assertEqual(self.check(NEW_SEED, "pkg-new", PINNED_SHA), "ok")

    def test_b_old_kid_pinned_artifact_verifies(self):
        self.assertEqual(self.check(OLD_SEED, "pkg-old", PINNED_SHA), "ok")

    def test_c_old_kid_unpinned_artifact_rejected(self):
        self.assertEqual(self.check(OLD_SEED, "pkg-old", FRESH_SHA), "retired_key")

    def test_c_old_key_under_another_name_or_env_is_still_pinned(self):
        PK.PACKAGE_PUBKEYS["pkg-alias"] = OLD_HEX                       # the pin is bound to the key, not the kid name
        self.assertEqual(self.check(OLD_SEED, "pkg-alias", FRESH_SHA), "retired_key")
        os.environ["PXA_PACKAGE_PUBKEY"] = OLD_HEX                      # the test override cannot lift it either
        self.assertEqual(self.check(OLD_SEED, "anything", FRESH_SHA), "retired_key")
        self.assertEqual(self.check(OLD_SEED, "anything", PINNED_SHA), "ok")

    def test_d_unknown_kid_rejected(self):
        self.assertEqual(self.check(STRANGER_SEED, "pkg-2099-01", FRESH_SHA), "no_pubkey")
        self.assertEqual(self.check(OLD_SEED, "pkg-2099-01", PINNED_SHA), "no_pubkey")

    def test_kid_picks_the_key(self):
        self.assertEqual(self.check(OLD_SEED, "pkg-new", FRESH_SHA), "bad_signature")   # old key's signature under the new kid
        self.assertEqual(self.check(NEW_SEED, "pkg-old", PINNED_SHA), "bad_signature")
        self.assertEqual(self.check(STRANGER_SEED, "pkg-new", FRESH_SHA), "bad_signature")

    def test_verify_package_rechecks_the_pin_on_the_signed_sha(self):
        with tempfile.TemporaryDirectory() as d:
            blob = b"x" * 1234
            p = os.path.join(d, "pkg.tar.gz")
            with open(p, "wb") as f:
                f.write(blob)
            sha = hashlib.sha256(blob).hexdigest()
            m = PK.check_statement(pkg_answer(NEW_SEED, "pkg-new", sha), "free", "linux-x86_64", 12)
            self.assertEqual(self.code(PK.verify_package, p, m), "ok")
            forged = dict(pkg_answer(OLD_SEED, "pkg-old", sha))            # the file's real sha, old key: not pinned
            mm = dict(m, statement=forged["statement"], signature=forged["signature"], kid="pkg-old")
            self.assertEqual(self.code(PK.verify_package, p, mm), "retired_key")
            pinned = PK.check_statement(pkg_answer(OLD_SEED, "pkg-old", PINNED_SHA), "free", "linux-x86_64", 12)
            swapped = dict(pinned, sha256=sha)                              # a local manifest whose sha differs from the signed one
            self.assertEqual(self.code(PK.verify_package, p, swapped), "bad_manifest")

    def test_pinned_ok_rules(self):
        self.assertTrue(PK.pinned_ok(bytes.fromhex(NEW_HEX), []))           # not retired: no list involved
        self.assertFalse(PK.pinned_ok(bytes.fromhex(OLD_HEX), []))          # retired: an empty list never passes
        self.assertFalse(PK.pinned_ok(bytes.fromhex(OLD_HEX), [PINNED_SHA, FRESH_SHA]))   # every artifact must be pinned
        self.assertTrue(PK.pinned_ok(bytes.fromhex(OLD_HEX), [PINNED_SHA.upper()]))


def engine_answer(seed, kid, tar_sha, notes_sha):
    st = json.dumps({"edition": "engine", "channel": "beta", "platform": "linux-x86_64", "cuda_major": 12, "build_id": "e4.0.1-test",
                     "version": "4.0.1-beta.1", "kid": kid,
                     "files": [{"name": "pxa-v4.0.1-linux.tar.gz", "role": "tarball", "sha256": tar_sha, "size": 1000},
                               {"name": "NOTES.md", "role": "notes", "sha256": notes_sha, "size": 10}]}, sort_keys=True)
    return {"manifest": st, "signature": b64u(F.ed_sign(seed, EU.SIGN_PREFIX + st.encode())), "kid": kid}


def beta_files_answer(seed, kid, shas):
    roles = ("tarball", "notes", "docker")
    st = json.dumps({"type": "beta-files", "build_id": "e4.0.1-test", "version": "4.0.1-beta.1", "kid": kid,
                     "files": [{"name": "f%d%s" % (i, ".tar.gz" if r != "notes" else ".md"), "role": r, "sha256": s, "size": 5}
                               for i, (r, s) in enumerate(zip(roles, shas))]}, sort_keys=True)
    return {"manifest": st, "signature": b64u(F.ed_sign(seed, EU.BETA_FILES_PREFIX + st.encode())), "kid": kid, "files": []}


def lib_answer(seed, kid, shas):
    st = json.dumps({"edition": "lib", "lib": LU.LIB_NAME, "channel": "stable", "platform": "linux-x86_64", "cuda_major": 12,
                     "lib_id": "l4.0.1", "version": "4.0.1", "kid": kid,
                     "files": [{"name": "lib/f%d.so" % i, "sha256": s, "size": 5} for i, s in enumerate(shas)]}, sort_keys=True)
    return {"manifest": st, "signature": b64u(F.ed_sign(seed, LU.SIGN_PREFIX + st.encode())), "kid": kid}


class EngineAndLibraryUpdates(_Keys, unittest.TestCase):
    BASE = "https://lic.pxanetwork.com"

    def rel(self, o):
        return self.code(EU.verify_release, o, self.BASE, "beta", "linux-x86_64", 12)

    def test_engine_release(self):
        self.assertEqual(self.rel(engine_answer(NEW_SEED, "pkg-new", FRESH_SHA, FRESH_SHA)), "ok")                 # (a)
        PK.PACKAGE_PINNED = {OLD_HEX: frozenset((PINNED_SHA, FRESH_SHA))}
        self.assertEqual(self.rel(engine_answer(OLD_SEED, "pkg-old", PINNED_SHA, FRESH_SHA)), "ok")               # (b)
        PK.PACKAGE_PINNED = {OLD_HEX: frozenset((PINNED_SHA,))}
        self.assertEqual(self.rel(engine_answer(OLD_SEED, "pkg-old", PINNED_SHA, FRESH_SHA)), "retired_key")      # (c) notes not pinned
        self.assertEqual(self.rel(engine_answer(OLD_SEED, "pkg-old", FRESH_SHA, PINNED_SHA)), "retired_key")      # (c) tarball not pinned
        self.assertEqual(self.rel(engine_answer(STRANGER_SEED, "pkg-x", FRESH_SHA, FRESH_SHA)), "no_pubkey")     # (d)

    def test_beta_file_list(self):
        v = lambda o: self.code(EU.verify_beta_files, o)
        self.assertEqual(v(beta_files_answer(NEW_SEED, "pkg-new", [FRESH_SHA] * 3)), "ok")
        self.assertEqual(v(beta_files_answer(OLD_SEED, "pkg-old", [PINNED_SHA] * 3)), "ok")
        self.assertEqual(v(beta_files_answer(OLD_SEED, "pkg-old", [PINNED_SHA, PINNED_SHA, FRESH_SHA])), "retired_key")
        self.assertEqual(v(beta_files_answer(STRANGER_SEED, "pkg-x", [FRESH_SHA] * 3)), "bad_signature")   # unknown kid: no key

    def test_library_release(self):
        v = lambda o: self.code(LU.verify_manifest, o, self.BASE, "stable", "linux-x86_64", 12)
        self.assertEqual(v(lib_answer(NEW_SEED, "pkg-new", [FRESH_SHA, PINNED_SHA])), "ok")
        self.assertEqual(v(lib_answer(OLD_SEED, "pkg-old", [PINNED_SHA])), "ok")
        self.assertEqual(v(lib_answer(OLD_SEED, "pkg-old", [PINNED_SHA, FRESH_SHA])), "retired_key")
        self.assertEqual(v(lib_answer(STRANGER_SEED, "pkg-x", [FRESH_SHA])), "no_pubkey")


def manifest(seed, kid, version="0.1.0"):
    body = {"schema": 1, "plugin_id": "dualtest", "name": "Dual", "version": version, "publisher": "pxa", "edition": "free",
            "min_host": "4.0", "entry": {"kind": "process", "argv": ["bin/x"], "protocol": "pxa-plugin/1"},
            "routes": ["/api/plugins/dualtest/"], "permissions": ["net.loopback"], "entitlement": None,
            "files": {"bin/x": hashlib.sha256(b"x").hexdigest()}}
    m = dict(body, signature={"kid": kid, "sig": b64u(F.ed_sign(seed, P.MANIFEST_PREFIX + P.canonical(body)))})
    return m


class Plugins(unittest.TestCase):
    def setUp(self):
        self.saved = (dict(P.PLUGIN_PUBKEYS), copy.deepcopy(P.PLUGIN_PINNED), P.RETIRED_PLUGIN_KEYS)
        P.PLUGIN_PUBKEYS.clear()
        P.PLUGIN_PUBKEYS.update({"plug-old": OLD_HEX, "plug-new": NEW_HEX})
        P.PLUGIN_PINNED.clear()
        P.RETIRED_PLUGIN_KEYS = frozenset((OLD_HEX,))
        self.published = manifest(OLD_SEED, "plug-old", "0.1.0")
        P.PLUGIN_PINNED[OLD_HEX] = {P.manifest_digest(self.published)}

    def tearDown(self):
        P.PLUGIN_PUBKEYS.clear()
        P.PLUGIN_PUBKEYS.update(self.saved[0])
        P.PLUGIN_PINNED.clear()
        P.PLUGIN_PINNED.update(self.saved[1])
        P.RETIRED_PLUGIN_KEYS = self.saved[2]

    def code(self, m, pubkeys=None):
        try:
            P.verify_manifest(m, pubkeys)
        except P.PluginError as e:
            return getattr(e, "code", "?")
        return "ok"

    def test_a_new_kid(self):
        self.assertEqual(self.code(manifest(NEW_SEED, "plug-new", "0.2.0")), "ok")

    def test_b_old_kid_published_plugin(self):
        self.assertEqual(self.code(self.published), "ok")

    def test_c_old_kid_new_plugin_rejected(self):
        self.assertEqual(self.code(manifest(OLD_SEED, "plug-old", "0.1.1")), "retired_key")
        changed = copy.deepcopy(self.published)          # a pinned plugin's manifest with one file swapped, re-signed with the old seed
        changed["files"]["bin/x"] = hashlib.sha256(b"evil").hexdigest()
        body = {k: v for k, v in changed.items() if k != "signature"}
        changed["signature"]["sig"] = b64u(F.ed_sign(OLD_SEED, P.MANIFEST_PREFIX + P.canonical(body)))
        self.assertEqual(self.code(changed), "retired_key")
        self.assertEqual(self.code(manifest(OLD_SEED, "plug-alias", "0.1.1"), {"plug-alias": OLD_HEX}), "retired_key")  # bound to the key

    def test_c_retired_key_with_no_list_verifies_nothing(self):
        P.PLUGIN_PINNED.clear()
        self.assertEqual(self.code(self.published), "retired_key")

    def test_d_unknown_kid(self):
        self.assertEqual(self.code(manifest(STRANGER_SEED, "plug-2099-01")), "unknown_key")
        self.assertEqual(self.code(manifest(OLD_SEED, "plug-2099-01")), "unknown_key")

    def test_kid_picks_the_key(self):
        self.assertEqual(self.code(manifest(OLD_SEED, "plug-new")), "bad_signature")

    def test_load_pubkeys_fills_the_pins_by_public_key(self):
        P.PLUGIN_PUBKEYS.clear()
        P.PLUGIN_PINNED.clear()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "pubkeys.json")
            with open(p, "w") as f:
                json.dump({"GRANT_PUBKEYS": {}, "PLUGIN_PUBKEYS": {"plug-old": OLD_HEX, "plug-new": NEW_HEX},
                           "PLUGIN_PINNED": {"plug-old": {P.manifest_digest(self.published): "dualtest 0.1.0"}, "plug-gone": ["00" * 32]}},
                          f)
            S.load_pubkeys(p)
        self.assertEqual(P.PLUGIN_PINNED, {OLD_HEX: {P.manifest_digest(self.published)}})   # a kid with no key adds nothing
        self.assertEqual(self.code(self.published), "ok")
        self.assertEqual(self.code(manifest(OLD_SEED, "plug-old", "0.1.1")), "retired_key")
        self.assertEqual(self.code(manifest(NEW_SEED, "plug-new", "0.1.1")), "ok")


class ShippedLists(unittest.TestCase):
    """what 4.0.1 actually ships"""

    def setUp(self):
        with open(os.path.join(TOOLS, "pxa_control", "store", "pubkeys.json")) as f:
            self.pub = json.load(f)

    def test_package_keys(self):
        pk = PK                                                                    # the other classes restore what they swap
        self.assertEqual(pk.PACKAGE_PUBKEYS, {k: ROT[k] for k in ("pkg-2026-10", "pkg-2026-10b")})
        self.assertEqual(set(pk.PACKAGE_PINNED), {ROT["pkg-2026-10"]})              # only the OLD key is pin-bound
        pins = pk.PACKAGE_PINNED[ROT["pkg-2026-10"]]
        self.assertTrue(pins and all(len(h) == 64 and h == h.lower() and int(h, 16) >= 0 for h in pins))
        self.assertIn("e9a6060cc9f6d9db4b875f30fcd0fa69917ae48bc98b5cdd6c43d54f9e4739d0", pins)   # the live Free build
        self.assertTrue(pk.pinned_ok(bytes.fromhex(ROT["pkg-2026-10b"]), [FRESH_SHA]))
        self.assertFalse(pk.pinned_ok(bytes.fromhex(ROT["pkg-2026-10"]), [FRESH_SHA]))

    def test_store_and_plugin_keys(self):
        self.assertEqual(self.pub["GRANT_PUBKEYS"], {k: ROT[k] for k in ("ent-2026-10", "ent-2026-10b")})
        self.assertEqual(self.pub["PLUGIN_PUBKEYS"], {k: ROT[k] for k in ("plug-2026-10", "plug-2026-10b")})
        self.assertEqual(set(self.pub["PLUGIN_PINNED"]), {"plug-2026-10"})
        pins = self.pub["PLUGIN_PINNED"]["plug-2026-10"]
        self.assertGreaterEqual(len(pins), 6)                                          # the six plugins live in the Store 10-30
        self.assertTrue(all(len(h) == 64 and h == h.lower() for h in pins))
        self.assertEqual(P.RETIRED_PLUGIN_KEYS, frozenset((ROT["plug-2026-10"],)))
        self.assertNotIn(ROT["plug-2026-10b"], P.RETIRED_PLUGIN_KEYS)

    def test_release_kid_names(self):
        from pxa_plugins import common as C
        self.assertTrue(C.is_release_kid("plug-2026-10") and C.is_release_kid("plug-2026-10b"))
        self.assertFalse(C.is_release_kid("plug-DEV-2026-10") or C.is_release_kid("test-ephemeral"))

    @unittest.skipUnless(os.environ.get("PXA_TEST_PXP_DIR"), "PXA_TEST_PXP_DIR not set (the published .pxp files)")
    def test_published_plugins_still_verify(self):
        saved = (dict(P.PLUGIN_PUBKEYS), copy.deepcopy(P.PLUGIN_PINNED))
        try:
            P.PLUGIN_PUBKEYS.clear()
            P.PLUGIN_PINNED.clear()
            S.load_pubkeys(os.path.join(TOOLS, "pxa_control", "store", "pubkeys.json"))
            files = sorted(glob.glob(os.path.join(os.environ["PXA_TEST_PXP_DIR"], "*.pxp")))
            self.assertTrue(files)
            for f in files:
                with zipfile.ZipFile(f) as z:
                    m = json.loads(z.read("plugin.json").decode("utf-8"))
                P.verify_manifest(m)                                           # raises on a retired_key / bad signature
        finally:
            P.PLUGIN_PUBKEYS.clear()
            P.PLUGIN_PUBKEYS.update(saved[0])
            P.PLUGIN_PINNED.clear()
            P.PLUGIN_PINNED.update(saved[1])


if __name__ == "__main__":
    unittest.main(verbosity=1)
