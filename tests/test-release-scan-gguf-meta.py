#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
# test-release-scan-gguf-meta.py -- scripts/release-scan.py check 5, GGUF metadata. Every file that starts with GGUF has
# its header KV section read: a general.quantized_by, or a quantize.imatrix.* string, that does not name us is a
# gguf-meta block (exit 2), reported by KEY NAME only (the value never reaches stdout, stderr or --json); numeric
# imatrix keys and our own values pass; a header that does not parse blocks too (fail closed); a GGUF inside a
# tarball input is caught the same way. Synthetic files, temp dirs only. No GPU, no network.
#
#     python3 tests/test-release-scan-gguf-meta.py
import io, json, os, struct, subprocess, sys, tarfile, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(ROOT, "scripts", "release-scan.py")
CAT = "gguf-meta"
BLOCKLIST = "@category demo scope=text\n\\bDEMOBLOCKWORD\\b\n@selftest demo DEMOBLOCKWORD\n"
PLANTED = ("ExampleQuantCo", "example-org-calibration")          # third-party values that must never be printed

# GGUF value types (gguf.md): code, struct format
FIXED = {"u8": (0, "B"), "i8": (1, "b"), "u16": (2, "H"), "i16": (3, "h"), "u32": (4, "I"), "i32": (5, "i"),
         "f32": (6, "f"), "bool": (7, "?"), "u64": (10, "Q"), "i64": (11, "q"), "f64": (12, "d")}


def gstr(s, bo):
    b = s.encode("utf-8") if isinstance(s, str) else s
    return struct.pack(bo + "Q", len(b)) + b


def gval(t, v, bo):
    if t == "str":
        return struct.pack(bo + "I", 8) + gstr(v, bo)
    if t == "arr:str":
        return struct.pack(bo + "IIQ", 9, 8, len(v)) + b"".join(gstr(x, bo) for x in v)
    if t.startswith("arr:"):
        code, fmt = FIXED[t[4:]]
        return struct.pack(bo + "IIQ", 9, code, len(v)) + b"".join(struct.pack(bo + fmt, x) for x in v)
    code, fmt = FIXED[t]
    return struct.pack(bo + "I", code) + struct.pack(bo + fmt, v)


def gguf(kvs, version=3, bo="<"):
    """A GGUF header (magic, version, 0 tensors, the KVs) and some bytes standing in for the tensor data."""
    out = b"GGUF" + struct.pack(bo + "IQQ", version, 0, len(kvs))
    for k, t, v in kvs:
        out += gstr(k, bo) + gval(t, v, bo)
    return out + bytes(range(256)) * 4


VOCAB = ["<s>", "</s>"] + ["tok%d" % i for i in range(4000)]
BASE = [("general.architecture", "str", "llama"),          # no general.name: it is a checked key since #55933
        ("llama.block_count", "u32", 2), ("llama.rope.freq_base", "f32", 10000.0), ("test.u8", "u8", 1),
        ("test.i8", "i8", -1), ("test.u16", "u16", 2), ("test.i16", "i16", -2), ("test.u64", "u64", 3),
        ("test.i64", "i64", -3), ("test.f64", "f64", 1.5), ("tokenizer.ggml.model", "str", "gpt2"),
        ("tokenizer.ggml.tokens", "arr:str", VOCAB), ("tokenizer.ggml.token_type", "arr:i32", [1] * len(VOCAB)),
        ("tokenizer.ggml.scores", "arr:f32", [0.0] * len(VOCAB)), ("tokenizer.ggml.add_bos_token", "bool", True),
        ("general.file_type", "u32", 7)]
NUMERIC_IMX = [("quantize.imatrix.entries_count", "i32", 280), ("quantize.imatrix.chunks_count", "i32", 125)]
DIRTY_A = BASE + [("general.quantized_by", "str", PLANTED[0])]
DIRTY_B = BASE + [("quantize.imatrix.dataset", "str", PLANTED[1]), ("quantize.imatrix.chunks_count", "i32", 125)]
# attribution keys (general.repo_url / url / author / homepage / license.link / organization / description / name /
# source*): a value must be us (fullmatch), an allow-listed upstream org (exact), or EXACTLY a declared
# general.base_model.N.repo_url that is https on huggingface.co / github.com with an allow-listed org
# (scripts/release-scan-upstream-orgs.txt). ExampleBase is a self-declared base that is NOT on that list.
BORG = "ExampleBase"
BURL = "https://huggingface.co/ExampleBase/Model-7B"
QURL = "https://huggingface.co/ExampleQuantCo/Model-7B-GGUF"
BM_ORG = [("general.base_model.count", "u32", 1), ("general.base_model.0.name", "str", "Model 7B"),
          ("general.base_model.0.organization", "str", BORG)]
BM_FULL = BM_ORG + [("general.base_model.0.repo_url", "str", BURL)]


def run(t, target, extra=(), env_ours=None, scan=SCAN):
    out = os.path.join(t, "out.json")
    if os.path.exists(out):
        os.remove(out)
    bl = os.path.join(t, "bl.txt")
    open(bl, "w").write(BLOCKLIST)
    env = dict(os.environ)
    env.pop("PXA_GGUF_OURS", None)
    if env_ours is not None:
        env["PXA_GGUF_OURS"] = env_ours
    p = subprocess.run([sys.executable, "-I", scan, "--blocklist", bl, "--json", out, "--work", os.path.join(t, "w"),
                        "--jobs", "1"] + list(extra) + [target], capture_output=True, text=True, env=env, timeout=300)
    js_text = open(out).read() if os.path.exists(out) else ""
    return p.returncode, (json.loads(js_text) if js_text else {}), p.stdout, p.stderr, js_text


def hits(js, cat=CAT):
    d = js[next(iter(js))] if js and "blocks" not in js else js
    return d.get("blocks", {}).get(cat, {}).get("by_file", {}) if d else {}


def pkg_with(t, files):
    """A fresh package dir holding {name: bytes}."""
    import shutil
    pkg = os.path.join(t, "pkg")
    shutil.rmtree(pkg, ignore_errors=True)
    for name, data in files.items():
        fp = os.path.join(pkg, name)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, "wb").write(data)
    return pkg


def main():
    fails = 0

    def check(name, ok, detail):
        nonlocal fails
        print("%s  %-46s %s" % ("ok  " if ok else "FAIL", name, detail))
        fails += not ok

    with tempfile.TemporaryDirectory(prefix="rscan-gguf-") as t:
        trunc = gguf(BASE)
        trunc = trunc[:trunc.index(b"tok2000")]                       # cut inside the tokenizer array
        hostile_key = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1 << 63) + b"general.name"
        cases = [
            # name, files, want rc, want {file: gguf-meta hit count} (None = no gguf-meta hit)
            ("(a) planted general.quantized_by", {"m/a.gguf": gguf(DIRTY_A)}, 2, {"m/a.gguf": 1}),
            ("(a) same, GGUF v2", {"a2.gguf": gguf(DIRTY_A, version=2)}, 2, {"a2.gguf": 1}),
            ("(a) quantized_by not a string", {"a3.gguf": gguf(BASE + [("general.quantized_by", "u32", 7)])}, 2,
             {"a3.gguf": 1}),
            ("(a) same, big-endian file", {"a4.gguf": gguf(DIRTY_A, bo=">")}, 2, {"a4.gguf": 1}),
            ("(b) planted quantize.imatrix.dataset", {"b.gguf": gguf(DIRTY_B)}, 2, {"b.gguf": 1}),
            ("(b) planted string inside an imatrix array",
             {"b2.gguf": gguf(BASE + [("quantize.imatrix.datasets", "arr:str", ["pxa-calib-v2.txt", PLANTED[1]])])}, 2,
             {"b2.gguf": 1}),
            ("(b) both keys in one file", {"b3.gguf": gguf(DIRTY_A + DIRTY_B[len(BASE):])}, 2, {"b3.gguf": 2}),
            ("(c) clean GGUF, neither key", {"c.gguf": gguf(BASE)}, 0, None),
            ("(c) numeric imatrix keys only", {"c2.gguf": gguf(BASE + NUMERIC_IMX)}, 0, None),
            ("(c) clean big-endian file", {"c3.gguf": gguf(BASE, bo=">")}, 0, None),
            ("(d) ours: quantized_by PXA + numeric imatrix keys",
             {"d.gguf": gguf(BASE + [("general.quantized_by", "str", "PXA")] + NUMERIC_IMX)}, 0, None),
            ("(d) imatrix.file pxa-calib.imatrix -> now blocks (MED-1)",
             {"d1.gguf": gguf(BASE + [("general.quantized_by", "str", "PXA"),
                                      ("quantize.imatrix.file", "str", "pxa-calib.imatrix")] + NUMERIC_IMX)}, 2, {"d1.gguf": 1}),
            ("(d) ours: PXQ ignored-imatrix shape (empty values)",
             {"d2.gguf": gguf(BASE + [("quantize.imatrix.file", "str", ""), ("quantize.imatrix.dataset", "str", ""),
                                      ("quantize.imatrix.entries_count", "i64", 0),
                                      ("quantize.imatrix.ignored_by", "str", "pxq-tiers (PXA_PXQ_IMX unset)")])}, 0, None),
            ("(e) truncated inside the KV section", {"e1.gguf": trunc}, 2, {"e1.gguf": 1}),
            ("(e) magic and half a version", {"e2.bin": b"GGUF\x03\x00"}, 2, {"e2.bin": 1}),
            ("(e) unsupported version", {"e3.gguf": gguf(BASE, version=99)}, 2, {"e3.gguf": 1}),
            ("(e) key length 2^63", {"e4.gguf": hostile_key}, 2, {"e4.gguf": 1}),
            ("(e) unknown value type",
             {"e5.gguf": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + gstr("general.name", "<") + struct.pack("<I", 13)}, 2,
             {"e5.gguf": 1}),
            ("(e) array count over the bound",
             {"e6.gguf": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + gstr("tokenizer.ggml.tokens", "<") +
              struct.pack("<IIQ", 9, 8, 10 ** 9)}, 2, {"e6.gguf": 1}),
            ("(e) KV count over the bound", {"e7.gguf": b"GGUF" + struct.pack("<IQQ", 3, 0, 10 ** 7)}, 2, {"e7.gguf": 1}),
            ("(e) no GGUF magic: not parsed", {"x.gguf": b"GGUX" + gguf(DIRTY_A)[4:]}, 0, None),
        ]
        for name, files, want_rc, want in cases:
            rc, js, out, err, _ = run(t, pkg_with(t, files))
            got = hits(js)
            ok = rc == want_rc and (got == want if want else not got)
            check(name, ok, "rc=%d %s=%s" % (rc, CAT, got))

        # (f) the report names the keys, never the values: stdout, stderr and --json
        pkg = pkg_with(t, {"a.gguf": gguf(DIRTY_A), "b.gguf": gguf(DIRTY_B),
                           "b2.gguf": gguf(BASE + [("quantize.imatrix.datasets", "arr:str", [PLANTED[1]])])})
        rc, js, out, err, js_text = run(t, pkg)
        blob = (out + err + js_text).lower()
        leaked = [v for v in PLANTED if v.lower() in blob]
        named = all(k in out for k in ("general.quantized_by", "quantize.imatrix.dataset", "quantize.imatrix.datasets"))
        check("(f) key names shown, values never shown", rc == 2 and named and not leaked and len(hits(js)) == 3,
              "rc=%d key names in stdout=%s leaked=%d" % (rc, named, len(leaked)))

        # (g) the same GGUF inside a .tar.gz input
        tgz = os.path.join(t, "pkg.tar.gz")
        src = pkg_with(t, {"models/a.gguf": gguf(DIRTY_A), "README.md": b"a model package\n"})
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(src, arcname="pkg")
        rc, js, out, err, _ = run(t, tgz)
        got = hits(js)
        check("(g) GGUF inside a .tar.gz input", rc == 2 and got == {"pkg/models/a.gguf": 1}, "rc=%d %s=%s" % (rc, CAT, got))

        # (h) who "us" is: --gguf-ours REGEX, else $PXA_GGUF_OURS, else the default; a bad regex stops the gate
        pkg = pkg_with(t, {"a.gguf": gguf(DIRTY_A)})
        them = r"(?i)\bexamplequantco\b"
        for name, extra, env_ours, want_rc, want in (
                ("(h) --gguf-ours naming the planted value", ["--gguf-ours", them], None, 0, None),
                ("(h) PXA_GGUF_OURS naming the planted value", [], them, 0, None),
                ("(h) --gguf-ours wins over PXA_GGUF_OURS", ["--gguf-ours", r"(?i)\bpx(?:a|q)"], them, 2, {"a.gguf": 1}),
                ("(h) bad --gguf-ours regex: cannot run", ["--gguf-ours", "("], None, 3, None)):
            rc, js, out, err, _ = run(t, pkg, extra, env_ours)
            got = hits(js)
            ok = rc == want_rc and (got == want if want else not got)
            check(name, ok, "rc=%d %s=%s" % (rc, CAT, got))

        # (i)-(vii) attribution keys. "-> now blocks": passed under the earlier base-model rules, blocks since #55933
        # (ExampleBase is a self-declared base whose org is not on the upstream list, so it names nobody)
        mirrors = [BURL + "/m%d" % i for i in range(5000)]
        QW = "https://huggingface.co/Qwen/Qwen3.8-27B"
        QW_BASE = [("general.base_model.count", "u32", 1), ("general.base_model.0.organization", "str", "Qwen"),
                   ("general.base_model.0.repo_url", "str", QW)]
        for name, kvs, want_rc, want in (
                ("(i) repo_url naming the quantizer, base declared", BM_ORG + [("general.repo_url", "str", QURL)], 2, 1),
                ("(ii) repo_url = a self-declared base -> now blocks", BM_FULL + [("general.repo_url", "str", BURL)], 2, 1),
                ("(ii) under a base repo_url (case, /) -> now blocks",
                 [("general.base_model.0.repo_url", "str", BURL + "/"), ("general.repo_url", "str", BURL.lower() + "/tree/main")],
                 2, 1),
                ("(ii) url naming the base org -> now blocks", BM_ORG + [("general.url", "str", "https://huggingface.co/ExampleBase")],
                 2, 1),
                ("(iii) author = the quantizer", BM_ORG + [("general.author", "str", PLANTED[0])], 2, 1),
                ("(iii) author = PXA", BM_ORG + [("general.author", "str", "PXA")], 0, None),
                ("(iv) source.url+repo_url naming the base -> now blocks",
                 BM_ORG + [("general.source.url", "str", BURL), ("general.source.repo_url", "str", BURL)], 2, 2),
                ("(iv) source.url + source.repo_url naming the quantizer",
                 BM_ORG + [("general.source.url", "str", QURL), ("general.source.repo_url", "str", QURL)], 2, 2),
                ("(iv) quantizer URL inside a general.source* array", BM_ORG + [("general.source.urls", "arr:str", [BURL, QURL])],
                 2, 1),
                ("(v) upstream base declared AFTER the keys: pass",
                 [("general.repo_url", "str", QW), ("general.source.url", "str", QW)] + QW_BASE, 0, None),
                ("(v) base model declared AFTER the keys: hit", [("general.repo_url", "str", QURL)] + QW_BASE, 2, 1),
                ("(vii) names the base org, no base model declared", [("general.repo_url", "str", BURL)], 2, 1),
                ("blank attribution values name nobody", [("general.url", "str", ""), ("general.author", "str", "  ")], 0, None),
                ("attribution strings over the cap: fail closed", BM_ORG + [("general.source.mirrors", "arr:str", mirrors)], 2, 1)):
            rc, js, out, err, _ = run(t, pkg_with(t, {"m.gguf": gguf(BASE + kvs)}))
            got = hits(js)
            ok = rc == want_rc and (got == {"m.gguf": want} if want else not got)
            check(name, ok, "rc=%d %s=%s" % (rc, CAT, got))

        # URLs (e4ec0330ff round): a substring never names the base model
        HF = "https://huggingface.co/"
        BREPO = [("general.base_model.0.repo_url", "str", HF + "ExampleBase/Model")]
        for name, kvs, want_rc, want in (
                ("URL: quantizer owner, base org in the repo name",
                 BM_ORG + [("general.repo_url", "str", HF + "ExampleQuantCo/ExampleBase-Model-7B-GGUF")], 2, 1),
                ("URL: owner only starts with the base org", BM_ORG + [("general.repo_url", "str", HF + "ExampleBase-Fans/Model")],
                 2, 1),
                ("URL: base org's namespace -> now blocks", BM_ORG + [("general.repo_url", "str", HF + "ExampleBase/Model-7B-GGUF")],
                 2, 1),
                ("URL: other owner past a base repo_url", BM_ORG + BREPO + [("general.repo_url", "str", HF + "ExampleBaseX/Model-GGUF")],
                 2, 1),
                ("URL: base repo_url prefix, no path boundary", BREPO + [("general.repo_url", "str", HF + "ExampleBase/Model-GGUF")],
                 2, 1),
                ("URL: base prefix then ?query -> now blocks", BREPO + [("general.repo_url", "str", HF + "ExampleBase/Model?revision=main")],
                 2, 1),
                ("host without a scheme, quantizer owner",
                 BM_ORG + [("general.source.url", "str", "huggingface.co/ExampleQuantCo/ExampleBase-Model-GGUF")], 2, 1),
                ("bare repo id, quantizer owner",
                 BM_ORG + [("general.source.huggingface.repository", "str", "ExampleQuantCo/ExampleBase-Model-7B-GGUF")], 2, 1),
                ("bare repo id, base owner -> now blocks", BM_ORG + [("general.source.huggingface.repository", "str", "ExampleBase/Model-7B")],
                 2, 1),
                ("author 'ExampleBase Team' -> now blocks", BM_ORG + [("general.author", "str", "ExampleBase Team")], 2, 1),
                ("author 'ExampleBaseFans': not a whole word", BM_ORG + [("general.author", "str", "ExampleBaseFans")], 2, 1),
                ("source.doi / source.uuid: identifiers, never hits",
                 [("general.source.doi", "str", "10.1234/ExampleQuantCo.5678"), ("general.source.uuid", "str", "ExampleQuantCo-0000")],
                 0, None)):
            rc, js, out, err, _ = run(t, pkg_with(t, {"m.gguf": gguf(BASE + kvs)}))
            got = hits(js)
            ok = rc == want_rc and (got == {"m.gguf": want} if want else not got)
            check(name, ok, "rc=%d %s=%s" % (rc, CAT, got))

        # #55895 MED-1: "us" is a FULLMATCH of the allow-list (?i)\s*(PXA|PXA Network|PXQ)\s*, never a substring;
        # #55933 (2): only EXACTLY a declared base repo_url that is https on huggingface.co / github.com, without
        # userinfo, '..', '\', '%', '?', '#', and with its org on the upstream list; plain values: us or an upstream org
        def based(url, org="meta-llama"):
            return [("general.base_model.count", "u32", 1), ("general.base_model.0.organization", "str", org),
                    ("general.base_model.0.repo_url", "str", url)]
        ML = "https://huggingface.co/meta-llama/Llama-3-8B"
        probes = [
            ("MED-1 quantized_by 'evil.com/pxa'", [("general.quantized_by", "str", "evil.com/pxa")], 2, 1),
            ("MED-1 quantized_by 'examplequant (not pxa)'", [("general.quantized_by", "str", "examplequant (not pxa)")], 2, 1),
            ("MED-1 repo_url 'huggingface.co/evil/x-pxa'", [("general.repo_url", "str", "huggingface.co/evil/x-pxa")], 2, 1),
            ("MED-1 'PXA Network' / ' pxq ' are us", [("general.quantized_by", "str", "PXA Network"),
                                                      ("general.author", "str", " pxq ")], 0, None),
            ("(2) self-declared base not on the list (examplequant)",
             based(HF + "examplequant/Model-7B-GGUF", "examplequant") + [("general.repo_url", "str", HF + "examplequant/Model-7B-GGUF")],
             2, 1)]
        for url in ("huggingface.co/meta-llama/../evil", "https://huggingface.co/meta-llama/../evil",
                    "meta-llama/llama/../../evil", "evil.com/meta-llama/x", "https://huggingface.co@evil.com/meta-llama/x",
                    "meta-llama/x", ML + "?x=https://evil", "https://meta-llama.evil.com", "http://huggingface.co/meta-llama/x",
                    "https://huggingface.co:8443/meta-llama/x", "https://huggingface.co/meta-llama/x%2F..%2Fevil",
                    "https://huggingface.co/meta-llama/x#frag", "https://huggingface.co\\meta-llama\\x"):
            probes.append(("(2) declared base %s" % url[:40], based(url) + [("general.repo_url", "str", url)], 2, 1))
        probes += [
            ("(2) author 'evil, derived from meta-llama'", based(ML) + [("general.author", "str", "evil, derived from meta-llama")],
             2, 1),
            ("(2) PASS exact declared upstream base repo_url", QW_BASE + [("general.repo_url", "str", QW)], 0, None),
            ("(2) PASS author 'PXA', organization 'Qwen'", [("general.author", "str", "PXA"), ("general.organization", "str", "Qwen")],
             0, None),
            ("(3) homepage/license.link/organization/description/name",
             [("general.homepage", "str", QURL), ("general.license.link", "str", QURL + "/LICENSE"),
              ("general.organization", "str", PLANTED[0]), ("general.description", "str", "Quantized by " + PLANTED[0]),
              ("general.name", "str", PLANTED[0] + " Model 7B")], 2, 5),
            ("(3) general.name 'PXA' passes", [("general.name", "str", "PXA")], 0, None)]
        for name, kvs, want_rc, want in probes:
            rc, js, out, err, _ = run(t, pkg_with(t, {"m.gguf": gguf(BASE + kvs)}))
            got = hits(js)
            ok = rc == want_rc and (got == {"m.gguf": want} if want else not got)
            check(name, ok, "rc=%d %s=%s" % (rc, CAT, got))

        # printed keys: control and format characters stripped (the value is never printed)
        rc, js, out, err, _ = run(t, pkg_with(t, {"m.gguf": gguf(BASE + [("quantize.imatrix.\x1b[31mfi‮le", "str", PLANTED[1])])}))
        ok = rc == 2 and "'quantize.imatrix.[31mfile: value does not name us'" in out and "\\x1b" not in out and "\\u202e" not in out
        check("printed key: control characters stripped", ok, "rc=%d" % rc)

        # LOW-5: one element budget (5M) for the whole KV walk: a 6M-element string array fails closed
        n = 6_000_000
        big = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + gstr("tokenizer.ggml.tokens", "<") + struct.pack("<IIQ", 9, 8, n) + bytes(8 * n)
        rc, js, out, err, _ = run(t, pkg_with(t, {"big.gguf": big}))
        got = hits(js)
        check("LOW-5 6M array elements: over the walk budget", rc == 2 and got == {"big.gguf": 1}, "rc=%d %s=%s" % (rc, CAT, got))
        del big

        # LOW-3: the docker image skip lists (/etc, /usr/share/doc, dpkg-owned files) never hide a GGUF
        img = os.path.join(t, "img.tar")
        layer = io.BytesIO()
        with tarfile.open(fileobj=layer, mode="w") as lt:
            for nm, data in (("etc/m.gguf", gguf(DIRTY_A)), ("usr/share/doc/x/m.gguf", gguf(DIRTY_A)),
                             ("usr/lib/owned.gguf", gguf(DIRTY_A)), ("var/lib/dpkg/info/x.list", b"/usr/lib/owned.gguf\n"),
                             ("opt/app/clean.gguf", gguf(BASE)), ("opt/app/dirty.gguf", gguf(DIRTY_A))):
                ti = tarfile.TarInfo(nm)
                ti.size = len(data)
                lt.addfile(ti, io.BytesIO(data))
        with tarfile.open(img, "w") as tf:
            for nm, data in (("manifest.json", json.dumps([{"Config": "cfg.json", "Layers": ["l1/layer.tar"]}]).encode()),
                             ("cfg.json", b'{"config": {}}'), ("l1/layer.tar", layer.getvalue())):
                ti = tarfile.TarInfo(nm)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        rc, js, out, err, _ = run(t, img)
        got = hits(js)
        want = {"image:/etc/m.gguf": 1, "image:/usr/share/doc/x/m.gguf": 1, "image:/usr/lib/owned.gguf": 1, "image:/opt/app/dirty.gguf": 1}
        check("LOW-3 GGUF under the docker skip lists", rc == 2 and got == want, "rc=%d %s=%s" % (rc, CAT, got))

        # (vi) attribution hits name the keys, never the values
        pkg = pkg_with(t, {"i.gguf": gguf(BASE + BM_ORG + [("general.repo_url", "str", QURL)]),
                           "iii.gguf": gguf(BASE + BM_ORG + [("general.author", "str", PLANTED[0])]),
                           "iv.gguf": gguf(BASE + BM_ORG + [("general.source.url", "str", QURL), ("general.source.repo_url", "str", QURL)]),
                           "vii.gguf": gguf(BASE + [("general.url", "str", BURL)])})
        rc, js, out, err, js_text = run(t, pkg)
        blob = (out + err + js_text).lower()
        leaked = [v for v in (PLANTED[0], BORG, QURL, BURL, "huggingface") if v.lower() in blob]
        keys = ("general.repo_url", "general.author", "general.source.url", "general.source.repo_url", "general.url")
        named = all(k + ":" in out for k in keys)
        check("(vi) attribution: key names shown, values never",
              rc == 2 and named and not leaked and hits(js) == {"i.gguf": 1, "iii.gguf": 1, "iv.gguf": 2, "vii.gguf": 1},
              "rc=%d key names in stdout=%s leaked=%d" % (rc, named, len(leaked)))

        # #55958 (a): the own-tool note is ours only byte for byte, and only under its own key
        NOTE = "pxq-tiers (PXA_PXQ_IMX unset)"
        for name, kvs, want_rc, want in (
                ("(a) own note, exact: ours", [("quantize.imatrix.ignored_by", "str", NOTE)], 0, None),
                ("(a) near miss: an extra character", [("quantize.imatrix.ignored_by", "str", NOTE + ".")], 2, 1),
                ("(a) near miss: a trailing space", [("quantize.imatrix.ignored_by", "str", NOTE + " ")], 2, 1),
                ("(a) near miss: another case", [("quantize.imatrix.ignored_by", "str", NOTE.upper())], 2, 1),
                ("(a) the note under another key", [("quantize.imatrix.file", "str", NOTE)], 2, 1),
                ("(c) LiquidAI upstream base passes", [("general.base_model.0.repo_url", "str", HF + "LiquidAI/Example-1B"),
                                                       ("general.repo_url", "str", HF + "LiquidAI/Example-1B")], 0, None),
                # #55961: general.name / general.description pass as 'PXA ' + text: anchored at byte 0 (no strip),
                # case-sensitive, and with no '/' (so no '://') and no 'www.'
                ("name 'PXA Qwen3 8B': ours", [("general.name", "str", "PXA Qwen3 8B")], 0, None),
                ("name 'PXA Qwen3.8 27B PXQN5': ours", [("general.name", "str", "PXA Qwen3.8 27B PXQN5")], 0, None),
                ("description 'PXA build ...': ours", [("general.description", "str", "PXA build of an upstream model")], 0, None),
                ("name 'Qwen3 8B Instruct': blocks", [("general.name", "str", "Qwen3 8B Instruct")], 2, 1),
                ("name 'PXA by evil.com/x': blocks", [("general.name", "str", "PXA by evil.com/x")], 2, 1),
                ("name 'examplequant PXA': blocks", [("general.name", "str", "examplequant PXA")], 2, 1),
                ("name 'pxa qwen' (lower case): blocks", [("general.name", "str", "pxa qwen")], 2, 1),
                ("name ' PXA Qwen3 8B' (leading space): blocks", [("general.name", "str", " PXA Qwen3 8B")], 2, 1),
                ("description 'PXA ... www.': blocks", [("general.description", "str", "PXA see www.example.org")], 2, 1),
                ("author 'PXA Qwen3 8B' (name rule only): blocks", [("general.author", "str", "PXA Qwen3 8B")], 2, 1),
                # #55984: no domain-like token either (a dot, then 2+ letters that end the token)
                ("name 'PXA model v4.0.1': ours", [("general.name", "str", "PXA model v4.0.1")], 0, None),
                ("name 'PXA Llama-3.1-8B.Q4_K_M': ours", [("general.name", "str", "PXA Llama-3.1-8B.Q4_K_M")], 0, None),
                ("name 'PXA x evil.com': blocks", [("general.name", "str", "PXA x evil.com")], 2, 1),
                ("name 'PXA mirror.example.org': blocks", [("general.name", "str", "PXA mirror.example.org")], 2, 1),
                ("name 'PXA x.io build': blocks", [("general.name", "str", "PXA x.io build")], 2, 1),
                ("description 'PXA see EVIL.COM:8080': blocks", [("general.description", "str", "PXA see EVIL.COM:8080")], 2, 1),
                ("name 'PXA x evil．com' (fullwidth dot): blocks", [("general.name", "str", "PXA x evil．com")], 2, 1),
                ("name 'PXA x.xn--p1ai' (punycode TLD): blocks", [("general.name", "str", "PXA x.xn--p1ai")], 2, 1)):
            rc, js, out, err, _ = run(t, pkg_with(t, {"m.gguf": gguf(BASE + kvs)}))
            got = hits(js)
            ok = rc == want_rc and (got == {"m.gguf": want} if want else not got)
            check(name, ok, "rc=%d %s=%s" % (rc, CAT, got))

        # #55958 (b): the upstream vocab test GGUFs in models/ are exempt by EXACT sha256 + basename from
        # scripts/release-scan-gguf-exempt.txt; a changed byte, another name or an unlisted file blocks
        import shutil
        vname = "ggml-vocab-llama-spm.gguf"
        vocab = open(os.path.join(ROOT, "models", vname), "rb").read()
        flipped = vocab[:-1] + bytes([vocab[-1] ^ 1])
        for name, rel, data, want_rc in (
                ("(b) listed vocab file passes", "models/" + vname, vocab, 0),
                ("(b) the same file, one byte changed: blocks", "models/" + vname, flipped, 2),
                ("(b) a copy under another name: blocks", "models/ggml-vocab-renamed.gguf", vocab, 2),
                ("(b) an unlisted GGUF with hits: blocks", "models/ggml-vocab-extra.gguf",
                 gguf(BASE + [("general.name", "str", PLANTED[0] + " vocab")]), 2)):
            rc, js, out, err, _ = run(t, pkg_with(t, {rel: data}))
            got = hits(js)
            ok = rc == want_rc and ((rel in got) if want_rc == 2 else not got)
            check(name, ok, "rc=%d %s=%s" % (rc, CAT, got))
        rc, js, out, err, _ = run(t, os.path.join(ROOT, "models"))
        got = hits(js)
        check("(b) the repo's models/ dir: no gguf-meta hit", rc == 0 and not got, "rc=%d %s files=%d" % (rc, CAT, len(got)))
        # a missing or malformed list stops the gate (exit 3): a copy of the scanner next to edited lists
        alt = os.path.join(t, "alt")
        os.makedirs(alt)
        shutil.copy(SCAN, alt)
        orgs = os.path.join(ROOT, "scripts", "release-scan-upstream-orgs.txt")
        exempt = os.path.join(ROOT, "scripts", "release-scan-gguf-exempt.txt")
        small = pkg_with(t, {"readme.txt": b"hello\n"})
        for name, orgs_src, exempt_text, want_rc in (
                ("(b) exemption list missing: cannot run", orgs, None, 3),
                ("(b) exemption list malformed: cannot run", orgs, "0123  models/x.gguf\n", 3),
                ("(b) exemption list absolute path: cannot run", orgs, "%s  /models/x.gguf\n" % ("0" * 64), 3),
                ("upstream org list missing: cannot run", None, open(exempt).read() if os.path.exists(exempt) else "", 3),
                ("both lists present: runs", orgs, open(exempt).read() if os.path.exists(exempt) else "", 0)):
            for f in ("release-scan-upstream-orgs.txt", "release-scan-gguf-exempt.txt"):
                if os.path.exists(os.path.join(alt, f)):
                    os.remove(os.path.join(alt, f))
            if orgs_src:
                shutil.copy(orgs_src, alt)
            if exempt_text is not None:
                open(os.path.join(alt, "release-scan-gguf-exempt.txt"), "w").write(exempt_text)
            rc, js, out, err, _ = run(t, small, scan=os.path.join(alt, "release-scan.py"))
            check(name, rc == want_rc, "rc=%d" % rc)

    # LOW-1: a file that cannot be read (EIO / EACCES on the magic read) is a block, in process with a failing open();
    # a short file and a file without the magic are still not GGUFs
    import importlib.util
    spec = importlib.util.spec_from_file_location("rscan_gguf_low1", SCAN)
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    with tempfile.TemporaryDirectory(prefix="rscan-gguf-low1-") as t:
        bad, short, plain = (os.path.join(t, n) for n in ("bad.gguf", "short.bin", "plain.txt"))
        for p, data in ((bad, gguf(DIRTY_A)), (short, b"GGU"), (plain, b"hello\n")):
            open(p, "wb").write(data)

        def failing_open(path, *a, **k):
            if path == bad:
                raise OSError(5, "Input/output error")
            return open(path, *a, **k)
        rs.open = failing_open
        got = [rs.gguf_meta_hits(p) for p in (bad, short, plain)]
        ok = len(got[0]) == 1 and got[0][0].startswith("GGUF header unreadable") and got[1] == [] and got[2] == []
        check("LOW-1 unreadable file blocks; short / plain do not", ok, "hits=%s" % [len(g) for g in got])

    print("test-release-scan-gguf-meta: %s" % ("PASS" if not fails else "FAIL %d" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
