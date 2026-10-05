"""Fakes for the Encode tab's tests: a Hugging Face server, a licence server, a CDN, fake encoder / engine tool folders, a package
builder and an Ed25519 signer. Everything runs on 127.0.0.1 with no GPU, no real model and no real encoder.

Used by tests/test-pxa-encode.py, tests/test-pxa-control.py (the HTTP routes) and tests/encode-gui-check.py (the headless browser)."""
import base64
import hashlib
import http.server
import io
import json
import os
import socketserver
import struct
import sys
import tarfile
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import pxa_encode_pkg as PK  # noqa: E402

TOOLS = os.path.join(HERE, "fake_encode_tools.py")


# ---------------------------------------------------------------------------------------------
# Ed25519 signing (tests only; Control itself only verifies)
# ---------------------------------------------------------------------------------------------
def _compress(P):
    zi = pow(P[2], PK._P - 2, PK._P)
    x, y = P[0] * zi % PK._P, P[1] * zi % PK._P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def ed_pub(seed):
    h = hashlib.sha512(seed).digest()
    a = (int.from_bytes(h[:32], "little") & ((1 << 254) - 8)) | (1 << 254)
    return _compress(PK._mul(a, PK._G))


def ed_sign(seed, msg):
    h = hashlib.sha512(seed).digest()
    a = (int.from_bytes(h[:32], "little") & ((1 << 254) - 8)) | (1 << 254)
    A = _compress(PK._mul(a, PK._G))
    r = int.from_bytes(hashlib.sha512(h[32:] + msg).digest(), "little") % PK._Q
    R = _compress(PK._mul(r, PK._G))
    k = int.from_bytes(hashlib.sha512(R + A + msg).digest(), "little") % PK._Q
    return R + ((r + k * a) % PK._Q).to_bytes(32, "little")


SEED = hashlib.sha256(b"pxa-encode-test-signing-seed").digest()
OTHER_SEED = hashlib.sha256(b"someone-else").digest()
PUB_HEX = ed_pub(SEED).hex()
GOOD_KEY = "pxk1.K-TEST0001." + "A1b2C3d4" * 4
OTHER_KEY = "pxk1.K-TEST0002." + "Z9y8X7w6" * 4


# ---------------------------------------------------------------------------------------------
# fake tool folders
# ---------------------------------------------------------------------------------------------
def _launcher(path, tool, fake_dir):
    with open(path, "w") as f:
        f.write('#!/bin/sh\nexec "%s" "%s" "%s" "%s" "$@"\n' % (sys.executable, TOOLS, tool, fake_dir))
    os.chmod(path, 0o755)


def _b64u(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def make_fake_encoder(d, edition="free", tiers=None, build_id="b-free-1", version="1.0.0", features=None, licence=None, **extra):
    """A folder with a `pxqe` command line that behaves like the encoder's `info --json` / `make` / `quantize` / `run`, plus the calibration text.
    `cli="1"` (via **extra) makes it an older encoder: no `make`, no `stages[]`, Control then drives the multi-command flow.
    `licence` is the offline view (`info --json`); `licence_check` (via **extra) is what `--check-licence` answers."""
    os.makedirs(d, exist_ok=True)
    classic = ["pxq1", "pxq2", "pxq3", "pxq4", "pxq4hq", "pxq6", "pxq_universal"]
    cfg = {"edition": edition, "version": version, "build_id": build_id, "cli": "2",
           "tiers": tiers if tiers is not None else (classic if edition == "free" else classic + ["pxqn1", "pxqn2", "pxqn3", "pxqn3s8", "pxqn4", "pxqn4s8", "pxqn5"]),
           "features": features if features is not None else (["classic-quantizer", "make", "skeleton", "dump", "ldlq", "hessian", "watermark", "licence"] if edition == "pro" else ["classic-quantizer", "make"])}
    if licence is not None:
        cfg["licence"] = licence
    cfg.update(extra)
    with open(os.path.join(d, "fake.json"), "w") as f:
        json.dump(cfg, f)
    p = os.path.join(d, "pxqe")
    with open(p, "w") as f:
        f.write('#!/bin/sh\nD=$(cd "$(dirname "$0")" && pwd)\nexec "%s" "%s" pxqe "$D" "$@"\n' % (sys.executable, TOOLS))
    os.chmod(p, 0o755)
    with open(os.path.join(d, "calib.txt"), "w") as f:
        f.write("a small calibration text for tests\n" * 20)
    return p


def set_fake(d, **kw):
    p = os.path.join(d, "fake.json")
    with open(p) as f:
        c = json.load(f)
    c.update(kw)
    with open(p, "w") as f:
        json.dump(c, f)


def make_engine_dir(d, delay=0.0):
    """An engine folder: bin/llama-server (answers --version), bin/llama-quantize, bin/llama-imatrix, convert_hf_to_gguf.py."""
    fake = os.path.join(d, ".fake")
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    os.makedirs(fake, exist_ok=True)
    with open(os.path.join(fake, "fake.json"), "w") as f:
        json.dump({"delay": delay}, f)
    _launcher(os.path.join(d, "bin", "llama-quantize"), "llama-quantize", fake)
    _launcher(os.path.join(d, "bin", "llama-imatrix"), "llama-imatrix", fake)
    with open(os.path.join(d, "convert_hf_to_gguf.py"), "w") as f:       # run by Python, like the real converter
        f.write("import subprocess, sys\nsys.exit(subprocess.call([sys.executable, %r, 'convert_hf_to_gguf', %r] + sys.argv[1:]))\n" % (TOOLS, fake))
    _launcher(os.path.join(d, "bin", "llama-server"), "llama-server", fake)
    return d


def set_engine_delay(d, delay):
    with open(os.path.join(d, ".fake", "fake.json"), "w") as f:
        json.dump({"delay": delay}, f)


# ---------------------------------------------------------------------------------------------
# a package (tar.gz, top folder pxqe/) + the licence server's signed answer
# ---------------------------------------------------------------------------------------------
KID = "pkg-2026-10"


def statement_for(edition, build_id, sha, size, platform="linux-x86_64", cuda_major=12, pkg=""):
    return json.dumps({"build_id": build_id, "cuda_major": cuda_major, "edition": edition, "kid": KID, "pkg": pkg, "platform": platform, "sha256": sha, "size": size},
                      sort_keys=True, separators=(",", ":"))


def sign_answer(answer, seed=SEED, platform="linux-x86_64", cuda_major=12, sha=None, size=None):
    """Add `statement` + `signature` to an answer dict, the way pxa-licd does: Ed25519 over b'pxqe-pkg|' + statement, base64url."""
    st = statement_for(answer["edition"], answer["build_id"], sha or answer["sha256"], size if size is not None else answer["size"], platform, cuda_major)
    answer = dict(answer, statement=st, signature=_b64u(ed_sign(seed, b"pxqe-pkg|" + st.encode())), kid=KID, platform=platform, cuda_major=cuda_major)
    return answer


def build_package(dest_dir, edition, build_id, version="1.0.0", seed=SEED, tiers=None, base_url="", tamper_file=False, wrong_platform=False, **fake):
    """-> (answer dict as the licence server returns it, archive path). The archive has the real layout: top folder `pxqe/` with a fake
    `pxqe` launcher, package.json, fake.json (the fake's behaviour), calib.txt, MANIFEST.json and MANIFEST.sig."""
    os.makedirs(dest_dir, exist_ok=True)
    name = "pxqe-%s-%s-linux-x86_64-cu12.tar.gz" % (edition, build_id)
    path = os.path.join(dest_dir, name)
    classic = ["pxq1", "pxq2", "pxq3", "pxq4", "pxq4hq", "pxq6", "pxq_universal"]
    cfg = {"edition": edition, "version": version, "build_id": build_id, "cli": "2",
           "tiers": tiers or (classic if edition == "free" else classic + ["pxqn1", "pxqn2", "pxqn3", "pxqn3s8", "pxqn4", "pxqn4s8", "pxqn5"]),
           "features": ["classic-quantizer", "make", "skeleton", "dump", "ldlq", "hessian", "watermark", "licence"] if edition == "pro" else ["classic-quantizer", "make"]}
    cfg.update(fake)
    launcher = ('#!/bin/sh\nD=$(cd "$(dirname "$0")" && pwd)\nexec "%s" "%s" pxqe "$D" "$@"\n' % (sys.executable, TOOLS)).encode()
    pj = {"build_id": build_id, "edition": edition, "version": "%s-%s" % (edition, build_id), "platform": "linux-x86_64", "cuda_major": 12, "tiers": cfg["tiers"], "features": cfg["features"]}
    files = {"pxqe": (launcher, 0o755), "package.json": (json.dumps(pj).encode(), 0o644), "fake.json": (json.dumps(cfg).encode(), 0o644), "calib.txt": (b"calibration\n" * 50, 0o644)}
    man = json.dumps({"edition": edition, "build_id": build_id, "files": {k: hashlib.sha256(v[0]).hexdigest() for k, v in files.items()}}, sort_keys=True, separators=(",", ":")).encode()
    if tamper_file:
        files["calib.txt"] = (b"tampered after the manifest was made\n", 0o644)
    files["MANIFEST.json"] = (man, 0o644)
    files["MANIFEST.sig"] = ((_b64u(ed_sign(seed, b"pxqe-manifest|" + man)) + "\n").encode(), 0o644)
    with tarfile.open(path, "w:gz") as tf:
        top = tarfile.TarInfo("pxqe")
        top.type = tarfile.DIRTYPE
        top.mode = 0o755
        tf.addfile(top)
        for fname, (data, mode) in files.items():
            ti = tarfile.TarInfo("pxqe/" + fname)
            ti.size = len(data)
            ti.mode = mode
            tf.addfile(ti, io.BytesIO(data))
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    size = os.path.getsize(path)
    ans = {"build_id": build_id, "edition": edition, "sha256": sha, "size": size, "package": "pkg-test" if edition == "pro" else "",
           "url": "/v1/download/tok-%s" % build_id if edition == "pro" else "/v1/free/download/%s" % build_id, "expires": int(time.time()) + 900}
    if edition == "free":
        ans["tiers"] = cfg["tiers"]
    ans = sign_answer(ans, seed, platform="linux-arm64" if wrong_platform else "linux-x86_64")
    return ans, path


def build_runtime_pack(dest_dir, rid="cuda12.8.1-r1", seed=SEED, tamper=False, missing_lib=False, wrong_id=False, bad_checksum=False, bulk=0):
    """The shared GPU runtime pack as the licence server publishes it: tar.xz, top folder `pxqe-runtime/` with lib/<the five CUDA libraries as real files>,
    LICENSE-NVIDIA-CUDA-EULA.txt, NOTICE.txt, runtime.json (id, libs, sizes), MANIFEST.json (sha256 of every file, unsigned like the real one).
    -> (answer dict as GET /v1/runtime/latest returns it, archive path)."""
    os.makedirs(dest_dir, exist_ok=True)
    name = "pxqe-runtime-%s-linux-x86_64-cu12.tar.xz" % rid
    path = os.path.join(dest_dir, name)
    libs = ["libcublas.so.12", "libcublasLt.so.12", "libcusolver.so.11", "libcusparse.so.12", "libnvJitLink.so.12"]
    if missing_lib:
        libs = [l for l in libs if l != "libcusolver.so.11"]
    files = {"lib/" + l: (b"\x7fELF fake %s\n" % l.encode()) * 200 for l in libs}
    if bulk:
        files["lib/libcublasLt.so.12"] = os.urandom(bulk)          # incompressible: a download of real size, for the cancel / resume tests
    files["LICENSE-NVIDIA-CUDA-EULA.txt"] = b"End User License Agreement (fake)\n" * 10
    files["NOTICE.txt"] = b"PXA GPU runtime (fake)\n"
    sizes = {k: len(v) for k, v in files.items()}
    files["runtime.json"] = json.dumps({"runtime_id": "other-id" if wrong_id else rid, "edition": "runtime", "cuda": "12.8.1", "libs": libs, "sizes": sizes}, sort_keys=True).encode()
    man = json.dumps({"edition": "runtime", "build_id": rid, "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}, sort_keys=True, separators=(",", ":")).encode()
    if tamper:
        files["lib/libcublas.so.12"] = b"tampered after the manifest was made\n"
    files["MANIFEST.json"] = man
    with tarfile.open(path, "w:xz") as tf:
        top = tarfile.TarInfo("pxqe-runtime")
        top.type = tarfile.DIRTYPE
        top.mode = 0o755
        tf.addfile(top)
        for fname, data in files.items():
            ti = tarfile.TarInfo("pxqe-runtime/" + fname)
            ti.size = len(data)
            ti.mode = 0o755 if fname.startswith("lib/") else 0o644
            tf.addfile(ti, io.BytesIO(data))
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    ans = {"build_id": rid, "runtime_id": rid, "edition": "runtime", "version": "cuda-12.8.1-r1", "sha256": "0" * 64 if bad_checksum else sha, "size": os.path.getsize(path),
           "unpacked": sum(sizes.values()), "libs": libs, "cuda": "12.8.1", "url": "/v1/runtime/download/%s" % rid,
           "notice": "NVIDIA CUDA libraries, redistributed under the NVIDIA CUDA Toolkit EULA (the licence text is inside the pack). For use with the PXA Quantizer."}
    ans = sign_answer(ans, seed)
    return ans, path


def build_evil_package(dest_dir, kind="traversal"):
    """A package whose statement signature is VALID but whose contents are hostile: path traversal / a symlink. Extraction must still refuse."""
    path = os.path.join(dest_dir, "evil-%s.tar.gz" % kind)
    with tarfile.open(path, "w:gz") as tf:
        if kind == "traversal":
            ti = tarfile.TarInfo("pxqe/../../escaped.txt")
            ti.size = 3
            tf.addfile(ti, io.BytesIO(b"bad"))
        else:
            ti = tarfile.TarInfo("pxqe/pxqe")
            ti.type = tarfile.SYMTYPE
            ti.linkname = "/bin/sh"
            tf.addfile(ti)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    ans = sign_answer({"build_id": "evil-" + kind, "edition": "pro", "sha256": sha, "size": os.path.getsize(path), "url": "/v1/download/evil"})
    return ans, path


# ---------------------------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------------------------
class _Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Base(object):
    def __init__(self):
        self.requests = []
        self.srv = None

    def handler(self):
        raise NotImplementedError

    def start(self):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _rec(self, body=b""):
                outer.requests.append({"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body.decode("utf-8", "replace")})

            def do_GET(self):
                self._rec()
                outer.handle(self, "GET", b"")

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                self._rec(body)
                outer.handle(self, "POST", body)
        self.srv = _Srv(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()
            self.srv = None

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port

    def reply(self, h, code, obj, ctype="application/json", extra=None):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        h.send_response(code)
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            h.send_header(k, v)
        h.end_headers()
        try:
            h.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def serve_range(self, h, data, delay=0.0):
        rng = h.headers.get("Range")
        start = 0
        code = 200
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0] or 0)
            if start >= len(data):
                h.send_response(416)
                h.send_header("Content-Length", "0")
                h.end_headers()
                return
            code = 206
        body = data[start:]
        h.send_response(code)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Content-Length", str(len(body)))
        if code == 206:
            h.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
        h.end_headers()
        try:
            for i in range(0, len(body), 8192):
                h.wfile.write(body[i:i + 8192])
                if delay:
                    h.wfile.flush()
                    time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            pass


def safetensors_bytes(n_layers=4, hidden=64, vocab=128, inter=128):
    """A real (small) safetensors file: a JSON header + zero data, BF16."""
    shapes = {"model.embed_tokens.weight": [vocab, hidden], "lm_head.weight": [vocab, hidden]}
    for l in range(n_layers):
        for n, s in (("self_attn.q_proj", [hidden, hidden]), ("self_attn.k_proj", [hidden // 2, hidden]), ("self_attn.v_proj", [hidden // 2, hidden]),
                     ("self_attn.o_proj", [hidden, hidden]), ("mlp.gate_proj", [inter, hidden]), ("mlp.up_proj", [inter, hidden]),
                     ("mlp.down_proj", [hidden, inter])):
            shapes["model.layers.%d.%s.weight" % (l, n)] = s
    hdr, off = {}, 0
    for k, s in shapes.items():
        n = 2
        for d in s:
            n *= d
        hdr[k] = {"dtype": "BF16", "shape": s, "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr).encode()
    hb += b" " * ((-len(hb)) % 8)
    return struct.pack("<Q", len(hb)) + hb + b"\0" * off, sum(_n(s) for s in shapes.values())


def _n(s):
    n = 1
    for d in s:
        n *= d
    return n


def tiny_config(layers=4, hidden=64, vocab=128, inter=128, arch="Qwen3ForCausalLM"):
    return {"architectures": [arch], "model_type": "qwen3", "num_hidden_layers": layers, "hidden_size": hidden, "vocab_size": vocab,
            "intermediate_size": inter, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": hidden // 4,
            "max_position_embeddings": 4096, "tie_word_embeddings": False}


def tiny_repo(license="apache-2.0", layers=4, arch="Qwen3ForCausalLM", gated=False, readme=True):
    st, n = safetensors_bytes(layers)
    files = {"config.json": json.dumps(tiny_config(layers, arch=arch)).encode(), "model.safetensors": st, "tokenizer.json": b"{}"}
    return {"license": license, "files": files, "params": n, "gated": gated, "need_token": False, "redirect": False, "delay": 0.0}


class FakeHF(_Base):
    """repos: {'org/name': tiny_repo()}. The big file can redirect to a CDN on another port (to prove the token stays home)."""

    def __init__(self, repos, cdn=None):
        _Base.__init__(self)
        self.repos = repos
        self.cdn = cdn

    def handle(self, h, method, body):
        u = urllib.parse.urlparse(h.path)
        parts = u.path.strip("/").split("/")
        if parts[:2] == ["api", "models"] and len(parts) >= 4:
            repo = "/".join(parts[2:4])
            r = self.repos.get(repo)
            if r is None:
                return self.reply(h, 404, {"error": "Repository not found"})
            if r.get("need_token") and "authorization" not in {k.lower() for k in h.headers.keys()}:
                return self.reply(h, 401, {"error": "Invalid credentials"})
            return self.reply(h, 200, {"id": repo, "gated": r.get("gated") or False, "private": False, "cardData": {"license": r["license"]},
                                       "tags": ["license:%s" % r["license"]],
                                       "safetensors": {"total": r["params"], "parameters": {"BF16": r["params"]}},
                                       "siblings": [{"rfilename": n, "size": len(b)} for n, b in r["files"].items()]})
        if len(parts) >= 5 and parts[2] == "resolve":
            repo = "/".join(parts[:2])
            fn = urllib.parse.unquote("/".join(parts[4:]))
            r = self.repos.get(repo)
            if r is None or fn not in r["files"]:
                return self.reply(h, 404, {"error": "not found"})
            if r.get("need_token") and "authorization" not in {k.lower() for k in h.headers.keys()}:
                return self.reply(h, 401, {"error": "Invalid credentials"})
            if r.get("redirect") and fn.endswith(".safetensors") and self.cdn:
                h.send_response(302)
                h.send_header("Location", "%s/cdn/%s/%s" % (self.cdn.url, repo, fn))
                h.send_header("Content-Length", "0")
                h.end_headers()
                return
            return self.serve_range(h, r["files"][fn], r.get("delay", 0.0))
        return self.reply(h, 404, {"error": "no route"})


class FakeCDN(_Base):
    def __init__(self, hf):
        _Base.__init__(self)
        self.hf = hf

    def handle(self, h, method, body):
        parts = urllib.parse.urlparse(h.path).path.strip("/").split("/")
        repo, fn = "/".join(parts[1:3]), "/".join(parts[3:])
        r = self.hf.repos.get(repo)
        if r is None or fn not in r["files"]:
            return self.reply(h, 404, {"error": "no"})
        return self.serve_range(h, r["files"][fn], r.get("delay", 0.0))


class FakeLicence(_Base):
    """The licence server's package endpoints, as tools/pxa-licd/PACKAGE.md describes them.
    mode: ok | revoked (403) | noquota (402) | invalid_key (401) | bad_signature | bad_checksum | wrong_platform | bad_manifest_file |
          server_error | link_expired | wrong_edition"""

    def __init__(self, pkgdir, free_build="b-free-2", pro_build="b-pro-2", mode="ok", valid_keys=(GOOD_KEY,)):
        _Base.__init__(self)
        self.pkgdir, self.mode, self.valid_keys = pkgdir, mode, set(valid_keys)
        self.free_build, self.pro_build = free_build, pro_build
        self.archives = {}
        self.pro_fake = {}                    # extra fake.json settings for the Pro package (e.g. needs_runtime=True)
        self.rt_mode = "ok"                   # the GPU runtime: ok | none (404) | tamper | missing_lib | wrong_id | bad_checksum | bad_signature
        self.runtime = None
        self.rt_downloads = 0
        self.rt_delay = 0.0
        self.rt_bulk = 0                      # bytes of incompressible data in the runtime pack (a download big enough to cancel)
        self.downloads = 0
        self.allowed_tiers = ["pxqn2", "pxqn3", "pxqn4", "pxqn5"]      # the key's plan, as `pxqe status` reports it (None = every tier)

    def start(self):
        _Base.start(self)
        self.rebuild()
        return self

    def rebuild(self):
        self.archives = {}
        for ed, b in (("free", self.free_build), ("pro", self.pro_build)):
            seed = OTHER_SEED if self.mode == "bad_signature" else SEED
            extra = {"allowed_tiers": self.allowed_tiers} if (ed == "pro" and self.allowed_tiers is not None) else {}
            if ed == "pro":
                extra.update(self.pro_fake)
            ans, path = build_package(self.pkgdir, ed, b, seed=seed, tamper_file=self.mode == "bad_manifest_file", wrong_platform=self.mode == "wrong_platform", **extra)
            if self.mode == "bad_checksum":            # signed, but for other bytes
                ans = sign_answer(dict(ans, sha256="0" * 64), SEED, sha="0" * 64)
            self.archives[ed] = (ans, path)
        self.runtime = None
        if self.rt_mode != "none":
            self.runtime = build_runtime_pack(self.pkgdir, seed=OTHER_SEED if self.rt_mode == "bad_signature" else SEED, tamper=self.rt_mode == "tamper", bulk=self.rt_bulk,
                                              missing_lib=self.rt_mode == "missing_lib", wrong_id=self.rt_mode == "wrong_id", bad_checksum=self.rt_mode == "bad_checksum")

    def handle(self, h, method, body):
        u = urllib.parse.urlparse(h.path)
        path = u.path
        if self.mode == "server_error":
            return self.reply(h, 503, {"error": "maintenance"})
        if path.startswith("/v1/runtime/download/"):
            if self.mode == "link_expired":
                return self.reply(h, 410, {"error": "this download link has expired"})
            if not self.runtime:
                return self.reply(h, 404, {"error": "unknown runtime pack"})
            self.rt_downloads += 1
            with open(self.runtime[1], "rb") as f:
                return self.serve_range(h, f.read(), self.rt_delay)
        if path == "/v1/runtime/latest" and method == "GET":
            if not self.runtime:
                return self.reply(h, 404, {"error": "no GPU runtime pack is published for linux-x86_64 CUDA 12 yet"})
            return self.reply(h, 200, self.runtime[0])
        if path.startswith("/v1/download/") or path.startswith("/v1/free/download/"):
            if self.mode == "link_expired":
                return self.reply(h, 410, {"error": "this download link has expired"})
            self.downloads += 1
            ed = "pro" if path.startswith("/v1/download/") else "free"
            with open(self.archives[ed][1], "rb") as f:
                return self.serve_range(h, f.read())
        if path == "/v1/free/latest" and method == "GET":
            return self.reply(h, 200, self.archives["free"][0])
        if path in ("/v1/package", "/v1/package/latest") and method == "POST":
            try:
                b = json.loads(body.decode() or "{}")
            except ValueError:
                return self.reply(h, 400, {"error": "bad json"})
            if b.get("key") not in self.valid_keys or self.mode == "invalid_key":
                return self.reply(h, 401, {"error": "invalid key"})
            if self.mode == "revoked":
                return self.reply(h, 403, {"error": "key revoked or expired"})
            if self.mode == "noquota":
                return self.reply(h, 402, {"error": "no encodes left"})
            if path == "/v1/package/latest":
                cur = b.get("build_id")
                return self.reply(h, 200, {"latest": self.pro_build, "edition": "pro", "update_available": cur != self.pro_build, "current": cur})
            m = dict(self.archives["pro"][0])
            if self.mode == "wrong_edition":
                m["edition"] = "free"
            return self.reply(h, 200, m)
        return self.reply(h, 404, {"error": "no route"})


class FakeBoard(_Base):
    """The community high-score board: /v1/scores?bracket= and POST /v1/score. Records every request."""

    def handle(self, h, method, body):
        path = urllib.parse.urlparse(h.path).path
        if path == "/v1/scores":
            return self.reply(h, 200, {"record": None, "top": []})
        if path == "/v1/score" and method == "POST":
            return self.reply(h, 200, {"ok": True, "id": 1, "rank": 1, "status": "ok", "record": None})
        return self.reply(h, 404, {"error": "no route"})
