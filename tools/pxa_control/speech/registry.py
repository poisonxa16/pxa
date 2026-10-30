"""Pinned model registry + safe download for the shared speech service (ONE download for every plugin).

Vendored from Anibal's voice model_registry.py with the same rules:
- licence must be on ALLOWED_LICENCES (the voice licence memo); anything else is refused unless an explicit
  override with a reason is passed (appended to overrides.jsonl next to the cache);
- formats: .onnx / .gguf / .safetensors / ggml whisper .bin (magic-checked) / raw f32 Kokoro voice .bin only;
  pickle checkpoints (.pt/.pth/.ckpt/.pkl or a pickle/zip header) are refused; nothing calls torch.load;
- every file is sha256-verified; the revision must be a 40-hex commit; a file without a pinned sha256 is refused before
  anything is downloaded (no unchecked fetch, and no floating-revision escape);
- an HF token, if any, is read from the environment only and redacted from every error.

Cache: $PXA_SPEECH_MODELS, else <Control config dir>/models/speech (the same folder for Dictate, Scribe,
Satellite and Co-host, so a model is downloaded once).
"""
import hashlib
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "models.json")
ALLOWED_LICENCES = frozenset({"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0",
                              "cc-by-4.0", "cc-by-sa-4.0", "public-domain", "unlicense"})
SAFE_EXT = (".safetensors", ".onnx", ".gguf", ".bin", ".json", ".txt")
PICKLE_EXT = (".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib")
TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_GGML_MAGIC = {b"lmgg", b"ggml", b"ggjt", b"GGUF"}


class RegistryError(Exception):
    pass


class LicenceRefused(RegistryError):
    pass


class UnsafeFormat(RegistryError):
    pass


def config_dir():
    """= pxa_control.core.config_dir() (kept local so the service imports nothing heavy)."""
    d = os.environ.get("PXA_CONTROL_CONFIG_DIR")
    if d:
        return d
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "pxa")


def default_cache():
    return os.environ.get("PXA_SPEECH_MODELS") or os.path.join(config_dir(), "models", "speech")


def load_registry(path=REGISTRY):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for m in data["models"]:
        for k in ("id", "kind", "engine", "repo", "revision", "licence", "files"):
            if k not in m:
                raise RegistryError("registry entry missing %r: %r" % (k, m.get("id")))
        out[m["id"]] = m
    return out


def _token():
    for k in TOKEN_ENV:
        v = os.environ.get(k)
        if v:
            return v
    return None


def redact(msg, token=None):
    s = str(msg)
    t = token or _token()
    if t:
        s = s.replace(t, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]{10,}", "hf_[REDACTED]", s)


def check_licence(entry, override=None, cache=None, clock=time.time):
    """-> 'allowed' | 'override'. override = {"reason": "...", "by": "..."} (recorded)."""
    lic = str(entry.get("licence") or "").lower()
    if entry.get("blocked") and not override:
        raise LicenceRefused("%s is blocked: %s" % (entry["id"], entry["blocked"]))
    if lic in ALLOWED_LICENCES and not entry.get("blocked"):
        return "allowed"
    if not override or not str(override.get("reason") or "").strip():
        raise LicenceRefused("%s: licence %r is not allowed by the voice licence memo; pass an explicit "
                             "override with a reason to use it anyway" % (entry["id"], lic or "unknown"))
    cache = cache or default_cache()
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, "overrides.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": clock(), "id": entry["id"], "repo": entry["repo"], "licence": lic,
                            "reason": override["reason"], "by": override.get("by", "")}) + "\n")
    return "override"


def check_format(path, entry=None):
    """Refuse anything pickle-ish. -> the detected kind."""
    name = path.lower()
    base = os.path.basename(path)
    if name.endswith(PICKLE_EXT):
        raise UnsafeFormat("%s: pickle-based checkpoint refused (use safetensors/onnx/gguf)" % base)
    if not name.endswith(SAFE_EXT):
        raise UnsafeFormat("%s: unsupported file type" % base)
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:1] == b"\x80" or head[:2] == b"PK":
        raise UnsafeFormat("%s: looks like a pickle/torch zip, refused" % base)
    if name.endswith(".gguf"):
        if head[:4] != b"GGUF":
            raise UnsafeFormat("%s: bad GGUF magic" % base)
        return "gguf"
    if name.endswith(".bin"):
        engine = (entry or {}).get("engine", "")
        if engine == "whisper.cpp":
            if head[:4] not in _GGML_MAGIC:
                raise UnsafeFormat("%s: not a ggml model" % base)
            return "ggml"
        if engine == "kokoro-onnx-hf" and "/voices/" in "/" + path.replace(os.sep, "/"):
            return "raw-f32"
        raise UnsafeFormat("%s: .bin only allowed as ggml (whisper.cpp) or Kokoro raw voices" % base)
    return name.rsplit(".", 1)[-1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _default_downloader(repo, filename, revision, token, local_dir):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RegistryError("pip install huggingface_hub (optional extra) to download speech models")
    return hf_hub_download(repo_id=repo, filename=filename, revision=revision, token=token, local_dir=local_dir)


def local_dir(entry, cache=None):
    return os.path.join(cache or default_cache(), entry["id"])


def is_fetched(entry, cache=None, verify=False):
    """All files present (and sha-verified when verify=True)."""
    root = local_dir(entry, cache)
    for f in entry["files"]:
        p = os.path.join(root, f["path"])
        if not os.path.isfile(p) or (f.get("size") and os.path.getsize(p) != f["size"]):
            return False
        if verify and sha256(p) != f["sha256"]:
            return False
    return True


def fetch(entry, cache=None, override=None, downloader=None, log=None):
    """Download (or reuse) every file of an entry; verify licence, format and sha256. -> local directory."""
    cache = cache or default_cache()
    check_licence(entry, override, cache)
    if not re.fullmatch(r"[0-9a-f]{40}", str(entry["revision"])):
        raise RegistryError("%s: revision must be a pinned commit sha" % entry["id"])
    for f in entry["files"]:
        if f["path"].lower().endswith(PICKLE_EXT):
            raise UnsafeFormat("%s: pickle-based checkpoint refused" % f["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256") or "")):
            raise RegistryError("%s: no pinned sha256, refused" % f["path"])
    downloader = downloader or _default_downloader
    root = local_dir(entry, cache)
    os.makedirs(root, exist_ok=True)
    token = _token()
    for f in entry["files"]:
        dest = os.path.join(root, f["path"])
        if not (os.path.exists(dest) and sha256(dest) == f["sha256"]):
            try:
                got = downloader(entry["repo"], f["path"], entry["revision"], token, root)
            except RegistryError:
                raise
            except Exception as e:      # never let the token reach a log line
                raise RegistryError("download %s/%s failed: %s" % (entry["repo"], f["path"], redact(e, token)))
            if os.path.abspath(got) != os.path.abspath(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.replace(got, dest)
            if sha256(dest) != f["sha256"]:
                os.remove(dest)         # a partial/corrupt download of OUR cache file, never a user model
                raise RegistryError("%s: sha256 mismatch, file removed" % f["path"])
        check_format(dest, entry)
        if log:
            log("model %s: %s ok" % (entry["id"], f["path"]))
    return root
