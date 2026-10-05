"""pxa_encode_plan.py - the arithmetic and the facts behind PXA Control's Encode tab. Pure functions, stdlib only,
no GPU and no encoder: everything here is testable without either.

  * the tier catalog (bits per weight, quality-class label, measured KLD where the PXA scoreboard has one);
  * reading a source: a Hugging Face repo id (API + config.json), a local Hugging Face folder (safetensors headers
    + config.json) or a local GGUF; its size, architecture, whether PXA supports it, its licence;
  * fit math: output file size per tier, the context that fits on the target cards, the verdict;
  * the tier recommendation (Free = classic PXQ only; Pro = PXQN first), with every locked tier still shown;
  * the disk / RAM / time estimates for the whole pipeline, including the peak disk with clean-up as it goes.

Numbers that are not obvious carry their source in a comment. Where the measurement does not exist the field is None
and the page says so: this file never invents a speed or a quality figure.
"""

import json
import os
import re
import struct
import urllib.error
import urllib.parse
import urllib.request

GIB = float(1 << 30)
MIB = float(1 << 20)

# ---------------------------------------------------------------------------------------------
# tier catalog
# ---------------------------------------------------------------------------------------------
# bpw = the tier's bits per weight on the quantized linear layers ( and 8; the classic
# PXQ tiers are the equal-bytes originals). cls = the quality-class label users see (owner 2026-09-30: keep the tier names,
# add a measured class). kld = assistant-token KLD vs the BF16 model, Qwen3.8-27B, pxqn-ladder scoreboard (2026-09); None
# means it has not been measured, and the label then says "about" and measured=False.
TIERS = [
    # classic PXQ (the Free encoder)
    {"key": "pxq1", "name": "PXQ1", "family": "pxq", "bpw": 1.25, "cls": "1-bit class", "kld": 10.47, "measured": True,
     "note": "experimental, very lossy"},
    {"key": "pxq2", "name": "PXQ2", "family": "pxq", "bpw": 2.25, "cls": "2-bit class", "kld": 1.155, "measured": True,
     "note": "smallest usable, visibly lossy"},
    {"key": "pxq3", "name": "PXQ3", "family": "pxq", "bpw": 3.25, "cls": "3-bit class", "kld": 0.197, "measured": True,
     "note": "good size/quality trade"},
    {"key": "pxq4", "name": "PXQ4", "family": "pxq", "bpw": 4.25, "cls": "4-bit class", "kld": 0.0189, "measured": True,
     "note": "near-lossless for chat"},
    {"key": "pxq4hq", "name": "PXQ4-HQ", "family": "pxq", "bpw": 4.5, "cls": "4.5-bit class", "kld": None, "measured": False,
     "note": "4-bit with finer scales"},
    {"key": "pxq6", "name": "PXQ6", "family": "pxq", "bpw": 5.25, "cls": "6-bit class", "kld": 0.0076, "measured": True,
     "note": "closest to the original"},
    {"key": "pxquniversal", "name": "PXQ_UNIVERSAL", "family": "pxq", "bpw": 4.5, "cls": "4-bit class", "kld": None, "measured": False,
     "note": "a mix of tiers chosen per tensor; the size varies with the model", "needs_map": True},
    # PXQN (Pro): LDLQ-encoded, same bytes as the classic tier above, lower error
    {"key": "pxqn1", "name": "PXQN1", "family": "pxqn", "bpw": 1.25, "cls": "1-bit class", "kld": 2.205, "measured": True,
     "note": "experimental, lossy"},
    {"key": "pxqn2", "name": "PXQN2", "family": "pxqn", "bpw": 2.25, "cls": "3-bit class", "kld": 0.161, "measured": True,
     "note": "performs like a classic 3-bit at 2-bit size"},
    {"key": "pxqn3", "name": "PXQN3", "family": "pxqn", "bpw": 3.25, "cls": "3.5-bit class", "kld": 0.035, "measured": True,
     "note": ""},
    {"key": "pxqn3s8", "name": "PXQN3S8", "family": "pxqn", "bpw": 3.5, "cls": "3.5-bit class", "kld": None, "measured": False,
     "note": "finer scales than PXQN3"},
    {"key": "pxqn3bal", "name": "PXQN3bal", "family": "pxqn", "bpw": 3.6, "cls": "4-bit class", "kld": 0.0238, "measured": True,
     "note": "a mix of PXQN3 and PXQN4 per tensor", "only_if_listed": True},
    {"key": "pxqn4", "name": "PXQN4", "family": "pxqn", "bpw": 4.25, "cls": "6-bit class", "kld": 0.0079, "measured": True,
     "note": "beats a classic 6-bit at 4-bit size"},
    {"key": "pxqn4s8", "name": "PXQN4S8", "family": "pxqn", "bpw": 4.5, "cls": "6-bit class", "kld": None, "measured": False,
     "note": "finer scales than PXQN4"},
    {"key": "pxqn5", "name": "PXQN5", "family": "pxqn", "bpw": 5.25, "cls": "Q6_K class", "kld": 0.0022, "measured": True,
     "note": "the highest quality"},
]
TIER_BY_KEY = {t["key"]: t for t in TIERS}
_ALIASES = {"pxq4u": "pxq4", "pxq4-hq": "pxq4hq", "pxqu": "pxq4"}
KLD_SOURCE = "Qwen3.8-27B, assistant-token KLD vs BF16, PXA pxqn-ladder scoreboard (2026-09)"


def tier_key(name):
    k = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    return _ALIASES.get(k, k)


def tier_entry(name):
    """The catalog row for a tier name, or a generic row for a name this Control has never heard of (a newer encoder)."""
    k = tier_key(name)
    if k in TIER_BY_KEY:
        return dict(TIER_BY_KEY[k])
    m = re.search(r"(\d+(?:\.\d+)?)", k)
    bpw = float(m.group(1)) + 0.25 if m and float(m.group(1)) < 8 else None
    return {"key": k, "name": str(name), "family": "pxqn" if k.startswith("pxqn") else "pxq", "bpw": bpw,
            "cls": "unrated", "kld": None, "measured": False, "note": "a tier newer than this PXA Control knows"}


# ---------------------------------------------------------------------------------------------
# source: parsing, Hugging Face, local folders
# ---------------------------------------------------------------------------------------------
HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,95}/[A-Za-z0-9][A-Za-z0-9._\-]{0,95}$")
HF_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/([A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-]*)(?:[/?#].*)?$")
WANT_FILE_RE = re.compile(r"(\.safetensors|\.safetensors\.index\.json|config\.json|generation_config\.json|tokenizer[^/]*|"
                          r"vocab[^/]*|merges\.txt|special_tokens_map\.json|added_tokens\.json|chat_template[^/]*|preprocessor_config\.json|"
                          r"[^/]*\.tiktoken|[^/]*\.model)$", re.I)


class SourceError(Exception):
    """A plain sentence for the user."""


def parse_source(text):
    """'org/name', a huggingface.co URL, or a path -> ('hf', repo) | ('path', absolute path)."""
    t = (text or "").strip().strip("\"'")
    if not t:
        raise SourceError("Paste a Hugging Face model (like Qwen/Qwen3-1.7B) or pick a folder or GGUF file on this computer.")
    if len(t) > 4096 or "\x00" in t:
        raise SourceError("That is not a model name or a path.")
    m = HF_URL_RE.match(t)
    if m:
        return "hf", m.group(1)
    if t.startswith(("/", "~", "./", "../")) or re.match(r"^[A-Za-z]:[\\/]", t):
        return "path", os.path.abspath(os.path.expanduser(t))
    if HF_ID_RE.match(t) and not os.path.exists(t):
        return "hf", t
    if os.path.exists(t):
        return "path", os.path.abspath(t)
    raise SourceError("That is neither a Hugging Face model id (it looks like organisation/name) nor a path that exists on this computer.")


def hf_endpoint():
    return (os.environ.get("HF_ENDPOINT") or os.environ.get("PXA_HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    for p in (os.path.join(os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "token"),):
        try:
            with open(p) as f:
                return f.read().strip()
        except OSError:
            pass
    return ""


class _StripAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face sends file downloads to a CDN on another host. The token must not follow it there (urllib would copy
    the Authorization header to the new host)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            for h in list(new.headers):
                if h.lower() == "authorization":
                    del new.headers[h]
            for h in list(new.unredirected_hdrs):
                if h.lower() == "authorization":
                    del new.unredirected_hdrs[h]
        return new


class HF(object):
    """The little of the Hugging Face API the wizard needs. The token (if any) goes only to this endpoint."""

    def __init__(self, endpoint=None, token=None):
        self.endpoint = (endpoint or hf_endpoint()).rstrip("/")
        self.token = hf_token() if token is None else token

    def _req(self, url, headers=None):
        h = {"User-Agent": "pxa-control-encode"}
        h.update(headers or {})
        if self.token and url.startswith(self.endpoint):
            h["Authorization"] = "Bearer " + self.token
        return urllib.request.Request(url, headers=h)

    def open(self, url, headers=None, timeout=30):
        return urllib.request.build_opener(_StripAuthRedirect()).open(self._req(url, headers), timeout=timeout)

    def _fail(self, e, repo):
        if isinstance(e, urllib.error.HTTPError):
            if e.code == 404:
                return SourceError("Hugging Face has no model called %s (check the spelling, or it may be private: set HF_TOKEN)." % repo)
            if e.code in (401, 403):
                return SourceError("%s is gated or private. Accept its licence on huggingface.co with your account, then set "
                                   "HF_TOKEN to a read token and try again." % repo)
            if e.code == 429:
                return SourceError("Hugging Face is rate-limiting this computer. Wait a minute and try again.")
            return SourceError("Hugging Face answered %d for %s." % (e.code, repo))
        return SourceError("Cannot reach Hugging Face (%s). Check the connection, or pick a model already on this computer." % type(e).__name__)

    def model(self, repo):
        url = "%s/api/models/%s?blobs=true" % (self.endpoint, urllib.parse.quote(repo, safe="/"))
        try:
            with self.open(url) as r:
                return json.loads(r.read(8 << 20).decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
            raise self._fail(e, repo)

    def text(self, repo, path, limit=1 << 20):
        url = "%s/%s/resolve/main/%s" % (self.endpoint, urllib.parse.quote(repo, safe="/"), urllib.parse.quote(path))
        try:
            with self.open(url) as r:
                return r.read(limit).decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return None
            raise self._fail(e, repo)


# ---------------------------------------------------------------------------------------------
# architecture support
# ---------------------------------------------------------------------------------------------
# (regex on the HF `architectures[0]`, family, level, plain note). First match wins. supported = PXA tests it; beta = it runs, rougher;
# untested = the encoder handles any plain linear layers but PXA has not run this family. Owner 2026-09-13: Qwen first release,
# Gemma supported, GLM beta.
ARCH_SUPPORT = [
    (r"^Qwen[23]", "Qwen", "supported", "Qwen3 / Qwen3.8 dense, MoE and hybrid models are PXA's best-tested family."),
    (r"^Gemma", "Gemma", "supported", "Gemma models are supported."),
    (r"^(Glm|ChatGLM)", "GLM", "beta", "GLM runs, but it is a beta family: expect rough edges."),
    (r"^(Llama|Mistral|Mixtral|Ministral)", "Llama-style", "untested", "Llama-style models should work; PXA has not tested them as much."),
]


def arch_support(arch, converter_names=None):
    """-> {family, level, text}. level: supported | beta | untested | unsupported | unknown."""
    if not arch:
        return {"family": None, "level": "unknown", "text": "Cannot tell the architecture from this model."}
    for rx, fam, level, note in ARCH_SUPPORT:
        if re.match(rx, arch, re.I):
            return {"family": fam, "level": level, "text": note}
    if converter_names is not None and arch not in converter_names:
        return {"family": arch, "level": "unsupported",
                "text": "PXA's converter does not know the %s architecture, so it cannot be turned into a GGUF." % arch}
    return {"family": arch, "level": "untested", "text": "PXA has not tested the %s architecture; it may work." % arch}


def converter_arch_names(path):
    """Architecture names the engine's convert_hf_to_gguf.py registers (read as text, never imported)."""
    try:
        with open(path, errors="replace") as f:
            blob = f.read()
    except OSError:
        return None
    names = set()
    for m in re.finditer(r"\.register\(([^)]*)\)", blob):
        names.update(re.findall(r"[\"']([A-Za-z0-9_]+)[\"']", m.group(1)))
    return names or None


# ---------------------------------------------------------------------------------------------
# licence of the SOURCE model (not the quantizer's)
# ---------------------------------------------------------------------------------------------
# level: ok = may be shared; conditions = may be shared under the model's own terms; noncommercial; noderivs = a quantized
# copy may not be shared; unknown. forbids = True when sharing the quantized file is not allowed or not clearly allowed.
_LIC = [
    (r"^(apache|mit|bsd|isc|unlicense|cc0|cc-by-4|cc-by-3|bsl|zlib|openrail\+\+?$)", "ok", "A permissive licence: you may share a quantized copy."),
    (r"^(cc-by-sa|gpl|agpl|lgpl|mpl|odc|ofl)", "conditions", "A share-alike / copyleft licence: you may share a quantized copy under the same terms."),
    (r"^(llama|gemma|qwen$|qwen-|deepseek|openrail|bigscience|creativeml|tongyi|yi-|glm|mistral|ai2|falcon|stabilityai|nvidia|health|c-uda|cc-by-sa)",
     "conditions", "A custom model licence: sharing is allowed under its own conditions (attribution, naming, acceptable use). Read it."),
    (r"nc-nd|-nd(-|$)|no-?deriv|polyform|noredist|no-redist|proprietary|all-rights|research-only|non-?redistrib", "noderivs",
     "This licence does not let you share derived files such as a quantized copy."),
    (r"-nc|noncommercial|non-commercial|research", "noncommercial", "Non-commercial: you may share a quantized copy, but not sell it or use it commercially."),
]


def classify_licence(lic):
    lic = (lic or "").strip().lower()
    if not lic or lic in ("other", "unknown", "none"):
        return {"id": lic or "unknown", "level": "unknown", "forbids": None,
                "text": "The licence is not stated or is custom. Read the model's own licence before you share a quantized copy."}
    for rx, level, text in _LIC:
        if level == "noderivs" and re.search(rx, lic):
            return {"id": lic, "level": level, "forbids": True, "text": text}
    for rx, level, text in _LIC:
        if level != "noderivs" and re.search(rx, lic):
            return {"id": lic, "level": level, "forbids": False, "text": text}
    return {"id": lic, "level": "unknown", "forbids": None,
            "text": "Unfamiliar licence (%s). Read it before you share a quantized copy." % lic[:40]}


def licence_from_card(text):
    """The `license:` of a README/model-card front-matter, or None."""
    if not text:
        return None
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return None
    for ln in m.group(1).splitlines():
        mm = re.match(r"^license\s*:\s*['\"]?([^'\"#]+?)['\"]?\s*$", ln.strip(), re.I)
        if mm:
            return mm.group(1).strip()
    return None


# ---------------------------------------------------------------------------------------------
# reading a model: config.json, safetensors headers, GGUF header
# ---------------------------------------------------------------------------------------------
def read_safetensors_header(path):
    """-> {tensor name: (dtype, shape)}; reads 8 bytes + the JSON header only."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > (64 << 20):
            raise SourceError("%s is not a valid safetensors file." % os.path.basename(path))
        h = json.loads(f.read(n).decode("utf-8"))
    return {k: (v.get("dtype"), v.get("shape") or []) for k, v in h.items() if k != "__metadata__" and isinstance(v, dict)}


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def params_from_headers(headers):
    """{tensor: (dtype, shape)} -> (total params, embedding params, lm_head params)."""
    total = emb = head = 0
    for name, (dt, shape) in headers.items():
        n = _numel(shape)
        total += n
        low = name.lower()
        if "embed_tokens" in low or low.endswith("tok_embeddings.weight") or low.endswith("wte.weight") or low.endswith("token_embd.weight"):
            emb += n
        elif low.endswith("lm_head.weight") or low == "output.weight":
            head += n
    return total, emb, head


def kv_bytes_per_token_config(cfg, kv_bytes=2):
    """KV cache bytes per token from a Hugging Face config.json (K and V, `kv_bytes` bytes per element). Only the layers that
    keep a KV cache count: hybrid models (full attention every Nth layer, the rest linear attention) keep far fewer. MLA models
    (a compressed latent) use their latent size. Arithmetic, not a measurement -> returns None when the fields are missing."""
    t = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    try:
        L = int(t.get("num_hidden_layers") or t.get("n_layer") or 0)
        if not L:
            return None
        kvl = L
        lt = t.get("layer_types")
        if isinstance(lt, list) and lt:
            kvl = sum(1 for x in lt if "full" in str(x) or str(x) in ("attention", "global"))
            kvl = kvl or L
        elif t.get("full_attention_interval"):
            kvl = max(1, L // int(t["full_attention_interval"]))
        if t.get("kv_lora_rank"):
            return int(kvl * (int(t["kv_lora_rank"]) + int(t.get("qk_rope_head_dim") or 0)) * kv_bytes)
        heads = int(t.get("num_attention_heads") or 0)
        kvh = int(t.get("num_key_value_heads") or heads or 0)
        hd = t.get("head_dim")
        if hd is None and t.get("hidden_size") and heads:
            hd = int(t["hidden_size"]) // heads
        if not (kvh and hd):
            return None
        return int(kvl * 2 * kvh * int(hd) * kv_bytes)
    except (TypeError, ValueError):
        return None


def facts_from_config(cfg):
    t = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    arch = None
    a = cfg.get("architectures")
    if isinstance(a, list) and a:
        arch = str(a[0])
    return {"arch": arch, "model_type": cfg.get("model_type"), "layers": t.get("num_hidden_layers"),
            "hidden": t.get("hidden_size"), "inter": t.get("intermediate_size") or t.get("moe_intermediate_size"),
            "vocab": t.get("vocab_size") or cfg.get("vocab_size"),
            "tie": bool(t.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))),
            "n_ctx_train": t.get("max_position_embeddings"), "kv_bytes_tok": kv_bytes_per_token_config(cfg),
            "experts": t.get("num_experts") or t.get("n_routed_experts") or 0}


def param_label(n):
    if not n:
        return "?"
    if n >= 1e9:
        return ("%.0fB" if n >= 1e10 else "%.1fB") % (n / 1e9)
    return "%.0fM" % (n / 1e6)


# ---------------------------------------------------------------------------------------------
# fit math
# ---------------------------------------------------------------------------------------------
# What a tier file weighs, from the tensor counts (checked against the Swift 27B PXQN4 skeleton: estimate 15.6 GB, file 15.72 GB):
# the quantized linear layers at bpw; the embedding table at q6_K (6.5625 bpw); the output head at q8_0 (8.5 bpw) unless tied;
# a few small tensors (norms, router) at 1.5% of the quantized part.
EMBED_BPW = 6.5625
HEAD_BPW = 8.5
SMALL_FRACTION = 0.015
HEADROOM_MIB_PER_CARD = 1200      # launcher HEADROOM RULE (measured 2026-08-28): ~1200 MiB free per card after load


def file_bytes(params, emb, head, bpw, tied=False):
    """Estimated GGUF size of one tier."""
    if not params or not bpw:
        return None
    emb = emb or 0
    head = 0 if tied else (head or 0)
    q = max(0, params - emb - head)
    return int(q * bpw / 8 * (1 + SMALL_FRACTION) + emb * EMBED_BPW / 8 + head * HEAD_BPW / 8)


def tier_fraction(params, emb, head, bpw, tied=False):
    """The share of the file that would be in the chosen PXQ tier (the public quantizer refuses a file under 50%: on a small model the
    embedding table and the head are a large part of it). None when it cannot be told."""
    if not params or not bpw:
        return None
    total = file_bytes(params, emb, head, bpw, tied)
    q = max(0, params - (emb or 0) - (0 if tied else (head or 0))) * bpw / 8
    return (q / total) if total else None


def fit(size_bytes, kv_bytes_tok, total_vram_mib, n_cards, n_ctx_train=None):
    """-> {ctx, verdict, text}. ctx = the context (tokens) that fits after the weights and the headroom; verdict fits|tight|no."""
    if not size_bytes or not total_vram_mib:
        return {"ctx": None, "verdict": "unknown", "text": "no cards selected"}
    free = total_vram_mib * MIB - size_bytes - HEADROOM_MIB_PER_CARD * MIB * max(1, n_cards)
    if free <= 0:
        return {"ctx": 0, "verdict": "no", "text": "the weights alone (%.1f GiB) do not fit in %.1f GiB with headroom" % (
            size_bytes / GIB, total_vram_mib / 1024.0)}
    if not kv_bytes_tok:
        return {"ctx": None, "verdict": "fits", "text": "weights fit; the context depends on the model"}
    ctx = int(free // kv_bytes_tok)
    if n_ctx_train:
        ctx = min(ctx, int(n_ctx_train))
    ctx = (ctx // 1024) * 1024
    # a context limited by what the model was trained for, not by memory, is a fit
    verdict = "fits" if (ctx >= 8192 or (n_ctx_train and ctx >= (int(n_ctx_train) // 1024) * 1024)) else ("tight" if ctx >= 2048 else "no")
    return {"ctx": ctx, "verdict": verdict, "text": "about %s tokens of context (f16 KV estimate)" % ("%dk" % (ctx // 1024) if ctx >= 1024 else ctx)}


def ctx_label(ctx):
    if ctx is None:
        return "?"
    return "%dk" % (ctx // 1024) if ctx >= 1024 else str(ctx)


# ---------------------------------------------------------------------------------------------
# measured speed cells (only what PXA has published; None otherwise)
# ---------------------------------------------------------------------------------------------
def load_cells(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return [c for c in d.get("cells", []) if isinstance(c, dict)]
    except (OSError, ValueError):
        return []


def find_cell(cells, tier, params, card_class, n_cards, moe=None):
    """A measured decode cell for this tier on this kind of card set, or None. A cell matches when the tier (or its family
    for 'any' cells), the card class and count are equal and the model size is within 15%."""
    if not params:
        return None
    for c in cells:
        if tier_key(c.get("tier", "")) != tier:
            continue
        if c.get("card_class") != card_class or int(c.get("n_cards", 0)) != int(n_cards):
            continue
        cp = c.get("params")
        if not cp or abs(float(cp) - params) / float(cp) > 0.15:
            continue
        if moe is not None and c.get("moe") is not None and bool(c["moe"]) != bool(moe):
            continue
        return c
    return None


# ---------------------------------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------------------------------
def build_tiers(src, cards, info, cells=None, allowed=None):
    """The tier table for step 2. src: the inspection dict (params, emb, head, tied, kv_bytes_tok, n_ctx_train); cards: the
    selected cards [{vram_mib, class}]; info: the chosen encoder's info (edition, tiers) or None.
    Every catalog tier is listed (a tier flagged only_if_listed appears only when the build lists it). A tier is `available` when the
    encoder lists it; otherwise `locked` (Free + a PXQN tier: 'Supporter feature'; Pro without it: 'not in your plan'; no encoder at
    all: 'needs an encoder'). `allowed` (Pro: the tiers the licence key's plan includes, from the licence server) locks the PXQN tiers
    outside it with the same 'not in your plan' reason; None = no information, nothing is locked on that account."""
    have = set((info or {}).get("tiers") or [])
    edition = (info or {}).get("edition")
    total_mib = sum(c.get("vram_mib", 0) for c in cards)
    n = len(cards)
    cls = cards[0].get("class") if cards and len({c.get("class") for c in cards}) == 1 else None
    rows = []
    # tiers the encoder lists that this Control does not know are appended (the build decides, not this table)
    keys = [t["key"] for t in TIERS if not t.get("only_if_listed") or t["key"] in have] + [k for k in have if k not in TIER_BY_KEY]
    for k in keys:
        t = tier_entry(k)
        size = file_bytes(src.get("params"), src.get("emb"), src.get("head"), t.get("bpw"), src.get("tied"))
        f = fit(size, src.get("kv_bytes_tok"), total_mib, n, src.get("n_ctx_train")) if size else {"ctx": None, "verdict": "unknown", "text": ""}
        avail = k in have
        locked, why = False, None
        if avail and allowed is not None and t["family"] == "pxqn" and k not in allowed:
            avail = False
            locked, why = True, "Not included in your plan"
        if avail and t.get("needs_map"):
            avail = False                      # offered by the build, but it needs a per-tensor tier map Control cannot write
            why = "Needs a per-tensor tier map: use the pxqe command line for this one."
        elif not avail and not locked:
            locked = True
            if not info:
                why = "Install an encoder to make this tier."
            elif edition == "free" and t["family"] == "pxqn":
                why = "Supporter feature"
            elif edition == "pro":
                why = "Not included in your plan"
            else:
                why = "This encoder does not make this tier"
        cell = find_cell(cells or [], k, src.get("params"), cls, n, src.get("moe")) if (cells and cls) else None
        rows.append({"key": k, "name": t["name"], "family": t["family"], "bpw": t["bpw"], "cls": t["cls"], "kld": t["kld"],
                     "measured": t["measured"], "note": t["note"], "size_bytes": size, "ctx": f["ctx"], "ctx_text": f["text"],
                     "verdict": f["verdict"], "available": avail, "locked": locked, "locked_reason": why,
                     "tier_fraction": tier_fraction(src.get("params"), src.get("emb"), src.get("head"), t.get("bpw"), src.get("tied")),
                     "decode_tps": cell.get("decode_tps") if cell else None, "cell_source": cell.get("source") if cell else None})
    return rows


def recommend(rows, edition):
    """Mark up to three tiers: 'best' (highest quality that fits with a usable context), 'roomy' (best quality that leaves
    32k+ context), 'small' (the smallest sensible one). Free picks among classic tiers only (that is all it lists as available);
    Pro prefers PXQN. Returns the list of keys in display order and annotates rows with `role`."""
    ok = [r for r in rows if r["available"] and r["verdict"] in ("fits", "tight") and r["bpw"]]

    def pref(r):                       # higher is better: family preference first (Pro: PXQN), then bits per weight
        fam = 1 if (edition == "pro" and r["family"] == "pxqn") else 0
        return (fam, r["bpw"])
    roles = {}
    if ok:
        good = [r for r in ok if r["verdict"] == "fits"] or ok
        best = max(good, key=pref)
        roles[best["key"]] = "best"
        roomy = [r for r in ok if (r["ctx"] or 0) >= 32768 and r["key"] not in roles]
        if roomy and (best["ctx"] or 0) < 32768:
            roles[max(roomy, key=pref)["key"]] = "roomy"
        small = [r for r in ok if r["bpw"] >= 2 and r["key"] not in roles] or [r for r in ok if r["key"] not in roles]
        if small:
            s = min(small, key=lambda r: (r["bpw"], 0 if r["family"] == "pxqn" and edition == "pro" else 1))
            if s["bpw"] < best["bpw"]:
                roles[s["key"]] = "small"
    for r in rows:
        r["role"] = roles.get(r["key"])
    order = {"best": 0, "roomy": 1, "small": 2}
    return sorted(roles, key=lambda k: order[roles[k]])


# ---------------------------------------------------------------------------------------------
# disk, RAM, VRAM and time for the whole pipeline
# ---------------------------------------------------------------------------------------------
# Measured on the 1.7B reference (7.7 GB dump, 5.3 GB Hessians, 2.0B params) and the Swift 27B encode (119 GB dump, 94 GB Hessians,
# 27.3B params, 10240-token calibration): dump ~4.4 GB per billion parameters, Hessians ~3.4 GB per billion.
DUMP_GB_PER_B = 4.4
HESS_GB_PER_B = 3.4
BF16_BYTES = 2.0
Q8_BYTES = 1.0625                 # q8_0: 34 bytes per 32 weights

# Seconds per billion parameters (Swift 27B chain, pipe.log 2026-09-30/10-01, 36-thread host, V100 cards): q8 ~88, skeleton ~56,
# dump 48-215 on two V100 (the slow end was a contended window), Hessians ~132, LDLQ ~110 (~55 per output file). Convert is
# I/O bound: ~30 assumed. Ranges are the measured spread, widened by 20% at the top.
SEC_PER_B = {"convert": (20, 45), "reference": (70, 110), "skeleton": (45, 70), "dump": (50, 220), "hessians": (100, 160), "encode": (50, 120),
             "quantize": (60, 110)}
DOWNLOAD_MBPS_ASSUMED = (30, 120)   # MB/s, home connection spread


def stage_sizes(src, tier_bytes, plan_stages):
    """Bytes live on disk per stage (inputs + outputs it touches), and the peak with clean-up after each stage.
    -> {"stage": {...}, "peak": bytes, "work_total": bytes, "out": bytes}"""
    p = (src.get("params") or 0) / 1e9
    dl = src.get("download_bytes") or 0
    bf16 = int(p * 1e9 * BF16_BYTES)
    q8 = int(p * 1e9 * Q8_BYTES)
    skel = tier_bytes or 0
    dump = int(p * DUMP_GB_PER_B * 1e9)
    hess = int(p * HESS_GB_PER_B * 1e9)
    # live set while each stage runs (clean-up: the download goes after convert, the BF16 after the reference, the dump after hessians)
    ldlq = "hessians" in plan_stages
    live = {"download": dl, "convert": dl + bf16, "reference": bf16 + q8, "skeleton": q8 + skel, "dump": q8 + skel + dump,
            "hessians": q8 + skel + dump + hess, "encode": q8 + skel + (hess if ldlq else 0), "verify": q8 + skel + (hess if ldlq else 0)}
    if src.get("kind") == "gguf":
        live["reference"] = (src.get("download_bytes") or bf16) + q8 if "reference" in plan_stages else 0
    # the classic path: one CPU quantize from the BF16 (or the source GGUF) straight to the tier file
    live["quantize"] = (bf16 if src.get("kind") != "gguf" else (src.get("download_bytes") or src.get("size_bytes") or bf16)) + skel
    if "quantize" in plan_stages:
        live["verify"] = skel
    live = {k: v for k, v in live.items() if k in plan_stages}
    peak = max(live.values()) if live else 0
    return {"live": live, "peak": peak, "out": skel, "keep_extra": 0 if "quantize" in plan_stages else q8 + (hess if ldlq else 0)}


def time_estimate(src, plan_stages, n_gpu_cards=1, bpw_factor=1.0):
    """-> {"stages": {stage: (low_s, high_s)}, "low", "high"}. Rough, from the measured spread (see SEC_PER_B)."""
    p = (src.get("params") or 0) / 1e9
    out = {}
    for s in plan_stages:
        if s == "download":
            b = src.get("download_bytes") or 0
            out[s] = (b / (DOWNLOAD_MBPS_ASSUMED[1] * 1e6), b / (DOWNLOAD_MBPS_ASSUMED[0] * 1e6))
        elif s in SEC_PER_B:
            lo, hi = SEC_PER_B[s]
            scale = (1.0 / max(1, n_gpu_cards)) if s == "dump" else 1.0
            out[s] = (p * lo * scale, p * hi * scale)
        elif s == "verify":
            out[s] = (10 + p * 1.5, 30 + p * 4)
    return {"stages": out, "low": sum(v[0] for v in out.values()), "high": sum(v[1] for v in out.values())}


def human_dur(s):
    s = int(max(0, s))
    if s < 90:
        return "%d s" % s
    if s < 5400:
        return "%d min" % round(s / 60.0)
    h = s / 3600.0
    return ("%.1f h" % h) if h < 36 else ("%.1f days" % (h / 24.0))


def human_bytes(n):
    n = float(n or 0)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return ("%.0f %s" % (n, u)) if u == "B" else ("%.1f %s" % (n, u))
        n /= 1024.0


# VRAM the encode (LDLQ) needs on ONE card: three K x K fp32 matrices for the largest Hessian plus the biggest tensor in fp32 x4.
# (measured: a 27B encode ran on a 16 GB V100; this formula gives ~5 GiB for it).
def encode_vram_bytes(hidden, inter, rows=None):
    k = max(int(inter or 0), int(hidden or 0)) or 8192
    r = int(rows or k)
    return int(3 * k * k * 4 + 4 * r * k * 4)


def ram_need_bytes(hidden, inter):
    """Host RAM for the Hessian stage and the encode: the largest Hessian twice plus slack."""
    k = max(int(inter or 0), int(hidden or 0)) or 8192
    return int(2 * k * k * 4 + (2 << 30))
