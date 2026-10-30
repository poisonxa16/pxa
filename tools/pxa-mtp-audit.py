#!/usr/bin/env python3
"""pxa-mtp-audit: does every model file carry the draft (MTP / nextn) head its name, card and preset say it does?

Read-only. Reads only the GGUF header (KV block + tensor directory), either from a local file or from a
published file by HTTP range requests (a few MiB per file, never the weights). Nothing is written,
downloaded in full, or uploaded.

What it checks, per file (a split file is read as one: every shard is walked):
  facts     arch, KV <arch>.nextn_predict_layers, number of nextn/mtp tensors
  claims    "implies MTP"  = the file name says mtp/nextn/draft head, a model card next to it (or given with
            --card) says the file carries MTP, or a launcher preset that names the file passes --spec-type mtp
            "implies no MTP" = name/card says no MTP / no draft head / stripped
Findings (exit 1 when any ERROR is found; --strict makes WARN fail too):
  ERROR  LIAR-KV        KV says nextn_predict_layers > 0 but there are no nextn/mtp tensors (head dropped, flag survived)
  ERROR  CLAIM-NO-HEAD  name / card / preset implies MTP but the file has no nextn/mtp tensors
  WARN   HEAD-NO-KV     nextn/mtp tensors present but the KV says 0 or nothing (the engine may ignore the head)
  WARN   NOCLAIM-HEAD   name / card says no MTP but the file carries the head
  INFO   SEPARATE-HEAD  the architecture ships its head as a separate file (e.g. nemotron_h, gemma4): nothing to find here
  OK     the facts and the claims agree

usage:
  pxa-mtp-audit.py PATH_OR_DIR [...]            local .gguf files / directories (recursive)
  pxa-mtp-audit.py --url URL [--url URL ...]    a published file by HTTP range (shards of a split file are found by name)
  pxa-mtp-audit.py --hf poisonxa                every .gguf of every model repo of a Hugging Face account (public API, GET only)
  pxa-mtp-audit.py --manifest FILE.md           every *.gguf path named in a release manifest / ship list (local paths)
  options: --presets DIR (default: presets/ next to this tool's repo), --card FILE (applies to every file),
           --json, --strict, --quiet (only problems)
"""
import argparse, glob, json, os, re, struct, sys, urllib.error, urllib.parse, urllib.request

# arch -> True when the architecture has no NextN layer in the file (the head, if any, is a separate file)
SEPARATE_HEAD_ARCHS = {"nemotron_h", "nemotron_h_moe", "gemma4", "gemma3", "llama", "gemma2"}
GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLOCK = 4 << 20

CLAIM_MTP = re.compile(r"\b(mtp|nextn|next-n|draft[ -]head|multi[ -]token[ -]prediction)\b", re.I)
CLAIM_NO = re.compile(r"(no[ -]?mtp|without[ -]+(the[ -]+)?(mtp|draft)|no[ -]+draft[ -]+head|(mtp|draft head)[ -]+(is[ -]+)?(not[ -]+(included|present|carried)|stripped|removed)|"
                      r"stripped[ -]+(of[ -]+)?(the[ -]+)?(mtp|draft)|nohead|no[ -]+nextn)", re.I)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_OPEN = urllib.request.build_opener(_NoRedirect)
TOKEN = {"v": None}            # a Hugging Face read token (by file path or PXA_HF_TOKEN), sent to huggingface.co ONLY, never printed


def http_open(url, headers=None, timeout=60, hops=5):
    """GET with manual redirects: the bearer token goes to the first host only, never to the CDN it redirects to"""
    first = urllib.parse.urlsplit(url).netloc
    for _ in range(hops):
        h = {"User-Agent": "pxa-mtp-audit/1"}
        h.update(headers or {})
        if TOKEN["v"] and urllib.parse.urlsplit(url).netloc == first and first.endswith("huggingface.co"):
            h["Authorization"] = "Bearer " + TOKEN["v"]
        try:
            return _OPEN.open(urllib.request.Request(url, headers=h), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise
    raise OSError("too many redirects")


class Src:
    """a seekable byte source: a local file or an HTTP URL read in 4 MiB range blocks"""

    def __init__(self, where):
        self.where, self.pos, self.cache, self.size = where, 0, {}, None
        self.http = where.startswith(("http://", "https://"))
        self.requests = 0
        if not self.http:
            self.f = open(where, "rb")
            self.size = os.fstat(self.f.fileno()).st_size

    def _block(self, i):
        b = self.cache.get(i)
        if b is None:
            a = i * BLOCK
            with http_open(self.where, {"Range": f"bytes={a}-{a + BLOCK - 1}"}) as r:
                b = r.read()
                cr = r.headers.get("Content-Range", "")
                if "/" in cr:
                    self.size = int(cr.rsplit("/", 1)[1])
            self.requests += 1
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = b
        return b

    def read(self, n):
        if not self.http:
            self.f.seek(self.pos)
            b = self.f.read(n)
        else:
            out, p, left = [], self.pos, n
            while left > 0:
                i, o = divmod(p, BLOCK)
                blk = self._block(i)
                if o >= len(blk):
                    break
                part = blk[o:o + left]
                out.append(part)
                p += len(part)
                left -= len(part)
            b = b"".join(out)
        if len(b) < n:
            raise EOFError("short read: the header runs past the end of the file")
        self.pos += n
        return b

    def skip(self, n):
        self.pos += n

    def close(self):
        if not self.http:
            self.f.close()


def read_header(where, max_tensors=500000):
    """{"kv": {...}, "names": [...], "n_tensors": N} - strings and small scalars only (long arrays are skipped)"""
    s = Src(where)
    try:
        if s.read(4) != b"GGUF":
            raise ValueError("not a GGUF file (bad magic)")
        s.read(4)
        nt, nkv = struct.unpack("<QQ", s.read(16))
        if nt > max_tensors or nkv > 100000:
            raise ValueError(f"implausible header (n_tensors={nt}, n_kv={nkv})")

        def rstr():
            n, = struct.unpack("<Q", s.read(8))
            if n > (1 << 24):
                raise ValueError("absurd string length")
            return s.read(n).decode("utf-8", "replace")

        def rval(t, keep=True):
            if t == 8:
                return rstr()
            if t == 9:
                et, ln = struct.unpack("<IQ", s.read(12))
                if et in GGUF_FIXED:
                    s.skip(GGUF_FIXED[et] * ln)
                    return None
                vals = [rval(et) for _ in range(ln)]
                return vals if len(vals) <= 8 else None
            b = s.read(GGUF_FIXED[t])
            return {4: lambda: struct.unpack("<I", b)[0], 5: lambda: struct.unpack("<i", b)[0],
                    6: lambda: struct.unpack("<f", b)[0], 7: lambda: bool(b[0]),
                    10: lambda: struct.unpack("<Q", b)[0], 11: lambda: struct.unpack("<q", b)[0]}.get(t, lambda: None)()

        kv = {}
        for _ in range(nkv):
            k = rstr()
            t, = struct.unpack("<I", s.read(4))
            kv[k] = rval(t)
        names = []
        for _ in range(nt):
            names.append(rstr())
            nd, = struct.unpack("<I", s.read(4))
            s.skip(8 * nd + 4 + 8)
        return {"kv": kv, "names": names, "n_tensors": nt, "http_requests": s.requests if s.http else 0}
    finally:
        s.close()


SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$")


def shard_set(where):
    """[paths/urls] of every shard of a split file (the given one when it is not split)"""
    base = where.rsplit("/", 1)[-1] if "/" in where else where
    m = SHARD.match(base)
    if not m:
        return [where]
    head = where[:len(where) - len(base)]
    n = int(m.group(3))
    return [f"{head}{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


def facts_for(where):
    """combined facts of a file or all its shards"""
    shards = shard_set(where)
    kv, names, errs = {}, [], []
    for i, sh in enumerate(shards):
        try:
            h = read_header(sh)
        except Exception as e:                       # noqa: BLE001 - one unreadable shard must not hide the rest
            errs.append(f"{sh.rsplit('/', 1)[-1]}: {e.__class__.__name__}: {e}")
            continue
        if i == 0 or not kv:
            kv = h["kv"]
        names += h["names"]
    arch = kv.get("general.architecture")
    kvn = kv.get(f"{arch}.nextn_predict_layers") if arch else None
    heads = [n for n in names if "nextn" in n or ".mtp" in n or n.startswith("mtp.")]
    return {"arch": arch, "nextn_kv": kvn, "head_tensors": len(heads), "n_tensors": len(names),
            "shards": len(shards), "errors": errs, "name_kv": kv.get("general.name")}


def claims_for(name, card_text, preset_hits, siblings=1):
    """([why it implies MTP], [why it implies no MTP]).
    A card line counts for THIS file when it names the file, or - when the card names no gguf file at all, or this
    is the only model file beside it - when it speaks about the model in general."""
    yes, no = [], []
    base = re.sub(r"\.gguf$", "", name)
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", base)
    if CLAIM_MTP.search(base):
        yes.append(f"file name '{name}'")
    if CLAIM_NO.search(base):
        no.append(f"file name '{name}'")
    lines = (card_text or "").splitlines()
    names_files = any(re.search(r"\.gguf", ln) for ln in lines)
    for ln in lines:
        mentions_me = bool(re.search(r"(?<![\w.-])" + re.escape(stem) + r"(?![\w-])", ln, re.I))
        mentions_other = (not mentions_me) and re.search(r"\.gguf", ln)
        if mentions_other or (names_files and not mentions_me and siblings > 1):
            continue
        low = ln.lower()
        if CLAIM_NO.search(ln):
            no.append("card: " + ln.strip()[:120])
        elif CLAIM_MTP.search(ln) and not re.search(r"(only|not|n't|never|without|when|if|needs?|requires?)\b", low):
            yes.append("card: " + ln.strip()[:120])
    yes += [f"preset {p}" for p in preset_hits]
    return yes, no


def presets_naming(fname, presets_dir):
    """presets whose text names this file (or its base name) AND passes --spec-type mtp"""
    hits = []
    if not presets_dir or not os.path.isdir(presets_dir):
        return hits
    stem = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", fname)
    for p in sorted(glob.glob(os.path.join(presets_dir, "*.env"))):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                t = fh.read()
        except OSError:
            continue
        if stem in t and re.search(r"--spec(-type)?[ =]+mtp|PXA_SPEC\w*=.*mtp", t):
            hits.append(os.path.basename(p))
    return hits


def judge(name, f, card_text="", presets_dir=None, siblings=1):
    """the findings list [(level, code, text)] for one file's facts"""
    out = []
    if f["errors"] and not f["head_tensors"] and f["n_tensors"] == 0:
        gated = any("HTTP Error 40" in e and ("401" in e or "403" in e) for e in f["errors"])
        return [("WARN", "NOT-READ", "gated repository: needs a Hugging Face read token (--hf-token-file); audit the local copy instead")] if gated \
            else [("ERROR", "UNREADABLE", "; ".join(f["errors"]))]
    if f["arch"] == "clip":
        return [("INFO", "PROJECTOR", "vision projector file: no draft head by design")]
    for e in f["errors"]:
        out.append(("WARN", "SHARD-UNREADABLE", e))
    yes, no = claims_for(name, card_text, presets_naming(name, presets_dir), siblings)
    kvn = f["nextn_kv"] or 0
    ht = f["head_tensors"]
    if kvn and not ht:
        out.append(("ERROR", "LIAR-KV", f"nextn_predict_layers={kvn} but 0 nextn/mtp tensors (the head was dropped, the flag survived)"))
    if yes and not ht:
        if f["arch"] in SEPARATE_HEAD_ARCHS:
            out.append(("INFO", "SEPARATE-HEAD", f"arch {f['arch']} ships its head as a separate file ({yes[0]})"))
        else:
            out.append(("ERROR", "CLAIM-NO-HEAD", f"implies MTP ({yes[0]}) but 0 nextn/mtp tensors"))
    if ht and not kvn:
        out.append(("WARN", "HEAD-NO-KV", f"{ht} nextn/mtp tensors but nextn_predict_layers is {f['nextn_kv']!r}"))
    if ht and no:
        out.append(("WARN", "NOCLAIM-HEAD", f"says no MTP ({no[0]}) but carries {ht} head tensors"))
    if not out:
        if f["arch"] in SEPARATE_HEAD_ARCHS and not ht:
            out.append(("INFO", "SEPARATE-HEAD", f"arch {f['arch']}: no head in the file by design"))
        else:
            out.append(("OK", "OK", f"head tensors {ht}, KV {f['nextn_kv']!r}, claims {'MTP' if yes else 'none'}"))
    return out


# ---- sources -----------------------------------------------------------------------------------

def local_files(paths):
    seen, out = set(), []
    for p in paths:
        cands = ([os.path.join(r, n) for r, _, ns in os.walk(p) for n in ns if n.endswith(".gguf")]
                 if os.path.isdir(p) else glob.glob(p) or [p])
        for c in sorted(cands):
            m = SHARD.match(os.path.basename(c))
            key = os.path.join(os.path.dirname(c), m.group(1)) if m else c
            if key not in seen:                      # one entry per split file (shard 1 stands for the set)
                seen.add(key)
                out.append(shard_set(c)[0])
    return out


def manifest_files(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    return sorted({m.group(0).strip("`'\"| ,;)") for m in re.finditer(r"/[^\s|`'\"]+\.gguf", txt)
                   if "..." not in m.group(0) and "*" not in m.group(0) and "$" not in m.group(0)})


def hf_get(url):
    with http_open(url) as r:
        return r.read()


hf_paths = {}


def n_siblings(path):
    """how many model files sit beside this one (a split file counts once, projectors not at all)"""
    d = os.path.dirname(path) or "."
    try:
        return len({SHARD.sub(r"\1", n) for n in os.listdir(d) if n.endswith(".gguf") and "mmproj" not in n}) or 1
    except OSError:
        return 1


def hf_files(account):
    """[(url, repo, path, readme_url or None)] for every .gguf of every model repo of the account (shard 1 only)"""
    out = []
    hf_paths.clear()
    for m in json.loads(hf_get(f"https://huggingface.co/api/models?author={account}&limit=200")):
        repo = m["id"]
        tree = json.loads(hf_get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"))
        paths = [t["path"] for t in tree if t.get("type") == "file"]
        hf_paths[repo] = paths
        for p in paths:
            if p.endswith(".gguf"):
                sm = SHARD.match(os.path.basename(p))
                if sm and int(sm.group(2)) != 1:
                    continue
                d = os.path.dirname(p)
                cards = [c for c in (f"{d}/README.md" if d else "README.md", "README.md") if c in paths]
                out.append((f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(p)}", repo, p,
                            f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(cards[0])}" if cards else None))
    return out


def local_card(path):
    for c in (os.path.join(os.path.dirname(path), "README.md"), os.path.join(os.path.dirname(path), "MODEL-CARD.md")):
        if os.path.isfile(c):
            with open(c, encoding="utf-8", errors="replace") as fh:
                return fh.read()
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--hf", help="Hugging Face account name (public API, GET only)")
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--presets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presets"))
    ap.add_argument("--hf-token-file", help="a file holding a Hugging Face read token (for gated repos); read by path, never printed, sent to huggingface.co only")
    ap.add_argument("--card", help="a model card text applied to every file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.hf_token_file:
        TOKEN["v"] = open(a.hf_token_file, encoding="utf-8").read().strip() or None
    elif os.environ.get("PXA_HF_TOKEN"):
        TOKEN["v"] = os.environ["PXA_HF_TOKEN"].strip()
    jobs = []                                         # (label, where, card_text, siblings)
    extra = open(a.card, encoding="utf-8", errors="replace").read() if a.card else ""
    for f in local_files(a.paths):
        jobs.append((f, f, extra or local_card(f), n_siblings(f)))
    for mf in a.manifest:
        for f in manifest_files(mf):
            jobs.append((f, f, extra or (local_card(f) if os.path.isfile(f) else ""), n_siblings(f)))
    for u in a.url:
        jobs.append((u, u, extra, 1))
    cards = {}
    if a.hf:
        for url, repo, p, cu in hf_files(a.hf):
            if cu and cu not in cards:
                try:
                    cards[cu] = hf_get(cu).decode("utf-8", "replace")
                except Exception:                     # noqa: BLE001
                    cards[cu] = ""
            d = os.path.dirname(p)
            sib = len({SHARD.sub(r"\1", os.path.basename(q)) for q in hf_paths[repo] if q.endswith(".gguf") and os.path.dirname(q) == d
                       and "mmproj" not in q})
            jobs.append((f"{repo}/{p}", url, extra or cards.get(cu, ""), sib))
    results, worst, done = [], 0, set()
    for label, where, card, sib in jobs:
        if label in done:                             # the same file named in several manifests is read once
            continue
        done.add(label)
        if not where.startswith("http") and not os.path.isfile(where):
            res = {"file": label, "findings": [("ERROR", "MISSING-FILE", "path not found (not read)")], "facts": {}}
        else:
            name = os.path.basename(label.split("?")[0])
            fct = facts_for(where)
            res = {"file": label, "facts": {k: fct[k] for k in ("arch", "nextn_kv", "head_tensors", "n_tensors", "shards")},
                   "findings": judge(name, fct, card, a.presets, sib)}
        results.append(res)
        lv = {x[0] for x in res["findings"]}
        worst = max(worst, 2 if "ERROR" in lv else 1 if "WARN" in lv else 0)
    if a.json:
        print(json.dumps(results, indent=1))
    else:
        for r in results:
            top = max((x[0] for x in r["findings"]), key=["OK", "INFO", "WARN", "ERROR"].index)
            if a.quiet and top in ("OK", "INFO"):
                continue
            f = r["facts"]
            print(f"{top:5} {r['file']}  [{f.get('arch')}, kv {f.get('nextn_kv')}, head tensors {f.get('head_tensors')}, shards {f.get('shards')}]")
            for lv, code, txt in r["findings"]:
                if lv not in ("OK",) or not a.quiet:
                    print(f"      {lv} {code}: {txt}")
        n = len(results)
        print(f"-- {n} file(s): {sum(1 for r in results if any(x[0] == 'ERROR' for x in r['findings']))} error, "
              f"{sum(1 for r in results if not any(x[0] == 'ERROR' for x in r['findings']) and any(x[0] == 'WARN' for x in r['findings']))} warn")
    return 1 if worst == 2 or (a.strict and worst == 1) else 0


if __name__ == "__main__":
    sys.exit(main())
