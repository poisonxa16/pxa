"""PXA Control: one-click downloads of the PXA models published on Hugging Face.

Lists the repositories under a Hugging Face user (`poisonxa` by default), shows the GGUF files each holds
with their size, and downloads one into a model folder. The download resumes where it stopped, checks the
sha256 the repository itself publishes, fetches every shard of a multi-part GGUF and puts the
`.expert-counts.csv` that must travel beside a GGUF next to it (a missing .csv is not a failure: the engine
falls back to the map it ships).

Auth: the page may hold a Hugging Face token (control.json `hf_token`, or PXA_CONTROL_HF_TOKEN). It is sent
to huggingface.co and to nothing else - the redirect to the file's CDN drops it - it is never logged, and it
is never handed back to the page. A repository the token cannot read is reported as supporter-only, so the
page can show a padlock and the Ko-fi link instead of a Download button that would fail.

Stdlib only (urllib), like the rest of PXA Control.
"""

import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_ORG = "poisonxa"
HF = "https://huggingface.co"
LIST_TTL = 600.0                  # seconds a listing is reused before it is asked for again
CHUNK = 1024 * 256
UA = "pxa-control"
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


class HFError(Exception):
    """Something the page should show as a sentence: a repo it cannot read, a file it cannot reach."""

    def __init__(self, msg, code="hf_error", status=502):
        super().__init__(msg)
        self.code = code
        self.status = status


class _NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Follow HF's redirect to the file's CDN, but do not carry the bearer token to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(HF).netloc:
            new.headers.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_NoAuthRedirect)


def _open(url, token=None, headers=None, timeout=60, method=None):
    h = {"User-Agent": UA, "Accept": "application/json"}
    h.update(headers or {})
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=h, method=method)
    try:
        return _OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HFError("this model is for supporters: its repository needs a Hugging Face token that can read it",
                          code="needs_token", status=403)
        if e.code == 404:
            raise HFError("Hugging Face has no such file any more (404)", code="not_found", status=404)
        raise HFError("Hugging Face answered %s %s" % (e.code, e.reason), status=502)
    except urllib.error.URLError as e:
        raise HFError("cannot reach huggingface.co: %s" % (getattr(e, "reason", e),), code="offline", status=502)


def _json(url, token=None, timeout=45):
    with _open(url, token=token, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _authored(url, token):
    """Bearer token, unless HF would reject an empty one (it wants the header absent, not blank)."""
    return token or None


def list_repos(org=DEFAULT_ORG, token=None):
    """Every repository under this user, newest first, with the GGUF files each one holds.

    Sizes and sha256 come from the repository's own tree listing, so they are the publisher's numbers,
    not ours. A repository whose first GGUF cannot be read anonymously is marked supporter-only.
    """
    url = "%s/api/models?author=%s&limit=200&full=false" % (HF, urllib.parse.quote(org))
    repos = _json(url, token=_authored(url, token))
    if not isinstance(repos, list):
        raise HFError("huggingface.co answered something that is not a list of repositories")
    out = []
    for r in sorted(repos, key=lambda m: (m.get("lastModified") or ""), reverse=True):
        rid = r.get("id") or ""
        if not rid or (r.get("private") and not token):
            continue
        try:
            files = _files(rid, token)
        except HFError as e:
            out.append({"repo": rid, "files": [], "error": str(e), "downloads": r.get("downloads"),
                        "updated": r.get("lastModified")})
            continue
        if not files:
            continue
        out.append({"repo": rid, "files": files, "downloads": r.get("downloads"), "updated": r.get("lastModified"),
                    "supporter": _supporter_only(rid, files, token)})
    return out


def _files(repo, token):
    """The .gguf files in a repository and the .expert-counts.csv that sits beside each one."""
    url = "%s/api/models/%s/tree/main?recursive=1" % (HF, repo)
    tree = _json(url, token=_authored(url, token))
    if not isinstance(tree, list):
        return []
    found = []
    for e in tree:
        if e.get("type") != "file":
            continue
        p = e.get("path") or ""
        base = os.path.basename(p)
        if not base:
            continue
        lfs = e.get("lfs") or {}
        item = {"path": p, "name": base, "size": lfs.get("size") or e.get("size") or 0,
                "sha256": (lfs.get("oid") or "") or None}
        if base.lower().endswith(".gguf"):
            item["kind"] = "model"
        elif base.lower().endswith(".gguf.expert-counts.csv"):
            item["kind"] = "counts"
            item["model"] = base[:-len(".expert-counts.csv")]
        else:
            continue
        found.append(item)
    return found


def _supporter_only(repo, files, token):
    """True when this repository's model files cannot be read without a token.

    One small ranged GET on the first model file answers it for the whole repository (every file in a
    gated repo behaves the same way), and costs a few bytes instead of a probe per file.
    """
    first = next((f for f in files if f.get("kind") == "model"), None)
    if not first:
        return False
    url = "%s/%s/resolve/main/%s" % (HF, repo, urllib.parse.quote(first["path"]))
    try:
        with _open(url, token=token, headers={"Range": "bytes=0-0"}, timeout=30) as r:
            r.read(1)
        return False
    except HFError as e:
        return e.code == "needs_token"


def safe_file_name(name):
    """a plain file name to write inside the chosen folder: no separator, not . or .., no NUL, no leading dot
    (no hidden file), a .gguf or a .gguf.expert-counts.csv, at most 255 bytes. Downloads are data: nothing
    fetched here is ever made executable or run."""
    if not isinstance(name, str) or not name or len(name.encode("utf-8", "replace")) > 255:
        return False
    if "/" in name or "\\" in name or "\x00" in name or name in (".", "..") or name.startswith("."):
        return False
    if os.path.basename(name) != name:
        return False
    low = name.lower()
    return low.endswith(".gguf") or low.endswith(".gguf.expert-counts.csv")


def download_plan(files, want):
    """Every file one download has to fetch: the GGUF asked for, its other shards, and its counts csv."""
    hit = next((f for f in files if f["path"] == want or f["name"] == want), None)
    if hit is None:
        raise HFError("that file is not in the repository any more", code="not_found", status=404)
    if hit.get("kind") != "model":
        raise HFError("that is not a model file", code="invalid", status=400)
    out = [hit]
    m = SHARD_RE.search(hit["name"])
    if m:                                   # a split GGUF: one shard on its own is not a usable model
        n = int(m.group(2))
        stem = hit["name"][:m.start()] + "-"
        out = [f for f in files if f.get("kind") == "model" and f["name"].startswith(stem)
               and (SHARD_RE.search(f["name"]) or (None, None))[0] and len(f["name"]) == len(hit["name"])]
        out.sort(key=lambda f: f["name"])
        if len(out) != n:
            raise HFError("this model is split into %d parts but the repository lists %d of them"
                          % (n, len(out)), code="incomplete", status=502)
    # The expert map. The engine looks for <the -m path>.expert-counts.csv, and -m is the file asked for (the
    # first shard of a split model). A repository may name its map after that file, after the unsplit model
    # name, or (a release dir built from a symlink) after another name of the same model: take the map named
    # for this file, else for its unsplit name, else the repository's only map when it holds only one model,
    # and SAVE it under the name the engine looks for - a map saved under another name is a map the engine
    # never reads, and the expert cache falls back to whole-layer streaming (2026-10-12).
    counts = [f for f in files if f.get("kind") == "counts"]
    unsplit = hit["name"][:m.start()] + ".gguf" if m else hit["name"]
    models = {(f["name"][:SHARD_RE.search(f["name"]).start()] + ".gguf") if SHARD_RE.search(f["name"]) else f["name"]
              for f in files if f.get("kind") == "model"}
    pick = (next((f for f in counts if f.get("model") == hit["name"]), None)
            or next((f for f in counts if f.get("model") == unsplit), None)
            or (counts[0] if len(counts) == 1 and len(models) == 1 else None))
    if pick is not None:
        want = hit["name"] + ".expert-counts.csv"
        out.append(dict(pick, save_as=want) if pick["name"] != want else pick)
    return out


def _sha256(path, stop=None, progress=None, h=None, limit=None):
    """sha256 of a file (or of its first `limit` bytes). progress(bytes_hashed) is called as it goes, so a page can show how far the
    check is and when it ends (beta 2: an 80 GB file sat on 'checking' for minutes with nothing moving). h: a hashlib object to
    continue (the resume prefix of a download). Returns the hex digest, or None when stopped."""
    import hashlib
    h = h if h is not None else hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            if stop is not None and stop.is_set():
                return None
            n = CHUNK if limit is None else min(CHUNK, limit - done)
            if n <= 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress is not None:
                progress(done)
    return h.hexdigest()


class Download:
    """One download, run on its own thread so the page can watch it and cancel it."""

    def __init__(self, repo, items, folder, token=None):
        self.repo = repo
        self.items = items
        self.folder = folder
        self.token = token
        self.state = "starting"          # starting | running | checking | done | failed | cancelled
        self.error = None
        self.i = 0
        self.done = 0
        self.total = sum(int(f.get("size") or 0) for f in items)
        self.file_done = 0
        self.file_total = int(items[0].get("size") or 0) if items else 0
        self.what = items[0]["name"] if items else ""
        self.bytes_now = 0
        self.rate = 0.0
        self.started = time.time()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.written = []
        # the sha256 check (beta 2): hashed WHILE the bytes arrive, so the end of a download is a compare, not a second read of
        # the whole file; only a resumed download hashes the part already on disk first, and that pass reports its progress
        self.hash_done = 0
        self.hash_total = 0
        self.hash_started = None
        self.hash_phase = None           # None | "resume" (the part already on disk) | "final" (the compare at the end)

    def _hash_progress(self, n):
        with self.lock:
            self.hash_done = n

    def status(self):
        with self.lock:
            el = max(0.001, time.time() - self.started)
            hrate = heta = None
            if self.hash_started is not None and self.hash_done:
                hel = max(0.001, time.time() - self.hash_started)
                hrate = self.hash_done / hel
                heta = max(0.0, (self.hash_total - self.hash_done) / hrate) if hrate > 0 else None
            return {"repo": self.repo, "folder": self.folder, "state": self.state, "error": self.error,
                    "what": self.what, "done": self.done, "total": self.total, "file_done": self.file_done,
                    "file_total": self.file_total, "rate": self.rate, "seconds": round(el, 1),
                    "files": len(self.items), "file_index": min(self.i + 1, len(self.items)),
                    "written": list(self.written),
                    "hash_phase": self.hash_phase, "hash_done": self.hash_done, "hash_total": self.hash_total,
                    "hash_rate": round(hrate, 1) if hrate else None,
                    "hash_eta_s": round(heta, 1) if heta is not None else None}

    # -- the download itself ----------------------------------------------------------------
    def _one(self, item):
        name = item.get("save_as") or item["name"]      # the counts csv is saved under the name the engine reads
        if not safe_file_name(name):                    # a repository listing is data: it never picks the folder
            raise HFError("refusing a file name with a path in it: %r" % str(name)[:80], code="invalid", status=502)
        target = os.path.join(self.folder, name)
        part = target + ".part"
        have = os.path.getsize(part) if os.path.exists(part) else 0
        url = "%s/%s/resolve/main/%s" % (HF, self.repo, urllib.parse.quote(item["path"]))
        headers = {}
        if have:
            headers["Range"] = "bytes=%d-" % have
        with self.lock:
            self.what = name
            self.file_total = int(item.get("size") or 0)
            self.file_done = have
        import hashlib
        h = hashlib.sha256() if item.get("sha256") else None
        r = _open(url, token=self.token, headers=headers, timeout=120)
        try:
            if have and getattr(r, "status", 200) != 206:  # the server ignored the range: start again
                have = 0
                with self.lock:
                    self.file_done = 0
            if h is not None and have:          # a resume: the part already on disk goes into the hash first (with progress)
                with self.lock:
                    self.state, self.hash_phase = "checking", "resume"
                    self.hash_done, self.hash_total, self.hash_started = 0, have, time.time()
                if _sha256(part, self.stop, self._hash_progress, h=h, limit=have) is None:
                    raise HFError("cancelled", code="cancelled", status=499)
                with self.lock:
                    self.state, self.hash_phase, self.hash_started = "running", None, None
            mode = "ab" if have else "wb"
            started, base, last = time.time(), have, have
            with open(part, mode) as f:
                while True:
                    if self.stop.is_set():
                        raise HFError("cancelled", code="cancelled", status=499)
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    if h is not None:
                        h.update(b)
                    f.flush()           # the bytes are on disk as they arrive, so a kill cannot lose the
                    last += len(b)      # fetched half and the size the resume reads is a real number
                    now = time.time()
                    if now - started >= 0.4:                # an average over the whole file, not the last
                        with self.lock:                     # chunk: a last-chunk rate reads 2000 MiB/s on a
                            self.file_done = last            # buffered tail and means nothing
                            self.rate = (last - base) / max(0.001, now - started)
        finally:
            r.close()                   # a cancel or a broken link must not leave the connection hanging
        with self.lock:
            self.file_done = last
        if item.get("sha256"):
            with self.lock:
                self.state, self.hash_phase = "checking", "final"
                self.hash_done = self.hash_total = last
                self.hash_started = time.time()
            got = h.hexdigest()                 # hashed as it arrived: no second read of the file
            if got != item["sha256"]:
                os.unlink(part)
                raise HFError("%s did not match the sha256 the repository publishes, so it was thrown away and "
                              "can be fetched again" % name, code="sha_mismatch", status=502)
        os.replace(part, target)
        with self.lock:
            self.written.append(name)
            self.state = "running"
            self.hash_phase, self.hash_started = None, None

    def run(self):
        with self.lock:
            self.state = "running"
        try:
            for self.i, item in enumerate(self.items):
                self._one(item)
                with self.lock:
                    self.done += int(item.get("size") or 0)
            with self.lock:
                self.state = "done"
        except HFError as e:
            with self.lock:
                self.state = "cancelled" if e.code == "cancelled" else "failed"
                if e.code != "cancelled":
                    self.error = str(e)
        except Exception as e:                      # noqa: BLE001 - a socket error must not kill the page
            with self.lock:
                self.state = "failed"
                self.error = "%s: %s" % (e.__class__.__name__, e)

    def cancel(self):
        self.stop.set()


class Downloads:
    """The one download this Control is running at a time (more would fight for the same link)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.job = None

    def start(self, repo, items, folder, token=None):
        with self.lock:
            if self.job is not None and self.job.state in ("starting", "running", "checking"):
                raise HFError("a download is already running; wait for it or cancel it first",
                              code="busy", status=409)
            free = shutil.disk_usage(folder).free
            need = sum(int(f.get("size") or 0) for f in items)
            if need and free < need + (256 << 20):
                raise HFError("not enough room in %s: %s needed, %s free" % (
                    folder, _h(need), _h(free)), code="no_space", status=507)
            job = Download(repo, items, folder, token=token)
            self.job = job
        threading.Thread(target=job.run, daemon=True, name="pxa-hf-download").start()
        return job

    def status(self):
        with self.lock:
            return self.job.status() if self.job is not None else None

    def cancel(self):
        with self.lock:
            job = self.job
        if job is not None and job.state in ("starting", "running", "checking"):
            job.cancel()
            return True
        return False

    def busy(self):
        return self.status() is not None and self.status()["state"] in ("starting", "running", "checking")


def _h(n):
    n = float(n or 0)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return ("%.0f %s" if u == "B" else "%.1f %s") % (n, u)
        n /= 1024


class Catalog:
    """The listing, remembered for LIST_TTL seconds so opening the tab twice does not hammer Hugging Face."""

    def __init__(self):
        self.lock = threading.Lock()
        self.at = 0.0
        self.data = None
        self.err = None
        self.token_seen = None

    def get(self, org, token, force=False):
        with self.lock:
            fresh = self.data is not None and (time.time() - self.at) < LIST_TTL
            same = self.token_seen == bool(token)
            if fresh and same and not force:
                return self.data
        try:
            repos = list_repos(org, token)
        except HFError as e:
            with self.lock:
                if self.data is not None:                  # a stale list beats an empty page
                    self.err = str(e)
                    return self.data
                self.err = str(e)
                self.data = {"org": org, "ts": time.time(), "repos": [], "error": str(e)}
            return self.data
        data = {"org": org, "ts": time.time(), "repos": repos, "error": None,
                "models": sum(1 for r in repos for f in r["files"] if f.get("kind") == "model")}
        with self.lock:
            self.data, self.err, self.at, self.token_seen = data, None, time.time(), bool(token)
        return data
