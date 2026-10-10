"""Folder fence for the chat agent's file tools.

Python port of the owner's hive sandbox (PXACLAW tools/local-lane/harness/sandbox.mjs resolveInside, ls, read,
write; tools2.mjs readRange, grep, append). The rule that matters: resolve the deepest part of the path that
EXISTS, so a file about to be created is fine but a symlink anywhere above it that points outside is not.
Absolute paths and '..' escapes are refused.
"""
import os
import re

MAX_FILE = 2 * 1024 * 1024
MAX_READ = 200 * 1024
SKIP = {".git", "node_modules", "__pycache__", ".pytest_cache", ".venv"}


class Outside(ValueError):
    pass


def resolve_inside(workdir, rel):
    if not isinstance(rel, str) or not rel or "\x00" in rel:
        raise Outside(f"path {rel!r} is outside the sandbox folder")
    if os.path.isabs(rel) or re.match(r"^[A-Za-z]:[\\/]", rel):
        raise Outside(f"path {rel!r} is outside the sandbox folder (use a path inside it, like notes.txt)")
    try:
        root = os.path.realpath(workdir)
    except OSError:
        root = os.path.abspath(workdir)
    target = os.path.normpath(os.path.join(root, rel))
    probe = target
    while not os.path.lexists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    real = os.path.realpath(probe)
    rest = os.path.relpath(target, probe) if target != probe else ""
    resolved = os.path.join(real, rest) if rest and rest != "." else real
    if resolved != root and not resolved.startswith(root + os.sep):
        raise Outside(f"path {rel!r} is outside the sandbox folder")
    return resolved


def _lines(text):
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def ls(workdir, rel=".", limit=400):
    root = os.path.realpath(workdir)
    base = resolve_inside(workdir, rel or ".")
    out = []
    for dp, dns, fns in os.walk(base):
        dns[:] = sorted(d for d in dns if d not in SKIP and not os.path.islink(os.path.join(dp, d)))
        for fn in sorted(fns):
            full = os.path.join(dp, fn)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            out.append(f"{os.path.relpath(full, root)} ({size} bytes)")
            if len(out) >= limit:
                out.append(f"[stopped at {limit} files]")
                return "\n".join(out)
    return "\n".join(out) if out else "(the folder is empty)"


def read_range(workdir, rel, offset=1, limit=200):
    p = resolve_inside(workdir, rel)
    if os.path.getsize(p) > MAX_FILE:
        raise ValueError(f"{rel} is larger than {MAX_FILE // (1024 * 1024)} MB")
    with open(p, encoding="utf-8", errors="replace") as f:
        allv = _lines(f.read())
    start = max(1, int(offset or 1))
    lim = max(1, min(int(limit or 200), 2000))
    sl = allv[start - 1:start - 1 + lim]
    out = [f"{start + i}: {t}" for i, t in enumerate(sl)]
    out.append(f"[lines {start}-{start - 1 + len(sl)} of {len(allv)}]")
    s = "\n".join(out)
    return s if len(s) <= MAX_READ else s[:MAX_READ] + f"\n[truncated at {MAX_READ} bytes]"


def write(workdir, rel, content, append=False):
    p = resolve_inside(workdir, rel)
    body = content if isinstance(content, str) else ""
    if len(body.encode("utf-8")) > MAX_FILE:
        raise ValueError("content is larger than 2 MB")
    if not os.path.isdir(workdir):           # the chat's own folder: owner-only, like the saved sessions
        os.makedirs(os.path.dirname(os.path.abspath(workdir)), mode=0o700, exist_ok=True)
        os.makedirs(workdir, mode=0o700, exist_ok=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    existed = os.path.exists(p)
    with open(p, "a" if append else "w", encoding="utf-8") as f:
        f.write(body)
    n = len(body.encode("utf-8"))
    if append:
        with open(p, encoding="utf-8", errors="replace") as f:
            total = len(_lines(f.read()))
        return f"appended {n} bytes to {rel} ({total} lines now)"
    return f"{'replaced' if existed else 'created'} {rel} ({n} bytes)"


def grep(workdir, pattern, rel=None, max_hits=60):
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        return f"error: bad pattern {e}"
    root = os.path.realpath(workdir)
    hits = []

    def scan(path, shown):
        try:
            if os.path.getsize(path) > MAX_FILE:
                return
            with open(path, "rb") as f:
                if b"\x00" in f.read(512):
                    return
            with open(path, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append(f"{shown}:{i}: {line.strip()[:200]}")
                        if len(hits) >= max_hits:
                            return
        except OSError:
            return

    if rel:
        scan(resolve_inside(workdir, rel), rel)
    else:
        for dp, dns, fns in os.walk(root):
            dns[:] = sorted(d for d in dns if d not in SKIP and not os.path.islink(os.path.join(dp, d)))
            for fn in sorted(fns):
                if len(hits) >= max_hits:
                    break
                full = os.path.join(dp, fn)
                if not os.path.islink(full):
                    scan(full, os.path.relpath(full, root))
    if not hits:
        return "(no matches)"
    if len(hits) >= max_hits:
        hits.append(f"[stopped at {max_hits} matches]")
    return "\n".join(hits)
