"""Append-only record of every GPU change and auto-start action: gpu_audit.jsonl in the config dir (0600).

    {"ts", "who", "action", "target", "before", "after", "ok", "dry_run", "error", "detail"}

who is "page <client address>", "schedule <id>", "supervisor", "boot" or "api"; never a token. The file rotates at 2 MiB
(one older copy, .1), so it cannot fill a disk.
"""
import json
import os
import threading
import time

MAX_BYTES = 2 * 1024 * 1024


class Audit(object):
    def __init__(self, path, clock=time.time, max_bytes=MAX_BYTES):
        self.path = path
        self.clock = clock
        self.max_bytes = max_bytes
        self.lock = threading.Lock()

    def write(self, action, who="api", target=None, before=None, after=None, ok=True, dry_run=False, error=None, detail=None):
        rec = {"ts": round(self.clock(), 3), "who": str(who)[:120], "action": str(action)[:60], "target": target,
               "before": before, "after": after, "ok": bool(ok), "dry_run": bool(dry_run)}
        if error:
            rec["error"] = str(error)[:500]
        if detail is not None:
            rec["detail"] = detail
        line = json.dumps(rec, separators=(",", ":"), default=str)
        with self.lock:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                try:
                    if os.path.getsize(self.path) > self.max_bytes:
                        os.replace(self.path, self.path + ".1")
                except OSError:
                    pass
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        return rec

    def read(self, limit=200, since=0.0, action=None):
        out = []
        for p in (self.path + ".1", self.path):
            try:
                with open(p) as f:
                    for ln in f:
                        try:
                            r = json.loads(ln)
                        except ValueError:
                            continue
                        if isinstance(r, dict) and r.get("ts", 0) >= since and (not action or r.get("action") == action):
                            out.append(r)
            except OSError:
                continue
        return out[-limit:]
