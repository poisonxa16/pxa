"""Who must NOT be touched right now. Asked before every GPU change, every auto-start, and every launch PXA Control makes
onto a card (App.start), so a benchmark card stays untouched whoever clicks what.

A card is locked when any of these holds:
  * maintenance mode is on (every card);
  * the card is reserved (store "reserved", optional expiry);
  * a LOCK FILE exists (gpu_lock_files): e.g. /run/pxa/bench.lock. Its content may scope it:
        JSON {"gpus": ["GPU-...", 0, 1], "why": "speed race"}   or a line "gpus=0,1"   (no scope = every card);
  * a WAIT-FOR file is MISSING (gpu_lock_if_missing): e.g. the race's GATE-DONE marker; absent = the race still runs.
Lock files are read on every check (they are tiny), so creating or removing one takes effect at once.
"""
import json
import os
import re
import time

from .errors import Locked

_GPUS_LINE = re.compile(r"gpus\s*[=:]\s*([A-Za-z0-9,\- ]+)", re.I)


def read_scope(path):
    """-> (scope, why): scope is None (every card) or a set of UUIDs / indexes."""
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read(8192)
    except OSError:
        return None, ""
    t = text.strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
            g = d.get("gpus")
            scope = None
            if isinstance(g, list) and g:
                scope = {x for x in g if isinstance(x, (str, int)) and not isinstance(x, bool)}
            return scope or None, str(d.get("why") or "")[:200]
        except ValueError:
            pass
    m = _GPUS_LINE.search(t)
    scope = None
    if m:
        scope = set()
        for x in m.group(1).replace(" ", "").split(","):
            if x.isdigit():
                scope.add(int(x))
            elif x:
                scope.add(x)
    first = t.splitlines()[0][:200] if t else ""
    return scope or None, ("" if m and first == m.group(0) else first)


class Guard(object):
    def __init__(self, store, lock_files=(), lock_if_missing=(), clock=time.time):
        self.store = store
        self.lock_files = [p for p in lock_files if p]
        self.lock_if_missing = [p for p in lock_if_missing if p]
        self.clock = clock

    def locks(self):
        """every lock in force now: [{kind, why, scope (None = all), path?}]"""
        out = []
        d = self.store.snapshot()
        m = d.get("maintenance") or {}
        if m.get("on"):
            out.append({"kind": "maintenance", "why": m.get("why") or "maintenance mode", "scope": None})
        now = self.clock()
        for u, r in (d.get("reserved") or {}).items():
            until = r.get("until") if isinstance(r, dict) else None
            if until and until < now:
                continue
            out.append({"kind": "reserved", "why": (r.get("why") if isinstance(r, dict) else str(r)) or "reserved",
                        "scope": [u], "until": until})
        for p in self.lock_files:
            if os.path.exists(p):
                scope, why = read_scope(p)
                out.append({"kind": "lock_file", "path": p, "why": why or f"lock file {os.path.basename(p)} exists",
                            "scope": sorted(scope, key=str) if scope else None})
        for p in self.lock_if_missing:
            if not os.path.exists(p):
                out.append({"kind": "waiting_for", "path": p, "why": f"{os.path.basename(p)} is missing (a benchmark is running)",
                            "scope": None})
        return out

    @staticmethod
    def _hits(lock, uuid, index):
        sc = lock.get("scope")
        return sc is None or uuid in sc or (index is not None and index in sc)

    def reasons(self, uuid, index=None, locks=None):
        locks = self.locks() if locks is None else locks
        return [lk for lk in locks if self._hits(lk, uuid, index)]

    def check(self, cards, action):
        """raise Locked when any of cards [(uuid, index)] is locked; cards may also be plain UUIDs."""
        locks = self.locks()
        bad = []
        for c in cards:
            uuid, index = (c if isinstance(c, tuple) else (c, None))
            for lk in self.reasons(uuid, index, locks):
                bad.append(f"card {index if index is not None else uuid}: {lk['why']}")
        if bad:
            uniq = list(dict.fromkeys(bad))
            raise Locked(f"{action} refused: " + "; ".join(uniq[:6]) + (f" (+{len(uniq) - 6} more)" if len(uniq) > 6 else ""),
                         detail={"reasons": uniq})

    def all_locked(self):
        return any(lk.get("scope") is None for lk in self.locks())
