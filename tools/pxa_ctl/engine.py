"""Apply a profile to cards: plan (a dry-run diff), apply in a fixed order, read every value back, roll everything back
if any step fails or reads back wrong. Applying the same profile twice is a no-op (fields already at the target are
dropped from the plan), so a re-apply after a reboot or a schedule is safe.

Order: persistence -> application clocks -> locked clocks -> power limit (the power limit last, so a card never runs a
new clock at an old, lower limit for longer than one step). The plan is transactional across ALL target cards: one
failure rolls back every card already changed, newest first, and the audit log has each step and each undo.
"""
from .errors import DriverError, Invalid

FIELDS = ("persistence", "app_clocks", "locked_clocks", "power_limit")
POWER_TOL_W = 1.0


def resolve_power(spec, c):
    """-> (watts or None, note or None) for one card: the profile's power setting clamped to what the card allows."""
    if not spec:
        return None, None
    if not c["caps"].get("power_limit"):
        return None, "power limit: not adjustable on this card"
    mode, v = spec["mode"], spec.get("value")
    lo, hi, dflt = c.get("min_w"), c.get("max_w"), c.get("default_w") or c.get("max_w")
    if mode == "watts":
        want = float(v)
    elif mode == "pct_default":
        want = dflt * float(v) / 100.0
    elif mode == "pct_max":
        want = hi * float(v) / 100.0
    else:
        want = dflt
    w = float(round(min(max(want, lo), hi)))
    note = None
    if abs(w - want) > 0.5 and (want < lo or want > hi):
        note = f"power limit: {want:.0f} W asked, this card allows {lo:.0f}-{hi:.0f} W, so {w:.0f} W"
    return w, note


def snap_clock(want, allowed):
    """the nearest allowed clock at or below want (else the lowest allowed); allowed empty -> want."""
    if not allowed:
        return want
    below = [a for a in allowed if a <= want]
    return max(below) if below else min(allowed)


def plan_card(profile_gpu, c, supported=None):
    """-> (steps, notes) for one card. A step: {field, before, after, op} (op tells apply which call to make)."""
    steps, notes = [], []
    g = profile_gpu or {}
    caps = c["caps"]
    pm = g.get("persistence")
    if pm is not None:
        if not caps.get("persistence"):
            notes.append("persistence mode: not available here")
        elif c.get("persistence") != pm:
            steps.append({"field": "persistence", "before": c.get("persistence"), "after": pm, "op": "set"})
    ac = g.get("app_clocks")
    if ac is not None:
        if not caps.get("app_clocks"):
            notes.append("application clocks: not supported on this card (hidden)")
        else:
            before = [c.get("app_mem"), c.get("app_sm")]
            if ac == "default":
                after = [c.get("def_app_mem"), c.get("def_app_sm")]
                if None not in after and before != after:
                    steps.append({"field": "app_clocks", "before": before, "after": after, "op": "reset"})
            else:
                mem, sm = ac["mem"], ac["sm"]
                if supported:
                    mems = sorted(supported)
                    m2 = mem if mem in supported else snap_clock(mem, mems)
                    s2 = snap_clock(sm, sorted(supported.get(m2, [])))
                    if (m2, s2) != (mem, sm):
                        notes.append(f"application clocks: {mem},{sm} MHz is not a supported pair here, so {m2},{s2} MHz")
                    mem, sm = m2, s2
                after = [float(mem), float(sm)]
                if before != after:
                    steps.append({"field": "app_clocks", "before": before, "after": after, "op": "set",
                                  "default": [c.get("def_app_mem"), c.get("def_app_sm")]})
    lc = g.get("locked_clocks")
    if lc is not None:
        if not caps.get("locked_clocks"):
            notes.append("locked clocks: need Volta or newer (hidden on this card)")
        elif lc == "reset":
            if c.get("locked") is not None:
                steps.append({"field": "locked_clocks", "before": list(c["locked"]), "after": None, "op": "reset"})
        else:
            hi = lc["max"]
            if c.get("max_sm") and hi > c["max_sm"]:
                notes.append(f"locked clocks: {hi} MHz is above this card's {c['max_sm']:.0f} MHz, so {c['max_sm']:.0f}")
                hi = int(c["max_sm"])
            lo = min(lc["min"], hi)
            after = [lo, hi]
            if list(c.get("locked") or []) != after:
                steps.append({"field": "locked_clocks", "before": list(c["locked"]) if c.get("locked") else None,
                              "after": after, "op": "set"})
    w, note = resolve_power(g.get("power"), c)
    if note:
        notes.append(note)
    if w is not None and (c.get("limit_w") is None or abs(c["limit_w"] - w) > 0.5):
        steps.append({"field": "power_limit", "before": c.get("limit_w"), "after": w, "op": "set"})
    steps.sort(key=lambda s: FIELDS.index(s["field"]))
    return steps, notes


class ApplyEngine(object):
    def __init__(self, adapter, guard, audit):
        self.adapter = adapter
        self.guard = guard
        self.audit = audit

    # ---- plan -------------------------------------------------------------------------------
    def plan(self, profile, cards=None):
        cards = self.adapter.list(max_age=0.0) if cards is None else cards
        by = {c["uuid"]: c for c in cards}
        locks = self.guard.locks()
        out, n = [], 0
        for u in profile["targets"]:
            c = by.get(u)
            if c is None:
                out.append({"uuid": u, "index": None, "name": None, "missing": True, "steps": [], "locked": [],
                            "notes": ["this card is not in the machine now: skipped"]})
                continue
            sup = None
            ac = (profile.get("gpu") or {}).get("app_clocks")
            if isinstance(ac, dict) and c["caps"].get("app_clocks"):
                try:
                    sup = self.adapter.supported_clocks(u)
                except Exception:          # noqa: BLE001 - unknown list: the driver itself will refuse a bad pair
                    sup = None
            steps, notes = plan_card(profile.get("gpu"), c, sup)
            n += len(steps)
            out.append({"uuid": u, "index": c["index"], "name": c["name"], "missing": False, "steps": steps, "notes": notes,
                        "locked": [lk["why"] for lk in self.guard.reasons(u, c["index"], locks)]})
        return {"profile": profile["id"], "cards": out, "changes": n,
                "locked": any(x["locked"] for x in out if x["steps"])}

    # ---- apply ------------------------------------------------------------------------------
    def _do(self, uuid, s, undo=False):
        a = self.adapter
        val = s["before"] if undo else s["after"]
        f = s["field"]
        if f == "persistence":
            a.set_persistence(uuid, bool(val))
        elif f == "power_limit":
            a.set_power_limit(uuid, val)
        elif f == "app_clocks":
            if (not undo and s["op"] == "reset") or (undo and val == s.get("default")):
                a.reset_app_clocks(uuid)
            else:
                a.set_app_clocks(uuid, val[0], val[1])
        elif f == "locked_clocks":
            if val is None:
                a.reset_locked_clocks(uuid)
            else:
                a.lock_clocks(uuid, val[0], val[1])

    @staticmethod
    def _readback_ok(c, s, undo=False):
        val = s["before"] if undo else s["after"]
        f = s["field"]
        if f == "power_limit":
            return c.get("limit_w") is not None and abs(c["limit_w"] - val) <= POWER_TOL_W, c.get("limit_w")
        if f == "persistence":
            return c.get("persistence") == val, c.get("persistence")
        if f == "app_clocks":
            got = [c.get("app_mem"), c.get("app_sm")]
            return got == list(val), got
        if f == "locked_clocks":
            got = list(c["locked"]) if c.get("locked") else None
            return got == (list(val) if val else None), got
        return True, None

    def revert(self, changes, who="api", action="undo"):
        """put back what a successful apply changed: changes = [{"uuid", "index", "step"}] in the order they were made.
        Newest first, each verified; best effort (one failure does not stop the others) and every step is logged."""
        out = []
        for ch in reversed(changes):
            s = ch["step"]
            r = {"uuid": ch["uuid"], "index": ch["index"], "field": s["field"], "to": s["before"], "ok": True}
            try:
                self._do(ch["uuid"], s, undo=True)
                c = self.adapter.get(ch["uuid"], fresh=True)
                ok, got = self._readback_ok(c, s, undo=True)
                if not ok:
                    r.update(ok=False, error=f"read back {got!r}")
            except Exception as e:      # noqa: BLE001
                r.update(ok=False, error=f"{e.__class__.__name__}: {e}")
            out.append(r)
            self.audit.write(action + ".step", who, target={"uuid": ch["uuid"], "index": ch["index"], "field": s["field"]},
                             before=s["after"], after=s["before"], ok=r["ok"], error=r.get("error"))
        return out

    def apply(self, profile, who="api", cards=None, action="apply"):
        """-> result dict. Raises Locked before touching anything if a target card is locked."""
        p = self.plan(profile, cards)
        todo = [x for x in p["cards"] if x["steps"]]
        self.guard.check([(x["uuid"], x["index"]) for x in todo], f"{action} {profile['id']}")
        done = []                      # (uuid, step) applied and verified, in order
        failure = None
        for x in todo:
            for s in x["steps"]:
                try:
                    self._do(x["uuid"], s)
                    c = self.adapter.get(x["uuid"], fresh=True)
                    ok, got = self._readback_ok(c, s)
                    done.append((x, s))
                    if not ok:
                        raise DriverError(f"card {x['index']} {s['field']}: read back {got!r}, expected {s['after']!r}",
                                          code="verify_failed")
                    self.audit.write(action + ".step", who, target={"uuid": x["uuid"], "index": x["index"],
                                     "profile": profile["id"], "field": s["field"]}, before=s["before"], after=s["after"])
                except Exception as e:  # noqa: BLE001 - every failure means: undo everything
                    failure = {"uuid": x["uuid"], "index": x["index"], "field": s["field"],
                               "error": str(e) if isinstance(e, (DriverError, Invalid)) else f"{e.__class__.__name__}: {e}"}
                    self.audit.write(action + ".step", who, target={"uuid": x["uuid"], "index": x["index"],
                                     "profile": profile["id"], "field": s["field"]}, before=s["before"], after=s["after"],
                                     ok=False, error=failure["error"])
                    break
            if failure:
                break
        rolled = []
        if failure:
            for x, s in reversed(done):
                r = {"uuid": x["uuid"], "index": x["index"], "field": s["field"], "to": s["before"], "ok": True}
                try:
                    self._do(x["uuid"], s, undo=True)
                    c = self.adapter.get(x["uuid"], fresh=True)
                    ok, got = self._readback_ok(c, s, undo=True)
                    if not ok:
                        r.update(ok=False, error=f"read back {got!r}")
                except Exception as e:  # noqa: BLE001
                    r.update(ok=False, error=f"{e.__class__.__name__}: {e}")
                rolled.append(r)
                self.audit.write(action + ".rollback", who, target={"uuid": x["uuid"], "index": x["index"],
                                 "profile": profile["id"], "field": s["field"]}, before=s["after"], after=s["before"],
                                 ok=r["ok"], error=r.get("error"))
        res = {"ok": failure is None, "profile": profile["id"], "plan": p, "applied": len(done) if not failure else 0,
               "changes": [] if failure else [{"uuid": x["uuid"], "index": x["index"], "step": s} for x, s in done],
               "failure": failure, "rolled_back": rolled, "rollback_ok": all(r["ok"] for r in rolled)}
        self.audit.write(action, who, target={"profile": profile["id"], "cards": [x["uuid"] for x in todo]},
                         after={"changes": p["changes"]}, ok=res["ok"], error=failure and failure["error"],
                         detail={"rolled_back": len(rolled), "rollback_ok": res["rollback_ok"]} if failure else None)
        return res
