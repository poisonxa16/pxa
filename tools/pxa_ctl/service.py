"""GpuControl: the one object PXA Control's routes talk to. It owns the store, guard, audit log, apply engine,
supervisor and scheduler, and enforces the switches:

    allow_gpu_control   power / clocks / persistence writes (default False; the page also asks a typed confirm)
    autostart           the supervisor may start servers (default False)
    schedules           schedules fire (default False)
    boot_apply          at Control start, re-apply the profiles that were active (power limits reset with the driver)

None of these can be switched on from the page: control.json or the environment only.
"""
import os
import threading
import time

from . import schedule as SCH
from .audit import Audit
from .driver import NullAdapter, UUID_RE
from .engine import ApplyEngine, resolve_power
from .errors import Invalid, Locked, Refused
from .guard import Guard
from .store import PRESETS, ProfileStore, validate_targets
from .supervisor import Supervisor

TRUE = ("1", "on", "yes", "true")
HYST_C = 5
PROBATION_S = 30.0            # after a change, watch the cards this long and undo it by itself if one turns unhealthy
UNDO_KEEP_S = 24 * 3600


def settings_from(cfg, env=None):
    """the switches from control.json, overridden by the environment."""
    env = os.environ if env is None else env
    cfg = cfg if isinstance(cfg, dict) else {}

    def flag(key, envk):
        v = env.get(envk)
        if v is not None and v.strip() != "":
            return v.strip().lower() in TRUE
        return cfg.get(key) is True

    def paths(key, envk):
        out = [p for p in (cfg.get(key) or []) if isinstance(p, str) and p.strip()] if isinstance(cfg.get(key), list) else []
        out += [p for p in (env.get(envk) or "").split(":") if p.strip()]
        return list(dict.fromkeys(os.path.expanduser(p.strip()) for p in out))
    return {"allow_gpu_control": flag("allow_gpu_control", "PXA_CONTROL_ALLOW_GPU"),
            "autostart": flag("gpu_autostart", "PXA_CONTROL_AUTOSTART"),
            "schedules": flag("gpu_schedules", "PXA_CONTROL_SCHEDULES"),
            "boot_apply": flag("gpu_boot_apply", "PXA_CONTROL_BOOT_APPLY"),
            "lock_files": paths("gpu_lock_files", "PXA_CONTROL_LOCK_FILES"),
            "lock_if_missing": paths("gpu_lock_if_missing", "PXA_CONTROL_LOCK_IF_MISSING"),
            "ui_mode": cfg.get("ui_mode") if cfg.get("ui_mode") in ("simple", "advanced") else "simple"}


def _who(who):
    return str(who or "api")[:120]


class GpuControl(object):
    def __init__(self, config_dir, settings, adapter=None, launcher=None, clock=time.time):
        self.settings = dict(settings)
        self.adapter = adapter or NullAdapter()
        self.clock = clock
        self.store = ProfileStore(os.path.join(config_dir, "gpu_profiles.json"))
        self.audit = Audit(os.path.join(config_dir, "gpu_audit.jsonl"), clock=clock)
        self.guard = Guard(self.store, self.settings["lock_files"], self.settings["lock_if_missing"], clock=clock)
        self.engine = ApplyEngine(self.adapter, self.guard, self.audit)
        self.launcher = launcher
        self.supervisor = Supervisor(launcher, self.guard, self.audit, clock=clock,
                                     enabled=lambda: self.settings["autostart"]) if launcher else None
        self.fired = {}
        self.thermal_state = {}       # uuid -> "warn" | "act" while above, for one alert / action per crossing
        self._thread = None
        self._stop = threading.Event()
        self.started = clock()
        self.last_tick = None
        self.tick_error = None
        self.undo_rec = None          # the last change that can be undone: {id, label, ts, changes, active_before}
        self.undo_seq = 0
        self.probation = None         # {until, undo_id, cards}: watched by tick(); an unhealthy card undoes the change
        self.last_auto_rollback = None

    # ---- reading ----------------------------------------------------------------------------
    def cards(self, max_age=1.0):
        try:
            cards = self.adapter.list(max_age=max_age)
        except Exception as e:                  # noqa: BLE001 - a broken driver must not break the page
            self.adapter.error = f"{e.__class__.__name__}: {e}"
            cards = []
        d = self.store.snapshot()
        locks = self.guard.locks()
        profs = d["profiles"]
        for c in cards:
            pid = d["active"].get(c["uuid"])
            c["active_profile"] = pid if pid in profs else None
            c["locked"] = [{"kind": lk["kind"], "why": lk["why"]} for lk in self.guard.reasons(c["uuid"], c["index"], locks)]
            c["reserved"] = c["uuid"] in d["reserved"]
            th = (profs.get(pid) or {}).get("thermal") or {}
            warn = th.get("warn_c") or (c.get("slowdown_c") - 5 if c.get("slowdown_c") else 80)
            c["thermal"] = {"warn_c": warn, "act_c": th.get("act_c"), "action": th.get("action") or "alert",
                            "slowdown_c": c.get("slowdown_c"), "shutdown_c": c.get("shutdown_c")}
            t = c.get("temp_c")
            c["alert"] = None
            if t is not None:
                if th.get("act_c") and t >= th["act_c"]:
                    c["alert"] = {"level": "bad", "text": f"{t:.0f} \u00b0C: at or above the profile's action point "
                                  f"({th['act_c']} \u00b0C, {c['thermal']['action'].replace('_', ' ')})"}
                elif t >= warn:
                    c["alert"] = {"level": "warn", "text": f"{t:.0f} \u00b0C: at or above the warning point ({warn:.0f} \u00b0C)"}
        return cards

    def view_settings(self):
        s = dict(self.settings)
        s["adapter"] = self.adapter.info()
        s["store_error"] = self.store.error
        s["store_writable"] = self.store.writable
        return s

    def state(self):
        d = self.store.snapshot()
        u = self.undo_rec
        undo = None if not u or self.clock() - u["ts"] > UNDO_KEEP_S else {
            "id": u["id"], "label": u["label"], "ts": u["ts"], "lines": u["lines"]}
        return {"settings": self.view_settings(), "cards": self.cards(), "profiles": list(d["profiles"].values()),
                "undo": undo, "probation": self.probation and {"until": self.probation["until"], "cards": self.probation["cards"]},
                "auto_rollback": self.last_auto_rollback,
                "presets": PRESETS, "schedules": d["schedules"], "reserved": d["reserved"],
                "maintenance": d["maintenance"], "quiet": d["quiet"], "locks": self.guard.locks(),
                "supervisor": self.supervisor.view() if self.supervisor else None,
                "background": {"running": bool(self._thread and self._thread.is_alive()), "last_tick": self.last_tick,
                               "error": self.tick_error},
                "ts": self.clock()}

    def health(self):
        a = self.adapter.info()
        return {"gpu_adapter": a, "profiles": len(self.store.data["profiles"]), "locks": len(self.guard.locks()),
                "allow_gpu_control": self.settings["allow_gpu_control"], "autostart": self.settings["autostart"],
                "schedules": self.settings["schedules"], "store_writable": self.store.writable,
                "background": bool(self._thread and self._thread.is_alive()), "last_tick": self.last_tick,
                "tick_error": self.tick_error}

    # ---- profiles -----------------------------------------------------------------------------
    def save_profile(self, body, who=None):
        if not isinstance(body, dict):
            raise Invalid("profile: a JSON object")
        p = dict(body.get("profile") if isinstance(body.get("profile"), dict) else body)
        if not p.get("id"):
            p["id"] = self.store.new_id(p.get("name"))
        before = self.store.data["profiles"].get(p["id"])
        saved = self.store.put(p)
        self.audit.write("profile.save", _who(who), target={"profile": saved["id"]}, before=before, after=saved)
        return saved

    def delete_profile(self, body, who=None):
        pid = (body or {}).get("id")
        before = self.store.get(pid)
        self.store.delete(pid)
        if self.supervisor:
            self.supervisor.forget_profile(pid, who=_who(who))
        self.audit.write("profile.delete", _who(who), target={"profile": pid}, before=before)
        return {"deleted": pid}

    def _profile_from(self, body):
        if not isinstance(body, dict):
            raise Invalid("a JSON object")
        if isinstance(body.get("profile"), dict):          # an unsaved draft from the editor: validated, not stored
            from .store import validate_profile
            p = dict(body["profile"])
            p.setdefault("id", "draft")
            return validate_profile(p)
        return self.store.get(body.get("id"))

    def plan(self, body):
        p = self._profile_from(body)
        r = self.engine.plan(p)
        r["allowed"] = self.settings["allow_gpu_control"]
        r["autostart"] = [s["server"] for s in p.get("autostart") or []]
        return r

    def _need_write(self, what):
        if not self.settings["allow_gpu_control"]:
            raise Refused("Changing cards is switched off on this machine, so nothing was changed. The Profiles page "
                          "explains how to switch it on.", code="gpu_control_off")

    def apply(self, body, who=None, confirm_needed=True, action="apply"):
        p = self._profile_from(body)
        if p["id"] == "draft":
            raise Invalid("save the profile before applying it")
        if confirm_needed and (body or {}).get("confirm") != p["id"]:
            raise Refused(f"type the profile id ({p['id']}) to confirm", code="confirm")
        plan = self.engine.plan(p)
        res = {"ok": True, "profile": p["id"], "plan": plan, "applied": 0, "autostart": None, "notes": []}
        if plan["changes"]:
            self._need_write(f"apply {p['id']}")
            res = dict(res, **self.engine.apply(p, who=_who(who), action=action))
        else:
            self.audit.write(action, _who(who), target={"profile": p["id"]}, detail={"changes": 0})
            res["notes"].append("every card already matches: nothing to change")
        if res["ok"]:
            present = [c["uuid"] for c in plan["cards"] if not c["missing"]]
            before_active = {u: self.store.data["active"].get(u) for u in present}
            self.store.set_active(present, p["id"])
            if res.get("changes"):
                res["undo_id"] = self._remember_undo(f"{p['name']} on {len(present)} card(s)", res["changes"], before_active)
            if p.get("autostart"):
                if self.supervisor and self.settings["autostart"]:
                    res["autostart"] = self.supervisor.run_profile(p, boot=(action == "boot"), who=_who(who))
                else:
                    res["notes"].append("auto-start is off (\"gpu_autostart\": true in control.json switches it on): "
                                        "no server was started")
        return res

    # ---- undo + the watch after a change ---------------------------------------------------------
    @staticmethod
    def describe(ch):
        s = ch["step"]
        f = s["field"]

        def v(x):
            if f == "power_limit":
                return f"{x:.0f} W" if isinstance(x, (int, float)) else "?"
            if f == "persistence":
                return "on" if x else "off"
            if f in ("app_clocks", "locked_clocks"):
                return "default" if x is None else ("/".join(f"{int(y)}" for y in x) + " MHz")
            return str(x)
        name = {"power_limit": "power", "persistence": "keep driver loaded", "app_clocks": "clocks",
                "locked_clocks": "locked clocks"}[f]
        return f"Card {ch['index']} {name}: {v(s['before'])} \u2192 {v(s['after'])}"

    def _remember_undo(self, label, changes, active_before):
        self.undo_seq += 1
        self.undo_rec = {"id": self.undo_seq, "label": label, "ts": self.clock(), "changes": changes,
                         "active_before": active_before, "lines": [self.describe(c) for c in changes]}
        self.probation = {"until": self.clock() + PROBATION_S, "undo_id": self.undo_seq,
                          "cards": sorted({c["uuid"] for c in changes})}
        return self.undo_seq

    def undo(self, body=None, who=None, action="undo"):
        u = self.undo_rec
        want = (body or {}).get("id")
        if not u or (want is not None and want != u["id"]):
            raise Invalid("There is nothing to undo (or a newer change replaced it). Refresh the page.", code="nothing_to_undo",
                          status=409)
        self._need_write("undo")
        self.guard.check([(c["uuid"], c["index"]) for c in u["changes"]], "undo")
        rolled = self.engine.revert(u["changes"], who=_who(who), action=action)
        ok = all(r["ok"] for r in rolled)
        if ok:
            for uu, pid in u["active_before"].items():
                self.store.set_active([uu], pid)
        self.undo_rec = None
        if self.probation and self.probation.get("undo_id") == u["id"]:
            self.probation = None
        self.audit.write(action, _who(who), target={"label": u["label"]}, ok=ok, detail={"steps": len(rolled)})
        return {"ok": ok, "reverted": rolled, "lines": u["lines"]}

    def _watch(self, now):
        """the minute after a change: a card that vanished, stopped answering or runs into its slowdown temperature
        undoes the change by itself (a beginner cannot brick a card with a profile)."""
        pr = self.probation
        if not pr:
            return
        if now >= pr["until"]:
            self.probation = None
            return
        try:
            cards = {c["uuid"]: c for c in self.adapter.list(max_age=0.0)}
        except Exception as e:                  # noqa: BLE001
            cards, err = {}, str(e)
        else:
            err = self.adapter.error
        bad = []
        for u in pr["cards"]:
            c = cards.get(u)
            if c is None:
                bad.append(f"a card stopped answering the driver ({err or 'not listed'})")
                continue
            hot = c.get("slowdown_c") or 90
            if c.get("temp_c") is not None and c["temp_c"] >= hot:
                bad.append(f"card {c['index']} reached {c['temp_c']:.0f} \u00b0C")
        if bad:
            self.probation = None
            try:
                r = self.undo({"id": pr["undo_id"]}, who="safety watch", action="auto_rollback")
                self.last_auto_rollback = {"ts": now, "why": "; ".join(bad), "ok": r["ok"], "lines": r["lines"]}
            except Exception as e:              # noqa: BLE001
                self.last_auto_rollback = {"ts": now, "why": "; ".join(bad), "ok": False, "error": str(e)}

    def reset(self, body, who=None):
        """power limit back to the card's default, application clocks to default, clock lock released."""
        body = body or {}
        uuids = validate_targets(body.get("targets"))
        if body.get("confirm") != "reset":
            raise Refused("type reset to confirm", code="confirm")
        self._need_write("reset")
        p = {"id": "reset", "name": "Reset", "targets": uuids,
             "gpu": {"power": {"mode": "default", "value": None}, "persistence": None, "app_clocks": "default",
                     "locked_clocks": "reset"}}
        before_active = {u: self.store.data["active"].get(u) for u in uuids}
        r = self.engine.apply(p, who=_who(who), action="reset")
        if r["ok"]:
            self.store.set_active(uuids, None)
            if r.get("changes"):
                r["undo_id"] = self._remember_undo(f"reset of {len(uuids)} card(s)", r["changes"], before_active)
        return r

    def set_power(self, body, who=None):
        """the quick per-card power limit (A/B comparisons from a card tile): {uuid, watts | "default", confirm}.
        Clamped to the card's min..max, gated like a profile (allow_gpu_control + guard), verified, rolled back on failure.
        confirm must repeat the watts asked (or "default"), so a stray request cannot change a card."""
        body = body or {}
        uuid = body.get("uuid")
        if not isinstance(uuid, str) or not UUID_RE.match(uuid):
            raise Invalid("uuid: a card UUID (GPU-...)")
        w = body.get("watts")
        if w == "default":
            spec, word = {"mode": "default", "value": None}, "default"
        else:
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not 30 <= w <= 1000:
                raise Invalid("watts: a number of watts (30..1000) or \"default\"")
            spec, word = {"mode": "watts", "value": float(w)}, str(int(round(w)))
        if str(body.get("confirm")) != word:
            raise Refused(f"confirm must repeat the limit ({word})", code="confirm")
        self._need_write("power limit")
        c = self.adapter.get(uuid, fresh=True)
        target, note = resolve_power(spec, c)
        if target is None:
            raise Invalid(note or "power limit: not adjustable on this card", code="unsupported")
        p = {"id": "quick-power", "name": "Power limit", "targets": [uuid],
             "gpu": {"power": spec, "persistence": None, "app_clocks": None, "locked_clocks": None}}
        r = self.engine.apply(p, who=_who(who), action="power")
        r["target_w"], r["clamp_note"] = target, note
        if r["ok"] and r.get("changes"):
            r["undo_id"] = self._remember_undo(f"card {c['index']} power limit", r["changes"],
                                               {uuid: self.store.data["active"].get(uuid)})
        r["card"] = {"index": c["index"], "min_w": c.get("min_w"), "max_w": c.get("max_w"), "default_w": c.get("default_w"),
                     "before_w": c.get("limit_w")}
        if r["ok"] and r["plan"]["changes"]:
            self.store.set_active([uuid], None)          # a hand-set limit: no profile describes this card any more
        return r

    def clocks(self, uuid):
        if not isinstance(uuid, str) or not UUID_RE.match(uuid):
            raise Invalid("uuid: a card UUID (GPU-...)")
        c = self.adapter.get(uuid)
        sup = self.adapter.supported_clocks(uuid) if c["caps"].get("app_clocks") else {}
        return {"uuid": uuid, "supported": {str(k): v for k, v in sup.items()}, "caps": c["caps"],
                "default": [c.get("def_app_mem"), c.get("def_app_sm")], "max_sm": c.get("max_sm")}

    # ---- locks ------------------------------------------------------------------------------
    def reserve(self, body, who=None):
        body = body or {}
        on = body.get("on", True)
        if not isinstance(on, bool):
            raise Invalid("on: true or false")
        until = body.get("until")
        if until is not None and (isinstance(until, bool) or not isinstance(until, (int, float))):
            raise Invalid("until: a unix time or null")
        self.store.reserve(body.get("targets"), on, why=body.get("why") or "", by=_who(who), until=until)
        self.audit.write("reserve" if on else "unreserve", _who(who), target={"cards": body.get("targets")},
                         after={"why": body.get("why") or ""})
        return {"reserved": self.store.snapshot()["reserved"]}

    def maintenance(self, body, who=None):
        body = body or {}
        self.store.set_maintenance(body.get("on"), why=body.get("why") or "", by=_who(who))
        self.audit.write("maintenance", _who(who), after={"on": body.get("on"), "why": body.get("why") or ""})
        return {"maintenance": self.store.snapshot()["maintenance"]}

    def quiet(self, body, who=None):
        """Benchmark mode (the page's name; the route is /api/gpu/quiet): the bench cards are reserved, auto-starts pause, and (only with
        allow_gpu_control) every other card is capped to cap_pct of its stock limit. Off puts it all back."""
        body = body or {}
        on = body.get("on")
        if not isinstance(on, bool):
            raise Invalid("on: true or false")
        q = self.store.snapshot().get("quiet")
        notes = []
        if on:
            if q and q.get("on"):
                raise Invalid("Benchmark mode is already on.", code="already", status=409)
            bench = validate_targets(body.get("bench"))
            cap = body.get("cap_pct", 70)
            if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 30 <= cap <= 100:
                raise Invalid("cap_pct: 30..100 (% of each card's stock power limit)")
            if body.get("confirm") != "quiet":
                raise Refused("type quiet to confirm", code="confirm")
            cards = self.adapter.list(max_age=0.0)
            others = [c for c in cards if c["uuid"] not in bench and c["caps"].get("power_limit")]
            prior, capped = {}, []
            already = set(self.store.snapshot()["reserved"])
            self.store.reserve(bench, True, why="Benchmark mode: kept free for a speed test", by=_who(who))
            if self.supervisor:
                self.supervisor.pause(who=_who(who))
            if others and self.settings["allow_gpu_control"]:
                free = [c for c in others if not self.guard.reasons(c["uuid"], c["index"])]
                for c in others:
                    if c not in free:
                        notes.append(f"card {c['index']} is locked ({self.guard.reasons(c['uuid'], c['index'])[0]['why']}): not capped")
                todo = [c for c in free if (resolve_power({"mode": "pct_default", "value": cap}, c)[0] or 0) < (c.get("limit_w") or 0) - 0.5]
                if todo:
                    p = {"id": "quiet", "name": "Quiet", "targets": [c["uuid"] for c in todo],
                         "gpu": {"power": {"mode": "pct_default", "value": cap}, "persistence": None, "app_clocks": None,
                                 "locked_clocks": None}}
                    r = self.engine.apply(p, who=_who(who), action="quiet.cap")
                    if r["ok"]:
                        prior = {c["uuid"]: c["limit_w"] for c in todo}
                        capped = [c["uuid"] for c in todo]
                    else:
                        notes.append("capping the other cards failed and was rolled back: " + (r["failure"] or {}).get("error", ""))
            elif others:
                notes.append("Changing cards is switched off here, so the other cards were not slowed down (they were only kept free of auto-starts).")
            q = {"on": True, "since": self.clock(), "bench": bench, "cap_pct": cap, "prior": prior, "capped": capped,
                 "reserved_here": [u for u in bench if u not in already], "by": _who(who)}
            self.store.set_quiet(q)
            self.audit.write("quiet.on", _who(who), target={"bench": bench}, after={"cap_pct": cap, "capped": capped},
                             detail={"notes": notes} if notes else None)
            return {"quiet": q, "notes": notes}
        if not q or not q.get("on"):
            raise Invalid("Benchmark mode is not on.", code="already", status=409)
        if q.get("capped"):
            if self.settings["allow_gpu_control"]:
                cards = {c["uuid"]: c for c in self.adapter.list(max_age=0.0)}
                for u in q["capped"]:
                    c = cards.get(u)
                    want = (q.get("prior") or {}).get(u)
                    if c is None or want is None:
                        continue
                    p = {"id": "quiet-restore", "name": "Quiet restore", "targets": [u],
                         "gpu": {"power": {"mode": "watts", "value": want}, "persistence": None, "app_clocks": None,
                                 "locked_clocks": None}}
                    try:
                        r = self.engine.apply(p, who=_who(who), action="quiet.restore")
                        if not r["ok"]:
                            notes.append(f"card {c['index']}: restore failed ({(r['failure'] or {}).get('error')})")
                    except Locked as e:
                        notes.append(f"card {c['index']}: {e}")
            else:
                notes.append("GPU control is off now: the capped cards keep their quiet limits")
        if q.get("reserved_here"):
            self.store.reserve(q["reserved_here"], False, by=_who(who))
        if self.supervisor:
            self.supervisor.resume(who=_who(who))
        self.store.set_quiet(None)
        self.audit.write("quiet.off", _who(who), before={"bench": q.get("bench"), "capped": q.get("capped")},
                         detail={"notes": notes} if notes else None)
        return {"quiet": None, "notes": notes}

    # ---- schedules, supervisor, import/export -------------------------------------------------
    def set_schedules(self, body, who=None):
        items = (body or {}).get("schedules")
        before = self.store.snapshot()["schedules"]
        out = self.store.set_schedules(items)
        self.audit.write("schedules.save", _who(who), before=before, after=out)
        return {"schedules": out, "active": self.settings["schedules"]}

    def supervisor_action(self, body, who=None):
        if not self.supervisor:
            raise Invalid("no supervisor in this Control")
        a = (body or {}).get("action")
        if a == "pause":
            self.supervisor.pause(who=_who(who))
        elif a == "resume":
            self.supervisor.resume(who=_who(who))
        elif a == "forget":
            self.supervisor.forget_profile((body or {}).get("profile"), who=_who(who),
                                           stop_servers=(body or {}).get("stop") is True)
        elif a == "run":
            if not self.settings["autostart"]:
                raise Refused("auto-start is off: \"gpu_autostart\": true in control.json switches it on", code="autostart_off")
            p = self.store.get((body or {}).get("profile"))
            self.supervisor.run_profile(p, who=_who(who))
        else:
            raise Invalid("action: pause, resume, run or forget")
        return self.supervisor.view()

    def export(self):
        return self.store.export()

    def import_(self, body, who=None):
        body = body or {}
        replace = body.get("replace", False)
        if not isinstance(replace, bool):
            raise Invalid("replace: true or false")
        r = self.store.import_(body.get("data"), replace=replace)
        self.audit.write("profiles.import", _who(who), after=r, detail={"replace": replace})
        return r

    def audit_read(self, limit=200):
        return {"entries": self.audit.read(limit=limit)}

    # ---- launches from the rest of Control -------------------------------------------------------
    def check_launch(self, cards, what="start a server"):
        """App.start asks this with [(uuid, index)] before it spawns anything onto those cards."""
        self.guard.check(cards, what)

    # ---- background: supervisor, schedules, thermal ----------------------------------------------
    def tick(self, now=None):
        now = self.clock() if now is None else now
        self.last_tick = now
        try:
            self._watch(now)
            if self.supervisor:
                self.supervisor.tick(now)
            if self.settings["schedules"]:
                self._schedules(now)
            if self.settings["allow_gpu_control"] or self.settings["autostart"]:
                self._thermal(now)
            self.tick_error = None
        except Exception as e:                  # noqa: BLE001 - the loop must survive anything
            self.tick_error = f"{e.__class__.__name__}: {e}"

    def _schedules(self, now):
        d = self.store.snapshot()
        for s in SCH.due(d["schedules"], now, self.fired):
            self.fired[s["id"]] = SCH.today_key(now)
            try:
                p = self.store.get(s["profile"])
                self.apply({"id": p["id"]}, who=f"schedule {s['id']}", confirm_needed=False, action="schedule")
            except Exception as e:              # noqa: BLE001 - logged, never raised into the loop
                self.audit.write("schedule", f"schedule {s['id']}", target={"profile": s["profile"]}, ok=False, error=str(e))

    def _thermal(self, now):
        d = self.store.snapshot()
        for c in self.cards(max_age=10.0):
            pid = c.get("active_profile")
            t = c.get("temp_c")
            if not pid or t is None:
                continue
            th = d["profiles"][pid]["thermal"]
            warn, act = c["thermal"]["warn_c"], th.get("act_c")
            prev = self.thermal_state.get(c["uuid"])
            if act and t >= act:
                if prev != "act":
                    self.thermal_state[c["uuid"]] = "act"
                    self._thermal_act(c, th, pid, t)
            elif warn and t >= warn:
                if prev is None:
                    self.thermal_state[c["uuid"]] = "warn"
                    self.audit.write("thermal.alert", "supervisor", target={"uuid": c["uuid"], "index": c["index"]},
                                     after={"temp_c": t, "warn_c": warn}, ok=False)
            elif prev and t < (warn or 0) - HYST_C:
                self.thermal_state.pop(c["uuid"], None)

    def _thermal_act(self, c, th, pid, t):
        tgt = {"uuid": c["uuid"], "index": c["index"], "profile": pid}
        if th["action"] == "cap_power" and self.settings["allow_gpu_control"]:
            p = {"id": "thermal", "name": "Thermal cap", "targets": [c["uuid"]],
                 "gpu": {"power": {"mode": "pct_default", "value": th.get("cap_pct") or 70}, "persistence": None,
                         "app_clocks": None, "locked_clocks": None}}
            try:
                w, _n = resolve_power(p["gpu"]["power"], c)
                if w is not None and (c.get("limit_w") or 0) > w + 0.5:
                    self.engine.apply(p, who="supervisor (thermal)", action="thermal.cap")
                    return
            except Locked as e:
                self.audit.write("thermal.cap", "supervisor (thermal)", target=tgt, ok=False, error=str(e))
                return
        elif th["action"] == "stop_autostart" and self.supervisor:
            self.supervisor.pause(who="supervisor (thermal)")
        self.audit.write("thermal.act", "supervisor (thermal)", target=tgt, after={"temp_c": t, "act_c": th.get("act_c"),
                         "action": th["action"]}, ok=False)

    def needs_background(self):
        s = self.settings
        return bool(s["autostart"] or s["schedules"] or s["allow_gpu_control"])

    def start_background(self, every=5.0):
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()

        def loop():
            while not self._stop.wait(every):
                self.tick()
        self._thread = threading.Thread(target=loop, daemon=True, name="pxa-gpuctl")
        self._thread.start()
        return True

    def stop(self, timeout=5.0):
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout)

    def boot(self):
        """at Control start: re-apply the profiles that were active (gpu_boot_apply) and queue their on_boot auto-starts."""
        out = {"reapplied": [], "queued": [], "errors": []}
        d = self.store.snapshot()
        for pid in sorted(set(d["active"].values())):
            p = d["profiles"].get(pid)
            if not p:
                continue
            if self.settings["boot_apply"] and self.settings["allow_gpu_control"]:
                try:
                    r = self.apply({"id": pid}, who="boot", confirm_needed=False, action="boot")
                    out["reapplied"].append({"profile": pid, "ok": r["ok"], "changes": r["plan"]["changes"]})
                    continue
                except Exception as e:          # noqa: BLE001
                    out["errors"].append(f"{pid}: {e}")
            if self.supervisor and self.settings["autostart"]:
                out["queued"] += self.supervisor.run_profile(p, boot=True, who="boot")
        self.store.set_boot(dict(out, ts=self.clock())) if self.store.writable and (out["reapplied"] or out["queued"]) else None
        return out
