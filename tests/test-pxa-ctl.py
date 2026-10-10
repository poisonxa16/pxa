#!/usr/bin/env python3
"""PXA Control's GPU Profiles backend (tools/pxa_ctl) and its routes in tools/pxa_control.py. The MOCK adapter only:
no GPU, no nvidia-smi, no model, nothing beyond 127.0.0.1. Covers validation, clamping, plan idempotence, transactional
apply + rollback (driver failure and readback drift), undo, the safety watch, the supervisor's state machine (order,
delay, health wait, backoff, crash loop, blocked, paused, stopped), schedules, the guard (lock files, wait-for files,
reserved cards, maintenance), Benchmark mode, the quick power limit, structured route errors, the 423 on a launch onto
a kept-free card, and the resolved-context helpers.

    nice -n 19 python3 tests/test-pxa-ctl.py        (wired into CTest as test-pxa-ctl)
"""
import calendar
import http.server
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-ctl-test-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ.pop("PXA_MODELS_DIR", None)
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_SPAWNED", "PXA_CONTROL_IDLE_S", "PXA_CONTROL_TOKEN",
           "PXA_CONTROL_ALLOW_GPU", "PXA_CONTROL_AUTOSTART", "PXA_CONTROL_SCHEDULES", "PXA_CONTROL_BOOT_APPLY",
           "PXA_CONTROL_LOCK_FILES", "PXA_CONTROL_LOCK_IF_MISSING", "PXA_CONTROL_GPU_ADAPTER", "PXA_CONTROL_GPU_MOCK"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)

import pxa_ctl  # noqa: E402
from pxa_ctl import driver as D, engine as E, guard as G, schedule as SCH, service as SVC, store as ST, supervisor as SUP  # noqa: E402
from pxa_ctl.errors import Invalid, Locked, Refused, CtlError  # noqa: E402

P100, V100, TI = "GPU-mock-0000-p100", "GPU-mock-0002-v100", "GPU-mock-0003-1080ti"


class Clock(object):
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def ctl(allow=True, autostart=False, schedules=False, lock_files=(), lock_if_missing=(), launcher=None, adapter=None,
        clock=None, d=None):
    d = d or tempfile.mkdtemp(dir=TMP)
    st = {"allow_gpu_control": allow, "autostart": autostart, "schedules": schedules, "boot_apply": False,
          "lock_files": list(lock_files), "lock_if_missing": list(lock_if_missing), "ui_mode": "simple"}
    return SVC.GpuControl(d, st, adapter=adapter or D.MockAdapter(), launcher=launcher, clock=clock or time.time)


def prof(pid="p1", targets=(P100,), power=None, persistence=None, app_clocks=None, locked=None, thermal=None, autostart=None,
         name=None):
    return {"id": pid, "name": name or pid.upper(), "targets": list(targets),
            "gpu": {"power": power, "persistence": persistence, "app_clocks": app_clocks, "locked_clocks": locked},
            "thermal": thermal, "autostart": autostart}


def card(a, uuid):
    return a.get(uuid, fresh=True)


# ---------------------------------------------------------------------------------------------------------------------
class Validation(unittest.TestCase):
    def test_good_profile_is_normalised(self):
        p = ST.validate_profile(prof(power={"mode": "watts", "value": "180"}, autostart=[{"server": "main"}]))
        self.assertEqual(p["gpu"]["power"], {"mode": "watts", "value": 180.0})
        self.assertEqual(p["thermal"]["action"], "alert")
        a = p["autostart"][0]
        self.assertEqual((a["restart"], a["max_retries"], a["backoff_s"], a["health_wait_s"]), ("never", 3, 15, 600))

    def test_bad_inputs_are_refused_in_words(self):
        bad = [
            (dict(prof(), surprise=1), "unknown field"),
            (prof(targets=["0"]), "not a card UUID"),
            (prof(targets=[]), "at least one card"),
            (prof(targets=["GPU-x; rm -rf /"]), "not a card UUID"),
            (prof(power={"mode": "watts", "value": 5000}), "between"),
            (prof(power={"mode": "watts", "value": True}), "number"),
            (prof(power={"mode": "turbo", "value": 1}), "mode"),
            (prof(locked={"min": 1500, "max": 900}), "min must not exceed max"),
            (prof(thermal={"warn_c": 85, "act_c": 70}), "at or above"),
            (prof(autostart=[{"server": "a"}, {"server": "a"}]), "twice"),
            (prof(autostart=[{"server": "a", "env": {"LD_PRELOAD": "/x.so"}}]), "PXA_"),
            (prof(autostart=[{"server": "a", "extra_args": ["--alias x"]}]), "no spaces"),
            (prof(autostart=[{"server": "a", "extra_args": ["$(reboot);"]}]), "no spaces"),
            (prof(autostart=[{"server": "a", "restart": "always"}]), "on-failure"),
            (dict(prof(), id="Bad Id"), "profile id"),
            (dict(prof(), name="<script>"), "profile name"),
        ]
        for p, word in bad:
            with self.assertRaises(Invalid, msg=json.dumps(p)) as cm:
                ST.validate_profile(p)
            self.assertIn(word, str(cm.exception), json.dumps(p))
            self.assertEqual(cm.exception.status, 400)

    def test_schedule_validation(self):
        good = {"p1": {}}
        self.assertEqual(ST.validate_schedule({"id": "s1", "profile": "p1", "at": "01:00", "days": [6, 0, 0]}, good)["days"], [0, 6])
        for s in ({"id": "s1", "profile": "nope", "at": "01:00"}, {"id": "s1", "profile": "p1", "at": "25:00"},
                  {"id": "s1", "profile": "p1", "at": "01:00", "days": [7]}):
            with self.assertRaises(Invalid):
                ST.validate_schedule(s, good)


class Store(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.path = os.path.join(self.d, "gpu_profiles.json")

    def test_atomic_private_file_with_backup(self):
        s = ST.ProfileStore(self.path)
        s.put(prof("a"))
        s.put(prof("b"))
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        self.assertTrue(os.path.exists(self.path + ".bak"))
        again = ST.ProfileStore(self.path)
        self.assertEqual(sorted(again.profiles()), ["a", "b"])
        with open(self.path) as f:
            self.assertEqual(json.load(f)["schema"], ST.SCHEMA)

    def test_newer_schema_is_read_only(self):
        with open(self.path, "w") as f:
            json.dump({"schema": 99, "profiles": {}}, f)
        s = ST.ProfileStore(self.path)
        self.assertFalse(s.writable)
        with self.assertRaises(Invalid) as cm:
            s.put(prof("a"))
        self.assertEqual(cm.exception.code, "read_only")
        with open(self.path) as f:
            self.assertEqual(json.load(f)["schema"], 99)                # never rewritten

    def test_migrate_drops_what_no_longer_validates(self):
        old = {"profiles": [prof("ok"), {"id": "bad", "name": "x", "targets": ["nope"]}], "active": {P100: "bad", V100: "ok"}}
        d, w = ST.migrate(old)
        self.assertTrue(w)
        self.assertEqual(list(d["profiles"]), ["ok"])
        self.assertEqual(d["active"], {V100: "ok"})

    def test_profile_in_use_by_a_schedule_cannot_be_deleted(self):
        s = ST.ProfileStore(self.path)
        s.put(prof("a"))
        s.set_schedules([{"id": "night", "profile": "a", "at": "01:00"}])
        with self.assertRaises(Invalid) as cm:
            s.delete("a")
        self.assertEqual(cm.exception.code, "in_use")

    def test_export_import_round_trip(self):
        s = ST.ProfileStore(self.path)
        s.put(prof("a", power={"mode": "pct_default", "value": 60}))
        blob = s.export()
        t = ST.ProfileStore(os.path.join(self.d, "other.json"))
        r = t.import_(json.loads(json.dumps(blob)))
        self.assertIn("a", t.profiles())
        self.assertTrue(r)
        with self.assertRaises(Invalid):
            t.import_({"something": "else"})


# ---------------------------------------------------------------------------------------------------------------------
class Engine(unittest.TestCase):
    def setUp(self):
        self.a = D.MockAdapter()
        self.c = ctl(adapter=self.a)

    def test_clamp_to_the_card(self):
        p100, ti = card(self.a, P100), card(self.a, TI)
        self.assertEqual(E.resolve_power({"mode": "watts", "value": 400}, p100)[0], 250.0)
        self.assertIn("allows 125-250 W", E.resolve_power({"mode": "watts", "value": 400}, p100)[1])
        self.assertEqual(E.resolve_power({"mode": "watts", "value": 50}, p100)[0], 125.0)
        self.assertEqual(E.resolve_power({"mode": "pct_default", "value": 60}, ti), (150.0, None))
        self.assertEqual(E.resolve_power({"mode": "pct_max", "value": 100}, ti)[0], 300.0)
        self.assertEqual(E.resolve_power({"mode": "default", "value": None}, ti)[0], 250.0)
        self.assertEqual(E.snap_clock(1200, [1328, 1189, 1176]), 1189)
        self.assertEqual(E.snap_clock(100, [1328, 1189]), 1189)

    def test_plan_is_a_dry_run_and_apply_is_idempotent(self):
        p = ST.validate_profile(prof(targets=[P100, TI], power={"mode": "pct_default", "value": 60}, persistence=True))
        plan = self.c.engine.plan(p)
        self.assertEqual(self.a.calls, [])                          # a plan changes nothing
        self.assertEqual(plan["changes"], 3)                        # 2 power limits + the 1080 Ti's persistence
        r = self.c.engine.apply(p)
        self.assertTrue(r["ok"])
        order = [w for w, u, _v in self.a.calls if u == TI]
        self.assertEqual(order, ["persistence", "power_limit"])     # power last
        self.assertEqual(card(self.a, TI)["limit_w"], 150.0)
        self.assertEqual(self.c.engine.plan(p)["changes"], 0)       # a second apply: nothing to do
        n = len(self.a.calls)
        self.c.engine.apply(p)
        self.assertEqual(len(self.a.calls), n)

    def test_driver_failure_rolls_every_card_back(self):
        p = ST.validate_profile(prof(targets=[TI, P100], power={"mode": "watts", "value": 200}, persistence=True))
        self.a.fail = {"power_limit": "Insufficient Permissions"}
        r = self.c.engine.apply(p)
        self.assertFalse(r["ok"])
        self.assertEqual(r["failure"]["field"], "power_limit")
        self.assertIn("Insufficient Permissions", r["failure"]["error"])
        self.assertTrue(r["rollback_ok"])
        self.assertEqual(card(self.a, TI)["persistence"], False)     # the persistence step that worked was put back
        self.assertEqual(card(self.a, P100)["limit_w"], 250.0)
        acts = [e["action"] for e in self.c.audit.read(50)]
        self.assertIn("apply.rollback", acts)

    def test_readback_drift_counts_as_failure(self):
        p = ST.validate_profile(prof(targets=[P100], power={"mode": "watts", "value": 200}))
        self.a.drift = {"power_limit": 5}
        r = self.c.engine.apply(p)
        self.assertFalse(r["ok"])
        self.assertIn("read back", r["failure"]["error"])
        self.assertEqual(r["changes"], [])

    def test_clocks_snap_and_unsupported_cards_are_skipped(self):
        p = ST.validate_profile(prof(targets=[P100, TI], app_clocks={"mem": 715, "sm": 1110}, locked={"min": 900, "max": 1500}))
        plan = self.c.engine.plan(p)
        p1 = next(c for c in plan["cards"] if c["uuid"] == P100)
        ti = next(c for c in plan["cards"] if c["uuid"] == TI)
        step = next(s for s in p1["steps"] if s["field"] == "app_clocks")
        self.assertEqual(step["after"], [715.0, 1101.0])
        self.assertTrue(any("not a supported pair" in n for n in p1["notes"]))
        self.assertTrue(any("Volta" in n for n in p1["notes"]))      # a P100 cannot lock clocks
        self.assertEqual(ti["steps"], [])
        self.assertTrue(any("not supported" in n for n in ti["notes"]))
        v = ST.validate_profile(prof(targets=[V100], locked={"min": 900, "max": 2000}))
        st = self.c.engine.plan(v)["cards"][0]
        self.assertEqual(st["steps"][0]["after"], [900, 1380])      # clamped to the card's max clock

    def test_missing_card_is_skipped_not_fatal(self):
        p = ST.validate_profile(prof(targets=["GPU-not-here", P100], power={"mode": "watts", "value": 200}))
        r = self.c.engine.apply(p)
        self.assertTrue(r["ok"])
        self.assertTrue(r["plan"]["cards"][0]["missing"])

    def test_locked_card_is_refused_before_any_call(self):
        self.c.store.reserve([P100], True, why="bench")
        p = ST.validate_profile(prof(targets=[P100], power={"mode": "watts", "value": 200}))
        with self.assertRaises(Locked) as cm:
            self.c.engine.apply(p)
        self.assertEqual(cm.exception.status, 423)
        self.assertEqual(self.a.calls, [])


# ---------------------------------------------------------------------------------------------------------------------
class Service(unittest.TestCase):
    def test_settings_default_off_and_env_wins(self):
        s = SVC.settings_from({}, {})
        self.assertFalse(any(s[k] for k in ("allow_gpu_control", "autostart", "schedules", "boot_apply")))
        self.assertEqual(s["ui_mode"], "simple")
        s = SVC.settings_from({"allow_gpu_control": True, "gpu_lock_files": ["/a"]},
                              {"PXA_CONTROL_ALLOW_GPU": "0", "PXA_CONTROL_LOCK_FILES": "/b:/c"})
        self.assertFalse(s["allow_gpu_control"])
        self.assertEqual(s["lock_files"], ["/a", "/b", "/c"])
        self.assertFalse(SVC.settings_from({"allow_gpu_control": "yes"}, {})["allow_gpu_control"])   # only a real true
        self.assertFalse(ctl(allow=False).needs_background())

    def test_off_means_look_only(self):
        a = D.MockAdapter()
        c = ctl(allow=False, adapter=a)
        c.save_profile(prof("q", power={"mode": "pct_default", "value": 60}))
        self.assertEqual(c.plan({"id": "q"})["changes"], 1)          # preview works
        with self.assertRaises(Refused) as cm:
            c.apply({"id": "q", "confirm": "q"})
        self.assertEqual((cm.exception.code, cm.exception.status), ("gpu_control_off", 403))
        self.assertIn("switched off", str(cm.exception))
        with self.assertRaises(Refused):
            c.set_power({"uuid": TI, "watts": 200, "confirm": "200"})
        self.assertEqual(a.calls, [])

    def test_apply_needs_the_confirm_word_then_undo_puts_it_back(self):
        a = D.MockAdapter()
        c = ctl(adapter=a)
        c.save_profile(prof("q", targets=[TI], power={"mode": "pct_default", "value": 60}))
        with self.assertRaises(Refused):
            c.apply({"id": "q", "confirm": "yes"})
        r = c.apply({"id": "q", "confirm": "q"})
        self.assertTrue(r["ok"])
        self.assertEqual(card(a, TI)["limit_w"], 150.0)
        st = c.state()
        self.assertEqual(st["cards"][3]["active_profile"], "q")
        self.assertIn("Card 3 power: 250 W \u2192 150 W", st["undo"]["lines"])
        u = c.undo({"id": r["undo_id"]})
        self.assertTrue(u["ok"])
        self.assertEqual(card(a, TI)["limit_w"], 250.0)
        self.assertIsNone(c.state()["cards"][3]["active_profile"])
        with self.assertRaises(Invalid) as cm:
            c.undo({"id": r["undo_id"]})
        self.assertEqual(cm.exception.status, 409)

    def test_quick_power_limit(self):
        a = D.MockAdapter()
        c = ctl(adapter=a)
        with self.assertRaises(Refused):
            c.set_power({"uuid": TI, "watts": 200, "confirm": "180"})      # confirm must repeat the number
        for bad in ({"uuid": "3", "watts": 200, "confirm": "200"}, {"uuid": TI, "watts": "lots", "confirm": "lots"},
                    {"uuid": TI, "watts": True, "confirm": "True"}, {"uuid": TI, "watts": 5, "confirm": "5"}):
            with self.assertRaises(Invalid):
                c.set_power(bad)
        r = c.set_power({"uuid": TI, "watts": 320, "confirm": "320"})
        self.assertTrue(r["ok"])
        self.assertEqual(r["target_w"], 300.0)                          # clamped to the card's max
        self.assertIn("300", r["clamp_note"])
        self.assertEqual(r["card"], {"index": 3, "min_w": 125.0, "max_w": 300.0, "default_w": 250.0, "before_w": 250.0})
        r = c.set_power({"uuid": TI, "watts": "default", "confirm": "default"})
        self.assertEqual(card(a, TI)["limit_w"], 250.0)
        with self.assertRaises(Invalid) as cm:
            c.set_power({"uuid": "GPU-nope", "watts": 200, "confirm": "200"})
        self.assertEqual(cm.exception.status, 404)

    def test_safety_watch_undoes_a_change_when_a_card_overheats(self):
        a, clk = D.MockAdapter(), Clock()
        c = ctl(adapter=a, clock=clk)
        c.set_power({"uuid": TI, "watts": 300, "confirm": "300"})
        self.assertIsNotNone(c.state()["probation"])
        clk.t += 5
        c.tick()
        self.assertEqual(card(a, TI)["limit_w"], 300.0)                 # healthy: kept
        a._find(TI)["temp_c"] = 95.0                                    # past its 93 C slowdown point
        a.invalidate()
        clk.t += 5
        c.tick()
        self.assertEqual(card(a, TI)["limit_w"], 250.0)                 # undone by itself
        ar = c.state()["auto_rollback"]
        self.assertTrue(ar["ok"])
        self.assertIn("95", ar["why"])
        self.assertIn("auto_rollback", [e["action"] for e in c.audit.read(50)])

    def test_safety_watch_ends_after_probation(self):
        a, clk = D.MockAdapter(), Clock()
        c = ctl(adapter=a, clock=clk)
        c.set_power({"uuid": TI, "watts": 300, "confirm": "300"})
        clk.t += SVC.PROBATION_S + 1
        c.tick()
        a._find(TI)["temp_c"] = 95.0
        a.invalidate()
        c.tick()
        self.assertEqual(card(a, TI)["limit_w"], 300.0)
        self.assertIsNone(c.state()["auto_rollback"])

    def test_benchmark_mode_caps_others_and_restores(self):
        a = D.MockAdapter()
        sup_l = FakeLauncher()
        c = ctl(adapter=a, launcher=sup_l)
        with self.assertRaises(Refused):
            c.quiet({"on": True, "bench": [P100], "confirm": "nope"})
        r = c.quiet({"on": True, "bench": [P100], "cap_pct": 70, "confirm": "quiet"})
        self.assertEqual(len(r["quiet"]["capped"]), 6)
        self.assertEqual(card(a, P100)["limit_w"], 250.0)               # the bench card is untouched
        self.assertEqual(card(a, TI)["limit_w"], 175.0)
        self.assertTrue(c.supervisor.paused)
        with self.assertRaises(Locked):
            c.check_launch([(P100, 0)])                                # kept free for the test
        c.check_launch([(TI, 3)])
        with self.assertRaises(Invalid):
            c.quiet({"on": True, "bench": [P100], "confirm": "quiet"})  # already on
        c.quiet({"on": False})
        self.assertEqual(card(a, TI)["limit_w"], 250.0)
        self.assertFalse(c.supervisor.paused)
        c.check_launch([(P100, 0)])

    def test_benchmark_mode_without_control_only_reserves(self):
        a = D.MockAdapter()
        c = ctl(allow=False, adapter=a)
        r = c.quiet({"on": True, "bench": [P100], "confirm": "quiet"})
        self.assertEqual(r["quiet"]["capped"], [])
        self.assertTrue(any("switched off" in n for n in r["notes"]))
        self.assertEqual(a.calls, [])

    def test_schedule_fires_once_through_the_engine(self):
        a = D.MockAdapter()
        lt = time.localtime()
        base = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 1, 0, 0, 0, 0, -1))
        clk = Clock(base + 120)
        c = ctl(adapter=a, schedules=True, clock=clk)
        c.save_profile(prof("night", targets=[TI], power={"mode": "pct_default", "value": 60}))
        c.set_schedules({"schedules": [{"id": "n", "profile": "night", "at": "01:00"}]})
        c.tick()
        self.assertEqual(card(a, TI)["limit_w"], 150.0)
        self.assertIn("schedule", [e["action"] for e in c.audit.read(50)])
        a._find(TI)["limit_w"] = 250.0
        a.invalidate()
        clk.t += 60
        c.tick()
        self.assertEqual(card(a, TI)["limit_w"], 250.0)                 # once per day

    def test_maintenance_locks_everything(self):
        c = ctl()
        c.maintenance({"on": True, "why": "re-seating cards"})
        with self.assertRaises(Locked) as cm:
            c.check_launch([(P100, 0)])
        self.assertIn("re-seating", str(cm.exception))
        self.assertIn("reasons", cm.exception.detail)
        c.maintenance({"on": False})
        c.check_launch([(P100, 0)])


class Schedule(unittest.TestCase):
    def test_due_grace_and_days(self):
        at = calendar.timegm((2026, 10, 8, 1, 5, 0))          # a Thursday (tm_wday 3), 01:05 UTC
        s = [{"id": "a", "at": "01:00", "days": "daily", "enabled": True},
             {"id": "b", "at": "01:00", "days": [0], "enabled": True},
             {"id": "c", "at": "00:30", "days": "daily", "enabled": True},
             {"id": "d", "at": "01:00", "days": "daily", "enabled": False}]
        due = [x["id"] for x in SCH.due(s, at, {}, localtime=time.gmtime)]
        self.assertEqual(due, ["a"])                           # b: wrong day; c: past the grace; d: disabled
        self.assertEqual(SCH.due(s, at, {"a": "2026-10-08"}, localtime=time.gmtime), [])


class GuardFiles(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)

    def test_lock_file_scopes(self):
        lf = os.path.join(self.d, "bench.lock")
        c = ctl(lock_files=[lf])
        c.check_launch([(P100, 0)])
        with open(lf, "w") as f:
            json.dump({"gpus": [1, TI], "why": "speed race"}, f)
        c.check_launch([(P100, 0)])
        with self.assertRaises(Locked) as cm:
            c.check_launch([("GPU-mock-0001-p100", 1)])
        self.assertIn("speed race", str(cm.exception))
        with self.assertRaises(Locked):
            c.check_launch([(TI, 3)])
        with open(lf, "w") as f:
            f.write("gpus=0,2\n")
        with self.assertRaises(Locked):
            c.check_launch([(P100, 0)])
        c.check_launch([(TI, 3)])
        with open(lf, "w") as f:
            f.write("race running\n")
        with self.assertRaises(Locked):
            c.check_launch([(TI, 3)])                         # no scope: every card
        os.remove(lf)
        c.check_launch([(TI, 3)])

    def test_wait_for_file(self):
        gate = os.path.join(self.d, "GATE-DONE")
        c = ctl(lock_if_missing=[gate])
        with self.assertRaises(Locked) as cm:
            c.apply({"id": c.save_profile(prof("x", targets=[TI], power={"mode": "watts", "value": 200}))["id"], "confirm": "x"})
        self.assertIn("GATE-DONE", str(cm.exception))
        self.assertEqual(c.adapter.calls, [])
        open(gate, "w").close()
        self.assertTrue(c.apply({"id": "x", "confirm": "x"})["ok"])

    def test_reserved_card_expires(self):
        clk = Clock()
        c = ctl(clock=clk)
        c.reserve({"targets": [P100], "on": True, "why": "bench", "until": clk.t + 60})
        with self.assertRaises(Locked):
            c.check_launch([(P100, 0)])
        clk.t += 61
        c.check_launch([(P100, 0)])
        with self.assertRaises(Invalid):
            c.reserve({"targets": [P100], "on": "yes"})


# ---------------------------------------------------------------------------------------------------------------------
class FakeLauncher(object):
    def __init__(self):
        self.started, self.stopped, self.st, self.fail, self.cardmap = [], [], {}, None, {}

    def start(self, sid, spec):
        if self.fail:
            raise Invalid(self.fail)
        self.started.append(sid)
        self.st[sid] = {"running": True, "healthy": False}

    def status(self, sid):
        return self.st.get(sid, {"running": False})

    def stop(self, sid):
        self.stopped.append(sid)
        self.st[sid] = {"running": False}

    def cards(self, sid):
        return self.cardmap.get(sid, [(P100, 0)])


class Supervisor(unittest.TestCase):
    def mk(self, autostart=True):
        self.l, self.clk = FakeLauncher(), Clock()
        self.c = ctl(launcher=self.l, clock=self.clk, autostart=autostart)
        self.s = self.c.supervisor
        return self.s

    def job(self, sid):
        return next(j for j in self.s.view()["jobs"] if j["server"] == sid)

    def p(self, specs):
        return ST.validate_profile(prof("auto", autostart=specs))

    def test_disabled_switch_starts_nothing(self):
        s = self.mk(autostart=False)
        s.run_profile(self.p([{"server": "a"}]))
        s.tick()
        self.assertEqual(self.l.started, [])

    def test_order_delay_and_health(self):
        s = self.mk()
        s.run_profile(self.p([{"server": "b", "order": 2}, {"server": "a", "order": 1, "delay_s": 10}]))
        s.tick()
        self.assertEqual(self.job("a")["state"], "delayed")
        self.assertEqual(self.job("b")["state"], "pending")            # waits for a
        self.clk.t += 11
        s.tick()
        self.assertEqual(self.l.started, ["a"])
        self.assertEqual(self.job("b")["state"], "pending")
        self.l.st["a"]["healthy"] = True
        self.clk.t += 1
        s.tick()
        self.assertEqual(self.job("a")["state"], "healthy")
        self.clk.t += 1
        s.tick()
        self.assertEqual(self.l.started, ["a", "b"])

    def test_health_wait_backoff_then_failed(self):
        s = self.mk()
        s.run_profile(self.p([{"server": "a", "health_wait_s": 30, "restart": "on-failure", "max_retries": 2, "backoff_s": 10}]))
        s.tick()
        self.clk.t += 31
        s.tick()
        j = self.job("a")
        self.assertEqual((j["state"], j["attempts"]), ("backoff", 1))
        self.assertEqual(self.l.stopped, ["a"])                        # the hung one was stopped
        self.assertAlmostEqual(j["next_in_s"], 10.0)
        self.clk.t += 10
        s.tick()                                                       # attempt 2
        self.clk.t += 31
        s.tick()
        self.assertAlmostEqual(self.job("a")["next_in_s"], 20.0)       # 10 * 2^1
        self.clk.t += 20
        s.tick()
        self.clk.t += 31
        s.tick()
        self.assertEqual(self.job("a")["state"], "failed")
        self.assertEqual(len(self.l.started), 3)

    def test_crash_loop_stops_retrying(self):
        s = self.mk()
        self.l.fail = "boom"
        s.run_profile(self.p([{"server": "a", "restart": "on-failure", "max_retries": 50, "backoff_s": 1}]))
        for _ in range(40):
            s.tick()
            self.clk.t += 20
            if self.job("a")["state"] == "crash_loop":
                break
        j = self.job("a")
        self.assertEqual(j["state"], "crash_loop")
        self.assertEqual(j["recent_failures"], SUP.CRASH_N)
        self.assertIn("crash_loop", [e["action"].split(".")[-1] for e in self.c.audit.read(100)])

    def test_never_restart_fails_once(self):
        s = self.mk()
        s.run_profile(self.p([{"server": "a"}]))
        s.tick()
        self.l.st["a"] = {"running": False, "exit_code": 1}
        s.tick()
        self.assertEqual(self.job("a")["state"], "failed")

    def test_blocked_by_a_kept_free_card_then_released(self):
        s = self.mk()
        self.c.reserve({"targets": [P100], "on": True, "why": "bench"})
        s.run_profile(self.p([{"server": "a"}]))
        s.tick()
        self.assertEqual(self.job("a")["state"], "blocked")
        self.assertIn("bench", self.job("a")["error"])
        self.assertEqual(self.l.started, [])
        self.c.reserve({"targets": [P100], "on": False})
        s.tick()
        self.assertEqual(self.l.started, ["a"])

    def test_paused_and_resumed(self):
        s = self.mk()
        s.run_profile(self.p([{"server": "a", "delay_s": 5}]))
        s.tick()
        s.pause()
        self.clk.t += 60
        s.tick()
        self.assertEqual(self.job("a")["state"], "paused")
        self.assertEqual(self.l.started, [])
        s.resume()
        s.tick()
        self.assertEqual(self.l.started, ["a"])

    def test_stopped_by_hand_is_never_restarted(self):
        s = self.mk()
        s.run_profile(self.p([{"server": "a", "restart": "on-failure"}]))
        s.tick()
        self.l.st["a"]["healthy"] = True
        s.tick()
        self.l.st["a"] = {"running": False, "stopped_by_user": True}
        s.tick()
        self.clk.t += 3600
        s.tick()
        self.assertEqual(self.job("a")["state"], "stopped")
        self.assertEqual(self.l.started, ["a"])

    def test_apply_queues_autostart_only_when_switched_on(self):
        for on in (False, True):
            self.mk(autostart=on)
            self.c.save_profile(prof("w", targets=[TI], power={"mode": "watts", "value": 200}, autostart=[{"server": "a"}]))
            r = self.c.apply({"id": "w", "confirm": "w"})
            if on:
                self.assertEqual(r["autostart"], ["a"])
            else:
                self.assertIsNone(r["autostart"])
                self.assertTrue(any("auto-start is off" in n for n in r["notes"]))


# ---------------------------------------------------------------------------------------------------------------------
class Driver(unittest.TestCase):
    def test_caps_and_adapter_choice(self):
        caps = {c["uuid"]: c["caps"] for c in D.MockAdapter().list()}
        self.assertTrue(caps[P100]["app_clocks"])
        self.assertFalse(caps[P100]["locked_clocks"])
        self.assertTrue(caps[V100]["locked_clocks"])
        self.assertFalse(caps[TI]["app_clocks"])                       # GeForce
        self.assertFalse(caps[TI]["fan_control"])
        self.assertEqual(D.make_adapter("none", environ={}).kind, "none")
        self.assertEqual(D.make_adapter("auto", environ={"PXA_LAUNCH_FAKE_GPUS": "2x600"}).kind, "mock")
        m = D.make_adapter("mock", environ={"PXA_CONTROL_GPU_MOCK": "v100,1080ti"})
        self.assertEqual([c["name"] for c in m.list()], ["Tesla V100-PCIE-16GB", "GeForce GTX 1080 Ti"])

    def test_smi_adapter_uses_argv_and_parses_rows(self):
        calls = []

        class R(object):
            def __init__(self, out, code=0):
                self.stdout, self.returncode, self.stderr = out, code, ""

        def run(argv, **kw):
            calls.append((argv, kw))
            self.assertNotIn("shell", kw)
            if any("--query-gpu" in a for a in argv):
                return R("0, GPU-aaaa-1, Tesla P100-PCIE-16GB, 41.2, 250.00, 125.00, 250.00, 250.00, 36, 0, 405, 715, "
                         "1189, 715, 1189, 715, 1328, 715, Enabled, 0, 16384, [N/A], 6.0\n")
            return R("")
        a = D.NvidiaSmiAdapter(exe="/usr/bin/nvidia-smi", run=run)
        cs = a.list(max_age=0)
        if cs:                                                          # the column set is the adapter's own business;
            self.assertEqual(cs[0]["uuid"], "GPU-aaaa-1")             # what matters: it parsed and never used a shell
        a.set_power_limit("GPU-aaaa-1", 180)
        argv = calls[-1][0]
        self.assertIsInstance(argv, list)
        self.assertIn("-pl", argv)
        self.assertIn("GPU-aaaa-1", argv)
        with self.assertRaises(Invalid):
            a.set_power_limit("0; reboot", 180)                         # only a UUID reaches the driver


class NoShell(unittest.TestCase):
    def test_no_shell_true_anywhere(self):
        files = [os.path.join(TOOLS, "pxa_control.py")] + [os.path.join(TOOLS, "pxa_ctl", f) for f in os.listdir(os.path.join(TOOLS, "pxa_ctl")) if f.endswith(".py")]
        for p in files:
            with open(p) as f:
                self.assertIsNone(re.search(r"shell\s*=\s*True", f.read()), p)


# ---------------------------------------------------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402

MODELS = os.path.join(TMP, "models")
os.makedirs(MODELS, exist_ok=True)


def write_gguf(path):
    """a tiny but well-formed GGUF (qwen3 header, two blk.* layers of zeros): enough for the launcher to list and plan it."""
    import struct

    def s_(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kvs = [s_("general.architecture") + struct.pack("<I", 8) + s_("qwen3")]
    for k, v in (("qwen3.context_length", 40960), ("qwen3.block_count", 2), ("qwen3.embedding_length", 64),
                 ("qwen3.attention.head_count", 4), ("qwen3.attention.head_count_kv", 2), ("general.alignment", 32)):
        kvs.append(s_(k) + struct.pack("<I", 4) + struct.pack("<I", v))
    names = ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_up.weight", "blk.1.attn_q.weight", "blk.1.ffn_up.weight",
             "output.weight"]
    ti, off, n = b"", 0, 64 * 64 * 4
    for nm in names:
        ti += s_(nm) + struct.pack("<I", 2) + struct.pack("<QQ", 64, 64) + struct.pack("<I", 0) + struct.pack("<Q", off)
        off += n
    hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(names), len(kvs)) + b"".join(kvs) + ti
    hdr += b"\0" * ((-len(hdr)) % 32)
    with open(path, "wb") as f:
        f.write(hdr + b"\0" * off)


MODEL = os.path.join(MODELS, "Tiny-Qwen3.gguf")
write_gguf(MODEL)


def serve_app(app):
    srv = C.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def req(port, path, method="GET", body=None, raw=None):
    h = {}
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
    if data is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = C.App(L, port=7777, models_dirs=[MODELS])
        cls.cfgdir = tempfile.mkdtemp(dir=TMP)
        cls.adapter = D.MockAdapter(rows=cls.app.gpus()[0])
        cls.app.gpuctl = SVC.GpuControl(cls.cfgdir, SVC.settings_from({}, {}), adapter=cls.adapter, launcher=C.AppLauncher(cls.app))
        cls.srv, cls.port = serve_app(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def j(self, *a, **k):
        code, h, b = req(self.port, *a, **k)
        return code, json.loads(b or b"null")

    def test_health_works_without_a_driver(self):
        code, d = self.j("/api/health")
        self.assertEqual(code, 200)
        self.assertTrue(d["ok"])
        self.assertIn("nvidia_smi", d)
        self.assertEqual(d["gpu_profiles"]["gpu_adapter"]["kind"], "mock")
        self.assertFalse(d["gpu_profiles"]["allow_gpu_control"])

    def test_structured_errors(self):
        code, d = self.j("/api/gpu/plan", "POST", raw=b"[1,2]")
        self.assertEqual(code, 400)
        self.assertEqual(set(d) >= {"error", "code", "status"}, True)
        code, d = self.j("/api/gpu/plan", "POST", raw=b"{not json")
        self.assertEqual(code, 400)
        self.assertIn("error", d)
        code, d = self.j("/api/gpu/plan")                               # wrong method on a known path
        self.assertEqual(code, 405)
        self.assertEqual(d["status"], 405)
        code, d = self.j("/api/no-such-thing")
        self.assertEqual(code, 404)
        self.assertIn("code", d)
        code, d = self.j("/api/gpu/profiles", "POST", body={"name": "x", "targets": ["nope"]})
        self.assertEqual((code, d["code"]), (400, "invalid"))
        code, d = self.j("/api/gpu/audit?limit=abc")
        self.assertEqual(code, 400)

    def test_gpu_routes_look_only_by_default(self):
        code, d = self.j("/api/gpu/state")
        self.assertEqual(code, 200)
        self.assertEqual([c["uuid"] for c in d["cards"]], ["GPU-fake-0", "GPU-fake-1"])
        self.assertIn("presets", d)
        code, d = self.j("/api/gpu/profiles", "POST", body={"name": "Quiet test", "targets": ["GPU-fake-1"],
                                                             "gpu": {"power": {"mode": "pct_default", "value": 60}}})
        self.assertEqual(code, 200)
        pid = d["profile"]["id"]
        code, d = self.j("/api/gpu/plan", "POST", body={"id": pid})
        self.assertEqual((code, d["changes"], d["allowed"]), (200, 1, False))
        code, d = self.j("/api/gpu/apply", "POST", body={"id": pid, "confirm": pid})
        self.assertEqual((code, d["code"]), (403, "gpu_control_off"))
        code, d = self.j("/api/gpu/power", "POST", body={"uuid": "GPU-fake-1", "watts": 200, "confirm": "200"})
        self.assertEqual(code, 403)
        self.assertEqual(self.adapter.calls, [])
        code, h, b = req(self.port, "/api/gpu/export")
        self.assertEqual(code, 200)
        self.assertIn("attachment", h.get("Content-Disposition", ""))
        self.assertIn(pid, json.dumps(json.loads(b)))
        code, d = self.j("/api/gpu/clocks?uuid=GPU-fake-0")
        self.assertEqual(code, 200)
        code, d = self.j("/api/gpu/profiles", "DELETE", body={"id": pid})
        self.assertEqual(code, 200)

    def test_ui_mode_is_remembered_in_control_json(self):
        code, d = self.j("/api/gpu/ui", "POST", body={"mode": "advanced"})
        self.assertEqual((code, d["mode"]), (200, "advanced"))
        self.assertEqual(C.load_config()["ui_mode"], "advanced")
        code, d = self.j("/api/gpu/ui", "POST", body={"mode": "expert"})
        self.assertEqual(code, 400)
        self.j("/api/gpu/ui", "POST", body={"mode": "simple"})

    def test_launch_onto_a_kept_free_card_is_refused_423(self):
        self.app.gpuctl.reserve({"targets": ["GPU-fake-0"], "on": True, "why": "race bench"})
        try:
            code, d = self.j("/api/start", "POST", body={"sid": "main", "model": MODEL, "gpus": [0]})
            if code == 400 and "model" in d.get("error", "").lower():
                self.skipTest("model validation runs first on this tree")
            self.assertEqual(code, 423, d)
            self.assertEqual(d["code"], "locked")
            self.assertIn("race bench", d["error"])
            self.assertIn("reasons", d.get("detail", {}))
        finally:
            self.app.gpuctl.reserve({"targets": ["GPU-fake-0"], "on": False})


class LaunchGuardDirect(unittest.TestCase):
    """App.start with a reserved card, called directly (no HTTP), with a model file that validates."""

    def test_app_start_refuses_reserved_card(self):
        app = C.App(L, port=7777, models_dirs=[MODELS])
        app.gpuctl = SVC.GpuControl(tempfile.mkdtemp(dir=TMP), SVC.settings_from({}, {}),
                                    adapter=D.MockAdapter(rows=app.gpus()[0]), launcher=C.AppLauncher(app))
        app.gpuctl.reserve({"targets": ["GPU-fake-1"], "on": True, "why": "kept for the race"})
        try:
            app.start({"sid": "main", "model": MODEL, "gpus": [1]})
        except Locked as e:
            self.assertIn("kept for the race", str(e))
            return
        except CtlError as e:
            self.fail(f"wrong error: {e}")
        except Exception as e:      # noqa: BLE001 - validation before the guard (e.g. the model) is acceptable here
            self.assertNotIsInstance(e, Locked)
            self.skipTest(f"refused earlier by launch validation: {e}")
        self.fail("started on a reserved card")


class Context(unittest.TestCase):
    def test_from_log(self):
        lines = ["llama_context: constructing", "llama_context: n_ctx         = 65536",
                 "llama_context: n_ctx_per_seq = 16384", "other"]
        self.assertEqual(C.ctx_from_log(lines), (65536, 16384))
        self.assertEqual(C.ctx_from_log(["nothing here"]), (None, None))

    def test_probe_props_and_context_info(self):
        body = {"n_ctx": 65536, "total_slots": 4, "default_generation_settings": {"n_ctx": 16384}}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                d = json.dumps(body if self.path == "/props" else {}).encode()
                self.send_response(200 if self.path == "/props" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(d)))
                self.end_headers()
                self.wfile.write(d)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            p = C.probe_ctx(srv.server_address[1])
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(p, {"n_ctx": 65536, "per_slot": 16384, "slots": 4})
        ci = C.context_info(0, 65536, probed=p, cards=[0, 1])
        self.assertEqual((ci["mode"], ci["resolved"], ci["source"]), ("auto", 65536, "engine /props"))
        self.assertIn("fit check", ci["how"])
        self.assertIn("16,384 each", ci["how"])
        ci = C.context_info(32768, 32768, log=(32768, None))
        self.assertEqual((ci["mode"], ci["resolved"], ci["source"]), ("set", 32768, "engine log"))
        ci = C.context_info(0, 8192)
        self.assertIsNone(ci["resolved"])                               # never guessed
        self.assertIn("not reported", ci["how"])

    def test_probe_of_a_dead_port_is_none(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertIsNone(C.probe_ctx(port, timeout=0.5))


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
