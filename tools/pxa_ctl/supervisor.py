"""Auto-start of servers a profile names, through PXA Control's own launch path (App.start of a saved server).

One job per (profile, server). States:
    pending -> delayed (delay_s) -> starting -> healthy
                                       |  no /health within health_wait_s, or the process exited
                                       v
                 on-failure: backoff (backoff_s * 2^n, at most 600 s) -> starting ... up to max_retries -> failed
                 CRASH_N failures within CRASH_WINDOW_S -> crash_loop (stops retrying, says so)
    blocked  : the guard refuses one of its cards (race lock, reserved card, maintenance); re-checked every tick and
               started as soon as the lock clears
    paused   : Quiet mode (or the page) paused auto-starts; resumes where it was
    stopped  : somebody stopped the server by hand: never restarted behind their back
Jobs of one profile start in their order; a job waits until the ones before it are healthy or have given up.
Nothing here sleeps: tick(now) is called by PXA Control's background thread (and directly by the tests).
"""
import threading
import time

CRASH_N = 5
CRASH_WINDOW_S = 600.0
BACKOFF_CAP_S = 600.0
DONE = ("healthy", "failed", "crash_loop", "stopped")
ACTIVE = ("starting",)


class Job(object):
    def __init__(self, profile_id, spec):
        self.profile = profile_id
        self.spec = dict(spec)
        self.sid = spec["server"]
        self.state = "pending"
        self.attempts = 0
        self.failures = []
        self.next_at = None
        self.started_at = None
        self.error = None
        self.history = []
        self.paused_from = None

    def to(self, state, now, why=None):
        if state != self.state:
            self.history.append({"ts": round(now, 2), "from": self.state, "to": state, "why": why})
            self.history = self.history[-30:]
        self.state = state
        if why:
            self.error = why if state in ("backoff", "failed", "crash_loop", "blocked") else self.error

    def view(self, now):
        return {"profile": self.profile, "server": self.sid, "state": self.state, "attempts": self.attempts,
                "max_retries": self.spec["max_retries"], "restart": self.spec["restart"], "order": self.spec["order"],
                "next_in_s": round(max(0.0, self.next_at - now), 1) if self.next_at and self.state in ("delayed", "backoff") else None,
                "error": self.error, "history": self.history[-8:],
                "recent_failures": len([t for t in self.failures if now - t <= CRASH_WINDOW_S])}


class Supervisor(object):
    def __init__(self, launcher, guard, audit, clock=time.time, enabled=lambda: False):
        """launcher: .start(sid, spec) (raises on failure), .status(sid) -> {running, healthy, stopped_by_user, exit_code},
        .stop(sid), .cards(sid) -> [(uuid, index)]. enabled(): the gpu_autostart switch, asked on every tick."""
        self.launcher = launcher
        self.guard = guard
        self.audit = audit
        self.clock = clock
        self.enabled = enabled
        self.jobs = []
        self.paused = False
        self.lock = threading.RLock()

    # ---- control ----------------------------------------------------------------------------
    def run_profile(self, profile, boot=False, who="api"):
        """queue the profile's auto-starts (only the on_boot ones at boot). A server already queued keeps its job."""
        with self.lock:
            have = {(j.profile, j.sid) for j in self.jobs if j.state not in ("failed", "crash_loop", "stopped")}
            added = []
            for s in profile.get("autostart") or []:
                if boot and not s.get("on_boot"):
                    continue
                if (profile["id"], s["server"]) in have:
                    continue
                self.jobs = [j for j in self.jobs if not (j.profile == profile["id"] and j.sid == s["server"])]
                j = Job(profile["id"], s)
                if self.paused:
                    j.paused_from, j.state = "pending", "paused"
                self.jobs.append(j)
                added.append(s["server"])
            if added:
                self.audit.write("autostart.queue", who, target={"profile": profile["id"], "servers": added},
                                 detail={"boot": boot})
            return added

    def forget_profile(self, pid, who="api", stop_servers=False):
        with self.lock:
            gone = [j for j in self.jobs if j.profile == pid]
            self.jobs = [j for j in self.jobs if j.profile != pid]
        if stop_servers:
            for j in gone:
                try:
                    self.launcher.stop(j.sid)
                except Exception:          # noqa: BLE001
                    pass
        if gone:
            self.audit.write("autostart.forget", who, target={"profile": pid, "servers": [j.sid for j in gone]},
                             detail={"stopped": stop_servers})
        return len(gone)

    def pause(self, who="api"):
        with self.lock:
            self.paused = True
            now = self.clock()
            for j in self.jobs:
                if j.state not in DONE and j.state != "paused":
                    j.paused_from = j.state if j.state not in ACTIVE else "starting"
                    j.to("paused", now, "auto-starts paused")
        self.audit.write("autostart.pause", who)

    def resume(self, who="api"):
        with self.lock:
            self.paused = False
            now = self.clock()
            for j in self.jobs:
                if j.state == "paused":
                    back = j.paused_from or "pending"
                    if back in ("delayed", "backoff") and j.next_at is not None and j.next_at < now:
                        j.next_at = now
                    if back == "starting":
                        j.started_at = now          # its health wait starts over
                    j.to(back, now, "resumed")
        self.audit.write("autostart.resume", who)

    # ---- the state machine ------------------------------------------------------------------
    def _blocked(self, j):
        try:
            cards = self.launcher.cards(j.sid)
            self.guard.check(cards, f"auto-start of {j.sid}")
            return None
        except Exception as e:             # noqa: BLE001 - Locked, or the server is gone from the config
            return str(e)

    def _fail(self, j, now, why):
        j.failures = [t for t in j.failures if now - t <= CRASH_WINDOW_S] + [now]
        self.audit.write("autostart.fail", "supervisor", target={"profile": j.profile, "server": j.sid}, ok=False,
                         error=why, detail={"attempts": j.attempts})
        if j.spec["restart"] == "on-failure" and len(j.failures) >= CRASH_N:
            j.to("crash_loop", now, f"{len(j.failures)} failures in {CRASH_WINDOW_S / 60:.0f} min: stopped retrying ({why})")
            self.audit.write("autostart.crash_loop", "supervisor", target={"profile": j.profile, "server": j.sid}, ok=False,
                             error=why)
            return
        if j.spec["restart"] == "on-failure" and j.attempts < j.spec["max_retries"]:
            wait = min(BACKOFF_CAP_S, j.spec["backoff_s"] * (2 ** j.attempts))
            j.attempts += 1
            j.next_at = now + wait
            j.to("backoff", now, f"{why}; retry {j.attempts}/{j.spec['max_retries']} in {wait:.0f} s")
        else:
            j.to("failed", now, why)

    def _start(self, j, now):
        try:
            self.launcher.start(j.sid, j.spec)
        except Exception as e:             # noqa: BLE001
            self._fail(j, now, f"start refused: {str(e)[:300]}")
            return
        j.started_at = now
        j.to("starting", now)
        self.audit.write("autostart.start", "supervisor", target={"profile": j.profile, "server": j.sid},
                         detail={"attempt": j.attempts + 1})

    def tick(self, now=None):
        now = self.clock() if now is None else now
        if not self.enabled():
            return
        with self.lock:
            jobs = sorted(self.jobs, key=lambda j: (j.profile, j.spec["order"]))
            gate = {}                           # profile -> True while an earlier job is not done yet
            for j in jobs:
                waiting = gate.get(j.profile, False)
                st = j.state
                if st in ("paused",) or st in ("failed", "crash_loop", "stopped"):
                    pass
                elif st in ("pending", "blocked"):
                    if not waiting and not self.paused:
                        why = self._blocked(j)
                        if why:
                            j.to("blocked", now, why)
                        else:
                            j.next_at = now + j.spec["delay_s"]
                            j.to("delayed", now)
                            if j.spec["delay_s"] <= 0:
                                self._start(j, now)
                elif st in ("delayed", "backoff"):
                    if now >= (j.next_at or 0):
                        why = self._blocked(j)
                        if why:
                            j.to("blocked", now, why)
                        else:
                            self._start(j, now)
                elif st == "starting":
                    s = self._status(j)
                    if s.get("healthy"):
                        j.to("healthy", now)
                        self.audit.write("autostart.healthy", "supervisor", target={"profile": j.profile, "server": j.sid},
                                         detail={"after_s": round(now - (j.started_at or now), 1)})
                    elif not s.get("running"):
                        self._fail(j, now, f"exited before it was healthy (code {s.get('exit_code')})")
                    elif now - (j.started_at or now) > j.spec["health_wait_s"]:
                        try:
                            self.launcher.stop(j.sid)
                        except Exception:  # noqa: BLE001
                            pass
                        self._fail(j, now, f"no healthy /health within {j.spec['health_wait_s']} s: stopped it")
                elif st == "healthy":
                    s = self._status(j)
                    if not s.get("running"):
                        if s.get("stopped_by_user"):
                            j.to("stopped", now, "stopped by hand: not restarted")
                        else:
                            self._fail(j, now, f"exited while serving (code {s.get('exit_code')})")
                if j.state not in DONE and j.state != "paused":
                    gate[j.profile] = True

    def _status(self, j):
        try:
            return self.launcher.status(j.sid) or {}
        except Exception:                  # noqa: BLE001
            return {"running": False}

    def view(self):
        now = self.clock()
        with self.lock:
            return {"paused": self.paused, "enabled": bool(self.enabled()),
                    "jobs": [j.view(now) for j in sorted(self.jobs, key=lambda j: (j.profile, j.spec["order"]))]}
