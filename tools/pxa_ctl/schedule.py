"""Time-of-day profile switches: {"id", "profile", "at": "HH:MM", "days": "daily" | [0 (Mon) .. 6], "enabled"}.

A schedule fires once per day, at or after its minute, within GRACE_S (so a Control started at 15:00 does not fire the
01:00 switch late), in the machine's local time. Firing = applying the profile through the same engine and guard as
a click (a locked card is refused and logged, never forced).
"""
import time

GRACE_S = 15 * 60


def due(schedules, now=None, fired=None, localtime=time.localtime):
    """-> [schedule] that should fire now; fired = {id: "YYYY-MM-DD"} of the last firing (updated by the caller)."""
    now = time.time() if now is None else now
    fired = fired or {}
    t = localtime(now)
    today = time.strftime("%Y-%m-%d", t)
    mins_now = t.tm_hour * 60 + t.tm_min
    out = []
    for s in schedules or []:
        if not s.get("enabled", True):
            continue
        if s.get("days", "daily") != "daily" and t.tm_wday not in s["days"]:
            continue
        hh, mm = (int(x) for x in s["at"].split(":"))
        start = hh * 60 + mm
        if mins_now < start or (mins_now - start) * 60 > GRACE_S:
            continue
        if fired.get(s["id"]) == today:
            continue
        out.append(s)
    return out


def today_key(now=None, localtime=time.localtime):
    return time.strftime("%Y-%m-%d", localtime(time.time() if now is None else now))
