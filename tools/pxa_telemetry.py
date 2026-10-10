#!/usr/bin/env python3
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
"""PXA Control telemetry history: a small time-series store for the Live tab.

The Live tab used to know only what happened while a page was open (the sampler stopped ten minutes after
the last viewer, and three hours were kept in memory). This module keeps the same numbers on disk, filled
by the same sampler running in the background inside the Control process, so a card's or a server's speed
can be graphed over days and weeks whether or not anybody was looking.

  * one SQLite file in the Control config dir (telemetry.db, mode 0600), WAL journal, ONE writer thread
    (every write goes through a queue to that thread; readers open their own read-only connection);
  * per card: memory used, load, temperature, power, power limit, SM clock;
    per server: decode / prefill t/s, busy slots, slot count, KV cells in use, expert-cache hit rate and
    swaps, draft acceptance, plus the requests that finished in each interval (count, tokens, time);
    this machine: RAM, CPU, swap; and every finished request's own record (raw retention only);
  * retention: raw samples (default every 10 s) kept raw_days (7), one-minute rollups kept rollup_days (90),
    a size cap (max_mb, 512) that trims the oldest raw rows first; all configurable;
  * no new dependency: sqlite3 is in the Python standard library.

Nothing in it leaves the machine. A prompt's text never reaches it (the sampler drops it on the way in).
Values are numbers only; a server is named by the label PXA Control shows, a card by its index and model name.
"""
import csv
import io
import json
import math
import os
import queue
import re
import sqlite3
import threading
import time

SCHEMA_VERSION = 1
CARD_COLS = ("mem", "util", "temp", "power", "limit", "clock")
SRV_COLS = ("dec", "pre", "busy", "slots", "ctx", "xhit", "xswap", "acc", "ema",
            "req", "gen_tok", "prompt_tok", "gen_ms", "prompt_ms")
HOST_COLS = ("ram", "cpu", "swap")
KINDS = {"card": CARD_COLS, "server": SRV_COLS, "host": HOST_COLS}
# digits kept on disk: SQLite writes a REAL with no fraction as a 1-3 byte integer, so whole MiB / % / C / W / MHz
# and token counts cost a third of a full double (a week of raw 10 s rows for 8 cards stays in the tens of MB)
PREC = {"mem": 0, "util": 0, "temp": 0, "power": 0, "limit": 0, "clock": 0, "ram": 0, "cpu": 1, "swap": 0,
        "dec": 2, "pre": 1, "busy": 2, "slots": 0, "ctx": 3, "xhit": 3, "xswap": 2, "acc": 3, "ema": 3,
        "req": 0, "gen_tok": 0, "prompt_tok": 0, "gen_ms": 0, "prompt_ms": 0}
SUM_COLS = frozenset(("req", "gen_tok", "prompt_tok", "gen_ms", "prompt_ms"))   # added up over a bucket, not averaged
REQ_COLS = ("slot", "n_prompt", "n_cached", "prompt_n", "prompt_ms", "prefill_tps", "n_gen", "gen_ms",
            "decode_tps", "draft_n", "draft_acc")
DERIVED_SRV = ("pre_req", "dec_req")        # per bucket: prompt_tok / prompt_ms, gen_tok / gen_ms (t/s of finished requests)
NICE_STEPS = (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)

DEFAULTS = {"enabled": True, "sample_s": 10.0, "raw_days": 7.0, "rollup_days": 90.0, "max_mb": 512.0,
            "prometheus": False}
_ENV = {"enabled": "PXA_CONTROL_TELEMETRY", "sample_s": "PXA_CONTROL_TELEMETRY_SAMPLE_S",
        "raw_days": "PXA_CONTROL_TELEMETRY_RAW_DAYS", "rollup_days": "PXA_CONTROL_TELEMETRY_ROLLUP_DAYS",
        "max_mb": "PXA_CONTROL_TELEMETRY_MAX_MB", "prometheus": "PXA_CONTROL_METRICS"}
_LIMITS = {"sample_s": (2.0, 3600.0), "raw_days": (0.05, 365.0), "rollup_days": (1.0, 3650.0), "max_mb": (16.0, 1 << 20)}
_OFF = ("0", "off", "no", "false", "never")
_ON = ("1", "on", "yes", "true", "always")


def settings(cfg=None, env=None):
    """the effective settings: DEFAULTS, then the control.json "telemetry" object, then the environment
    (PXA_CONTROL_TELEMETRY=0 turns the store off, PXA_CONTROL_METRICS=1 opens /metrics, ...). Out-of-range
    numbers are clamped; text that is not a number is ignored."""
    env = os.environ if env is None else env
    out = dict(DEFAULTS)
    src = (cfg or {}).get("telemetry") if isinstance((cfg or {}).get("telemetry"), dict) else {}
    for k in DEFAULTS:
        v = src.get(k)
        e = env.get(_ENV[k])
        if e is not None and str(e).strip() != "":
            v = e
        if v is None:
            continue
        if isinstance(DEFAULTS[k], bool):
            s = str(v).strip().lower()
            if isinstance(v, bool):
                out[k] = v
            elif s in _ON:
                out[k] = True
            elif s in _OFF:
                out[k] = False
        else:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f != f:
                continue
            lo, hi = _LIMITS[k]
            out[k] = min(hi, max(lo, f))
    out["raw_days"] = min(out["raw_days"], out["rollup_days"])
    return out


def _nice_step(x, floor):
    x = max(float(floor), float(x))
    for s in NICE_STEPS:
        if s >= x:
            return s
    return int(math.ceil(x / 86400.0)) * 86400


def _fin(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _r(v, nd):
    return None if v is None else round(v, nd)


class Store(object):
    """the on-disk history. Writes: record() / record_requests() / flush() queue work for the writer thread.
    Reads: series() / requests() / list_series() / csv_text() open their own connection."""

    def __init__(self, path, cfg=None, clock=time.time):
        self.path = path
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.clock = clock
        self.q = queue.Queue(maxsize=10000)
        self.acc = {}                   # (kind, key) -> [n, sums..., counts..., t_sum, label, info]
        self.acc_lock = threading.Lock()
        self.last_flush = 0.0
        self.sid_cache = {}
        self.rolled = {}                          # kind -> rollup watermark (meta rolled_<kind>), writer thread only
        self.late = {}                            # kind -> oldest raw ts written behind the watermark since the last pass
        self.thread = None
        self.stop_ev = threading.Event()
        self.error = None
        self.writes = 0
        self.dropped = 0
        self.last_maint = 0.0
        self.counters = {}              # server key -> {"req": n, "gen_tok": n, "prompt_tok": n} since this process started
        self.is_writer = False
        self._lock_fd = None
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        con = self._connect()
        try:
            self._init_schema(con)
        finally:
            con.close()
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    # ---- connection and schema --------------------------------------------------------------------
    def _connect(self, readonly=False):
        if readonly:
            # a plain connection that refuses writes: a mode=ro URI cannot open a WAL file whose -shm is gone
            con = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            con.execute("PRAGMA query_only=1")
            return con
        con = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        con.execute("PRAGMA busy_timeout=10000")
        return con

    def _init_schema(self, con):
        new = con.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] == 0
        if new:
            con.execute("PRAGMA auto_vacuum=INCREMENTAL")      # only takes effect before the first table
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA journal_size_limit=4194304")
        con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS series(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, key TEXT NOT NULL, "
                    "label TEXT, info TEXT, first_ts INTEGER, last_ts INTEGER, UNIQUE(kind, key))")
        for kind, cols in KINDS.items():
            body = ", ".join(f'"{c}" REAL' for c in cols)
            for suf in ("raw", "1m"):
                con.execute(f"CREATE TABLE IF NOT EXISTS {kind}_{suf}(sid INTEGER NOT NULL, ts INTEGER NOT NULL, n INTEGER, "
                            f"{body}, PRIMARY KEY(sid, ts)) WITHOUT ROWID")
            # no ts index: reads go by (sid, ts), the primary key; the 5-minute retention pass scans instead
            con.execute(f"DROP INDEX IF EXISTS {kind}_raw_ts")
            con.execute(f"DROP INDEX IF EXISTS {kind}_1m_ts")
        rq = ", ".join(f'"{c}" REAL' for c in REQ_COLS if c != "slot")
        con.execute(f"CREATE TABLE IF NOT EXISTS requests(sid INTEGER NOT NULL, ts REAL NOT NULL, slot INTEGER NOT NULL, {rq}, "
                    "PRIMARY KEY(sid, ts, slot)) WITHOUT ROWID")
        con.execute("CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts)")
        # throttle reasons are not a gauge column: a card can have several at once, and older
        # databases simply gain this table the next time the writer opens them.
        con.execute("CREATE TABLE IF NOT EXISTS throttle(ts INTEGER NOT NULL, card TEXT NOT NULL, reason TEXT NOT NULL)")
        con.execute("CREATE INDEX IF NOT EXISTS throttle_ts ON throttle(ts)")
        con.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('schema', ?)", (str(SCHEMA_VERSION),))
        con.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('created', ?)", (str(int(self.clock())),))
        con.commit()

    # ---- one writer per file ----------------------------------------------------------------------------
    def lock_writer(self):
        """take <db>.lock (flock, released when this process exits). False when another process holds it:
        that PXA Control records, this one only reads."""
        try:
            import fcntl
        except ImportError:          # not Linux: no second-process guard, still one writer thread here
            self.is_writer = True
            return True
        try:
            fd = os.open(self.path + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._lock_fd = fd
        self.is_writer = True
        return True

    def writer_pid(self):
        if self.is_writer:
            return os.getpid()
        try:
            with open(self.path + ".lock") as f:
                return int(f.read().strip() or 0) or None
        except (OSError, ValueError):
            return None

    def _unlock(self):
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        self.is_writer = False

    # ---- the writer thread -------------------------------------------------------------------------
    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop_ev.clear()
            self.thread = threading.Thread(target=self._writer, daemon=True, name="pxa-telemetry")
            self.thread.start()
        return self

    def close(self, timeout=5.0):
        if self.thread is not None and self.thread.is_alive():
            self.flush(force=True)
            self._put(("stop",))
            self.thread.join(timeout)
        self.stop_ev.set()
        self._unlock()

    def _put(self, item):
        try:
            self.q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _writer(self):
        con = self._connect()
        try:
            for k, v in con.execute("SELECT k, v FROM meta WHERE k LIKE 'rolled_%'"):
                self.rolled[k[len("rolled_"):]] = int(v)
            while True:
                try:
                    item = self.q.get(timeout=1.0)
                except queue.Empty:
                    item = None
                batch = [item] if item is not None else []
                while len(batch) < 500:          # one transaction for whatever is waiting
                    try:
                        batch.append(self.q.get_nowait())
                    except queue.Empty:
                        break
                stop = False
                try:
                    for it in batch:
                        if it[0] == "stop":
                            stop = True
                        elif it[0] == "rows":
                            self._write_rows(con, it[1])
                        elif it[0] == "reqs":
                            self._write_reqs(con, it[1], it[2], it[3], it[4])
                        elif it[0] == "thr":
                            self._write_throttle(con, it[1])
                        elif it[0] == "maint":
                            con.commit()
                            self._maintain(con, it[1])
                        elif it[0] == "sync":
                            con.commit()
                            it[1].set()
                    con.commit()
                    self.error = None
                except sqlite3.Error as e:          # a full disk, a locked file: a gap in the history, never a crash
                    self.error = f"{e.__class__.__name__}: {e}"
                    try:
                        con.rollback()
                    except sqlite3.Error:
                        pass
                    for it in batch:
                        if it[0] == "sync":
                            it[1].set()
                now = self.clock()
                if not stop and now - self.last_maint >= 300:
                    self.last_maint = now
                    try:
                        self._maintain(con, now)
                        con.commit()
                    except sqlite3.Error as e:
                        self.error = f"{e.__class__.__name__}: {e}"
                if stop:
                    return
        finally:
            try:
                con.commit()
                con.close()
            except sqlite3.Error:
                pass

    def sync(self, timeout=10.0):
        """wait until everything queued so far is on disk (tests, CSV right after a flush)."""
        if self.thread is None or not self.thread.is_alive():
            return False
        ev = threading.Event()
        if not self._put(("sync", ev)):
            return False
        return ev.wait(timeout)

    def _sid(self, con, kind, key, label, info, ts):
        k = (kind, key)
        sid = self.sid_cache.get(k)
        info_s = json.dumps(info, sort_keys=True, separators=(",", ":")) if info else None
        if sid is None:
            row = con.execute("SELECT id FROM series WHERE kind=? AND key=?", (kind, key)).fetchone()
            if row is None:
                cur = con.execute("INSERT INTO series(kind, key, label, info, first_ts, last_ts) VALUES(?,?,?,?,?,?)",
                                  (kind, key, label, info_s, ts, ts))
                sid = cur.lastrowid
            else:
                sid = row[0]
            self.sid_cache[k] = sid
        con.execute("UPDATE series SET last_ts=max(coalesce(last_ts, 0), ?), first_ts=min(coalesce(first_ts, ?), ?), "
                    "label=coalesce(?, label), info=coalesce(?, info) WHERE id=?", (ts, ts, ts, label, info_s, sid))
        return sid

    def _write_rows(self, con, rows):
        for kind, key, label, info, ts, vals, n in rows:
            sid = self._sid(con, kind, key, label, info, ts)
            cols = KINDS[kind]
            q = ", ".join(f'"{c}"' for c in cols)
            con.execute(f"INSERT OR REPLACE INTO {kind}_raw(sid, ts, n, {q}) VALUES(?,?,?{',?' * len(cols)})",
                        (sid, ts, n) + tuple(vals))
            self.writes += 1
            if ts < self.rolled.get(kind, -1):        # behind the rollup (the clock went back, an import): roll again from there
                self.late[kind] = min(self.late.get(kind, ts), ts)

    def _write_reqs(self, con, key, label, info, reqs):
        if not reqs:
            return
        sid = self._sid(con, "server", key, label, info, int(max(r[0] for r in reqs)))
        q = ", ".join(f'"{c}"' for c in REQ_COLS)
        for r in reqs:
            con.execute(f"INSERT OR IGNORE INTO requests(sid, ts, {q}) VALUES(?,?{',?' * len(REQ_COLS)})",
                        (sid, r[0], int(r[1]) if r[1] is not None else -1) + tuple(r[2:2 + len(REQ_COLS) - 1]))

    # ---- rollups, retention, size cap ---------------------------------------------------------------
    def _maintain(self, con, now):
        lag = 60 + 2 * float(self.cfg.get("sample_s") or 10)   # a flushed row is stamped with the mean time of its
        upto = int((now - lag) // 60) * 60                   # samples: stay behind it, whole minutes only
        for kind, cols in KINDS.items():
            mk = f"rolled_{kind}"
            row = con.execute("SELECT v FROM meta WHERE k=?", (mk,)).fetchone()
            start = int(row[0]) if row else None
            if start is None:
                r = con.execute(f"SELECT min(ts) FROM {kind}_raw").fetchone()
                if r[0] is None:
                    continue
                start = int(r[0] // 60) * 60
            if kind in self.late:
                start = min(start, int(self.late.pop(kind) // 60) * 60)
            if start >= upto:
                continue
            agg = ", ".join((f'sum("{c}")' if c in SUM_COLS else f'round(avg("{c}"), {PREC.get(c, 3)})') for c in cols)
            q = ", ".join(f'"{c}"' for c in cols)
            con.execute(f"INSERT OR REPLACE INTO {kind}_1m(sid, ts, n, {q}) "
                        f"SELECT sid, (ts / 60) * 60 AS m, sum(coalesce(n, 1)), {agg} FROM {kind}_raw "
                        f"WHERE ts >= ? AND ts < ? GROUP BY sid, m", (start, upto))
            con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)", (mk, str(upto)))
            self.rolled[kind] = upto
        raw_cut = int(now - self.cfg["raw_days"] * 86400)
        roll_cut = int(now - self.cfg["rollup_days"] * 86400)
        deleted = 0
        for kind in KINDS:
            deleted += con.execute(f"DELETE FROM {kind}_raw WHERE ts < ?", (raw_cut,)).rowcount
            deleted += con.execute(f"DELETE FROM {kind}_1m WHERE ts < ?", (roll_cut,)).rowcount
        deleted += con.execute("DELETE FROM requests WHERE ts < ?", (raw_cut,)).rowcount
        deleted += con.execute("DELETE FROM throttle WHERE ts < ?", (raw_cut,)).rowcount
        con.execute("DELETE FROM series WHERE last_ts < ?", (roll_cut,))
        # size cap: the oldest raw day goes first (the rollups still hold its minutes), then the oldest rollup day
        cap = self.cfg["max_mb"] * 1048576
        raw_t = [f"{k}_raw" for k in KINDS] + ["requests", "throttle"]
        roll_t = [f"{k}_1m" for k in KINDS]
        for _ in range(400):
            if self._db_bytes(con) <= cap:
                break
            for group in (raw_t, roll_t):
                mins = [r for r in (con.execute(f"SELECT min(ts) FROM {t}").fetchone()[0] for t in group) if r is not None]
                if mins:
                    cut = min(mins) + 86400
                    for t in group:
                        deleted += con.execute(f"DELETE FROM {t} WHERE ts < ?", (cut,)).rowcount
                    break
            else:
                break
            con.commit()
            con.execute("PRAGMA incremental_vacuum")
        if deleted:
            con.commit()
            con.execute("PRAGMA incremental_vacuum(2000)")
        return deleted

    @staticmethod
    def _db_bytes(con):
        pc = con.execute("PRAGMA page_count").fetchone()[0]
        fl = con.execute("PRAGMA freelist_count").fetchone()[0]
        ps = con.execute("PRAGMA page_size").fetchone()[0]
        return (pc - fl) * ps

    def maintain_now(self, now=None):
        """run the rollup/retention pass on the writer thread and wait for it (tests, the settings page)."""
        self._put(("maint", self.clock() if now is None else now))
        return self.sync()

    # ---- what the sampler hands in -------------------------------------------------------------------
    def record(self, kind, key, ts, vals, label=None, info=None):
        """one sample of one card / server / host. Samples are averaged (SUM_COLS added up) until the next
        flush, so a Live tab open at 2 s does not write five rows where the background writes one."""
        if not self.is_writer:
            return
        cols = KINDS[kind]
        v = [_fin(x) for x in (list(vals) + [None] * len(cols))[:len(cols)]]
        k = (kind, str(key))
        with self.acc_lock:
            a = self.acc.get(k)
            if a is None:
                a = self.acc[k] = {"n": 0, "t": 0.0, "s": [0.0] * len(cols), "c": [0] * len(cols), "label": None, "info": None}
            a["n"] += 1
            a["t"] += float(ts)
            for i, x in enumerate(v):
                if x is not None:
                    a["s"][i] += x
                    a["c"][i] += 1
            if label is not None:
                a["label"] = str(label)[:120]
            if info:
                a["info"] = info
            if kind == "server":
                c = self.counters.setdefault(str(key), {"req": 0.0, "gen_tok": 0.0, "prompt_tok": 0.0})
                for col in ("req", "gen_tok", "prompt_tok"):
                    x = v[cols.index(col)]
                    if x:
                        c[col] += x

    def note_throttle(self, card, ts, reasons):
        """One sample of the throttle reasons active on a card. Empty reasons are not stored."""
        if not self.is_writer:
            return
        rows = []
        for reason in reasons or []:
            if not reason:
                continue
            rows.append((int(ts), str(card)[:40], str(reason)[:80]))
        if rows:
            self._put(("thr", rows))

    def _write_throttle(self, con, rows):
        con.executemany("INSERT INTO throttle(ts, card, reason) VALUES(?,?,?)", rows)

    def heat_summary(self, since=None, until=None, hot_c=80.0):
        """Per card, over [since, until] (default the last 24 h): max temperature, minutes at or
        above hot_c, and how many samples saw each throttle reason.

        A raw row stands for one flushed sample (about sample_s seconds). The gap to the next row
        is the time that reading covers, capped at two sample intervals so a long hole is not
        counted as continuous heat.
        """
        now = self.clock()
        until = now if _fin(until) is None else float(until)
        since = (until - 86400.0) if _fin(since) is None else float(since)
        step = float(self.cfg["sample_s"])
        con = self._ro()
        try:
            try:
                samples = list(con.execute(
                    "SELECT s.key, coalesce(s.label, ''), c.ts, c.temp FROM card_raw c "
                    "JOIN series s ON s.id = c.sid WHERE s.kind = 'card' AND c.ts >= ? AND c.ts <= ? "
                    "AND c.temp IS NOT NULL ORDER BY s.key, c.ts",
                    (int(since), int(until))))
            except sqlite3.Error:
                samples = []
            try:
                thr = list(con.execute(
                    "SELECT card, reason, count(*) FROM throttle WHERE ts >= ? AND ts <= ? "
                    "GROUP BY card, reason",
                    (int(since), int(until))))
            except sqlite3.Error:
                thr = []
        finally:
            con.close()
        grouped = {}
        for key, label, ts, temp in samples:
            grouped.setdefault(key, {"label": label, "rows": []})["rows"].append((int(ts), float(temp)))
        reasons = {}
        for card, reason, n in thr:
            reasons.setdefault(card, {})[reason] = int(n)
        cards = []
        keys = list(dict.fromkeys(list(grouped) + list(reasons)))
        for key in keys:
            info = grouped.get(key) or {"label": "", "rows": []}
            rows = info["rows"]
            hot_s = 0.0
            mx = None
            for i, (ts, temp) in enumerate(rows):
                mx = temp if mx is None else max(mx, temp)
                if i + 1 < len(rows):
                    dt = min(step * 2.0, max(0.0, rows[i + 1][0] - ts))
                else:
                    dt = step
                if temp >= hot_c:
                    hot_s += dt
            counts = reasons.get(key) or {}
            cards.append({
                "card": key,
                "label": info["label"] or None,
                "max_c": None if mx is None else round(mx, 1),
                "minutes_ge_80": round(hot_s / 60.0, 2),
                "samples": len(rows),
                "throttle": {k: {"samples": v, "minutes": round(v * step / 60.0, 2)} for k, v in sorted(counts.items())},
            })
        cards.sort(key=lambda c: c["card"])
        return {"since": since, "until": until, "hot_c": hot_c, "sample_s": step, "cards": cards}

    def record_requests(self, key, reqs, label=None, info=None):
        """finished requests of one server: (ts, slot, n_prompt, n_cached, prompt_n, prompt_ms, prefill_tps, n_gen,
        gen_ms, decode_tps, draft_n, draft_acc) rows as the Live sampler keeps them."""
        rows = [tuple(r) for r in reqs if r and _fin(r[0]) is not None]
        if rows and self.is_writer:
            self._put(("reqs", str(key), None if label is None else str(label)[:120], info, rows))

    def flush(self, now=None, force=False):
        """write the averaged samples once sample_s has passed since the last write."""
        now = self.clock() if now is None else now
        if not self.is_writer or (not force and now - self.last_flush < self.cfg["sample_s"] * 0.95):
            return 0
        with self.acc_lock:
            acc, self.acc = self.acc, {}
        self.last_flush = now
        rows = []
        for (kind, key), a in acc.items():
            if not a["n"]:
                continue
            cols = KINDS[kind]
            vals = []
            for i, c in enumerate(cols):
                if not a["c"][i]:
                    vals.append(None)
                elif c in SUM_COLS:
                    vals.append(round(a["s"][i], PREC.get(c, 3)))
                else:
                    vals.append(round(a["s"][i] / a["c"][i], PREC.get(c, 3)))
            rows.append((kind, key, a["label"], a["info"], int(round(a["t"] / a["n"])), vals, a["n"]))
        if rows:
            self._put(("rows", rows))
        return len(rows)

    # ---- reads ---------------------------------------------------------------------------------------------
    def _ro(self):
        return self._connect(readonly=True)

    def list_series(self, kind=None):
        con = self._ro()
        try:
            q = "SELECT id, kind, key, label, info, first_ts, last_ts FROM series"
            args = ()
            if kind:
                q += " WHERE kind=?"
                args = (kind,)
            out = []
            for sid, k, key, label, info, f, l in con.execute(q + " ORDER BY kind, key", args):
                try:
                    inf = json.loads(info) if info else {}
                except ValueError:
                    inf = {}
                out.append({"kind": k, "key": key, "label": label, "info": inf, "first_ts": f, "last_ts": l})
            return out
        finally:
            con.close()

    def status(self):
        info = {"path": self.path, "settings": dict(self.cfg), "writer_alive": bool(self.thread and self.thread.is_alive()),
                "error": self.error, "rows_written": self.writes, "dropped": self.dropped}
        try:
            info["bytes"] = sum(os.path.getsize(p) for p in (self.path, self.path + "-wal") if os.path.exists(p))
            con = self._ro()
            try:
                r = con.execute("SELECT min(first_ts), max(last_ts), count(*) FROM series").fetchone()
                info["first_ts"], info["last_ts"], info["series"] = r[0], r[1], r[2]
                info["raw_rows"] = sum(con.execute(f"SELECT count(*) FROM {k}_raw").fetchone()[0] for k in KINDS)
                info["rollup_rows"] = sum(con.execute(f"SELECT count(*) FROM {k}_1m").fetchone()[0] for k in KINDS)
                info["requests"] = con.execute("SELECT count(*) FROM requests").fetchone()[0]
            finally:
                con.close()
        except (OSError, sqlite3.Error) as e:
            info["error"] = info["error"] or f"{e.__class__.__name__}: {e}"
        return info

    def plan(self, since, until, step=None, max_points=600, source="auto"):
        """-> (table suffix, step): raw when the whole range is still inside raw retention and the step is
        under a few minutes, else the one-minute rollups."""
        now = self.clock()
        span = max(1.0, until - since)
        raw_from = now - self.cfg["raw_days"] * 86400
        want = _fin(step) or 0.0
        auto = span / max(10, int(max_points))
        if source == "raw" or (source == "auto" and since >= raw_from and max(want, auto) < 300):
            return "raw", _nice_step(max(want, auto), max(1.0, self.cfg["sample_s"]))
        return "1m", _nice_step(max(want, auto), 60)

    def series(self, kind, keys=None, since=None, until=None, step=None, max_points=600, source="auto", cols=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        now = self.clock()
        until = now if _fin(until) is None else float(until)
        since = (until - 86400) if _fin(since) is None else float(since)
        if since >= until:
            since = until - 60
        suf, st = self.plan(since, until, step, max_points, source)
        allc = KINDS[kind]
        want = [c for c in (cols or allc) if c in allc]
        if kind == "server":
            for need in ("gen_tok", "gen_ms", "prompt_tok", "prompt_ms"):
                if need not in want:
                    want.append(need)
        con = self._ro()
        try:
            meta = {}
            q = "SELECT id, key, label, info, first_ts, last_ts FROM series WHERE kind=?"
            args = [kind]
            if keys:
                keys = [str(k) for k in keys][:64]
                q += " AND key IN (%s)" % ",".join("?" * len(keys))
                args += keys
            for sid, key, label, info, f, l in con.execute(q, args):
                try:
                    inf = json.loads(info) if info else {}
                except ValueError:
                    inf = {}
                meta[sid] = {"key": key, "label": label, "info": inf, "first_ts": f, "last_ts": l, "rows": []}
            if not meta:
                return self._shape(kind, want, suf, st, since, until, [])
            agg = ", ".join((f'sum("{c}")' if c in SUM_COLS else f'avg("{c}")') for c in want)
            sql = (f"SELECT sid, (ts / ?) AS b, avg(ts), {agg} FROM {kind}_{suf} WHERE ts >= ? AND ts <= ? "
                   f"AND sid IN ({','.join('?' * len(meta))}) GROUP BY sid, b ORDER BY sid, b")
            for row in con.execute(sql, [int(st), int(since), int(until)] + list(meta)):
                meta[row[0]]["rows"].append([round(row[2], 1)] + [None if v is None else float(v) for v in row[3:]])
        finally:
            con.close()
        out = []
        for m in meta.values():
            rows = m["rows"]
            if kind == "server":
                ix = {c: i + 1 for i, c in enumerate(want)}
                for r in rows:
                    pt, pm, gt, gm = r[ix["prompt_tok"]], r[ix["prompt_ms"]], r[ix["gen_tok"]], r[ix["gen_ms"]]
                    r.append(pt * 1000.0 / pm if pt and pm and pt >= 64 else None)
                    r.append(gt * 1000.0 / gm if gt and gm else None)
            m["rows"] = [[r[0]] + [_r(v, 3) for v in r[1:]] for r in rows]
            out.append(m)
        out.sort(key=lambda m: (m["key"]))
        return self._shape(kind, want, suf, st, since, until, out)

    @staticmethod
    def _shape(kind, want, suf, st, since, until, series):
        cols = ["t"] + list(want) + (list(DERIVED_SRV) if kind == "server" else [])
        return {"kind": kind, "cols": cols, "source": suf, "step": st, "since": since, "until": until, "series": series}

    def requests(self, keys=None, since=None, until=None, limit=2000):
        now = self.clock()
        until = now if _fin(until) is None else float(until)
        since = (until - 3600) if _fin(since) is None else float(since)
        limit = max(1, min(int(limit or 2000), 20000))
        con = self._ro()
        try:
            q = ("SELECT s.key, s.label, r.ts, r.slot, " + ", ".join(f'r."{c}"' for c in REQ_COLS if c != "slot") +
                 " FROM requests r JOIN series s ON s.id = r.sid WHERE r.ts >= ? AND r.ts <= ?")
            args = [since, until]
            if keys:
                keys = [str(k) for k in keys][:64]
                q += " AND s.key IN (%s)" % ",".join("?" * len(keys))
                args += keys
            q += " ORDER BY r.ts DESC LIMIT ?"
            args.append(limit)
            rows = [list(r) for r in con.execute(q, args)]
        finally:
            con.close()
        rows.reverse()
        return {"cols": ["key", "label", "t"] + list(REQ_COLS), "rows": rows}

    def csv_text(self, kind, keys=None, since=None, until=None, step=None, source="auto"):
        """the same numbers as series(), one line per series and time, for a spreadsheet."""
        if kind == "requests":
            d = self.requests(keys, since, until, limit=20000)
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(["time_utc"] + d["cols"])
            for r in d["rows"]:
                w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r[2]))] + r)
            return buf.getvalue()
        d = self.series(kind, keys, since, until, step, max_points=200000, source=source)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["time_utc", "unix_s", kind, "label"] + d["cols"][1:])
        for s in d["series"]:
            for r in s["rows"]:
                w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r[0])), r[0], s["key"], s["label"] or ""] +
                           ["" if v is None else v for v in r[1:]])
        return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------
# Prometheus text exposition: PXA Control's own gauges plus each engine server's /metrics, relabeled
# ---------------------------------------------------------------------------------------------------------
_PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)(\s+\S+)?\s*$')


def _esc(v):
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(d):
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in d.items() if v is not None) + "}"


CARD_GAUGES = (("mem", "pxa_card_memory_used_mib", "GPU memory in use, MiB"),
               ("util", "pxa_card_utilization_percent", "GPU load, percent"),
               ("temp", "pxa_card_temperature_celsius", "GPU temperature, degrees C"),
               ("power", "pxa_card_power_watts", "GPU power draw, W"),
               ("limit", "pxa_card_power_limit_watts", "GPU power limit, W"),
               ("clock", "pxa_card_sm_clock_mhz", "GPU SM clock, MHz"))
SRV_GAUGES = (("dec", "pxa_server_decode_tokens_per_second", "tokens written per second, all slots"),
              ("pre", "pxa_server_prefill_tokens_per_second", "prompt tokens read per second (engine counters)"),
              ("busy", "pxa_server_slots_busy", "slots working on a request"),
              ("slots", "pxa_server_slots", "slots the server has"),
              ("ctx", "pxa_server_kv_used_ratio", "share of the KV cache cells in use"),
              ("xhit", "pxa_server_expert_cache_hit_ratio", "expert lookups served from VRAM"),
              ("acc", "pxa_server_draft_acceptance_ratio", "drafted tokens the model accepted"))
SRV_COUNTERS = (("req", "pxa_server_requests_total", "requests finished since PXA Control started"),
                ("gen_tok", "pxa_server_generated_tokens_total", "tokens written since PXA Control started"),
                ("prompt_tok", "pxa_server_prompt_tokens_total", "prompt tokens read since PXA Control started"))
HOST_GAUGES = (("ram", "pxa_host_memory_used_mib", "this machine's RAM in use, MiB"),
               ("cpu", "pxa_host_cpu_percent", "this machine's CPU load, percent"),
               ("swap", "pxa_host_swap_used_mib", "swap in use, MiB"))


def prom_text(cards, servers, host, counters=None, engine=None):
    """cards: [{index, name, mem, util, temp, power, limit, clock}], servers: [{key, name, dec, pre, ...}],
    host: {ram, cpu, swap} or None, counters: {server key: {req, gen_tok, prompt_tok}}, engine: {server key:
    the text of that server's own /metrics}. Every engine sample gets a server="<key>" label; a HELP/TYPE
    line is written once per metric name."""
    out = []

    def block(name, help_, typ, samples):
        if not samples:
            return
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} {typ}")
        for lab, v in samples:
            out.append(f"{name}{_labels(lab)} {float(v):.10g}")

    for col, name, help_ in CARD_GAUGES:
        block(name, help_, "gauge", [({"card": str(c.get("index")), "name": c.get("name")}, c[col]) for c in cards
                                     if _fin(c.get(col)) is not None])
    for col, name, help_ in SRV_GAUGES:
        block(name, help_, "gauge", [({"server": s.get("key"), "name": s.get("name")}, s[col]) for s in servers
                                     if _fin(s.get(col)) is not None])
    for col, name, help_ in SRV_COUNTERS:
        block(name, help_, "counter", [({"server": k}, v.get(col) or 0.0) for k, v in sorted((counters or {}).items())])
    if host:
        for col, name, help_ in HOST_GAUGES:
            if _fin(host.get(col)) is not None:
                block(name, help_, "gauge", [({}, host[col])])
    seen_meta = set()
    for key in sorted(engine or {}):
        for line in (engine[key] or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.split(None, 3)
                if len(parts) >= 3 and parts[1] in ("HELP", "TYPE") and (parts[1], parts[2]) not in seen_meta:
                    seen_meta.add((parts[1], parts[2]))
                    out.append(line)
                continue
            m = _PROM_LINE.match(line)
            if not m:
                continue
            name, lab, val = m.group(1), m.group(2), m.group(3)
            inner = lab[1:-1].strip().rstrip(",") if lab else ""
            add = f'server="{_esc(key)}"'
            out.append(f"{name}{{{add}{',' + inner if inner else ''}}} {val}")
    return "\n".join(out) + "\n"
