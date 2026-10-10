#!/usr/bin/env python3
"""PXA Control's telemetry history (tools/pxa_telemetry.py): the SQLite time-series store behind the Live tab's
long ranges. Settings and their environment overrides, sample averaging, the one-minute rollups, retention and the
size cap, bucketed reads, CSV, the one-writer lock and the Prometheus text. No GPU, no network, a temp dir only.

    python3 tests/test-pxa-telemetry.py          (wired into CTest as test-pxa-telemetry)
"""
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import pxa_telemetry as T  # noqa: E402


class Clock(object):
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


def card(mem=1000.0, util=50.0, temp=60.0, power=150.0, limit=250.0, clock=1300.0):
    return [mem, util, temp, power, limit, clock]


def srv(dec=30.0, pre=None, busy=1, slots=1, ctx=0.1, xhit=None, xswap=None, acc=None, ema=None,
        req=0, gen_tok=0, prompt_tok=0, gen_ms=0, prompt_ms=0):
    return [dec, pre, busy, slots, ctx, xhit, xswap, acc, ema, req, gen_tok, prompt_tok, gen_ms, prompt_ms]


class Base(unittest.TestCase):
    T0 = 1_800_000_000.0         # a fixed "now" (2027) so day arithmetic is exact

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="pxa-tel-")
        self.path = os.path.join(self.dir, "telemetry.db")
        self.clock = Clock(self.T0)
        self.st = None

    def tearDown(self):
        if self.st is not None:
            self.st.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def store(self, **cfg):
        c = dict(sample_s=10.0)
        c.update(cfg)
        self.st = T.Store(self.path, c, clock=self.clock)
        self.st.last_maint = float("inf")          # no timed pass: each test runs maintain_now() where it means to
        self.assertTrue(self.st.lock_writer())
        self.st.start()
        return self.st

    def put(self, kind, key, ts, vals, label=None):
        """one sample written straight away (a flush per sample)."""
        self.st.record(kind, key, ts, vals, label=label)
        self.st.flush(now=ts, force=True)


class Settings(Base):
    def test_defaults(self):
        s = T.settings({}, env={})
        self.assertEqual(s, T.DEFAULTS)

    def test_config_then_env_wins_and_numbers_are_clamped(self):
        cfg = {"telemetry": {"raw_days": 3, "rollup_days": 30, "sample_s": 0.1, "prometheus": True}}
        s = T.settings(cfg, env={})
        self.assertEqual((s["raw_days"], s["rollup_days"], s["sample_s"], s["prometheus"]), (3.0, 30.0, 2.0, True))
        s = T.settings(cfg, env={"PXA_CONTROL_TELEMETRY_RAW_DAYS": "1.5", "PXA_CONTROL_METRICS": "off",
                                 "PXA_CONTROL_TELEMETRY": "0", "PXA_CONTROL_TELEMETRY_MAX_MB": "lots"})
        self.assertEqual((s["raw_days"], s["prometheus"], s["enabled"], s["max_mb"]), (1.5, False, False, 512.0))

    def test_raw_never_outlives_the_rollups(self):
        s = T.settings({"telemetry": {"raw_days": 60, "rollup_days": 7}}, env={})
        self.assertEqual(s["raw_days"], 7.0)

    def test_non_dict_telemetry_section_is_ignored(self):
        self.assertEqual(T.settings({"telemetry": "yes"}, env={}), T.DEFAULTS)


class Writes(Base):
    def test_file_is_private_wal_and_versioned(self):
        self.store()
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        con = sqlite3.connect(self.path)
        self.assertEqual(con.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.assertEqual(con.execute("SELECT v FROM meta WHERE k='schema'").fetchone()[0], str(T.SCHEMA_VERSION))
        con.close()

    def test_samples_are_averaged_until_the_flush_and_sums_added_up(self):
        st = self.store()
        t = self.T0 - 100
        st.record("card", "0", t, card(util=40), label="Tesla P100")
        st.record("card", "0", t + 2, card(util=60))
        st.record("server", "m:main", t, srv(dec=20, req=1, gen_tok=100, gen_ms=4000), label="Server 1")
        st.record("server", "m:main", t + 2, srv(dec=40, req=2, gen_tok=50, gen_ms=1000))
        self.assertEqual(st.flush(now=t + 2), 2)
        self.assertEqual(st.flush(now=t + 4), 0)                  # inside sample_s: nothing new to write
        st.sync()
        c = st.series("card", since=t - 10, until=t + 10, step=60)["series"][0]
        self.assertEqual((c["key"], c["label"]), ("0", "Tesla P100"))
        cols = st.series("card", since=t - 10, until=t + 10)["cols"]
        self.assertAlmostEqual(c["rows"][0][cols.index("util")], 50.0)
        s = st.series("server", since=t - 10, until=t + 10)
        r = s["series"][0]["rows"][0]
        self.assertAlmostEqual(r[s["cols"].index("dec")], 30.0)
        self.assertEqual(r[s["cols"].index("req")], 3.0)
        self.assertEqual(r[s["cols"].index("gen_tok")], 150.0)
        self.assertAlmostEqual(r[s["cols"].index("dec_req")], 30.0)        # 150 tokens in 5 s of decode
        self.assertIsNone(r[s["cols"].index("pre_req")])                    # no prompt read in that bucket

    def test_non_numbers_are_gaps(self):
        st = self.store()
        self.put("card", "1", self.T0 - 50, [float("nan"), "[N/A]", None, float("inf"), 250, 1300])
        st.sync()
        d = st.series("card", since=self.T0 - 60, until=self.T0)
        row = d["series"][0]["rows"][0]
        self.assertEqual(row[1:5], [None, None, None, None])
        self.assertEqual(row[d["cols"].index("limit")], 250.0)

    def test_counters_for_prometheus_accumulate(self):
        st = self.store()
        st.record("server", "m:a", self.T0, srv(req=2, gen_tok=10, prompt_tok=500))
        st.record("server", "m:a", self.T0 + 1, srv(req=1, gen_tok=5))
        self.assertEqual(st.counters["m:a"], {"req": 3.0, "gen_tok": 15.0, "prompt_tok": 500.0})

    def test_reader_works_while_the_writer_writes(self):
        st = self.store()
        errs = []

        def reader():
            for _ in range(30):
                try:
                    st.series("card", since=self.T0 - 4000, until=self.T0)
                except Exception as e:        # noqa: BLE001
                    errs.append(e)
        th = threading.Thread(target=reader)
        th.start()
        for i in range(300):
            self.put("card", "0", self.T0 - 3000 + i * 10, card(util=i % 100))
        th.join()
        st.sync()
        self.assertEqual(errs, [])
        self.assertEqual(len(st.series("card", since=self.T0 - 4000, until=self.T0, step=10)["series"][0]["rows"]), 300)


class Rollups(Base):
    def test_minute_rollups_average_gauges_and_add_sums(self):
        st = self.store()
        base = int(self.T0 // 60) * 60 - 600
        for i in range(18):                                        # three minutes at 10 s
            self.put("server", "m:main", base + i * 10, srv(dec=10 + i, req=1, gen_tok=10, gen_ms=500))
        st.maintain_now(now=base + 1000)
        con = sqlite3.connect(self.path)
        rows = con.execute("SELECT ts, n, dec, req, gen_tok FROM server_1m ORDER BY ts").fetchall()
        con.close()
        self.assertEqual([r[0] for r in rows], [base, base + 60, base + 120])
        self.assertEqual(rows[0][1], 6)
        self.assertAlmostEqual(rows[0][2], sum(10 + i for i in range(6)) / 6)
        self.assertEqual((rows[0][3], rows[0][4]), (6.0, 60.0))
        # a second pass does not roll the same minutes twice
        st.maintain_now(now=base + 1000)
        con = sqlite3.connect(self.path)
        self.assertEqual(con.execute("SELECT count(*) FROM server_1m").fetchone()[0], 3)
        con.close()

    def test_rows_written_behind_the_rollup_are_rolled_again(self):
        st = self.store()
        base = int(self.T0 // 60) * 60 - 600
        self.put("card", "0", base + 120, card(temp=60))
        st.maintain_now(now=base + 1000)                           # watermark past base + 120
        self.put("card", "0", base, card(temp=50))                 # the clock went back / an import
        self.put("card", "0", base + 125, card(temp=80))           # a minute that was already rolled
        st.maintain_now(now=base + 1000)
        con = sqlite3.connect(self.path)
        rows = con.execute("SELECT ts, temp FROM card_1m ORDER BY ts").fetchall()
        first = con.execute("SELECT first_ts FROM series").fetchone()[0]
        con.close()
        self.assertEqual(rows, [(base, 50.0), (base + 120, 70.0)])
        self.assertEqual(first, base)

    def test_old_ranges_read_from_the_rollups(self):
        st = self.store(raw_days=1, rollup_days=30)
        old = self.T0 - 5 * 86400
        for i in range(12):
            self.put("card", "0", old + i * 10, card(temp=70 + (i % 2)))
        st.maintain_now(now=old + 3600)                            # rolled while still inside raw retention
        st.maintain_now()                                           # now: the raw rows are past 1 day and go
        d = st.series("card", since=old - 60, until=old + 600)
        self.assertEqual(d["source"], "1m")
        self.assertGreaterEqual(d["step"], 60)
        temps = [r[d["cols"].index("temp")] for r in d["series"][0]["rows"]]
        self.assertTrue(all(abs(t - 70.5) < 0.6 for t in temps), temps)
        con = sqlite3.connect(self.path)
        self.assertEqual(con.execute("SELECT count(*) FROM card_raw").fetchone()[0], 0)
        con.close()

    def test_recent_short_ranges_read_raw(self):
        st = self.store()
        self.put("card", "0", self.T0 - 30, card())
        st.sync()
        self.assertEqual(st.series("card", since=self.T0 - 600, until=self.T0)["source"], "raw")
        self.assertEqual(st.plan(self.T0 - 30 * 86400, self.T0)[0], "1m")


class Retention(Base):
    def test_raw_rollup_and_series_retention(self):
        st = self.store(raw_days=2, rollup_days=10)
        for d in (12, 5, 1):
            t = self.T0 - d * 86400
            self.clock.t = t + 200                                 # the data arrives in real time
            self.put("card", "dead" if d == 12 else "0", t, card())
            st.record_requests("m:main", [(t, 0, 10, 0, 10, 5.0, 2000.0, 20, 400.0, 50.0, 0, 0)])
            st.maintain_now(now=t + 200)
        self.clock.t = self.T0
        st.maintain_now()
        con = sqlite3.connect(self.path)
        raw = [r[0] for r in con.execute("SELECT ts FROM card_raw")]
        roll = [r[0] for r in con.execute("SELECT ts FROM card_1m")]
        keys = [r[0] for r in con.execute("SELECT key FROM series WHERE kind='card'")]
        reqs = con.execute("SELECT count(*) FROM requests").fetchone()[0]
        con.close()
        self.assertEqual(len(raw), 1)                              # only the 1-day-old sample is still raw
        self.assertEqual(len(roll), 2)                             # 5 and 1 days old; 12 days is past the rollups
        self.assertNotIn("dead", keys)                             # a card unseen for longer than the rollups is forgotten
        self.assertEqual(reqs, 1)

    def test_size_cap_drops_the_oldest_raw_day_first(self):
        st = self.store(raw_days=30, rollup_days=60, max_mb=1e6)
        for day in range(6, 0, -1):
            t0 = self.T0 - day * 86400
            for i in range(0, 86400, 120):
                st.record("card", "0", t0 + i, card())
                st.record("card", "1", t0 + i, card())
                st.flush(now=t0 + i, force=True)
        st.maintain_now()                                          # rollups made, nothing over the cap yet
        con = sqlite3.connect(self.path)
        before = T.Store._db_bytes(con)
        rolled0 = con.execute("SELECT count(*) FROM card_1m").fetchone()[0]
        con.close()
        cap = before * 0.85
        st.cfg["max_mb"] = cap / 1048576
        st.maintain_now()
        con = sqlite3.connect(self.path)
        used = T.Store._db_bytes(con)
        oldest = con.execute("SELECT min(ts) FROM card_raw").fetchone()[0]
        newest = con.execute("SELECT max(ts) FROM card_raw").fetchone()[0]
        rolled = con.execute("SELECT count(*) FROM card_1m").fetchone()[0]
        con.close()
        self.assertLessEqual(used, cap)
        self.assertGreater(oldest, self.T0 - 6 * 86400 + 86000)    # the first raw day went ...
        self.assertGreaterEqual(newest, self.T0 - 86400)           # ... the newest stayed
        self.assertEqual(rolled, rolled0)                          # and the minutes of the dropped days are still there


class Reads(Base):
    def test_bucketing_respects_max_points_and_keys(self):
        st = self.store()
        for i in range(360):
            st.record("card", "0", self.T0 - 3600 + i * 10, card(util=i % 100))
            st.record("card", "1", self.T0 - 3600 + i * 10, card(util=10))
            st.flush(now=self.T0 - 3600 + i * 10, force=True)
        st.sync()
        d = st.series("card", keys=["1"], since=self.T0 - 3600, until=self.T0, max_points=60)
        self.assertEqual([s["key"] for s in d["series"]], ["1"])
        self.assertLessEqual(len(d["series"][0]["rows"]), 61)
        self.assertEqual(d["step"], 60)
        self.assertEqual(st.series("card", keys=["nope"], since=self.T0 - 60, until=self.T0)["series"], [])
        with self.assertRaises(ValueError):
            st.series("disk")

    def test_requests_are_kept_once_and_exported(self):
        st = self.store()
        r = (self.T0 - 20, 0, 900, 100, 800, 1000.0, 800.0, 100, 3000.0, 33.3, 40, 30)
        st.record_requests("m:main", [r], label="Server 1")
        st.record_requests("m:main", [r, (self.T0 - 10, 1) + r[2:]])
        st.sync()
        d = st.requests(since=self.T0 - 60, until=self.T0)
        self.assertEqual(len(d["rows"]), 2)
        row = dict(zip(d["cols"], d["rows"][0]))
        self.assertEqual((row["key"], row["label"], row["decode_tps"], row["slot"]), ("m:main", "Server 1", 33.3, 0))
        csv = st.csv_text("requests", since=self.T0 - 60, until=self.T0).splitlines()
        self.assertEqual(len(csv), 3)
        self.assertTrue(csv[0].startswith("time_utc,key,label,t,slot"))

    def test_csv_has_one_line_per_series_and_time(self):
        st = self.store()
        for i in range(5):
            st.record("card", "0", self.T0 - 50 + i * 10, card())
            st.record("card", "1", self.T0 - 50 + i * 10, card())
            st.flush(now=self.T0 - 50 + i * 10, force=True)
        st.sync()
        lines = st.csv_text("card", since=self.T0 - 100, until=self.T0, step=10).splitlines()
        self.assertEqual(lines[0].split(",")[:5], ["time_utc", "unix_s", "card", "label", "mem"])
        self.assertEqual(len(lines), 1 + 10)

    def test_status_and_list(self):
        st = self.store()
        self.put("host", "host", self.T0 - 5, [32000.0, 12.0, None], label="this machine")
        st.sync()
        s = st.status()
        self.assertTrue(s["writer_alive"])
        self.assertEqual((s["series"], s["raw_rows"]), (1, 1))
        self.assertEqual(st.list_series("host")[0]["label"], "this machine")


class OneWriter(Base):
    def test_a_second_store_on_the_same_file_only_reads(self):
        st = self.store()
        self.put("card", "0", self.T0 - 5, card())
        st.sync()
        other = T.Store(self.path, {"sample_s": 10.0}, clock=self.clock)
        try:
            self.assertFalse(other.lock_writer())
            self.assertEqual(other.writer_pid(), os.getpid())
            other.record("card", "0", self.T0, card())
            self.assertEqual(other.flush(force=True), 0)
            self.assertEqual(len(other.series("card", since=self.T0 - 60, until=self.T0)["series"]), 1)
        finally:
            other.close()
        st.close()
        self.st = None
        again = T.Store(self.path, {}, clock=self.clock)          # released at close: the next one may write
        try:
            self.assertTrue(again.lock_writer())
        finally:
            again.close()


class Prometheus(unittest.TestCase):
    def test_gauges_counters_and_relabeled_engine_text(self):
        txt = T.prom_text(
            [{"index": 0, "name": 'Tesla "P100"', "mem": 1234.0, "util": 97.0, "temp": None}],
            [{"key": "m:main", "name": "Server 1", "dec": 31.5, "busy": 1, "ctx": 0.25}],
            {"ram": 20000.0, "cpu": 5.0, "swap": None},
            counters={"m:main": {"req": 7.0, "gen_tok": 1234567.0, "prompt_tok": 0.0}},
            engine={"m:main": "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
                              "# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total 5000\n"
                              'llamacpp:x{a="b"} 1\nnot a metric line\n',
                    "m:two": "# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total 7\n"})
        self.assertIn('pxa_card_utilization_percent{card="0",name="Tesla \\"P100\\""} 97', txt)
        self.assertNotIn("pxa_card_temperature_celsius{", txt)            # no reading, no sample
        self.assertIn('pxa_server_decode_tokens_per_second{server="m:main",name="Server 1"} 31.5', txt)
        self.assertIn('pxa_server_generated_tokens_total{server="m:main"} 1234567', txt)
        self.assertIn("pxa_host_memory_used_mib{} 20000", txt)
        self.assertIn('llamacpp:prompt_tokens_total{server="m:main"} 5000', txt)
        self.assertIn('llamacpp:prompt_tokens_total{server="m:two"} 7', txt)
        self.assertIn('llamacpp:x{server="m:main",a="b"} 1', txt)
        self.assertEqual(txt.count("# TYPE llamacpp:prompt_tokens_total"), 1)
        self.assertNotIn("not a metric", txt)
        self.assertTrue(txt.endswith("\n"))


class HeatHistory(Base):
    def test_24h_heat_and_throttle_counts(self):
        st = self.store(sample_s=10.0)
        t = self.T0 - 100
        self.clock.t = t
        self.put("card", "0", t, card(temp=90), label="V100")
        self.put("card", "0", t + 10, card(temp=90))
        self.put("card", "0", t + 20, card(temp=70))
        self.put("card", "1", t, card(temp=60), label="P100")
        st.note_throttle("0", t, ["hw_thermal", "hw_power_brake"])
        st.note_throttle("0", t + 10, ["hw_thermal"])
        st.sync()
        self.clock.t = self.T0
        s = st.heat_summary(since=t - 1, until=self.T0)
        by = {c["card"]: c for c in s["cards"]}
        self.assertEqual(by["0"]["label"], "V100")
        self.assertEqual(by["0"]["max_c"], 90.0)
        # two hot samples, each covering the 10 s until the next reading
        self.assertEqual(by["0"]["minutes_ge_80"], 0.33)  # 20 s of the 10 s sample interval, rounded
        self.assertEqual(by["0"]["throttle"]["hw_thermal"]["samples"], 2)
        self.assertEqual(by["0"]["throttle"]["hw_power_brake"]["samples"], 1)
        self.assertEqual(by["1"]["max_c"], 60.0)
        self.assertEqual(by["1"]["minutes_ge_80"], 0.0)
        self.assertEqual(by["1"]["throttle"], {})

if __name__ == "__main__":
    unittest.main(verbosity=2)
