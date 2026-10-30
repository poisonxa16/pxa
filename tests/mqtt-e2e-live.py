#!/usr/bin/env python3
"""End to end: PXA Control -> a broker -> the Home Assistant discovery payloads, on the wire.

The unit test beside this one (test-pxa-mqtt.py) checks the payloads a snapshot PRODUCES; this
harness checks the payloads Control actually PUBLISHES, which is the half that was wrong on
2026-10-11: the Live sampler's host row is a count of MiB, so a healthy box announced
"RAM used 8386.1640625 %" to Home Assistant, and its CPU as 2.7023645689978726 %.

Nothing here is mocked except the broker socket: the Control that runs is the real App, the Live
sampler reads this machine's /proc/meminfo, the MQTT client is the real one, and the payloads
below are decoded from the bytes on the wire.  The test passes only if the published numbers agree
with /proc -- a percentage that is a percentage, an amount in MiB, and a rounded CPU share.

    python3 tests/mqtt-e2e-live.py          (prints RESULT: PASS, exit 0)

Loopback only: it binds a broker on 127.0.0.1 and points Control at it.  No broker is required to
be installed, and no network is used.
"""
import importlib.util
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-mqtt-e2e-")
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_LAUNCH_STATE"] = os.path.join(TMP, "state")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "2x1024"
sys.path.insert(0, TOOLS)

# the licence/launcher module the App is built on, loaded the way the Control tests load it
_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)


class Broker(object):
    """A broker that speaks just enough MQTT 3.1.1 to take Control's CONNECT and record its
    PUBLISHes.  Payload bytes are kept as text, exactly as they arrived."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.msgs = {}                      # topic -> last payload
        self.lock = threading.Lock()
        self.dead = False
        self.connected = 0
        threading.Thread(target=self._serve, daemon=True, name="fake-broker").start()

    def _serve(self):
        while not self.dead:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._conn, args=(c,), daemon=True).start()

    def _conn(self, c):
        c.settimeout(30)
        buf = bytearray()

        def read(n):
            while len(buf) < n:
                b = c.recv(65536)
                if not b:
                    raise EOFError
                buf.extend(b)
            out = bytes(buf[:n])
            del buf[:n]
            return out

        try:
            while True:
                kind = read(1)[0] >> 4
                n, shift = 0, 0
                while True:
                    b = read(1)[0]
                    n |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                body = read(n) if n else b""
                if kind == 1:                                    # CONNECT
                    with self.lock:
                        self.connected += 1
                    c.sendall(b"\x20\x02\x00\x00")
                elif kind == 3:                                  # PUBLISH
                    tl = struct.unpack("!H", body[:2])[0]
                    topic = body[2:2 + tl].decode("utf-8", "replace")
                    with self.lock:
                        self.msgs[topic] = body[2 + tl:].decode("utf-8", "replace")
                elif kind == 12:                                 # PINGREQ
                    c.sendall(b"\xd0\x00")
                elif kind == 14:                                 # DISCONNECT
                    break
        except (OSError, EOFError):
            pass
        finally:
            try:
                c.close()
            except OSError:
                pass

    def close(self):
        self.dead = True
        try:
            self.sock.close()
        except OSError:
            pass

    def get(self, topic):
        with self.lock:
            return self.msgs.get(topic)


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[k] = int(v.split()[0]) / 1024.0        # KiB -> MiB, the same conversion the sampler does
    return out


def main():
    import pxa_control as C                                       # noqa: PLC0415 - after the env above

    br = Broker()
    os.environ.update({"PXA_CONTROL_MQTT": "1", "PXA_CONTROL_MQTT_HOST": "127.0.0.1",
                       "PXA_CONTROL_MQTT_PORT": str(br.port), "PXA_CONTROL_MQTT_INTERVAL": "2",
                       "PXA_CONTROL_MQTT_BASE": "pxa", "PXA_CONTROL_MQTT_DISCOVERY": "1"})
    app = C.App(L, port=0)
    pub = app.start_mqtt()
    fails, lines = [], []

    def check(ok, what):
        lines.append(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    try:
        if pub is None:
            print("FAIL Control did not start the MQTT publisher: %s" % app.mqtt_error)
            return 1
        # CPU needs two samples (it is a delta between ticks), so wait for it as well rather than
        # reading the first cycle and blaming the publisher for the sampler's warm-up.
        deadline = time.time() + 40
        while time.time() < deadline:
            if (br.get("pxa/host/ram_used") is not None and br.get("pxa/host/ram_total_mib") is not None
                    and br.get("pxa/host/cpu_used") is not None):
                break
            time.sleep(0.25)
        m = meminfo()
        used_mib = m["MemTotal"] - m["MemAvailable"]
        want_pct = round(used_mib / m["MemTotal"] * 100.0, 1)

        def num(topic):
            v = br.get(topic)
            return None if v is None else float(v)

        def cfg(topic):
            v = br.get(topic)
            return None if v is None else json.loads(v)

        ram_pct, ram_mib, ram_total = (num("pxa/host/ram_used"),
                                       num("pxa/host/ram_used_mib"),
                                       num("pxa/host/ram_total_mib"))
        cpu = num("pxa/host/cpu_used")
        lines.append("published: ram_used=%s ram_used_mib=%s ram_total_mib=%s cpu_used=%s"
                     % (ram_pct, ram_mib, ram_total, cpu))
        lines.append("this machine now: %.1f MiB used of %.0f MiB = %.1f %%"
                     % (used_mib, m["MemTotal"], want_pct))

        check(ram_pct is not None, "RAM used was published")
        if ram_pct is not None:
            # memory moves between the sample and this read, so the share is compared with a
            # tolerance -- but a MiB count published as a percent is hundreds of points away, not
            # a few, so this still fails loudly on the 2026-10-11 bug.
            check(0.0 <= ram_pct <= 100.0, "RAM used is a percentage in range (got %s)" % ram_pct)
            check(abs(ram_pct - want_pct) <= 3.0,
                  "RAM used agrees with /proc/meminfo (published %s, this machine %.1f)" % (ram_pct, want_pct))
        check(ram_total == round(m["MemTotal"], 1),
              "RAM total is the machine's MemTotal in MiB (%s vs %.0f)" % (ram_total, m["MemTotal"]))
        if ram_mib is not None:
            # the live comparison gets the same 3-point tolerance as the share above (memory moves between the sample and
            # this read: 512 MiB failed on a busy 188 GiB box, 71088 vs 72689); the units are pinned by the sample itself:
            # the amount and the share come from ONE reading, so amount / total must give the share to its one decimal
            tol = max(512.0, 0.03*m["MemTotal"])
            check(abs(ram_mib - used_mib) <= tol, "RAM used (amount) is MiB used (%s vs %.0f, tolerance %.0f)" % (ram_mib, used_mib, tol))
            if ram_pct is not None and ram_total:
                check(abs(100.0*ram_mib/ram_total - ram_pct) <= 0.051,
                      "RAM used (amount) / RAM total is the published share (%.3f vs %s)" % (100.0*ram_mib/ram_total, ram_pct))
        check(cpu is not None and 0.0 <= cpu <= 100.0, "CPU used is a percentage in range (got %s)" % cpu)
        if cpu is not None:
            check(cpu == round(cpu, 1), "CPU used is rounded to one decimal (got %r)" % cpu)

        if m.get("SwapTotal", 0) > 0:
            swap_pct = num("pxa/host/swap_used")
            swap_mib = num("pxa/host/swap_used_mib")
            check(swap_pct is not None and 0.0 <= swap_pct <= 100.0,
                  "Swap used is a percentage in range (got %s)" % swap_pct)
            check(swap_mib is not None and 0.0 <= swap_mib <= m["SwapTotal"] + 1.0,
                  "Swap used (amount) is within the machine's swap (got %s of %.0f)"
                  % (swap_mib, m["SwapTotal"]))
        else:
            check(br.get("pxa/host/swap_used") is None,
                  "no swap on this machine, so no swap entity is published at all")

        c = cfg("homeassistant/sensor/host/ram_used/config")
        check(c is not None, "the RAM config was published to the discovery topic")
        if c:
            check(c.get("unit_of_measurement") == "%", "RAM used is published with the %% unit")
            check("device_class" not in c, "RAM used has no device_class (HA has none for a percent)")
            check(c.get("state_class") == "measurement", "RAM used is a measurement")
            check(c.get("suggested_display_precision") == 1, "RAM used asks HA for one decimal")
            check((c.get("origin") or {}).get("sw_version") == C.CONTROL_VERSION,
                  "the config names the software and version that published it (%s)"
                  % (c.get("origin") or {}).get("sw_version"))
            check(c.get("state_topic") == "pxa/host/ram_used", "the config points at the value it publishes")

        c = cfg("homeassistant/sensor/host/ram_used_mib/config")
        if c:
            check(c.get("unit_of_measurement") == "MiB", "the amount sensor is published in MiB")
            check(c.get("device_class") == "data_size", "the amount sensor is a data size")

        # every percent entity Control published, whichever device it belongs to
        bad = []
        for topic, value in sorted(br.msgs.items()):
            if topic.startswith("homeassistant/"):
                continue
            if topic.rsplit("/", 1)[-1] in ("ram_used", "swap_used", "cpu_used", "utilization",
                                            "kv_cache_used", "expert_cache_hit", "draft_acceptance"):
                try:
                    if not (0.0 <= float(value) <= 100.0):
                        bad.append("%s=%s" % (topic, value))
                except ValueError:
                    bad.append("%s=%s" % (topic, value))
        check(not bad, "every published percentage is a percentage (offenders: %s)" % (bad or "none"))
        check(br.get("pxa/status") == "online", "the publisher announced itself online")
        check(any(t.startswith("pxa/gpu_") for t in br.msgs), "the per-GPU entities were published too")
    finally:
        try:
            app.stop_mqtt()
        finally:
            br.close()

    print("\n".join(lines))
    print("RESULT: %s%s" % ("PASS" if not fails else "FAIL: " + "; ".join(fails),
                            "" if not fails else " (%d check(s))" % len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
