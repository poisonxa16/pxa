"""pxa_mqtt -- an OPTIONAL MQTT publisher for PXA Control, with Home Assistant MQTT Discovery.

Off unless you turn it on (Control -> Settings -> Home Assistant, or PXA_CONTROL_MQTT=1).  When it
is on, once every `interval` seconds Control publishes the same numbers its Live sampler already
collects -- generation and prompt tokens/s, busy and total slots, KV-cache fill, expert-cache hit
rate, draft acceptance, per-card temperature / VRAM / utilisation / power, host RAM+CPU, and the
expert map's learned-session count -- as Home Assistant entities, each with its own discovery
config so nothing has to be written by hand in HA.

Everything stays on your LAN: this module talks to the broker you name and to nothing else.
Nothing is ever sent to us.

No third-party dependency.  The package carries a standard-library-only python, and `paho-mqtt`
would mean either a pip install at run time or a vendored tree, so the publisher implements the
handful of MQTT 3.1.1 packets an uplink needs (CONNECT with a last will, PUBLISH QoS 0, PINGREQ,
DISCONNECT) on a plain socket.  QoS 0 + retained discovery configs is exactly the shape Home
Assistant's discovery expects; a dropped state message is replaced by the next interval.

The broker password is a secret: it lives in control.json (mode 0600, like the licence key), it is
returned to the page only as `password_set: true`, and it never reaches a log line -- see
`public()` and pxa_control.redact_text.
"""

import json
import os
import socket
import ssl
import struct
import threading
import time

SCHEMA_VERSION = 1

DEFAULTS = {"enabled": False, "host": "", "port": 1883, "user": "", "password": "", "tls": False,
            "base": "pxa", "prefix": "homeassistant", "interval": 15.0, "discovery": True}
_ENV = {"enabled": "PXA_CONTROL_MQTT", "host": "PXA_CONTROL_MQTT_HOST", "port": "PXA_CONTROL_MQTT_PORT",
        "user": "PXA_CONTROL_MQTT_USER", "password": "PXA_CONTROL_MQTT_PASSWORD", "tls": "PXA_CONTROL_MQTT_TLS",
        "base": "PXA_CONTROL_MQTT_BASE", "prefix": "PXA_CONTROL_MQTT_PREFIX",
        "interval": "PXA_CONTROL_MQTT_INTERVAL", "discovery": "PXA_CONTROL_MQTT_DISCOVERY"}
_LIMITS = {"port": (1.0, 65535.0), "interval": (2.0, 3600.0)}
_TEXTS = ("host", "user", "base", "prefix")
_OFF = ("0", "off", "no", "false", "never")
_ON = ("1", "on", "yes", "true", "always")
KEEPALIVE = 60                       # seconds; the loop pings at half that on an idle link


def settings(cfg=None, env=None):
    """The effective settings: DEFAULTS, then the control.json "mqtt" object, then the environment
    (an env var wins).  Numbers are clamped; a topic that MQTT would reject is refused rather than
    published to."""
    env = os.environ if env is None else env
    out = dict(DEFAULTS)
    src = (cfg or {}).get("mqtt") if isinstance((cfg or {}).get("mqtt"), dict) else {}
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
        elif k in _LIMITS:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f != f:
                continue
            lo, hi = _LIMITS[k]
            out[k] = min(hi, max(lo, f))
        else:
            out[k] = str(v).strip()
    out["port"] = int(out["port"])
    out["interval"] = float(out["interval"])
    for k in ("base", "prefix"):
        out[k] = valid_topic_root(out[k]) or DEFAULTS[k]
    return out


def valid_topic_root(s):
    """A topic root, or "" -- MQTT reserves '+' '#' and a leading '$', and a space or a slash at
    either end is a typo that would silently publish to a topic the user cannot find."""
    s = (s or "").strip().strip("/")
    if not s or any(c in s for c in " +#\u0000") or s.startswith("$"):
        return ""
    return s


def public(cfg):
    """The settings as the page may see them: the password is replaced by a flag, never a value."""
    d = dict(cfg)
    d["password"] = ""
    d["password_set"] = bool(cfg.get("password"))
    return d


# --------------------------------------------------------------------------- the wire
def _utf8(s):
    b = str(s).encode("utf-8")
    return struct.pack("!H", len(b)) + b


def _remaining(n):
    out = bytearray()
    while True:
        b = n % 128
        n //= 128
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _packet(kind, flags, body):
    return bytes([(kind << 4) | flags]) + _remaining(len(body)) + body


class MQTTError(Exception):
    """A refusal from the broker, or a link that went away."""


class Client(object):
    """A minimal MQTT 3.1.1 client over one socket: CONNECT (with last will), PUBLISH QoS 0,
    PINGREQ, DISCONNECT.  Nothing here logs or holds the password beyond the CONNECT packet."""

    def __init__(self, host, port=1883, user="", password="", tls=False, client_id=None,
                 will=None, timeout=10.0):
        self.host, self.port, self.user, self.password = host, int(port), user or "", password or ""
        self.tls, self.timeout = bool(tls), float(timeout)
        self.client_id = client_id or ("pxa-control-" + str(os.getpid()))
        self.will = will                      # (topic, payload) published by the broker if we vanish
        self.sock = None
        self.last_ping = 0.0

    def connect(self):
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.tls:
            ctx = ssl.create_default_context()
            s = ctx.wrap_socket(s, server_hostname=self.host)
        s.settimeout(self.timeout)
        self.sock = s
        body = _utf8("MQTT") + bytes([4])
        flags = 0x02                                            # clean session
        if self.will:
            flags |= 0x04 | 0x20                                 # will flag, QoS 0, RETAINED (bit 5): a client that
            #                                                      is gone when it fails must not leave "online" behind
        if self.user:
            flags |= 0x80
        if self.password:
            flags |= 0x40
        body += bytes([flags]) + struct.pack("!H", KEEPALIVE) + _utf8(self.client_id)
        if self.will:
            body += _utf8(self.will[0]) + _utf8(self.will[1])
        if self.user:
            body += _utf8(self.user)
        if self.password:
            body += _utf8(self.password)
        s.sendall(_packet(1, 0, body))
        kind, payload = self._read_packet()
        if kind != 2:
            raise MQTTError("expected CONNACK, got packet type %d" % kind)
        if len(payload) < 2:
            raise MQTTError("short CONNACK")
        code = payload[1]
        if code:
            raise MQTTError("the broker refused the connection: %s" % _CONNACK.get(code, "code %d" % code))
        self.last_ping = time.time()
        return True

    def _read_packet(self):
        # ONE type byte, then the varint length.  Reading two bytes here swallowed the length byte, so a
        # CONNACK's own length became its first payload byte and every reply looked "short" -- found
        # 2026-10-09 by pointing this at a real broker.
        head = self._recv(1)
        kind = head[0] >> 4
        n, shift = 0, 0
        while True:
            b = self._recv(1)[0]
            n |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return kind, (self._recv(n) if n else b"")

    def _recv(self, n):
        out = b""
        while len(out) < n:
            try:
                chunk = self.sock.recv(n - len(out))
            except socket.timeout:
                raise MQTTError("the broker stopped answering")
            if not chunk:
                raise MQTTError("the broker closed the connection")
            out += chunk
        return out

    def publish(self, topic, payload, retain=False):
        if self.sock is None:
            raise MQTTError("not connected")
        if not isinstance(payload, (bytes, bytearray)):
            payload = str(payload).encode("utf-8")
        self.sock.sendall(_packet(3, 0x01 if retain else 0, _utf8(topic) + bytes(payload)))
        return True

    def keepalive(self):
        """PINGREQ at half the keepalive; also drains whatever the broker sent us, so a silent
        error is noticed instead of sitting in the socket buffer."""
        try:
            self.sock.settimeout(0.01)
            try:
                while self.sock.recv(4096):
                    pass
            except (socket.timeout, BlockingIOError, ssl.SSLWantReadError):
                pass
        except OSError:
            pass
        self.sock.settimeout(self.timeout)
        if time.time() - self.last_ping >= KEEPALIVE / 2.0:
            self.sock.sendall(_packet(12, 0, b""))
            self.last_ping = time.time()

    def ping(self):
        """A round trip that proves the link: PINGREQ, then wait for PINGRESP.  Used by the page's
        Test connection button after the CONNACK."""
        self.sock.settimeout(self.timeout)
        self.sock.sendall(_packet(12, 0, b""))
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            kind, _ = self._read_packet()
            if kind == 13:
                self.last_ping = time.time()
                return True
        raise MQTTError("no PINGRESP from the broker")

    def close(self):
        try:
            if self.sock is not None:
                self.sock.sendall(_packet(14, 0, b""))           # DISCONNECT: not a will, a goodbye
        except OSError:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        finally:
            self.sock = None


_CONNACK = {1: "unacceptable protocol version", 2: "identifier rejected", 3: "server unavailable",
            4: "bad user name or password", 5: "not authorised"}


def test_connection(cfg, timeout=8.0):
    """Connect, complete the CONNACK, prove the link with a PINGREQ/PINGRESP, disconnect.  Publishes
    nothing but the connect/disconnect, so a test can never leave a retained entity behind."""
    c = Client(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["tls"],
               client_id="pxa-control-test-" + str(os.getpid()), timeout=timeout)
    t0 = time.time()
    try:
        c.connect()
        c.ping()
        ms = int((time.time() - t0) * 1000)
        return {"ok": True, "ms": ms, "tls": bool(cfg["tls"]),
                "detail": "connected to %s:%s%s" % (cfg["host"], cfg["port"], " (TLS)" if cfg["tls"] else "")}
    except Exception as e:                                        # noqa: BLE001 - the message is the answer
        return {"ok": False, "ms": int((time.time() - t0) * 1000), "error": str(e)}
    finally:
        c.close()


# --------------------------------------------------------------------------- what we publish
def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _pct(v):
    f = _num(v)
    return None if f is None else round(f * 100.0, 1)


# (metric, label, unit, device_class, state_class, how to read it out of a server row)
SERVER_METRICS = (
    ("tokens_per_second", "Generation speed", "tok/s", None, "measurement", lambda s: _num(s.get("dec"))),
    ("prompt_tokens_per_second", "Prompt speed", "tok/s", None, "measurement", lambda s: _num(s.get("pre"))),
    ("slots_busy", "Slots busy", None, None, "measurement", lambda s: _num(s.get("busy"))),
    ("slots_total", "Slots", None, None, "measurement", lambda s: _num(s.get("slots"))),
    ("kv_cache_used", "KV cache used", "%", None, "measurement", lambda s: _pct(s.get("ctx"))),
    ("expert_cache_hit", "Expert cache hit", "%", None, "measurement", lambda s: _pct(s.get("xhit"))),
    ("draft_acceptance", "Draft acceptance", "%", None, "measurement", lambda s: _pct(s.get("acc"))),
    ("model", "Model", None, None, None, lambda s: (s.get("model") or None)),
    ("state", "State", None, None, None, lambda s: (s.get("phase") or None)),
)
CARD_METRICS = (
    ("temperature", "Temperature", "°C", "temperature", "measurement", lambda c: _num(c.get("temp"))),
    ("memory_used", "VRAM used", "MiB", None, "measurement", lambda c: _num(c.get("mem"))),
    ("memory_total", "VRAM total", "MiB", None, None, lambda c: _num(c.get("mem_total_mib"))),
    ("utilization", "Utilisation", "%", None, "measurement", lambda c: _num(c.get("util"))),
    ("power", "Power", "W", "power", "measurement", lambda c: _num(c.get("power"))),
)
HOST_METRICS = (
    ("ram_used", "RAM used", "%", None, "measurement", lambda h: _num(h.get("ram"))),
    ("cpu_used", "CPU used", "%", None, "measurement", lambda h: _num(h.get("cpu"))),
    ("swap_used", "Swap used", "%", None, "measurement", lambda h: _num(h.get("swap"))),
)
EXPERT_METRICS = (
    ("learned_sessions", "Expert map sessions", None, None, "measurement",
     lambda e: _num(e.get("sessions"))),
    ("learning", "Expert map", None, None, None,
     lambda e: ("learning" if e.get("learning") else "idle")),
)


def entities(snapshot, base, host_label="pxa-control"):
    """Every entity as (id, metric, value, config).

    `snapshot` is what PXA Control's Live sampler hands over:
        {"cards": [{"index", "name", "mem_total_mib", "mem", "util", "temp", "power", ...}],
         "servers": [{"key", "name", "dec", "pre", "busy", "slots", "ctx", "xhit", "acc",
                      "model", "phase", ...}],
         "host": {"ram", "cpu", "swap"},
         "expert": {"sessions", "learning"}}

    The config is the Home Assistant discovery payload; the caller publishes it retained to
    `<prefix>/sensor/<id>/<metric>/config` and the value to `<base>/<id>/<metric>`.
    """
    dev_id = _dev(host_label)
    device = {"identifiers": [dev_id], "name": host_label, "manufacturer": "PXA",
              "model": "PXA Control", "sw_version": "3.1"}
    avail = base + "/status"
    out = []

    def add(eid, metric, value, label, unit=None, device_class=None, state_class=None, dev=None,
            category=None, icon=None):
        if value is None:
            return                       # a metric this build does not report is not an entity
        obj = {"name": label, "unique_id": "%s_%s" % (eid, metric),
               "state_topic": "%s/%s/%s" % (base, eid, metric),
               "availability_topic": avail, "payload_available": "online",
               "payload_not_available": "offline", "device": dev or device}
        if unit:
            obj["unit_of_measurement"] = unit
        if device_class:
            obj["device_class"] = device_class
        if state_class:
            obj["state_class"] = state_class
        if category:
            obj["entity_category"] = category
        if icon:
            obj["icon"] = icon
        out.append((eid, metric, value, obj))

    for s in snapshot.get("servers") or []:
        eid = "server_" + _slug(s.get("key") or s.get("name") or "server")
        for metric, label, unit, dc, sc, get in SERVER_METRICS:
            name = label if not s.get("name") else "%s %s" % (s["name"], label)
            add(eid, metric, get(s), name, unit, dc, sc)
    for c in snapshot.get("cards") or []:
        i = c.get("index")
        eid = "gpu_%s" % i
        gdev = {"identifiers": ["%s_gpu_%s" % (dev_id, i)], "name": "%s GPU %s" % (host_label, i),
                "manufacturer": "PXA", "via_device": dev_id}
        if c.get("name"):
            gdev["model"] = c["name"]
        for metric, label, unit, dc, sc, get in CARD_METRICS:
            add(eid, metric, get(c), "GPU %s %s" % (i, label), unit, dc, sc, dev=gdev,
                category="diagnostic" if metric in ("memory_total",) else None)
    h = snapshot.get("host") or {}
    for metric, label, unit, dc, sc, get in HOST_METRICS:
        add("host", metric, get(h), label, unit, dc, sc, category="diagnostic")
    if "expert" in snapshot:
        # guarded by PRESENCE, not by an empty dict: the map's state metric is never None, so an absent
        # section would otherwise publish "idle" and tell Home Assistant the map is idle when there is
        # no map at all.
        e = snapshot.get("expert") or {}
        for metric, label, unit, dc, sc, get in EXPERT_METRICS:
            add("expert_map", metric, get(e), label, unit, dc, sc, icon="mdi:brain")
    return out


def discovery_topics(ents, prefix):
    """(config topic, payload) for each entity, in a stable order."""
    out = []
    for eid, metric, _value, cfg in ents:
        out.append(("%s/sensor/%s/%s/config" % (prefix, eid, metric), json.dumps(cfg, sort_keys=True)))
    return out


def state_topics(ents, base):
    """(state topic, payload, retain) for each entity's value."""
    out = []
    for eid, metric, value, _cfg in ents:
        if isinstance(value, float) and value == int(value):
            value = int(value)                          # 70.0 -> "70": prettier, and HA reads both
        out.append(("%s/%s/%s" % (base, eid, metric), value, True))
    return out


def _dev(label):
    return "pxa_" + _slug(label)


def _slug(s):
    out = []
    for ch in str(s).lower():
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "server"


# --------------------------------------------------------------------------- the pump
class Publisher(threading.Thread):
    """Publishes discovery + state every `interval` seconds until stop().  One thread, daemon, and
    every failure is recorded in `status()` rather than raised: a broker that went away must not
    take PXA Control with it."""

    def __init__(self, cfg, snapshot_fn, host_label, log=None, clock=time.time):
        threading.Thread.__init__(self, name="pxa-mqtt", daemon=True)
        self.cfg = dict(cfg)
        self.snapshot_fn = snapshot_fn
        self.host_label = host_label
        self.log = log or (lambda *_: None)
        self.clock = clock
        self.stop_ev = threading.Event()
        self.client = None
        self.state = "starting"
        self.error = None
        self.last_ok = None
        self.published = 0
        self.reconnects = 0
        self._ids_seen = set()

    # ---- the loop ---------------------------------------------------------------------------
    def run(self):
        while not self.stop_ev.is_set():
            try:
                self._cycle()
            except Exception as e:                                    # noqa: BLE001 - never die
                self.state, self.error = "error", str(e)
                self.log("mqtt: %s" % e)
                self._drop()
            self.stop_ev.wait(self.cfg["interval"])

    def _cycle(self):
        cfg = self.cfg
        if self.client is None:
            will = (cfg["base"] + "/status", "offline")
            c = Client(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["tls"], will=will)
            c.connect()
            self.client = c
            self.state, self.error = "connected", None
            self.published = 0
            self.log("mqtt: connected to %s:%s" % (cfg["host"], cfg["port"]))
        c = self.client
        if cfg["discovery"]:
            ents = entities(self._snapshot(), cfg["base"], self.host_label)
            for topic, payload in discovery_topics(ents, cfg["prefix"]):
                c.publish(topic, payload, retain=True)
        ents = entities(self._snapshot(), cfg["base"], self.host_label)
        for topic, payload, retain in state_topics(ents, cfg["base"]):
            c.publish(topic, payload, retain=retain)
            self.published += 1
        c.publish(cfg["base"] + "/status", "online", retain=True)
        c.keepalive()
        self.last_ok = self.clock()
        self.state = "connected"
        self._ids_seen = set((e[0], e[1]) for e in ents)

    def _snapshot(self):
        try:
            return self.snapshot_fn() or {}
        except Exception as e:                                        # noqa: BLE001
            self.log("mqtt: snapshot failed: %s" % e)
            return {}

    def _drop(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:                                         # noqa: BLE001
                pass
            self.client = None
            self.reconnects += 1

    def stop(self):
        self.stop_ev.set()
        if self.is_alive() and threading.current_thread() is not self:
            # Let an in-flight cycle finish first, or it could publish its own "online" after the goodbye below.
            self.join(timeout=2.0)
        if self.client is not None:
            # Say goodbye out loud. A clean DISCONNECT makes the broker DISCARD the last will, so without this
            # a graceful shutdown leaves the retained "online" standing and Home Assistant keeps showing the
            # last reading forever. The will still covers the crash / cable-out case, where nothing gets to speak.
            try:
                self.client.publish(self.cfg["base"] + "/status", "offline", retain=True)
            except Exception:                                     # noqa: BLE001 - we are shutting down regardless
                pass
        self._drop()
        self.state = "stopped"

    def status(self):
        return {"state": self.state, "error": self.error, "last_ok": self.last_ok,
                "published": self.published, "reconnects": self.reconnects,
                "entities": len(self._ids_seen), "alive": self.is_alive()}
