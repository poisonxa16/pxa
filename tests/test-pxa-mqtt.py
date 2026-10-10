#!/usr/bin/env python3
"""PXA Control's MQTT publisher (v3.1, tools/pxa_mqtt.py): the optional bridge that puts Control's own
readings -- per-GPU temperature/VRAM/utilisation/power, host load, the expert map, and each server's
speed and slots -- into Home Assistant by MQTT discovery.

What is checked here is the CONTRACT Home Assistant consumes: the settings and their environment
overrides with the password masked on the way out, the topic root validation, the packet framing on
the wire (the fixed header is ONE type byte then a varint length -- a two-byte read there silently
breaks every reply), the discovery payload (unique_id, device block, unit, device_class,
state_class, the availability wiring), the topic shapes and retain flags, and the availability
behaviour on both shutdown paths.

No broker and no network: a fake socket for the wire, a fake snapshot for the entities. The live
end-to-end run against a real mosquitto is a separate harness (the release notes' evidence).

    python3 tests/test-pxa-mqtt.py          (wired into CTest as test-pxa-mqtt)
"""
import json
import os
import socket
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import pxa_mqtt as M  # noqa: E402


def snapshot(cards=True, host=True, expert=True, servers=True, card_power=True):
    """A Live-sampler snapshot shaped like the real one (tools/pxa_control.py Live.latest(with_meta=True))."""
    s = {}
    if cards:
        c = {"index": 0, "name": "Tesla P100-PCIE-16GB", "mem_total_mib": 16384,
             "mem": 6400, "util": 42, "temp": 61, "power": 120.0 if card_power else None}
        c2 = {"index": 1, "name": "Tesla V100-PCIE-16GB", "mem_total_mib": 16384,
              "mem": 120, "util": 0, "temp": 33, "power": 28.0}
        s["cards"] = [c, c2]
    if servers:
        s["servers"] = [{"key": "d:8080", "name": "brain", "port": 8080, "dec": 41.6, "pre": 512.0,
                         "busy": 0, "slots": 2, "ctx": 0.25, "xhit": None, "acc": None,
                         "model": "qwen3-0.6b-q8", "phase": "idle", "kind": "docker",
                         "n_ctx": 4096, "model_file": "/models/qwen3-0.6b-q8.gguf"}]
    if host:
        s["host"] = {"ram": 31.5, "cpu": 8.2, "swap": 0.0}
    if expert:
        s["expert"] = {"sessions": 12, "learning": False}
    return s


class SettingsTests(unittest.TestCase):
    def test_defaults_are_off(self):
        s = M.settings({})
        self.assertFalse(s["enabled"], "the publisher must be OFF unless the user turns it on")
        self.assertEqual(s["port"], 1883)
        self.assertEqual(s["base"], "pxa")
        self.assertEqual(s["prefix"], "homeassistant")
        self.assertEqual(s["interval"], 15.0)
        self.assertTrue(s["discovery"])
        self.assertFalse(s["tls"])
        self.assertEqual(s["host"], "")

    def test_the_saved_block_is_read(self):
        s = M.settings({"mqtt": {"enabled": True, "host": "broker.lan", "port": 8883, "tls": True,
                                 "base": "pxa/control", "interval": 30, "discovery": False}})
        self.assertTrue(s["enabled"])
        self.assertEqual((s["host"], s["port"], s["tls"]), ("broker.lan", 8883, True))
        self.assertEqual(s["base"], "pxa/control")
        self.assertEqual(s["interval"], 30.0)
        self.assertFalse(s["discovery"])

    def test_env_wins_over_the_page(self):
        env = {"PXA_CONTROL_MQTT": "1", "PXA_CONTROL_MQTT_HOST": "env.lan",
               "PXA_CONTROL_MQTT_PORT": "18830", "PXA_CONTROL_MQTT_BASE": "pxa/env",
               "PXA_CONTROL_MQTT_INTERVAL": "5", "PXA_CONTROL_MQTT_TLS": "0",
               "PXA_CONTROL_MQTT_DISCOVERY": "0"}
        s = M.settings({"mqtt": {"enabled": False, "host": "page.lan", "port": 1883}}, env)
        self.assertTrue(s["enabled"], "PXA_CONTROL_MQTT=1 turns it on")
        self.assertEqual(s["host"], "env.lan")
        self.assertEqual(s["port"], 18830)
        self.assertEqual(s["base"], "pxa/env")
        self.assertEqual(s["interval"], 5.0)
        self.assertFalse(s["tls"])
        self.assertFalse(s["discovery"])

    def test_limits_are_clamped(self):
        s = M.settings({"mqtt": {"port": 99999, "interval": 0.01}})
        self.assertTrue(1 <= s["port"] <= 65535, "a port out of range must be clamped, not passed on")
        self.assertGreaterEqual(s["interval"], M._LIMITS["interval"][0])

    def test_public_masks_the_password(self):
        p = M.public({"enabled": True, "host": "b", "password": "hunter2"})
        self.assertTrue(p.get("password_set"))
        self.assertEqual(p.get("password"), "", "the value is blanked, never sent back to the page")
        self.assertNotIn("hunter2", json.dumps(p))

    def test_public_reports_a_blank_password_as_unset(self):
        self.assertFalse(M.public({"enabled": True, "password": ""}).get("password_set"))


class TopicTests(unittest.TestCase):
    def test_a_plain_root_is_legal(self):
        for t in ("pxa", "pxa/control", "home/pxa_1", "a-b_c"):
            self.assertTrue(M.valid_topic_root(t), t)

    def test_wildcards_and_empties_are_refused(self):
        for t in ("", " ", "pxa/+", "pxa/#", "+", "#", "a b", "$SYS", None, "   "):
            self.assertFalse(M.valid_topic_root(t), repr(t))

    def test_a_stray_edge_slash_is_trimmed_not_refused(self):
        """A slash at either end is a typo the user cannot see the effect of, so it is fixed rather
        than rejected: the setting reads back as the topic that is actually used."""
        self.assertEqual(M.valid_topic_root("/pxa"), "pxa")
        self.assertEqual(M.valid_topic_root("pxa/"), "pxa")
        self.assertEqual(M.valid_topic_root("  pxa/control  "), "pxa/control")


class FakeSock(object):
    """A byte stream, not a list of chunks: recv(n) hands back up to n bytes and KEEPS the rest, the
    way a socket does.  A chunk list that popped each item whole would drop the tail of every read."""

    def __init__(self, chunks):
        self.buf = bytearray(b"".join(chunks))
        self.sent = []

    def settimeout(self, _t):
        pass

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, n):
        if not self.buf:
            raise socket.timeout("nothing more")
        out, self.buf = bytes(self.buf[:n]), self.buf[n:]
        return out

    def close(self):
        pass


class WireTests(unittest.TestCase):
    def test_varint_length(self):
        self.assertEqual(M._remaining(0), b"\x00")
        self.assertEqual(M._remaining(127), b"\x7f")
        self.assertEqual(M._remaining(128), b"\x80\x01")
        self.assertEqual(M._remaining(16383), b"\xff\x7f")
        self.assertEqual(M._remaining(16384), b"\x80\x80\x01")

    def test_packet_header_is_type_then_length(self):
        p = M._packet(3, 0, b"hello")
        self.assertEqual(p[0] >> 4, 3)
        self.assertEqual(p[1], 5, "one length byte for a 5-byte body")
        self.assertEqual(p[2:], b"hello")

    def test_utf8_is_length_prefixed(self):
        self.assertEqual(M._utf8("ab"), b"\x00\x02ab")

    def test_a_connack_reads_back_intact(self):
        """THE regression: the fixed header is one type byte then the varint length. Reading two bytes
        swallowed the length byte (0x02), so a CONNACK -- 20 02 00 00 -- parsed as a zero-length
        payload and every reply came back "short" against a perfectly healthy broker."""
        c = M.Client("127.0.0.1", 1883, "", "", False)
        c.sock = FakeSock([b"\x20", b"\x02\x00\x00"])
        kind, payload = c._read_packet()
        self.assertEqual(kind, 2, "CONNACK")
        self.assertEqual(payload, b"\x00\x00")
        self.assertEqual(payload[1], 0, "the return code is the SECOND payload byte -- 0 means accepted")

    def test_a_multi_byte_length_reads_back_intact(self):
        body = b"x" * 200
        c = M.Client("127.0.0.1", 1883, "", "", False)
        c.sock = FakeSock([b"\x30", bytes([200 & 0x7F | 0x80, 200 >> 7]), body])
        kind, payload = c._read_packet()
        self.assertEqual(kind, 3)
        self.assertEqual(payload, body)

    def test_a_connect_with_a_will_retains_it(self):
        sock = FakeSock([b"\x20\x02\x00\x00"])
        real = M.socket.create_connection
        M.socket.create_connection = lambda *a, **k: sock
        try:
            c = M.Client("127.0.0.1", 1883, "u", "p", False, will=("pxa/status", "offline"))
            c.connect()
        finally:
            M.socket.create_connection = real
        hello = sock.sent[0]
        self.assertEqual(hello[0] >> 4, 1, "CONNECT")
        body = hello[2:]                      # 0x10 type, one length byte
        level = body.index(b"MQTT") + 4       # past the length-prefixed "MQTT"
        flags_byte = body[level + 1]
        self.assertTrue(flags_byte & 0x04, "will flag must be set")
        self.assertTrue(flags_byte & 0x20,
                        "the will must be RETAINED: a client that is gone when it fails cannot leave "
                        "a retained 'online' behind")
        self.assertIn(b"pxa/status", body)
        self.assertIn(b"offline", body)

    def test_disconnect_is_the_goodbye_packet(self):
        c = M.Client("127.0.0.1", 1883, "", "", False)
        sock = FakeSock([])
        c.sock = sock
        c.close()
        self.assertEqual(sock.sent[-1], bytes([0xE0, 0x00]))


class EntityTests(unittest.TestCase):
    def setUp(self):
        self.ents = M.entities(snapshot(), "pxa", "PXANET")
        self.by = {(e, m): (v, c) for e, m, v, c in self.ents}

    def test_the_device_and_the_gpu_sub_devices(self):
        _v, cfg = self.by[("host", "ram_used")]
        self.assertEqual(cfg["device"]["identifiers"], ["pxa_pxanet"])
        self.assertEqual(cfg["device"]["name"], "PXANET")
        self.assertEqual(cfg["device"]["manufacturer"], "PXA")
        _v, g = self.by[("gpu_0", "temperature")]
        self.assertEqual(g["device"]["name"], "PXANET GPU 0")
        self.assertEqual(g["device"]["via_device"], "pxa_pxanet",
                         "a GPU attaches to the Control device, not to the LAN")

    def test_unique_id_names_its_own_state_topic(self):
        for (eid, metric), (_v, cfg) in self.by.items():
            self.assertEqual(cfg["unique_id"], "%s_%s" % (eid, metric))
            self.assertEqual(cfg["state_topic"], "pxa/%s/%s" % (eid, metric))

    def test_every_entity_carries_the_availability_wiring(self):
        for (_eid, _m), (_v, cfg) in self.by.items():
            self.assertEqual(cfg["availability_topic"], "pxa/status")
            self.assertEqual(cfg["payload_available"], "online")
            self.assertEqual(cfg["payload_not_available"], "offline")

    def test_units_and_classes(self):
        u = lambda k: self.by[k][1]          # noqa: E731
        self.assertEqual(u(("gpu_0", "temperature"))["unit_of_measurement"], "°C")
        self.assertEqual(u(("gpu_0", "temperature"))["device_class"], "temperature")
        self.assertEqual(u(("gpu_0", "temperature"))["state_class"], "measurement")
        self.assertEqual(u(("gpu_1", "power"))["device_class"], "power")
        self.assertEqual(u(("gpu_1", "memory_used"))["unit_of_measurement"], "MiB")
        self.assertEqual(u(("gpu_1", "utilization"))["unit_of_measurement"], "%")
        self.assertEqual(u(("host", "ram_used"))["unit_of_measurement"], "%")
        self.assertEqual(u(("server_d_8080", "tokens_per_second"))["unit_of_measurement"], "tok/s")
        self.assertEqual(u(("server_d_8080", "kv_cache_used"))["unit_of_measurement"], "%")
        self.assertEqual(u(("server_d_8080", "tokens_per_second"))["state_class"], "measurement")

    def test_a_fixed_capacity_is_not_a_measurement(self):
        """VRAM total never moves; a state_class there would have Home Assistant build long-term
        statistics on a constant."""
        cfg = self.by[("gpu_0", "memory_total")][1]
        self.assertNotIn("state_class", cfg)
        self.assertEqual(cfg["unit_of_measurement"], "MiB")

    def test_text_metrics_carry_no_unit(self):
        for key in (("server_d_8080", "model"), ("server_d_8080", "state"),
                    ("expert_map", "learning")):
            self.assertNotIn("unit_of_measurement", self.by[key][1], key)

    def test_the_percentages_are_percentages(self):
        self.assertEqual(self.by[("server_d_8080", "kv_cache_used")][0], 25.0,
                         "ctx 0.25 is 25 percent, not 0.25")
        self.assertEqual(self.by[("host", "ram_used")][0], 31.5)

    def test_a_metric_the_build_does_not_report_makes_no_entity(self):
        """The documented rule: a reading that is not there produces no sensor, rather than one frozen
        at zero.  With no power reading for card 0 that entity is absent; card 1 still has its own."""
        got = {(e, m) for e, m, _v, _c in M.entities(snapshot(card_power=False), "pxa", "PXANET")}
        self.assertNotIn(("gpu_0", "power"), got)
        self.assertIn(("gpu_0", "temperature"), got, "only the missing metric goes, not the card")
        self.assertIn(("gpu_1", "power"), got)

    def test_a_silent_server_still_names_itself(self):
        """dec/pre are None while nothing has generated; the model, slots and state are still real."""
        self.assertNotIn(("server_d_8080", "draft_acceptance"), self.by,
                         "no speculative decoding configured, so no acceptance entity")
        self.assertIn(("server_d_8080", "model"), self.by)

    def test_no_section_no_entities(self):
        self.assertEqual(M.entities({}, "pxa", "H"), [])

    def test_an_absent_expert_map_makes_no_entities(self):
        """The state metric is never None, so an absent section would otherwise publish 'idle' -- and
        tell Home Assistant the map is idle on a build that has no map at all."""
        got = M.entities(snapshot(expert=False), "pxa", "H")
        self.assertEqual([m for e, m, _v, _c in got if e == "expert_map"], [])

    def test_float_values_print_as_integers_when_whole(self):
        st = dict((t, (v, r)) for t, v, r in M.state_topics(self.ents, "pxa"))
        self.assertEqual(st["pxa/gpu_1/memory_used"][0], 120, "120.0 -> 120")
        self.assertIsInstance(st["pxa/gpu_0/temperature"][0], int)


class TopicPublishTests(unittest.TestCase):
    def setUp(self):
        self.ents = M.entities(snapshot(), "pxa", "PXANET")

    def test_discovery_topics_and_retain(self):
        d = dict(M.discovery_topics(self.ents, "homeassistant"))
        self.assertIn("homeassistant/sensor/gpu_0/temperature/config", d)
        self.assertIn("homeassistant/sensor/server_d_8080/tokens_per_second/config", d)
        self.assertEqual(len(d), len(self.ents))
        for topic, payload in d.items():
            self.assertTrue(topic.startswith("homeassistant/sensor/"), topic)
            self.assertTrue(topic.endswith("/config"), topic)
            self.assertIsInstance(json.loads(payload), dict)

    def test_a_custom_prefix_moves_every_config(self):
        d = dict(M.discovery_topics(self.ents, "ha"))
        self.assertTrue(all(t.startswith("ha/sensor/") for t in d))

    def test_state_topics_are_retained(self):
        for topic, _value, retain in M.state_topics(self.ents, "pxa"):
            self.assertTrue(retain, "a state value is retained so a restarting HA has the last reading")
            self.assertTrue(topic.startswith("pxa/"), topic)

    def test_the_contract_the_broker_capture_asserts(self):
        """The same shape the live harness checks against mosquitto: config topic, unique_id, device,
        and a matching state topic for every entity."""
        states = set(t for t, _v, _r in M.state_topics(self.ents, "pxa"))
        for topic, payload in M.discovery_topics(self.ents, "homeassistant"):
            cfg = json.loads(payload)
            self.assertIn(cfg["state_topic"], states)
            self.assertTrue(cfg["device"]["identifiers"])
            self.assertEqual(cfg["unique_id"], topic.split("/")[2] + "_" + topic.split("/")[3])


class FakeClient(object):
    def __init__(self):
        self.published = []
        self.closed = False

    def publish(self, topic, payload, retain=False):
        self.published.append((topic, payload, retain))

    def keepalive(self):
        pass

    def close(self):
        self.closed = True


class AvailabilityTests(unittest.TestCase):
    def pub(self):
        p = M.Publisher(M.settings({"mqtt": {"enabled": True, "host": "h", "base": "pxa"}}),
                        lambda: snapshot(), "PXANET")
        p.client = FakeClient()
        return p

    def test_a_clean_stop_says_offline_out_loud(self):
        """A clean DISCONNECT makes the broker DISCARD the will, so without an explicit goodbye a
        graceful shutdown leaves the retained 'online' standing and Home Assistant holds the last
        reading forever."""
        p = self.pub()
        c = p.client                       # stop() drops the reference, so hold it to read what was sent
        p.stop()
        avail = [t for t in c.published if t[0] == "pxa/status"]
        self.assertTrue(avail, "stop() must publish to the availability topic")
        self.assertEqual(avail[-1], ("pxa/status", "offline", True))
        self.assertTrue(c.closed, "and then close the socket")
        self.assertIsNone(p.client, "the client is dropped")
        self.assertEqual(p.state, "stopped")

    def test_a_cycle_reports_online(self):
        p = self.pub()
        p._cycle()
        avail = [t for t in p.client.published if t[0] == "pxa/status"]
        self.assertEqual(avail[-1], ("pxa/status", "online", True),
                         "retained, so an HA that starts later still learns it is up")
        self.assertTrue(p.published > 0)
        self.assertEqual(p.state, "connected")

    def test_the_status_names_the_entities_it_published(self):
        p = self.pub()
        p._cycle()
        st = p.status()
        self.assertEqual(st["entities"], len(M.entities(snapshot(), "pxa", "PXANET")))
        self.assertTrue(st["alive"]) if p.is_alive() else None
        self.assertIsNone(st["error"])

    def test_a_failing_snapshot_does_not_kill_the_thread(self):
        def boom():
            raise RuntimeError("the sampler is not ready")
        p = M.Publisher(M.settings({"mqtt": {"enabled": True, "host": "h"}}), boom, "H")
        p.client = FakeClient()
        p._cycle()                       # discovery and state come out empty; no exception escapes
        self.assertEqual(p.published, 0)
        self.assertEqual(p.state, "connected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
