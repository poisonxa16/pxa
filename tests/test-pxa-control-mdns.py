#!/usr/bin/env python3
"""PXA Control's mDNS advert (tools/pxa_control/mdns.py; Router plugin "find rigs"): the DNS wire format, which questions it answers
and how (multicast with the 1 s rule, QU and legacy-unicast straight to the asker, known-answer suppression, off-link refused), that
the TXT record carries no secret, the interface choice (container / VM bridges skipped, PXA_CONTROL_MDNS_IFACES), the off switch
PXA_CONTROL_MDNS=0 and --lan only, and a live advertise -> browse round trip on this computer's home-network interface on a PRIVATE
port (never 5353, so nothing leaves on the real mDNS port during a test run). No GPU.

    python3 tests/test-pxa-control-mdns.py          (wired into CTest as test-pxa-control-mdns)
"""
import ipaddress
import os
import random
import socket
import struct
import sys
import time
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.dont_write_bytecode = True

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("pxa_mdns_under_test", os.path.join(ROOT, "tools", "pxa_control", "mdns.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

LAN = ipaddress.ip_interface("192.168.50.10/24")
WIFI = ipaddress.ip_interface("10.9.0.7/16")
SECRET = "s3cret-token-0123456789-abcdef"


class FakeSock(object):
    def __init__(self):
        self.sent = []
        self.opts = []

    def setsockopt(self, *a):
        self.opts.append(a)

    def sendto(self, data, addr):
        self.sent.append((data, addr))


def adv(port=7777, ifaces=((("br0", LAN)), ("wlan0", WIFI))):
    a = M.Advertiser(port, "4.1.0", hostname="Tower", ifaces=list(ifaces), mport=5353)
    a.sock = FakeSock()
    return a


class Wire(unittest.TestCase):
    def test_roundtrip_every_record(self):
        a = adv()
        rec = a.records(LAN)
        ans = rec["ptr"] + rec["srv"] + rec["txt"] + rec["a"] + rec["meta"]
        m = M.parse(M.build(0, 0x8400, [], ans))
        self.assertEqual(m["flags"], 0x8400)
        got = {(n, t): (c, ttl, d) for (n, t, c, ttl, d) in m["rr"]}
        self.assertEqual(got[(M.SERVICE, M.T_PTR)][2]["target"], "PXA Control on Tower." + M.SERVICE)
        srv = got[("PXA Control on Tower." + M.SERVICE, M.T_SRV)]
        self.assertEqual((srv[2]["port"], srv[2]["target"]), (7777, "pxa-tower.local"))
        self.assertEqual(srv[0], 0x8001, "SRV carries the cache-flush bit")
        self.assertEqual(got[("pxa-tower.local", M.T_A)][2]["addr"], "192.168.50.10")
        self.assertEqual(got[(M.META, M.T_PTR)][2]["target"], M.SERVICE)

    def test_compression_pointer_and_garbage(self):
        # a hand-made answer that points back at the question's name (what Bonjour / avahi send)
        q = M._labels(M.SERVICE)
        pkt = struct.pack("!HHHHHH", 0, 0x8400, 1, 1, 0, 0) + q + struct.pack("!HH", 12, 1)
        target = b"\x04Box1" + b"\xc0\x0c"
        pkt += b"\xc0\x0c" + struct.pack("!HHIH", 12, 1, 120, len(target)) + target
        m = M.parse(pkt)
        self.assertEqual(m["rr"][0][4]["target"], "Box1." + M.SERVICE)
        for bad in (b"", b"\x00" * 5, struct.pack("!HHHHHH", 0, 0, 1, 0, 0, 0) + b"\xc0\x0c",
                    struct.pack("!HHHHHH", 0, 0, 1, 0, 0, 0) + b"\x3f" + b"a" * 10):
            with self.assertRaises((ValueError, struct.error, IndexError)):
                M.parse(bad)

    def test_instance_name_and_hostname(self):
        self.assertEqual(adv().instance, "PXA Control on Tower")
        self.assertEqual(adv(port=7801).instance, "PXA Control on Tower (7801)", "a second Control on one box gets its own name")
        self.assertEqual(M.Advertiser(7777, hostname="My Rig_2!", ifaces=[("e", LAN)]).hostname, "pxa-my-rig-2.local")


class NoSecret(unittest.TestCase):
    def test_txt_has_no_secret(self):
        os.environ["PXA_CONTROL_TOKEN"] = SECRET
        try:
            a = adv()
            self.assertEqual(a.txt(), ["v=1", "path=/", "ver=4.1.0", "auth=token"])
            for itf in (LAN, WIFI):
                rec = a.records(itf)
                blob = M.build(0, 0x8400, [], sum(rec.values(), []))
                self.assertNotIn(SECRET.encode(), blob)
                self.assertNotIn(b"token=", blob)
        finally:
            os.environ.pop("PXA_CONTROL_TOKEN", None)

    def test_version_is_sanitised(self):
        a = M.Advertiser(7777, "4.1.0; rm -rf / \x00 token=abc", hostname="x", ifaces=[("e", LAN)])
        self.assertTrue(all(c.isalnum() or c in "._+-" for c in a.version), a.version)


class Answers(unittest.TestCase):
    def q(self, name, t, qu=False, pid=0, known=()):
        pkt = M.build(pid, 0, [(name, t, 0x8001 if qu else 1)], [])
        if known:
            hdr = list(struct.unpack("!HHHHHH", pkt[:12]))
            hdr[3] = len(known)
            pkt = struct.pack("!HHHHHH", *hdr) + pkt[12:] + b"".join(M._rr(*k[:4]) for k in known)
        return pkt

    def test_ptr_query_multicast_with_additionals(self):
        a = adv()
        self.assertEqual(a.handle(self.q(M.SERVICE, M.T_PTR), ("192.168.50.77", 5353)), "multicast")
        data, dst = a.sock.sent[-1]
        self.assertEqual(dst, (M.GROUP, 5353))
        m = M.parse(data)
        types_ = sorted(t for (_n, t, _c, _ttl, _d) in m["rr"])
        self.assertEqual(types_, sorted([M.T_PTR, M.T_SRV, M.T_TXT, M.T_A]))
        addrs = [d["addr"] for (_n, t, _c, _ttl, d) in m["rr"] if t == M.T_A]
        self.assertEqual(addrs, ["192.168.50.10"], "only the address on the asker's network")
        mc_if = [o for o in a.sock.opts if o[1] == socket.IP_MULTICAST_IF][-1]
        self.assertEqual(mc_if[2], socket.inet_aton("192.168.50.10"), "sent out of the link the question came on")

    def test_one_second_rule(self):
        a = adv()
        self.assertEqual(a.handle(self.q(M.SERVICE, M.T_PTR), ("192.168.50.77", 5353)), "multicast")
        self.assertIsNone(a.handle(self.q(M.SERVICE, M.T_PTR), ("192.168.50.78", 5353)))
        self.assertEqual(len(a.sock.sent), 1)
        self.assertEqual(a.handle(self.q(M.SERVICE, M.T_PTR), ("10.9.3.3", 5353)), "multicast", "another link has its own clock")

    def test_legacy_unicast(self):
        a = adv()
        self.assertEqual(a.handle(self.q(M.SERVICE, M.T_PTR, pid=0x4242), ("192.168.50.77", 40000)), "unicast")
        data, dst = a.sock.sent[-1]
        self.assertEqual(dst, ("192.168.50.77", 40000))
        m = M.parse(data)
        self.assertEqual(m["id"], 0x4242)
        self.assertEqual(m["qd"][0][0], M.SERVICE, "the question is echoed")
        self.assertTrue(all(ttl <= 10 for (_n, _t, _c, ttl, _d) in m["rr"]))
        self.assertTrue(all(not (c & 0x8000) for (_n, _t, c, _ttl, _d) in m["rr"]), "no cache-flush bit in a legacy answer")

    def test_qu_question_answered_unicast_on_5353(self):
        a = adv()
        self.assertEqual(a.handle(self.q(M.SERVICE, M.T_PTR, qu=True), ("192.168.50.77", 5353)), "unicast")
        self.assertEqual(a.sock.sent[-1][1], ("192.168.50.77", 5353))

    def test_off_link_and_not_ours_and_responses(self):
        a = adv()
        self.assertIsNone(a.handle(self.q(M.SERVICE, M.T_PTR), ("8.8.8.8", 5353)), "off-link asker")
        self.assertIsNone(a.handle(self.q(M.SERVICE, M.T_PTR), ("172.17.0.2", 5353)), "a container bridge we do not serve")
        self.assertIsNone(a.handle(self.q("_http._tcp.local", M.T_PTR), ("192.168.50.77", 5353)), "not our service")
        resp = M.build(0, 0x8400, [], a.records(LAN)["ptr"])
        self.assertIsNone(a.handle(resp, ("192.168.50.77", 5353)), "a response is never answered")
        self.assertIsNone(a.handle(b"\x01\x02junk", ("192.168.50.77", 5353)))
        self.assertEqual(a.sock.sent, [])

    def test_meta_srv_txt_a_questions(self):
        a = adv()
        for (name, t, want) in ((M.META, M.T_PTR, M.T_PTR), (a.fqdn_instance, M.T_SRV, M.T_SRV),
                                (a.fqdn_instance, M.T_TXT, M.T_TXT), ("PXA-TOWER.local", M.T_A, M.T_A)):
            a._last.clear()
            self.assertEqual(a.handle(self.q(name, t), ("192.168.50.77", 40001)), "unicast", name)
            m = M.parse(a.sock.sent[-1][0])
            self.assertIn(want, [rt for (_n, rt, _c, _ttl, _d) in m["rr"]])

    def test_known_answer_suppression(self):
        a = adv()
        known = [(M.SERVICE, M.T_PTR, 4500, M._labels(a.fqdn_instance))]
        self.assertIsNone(a.handle(self.q(M.SERVICE, M.T_PTR, known=known), ("192.168.50.77", 5353)))
        stale = [(M.SERVICE, M.T_PTR, 100, M._labels(a.fqdn_instance))]
        self.assertEqual(a.handle(self.q(M.SERVICE, M.T_PTR, known=stale), ("192.168.50.77", 5353)), "multicast",
                         "a known answer with less than half its TTL left is refreshed")

    def test_announce_and_goodbye(self):
        a = adv()
        a.announce()
        self.assertEqual(len(a.sock.sent), 2, "one per interface")
        self.assertTrue(all(ttl > 0 for (_n, _t, _c, ttl, _d) in M.parse(a.sock.sent[0][0])["rr"]))
        a.sock.sent.clear()
        a.announce(goodbye=True)
        self.assertTrue(all(ttl == 0 for (_n, _t, _c, ttl, _d) in M.parse(a.sock.sent[0][0])["rr"]))


class Interfaces(unittest.TestCase):
    OUT = ("1: lo    inet 127.0.0.1/8 scope host lo\n"
           "8: br0    inet 192.168.1.58/24 brd 192.168.1.255 scope global br0\n"
           "9: wlan0    inet 192.168.1.57/24 brd 192.168.1.255 scope global wlan0\n"
           "10: docker0    inet 172.17.0.1/16 scope global docker0\n"
           "11: br-2f6be89f7b2b    inet 172.19.0.1/16 scope global br-2f6\n"
           "12: virbr0    inet 192.168.122.1/24 scope global virbr0\n"
           "13: eth1@if5    inet 10.1.2.3/8 scope global eth1\n"
           "14: eth2    inet 203.0.113.9/24 scope global eth2\n")

    def run_ip(self, *a, **k):
        return types.SimpleNamespace(stdout=self.OUT, returncode=0)

    def test_home_network_only_and_bridges_skipped(self):
        got = [(n, str(i)) for (n, i) in M.interfaces({}, run=self.run_ip)]
        self.assertEqual(got, [("br0", "192.168.1.58/24"), ("wlan0", "192.168.1.57/24"), ("eth1", "10.1.2.3/8")])

    def test_named_interfaces(self):
        got = [n for (n, _i) in M.interfaces({"PXA_CONTROL_MDNS_IFACES": "wlan0, docker0"}, run=self.run_ip)]
        self.assertEqual(got, ["wlan0", "docker0"], "picking by name is the owner's choice (still home-network addresses only)")


class Switches(unittest.TestCase):
    def test_enabled(self):
        self.assertTrue(M.enabled({}))
        for v in ("0", "off", "false", "no", " OFF "):
            self.assertFalse(M.enabled({"PXA_CONTROL_MDNS": v}), v)

    def test_start_for(self):
        self.assertEqual(M.start_for(7777, "x", lan=False), (None, None), "only a Control on the network advertises")
        a, line = M.start_for(7777, "x", lan=True, environ={"PXA_CONTROL_MDNS": "0"})
        self.assertIsNone(a)
        self.assertIn("PXA_CONTROL_MDNS=0", line)

    def test_serve_wires_it_behind_lan(self):
        src = open(os.path.join(ROOT, "tools", "pxa_control", "server.py"), encoding="utf-8").read()
        i = src.index("_mdns.start_for(port, CONTROL_VERSION, lan)")
        self.assertIn("if lan:", src[i - 300:i])
        self.assertIn("mdns_adv.stop", src, "goodbye at exit")


def _private_port():
    for _ in range(20):
        p = random.randint(20000, 60000)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(("", p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    return None


class Live(unittest.TestCase):
    """a real socket pair on this computer's home-network interface, on a private port and group port (not 5353)."""

    def test_advertise_then_browse(self):
        ifs = M.interfaces()
        if not ifs:
            self.skipTest("no home-network interface here")
        port = _private_port()
        a = M.Advertiser(17777, "4.1.0", hostname="mdns-test", ifaces=ifs[:1], mport=port).start()
        try:
            time.sleep(0.3)
            got = M.browse(1.5, ifaces=ifs[:1], mport=port)
        finally:
            a.stop()
        hit = [g for g in got if g["name"] == "PXA Control on mdns-test (17777)"]
        self.assertEqual(len(hit), 1, got)
        self.assertEqual(hit[0]["port"], 17777)
        self.assertEqual(hit[0]["host"], str(ifs[0][1].ip))
        self.assertIn("auth=token", hit[0]["txt"])

    def test_goodbye_removes(self):
        ifs = M.interfaces()
        if not ifs:
            self.skipTest("no home-network interface here")
        port = _private_port()
        a = M.Advertiser(17778, "4.1.0", hostname="mdns-test", ifaces=ifs[:1], mport=port).start()
        a.stop()
        self.assertIsNone(a.sock)
        self.assertEqual([g for g in M.browse(0.8, ifaces=ifs[:1], mport=port) if "17778" in g["name"]], [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
