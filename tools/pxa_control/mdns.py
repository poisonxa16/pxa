"""mDNS / DNS-SD advert for a PXA Control that is on the network (`--lan`), and a one-shot browser for it. Standard library only.

    Advertiser(port, version).start()   answers `_pxa-control._tcp.local` (PTR, SRV, TXT, A) on this computer's home-network
                                        interfaces; announces twice at start and says goodbye (TTL 0) at stop.
    browse(timeout)                     -> [{"name", "host", "port", "txt", "addrs"}]: every PXA Control that answers a query sent
                                        from an ephemeral port (legacy unicast, RFC 6762 s6.7), so it works beside avahi-daemon.

Rules:
* Started by server.serve() only when Control listens on the network (--lan). Off switch: PXA_CONTROL_MDNS=0.
* TXT carries NO secret: v=1, path=/, ver=<Control version>, auth=token (the token is required; the token itself never leaves Control).
* The socket shares UDP 5353 with avahi-daemon / Bonjour / Home Assistant (SO_REUSEADDR, plus SO_REUSEPORT where it exists).
* Answers only queries from an address on one of the advertised interfaces' networks (on-link, RFC 6762 s11); the A records in an answer
  are the addresses of that interface's network, so a rig with a wired and a wireless leg never hands out an unreachable address.
* Interfaces: private (RFC 1918) and link-local IPv4 only; container / VM bridges (docker*, br-*, veth*, virbr*, ...) are skipped.
  PXA_CONTROL_MDNS_IFACES=br0,eth0 picks them by name instead.
* No probing / conflict renaming (RFC 6762 s8): the instance name carries the host name (and the port when it is not 7777), so two
  Controls on one LAN or on one box never claim the same name. The host name is pxa-<hostname>.local (never <hostname>.local, which
  the OS responder owns).
"""
import ipaddress
import os
import random
import re
import select
import socket
import struct
import subprocess
import threading
import time

GROUP = "224.0.0.251"
PORT = 5353
SERVICE = "_pxa-control._tcp.local"
META = "_services._dns-sd._udp.local"
DEFAULT_CONTROL_PORT = 7777
T_A, T_PTR, T_TXT, T_SRV, T_ANY = 1, 12, 16, 33, 255
TTL_HOST, TTL_OTHER = 120, 4500                 # RFC 6762 s10 recommendations
SKIP_IFACES = ("lo", "docker", "br-", "veth", "virbr", "vnet", "lxc", "lxd", "cni", "flannel", "cali", "podman", "tun", "tap",
               "wg", "tailscale", "zt", "pxa-")
NET_RE = re.compile(r"^\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)/(\d+)")


def enabled(environ=None):
    env = os.environ if environ is None else environ
    return str(env.get("PXA_CONTROL_MDNS", "1")).strip().lower() not in ("0", "off", "false", "no")


# ---- interfaces ------------------------------------------------------------------------------------------------------------
HOME_NETS = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"))


def _homenet(ip):
    a = ipaddress.ip_address(ip)
    return any(a in n for n in HOME_NETS)


def interfaces(environ=None, run=subprocess.run):
    """-> [(ifname, IPv4Interface)] this computer's home-network IPv4 interfaces. Linux: `ip -4 -o addr`; elsewhere (or when `ip` is
    missing) the address the default route uses, taken as a /24."""
    env = os.environ if environ is None else environ
    want = [x.strip() for x in str(env.get("PXA_CONTROL_MDNS_IFACES") or "").split(",") if x.strip()]
    out = []
    try:
        r = run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=3)
        for ln in (r.stdout or "").splitlines():
            m = NET_RE.match(ln.strip())
            if not m:
                continue
            name = m.group(1).split("@", 1)[0]
            itf = ipaddress.ip_interface("%s/%s" % (m.group(2), m.group(3)))
            if want:
                if name not in want:
                    continue
            elif name.startswith(SKIP_IFACES):
                continue
            if _homenet(str(itf.ip)):
                out.append((name, itf))
    except (OSError, ValueError, subprocess.SubprocessError):
        out = []
    if not out:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.168.1.1", 9))
            ip = s.getsockname()[0]
            if _homenet(ip):
                out.append(("default", ipaddress.ip_interface(ip + "/24")))
        except OSError:
            pass
        finally:
            s.close()
    return out


# ---- DNS wire format -------------------------------------------------------------------------------------------------------
def _labels(name):
    out = b""
    for part in name.rstrip(".").split("."):
        raw = part.encode("utf-8")[:63]
        out += bytes([len(raw)]) + raw
    return out + b"\0"


def _read_name(buf, off, depth=0):
    """-> (name, next offset). Follows compression pointers (bounded)."""
    parts, jumped, nxt = [], False, off
    for _ in range(128):
        if off >= len(buf):
            raise ValueError("name past the end")
        n = buf[off]
        if n == 0:
            off += 1
            break
        if n & 0xC0 == 0xC0:
            if off + 1 >= len(buf) or depth > 16:
                raise ValueError("bad pointer")
            ptr = ((n & 0x3F) << 8) | buf[off + 1]
            if not jumped:
                nxt = off + 2
            jumped = True
            off, depth = ptr, depth + 1
            continue
        if n & 0xC0:
            raise ValueError("bad label")
        parts.append(buf[off + 1:off + 1 + n].decode("utf-8", "replace"))
        off += 1 + n
    else:
        raise ValueError("name too long")
    return ".".join(parts), (nxt if jumped else off)


def parse(buf):
    """-> {"id", "flags", "qd": [(name, type, qclass)], "rr": [(name, type, rclass, ttl, rdata dict)]} (answers + authority +
    additional). ValueError on a malformed packet."""
    if len(buf) < 12:
        raise ValueError("short")
    pid, flags, nq, na, nn, nr = struct.unpack("!HHHHHH", buf[:12])
    off, qd, rr = 12, [], []
    for _ in range(nq):
        name, off = _read_name(buf, off)
        t, c = struct.unpack("!HH", buf[off:off + 4])
        off += 4
        qd.append((name, t, c))
    for _ in range(na + nn + nr):
        name, off = _read_name(buf, off)
        t, c, ttl, ln = struct.unpack("!HHIH", buf[off:off + 10])
        off += 10
        rd = buf[off:off + ln]
        if len(rd) != ln:
            raise ValueError("short rdata")
        d = {}
        if t == T_PTR:
            d["target"] = _read_name(buf, off)[0]
        elif t == T_SRV:
            d["priority"], d["weight"], d["port"] = struct.unpack("!HHH", rd[:6])
            d["target"] = _read_name(buf, off + 6)[0]
        elif t == T_A and ln == 4:
            d["addr"] = socket.inet_ntoa(rd)
        elif t == T_TXT:
            i, items = 0, []
            while i < ln:
                k = rd[i]
                items.append(rd[i + 1:i + 1 + k].decode("utf-8", "replace"))
                i += 1 + k
            d["txt"] = items
        off += ln
        rr.append((name, t, c, ttl, d))
    return {"id": pid, "flags": flags, "qd": qd, "rr": rr}


def _rr(name, t, ttl, rdata, flush=False):
    return _labels(name) + struct.pack("!HHIH", t, 0x8001 if flush else 0x0001, ttl, len(rdata)) + rdata


def _txt(items):
    out = b""
    for it in items:
        raw = it.encode("utf-8")[:255]
        out += bytes([len(raw)]) + raw
    return out or b"\0"


def build(pid, flags, questions, answers, additional=()):
    """answers / additional: [(name, type, ttl, rdata bytes, cache-flush)]."""
    out = struct.pack("!HHHHHH", pid, flags, len(questions), len(answers), 0, len(additional))
    for (name, t, c) in questions:
        out += _labels(name) + struct.pack("!HH", t, c)
    for (name, t, ttl, rd, fl) in list(answers) + list(additional):
        out += _rr(name, t, ttl, rd, fl)
    return out


def query(names_types, pid=0, unicast=False):
    return build(pid, 0, [(n, t, 0x8001 if unicast else 0x0001) for (n, t) in names_types], [])


# ---- the advertiser --------------------------------------------------------------------------------------------------------
def _host_label(name):
    s = re.sub(r"[^a-z0-9-]+", "-", str(name or "pxa").lower()).strip("-")[:40]
    return s or "pxa"


class Advertiser(object):
    def __init__(self, port, version="", hostname=None, ifaces=None, environ=None, mport=PORT, group=GROUP):
        self.port = int(port)
        self.version = re.sub(r"[^A-Za-z0-9._+-]", "", str(version or ""))[:40]
        host = hostname or socket.gethostname().split(".", 1)[0]
        self.instance = "PXA Control on %s" % str(host)[:40] + ("" if self.port == DEFAULT_CONTROL_PORT else " (%d)" % self.port)
        self.fqdn_instance = "%s.%s" % (self.instance, SERVICE)
        self.hostname = "pxa-%s.local" % _host_label(host)
        self.ifaces = list(ifaces) if ifaces is not None else interfaces(environ)
        self.mport, self.group = mport, group
        self.sock = None
        self._th = None
        self._stop = threading.Event()
        self._last = {}                         # (iface ip, record name) -> last multicast answer (1 s rule)
        self.answered = 0

    # records ---------------------------------------------------------------------------------------------------------------
    def txt(self):
        return ["v=1", "path=/", "ver=%s" % (self.version or "?"), "auth=token"]

    def records(self, itf, ttl_scale=1):
        """-> dict kind -> [(name, type, ttl, rdata, flush)] for the interface `itf` (its network's addresses only)."""
        host_ttl, other_ttl = TTL_HOST * ttl_scale, TTL_OTHER * ttl_scale
        addrs = [i for (_n, i) in self.ifaces if i.network == itf.network] or [itf]
        srv = struct.pack("!HHH", 0, 0, self.port) + _labels(self.hostname)
        return {
            "meta": [(META, T_PTR, other_ttl, _labels(SERVICE), False)],
            "ptr": [(SERVICE, T_PTR, other_ttl, _labels(self.fqdn_instance), False)],
            "srv": [(self.fqdn_instance, T_SRV, host_ttl, srv, True)],
            "txt": [(self.fqdn_instance, T_TXT, other_ttl, _txt(self.txt()), True)],
            "a": [(self.hostname, T_A, host_ttl, socket.inet_aton(str(i.ip)), True) for i in addrs],
        }

    def answer_for(self, questions, itf):
        """-> (answers, additional) for the questions this Control owns; ([], []) when none is ours."""
        rec = self.records(itf)
        ans, add = [], []
        for (name, t, _c) in questions:
            n = name.lower().rstrip(".")
            if n == META.lower() and t in (T_PTR, T_ANY):
                ans += rec["meta"]
            elif n == SERVICE.lower() and t in (T_PTR, T_ANY):
                ans += rec["ptr"]
                add += rec["srv"] + rec["txt"] + rec["a"]
            elif n == self.fqdn_instance.lower():
                if t in (T_SRV, T_ANY):
                    ans += rec["srv"]
                    add += rec["a"]
                if t in (T_TXT, T_ANY):
                    ans += rec["txt"]
            elif n == self.hostname.lower() and t in (T_A, T_ANY):
                ans += rec["a"]
        seen, add2 = set(), []
        for r in ans:
            seen.add((r[0].lower(), r[1], r[3]))
        for r in add:
            k = (r[0].lower(), r[1], r[3])
            if k not in seen:
                seen.add(k)
                add2.append(r)
        return ans, add2

    def iface_for(self, src_ip):
        try:
            a = ipaddress.ip_address(src_ip)
        except ValueError:
            return None
        return next((i for (_n, i) in self.ifaces if a in i.network), None)

    # socket ----------------------------------------------------------------------------------------------------------------
    def _open(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        s.bind(("", self.mport))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        joined = 0
        for (_n, i) in self.ifaces:
            try:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.group) + socket.inet_aton(str(i.ip)))
                joined += 1
            except OSError:
                pass
        if not joined:
            s.close()
            raise OSError("could not join the mDNS group on any interface")
        return s

    def _send_multicast(self, itf, payload):
        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(str(itf.ip)))
            self.sock.sendto(payload, (self.group, self.mport))
        except OSError:
            pass

    def announce(self, goodbye=False):
        """unsolicited response with every record, on every interface (goodbye: TTL 0)."""
        for (_n, itf) in self.ifaces:
            rec = self.records(itf, ttl_scale=0 if goodbye else 1)
            ans = rec["ptr"] + rec["srv"] + rec["txt"] + rec["a"] + rec["meta"]
            self._send_multicast(itf, build(0, 0x8400, [], ans))

    def handle(self, data, src):
        """one datagram -> sends what it must; returns 'multicast' | 'unicast' | None (for tests)."""
        try:
            msg = parse(data)
        except (ValueError, struct.error, IndexError):
            return None
        if msg["flags"] & 0x8000 or not msg["qd"]:
            return None                         # a response, or nothing asked
        itf = self.iface_for(src[0])
        if itf is None:
            return None                         # not on one of our links
        ans, add = self.answer_for(msg["qd"], itf)
        if not ans:
            return None
        known = {(n.lower(), t, d.get("target") or d.get("addr")) for (n, t, _c, ttl, d) in msg["rr"] if ttl > TTL_OTHER // 2}
        ans = [r for r in ans if (r[0].lower(), r[1], (parse_target(r))) not in known]
        if not ans:
            return None                         # known-answer suppression (RFC 6762 s7.1)
        self.answered += 1
        if src[1] != self.mport:                # legacy unicast: same id, the question echoed, short TTLs, no cache-flush bit
            cap = lambda rs: [(n, t, min(ttl, 10), rd, False) for (n, t, ttl, rd, _f) in rs]   # noqa: E731
            pkt = build(msg["id"], 0x8400, [(q[0], q[1], 1) for q in msg["qd"]], cap(ans), cap(add))
            try:
                self.sock.sendto(pkt, src)
            except OSError:
                pass
            return "unicast"
        if any(q[2] & 0x8000 for q in msg["qd"]):           # QU question: answer straight to the asker
            try:
                self.sock.sendto(build(0, 0x8400, [], ans, add), src)
            except OSError:
                pass
            return "unicast"
        now = time.monotonic()
        key = (str(itf.ip), tuple(sorted({r[0].lower() for r in ans})))
        if now - self._last.get(key, 0.0) < 1.0:
            return None                         # at most once a second per record set and link (RFC 6762 s6)
        self._last[key] = now
        time.sleep(random.uniform(0.02, 0.12))  # shared-record delay (s6)
        self._send_multicast(itf, build(0, 0x8400, [], ans, add))
        return "multicast"

    def start(self):
        if not self.ifaces:
            raise OSError("no home-network interface to advertise on")
        self.sock = self._open()
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True, name="pxa-control-mdns")
        self._th.start()
        return self

    def _loop(self):
        self.announce()
        next_ann = time.monotonic() + 1.0
        anns = 1
        while not self._stop.is_set():
            try:
                r, _w, _x = select.select([self.sock], [], [], 0.5)
            except (OSError, ValueError):
                return
            if r:
                try:
                    data, src = self.sock.recvfrom(9000)
                except OSError:
                    continue
                self.handle(data, src)
            if anns < 2 and time.monotonic() >= next_ann:
                self.announce()
                anns += 1

    def stop(self):
        if self.sock is None:
            return
        self._stop.set()
        try:
            self.announce(goodbye=True)
        except Exception:       # noqa: BLE001
            pass
        if self._th is not None:
            self._th.join(2)
        try:
            self.sock.close()
        except OSError:
            pass
        self.sock = None

    def describe(self):
        return "%s on %s" % (self.instance, ", ".join("%s %s" % (n, i.ip) for (n, i) in self.ifaces))


def parse_target(r):
    """the comparable rdata of one of our records (PTR target name or A address) for known-answer suppression."""
    (_name, t, _ttl, rd, _f) = r
    if t == T_A and len(rd) == 4:
        return socket.inet_ntoa(rd)
    if t == T_PTR:
        try:
            return _read_name(rd, 0)[0]
        except ValueError:
            return None
    return None


def start_for(port, version, lan, environ=None):
    """server.serve()'s hook: -> (Advertiser or None, one status line)."""
    if not lan:
        return None, None
    if not enabled(environ):
        return None, "off (PXA_CONTROL_MDNS=0)"
    try:
        adv = Advertiser(port, version, environ=environ).start()
    except OSError as e:
        return None, "off (%s)" % e
    return adv, "advertising _pxa-control._tcp as '%s'" % adv.describe()


# ---- the browser -----------------------------------------------------------------------------------------------------------
def browse(timeout=1.5, ifaces=None, environ=None, mport=PORT, group=GROUP, service=SERVICE):
    """-> [{"name", "host", "port", "txt", "addrs", "src"}] every responder of `service` heard within `timeout` seconds. One query per
    interface from an ephemeral port (legacy unicast: responders answer this socket directly, so avahi-daemon keeps 5353)."""
    ifaces = list(ifaces) if ifaces is not None else interfaces(environ)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    found = {}
    try:
        s.bind(("", 0))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        pid = random.randint(1, 0xFFFF)
        q = query([(service, T_PTR)], pid=pid)
        for (_n, i) in ifaces or [("default", None)]:
            try:
                if i is not None:
                    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(str(i.ip)))
                s.sendto(q, (group, mport))
            except OSError:
                pass
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                break
            r, _w, _x = select.select([s], [], [], left)
            if not r:
                break
            try:
                data, src = s.recvfrom(9000)
            except OSError:
                break
            try:
                msg = parse(data)
            except (ValueError, struct.error, IndexError):
                continue
            if not msg["flags"] & 0x8000:
                continue
            _collect(msg, src, service, found)
    finally:
        s.close()
    return sorted(found.values(), key=lambda x: (x["host"] or "", x["port"] or 0, x["name"]))


def _collect(msg, src, service, found):
    rr = msg["rr"]
    insts = [d["target"] for (n, t, _c, ttl, d) in rr if t == T_PTR and n.lower() == service.lower() and ttl > 0 and d.get("target")]
    for inst in insts:
        srv = next((d for (n, t, _c, _ttl, d) in rr if t == T_SRV and n.lower() == inst.lower()), None)
        txt = next((d.get("txt") for (n, t, _c, _ttl, d) in rr if t == T_TXT and n.lower() == inst.lower()), None) or []
        addrs = []
        if srv:
            addrs = [d["addr"] for (n, t, _c, _ttl, d) in rr if t == T_A and n.lower() == srv["target"].lower() and d.get("addr")]
        # the address that answered is reachable for sure; prefer it when it is one of the advertised ones (or none came)
        host = src[0] if (src[0] in addrs or not addrs) else addrs[0]
        label = inst[:-(len(service) + 1)] if inst.lower().endswith("." + service.lower()) else inst
        key = (label, srv["port"] if srv else None)
        cur = found.get(key)
        if cur is None:
            found[key] = {"name": label, "host": host, "port": srv["port"] if srv else None, "txt": txt, "addrs": addrs, "src": src[0]}
        elif src[0] in addrs and cur["host"] != src[0] and cur["host"] not in addrs:
            cur["host"] = src[0]
