"""The host's egress proxy for scoped internet (PLUGINS-PROTOTYPE-PLAN S4; Co-host design s4).

A plugin that asks for `net.internet:<host>` gets ONE door out: an HTTP CONNECT proxy the host runs for that plugin alone.
* Where: POSIX = a unix socket in the host's 0700 run folder (the peer's uid and process group are checked, as on the plugin
  channel); Windows = 127.0.0.1:<ephemeral> with a one-time token (Proxy-Authorization: Bearer). The child gets
  PXA_PLUGIN_EGRESS ("unix:<path>" or "http://127.0.0.1:<port>") and, on Windows, PXA_PLUGIN_EGRESS_TOKEN.
* What passes: `CONNECT <host>:<port>` only (no plain GET: everything is TLS end to end, the proxy never sees content), the host
  one of the plugin's CONSENTED scoped hosts (read again on every CONNECT: a withdrawn consent closes the door at once), the port
  a TLS port (443, 6697), every address the name resolves to public (no private, loopback, link-local: DNS rebinding cannot turn
  an allowed name into a probe of the home network); the address checked is the address connected to.
* Limits: 16 open tunnels per plugin; a stop (grant lost, consent withdrawn, Stop, Control exit) closes the door and every tunnel.
* Audit: a refused CONNECT writes {event: egress_denied, host, port, code} (a host name, never content); counts in status.
* The wall on Linux: a plugin whose only network permissions are scoped hosts (no listen, no net.loopback / net.lan / net.internet;
  net.http is the host's own web fetch and keeps the seal) starts in its own network namespace (util-linux `unshare --net`, user namespace when not root): it has no network
  at all except this proxy's unix socket. Where that is not possible (Windows, macOS, a kernel without user namespaces, or
  PXA_PLUGIN_NETNS=0) the proxy is the door the first-party code uses and status says "proxy" instead of "sealed".
"""
import hmac
import ipaddress
import os
import secrets
import shutil
import socket
import struct
import subprocess
import threading
import time

PORTS = (443, 6697)
MAX_TUNNELS = 16
HEAD_MAX = 8 << 10
HEAD_TIMEOUT_S = 10.0
CONNECT_TIMEOUT_S = 15.0
BUF = 64 << 10
WINDOWS = os.name == "nt"
# net.http is NOT here: net.fetch / net.search run inside the host (host.py HOST_CALLS), so the child gets no network from it
DIRECT_NET = ("net.loopback", "net.lan", "net.internet")


def scoped_hosts(perms):
    from . import scoped_host
    return sorted({h for h in (scoped_host(p) for p in perms or []) if h})


def wants_proxy(m, perms):
    """does this plugin (manifest + consented permissions) get an egress proxy?"""
    return bool(scoped_hosts(perms))


def sealable(m, perms):
    """may it start with no network but the proxy? Only plugins whose whole network reach is scoped hosts."""
    return wants_proxy(m, perms) and not m.get("listen") and not any(p in DIRECT_NET for p in perms or [])


_NETNS = {}


def netns_prefix():
    """the argv prefix that starts a child in a fresh network namespace, or None (probed once)."""
    if "v" in _NETNS:
        return _NETNS["v"]
    v = None
    exe = shutil.which("unshare")
    if exe and not WINDOWS and os.environ.get("PXA_PLUGIN_NETNS", "1") != "0":
        cands = [["--net"]] if os.geteuid() == 0 else []
        cands += [["--user", "--map-current-user", "--net"], ["--user", "--map-root-user", "--net"]]
        for flags in cands:
            try:
                r = subprocess.run([exe] + flags + ["--", "true"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                v = [exe] + flags + ["--"]
                break
    _NETNS["v"] = v
    return v


_NAT64 = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


def _public(infos):
    for _f, _t, _p, _c, sa in infos:
        ip = ipaddress.ip_address(sa[0].split("%", 1)[0])
        if not ip.is_global:
            return False
        if ip.version == 6:
            if any(ip in n for n in _NAT64):                     # 64:ff9b::10.0.0.1 is whatever the NAT64 gateway makes of it
                return False
            inner = [ip.ipv4_mapped, ip.sixtofour, ip.teredo[1] if ip.teredo else None]   # ::ffff:a.b.c.d, 2002:a.b.c.d::, teredo client
            if any(x is not None and not x.is_global for x in inner):
                return False
    return bool(infos)


class EgressProxy(object):
    """one plugin's door out. allowed() -> the hosts it may reach now; audit(event, **kw) -> the host's audit."""

    def __init__(self, pid, allowed, audit=None, run_dir=None, pgid=None, ports=PORTS, resolver=None, public_only=True,
                 max_tunnels=MAX_TUNNELS):
        self.pid, self.allowed, self.audit_fn = pid, allowed, audit
        self.run_dir, self.pgid = run_dir, pgid
        self.ports = tuple(ports)
        self.resolver = resolver or (lambda h, p: socket.getaddrinfo(h, p, type=socket.SOCK_STREAM))
        self.public_only = public_only          # tests point a name at 127.0.0.1 with public_only=False
        self.max_tunnels = max_tunnels
        self.token = None
        self.address = None
        self.sock = None
        self.tunnels = set()
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.stats = {"tunnels": 0, "refused": 0, "up": 0, "down": 0, "open": 0}

    def __repr__(self):
        return "<EgressProxy %s>" % self.pid

    # ---- listen ---------------------------------------------------------------------------------------------------
    def start(self):
        if WINDOWS or not hasattr(socket, "AF_UNIX"):
            self.token = secrets.token_urlsafe(24)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            self.address = "http://127.0.0.1:%d" % s.getsockname()[1]
        else:
            d = self.run_dir
            os.makedirs(d, mode=0o700, exist_ok=True)
            path = os.path.join(d, "%s-egress-%s.sock" % (self.pid, secrets.token_hex(4)))
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(path)
            os.chmod(path, 0o600)
            self.address = "unix:" + path
        s.listen(16)
        s.settimeout(0.5)
        self.sock = s
        threading.Thread(target=self._accept_loop, daemon=True, name="pxa-egress-" + self.pid).start()
        return self

    def env(self):
        e = {"PXA_PLUGIN_EGRESS": self.address}
        if self.token:
            e["PXA_PLUGIN_EGRESS_TOKEN"] = self.token
        return e

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.sock.close()
        except (OSError, AttributeError):
            pass
        if self.address and self.address.startswith("unix:"):
            try:
                os.unlink(self.address[5:])
            except OSError:
                pass
        with self.lock:
            ts = list(self.tunnels)
            self.tunnels.clear()
        for s in ts:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass

    def view(self):
        return dict(self.stats, address_kind="unix" if (self.address or "").startswith("unix:") else "tcp")

    # ---- serve ----------------------------------------------------------------------------------------------------
    def _accept_loop(self):
        while not self.closed.is_set():
            try:
                c, _a = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _peer_ok(self, c):
        if self.token is not None or not hasattr(socket, "SO_PEERCRED"):
            return True                                  # TCP: the token decides
        try:
            ppid, puid, _g = struct.unpack("3i", c.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        except OSError:
            return False
        if puid != os.getuid():
            return False
        if self.pgid is None:
            return False                                 # set right after the child starts; nothing else may knock first
        try:
            return ppid == self.pgid or os.getpgid(ppid) == self.pgid
        except OSError:
            return False

    def _deny(self, c, code, status, host="", port=0):
        self.stats["refused"] += 1
        if self.audit_fn:
            try:
                self.audit_fn("egress_denied", host=host[:100] or None, port=port or None, code=code)
            except Exception:  # noqa: BLE001
                pass
        try:
            c.sendall(("HTTP/1.1 %d %s\r\nContent-Length: 0\r\nConnection: close\r\nX-PXA-Egress: %s\r\n\r\n"
                       % (status, "Forbidden" if status == 403 else "Refused", code)).encode())
        except OSError:
            pass
        try:
            c.close()
        except OSError:
            pass

    def _head(self, c):
        c.settimeout(HEAD_TIMEOUT_S)
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = c.recv(1024)
            if not chunk:
                return None
            buf += chunk
            if len(buf) > HEAD_MAX:
                return None
        head, _sep, rest = buf.partition(b"\r\n\r\n")
        return head.decode("latin-1"), rest

    def _serve(self, c):
        try:
            if not self._peer_ok(c):
                return self._deny(c, "bad_peer", 403)
            got = self._head(c)
            if got is None:
                return self._deny(c, "bad_request", 400)
            head, rest = got
            lines = head.split("\r\n")
            parts = lines[0].split(" ")
            if len(parts) != 3 or parts[0] != "CONNECT" or not parts[2].startswith("HTTP/1."):
                return self._deny(c, "connect_only", 405)
            if self.token is not None:
                ok = False
                for ln in lines[1:]:
                    k, _s, v = ln.partition(":")
                    if (k.strip().lower() == "proxy-authorization" and v.strip()[:7].lower() == "bearer "
                            and hmac.compare_digest(v.strip()[7:].encode("utf-8", "replace"), self.token.encode())):
                        ok = True
                if not ok:
                    return self._deny(c, "bad_token", 407)
            target = parts[1]
            host, _s, port_s = target.rpartition(":")
            host = host.strip().lower().rstrip(".")
            if host.startswith("["):
                return self._deny(c, "not_allowed", 403, host)
            try:
                port = int(port_s)
            except ValueError:
                return self._deny(c, "bad_request", 400, host)
            if host not in set(self.allowed() or ()):
                return self._deny(c, "not_allowed", 403, host, port)
            if port not in self.ports:
                return self._deny(c, "port", 403, host, port)
            with self.lock:
                if len(self.tunnels) >= self.max_tunnels * 2:
                    return self._deny(c, "too_many", 429, host, port)
            try:
                infos = self.resolver(host, port)
            except OSError:
                return self._deny(c, "dns", 502, host, port)
            if self.public_only and not _public(infos):
                return self._deny(c, "private_address", 403, host, port)
            fam, _t, _p, _cn, sa = infos[0]
            up = socket.socket(fam, socket.SOCK_STREAM)
            up.settimeout(CONNECT_TIMEOUT_S)
            try:
                up.connect(sa)
            except OSError:
                up.close()
                return self._deny(c, "connect_failed", 502, host, port)
            up.settimeout(None)
            c.settimeout(None)
            with self.lock:
                if self.closed.is_set():
                    up.close()
                    return self._deny(c, "closed", 503, host, port)
                self.tunnels.update((c, up))
                self.stats["tunnels"] += 1
                self.stats["open"] += 1
            c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            if rest:
                up.sendall(rest)
                self.stats["up"] += len(rest)
            t = threading.Thread(target=self._pipe, args=(up, c, "down"), daemon=True)
            t.start()
            self._pipe(c, up, "up")
            t.join()
            with self.lock:
                self.tunnels.discard(c)
                self.tunnels.discard(up)
                self.stats["open"] -= 1
            for s in (c, up):
                try:
                    s.close()
                except OSError:
                    pass
        except OSError:
            try:
                c.close()
            except OSError:
                pass

    def _pipe(self, src, dst, way):
        try:
            while True:
                b = src.recv(BUF)
                if not b:
                    break
                dst.sendall(b)
                self.stats[way] += len(b)
        except OSError:
            pass
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def open_tunnel(address, host, port, token=None, timeout=20.0):
    """the plugin side: a socket through the host's egress proxy to host:port (TLS is the caller's). Standard library only, so a
    plugin can copy it; raises OSError with the proxy's refusal code."""
    if address.startswith("unix:"):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(address[5:])
    else:
        hp = address.split("://", 1)[-1]
        h, _s, p = hp.rpartition(":")
        s = socket.create_connection((h, int(p)), timeout=timeout)
    req = "CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n" % (host, int(port), host, int(port))
    if token:
        req += "Proxy-Authorization: Bearer %s\r\n" % token
    try:
        s.sendall((req + "\r\n").encode())
    except (BrokenPipeError, ConnectionResetError):
        pass                                    # a proxy that refuses at once (bad_peer) may close first: its answer is still read
    buf = b""
    t0 = time.monotonic()
    while b"\r\n\r\n" not in buf:
        b = s.recv(1)                           # byte by byte: nothing of the tunnel is read past the answer
        if not b or len(buf) > HEAD_MAX or time.monotonic() - t0 > timeout:
            s.close()
            raise OSError("the egress proxy closed")
        buf += b
    head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1")
    first = head.split("\r\n", 1)[0].split(" ")
    if len(first) < 2 or first[1] != "200":
        code = ""
        for ln in head.split("\r\n")[1:]:
            if ln.lower().startswith("x-pxa-egress:"):
                code = ln.split(":", 1)[1].strip()
        s.close()
        raise OSError("egress refused (%s %s)" % (first[1] if len(first) > 1 else "?", code or "refused"))
    return s
