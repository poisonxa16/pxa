"""A fake huggingface.co for the browser checks (tests/models-gui-check.py): the listing endpoints only, with a counter of
every request it receives, so a check can prove that a page visit does (or does not) reach "the network". No real
network: PXA Control is pointed at it by setting pxa_hf.HF to its address."""
import hashlib
import http.server
import json
import re
import socketserver
import threading

REPOS = {
    "poisonxa/Pub-Model-PXQ": {"Pub-Model-PXQ4.gguf": b"GGUF" + b"\x11" * 508, "Pub-Model-PXQ4.gguf.expert-counts.csv": b"layer,expert\n0,3\n"},
    "poisonxa/Pub-Dense-PXQ": {"Pub-Dense-PXQ3.gguf": b"GGUF" + b"\x22" * 4096},
    "poisonxa/Supporter-Model-PXQN": {"Supporter-Model-PXQN5.gguf": b"GGUF" + b"\x33" * 2048},
}


class Origin(object):
    def __init__(self):
        self.hits = []                       # request paths, in order
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/_listings":         # the browser check reads the count here (not itself a hit)
                    return self._json({"listings": outer.listings(), "hits": len(outer.hits)})
                outer.hits.append(path)
                if path == "/api/models":
                    return self._json([{"id": r, "private": False, "downloads": 10 + i, "lastModified": "2026-10-%02dT00:00:00Z" % (10 + i)}
                                       for i, r in enumerate(REPOS)])
                if path.startswith("/api/models/") and path.endswith("/tree/main"):
                    files = REPOS.get(path[len("/api/models/"):-len("/tree/main")])
                    if files is None:
                        return self._send(404, b"{}")
                    return self._json([{"type": "file", "path": n, "size": len(d), "lfs": {"oid": hashlib.sha256(d).hexdigest(), "size": len(d)}}
                                       for n, d in files.items()])
                m = re.match(r"^/([^/]+/[^/]+)/resolve/main/(.+)$", path)
                if m:                        # the one-byte probe that marks a repository supporter-only
                    if "Supporter" in m.group(1) and not self.headers.get("Authorization"):
                        return self._send(401, b'{"error":"unauthorized"}')
                    return self._send(200, b"G")
                self._send(404, b"{}")

            def _json(self, obj):
                self._send(200, json.dumps(obj).encode())

            def _send(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.srv = S(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def listings(self):
        """how many times the page's visit made Control ask the origin for the repository list"""
        return sum(1 for p in self.hits if p == "/api/models")

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
