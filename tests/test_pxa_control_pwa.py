"""PXA Control as an installable app (PWA, the Control app pass): the manifest, the shell service worker, its
registration, the icons and the host picker are served by a real in-process Control, with the right types, on
loopback without a token (public code, nothing of the user's) and only with the token on --lan. No GPU.

    python3 -m pytest -q tests/test_pxa_control_pwa.py
"""
import http.client
import importlib.util
import json
import os
import sys
import tempfile
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TMP = tempfile.mkdtemp(prefix="pxa-control-pwa-")
os.environ["PXA_LAUNCH_FAKE_GPUS"] = "1x600"
os.environ["PXA_CONTROL_CONFIG_DIR"] = os.path.join(TMP, "cfg")
os.environ["PXA_CONTROL_DISCOVER"] = "0"
os.environ["PXA_CONTROL_TELEMETRY"] = "0"
for _k in ("DISPLAY", "WAYLAND_DISPLAY", "PXA_CONTROL", "PXA_CONTROL_TOKEN"):
    os.environ.pop(_k, None)
sys.path.insert(0, TOOLS)
_spec = importlib.util.spec_from_file_location("pxa_launch", os.path.join(TOOLS, "pxa-launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
import pxa_control as C  # noqa: E402

PWA = {"/manifest.webmanifest": "application/manifest+json", "/sw.js": "application/javascript",
       "/pwa.js": "application/javascript", "/hosts.html": "text/html", "/icon-192.png": "image/png",
       "/icon-512.png": "image/png"}


def serve(lan=False):
    app = C.App(L, port=0, lan=lan, token="t0ken-for-the-pwa-test-" + ("lan" if lan else "local"))
    srv = C.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return app, srv, srv.server_address[1]


def get(port, path, token=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    h = {"Host": f"127.0.0.1:{port}"}
    if token:
        h["X-PXA-Token"] = token
    c.request("GET", path, headers=h)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, body, {k.lower(): v for k, v in r.getheaders()}


@pytest.fixture(scope="module")
def local():
    app, srv, port = serve()
    yield app, port
    srv.shutdown()


def test_every_pwa_file_is_served_with_its_type_and_no_token_on_loopback(local):
    _, port = local
    for path, ctype in PWA.items():
        st, body, hd = get(port, path)
        assert st == 200, path
        assert hd["content-type"].startswith(ctype), (path, hd["content-type"])
        assert body, path


def test_manifest_is_installable(local):
    _, port = local
    st, body, _ = get(port, "/manifest.webmanifest")
    m = json.loads(body)
    assert m["name"] == "PXA Control" and m["short_name"]
    assert m["start_url"] == "/" and m["scope"] == "/" and m["display"] == "standalone"
    sizes = {(i["sizes"], i["purpose"]) for i in m["icons"]}
    assert {("192x192", "any"), ("512x512", "any"), ("192x192", "maskable"), ("512x512", "maskable")} <= sizes
    for icon in m["icons"]:                     # every icon the manifest names is served, a PNG of the stated size
        st, png, _ = get(port, icon["src"])
        side = int(icon["sizes"].split("x")[0])
        assert st == 200 and png.startswith(b"\x89PNG")
        assert png[16:24] == side.to_bytes(4, "big") * 2
    for sc in m.get("shortcuts", []):
        assert get(port, sc["url"])[0] == 200


def test_service_worker_keeps_off_the_api_and_falls_back_to_the_host_picker(local):
    _, port = local
    st, sw, _ = get(port, "/sw.js")
    sw = sw.decode()
    assert 'addEventListener("fetch"' in sw and "respondWith" in sw
    assert '"/api/"' in sw and "/metrics" in sw             # never answers the API or the metrics
    assert sw.index('"/api/"') < sw.index("respondWith")
    assert '"/hosts.html"' in sw                            # a page load that cannot reach Control gets the picker
    for p in ("/", "/hosts.html", "/manifest.webmanifest", "/pwa.js", "/icon-192.png", "/icon-512.png"):
        assert f'"{p}"' in sw, p
    st, js, _ = get(port, "/pwa.js")
    assert b'serviceWorker.register("/sw.js"' in js and b"isSecureContext" in js


def test_the_page_links_the_manifest_and_registers_the_worker(local):
    app, port = local
    st, page, hd = get(port, "/", token=app.token)
    assert st == 200
    assert b'<link rel="manifest" href="/manifest.webmanifest" crossorigin="use-credentials">' in page
    assert b'<script src="/pwa.js" defer></script>' in page
    csp = hd["content-security-policy"]
    assert "manifest-src 'self'" in csp and "worker-src 'self'" in csp


def test_host_picker_page_has_a_csp_that_admits_its_own_script(local):
    _, port = local
    st, page, hd = get(port, "/hosts.html")
    assert st == 200 and b"pxa.hosts" in page
    csp = hd["content-security-policy"]
    assert "'sha256-" in csp and "'unsafe-inline'" not in csp.split("script-src", 1)[1].split(";", 1)[0]


def test_on_lan_the_pwa_files_need_the_token_like_everything_else():
    app, srv, port = serve(lan=True)
    try:
        for path in PWA:
            assert get(port, path)[0] == 401, path
            assert get(port, path, token=app.token)[0] == 200, path
    finally:
        srv.shutdown()


def test_host_picker_offline_screen_names_the_rig_and_retries_through_api():
    """the offline screen (PWA fallback, and the Android app's ?offline=1&from=<rig>) names the rig, offers Try again and
    probes the rig through /api/ (never answered by the service worker, so a cached copy cannot fake a live rig);
    the browser behaviour itself is tests/hosts-offline-gui-check.py"""
    page = open(os.path.join(TOOLS, "pxa_control_ui", "hosts.html"), encoding="utf-8").read()
    assert 'qs.offline === "1"' in page and "qs.from" in page and 'id = "again"' in page
    assert 'rig + "/api/health?probe="' in page and 'mode: "no-cors"' in page and 'credentials: "omit"' in page
    assert "Math.min(30, wait * 2)" in page and "document.hidden" in page
    sw = open(os.path.join(TOOLS, "pxa_control_ui", "sw.js"), encoding="utf-8").read()
    assert 'p.indexOf("/api/") === 0' in sw                 # the probe path is never cached


def test_android_wrapper_offline_redirect_and_audio_only_mic():
    """the APK sends a failed rig page load to the picker's offline screen, and grants a rig page the microphone only
    (audio capture, after Android's own prompt), never the camera or anything else, never the bundled picker"""
    a = os.path.join(ROOT, "packaging", "android")
    src = open(os.path.join(a, "src", "net", "pxanetwork", "control", "MainActivity.java"), encoding="utf-8").read()
    man = open(os.path.join(a, "AndroidManifest.xml"), encoding="utf-8").read()
    assert "r.isForMainFrame()" in src and 'HOSTS + "?offline=1&from=" + Uri.encode(o)' in src
    assert src.count("r.grant(") + src.count(".grant(") >= 2
    for g in [ln for ln in src.splitlines() if ".grant(" in ln]:
        assert "new String[] {PermissionRequest.RESOURCE_AUDIO_CAPTURE}" in g, g
    assert "RESOURCE_VIDEO_CAPTURE" not in src and "getResources()" in src
    # security review P10 M1: rigs only (origins the picker opened); the file:// picker, other sites and frames never get the mic
    assert "if (!audio || ro == null || !rigs.contains(ro)) { r.deny(); return; }" in src
    assert "if (cur != null && cur.startsWith(HOSTS)) rigs.add(o);" in src and "if (rigs.contains(o)) return false;" in src
    assert "if (!r.isForMainFrame()) return false;" in src
    assert 'return !u.toString().startsWith("file:///android_asset/");' in src
    assert 'if (sch.equals("http") || sch.equals("https") || sch.equals("file")) return false;' not in src   # no blanket stay-in-app
    assert "if (o == null || !rigs.contains(o)) return;" in src             # the offline screen and the probe: rigs only
    assert 'out.putStringArrayList("pxa.rigs"' in src and 'state.getStringArrayList("pxa.rigs")' in src
    assert 'android.permission.RECORD_AUDIO' in man and 'android.permission.CAMERA' not in man
    # file:// cannot fetch an http rig: the app probes natively and hands the reopen to the picker's own code (tokens)
    assert "s.connect(new InetSocketAddress(u.getHost(), port), 3000)" in src and "startProbe(o, 3000);" in src
    assert 'web.evaluateJavascript("window.pxaRigBack && window.pxaRigBack()", null)' in src
    assert "Math.min(30000, probeWait * 2)" in src and "ui.removeCallbacks(probeTick)" in src
    page = open(os.path.join(TOOLS, "pxa_control_ui", "hosts.html"), encoding="utf-8").read()
    assert "window.pxaRigBack = function ()" in page and 'location.protocol === "file:"' in page
    # security review P10 L1: an unknown ?from= is dropped (no open redirect from the offline screen)
    assert "var dropped = !!(from && from !== location.origin && !list.some(function (x) { return x && x.url === from; }));" in page
    assert "if (dropped) from = null;" in page and "var rig = from || (!dropped && " in page
    assert page.index("if (dropped) from = null;") < page.index("var rig = from ||")
