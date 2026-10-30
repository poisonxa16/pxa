/* PXA Control service worker (the PWA shell). Network first, always: a reachable Control answers every request
   itself. The cache only stands in when the network fails: the page's code and styles come from the last good copy,
   and a page load that cannot reach this Control gets the host picker (/hosts.html) instead of a browser error.
   Never touched: /api/ (chat streams, state, tokens), /metrics, the engine proxy, anything on another origin. */
var CACHE = "pxa-control-shell-v3";
var PRECACHE = [
  "/", "/hosts.html", "/manifest.webmanifest", "/pwa.js", "/icon-192.png", "/icon-512.png", "/favicon.png", "/mark.png",
  "/home.js", "/plugins.js", "/plugin-host.js", "/plugin-host.css", "/store.js", "/wallet.js", "/wallet.css", "/encode.js", "/encode.css", "/live.js", "/live.css",
  "/profiles.js", "/profiles.css", "/mqtt.js", "/chat.js", "/chat.css"
];
var STATIC_EXT = /\.(js|css|png|svg|ico|json|webmanifest|html|woff2?|ttf)$/;

function cacheable(url) {
  if (url.origin !== self.location.origin) return false;
  var p = url.pathname;
  if (p.indexOf("/api/") === 0 || p === "/metrics" || p === "/sw.js") return false;
  return p === "/" || STATIC_EXT.test(p);
}

function put(cache, key, res) {
  // Control serves everything no-store; a copy without that header can still sit in the Cache API.
  return res.blob().then(function (body) {
    var h = new Headers(res.headers);
    h.delete("cache-control");
    h.delete("pragma");
    return cache.put(key, new Response(body, {status: res.status, statusText: res.statusText, headers: h}));
  });
}

self.addEventListener("install", function (e) {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then(function (cache) {
    return Promise.all(PRECACHE.map(function (u) {
      return fetch(u, {credentials: "same-origin", redirect: "follow"}).then(function (res) {
        return res && res.ok && !res.redirected ? put(cache, u, res) : null;
      }).catch(function () { return null; });
    }));
  }).catch(function () { return null; }));
});

self.addEventListener("activate", function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k.indexOf("pxa-") === 0 && k !== CACHE; })
      .map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});

self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (!req || req.method !== "GET") return;
  var url;
  try { url = new URL(req.url); } catch (err) { return; }
  if (!cacheable(url)) return;
  var nav = req.mode === "navigate";
  e.respondWith(fetch(req).then(function (res) {
    if (res && res.ok && !res.redirected && res.type === "basic") {
      var copy = res.clone();
      caches.open(CACHE).then(function (c) { return put(c, url.pathname, copy); }).catch(function () {});
    }
    return res;
  }).catch(function () {
    var key = nav ? "/hosts.html" : url.pathname;
    return caches.match(key).then(function (hit) {
      return hit || new Response(nav ? "PXA Control is not reachable." : "", {
        status: 504, statusText: "offline", headers: {"Content-Type": "text/plain; charset=utf-8"}});
    });
  }));
});
