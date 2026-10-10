/* PXA Control service worker. Network first for the app shell, then this cache.
   A request under /api/ is never answered here, so chat streams stay on the network. */
var CACHE = "pxa-shell-v1";
var SHELL = [
  "/", "/favicon.png", "/mark.png", "/agent-manifest.json",
  "/agent-icon-192.png", "/agent-icon-512.png",
  "/encode.css", "/live.css", "/profiles.css", "/agent.css",
  "/vendor/katex/katex.min.css", "/vendor/katex/katex.min.js",
  "/agent-markdown.css", "/agent-attach.css", "/agent-actions.css", "/agent-params.css",
  "/agent-context.css", "/agent-ux.css", "/agent-stream.css", "/agent-memory.css",
  "/chat.css", "/chat-history.css", "/chat-markdown.css", "/chat-actions.css",
  "/chat-stream.css", "/chat-attach.css", "/chat-params.css", "/chat-context.css", "/chat-ux.css",
  "/encode.js", "/live.js", "/profiles.js", "/mqtt.js", "/agent.js",
  "/agent-markdown.js", "/agent-attach.js", "/agent-actions.js", "/agent-params.js",
  "/agent-context.js", "/agent-ux.js", "/agent-stream.js", "/agent-memory.js",
  "/chat.js", "/chat-history.js", "/chat-markdown.js", "/chat-actions.js",
  "/chat-stream.js", "/chat-attach.js", "/chat-params.js", "/chat-context.js", "/chat-ux.js"
];

function shellPath(path) {
  if (path.indexOf("/api/") === 0) return false;
  if (path === "/" || path.indexOf("/vendor/katex/") === 0) return true;
  return SHELL.indexOf(path) !== -1;
}

function put(cache, url, res) {
  // The page is served no-store. A new Response can still go into the Cache API.
  return res.blob().then(function (body) {
    var headers = new Headers(res.headers);
    headers.delete("cache-control");
    headers.delete("pragma");
    return cache.put(url, new Response(body, {status: res.status, statusText: res.statusText, headers: headers}));
  });
}

self.addEventListener("install", function (e) {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then(function (cache) {
    return Promise.all(SHELL.map(function (url) {
      return fetch(url).then(function (res) {
        if (!res || !res.ok) return null;
        return put(cache, url, res);
      }).catch(function () { return null; });
    }));
  }).catch(function () { return null; }));
});

self.addEventListener("activate", function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) {
      return caches.delete(k);
    }));
  }).then(function () { return self.clients.claim(); }));
});

self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (!req || req.method !== "GET") return;
  var url;
  try { url = new URL(req.url); } catch (err) { return; }
  if (url.origin !== self.location.origin) return;
  if (url.pathname.indexOf("/api/") === 0) return;
  if (!shellPath(url.pathname)) return;
  e.respondWith(fetch(req).then(function (res) {
    if (res && res.ok) {
      var copy = res.clone();
      caches.open(CACHE).then(function (cache) { return put(cache, req, copy); }).catch(function () {});
    }
    return res;
  }).catch(function () {
    return caches.match(req).then(function (hit) {
      return hit || caches.match(url.pathname) || caches.match("/");
    }).then(function (hit) {
      return hit || new Response("", {status: 504, statusText: "offline"});
    });
  }));
});
