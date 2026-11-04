// PXA Control: signs this tab in (todo control-cookie-port-scope).
//
// The session token is never a cookie: browsers send a host's cookies to EVERY port of it, so any other program
// listening on this machine (or on the rig, with --lan) would receive it and could replay it. The token lives in this
// tab's sessionStorage and rides the X-PXA-Token header on every same-origin fetch and EventSource. It arrives once in
// the address, /#token=... (Control's redirect after a ?token= link, a --lan bookmark or the sign-in form; a fragment
// is never sent to a server), and is dropped from the address bar at once. A tab without it goes to /login.
// Navigations that cannot carry a header (a frame, a <script src>, a download link, a page in a new tab) use
// PXA_AUTH.url(path): the same path with a single-use ticket for that one GET.
// Loaded first and synchronously in <head>, before every other script; no inline script (the CSP).
(function () {
  "use strict";
  var KEY = "pxa_control_token", HDR = "X-PXA-Token", LOGIN = "/login";
  // public shells Control serves under /api/ (the Vigil page and HUD and their code): they load this file and sign
  // in from the tab, so they open without a ticket (a reload or a PWA launch works the same way)
  var PUBLIC = /^\/api\/chat\/vigil\/(page|hud|ui\/[^\/]+)$/;
  var tok = null;

  function save(t) {
    tok = t || null;
    try { if (tok) sessionStorage.setItem(KEY, tok); else sessionStorage.removeItem(KEY); }
    catch (e) { /* storage blocked: the token lives for this page only */ }
  }
  try { tok = sessionStorage.getItem(KEY); } catch (e) { tok = null; }

  // #token=... -> this tab, then out of the address bar (and of the history entry)
  var hash = location.hash || "", m = /^#(?:.*&)?token=([^&]*)/.exec(hash);
  if (m) {
    var t = "";
    try { t = decodeURIComponent(m[1]); } catch (e) { t = ""; }
    if (t) save(t);
    var rest = hash.slice(1).split("&").filter(function (kv) { return kv && kv.split("=")[0] !== "token"; }).join("&");
    try { history.replaceState(history.state, "", location.pathname + location.search + (rest ? "#" + rest : "")); }
    catch (e) { location.hash = rest; }
  }

  var nativeFetch = window.fetch ? window.fetch.bind(window) : null;
  var NativeES = window.EventSource;
  function sameOrigin(u) {
    try { return new URL(u, location.href).origin === location.origin; } catch (e) { return false; }
  }

  // No token in this tab: a Control in the no-token test mode answers anyway; any other sends the tab to sign in.
  // Same-origin requests wait for this check, so the page never fires a burst of refused calls first.
  var ready = Promise.resolve();
  if (!tok && nativeFetch) {
    ready = nativeFetch("/api/info", {cache: "no-store", credentials: "omit"}).then(function (r) {
      if (r.status === 401) { location.replace(LOGIN); return new Promise(function () {}); }
    }, function () { /* Control unreachable: the page says so itself */ });
  }

  function withToken(input, init) {
    var o = {}, k;
    init = init || {};
    for (k in init) if (Object.prototype.hasOwnProperty.call(init, k)) o[k] = init[k];
    var h = new Headers(init.headers || (input && typeof input === "object" && input.headers) || undefined);
    if (tok && !h.has(HDR)) h.set(HDR, tok);
    o.headers = h;
    return o;
  }
  if (nativeFetch) {
    window.fetch = function (input, init) {
      var u = input && typeof input === "object" && "url" in input ? input.url : String(input);
      if (!sameOrigin(u)) return nativeFetch(input, init);              // the token never leaves this origin
      return ready.then(function () { return nativeFetch(input, withToken(input, init)); });
    };
  }

  // EventSource cannot send a header: same-origin streams are read with fetch instead. Same events, onmessage /
  // addEventListener / onerror / close(); on an error or the end of the stream it closes (every caller here reopens
  // the stream itself from its last seq, which is what a native reconnect would not do anyway).
  function FetchEventSource(url) {
    var self = this, ls = {}, ac = new AbortController();
    self.url = new URL(url, location.href).href;
    self.readyState = 0;
    self.withCredentials = false;
    self.onopen = self.onmessage = self.onerror = null;
    self.addEventListener = function (type, fn) { (ls[type] = ls[type] || []).push(fn); };
    self.removeEventListener = function (type, fn) { if (ls[type]) ls[type] = ls[type].filter(function (g) { return g !== fn; }); };
    function call(fn, ev) { try { fn.call(self, ev); } catch (e) { setTimeout(function () { throw e; }); } }
    function emit(type, ev) {
      if (type === "message" || type === "open" || type === "error") { var on = self["on" + type]; if (typeof on === "function") call(on, ev); }
      (ls[type] || []).slice().forEach(function (fn) { call(fn, ev); });
    }
    function fail() {
      if (self.readyState === 2) return;
      self.readyState = 2;
      try { ac.abort(); } catch (e) { /* done */ }
      emit("error", new Event("error"));
    }
    self.close = function () { self.readyState = 2; try { ac.abort(); } catch (e) { /* done */ } };
    window.fetch(url, {headers: {Accept: "text/event-stream"}, cache: "no-store", signal: ac.signal}).then(function (r) {
      if (self.readyState === 2) return;
      if (!r.ok || !r.body || !/^text\/event-stream/i.test(r.headers.get("Content-Type") || "")) return fail();
      self.readyState = 1;
      emit("open", new Event("open"));
      var rd = r.body.getReader(), dec = new TextDecoder(), buf = "", data = [], type = "", id = "";
      function line(l) {
        if (l === "") {
          if (data.length) {
            var name = type || "message";
            emit(name, new MessageEvent(name, {data: data.join("\n"), lastEventId: id, origin: location.origin}));
          }
          data = []; type = "";
          return;
        }
        if (l.charAt(0) === ":") return;
        var i = l.indexOf(":"), f = i < 0 ? l : l.slice(0, i), v = i < 0 ? "" : l.slice(i + 1);
        if (v.charAt(0) === " ") v = v.slice(1);
        if (f === "data") data.push(v); else if (f === "event") type = v; else if (f === "id") id = v;
      }
      function pump() {
        return rd.read().then(function (x) {
          if (self.readyState === 2) return;
          if (x.done) return fail();
          buf += dec.decode(x.value, {stream: true});
          var parts = buf.split("\n");
          buf = parts.pop();
          for (var j = 0; j < parts.length && self.readyState !== 2; j++) line(parts[j].replace(/\r$/, ""));
          return pump();
        });
      }
      return pump();
    }).catch(fail);
  }
  if (NativeES) {
    var ES = function (url, init) { return sameOrigin(url) ? new FetchEventSource(url) : new NativeES(url, init); };
    ES.CONNECTING = 0; ES.OPEN = 1; ES.CLOSED = 2;
    ES.prototype = NativeES.prototype;
    window.EventSource = ES;
  }

  // path -> Promise of the same address with a single-use ticket for one GET of exactly that path and query (a frame,
  // a script, a download, a page in a new tab)
  function url(path) {
    var u = new URL(path, location.href);
    if (u.origin !== location.origin) return Promise.resolve(u.href);
    if (PUBLIC.test(u.pathname)) return Promise.resolve(u.pathname + u.search + u.hash);
    return ready.then(function () {
      if (!tok) return u.pathname + u.search + u.hash;                    // the no-token test mode
      return nativeFetch("/api/auth/ticket", {method: "POST", cache: "no-store", credentials: "omit",
        headers: {"Content-Type": "application/json", "X-PXA-Token": tok}, body: JSON.stringify({path: u.pathname, query: u.search.slice(1)})})
        .then(function (r) {
          if (!r.ok) throw new Error(r.status === 401 ? "not signed in: open the address PXA Control printed again" : "HTTP " + r.status);
          return r.json();
        })
        .then(function (j) { if (j && j.ticket) u.searchParams.set("ticket", j.ticket); return u.pathname + u.search + u.hash; });
    });
  }

  // a plain link to the API (a CSV download, the engine's own page in a new tab) cannot carry the header: on the
  // click it gets a ticket and goes there itself
  function apiLink(ev) {
    var a = ev.target && ev.target.closest ? ev.target.closest("a[href]") : null;
    if (!a || a.hasAttribute("data-pxa-ticketed") || ev.defaultPrevented) return;
    var u;
    try { u = new URL(a.getAttribute("href"), location.href); } catch (e) { return; }
    if (u.origin !== location.origin || !(u.pathname.indexOf("/api/") === 0 || u.pathname === "/metrics")) return;
    if (!tok) return;
    ev.preventDefault();
    var dl = a.hasAttribute("download");
    var blank = !dl && (ev.type === "auxclick" || a.target === "_blank" || ev.ctrlKey || ev.metaKey || ev.shiftKey);
    var w = blank ? window.open("", "_blank") : null;               // opened in the click, so no pop-up blocker
    if (w) { try { w.opener = null; } catch (e) { /* fine */ } }
    url(u.pathname + u.search).then(function (v) {
      if (dl) {
        var b = document.createElement("a");
        b.href = v; b.download = a.getAttribute("download") || ""; b.setAttribute("data-pxa-ticketed", "");
        document.body.appendChild(b); b.click(); b.remove();
      } else if (w) w.location.replace(v);
      else if (blank) window.open(v, "_blank", "noopener");
      else location.assign(v);
    }, function () { if (w) w.close(); });
  }
  document.addEventListener("click", function (ev) { if (ev.button === 0) apiLink(ev); }, true);
  document.addEventListener("auxclick", function (ev) { if (ev.button === 1) apiLink(ev); }, true);

  window.PXA_AUTH = {
    header: HDR,
    token: function () { return tok; },
    set: function (t) { save(String(t || "").trim()); },
    clear: function () { save(null); },
    ready: function () { return ready; },
    url: url
  };
})();
