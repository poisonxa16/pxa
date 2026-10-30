/* PXA Control as an installable app (PWA): register the shell service worker (/sw.js) and remember this Control's
   address for the host picker (/hosts.html). Nothing here talks to /api/. */
(function () {
  "use strict";
  try {
    var KEY = "pxa.hosts", now = Date.now(), list = [];
    try { list = JSON.parse(localStorage.getItem(KEY) || "[]") || []; } catch (e) { list = []; }
    if (!Array.isArray(list)) list = [];
    var here = location.origin, hit = null;
    for (var i = 0; i < list.length; i++) if (list[i] && list[i].url === here) hit = list[i];
    if (!hit) { hit = {url: here, name: location.hostname}; list.unshift(hit); }
    hit.last = now;
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, 20)));
  } catch (e) { /* storage off: the picker still works by typing the address */ }
  if (!("serviceWorker" in navigator) || !window.isSecureContext) return;   // http on a LAN address: no SW (browser rule)
  window.addEventListener("load", function () {
    var p = navigator.serviceWorker.register("/sw.js", {scope: "/"});
    if (p && p.catch) p.catch(function () {});
  });
})();
