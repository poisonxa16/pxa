// The share page (todo control-share-report): a FREE one-click image of a benchmark run. Reads GET /api/share/runs and
// /api/share/card(.svg); nothing is sent anywhere, the user posts the image themselves. No inline script (the CSP).
"use strict";
(function () {
  const $ = s => document.querySelector(s);
  const msg = (t, err) => { const m = $("#sh-msg"); m.textContent = t || ""; m.classList.toggle("err", !!err); };
  let card = null;

  async function api(path) {
    const r = await fetch(path, {credentials: "same-origin", headers: {Accept: "application/json"}});
    if (r.status === 401) throw new Error("Open this page from PXA Control (Help menu > Share your rig's speed) so it is signed in.");
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error((j && (j.error || j.message)) || ("HTTP " + r.status));
    return j;
  }
  function q() {
    const p = new URLSearchParams();
    const id = $("#sh-run").value;
    if (id) p.set("id", id);
    if ($("#sh-name").checked) p.set("name", "1");
    return p.toString();
  }
  function fmt(v) { return v >= 100 ? v.toFixed(0) : v.toFixed(1); }

  async function show() {
    try {
      card = await api("/api/share/card?" + q());
      const src = "/api/share/card.svg?" + q();
      $("#sh-img").src = src;
      $("#sh-img").alt = card.headline;
      $("#sh-svg").href = src;
      $("#sh-fig").hidden = false;
      $("#sh-actions").hidden = false;
      msg("");
    } catch (e) { msg(e.message, true); }
  }
  function pngBlob() {
    return new Promise((ok, bad) => {
      const img = $("#sh-img");
      const c = document.createElement("canvas");
      c.width = 1200; c.height = 630;
      try { c.getContext("2d").drawImage(img, 0, 0, 1200, 630); } catch (e) { return bad(e); }
      c.toBlob(b => b ? ok(b) : bad(new Error("the browser could not make a PNG; use Save SVG")), "image/png");
    });
  }
  async function savePng() {
    try {
      const b = await pngBlob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(b);
      a.download = "pxa-bench-card-" + (card && card.when ? card.when : "run") + ".png";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      msg("Saved. Post it in the PXA Network Discord or anywhere you like.");
    } catch (e) { msg(e.message, true); }
  }
  async function copyPng() {
    try {
      if (!window.ClipboardItem || !navigator.clipboard || !navigator.clipboard.write) throw new Error("this browser cannot copy images; use Save image");
      await navigator.clipboard.write([new ClipboardItem({"image/png": pngBlob()})]);
      msg("Image copied: paste it into Discord.");
    } catch (e) { msg(e.message, true); }
  }
  async function copyText() {
    try { await navigator.clipboard.writeText(card ? card.text : ""); msg("Text copied."); }
    catch (e) { msg("this browser blocked the clipboard: " + e.message, true); }
  }

  async function init() {
    $("#sh-run").addEventListener("change", show);
    $("#sh-name").addEventListener("change", show);
    $("#sh-png").addEventListener("click", savePng);
    $("#sh-copy").addEventListener("click", copyPng);
    $("#sh-text").addEventListener("click", copyText);
    try {
      const j = await api("/api/share/runs");
      const sel = $("#sh-run");
      sel.textContent = "";
      for (const r of j.runs || []) {
        const o = document.createElement("option");
        o.value = String(r.id);
        o.textContent = r.when + "  " + r.model + (r.quant ? " " + r.quant : "") + "  " + fmt(r.decode_tps) + " t/s";
        sel.appendChild(o);
      }
      if (!(j.runs || []).length) { msg("Run a benchmark first (the Benchmarks tab): the card is made from one of your runs."); return; }
      await show();
    } catch (e) { msg(e.message, true); }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
