// plugins.js: the Plugins and Store sections of PXA Control (2026-10 revamp). Data comes only from store.js
// (window.PXAStore); this file draws it. Plugins are first-party, made by the PXA team from Discord requests, so there is
// no publisher UI: each card says where the request came from and the page links to Discord for new requests.
// Consent follows the PXA Anibal plugins flow (anibal/f-control tools/vigil/ui/plugins.js): a plugin says what it does,
// what it can reach and where it can send data before it runs, and asks again when an update wants more.
(function () {
  "use strict";
  const DISCORD = "https://discord.pxanetwork.com", KOFI_URL = "https://ko-fi.com/shatteredrealms1";
  const KIND = {plugin: "Plugin", "quant-tier": "Quant tier", preset: "Preset", feature: "Feature"};
  const P = window.PXAStore;
  let kindFilter = "all";
  const fmtDate = s => { const t = Date.parse(s); return isNaN(t) ? String(s || "") : new Date(t).toLocaleDateString(undefined, {year: "numeric", month: "short", day: "numeric"}); };
  const coinsWord = n => n + " coin" + (Math.abs(n) === 1 ? "" : "s");
  const kofi = (label, cls) => el("a", {class: "b " + (cls || ""), href: KOFI_URL, target: "_blank", rel: "noopener"}, icon("heart"), label || "Support on Ko-fi");

  // ---------------- dialog (one <dialog id="sd"> for activate and consent) ----------------
  function dialog(title, body, actions) {
    const d = $("#sd"); $("#sd-h").textContent = title; $("#sd-body").replaceChildren(...body); $("#sd-acts").replaceChildren(...actions);
    if (!d.open) d.showModal();
    const f = $("#sd-acts .pri") || $("#sd-acts button"); if (f) f.focus();
    return d;
  }
  const closeDlg = () => { const d = $("#sd"); if (d.open) d.close(); };

  function previewNote() {
    return P.source === "preview" ? el("div", {class: "alert info", role: "note"}, icon("spark"),
      el("span", {}, el("b", {text: "Preview. "}), "These plugins, coins and receipts are example data until the PXA Store opens. Nothing here is charged, downloaded or installed, and a reload starts over.")) : null;
  }
  function syncNote() {
    const y = P.sync(); if (y.online) return null;
    if (y.expired) return el("div", {class: "alert bad", role: "alert"}, icon("cloud"), el("span", {},
      el("b", {text: "Offline too long. "}), "The Store has not answered since " + fmtDate(y.checked) + ", so supporter features are paused until it does. Everything free works as usual."),
      el("span", {class: "acts"}, el("button", {class: "b sm", type: "button", text: "Try again", onclick: () => P.load(true)})));
    return el("div", {class: "alert warn", role: "status"}, icon("cloud"), el("span", {},
      el("b", {text: "The Store can't be reached. "}), "What you unlocked keeps working until " + fmtDate(y.graceUntil) + ". Check the internet connection; PXA Control tries again on its own."),
      el("span", {class: "acts"}, el("button", {class: "b sm", type: "button", text: "Try again", onclick: () => P.load(true)})));
  }
  function provenance(it) {
    const p = it.provenance; if (!p) return null;
    return el("span", {class: "chip", title: (p.where ? p.where + " on the PXA Network Discord" : "PXA Network Discord")}, icon("chat"),
      (p.label || "Requested on Discord") + (p.votes ? " · " + p.votes + " votes" : ""));
  }
  function stateChip(s) {
    if (s.state === "active") return el("span", {class: "chip ok"}, icon("check"), s.grace ? "Active (offline grace)" : "Active");
    if (s.state === "owned") return el("span", {class: "chip"}, icon("unlock"), "Unlocked · off");
    if (s.state === "free") return el("span", {class: "chip ok"}, "Free");
    return el("span", {class: "chip sup"}, icon("lock"), s.lapsed ? "Ended · locked" : "Locked");
  }
  const coinLine = it => el("span", {class: "coin"}, +it.price ? coinsWord(+it.price) : "Free",
    it.duration_days ? el("small", {text: " · " + it.duration_days + " days"}) : null);

  // ---------------- activate (credits) ----------------
  function askActivate(it) {
    const s = P.state(it.id);
    if (s.state !== "locked" && s.state !== "free") return;
    if (s.expired) { toast("The Store is offline: activation waits until it answers", true); return; }
    if (s.state === "locked" && !s.afford) {
      dialog("Not enough coins yet", [el("p", {class: "muted", text: it.name + " needs " + coinsWord(s.price) + "; you have " + coinsWord(P.balance()) + ". Coins come from donations on Ko-fi and from helping PXA, and show up here once the Store sees them."})],
        [el("button", {class: "b", type: "button", text: "Close", onclick: closeDlg}), kofi("Support on Ko-fi", "pri")]);
      return;
    }
    const after = P.balance() - (+it.price || 0);
    dialog((+it.price ? "Activate " : "Get ") + it.name + (+it.price ? " for " + coinsWord(+it.price) + "?" : "?"), [
      el("p", {class: "muted", text: it.description || it.tagline || ""}),
      it.highlights ? el("ul", {}, ...it.highlights.map(h => el("li", {text: h}))) : null,
      el("div", {class: "kv"}, "Lasts", el("b", {text: it.duration_days ? it.duration_days + " days from today" : "for good"})),
      +it.price ? el("div", {class: "kv"}, "Your balance after", el("b", {text: coinsWord(after)})) : null,
      it.kind === "plugin" ? el("p", {class: "tiny", text: "It installs switched off. Turning it on shows what it can reach and asks for your OK first."}) : null,
      P.source === "preview" ? el("p", {class: "tiny", text: "Preview: this activation is simulated and resets when you reload."}) : null].filter(Boolean),
      [el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
       el("button", {class: "b pri", type: "button", text: +it.price ? "Activate for " + coinsWord(+it.price) : "Get it", onclick: async ev => {
         ev.target.disabled = true;
         try {
           await P.activate(it.id); closeDlg();
           toast(it.kind === "plugin" ? it.name + " is installed. Turn it on in Plugins." : it.name + " is active" + (it.duration_days ? " for " + it.duration_days + " days" : ""));
           if (it.kind === "plugin" && S.tab !== "plugins") showTab("plugins");
         } catch (e) { ev.target.disabled = false; toast(e.message, true); }
       }})]);
  }
  function actionsFor(it, where) {
    const s = P.state(it.id), out = [];
    if (s.state === "locked") {
      if (s.expired) out.push(el("button", {class: "b sm", type: "button", text: "Try the Store again", onclick: () => P.load(true)}));
      else {
        // a live store with a wallet spends the wallet's coins through the wallet dialog (it checks the balance, a hold and an offline store);
        // the preview fixture keeps its own in-page activation
        const go = P.source === "server" && window.walletUnlock ? () => walletUnlock(it) : () => askActivate(it);
        if (s.afford) out.push(el("button", {class: "b sup", type: "button", onclick: go}, icon("unlock"), (P.source === "server" ? "Unlock for " : "Activate for ") + coinsWord(s.price)));
        else out.push(el("button", {class: "b sup", type: "button", onclick: go}, icon("lock"), "Needs " + coinsWord(s.need) + " more"), kofi(null, "ghost sm"));
      }
    } else if (s.state === "free") out.push(el("button", {class: "b pri", type: "button", text: "Get it free", onclick: () => askActivate(it)}));
    else if (it.kind === "plugin" && where === "store") out.push(el("button", {class: "b", type: "button", text: s.state === "active" ? "Manage in Plugins" : "Turn on in Plugins", onclick: () => { showTab("plugins"); focusItem(it.id); }}));
    else if (it.kind === "quant-tier" && s.state === "active") out.push(el("button", {class: "b", type: "button", text: "Download in Models", onclick: () => showTab("models")}),
      el("button", {class: "b", type: "button", text: "Make one in Encode", onclick: () => showTab("encode")}));
    else if (it.kind === "preset" && s.state === "active") out.push(el("button", {class: "b", type: "button", text: "Use in Launch", onclick: () => showTab("launch")}));
    if (s.state === "active" && s.expires) out.push(el("span", {class: "tiny", text: "until " + fmtDate(s.expires)}));
    return out;
  }

  // ---------------- consent (aligned with the Anibal plugins flow) ----------------
  function needsConsent(it, s) { const c = s.consent; return !!(it.permissions && it.permissions.length) && (!c || (+c.version || 0) < (+it.perm_version || 1)); }
  function permList(it, s) {
    const had = new Set((s.consent && s.consent.granted) || []), groups = {};
    for (const p of it.permissions || []) (groups[p.group || "What it can reach"] = groups[p.group || "What it can reach"] || []).push(p);
    const out = [];
    for (const [g, ps] of Object.entries(groups)) {
      out.push(el("h4", {style: "margin:12px 0 4px;font-size:var(--fs-1)", text: g}));
      out.push(el("ul", {}, ...ps.map(p => { const fresh = s.consent && (p.new || !had.has(p.id));
        return el("li", {style: fresh ? "border-color:var(--warn)" : ""}, icon(p.icon || "shield"), el("div", {}, el("b", {text: p.label + (fresh ? "  (new)" : "")}), p.why ? el("span", {text: p.why}) : null)); })));
    }
    return out;
  }
  function askConsent(it, thenOn) {
    const s = P.state(it.id), again = !!s.consent;
    dialog(again ? it.name + " wants your OK for an update" : "Turn on " + it.name + "?", [el("div", {class: "consent"},
      el("p", {class: "muted", style: "margin:0", text: "Plugins that reach outside PXA ask you first, and ask again when an update wants more." + (again ? " Marked rows are new or changed since you last approved." : "")}),
      ...permList(it, s))],
      [el("button", {class: "b ghost", type: "button", text: "Not now", title: "it stays off and nothing of it runs", onclick: () => { closeDlg(); toast("Not now: " + it.name + " stays off and nothing of it runs"); }}),
       el("button", {class: "b pri", type: "button", text: again ? "Approve the update" : "Approve and turn on", onclick: async ev => {
         ev.target.disabled = true;
         try { await P.consent(it.id, +it.perm_version || 1, (it.permissions || []).map(p => p.id)); if (thenOn) await P.setEnabled(it.id, true); closeDlg(); toast(it.name + " is on"); }
         catch (e) { ev.target.disabled = false; toast(e.message, true); }
       }})]);
  }
  async function toggle(it, on, input) {
    const s = P.state(it.id);
    if (on && needsConsent(it, s)) { input.checked = false; askConsent(it, true); return; }
    try { await P.setEnabled(it.id, on); toast(it.name + (on ? " is on" : " is off: nothing of it runs")); } catch (e) { input.checked = !on; toast(e.message, true); }
  }
  async function install(it, btn) {
    btn.disabled = true; const was = btn.textContent; btn.textContent = "Downloading and checking...";
    try {
      const r = await fetch("/api/plugins/install", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({item_id: it.id})});
      let j = null; try { j = await r.json(); } catch (e) { /* not JSON */ }
      if (!r.ok) throw new Error((j && j.error) || ("HTTP " + r.status));
      toast(it.name + " is installed, switched off. Turn it on in its card.");
      if (window.PXAPlugHost) await PXAPlugHost.load(true);
      renderPlugins();
    } catch (e) { btn.disabled = false; btn.textContent = was; toast(e.message, true); }
  }
  function pluginControls(it) {
    const s = P.state(it.id), out = [];
    if (P.source === "server" && (s.state === "owned" || s.state === "active")) {
      // unlocked on the wallet, not on this computer yet: the signed package comes from the Store (the catalogue lists it)
      if (it.bundle) out.push(el("button", {class: "b pri", type: "button", "data-act": "install", onclick: ev => install(it, ev.currentTarget)}, icon("dl"), "Download & install"));
      else out.push(el("span", {class: "tiny", text: "Unlocked. Its download is not ready yet; it shows up here when it is."}));
      out.push(el("button", {class: "b ghost sm", type: "button", text: "What it can reach", onclick: () => showReach(it)}));
    } else if (s.state === "owned" || s.state === "active") {
      const inp = el("input", {type: "checkbox", role: "switch", "aria-label": "turn " + it.name + " on or off", onchange: ev => toggle(it, ev.target.checked, ev.target)});
      inp.checked = s.state === "active";
      out.push(el("label", {class: "sw"}, inp, s.state === "active" ? "On" : "Off"));
      if (s.state === "active" && needsConsent(it, s)) out.push(el("button", {class: "b sm", type: "button", text: "Update waiting for your OK", onclick: () => askConsent(it, false)}));
      out.push(el("button", {class: "b ghost sm", type: "button", text: "What it can reach", onclick: () => showReach(it)}));
      if (s.consent) out.push(el("button", {class: "b ghost sm", type: "button", text: "Withdraw consent", onclick: async () => {
        try { await P.consent(it.id, 0, null); toast("Consent withdrawn. " + it.name + " is off and asks again before it runs."); } catch (e) { toast(e.message, true); } }}));
    } else {
      out.push(...actionsFor(it, "plugins"));
      out.push(el("button", {class: "b ghost sm", type: "button", text: "What it can reach", onclick: () => showReach(it)}));
    }
    return out;
  }
  function showReach(it) {
    dialog("What " + it.name + " can reach", [el("div", {class: "consent"}, ...permList(it, {consent: null}))],
      [el("button", {class: "b pri", type: "button", text: "Close", onclick: closeDlg})]);
  }

  // ---------------- cards ----------------
  function card(it, where) {
    const s = P.state(it.id);
    return el("article", {class: "pcard " + s.state, "data-item": it.id},
      el("div", {class: "top"}, el("span", {class: "pi" + (s.state === "locked" ? " sup" : "")}, icon(it.icon || (it.kind === "plugin" ? "plug" : "spark"))),
        el("div", {}, el("h4", {text: it.name}), el("div", {class: "by"}, el("span", {class: "kind", text: KIND[it.kind] || it.kind}), it.version ? el("span", {text: "v" + it.version}) : null, it.size ? el("span", {text: it.size}) : null)),
        stateChip(s)),
      el("p", {text: it.tagline || ""}),
      it.description ? el("p", {class: "tiny", text: it.description}) : null,
      el("div", {class: "row"}, coinLine(it), provenance(it)),
      el("div", {class: "acts"}, ...(where === "plugins" && it.kind === "plugin" ? pluginControls(it) : actionsFor(it, where))));
  }
  function hero(it) {
    const s = P.state(it.id);
    return el("section", {class: "phero", "data-item": it.id, "aria-label": it.name},
      el("div", {},
        el("div", {class: "kicker", text: "The first PXA plugin"}),
        el("h3", {}, it.name, " ", stateChip(s)),
        el("p", {text: it.tagline + ". " + (it.description || "")}),
        it.highlights ? el("ul", {}, ...it.highlights.map(h => el("li", {text: h}))) : null,
        el("div", {class: "row", style: "margin-bottom:10px"}, coinLine(it), provenance(it)),
        el("div", {class: "acts row"}, ...pluginControls(it))),
      el("div", {class: "art", "aria-hidden": "true"}, el("div", {class: "orb"}),
        el("div", {class: "wave"}, ...[8, 16, 24, 14, 20, 10, 26, 12, 18, 9].map(v => el("i", {style: "height:" + v + "px"})))));
  }
  function focusItem(id) {
    setTimeout(() => {
      const c = document.querySelector('section:not([hidden]) [data-item="' + id + '"]'); if (!c) return;
      c.scrollIntoView({block: "center", behavior: "smooth"}); c.classList.remove("flash"); void c.offsetWidth; c.classList.add("flash");
      setTimeout(() => c.classList.remove("flash"), 1700);
    }, 60);
  }
  window.storeFocus = focusItem;

  // ---------------- pages ----------------
  function failed(root) {
    root.replaceChildren(stateBox("alert", "Could not read the Store", P.error + ". The rest of PXA Control works without it.",
      [el("button", {class: "b", type: "button", text: "Try again", onclick: () => P.load(true)})], "bad"));
  }
  function renderPlugins() {
    const root = $("#pl-root"); if (!root) return;
    if (!P.catalog) { if (P.error) failed(root); return; }
    const hereAny = () => !!(window.PXAPlugHost && PXAPlugHost.any());
    const here = id => !!(window.PXAPlugHost && PXAPlugHost.has(id));   // installed on this computer: drawn above by plugin-host.js
    const ps = P.items().filter(x => x.kind === "plugin" && !here(x.id)), h = ps.find(x => x.hero) || null;
    const rest = ps.filter(x => x !== h), mine = rest.filter(x => ["owned", "active"].includes(P.state(x.id).state)), avail = rest.filter(x => !mine.includes(x));
    uiKeep(root, () => root.replaceChildren(...[previewNote(), syncNote(), h ? hero(h) : null,
      mine.length || !hereAny() ? el("h3", {class: "sect", text: "Installed"}) : null,
      mine.length ? el("div", {class: "pgrid"}, ...mine.map(x => card(x, "plugins"))) : hereAny() ? null :
        el("p", {class: "muted", text: h && ["owned", "active"].includes(P.state(h.id).state) ? "Nothing else installed yet." : "No plugins installed yet. Get one below; it installs switched off."}),
      el("h3", {class: "sect", text: "Available"}),
      avail.length ? el("div", {class: "pgrid"}, ...avail.map(x => card(x, "plugins"))) : el("p", {class: "muted", text: "You have every plugin there is. More come from Discord requests."}),
      el("div", {class: "alert info", style: "margin-top:var(--s5)"}, icon("chat"), el("span", {}, el("b", {text: "Missing something? "}),
        "Every PXA plugin starts as a request on the PXA Network Discord, and the team builds the ones people ask for most."),
        el("span", {class: "acts"}, el("a", {class: "b sm", href: DISCORD, target: "_blank", rel: "noopener", text: "Request a plugin"})))].filter(Boolean)));
  }
  const rcDelta = r => Math.round(Number(r.credits != null ? r.credits : r.delta) || 0);
  function renderStore() {
    const root = $("#st-root"); if (!root) return;
    if (!P.catalog) { if (P.error) failed(root); return; }
    const e = P.ent || {}, cr = e.credits || {}, kinds = ["all", "quant-tier", "plugin", "preset", "feature"];
    const list = P.items().filter(x => kindFilter === "all" || x.kind === kindFilter)
      .sort((a, b) => (a.kind === "quant-tier" ? -1 : 0) - (b.kind === "quant-tier" ? -1 : 0));
    const rc = (e.receipts || []).slice(0, 12);
    const liveWallet = P.source === "server" && P.urls && P.urls.wallet;
    uiKeep(root, () => root.replaceChildren(...[previewNote(), syncNote(),
      liveWallet ? null : el("div", {class: "credits", id: "st-credits"},
        el("div", {}, el("div", {class: "tiny", style: "text-transform:uppercase;letter-spacing:.08em;font-weight:700", text: "Your coins"}),
          el("div", {class: "bal"}, String(P.balance()), el("small", {text: "coins"}))),
        el("p", {}, "Coins come from donations on Ko-fi and from helping PXA, and show up here once the Store sees them" +
          (cr.last_topup ? " (last: " + coinsWord(cr.last_topup.credits) + " on " + fmtDate(cr.last_topup.at) + ")" : "") + ". They unlock items in the Store; nothing renews by itself."),
        el("div", {class: "acts"}, kofi(null, "pri"), el("button", {class: "b", type: "button", text: "Receipts", onclick: () => $("#st-receipts").scrollIntoView({behavior: "smooth"})}))),
      el("div", {class: "kfilter", role: "group", "aria-label": "show"}, ...kinds.map(k => el("button", {type: "button", "aria-pressed": String(kindFilter === k),
        text: k === "all" ? "Everything" : (KIND[k] || k) + "s", onclick: () => { kindFilter = k; renderStore(); }}))),
      el("div", {class: "pgrid"}, ...list.map(x => card(x, "store"))),
      liveWallet ? null : el("h3", {class: "sect", id: "st-receipts", text: "Receipts"}),
      liveWallet ? null : rc.length ? el("div", {class: "card tw"}, el("table", {}, el("thead", {}, el("tr", {}, el("th", {text: "When"}), el("th", {text: "What"}), el("th", {text: "Coins"}), el("th", {text: "Receipt"}))),
        el("tbody", {}, ...rc.map(r => el("tr", {}, el("td", {text: fmtDate(r.at || r.ts)}), el("td", {text: (r.what || "") + (r.preview ? " (preview)" : "")}),
          el("td", {class: "mono", style: "color:" + (rcDelta(r) > 0 ? "var(--ok)" : "var(--text)"), text: (rcDelta(r) > 0 ? "+" : "") + rcDelta(r)}), el("td", {class: "mono tiny", text: r.id || ""}))))))
        : el("p", {class: "muted", text: "No receipts yet."}),
      el("p", {class: "tiny", style: "margin-top:var(--s4)", text: "PXA Control and the PXQ quants are free for everyone. Supporter coins come from donations and from helping PXA, and unlock the PXQN quants, plugins and extras. Plugins are made by the PXA team from Discord requests."})].filter(Boolean)));
  }
  function navBadge() {
    const b = $("#nav-credits"); if (!b) return;
    const ok = P.catalog && ((P.ent && P.ent.credits) || P.hasWallet);
    b.hidden = !ok; if (ok) { b.textContent = String(P.balance()); b.title = coinsWord(P.balance()); }
  }
  P.on(() => {
    navBadge();
    if (S.tab === "plugins") renderPlugins();
    if (S.tab === "store") renderStore();
    if (S.tab === "models") { if (typeof renderModels === "function") renderModels(); if (typeof renderAvailable === "function") renderAvailable(); }
    if (S.tab === "home" && window.homeRender) homeRender();
  });
  window.pluginsOpen = () => { if (window.pluginHostOpen) pluginHostOpen(); if (!P.catalog && !P.error) P.load(); else renderPlugins(); };
  window.pluginsRedraw = () => { if (S.tab === "plugins") renderPlugins(); };
  window.storeOpen = () => { if (window.walletOpen) walletOpen(); if (!P.catalog && !P.error) P.load(); else renderStore(); };
  // a page left open reads the signed catalogue again once it is 6 h old: at the next quarter-hour tick while it is shown, or the
  // moment a background tab is shown again, so a plugin update reaches the card and the Plugins tab without a reload. A read that
  // fails keeps the old catalogue, so it is tried again at the next tick; the preview's sample Store is never re-read (that would
  // reset its simulated unlocks)
  const CATALOG_EVERY_MS = 6 * 3600 * 1000, CATALOG_CHECK_MS = 15 * 60 * 1000;
  let catalogAt = Date.now(), seenCatalog = null;
  function catalogSeen(s) {
    if (s.catalog && s.catalog !== seenCatalog) { seenCatalog = s.catalog; catalogAt = Date.now(); }
  }
  function catalogIfOld() {
    if (P.source === "preview" || document.visibilityState === "hidden" || Date.now() - catalogAt < CATALOG_EVERY_MS) return;
    catalogAt = Date.now() - CATALOG_EVERY_MS + CATALOG_CHECK_MS;          // a new catalogue sets it to now (catalogSeen)
    P.load(true);
  }
  P.on(catalogSeen);
  P.load();
  setInterval(catalogIfOld, CATALOG_CHECK_MS);
  document.addEventListener("visibilitychange", catalogIfOld);
})();
