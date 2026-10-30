// plugin-host.js: "On this computer" in the Plugins tab (PLUGINS-PROTOTYPE-PLAN S3). The plugins the host (plugins/host.py) actually has:
// name, version, who signed it, what you allowed, where it stands with the unlock gate, and the buttons for each step.
//   Unlock with N coins  -> POST /api/store/wallet/redeem (the signed wallet redeem; Control then keeps the statement, pulls this computer's
//                           grant and installs a supporter plugin's package): the gate (plugins/gate.py) is what lets it run
//   Review permissions   -> the consent card: every permission in plain words, tick what you allow -> POST /api/plugins/host/consent
//   Start / Stop         -> POST /api/plugins/host/start|stop
//   Open page            -> the plugin's own page in <iframe sandbox="allow-scripts"> (opaque origin: no cookie, no token, no Control API)
//                           with a postMessage bridge that only reaches /api/plugins/<that plugin>/* (plugins/ui.py has the contract)
//   Its own tab          -> a plugin whose manifest says ui.tab.nav gets a button in Control's section bar once it is allowed and
//                           unlocked ("plug-<id>"): the same sealed frame, full height, kept open while you visit other sections
//   Speech on this computer -> the engines + voice models the speech plugins use (cards' "speech" block): Set up speech ->
//                           POST /api/plugins/host/speech/setup {languages?}; the 5 s refresh shows the setup's progress
// Words: coins and supporter rewards. Nothing here prices anything.
(function () {
  "use strict";
  const KOFI = "https://ko-fi.com/shatteredrealms1";
  const coinsWord = n => n + " coin" + (Math.abs(n) === 1 ? "" : "s");
  const GATE_WORDS = {active: "Unlocked on this computer", refresh_due: "Unlocked on this computer", missing: "Locked", expired: "Needs one connection to the Store",
    revoked: "No longer unlocked", wrong_machine: "Unlocked for another computer", clock: "This computer's clock is behind", invalid: "Could not be checked"};
  const CONSENT_WORDS = {none: "Not asked yet", granted: "Allowed", partial: "Partly allowed", needs_reconsent: "An update wants more", revoked: "Withdrawn"};
  const RUN_WORDS = {running: "Running", starting: "Starting", stopped: "Stopped", locked: "Locked", failed: "Stopped after repeated failures", stopping: "Stopping", restarting: "Restarting"};
  let data = null, last = "", err = null, timer = null, frame = null, loading = null, loadSeq = 0, doneOk = 0;
  const tabs = {};          // plugin id -> {btn, sec, frame: {id, el} | null}: the plugins that have their own tab in the bar

  const dlg = () => $("#sd");
  function dialog(title, body, actions) {
    const d = dlg(); $("#sd-h").textContent = title; $("#sd-body").replaceChildren(...body); $("#sd-acts").replaceChildren(...actions);
    if (!d.open) d.showModal();
    const f = $("#sd-acts .pri") || $("#sd-acts button"); if (f) f.focus();
    return d;
  }
  const closeDlg = () => { const d = dlg(); if (d.open) d.close(); };
  async function post(path, body) {
    const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
    let j = null; try { j = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok) throw Object.assign(new Error((j && j.error) || ("HTTP " + r.status)), {status: r.status, code: j && j.code});
    return j || {};
  }

  // ---------------- data ----------------
  async function load(force, fresh) {
    if (loading && !force) return loading;
    // reads are numbered and doneOk is the read the cards came from: a late answer older than the cards is dropped, a late OK
    // answer newer than them is drawn even when a still newer read failed, and only such a newer read's failure is an error
    const my = ++loadSeq;
    loading = (async () => {
      try {
        const r = await fetch("/api/plugins/host/cards" + (fresh ? "?fresh=1" : ""), {cache: "no-store"});
        if (!r.ok) throw new Error("HTTP " + r.status);
        const j = await r.json();
        if (my <= doneOk) return;
        doneOk = my; err = null;
        const s = JSON.stringify(j);
        if (s !== last) { last = s; data = j; navUpdates(updatesReady(data.plugins)); render(); }
      } catch (e) { if (my > doneOk) { err = e.message; if (!data) render(); } }
      finally { if (my === loadSeq) loading = null; }
    })();
    return loading;
  }
  const row = id => data && data.plugins.find(p => p.plugin_id === id);
  window.PXAPlugHost = {any: () => !!(data && data.plugins.length), has: id => !!(data && data.plugins.some(p => p.plugin_id === id || (p.entitlement && p.entitlement.item_id === id))),
                        load, open: () => { start(); return load(true, true); }, close: () => stop()};

  // ---------------- pieces ----------------
  function chip(cls, text, ico) { return el("span", {class: "chip " + cls}, ico ? icon(ico) : null, text); }
  function sigChip(p) { return p.signature.ok ? chip("ok", "Signed by PXA", "shield") : chip("bad", "Will not run", "alert"); }
  function consentChip(p) {
    const s = p.consent.status, w = CONSENT_WORDS[s] || s;
    if (s === "granted") return chip("ok", w, "check");
    if (s === "none" || s === "revoked" || s === "needs_reconsent" || s === "partial") return chip(s === "none" ? "" : "warn", w);
    return chip("", w);
  }
  function gateChip(e) {
    const ok = e.state === "active" || e.state === "refresh_due";
    return chip(ok ? "ok" : e.state === "missing" ? "sup" : "bad", GATE_WORDS[e.state] || e.state, ok ? "unlock" : "lock");
  }
  function runChip(p) {
    const s = p.state || "stopped";
    return chip(s === "running" ? "ok" : s === "failed" || s === "locked" ? "bad" : "", RUN_WORDS[s] || s, s === "running" ? "check" : null);
  }
  function kv(k, v) { return el("div", {class: "kv"}, el("span", {text: k}), v); }
  function gateRun(p) { const e = p.entitlement; return !e || (e.active && e.how === "grant"); }
  // the card's "Update to v..." rule: signed ok, a newer version in the Store, and this computer may run it
  function canUpdate(p) { return !!(p.ok && p.store && p.store.update && gateRun(p)); }
  function updatesReady(plugins) {
    return (plugins || []).filter(canUpdate).length;
  }
  // the count on the Plugins tab, so an update is seen without opening the tab
  function navUpdates(n) {
    const b = $("#nav-plug-upd"); if (!b) return;
    b.hidden = !n; b.textContent = n ? String(n) : "";
    b.title = !n ? "" : n === 1 ? "1 plugin update ready" : n + " plugin updates ready";
    const sr = $("#nav-plug-upd-sr"); if (sr) sr.textContent = n ? ", " + b.title : "";      // what a screen reader hears after "Plugins"
  }

  // ---------------- consent card ----------------
  function consentCard(p, thenStart) {
    const again = p.consent.status === "needs_reconsent", groups = {}, ticks = [];
    for (const l of p.lines) (groups[l.group] = groups[l.group] || []).push(l);
    const had = new Set(p.consent.permissions || []);
    const body = [el("div", {class: "consent", "data-consent": p.plugin_id},
      el("p", {class: "muted", style: "margin:0", text: p.name + " is made by PXA. This is everything it asks to do on this computer." +
        (again ? " Rows marked New were not in the version you allowed before." : "") + " Untick anything you do not want to allow."}))];
    for (const [g, ls] of Object.entries(groups)) {
      body[0].append(el("h4", {class: "cg", text: g}));
      body[0].append(el("ul", {}, ...ls.map(l => {
        const li = el("li", {class: l.care ? "care" : "", "data-perm": l.key, "data-reach": l.reach});
        if (l.ask) {
          const cb = el("input", {type: "checkbox", id: "cc-" + l.key.replace(/[^a-z0-9]/gi, "_"), "data-perm": l.key, "aria-label": l.words});
          cb.checked = p.consent.status === "none" || p.consent.status === "revoked" ? true : (had.has(l.key) || l.new === true);
          cb.addEventListener("change", sync); ticks.push(cb);
          li.append(el("label", {class: "pk", for: cb.id}, cb, icon(l.icon)),
            el("div", {}, el("b", {text: cap(l.words)}), el("span", {}, l.new && again ? chip("warn", "New") : null,
              chip(l.reach === "local" ? "" : "warn", l.tag))));
        } else {
          li.append(el("span", {class: "pk"}, icon(l.icon)), el("div", {}, el("b", {text: cap(l.words)}), el("span", {}, "follows from what you allow above", chip(l.reach === "local" ? "" : "warn", l.tag))));
        }
        return li;
      })));
    }
    if (!p.lines.length) body[0].append(el("p", {style: "margin:10px 0 0", text: p.name + " does not ask for any permission."}));
    body[0].append(el("p", {class: "tiny", style: "margin:10px 0 0", text: "Nothing runs until you press the button. You can withdraw this at any time on the plugin's card."}));
    const ok = el("button", {class: "b pri", type: "button", onclick: ev => go(ev.target)});
    function sync() {
      const n = ticks.filter(t => t.checked).length; ok.disabled = ticks.length > 0 && n === 0;
      ok.textContent = ticks.length ? "Allow " + n + " of " + ticks.length + (thenStart ? " and turn on" : "") : (thenStart ? "Allow and turn on" : "Allow");
    }
    async function go(btn) {
      btn.disabled = true;
      try {
        await post("/api/plugins/host/consent", {plugin_id: p.plugin_id, permissions: ticks.filter(t => t.checked).map(t => t.dataset.perm)});
        if (thenStart && gateRun(p)) { try { await post("/api/plugins/host/start", {plugin_id: p.plugin_id}); } catch (e) { toast(e.message, true); } }
        closeDlg(); toast(p.name + (thenStart ? " is on" : ": your choice is saved")); await load(true);
      } catch (e) { btn.disabled = false; toast(e.code === "consent_local_only" ? "Allow it at the computer itself: this page is not on it." : e.message, true); }
    }
    sync();
    dialog(again ? p.name + " wants your OK for an update" : "Allow " + p.name + "?", body,
      [el("button", {class: "b ghost", type: "button", text: "Not now", onclick: () => { closeDlg(); toast(p.name + " stays off and nothing of it runs"); }}), ok]);
  }
  const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
  function reach(p) {
    const body = [el("div", {class: "consent"}, el("ul", {}, ...p.lines.map(l => el("li", {class: l.care ? "care" : ""},
      el("span", {class: "pk"}, icon(l.icon)), el("div", {}, el("b", {text: cap(l.words)}),
        el("span", {text: (p.consent.permissions || []).includes(l.key) ? "allowed" : l.ask ? "not allowed" : "follows from what is allowed"}))))))];
    dialog("What " + p.name + " asks to do", body, [el("button", {class: "b pri", type: "button", text: "Close", onclick: closeDlg})]);
  }

  // ---------------- unlock (the gate) ----------------
  function unlockDlg(p) {
    const e = p.entitlement, have = e.have, cost = e.cost;
    if (have != null && have < 0)
      return dialog("Unlocks are paused", [el("p", {class: "muted", text: "A donation was reversed, so the coins balance is below zero. Unlocks are paused until it is back above zero. Extras you already unlocked keep working."})],
        [el("button", {class: "b pri", type: "button", text: "Close", onclick: closeDlg})]);
    if (have == null)
      return dialog("Make your coins wallet first", [el("p", {class: "muted", text: "Unlocking " + p.name + " spends " + coinsWord(cost) + " from your PXA Coins wallet. The wallet is free and lives in the Store tab; no wallet is on this computer yet."})],
        [el("button", {class: "b", type: "button", text: "Not now", onclick: closeDlg}),
         el("button", {class: "b pri", type: "button", text: "Open the Store", onclick: () => { closeDlg(); showTab("store"); }})]);
    if (have < cost)
      return dialog("Not enough coins yet", [el("p", {class: "muted", text: p.name + " needs " + coinsWord(cost) + "; you have " + coinsWord(have) + ". Coins come from donations on Ko-fi and from helping PXA, and show up here once the Store sees them."})],
        [el("button", {class: "b", type: "button", text: "Close", onclick: closeDlg}),
         el("a", {class: "b pri", href: KOFI, target: "_blank", rel: "noopener", text: "Support on Ko-fi"})]);
    dialog("Unlock " + p.name + " with " + coinsWord(cost) + "?", [
      el("p", {class: "muted", text: "It unlocks for good on this wallet and runs on this computer after your OK on the next card."}),
      el("div", {class: "kv"}, el("span", {text: "Your balance after"}), el("b", {text: coinsWord(have - cost)}))], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Unlock with " + coinsWord(cost), onclick: async ev => {
        ev.target.disabled = true;
        try {
          const r = await post("/api/store/wallet/redeem", {item_id: e.item_id});
          closeDlg();
          toast(r && r.redeemed === false ? p.name + " was already unlocked on this wallet. Nothing was charged." : p.name + " is unlocked.");
          if (window.PXAStore) PXAStore.load(true);
          if (window.walletRefresh) walletRefresh();
          await load(true);
        } catch (er) { ev.target.disabled = false; toast(er.status === 402 ? "Not enough coins: " + er.message : er.message, true); }
      }})]);
  }
  async function syncStore(btn) {
    btn.disabled = true;
    try { await post("/api/store/refresh", {}); toast("Checked with the Store"); await load(true); }
    catch (e) { toast(e.message, true); }
    finally { btn.disabled = false; }
  }

  // ---------------- start, stop, withdraw ----------------
  async function act(path, p, ok, btn) {
    if (btn) btn.disabled = true;
    try { await post(path, {plugin_id: p.plugin_id}); if (ok) toast(ok); await load(true); }
    catch (e) { toast(e.message, true); if (btn) btn.disabled = false; }
  }
  function actions(p) {
    const out = [], e = p.entitlement, cs = p.consent.status;
    if (!p.ok) return out;
    if (e && !(e.active && e.how === "grant")) {
      if (e.action === "unlock") {
        const short = e.have != null && e.have >= 0 && e.have < e.cost;
        out.push(el("button", {class: "b sup", type: "button", "data-act": "unlock", onclick: () => unlockDlg(p)}, icon(short ? "lock" : "unlock"),
          short ? "Needs " + coinsWord(e.cost - e.have) + " more" : "Unlock with " + coinsWord(e.cost)));
      } else if (e.action === "store") out.push(el("button", {class: "b sup", type: "button", "data-act": "store", onclick: () => { showTab("store"); if (window.storeFocus) storeFocus(e.item_id); }}, icon("unlock"), "Unlock in the Store"));
      else out.push(el("button", {class: "b", type: "button", "data-act": "sync", onclick: ev => syncStore(ev.currentTarget)}, "Check with the Store"));
    } else if (cs === "none" || cs === "revoked" || cs === "needs_reconsent" || cs === "partial") {
      out.push(el("button", {class: "b pri", type: "button", "data-act": "consent", onclick: () => consentCard(p, true)}, cs === "needs_reconsent" ? "Review the update" : cs === "partial" ? "Review permissions" : "Review and allow"));
    } else if (p.state === "running" || p.state === "starting") {
      out.push(el("button", {class: "b", type: "button", "data-act": "stop", onclick: ev => act("/api/plugins/host/stop", p, p.name + " is off: nothing of it runs", ev.currentTarget)}, "Turn off"));
      if (p.page) out.push(el("button", {class: "b pri", type: "button", "data-act": "page", onclick: () => openPage(p)}, icon("play"), "Open " + p.page.title));
    } else out.push(el("button", {class: "b pri", type: "button", "data-act": "start", onclick: ev => act("/api/plugins/host/start", p, p.name + " is on", ev.currentTarget)}, "Turn on"));
    const st = p.store || {};
    if (canUpdate(p)) out.push(el("button", {class: "b", type: "button", "data-act": "update", onclick: ev => update(p, ev.currentTarget)}, icon("dl"), "Update to v" + st.latest));
    out.push(el("button", {class: "b ghost sm", type: "button", "data-act": "reach", onclick: () => reach(p)}, "What it can do"));
    if (cs === "granted" || cs === "partial" || cs === "needs_reconsent") out.push(el("button", {class: "b ghost sm", type: "button", "data-act": "withdraw",
      onclick: ev => act("/api/plugins/host/consent/revoke", p, "Consent withdrawn. " + p.name + " is off and asks again before it runs.", ev.currentTarget)}, "Withdraw consent"));
    if (p.keys > 0) out.push(el("button", {class: "b ghost sm", type: "button", "data-act": "forget-keys", title: "Sign-in keys " + p.name + " kept in PXA's key store on this computer",
      onclick: ev => act("/api/plugins/host/keys/forget", p, p.name + "'s saved sign-in keys are gone from this computer.", ev.currentTarget)}, "Forget saved keys"));
    out.push(el("button", {class: "b ghost sm", type: "button", "data-act": "remove", onclick: () => removeDlg(p)}, "Remove"));
    return out;
  }
  async function update(p, btn) {
    btn.disabled = true;
    try { const j = await post("/api/plugins/install", {item_id: (p.store && p.store.item_id) || p.plugin_id}); toast(p.name + " is now v" + ((j.installed && j.installed.version) || "")); await load(true); }
    catch (e) { toast(e.message, true); btn.disabled = false; }
  }
  function removeDlg(p) {
    dialog("Remove " + p.name + "?", [el("p", {class: "muted", text: "It stops and its files leave this computer. Your unlock stays on your wallet, so you can download and install it again any time."})],
      [el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
       el("button", {class: "b pri", type: "button", text: "Remove", onclick: async ev => {
         ev.target.disabled = true;
         try { await post("/api/plugins/uninstall", {plugin_id: p.plugin_id}); closeDlg(); toast(p.name + " is removed from this computer"); last = ""; await load(true); }
         catch (e) { ev.target.disabled = false; toast(e.message, true); }
       }})]);
  }

  // why a stopped plugin is stopped, in words (a stop you asked for needs no note)
  function note(p) {
    if (p.state === "running" || !p.why) return "";
    if (p.why === "grant") return "Stopped because its unlock ended.";
    if (p.why === "consent") return "Stopped because consent was withdrawn.";
    if (p.state === "failed") return "It stopped on its own too many times (" + p.why + "). Turn it on to try again.";
    if (p.state === "restarting") return "It stopped on its own (" + p.why + ") and is starting again.";
    return "";
  }

  // ---------------- speech on this computer (plugins/host.py speech_setup_*) ----------------
  async function speechSetup(btn, body) {
    btn.disabled = true;
    try { await post("/api/plugins/host/speech/setup", body); toast("Speech setup started. It runs in the background."); await load(true); }
    catch (e) { btn.disabled = false; toast(e.code === "setup_local_only" ? "Set up speech at the computer itself: this page is not on it." : e.message, true); }
  }
  function speechPanel(sp) {
    const job = sp.job || {}, running = job.state === "running", mb = sp.download_mb || {}, langs = !!(sp.models && sp.models.languages);
    const fail = !running && (job.state === "failed" || job.state === "stopped") ? job.message : "";
    const btn = (cls, act, text, body) => el("button", {class: "b" + (cls ? " " + cls : ""), type: "button", "data-act": act, onclick: ev => speechSetup(ev.currentTarget, body)}, text);
    let state, text;
    if (running) { state = chip("", "Setting up", "dl"); text = (job.message || "Starting") + ". It runs in the background: you can keep using PXA Control, and speech plugins wait until it is done."; }
    else if (!sp.supported) { state = chip("bad", "Not available", "alert"); text = sp.why || "Speech cannot run on this computer."; }
    else if (sp.ready) { state = chip("ok", "Ready", "check"); text = "The speech engines and English voices are on this computer" + (langs ? ", and the other languages too." : "."); }
    else { state = chip("warn", "Needs a one-time setup"); text = "Speech plugins need the speech engines and voice models on this computer: a one-time download of about " + mb.english + " MB."; }
    const acts = [];
    if (sp.supported && !running) {
      if (!sp.ready) {
        acts.push(btn("pri", "speech-setup", fail ? "Try again" : "Set up speech", fail && job.languages ? {languages: true} : {}));   // a retry is the same kind of run
        acts.push(btn("", "speech-languages", "With other languages (about " + (mb.english + mb.languages) + " MB)", {languages: true}));
      } else {
        if (sp.engines === "update") acts.push(btn("pri", "speech-update", "Update the speech engines", {languages: langs}));
        if (!langs) acts.push(btn("", "speech-languages", "Add other languages (about " + (mb.add_languages != null ? mb.add_languages : mb.languages) + " MB)", {languages: true}));
      }
    }
    const by = (sp.needed_by || []).length ? "Used by " + sp.needed_by.join(", ") : "For the speech plugins";
    return el("section", {class: "phspeech", "data-ph": "speech", "data-state": running ? "running" : !sp.supported ? "unsupported" : sp.ready ? "ready" : "setup"},
      el("div", {class: "top"}, el("span", {class: "pi"}, icon("mic")),
        el("div", {}, el("h4", {text: "Speech on this computer"}), el("div", {class: "by", text: by})), state),
      el("p", {"data-k": "speech-text", text: text}),
      fail ? el("div", {class: "alert warn", role: "status"}, icon("alert"), el("span", {"data-k": "speech-error", text: fail})) : null,
      acts.length ? el("div", {class: "acts"}, ...acts) : null);
  }

  // ---------------- card + list ----------------
  function card(p) {
    const e = p.entitlement;
    const state = !p.ok ? "bad" : p.state === "running" ? "active" : e && !(e.active && e.how === "grant") ? "locked" : "";
    return el("article", {class: "pcard ph " + state, "data-plugin": p.plugin_id, "data-gate": e ? e.state : "free", "data-consent": p.consent.status, "data-state": p.state || "stopped"},
      el("div", {class: "top"}, el("span", {class: "pi" + (e && !(e.active && e.how === "grant") ? " sup" : "")}, icon("plug")),
        el("div", {}, el("h4", {text: p.name}), el("div", {class: "by"}, el("span", {class: "kind", text: "Plugin"}), p.version ? el("span", {text: "v" + p.version}) : null,
          el("span", {text: "made by PXA"}), p.edition === "supporter" ? el("span", {class: "kind", text: "Unlocks with coins"}) : null)),
        p.ok ? runChip(p) : null),
      el("div", {class: "phrows"},
        kv("Signature", el("span", {class: "v", "data-k": "sig"}, sigChip(p), el("small", {text: p.signature.text}))),
        p.ok ? kv("Your OK", el("span", {class: "v", "data-k": "consent"}, consentChip(p), el("small", {text: p.lines.some(l => l.ask) ? (p.consent.permissions || []).length + " of " + p.lines.filter(l => l.ask).length + " allowed" : "asks for nothing"}))) : null,
        e ? kv("Unlock", el("span", {class: "v", "data-k": "gate"}, gateChip(e), el("small", {text: e.state === "missing" ? "Unlocks with PXA coins in the Store." : (e.message || (e.cost ? coinsWord(e.cost) : ""))}))) : (p.ok ? kv("Unlock", el("span", {class: "v", "data-k": "gate"}, chip("ok", "Free"))) : null)),
      note(p) ? el("p", {class: "tiny", "data-k": "note", text: note(p)}) : null,
      el("div", {class: "acts"}, ...actions(p)));
  }
  function render() {
    const root = $("#ph-root"); if (!root) return;
    if (dlg() && dlg().open) { last = ""; navSync(); return; }        // a card is open: draw again after it closes
    const ps = (data && data.plugins) || [];
    const kids = [];
    if (!data && err) kids.push(el("div", {class: "alert warn", role: "status"}, icon("alert"), el("span", {text: "The plugin list could not be read (" + err + ")."})));
    if (ps.length) {
      kids.push(el("h3", {class: "sect", text: "On this computer"}));
      if (data.lan) kids.push(el("p", {class: "tiny", text: "This page is open from another device: permissions can only be given at the computer itself."}));
      kids.push(el("div", {class: "pgrid", "data-ph": "list"}, ...ps.map(card)));
    }
    if (data && data.speech) {
      if (!ps.length) kids.push(el("h3", {class: "sect", text: "On this computer"}));
      kids.push(speechPanel(data.speech));
    }
    root.replaceChildren(...kids);                                   // the open page lives in #ph-page: a redraw never reloads its frame
    navSync();
    if (window.pluginsRedraw) window.pluginsRedraw();                 // the Store's cards skip the plugins that are shown here
  }

  // ---------------- the plugin's page: sandboxed frame + postMessage bridge ----------------
  const themeMsg = () => {
    const cs = getComputedStyle(document.documentElement), vars = {};
    for (const k of ["--bg", "--surface", "--surface-2", "--line", "--text", "--dim", "--accent", "--accent-text", "--ok", "--warn", "--bad", "--font", "--mono"]) vars[k] = cs.getPropertyValue(k).trim();
    return {name: document.documentElement.dataset.theme || "dark", vars};
  };
  function openPage(p) {
    if (tabs[p.plugin_id]) { showTab("plug-" + p.plugin_id); return; }     // it has its own tab: open it there
    closePage();
    const src = "/api/plugins/host/frame?plugin_id=" + encodeURIComponent(p.plugin_id);
    const f = el("iframe", {class: "phframe", title: p.page.title, sandbox: "allow-scripts", src, referrerpolicy: "no-referrer", "data-plugin": p.plugin_id});
    const panel = el("section", {class: "phpanel", "data-ph": "page", "aria-label": p.page.title},
      el("div", {class: "phbar"}, el("b", {text: p.page.title}), el("span", {class: "tiny", text: "runs in a sealed frame: it cannot see this page or your token, and can only talk to " + p.name}),
        el("span", {class: "sp"}), el("button", {class: "b sm", type: "button", text: "Close", onclick: closePage})), f);
    frame = {id: p.plugin_id, el: f, panel};
    $("#ph-page").append(panel);
    panel.scrollIntoView({block: "nearest"});
  }
  function closePage() { if (frame && frame.panel) frame.panel.remove(); frame = null; }
  const b64 = buf => { let s = "", u = new Uint8Array(buf); for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
  function pathOk(id, path) {
    if (typeof path !== "string" || path.length > 2000 || /[\u0000-\u001f\\]/.test(path)) return false;
    let u; try { u = new URL(path, location.origin); } catch (e) { return false; }
    if (u.origin !== location.origin) return false;
    const dec = decodeURIComponent(u.pathname);
    return u.pathname.startsWith("/api/plugins/" + id + "/") && !dec.split("/").includes("..") && !dec.includes("//");
  }
  function frameOf(src) {                                            // the open page in the Plugins tab, or a plugin tab's frame
    if (!src) return null;
    if (frame && src === frame.el.contentWindow) return frame;
    for (const t of Object.values(tabs)) if (t.frame && src === t.frame.el.contentWindow) return t.frame;
    return null;
  }
  window.addEventListener("message", async ev => {
    const frame = frameOf(ev.source);
    if (!frame || ev.source !== frame.el.contentWindow) return;       // only the frame we opened, nothing else on the screen
    const m = ev.data, target = frame.el.contentWindow;
    if (!m || m.pxa !== "plugin/1") return;
    const say = o => { try { target.postMessage(Object.assign({pxa: "plugin/1"}, o), "*"); } catch (e) { /* frame gone */ } };
    if (m.op === "ready") { say({op: "hello", plugin_id: frame.id, theme: themeMsg()}); return; }
    if (m.op === "resize") { if (frame.full) return; const h = Math.max(160, Math.min(2400, Number(m.height) || 0)); if (h) frame.el.style.height = h + "px"; return; }
    if (m.op !== "fetch") return;
    const id = m.id, method = String(m.method || "GET").toUpperCase();
    if (!["GET", "POST", "DELETE"].includes(method)) return say({op: "error", id, error: "method refused"});
    if (!pathOk(frame.id, m.path)) return say({op: "error", id, error: "a plugin page may only call /api/plugins/" + frame.id + "/..."});
    const headers = {}, ct = m.headers && (m.headers["Content-Type"] || m.headers["content-type"]);
    if (ct && /^[\w.+\/-]{3,80}$/.test(ct)) headers["Content-Type"] = ct;
    const body = method === "GET" || m.body == null ? undefined : String(m.body);
    if (body !== undefined && body.length > (1 << 20)) return say({op: "error", id, error: "body too large"});
    try {
      const r = await fetch(m.path, {method, headers, body, credentials: "same-origin", redirect: "error"});
      const ctype = r.headers.get("Content-Type") || "", buf = await r.arrayBuffer();
      if (buf.byteLength > (8 << 20)) return say({op: "error", id, error: "answer too large"});
      const textual = /^(application\/json|text\/)/.test(ctype);
      say(textual ? {op: "result", id, status: r.status, ctype, text: new TextDecoder().decode(buf)} : {op: "result", id, status: r.status, ctype, b64: b64(buf)});
    } catch (e) { say({op: "error", id, error: "the call failed"}); }
  });
  new MutationObserver(() => {
    for (const f of [frame].concat(Object.values(tabs).map(t => t.frame))) if (f && f.el.contentWindow) try { f.el.contentWindow.postMessage({pxa: "plugin/1", op: "theme", plugin_id: f.id, theme: themeMsg()}, "*"); } catch (e) { /* gone */ }
  }).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});

  // ---------------- a plugin's own tab in the section bar ----------------
  // only for a plugin that asks for it (ui.tab.nav), verifies, is allowed (consent) and may run (free, or unlocked): the button and its
  // section are made from the signed manifest's title and one of Control's own icons; the frame is the same sealed frame as above
  const wantsTab = p => p.ok && p.page && p.page.nav === true && (p.consent.status === "granted" || p.consent.status === "partial") && gateRun(p);
  function navSync() {
    const bar = $("#tabs"), before = bar && bar.querySelector('button[data-tab="plugins"]'), main = $("#main");
    if (!bar || !before || !main) return;
    const want = new Map(((data && data.plugins) || []).filter(wantsTab).map(p => [p.plugin_id, p]));
    for (const id of Object.keys(tabs)) if (!want.has(id)) {
      const t = tabs[id]; t.btn.remove(); t.sec.remove(); delete tabs[id];
      if (typeof S !== "undefined" && S.tab === "plug-" + id) showTab("plugins");
    }
    for (const [id, p] of want) {
      let t = tabs[id];
      if (!t) {
        const btn = el("button", {role: "tab", "data-tab": "plug-" + id, "aria-selected": "false", tabindex: "-1", "data-plugin": id},
          icon(p.page.glyph || "plug"), el("span", {class: "nl", text: p.page.title}));
        btn.addEventListener("click", () => showTab("plug-" + id));
        bar.insertBefore(btn, before);
        const sec = el("section", {id: "p-plug-" + id, "data-page": "plug-" + id, class: "ptab", hidden: ""});
        main.insertBefore(sec, main.querySelector(":scope > .foot"));   // before the app footer, which closes every page
        t = tabs[id] = {btn, sec, frame: null, state: null};
      }
      t.p = p;
      if (typeof S !== "undefined" && S.tab === "plug-" + id) drawTab(id);
    }
    if (window.pxaWantTab && tabs[window.pxaWantTab.slice(5)]) { const w = window.pxaWantTab; window.pxaWantTab = null; showTab(w); }
  }
  function drawTab(id) {
    const t = tabs[id]; if (!t) return;
    const p = t.p, run = p.state === "running";
    if (run && t.frame && t.state === "running") return;                 // already showing the page: never reload a working frame
    t.state = p.state || "stopped";
    if (run) {
      const src = "/api/plugins/host/frame?plugin_id=" + encodeURIComponent(id);
      const f = el("iframe", {class: "phframe ptframe", title: p.page.title, sandbox: "allow-scripts", src, referrerpolicy: "no-referrer", "data-plugin": id});
      t.frame = {id, el: f, full: true};
      t.sec.replaceChildren(el("div", {class: "ptbar"}, el("h2", {text: p.page.title}),
        el("span", {class: "tiny", text: "runs in a sealed frame: it cannot see the rest of PXA Control or your token, and only talks to " + p.name}),
        el("span", {class: "sp"}), el("button", {class: "b ghost sm", type: "button", "data-act": "manage", text: "Manage in Plugins", onclick: () => { showTab("plugins"); }})), f);
      return;
    }
    t.frame = null;
    const busy = p.state === "starting" || p.state === "restarting";
    t.sec.replaceChildren(el("div", {class: "ptbar"}, el("h2", {text: p.page.title})),
      el("div", {class: "ptoff"}, el("p", {class: "muted", text: busy ? p.name + " is starting." : p.name + " is off." + (note(p) ? " " + note(p) : "")}),
        busy ? null : el("button", {class: "b pri", type: "button", "data-act": "start", onclick: async ev => {
          ev.currentTarget.disabled = true;
          try { await post("/api/plugins/host/start", {plugin_id: id}); await load(true); }
          catch (e) { toast(e.message, true); ev.target.disabled = false; }
        }}, icon("play"), "Turn on " + p.name),
        el("button", {class: "b ghost", type: "button", text: "Manage in Plugins", onclick: () => showTab("plugins")})));
  }
  window.pluginTabOpen = id => { if (tabs[id]) drawTab(id); start(); load(true); };

  // ---------------- refresh ----------------
  function start() { if (!timer) timer = setInterval(() => { if (typeof S !== "undefined" && (S.tab === "plugins" || S.tab.startsWith("plug-")) && document.visibilityState !== "hidden") load(); }, 5000); }
  function stop() { clearInterval(timer); timer = null; closePage(); }
  window.pluginHostOpen = () => { start(); return load(true, true); };
  window.pluginHostClose = stop;
  load(true);                                                         // once at start: the plugins with their own tab get their button
  let seenCatalog = null;                                             // a fresh signed catalogue (read each time the page opens, and every
  if (window.PXAStore) PXAStore.on(st => {                            // 6 h while it stays open): the cards are asked again, so an update
    if (st.catalog && st.catalog !== seenCatalog) { seenCatalog = st.catalog; load(true); }    // shows on its card and on the tab
  });
  dlg() && dlg().addEventListener("close", () => { if (typeof S !== "undefined" && S.tab === "plugins") { last = ""; load(true); } });
})();
