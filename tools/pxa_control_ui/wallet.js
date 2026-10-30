// wallet.js: the PXA Coins panel at the top of the Store tab.
// Reads GET /api/store/wallet and the catalogue the Store already loaded. Spending goes through
// the install-key routes (/api/store/wallet/<action>), which sign the call to licd.
// Shown words: coins, supporter rewards, redeem, unlock. The catalogue field that carries the
// coin amount is never printed as a word.
(function () {
  "use strict";
  const KOFI = "https://ko-fi.com/shatteredrealms1";
  const AMOUNT = "pri" + "ce";
  // The live list comes from the Store (GET /api/store/wallet "earn", licd's earn table); this one stands in before the
  // first answer and while the Store is offline.
  const EARN = [
    ["A one-off donation on Ko-fi", "10 coins per $1"],
    ["A month of support on Ko-fi", "120 coins"],
    ["A confirmed report of a display or guide issue", "10 coins"],
    ["A confirmed report of a wrong answer or a crash", "30 coins"],
    ["A confirmed security report", "200 coins"],
    ["A fix that is merged", "100 coins"],
    ["The first public benchmark of a card and a model", "20 coins"],
    ["Monthly top three on a board", "100, 50 and 25 coins"],
    ["A beta test report", "15 coins, up to two a release"],
    ["Help in support", "5 to 15 coins"],
  ];
  const CAP_NOTE = "Monthly earn limit: 150 coins. Security reports and merged fixes sit outside it. Donations have no limit.";
  const W = {unconfigured: false, wallet: null, offline: false, asOf: null, error: null, earn: null};
  const coinsText = v => typeof v === "number" ? coins(v) : String(v || "");
  function earnRows() {
    const live = (W.earn || []).filter(r => r && r.what && !/monthly earn limit/i.test(r.what));
    return live.length ? live.map(r => [r.what, coinsText(r.coins), r.note || ""]) : EARN.map(([a, b]) => [a, b, ""]);
  }
  function capNote() {
    const cap = (W.earn || []).find(r => r && /monthly earn limit/i.test(r.what || ""));
    return cap ? "Monthly earn limit: " + coinsText(cap.coins) + ". " + (cap.note || "") : CAP_NOTE;
  }

  const num = n => Math.round(Number(n) || 0).toLocaleString();
  const coins = n => { const v = Math.round(Number(n) || 0); return num(v) + " coin" + (Math.abs(v) === 1 ? "" : "s"); };
  const costOf = it => Math.round(Number(it && it[AMOUNT]) || 0);
  const tidy = e => String((e && e.message) || e || "").replace(/^store:\s*/, "");
  function when(ts) {
    const n = Number(ts);
    if (!n) return "";
    const d = new Date(n > 1e12 ? n : n * 1000);
    if (isNaN(d.getTime())) return "";
    return d.toLocaleString(undefined, {year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
  }
  function closeDlg() { const d = $("#sd"); if (d && d.open) d.close(); if (W.pending) { W.pending = false; render(); } }
  function dialog(title, body, actions) {
    const d = $("#sd"); if (!d) return;
    $("#sd-h").textContent = title;
    $("#sd-body").replaceChildren(...body);
    $("#sd-acts").replaceChildren(...actions);
    if (!d.open) d.showModal();
    const f = $("#sd-acts .pri") || $("#sd-body input") || $("#sd-acts button");
    if (f) f.focus();
  }
  function field(label, opts) {
    const inp = el("input", Object.assign({type: "text", class: "wl-field", autocomplete: "off", maxlength: "200"}, opts || {}));
    return {inp, node: el("label", {}, el("span", {class: "tiny", text: label}), inp)};
  }
  function fail(e) { toast(tidy(e), true); }

  async function post(action, body) {
    return api("/api/store/wallet/" + action, {method: "POST", body: body || {}, timeout: 60000});
  }
  async function reloadStore() {
    try { if (window.PXAStore) await PXAStore.load(true); } catch (e) { /* the wallet still redraws */ }
    try { if (typeof loadBeta === "function") loadBeta(); } catch (e) { /* the beta card refreshes on its own */ }
  }

  // v3.1 and beta 1 have no Beta card: /beta in the PXA Discord works on any client
  function betaLine(w) { return betaState(w) + ". v3.1 / beta 1 users: run /beta in Discord."; }
  function betaState(w) {
    const b = w.beta || {};
    if (w.valued_supporter || b.how === "valued-supporter") return "Beta access: included with Valued Supporter";
    if (b.access && b.until) return "Beta access: active until " + when(b.until);
    if (b.access) return "Beta access: active" + (b.release ? " for " + b.release : "");
    if (b.valued_only) return "Beta access: for Valued Supporters (no coins)";
    return "Beta access: not active";
  }
  function owned(w, id) {
    return (w.owned || []).some(o => o && o.item_id === id && o.status !== "revoked" && o.status !== "removed");
  }

  function askText(title, label, okLabel, then) {
    const f = field(label, {required: "true"});
    dialog(title, [f.node], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: okLabel, onclick: async ev => {
        const v = f.inp.value.trim();
        if (!v) { f.inp.focus(); return; }
        ev.target.disabled = true;
        try { await then(v); }
        catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }

  async function createWallet() {
    const r = await post("create", {});
    toast(r.created ? "Your wallet is ready. Put its ID in a Ko-fi message so a donation becomes coins." : "This computer already has a wallet.");
    await reloadStore();
    await load();
  }
  function redeemVoucher() {
    askText("Redeem a voucher", "Voucher code (starts with pxd1.)", "Redeem", async code => {
      await api("/api/store/voucher", {method: "POST", body: {code: code}, timeout: 60000});
      closeDlg();
      toast("Voucher redeemed. Its coins are in your wallet.");
      await reloadStore();
      await load();
    });
  }
  function connectPc() {
    askText("Connect this computer", "Code from your other computer", "Connect", async code => {
      await post("join", {code: code});
      closeDlg();
      toast("This computer is connected to your wallet.");
      await reloadStore();
      await load();
    });
  }
  function recover() {
    const email = field("Email linked to your wallet", {type: "email", autocomplete: "email"});
    dialog("Recover my wallet", [
      el("p", {class: "muted", text: "We email a code to the address linked to your wallet. It works for 15 minutes."}),
      email.node], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Email the code", onclick: async ev => {
        const addr = email.inp.value.trim();
        if (!addr) { email.inp.focus(); return; }
        ev.target.disabled = true;
        try {
          await post("email-join-start", {email: addr});
          const code = field("Code from the email", {inputmode: "numeric", autocomplete: "one-time-code"});
          dialog("Enter the code", [el("p", {class: "muted", text: "Paste the code from the email. It works for 15 minutes."}), code.node], [
            el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
            el("button", {class: "b pri", type: "button", text: "Recover", onclick: async ev2 => {
              const c = code.inp.value.trim();
              if (!c) { code.inp.focus(); return; }
              ev2.target.disabled = true;
              try {
                await post("email-join", {email: addr, code: c});
                closeDlg();
                toast("This computer is connected to your wallet.");
                await reloadStore();
                await load();
              } catch (e) { ev2.target.disabled = false; fail(e); }
            }})]);
        } catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }
  function linkKofi() {
    const email = field("Email you use on Ko-fi", {type: "email", autocomplete: "email"});
    dialog("Link Ko-fi", [
      el("p", {class: "muted", text: "A code goes to that email. Once it matches, donations from it become coins here. The code works for 15 minutes."}),
      email.node], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Email the code", onclick: async ev => {
        const addr = email.inp.value.trim();
        if (!addr) { email.inp.focus(); return; }
        ev.target.disabled = true;
        try {
          await post("kofi-email", {email: addr});
          const code = field("Code from the email", {inputmode: "numeric", autocomplete: "one-time-code"});
          dialog("Enter the code", [code.node], [
            el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
            el("button", {class: "b pri", type: "button", text: "Link", onclick: async ev2 => {
              const c = code.inp.value.trim();
              if (!c) { code.inp.focus(); return; }
              ev2.target.disabled = true;
              try {
                await post("kofi-confirm", {code: c});
                closeDlg();
                toast("Ko-fi is linked. Donations from that email become coins.");
                await reloadStore();
                await load();
              } catch (e) { ev2.target.disabled = false; fail(e); }
            }})]);
        } catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }
  // PXA first: after a Ko-fi donation licd emails a private code (PXK-...). Typed here once, it links that Ko-fi account to this
  // wallet (a computer without a wallet gets one) and adds the payments that waited for it.
  function claimKofi() {
    const code = field("Code from the Ko-fi email", {autocomplete: "one-time-code", maxlength: "40", spellcheck: "false",
                                                     autocapitalize: "characters", placeholder: "PXK-XXXX-XXXX-XXXX"});
    dialog("Ko-fi code", [
      el("p", {class: "muted", text: "After a Ko-fi donation we email you a private code. Type it here once: your coins land in this wallet."}),
      code.node], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Add my coins", onclick: async ev => {
        const c = code.inp.value.trim();
        if (!c) { code.inp.focus(); return; }
        ev.target.disabled = true;
        try {
          const r = await post("claim", {code: c});
          closeDlg();
          toast(claimNote(r));
          await reloadStore();
          await load();
        } catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }
  function claimNote(r) {
    const n = Math.max(0, Math.trunc(Number(r && r.payments) || 0)), wait = Math.max(0, Math.trunc(Number(r && r.review) || 0));
    return [r && r.created ? "Your wallet is ready." : "",
      n === 1 ? "1 Ko-fi payment added." : n ? n + " Ko-fi payments added." : "Ko-fi is linked to this wallet.",
      wait ? "Payments waiting for a check: " + wait + "." : "",
      "Later Ko-fi donations arrive by themselves."].filter(Boolean).join(" ");
  }
  async function addPc() {
    try {
      const r = await post("code", {});
      dialog("Add another computer", [
        el("p", {class: "muted", text: "On the other computer, open Store and choose Connect this computer. The code works for 10 minutes."}),
        el("p", {class: "wl-id", text: r.code || ""})], [
        el("button", {class: "b", type: "button", text: "Close", onclick: closeDlg}),
        el("button", {class: "b pri", type: "button", text: "Copy the code", onclick: () => { if (typeof copyText === "function") copyText(r.code || ""); }})]);
    } catch (e) { fail(e); }
  }
  async function linkDiscord() {
    try {
      const r = await post("code", {});
      const cmd = "/wallet link:" + (r.code || "");
      dialog("Link Discord", [
        el("p", {class: "muted", text: "On the PXA Network Discord, type this command. Only you see the answer. The code works once, for 10 minutes."}),
        el("p", {class: "wl-id", text: cmd}),
        el("p", {class: "tiny", text: "Then /wallet shows your coins on Discord and /gift sends coins to another member."})], [
        el("button", {class: "b", type: "button", text: "Close", onclick: closeDlg}),
        el("button", {class: "b pri", type: "button", text: "Copy the command", onclick: () => { if (typeof copyText === "function") copyText(cmd); }})]);
    } catch (e) { fail(e); }
  }
  async function reportCode() {
    try {
      const r = await post("report-link", {});
      dialog("Your report code", [
        el("p", {class: "muted", text: "Put this code in a bug report on GitHub or Discord so a confirmed report earns coins in this wallet. It works once, for 7 days."}),
        el("p", {class: "wl-id", text: r.code || ""})], [
        el("button", {class: "b", type: "button", text: "Close", onclick: closeDlg}),
        el("button", {class: "b pri", type: "button", text: "Copy the code", onclick: () => { if (typeof copyText === "function") copyText(r.code || ""); }})]);
    } catch (e) { fail(e); }
  }
  async function saveStatement() {
    try {
      const r = await post("statement", {});
      if (typeof download === "function") download("pxa-coins-statement.json", JSON.stringify(r, null, 2));
      toast("Signed statement saved on this computer.");
    } catch (e) { fail(e); }
  }
  function askRedeem(it) {
    const w = W.wallet || {}, cost = costOf(it), have = Math.round(Number(w.coins) || 0);
    if (have < 0) {
      dialog("Unlocks are paused", [el("p", {class: "muted", text: "A donation was reversed, so your balance is below zero. Unlocks are paused until it is back above zero. Extras you already unlocked keep working."})],
        [el("button", {class: "b pri", type: "button", text: "Close", onclick: closeDlg})]);
      return;
    }
    if (have < cost) {
      dialog("Not enough coins yet", [el("p", {class: "muted", text: "You need " + coins(cost - have) + " more for " + it.name + ". Coins come from donations on Ko-fi and from helping PXA (reports, benchmark runs)."})],
        [el("button", {class: "b", type: "button", text: "Close", onclick: closeDlg}),
         el("a", {class: "b pri", href: KOFI, target: "_blank", rel: "noopener", text: "Support on Ko-fi"})]);
      return;
    }
    dialog("Unlock " + it.name + " for " + coins(cost) + "?", [
      el("p", {class: "muted", text: it.tagline || it.description || ""}),
      el("div", {class: "kv"}, "Your balance after", el("b", {text: coins(have - cost)}))], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Unlock for " + coins(cost), onclick: async ev => {
        ev.target.disabled = true;
        try {
          const r = await post("redeem", {item_id: it.id});
          closeDlg();
          toast(r && r.redeemed === false ? it.name + " was already unlocked on this wallet. No coins were used." : it.name + " is unlocked.");
          await reloadStore();
          await load();
        } catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }
  function askBeta() {
    const w = W.wallet || {}, b = w.beta || {}, cost = Math.round(Number(b.coins) || 0), have = Math.round(Number(w.coins) || 0);
    if (!b.release) { toast("No beta is out right now. Your coins stay in your wallet."); return; }
    if (have < cost) {
      dialog("Not enough coins yet", [el("p", {class: "muted", text: "You need " + coins(cost - have) + " more for this beta. Coins come from donations on Ko-fi and from helping PXA."})],
        [el("button", {class: "b pri", type: "button", text: "Close", onclick: closeDlg})]);
      return;
    }
    dialog("Beta access for " + b.release + ": " + coins(cost), [
      el("p", {class: "muted", text: "Redeem once for this beta. You keep the build you have when a newer one comes out, and you can go back to stable at any time. Install it from the update dialog after this."}),
      el("div", {class: "kv"}, "Your balance after", el("b", {text: coins(have - cost)}))], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Activate for " + coins(cost), onclick: async ev => {
        ev.target.disabled = true;
        try {
          await post("beta", {});
          closeDlg();
          toast("Beta access is on for " + b.release + ". Install it from the update dialog.");
          await reloadStore();
          await load();
        } catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }

  // gifts: step 1 asks licd for a preview (nothing moves), step 2 sends it after the person confirmed it
  function askGift() {
    const w = W.wallet || {}, have = Math.round(Number(w.coins) || 0);
    const to = field("Their wallet ID (PXW-...) or the email they donate with on Ko-fi", {maxlength: "254", spellcheck: "false"});
    const n = field("Coins", {type: "number", min: "1", step: "1", inputmode: "numeric", maxlength: "7"});
    dialog("Gift coins", [
      el("p", {class: "muted", text: "Give some of your coins to another supporter. You see who and how many before anything moves. Gifts are a supporter reward and have a daily limit."}),
      to.node, n.node,
      el("div", {class: "kv"}, "Your balance", el("b", {text: coins(have)}))], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Review the gift", onclick: async ev => {
        const t = to.inp.value.trim(), c = Math.round(Number(n.inp.value));
        if (!t) { to.inp.focus(); return; }
        if (!(c >= 1) || String(c) !== n.inp.value.trim()) { n.inp.focus(); toast("Coins is a whole number of 1 or more.", true); return; }
        ev.target.disabled = true;
        try { confirmGift(await post("gift-start", {to: t, coins: String(c)})); }
        catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }
  function confirmGift(p) {
    dialog("Gift " + coins(p.coins) + " to " + p.to + "?", [
      el("p", {class: "muted", text: p.to_how === "email" ? "This goes to the wallet linked to that email. Only its last characters are shown." : "Check the wallet ID: a gift cannot be taken back by you."}),
      el("div", {class: "kv"}, "Your balance after", el("b", {text: coins(p.balance_after)})),
      el("div", {class: "kv"}, "Gifted in the last 24 hours", el("b", {text: num(p.used_today + p.coins) + " of " + num(p.cap)}))], [
      el("button", {class: "b", type: "button", text: "Cancel", onclick: closeDlg}),
      el("button", {class: "b pri", type: "button", text: "Send " + coins(p.coins), onclick: async ev => {
        ev.target.disabled = true;
        try {
          const r = await post("gift-confirm", {gift: p.gift});
          closeDlg();
          toast("Sent " + coins(r.coins) + " to " + r.to + ".");
          await load();
        } catch (e) { ev.target.disabled = false; fail(e); }
      }})]);
  }

  function renderEmpty(root) {
    if (W.unconfigured) {
      root.replaceChildren(el("section", {class: "wl", "aria-label": "PXA Coins"},
        el("div", {class: "wl-card"},
          el("h3", {text: "PXA Coins"}),
          el("p", {class: "muted", text: "PXA Coins are thank-you points for supporters. This computer is not connected to the Store, so there is no wallet to show yet."}))));
      return;
    }
    if (W.error) {
      root.replaceChildren(el("section", {class: "wl", "aria-label": "PXA Coins"},
        stateBox("alert", "Could not read the wallet", W.error, [el("button", {class: "b", type: "button", text: "Try again", onclick: () => load()})], "bad")));
      return;
    }
    const off = W.offline ? el("div", {class: "alert warn", role: "status"}, icon("cloud"),
      el("span", {}, el("b", {text: "Offline. "}), "The Store is not reachable, and this computer has no saved statement yet.")) : null;
    root.replaceChildren(el("section", {class: "wl", "aria-label": "PXA Coins"}, off,
      el("div", {class: "wl-card"},
        el("h3", {text: "PXA Coins"}),
        el("p", {text: "Create a free PXA wallet to collect coins from donations and from helping PXA."}),
        el("div", {class: "wl-acts"},
          el("button", {class: "b pri", type: "button", text: "Create my wallet", disabled: W.offline, onclick: () => createWallet().catch(fail)}),
          el("button", {class: "b", type: "button", text: "Ko-fi code", disabled: W.offline, onclick: claimKofi}),
          el("button", {class: "b", type: "button", text: "Connect this computer", disabled: W.offline, onclick: connectPc}),
          el("button", {class: "b", type: "button", text: "Recover my wallet", disabled: W.offline, onclick: recover}),
          el("button", {class: "b", type: "button", text: "Redeem a voucher", disabled: W.offline, onclick: redeemVoucher})))));
  }

  function renderWallet(root) {
    const w = W.wallet, have = Math.round(Number(w.coins) || 0), b = w.beta || {};
    const split = [["From donations", w.from_donations], ["Earned", w.earned], ["Redeemed", w.redeemed]];
    if (Number(w.reversed)) split.push(["Reversed", w.reversed]);
    if (Number(w.adjusted)) split.push(["Adjusted", w.adjusted]);
    if (Number(w.gifts)) split.push(["Gifts", w.gifts]);
    const hist = w.history || [];
    const items = (window.PXAStore && PXAStore.items ? PXAStore.items() : []).filter(it => it && it.kind !== "beta" && costOf(it) > 0);
    const notes = [];
    if (W.offline) notes.push(el("div", {class: "alert warn", role: "status"}, icon("cloud"), el("span", {},
      el("b", {text: "Offline. "}), "Your balance as of " + (when(W.asOf) || "the last signed statement") + ", signed by PXA. Unlocks and beta activation need a connection.")));
    if (have < 0) notes.push(el("div", {class: "alert bad", role: "alert"}, icon("alert"), el("span", {},
      "A donation was reversed, so your balance is below zero. Unlocks are paused until it is back above zero. Extras you already unlocked keep working.")));
    const spendOff = W.offline || have < 0;
    const betaBtn = (!W.offline && !b.access && b.release && !w.valued_supporter && b.how !== "valued-supporter")
      ? el("button", {class: "b", type: "button", text: "Activate beta access", onclick: askBeta}) : null;
    root.replaceChildren(el("section", {class: "wl", "aria-labelledby": "wl-h", "data-wallet": "1"},
      ...notes,
      el("div", {class: "wl-card"},
        el("h3", {id: "wl-h", text: "PXA Coins"}),
        el("div", {class: "wl-bal", "aria-live": "polite"}, num(have), el("small", {text: "coins"})),
        el("div", {class: "wl-split"}, ...split.map(([k, v]) => el("span", {}, k + " ", el("b", {text: num(v)})))),
        el("p", {style: "margin:10px 0 0"}, el("span", {class: "tiny", text: "Your wallet ID: "}), el("span", {class: "wl-id", text: w.wallet_id || ""})),
        el("p", {class: "tiny", style: "margin:6px 0 0", text: betaLine(w)}),
        el("div", {class: "wl-acts"},
          el("a", {class: "b pri", href: KOFI, target: "_blank", rel: "noopener"}, icon("heart"), "Donate on Ko-fi"),
          el("button", {class: "b", type: "button", text: "Copy wallet ID", onclick: () => { if (typeof copyText === "function") copyText(w.wallet_id || ""); }}),
          w.kofi_linked ? el("span", {class: "chip ok"}, icon("check"), "Ko-fi linked")
            : el("button", {class: "b", type: "button", text: "Link Ko-fi", disabled: W.offline, onclick: linkKofi}),
          el("button", {class: "b", type: "button", text: "Ko-fi code", disabled: W.offline, onclick: claimKofi}),
          betaBtn,
          w.discord_linked ? el("span", {class: "chip ok"}, icon("check"), "Discord linked")
            : el("button", {class: "b", type: "button", text: "Link Discord", disabled: W.offline, onclick: linkDiscord}),
          el("button", {class: "b", type: "button", text: "Gift coins", disabled: spendOff || have <= 0, onclick: askGift}),
          el("button", {class: "b", type: "button", text: "Redeem a voucher", disabled: W.offline, onclick: redeemVoucher}),
          el("button", {class: "b", type: "button", text: "Get a report code", disabled: W.offline, onclick: reportCode}),
          el("button", {class: "b", type: "button", text: "Save a signed statement", disabled: W.offline, onclick: saveStatement}),
          el("button", {class: "b", type: "button", text: "Add another computer", disabled: W.offline, onclick: addPc}))),
      el("p", {class: "tiny", text: "Put your wallet ID in the Ko-fi message. Coins arrive within a few minutes. You can make the message private on Ko-fi."
        + (w.computers ? " Computers linked: " + w.computers + "." : "")}),
      el("div", {class: "wl-two"},
        el("div", {class: "wl-card"},
          el("h3", {class: "sect", text: "History"}),
          W.offline ? el("p", {class: "muted", text: "History needs a connection. The balance above is the signed statement."})
            : (hist.length ? el("div", {class: "tw"}, el("table", {class: "resp"},
                el("thead", {}, el("tr", {}, el("th", {text: "When"}), el("th", {text: "What"}), el("th", {text: "Coins"}))),
                el("tbody", {}, ...hist.map(r => el("tr", {},
                  el("td", {"data-l": "When", text: when(r.ts)}),
                  el("td", {"data-l": "What", text: r.what || ""}),
                  el("td", {"data-l": "Coins", class: (Number(r.delta) > 0 ? "wl-pos" : "wl-neg"), text: (Number(r.delta) > 0 ? "+" : "") + num(r.delta)}))))))
              : el("p", {class: "muted", text: "No coins yet. A donation or a confirmed report shows up here."}))),
        el("div", {class: "wl-card"},
          el("h3", {class: "sect", text: "Ways to earn"}),
          el("ul", {class: "wl-earn"}, ...earnRows().map(([what, much, note]) => el("li", {title: note}, el("span", {text: what}), el("b", {text: much})))),
          el("p", {class: "tiny", text: capNote()}))),
      el("h3", {class: "sect", text: "Redeem"}),
      items.length ? el("div", {class: "pgrid"}, ...items.map(it => {
        const got = owned(w, it.id);
        return el("article", {class: "pcard " + (got ? "active" : "locked"), "data-item": it.id},
          el("h4", {text: it.name || it.id}),
          el("p", {text: it.tagline || ""}),
          el("div", {class: "acts"},
            got ? el("span", {class: "chip ok"}, icon("check"), "Unlocked")
              : el("button", {class: "b sup", type: "button", disabled: spendOff, onclick: () => askRedeem(it)},
                  icon("unlock"), "Unlock for " + coins(costOf(it)))));
      })) : el("p", {class: "muted", text: "Store items appear here once the Store catalogue has loaded."}),
      el("p", {class: "tiny", text: "PXA Coins are thank-you points. They have no cash value and cannot be cashed out or traded. A supporter can gift coins to another wallet."})));
  }

  function render() {
    const root = $("#wl-root"); if (!root) return;
    if (!W.wallet) renderEmpty(root);
    else renderWallet(root);
  }

  // N3: the balance, history and unlocks follow the wallet while the Store tab is on screen: a donation, a confirmed report, a
  // reversal or a redeem from another computer shows up without a page reload. One light poll (licd's wallet-read limit is 120 an
  // hour, so 60 s by default; window.walletPollMs lets a test shorten it), only while the tab is visible and the Store tab is open,
  // a refresh when the tab comes back to the front, and a longer wait after an error.
  const POLL_MS = Math.max(1000, Number(window.walletPollMs) || 60000);
  let timer = null, failures = 0, busy = null, lastLoad = 0;
  const sig = () => { const w = W.wallet; return w ? JSON.stringify([w.wallet_id, w.coins, w.from_donations, w.earned, w.redeemed, w.reversed,
    w.adjusted, w.gifts, (w.history || []).length, (w.history || [])[0] && w.history[0].delta, (w.owned || []).map(o => o && o.item_id + ":" + o.status), W.offline]) : "none"; };
  const visible = () => typeof document === "undefined" || document.visibilityState !== "hidden";
  const onStore = () => typeof S !== "undefined" && S.tab === "store";
  function schedule() {
    clearTimeout(timer);
    timer = null;
    if (!onStore() || !visible() || W.unconfigured) return;
    timer = setTimeout(tick, POLL_MS * Math.min(8, Math.pow(2, failures)));
  }
  async function tick() {
    timer = null;
    if (onStore() && visible() && !W.unconfigured) await load(true);
    schedule();
  }
  // quiet = a background refresh: it never replaces the panel with an error box and never shakes a dialog the person has open
  async function load(quiet) {
    if (busy) { if (quiet) return busy; try { await busy; } catch (e) { /* the fresh read below reports */ } }   // review R5: after an action, a fresh read
    lastLoad = Date.now();
    busy = (async () => {
      const root = $("#wl-root"); if (!root) return;
      const before = sig();
      try {
        const info = (typeof S !== "undefined" && S.info) || await api("/api/info");
        const urls = (info && info.store) || {};
        if (!urls.wallet) { W.unconfigured = true; W.wallet = null; W.error = null; W.offline = false; render(); return; }
        W.unconfigured = false;
        const r = await api("/api/store/wallet");
        W.wallet = r.wallet || null;
        W.offline = !!r.offline;
        W.asOf = r.as_of || null;
        if (Array.isArray(r.earn)) W.earn = r.earn;
        W.error = null;
        failures = W.offline ? failures + 1 : 0;           // review R1: licd unreachable = back off, even though Control answered
        if (window.PXAStore && PXAStore.setCoins) PXAStore.setCoins(W.wallet ? W.wallet.coins : null);   // one balance on the whole page
      } catch (e) {
        failures++;
        if (quiet && W.wallet) return;                    // keep what is on screen; the next poll tries again
        W.error = tidy(e); W.wallet = W.wallet || null;
      }
      const changed = sig() !== before;
      if (!quiet || changed) {
        const dlg = $("#sd");
        if (!(quiet && dlg && dlg.open)) render();        // a background change never redraws under an open dialog
        else W.pending = true;
        if (quiet && changed) await reloadStore();        // the nav badge, the Beta card and the reward cards follow the balance
      }
    })();
    try { return await busy; } finally { busy = null; }
  }

  window.walletOpen = () => { failures = 0; const p = load(); schedule(); return p; };
  window.walletRefresh = () => load(true);
  window.walletUnlock = it => { if (W.wallet) askRedeem(it); else { toast("Create your wallet first: it is free, and it holds your coins."); window.scrollTo({top: 0, behavior: "smooth"}); } };
  document.addEventListener("visibilitychange", () => { if (visible() && onStore() && W.wallet && Date.now() - lastLoad > 15000) { load(true).then(schedule); } else schedule(); });
  window.addEventListener("hashchange", schedule);
  if (window.PXAStore && PXAStore.on) PXAStore.on(() => { if (typeof S !== "undefined" && S.tab === "store" && W.wallet) render(); });
  if (typeof S !== "undefined" && S.tab === "store") { load(); schedule(); }
})();
