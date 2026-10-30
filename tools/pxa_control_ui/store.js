// store.js: the one adapter between PXA Control's page and the PXA Store (the Plugins and Store sections, the tier
// badges and locks in Models and Encode). Nothing else in the page reads store data.
//
// Source of truth, in order:
//   1. the store backend, when PXA Control announces it: GET /api/info carries {"store": {"catalog": "/api/store/catalog",
//      "entitlements": "/api/store/entitlements", "activate": "/api/store/activate", "plugin": "/api/store/plugin",
//      "consent": "/api/store/consent"}} (Control's backend fills these in; the page never probes a route that is not
//      announced, so an install without a store backend makes no failing requests);
//   2. otherwise the PREVIEW fixture /store-mock.json. In preview, Activate / Turn on / consent change an in-memory copy
//      for this page view only (a reload starts over) and every view says "Preview".
// No payments and no entitlement checks happen here or anywhere in the page: what the server says is drawn as is.
//
// Shapes the page renders (Control's backend posts the authoritative ones):
//   catalog      {version, updated, currency: {name, source, rate_note}, items: [item]}
//   item         {id, kind: "plugin" | "quant-tier" | "preset" | "feature", name, tagline, description, highlights?, price
//                 (credits), duration_days?, grants?: [feature], icon?, hero?, version?, size?,
//                 provenance?: {source: "discord", label, where, votes, url}, perm_version?,
//                 permissions?: [{id, group, label, why?, icon?, new?}]}
//   entitlements {account: {supporter, name, since}, credits: {balance, lifetime, updated, last_topup},
//                 items: {<id>: {state: "owned" | "active", since, expires?, consent?: {version, granted: [perm id]}}},
//                 grants: [feature], receipts: [{id, at, item, credits, what}], sync: {online, checked, grace_days, grace_until?}}
window.PXAStore = (function () {
  "use strict";
  const st = {catalog: null, ent: null, source: null, error: null, loading: null, subs: [], urls: null, coins: null};
  const DAY = 86400000;
  const emit = () => st.subs.slice().forEach(fn => { try { fn(st); } catch (e) { console.error("PXAStore listener failed:", e); } });
  async function getJSON(u) {
    const r = await fetch(u, {cache: "no-store"});
    if (!r.ok) { let j = null; try { j = await r.json(); } catch (e) { /* not JSON */ } throw new Error((j && j.error) || ("HTTP " + r.status)); }
    return r.json();
  }
  async function announced() {
    try { const info = (typeof S !== "undefined" && S.info) || await getJSON("/api/info"); return (info && info.store && info.store.catalog) ? info.store : null; }
    catch (e) { return null; }
  }
  function load(force) {
    if (st.loading && !force) return st.loading;
    st.loading = (async () => {
      st.urls = await announced();
      try {
        if (st.urls) {
          const [c, e] = await Promise.all([getJSON(st.urls.catalog), getJSON(st.urls.entitlements)]);
          st.catalog = c; st.ent = e; st.source = "server";
        } else {
          const f = await getJSON("/store-mock.json");
          st.catalog = f.catalog; st.ent = JSON.parse(JSON.stringify(f.entitlements)); st.source = "preview";
          // ?store=offline|expired shows the offline / grace states of the design (preview only)
          const m = new URLSearchParams(location.search).get("store");
          if (m === "offline" || m === "expired") {
            const g = (st.ent.sync && st.ent.sync.grace_days) || 14, back = m === "offline" ? 3 : g + 2;
            st.ent.sync = Object.assign({}, st.ent.sync, {online: false, checked: new Date(Date.now() - back * DAY).toISOString(),
              grace_until: new Date(Date.now() + (g - back) * DAY).toISOString()});
          }
        }
        st.error = null;
      } catch (e) { st.error = e.message || String(e); }
      emit();
      return st;
    })();
    return st.loading;
  }
  const items = () => (st.catalog && st.catalog.items) || [];
  const item = id => items().find(x => x.id === id) || null;
  // the coins of the PXA Coins wallet (wallet.js sets them) are THE balance whenever this computer holds a wallet: the cards and the nav
  // badge read the same number the panel shows, never the older account snapshot (which does not know the wallet)
  const balance = () => st.coins != null ? st.coins : ((st.ent && st.ent.credits && +st.ent.credits.balance) || 0);
  function setCoins(n) {
    const v = n == null ? null : Math.round(Number(n) || 0);
    if (v === st.coins) return;
    st.coins = v; emit();
  }
  // offline: what was granted keeps working until grace_until; after that it reads as locked until the store answers
  function sync() {
    const s = (st.ent && st.ent.sync) || {online: true};
    const until = s.grace_until ? Date.parse(s.grace_until) : null;
    // never answered and no account linked = nothing unlocked to lose: a fresh install, not an outage (no offline banner)
    if (!s.checked && !(st.ent && st.ent.account)) return {online: true, checked: null, graceUntil: null, expired: false};
    return {online: s.online !== false, checked: s.checked, graceUntil: until, expired: s.online === false && until != null && until < Date.now()};
  }
  // locked | free | owned | active, plus what it would take: the page never shows a control without its reason
  function state(id) {
    const it = item(id), e = st.ent && st.ent.items && st.ent.items[id], y = sync();
    if (!it) return {state: "missing"};
    const exp = e && e.expires ? Date.parse(e.expires) : null;
    if (e && !y.expired && (!exp || exp > Date.now())) return {state: e.state === "active" ? "active" : "owned", since: e.since, expires: e.expires, consent: e.consent || null, grace: !y.online};
    const price = +it.price || 0;
    if (!price) return {state: "free", price: 0};
    const bal = balance();
    return {state: "locked", price, afford: bal >= price, need: Math.max(0, price - bal), lapsed: !!(e && exp && exp <= Date.now()), expired: y.expired};
  }
  function entitled(feature) {
    if (!st.ent || sync().expired) return false;
    if ((st.ent.grants || []).includes(feature)) return true;
    return items().some(it => (it.grants || []).includes(feature) && state(it.id).state === "active");
  }
  async function post(kind, body) {
    if (st.source === "server" && st.urls && st.urls[kind]) {
      const r = await fetch(st.urls[kind], {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      let j = null; try { j = await r.json(); } catch (e) { /* not JSON */ }
      if (!r.ok) throw new Error((j && j.error) || ("HTTP " + r.status));
      await load(true); return j || {ok: true};
    }
    return null;                                       // preview: the caller changes the in-memory copy
  }
  const today = () => new Date().toISOString().slice(0, 10);
  async function activate(id) {
    const it = item(id); if (!it) throw new Error("not in the catalogue: " + id);
    const srv = await post("activate", {item: id}); if (srv) return srv;
    const price = +it.price || 0;
    if (balance() < price) throw new Error("not enough coins: " + it.name + " needs " + price + ", you have " + balance());
    st.ent.credits.balance = balance() - price;
    st.ent.items = st.ent.items || {};
    st.ent.items[id] = {state: it.kind === "plugin" ? "owned" : "active", since: today(),
      expires: it.duration_days ? new Date(Date.now() + it.duration_days * DAY).toISOString().slice(0, 10) : undefined};
    const rc = {id: "preview-" + Date.now(), at: new Date().toISOString(), item: id, credits: -price, what: it.name, preview: true};
    (st.ent.receipts = st.ent.receipts || []).unshift(rc);
    emit(); return {ok: true, receipt: rc, preview: true};
  }
  async function setEnabled(id, on) {
    const srv = await post("plugin", {id, enabled: !!on}); if (srv) return srv;
    const e = st.ent.items && st.ent.items[id]; if (!e) throw new Error("not installed: " + id);
    e.state = on ? "active" : "owned"; emit(); return {ok: true, preview: true};
  }
  async function consent(id, version, granted) {
    const srv = await post("consent", {id, version, granted}); if (srv) return srv;
    const e = st.ent.items && st.ent.items[id]; if (!e) throw new Error("not installed: " + id);
    e.consent = granted ? {version, granted} : null; if (!granted) e.state = "owned"; emit(); return {ok: true, preview: true};
  }
  return {load, item, items, balance, setCoins, state, entitled, sync, activate, setEnabled, consent,
          on(fn) { st.subs.push(fn); }, get source() { return st.source; }, get urls() { return st.urls; }, get hasWallet() { return st.coins != null; }, get error() { return st.error; },
          get ent() { return st.ent; }, get catalog() { return st.catalog; }};
})();
