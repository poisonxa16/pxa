"use strict";
/* PXA Control: the Profiles tab (v3.1). Per-card power limit, clocks and auto-starts, in plain words.
   Loaded by index.html after its own script, so it uses that script's helpers ($, $$, el, api, toast, store, badge, fmtNum,
   fmtAgo, copyText, download, S). Reads GET /api/gpu/state (tools/pxa_ctl via PXA Control) every 3 s while the tab is open.
   Two modes, remembered per browser and in control.json: Simple (presets, one power slider, nothing technical) and Advanced
   (exact numbers, clocks, persistence, UUID groups, auto-start specs, schedules, raw diff, change history, copy as curl).
   Everything that came from the server is put on the page as text; no innerHTML of server data anywhere. */

const PF = {state: null, timer: null, mode: store.get("pf.mode", null), open: false, info: null, draft: null, plan: null,
  sched: null, schedDirty: false, audit: [], auditAt: 0, pw: null, built: false};
const PF_RECO_LO = 0.6;                      // the recommended band: 60 % .. 100 % of a card's stock power limit
const PF_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const pfAdv = () => PF.mode === "advanced";
const pfW = v => v == null ? "?" : Math.round(v) + " W";
const pfShort = n => String(n || "").replace(/^(Tesla |GeForce |NVIDIA )/, "").replace(/-PCIE.*$/, "");

function pfFill(node, ...kids) {             // replaceChildren() would print a null child as the text "null"
  node.replaceChildren(...kids.flat().filter(k => k != null && k !== false));
}
function pfInfo(text) {                      // the (i) with a plain-language tooltip
  return el("span", {class: "pf-i", title: text, tabindex: "0", role: "img", "aria-label": text, text: "i"});
}
function pfCurl(method, path, body) {
  let c = `curl -s${method !== "GET" ? " -X " + method : ""} '${location.origin}${path}'`;
  if (PF.info && PF.info.lan) c += ` -H "X-PXA-Token: $PXA_CONTROL_TOKEN"`;
  if (body !== undefined) c += ` -H 'Content-Type: application/json' -d '${JSON.stringify(body).replace(/'/g, "'\\''")}'`;
  return c;
}
function pfCurlBtn(method, path, bodyFn) {
  return pfAdv() ? el("button", {class: "b", title: "the same action from a terminal", onclick: () => copyText(pfCurl(method, path, bodyFn ? bodyFn() : undefined)), text: "Copy as curl"}) : null;
}
function pfFriendly(e) {                     // what the server said, plus what to do next
  const m = (e && e.message) || String(e);
  if (/switched off/i.test(m)) return m;
  if (/refused: /i.test(m)) return m.replace(/^.*? refused: /, "Not changed: ") + ". It is kept free on purpose; try again when it is released.";
  if (/unreachable|no answer/i.test(m)) return m + ". Check that PXA Control is still running, then press the button again.";
  return m;
}
async function pfCall(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg); return r; }
  catch (e) { toast(pfFriendly(e), true); return null; }
  finally { pfLoad(); }
}

// ---------------- open / close / load ----------------
async function profilesOpen() {
  PF.open = true;
  if (!PF.built) pfBuild();
  if (!PF.info) { try { PF.info = await api("/api/info"); } catch (e) { PF.info = {}; } }
  await pfLoad();
  clearInterval(PF.timer);
  PF.timer = setInterval(() => { if (!document.hidden && PF.open) pfLoad(); }, 3000);
}
function profilesClose() { PF.open = false; clearInterval(PF.timer); PF.timer = null; }
async function pfLoad() {
  try {
    const s = await api("/api/gpu/state", {timeout: 20000});
    PF.state = s;
    if (PF.mode == null) PF.mode = s.settings.ui_mode || "simple";
    pfRender();
  } catch (e) {
    pfFill($("#pf-banner"), el("div", {class: "alert bad"}, badge("bad", "unavailable"),
      el("span", {text: "The Profiles page could not read this machine's cards: " + pfFriendly(e)})));
    pfFill($("#pf-cards"));
  }
}
function pfBuild() {
  PF.built = true;
  const seg = el("span", {class: "seg", id: "pf-mode", "aria-label": "Simple or Advanced"},
    el("button", {"data-v": "simple", title: "presets and one power slider: nothing technical"}, "Simple"),
    el("button", {"data-v": "advanced", title: "exact numbers, clocks, card groups, auto-starts, schedules, history"}, "Advanced"));
  $("#p-profiles .ph .sp").after(seg);
  seg.dataset.v = PF.mode || "simple";
  segInit(seg, v => { PF.mode = v; store.set("pf.mode", v); api("/api/gpu/ui", {body: {mode: v}}).catch(() => {}); pfRender(); });
  $("#pf-new").addEventListener("click", () => pfEdit(null, "balanced"));
  $("#pf-quiet").addEventListener("click", pfBench);
  $("#pf-export").addEventListener("click", pfExport);
  $("#pf-import").addEventListener("click", () => { $("#pf-imp-text").value = ""; $("#pf-imp").showModal(); });
  $("#pf-imp-cancel").addEventListener("click", () => $("#pf-imp").close());
  $("#pf-imp-file").addEventListener("change", async ev => { const f = ev.target.files[0]; if (f) $("#pf-imp-text").value = await f.text(); });
  $("#pf-imp-ok").addEventListener("click", pfImport);
  $("#pf-ed-cancel").addEventListener("click", () => $("#pf-ed").close());
  $("#pf-ed-save").addEventListener("click", () => pfSave(false));
  $("#pf-ed-preview").addEventListener("click", () => pfSave(true));
  $("#pf-plan-close").addEventListener("click", () => $("#pf-plan").close());
  $("#pf-pw-cancel").addEventListener("click", () => $("#pf-pw").close());
  $("#pf-qd-cancel").addEventListener("click", () => $("#pf-qd").close());
  $("#pf-sched-add").addEventListener("click", () => { pfSchedRows().push({id: "s" + Date.now().toString(36), profile: (PF.state.profiles[0] || {}).id, at: "01:00", days: "daily", enabled: true}); PF.schedDirty = true; pfRenderSched(); });
  $("#pf-sched-save").addEventListener("click", () => pfCall(async () => { await api("/api/gpu/schedules", {body: {schedules: pfSchedRows()}}); PF.schedDirty = false; PF.sched = null; }, "Schedules saved"));
  $("#pf-sup-toggle").addEventListener("click", () => pfCall(() => api("/api/gpu/supervisor", {body: {action: PF.state.supervisor && PF.state.supervisor.paused ? "resume" : "pause"}})));
  $("#pf-audit-reload").addEventListener("click", () => pfLoadAudit(true));
}

// ---------------- render ----------------
function pfRender() {
  const s = PF.state; if (!s) return;
  const adv = pfAdv();
  const m = $("#pf-mode"); if (m && m.setv) m.setv(PF.mode);
  $("#pf-quiet").hidden = !adv; $("#pf-export").hidden = !adv; $("#pf-import").hidden = !adv;
  $("#pf-quiet").textContent = s.quiet && s.quiet.on ? "End benchmark mode" : "Benchmark mode";
  pfRenderBanner();
  pfRenderCards();
  pfRenderProfiles();
  const two = $("#p-profiles .pf-two"), aud = $("#pf-audit").closest(".card");
  two.hidden = !adv; aud.hidden = !adv;
  if (adv) { pfRenderSched(); pfRenderSup(); pfLoadAudit(false); }
}

function pfHowToEnable() {
  const path = (PF.info && PF.info.config) || "~/.config/pxa/control.json";
  const line = '"allow_gpu_control": true';
  return el("details", {style: "margin-top:6px"}, el("summary", {text: "How to switch it on"}),
    el("ol", {class: "tiny", style: "margin:4px 0 0;padding-left:1.3em;line-height:1.7"},
      el("li", {}, "Open PXA Control's settings file: ", el("span", {class: "mono", text: path})),
      el("li", {}, "Add this line inside the { }: ", el("span", {class: "mono", text: line}), " ",
        el("button", {class: "link", onclick: () => copyText(line), text: "copy"})),
      el("li", {text: "Restart PXA Control. Every change still asks you first, and you can undo it."})));
}

function pfRenderBanner() {
  const s = PF.state, out = [], adv = pfAdv();
  if (!s.settings.adapter.available) {
    out.push(el("div", {class: "alert warn"}, badge("warn", "read-only"), el("span", {text:
      "PXA Control cannot reach the NVIDIA driver tools on this machine, so it shows what it can and changes nothing. Everything else in PXA Control works as usual."})));
  } else if (!s.settings.allow_gpu_control) {
    out.push(el("div", {class: "alert warn"}, badge("warn", "look only"), el("div", {},
      el("div", {text: "Changing cards is switched off on this machine (the safe default). You can look around, make profiles and preview them; nothing is applied."}),
      pfHowToEnable())));
  }
  if (s.auto_rollback && Date.now() / 1000 - s.auto_rollback.ts < 3600) {
    out.push(el("div", {class: "alert bad"}, badge("bad", "undone"), el("span", {text:
      `PXA Control undid your last change by itself ${fmtAgo(s.auto_rollback.ts)}, because ${s.auto_rollback.why}. ` +
      (s.auto_rollback.ok ? "The cards are back as they were." : "Some values could not be put back: " + (s.auto_rollback.error || "see Changes") + ".")})));
  }
  if (s.undo) {
    out.push(el("div", {class: "alert ok"}, badge("ok", "changed"), el("div", {style: "flex:1"},
      el("div", {text: `${s.undo.label}, ${fmtAgo(s.undo.ts)}` + (s.probation ? ". Watching the cards for a few seconds; anything wrong is undone by itself." : ".")}),
      el("div", {class: "tiny", text: s.undo.lines.join(" · ")})),
      el("button", {class: "b", onclick: () => pfCall(() => api("/api/gpu/undo", {body: {id: s.undo.id}}), "Undone: the cards are back as they were"), text: "Undo"})));
  }
  if (s.quiet && s.quiet.on) {
    out.push(el("div", {class: "alert warn"}, badge("warn", "benchmark"), el("span", {style: "flex:1", text:
      `Benchmark mode is on since ${fmtAgo(s.quiet.since)}: card(s) ${pfIdx(s.quiet.bench)} are kept free for a speed test and auto-starts wait.`}),
      el("button", {class: "b", onclick: pfBench, text: "End"})));
  }
  for (const lk of s.locks) {
    if (lk.kind === "reserved" && s.quiet && s.quiet.on) continue;
    const who = lk.scope ? "Card(s) " + pfIdx(lk.scope) : "Every card";
    const what = {maintenance: "maintenance mode is on", reserved: "kept free", lock_file: "a speed test is running", waiting_for: "a speed test is running"}[lk.kind] || lk.kind;
    out.push(el("div", {class: "alert warn"}, badge("warn", "kept free"), el("span", {style: "flex:1", text:
      `${who}: ${what}${lk.why && lk.kind !== "waiting_for" ? " (" + lk.why + ")" : ""}. PXA Control will not change ${lk.scope ? "it" : "them"} or start servers there until that ends.` +
      (adv && lk.path ? ` [${lk.path}]` : "")}),
      lk.kind === "reserved" && adv ? el("button", {class: "b", onclick: () => pfCall(() => api("/api/gpu/reserve", {body: {targets: lk.scope, on: false}}), "released"), text: "Release"}) : null,
      lk.kind === "maintenance" ? el("button", {class: "b", onclick: () => pfCall(() => api("/api/gpu/maintenance", {body: {on: false}})), text: "End"}) : null));
  }
  if (adv && s.settings.store_error) out.push(el("div", {class: "alert bad"}, badge("bad", "file"), el("span", {text: s.settings.store_error})));
  pfFill($("#pf-banner"), ...out);
}
function pfIdx(uuids) {
  const by = {}; for (const c of PF.state.cards) by[c.uuid] = c.index;
  return (uuids || []).map(u => by[u] != null ? "#" + by[u] : (typeof u === "number" ? "#" + u : String(u).slice(0, 12) + "…")).join(", ");
}

// ---- the live cards ----
function pfMeter(label, help, right, pct, marks, sub, cls) {
  return el("div", {class: "pf-meter"},
    el("div", {class: "top"}, el("span", {}, label, " ", pfInfo(help)), el("b", {text: right})),
    el("div", {class: "pf-track"}, ...(marks.band ? [el("span", {class: "rng", style: `left:${marks.band[0]}%;width:${Math.max(0, marks.band[1] - marks.band[0])}%`})] : []),
      el("i", {class: cls || "", style: `width:${Math.max(0, Math.min(100, pct))}%`}),
      ...(marks.list || []).map(m => el("span", {class: "mk " + (m.cls || ""), style: `left:${Math.max(0, Math.min(100, m.at))}%`, title: m.title}))),
    el("div", {class: "pf-sub"}, ...sub.map(x => el("span", {text: x}))));
}
function pfSpark(rows) {
  const W = 300, H = 46;
  if (!rows || rows.filter(r => r[1] != null || r[2] != null).length < 2) return el("div", {class: "tiny", style: "height:46px;display:flex;align-items:center", text: "History appears after a minute or two on this page."});
  const t0 = rows[0][0], t1 = rows[rows.length - 1][0] || t0 + 1;
  const pmax = Math.max(1, ...rows.map(r => r[1] || 0)) * 1.1, tmax = 100;
  const line = (i, max) => rows.filter(r => r[i] != null).map(r => `${((r[0] - t0) / Math.max(1, t1 - t0) * W).toFixed(1)},${(H - (r[i] / max) * H).toFixed(1)}`).join(" ");
  return lvSvg("svg", {class: "pf-spark", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img", "aria-label": "power and temperature over the last hour"},
    lvSvg("line", {class: "g", x1: 0, x2: W, y1: H - 0.5, y2: H - 0.5}),
    lvSvg("polyline", {class: "l", points: line(1, pmax), stroke: "var(--series-1)"}),
    lvSvg("polyline", {class: "l", points: line(2, tmax), stroke: "var(--series-2)"}));
}
function pfRenderCards() {
  const s = PF.state, adv = pfAdv(), box = $("#pf-cards");
  $("#pf-cards-sub").textContent = s.cards.length ? `live, every 3 s${s.telemetry ? " · history from the recorder" : ""}` : "";
  if (!s.cards.length) { pfFill(box, el("div", {class: "empty center"}, el("b", {text: "No NVIDIA cards found"}), el("span", {text: s.settings.adapter.error || "PXA Control sees no cards on this machine."}))); return; }
  const profs = {}; for (const p of s.profiles) profs[p.id] = p;
  pfFill(box, ...s.cards.map(c => {
    const max = c.max_w || c.limit_w || 1, dflt = c.default_w || max;
    const custom = !c.active_profile && c.limit_w != null && Math.abs(c.limit_w - dflt) > 0.5;
    const hot = c.thermal || {}, tmax = c.shutdown_c || 100, t = c.temp_c;
    const tcls = c.alert ? c.alert.level : "";
    const fan = c.caps.fan_read ? (c.fan_pct != null ? c.fan_pct + " %" : "?") : "no fan (needs case airflow)";
    const used = c.users.map(u => `${u.name} (${((u.mib || 0) / 1024).toFixed(1)} GiB)`).concat(c.other_mib ? [`other programs (${(c.other_mib / 1024).toFixed(1)} GiB)`] : []);
    return el("div", {class: "card pf-card"},
      el("h3", {}, el("span", {class: "mono", text: "#" + c.index}), el("span", {class: "nm", text: pfShort(c.name)}), el("span", {class: "sp"}),
        c.active_profile ? el("span", {class: "chip pxq", title: "the profile applied to this card", text: (profs[c.active_profile] || {}).name || c.active_profile}) : custom ? el("span", {class: "chip", text: "custom " + pfW(c.limit_w)}) : el("span", {class: "chip", text: "stock"}),
        c.locked.length ? badge("warn", "kept free") : null),
      c.alert ? el("div", {class: "pf-alert " + c.alert.level, role: "alert", text: c.alert.text}) : null,
      c.caps.power_limit ? pfMeter("Power", "How much electricity the card may use. Lower = cooler, quieter and cheaper, a little slower.",
        `${pfW(c.power_w)} of ${pfW(c.limit_w)}`, 100 * (c.power_w || 0) / max,
        {band: [100 * (c.min_w || 0) / max, 100], list: [{at: 100 * (c.limit_w || 0) / max, title: "limit " + pfW(c.limit_w)}, {at: 100 * dflt / max, cls: "warn", title: "stock " + pfW(dflt)}]},
        [`lowest ${pfW(c.min_w)}`, `stock ${pfW(dflt)}`, `highest ${pfW(c.max_w)}`])
        : el("div", {class: "kv"}, "Power", el("b", {text: c.power_w != null ? pfW(c.power_w) : "not reported"})),
      pfMeter("Temperature", "Cards slow themselves down when hot. A warning shows before that.", t != null ? Math.round(t) + " \u00b0C" : "?",
        100 * (t || 0) / tmax, {list: [hot.warn_c ? {at: 100 * hot.warn_c / tmax, cls: "warn", title: "warning " + hot.warn_c + " \u00b0C"} : null, c.slowdown_c ? {at: 100 * c.slowdown_c / tmax, cls: "bad", title: "the card slows down at " + c.slowdown_c + " \u00b0C"} : null].filter(Boolean)},
        [hot.warn_c ? `warns at ${Math.round(hot.warn_c)} \u00b0C` : "", c.slowdown_c ? `slows itself at ${c.slowdown_c} \u00b0C` : ""], "temp " + tcls),
      el("div", {class: "pf-kvs"},
        el("div", {class: "kv"}, "Memory", el("b", {text: c.mem_total_mib ? `${((c.mem_used_mib || 0) / 1024).toFixed(1)} / ${(c.mem_total_mib / 1024).toFixed(0)} GiB` : "?"})),
        el("div", {class: "kv"}, "Busy", el("b", {text: c.util_pct != null ? Math.round(c.util_pct) + " %" : "?"})),
        el("div", {class: "kv", title: "how fast the chip is ticking right now"}, "Speed", el("b", {text: c.sm_clock != null ? Math.round(c.sm_clock) + " MHz" : "?"})),
        el("div", {class: "kv"}, "Fan", el("b", {text: fan})),
        adv ? el("div", {class: "kv"}, "Memory clock", el("b", {text: c.mem_clock != null ? Math.round(c.mem_clock) + " MHz" : "?"})) : null,
        adv ? el("div", {class: "kv"}, "App clocks", el("b", {text: c.caps.app_clocks ? `${c.app_mem}/${c.app_sm} MHz` : "not supported"})) : null,
        adv ? el("div", {class: "kv"}, "Persistence", el("b", {text: c.persistence == null ? "n/a" : c.persistence ? "on" : "off"})) : null,
        adv ? el("div", {class: "kv", title: "the driver does not report a clock lock back, so this shows what the active profile set"}, "Clock lock", el("b", {text: pfLockText(c, profs)})) : null),
      el("div", {class: "pf-eff"},
        el("div", {title: "tokens per second the servers on this card are making right now"}, el("span", {text: "tok/s here"}), el("b", {text: c.tps != null ? fmtNum(c.tps) : "idle"})),
        el("div", {title: "speed per watt: higher means more work for the same electricity. Compare it before and after a power change."}, el("span", {text: "tok/s per watt"}), el("b", {text: c.tps_per_w != null ? fmtNum(c.tps_per_w, 3) : "-"}))),
      pfSpark(c.spark),
      el("div", {class: "pf-leg"}, el("span", {}, el("i", {style: "background:var(--series-1)"}), "power"), el("span", {}, el("i", {style: "background:var(--series-2)"}), "temperature"), el("span", {class: "tiny", text: "last hour"})),
      el("div", {class: "pf-users", text: used.length ? "Running here: " + used.join(", ") : "Nothing running on this card."}),
      adv ? el("div", {class: "tiny mono", style: "overflow-wrap:anywhere", text: c.uuid}) : null,
      el("div", {class: "pf-acts"},
        c.caps.power_limit ? el("button", {class: "b", onclick: () => pfPower(c), text: "Power limit\u2026"}) : null,
        adv && s.settings.allow_gpu_control ? el("button", {class: "b", onclick: () => pfReset(c), title: "stock power limit, default clocks", text: "Reset to default"}) : null,
        adv ? el("button", {class: "b", onclick: () => pfCall(() => api("/api/gpu/reserve", {body: {targets: [c.uuid], on: !c.reserved, why: "kept free from the Profiles page"}}), c.reserved ? "released" : "kept free"), title: "keep this card free: no changes and no server starts on it", text: c.reserved ? "Release" : "Keep free"}) : null));
  }));
}

function pfLockText(c, profs) {
  if (!c.caps.locked_clocks) return "not supported";
  const lc = c.active_profile && profs[c.active_profile] && profs[c.active_profile].gpu.locked_clocks;
  return lc && lc.min ? `${lc.min}\u2013${lc.max} MHz (profile)` : "driver default";
}

// ---- profiles ----
function pfPowerText(g) {
  const p = g && g.power;
  if (!p) return "power left as it is";
  return {default: "stock power", pct_default: `${p.value} % of stock power`, pct_max: p.value === 100 ? "the highest power allowed" : `${p.value} % of the highest power`, watts: `${p.value} W power limit`}[p.mode];
}
function pfPresetTile(id, onpick) {
  const pr = PF.state.presets[id];
  return el("button", {class: "tile", style: "text-align:left;font:inherit;color:inherit", onclick: () => onpick(id)},
    el("span", {class: "t1", text: pr.name}), el("span", {class: "t2", text: pr.blurb}));
}
function pfPresetIds(all) {
  return Object.entries(PF.state.presets).filter(([, p]) => all || p.simple).sort((a, b) => a[1].order - b[1].order).map(([k]) => k);
}
function pfRenderProfiles() {
  const s = PF.state, adv = pfAdv(), box = $("#pf-list");
  if (!s.profiles.length) {
    pfFill(box, el("div", {class: "card", style: "grid-column:1/-1"},
      el("div", {class: "empty center", style: "padding-bottom:var(--s4)"}, el("b", {text: "No profiles yet. Pick a preset to get started."}),
        el("span", {text: "A profile is a saved setting for your cards. You can preview it before anything changes, and undo it after."})),
      el("div", {class: "tiles", style: "grid-template-columns:repeat(auto-fit,minmax(200px,1fr))"}, ...pfPresetIds(adv).map(id => pfPresetTile(id, k => pfEdit(null, k))))));
    return;
  }
  const by = {}; for (const c of s.cards) by[c.uuid] = c;
  pfFill(box, ...s.profiles.map(p => {
    const active = s.cards.filter(c => c.active_profile === p.id).length;
    const pr = p.preset ? s.presets[p.preset] : null;
    return el("div", {class: "card pf-prof"},
      el("h3", {}, el("span", {style: "overflow-wrap:anywhere", text: p.name}), el("span", {class: "sp"}),
        pr ? el("span", {class: "chip", text: pr.name}) : el("span", {class: "chip", text: "custom"}),
        active ? badge("ok", `on ${active} card${active > 1 ? "s" : ""}`) : null),
      el("div", {class: "blurb", text: `${pfPowerText(p.gpu)[0].toUpperCase() + pfPowerText(p.gpu).slice(1)}` +
        (p.thermal && p.thermal.act_c && p.thermal.action === "cap_power" ? `, slows down by itself above ${p.thermal.act_c} \u00b0C` : "") + "."}),
      el("div", {class: "pf-chips"}, ...p.targets.map(u => by[u] ? el("span", {class: "chip", title: u, text: `#${by[u].index} ${pfShort(by[u].name)}`}) : el("span", {class: "chip warn", title: u, text: "card not present"}))),
      p.autostart.length ? el("div", {class: "tiny", text: "Starts: " + p.autostart.map(a => (s.servers.find(x => x.sid === a.server) || {name: a.server}).name).join(", ")}) : null,
      adv && p.notes ? el("div", {class: "tiny", text: p.notes}) : null,
      el("div", {class: "pf-acts"},
        el("button", {class: "b pri", onclick: () => pfPreview({id: p.id}, p), text: "Apply\u2026"}),
        el("button", {class: "b", onclick: () => pfEdit(p), text: "Edit"}),
        adv ? el("button", {class: "b", onclick: () => pfEdit(Object.assign(JSON.parse(JSON.stringify(p)), {id: "", name: (p.name + " copy").slice(0, 48)})), text: "Clone"}) : null,
        el("button", {class: "b danger", onclick: () => pfDelete(p), text: "Delete"}),
        pfCurlBtn("POST", "/api/gpu/apply", () => ({id: p.id, confirm: p.id}))));
  }));
}
async function pfDelete(p) {
  if (!confirm(`Delete the profile "${p.name}"? The cards keep their current settings.`)) return;
  pfCall(() => api("/api/gpu/profiles", {method: "DELETE", body: {id: p.id}}), "deleted");
}

// ---- the editor ----
function pfBlank(presetId) {
  const pr = PF.state.presets[presetId] || PF.state.presets.balanced;
  return {id: "", name: pr.name, preset: presetId, targets: PF.state.cards.map(c => c.uuid), notes: "",
    gpu: JSON.parse(JSON.stringify(pr.gpu)), thermal: JSON.parse(JSON.stringify(pr.thermal)), autostart: []};
}
function pfEdit(p, presetId) {
  PF.draft = p ? JSON.parse(JSON.stringify(p)) : pfBlank(presetId || "balanced");
  PF.editing = p && p.id ? p.id : null;
  $("#pf-ed-h").textContent = PF.editing ? "Edit " + p.name : "New profile";
  $("#pf-ed-msg").textContent = "";
  pfRenderEditor();
  $("#pf-ed").showModal();
}
function pfSelCards() { const d = PF.draft; return PF.state.cards.filter(c => d.targets.includes(c.uuid)); }
function pfRenderEditor() {
  const d = PF.draft, s = PF.state, adv = pfAdv(), body = $("#pf-ed-body");
  const sec = (title, help, ...kids) => el("div", {class: "pf-sec"}, el("h4", {}, title, help ? pfInfo(help) : null), ...kids);
  const presets = el("div", {class: "tiles", style: "grid-template-columns:repeat(auto-fit,minmax(180px,1fr))"}, ...pfPresetIds(adv).map(id => {
    const t = pfPresetTile(id, k => { const keep = {id: d.id, name: d.name, targets: d.targets, autostart: d.autostart, notes: d.notes}; const b = pfBlank(k);
      PF.draft = Object.assign(b, keep, {preset: k, name: (!d.name || Object.values(s.presets).some(x => x.name === d.name)) ? b.name : d.name}); pfRenderEditor(); });
    if (d.preset === id) t.style.boxShadow = "inset 0 0 0 1px var(--accent)", t.style.borderColor = "var(--accent)";
    return t;
  }));
  const name = el("input", {type: "text", value: d.name, maxlength: "48", "aria-label": "profile name", style: "width:100%"});
  name.addEventListener("input", () => { d.name = name.value; });
  const cards = el("div", {class: "tiles"}, ...s.cards.map(c => {
    const cb = el("input", {type: "checkbox"}); cb.checked = d.targets.includes(c.uuid);
    cb.addEventListener("change", () => { d.targets = cb.checked ? d.targets.concat([c.uuid]) : d.targets.filter(u => u !== c.uuid); pfRenderEditor(); });
    return el("label", {class: "tile"}, cb, el("span", {class: "tick"}), el("span", {class: "t1", text: `#${c.index} ${pfShort(c.name)}`}),
      el("span", {class: "t2", text: adv ? c.uuid : `${pfW(c.min_w)} \u2013 ${pfW(c.max_w)}, stock ${pfW(c.default_w)}`}));
  }));
  const groups = adv ? el("div", {class: "row", style: "margin-top:8px"}, el("span", {class: "tiny", text: "Select a group:"}),
    ...Object.entries(s.cards.reduce((a, c) => { (a[pfShort(c.name)] = a[pfShort(c.name)] || []).push(c.uuid); return a; }, {})).map(([n, us]) =>
      el("button", {class: "b", style: "height:30px", onclick: () => { d.targets = us; pfRenderEditor(); }, text: `all ${n} (${us.length})`})),
    el("button", {class: "b", style: "height:30px", onclick: () => { d.targets = s.cards.map(c => c.uuid); pfRenderEditor(); }, text: "every card"})) : null;
  const uuidBox = adv ? el("details", {style: "margin-top:6px"}, el("summary", {text: "Cards by UUID (also cards not plugged in now)"}), (() => {
    const ta = el("textarea", {rows: "3", class: "mono", spellcheck: "false"}); ta.value = d.targets.join("\n");
    ta.addEventListener("change", () => { d.targets = ta.value.split(/[\s,]+/).filter(Boolean); pfRenderEditor(); }); return ta; })()) : null;
  pfFill(body, 
    el("div", {class: "pf-sec", style: "border-top:0;margin-top:0;padding-top:0"}, el("h4", {}, "Start from"), presets),
    sec("Name", null, name),
    sec("Which cards", "The profile changes only the cards ticked here.", cards, groups, uuidBox),
    sec("Power limit", "How much electricity each card may use. Lower = cooler, quieter and cheaper, a little slower. Higher = faster, hotter, louder.", pfPowerEditor()),
    pfThermalSimple(),
    adv ? pfAdvancedEditor() : null);
}
function pfPowerEditor() {
  const d = PF.draft, adv = pfAdv(), sel = pfSelCards();
  const p = d.gpu.power || {mode: "default", value: null};
  const wrap = el("div", {});
  const pctOf = c => p.mode === "pct_default" ? (c.default_w || c.max_w) * p.value / 100 : p.mode === "pct_max" ? c.max_w * p.value / 100 : p.mode === "watts" ? p.value : (c.default_w || c.max_w);
  const clampLine = () => sel.length ? el("div", {class: "pf-clamp"}, ...sel.map(c => {
    const want = pctOf(c), w = Math.round(Math.min(Math.max(want, c.min_w || 0), c.max_w || want));
    const out = w < (c.default_w || c.max_w) * PF_RECO_LO - 0.5 || w > (c.default_w || c.max_w) + 0.5;
    return el("div", {}, `#${c.index} ${pfShort(c.name)}: `, el("b", {text: `${pfW(c.limit_w)} \u2192 ${pfW(w)}`}),
      Math.abs(w - want) > 0.5 ? ` (the card allows ${pfW(c.min_w)}\u2013${pfW(c.max_w)})` : "",
      out ? el("span", {style: "color:var(--warn)", text: "  outside the recommended range: " + (w > (c.default_w || 0) ? "runs hotter than stock" : "noticeably slower")}) : null);
  })) : el("div", {class: "pf-clamp", text: "Tick at least one card above."});
  if (!adv) {                                    // Simple: one slider, % of stock, inside the recommended band
    if (p.mode !== "pct_default") { d.gpu.power = {mode: "pct_default", value: p.mode === "default" || p.mode === "pct_max" ? 100 : 80}; return pfPowerEditor(); }
    const lo = Math.round(PF_RECO_LO * 100);
    const r = el("input", {type: "range", min: String(lo), max: "100", step: "5", value: String(Math.min(100, Math.max(lo, p.value))), "aria-label": "power limit, percent of stock"});
    const v = el("b", {style: "min-width:120px", text: `${r.value} % of stock`});
    const cl = el("div", {}, clampLine());
    r.addEventListener("input", () => { d.gpu.power = {mode: "pct_default", value: +r.value}; v.textContent = `${r.value} % of stock`; pfFill(cl, clampLine()); });
    wrap.append(el("div", {class: "pf-row"}, el("span", {class: "tiny", text: "cooler"}), r, el("span", {class: "tiny", text: "faster"}), v,
      el("button", {class: "b", onclick: () => { d.gpu.power = {mode: "pct_default", value: 100}; pfRenderEditor(); }, text: "Reset to default"})),
      el("div", {class: "pf-sub", style: "margin-top:2px"}, el("span", {text: `${lo} % (recommended lowest)`}), el("span", {text: "100 % = stock"})), cl);
    return wrap;
  }
  // Advanced: every mode, exact numbers, the hardware range (warned outside the recommended band)
  const seg = el("span", {class: "seg"}, ...[["leave", "Leave"], ["default", "Stock"], ["pct_default", "% of stock"], ["pct_max", "% of max"], ["watts", "Watts"]].map(([k, t]) =>
    el("button", {"data-v": k, "aria-pressed": String(d.gpu.power ? d.gpu.power.mode === k : k === "leave"), text: t,
      onclick: () => { d.gpu.power = k === "leave" ? null : {mode: k, value: k === "watts" ? Math.round((sel[0] && (sel[0].limit_w || sel[0].default_w)) || 200) : k === "default" ? null : 100}; pfRenderEditor(); }})));
  wrap.append(seg);
  if (d.gpu.power && ["watts", "pct_default", "pct_max"].includes(d.gpu.power.mode)) {
    const isW = d.gpu.power.mode === "watts";
    const lo = isW ? Math.min(...sel.map(c => c.min_w || 30), 1000) : 10, hi = isW ? Math.max(...sel.map(c => c.max_w || 300), 30) : 100;
    const num = el("input", {type: "number", min: String(lo), max: String(hi), step: "1", value: String(d.gpu.power.value)});
    const r = el("input", {type: "range", min: String(lo), max: String(hi), step: "1", value: String(d.gpu.power.value)});
    const cl = el("div", {}, clampLine());
    const set = v => { d.gpu.power.value = Math.max(lo, Math.min(hi, +v || lo)); num.value = r.value = d.gpu.power.value; pfFill(cl, clampLine()); };
    r.addEventListener("input", () => set(r.value)); num.addEventListener("change", () => set(num.value));
    wrap.append(el("div", {class: "pf-row", style: "margin-top:8px"}, r, num, el("span", {class: "tiny", text: isW ? "W" : "%"}),
      el("button", {class: "b", onclick: () => { d.gpu.power = {mode: "default", value: null}; pfRenderEditor(); }, text: "Reset to default"})),
      el("div", {class: "pf-sub"}, el("span", {text: (isW ? lo + " W" : lo + " %") + " hardware lowest"}), el("span", {text: isW ? "stock " + sel.map(c => pfW(c.default_w)).filter((x, i, a) => a.indexOf(x) === i).join(" / ") : ""}), el("span", {text: (isW ? hi + " W" : hi + " %") + " highest"})), cl);
  } else wrap.append(clampLine());
  return wrap;
}
function pfThermalSimple() {
  const d = PF.draft, adv = pfAdv();
  if (adv) return null;
  const on = d.thermal && d.thermal.action === "cap_power" && d.thermal.act_c;
  const cb = el("input", {type: "checkbox"}); cb.checked = !!on;
  cb.addEventListener("change", () => { d.thermal = cb.checked ? {warn_c: 78, act_c: 85, action: "cap_power", cap_pct: 60} : {warn_c: 80, act_c: null, action: "alert", cap_pct: 70}; });
  return el("div", {class: "pf-sec"}, el("label", {class: "check"}, cb, "Slow the card down by itself if it gets too hot ", pfInfo("Above the safety temperature the card's power is lowered automatically, then you get a note on this page.")));
}
function pfAdvancedEditor() {
  const d = PF.draft, s = PF.state, sel = pfSelCards();
  const anyCap = k => sel.some(c => c.caps[k]);
  const segOf = (opts, cur, on) => el("span", {class: "seg"}, ...opts.map(([k, t]) => el("button", {"data-v": k, "aria-pressed": String(cur === k), text: t, onclick: () => { on(k); pfRenderEditor(); }})));
  const g = d.gpu;
  const pmCur = g.persistence === true ? "on" : g.persistence === false ? "off" : "leave";
  const acCur = g.app_clocks === "default" ? "default" : g.app_clocks ? "set" : "leave";
  const lcCur = g.locked_clocks === "reset" ? "reset" : g.locked_clocks ? "set" : "leave";
  const numIn = (val, on, min, max, w) => { const i = el("input", {type: "number", value: val == null ? "" : String(val), min: String(min), max: String(max), style: `width:${w || 96}px`}); i.addEventListener("change", () => on(i.value === "" ? null : +i.value)); return i; };
  const th = d.thermal || {};
  const rows = d.autostart.map((a, i) => {
    const srv = el("select", {}, ...s.servers.map(x => el("option", {value: x.sid, text: `${x.name}${x.model ? " · " + x.model : ""}`}))); srv.value = a.server;
    srv.addEventListener("change", () => a.server = srv.value);
    const rs = el("select", {}, el("option", {value: "never", text: "never"}), el("option", {value: "on-failure", text: "on failure"})); rs.value = a.restart;
    rs.addEventListener("change", () => a.restart = rs.value);
    const ob = el("input", {type: "checkbox"}); ob.checked = !!a.on_boot; ob.addEventListener("change", () => a.on_boot = ob.checked);
    const xa = el("input", {type: "text", placeholder: "--flag value ...", value: (a.extra_args || []).join(" "), style: "width:100%"});
    xa.addEventListener("change", () => a.extra_args = xa.value.split(/\s+/).filter(Boolean));
    const ev = el("input", {type: "text", placeholder: "PXA_NAME=value ...", value: Object.entries(a.env || {}).map(([k, v]) => k + "=" + v).join(" "), style: "width:100%"});
    ev.addEventListener("change", () => { a.env = {}; for (const kv of ev.value.split(/\s+/).filter(Boolean)) { const j = kv.indexOf("="); if (j > 0) a.env[kv.slice(0, j)] = kv.slice(j + 1); } });
    return [el("tr", {}, el("td", {}, srv), el("td", {}, numIn(a.order, v => a.order = v ?? i + 1, 0, 999, 64)), el("td", {}, numIn(a.delay_s, v => a.delay_s = v ?? 0, 0, 3600, 74)),
      el("td", {}, numIn(a.health_wait_s, v => a.health_wait_s = v ?? 600, 5, 7200, 80)), el("td", {}, rs), el("td", {}, numIn(a.max_retries, v => a.max_retries = v ?? 3, 0, 50, 64)),
      el("td", {}, numIn(a.backoff_s, v => a.backoff_s = v ?? 15, 1, 3600, 74)), el("td", {}, ob),
      el("td", {}, el("button", {class: "b danger", style: "height:30px", onclick: () => { d.autostart.splice(i, 1); pfRenderEditor(); }, text: "\u2715"}))),
      el("tr", {}, el("td", {colspan: "9", style: "border-bottom:1px solid var(--line-2)"}, el("div", {class: "pf-row"}, el("span", {class: "tiny", text: "extra engine args"}), el("div", {style: "flex:1;min-width:160px"}, xa), el("span", {class: "tiny", text: "env"}), el("div", {style: "flex:1;min-width:160px"}, ev))))];
  });
  const notes = el("textarea", {rows: "2", maxlength: "500"}); notes.value = d.notes || ""; notes.addEventListener("input", () => d.notes = notes.value);
  return el("details", {class: "pf-sec", open: true}, el("summary", {text: "Advanced"}),
    el("div", {class: "form", style: "grid-template-columns:repeat(auto-fill,minmax(260px,1fr));margin-top:8px"},
      anyCap("persistence") ? el("div", {}, el("div", {class: "tiny", style: "margin-bottom:4px"}, "Persistence mode ", pfInfo("keeps the driver loaded between runs: faster starts, a few watts at idle")),
        segOf([["leave", "Leave"], ["on", "On"], ["off", "Off"]], pmCur, k => g.persistence = k === "leave" ? null : k === "on")) : null,
      anyCap("app_clocks") ? el("div", {}, el("div", {class: "tiny", style: "margin-bottom:4px"}, "Application clocks (memory / chip) ", pfInfo("the clocks the card runs at under load. 'Default' puts the factory pair back. Not available on GeForce cards.")),
        segOf([["leave", "Leave"], ["default", "Default"], ["set", "Set"]], acCur, k => g.app_clocks = k === "leave" ? null : k === "default" ? "default" : {mem: (sel.find(c => c.caps.app_clocks) || {}).def_app_mem || 715, sm: (sel.find(c => c.caps.app_clocks) || {}).def_app_sm || 1189}),
        acCur === "set" ? el("div", {class: "pf-row", style: "margin-top:6px"}, numIn(g.app_clocks.mem, v => g.app_clocks.mem = v, 100, 20000), el("span", {class: "tiny", text: "/"}), numIn(g.app_clocks.sm, v => g.app_clocks.sm = v, 100, 5000), el("span", {class: "tiny", text: "MHz (snapped to a pair the card supports)"})) : null) : null,
      anyCap("locked_clocks") ? el("div", {}, el("div", {class: "tiny", style: "margin-bottom:4px"}, "Locked chip clock ", pfInfo("pins the chip clock to a range (Volta and newer). Steadier benchmark numbers.")),
        segOf([["leave", "Leave"], ["reset", "Unlock"], ["set", "Lock"]], lcCur, k => g.locked_clocks = k === "leave" ? null : k === "reset" ? "reset" : {min: 1000, max: Math.round((sel.find(c => c.caps.locked_clocks) || {}).max_sm || 1380)}),
        lcCur === "set" ? el("div", {class: "pf-row", style: "margin-top:6px"}, numIn(g.locked_clocks.min, v => g.locked_clocks.min = v, 100, 5000), el("span", {class: "tiny", text: "to"}), numIn(g.locked_clocks.max, v => g.locked_clocks.max = v, 100, 5000), el("span", {class: "tiny", text: "MHz"})) : null) : null,
      el("div", {}, el("div", {class: "tiny", style: "margin-bottom:4px"}, "Temperature guard ", pfInfo("warn: a note on this page. act: what to do at the second temperature.")),
        el("div", {class: "pf-row"}, el("span", {class: "tiny", text: "warn"}), numIn(th.warn_c, v => th.warn_c = v, 40, 100, 70), el("span", {class: "tiny", text: "act"}), numIn(th.act_c, v => th.act_c = v, 40, 105, 70), el("span", {class: "tiny", text: "\u00b0C"})),
        el("div", {class: "pf-row", style: "margin-top:6px"}, (() => { const x = el("select", {}, el("option", {value: "alert", text: "only warn"}), el("option", {value: "cap_power", text: "lower the power"}), el("option", {value: "stop_autostart", text: "pause auto-starts"})); x.value = th.action || "alert"; x.addEventListener("change", () => th.action = x.value); return x; })(),
          el("span", {class: "tiny", text: "to"}), numIn(th.cap_pct, v => th.cap_pct = v, 10, 100, 70), el("span", {class: "tiny", text: "% of stock"})))),
    el("div", {class: "pf-sec"}, el("h4", {}, "Servers that start with this profile", pfInfo("in order; each waits until the one before is healthy. 'on failure' restarts a crashed server with growing pauses, and stops after the retries or 5 crashes in 10 minutes.")),
      s.servers.length ? el("div", {class: "tw"}, el("table", {class: "pf-as"}, el("thead", {}, el("tr", {}, ...["Server", "Order", "Delay s", "Health wait s", "Restart", "Retries", "Backoff s", "At boot", ""].map(h => el("th", {text: h})))), el("tbody", {}, ...rows.flat()))) : el("div", {class: "tiny", text: "Save a server on the Launch tab first; it then appears here."}),
      s.servers.length ? el("button", {class: "b", style: "margin-top:8px", onclick: () => { d.autostart.push({server: s.servers[0].sid, order: d.autostart.length + 1, delay_s: 0, health_wait_s: 600, restart: "on-failure", max_retries: 3, backoff_s: 15, on_boot: false, extra_args: [], env: {}}); pfRenderEditor(); }, text: "Add a server"}) : null),
    el("div", {class: "pf-sec"}, el("h4", {}, "Notes"), notes,
      el("div", {class: "row", style: "margin-top:8px"}, el("button", {class: "b", onclick: () => copyText(JSON.stringify(PF.draft, null, 1)), text: "Copy as JSON"}),
        el("button", {class: "b", onclick: () => copyText(pfCurl("POST", "/api/gpu/profiles", PF.draft)), text: "Copy as curl"}))));
}
async function pfSave(previewOnly) {
  const d = PF.draft;
  if (!d.targets.length) { $("#pf-ed-msg").textContent = "Tick at least one card."; return; }
  if (!(d.name || "").trim()) { $("#pf-ed-msg").textContent = "Give the profile a name."; return; }
  const body = JSON.parse(JSON.stringify(d)); if (!body.id) delete body.id;
  for (const a of body.autostart || []) { a.order = a.order ?? 1; }
  try {
    const r = await api("/api/gpu/profiles", {body});
    PF.draft.id = r.profile.id; PF.editing = r.profile.id;
    $("#pf-ed").close(); toast(`Saved "${r.profile.name}"` + (previewOnly ? "" : ". Press Apply when you want it on the cards."));
    await pfLoad();
    if (previewOnly) pfPreview({id: r.profile.id}, r.profile);
  } catch (e) { $("#pf-ed-msg").textContent = pfFriendly(e); }
}

// ---- preview + apply (the confirm, in plain words) ----
function pfStepText(c, st) {
  const v = (f, x) => f === "power_limit" ? pfW(x) : f === "persistence" ? (x ? "on" : "off") : x == null ? "default" : x.map(Math.round).join("/") + " MHz";
  const n = {power_limit: "power", persistence: "keep driver loaded", app_clocks: "clocks", locked_clocks: "locked clock"}[st.field];
  return `Card ${c.index} ${n}: ${v(st.field, st.before)} \u2192 ${v(st.field, st.after)}`;
}
async function pfPreview(req, p) {
  const s = PF.state, adv = pfAdv();
  let plan;
  try { plan = await api("/api/gpu/plan", {body: req}); } catch (e) { toast(pfFriendly(e), true); return; }
  PF.plan = {req, p, plan};
  $("#pf-plan-h").textContent = "Apply " + p.name + "?";
  const lines = [], notes = [], locked = [];
  for (const c of plan.cards) {
    for (const st of c.steps) lines.push(pfStepText(c, st));
    for (const n of c.notes) notes.push((c.index != null ? `Card ${c.index}: ` : "") + n);
    if (c.locked.length && c.steps.length) locked.push(`Card ${c.index}: ${c.locked.join("; ")}`);
  }
  const canWrite = s.settings.allow_gpu_control;
  const box = [];
  if (lines.length) box.push(el("div", {class: "card", style: "margin:0;box-shadow:none"}, ...lines.map(t => el("div", {class: "kv", style: "font-size:var(--fs-2);color:var(--text)", text: t})),
    el("div", {class: "tiny", style: "margin-top:6px", text: "You can undo this anytime from the top of the Profiles page. If a card misbehaves in the first seconds, PXA Control undoes it by itself."})));
  else box.push(el("div", {class: "alert ok"}, badge("ok", "nothing to do"), el("span", {text: "The cards already match this profile."})));
  if (plan.autostart.length) box.push(el("div", {class: "kv"}, "Then starts", el("b", {text: plan.autostart.map(a => (s.servers.find(x => x.sid === a) || {name: a}).name).join(", ") + (s.settings.autostart ? "" : " (auto-start is switched off here: nothing will start)")})));
  if (notes.length) box.push(el("div", {class: "pf-clamp"}, ...notes.map(n => el("div", {text: n}))));
  if (locked.length) box.push(el("div", {class: "alert warn"}, badge("warn", "kept free"), el("span", {text: locked.join(" · ") + ". These cards cannot be changed right now."})));
  if (lines.length && !canWrite) box.push(el("div", {class: "alert warn"}, badge("warn", "look only"), el("div", {}, el("div", {text: "Changing cards is switched off on this machine, so this is a preview only."}), pfHowToEnable())));
  if (adv) box.push(el("details", {class: "pf-sec"}, el("summary", {text: "Exact changes (raw)"}),
    !lines.length ? el("div", {class: "tiny", text: "No differences: every field already has the value this profile asks for."}) :
    el("div", {class: "tw"}, el("table", {class: "pf-diff"}, el("thead", {}, el("tr", {}, ...["Card", "UUID", "Field", "Now", "After"].map(h => el("th", {text: h})))),
      el("tbody", {}, ...plan.cards.flatMap(c => c.steps.map(st => el("tr", {}, el("td", {class: "mono", text: c.index == null ? "-" : "#" + c.index}), el("td", {class: "mono", text: c.uuid}),
        el("td", {class: "mono", text: st.field}), el("td", {class: "mono from", text: JSON.stringify(st.before)}), el("td", {class: "mono to", text: JSON.stringify(st.after)}))))))),
    el("div", {class: "row", style: "margin-top:8px"}, el("button", {class: "b", onclick: () => copyText(pfCurl("POST", "/api/gpu/plan", req)), text: "Copy preview as curl"}),
      el("button", {class: "b", onclick: () => copyText(pfCurl("POST", "/api/gpu/apply", Object.assign({}, req, {confirm: req.id}))), text: "Copy apply as curl"}))));
  box.push(el("div", {class: "pf-res", id: "pf-plan-res"}));
  pfFill($("#pf-plan-body"), ...box);
  const btn = $("#pf-plan-apply");
  const blocked = (lines.length && !canWrite) || locked.length;
  btn.hidden = !!blocked; btn.disabled = false;
  btn.textContent = lines.length ? `Apply to ${new Set(plan.cards.filter(c => c.steps.length).map(c => c.uuid)).size} card(s)` : "Mark as applied";
  btn.onclick = async () => {
    btn.disabled = true;
    try {
      const r = await api("/api/gpu/apply", {body: {id: req.id, confirm: req.id}, timeout: 120000});
      if (r.ok) { $("#pf-plan").close(); toast(lines.length ? "Applied. Undo is at the top of the page." : "Done"); }
      else {
        const f = r.failure || {};
        pfFill($("#pf-plan-res"), el("div", {class: "alert bad"}, badge("bad", "put back"), el("span", {text:
          `Card ${f.index} did not take the change (${f.error}), so PXA Control put every card back the way it was` + (r.rollback_ok ? "." : ", except: " + r.rolled_back.filter(x => !x.ok).map(x => `card ${x.index} ${x.field}`).join(", ") + ". Check those cards.")})));
      }
    } catch (e) { pfFill($("#pf-plan-res"), el("div", {class: "alert bad"}, badge("bad", "not applied"), el("span", {text: pfFriendly(e)}))); }
    btn.disabled = false; pfLoad();
  };
  $("#pf-plan").showModal();
}

// ---- the quick power limit on a card (A/B comparisons) ----
function pfPower(c) {
  const adv = pfAdv(), s = PF.state, dflt = c.default_w || c.max_w;
  const lo = adv ? c.min_w : Math.max(c.min_w, Math.round(dflt * PF_RECO_LO)), hi = adv ? c.max_w : Math.min(c.max_w, dflt);
  let want = Math.round(Math.min(hi, Math.max(lo, c.limit_w || dflt)));
  $("#pf-pw-h").textContent = `Card ${c.index} \u00b7 ${pfShort(c.name)}: power limit`;
  const r = el("input", {type: "range", min: String(lo), max: String(hi), step: "1", value: String(want), "aria-label": "power limit in watts"});
  const num = el("input", {type: "number", min: String(lo), max: String(hi), step: "1", value: String(want), hidden: !adv});
  const big = el("b", {style: "font-size:var(--fs-4);font-variant-numeric:tabular-nums;min-width:90px", text: pfW(want)});
  const line = el("div", {class: "kv", style: "font-size:var(--fs-2);color:var(--text)"});
  const warn = el("div", {class: "pf-clamp"});
  const pos = w => `${(100 * (w - lo) / Math.max(1, hi - lo)).toFixed(1)}%`;
  const upd = v => {
    want = Math.round(Math.min(hi, Math.max(lo, +v || lo))); r.value = num.value = want; big.textContent = pfW(want);
    line.textContent = `Card ${c.index} power: ${pfW(c.limit_w)} \u2192 ${pfW(want)}. You can undo this anytime.`;
    const out = want < dflt * PF_RECO_LO - 0.5 || want > dflt + 0.5;
    warn.textContent = out ? (want > dflt ? "Above stock: hotter and louder; watch the temperature." : "Well below stock: noticeably slower.") : "";
    warn.style.color = out ? "var(--warn)" : "";
    $("#pf-pw-ok").disabled = Math.abs(want - (c.limit_w || 0)) < 0.5;
  };
  r.addEventListener("input", () => upd(r.value)); num.addEventListener("change", () => upd(num.value));
  const marks = el("div", {style: "position:relative;height:16px;margin:0 8px", class: "tiny"},
    el("span", {style: "position:absolute;left:0", text: (adv ? "lowest " : "") + pfW(lo)}),
    dflt > lo + (hi - lo) * 0.18 && dflt < hi - (hi - lo) * 0.18 ? el("span", {style: `position:absolute;left:${pos(dflt)};transform:translateX(-50%);color:var(--warn)`, text: "\u25B2 stock " + pfW(dflt)}) : null,
    el("span", {style: "position:absolute;right:0" + (Math.abs(dflt - hi) < 0.5 ? ";color:var(--warn)" : ""), text: (Math.abs(dflt - hi) < 0.5 ? "stock " : "") + pfW(hi)}));
  const can = s.settings.allow_gpu_control && !c.locked.length;
  pfFill($("#pf-pw-body"), 
    el("p", {class: "tiny", style: "margin:0 0 10px", text: "Lower = cooler, quieter and cheaper to run, a little slower. Higher = faster, hotter, louder. Try a value, watch \"tok/s per watt\" on the card, and undo if you do not like it."}),
    el("div", {class: "pf-row"}, r, num, big), marks,
    el("div", {class: "pf-sub", style: "margin:4px 0 10px"}, el("span", {text: `now ${pfW(c.limit_w)}`}), el("span", {text: adv ? `card allows ${pfW(c.min_w)}\u2013${pfW(c.max_w)}` : `recommended ${pfW(lo)}\u2013${pfW(hi)}`})),
    line, warn,
    c.locked.length ? el("div", {class: "alert warn"}, badge("warn", "kept free"), el("span", {text: c.locked.map(x => x.why).join("; ") + ". This card cannot be changed right now."})) : null,
    !s.settings.allow_gpu_control ? el("div", {class: "alert warn"}, badge("warn", "look only"), el("div", {}, el("div", {text: "Changing cards is switched off on this machine."}), pfHowToEnable())) : null,
    adv ? el("div", {class: "row", style: "margin-top:8px"}, el("button", {class: "b", onclick: () => copyText(pfCurl("POST", "/api/gpu/power", {uuid: c.uuid, watts: want, confirm: String(want)})), text: "Copy as curl"})) : null);
  upd(want);
  $("#pf-pw-ok").hidden = $("#pf-pw-default").hidden = !can;
  $("#pf-pw-ok").onclick = async () => {
    const r2 = await pfCall(() => api("/api/gpu/power", {body: {uuid: c.uuid, watts: want, confirm: String(want)}, timeout: 60000}));
    if (r2) { if (r2.ok) { toast(`Card ${c.index}: ${pfW(r2.target_w)}. Undo is at the top of the page.`); $("#pf-pw").close(); } else toast(`Card ${c.index} did not take it, so it was put back (${(r2.failure || {}).error})`, true); }
  };
  $("#pf-pw-default").onclick = async () => {
    const r2 = await pfCall(() => api("/api/gpu/power", {body: {uuid: c.uuid, watts: "default", confirm: "default"}, timeout: 60000}));
    if (r2 && r2.ok) { toast(`Card ${c.index}: back to stock ${pfW(r2.target_w)}`); $("#pf-pw").close(); }
  };
  $("#pf-pw").showModal();
}
async function pfReset(c) {
  if (!confirm(`Card ${c.index}: stock power limit and default clocks. You can undo this anytime. Go ahead?`)) return;
  pfCall(() => api("/api/gpu/reset", {body: {targets: [c.uuid], confirm: "reset"}}), `Card ${c.index} is back to its defaults`);
}

// ---- benchmark mode ----
function pfBench() {
  const s = PF.state;
  if (s.quiet && s.quiet.on) {
    if (!confirm("End benchmark mode? The cards that were slowed down get their limits back and auto-starts continue.")) return;
    pfCall(() => api("/api/gpu/quiet", {body: {on: false}}), "Benchmark mode ended");
    return;
  }
  const pick = new Set();
  const tiles = el("div", {class: "tiles"}, ...s.cards.map(c => { const cb = el("input", {type: "checkbox"}); cb.addEventListener("change", () => cb.checked ? pick.add(c.uuid) : pick.delete(c.uuid));
    return el("label", {class: "tile"}, cb, el("span", {class: "tick"}), el("span", {class: "t1", text: `#${c.index} ${pfShort(c.name)}`}), el("span", {class: "t2", text: c.users.length ? "busy" : "free"})); }));
  const cap = el("input", {type: "number", min: "30", max: "100", value: "70", style: "width:90px"});
  pfFill($("#pf-qd-body"), 
    el("p", {class: "tiny", style: "margin:0 0 10px", text: "For a fair speed test: the cards you tick are kept free (no changes, no server starts), auto-starts wait, and the other cards are slowed down so they make less heat and noise. Ending it puts everything back."}),
    el("h4", {style: "margin:0 0 6px", text: "Cards used by the test"}), tiles,
    el("div", {class: "pf-row", style: "margin-top:10px"}, el("span", {class: "tiny", text: "Other cards run at"}), cap, el("span", {class: "tiny", text: "% of stock power"}),
      s.settings.allow_gpu_control ? null : el("span", {class: "tiny", style: "color:var(--warn)", text: "(changing cards is switched off: they will only be kept free)"})),
    el("div", {class: "row", style: "margin-top:8px"}, el("button", {class: "b", onclick: () => copyText(pfCurl("POST", "/api/gpu/quiet", {on: true, bench: [...pick], cap_pct: +cap.value, confirm: "quiet"})), text: "Copy as curl"})));
  $("#pf-qd-ok").onclick = async () => {
    if (!pick.size) { toast("Tick the card(s) the speed test uses", true); return; }
    const r = await pfCall(() => api("/api/gpu/quiet", {body: {on: true, bench: [...pick], cap_pct: +cap.value, confirm: "quiet"}}));
    if (r) { $("#pf-qd").close(); toast("Benchmark mode is on" + (r.notes.length ? ": " + r.notes.join(" ") : "")); }
  };
  $("#pf-qd").showModal();
}

// ---- schedules, auto-start, history, import/export (Advanced) ----
function pfSchedRows() { if (!PF.sched) PF.sched = JSON.parse(JSON.stringify(PF.state.schedules)); return PF.sched; }
function pfRenderSched() {
  const s = PF.state, rows = PF.schedDirty ? pfSchedRows() : (PF.sched = JSON.parse(JSON.stringify(s.schedules)));
  $("#pf-sched-sub").textContent = s.settings.schedules ? "switch profiles at a time of day" : "saved, but they only run once schedules are switched on in control.json (gpu_schedules)";
  if (!s.profiles.length) { pfFill($("#pf-sched"), el("div", {class: "tiny", text: "Make a profile first."})); return; }
  pfFill($("#pf-sched"), ...(rows.length ? rows.map((r, i) => {
    const ps = el("select", {}, ...s.profiles.map(p => el("option", {value: p.id, text: p.name}))); ps.value = r.profile; ps.addEventListener("change", () => { r.profile = ps.value; PF.schedDirty = true; });
    const tm = el("input", {type: "time", value: r.at}); tm.addEventListener("change", () => { r.at = tm.value; PF.schedDirty = true; });
    const days = el("span", {class: "pf-days"}, el("button", {"aria-pressed": String(r.days === "daily"), onclick: () => { r.days = "daily"; PF.schedDirty = true; pfRenderSched(); }, text: "Every day"}),
      ...PF_DAYS.map((n, d) => el("button", {"aria-pressed": String(r.days !== "daily" && r.days.includes(d)), title: n, onclick: () => {
        let ds = r.days === "daily" ? [] : r.days.slice(); ds = ds.includes(d) ? ds.filter(x => x !== d) : ds.concat([d]); r.days = ds.length ? ds.sort() : "daily"; PF.schedDirty = true; pfRenderSched(); }, text: n[0]})));
    const en = el("input", {type: "checkbox"}); en.checked = r.enabled; en.addEventListener("change", () => { r.enabled = en.checked; PF.schedDirty = true; });
    return el("div", {class: "pf-sched"}, ps, tm, days, el("label", {class: "check"}, en, "on"), el("button", {class: "b danger", style: "height:30px", onclick: () => { rows.splice(i, 1); PF.schedDirty = true; pfRenderSched(); }, text: "\u2715"}));
  }) : [el("div", {class: "tiny", text: "No schedules. Example: Overnight at 01:00, Balanced at 08:00."})]));
}
function pfRenderSup() {
  const sv = PF.state.supervisor, s = PF.state;
  $("#pf-sup-sub").textContent = s.settings.autostart ? (sv && sv.paused ? "paused" : "on") : "off: switch on with gpu_autostart in control.json";
  $("#pf-sup-toggle").hidden = !sv || !s.settings.autostart; $("#pf-sup-toggle").textContent = sv && sv.paused ? "Resume" : "Pause";
  const lvl = {healthy: "ok", starting: "warn", delayed: "", pending: "", backoff: "warn", blocked: "warn", paused: "", failed: "bad", crash_loop: "bad", stopped: ""};
  const words = {healthy: "serving", starting: "starting", delayed: "waiting", pending: "queued", backoff: "retrying", blocked: "kept free", paused: "paused", failed: "gave up", crash_loop: "crash loop", stopped: "stopped by you"};
  pfFill($("#pf-sup"), ...(sv && sv.jobs.length ? sv.jobs.map(j => el("div", {class: "pf-job"}, badge(lvl[j.state] || "", words[j.state] || j.state),
    el("div", {style: "flex:1;min-width:0"}, el("b", {text: (s.servers.find(x => x.sid === j.server) || {name: j.server}).name}), el("span", {class: "tiny", text: ` · ${(s.profiles.find(p => p.id === j.profile) || {name: j.profile}).name} · try ${j.attempts + 1}${j.next_in_s != null ? ` · next in ${Math.round(j.next_in_s)} s` : ""}`}),
      j.error ? el("div", {class: "tiny", style: "overflow-wrap:anywhere", text: j.error}) : null))) : [el("div", {class: "tiny", text: "Nothing queued. Servers listed in a profile start when you apply it."})]));
}
async function pfLoadAudit(force) {
  if (!force && Date.now() - PF.auditAt < 15000) return;
  PF.auditAt = Date.now();
  try { PF.audit = (await api("/api/gpu/audit?limit=200")).entries; } catch (e) { return; }
  const tgt = t => !t ? "" : t.index != null ? `#${t.index}${t.field ? " " + t.field : ""}` : t.profile ? t.profile : t.cards ? pfIdx(t.cards) : t.server ? t.server : JSON.stringify(t).slice(0, 80);
  const val = (b, a) => (b == null && a == null) ? "" : `${JSON.stringify(b)} \u2192 ${JSON.stringify(a)}`.slice(0, 140);
  pfFill($("#pf-audit"), ...(PF.audit.length ? PF.audit.slice().reverse().map(r => el("tr", {},
    el("td", {text: new Date(r.ts * 1000).toLocaleString()}), el("td", {text: r.who}), el("td", {class: "mono", text: r.action + (r.dry_run ? " (preview)" : "")}),
    el("td", {class: "mono", title: JSON.stringify(r.target || {}), text: tgt(r.target) + (r.action.endsWith(".step") || r.action.endsWith(".rollback") ? "  " + val(r.before, r.after) : "")}),
    el("td", {}, r.ok ? badge("ok", "ok") : badge("bad", r.error ? r.error.slice(0, 80) : "failed")))) : [el("tr", {}, el("td", {colspan: "5", class: "empty", text: "No changes yet."}))]));
}
async function pfExport() {
  try { const r = await fetch("/api/gpu/export"); if (!r.ok) throw new Error("HTTP " + r.status); download("pxa-gpu-profiles.json", await r.text()); }
  catch (e) { toast(pfFriendly(e), true); }
}
async function pfImport() {
  let data; try { data = JSON.parse($("#pf-imp-text").value); } catch (e) { toast("That is not valid JSON: " + e.message, true); return; }
  const r = await pfCall(() => api("/api/gpu/import", {body: {data, replace: $("#pf-imp-replace").checked}}));
  if (r) { $("#pf-imp").close(); toast(`Imported ${r.profiles} profile(s) and ${r.schedules} schedule(s)`); }
}
