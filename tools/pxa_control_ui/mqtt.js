"use strict";
/* PXA Control: the Home Assistant panel (v3.1). Publish the Live numbers to an MQTT broker with Home
   Assistant discovery, so HA grows the entities by itself. Loaded by index.html after its own script, so
   it uses that script's helpers ($, el, api, toast). Reads GET /api/mqtt, saves with POST /api/mqtt/settings
   and proves a broker with POST /api/mqtt/test. The broker password is never rendered back: the server
   answers `password_set` and the field only sends a value when the user types one. */

const MQ = {status: null, open: false, timer: null, dirty: false};

function mqRow(label, node, hint) {
  return el("label", {class: "f"}, label, node, hint ? el("span", {class: "tiny", text: hint}) : null);
}

function mqInput(id, value, attrs) {
  const i = el("input", Object.assign({type: "text", id, value: value == null ? "" : String(value),
    autocomplete: "off", spellcheck: "false"}, attrs || {}));
  i.addEventListener("input", () => { MQ.dirty = true; });
  return i;
}

function mqCheck(id, label, checked, hint) {
  const i = el("input", {type: "checkbox", id});
  i.checked = !!checked;
  i.addEventListener("change", () => { MQ.dirty = true; });
  return el("label", {class: "check"}, i, " " + label, hint ? el("span", {class: "tiny", text: " " + hint}) : null);
}

function mqForm() {
  const s = (MQ.status && MQ.status.settings) || {};
  const host = mqInput("mq-host", s.host, {placeholder: "192.168.1.10 or a hostname"});
  const port = mqInput("mq-port", s.port == null ? 1883 : s.port, {type: "number", min: "1", max: "65535"});
  const user = mqInput("mq-user", s.user, {placeholder: "empty if the broker allows anonymous"});
  const pass = mqInput("mq-pass", "", {type: "password",
    placeholder: s.password_set ? "saved (type to replace)" : "none"});
  const base = mqInput("mq-base", s.base || "pxa");
  const interval = mqInput("mq-interval", s.interval == null ? 15 : s.interval, {type: "number", min: "2", max: "3600"});
  const tls = mqCheck("mq-tls", "Encrypted (TLS)", s.tls);
  const disc = mqCheck("mq-discovery", "Home Assistant discovery", s.discovery !== false,
    "lets HA create the entities by itself");
  const enabled = mqCheck("mq-enabled", "Publish to Home Assistant", s.enabled);
  return [enabled, host, port, user, pass, tls, base, interval, disc];
}

function mqBody() {
  const body = {host: $("#mq-host").value.trim(), port: +$("#mq-port").value || 1883,
    user: $("#mq-user").value.trim(), tls: $("#mq-tls").checked, base: $("#mq-base").value.trim(),
    interval: +$("#mq-interval").value || 15, discovery: $("#mq-discovery").checked,
    enabled: $("#mq-enabled").checked};
  const pw = $("#mq-pass").value;                    // only sent when typed: an untouched field keeps the saved one
  if (pw) body.password = pw;
  return body;
}

function mqStatusLine(st) {
  st = st || MQ.status || {};
  if (st.available === false) {
    return el("div", {class: "tiny", style: "color:var(--warn)"},
      "The MQTT module is missing from this PXA Control: " + (st.error || "unknown reason"));
  }
  const s = st.settings || {};
  if (!s.enabled) return el("div", {class: "tiny", text: "Off. Nothing is sent anywhere until you turn this on."});
  if (!s.host) return el("div", {class: "tiny", style: "color:var(--warn)", text: "On, but no broker host is set yet."});
  const state = st.state || "starting";
  const bits = [];
  if (st.error) bits.push(st.error);
  if (st.published) bits.push(`${st.published} value(s) published`);
  if (st.entities) bits.push(`${st.entities} entities`);
  if (st.reconnects) bits.push(`${st.reconnects} reconnect(s)`);
  if (st.last_ok) bits.push("last update " + new Date(st.last_ok * 1000).toLocaleTimeString());
  const good = state === "connected";
  return el("div", {class: "tiny", style: good ? "" : "color:var(--warn)"},
    `${good ? "Connected" : state} to ${s.host}:${s.port}${bits.length ? " · " + bits.join(" · ") : ""}`);
}

function mqRender() {
  const box = $("#mq-panel");
  if (!box) return;
  const st = MQ.status || {};
  const s = st.settings || {};
  const save = el("button", {class: "b pri", type: "button", id: "mq-save", text: "Save"});
  const test = el("button", {class: "b", type: "button", id: "mq-test", text: "Test connection"});
  box.replaceChildren(...mqForm(),
    el("div", {class: "row", style: "margin-top:10px"}, save, test),
    el("div", {id: "mq-status", style: "margin-top:8px"}, mqStatusLine(st)),
    el("div", {class: "tiny", style: "margin-top:8px"},
      "The values are the same ones the Live tab shows: speed, slots, KV cache, per-card temperature, VRAM and",
      " utilisation, and the expert map. Nothing leaves your network. The broker password is stored with your other",
      " PXA Control settings and is never shown back to this page."));
  save.addEventListener("click", mqSave);
  test.addEventListener("click", mqTest);
  MQ.dirty = false;
  void s;
}

async function mqLoad(force) {
  if (!force && MQ.open && Date.now() - (MQ.at || 0) < 3000) return;
  MQ.at = Date.now();
  try { MQ.status = await api("/api/mqtt"); }
  catch (e) { MQ.status = {available: false, error: e.message}; }
  if (!MQ.dirty) mqRender();                 // never overwrite a field the user is editing
}

async function mqSave() {
  const b = $("#mq-save");
  b.disabled = true;
  try {
    MQ.status = await api("/api/mqtt/settings", {body: mqBody()});
    MQ.dirty = false;
    mqRender();
    toast((MQ.status.settings || {}).enabled ? "Home Assistant publishing is on" : "Home Assistant publishing is off");
  } catch (e) { toast(e.message, true); }
  finally { b.disabled = false; }
}

async function mqTest() {
  const b = $("#mq-test");
  b.disabled = true;
  b.textContent = "Testing…";
  try {
    const r = await api("/api/mqtt/test", {body: mqBody(), timeout: 20000});
    if (r.ok) toast(`Broker answered in ${r.ms} ms${r.tls ? " (TLS)" : ""}`);
    else toast("Not connected: " + (r.error || "the broker refused"), true);
    const line = $("#mq-status");
    if (line) line.replaceChildren(el("div", {class: "tiny", style: r.ok ? "" : "color:var(--warn)"},
      r.ok ? `Test passed: ${r.detail}, answered in ${r.ms} ms. Nothing was published.`
           : `Test failed: ${r.error}. Check the host, port and password; the broker must be reachable from this machine.`));
  } catch (e) { toast(e.message, true); }
  finally { b.disabled = false; b.textContent = "Test connection"; }
}

function mqOpen() { MQ.open = true; mqLoad(true); if (!MQ.timer) MQ.timer = setInterval(() => { if (!document.hidden && MQ.open && !MQ.dirty) mqLoad(); }, 5000); }
function mqClose() { MQ.open = false; clearInterval(MQ.timer); MQ.timer = null; }

(function () {
  const d = $("#adv-settings");
  if (!d) return;
  if (d.open) mqOpen();
  d.addEventListener("toggle", () => (d.open ? mqOpen() : mqClose()));
})();
