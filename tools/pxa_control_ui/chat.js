// PXA Control: the Chat tab's classic chat, as a small core (window.PXAChat) that plugins extend.
// Loaded after the page's inline script: it uses that script's helpers ($, el, S, store, api, toast, fmtNum, withQ,
// qsTarget, pollStatus, and the thinking controls TK / thinkVal). The hook API is described in CHAT-PLUGINS.md.
// Plugins are the chat-*.js files loaded right after this one; each owns its own file (and its own chat-*.css).
(function () {
  "use strict";
  if (typeof el !== "function" || !document.getElementById("p-chat")) return;

  // ---------------- markdown (the default renderer; a plugin may replace it with setRenderer) ----------------
  function esc(s) { return s.replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }
  function inline(s) {
    // s is already escaped
    return s.replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  }
  function md(src) {
    const out = [], parts = src.split(/```/);
    parts.forEach((p, i) => {
      if (i % 2) { const nl = p.indexOf("\n"); const body = nl >= 0 ? p.slice(nl + 1) : p; out.push("<pre><code>" + esc(body.replace(/\n$/, "")) + "</code></pre>"); return; }
      let list = null, para = [];
      const flush = () => { if (para.length) { out.push("<p>" + inline(esc(para.join("\n"))).replace(/\n/g, "<br>") + "</p>"); para = []; } };
      const endList = () => { if (list) { out.push(`</${list}>`); list = null; } };
      for (const line of p.split("\n")) {
        let m;
        if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { flush(); endList(); out.push(`<h4>${inline(esc(m[2]))}</h4>`); }
        else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { flush(); if (list !== "ul") { endList(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(esc(m[1]))}</li>`); }
        else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { flush(); if (list !== "ol") { endList(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(esc(m[1]))}</li>`); }
        else if (!line.trim()) { flush(); endList(); }
        else { endList(); para.push(line); }
      }
      flush(); endList();
    });
    return out.join("");
  }

  // Shared by the Classic (chat.js) and Assistant (agent.js) views: functions run on the DOM the renderer produced
  // (syntax highlighting, math). fn(root, {view: "classic"|"assistant", live, msg}). See CHAT-PLUGINS.md.
  window.PXAMarkdown = window.PXAMarkdown || (function () {
    const P = [];
    return {add(fn) { if (typeof fn === "function" && !P.includes(fn)) P.push(fn); },
            remove(fn) { const i = P.indexOf(fn); if (i >= 0) P.splice(i, 1); },
            process(root, ctx) { for (const fn of P.slice()) { try { fn(root, ctx || {}); } catch (e) { console.error("PXAMarkdown post-processor failed:", e); } } },
            get count() { return P.length; }};
  })();

  // ---------------- event bus ----------------
  const EVENTS = ["init", "beforeSend", "delta", "messageDone", "error", "convChange", "render"];
  const H = {}; EVENTS.forEach(e => { H[e] = []; });
  let inited = false;
  function on(evt, fn) {
    if (!H[evt]) throw new Error("PXAChat: unknown event " + evt + " (known: " + EVENTS.join(", ") + ")");
    H[evt].push(fn);
    if (evt === "init" && inited) setTimeout(() => call(fn, "init", [C]), 0);
    return () => { const i = H[evt].indexOf(fn); if (i >= 0) H[evt].splice(i, 1); };
  }
  function call(fn, evt, args) {
    try { return fn(...args); } catch (e) { console.error("PXAChat " + evt + " handler failed:", e); }
  }
  function emit(evt, ...args) { for (const fn of H[evt].slice()) call(fn, evt, args); }
  async function emitAsync(evt, ...args) {   // beforeSend: handlers may be async; they run one after another
    for (const fn of H[evt].slice()) { try { await fn(...args); } catch (e) { console.error("PXAChat " + evt + " handler failed:", e); } }
  }

  // ---------------- storage: in memory + one localStorage blob; a plugin may replace C.store ----------------
  const BLOB = "pxa-control.chat.convs", KEEP = 100;
  function memoryStore() {
    const m = new Map();
    try { const a = JSON.parse(localStorage.getItem(BLOB) || "[]"); if (Array.isArray(a)) a.forEach(c => c && c.id && m.set(c.id, c)); } catch (e) { /* private mode / bad blob */ }
    function persist() {
      let all = Array.from(m.values()).sort((a, b) => (b.updated || 0) - (a.updated || 0)).slice(0, KEEP);
      for (;;) {   // a full quota drops the oldest conversations from the blob, never the page
        try { localStorage.setItem(BLOB, JSON.stringify(all)); return; } catch (e) { if (all.length <= 1) return; all = all.slice(0, Math.ceil(all.length / 2)); }
      }
    }
    return {
      get(id) { const c = m.get(id); return c ? JSON.parse(JSON.stringify(c)) : null; },
      put(conv) { m.set(conv.id, JSON.parse(JSON.stringify(conv))); persist(); return true; },
      list() { return Array.from(m.values()).map(c => ({id: c.id, title: c.title, created: c.created, updated: c.updated, count: (c.messages || []).length}))
        .sort((a, b) => (b.updated || 0) - (a.updated || 0)); },
      delete(id) { const had = m.delete(id); persist(); return had; },
    };
  }

  // ---------------- model ----------------
  const uid = p => (p || "m") + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  function makeConv() { const t = Date.now(); return {id: uid("c"), title: "New chat", created: t, updated: t, settings: {}, messages: []}; }
  function makeMsg(role, content, parentId) { return {id: uid("m"), role, content: content || "", reasoning: "", meta: {}, parentId: parentId || null}; }
  // what goes to the server: user turns, and assistant turns that produced something (a failed reply is left out)
  function history(conv) {
    return conv.messages.filter(m => m.role === "user" || m.role === "system" || (m.role === "assistant" && (m.content || m.reasoning)) || m.role === "tool")
      .map(m => { const o = {role: m.role, content: m.content}; if (m.parts) o.content = m.parts; return o; });
  }

  // ---------------- registries ----------------
  const R = {actions: [], sections: [], panels: [], buttons: [], slash: new Map(), renderer: md};

  // ---------------- DOM ----------------
  const D = {page: $("#p-chat"), msgs: $("#msgs"), input: $("#c-in"), send: $("#c-send"), stop: $("#c-stop"),
             cols: $("#p-chat .cols.chat"), side: $("#chat-sidebar"), sideToggle: $("#chat-side-toggle"),
             tools: $("#chat-composer-tools"), sections: $("#chat-settings-sections")};
  const views = new Map();   // msg id -> {m, body, think, meta, actions}
  const scrollEnd = () => { D.msgs.scrollTop = D.msgs.scrollHeight; };
  function renderBody(msg, v, live) {
    if (msg.role === "user") v.body.textContent = msg.content;
    else {
      let h = ""; try { h = R.renderer(msg.content || "", msg); } catch (e) { console.error("PXAChat renderer failed:", e); h = md(msg.content || ""); }
      v.body.innerHTML = h; window.PXAMarkdown.process(v.body, {view: "classic", live: !!live, msg});
    }
  }
  function addMsgView(msg) {
    $("#msgs-empty") && $("#msgs-empty").remove();
    const body = el("div", {class: "body"}), think = el("div", {class: "think", hidden: true}), meta = el("div", {class: "meta"}),
      actions = el("div", {class: "msg-actions"});
    const m = el("div", {class: "msg " + msg.role, "data-id": msg.id}, think, body, meta, actions);
    const v = {m, body, think, meta, actions}; views.set(msg.id, v);
    renderBody(msg, v);
    if (msg.reasoning) { think.hidden = false; think.textContent = msg.reasoning; }
    if (msg.meta && msg.meta.text) meta.textContent = msg.meta.text;
    if (msg.meta && msg.meta.error) m.style.borderColor = "var(--bad)";
    D.msgs.append(m); scrollEnd();
    return v;
  }
  function renderActions(msg) {
    const v = views.get(msg.id); if (!v) return;
    v.actions.replaceChildren(...R.actions.filter(a => { try { return !a.when || a.when(msg, C.conv); } catch (e) { return false; } })
      .map(a => el("button", {class: "b msg-act", type: "button", "data-act": a.id, title: a.title || a.label, "aria-label": a.label,
        onclick: () => { try { const r = a.run(msg, C.conv, v); if (r && r.catch) r.catch(e => toast(e.message || String(e), true)); } catch (e) { toast(e.message || String(e), true); } }},
        a.icon ? el("span", {class: "ic", "aria-hidden": "true", text: a.icon}) : null, a.icon && a.iconOnly ? null : a.label)));
  }
  function finishView(msg) { renderActions(msg); const v = views.get(msg.id); if (v) emit("render", v.m, msg); }
  function emptyView(title, sub) {
    D.msgs.replaceChildren(el("div", {class: "empty center", id: "msgs-empty"}, el("b", {text: title}), el("span", {text: sub})));
  }
  function renderConv() {
    views.clear();
    if (!C.conv.messages.length) { emptyView("New chat", "Say something."); return; }
    D.msgs.replaceChildren();
    for (const msg of C.conv.messages) { addMsgView(msg); finishView(msg); }
  }
  function setBusy(on) { D.send.disabled = on; D.stop.disabled = !on; }
  function sideLayout() {
    const has = R.panels.length > 0, open = has && C.ui.sideOpen;
    D.side.hidden = !open; D.cols.classList.toggle("has-side", open);
    if (D.sideToggle) { D.sideToggle.hidden = !has; D.sideToggle.setAttribute("aria-expanded", open ? "true" : "false"); }
  }

  // ---------------- one robust SSE reader for OpenAI-style streams ----------------
  // onDelta(json) for every data event; resolves {response, thinkNotes, timings, usage}. Throws on HTTP errors and on an
  // error object sent mid-stream. Handles chunk boundaries anywhere, CRLF, "data:" with or without a space, multi-line
  // events, comments, [DONE], and a last event without its trailing blank line.
  async function streamCompletion(body, onDelta, signal) {
    const r = await fetch(withQ("/api/engine/v1/chat/completions", qsTarget(S.target)), {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), signal});
    if (!r.ok) { let j = null; try { j = await r.json(); } catch (e) { /* */ } throw new Error((j && (j.error && (j.error.message || j.error))) || "HTTP " + r.status); }
    let thinkNotes = "";
    try { const th = JSON.parse(r.headers.get("X-PXA-Thinking") || "null"); if (th && th.notes && th.notes.length) thinkNotes = th.notes.join("; "); } catch (e) { /* */ }
    const out = {response: r, thinkNotes, timings: null, usage: null, done: false};
    const ctype = r.headers.get("Content-Type") || "";
    if (!/event-stream/.test(ctype)) {   // a server that ignored stream:true answers with one JSON object
      const j = await r.json();
      if (j.error) throw new Error(j.error.message || String(j.error));
      const ch = j.choices && j.choices[0], m = ch && ch.message;
      if (j.timings) out.timings = j.timings; if (j.usage) out.usage = j.usage;
      if (m) onDelta({choices: [{index: 0, delta: {content: m.content || "", reasoning_content: m.reasoning_content || ""}, finish_reason: ch.finish_reason}], timings: j.timings, usage: j.usage});
      out.done = true; return out;
    }
    const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "", data = [];
    const dispatch = () => {
      if (!data.length) return; const s = data.join("\n"); data = [];
      if (s.trim() === "[DONE]") { out.done = true; return; }
      let j; try { j = JSON.parse(s); } catch (e) { return; }
      if (j && j.error) { const e = new Error(j.error.message || (typeof j.error === "string" ? j.error : JSON.stringify(j.error))); e.serverError = j.error; throw e; }
      if (j.timings) out.timings = j.timings; if (j.usage) out.usage = j.usage;
      onDelta(j);
    };
    const line = l => {
      if (l.endsWith("\r")) l = l.slice(0, -1);
      if (l === "") { dispatch(); return; }
      if (l[0] === ":") return;                                   // comment / keep-alive
      const c = l.indexOf(":"), f = c < 0 ? l : l.slice(0, c);
      if (f !== "data") return;                                   // event:, id:, retry:
      let v = c < 0 ? "" : l.slice(c + 1); if (v[0] === " ") v = v.slice(1);
      data.push(v);
    };
    for (;;) {
      const {value, done} = await rd.read();
      if (done) { buf += dec.decode(); break; }
      buf += dec.decode(value, {stream: true});
      let i; while ((i = buf.indexOf("\n")) >= 0) { const l = buf.slice(0, i); buf = buf.slice(i + 1); line(l); }
    }
    if (buf) line(buf); dispatch();
    return out;
  }

  // ---------------- send / generate ----------------
  async function generate(conv, parentId) {
    const msg = makeMsg("assistant", "", parentId); conv.messages.push(msg);
    const ui = addMsgView(msg); ui.meta.textContent = "thinking ...";
    const msgs = []; const sys = $("#c-sys").value.trim(); if (sys) msgs.push({role: "system", content: sys});
    msgs.push(...history({messages: conv.messages.slice(0, -1)}));
    const body = {messages: msgs, stream: true, temperature: +$("#c-temp").value, max_tokens: +$("#c-max").value || 1024,
                  stream_options: {include_usage: true}};
    if (typeof TK !== "undefined" && TK.c.prof && TK.c.prof.supported) body.pxa_thinking = thinkVal("c");
    const ac = new AbortController(); S.abort = ac; setBusy(true);
    let n = 0, res = null; const t0 = performance.now(); let tFirst = null;
    try {
      await emitAsync("beforeSend", body, conv);
      res = await streamCompletion(body, j => {
        const d = j.choices && j.choices[0] && j.choices[0].delta;
        if (d) {
          if (d.reasoning_content) { msg.reasoning += d.reasoning_content; ui.think.hidden = false; ui.think.textContent = msg.reasoning; n++; }
          if (d.content) { msg.content += d.content; n++; renderBody(msg, ui, true); }
          if (tFirst == null && (d.content || d.reasoning_content)) tFirst = performance.now();
          if (j.choices[0].finish_reason) msg.meta.finish_reason = j.choices[0].finish_reason;
          ui.meta.textContent = `${n} tokens ...`; scrollEnd();
        }
        emit("delta", j, msg);
      }, ac.signal);
    } catch (e) {
      if (e.name === "AbortError") msg.meta.stopped = true;
      else {
        msg.meta.error = e.message; ui.meta.textContent = "error: " + (/no server/.test(e.message) ? "no server is running: start one on the Launch tab" : e.message);
        msg.meta.text = ui.meta.textContent; ui.m.style.borderColor = "var(--bad)"; emit("error", e, msg);
      }
    } finally {
      S.abort = null; setBusy(false);
      if (msg.content || msg.reasoning) {
        const timings = res && res.timings, usage = res && res.usage, secs = ((performance.now() - (tFirst || t0)) / 1000);
        if (timings) msg.meta.timings = timings; if (usage) msg.meta.usage = usage;
        ui.meta.textContent = timings ? `${fmtNum(timings.predicted_per_second)} t/s decode · ${fmtNum(timings.prompt_per_second)} t/s prefill · ${timings.predicted_n} tokens`
          : `${fmtNum(n / Math.max(secs, 1e-3))} t/s (measured in the browser) · ${usage ? usage.completion_tokens : n} tokens`;
        if (res && res.thinkNotes) ui.meta.textContent += " · thinking: " + res.thinkNotes;
        if (msg.meta.error) { ui.meta.textContent += " · error: " + msg.meta.error; ui.m.style.borderColor = "var(--bad)"; }
        msg.meta.text = ui.meta.textContent;
      }
      msg.meta.model = targetLabel(S.target); msg.meta.ts = Date.now();
      conv.updated = Date.now(); save(conv);
      if (msg.content && window.PXAMarkdown.count) renderBody(msg, ui, false);
      finishView(msg); emit("messageDone", msg, conv);
    }
    return msg;
  }
  async function send(text, opts) {
    opts = opts || {};
    const fromBox = text == null; if (fromBox) text = D.input.value.trim(); else text = String(text).trim();
    if (!text || S.abort) return null;
    const sm = text.match(/^\/(\S+)(?:\s+([\s\S]*))?$/), cmd = sm && R.slash.get(sm[1]);
    if (cmd) {   // a registered slash command runs instead of sending; a string it returns is sent instead
      if (fromBox) D.input.value = "";
      let r; try { r = await cmd(sm[2] || "", C.conv); } catch (e) { toast(e.message || String(e), true); return null; }
      if (typeof r !== "string" || !r.trim()) return null; text = r.trim();
    } else if (fromBox) D.input.value = "";
    const conv = C.conv, last = conv.messages[conv.messages.length - 1];
    const u = makeMsg("user", text, last && last.id); if (opts.parts) u.parts = opts.parts; if (opts.meta) Object.assign(u.meta, opts.meta);
    conv.messages.push(u);
    if (conv.title === "New chat") conv.title = text.replace(/\s+/g, " ").slice(0, 60);
    addMsgView(u); finishView(u);
    return generate(conv, u.id);
  }
  async function regenerate(msgId) {
    if (S.abort) return null;
    const conv = C.conv, i = conv.messages.findIndex(m => m.id === msgId); if (i < 0) return null;
    const keep = conv.messages[i].role === "assistant" ? i : i + 1;   // an assistant turn is replaced; after a user turn, a new reply
    for (const m of conv.messages.slice(keep)) { const v = views.get(m.id); if (v) v.m.remove(); views.delete(m.id); }
    conv.messages.length = keep;
    const prev = conv.messages[keep - 1];
    return generate(conv, prev && prev.id);
  }
  function stop() { if (S.abort) S.abort.abort(); }
  function save(conv) {
    if (!conv.messages.length) return;
    try { const p = C.store.put(conv); if (p && p.catch) p.catch(e => console.warn("PXAChat store.put:", e)); } catch (e) { console.warn("PXAChat store.put:", e); }
  }
  function newChat() {
    stop(); C.conv = makeConv(); S.chat = C.conv.messages; renderConv(); emit("convChange", C.conv); return C.conv;
  }
  async function loadConv(id) {
    const c = await C.store.get(id); if (!c) return null;
    stop(); c.messages = c.messages || []; c.settings = c.settings || {}; C.conv = c; S.chat = c.messages;
    renderConv(); emit("convChange", c); return c;
  }

  // ---------------- public core ----------------
  const C = {
    version: 1,
    conv: makeConv(),
    get target() { return S.target; },
    get busy() { return !!S.abort; },
    ui: {sideOpen: !matchMedia("(max-width: 980px)").matches},
    store: memoryStore(),
    on, emit,
    send, regenerate, stop, newChat, loadConv, streamCompletion,
    history: conv => history(conv || C.conv),
    md, esc, inline,
    render: (text, msg) => R.renderer(text, msg),
    rerender(msg) { const v = views.get(msg.id); if (v) { renderBody(msg, v); finishView(msg); } },
    view: msgId => views.get(msgId) || null,
    addMarkdownPostProcessor(fn) { window.PXAMarkdown.add(fn); },
    setRenderer(fn) { R.renderer = typeof fn === "function" ? fn : md; for (const m of C.conv.messages) if (views.get(m.id)) renderBody(m, views.get(m.id)); },
    addMessageAction(a) {
      if (!a || !a.id || typeof a.run !== "function") throw new Error("addMessageAction needs {id, label, run}");
      R.actions = R.actions.filter(x => x.id !== a.id).concat([a]); for (const m of C.conv.messages) renderActions(m);
    },
    addSettingsSection(id, title, renderFn) {
      const old = $("#chat-sec-" + id); if (old) old.remove();
      const box = el("div", {class: "chat-sec-body"}), sec = el("div", {class: "chat-sec", id: "chat-sec-" + id}, title ? el("h4", {text: title}) : null, box);
      D.sections.append(sec); R.sections.push(id);
      try { renderFn && renderFn(box, C); } catch (e) { console.error("PXAChat section " + id + ":", e); }
      return box;
    },
    addSidebarPanel(id, renderFn) {
      const old = $("#chat-panel-" + id); if (old) old.remove();
      const box = el("div", {class: "chat-panel", id: "chat-panel-" + id}); D.side.append(box);
      if (!R.panels.includes(id)) R.panels.push(id); sideLayout();
      try { renderFn && renderFn(box, C); } catch (e) { console.error("PXAChat panel " + id + ":", e); }
      return box;
    },
    addComposerButton(b) {
      if (!b || !b.id || typeof b.onClick !== "function") throw new Error("addComposerButton needs {id, label, onClick}");
      const old = $("#chat-tool-" + b.id); if (old) old.remove();
      const btn = el("button", {class: "b chat-tool", type: "button", id: "chat-tool-" + b.id, title: b.title || b.label, "aria-label": b.label,
        onclick: e => { try { const r = b.onClick(C.conv, e); if (r && r.catch) r.catch(x => toast(x.message || String(x), true)); } catch (x) { toast(x.message || String(x), true); } }},
        b.icon ? el("span", {class: "ic", "aria-hidden": "true", text: b.icon}) : null, b.icon && b.iconOnly ? null : b.label);
      D.tools.append(btn); return btn;
    },
    addSlashCommand(name, fn) { R.slash.set(String(name).replace(/^\//, ""), fn); },
    slashCommands: () => Array.from(R.slash.keys()),
    toggleSidebar(open) { C.ui.sideOpen = open == null ? !C.ui.sideOpen : !!open; sideLayout(); },
  };
  S.chat = C.conv.messages;
  window.PXAChat = C;

  // ---------------- wiring (unchanged behaviour of the classic chat) ----------------
  D.send.addEventListener("click", () => send());
  D.stop.addEventListener("click", stop);
  D.input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  $("#c-clear").addEventListener("click", newChat);
  if (D.sideToggle) D.sideToggle.addEventListener("click", () => C.toggleSidebar());
  $("#c-attach-b").addEventListener("click", async () => {
    try { const d = await api("/api/attach", {body: {port: +$("#c-attach").value || 0}}); toast(d.attach_port ? "attached to :" + d.attach_port : "detached"); pollStatus(); } catch (e) { toast(e.message, true); }
  });
  ["c-sys", "c-temp", "c-max"].forEach(id => { const v = store.get(id); if (v != null) $("#" + id).value = v; $("#" + id).addEventListener("change", e => store.set(id, e.target.value)); });
  sideLayout();

  // init fires once, after every plugin script has run (they load after this file, before DOMContentLoaded)
  const fire = () => { if (inited) return; inited = true; emit("init", C); };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fire); else setTimeout(fire, 0);
})();
