// PXA Control chat agent (tools/pxa_chat, v3.1). The "Assistant" view of the Chat tab, laid out like a modern
// assistant: a sidebar of saved chats, a wide centred conversation (right-aligned user bubbles, full-width
// Markdown replies streamed token by token), compact tool rows that expand, a collapsed "Thought for 3s" row,
// inline approval cards, and a bottom composer with the server picker and a mode chip. Advanced adds a
// right-side settings drawer. Vanilla JS on the page's own helpers ($, $$, el, api, toast, store, segInit).
// If the backend package is missing (/api/chat/presets 404) this file does nothing: Chat stays as it was.
"use strict";

// md:begin -- the safe Markdown renderer: everything is HTML-escaped first, then only our own tags are added.
// Headings, lists (one nesting level), tables, block quotes, rules, fenced code (with a language label and a
// copy button), inline code, bold, italic, strikethrough, http(s)/mailto links. Streaming-safe: an unclosed
// fence renders as a code block so far.
const pxaMd = (function () {
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const unesc = s => s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  function safeUrl(u) {
    const raw = unesc(u).trim();
    return /^(https?:\/\/|mailto:)/i.test(raw) ? esc(raw) : null;
  }
  function inline(s) {                      // s is already escaped
    const keep = [];
    const hold = h => "\u0000" + (keep.push(h) - 1) + "\u0000";
    s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (m, t, c) => hold("<code>" + c.replace(/^ (.*) $/, "$1") + "</code>"));
    s = s.replace(/!?\[([^\]\n]{1,300})\]\(([^)\s]{1,2000})\)/g, (m, t, u) => {
      const h = safeUrl(u);
      return h ? hold(`<a href="${h}" target="_blank" rel="noopener noreferrer">${t}</a>`) : m;
    });
    s = s.replace(/(^|[\s(])((?:https?:\/\/)[^\s<]+[^\s<.,;:!?)'"&])/g, (m, pre, u) => {
      const h = safeUrl(u);
      return h ? pre + hold(`<a href="${h}" target="_blank" rel="noopener noreferrer">${u}</a>`) : m;
    });
    s = s.replace(/\*\*([^\s*](?:[\s\S]*?[^\s*])?)\*\*/g, "<strong>$1</strong>").replace(/__([^\s_](?:[\s\S]*?[^\s_])?)__/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^\s*](?:[^*]*?[^\s*])?)\*(?!\*)/g, "$1<em>$2</em>").replace(/(^|[^_\w])_([^\s_](?:[^_]*?[^\s_])?)_(?!\w)/g, "$1<em>$2</em>");
    s = s.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    return s.replace(/\u0000(\d+)\u0000/g, (m, i) => keep[+i]);
  }
  const cells = line => {
    let t = line.trim();
    if (t.startsWith("|")) t = t.slice(1);
    if (t.endsWith("|") && !t.endsWith("\\|")) t = t.slice(0, -1);
    return t.split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, "|"));
  };
  const isSep = line => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line) && line.includes("-");
  function codeBlock(lang, body) {
    const l = esc((lang || "").trim().split(/\s+/)[0].slice(0, 24));
    return `<div class="ag-code"><div class="ag-code-h"><span>${l || "code"}</span><button type="button" class="ag-copy" data-agcopy="1" aria-label="copy code">Copy</button></div><pre><code${l ? ` class="lang-${l}"` : ""}>${esc(body)}</code></pre></div>`;
  }
  function render(src) {
    const lines = String(src || "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let para = [], list = [];               // list: stack of {tag, indent}
    const flush = () => { if (para.length) { out.push("<p>" + inline(esc(para.join("\n"))).replace(/\n/g, "<br>") + "</p>"); para = []; } };
    const endLists = (indent) => { while (list.length && (indent == null || list[list.length - 1].indent > indent)) out.push(`</li></${list.pop().tag}>`); };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      let m;
      if ((m = line.match(/^\s{0,3}(`{3,}|~{3,})\s*([^`]*)$/))) {          // fenced code (unclosed = to the end)
        flush(); endLists();
        const fence = m[1], body = [];
        i++;
        while (i < lines.length && !new RegExp("^\\s{0,3}" + fence[0] + "{" + fence.length + ",}\\s*$").test(lines[i])) body.push(lines[i++]);
        out.push(codeBlock(m[2], body.join("\n")));
        continue;
      }
      if (!line.trim()) { flush(); if (list.length && !(lines[i + 1] || "").match(/^\s*([-*+]|\d+[.)])\s+/)) endLists(); continue; }
      if ((m = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/))) { flush(); endLists(); const n = Math.min(6, m[1].length + 2); out.push(`<h${n}>${inline(esc(m[2]))}</h${n}>`); continue; }
      if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); endLists(); out.push("<hr>"); continue; }
      if (line.includes("|") && isSep(lines[i + 1] || "") && !list.length) {
        flush();
        const head = cells(line), al = cells(lines[i + 1]).map(c => c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "");
        const row = (cs, tag) => "<tr>" + head.map((_, j) => `<${tag}${al[j] ? ` style="text-align:${al[j]}"` : ""}>${inline(esc(cs[j] || ""))}</${tag}>`).join("") + "</tr>";
        const body = [];
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) body.push(row(cells(lines[i++]), "td"));
        i--;
        out.push(`<div class="ag-table"><table><thead>${row(head, "th")}</thead><tbody>${body.join("")}</tbody></table></div>`);
        continue;
      }
      if ((m = line.match(/^\s{0,3}>\s?(.*)$/))) {
        flush(); endLists();
        const q = [m[1]];
        while (i + 1 < lines.length && (m = lines[i + 1].match(/^\s{0,3}>\s?(.*)$/))) { q.push(m[1]); i++; }
        out.push("<blockquote>" + render(q.join("\n")) + "</blockquote>");
        continue;
      }
      if ((m = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/))) {
        flush();
        const indent = m[1].replace(/\t/g, "    ").length, tag = /\d/.test(m[2]) ? "ol" : "ul";
        const top = list[list.length - 1];
        if (!top || indent > top.indent + 1) {
          const start = tag === "ol" && parseInt(m[2], 10) !== 1 ? ` start="${parseInt(m[2], 10)}"` : "";
          out.push(`<${tag}${start}><li>`); list.push({tag, indent});
        } else {
          endLists(indent);
          const t2 = list[list.length - 1];
          if (t2 && t2.tag !== tag && t2.indent === indent) { out.push(`</li></${list.pop().tag}><${tag}><li>`); list.push({tag, indent}); }
          else if (t2) out.push("</li><li>");
          else { out.push(`<${tag}><li>`); list.push({tag, indent}); }
        }
        out.push(inline(esc(m[3])));
        continue;
      }
      if (list.length && /^\s+\S/.test(line)) { out.push("<br>" + inline(esc(line.trim()))); continue; }
      endLists();
      para.push(line);
    }
    flush(); endLists();
    return out.join("");
  }
  function splitThink(text) {               // a server without a reasoning parser puts <think>..</think> in the text
    const m = String(text || "").match(/^\s*<think>([\s\S]*?)(<\/think>|$)/);
    if (!m) return {think: "", rest: text || "", open: false};
    return {think: m[1], rest: text.slice(m[0].length).replace(/^\s+/, ""), open: !m[2]};
  }
  return {render, inline, esc, splitThink};
})();
// md:end

// Shared by the Assistant (agent.js) and Classic (chat.js) views: functions run on the DOM the safe renderer produced
// (syntax highlighting, math). fn(root, {view: "assistant"|"classic", live, msg}). See CHAT-PLUGINS.md.
window.PXAMarkdown = window.PXAMarkdown || (function () {
  const P = [];
  return {add(fn) { if (typeof fn === "function" && !P.includes(fn)) P.push(fn); },
          remove(fn) { const i = P.indexOf(fn); if (i >= 0) P.splice(i, 1); },
          process(root, ctx) { for (const fn of P.slice()) { try { fn(root, ctx || {}); } catch (e) { console.error("PXAMarkdown post-processor failed:", e); } } },
          get count() { return P.length; }};
})();

(function () {
  const page = document.getElementById("p-chat");
  if (!page || typeof el !== "function") return;
  const ICON = {
    file: '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/>',
    pen: '<path d="M4 20h4L19 9l-4-4L4 16z"/>',
    globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/>',
    search: '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>',
    calc: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 12h2M14 12h2M8 16h2M14 16h2"/>',
    people: '<circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2.5"/><path d="M3 20c0-3.5 2.7-6 6-6s6 2.5 6 6M15 14.5c3 0 6 2 6 5.5"/>',
    term: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M13 15h4"/>',
    shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M12 8v5M12 16h.01"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
    redo: '<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4v5h-5"/>',
    edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/>',
    send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
    stop: '<rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    clip: '<path d="M21 11.5l-8.5 8.5a5 5 0 0 1-7-7L13 5.5a3.5 3.5 0 0 1 5 5L10.5 18a2 2 0 0 1-3-3L15 8"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8"/>',
    side: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
    gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    down: '<path d="M6 9l6 6 6-6"/>',
    check: '<path d="M5 12l5 5 9-10"/>',
    x: '<path d="M6 6l12 12M18 6L6 18"/>',
    bulb: '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3z"/>',
    dl: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
    close: '<path d="M6 6l12 12M18 6L6 18"/>',
    brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h1V4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h-1V4z"/>',
    pin: '<path d="M9 3h6l-1 6 4 3v2H6v-2l4-3z"/><path d="M12 14v7"/>',
    undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
  };
  const TOOL_ICON = {list_files: "file", read_file: "file", write_file: "pen", search_files: "search", web_fetch: "globe",
                     web_search: "search", calculate: "calc", ask_server: "people", spawn_agent: "people",
                     note_write: "pin", note_read: "pin", run_command: "term", remember: "brain", forget: "brain"};
  const MEM_TOOLS = ["remember", "forget", "memory_save"];
  const ico = (k, cls) => { const s = el("span", {class: "ag-svg " + (cls || ""), "aria-hidden": "true"});
    s.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICON[k] || ICON.file}</svg>`; return s; };
  const iconBtn = (k, label, onclick, extra) => el("button", Object.assign({type: "button", class: "ag-ib", title: label, "aria-label": label, onclick}, extra || {}), ico(k));
  const STARTERS = {
    chat: ["Explain what a GPU does, simply", "Write a short poem about my computer", "Give me three quick pasta dinner ideas", "Show me a table comparing tea and coffee"],
    assistant: ["What is 18% of 2,450?", "Make a shopping list file for a pasta dinner", "Summarize https://example.com", "What files do you have?"],
    researcher: ["What's new in open-source AI this month? Cite sources", "How do GPUs speed up AI? Keep it short", "Compare two popular laptops for students", "Find the population of Canada and cite it"],
    coder: ["Write a Python script that renames photos by date", "Make a tiny to-do web page", "Explain this regex: ^\\d{3}-\\d{4}$", "A bash one-liner to find the 10 biggest files"],
  };
  const A = {presets: [], tools: [], search: false, servers: [], selected: null, view: store.get("agent.view", "assistant"),
             adv: store.get("agent.adv", false), preset: store.get("agent.preset", "assistant"), session: store.get("agent.session", null),
             side: store.get("agent.side", window.innerWidth > 900), drawer: store.get("agent.drawer", false),
             useMem: true, incognito: false, web: true, refs: [], atOn: 0, atTok: 0, mem: {enabled: true, k: 8, budget: 400, facts: []},
             host: {available: false, on: false, why: "", allow: {commands: [], read_paths: []}},
             run: null, es: null, seq: 0, cur: null, turns: [], branchAt: {}, sessions: [], poll: null, lastStatus: {}, title: "New chat", retries: 0};
  const R = {};

  // ---------------- extension surface: window.PXAAgent (CHAT-PLUGINS.md) ----------------
  const EVENTS = ["init", "sessionChange", "beforeRun", "delta", "messageDone", "render"];
  const HK = {}; EVENTS.forEach(e => { HK[e] = []; });
  const X = {actions: [], buttons: [], sections: [], slash: new Map(), inited: false, runTypes: new Set(["compact"])};
  function hk(evt, ...args) { for (const fn of HK[evt].slice()) { try { fn(...args); } catch (e) { console.error("PXAAgent " + evt + " handler failed:", e); } } }
  async function hkAsync(evt, ...args) { for (const fn of HK[evt].slice()) { try { await fn(...args); } catch (e) { console.error("PXAAgent " + evt + " handler failed:", e); } } }
  function actBtn(a, msg) {
    return el("button", {type: "button", class: "ag-ib ag-act", "data-act": a.id, title: a.title || a.label, "aria-label": a.label,
      onclick: () => { try { const r = a.run(msg, G); if (r && r.catch) r.catch(e => toast(e.message || String(e), true)); } catch (e) { toast(e.message || String(e), true); } }},
      a.icon && ICON[a.icon] ? ico(a.icon) : el("span", {class: "ag-act-l", text: a.icon || a.label}));
  }
  function plugActs(msg) { return X.actions.filter(a => { try { return !a.when || a.when(msg, G); } catch (e) { return false; } }).map(a => actBtn(a, msg)); }
  function userPlug(m) { return {role: "user", idx: +m.dataset.turn, get text() { return m._text; }, el: m}; }
  function refreshActs() {
    if (!R.col) return;
    $$(".ag-u", R.col).forEach(m => {
      const tb = $(".ag-tb", m); if (!tb) return;
      $$("[data-act]", tb).forEach(b => b.remove());
      const acts = plugActs(userPlug(m)); if (acts.length) tb.append(...acts);
    });
    $$(".ag-a", R.col).forEach(m => {
      const tb = $(".ag-tb", m); if (!tb || !m._plug) return;
      $$("[data-act]", tb).forEach(b => b.remove());
      const acts = plugActs(m._plug); if (acts.length) tb.append(...acts);
    });
  }
  function mountButton(b) {
    if (!R.ctools) return;
    const old = $("#ag-tool-" + b.id); if (old) old.remove();
    R.ctools.append(el("button", {type: "button", class: "ag-chip ag-ctool", id: "ag-tool-" + b.id, title: b.title || b.label, "aria-label": b.label,
      onclick: e => { try { const r = b.onClick(G, e); if (r && r.catch) r.catch(x => toast(x.message || String(x), true)); } catch (x) { toast(x.message || String(x), true); } }},
      b.icon && ICON[b.icon] ? ico(b.icon) : null, b.icon && b.iconOnly ? null : el("span", {class: "ag-ctool-l", text: b.label})));
  }
  function mountSection(sc) {
    if (!R.secs) return;
    const old = $("#ag-sec-" + sc.id); if (old) old.remove();
    const box = el("div", {class: "ag-sec-b"});
    R.secs.append(el("div", {class: "ag-sec", id: "ag-sec-" + sc.id}, sc.title ? el("div", {class: "tiny", text: sc.title}) : null, box));
    try { sc.render && sc.render(box, G); } catch (e) { console.error("PXAAgent section " + sc.id + ":", e); }
  }
  const G = window.PXAAgent = {
    version: 1,
    on(evt, fn) {
      if (!HK[evt]) throw new Error("PXAAgent: unknown event " + evt + " (known: " + EVENTS.join(", ") + ")");
      HK[evt].push(fn); if (evt === "init" && X.inited) setTimeout(() => { try { fn(G); } catch (e) { console.error(e); } }, 0);
      return () => { const i = HK[evt].indexOf(fn); if (i >= 0) HK[evt].splice(i, 1); };
    },
    get ready() { return X.inited; },
    get sessionId() { return A.session; },
    // a marker line in the transcript (e.g. "Conversation compacted"): on screen only, not sent, not saved. Returns the element.
    insertMarker(text, opts) {
      opts = opts || {}; if (!R.col) return null;
      clearWelcome();
      const n = el("div", {class: "ag-marker " + (opts.kind || ""), role: "note", "data-marker": opts.id || ""}, el("span", {text: String(text)}));
      if (opts.before && opts.before.parentNode === R.col) R.col.insertBefore(n, opts.before); else R.col.append(n);
      scrollEnd(); return n;
    },
    get state() { return {session: A.session, run: A.run, turns: A.turns.slice(), servers: A.servers, selected: A.selected, preset: A.preset,
                          advanced: A.adv, view: A.view, title: A.title, web: A.web, useMem: A.useMem}; },
    get el() { return {root: R.root, col: R.col, scroll: R.scroll, input: R.input, send: R.send, composerTools: R.ctools, drawer: R.drawer, sections: R.secs}; },
    md: pxaMd,
    send(text) { if (text != null) { R.input.value = String(text); grow(); } return send(); },
    stop: () => stop(), newChat: () => newChat(), openSession: id => openSession(id),
    resend: (idx, text) => resend(idx, text == null ? A.turns[idx] : text),
    sessions: () => loadSessions().then(() => A.sessions.slice()),
    addMessageAction(a) {
      if (!a || !a.id || typeof a.run !== "function") throw new Error("addMessageAction needs {id, label, run}");
      X.actions = X.actions.filter(x => x.id !== a.id).concat([a]);
      refreshActs();   // messages already on screen pick up the new button, same as Classic
    },
    addComposerButton(b) {
      if (!b || !b.id || typeof b.onClick !== "function") throw new Error("addComposerButton needs {id, label, onClick}");
      X.buttons = X.buttons.filter(x => x.id !== b.id).concat([b]); mountButton(b);
    },
    addSettingsSection(id, title, render) { const sc = {id, title, render}; X.sections = X.sections.filter(x => x.id !== id).concat([sc]); mountSection(sc); },
    addSlashCommand(name, fn) { X.slash.set(String(name).replace(/^\//, ""), fn); },
    slashCommands: () => Array.from(X.slash.keys()),
    addRunEventType(type) { X.runTypes.add(String(type)); },     // an extra SSE event type from /api/chat/events, delivered to on("delta")
    setMarkdownPostProcessor(fn) { window.PXAMarkdown.add(fn); },
  };

  async function boot() {
    let p;
    try { p = await api("/api/chat/presets", {timeout: 8000}); } catch (e) { return; }   // no backend: leave Chat alone
    A.presets = p.presets; A.tools = p.tools; A.search = p.search;
    if (!A.presets.some(x => x.id === A.preset)) A.preset = "assistant";
    build();
    setView(A.view);
    if (A.session) await openSession(A.session, true); else welcome();
    loadSessions(); loadMemory(); loadHost();
    X.inited = true; hk("init", G);
  }

  // ---------------- layout ----------------
  function build() {
    const ph = $(".ph", page);
    const viewSeg = el("span", {class: "seg", id: "ag-view", "data-v": A.view, "aria-label": "assistant or classic chat"},
      el("button", {"data-v": "assistant", text: "Assistant"}), el("button", {"data-v": "classic", text: "Classic chat"}));
    const modeSeg = el("span", {class: "seg", id: "ag-mode", "data-v": A.adv ? "advanced" : "simple", "aria-label": "simple or advanced settings",
                                title: "Advanced adds a settings drawer: system prompt, sampling, tool switches, step limit, raw JSON and copy-as-curl"},
      el("button", {"data-v": "simple", text: "Simple"}), el("button", {"data-v": "advanced", text: "Advanced"}));
    ph.append(el("div", {class: "sp"}), el("div", {class: "ag-bar"}, viewSeg, el("span", {id: "ag-mode-w"}, modeSeg)));
    segInit(viewSeg, v => setView(v)); segInit(modeSeg, v => setAdv(v === "advanced"));

    // sidebar
    R.list = el("nav", {class: "ag-list", id: "ag-list", "aria-label": "your chats"});
    R.side = el("aside", {class: "ag-side", id: "ag-side"},
      el("div", {class: "ag-side-top"},
        el("button", {type: "button", class: "ag-new", id: "ag-new", onclick: newChat, title: "New chat (Ctrl+K)"}, ico("plus"), el("span", {text: "New chat"})),
        iconBtn("side", "Hide the chat list", () => setSide(false), {id: "ag-side-hide"})),
      R.list,
      el("button", {type: "button", class: "ag-memb", id: "ag-mem-open", onclick: () => openMemory(true), title: "What the assistant remembers about you"},
        ico("brain"), el("span", {text: "Memory"}), el("span", {class: "sp"}), el("span", {class: "ag-memn", id: "ag-mem-n"})),
      el("div", {class: "ag-side-foot"}, el("span", {text: "Chats and memory are saved on this machine only."})));
    R.scrim = el("div", {class: "ag-scrim", onclick: () => setSide(false)});

    // main
    R.title = el("div", {class: "ag-title", id: "ag-title", text: "New chat"});
    R.incogBadge = el("span", {class: "ag-incog", id: "ag-incog", hidden: true, text: "Incognito"});
    R.exp = el("span", {class: "ag-exp"},
      iconBtn("dl", "Save this chat (Markdown or JSON)", e => menu(e.currentTarget, [
        {label: "Save as Markdown", sub: "Readable notes file", act: () => exportChat("md")},
        {label: "Save as JSON", sub: "Everything, including tool steps", act: () => exportChat("json")}]), {id: "ag-export"}));
    R.gear = iconBtn("gear", "Settings for this mode", () => setDrawer(!A.drawer), {id: "ag-gear", class: "ag-ib"});
    R.top = el("div", {class: "ag-top"},
      iconBtn("side", "Show the chat list", () => setSide(!A.side), {id: "ag-side-show"}),
      R.title, R.incogBadge, el("span", {class: "sp"}),
      iconBtn("search", "Web search: on or off for this chat, and which search to use", () => openWeb(!(R.wsp && !R.wsp.hidden)), {id: "ag-web"}), R.exp, R.gear);
    R.col = el("div", {class: "ag-col", id: "ag-msgs", "aria-live": "polite"});
    R.scroll = el("div", {class: "ag-scroll", id: "ag-scroll"}, R.col);
    R.scroll.addEventListener("scroll", () => { A.stick = R.scroll.scrollHeight - R.scroll.scrollTop - R.scroll.clientHeight < 80; });
    R.col.addEventListener("click", e => {
      const b = e.target.closest("[data-agcopy]");
      if (b) { const c = b.closest(".ag-code"); copyText($("code", c).textContent); b.textContent = "Copied"; setTimeout(() => b.textContent = "Copy", 1500); }
    });

    // composer
    R.input = el("textarea", {id: "ag-in", rows: "1", placeholder: "Message the assistant", "aria-label": "message"});
    R.refs = el("div", {class: "ag-refs", id: "ag-refs", hidden: true});
    R.at = el("div", {class: "ag-at", id: "ag-at", hidden: true, role: "listbox"});
    R.input.addEventListener("keydown", onComposerKey);
    R.input.addEventListener("input", () => { grow(); refreshAt(); });
    R.send = el("button", {type: "button", class: "ag-send", id: "ag-send", "aria-label": "Send", title: "Send (Enter)", onclick: () => A.run ? stop() : send()}, ico("send"));
    R.stop = R.send;
    R.mode = el("button", {type: "button", class: "ag-chip", id: "ag-mode-chip", title: "What kind of help", onclick: e => presetMenu(e.currentTarget)});
    R.srv = el("button", {type: "button", class: "ag-chip ag-srvbtn", id: "ag-srv", title: "Which model answers", onclick: e => serverMenu(e.currentTarget)});
    R.more = el("button", {type: "button", class: "ag-chip ag-more", id: "ag-composer-more", title: "More", "aria-label": "More composer actions",
      onclick: e => composerMore(e.currentTarget)}, ico("plus"));
    R.comp = el("div", {class: "ag-comp"},
      el("div", {class: "ag-box"}, R.at, R.refs, R.input,
        el("div", {class: "ag-box-row"}, R.mode, R.srv, R.more, R.ctools = el("span", {class: "ag-ctools", id: "ag-composer-tools"}), el("span", {class: "sp"}), R.send)),
      el("div", {class: "ag-hint", text: "Enter to send · Shift+Enter for a new line · Ctrl+K new chat · Esc stops"}),
      el("div", {id: "ag-qnote"}));
    R.main = el("section", {class: "ag-main"}, R.top, R.scroll, R.comp);

    // settings drawer: the gear opens it in both modes; the knobs Simple mode hides carry .ag-advonly
    R.sys = el("textarea", {id: "ag-sys", rows: "7", "aria-label": "system prompt"});
    R.temp = el("input", {type: "number", id: "ag-temp", min: "0", max: "2", step: "0.05"});
    R.topp = el("input", {type: "number", id: "ag-topp", min: "0", max: "1", step: "0.05", placeholder: "server default"});
    R.topk = el("input", {type: "number", id: "ag-topk", min: "0", max: "500", step: "1", placeholder: "server default"});
    R.minp = el("input", {type: "number", id: "ag-minp", min: "0", max: "1", step: "0.01", placeholder: "server default"});
    R.maxtok = el("input", {type: "number", id: "ag-maxtok", min: "1", max: "262144", placeholder: "server default"});
    R.steps = el("input", {type: "number", id: "ag-steps", min: "1", max: "40"});
    R.think = el("select", {id: "ag-think"}, el("option", {value: "auto", text: "auto"}), el("option", {value: "on", text: "on"}), el("option", {value: "off", text: "off"}));
    R.tmode = el("select", {id: "ag-tmode", title: "auto = the server's own tool calls when its template has them, else the text protocol"},
      el("option", {value: "auto", text: "auto"}), el("option", {value: "native", text: "native"}), el("option", {value: "text", text: "text protocol"}));
    R.tools = el("div", {class: "ag-tools"});
    R.host = el("div", {class: "ag-host", id: "ag-host"});
    for (const t of A.tools) {
      const cb = el("input", {type: "checkbox", value: t.name, disabled: !t.available && !t.host_only});
      if (t.host_only) cb.dataset.host = "1";          // enabled or not by the Host access switch below, not here
      R.tools.append(el("label", {class: "check", title: t.available || t.host_only ? "" : "needs a search endpoint (chat_search_url in control.json)"}, cb,
        el("span", {}, el("b", {text: t.label}), " ", el("small", {text: t.available || t.host_only ? t.blurb : "not set up on this machine"}))));
    }
    R.drawer = el("aside", {class: "ag-drawer", id: "ag-drawer", "aria-label": "settings"},
      el("div", {class: "ag-drawer-h"}, el("b", {text: "Settings"}), el("span", {class: "tiny", id: "ag-drawer-preset"}), el("span", {class: "sp"}),
        iconBtn("close", "Close settings", () => setDrawer(false), {id: "ag-drawer-x"})),
      el("div", {class: "ag-drawer-b"},
        el("div", {class: "ag-advonly", style: "display:grid;gap:12px"},
          el("label", {class: "f"}, "System prompt", R.sys),
          el("div", {class: "ag-grid2"},
            el("label", {class: "f"}, "Temperature", R.temp), el("label", {class: "f"}, "Top P", R.topp),
            el("label", {class: "f"}, "Top K", R.topk), el("label", {class: "f"}, "Min P", R.minp),
            el("label", {class: "f"}, "Max tokens per step", R.maxtok), el("label", {class: "f"}, "Max steps", R.steps),
            el("label", {class: "f"}, "Thinking", R.think), el("label", {class: "f"}, "Tool calls", R.tmode)),
          el("div", {class: "tiny", style: "margin-top:4px"}, "Tools this chat may use"), R.tools,
          el("div", {class: "row", style: "gap:8px;flex-wrap:wrap"},
            el("button", {class: "b", id: "ag-curl", text: "Copy as curl", onclick: copyCurl}))),
        el("div", {class: "tiny", style: "margin-top:8px"}, "This machine"), R.host,
        el("label", {class: "check", title: "Off: this chat neither sees nor saves memories (the switch in Memory turns it off everywhere)"},
          R.usemem = el("input", {type: "checkbox", id: "ag-usemem", checked: true, onchange: () => { A.useMem = R.usemem.checked; }}),
          el("span", {}, el("b", {text: "Use memory in this chat"}), " ", el("small", {text: "recall and save facts about you"}))),
        el("label", {class: "check", title: "Incognito chats are not saved and are not searched. Closing this page drops the chat."},
          R.incog = el("input", {type: "checkbox", id: "ag-incognito", checked: false, onchange: () => {
            A.incognito = R.incog.checked;
            if (A.incognito) { A.useMem = false; R.usemem.checked = false; }
            syncIncog();
          }}),
          el("span", {}, el("b", {text: "Incognito"}), " ", el("small", {text: "this chat is not saved and is not searched"}))),
        el("div", {class: "row", style: "gap:8px;flex-wrap:wrap"},
          el("button", {class: "b", id: "ag-reset", text: "Reset to preset", onclick: () => { store.set("agent.adv." + A.preset, null); presetChanged(); toast("Back to the preset's settings"); }})),
        el("div", {class: "tiny", style: "margin-top:8px"}, "Sub-agents"),
        el("div", {class: "ag-grid2", id: "ag-sub-set"},
          el("label", {class: "f"}, "At once", R.subMax = el("input", {type: "number", id: "ag-sub-max", min: "1", max: "4", value: String(store.get("agent.subMax", 2))})),
          el("label", {class: "f"}, "Steps each", R.subSteps = el("input", {type: "number", id: "ag-sub-steps", min: "1", max: "8", value: String(store.get("agent.subSteps", 4))})),
          el("label", {class: "f"}, "Seconds each", R.subSecs = el("input", {type: "number", id: "ag-sub-secs", min: "15", max: "120", value: String(store.get("agent.subSecs", 60))}))),
        R.secs = el("div", {class: "ag-secs", id: "ag-settings-sections"})));
    X.buttons.forEach(mountButton); X.sections.forEach(mountSection);

    R.memp = buildMemory();
    R.root = el("div", {class: "ag", id: "ag-root"}, R.side, R.scrim, R.main, R.drawer, R.memp, R.wsp = buildWeb());
    page.append(R.root);
    [R.sys, R.temp, R.topp, R.topk, R.minp, R.maxtok, R.steps, R.think, R.tmode].forEach(x => x.addEventListener("change", saveAdv));
    [R.subMax, R.subSteps, R.subSecs].forEach(x => x.addEventListener("change", () => {
      subNum(R.subMax, "agent.subMax", 1, 4, 2);
      subNum(R.subSteps, "agent.subSteps", 1, 8, 4);
      subNum(R.subSecs, "agent.subSecs", 15, 120, 60);
    }));
    R.tools.addEventListener("change", saveAdv);
    page.dataset.agMode = A.adv ? "advanced" : "simple";
    setSide(A.side && window.innerWidth > 900, true); setDrawer(A.drawer && window.innerWidth > 900, true);
    presetChanged(); renderHost();
    window.addEventListener("resize", fit);
    new MutationObserver(() => requestAnimationFrame(fit)).observe(page, {attributes: true, attributeFilter: ["hidden"]});
    document.addEventListener("keydown", keys);
    document.addEventListener("mousedown", e => { if (R.at && !R.at.hidden && !R.at.contains(e.target) && e.target !== R.input) closeAt(); });
    document.addEventListener("click", e => { if (R.menu && !R.menu.contains(e.target) && !e.target.closest(".ag-menu-open")) closeMenu(); }, true);
  }
  function fit() {
    if (!R.root || R.root.hidden || page.hidden || !R.root.offsetParent) return;    // measured only when on screen
    // fill the window without making the page itself scroll (the conversation scrolls inside)
    const top = R.root.getBoundingClientRect().top + window.scrollY;
    // what sits under the panel inside <main> (its bottom padding, which also clears the mobile tab bar).
    // Not document.scrollHeight: that is never less than the window, so a short panel would stay short.
    const main = R.root.closest("main") || document.body;
    const below = Math.max(0, main.getBoundingClientRect().bottom - R.root.getBoundingClientRect().bottom);
    R.root.style.height = Math.max(420, window.innerHeight - top - below) + "px";
    if (window.innerWidth > 900 && window.scrollY) window.scrollTo(0, 0);
    if (window.innerWidth <= 900 && A.side && !R.root.classList.contains("ag-side-overlay")) R.root.classList.add("ag-side-overlay");
    if (window.innerWidth > 900) R.root.classList.remove("ag-side-overlay");
  }
  function keys(e) {
    if (e.key === "Escape" && R.memp && !R.memp.hidden) { e.preventDefault(); openMemory(false); return; }
    if (A.view !== "assistant" || S.tab !== "chat" || R.root.hidden) return;
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); return; }
    if (e.key === "Escape") {
      if (R.menu) { closeMenu(); return; }
      if (A.run && !document.querySelector("dialog[open]")) { e.preventDefault(); stop(); }
    }
  }
  function setView(v) {
    A.view = v; store.set("agent.view", v);
    const classic = $(".cols.chat:not(#ag-root)", page);
    if (classic) classic.hidden = v !== "classic";
    R.root.hidden = v !== "assistant";
    $("#ag-mode-w").hidden = v !== "assistant";
    const sub = $(".ph p", page); if (sub) sub.textContent = v === "assistant" ? "Your own AI assistant, running on your own cards. It can use tools and always asks before anything risky." : "Streams from the running server's OpenAI endpoint.";
    if (v === "assistant") { startPoll(); requestAnimationFrame(fit); } else stopPoll();
  }
  function setAdv(on) {
    A.adv = on; store.set("agent.adv", on); page.dataset.agMode = on ? "advanced" : "simple";
    if (on && store.get("agent.drawer.seen", false) === false) { store.set("agent.drawer.seen", true); setDrawer(true); }
    $$(".ag-tool", R.col).forEach(t => t._fill && t._fill());
    renderServerChip();
  }
  function setSide(on, quiet) {
    A.side = on; if (!quiet || window.innerWidth > 900) store.set("agent.side", on);
    R.root.classList.toggle("ag-side-off", !on);
    R.root.classList.toggle("ag-side-overlay", on && window.innerWidth <= 900);
  }
  function setDrawer(on, quiet) { A.drawer = on; if (!quiet) store.set("agent.drawer", on); R.root.classList.toggle("ag-drawer-off", !on); }

  // ---------------- presets / advanced ----------------
  const preset = () => A.presets.find(p => p.id === A.preset) || A.presets[0];
  function presetChanged() {
    const p = preset(), saved = store.get("agent.adv." + p.id, null);
    R.sys.value = saved && saved.system != null ? saved.system : p.system;
    R.temp.value = saved && saved.temperature != null ? saved.temperature : p.temperature;
    R.topp.value = saved && saved.top_p != null ? saved.top_p : ""; R.topk.value = saved && saved.top_k != null ? saved.top_k : "";
    R.minp.value = saved && saved.min_p != null ? saved.min_p : ""; R.maxtok.value = saved && saved.max_tokens ? saved.max_tokens : "";
    R.steps.value = saved && saved.max_steps ? saved.max_steps : p.max_steps;
    R.think.value = saved && saved.thinking ? saved.thinking : (p.thinking || "auto"); R.tmode.value = saved && saved.tool_mode ? saved.tool_mode : "auto";
    const on = new Set(saved && saved.tools ? saved.tools : p.tools);
    $$("input", R.tools).forEach(cb => cb.checked = on.has(cb.value) && !cb.disabled);
    R.mode.replaceChildren(el("span", {class: "ag-chip-k", text: "Mode"}), el("span", {text: p.name}), ico("down", "ag-caret"));
    $("#ag-drawer-preset").textContent = p.name;
    if (!A.turns.length && !A.run) welcome();
  }
  function setPreset(id) { A.preset = id; store.set("agent.preset", id); presetChanged(); }
  const numOrNull = v => v === "" ? null : +v;
  function subNum(node, key, lo, hi, fallback) {
    const n = Math.round(+node.value);
    const v = Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : fallback;
    node.value = String(v);
    store.set(key, v);
    return v;
  }
  function advSettings() {
    return {system: R.sys.value, temperature: numOrNull(R.temp.value), top_p: numOrNull(R.topp.value), top_k: numOrNull(R.topk.value),
            min_p: numOrNull(R.minp.value), max_tokens: numOrNull(R.maxtok.value), max_steps: numOrNull(R.steps.value),
            thinking: R.think.value, tool_mode: R.tmode.value, tools: $$("input:checked", R.tools).map(c => c.value)};
  }
  function saveAdv() { store.set("agent.adv." + A.preset, advSettings()); }

  // ---------------- host access ----------------
  // The switch is per chat and lasts only as long as this session of PXA Control, so what the panel shows is
  // read back from the server, never remembered here: `available` is the deployment's answer (a Control
  // reachable on the LAN refuses it, so the person approving is not necessarily the person at the keyboard),
  // `on` is this chat's switch, and `allow` is the file the owner writes by hand.
  const hostOn = () => !!(A.host.available && A.host.on);
  function setHost(d) { A.host = Object.assign({}, A.host, d, {allow: (d && d.allow) || A.host.allow}); }
  function renderHost() {
    if (!R.host) return;
    const h = A.host;
    const cb = el("input", {type: "checkbox", id: "ag-host-cb", checked: !!h.on, disabled: !h.available, onchange: hostToggle});
    const words = !h.available ? (h.why || "not available on this Control")
      : h.on ? "On for this chat, for this session only. Every command asks first and is written to the log."
      : "Off. The assistant cannot run commands or read files on this machine.";
    R.host.replaceChildren(
      el("label", {class: "check", title: h.available ? "Run allowlisted commands and read allowlisted files on the machine PXA Control is running on" : ""},
        cb, el("span", {}, el("b", {text: "Host access"}), " ", el("small", {text: "commands and files on this machine"}))),
      el("div", {class: "ag-host-w" + (h.on ? " on" : ""), id: "ag-host-w", text: words}),
      h.available ? hostAllowBox(h) : null);
    const on = hostOn();
    $$("input[data-host]", R.tools).forEach(b => { b.disabled = !on; if (!on) b.checked = false; });
  }
  function hostAllowBox(h) {
    const a = h.allow || {}, cmds = a.commands || [], reads = a.read_paths || [];
    const d = el("div", {class: "ag-host-d"});
    if (!cmds.length && !reads.length) {
      d.append(el("div", {class: "tiny", text: "Nothing is allowed yet. Write the commands and folders you permit into host-allow.json, then reopen this panel."}));
    } else {
      for (const c of cmds) d.append(el("code", {text: "run: " + (Array.isArray(c) ? c.join(" ") : c)}));
      for (const p of reads) d.append(el("code", {text: "read: " + p}));
    }
    d.append(el("div", {class: "tiny", text: "Only what is listed here, and only while the switch is on."}));
    return d;
  }
  async function hostToggle(e) {
    const cb = e.currentTarget, on = !!cb.checked;
    cb.disabled = true;                                  // no second toggle while the server is answering
    try { setHost(await api("/api/chat/host", {body: {on, session_id: A.session}})); }
    catch (err) { toast(err.message, true); }
    renderHost();
    if (hostOn()) toast("Host access is on for this chat. Every call asks first.");
  }
  async function loadHost() {
    try { setHost(await api("/api/chat/host" + (A.session ? "?session=" + encodeURIComponent(A.session) : ""))); }
    catch (e) { renderHost(); return; }
    renderHost();
  }

  // ---------------- menus ----------------
  function closeMenu() { if (R.menu) { R.menu.remove(); R.menu = null; } $$(".ag-menu-open").forEach(b => b.classList.remove("ag-menu-open")); }
  function menu(anchor, items, opts) {
    if (R.menu && anchor.classList.contains("ag-menu-open")) { closeMenu(); return; }
    closeMenu();
    const m = el("div", {class: "ag-menu", role: "menu"});
    for (const it of items) {
      if (it.head) { m.append(el("div", {class: "ag-menu-head", text: it.head})); continue; }
      if (it.node) { m.append(it.node); continue; }
      m.append(el("button", {type: "button", role: "menuitemradio", "aria-checked": it.on ? "true" : "false", class: "ag-menu-i" + (it.on ? " on" : ""), disabled: !!it.disabled,
        onclick: () => { closeMenu(); it.act && it.act(); }},
        it.dot ? el("span", {class: "dot " + it.dot}) : null,
        el("span", {class: "ag-menu-t"}, el("b", {text: it.label}), it.sub ? el("small", {text: it.sub}) : null),
        it.on ? ico("check", "ag-menu-ck") : null));
    }
    document.body.append(m);
    anchor.classList.add("ag-menu-open");
    const r = anchor.getBoundingClientRect(), mw = Math.min(340, window.innerWidth - 16);
    m.style.width = mw + "px";
    let left = (opts && opts.right) ? r.right - mw : r.left;
    left = Math.max(8, Math.min(left, window.innerWidth - mw - 8));
    m.style.left = left + "px";
    const h = m.offsetHeight;
    m.style.top = (r.top - h - 6 > 8 && !(opts && opts.below) ? r.top - h - 6 : r.bottom + 6) + window.scrollY + "px";
    R.menu = m;
    const first = $("button:not([disabled])", m); if (first) first.focus();
  }
  function composerMore(anchor) {
    const items = X.buttons.map(b => ({label: b.label || b.id, sub: b.title && b.title !== b.label ? b.title : "",
      act: () => { try { const r = b.onClick(G); if (r && r.catch) r.catch(x => toast(x.message || String(x), true)); } catch (x) { toast(x.message || String(x), true); } }}));
    menu(anchor, [{head: "Composer"}].concat(items.length ? items : [{label: "Nothing else yet", disabled: true}]));
  }
  function presetMenu(anchor) {
    menu(anchor, [{head: "What kind of help?"}].concat(A.presets.map(p => ({label: p.name, sub: p.blurb, on: p.id === A.preset, act: () => setPreset(p.id)}))));
  }
  const DOT = {ready: "ok", busy: "warn", starting: "warn pulse", error: "bad", down: "bad"};
  function serverMenu(anchor) {
    if (!A.servers.length) {
      menu(anchor, [{head: "No server running yet"}, {label: "Go to Launch", sub: "Start a model, then come back. Chat picks it up on its own.", act: () => showTab("launch")}]);
      return;
    }
    menu(anchor, [{head: "Talk to"}].concat(A.servers.map(s => ({
      label: s.name, dot: DOT[s.status] || "", on: s.key === A.selected, act: () => pickServer(s.key),
      sub: [s.status_words, s.cards_label, s.ctx && s.ctx.label, s.caps && s.caps.tools ? "tools" : "", s.caps && s.caps.thinking ? "thinking" : "", A.adv ? s.base_url : ""].filter(Boolean).join(" · ")})),
      [{label: "Refresh the list", sub: "Look for servers again", act: () => loadServers(true)}]));
  }

  // ---------------- servers ----------------
  function startPoll() { stopPoll(); loadServers(); A.poll = setInterval(() => { if (!document.hidden && S.tab === "chat" && A.view === "assistant") loadServers(); }, 5000); }
  function stopPoll() { if (A.poll) clearInterval(A.poll); A.poll = null; }
  async function loadServers(force) {
    let d;
    try { d = await api("/api/chat/servers", {timeout: 15000}); } catch (e) { if (force) toast(e.message, true); return; }
    A.servers = d.servers; A.search = d.search;
    if (d.host) { setHost(d.host); renderHost(); }        // the switch is per chat, so it follows the chat
    if (!A.selected || !A.servers.some(s => s.key === A.selected)) A.selected = d.selected;
    const cur = A.servers.find(s => s.key === A.selected), was = A.lastStatus[A.selected];
    const up = s => s === "ready" || s === "busy";
    if (cur && was && was !== cur.status && up(cur.status) && !up(was)) notice(`${cur.name} is back. Your chat is still here.`, "ok");
    A.lastStatus = Object.fromEntries(A.servers.map(s => [s.key, s.status]));
    renderServerChip();
    if (force) toast(A.servers.length ? `Found ${A.servers.length} server${A.servers.length > 1 ? "s" : ""}` : "No server running yet");
  }
  function renderServerChip() {
    const cur = A.servers.find(s => s.key === A.selected);
    R.srv.replaceChildren(el("span", {class: "dot " + (cur ? DOT[cur.status] || "" : "bad")}),
      el("span", {class: "ag-srv-n", text: cur ? cur.name : (A.servers.length ? "Pick a server" : "No server running")}), ico("down", "ag-caret"));
    R.srv.title = cur ? `${cur.name} · ${cur.status_words} · ${cur.ctx ? cur.ctx.label : ""}` : "Start a model on the Launch tab";
    sendState();
    const w = $("#ag-welcome-srv"); if (w) w.textContent = cur ? `${preset().name} mode · ${cur.name}` : "No model is running yet";
    if (window.chatQuantNote) chatQuantNote();
  }
  async function pickServer(key) {
    A.selected = key; renderServerChip();
    try { await api("/api/chat/select", {body: {key}}); } catch (e) { toast(e.message, true); }
  }
  function sendState() {
    const running = !!A.run;
    R.send.classList.toggle("stop", running);
    R.send.replaceChildren(ico(running ? "stop" : "send"));
    R.send.setAttribute("aria-label", running ? "Stop" : "Send"); R.send.title = running ? "Stop (Esc)" : "Send (Enter)";
    R.send.disabled = !running && !A.servers.some(s => s.key === A.selected);
    R.send.id = "ag-send";
  }
  function grow() { R.input.style.height = "auto"; R.input.style.height = Math.min(R.input.scrollHeight, Math.round(window.innerHeight * 0.4)) + "px"; }

  // ---------------- conversation ----------------
  function scrollEnd(force) { if (force || A.stick !== false) R.scroll.scrollTop = R.scroll.scrollHeight; }
  function welcome() {
    const p = preset(), cur = A.servers.find(s => s.key === A.selected);
    R.col.replaceChildren(el("div", {class: "ag-welcome", id: "ag-empty"},
      el("div", {class: "ag-hello", text: greeting()}),
      el("div", {class: "ag-sub", id: "ag-welcome-srv", text: cur ? `${p.name} mode · ${cur.name}` : "No model is running yet"}),
      el("p", {class: "ag-blurb", text: p.tools.length ? p.blurb + " It always asks before saving files or anything risky." : p.blurb}),
      el("div", {class: "ag-starters"}, (STARTERS[p.id] || []).map(t => el("button", {type: "button", class: "ag-starter", onclick: () => { R.input.value = t; grow(); send(); }}, el("span", {text: t}))))));
    A.title = "New chat"; R.title.textContent = "New chat";
  }
  function greeting() { const h = new Date().getHours(); return (h < 5 ? "Up late?" : h < 12 ? "Good morning." : h < 18 ? "Good afternoon." : "Good evening.") + " What can I help with?"; }
  function clearWelcome() { const w = $("#ag-empty", R.col); if (w) w.remove(); }

  function refTitle(t) { t = String(t || "Chat").replace(/\s+/g, " ").trim(); return t.length > 42 ? t.slice(0, 39) + "..." : (t || "Chat"); }
  function atQuery() {
    const ta = R.input, v = ta.value, i = ta.selectionStart;
    if (i == null) return null;
    const m = /(^|\s)@([^\s@]*)$/.exec(v.slice(0, i));
    if (!m) return null;
    return {start: i - m[2].length - 1, end: i, q: m[2]};
  }
  function closeAt() { if (!R.at) return; R.at.hidden = true; R.at.replaceChildren(); A.atOn = 0; }
  function paintRefs() {
    if (!R.refs) return;
    R.refs.replaceChildren(...A.refs.map(s => el("span", {class: "ag-ref"},
      el("span", {class: "ag-ref-t", text: "@ " + refTitle(s.title)}),
      el("button", {type: "button", class: "ag-ref-x", "aria-label": "Remove " + refTitle(s.title), text: "\u00d7",
        onclick: () => { A.refs = A.refs.filter(x => x.id !== s.id); paintRefs(); }}))));
    R.refs.hidden = !A.refs.length;
  }
  function addRef(s) {
    const q = atQuery();
    if (q) {
      const v = R.input.value;
      R.input.value = v.slice(0, q.start) + v.slice(q.end);
      grow();
    }
    closeAt();
    if (!s || !s.id || s.id === A.session || A.refs.some(r => r.id === s.id)) { R.input.focus(); return; }
    if (A.refs.length >= 4) { toast("Four earlier chats is the limit on one message.", true); R.input.focus(); return; }
    A.refs.push({id: s.id, title: s.title || "Chat"});
    paintRefs();
    R.input.focus();
  }
  async function refreshAt() {
    const q = atQuery(), tok = ++A.atTok;
    if (!q) { closeAt(); return; }
    let rows = A.sessions || [];
    try {
      const d = await api("/api/chat/sessions");
      if (tok !== A.atTok) return;
      rows = d.sessions || rows; A.sessions = rows;
    } catch (e) { /* the list we already have is enough to offer */ }
    if (tok !== A.atTok || !atQuery()) return;
    const needle = q.q.toLowerCase();
    const hits = rows.filter(s => s && s.id && s.id !== A.session && !s.incognito &&
      (!needle || String(s.title || "").toLowerCase().includes(needle) || String(s.id).toLowerCase().includes(needle))).slice(0, 8);
    A.atOn = 0;
    if (!hits.length) {
      R.at.replaceChildren(el("div", {class: "ag-at-none", text: needle ? "No saved chats match" : "No saved chats yet"}));
    } else {
      R.at.replaceChildren(...hits.map((s, i) => el("button", {type: "button", class: i === 0 ? "on" : "", role: "option",
        onclick: () => addRef(s)}, el("span", {class: "ag-at-t", text: s.title || "Chat"}))));
    }
    R.at.hidden = false;
  }
  function onComposerKey(e) {
    if (R.at && !R.at.hidden) {
      const buttons = R.at.querySelectorAll("button");
      if ((e.key === "ArrowDown" || e.key === "ArrowUp") && buttons.length) {
        e.preventDefault();
        A.atOn = (A.atOn + (e.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
        buttons.forEach((b, i) => b.classList.toggle("on", i === A.atOn));
        return;
      }
      if (e.key === "Enter" && !e.shiftKey && buttons.length) {
        e.preventDefault();
        (buttons[A.atOn] || buttons[0]).click();
        return;
      }
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeAt(); return; }
    }
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  }
  function branchBar(idx) {
    const b = A.branchAt[idx];
    if (!b || !(b.n > 1)) return null;
    const go = dir => () => switchBranch(idx, dir);
    return el("div", {class: "ag-branch", role: "group", "aria-label": "version " + b.i + " of " + b.n},
      el("button", {type: "button", class: "ag-branch-prev", text: "<", title: "Previous version", "aria-label": "Previous version",
        disabled: b.i <= 1, onclick: go(-1)}),
      el("span", {class: "ag-branch-n", text: b.i + "/" + b.n}),
      el("button", {type: "button", class: "ag-branch-next", text: ">", title: "Next version", "aria-label": "Next version",
        disabled: b.i >= b.n, onclick: go(1)}));
  }
  async function switchBranch(idx, dir) {
    if (A.run) { toast("Wait for the answer, or press Stop.", true); return; }
    const b = A.branchAt[idx];
    if (!b || !A.session) return;
    const index = b.i - 1 + dir;
    if (index < 0 || index >= b.n) return;
    try {
      await api("/api/chat/branch", {body: {session_id: A.session, turn: idx, index}});
      await openSession(A.session, true);
    } catch (e) { toast(e.message || String(e), true); }
  }
  function userMsg(text, idx, refs) {
    clearWelcome();
    const bubble = el("div", {class: "ag-bubble"});
    if (refs && refs.length) bubble.append(el("div", {class: "ag-msg-refs"},
      ...refs.map(r => el("span", {class: "ag-ref ag-ref-ro", text: "@ " + refTitle(r.title)}))));
    bubble.append(document.createTextNode(text));
    const m = el("div", {class: "ag-u", "data-turn": idx}, bubble, branchBar(idx),
      el("div", {class: "ag-tb"},
        iconBtn("copy", "Copy", () => copyText(m._text)),
        iconBtn("edit", "Edit and resend", () => editUser(m))));
    m._text = text; m._bubble = bubble;
    if (X.actions.length) $(".ag-tb", m).append(...plugActs(userPlug(m)));
    R.col.append(m);
    hk("render", m, {role: "user", idx, text, el: m});
    return m;
  }
  function editUser(m) {
    if (A.run) { toast("Wait for the answer, or press Stop.", true); return; }
    const idx = +m.dataset.turn, ta = el("textarea", {class: "ag-edit", rows: "2"});
    ta.value = m._text;
    const done = () => { box.replaceWith(m._bubble); m.classList.remove("editing"); };
    const go = () => { const v = ta.value.trim(); if (!v) return; done(); resend(idx, v); };
    ta.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(); } if (e.key === "Escape") { e.stopPropagation(); done(); } });
    const box = el("div", {class: "ag-editbox"}, ta, el("div", {class: "row", style: "justify-content:flex-end;gap:8px"},
      el("button", {type: "button", class: "b", text: "Cancel", onclick: done}), el("button", {type: "button", class: "b pri", text: "Send", onclick: go})));
    m._bubble.replaceWith(box); m.classList.add("editing");
    ta.focus(); ta.style.height = Math.min(ta.scrollHeight + 4, 320) + "px";
  }
  function asstMsg(idx) {
    const parts = el("div", {class: "ag-parts"}), foot = el("div", {class: "ag-foot"});
    const m = el("div", {class: "ag-a", "data-turn": idx}, parts, foot);
    R.col.append(m);
    return {m, parts, foot, idx, texts: {}, tools: {}, think: null, final: "", pending: el("span", {class: "ag-typing"}, el("i"), el("i"), el("i"))};
  }
  // --- parts
  function thinkRow(T) {
    if (T.think) return T.think;
    const body = el("div", {class: "ag-think-b"}), label = el("span", {text: "Thinking"});
    const d = el("details", {class: "ag-think"}, el("summary", {}, ico("bulb"), label, el("span", {class: "ag-dots"})), body);
    T.parts.append(d);
    T.think = {d, body, label, text: "", t0: null, t1: null, live: true};
    return T.think;
  }
  function thinkDone(T, secs) {
    const k = T.think; if (!k || !k.live) return;
    k.live = false; k.d.classList.add("done");
    const s = secs != null ? secs : (k.t1 && k.t0 ? k.t1 - k.t0 : null);
    k.label.textContent = s != null ? `Thought for ${s < 1 ? "a moment" : Math.round(s) + "s"}` : "Thought";
  }
  function textPart(T, step) {
    let p = T.texts[step];
    if (!p) { p = {node: el("div", {class: "ag-md"}), raw: ""}; T.texts[step] = p; T.parts.append(p.node); }
    return p;
  }
  function paint(T, p, live) {
    const sp = pxaMd.splitThink(p.raw);
    if (sp.think) { const k = thinkRow(T); k.body.textContent = sp.think; if (!sp.open) thinkDone(T); }
    p.node.innerHTML = pxaMd.render(sp.rest);
    window.PXAMarkdown.process(p.node, {view: "assistant", live: !!live, idx: T.idx});
    p.node.hidden = !sp.rest.trim();
    if (live) {
      $$(".ag-cursor", T.m).forEach(c => c.remove());
      const last = p.node.lastElementChild;
      const host = last && /^(P|LI|H\d|TD)$/.test(last.tagName) ? last : (last && last.tagName === "UL" || last && last.tagName === "OL") ? last.lastElementChild || p.node : p.node;
      host.append(el("span", {class: "ag-cursor", "aria-hidden": "true"}));
      p.node.hidden = false;
    }
  }
  function schedulePaint(T, p) {
    if (p._raf) return;
    p._raf = requestAnimationFrame(() => { p._raf = 0; paint(T, p, A.cur === T); scrollEnd(); });
  }
  function toolRow(T, d) {
    if (MEM_TOOLS.includes(d.name)) return memRow(T, d);
    const st = el("span", {class: "ag-st"}, el("span", {class: "spin"}));
    const say = el("span", {class: "ag-say", text: d.say});
    const body = el("div", {class: "ag-tool-b"});
    const row = el("details", {class: "ag-tool running"}, el("summary", {}, ico(TOOL_ICON[d.name] || "file"), say, st), body);
    if (d.name === "spawn_agent") { row.classList.add("ag-subcard"); row.open = d.ok == null; }
    row._d = Object.assign({}, d); row._say = say; row._st = st; row._body = body; row._T = T;
    row._fill = () => fillTool(row);
    row._fill();
    T.parts.append(row); T.tools[d.call_id] = row;
    return row;
  }
  function friendlyArgs(d) {
    const a = d.args || {};
    if (d.name === "calculate") return String(a.expression || "");
    if (d.name === "web_fetch") return String(a.url || "");
    if (d.name === "web_search") return String(a.query || "");
    if (d.name === "run_command") return String(a.command || "");
    if (d.name === "write_file") return `${a.path || ""}${a.append ? " (add to the end)" : ""}\n\n${String(a.content || "").slice(0, 2000)}`;
    if (d.name === "ask_server") return `${a.server || ""}: ${a.task || ""}`;
    if (d.name === "spawn_agent") return String(a.task || "");
    return Object.entries(a).map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`).join("\n") || "(nothing)";
  }
  function splitLegacyTrace(text) {
    const s = String(text || "");
    const cut = s.search(/\n\nTrace:\n/);
    if (cut < 0) return {answer: s.trim(), trace: []};
    const lines = s.slice(cut + 8).split("\n").map(l => l.replace(/^- /, "").trim()).filter(l => l && !/^\d+ tokens$/.test(l));
    return {answer: s.slice(0, cut).trim(), trace: lines};
  }
  function subShown(d) {
    if (!d) return "";
    const legacy = splitLegacyTrace(d.output);
    return (Array.isArray(d.trace) ? String(d.output == null ? "" : d.output) : legacy.answer).replace(/\n\nTrace:\n[\s\S]*$/, "").trim();
  }
  function paintRich(node, src) {
    node.innerHTML = pxaMd.render(pxaMd.splitThink(src).rest);
    window.PXAMarkdown.process(node, {view: "assistant", live: false});
  }
  function normAnswer(s) { return String(s || "").replace(/\s+/g, " ").trim(); }
  function isEcho(final, answer) {
    const a = normAnswer(answer);
    let f = normAnswer(final);
    if (!a || !f) return false;
    if (f === a) return true;
    for (let i = 0; i < 3 && f.length > a.length && /^result:\s*/i.test(f); i++) {
      f = f.replace(/^result:\s*/i, "");
      if (f === a) return true;
    }
    return false;
  }
  function dedupeSubAnswers(T) {
    // The card already shows the answer. Hide a follow-up that only repeats it, so the reader sees it once.
    if (!T || !T.parts) return;
    const rows = $$(".ag-subcard", T.parts);
    const answers = rows.map(row => subShown(row._d)).filter(Boolean);
    if (!answers.length) return;
    Object.values(T.texts).forEach(p => {
      if (!p || p._echo || !p.raw || !p.raw.trim()) return;
      if (!answers.some(a => isEcho(p.raw, a))) return;
      p._echo = true;
      p.node.hidden = true;
      p.node.replaceChildren();
    });
    const leftover = Object.values(T.texts).some(p => p && p.raw && p.raw.trim() && !p._echo);
    rows.forEach(row => {
      if (!row._d || row._d.ok == null) return;
      if (leftover) row.open = false;
      else if (row._d.ok) row.open = true;
    });
  }
  function fillSub(row, done) {
    const d = row._d;
    const task = String((d.args && d.args.task) || d.task || "").trim() || "Sub-agent";
    let status = d.status;
    if (!status) status = !done ? (d.waiting ? "waiting" : "running") : (d.ok ? "finished" : (d.decision === "deny" || d.decision === "cancelled" ? "stopped" : "failed"));
    const label = {running: "running", finished: "finished", stopped: "stopped", failed: "failed", waiting: "needs you"}[status] || status;
    row._say.textContent = task;
    row._st.replaceChildren(el("span", {class: "ag-sub-pill " + status, text: label}));
    const legacy = splitLegacyTrace(d.output);
    const trace = Array.isArray(d.trace) && d.trace.length ? d.trace.map(x => String(x)) : legacy.trace;
    const steps = d.steps != null ? d.steps : 0;
    const tokens = d.tokens != null ? d.tokens : 0;
    const shown = subShown(d);
    const nodes = [
      el("div", {class: "ag-sub-meta"}, el("span", {text: steps === 1 ? "1 step" : steps + " steps"}), el("span", {text: tokens + " tokens"})),
    ];
    if (shown) {
      const box = el("div", {class: "ag-sub-result ag-md"});
      paintRich(box, shown);
      nodes.push(box);
    }
    if (trace.length) nodes.push(el("details", {class: "ag-sub-trace"}, el("summary", {text: "Trace"}), el("ol", {}, ...trace.map(t => el("li", {text: t})))));
    if (A.adv) nodes.push(el("pre", {text: JSON.stringify({task, arguments: d.args || {}}, null, 2)}));
    if (!done) nodes.push(el("button", {type: "button", class: "b ag-sub-stop", text: "Stop sub-agent", onclick: () => stop()}));
    row._body.replaceChildren(...nodes.filter(n => n != null));
    if (!done) row.open = true;
    else if (!row._settled) row._settled = true;
  }
  function fillTool(row) {
    const d = row._d, b = row._body, out = d.output == null ? null : String(d.output);
    const done = d.ok != null;
    row.classList.toggle("running", !done && !d.waiting); row.classList.toggle("waiting", !!d.waiting);
    row.classList.toggle("bad", done && !d.ok);
    if (d.name === "spawn_agent") { fillSub(row, done); if (row._T) placeNotes(row._T); return; }
    row._say.textContent = done ? (d.done || d.say || "") : (d.say || "");
    row._st.replaceChildren(done ? ico(d.ok ? "check" : "x") : d.waiting ? el("span", {class: "ag-wait", text: "needs you"}) : el("span", {class: "spin"}));
    if (done && !d.ok) row._st.append(el("span", {class: "ag-why", text: d.decision === "deny" ? "skipped" : d.decision === "timeout" ? "no answer" : "failed"}));
    const nodes = [
      el("div", {class: "ag-io"}, el("span", {class: "ag-io-k", text: "Input"}),
        A.adv ? el("button", {type: "button", class: "link", text: "Copy JSON", onclick: () => copyText(JSON.stringify({name: d.name, arguments: d.args, result: d.output}, null, 2))}) : null),
      el("pre", {text: A.adv ? JSON.stringify({tool: d.name, arguments: d.args}, null, 2) : friendlyArgs(d)}),
      out != null ? el("div", {class: "ag-io"}, el("span", {class: "ag-io-k", text: d.ok ? "Output" : "What happened"}),
        d.secs != null ? el("span", {class: "tiny", text: `${d.secs}s`}) : null) : null,
      out != null ? el("pre", {text: A.adv ? out : out.slice(0, 1500) + (out.length > 1500 ? "\n..." : "")}) : null,
    ].filter(n => n != null);
    b.replaceChildren(...nodes);
  }
  function placeNotes(T) {
    const d = T && T.notesEl;
    if (!d || !T.parts) return;
    const cards = $$(".ag-subcard", T.parts);
    const last = cards.length ? cards[cards.length - 1] : null;
    if (last) {
      if (d.previousElementSibling !== last) T.parts.insertBefore(d, last.nextSibling);
      return;
    }
    if (d.parentNode !== T.parts) {
      if (T.pending && T.pending.parentNode === T.parts) T.parts.insertBefore(d, T.pending);
      else T.parts.append(d);
    }
  }
  function paintNotes(T, notes) {
    const list = (Array.isArray(notes) ? notes : []).filter(n => n && n.key);
    if (!list.length) {
      if (T.notesEl) T.notesEl.remove();
      T.notesEl = null;
      return;
    }
    if (!T.notesEl) {
      const body = el("div", {class: "ag-notes-b"});
      T.notesEl = el("details", {class: "ag-notes", open: true},
        el("summary", {}, ico("pin"), el("span", {class: "ag-say", text: "Shared notes"})), body);
      T.notesBody = body;
    }
    T.notesEl.open = true;
    T.notesBody.replaceChildren(...list.map(n => el("div", {class: "ag-note-row"},
      el("span", {class: "ag-note-k", text: String(n.key)}),
      el("span", {class: "ag-note-t", text: n.text == null ? "" : String(n.text)}))));
    placeNotes(T);
  }
  function approvalCard(T, d) {
    const left = el("span", {class: "ag-left"});
    const btns = el("div", {class: "ag-ap-btns"},
      el("button", {type: "button", class: "b pri", text: "Allow once", onclick: () => answer("once")}),
      d.always_ok ? el("button", {type: "button", class: "b", text: "Always for this chat", onclick: () => answer("always")}) : null,
      el("button", {type: "button", class: "b", text: "Deny", onclick: () => answer("deny")}));
    const card = el("div", {class: "ag-ap", id: "ag-ap-" + d.approval_id, role: "group", "aria-label": "approval needed"},
      el("div", {class: "ag-ap-h"}, ico("shield", "ag-ap-ic"), el("b", {class: "ag-q", text: d.question})),
      d.detail ? el("p", {class: "ag-d", text: d.detail}) : null,
      A.adv ? el("pre", {text: JSON.stringify({tool: d.name, arguments: d.args}, null, 2)}) : null,
      btns, left);
    const runId = A.run;
    async function answer(dec) {
      $$("button", btns).forEach(b => b.disabled = true);
      try { await api("/api/chat/approve", {body: {run_id: runId, approval_id: d.approval_id, decision: dec}}); }
      catch (e) { $$("button", btns).forEach(b => b.disabled = false); notice(e.message, "bad"); }
    }
    const until = Date.now() + (d.timeout_s || 300) * 1000;
    const tick = () => { const s = Math.max(0, Math.round((until - Date.now()) / 1000)); left.textContent = `Skipped automatically in ${fmtDur(s)} if you don't answer.`; };
    tick(); card._timer = setInterval(tick, 1000);
    const row = T.tools[d.call_id];
    if (row && row.nextSibling) T.parts.insertBefore(card, row.nextSibling); else T.parts.append(card);
    setTimeout(() => { const b = $("button", btns); if (b && document.activeElement === document.body) b.focus(); }, 50);
    return card;
  }
  function approvalAnswered(d) {
    const card = $("#ag-ap-" + d.approval_id, R.col); if (!card) return;
    clearInterval(card._timer);
    const words = {once: "Allowed once", always: "Allowed for this chat", deny: "Denied", timeout: "Skipped: no answer in time", cancelled: "Stopped"}[d.decision] || d.decision;
    card.replaceChildren(ico("shield", "ag-ap-ic"), el("span", {text: words}));
    card.className = "ag-ap answered " + (d.decision === "once" || d.decision === "always" ? "ok" : "no");
  }
  function notice(text, kind, retry, host) {
    const n = el("div", {class: "ag-note " + (kind || "")}, kind === "wait" ? el("span", {class: "spin"}) : null, el("span", {text}),
      retry ? el("button", {type: "button", class: "b", text: "Retry", onclick: () => { n.remove(); retry(); }}) : null);
    (host || (A.cur ? A.cur.parts : R.col)).append(n);
    scrollEnd();
    return n;
  }
  function metaText(d, secs) {
    const tm = d.timings || {}, parts = [];
    if (d.usage && d.usage.completion_tokens) parts.push(`${Math.round(d.usage.completion_tokens).toLocaleString("en-US")} tokens`);
    else if (tm.predicted_n) parts.push(`${Math.round(tm.predicted_n).toLocaleString("en-US")} tokens`);
    if (tm.predicted_per_second) parts.push(`${(+tm.predicted_per_second).toFixed(1)} tok/s`);
    if (d.steps > 1) parts.push(`${d.steps} steps`);
    if (secs) parts.push(`${secs < 10 ? secs.toFixed(1) : Math.round(secs)}s`);
    return parts.join(" · ");
  }
  function footer(T, d, secs, endType) {
    const isLast = () => T.idx === A.turns.length - 1;
    const copyBtn = iconBtn("copy", "Copy reply", () => copyText(T.final || Object.values(T.texts).map(p => pxaMd.splitThink(p.raw).rest).join("\n\n")));
    const regen = iconBtn("redo", "Regenerate", () => { if (isLast()) resend(T.idx, A.turns[T.idx]); else toast("Only the latest reply can be regenerated. Edit a message to branch from it.", true); });
    const msg = {role: "assistant", idx: T.idx, end: endType, data: d || null, el: T.m, get text() { return T.final || Object.values(T.texts).map(p => pxaMd.splitThink(p.raw).rest).join("\n\n"); }};
    T.m._plug = msg; T.msg = msg;
    T.foot.replaceChildren(el("div", {class: "ag-tb"}, copyBtn, regen, ...plugActs(msg)), el("span", {class: "ag-meta", text: endType === "run.done" ? metaText(d || {}, secs) : ""}));
    hk("render", T.m, msg);
  }

  // ---------------- turns ----------------
  function renderSaved(t, idx) {
    userMsg(t.user, idx);
    const T = asstMsg(idx), end = t.end || {}, byId = {};
    for (const s of t.steps || []) byId[s.call_id] = s;
    const parts = t.parts || [{k: "text", step: 0, text: t.assistant || ""}].concat((t.steps || []).map(s => ({k: "tool", call_id: s.call_id})));
    if (t.think) { const k = thinkRow(T); k.body.textContent = t.think; thinkDone(T, t.think_s); }
    for (const p of parts) {
      if (p.k === "tool" && byId[p.call_id]) { const s = byId[p.call_id]; toolRow(T, Object.assign({}, s, {waiting: false})); }
      else if (p.k === "text") { const tp = textPart(T, "s" + p.step + "_" + Object.keys(T.texts).length); tp.raw = p.text; paint(T, tp, false); }
    }
    if (Array.isArray(t.notes) && t.notes.length) paintNotes(T, t.notes);
    dedupeSubAnswers(T);
    T.final = t.assistant || "";
    if (end.type === "run.error") notice(friendlyErr(end.data || {}), "bad", () => resend(idx, t.user), T.parts);
    else if (end.type === "run.cancelled") notice("Stopped.", "quiet", null, T.parts);
    else if (end.type === "run.done" && (end.data || {}).empty) notice("The model sent an empty reply.", "quiet", () => resend(idx, t.user), T.parts);
    footer(T, end.data, null, end.type);
    return T;
  }
  function friendlyErr(d) {
    return d.error || "Something went wrong.";
  }
  async function send() {
    let text = R.input.value.trim();
    if (!text || A.run) return;
    const sm = text.match(/^\/(\S+)(?:\s+([\s\S]*))?$/), cmd = sm && X.slash.get(sm[1]);
    if (cmd) {           // a registered slash command runs instead of sending; a string it returns is sent instead
      R.input.value = ""; grow();
      let r; try { r = await cmd(sm[2] || "", G); } catch (e) { toast(e.message || String(e), true); return; }
      if (typeof r !== "string" || !r.trim() || A.run) return;
      text = r.trim(); R.input.value = text;
    }
    if (!A.servers.some(s => s.key === A.selected)) { notice("No model is running yet. Start one on the Launch tab, then come back.", "bad", null, R.col); return; }
    const refs = A.refs.slice();
    R.input.value = ""; grow();
    A.refs = []; paintRefs(); closeAt();
    startTurn(text, null, refs);
  }
  function resend(idx, text) {
    if (A.run) { toast("Wait for the answer, or press Stop.", true); return; }
    // drop that turn and everything after it, on screen and (with rewind) on disk
    $$(".ag-u, .ag-a, .ag-note", R.col).forEach(n => { if (n.dataset.turn == null ? false : +n.dataset.turn >= idx) n.remove(); });
    $$(":scope > .ag-note", R.col).forEach(n => n.remove());
    const saved = idx < A.turns.length && A.session;
    if (saved) {
      const prev = A.branchAt[idx];
      const n = prev && prev.n > 1 ? prev.n + 1 : 2;
      Object.keys(A.branchAt).forEach(k => { if (+k >= idx) delete A.branchAt[k]; });
      A.branchAt[idx] = {i: n, n: n};     // painted now: the run's save finishes after the reply, too late for the check
    }
    A.turns = A.turns.slice(0, idx);
    startTurn(text, saved ? idx : null);
  }
  async function startTurn(text, rewind, refs) {
    const idx = A.turns.length;
    A.turns.push(text);
    userMsg(text, idx, refs);
    const T = asstMsg(idx); A.cur = T; A.stick = true;
    T.parts.append(T.pending); T.t0 = Date.now();
    scrollEnd(true);
    const body = Object.assign({message: text, server: A.selected, preset: A.preset, session_id: A.session, advanced: A.adv}, A.adv ? advSettings() : {});
    if (A.adv) body.use_memory = A.useMem;
    body.incognito = !!A.incognito;
    if (!A.web) body.web = false;
    if (rewind != null) body.rewind = rewind;
    if (refs && refs.length) body.refs = refs.map(r => r.id);
    body.sub_max = subNum(R.subMax, "agent.subMax", 1, 4, 2);
    body.sub_steps = subNum(R.subSteps, "agent.subSteps", 1, 8, 4);
    body.sub_seconds = subNum(R.subSecs, "agent.subSecs", 15, 120, 60);
    A.run = "pending"; sendState();
    if (HK.beforeRun.length) await hkAsync("beforeRun", body, G);
    let r;
    try { r = await api("/api/chat/run", {body, timeout: 20000}); }
    catch (e) {
      A.run = null; A.cur = null; sendState(); T.pending.remove();
      A.turns.pop();                         // nothing was saved: Retry sends it again as a new turn
      notice(e.message, "bad", () => { T.m.remove(); $$(".ag-u", R.col).forEach(n => { if (+n.dataset.turn === idx) n.remove(); }); startTurn(text, rewind); }, T.parts);
      return;
    }
    if (A.session !== r.session_id) { A.session = r.session_id; hk("sessionChange", A.session, G); }
    store.set("agent.session", A.incognito ? null : A.session);
    A.run = r.run_id; A.seq = 0; A.retries = 0; sendState();
    if (!A.turns.slice(0, -1).length) { A.title = text.length > 60 ? text.slice(0, 57) + "..." : text; R.title.textContent = A.title; }
    listen();
    loadSessions();
  }
  function listen() {
    if (A.es) A.es.close();
    const es = new EventSource(`/api/chat/events?run=${encodeURIComponent(A.run)}&since=${A.seq}`);
    A.es = es;
    const types = ["run.started", "status", "step", "text.delta", "think.delta", "text.retract", "tool.call", "approval.request", "approval.answered", "tool.result", "subagent", "notes", "run.done", "run.error", "run.cancelled"]
      .concat(Array.from(X.runTypes).filter(t => !/^(run\.|status$|step$|text\.|think\.|tool\.|approval\.)/.test(t)));   // plugin types: only the delta hook sees them
    for (const t of types) es.addEventListener(t, m => {
      let ev; try { ev = JSON.parse(m.data); } catch (e) { return; }
      if (ev.seq <= A.seq) return;
      A.seq = ev.seq; A.retries = 0;
      if (A.reconn) { A.reconn.remove(); A.reconn = null; }
      const T = A.cur;
      onEvent(ev);
      if (T) { hk("delta", ev, T.m); if (/^run\.(done|error|cancelled)$/.test(ev.type)) hk("messageDone", T.msg || {role: "assistant", idx: T.idx, end: ev.type, data: ev.data, el: T.m}, G); }
    });
    es.onerror = () => {
      es.close(); if (A.es === es) A.es = null;
      if (!A.run || A.run === "pending") return;
      A.retries++;
      if (A.retries === 2 && !A.reconn) A.reconn = notice("Reconnecting to PXA Control...", "wait");
      if (A.retries > 12) {
        if (A.reconn) { A.reconn.remove(); A.reconn = null; }
        notice("Lost the connection to PXA Control. The answer keeps going on the server.", "bad", () => { A.retries = 0; listen(); });
        return;
      }
      setTimeout(() => { if (A.run && !A.es) listen(); }, Math.min(8000, 800 * A.retries));
    };
  }
  function onEvent(ev) {
    const T = A.cur, d = ev.data || {};
    if (!T) return;
    if (T.pending.parentNode && !["run.started", "step", "status"].includes(ev.type)) T.pending.remove();
    switch (ev.type) {
      case "run.started":
        if (d.user && T.userEl && !T.userEl._text) { T.userEl._text = d.user; T.userEl._bubble.textContent = d.user; A.turns[T.idx] = d.user; }
        break;
      case "status": {
        if (T.status) T.status.remove();
        T.status = notice(d.text, d.kind === "reattached" ? "ok" : "wait");
        if (d.kind === "reattached") { const n = T.status; T.status = null; setTimeout(() => n.remove(), 4000); }
        break;
      }
      case "step": T.stepT = ev.t; if (d.n > 1 && !T.pending.parentNode) T.parts.append(T.pending); break;
      case "think.delta": {
        const k = thinkRow(T); if (!k.t0) k.t0 = T.stepT || ev.t;
        k.text += (d.text == null ? "" : d.text); k.body.textContent = k.text; break;
      }
      case "text.delta": {
        if (T.status) { T.status.remove(); T.status = null; }
        if (T.think && T.think.live) { T.think.t1 = ev.t; thinkDone(T); }
        const p = textPart(T, d.step); p.raw += (d.text == null ? "" : d.text); if (d.fallback) p.node.classList.add("ag-fallback"); schedulePaint(T, p); break;
      }
      case "text.retract": { const p = textPart(T, d.step); p.raw = d.text || ""; paint(T, p, false); break; }
      case "tool.call":
        if (T.think && T.think.live) { T.think.t1 = ev.t; thinkDone(T); }
        $$(".ag-cursor", T.m).forEach(c => c.remove());
        toolRow(T, d); if (T.notesEl) placeNotes(T); break;
      case "approval.request": { const row = T.tools[d.call_id]; if (row) { row._d.waiting = true; row._fill(); } approvalCard(T, d); break; }
      case "approval.answered": { const row = T.tools[d.call_id]; if (row) { row._d.waiting = false; row._d.decision = d.decision; row._fill(); } approvalAnswered(d); break; }
      case "subagent": {
        const row = T.tools[d.call_id];
        if (!row) break;
        if (d.status) row._d.status = d.status;
        if (d.step != null) row._d.steps = d.step;
        if (d.tokens != null) row._d.tokens = d.tokens;
        if (Array.isArray(d.trace)) row._d.trace = d.trace;
        if (d.task) row._d.task = d.task;
        row._fill();
        break;
      }
      case "notes": paintNotes(T, d.notes); break;
      case "tool.result": {
        const row = T.tools[d.call_id];
        if (row) {
          Object.assign(row._d, {ok: d.ok, output: d.output, done: d.done, secs: d.secs, memory: d.memory});
          ["status", "steps", "tokens", "trace", "task"].forEach(k => { if (d[k] != null) row._d[k] = d[k]; });
          row._fill();
        }
        if (d.memory) loadMemory();
        T.parts.append(T.pending); break;
      }
      case "run.done": case "run.error": case "run.cancelled": {
        T.pending.remove(); if (T.status) { T.status.remove(); T.status = null; }
        if (T.think && T.think.live) { T.think.t1 = ev.t; thinkDone(T); }
        Object.values(T.texts).forEach(p => { if (p._raf) { cancelAnimationFrame(p._raf); p._raf = 0; } paint(T, p, false); });
        $$(".ag-cursor", T.m).forEach(c => c.remove());
        $$(".ag-ap:not(.answered)", T.m).forEach(c => { clearInterval(c._timer); c.classList.add("answered", "no"); c.replaceChildren(ico("shield", "ag-ap-ic"), el("span", {text: "Not needed any more"})); });
        if (ev.type === "run.done") {
          T.final = d.text || "";
          if (d.text && !Object.values(T.texts).some(p => p.raw.trim())) { const p = textPart(T, "final"); p.raw = d.text; paint(T, p, false); }
          dedupeSubAnswers(T);
          if (d.empty) notice("The model sent an empty reply.", "quiet", () => resend(T.idx, A.turns[T.idx]));
        } else if (ev.type === "run.error") notice(friendlyErr(d), "bad", () => resend(T.idx, A.turns[T.idx]));
        else notice("Stopped.", "quiet");
        footer(T, d, (Date.now() - T.t0) / 1000, ev.type);
        finish();
        break;
      }
    }
    scrollEnd();
  }
  function finish() {
    if (A.es) A.es.close(); A.es = null; A.run = null; A.cur = null;
    if (A.reconn) { A.reconn.remove(); A.reconn = null; }
    sendState(); loadSessions(); loadMemory();
    if (document.activeElement === document.body || document.activeElement === R.send) R.input.focus();
  }
  async function stop() {
    if (!A.run || A.run === "pending") return;
    try { await api("/api/chat/cancel", {body: {run_id: A.run}}); } catch (e) { notice(e.message, "bad"); }
  }

  // ---------------- sessions ----------------
  async function loadSessions() {
    let d; try { d = await api("/api/chat/sessions"); } catch (e) { return; }
    A.sessions = d.sessions || [];
    renderList();
  }
  function renderList() {
    if (R.list.querySelector(".ag-item.renaming, .ag-item.confirm")) return;     // don't yank an open rename/delete
    const rows = (A.sessions || []).slice(0, 200);
    if (A.incognito && A.session && !rows.some(s => s.id === A.session)) {
      rows.unshift({id: A.session, title: (A.title || "New chat") + " · incognito", updated: Date.now() / 1000, incognito: true});
    }
    if (!rows.length) { R.list.replaceChildren(el("div", {class: "ag-list-empty", text: "Your chats show up here."})); return; }
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const groups = [["Today", []], ["Earlier", []]];
    for (const s of rows) groups[(s.updated || 0) * 1000 >= today.getTime() ? 0 : 1][1].push(s);
    const out = [];
    for (const [name, items] of groups) {
      if (!items.length) continue;
      out.push(el("div", {class: "ag-group", text: name}));
      for (const s of items) out.push(listItem(s));
    }
    R.list.replaceChildren(...out);
  }
  function listItem(s) {
    const it = el("div", {class: "ag-item" + (s.id === A.session ? " on" : ""), "data-id": s.id});
    const open = el("button", {type: "button", class: "ag-item-t", title: s.title, "aria-current": s.id === A.session ? "true" : "false", onclick: () => openSession(s.id)}, el("span", {text: s.title}));
    const acts = el("span", {class: "ag-item-acts"},
      iconBtn("edit", "Rename", () => rename(it, s)), iconBtn("trash", "Delete", () => confirmDelete(it, s)));
    it.append(open, acts);
    return it;
  }
  function rename(it, s) {
    const inp = el("input", {type: "text", class: "ag-rename", value: s.title, maxlength: "80", "aria-label": "chat name"});
    it.classList.add("renaming"); it.replaceChildren(inp); inp.focus(); inp.select();
    let done = false;
    const end = async save => {
      if (done) return; done = true;
      const v = inp.value.trim();
      it.classList.remove("renaming");
      if (save && v && v !== s.title) {
        try { await api("/api/chat/rename", {body: {id: s.id, title: v}}); s.title = v; if (s.id === A.session) { A.title = v; R.title.textContent = v; } }
        catch (e) { toast(e.message, true); }
      }
      renderList();
    };
    inp.addEventListener("keydown", e => { if (e.key === "Enter") end(true); if (e.key === "Escape") { e.stopPropagation(); end(false); } });
    inp.addEventListener("blur", () => end(true));
  }
  function confirmDelete(it, s) {
    if (A.run && s.id === A.session) { toast("Wait for the answer, or press Stop.", true); return; }
    it.classList.add("confirm");
    it.replaceChildren(el("span", {class: "ag-item-q", text: "Delete this chat and its files?"}),
      el("button", {type: "button", class: "b danger ag-del-yes", text: "Delete", onclick: async () => {
        try { await api("/api/chat/session", {method: "DELETE", body: {id: s.id}}); } catch (e) { toast(e.message, true); }
        it.classList.remove("confirm");
        if (s.id === A.session) newChat(); else loadSessions();
      }}),
      el("button", {type: "button", class: "b", text: "Cancel", onclick: () => { it.classList.remove("confirm"); renderList(); }}));
  }
  async function openSession(id, quiet) {
    if (A.run) { toast("Wait for the current answer, or press Stop.", true); return; }
    let s; try { s = await api("/api/chat/session?id=" + encodeURIComponent(id)); }
    catch (e) { if (!quiet) toast(e.message, true); A.session = null; store.set("agent.session", null); welcome(); loadSessions(); return; }
    A.session = s.id; store.set("agent.session", s.incognito ? null : s.id);
    if (s.preset && s.preset !== A.preset && A.presets.some(p => p.id === s.preset)) setPreset(s.preset);
    if (s.server_key) A.selected = s.server_key;
    A.title = s.title || "Chat"; R.title.textContent = A.title;
    A.useMem = !s.no_memory; R.usemem.checked = A.useMem;
    A.incognito = !!s.incognito; syncIncog();
    R.col.replaceChildren();
    A.turns = (s.turns || []).map(t => t.user);
    A.branchAt = {};
    (s.turns || []).forEach((t, i) => {
      if (t.branch && t.branch.n > 1) A.branchAt[i] = {i: t.branch.i, n: t.branch.n};
      renderSaved(t, i);
    });
    if (!(s.turns || []).length && !s.running) welcome();
    if (s.running) {                         // re-attach to an answer still going (page reload, another tab)
      const idx = A.turns.length; A.turns.push("");
      const u = userMsg("", idx), T = asstMsg(idx); T.userEl = u; T.t0 = Date.now(); T.parts.append(T.pending);
      A.cur = T; A.run = s.running; A.seq = 0; listen();
    }
    renderServerChip(); renderList(); scrollEnd(true); loadHost();
    if (window.innerWidth <= 900) setSide(false);
    hk("sessionChange", A.session, G);
  }
  function newChat() {
    if (A.run) { toast("Wait for the current answer, or press Stop.", true); return; }
    A.session = null; store.set("agent.session", null); A.turns = []; A.branchAt = {};
    A.useMem = true; R.usemem.checked = true;
    A.incognito = false; syncIncog();
    A.refs = []; paintRefs(); closeAt();
    welcome(); renderList(); loadHost(); R.input.focus();
    if (window.innerWidth <= 900) setSide(false);
    hk("sessionChange", null, G);
  }
  function exportChat(fmt) { if (!A.session) { toast("Nothing to save yet: start a chat first.", true); return; } location.href = `/api/chat/export?id=${encodeURIComponent(A.session)}&format=${fmt}`; }
  function copyCurl() {
    const body = Object.assign({message: R.input.value.trim() || "Hello!", server: A.selected, preset: A.preset, advanced: true}, advSettings());
    const base = location.origin;
    const cmd = `# start an agent turn, then stream its events (add -H "X-PXA-Token: $PXA_TOKEN" when Control has a token)\n` +
      `RUN=$(curl -s ${base}/api/chat/run -H 'Content-Type: application/json' -d '${JSON.stringify(body).replace(/'/g, "'\\''")}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')\n` +
      `curl -N "${base}/api/chat/events?run=$RUN"`;
    copyText(cmd);
  }

  // ---------------- memory ----------------
  // "Memory updated" row: what was saved or forgotten, with Undo and a way into the Memory panel.
  function memRow(T, d) {
    const st = el("span", {class: "ag-st"}, el("span", {class: "spin"}));
    const say = el("span", {class: "ag-say", text: "Updating memory"});
    const body = el("div", {class: "ag-tool-b ag-mem-b"});
    const row = el("details", {class: "ag-tool ag-memrow running"}, el("summary", {}, ico("brain"), say, st), body);
    row._d = Object.assign({}, d); row._say = say; row._st = st; row._body = body;
    row._fill = () => fillMem(row);
    row._fill();
    T.parts.append(row); T.tools[d.call_id] = row;
    return row;
  }
  function fillMem(row) {
    const d = row._d, m = d.memory || {}, done = d.ok != null, b = row._body;
    row.classList.toggle("running", !done); row.classList.toggle("bad", done && !d.ok);
    row._st.replaceChildren(done ? ico(d.ok ? "check" : "x") : el("span", {class: "spin"}));
    const fact = (m.fact && m.fact.text) || (d.args && (d.args.fact || d.args.text)) || "";
    if (!done) { row._say.textContent = "Updating memory"; b.replaceChildren(el("div", {class: "ag-mem-q", text: fact})); return; }
    if (row._undone) {
      row._say.textContent = "Memory change undone";
      b.replaceChildren(el("div", {class: "ag-mem-k", text: m.op === "forget" ? "Put back:" : "Removed again:"}), el("div", {class: "ag-mem-q", text: fact}));
      return;
    }
    if (!d.ok) {
      row._say.textContent = m.op === "refused" ? "Not saved to memory" : "Memory not changed";
      const why = m.why || String(d.output || "").replace(/^error:\s*(not saved:\s*)?/, "");
      b.replaceChildren(el("div", {class: "ag-mem-k", text: why.charAt(0).toUpperCase() + why.slice(1) + "."}));
      return;
    }
    row._say.textContent = "Memory updated";
    const verb = m.op === "forget" ? "Forgot" : m.status === "same" ? "Already remembered" : m.status === "updated" ? "Updated" : "Remembered";
    const undo = el("button", {type: "button", class: "b ag-mem-undo", text: "Undo", onclick: async () => {
      undo.disabled = true;
      try {
        if (m.op === "forget") await api("/api/chat/memory", {body: {op: "restore", fact: m.fact}});
        else await api("/api/chat/memory", {body: {op: "delete", id: m.fact.id}});
        row._undone = true; row._fill(); loadMemory(); toast(m.op === "forget" ? "Put back in memory" : "Removed from memory");
      } catch (e) { undo.disabled = false; toast(e.message, true); }
    }});
    b.replaceChildren(
      el("div", {class: "ag-mem-k", text: verb + ":"}), el("div", {class: "ag-mem-q", text: fact}),
      el("div", {class: "ag-mem-acts"}, m.status === "same" || m.status === "updated" ? null : undo,
        el("button", {type: "button", class: "link", text: "Manage memory", onclick: () => openMemory(true)})));
  }
  async function loadMemory() {
    try { A.mem = await api("/api/chat/memory", {timeout: 8000}); } catch (e) { return; }
    renderMemory();
  }
  const fmtDay = t => { const d = new Date((t || 0) * 1000); return d.toLocaleDateString(undefined, {day: "numeric", month: "short", year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric"}); };
  function fmtWhen(t) {
    if (!t) return "Saved";
    const d = new Date(t * 1000), now = new Date();
    const day = d.toLocaleDateString(undefined, {day: "numeric", month: "short", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric"});
    const hm = d.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit"});
    return day + ", " + hm;
  }
  function memFrom(f) {
    const title = String(f.source_title || "").replace(/\[\[[^\]]*\]\]/g, " ").replace(/\s+/g, " ").trim();
    let where = "";
    if (f.source === "you") where = "by you";
    else if (f.source === "import") where = "from a file";
    else if (title) where = "from " + title;
    else if (f.source) where = "from a chat";
    const when = fmtWhen(f.updated);
    return [where, when].filter(Boolean).join(" · ");
  }
  function syncIncog() {
    if (R.incogBadge) R.incogBadge.hidden = !A.incognito;
    if (R.incog) R.incog.checked = !!A.incognito;
    if (A.incognito && R.usemem) { A.useMem = false; R.usemem.checked = false; }
  }
  // ---- web search: per-chat switch + provider settings (saved on this machine, in Control's chat folder) ----
  function buildWeb() {
    const F = {};
    const inp = (id, ph, type) => el("input", {type: type || "text", id: "ag-ws-" + id, placeholder: ph, autocomplete: "off"});
    F.on = el("input", {type: "checkbox", id: "ag-ws-on", role: "switch", checked: true, onchange: () => { A.web = F.on.checked; toast(A.web ? "Web search is on for this chat" : "Web search is off for this chat"); }});
    F.prov = el("select", {id: "ag-ws-prov", onchange: show},
      el("option", {value: "builtin", text: "Built-in (DuckDuckGo, no setup)"}), el("option", {value: "searxng", text: "SearXNG (self-hosted)"}),
      el("option", {value: "brave", text: "Brave Search API key"}), el("option", {value: "tavily", text: "Tavily API key"}));
    F.searxng_url = inp("searxng", "http://127.0.0.1:8080"); F.brave_key = inp("brave", "Brave API key", "password");
    F.tavily_key = inp("tavily", "Tavily API key", "password"); F.reader_url = inp("reader", "optional, e.g. http://127.0.0.1:11235");
    F.reader_token = inp("rtoken", "reader token, if it needs one", "password");
    F.found = el("div", {class: "tiny", id: "ag-ws-found"}); F.out = el("div", {class: "tiny", id: "ag-ws-out", style: "white-space:pre-wrap;margin-top:8px"});
    const row = (k, label, node) => { node.dataset.k = k; return el("label", {class: "f", id: "ag-ws-row-" + k}, label, node); };
    F.rows = {searxng: row("searxng", "SearXNG address", F.searxng_url), brave: row("brave", "Brave key", F.brave_key), tavily: row("tavily", "Tavily key", F.tavily_key)};
    function show() { for (const k in F.rows) F.rows[k].hidden = F.prov.value !== k; }
    const body = () => ({provider: F.prov.value, searxng_url: F.searxng_url.value, brave_key: F.brave_key.value, tavily_key: F.tavily_key.value,
                         reader_url: F.reader_url.value, reader_token: F.reader_token.value});
    F.save = async () => { try { await api("/api/chat/search", {body: body()}); toast("Search settings saved"); F.load(); } catch (e) { toast(e.message, true); } };
    F.test = async () => {
      F.out.textContent = "Searching...";
      try {
        const r = await api("/api/chat/search/test", {body: Object.assign(body(), {query: "pxa llama.cpp pascal"}), timeout: 40000});
        F.out.textContent = r.ok ? (r.note ? r.note + "\n" : "") + r.results.map(x => "- " + x.title + "\n  " + x.url).join("\n") : "Search failed: " + r.error;
      } catch (e) { F.out.textContent = "Search failed: " + e.message; }
    };
    F.load = async () => {
      try {
        const d = await api("/api/chat/search"), c = d.settings;
        F.prov.value = c.provider; F.searxng_url.value = c.searxng_url; F.reader_url.value = c.reader_url;
        F.brave_key.placeholder = c.brave_key_set ? "saved (leave blank to keep)" : "Brave API key";
        F.tavily_key.placeholder = c.tavily_key_set ? "saved (leave blank to keep)" : "Tavily API key";
        F.reader_token.placeholder = c.reader_token_set ? "saved (leave blank to keep)" : "reader token, if it needs one";
        F.found.replaceChildren();
        for (const [k, label] of [["searxng_url", "A SearXNG search"], ["reader_url", "A page reader (crawl4ai)"]]) {
          if (!d.found[k]) continue;
          F.found.append(el("div", {class: "row", style: "gap:8px;margin:6px 0"}, el("span", {text: label + " was found on this machine at " + d.found[k] + ". Use it?"}),
            el("button", {class: "b", text: "Use it", onclick: () => { F[k].value = d.found[k]; if (k === "searxng_url") F.prov.value = "searxng"; show(); F.save(); }})));
        }
        show();
      } catch (e) { F.out.textContent = e.message; }
    };
    R.ws = F;
    return el("div", {class: "ag-memp", id: "ag-wsp", hidden: true},
      el("div", {class: "ag-memp-card"},
        el("div", {class: "ag-memp-h"}, ico("search"), el("b", {text: "Web search"}), el("span", {class: "sp"}), iconBtn("close", "Close", () => openWeb(false))),
        el("div", {class: "ag-memp-b", style: "padding:12px 16px;display:grid;gap:8px;overflow:auto"},
          el("label", {class: "check"}, F.on, el("span", {}, el("b", {text: "Search the web in this chat"}), " ", el("small", {text: "off: the assistant does not search or open pages"}))),
          F.found,
          el("label", {class: "f"}, "Search with", F.prov), F.rows.searxng, F.rows.brave, F.rows.tavily,
          el("label", {class: "f"}, "Page reader (optional, better page text)", F.reader_url), el("label", {class: "f"}, "Reader token", F.reader_token),
          el("div", {class: "row", style: "gap:8px"}, el("button", {class: "b", id: "ag-ws-save", text: "Save", onclick: F.save}),
            el("button", {class: "b", id: "ag-ws-test", text: "Test search", onclick: F.test})),
          F.out,
          el("div", {class: "tiny", text: "Your questions go to the search you pick. The built-in one sends them to DuckDuckGo. Keys stay on this computer."}))));
  }
  function openWeb(on) { R.wsp.hidden = !on; if (on) { R.ws.on.checked = A.web; R.ws.out.textContent = ""; R.ws.load(); } }
  function buildMemory() {
    R.memOn = el("input", {type: "checkbox", id: "ag-mem-on", role: "switch", onchange: async () => {
      try { await api("/api/chat/memory", {body: {op: "settings", enabled: R.memOn.checked}}); toast(R.memOn.checked ? "Memory is on" : "Memory is off: chats won't see or save it"); loadMemory(); }
      catch (e) { R.memOn.checked = !R.memOn.checked; toast(e.message, true); }
    }});
    R.memAdd = el("input", {type: "text", id: "ag-mem-add", maxlength: "300", placeholder: "Add something, like \"I'm vegetarian\"", "aria-label": "add a memory"});
    const add = async () => {
      const t = R.memAdd.value.trim(); if (!t) return;
      try { const r = await api("/api/chat/memory", {body: {op: "add", text: t}}); R.memAdd.value = ""; toast(r.status === "added" ? "Saved to memory" : "Already in memory"); loadMemory(); }
      catch (e) { if (/secret/.test(e.message)) R.memAdd.value = ""; toast(e.message, true); }   // never leave a secret on screen
    };
    R.memAdd.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); add(); } });
    R.memList = el("div", {class: "ag-mem-list", id: "ag-mem-list"});
    R.memFoot = el("div", {class: "ag-mem-foot"});
    R.memK = el("input", {type: "number", id: "ag-mem-k", min: "0", max: "50"});
    R.memBudget = el("input", {type: "number", id: "ag-mem-budget", min: "50", max: "4000", step: "50"});
    const saveTune = async () => {
      try { await api("/api/chat/memory", {body: {op: "settings", k: R.memK.value, budget: R.memBudget.value}}); toast("Saved"); loadMemory(); }
      catch (e) { toast(e.message, true); loadMemory(); }
    };
    R.memK.addEventListener("change", saveTune); R.memBudget.addEventListener("change", saveTune);
    R.memRaw = el("pre", {class: "ag-mem-raw", id: "ag-mem-raw"});
    const file = el("input", {type: "file", accept: ".json,application/json", hidden: true, onchange: async () => {
      const f = file.files[0]; file.value = ""; if (!f) return;
      let j; try { j = JSON.parse(await f.text()); } catch (e) { toast("That file isn't JSON. Use a file from Export.", true); return; }
      try { const r = await api("/api/chat/memory", {body: {op: "import", facts: Array.isArray(j) ? j : j.facts}});
        toast(`Imported ${r.added}` + (r.skipped.length ? `, skipped ${r.skipped.length} (secrets or reply-style rules)` : "")); loadMemory(); }
      catch (e) { toast(e.message, true); }
    }});
    const p = el("div", {class: "ag-memp", id: "ag-memp", hidden: true, role: "dialog", "aria-label": "Memory"},
      el("div", {class: "ag-memp-card"},
        el("div", {class: "ag-memp-h"}, ico("brain"), el("b", {text: "Memory"}), el("span", {class: "sp"}), iconBtn("close", "Close memory", () => openMemory(false))),
        el("div", {class: "ag-memp-b"},
          el("p", {class: "ag-mem-intro", text: "Things the assistant remembers about you. Every chat can use them, on any server. They stay on this machine."}),
          el("label", {class: "ag-switch"}, R.memOn, el("span", {class: "ag-switch-ui", "aria-hidden": "true"}),
            el("span", {}, el("b", {text: "Remember things across chats"}), el("small", {text: "The assistant saves useful facts you tell it and uses them later."}))),
          el("div", {class: "ag-mem-addrow"}, R.memAdd, el("button", {type: "button", class: "b pri", id: "ag-mem-addb", text: "Add", onclick: add})),
          R.memQ = el("input", {type: "search", id: "ag-mem-q", placeholder: "Search memories", "aria-label": "search memories"}),
          R.memList, R.memFoot,
          el("details", {class: "ag-mem-adv ag-advonly", id: "ag-mem-adv"}, el("summary", {text: "Advanced"}),
            el("div", {class: "ag-grid2"},
              el("label", {class: "f", title: "how many relevant facts each chat sees, besides pinned ones"}, "Facts per message", R.memK),
              el("label", {class: "f", title: "the most tokens the memory block may take in the prompt"}, "Token budget", R.memBudget)),
            el("div", {class: "row", style: "gap:8px;flex-wrap:wrap;margin:8px 0"},
              el("button", {type: "button", class: "b", id: "ag-mem-export", text: "Export", onclick: () => { location.href = "/api/chat/memory/export"; }}),
              el("button", {type: "button", class: "b", id: "ag-mem-import", text: "Import", onclick: () => file.click()}), file,
              el("button", {type: "button", class: "b", text: "Copy JSON", onclick: () => copyText(JSON.stringify(A.mem.facts, null, 2))})),
            el("div", {class: "tiny", text: "Raw facts"}), R.memRaw))));
    p.addEventListener("click", e => { if (e.target === p) openMemory(false); });
    R.memQ.addEventListener("input", () => renderMemory());
    return p;
  }
  function openMemory(on) {
    R.memp.hidden = !on;
    if (on) { loadMemory(); if (window.innerWidth <= 900) setSide(false); R.memAdd.focus(); }
  }
  function renderMemory() {
    const m = A.mem || {facts: []}, facts = m.facts || [];
    const n = $("#ag-mem-n"); if (n) { n.textContent = m.enabled === false ? "off" : facts.length ? String(facts.length) : ""; n.classList.toggle("off", m.enabled === false); }
    if (!R.memList) return;
    R.memOn.checked = m.enabled !== false;
    R.memp.classList.toggle("ag-mem-disabled", m.enabled === false);
    if (document.activeElement !== R.memK) R.memK.value = m.k;
    if (document.activeElement !== R.memBudget) R.memBudget.value = m.budget;
    R.memRaw.textContent = JSON.stringify(facts, null, 1);
    if (R.memQ) R.memQ.hidden = !facts.length;
    if (!facts.length) {
      R.memList.replaceChildren(el("div", {class: "ag-mem-empty"}, ico("brain"),
        el("div", {text: "Nothing remembered yet."}),
        el("small", {text: "Tell the assistant something about you, like \"I live in Toronto\" or \"I'm vegetarian\", and it will keep it here."})));
      R.memFoot.replaceChildren();
      return;
    }
    const q = (R.memQ && R.memQ.value || "").trim().toLowerCase();
    const shown = q ? facts.filter(f => (f.text + " " + (f.source_title || "") + " " + (f.id || "")).toLowerCase().includes(q)) : facts;
    if (!shown.length) {
      R.memList.replaceChildren(el("div", {class: "ag-mem-empty", id: "ag-mem-none"},
        el("div", {text: "No memories match that search."})));
    } else R.memList.replaceChildren(...shown.map(memItem));
    R.memFoot.replaceChildren(el("button", {type: "button", class: "b ag-danger", id: "ag-mem-clear", text: "Clear all", onclick: confirmClear}),
      el("span", {class: "tiny", text: (q ? `${shown.length} match · ` : "") + `${facts.length} of ${m.limit || 200} · the oldest unpinned make room when full`}));
  }
  function memItem(f) {
    const it = el("div", {class: "ag-mem-it" + (f.pinned ? " pinned" : ""), "data-id": f.id});
    const view = () => {
      const pin = iconBtn("pin", f.pinned ? "Unpin (pinned facts are in every chat)" : "Pin: use in every chat", async () => {
        try { await api("/api/chat/memory", {body: {op: "pin", id: f.id, pinned: !f.pinned}}); loadMemory(); } catch (e) { toast(e.message, true); } }, {class: "ag-ib ag-mem-pin" + (f.pinned ? " on" : "")});
      it.replaceChildren(
        el("div", {class: "ag-mem-t"}, el("div", {class: "ag-mem-txt", text: f.text}),
          el("div", {class: "ag-mem-meta", text: (f.pinned ? "Pinned · " : "") + memFrom(f)})),
        el("div", {class: "ag-mem-btns"}, pin,
          iconBtn("edit", "Edit", edit, {class: "ag-ib ag-mem-edit"}),
          iconBtn("trash", "Delete", async () => {
            try { await api("/api/chat/memory", {body: {op: "delete", id: f.id}}); loadMemory();
              toastUndo("Deleted from memory", async () => { await api("/api/chat/memory", {body: {op: "restore", fact: f}}); loadMemory(); }); }
            catch (e) { toast(e.message, true); } }, {class: "ag-ib ag-mem-del"})));
    };
    const edit = () => {
      const inp = el("input", {type: "text", value: f.text, maxlength: "300", "aria-label": "edit memory"});
      const save = async () => {
        const t = inp.value.trim(); if (!t || t === f.text) { view(); return; }
        try { await api("/api/chat/memory", {body: {op: "update", id: f.id, text: t}}); loadMemory(); } catch (e) { toast(e.message, true); }
      };
      inp.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); save(); } if (e.key === "Escape") { e.stopPropagation(); e.preventDefault(); view(); } });
      it.replaceChildren(el("div", {class: "ag-mem-editrow"}, inp,
        el("button", {type: "button", class: "b pri", text: "Save", onclick: save}), el("button", {type: "button", class: "b", text: "Cancel", onclick: view})));
      inp.focus(); inp.select();
    };
    view();
    return it;
  }
  function toastUndo(text, fn) {
    const n = el("div", {class: "ag-mem-toast"}, el("span", {text}), el("button", {type: "button", class: "link", text: "Undo", onclick: async () => { n.remove(); try { await fn(); } catch (e) { toast(e.message, true); } }}));
    R.memp.querySelector(".ag-memp-card").append(n);
    setTimeout(() => n.remove(), 6000);
  }
  function confirmClear() {
    const n = (A.mem.facts || []).length;
    R.memFoot.replaceChildren(el("div", {class: "ag-mem-confirm", id: "ag-mem-confirm"},
      el("span", {text: `Forget all ${n} thing${n === 1 ? "" : "s"} the assistant remembers? This can't be undone. Tip: Export first if you might want them back.`}),
      el("div", {class: "row", style: "gap:8px"},
        el("button", {type: "button", class: "b ag-danger", id: "ag-mem-clear-yes", text: "Forget everything", onclick: async () => {
          try { await api("/api/chat/memory", {body: {op: "clear", confirm: true}}); toast("Memory cleared"); } catch (e) { toast(e.message, true); }
          loadMemory(); }}),
        el("button", {type: "button", class: "b", text: "Cancel", onclick: renderMemory}))));
  }

  window.pxaAgent = {state: A, reload: loadServers, md: pxaMd, newChat, openSession, openMemory, loadMemory};
  boot();
})();
