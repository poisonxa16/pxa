// agent-context.js: token meter against the selected server's context. /tokenize when it answers, chars/4 labelled when it does not.
(function () {
  "use strict";
  const G = window.PXAAgent;
  if (!G) return;
  const MAX = 400000;
  let timer = 0, seq = 0;
  function nctx() {
    const st = G.state || {};
    const s = (st.servers || []).filter(function (x) { return x.key === st.selected; })[0];
    const n = s && s.ctx && s.ctx.resolved;
    return n > 0 ? n : 0;
  }
  function transcript() {
    const box = (G.el && G.el.input && G.el.input.value) || "";
    const parts = [];
    const col = G.el && G.el.col;
    if (col) {
      const users = col.querySelectorAll(".ag-u");
      for (let i = 0; i < users.length; i++) if (users[i]._text) parts.push(String(users[i]._text));
      const replies = col.querySelectorAll(".ag-a .ag-md");
      for (let j = 0; j < replies.length; j++) {
        const t = replies[j].textContent || "";
        if (t.trim()) parts.push(t);
      }
    } else {
      const turns = (G.state && G.state.turns) || [];
      for (let k = 0; k < turns.length; k++) parts.push(String(turns[k]));
    }
    return parts.join("\n") + "\n" + box;
  }
  function paint(tokens, estimated) {
    const meter = document.getElementById("ag-ctx-meter");
    if (!meter) return;
    const n = nctx();
    const us = Math.max(0, Math.round(tokens) || 0).toLocaleString("en-US");
    const of = n ? " of " + n.toLocaleString("en-US") : "";
    meter.textContent = estimated ? "about " + us + of + " tokens (estimated)" : us + of + " tokens";
    meter.classList.toggle("ag-ctx-hot", !!(n && tokens / n > 0.75));
  }
  function count() {
    const mine = ++seq;
    const full = transcript();
    const local = Math.ceil(full.length / 4);
    paint(local, true);
    if (full.length > MAX) return;
    const server = (G.state && G.state.selected) || "";
    api("/api/chat/tokenize", {body: {server: server, text: full}, timeout: 8000}).then(function (d) {
      if (mine !== seq) return;
      if (d && typeof d.tokens === "number" && d.estimated === false) paint(d.tokens, false);
      else paint(local, true);
    }).catch(function () {
      if (mine === seq) paint(local, true);
    });
  }
  function schedule(now) {
    clearTimeout(timer);
    timer = setTimeout(count, now ? 0 : 400);
  }
  G.on("init", function () {
    const input = G.el && G.el.input;
    const host = input && (input.closest(".ag-comp") || input.parentNode);
    if (host && !document.getElementById("ag-ctx-meter")) {
      host.append(el("div", {id: "ag-ctx-meter", class: "ag-ctx", role: "status", "aria-live": "polite"}));
    }
    if (input) input.addEventListener("input", function () { schedule(false); });
    schedule(true);
    const t0 = Date.now();
    const wait = setInterval(function () {
      if (nctx() || Date.now() - t0 > 15000) {
        clearInterval(wait);
        if (nctx()) schedule(true);
      }
    }, 500);
  });
  G.on("messageDone", function () { schedule(true); });
  G.on("sessionChange", function () { schedule(true); });
})();
