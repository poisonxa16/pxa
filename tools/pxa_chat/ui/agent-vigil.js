// agent-vigil.js: Vigil lives in this Chat tab. The kill switch sits in the header.
(function () {
  "use strict";

  // Approval cards come from tools/anibal/ui/cards.js (QA F16/F19) and code changes from
  // diff.js, served by Control at /api/chat/vigil/ui/: one renderer for Control, the HUD and this tab.
  let cardsReady = null;
  function loadScript(name, global) {
    return new Promise(function (ok, bad) {
      if (window[global]) return ok(window[global]);
      const s = document.createElement("script");
      s.onload = function () { window[global] ? ok(window[global]) : bad(new Error(name)); };
      s.onerror = function () { bad(new Error(name)); };
      // a <script src> cannot carry the X-PXA-Token header: a single-use ticket for this one load (auth.js)
      const path = "/api/chat/vigil/ui/" + name;
      (window.PXA_AUTH ? window.PXA_AUTH.url(path) : Promise.resolve(path)).then(function (u) {
        s.src = u;
        document.head.appendChild(s);
      }, function () { bad(new Error(name)); });
    });
  }
  function loadCards() {
    if (window.VigilCards) return Promise.resolve(window.VigilCards);
    if (!cardsReady) cardsReady = loadScript("diff.js", "VigilDiff")
      .catch(function () { return null; })              // without diff.js a patch shows as text
      .then(function () { return loadScript("cards.js", "VigilCards"); })
      .catch(function (e) { cardsReady = null; throw e; });
    return cardsReady;
  }
  // if cards.js cannot load: every arg, with everything outside printable ASCII escaped
  function fallbackCard(p) {
    const pre = document.createElement("pre");
    pre.className = "ag-body";
    const raw = JSON.stringify({tool: p.tool, tier: p.tier, args: p.args || {}, reason: p.reason || ""}, null, 2);
    pre.textContent = raw.replace(/[^\x20-\x7e\n]/g, function (c) { return "\\u{" + c.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0") + "}"; });
    return pre;
  }

  const G = window.PXAAgent;
  if (!G) return;

  function post(path, body) {
    return fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body || {})
    }).then(r => r.json());
  }

  function paintPending(turn) {
    const col = G.el && G.el.col;
    if (!col || !turn || !turn.pending) return;
    const p = turn.pending;
    const card = document.createElement("div");
    card.className = "ag-ap";
    card.id = "ag-vigil-ap";
    card.setAttribute("role", "group");
    card.setAttribute("aria-label", "approval needed");
    const h = document.createElement("div");
    h.className = "ag-ap-h";
    const q = document.createElement("b");
    q.className = "ag-q";
    q.textContent = p.tool === "shell" ? "Run this command?" : ((p.tier || "") + " " + (p.tool || "tool"));
    h.appendChild(q);
    const detail = document.createElement("div");
    detail.className = "ag-d";
    loadCards().then(function (C) { detail.appendChild(C.render(document, p, {prefix: "ag"})); },
                     function () { detail.appendChild(fallbackCard(p)); });
    const btns = document.createElement("div");
    btns.className = "ag-ap-btns";
    const choices = (p.tool === "shell" || p.tier === "T3") ? ["once", "deny"] : ["once", "always", "deny"];
    choices.forEach(function (decision) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "b" + (decision === "once" ? " pri" : "");
      b.textContent = decision === "once" ? "Allow once" : decision === "always" ? "Always for this chat" : "Deny";
      b.onclick = function () {
        btns.querySelectorAll("button").forEach(x => x.disabled = true);
        post("/api/chat/vigil/approve", {id: p.id, decision: decision}).then(function () {
          return post("/api/chat/vigil/resume", {chat_id: turn.chat_id});
        }).then(function (next) {
          card.remove();
          if (next && next.text) G.insertMarker(next.text, {kind: "ok"});
          paintPending(next);
        });
      };
      btns.appendChild(b);
    });
    card.appendChild(h);
    card.appendChild(detail);
    card.appendChild(btns);
    col.appendChild(card);
  }

  function mountKill() {
    const top = document.querySelector(".ag-top");
    if (!top || document.getElementById("ag-vigil-kill")) return;
    const b = document.createElement("button");
    b.type = "button";
    b.id = "ag-vigil-kill";
    b.className = "b ag-vigil-kill";
    b.textContent = "Kill switch";
    b.title = "Stop Vigil. New steps are refused until Control is opened again.";
    b.onclick = function () {
      post("/api/chat/vigil/kill", {}).then(function () {
        if (G.stop) G.stop();
        G.insertMarker("Vigil stopped. New steps are refused.", {kind: "bad", id: "vigil-kill"});
      });
    };
    top.appendChild(b);
    const open = document.createElement("a");
    open.id = "ag-vigil-open";
    open.className = "b ag-vigil-open";
    open.href = "/api/chat/vigil/page";  // served by Control itself (fallback: :7878/vigil/)
    open.target = "_blank";
    open.rel = "noopener";
    open.textContent = "Vigil Control";
    open.title = "Approvals, audit, memory, jobs and settings";
    top.appendChild(open);
  }

  // Alt+Shift+V anywhere in Control opens or focuses the Vigil HUD window (same check as
  // control.js summonKey; a browser page cannot take an OS-wide hotkey, see vigil/ui/summon.py)
  function summonKey(e) { return e.altKey && e.shiftKey && !e.ctrlKey && !e.metaKey && (e.code === "KeyV" || e.key === "V" || e.key === "v"); }
  document.addEventListener("keydown", function (e) {
    if (!summonKey(e)) return;
    e.preventDefault();
    const w = window.open("/api/chat/vigil/hud", "vigil-hud");
    if (w && w.focus) w.focus();
  });

  G.on("init", function () {
    mountKill();
    G.addSlashCommand("vigil", function (rest) {
      const chat = G.sessionId || "chat";
      return post("/api/chat/vigil/turn", {chat_id: chat, message: rest || ""}).then(function (turn) {
        if (turn.status === "stopped") {
          G.insertMarker("Vigil is stopped.", {kind: "bad"});
          return;
        }
        if (turn.status === "denied") {
          const note = document.createElement("div");
          note.className = "ag-note bad";
          note.textContent = turn.text || "Blocked.";
          if (G.el && G.el.col) G.el.col.appendChild(note);
        } else if (turn.text) {
          G.insertMarker(turn.text, {kind: "ok"});
        }
        paintPending(turn);
        showAudit(chat);
      });
    });
  });

  function showAudit(chat) {
    const col = G.el && G.el.col;
    if (!col) return;
    let panel = document.getElementById("ag-vigil-audit");
    if (!panel) {
      panel = document.createElement("details");
      panel.id = "ag-vigil-audit";
      panel.className = "ag-notes ag-vigil-audit";
      panel.open = true;
      const sum = document.createElement("summary");
      sum.textContent = "Audit";
      const pre = document.createElement("pre");
      pre.id = "ag-vigil-audit-body";
      panel.appendChild(sum);
      panel.appendChild(pre);
      col.appendChild(panel);
    }
    fetch("/api/chat/vigil/audit?chat_id=" + encodeURIComponent(chat || "")).then(r => r.json()).then(function (data) {
      const pre = document.getElementById("ag-vigil-audit-body");
      if (!pre) return;
      const seen = {};
      const lines = [];
      (data.rows || []).forEach(function (row) {
        if (!row.human || seen[row.human]) return;
        seen[row.human] = 1;
        lines.push(row.human);
      });
      lines.reverse();
      panel.dataset.all = lines.join("\n");
      const shown = lines.slice(0, 5);
      pre.textContent = shown.join("\n") || "Nothing yet.";
      let more = document.getElementById("ag-vigil-audit-more");
      if (lines.length > 5) {
        if (!more) {
          more = document.createElement("button");
          more.type = "button";
          more.id = "ag-vigil-audit-more";
          more.className = "link";
          more.textContent = "show all";
          more.onclick = function () {
            pre.textContent = panel.dataset.all || pre.textContent;
            more.remove();
          };
          panel.appendChild(more);
        }
      } else if (more) more.remove();
    });
  }
})();
