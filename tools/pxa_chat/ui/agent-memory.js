// agent-memory.js: auto-compact marker, /compact, the on/off threshold, and memory retention.
(function(){
  const G = window.PXAAgent;
  if (!G) return;
  const S = (typeof store !== "undefined" && store) ? store : {get: () => null, set: () => {}};

  function pct() {
    const n = parseFloat(S.get("agent.compactAt", "75"));
    return Math.min(95, Math.max(2, isNaN(n) ? 75 : n));
  }
  function on() { return S.get("agent.compact", "on") !== "off"; }

  function label(d) {
    const n = d.turns || 0;
    const core = d.method === "dropped"
      ? "Conversation compacted (" + n + " turns dropped)"
      : "Conversation compacted (" + n + " turns → summary)";
    return d.estimated ? core + " · token count estimated" : core;
  }
  function marker(d, before) {
    const n = G.insertMarker(label(d), {kind: "compact", id: "compact", before: before || null});
    if (!n) return;
    const pre = el("pre", {class: "ag-compact-sum", text: d.summary || d.note || ""});
    const btn = el("button", {type: "button", class: "ag-compact-btn", text: "Show summary",
      onclick: () => {
        const open = n.classList.toggle("open");
        btn.textContent = open ? "Hide summary" : "Show summary";
      }});
    n.append(btn, pre);
  }

  G.on("beforeRun", body => {
    body.compact = on();
    body.compact_threshold = pct() / 100;
    if (G._forceCompact) { body.force_compact = true; G._forceCompact = false; }
  });
  G.addSlashCommand("compact", () => {
    G._forceCompact = true;
    return "Continue from the compacted history.";
  });
  G.on("delta", (ev, el) => {
    if (!ev || ev.type !== "compact") return;
    const before = el && el.previousElementSibling;
    marker(ev.data || {}, before || el);
  });
  G.on("sessionChange", async id => {
    if (!id || !G.el || !G.el.col) return;
    let s;
    try { s = await api("/api/chat/session?id=" + encodeURIComponent(id)); }
    catch (e) { return; }
    const c = s && s.compact;
    if (!c || !c.summary || document.querySelector("#ag-msgs .ag-marker.compact")) return;
    marker({turns: c.turns, method: c.method, summary: c.summary, estimated: c.estimated, note: c.note});
  });
  G.addSettingsSection("compact", "Context window", box => {
    const cb = el("input", {type: "checkbox", id: "ag-compact-on", checked: on(),
      onchange: () => { S.set("agent.compact", cb.checked ? "on" : "off"); hint.textContent = say(); }});
    const num = el("input", {type: "number", id: "ag-compact-at", min: "2", max: "95", step: "1", value: String(pct()),
      "aria-label": "compact when the context is this percent full",
      onchange: () => { S.set("agent.compactAt", num.value); hint.textContent = say(); }});
    const hint = el("p", {class: "tiny", id: "ag-compact-hint"});
    function say() {
      return on()
        ? "Older turns are summarized when the chat fills about " + pct() + "% of the context window. The full transcript stays saved. /compact does it now."
        : "Auto-compact is off. A context overflow still compacts once and retries. /compact still works.";
    }
    hint.textContent = say();
    box.append(
      el("label", {class: "check"}, cb, el("span", {}, el("b", {text: "Auto-compact long chats"}))),
      el("label", {class: "f"}, "Compact at this % of the context window", num),
      hint);
  });
})();
