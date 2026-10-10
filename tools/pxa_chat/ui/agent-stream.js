// agent-stream.js: the server that answered, and a button that sends the next message to another running server.
// Retry on the error note stays in the core.
(function () {
  "use strict";
  const G = window.PXAAgent;
  if (!G) return;
  let fallback = "";

  G.on("delta", function (ev, msgEl) {
    if (!ev || ev.type !== "run.started" || !msgEl) return;
    const d = ev.data || {};
    msgEl._agServer = d.server || d.server_key || "";
  });
  G.on("messageDone", function (msg) {
    if (!msg || !msg.el) return;
    if (msg.end === "run.done") {
      const name = msg.el._agServer || "";
      const meta = msg.el.querySelector(".ag-meta");
      if (name && meta && meta.textContent.indexOf(name) < 0)
        meta.textContent = meta.textContent ? meta.textContent + " · " + name : name;
    }
    if (msg.end !== "run.error") return;
    const note = msg.el.querySelector(".ag-note");
    if (!note || note.querySelector(".ag-try-other")) return;
    note.append(el("button", {type: "button", class: "b ag-try-other", text: "Try another server",
      onclick: function () {
        const st = G.state || {};
        const others = (st.servers || []).filter(function (s) {
          return s.key !== st.selected && (s.status === "ready" || s.status === "busy");
        });
        if (!others.length) { toast("No other server is running."); return; }
        fallback = others[0].key;
        toast("The next message goes to " + (others[0].name || others[0].key) + ".");
      }}));
  });
  G.on("beforeRun", function (body) {
    if (!fallback) return;
    body.server = fallback;
    fallback = "";
  });
})();
