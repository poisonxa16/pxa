// agent-params.js: repeat penalty, seed, stop sequences, JSON schema, grammar (Advanced), and system-prompt snippets. Temperature and the other sampling knobs stay in the core.
(function () {
  "use strict";
  const G = window.PXAAgent;
  if (!G) return;
  const S = (typeof store !== "undefined" && store) ? store : {get: function () { return null; }, set: function () {}};
  const SNIPS = [
    {id: "brief", label: "Brief", text: "Answer in a few sentences. Skip the preamble."},
    {id: "code", label: "Code", text: "Prefer complete code, with a short note on how to run it."},
    {id: "exact", label: "Exact", text: "Quote numbers and names exactly. Say when you are not sure."}
  ];
  function opt(v, lo, hi, integer) {
    if (v == null || v === "") return null;
    const n = integer ? Math.round(+v) : +v;
    if (!Number.isFinite(n) || n < lo || n > hi) return null;
    return n;
  }
  function sysBox() { return document.getElementById("ag-sys"); }
  function apply(id, text) {
    const sys = sysBox();
    if (!sys) { toast("The system prompt box is not on this page."); return; }
    if (!S.get("agent.snippet", "") && S.get("agent.snippetPrev", null) == null) S.set("agent.snippetPrev", sys.value);
    sys.value = text;
    S.set("agent.snippet", id);
    sys.dispatchEvent(new Event("change", {bubbles: true}));
  }
  function clearSnip() {
    const sys = sysBox();
    const prev = S.get("agent.snippetPrev", null);
    if (sys && typeof prev === "string") { sys.value = prev; sys.dispatchEvent(new Event("change", {bubbles: true})); }
    S.set("agent.snippet", "");
    S.set("agent.snippetPrev", null);
  }
  G.addSettingsSection("params", "Sampling and snippets", function (box) {
    const rp = el("input", {type: "number", id: "ag-repeat", min: "0.5", max: "2", step: "0.05",
      placeholder: "server default", "aria-label": "repeat penalty", value: S.get("agent.repeatPenalty", "") == null ? "" : String(S.get("agent.repeatPenalty", "")),
      onchange: function () { S.set("agent.repeatPenalty", rp.value); }});
    const seed = el("input", {type: "number", id: "ag-seed", min: "-1", max: "2147483647", step: "1",
      placeholder: "server default", "aria-label": "seed", value: S.get("agent.seed", "") == null ? "" : String(S.get("agent.seed", "")),
      onchange: function () { S.set("agent.seed", seed.value); }});
    const snips = el("div", {class: "ag-snips", id: "ag-snips"});
    SNIPS.forEach(function (s) {
      snips.append(el("button", {type: "button", class: "b", id: "ag-snip-" + s.id, text: s.label,
        onclick: function () { apply(s.id, s.text); }}));
    });
    snips.append(el("button", {type: "button", class: "b", id: "ag-snip-clear", text: "Mode default",
      onclick: clearSnip}));
    // P08: stop sequences, a JSON schema and a grammar (Advanced mode only; the server checks and size-caps them)
    function area(id, key, label, ph, rows) {
      const ta = el("textarea", {id: id, rows: String(rows), placeholder: ph, "aria-label": label, spellcheck: "false",
        value: String(S.get(key, "") || ""), onchange: function () { S.set(key, ta.value); }});
      ta.value = String(S.get(key, "") || "");
      return el("label", {class: "f"}, label, ta);
    }
    box.append(
      el("div", {class: "ag-grid2"},
        el("label", {class: "f"}, "Repeat penalty", rp),
        el("label", {class: "f"}, "Seed", seed)),
      el("div", {class: "ag-advonly ag-outcons"},
        area("ag-stop", "agent.stop", "Stop sequences (one per line, up to 4)", "END", 2),
        area("ag-jschema", "agent.jsonSchema", "JSON schema for the reply (turns tools off)", '{"type": "object"}', 3),
        area("ag-grammar", "agent.grammar", "Grammar (GBNF, turns tools off)", 'root ::= "yes" | "no"', 2)),
      el("div", {class: "tiny"}, "Snippets fill the system prompt and are sent with the next message."),
      snips);
  });
  G.on("beforeRun", function (body) {
    const rp = opt(S.get("agent.repeatPenalty", ""), 0.5, 2, false);
    if (rp != null) body.repeat_penalty = rp;
    const seed = opt(S.get("agent.seed", ""), -1, 2147483647, true);
    if (seed != null) body.seed = seed;
    const stop = String(S.get("agent.stop", "") || "").split("\n").map(function (x) { return x.replace(/\r$/, ""); }).filter(Boolean);
    if (stop.length) body.stop = stop.slice(0, 4);
    const js = String(S.get("agent.jsonSchema", "") || "").trim();
    if (js) {
      try { const o = JSON.parse(js); if (o && typeof o === "object" && !Array.isArray(o)) body.json_schema = o; else toast("The JSON schema must be an object; it was not sent."); }
      catch (e) { toast("The JSON schema is not valid JSON; it was not sent."); }
    }
    const gr = String(S.get("agent.grammar", "") || "").trim();
    if (gr && !body.json_schema) body.grammar = gr;
    if (S.get("agent.snippet", "")) {
      const sys = sysBox();
      if (sys) { body.system = sys.value; body.system_snippet = true; }
    }
  });
})();
