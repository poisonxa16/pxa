// agent-actions.js: Continue from the last reply. The < 1/3 > version list is drawn by the core.
(function () {
  "use strict";
  const G = window.PXAAgent;
  if (!G) return;
  G.addMessageAction({
    id: "continue",
    label: "Continue",
    title: "Ask the assistant to continue",
    when: function (msg) {
      return msg && msg.role === "assistant" && msg.end === "run.done" && msg.idx === G.state.turns.length - 1;
    },
    run: function () { return G.send("Continue"); }
  });
})();
