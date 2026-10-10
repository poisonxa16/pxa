// agent-ux.js: slash commands, one extra shortcut, voice in and out, and the app manifest's service worker.
// The phone layout already lives in agent.css at max-width 900px. This file does not move New chat.
(function () {
  "use strict";
  const G = window.PXAAgent;
  if (!G) return;

  G.addSlashCommand("new", function () { G.newChat(); });
  G.addSlashCommand("continue", function () { return "Continue"; });

  document.addEventListener("keydown", function (e) {
    if (e.defaultPrevented || e.key === "Enter" || e.key === "@") return;
    if (!(e.altKey && e.shiftKey) || e.ctrlKey || e.metaKey) return;
    if (String(e.key || "").toLowerCase() !== "n") return;
    e.preventDefault();
    G.newChat();
  });

  function voiceIn() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { toast("Voice input is not available in this browser."); return; }
    let rec;
    try { rec = new SR(); }
    catch (e) { toast("Voice input is not available in this browser."); return; }
    rec.lang = document.documentElement.lang || "en-US";
    rec.interimResults = false;
    rec.onerror = function () {};
    rec.onresult = function (ev) {
      try {
        const said = ev.results && ev.results[0] && ev.results[0][0] && ev.results[0][0].transcript;
        const input = G.el && G.el.input;
        if (!said || !input) return;
        input.value = (input.value ? input.value.replace(/\s*$/, " ") : "") + said;
        input.dispatchEvent(new Event("input", {bubbles: true}));
      } catch (e) { /* a result we cannot read is not a crash */ }
    };
    try { rec.start(); }
    catch (e) { toast("Voice input did not start."); }
  }
  function speak(text) {
    const say = String(text || "").replace(/\s+/g, " ").trim().slice(0, 2000);
    if (!say) { toast("There is nothing to read yet."); return; }
    const SS = window.speechSynthesis;
    if (!SS || typeof window.SpeechSynthesisUtterance !== "function") {
      toast("Voice output is not available in this browser.");
      return;
    }
    try {
      SS.cancel();
      const u = new window.SpeechSynthesisUtterance(say);
      u.onerror = function () {};
      SS.speak(u);
    } catch (e) { toast("Voice output did not start."); }
  }

  G.addComposerButton({id: "voice", label: "Voice", title: "Dictate into the composer", icon: "mic", onClick: voiceIn});
  G.addMessageAction({
    id: "read-aloud",
    label: "Read aloud",
    title: "Read this reply aloud",
    when: function (msg) { return msg && msg.role === "assistant" && msg.end === "run.done"; },
    run: function (msg) { speak(msg && msg.text); }
  });

  G.on("init", function () {
    if (!("serviceWorker" in navigator)) return;
    const p = navigator.serviceWorker.register("/agent-sw.js");
    if (p && p.catch) p.catch(function () {});
  });
})();
