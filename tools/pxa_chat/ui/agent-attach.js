// agent-attach.js: paste, drop, or pick files and images. Text lands in the composer. Images go with the next run.
(function () {
  "use strict";
  const G = window.PXAAgent;
  if (!G) return;
  const IMAGES = 4, MAX_BYTES = 1000000, MAX_TEXT = 100000;
  const images = [];
  let row = null, picker = null;

  function paint() {
    if (!row) return;
    row.replaceChildren();
    images.forEach(function (im, i) {
      const img = el("img", {alt: "", src: "data:" + im.media_type + ";base64," + im.data});
      const x = el("button", {type: "button", class: "ag-file-x", text: "Remove", "aria-label": "Remove " + im.name,
        onclick: function () { images.splice(i, 1); paint(); }});
      row.append(el("span", {class: "ag-file"}, img, el("span", {text: im.name}), x));
    });
  }
  function addImage(file) {
    const mt = (file.type || "").toLowerCase() === "image/jpg" ? "image/jpeg" : (file.type || "").toLowerCase();
    if (!/^image\/(png|jpeg|gif|webp)$/.test(mt)) { toast("Use a PNG, JPEG, GIF or WebP image."); return; }
    if (images.length >= IMAGES) { toast("Four images is the limit for one message."); return; }
    if (file.size > MAX_BYTES) { toast("That image is over 1 MB. Try a smaller one."); return; }
    const rd = new FileReader();
    rd.onload = function () {
      const b64 = String(rd.result || "").split(",")[1] || "";
      if (!b64 || images.length >= IMAGES) return;
      images.push({name: file.name || "image", media_type: mt, data: b64});
      paint();
    };
    rd.onerror = function () { toast("That image could not be read."); };
    rd.readAsDataURL(file);
  }
  function isText(file) {
    const t = file.type || "";
    if (/^text\//.test(t) || t === "application/json" || t === "application/javascript") return true;
    return /\.(txt|md|json|csv|py|js|mjs|css|html|log|yml|yaml|sh|toml)$/i.test(file.name || "");
  }
  function addText(file) {
    if (file.size > MAX_TEXT * 2) { toast("That file is too big to paste into the box."); return; }
    const rd = new FileReader();
    rd.onload = function () {
      const input = G.el && G.el.input;
      if (!input) return;
      const chunk = String(rd.result || "").slice(0, MAX_TEXT);
      if (!chunk) return;
      const cur = input.value;
      input.value = (cur && !/\n$/.test(cur) ? cur + "\n\n" : cur) + chunk;
      input.dispatchEvent(new Event("input", {bubbles: true}));
    };
    rd.onerror = function () { toast("That file could not be read."); };
    rd.readAsText(file);
  }
  function take(list) {
    Array.from(list || []).forEach(function (f) {
      if (!f) return;
      if ((f.type || "").indexOf("image/") === 0) addImage(f);
      else if (isText(f)) addText(f);
      else toast("Drop a text file or a PNG, JPEG, GIF or WebP image.");
    });
  }

  G.on("init", function () {
    const input = G.el && G.el.input;
    if (!input || picker) return;
    const box = input.closest(".ag-box") || input.parentNode;
    row = el("div", {class: "ag-files", id: "ag-files"});
    box.insertBefore(row, input);
    picker = el("input", {type: "file", id: "ag-attach", hidden: true, multiple: true,
      accept: "image/png,image/jpeg,image/gif,image/webp,text/plain,.txt,.md,.json,.py,.js,.css,.html",
      onchange: function () { take(picker.files); picker.value = ""; }});
    box.append(picker);
    input.addEventListener("paste", function (e) {
      const files = e.clipboardData && e.clipboardData.files;
      if (files && files.length) { e.preventDefault(); take(files); }
    });
    box.addEventListener("dragover", function (e) { if (e.dataTransfer) { e.preventDefault(); box.classList.add("ag-drop"); } });
    box.addEventListener("dragleave", function () { box.classList.remove("ag-drop"); });
    box.addEventListener("drop", function (e) {
      e.preventDefault();
      box.classList.remove("ag-drop");
      take(e.dataTransfer && e.dataTransfer.files);
    });
  });
  G.addComposerButton({id: "attach", label: "Attach", title: "Attach a file or an image", icon: "clip",
    onClick: function () { if (picker) picker.click(); }});
  G.on("beforeRun", function (body) {
    if (!images.length) return;
    body.images = images.map(function (im) { return {media_type: im.media_type, data: im.data}; });
  });
  G.on("messageDone", function (msg) {
    if (msg && msg.end === "run.done") { images.length = 0; paint(); }
  });
  G.on("sessionChange", function () {
    // a send that creates the chat fires this while the run is still going: keep the chips for Retry
    if (G.state && G.state.run) return;
    images.length = 0;
    if (row) paint();
  });
})();
