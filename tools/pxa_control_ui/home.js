// home.js: PXA Control's Home section (2026-10 revamp), the page a fresh visit opens. It adapts to where the user is:
//   first run  - no model yet: get one (download a recommended PXA quant, or point at a folder of .gguf files);
//   ready      - models, nothing serving: one model picker, cards picked for it, one Start;
//   loading    - the server this page started is loading: what is happening and how long it takes;
//   serving    - every running server, with Chat as the main thing to do next.
// It owns no logic of its own: downloads, folders, the fit check and Start are the Models and Launch code
// (startDownload, saveDirs, loadFits, doStart), so Home and Launch can never disagree about what a start does.
(function () {
  "use strict";
  const H = {folder: "", model: null, err: null, sig: "", busy: false, asked: {}};
  const short = n => String(n || "").replace(/^(NVIDIA |Tesla |GeForce )/, "").replace(/-PCIE.*$/, "");
  const gib = mib => (mib / 1024).toFixed(0) + " GiB";
  const models = () => (S.models && S.models.models) || [];
  const running = () => ((S.fleet && S.fleet.instances) || []).filter(x => x.running || x.attached);
  const busyDl = () => !!(HF.last && ["starting", "running", "checking"].includes(HF.last.state));

  function rigLine() {
    const cs = S.gpus || [];
    if (!cs.length) return el("div", {class: "rigline"}, el("span", {text: "Reading your cards ..."}));
    const groups = {};
    for (const c of cs) { const k = short(c.name) + " " + gib(c.mem_total_mib); groups[k] = (groups[k] || 0) + 1; }
    const total = cs.reduce((a, c) => a + c.mem_total_mib, 0);
    return el("div", {class: "rigline"}, el("span", {}, el("b", {text: cs.length + " card" + (cs.length === 1 ? "" : "s")}), " · " + gib(total) + " of GPU memory"),
      ...Object.entries(groups).map(([k, n]) => el("span", {text: n + "× " + k})));
  }
  // the card set Home starts on: what the user picked before, else the smallest set the fit check says holds the model
  function pickCards(path) {
    const saved = store.get("cards", []);
    if (saved.length) return {ids: saved, why: "your last choice"};
    const sets = (S.fitSets || []).filter(s => s.label !== "your pick"), fits = (S.fits && S.fits[path]) || [];
    const all = S.fitSets || [];
    for (const s of sets) {
      const f = fits[all.indexOf(s)];
      if (f && (f.verdict === "fits" || f.verdict === "ram")) return {ids: s.ids, why: s.label, verdict: f};
    }
    const idle = (S.gpus || []).filter(c => c.mem_used_mib < 512 && !(c.procs || []).length).map(c => c.index);
    return {ids: idle.length ? idle : (S.gpus || []).map(c => c.index), why: idle.length ? "every idle card" : "every card"};
  }
  function fitFor(path, ids) {
    const i = (S.fitSets || []).findIndex(s => s.ids.join(",") === ids.join(","));
    return i >= 0 && S.fits && S.fits[path] ? S.fits[path][i] : null;
  }
  // recommended downloads: models only (no vision projector, no second part of a split file), that can fit the cards at
  // all; free before supporter, PXA quants before other files, then smallest first: the quickest first success
  function recommended() {
    const d = HF.data; if (!d) return [];
    const vram = (S.gpus || []).reduce((a, c) => a + c.mem_total_mib, 0) * 1048576;
    const out = [];
    for (const r of d.repos || []) for (const f of r.files || []) {
      if (f.kind === "counts" || f.have || /mmproj|projector/i.test(f.name) || /-0000[2-9]-of-|-000[1-9]\d-of-/.test(f.name)) continue;
      if (vram && +f.size > vram * 0.9) continue;
      out.push({repo: r.repo, f, locked: !!r.supporter, pxa: !!quantTier(f.name)});
    }
    return out.sort((a, b) => (a.locked - b.locked) || (b.pxa - a.pxa) || ((+a.f.size || 0) - (+b.f.size || 0))).slice(0, 3);
  }

  // ---------------- steps ----------------
  function step(n, cls, title, sub, body) {
    return el("div", {class: "hstep " + cls}, el("span", {class: "num", "aria-hidden": "true"}, cls === "done" ? icon("check") : String(n)),
      el("div", {}, el("h4", {}, title), sub ? el("p", {class: "sub", text: sub}) : null, body ? el("div", {class: "body"}, body) : null));
  }
  function folderForm(primary) {
    const inp = el("input", {type: "text", id: "h-folder", placeholder: "/path/to/models", "aria-label": "model folder", spellcheck: "false", autocomplete: "off",
      value: H.folder, style: "flex:1;min-width:180px", oninput: ev => { H.folder = ev.target.value; },
      onkeydown: ev => { if (ev.key === "Enter") addFolder(); }});
    return el("div", {class: "row"}, inp, el("button", {class: "b" + (primary ? " pri" : ""), type: "button", id: "h-folder-add", text: "Use this folder", onclick: addFolder}));
  }
  async function addFolder() {
    const v = (H.folder || "").trim(); if (!v) { const f = $("#h-folder"); if (f) f.focus(); toast("type the folder that holds (or will hold) your models", true); return; }
    await saveDirs([...((S.models && S.models.roots) || []), v]); H.folder = ""; H.sig = ""; homeRender();
    if (S.models && S.models.roots.length && !HF.data) loadAvailable().then(() => { H.sig = ""; homeRender(); });
  }
  function getModelBody() {
    const roots = (S.models && S.models.roots) || [];
    const dl = HF.last, live = busyDl();
    const left = el("div", {class: "hopt"}, el("h5", {}, icon("dl"), "Download one from PXA"),
      el("p", {text: "Ready-made PXA quants from huggingface.co/poisonxa. A download resumes if it stops and is checked against the published sha256."}));
    if (live || (dl && HF.sawRun)) {
      const total = dl.total || 0, done = dl.done + (dl.file_done || 0), pct = total ? Math.min(100, 100 * done / total) : 0;
      left.append(el("div", {class: "hfprog"}, el("div", {class: "row", style: "justify-content:space-between"},
        el("b", {class: "tiny", text: dl.state === "failed" ? "Download failed: " + (dl.error || "") : dl.state === "done" ? "Downloaded " + (dl.written || []).join(", ") : "Downloading " + (dl.what || "")}),
        live ? el("button", {class: "b sm", type: "button", text: "Cancel", onclick: () => $("#hf-p-cancel").click()}) : null),
        el("div", {class: "bar"}, el("i", {style: "width:" + pct.toFixed(1) + "%"})),
        el("div", {class: "tiny", text: live ? hfBytes(done) + " of " + hfBytes(total) + (dl.rate ? " at " + hfBytes(dl.rate) + "/s" : "") + " · " + pct.toFixed(0) + "%" : ""})));
    } else if (!HF.data) {
      left.append(HF.err ? stateBox("cloud", "Can't reach huggingface.co", HF.err + ". Check the internet connection, or use a folder you already have.",
        [el("button", {class: "b sm", type: "button", text: "Try again", onclick: () => { HF.err = null; homeLoadHF(true); }})], "bad")
        : el("div", {class: "empty"}, el("span", {class: "spin"}), "Reading the PXA model list ..."));
    } else {
      const recs = recommended();
      if (!recs.length) left.append(el("p", {class: "muted", text: "Nothing new to download: every published model is already in your folders."}));
      for (const x of recs) {
        const btn = x.locked && !pxqnUnlocked()
          ? el("button", {class: "b sup sm", type: "button", onclick: () => { showTab("store"); if (window.storeFocus) storeFocus("pxqn-tier"); }}, icon("lock"), "Unlock")
          : el("button", {class: "b pri sm", type: "button", onclick: async ev => {
              if (!roots.length) { const f = $("#h-folder"); if (f) f.focus(); toast("first choose where models go: type a folder and press Use this folder", true); return; }
              await startDownload(x.repo, x.f.path, ev.currentTarget); H.sig = ""; homeRender(); }}, icon("dl"), "Download");
        left.append(el("div", {class: "rec"}, el("div", {}, el("div", {class: "nm", text: x.f.name}),
          el("div", {class: "meta"}, tierBadge(x.locked && !quantTier(x.f.name) ? "pxqn" : x.f.name, x.locked && !pxqnUnlocked()), el("span", {text: x.f.size_h || hfBytes(x.f.size)}), el("span", {class: "mono", text: x.repo}))), btn));
      }
      left.append(el("button", {class: "link", type: "button", style: "margin-top:6px", text: "See every published model", onclick: () => showTab("models")}));
    }
    const right = el("div", {class: "hopt"}, el("h5", {}, icon("folder"), roots.length ? "Your model folder" : "Where models live"),
      el("p", {text: roots.length ? "Downloads go to " + roots[0] + ". Add another folder that already holds .gguf files:" :
        "Name the folder that holds your .gguf files, or where downloads should go. It is remembered on this machine."}),
      folderForm(!roots.length),
      roots.length ? el("div", {class: "tiny", style: "margin-top:8px"}, "Already copied files in? ", el("button", {class: "link", type: "button", text: "Look again", onclick: () => loadModels(true).then(() => { H.sig = ""; homeRender(); })})) : null);
    return el("div", {class: "hopts"}, left, right);
  }
  function runBody() {
    const ms = models();
    const cur = H.model && ms.some(m => m.path === H.model) ? H.model : ($("#l-model").value || store.get("model", "") || (ms[0] && ms[0].path));
    H.model = cur;
    const sel = el("select", {id: "h-model", "aria-label": "model to run", onchange: ev => { H.model = ev.target.value; H.err = null; H.sig = ""; homeRender(); }},
      ...ms.map(m => el("option", {value: m.path, text: m.file + "  ·  " + (m.type || m.codec || "") + "  ·  " + (m.size_h || "")})));
    sel.value = cur;
    const pick = pickCards(cur), f = fitFor(cur, pick.ids);
    const byIdx = Object.fromEntries((S.gpus || []).map(c => [c.index, c]));
    const fit = f ? badge(f.verdict === "fits" || f.verdict === "ram" ? "ok" : f.verdict === "tight" ? "warn" : "bad",
      f.verdict === "ram" ? "fits with system RAM" : f.verdict === "fits" ? "fits on these cards" : f.verdict === "tight" ? "tight fit" : "does not fit") : null;
    if (fit && f.why) fit.title = f.why;
    const m = ms.find(x => x.path === cur);
    return el("div", {},
      el("div", {class: "runbox"}, el("label", {class: "f"}, "Model", sel),
        el("button", {class: "b pri lg", id: "h-start", type: "button", disabled: H.busy || !cur || !pick.ids.length, onclick: () => homeStart(cur, pick.ids)}, icon("play"), H.busy ? "Starting ..." : "Start")),
      m ? el("div", {class: "row", style: "margin-top:8px;gap:6px"}, tierBadge(m.codec || m.file), m.family ? el("span", {class: "tiny", text: m.family + (m.size_label ? " · " + m.size_label : "")}) : null, m.notice ? stdBadge(m.notice) : null) : null,
      el("div", {class: "cardsum", style: "margin-top:10px"}, el("span", {text: "On"}),
        ...pick.ids.map(i => el("span", {class: "chip", text: "#" + i + " " + (byIdx[i] ? short(byIdx[i].name) : "")})),
        el("span", {class: "tiny", text: "(" + pick.why + ")"}), fit,
        el("button", {class: "link", type: "button", text: "Change cards or settings", onclick: () => { homeToLaunch(cur, pick.ids); showTab("launch"); }})),
      H.err ? el("div", {class: "alert bad", role: "alert"}, badge("bad", "not started"), el("span", {text: H.err}),
        el("span", {class: "acts"}, el("button", {class: "b sm", type: "button", text: "See why", onclick: () => showTab("launch")}))) : null);
  }
  // Home drives the Launch form and its Start: one start path, so the plan, the log and the Servers card all follow
  function homeToLaunch(path, ids) {
    if (!$$("#l-cards input").length) renderLaunchCards();
    $$("#l-cards input").forEach(i => { i.checked = ids.includes(+i.value); });
    store.set("cards", ids);
    $("#l-model").value = path; store.set("model", path); modelInfo(); steps();
  }
  async function homeStart(path, ids) {
    H.busy = true; H.err = null; H.sig = ""; homeRender();
    try {
      homeToLaunch(path, ids);
      if (!selectedCards().length) throw new Error("no cards to start on: open Launch to pick them");
      const d = await doStart();
      if (!d || !d.ok) {
        // the refusal itself: a REFUSING / [R-n] rule line, else the launcher's LAST "!!" line (it ends on the fatal one,
        // e.g. "no WORKING llama-server found"; earlier "!!" lines are warnings the server would start with)
        const lines = (d && d.text || "").split("\n").map(l => l.trim());
        const why = lines.find(l => /REFUSING to|\[R-\d/.test(l)) || (d && (d.blockers || [])[0]) || lines.filter(l => /^!!/.test(l)).pop();
        H.err = ((d && d.error) || (why ? why.replace(/^!!\s*/, "") : "The launcher would not start this model on these cards.")).replace(/([^.!?])$/, "$1.") + " The Launch section shows the full plan.";
      }
    } catch (e) { H.err = e.message; }
    H.busy = false; H.sig = ""; await loadFleet(true); homeRender();
  }

  // ---------------- views ----------------
  function servingView(run) {
    const color = S.fleet ? fleetColors(S.fleet) : {};
    const ok = run.filter(x => x.health === "ok");
    return el("section", {class: "card hero", "aria-label": "running servers"},
      el("h2", {text: ok.length ? (ok.length === 1 ? "Your model is serving" : ok.length + " servers are serving") : "Your server is getting ready"}),
      el("p", {class: "lead", text: ok.length ? "Chat with it here, or point any OpenAI-compatible app at its port." : "Loading a model takes a minute or two. Its log shows each step."}),
      rigLine(),
      el("div", {class: "serving", style: "margin-top:var(--s5)"}, ...run.map(x => {
        const up = x.running && x.health === "ok", m = x.metrics || {};
        const acts = [];
        if (up && x.port) acts.push(el("button", {class: "b pri", type: "button", onclick: () => { setTarget(targetOf(x)); showTab("chat"); }}, icon("chat"), "Chat"),
          el("button", {class: "b", type: "button", text: "Benchmarks", onclick: () => { setTarget(targetOf(x)); showTab("speed"); }}));
        if ((x.actions || []).includes("log")) acts.push(el("button", {class: "b", type: "button", text: "Log", onclick: () => openLogDialog(x)}));
        if ((x.actions || []).includes("stop")) acts.push(el("button", {class: "b danger", type: "button", text: "Stop", onclick: () => fleetAct(x, "stop")}));
        return el("div", {class: "srvrow", style: "border-left-color:" + (color[x.key] || "var(--accent)")},
          el("span", {class: "dot " + healthLevel(x), "aria-hidden": "true"}),
          el("div", {}, el("div", {class: "nm", text: chipLabel(x)}),
            el("div", {class: "tiny", text: [x.label || x.name, srvPhase(x), x.port ? "port " + x.port : null, x.gpus && x.gpus.length ? "cards " + x.gpus.join(",") : null,
              up && m.decode_median ? fmtNum(m.decode_median) + " t/s decode" : null].filter(Boolean).join(" · ")})),
          el("div", {class: "acts"}, ...acts));
      })),
      el("div", {class: "row", style: "margin-top:var(--s4)"}, el("button", {class: "b", type: "button", text: "Run another model", onclick: () => { H.more = true; H.sig = ""; homeRender(); }}),
        el("button", {class: "b ghost", type: "button", text: "All servers", onclick: () => showTab("servers")})));
  }
  function stepsView(hasModel) {
    const first = !hasModel;
    return el("section", {class: "card hero", "aria-label": first ? "get started" : "run a model"},
      el("h2", {text: first ? "Run your first model" : "Run a model"}),
      el("p", {class: "lead", text: first ? "Three steps from here to a chat with a model on your own cards. Everything else is optional and lives in the other sections." :
        "Pick a model and press Start. The launcher chooses the engine, split, context and flags for these cards, and tells you why."}),
      rigLine(),
      el("div", {class: "hsteps"},
        step(1, hasModel ? "done" : "now", "Get a model", hasModel ? models().length + " model" + (models().length === 1 ? "" : "s") + " in " + ((S.models && S.models.roots) || []).join(", ") : null,
          hasModel ? null : getModelBody()),
        step(2, hasModel ? "now" : "later", "Pick cards and start", hasModel ? null : "Choose a model and press Start; the cards are picked for it.", hasModel ? runBody() : null),
        step(3, "later", "Chat with it", "Opens as soon as the model is serving.", null)),
      hasModel ? el("div", {class: "row", style: "margin-top:var(--s2)"}, el("button", {class: "link", type: "button", text: "Get another model", onclick: () => showTab("models")})) : null);
  }
  function nextTiles() {
    const t = (ico, title, text, tab, sup) => el("button", {class: "ntile", type: "button", onclick: () => showTab(tab)},
      el("span", {class: "ni" + (sup ? " sup" : "")}, icon(ico)), el("b", {text: title}), el("span", {text}));
    const anibal = window.PXAStore && PXAStore.item("anibal");
    return el("div", {},
      el("h3", {class: "sect", text: "More you can do"}),
      el("div", {class: "next"},
        t("encode", "Make your own quant", "Turn a Hugging Face model into a PXQ file sized for your cards.", "encode"),
        t("live", "Watch it work", "Live charts of every card and every server.", "live"),
        t("profiles", "Quieter or faster cards", "GPU profiles: power, clocks, auto-start.", "profiles"),
        anibal ? t("mic", "Talk to it: " + anibal.name, "The first PXA plugin: a voice assistant on your rig.", "plugins", true)
               : t("plug", "Plugins", "Made by the PXA team from Discord requests.", "plugins", true)));
  }
  function sigOf() {
    const run = running().map(x => [x.key, x.health, x.phase, x.port].join(":")).join("|");
    const dl = HF.last ? [HF.last.state, Math.round(((HF.last.done || 0) + (HF.last.file_done || 0)) / 1048576)].join(":") : "";
    return [S.models ? models().length + ":" + S.models.roots.join(",") : "-", run, dl, HF.data ? 1 : 0, HF.err || "", S.fits ? Object.keys(S.fits).length : 0,
      (S.gpus || []).length, H.model, H.err, H.busy, H.more, store.get("cards", []).join(","), window.PXAStore && PXAStore.catalog ? PXAStore.entitled("pxqn") : "-",
      document.documentElement.dataset.theme].join("~");
  }
  function homeRender(force) {
    const root = $("#home-root"); if (!root || S.tab !== "home") return;
    // no state is drawn before both the library and the servers are known: a running rig never flashes "first run"
    if (!S.models || (!S.fleet && !H.fleetTried)) { if (!root.querySelector(".skel")) root.replaceChildren(el("div", {class: "skel", style: "min-height:320px"})); return; }
    const sig = sigOf(); if (!force && sig === H.sig) return; H.sig = sig;
    const run = running(), hasModel = models().length > 0;
    if (!hasModel && (!HF.data || HF.cached) && !H.asked.hf) homeLoadHF();
    uiKeep(root, () => root.replaceChildren(...[
      run.length ? servingView(run) : null,
      !run.length || H.more ? stepsView(hasModel) : null,
      nextTiles()].filter(Boolean)));
  }
  async function homeLoadHF(force) {
    H.asked.hf = true;
    try { await loadAvailable(force); } catch (e) { HF.err = e.message; }
    H.sig = ""; homeRender();
  }
  window.homeRender = homeRender;
  window.homeOpen = async () => {
    H.sig = ""; homeRender();
    if (!S.fleet) { await loadFleet(); H.fleetTried = true; }
    if (!S.gpus || !S.gpus.length) { try { S.gpus = (await api("/api/rig/live")).cards; } catch (e) { /* the rig line says so */ } }
    await loadModels(); if (!S.fits && models().length) await loadFits();
    pollDownload(); H.sig = ""; homeRender();
  };
  if (S.tab === "home") window.homeOpen();      // the boot may have opened Home before this file arrived
})();
