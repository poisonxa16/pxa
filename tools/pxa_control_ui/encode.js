"use strict";
/* PXA Control: the Encode tab. A wizard that turns a Hugging Face model into a PXQ / PXQN file.
   Loaded by index.html after its own script, so it uses that script's helpers ($, $$, el, api, toast, store, badge, fmtDur,
   copyText, renderLogInto, showTab, setTarget, checkHighScore, S). It talks only to /api/encode/* on PXA Control.
   Everything the server sends is put on the page as text (never as HTML). The licence key is typed into one password box,
   sent once to this server, and never shown again except masked. */

const EN = {
  st: null, built: false, step: 1, src: null, srcText: store.get("enc.src", ""), srcErr: "", srcBusy: false,
  rig: null, cards: store.get("enc.cards", null), other: store.get("enc.other", false), vram: store.get("enc.vram", ""),
  plan: null, planErr: "", planBusy: false, tier: null,
  adv: {work_dir: "", out_dir: "", keep: false, use_hessians: null, encode_card: "", licence_ack: false}, lock: store.get("enc.lock", null),
  checks: null, checksErr: "", checksBusy: false,
  job: null, logLines: [], logSeq: 0, jobTimer: null, pkgTimer: null, pkgDismissed: false, rtTimer: null, rtDismissed: false, rtAsked: false, showGet: false, showPicker: false, test: null, updAsked: false,
};
const enApi = (p, o) => api("/api/encode/" + p, o);
const enBytes = n => { if (n == null) return "?"; const u = ["B", "KiB", "MiB", "GiB", "TiB"]; let i = 0, v = +n; while (v >= 1024 && i < 4) { v /= 1024; i++; } return (i ? v.toFixed(v >= 100 ? 0 : 1) : String(v)) + " " + u[i]; };
const enCtx = c => c == null ? "?" : c >= 1024 ? Math.floor(c / 1024) + "k" : String(c);
const ICON_LOCK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>';
function enIcon(svg) { const s = el("span", {class: "en-ico"}); s.innerHTML = svg; return s; }   // static markup from this file only
const EN_ROLE = {best: "Best quality that fits", roomy: "More room for long context", small: "Smallest that is still good"};
const EN_LIVE = ["running", "paused", "queued"];
const EN_STATUS = {queued: "waiting", running: "running", paused: "paused", interrupted: "interrupted", failed: "stopped with an error", cancelled: "cancelled", done: "finished"};

function enBuild() {
  EN.built = true;
  $("#en-root").replaceChildren(el("div", {id: "en-enc"}), el("div", {id: "en-get", hidden: true}), el("div", {id: "en-wiz"}), el("div", {id: "en-jobs"}));
}

async function encodeOpen() {
  if (!EN.built) enBuild();
  try { await enLoadState(); } catch (e) { $("#en-enc").replaceChildren(el("div", {class: "alert bad"}, "The Encode tab could not start: " + e.message)); return; }
  renderWiz();
  const j = (EN.st.jobs || [])[0];
  if (j && !EN.job && (EN_LIVE.includes(j.status) || ["interrupted", "failed", "cancelled"].includes(j.status))) await enOpenJob(j.id);
  if (!EN.updAsked) {      // at most one network check a day: the server throttles it, and the page asks once per visit
    EN.updAsked = true;
    enApi("update", {body: {}, timeout: 40000}).then(() => enLoadState(), () => {});
  }
}
if (typeof S !== "undefined" && S.tab === "encode") encodeOpen();

async function enLoadState(force) {
  EN.st = await enApi("state" + (force ? "?force=1" : ""), {timeout: 60000});
  renderEncoder(); renderGet(); renderJobs();
  if (EN.st.pkg && EN.st.pkg.running && !EN.pkgTimer) EN.pkgTimer = setInterval(enPollPkg, 800);
  if (EN.st.runtime && EN.st.runtime.job && EN.st.runtime.job.running && !EN.rtTimer) EN.rtTimer = setInterval(enPollRuntime, 800);
  enAutoCheckLicence();
  enRuntimeOffer();
}
async function enPollPkg() {
  try { EN.st = await enApi("state"); } catch (e) { return; }
  renderEncoder(); renderGet();
  if (!EN.st.pkg.running) { clearInterval(EN.pkgTimer); EN.pkgTimer = null; EN.licAsked = false; enAutoCheckLicence(); renderWiz(); if (EN.step === 2 && EN.src) enPlan(); }
}

// ------------------------------------------------------------------ the GPU runtime (cuBLAS / cuSOLVER the Pro encoder links)
// Once per visit, when the Pro encoder cannot find the NVIDIA CUDA libraries: ask the server what it offers (name + size), so the button can say it.
async function enRuntimeOffer() {
  const rt = EN.st && EN.st.runtime;
  if (!rt || !rt.need || EN.rtAsked || rt.offer) return;
  EN.rtAsked = true;
  try { const r = await enApi("runtime", {timeout: 40000}); if (EN.st) { EN.st.runtime = r; renderEncoder(); if (EN.step === 3) renderWiz(); } } catch (e) { /* the button still works without the size */ }
}
async function enPollRuntime() {
  try { EN.st = await enApi("state"); } catch (e) { return; }
  renderEncoder();
  const j = EN.st.runtime && EN.st.runtime.job;
  if (!j || !j.running) {
    clearInterval(EN.rtTimer); EN.rtTimer = null; EN.licAsked = false; EN.plan = null;
    renderWiz(); if (EN.step === 3) enRunChecks(); else if (EN.step === 2 && EN.src) enPlan();
  }
}
async function enGetRuntime() {
  EN.rtDismissed = false;
  try { await enApi("runtime/get", {body: {}, timeout: 30000}); await enLoadState(); } catch (e) { toast(e.message, true); }
}
function rtButton(id, rt) {
  const sz = rt.offer ? rt.offer.size_h : "about 1 GB";
  return el("button", {class: "b pri", id, text: "Download the GPU runtime (one time, " + sz + ")", disabled: !!(rt.job && rt.job.running), onclick: enGetRuntime});
}
const RT_SOURCE = {pack: "the PXA GPU runtime", system: "this computer's CUDA install", engine: "your PXA engine install", toolkit: "the CUDA toolkit", pip: "Python's nvidia packages", conda: "conda", env: "PXQE_CUDA_LIBS", resolved: "this computer"};
// the box under the licence line: what the Pro encoder is missing (with the one button that fixes it), the download in progress, or where its GPU libraries come from
function runtimeBox(sel, rt) {
  if (!rt || !rt.applies || sel.edition !== "pro") return [];
  const j = rt.job || {}, out = [];
  const show = j.phase && j.phase !== "idle" && !EN.rtDismissed;
  if (j.running) {
    out.push(el("div", {class: "alert warn", id: "en-rtmsg", role: "status", "aria-live": "polite", style: "display:block;margin-top:8px"}, el("div", {text: j.message || "Working ..."}),
      el("div", {class: "bar en-big"}, el("i", {style: "width:" + Math.round(100 * (j.pct || 0)) + "%"})),
      el("div", {class: "row", style: "margin-top:6px"}, el("button", {class: "b", id: "en-rtcancel", text: "Cancel", onclick: async () => { try { await enApi("runtime/cancel", {body: {}}); } catch (e) { toast(e.message, true); } }}))));
    return out;
  }
  if (show && (j.phase === "failed" || j.phase === "installed_but"))
    out.push(el("div", {class: "alert bad", id: "en-rtmsg", role: "alert", style: "display:block;margin-top:8px"}, el("div", {text: j.message}),
      el("div", {class: "row", style: "margin-top:8px"}, j.phase === "failed" ? el("button", {class: "b", id: "en-rtretry", text: "Try again", onclick: enGetRuntime}) : null,
        el("button", {class: "link", id: "en-rtclose", text: "Close", onclick: () => { EN.rtDismissed = true; renderEncoder(); }}))));
  else if (show && j.phase === "done")
    out.push(el("div", {class: "alert ok", id: "en-rtmsg", role: "status", style: "display:block;margin-top:8px"}, el("div", {text: j.message}),
      el("div", {class: "row", style: "margin-top:6px"}, el("button", {class: "link", id: "en-rtclose", text: "Close", onclick: () => { EN.rtDismissed = true; renderEncoder(); }}))));
  if (rt.need) {
    out.push(el("div", {class: "alert warn", id: "en-rtneed", style: "display:block;margin-top:8px"}, el("div", {text: rt.text}),
      el("div", {class: "tiny", style: "margin:4px 0 8px", text: "PXA Control can fetch them for you. It is a one-time download of NVIDIA's own libraries (their licence is inside); they are checked against PXA's signature, kept next to the encoder, and shared by every Pro encoder you install. You can also install the CUDA 12 toolkit yourself."}),
      el("div", {class: "row"}, rtButton("en-getrt", rt)),
      rt.offer_error ? el("div", {class: "tiny", id: "en-rterr", style: "margin-top:4px", text: rt.offer_error}) : null));
  } else if (rt.lib === "unloadable" || rt.lib === "missing") {
    out.push(el("div", {class: "alert bad", id: "en-rtbad", style: "display:block;margin-top:8px", text: rt.text || "The Pro encoder cannot load on this computer."}));
  } else if (rt.ok && rt.source) {
    out.push(el("div", {class: "kv", id: "en-rtsrc", style: "margin-top:6px"}, el("span", {text: "GPU libraries"}), el("b", {text: "from " + (RT_SOURCE[rt.source] || rt.source) + (rt.installed && rt.source === "pack" ? " (" + rt.installed.size_h + ", CUDA " + (rt.installed.cuda || "12") + ")" : "")})));
  }
  return out;
}

// ------------------------------------------------------------------ encoder panel
function licLine(e, st) {
  const l = e.licence || {}, out = [];
  if (!st.key_set) return [badge("bad", "No key saved"), el("span", {class: "tiny", text: " Paste your key under Get the encoder, or switch to the Free encoder."})];
  const left = l.encodes_left, exp = l.expires ? new Date(typeof l.expires === "number" ? l.expires * 1000 : l.expires) : null;
  const expTxt = exp && !isNaN(exp) ? ", expires " + exp.toLocaleDateString() : "";
  const sub = t => el("span", {class: "tiny", text: " " + t});
  switch (l.state) {
    case "valid": out.push(badge("ok", "Licence OK"), sub((l.unlimited ? "unlimited encodes" : left != null ? left + " encode" + (left === 1 ? "" : "s") + " left this month" : "") + expTxt + (l.user ? " \u00b7 " + l.user : ""))); break;
    case "expired": out.push(badge("bad", "Licence expired"), sub("Get a fresh key from the PXA Network Discord (/encoder). The Free encoder still works.")); break;
    case "revoked": case "suspended": out.push(badge("bad", l.state === "revoked" ? "Key revoked" : "Key suspended"), sub("The licence server no longer accepts this key. Ask in the PXA Network Discord (/encoder), or use the Free encoder.")); break;
    case "refused": out.push(badge("bad", "Key refused"), sub((l.reason ? l.reason + ". " : "") + "Check the key, or ask in the PXA Network Discord (/encoder). The Free encoder still works.")); break;
    case "no_key": out.push(badge("bad", "No key"), sub("The encoder does not have your key. Paste it under Get the encoder.")); break;
    case "no_server": out.push(badge("bad", "No licence server"), sub("The encoder does not know the licence server address.")); break;
    case "no_quota": out.push(badge("bad", "No encodes left"), sub("Your monthly quota is used up. The Free encoder still works.")); break;
    case "offline": out.push(badge("warn", "Licence server not reachable"), sub("Check your internet; the classic tiers still work."), el("button", {class: "link", text: " Try again", onclick: enCheckLicence})); break;
    default: out.push(badge("warn", "Licence not checked"), el("button", {class: "link", text: " Check now", onclick: enCheckLicence}));
  }
  const rt = e.runtime || {};
  if (rt.lib === "unloadable" || rt.lib === "missing") out.push(badge("warn", "Cannot load here"), sub(rt.fix === "runtime-pack" ? "The NVIDIA CUDA libraries are missing (see below)." : rt.fix === "driver" ? "No NVIDIA driver was found." : (rt.detail ? rt.detail + ". " : "") + "The PXQN tiers need an NVIDIA driver and the CUDA 12 runtime."));
  return out;
}
async function enAutoCheckLicence() {
  // once per visit, quietly: Pro + a saved key + a licence the encoder has not asked the server about yet
  const sel = EN.st && EN.st.encoders.find(e => e.selected);
  if (EN.licAsked || !sel || sel.edition !== "pro" || !EN.st.key_set || !sel.licence || sel.licence.checked) return;
  EN.licAsked = true;
  try { EN.st = await enApi("licence", {body: {}, timeout: 70000}); renderEncoder(); } catch (e) { /* the badge still offers Check now */ }
}
async function enCheckLicence() { try { EN.st = await enApi("licence", {body: {}, timeout: 70000}); renderEncoder(); } catch (e) { toast(e.message, true); } }

function renderEncoder() {
  const st = EN.st; if (!st) return;
  const ok = st.encoders.filter(e => e.ok), sel = ok.find(e => e.selected), bad = st.encoders.filter(e => !e.ok);
  const head = el("div", {class: "en-bar"}, el("h3", {style: "margin:0"}, "Encoder"),
    sel ? el("span", {class: "chip " + (sel.edition === "pro" ? "pxqn" : "pxq"), id: "en-edition", text: (sel.edition === "pro" ? "Pro" : "Free") + (sel.version ? " " + sel.version : "")}) : el("span", {class: "chip warn", id: "en-edition", text: "none installed"}),
    el("span", {class: "sp"}),
    el("button", {class: "b", id: "en-rescan", text: "Rescan", onclick: async () => { try { await enLoadState(true); renderWiz(); if (EN.step === 2 && EN.src) enPlan(); toast("Looked again: " + (EN.st.encoders.filter(e => e.ok).length) + " encoder(s) found"); } catch (e) { toast(e.message, true); } }}),
    el("button", {class: "b", id: "en-getbtn", text: "Get the encoder", onclick: () => { EN.showGet = !EN.showGet; renderGet(); if (EN.showGet) $("#en-get").scrollIntoView({block: "nearest"}); }}));
  const kids = [head];
  if (!sel) kids.push(el("div", {class: "alert warn", id: "en-none"}, "No PXA Quantizer was found on this computer. Download the Free one (no key), or paste your supporter key to get Pro."));
  else {
    if (ok.length > 1) kids.push(el("div", {class: "row", style: "margin-top:8px"}, el("label", {class: "tiny", for: "en-sel", text: "Use"}),
      el("select", {id: "en-sel", "aria-label": "which encoder to use", onchange: async ev => { try { EN.st = await enApi("use", {body: {path: ev.target.value}}); renderEncoder(); renderGet(); EN.plan = null; renderWiz(); if (EN.step === 2) enPlan(); } catch (e) { toast(e.message, true); } }},
        ...ok.map(e => el("option", {value: e.path, selected: e.selected, text: (e.edition === "pro" ? "Pro " : "Free ") + (e.version || "") + " (" + e.build_id + ")"})))));
    kids.push(el("div", {class: "kv", style: "margin-top:6px"}, el("span", {text: "Build"}), el("b", {class: "mono", text: sel.build_id + (sel.tiers.length ? " · " + sel.tiers.length + " tiers" : "")})));
    kids.push(el("div", {class: "row", id: "en-lic", style: "margin-top:6px"}, ...(sel.edition === "pro" ? licLine(sel, st) :
      [badge("ok", "Free"), el("span", {class: "tiny", text: " Makes the classic PXQ tiers. Pro (a supporter feature) adds the PXQN tiers: the same size with much lower error."}),
       el("button", {class: "link", id: "en-howpro", text: " How to get Pro", onclick: () => { EN.showGet = true; renderGet(); $("#en-get").scrollIntoView({block: "nearest"}); }})])));
    kids.push(...runtimeBox(sel, st.runtime));
  }
  for (const u of (st.update && st.update.available) || [])
    kids.push(el("div", {class: "alert ok en-upd"}, el("span", {}, "Update available: " + (u.edition === "pro" ? "Pro " : "Free ") + (u.version || u.build_id) + ". "),
      el("button", {class: "b", text: "Update", onclick: () => enGet(u.edition, true)})));
  for (const e of bad) kids.push(el("div", {class: "tiny", style: "margin-top:6px", text: "Found " + e.path + " but it did not answer: " + (e.error || "no details") + "."}));
  kids.push(el("details", {style: "margin-top:8px", open: EN.showPicker}, el("summary", {text: "Use a Pro encoder I already downloaded…"}),
    el("div", {class: "row", style: "margin-top:6px"}, el("input", {type: "text", id: "en-addpath", placeholder: "/path/to/pxqe or the folder you unpacked it into", "aria-label": "path to the encoder", style: "flex:1;min-width:200px"}),
      el("button", {class: "b", id: "en-addgo", text: "Use it", onclick: async () => {
        try { EN.st = await enApi("add-encoder", {body: {path: $("#en-addpath").value}, timeout: 60000}); renderEncoder(); renderGet(); renderWiz(); toast("Encoder added: its tiers are unlocked now."); }
        catch (e) { $("#en-addmsg").textContent = e.message; }
      }})),
    el("div", {class: "tiny", id: "en-addmsg", role: "status", "aria-live": "polite", style: "margin-top:4px"})));
  $("#en-enc").replaceChildren(el("div", {class: "card"}, ...kids));
}

// ------------------------------------------------------------------ get the encoder
async function enGet(edition, update) {
  EN.pkgDismissed = false;
  const body = {edition, update: !!update};
  if (edition === "pro") { const k = $("#en-key"); if (k && k.value.trim()) body.key = k.value.trim(); }
  try {
    await enApi("get", {body, timeout: 30000});
    const k = $("#en-key"); if (k) k.value = "";
    await enLoadState();
  } catch (e) { toast(e.message, true); }
}
function renderGet() {
  const st = EN.st, box = $("#en-get"); if (!st) return;
  const none = !st.encoders.some(e => e.ok);
  const phase = st.pkg && st.pkg.phase;
  box.hidden = !(EN.showGet || none || (phase && phase !== "idle" && !EN.pkgDismissed));
  if (box.hidden) return;
  const p = st.pkg || {}, running = !!p.running;
  const free = el("div", {class: "card", style: "margin:0"}, el("h4", {text: "Free encoder"}),
    el("p", {class: "tiny", text: "Makes the classic PXQ tiers. No key and no account."}),
    el("button", {class: "b pri", id: "en-getfree", text: "Download Free", disabled: running, onclick: () => enGet("free")}));
  const keyRow = st.key_set
    ? el("div", {class: "row"}, el("span", {class: "en-masked", id: "en-keymask", title: "your key, hidden", text: st.key_masked}),
        el("button", {class: "b", id: "en-keyforget", text: "Forget key", onclick: async () => { try { EN.st = await enApi("key", {body: {key: ""}}); renderEncoder(); renderGet(); } catch (e) { toast(e.message, true); } }}))
    : el("div", {class: "row"}, el("input", {type: "password", id: "en-key", autocomplete: "off", spellcheck: "false", "aria-label": "your PXA Quantizer key", placeholder: "pxk1.K-XXXXXXXX.…", style: "flex:1;min-width:200px"}));
  const pro = el("div", {class: "card", style: "margin:0"}, el("h4", {text: "Pro encoder (a supporter feature)"}),
    el("p", {class: "tiny"}, "The PXQN tiers: the same size as the classic ones with much lower error. Support PXA on ", el("a", {href: "https://ko-fi.com/shatteredrealms1", target: "_blank", rel: "noopener", text: "Ko-fi"}),
      ", then type ", el("b", {text: "/encoder"}), " in the ", el("a", {href: "https://discord.gg/EqazvV9tf", target: "_blank", rel: "noopener", text: "PXA Network Discord"}), ": the bot gives you a key."),
    el("div", {class: "tiny", style: "margin:6px 0"}, "I have a key:"), keyRow,
    el("div", {class: "tiny", style: "margin-top:6px"}, "The key is stored on this computer (file mode 600) and sent only to the PXA licence server (" + st.licence_server + ")."),
    el("div", {class: "row", style: "margin-top:8px"}, el("button", {class: "b pri", id: "en-getpro", text: "Download Pro", disabled: running, onclick: () => enGet("pro")})));
  const kids = [el("h3", {}, "Get the encoder"), el("div", {class: "en-grid2"}, free, pro)];
  if (p.phase && p.phase !== "idle" && !EN.pkgDismissed) {
    const failed = p.phase === "failed";
    kids.push(el("div", {class: "alert " + (failed ? "bad" : p.phase === "done" ? "ok" : "warn"), id: "en-pkgmsg", role: "status", "aria-live": "polite", style: "display:block"},
      el("div", {text: p.message || p.phase}),
      running ? el("div", {class: "bar en-big"}, el("i", {style: "width:" + Math.round(100 * (p.pct || 0)) + "%"})) : null,
      failed ? el("div", {class: "row", style: "margin-top:8px"},
        el("button", {class: "b", id: "en-pkgretry", text: "Try again", onclick: () => enGet(p.edition || "free")}),
        p.edition === "pro" ? el("button", {class: "b", id: "en-pkgfree", text: "Use Free instead", onclick: () => enGet("free")}) : null) : null,
      !running ? el("div", {class: "row", style: "margin-top:8px"}, el("button", {class: "link", id: "en-pkgclose", text: "Close", onclick: () => { EN.pkgDismissed = true; EN.showGet = false; renderGet(); }})) : null));
  }
  box.replaceChildren(el("div", {class: "card"}, ...kids));
}

// ------------------------------------------------------------------ the wizard shell
const EN_STEPS = ["Source", "Target", "Checks", "Run", "Done"];
function renderWiz() {
  if (!EN.st) return;
  const steps = el("ol", {class: "en-steps", "aria-label": "steps"}, ...EN_STEPS.map((n, i) =>
    el("li", {class: EN.step > i + 1 ? "done" : "", "aria-current": EN.step === i + 1 ? "step" : null, id: "en-step-" + (i + 1)}, el("span", {class: "n", text: String(i + 1)}), el("span", {class: "lb", text: n}))));
  const body = [stepSource, stepTarget, stepChecks, stepRun, stepDone][EN.step - 1]();
  $("#en-wiz").replaceChildren(steps, body);
  if (EN.step === 4 || EN.step === 5) renderJobTop();
}
function enGo(n) { EN.step = n; renderWiz(); if (n === 2) { enLoadRigThenPlan(); } if (n === 3) enRunChecks(); window.scrollTo({top: $("#p-encode").offsetTop - 70, behavior: "smooth"}); }

// ------------------------------------------------------------------ step 1: source
const SUP = {supported: "ok", beta: "warn", untested: "warn", unsupported: "bad", unknown: "warn"};
const LIC = {ok: "ok", conditions: "warn", noncommercial: "warn", noderivs: "bad", unknown: "warn"};
function stepSource() {
  const s = EN.src;
  const kids = [el("h3", {}, "1. Which model?"),
    el("p", {class: "tiny", text: "Paste a Hugging Face model (like Qwen/Qwen3-1.7B), or pick a folder or a GGUF file on this computer."}),
    el("div", {class: "row"}, el("input", {type: "text", id: "en-src", value: EN.srcText, placeholder: "Qwen/Qwen3-1.7B   or   /path/to/model", autocomplete: "off", spellcheck: "false", "aria-label": "Hugging Face model or local path", style: "flex:1;min-width:220px",
        onkeydown: ev => { if (ev.key === "Enter") enInspect(); }}),
      el("button", {class: "b pri", id: "en-inspect", text: "Look it up", disabled: EN.srcBusy, onclick: enInspect}),
      el("button", {class: "b", id: "en-pick", text: "Pick from this computer…", onclick: () => enPicker()}))];
  if (EN.srcBusy) kids.push(el("div", {class: "empty"}, el("span", {class: "spin"}), "Looking it up ..."));
  if (EN.srcErr) kids.push(el("div", {class: "alert bad", id: "en-srcerr", role: "alert"}, EN.srcErr));
  if (s) {
    const sup = s.support || {}, lic = s.licence || {};
    kids.push(el("div", {id: "en-srcinfo", style: "margin-top:12px"},
      el("div", {class: "en-facts"},
        fact("Model", s.name), fact("From", s.kind === "hf" ? "Hugging Face" : s.kind === "folder" ? "A folder here" : "A GGUF file here"),
        fact("Size", s.kind === "hf" ? s.size_h + " to download" : s.size_h), fact("Parameters", s.param_label),
        fact("Architecture", s.arch || "unknown")),
      el("div", {class: "row", style: "margin:4px 0"}, badge(SUP[sup.level] || "warn", "PXA support: " + (sup.level || "unknown")), el("span", {class: "tiny", text: sup.text || ""})),
      el("div", {class: "row", style: "margin:4px 0"}, badge(LIC[lic.level] || "warn", "Model licence: " + (lic.id || "unknown")), el("span", {class: "tiny", text: lic.text || ""})),
      lic.level === "noderivs" ? el("div", {class: "alert warn", id: "en-licwarn"}, "Heads up: this licence does not allow sharing a quantized copy. You can still make one for yourself; do not upload or share it.") : null,
      ...((s.notes || []).map(n => el("div", {class: "tiny", text: n})))));
  }
  const can = s && s.support && s.support.level !== "unsupported";
  kids.push(el("div", {class: "row en-act", style: "margin-top:14px;justify-content:flex-end"},
    el("button", {class: "b pri", id: "en-next1", text: "Next: where will it run?", disabled: !can, onclick: () => enGo(2)})));
  return el("div", {class: "card"}, ...kids);
}
function fact(l, v) { return el("div", {}, el("span", {text: l}), el("b", {text: v == null ? "?" : String(v)})); }
async function enInspect() {
  const v = ($("#en-src").value || "").trim();
  EN.srcText = v; EN.srcErr = ""; EN.src = null; EN.srcBusy = true; EN.plan = null; EN.checks = null; renderWiz();
  try { EN.src = await enApi("inspect", {body: {source: v, force: true}, timeout: 90000}); store.set("enc.src", v); }
  catch (e) { EN.srcErr = e.message; }
  EN.srcBusy = false; renderWiz();
}

// folder / file picker
async function enPicker(path) {
  let dlg = $("#en-dlg");
  if (!dlg) {
    dlg = el("dialog", {class: "dlg en-dlg", id: "en-dlg", "aria-labelledby": "en-dlg-h"}, el("h2", {id: "en-dlg-h", text: "Pick a model"}), el("div", {class: "tiny", id: "en-dlg-path", style: "margin-bottom:6px;overflow-wrap:anywhere"}),
      el("div", {class: "en-pick", id: "en-dlg-list"}), el("div", {class: "row", style: "justify-content:flex-end;margin-top:10px"},
        el("button", {class: "b", id: "en-dlg-use", text: "Use this folder", hidden: true}), el("button", {class: "b", id: "en-dlg-close", text: "Cancel", onclick: () => dlg.close()})));
    document.body.append(dlg);
  }
  if (!dlg.open) dlg.showModal();
  let d;
  try { d = await enApi("browse" + (path ? "?path=" + encodeURIComponent(path) : "")); } catch (e) { $("#en-dlg-list").replaceChildren(el("div", {class: "empty", text: e.message})); return; }
  $("#en-dlg-path").textContent = d.path;
  const rows = [];
  if (d.parent) rows.push(el("button", {onclick: () => enPicker(d.parent)}, el("span", {class: "k", text: "up"}), el("span", {class: "n", text: ".."})));
  for (const e of d.entries) rows.push(el("button", {onclick: () => e.kind === "gguf" ? enPicked(d.path.replace(/\/$/, "") + "/" + e.name) : e.kind === "hfdir" ? enPicker(d.path.replace(/\/$/, "") + "/" + e.name) : enPicker(d.path.replace(/\/$/, "") + "/" + e.name)},
    el("span", {class: "k", text: e.kind === "gguf" ? "gguf" : e.kind === "hfdir" ? "model" : "folder"}), el("span", {class: "n", text: e.name + (e.kind === "gguf" ? "  (" + enBytes(e.size) + ")" : "")})));
  $("#en-dlg-list").replaceChildren(...(rows.length ? rows : [el("div", {class: "empty", text: "Nothing to pick in here."})]));
  const use = $("#en-dlg-use"); use.hidden = !d.is_model; use.onclick = () => enPicked(d.path);
}
function enPicked(p) { $("#en-dlg").close(); EN.srcText = p; renderWiz(); $("#en-src").value = p; enInspect(); }

// ------------------------------------------------------------------ step 2: target + tiers
async function enLoadRigThenPlan() {
  try { EN.rig = await api("/api/rig/live", {timeout: 30000}); } catch (e) { EN.rig = {cards: []}; }
  const have = EN.rig.cards.map(c => c.index);
  let sel = Array.isArray(EN.cards) ? EN.cards.filter(i => have.includes(i)) : [];
  if (!sel.length && !EN.other) sel = have.slice();
  EN.cards = sel;
  renderWiz();
  await enPlan();
}
function enTarget() {
  if (EN.other && EN.vram) return {vram_gib: EN.vram};
  return {cards: EN.cards};
}
async function enPlan() {
  const t = enTarget();
  if (!(t.vram_gib || (t.cards && t.cards.length))) { EN.plan = null; EN.planErr = ""; renderTiers(); return; }
  EN.planBusy = true; EN.planErr = ""; renderTiers();
  try {
    EN.plan = await enApi("plan", Object.assign({body: Object.assign({source: EN.srcText}, t), timeout: 90000}));
    const avail = EN.plan.tiers.filter(r => r.available);
    if (!EN.tier || !avail.some(r => r.key === EN.tier)) EN.tier = (EN.plan.recommended[0]) || null;
  } catch (e) { EN.plan = null; EN.planErr = e.message; }
  EN.planBusy = false; renderTiers();
}
function stepTarget() {
  const cards = (EN.rig && EN.rig.cards) || [];
  const tiles = cards.map(c => el("label", {class: "tile"}, el("input", {type: "checkbox", value: String(c.index), checked: EN.cards.includes(c.index), "aria-label": "card " + c.index + " " + c.name,
      onchange: ev => { const i = +ev.target.value; EN.cards = ev.target.checked ? EN.cards.concat([i]) : EN.cards.filter(x => x !== i); store.set("enc.cards", EN.cards); enPlan(); }}),
    el("span", {class: "tick"}), el("span", {class: "t1", text: "#" + c.index + " " + c.name}), el("span", {class: "t2", text: (c.mem_total_mib / 1024).toFixed(0) + " GiB · " + (c.class || "")})));
  tiles.push(el("label", {class: "tile"}, el("input", {type: "checkbox", id: "en-other", checked: EN.other, "aria-label": "another machine",
      onchange: ev => { EN.other = ev.target.checked; store.set("enc.other", EN.other); renderWiz(); enPlan(); }}), el("span", {class: "tick"}), el("span", {class: "t1", text: "Another machine"}), el("span", {class: "t2", text: "type its GPU memory"})));
  const kids = [el("h3", {}, "2. Which cards will run it?"),
    el("p", {class: "tiny", text: "Tick the cards that will run the finished model. The tiers below are the ones that fit."}),
    el("div", {class: "tiles", id: "en-cards"}, ...tiles)];
  if (EN.other) kids.push(el("div", {class: "row", style: "margin-top:8px"}, el("label", {class: "f", style: "width:220px"}, "Total GPU memory (GiB)",
    el("input", {type: "number", id: "en-vram", min: "2", max: "4096", value: EN.vram, onchange: ev => { EN.vram = ev.target.value; store.set("enc.vram", EN.vram); enPlan(); }}))));
  kids.push(el("div", {id: "en-tiers", style: "margin-top:14px"}));
  kids.push(el("div", {id: "en-lock", hidden: true}));
  kids.push(el("details", {id: "en-adv", style: "margin-top:12px"}, el("summary", {text: "Advanced"}), advBody()));
  kids.push(el("div", {class: "row en-act", style: "margin-top:14px;justify-content:space-between"},
    el("button", {class: "b", id: "en-back2", text: "Back", onclick: () => enGo(1)}),
    el("button", {class: "b pri", id: "en-next2", text: "Check my machine", onclick: () => enGo(3)})));
  setTimeout(renderTiers, 0);
  return el("div", {class: "card"}, ...kids);
}
function advBody() {         // (an encoder with CLI version 2 always measures the weights first: `pxqe make` has no round-to-nearest mode, so no LDLQ switch)
  const st = EN.st, sel = (st.encoders || []).find(e => e.selected), feats = (sel && sel.features) || [];
  const cards = (EN.rig && EN.rig.cards) || [];
  const tierSel = el("select", {id: "en-advtier", "aria-label": "tier", onchange: ev => { EN.tier = ev.target.value; renderTiers(); }},
    ...((EN.plan ? EN.plan.tiers : []).filter(r => r.available).map(r => el("option", {value: r.key, selected: r.key === EN.tier, text: r.name + " · " + r.cls + " · " + enBytes(r.size_bytes)}))));
  return el("div", {style: "display:grid;gap:12px;margin-top:8px"},
    el("label", {class: "f"}, "Tier", tierSel),
    feats.includes("ldlq") && !((sel && sel.cli) >= 2) ? el("label", {class: "check"}, el("input", {type: "checkbox", id: "en-adv-hess", checked: EN.adv.use_hessians !== false, onchange: ev => { EN.adv.use_hessians = ev.target.checked; }}),
      "Use LDLQ (measures the weights first: slower, better quality)") : null,
    el("label", {class: "f"}, "Work folder (big temporary files; needs room)", el("input", {type: "text", id: "en-adv-work", value: EN.adv.work_dir, placeholder: (st.defaults || {}).work_dir || "", onchange: ev => { EN.adv.work_dir = ev.target.value.trim(); }})),
    el("label", {class: "f"}, "Output folder (where the finished file goes)", el("input", {type: "text", id: "en-adv-out", value: EN.adv.out_dir, placeholder: (st.defaults || {}).out_dir || "", onchange: ev => { EN.adv.out_dir = ev.target.value.trim(); }})),
    cards.length ? el("label", {class: "f"}, "Card for the encode", el("select", {id: "en-adv-card", onchange: ev => { EN.adv.encode_card = ev.target.value; }},
      el("option", {value: "", text: "automatic (the card with the most free memory)"}), ...cards.map(c => el("option", {value: String(c.index), selected: String(c.index) === EN.adv.encode_card, text: "#" + c.index + " " + c.name})))) : null,
    el("label", {class: "check"}, el("input", {type: "checkbox", id: "en-adv-keep", checked: EN.adv.keep, onchange: ev => { EN.adv.keep = ev.target.checked; }}), "Keep the intermediate files (to encode another tier of this model later; needs a lot of disk)"));
}
function tierOpt(r, role) {
  const speed = r.decode_tps ? "~" + r.decode_tps.toFixed(0) + " t/s decode (" + (r.cell_source || "measured") + ")" : "no measured speed for this setup yet";
  return el("label", {class: "en-opt", "data-tier": r.key},
    el("input", {type: "radio", name: "en-tier", value: r.key, checked: EN.tier === r.key, "aria-label": r.name, onchange: () => { EN.tier = r.key; renderLock(); }}),
    el("div", {}, el("div", {class: "t1"}, el("span", {text: r.name}), el("span", {class: "chip " + (r.family === "pxqn" ? "pxqn" : "pxq"), title: r.kld != null ? "assistant-token KLD " + r.kld + " (lower is closer to the original)" : "not measured yet", text: (r.measured ? "" : "about ") + r.cls}),
        role ? el("span", {class: "en-role", text: EN_ROLE[role]}) : null),
      el("div", {class: "t2"}, el("span", {}, "File ", el("b", {text: enBytes(r.size_bytes)})), el("span", {}, "Context ", el("b", {text: r.verdict === "no" ? "does not fit" : "about " + enCtx(r.ctx)})), el("span", {text: speed})),
      r.note ? el("div", {class: "t3", text: r.note}) : null));
}
function renderTiers() { renderTiersList(); renderLock(); }
function renderTiersList() {
  const box = $("#en-tiers"); if (!box) return;
  if (EN.planBusy) { box.replaceChildren(el("div", {class: "empty"}, el("span", {class: "spin"}), "Working out what fits ...")); return; }
  if (EN.planErr) { box.replaceChildren(el("div", {class: "alert bad", id: "en-planerr", role: "alert"}, EN.planErr)); return; }
  const p = EN.plan;
  if (!p) { box.replaceChildren(el("div", {class: "alert warn"}, "Pick at least one card (or type another machine's memory) to see what fits.")); return; }
  const ed = p.encoder ? p.encoder.edition : null;
  const kids = [el("h4", {}, "Recommended for " + p.target.vram_gib + " GiB of GPU memory")];
  const rec = p.recommended.map(k => p.tiers.find(r => r.key === k));
  if (!ed) kids.push(el("div", {class: "alert warn"}, "No encoder is installed yet, so no tier can be made. Use Get the encoder above."));
  else if (!rec.length) kids.push(el("div", {class: "alert bad", id: "en-nofit"}, "Nothing this encoder makes fits these cards with room for context. Pick more cards, a smaller model, or another machine."));
  kids.push(el("div", {class: "en-opts", id: "en-recs"}, ...rec.map(r => tierOpt(r, r.role))));
  const chosen = p.tiers.find(r => r.key === EN.tier);
  if (chosen && chosen.available && !rec.includes(chosen)) kids.push(el("div", {class: "en-opts", style: "margin-top:10px"}, tierOpt(chosen, null)));
  const locked = p.tiers.filter(r => r.locked && r.family === "pxqn");
  if (locked.length) {
    kids.push(el("h4", {style: "margin-top:16px"}, ed === "pro" ? "Not in your plan" : "Locked: Supporter features"));
    kids.push(el("p", {class: "tiny", id: "en-plannote", style: "margin:0 0 8px", text: ed === "pro" ? (p.plan_note || "These tiers are not part of your current key's plan.") : "The Free encoder makes the classic PXQ tiers. Pro adds these PXQN tiers: the same size, much lower error. This is what they would be for this model:"}));
    kids.push(el("div", {class: "en-opts", id: "en-locked"}, ...locked.map(r => el("div", {class: "en-lock", "data-tier": r.key}, enIcon(ICON_LOCK),
      el("div", {}, el("div", {class: "t1"}, el("span", {text: r.name}), el("span", {class: "chip pxqn", text: (r.measured ? "" : "about ") + r.cls}), el("span", {class: "chip warn", text: r.locked_reason || "Supporter feature"})),
        el("div", {class: "t2"}, el("span", {text: "File " + enBytes(r.size_bytes)}), el("span", {text: "Context " + (r.verdict === "no" ? "does not fit" : "about " + enCtx(r.ctx))})))))));
    kids.push(el("div", {class: "row", style: "margin-top:10px"},
      el("button", {class: "b", id: "en-howpro2", text: "How to get Pro", onclick: () => { EN.showGet = true; renderGet(); $("#en-get").scrollIntoView({block: "nearest"}); }}),
      el("button", {class: "b", id: "en-havekey", text: "I have a key", onclick: () => { EN.showGet = true; renderGet(); const k = $("#en-key"); if (k) k.focus(); $("#en-get").scrollIntoView({block: "nearest"}); }})));
  }
  box.replaceChildren(...kids);
  const at = $("#en-advtier"); if (at) at.replaceChildren(...p.tiers.filter(r => r.available).map(r => el("option", {value: r.key, selected: r.key === EN.tier, text: r.name + " · " + r.cls + " · " + enBytes(r.size_bytes)})));
  const nx = $("#en-next2"); if (nx) nx.disabled = !(chosen && chosen.available);
}

// ------------------------------------------------------------------ "Who can load this file" (Pro, PXQN tiers)
// The server (plan.lock) says which choices exist: "on" = the radios of the modes the key's plan allows; "off" (the encoder cannot lock yet, or the licence
// server's lock switch is off) and "unknown" (the licence server could not be asked) = the choice shown disabled with one line, and the file is written without a lock.
function enLockView() {
  const p = EN.plan, row = p && p.tiers.find(r => r.key === EN.tier);
  return p && p.lock && row && row.family === "pxqn" && row.available ? p.lock : null;
}
function enLockChoice(L) { return L.choices.some(c => c.id === EN.lock) ? EN.lock : L.default; }
function renderLock() {
  const box = $("#en-lock"); if (!box) return;
  const L = enLockView();
  box.hidden = !L;
  if (!L) { box.replaceChildren(); return; }
  const on = L.state === "on", kids = [el("legend", {text: "Who can load this file"})];
  if (on) {
    const cur = enLockChoice(L);
    kids.push(el("div", {class: "en-opts", id: "en-lockopts"}, ...L.choices.map(c => el("label", {class: "en-opt", "data-lock": c.id},
      el("input", {type: "radio", name: "en-lockmode", value: c.id, checked: c.id === cur, "aria-label": c.label, onchange: () => { EN.lock = c.id; store.set("enc.lock", c.id); }}),
      el("div", {}, el("div", {class: "t1"}, el("span", {text: c.label})), el("div", {class: "t3", text: c.note}))))));
    kids.push(el("div", {class: "tiny", id: "en-lockneeds", style: "margin-top:6px", text: "A locked file loads in PXA " + L.needs + " or newer."}));
  } else {
    kids.push(el("div", {class: "en-opts", id: "en-lockopts"}, el("label", {class: "en-opt", "data-lock": "personal"},
      el("input", {type: "radio", name: "en-lockmode", value: "personal", disabled: true, "aria-label": "Only me (recommended)", "aria-describedby": "en-lockoff"}),
      el("div", {}, el("div", {class: "t1"}, el("span", {text: "Only me (recommended)"}))))));
    kids.push(el("div", {class: "tiny", id: "en-lockoff", style: "margin-top:6px", text: L.text}));
  }
  box.replaceChildren(el("fieldset", {class: "en-lockset", id: "en-lockset", disabled: !on, "data-state": L.state}, ...kids));
}

// ------------------------------------------------------------------ step 3: checks
function enBody() {
  const b = {source: EN.srcText, tier: EN.tier, keep: EN.adv.keep, licence_ack: EN.adv.licence_ack};
  Object.assign(b, enTarget());
  if (EN.adv.work_dir) b.work_dir = EN.adv.work_dir;
  if (EN.adv.out_dir) b.out_dir = EN.adv.out_dir;
  if (EN.adv.use_hessians !== null) b.use_hessians = EN.adv.use_hessians;
  if (EN.adv.encode_card !== "") b.encode_card = EN.adv.encode_card;
  if (EN.other) { delete b.cards; }
  else if (b.cards) b.cards = b.cards.slice();
  if (EN.other && EN.rig && EN.rig.cards.length) b.cards = EN.rig.cards.map(c => c.index).slice(0, 1);   // the encode still runs on a card of this machine
  const L = enLockView();
  if (L && L.state === "on") b.lock = enLockChoice(L);            // while the choice is disabled nothing is sent: the server decides (and writes the file open)
  return b;
}
async function enRunChecks() {
  EN.checksBusy = true; EN.checksErr = ""; EN.checks = null; renderWiz();
  try { EN.checks = await enApi("checks", {body: enBody(), timeout: 150000}); } catch (e) { EN.checksErr = e.message; }
  EN.checksBusy = false; if (EN.step === 3) renderWiz();
}
function stepChecks() {
  const c = EN.checks, kids = [el("h3", {}, "3. Checks before we start")];
  if (EN.checksBusy) kids.push(el("div", {class: "empty"}, el("span", {class: "spin"}), "Checking disk, memory, graphics card, tools and your licence ..."));
  if (EN.checksErr) kids.push(el("div", {class: "alert bad", role: "alert", id: "en-checkserr"}, EN.checksErr));
  if (c) {
    kids.push(el("div", {id: "en-checks"}, ...c.checks.map(k => el("div", {class: "en-ck", "data-check": k.id},
      el("div", {}, badge(k.status, k.status === "ok" ? "OK" : k.status === "warn" ? "Check" : "Stop")),
      el("div", {}, el("div", {class: "lab", text: k.label}), el("div", {text: k.text}), k.fix ? el("div", {class: "fix", text: k.fix}) : null,
        k.action === "runtime" ? el("div", {class: "row", style: "margin-top:6px"}, rtButton("en-ck-getrt", (EN.st && EN.st.runtime) || {})) : null)))));
    const lic = c.checks.find(k => k.id === "src_licence");
    if (lic && lic.status !== "ok" && EN.src && EN.src.licence.level === "noderivs")
      kids.push(el("label", {class: "check", style: "margin-top:8px"}, el("input", {type: "checkbox", id: "en-ack", checked: EN.adv.licence_ack, onchange: ev => { EN.adv.licence_ack = ev.target.checked; enRunChecks(); }}), "I will not share the result"));
    kids.push(el("div", {class: "en-facts", id: "en-est"}, fact("About how long", c.estimate.text), fact("Disk at the peak", c.disk.peak_h), fact("Final file", c.disk.output_h),
      fact("Work folder", c.work_dir), fact("Output folder", c.out_dir)));
    kids.push(el("details", {}, el("summary", {text: "What will happen"}), el("ol", {style: "margin:6px 0 0;padding-left:20px"}, ...c.stages.map(s => el("li", {text: s.label + (c.estimate.stages[s.id] ? "  (" + fmtDur(c.estimate.stages[s.id][0]) + " to " + fmtDur(c.estimate.stages[s.id][1]) + ")" : "")})))));
    if (!c.can_start) kids.push(el("div", {class: "alert bad", id: "en-refuse", role: "alert"}, "Cannot start yet: " + c.refusal));
  }
  kids.push(el("div", {class: "row en-act", style: "margin-top:14px;justify-content:space-between"},
    el("button", {class: "b", id: "en-back3", text: "Back", onclick: () => enGo(2)}),
    el("div", {class: "row"}, el("button", {class: "b", id: "en-recheck", text: "Check again", onclick: enRunChecks}),
      el("button", {class: "b pri", id: "en-start", text: "Start encoding", disabled: !(c && c.can_start) || EN.checksBusy, onclick: enStart}))));
  return el("div", {class: "card"}, ...kids);
}

// ------------------------------------------------------------------ step 4: run
async function enStart() {
  try {
    const j = await enApi("start", {body: enBody(), timeout: 150000});
    EN.job = j; EN.logLines = []; EN.logSeq = 0; EN.step = 4; EN.test = null; renderWiz(); enPollJob(true);
  } catch (e) { toast(e.message, true); }
}
async function enOpenJob(id) {
  try { EN.job = await enApi("job?id=" + encodeURIComponent(id)); } catch (e) { toast(e.message, true); return; }
  EN.logLines = []; EN.logSeq = 0; EN.step = EN.job.status === "done" ? 5 : 4; EN.test = null; renderWiz(); enPollJob(true);
}
function enPollJob(now) {
  clearTimeout(EN.jobTimer);
  const tick = async () => {
    if (!EN.job) return;
    try {
      const j = await enApi("job?id=" + encodeURIComponent(EN.job.id) + "&since=" + EN.logSeq);
      EN.logLines.push(...(j.log || [])); EN.logSeq = j.log_seq; delete j.log; EN.job = j;
      if (j.status === "done" && EN.step === 4) { EN.step = 5; renderWiz(); } else renderJobTop();
      enLogRender();
      if (EN_LIVE.includes(j.status)) EN.jobTimer = setTimeout(tick, 1000); else enLoadState().catch(() => {});
    } catch (e) { EN.jobTimer = setTimeout(tick, 3000); }
  };
  if (now) tick(); else EN.jobTimer = setTimeout(tick, 1000);
}
function enLogRender() { const pre = $("#en-log"); if (pre) renderLogInto(pre, EN.logLines, "", false, true, null); }
function stageDot(s) { return s.status === "done" ? "✓" : s.status === "failed" ? "✕" : s.status === "running" ? "▶" : "·"; }
function renderJobTop() {
  const j = EN.job, box = $("#en-jobtop"); if (!j || !box) return;
  const kids = [el("div", {class: "en-bar"}, el("h3", {style: "margin:0"}, (j.params.name || "model") + " → " + (j.params.tier_name || "")),
    el("span", {class: "chip " + (j.status === "done" ? "ok" : j.status === "running" ? "warn" : j.status === "paused" ? "warn" : "bad"), id: "en-jstatus", text: EN_STATUS[j.status] || j.status}), el("span", {class: "sp"}))];
  const pct = Math.round(100 * (j.pct || 0));
  const done = j.status === "done" && EN.step === 5;
  const bar = el("div", {class: "bar en-big", role: "progressbar", "aria-valuenow": String(pct), "aria-valuemin": "0", "aria-valuemax": "100", "aria-label": "overall progress"}, el("i", {id: "en-jbar", style: "width:" + pct + "%"}));
  const text = el("div", {class: "tiny", id: "en-jtext", role: "status", "aria-live": "polite", text: pct + "%" + (j.eta_s != null && EN_LIVE.includes(j.status) ? " \u00b7 about " + fmtDur(j.eta_s) + " left" : "") + (j.status === "paused" ? " \u00b7 paused" : "")});
  const stages = el("div", {id: "en-stages", style: "margin-top:10px"}, ...j.stages.map(s => el("div", {class: "en-stage " + s.status, "data-stage": s.id},
    el("span", {class: "en-sdot", text: stageDot(s)}), el("div", {}, el("div", {class: "nm", text: s.label}), el("div", {class: "dt", text: s.status === "done" ? "done" : s.status === "pending" ? "waiting" : (s.detail || (s.status === "failed" ? "stopped" : "working"))})),
    el("div", {class: "tiny eta", text: s.status === "running" && s.eta_s != null ? "~" + fmtDur(s.eta_s) + " left" : s.status === "running" ? Math.round(100 * s.pct) + "%" : ""}),
    el("div", {class: "bar"}, el("i", {style: "width:" + Math.round(100 * (s.status === "done" ? 1 : s.pct)) + "%"})))));
  if (done) kids.push(el("details", {id: "en-steps-done", style: "margin-top:6px"}, el("summary", {text: "Show what was done"}), stages));     // a finished job leads with the result, the steps are one click away
  else kids.push(bar, text, stages);
  if (j.error && j.status !== "running") kids.push(el("div", {class: "alert " + (j.status === "cancelled" ? "warn" : "bad"), id: "en-jerr", role: "alert", style: "display:block"},
    el("div", {text: j.error.message}), j.error.hint ? el("div", {class: "tiny", text: j.error.hint}) : null));
  const act = [];
  if (j.status === "running") act.push(el("button", {class: "b", id: "en-pause", text: "Pause", onclick: () => enJobAct("pause")}));
  if (j.status === "paused") act.push(el("button", {class: "b pri", id: "en-resume", text: "Resume", onclick: () => enJobAct("resume")}));
  if (["interrupted", "failed", "cancelled"].includes(j.status)) act.push(el("button", {class: "b pri", id: "en-resume", text: "Resume", onclick: () => enJobAct("resume")}));
  if (EN_LIVE.includes(j.status)) act.push(el("button", {class: "b danger", id: "en-cancel", text: "Cancel", onclick: () => { if (confirm("Cancel this encode? Finished steps are kept and you can resume it later.")) enJobAct("cancel"); }}));
  if (["interrupted", "failed", "cancelled"].includes(j.status)) act.push(el("button", {class: "b danger", id: "en-discard", text: "Discard", onclick: () => { if (confirm("Delete this job and its temporary files?")) enJobAct("discard"); }}));
  if (j.status === "done") act.push(el("button", {class: "b", id: "en-another", text: "Encode another model", onclick: enAnother}));
  if (act.length) kids.push(el("div", {class: "row en-act", style: "margin-top:10px"}, ...act));
  box.replaceChildren(...kids);
}
async function enJobAct(a) {
  try {
    const r = await enApi(a, {body: {id: EN.job.id}, timeout: 60000});
    if (a === "discard") { EN.job = null; EN.step = 1; await enLoadState(); renderWiz(); return; }
    if (r && r.id) EN.job = r;
    renderJobTop(); enPollJob(true);
  } catch (e) { toast(e.message, true); }
}
function enAnother() { EN.job = null; EN.src = null; EN.plan = null; EN.checks = null; EN.tier = null; EN.test = null; EN.step = 1; clearTimeout(EN.jobTimer); enLoadState().catch(() => {}); renderWiz(); }
function stepRun() {
  const j = EN.job;
  if (!j) return el("div", {class: "card"}, el("div", {class: "empty", text: "No encode is open. Start one from step 1."}));
  return el("div", {class: "card"}, el("h3", {}, "4. Running"), el("div", {id: "en-jobtop"}),
    el("details", {id: "en-logd", style: "margin-top:12px"}, el("summary", {text: "Log"}), el("pre", {class: "log", id: "en-log", tabindex: "0", "aria-label": "encode log"})));
}

// ------------------------------------------------------------------ step 5: done
function stepDone() {
  const j = EN.job;
  if (!j || !j.result) return el("div", {class: "card"}, el("div", {class: "empty", text: "Nothing finished yet."}));
  const r = j.result, c = j.counts || {};
  const kids = [el("h3", {}, "5. Done"),
    el("div", {class: "alert ok", id: "en-doneok", style: "display:block"}, el("b", {text: "Your file is ready. "}), "It is a ", el("b", {text: r.tier}), " file in the ", el("b", {text: r.cls}), "."),
    r.lock ? el("div", {class: "en-lockdone " + (r.lock.locked ? "locked" : "open"), id: "en-lockdone"},
      el("div", {class: "t1"}, r.lock.locked ? badge("ok", r.lock.title) : el("span", {class: "chip", text: r.lock.title})),
      el("div", {id: "en-lockwho", text: r.lock.text}), r.lock.needs ? el("div", {class: "tiny", id: "en-lockreq", text: r.lock.needs}) : null) : null,
    el("div", {class: "en-facts"}, fact("Size", enBytes(r.size)), fact("Tensors", c.encoded ? c.exact + " of " + c.encoded + " decoded back exactly" : (r.tier_tensors || 0) + " in " + r.tier), fact("Quality class", r.cls)),
    el("div", {class: "tiny"}, "File"), el("div", {class: "en-path"}, el("div", {class: "cmd", id: "en-outpath", text: r.path}), el("button", {class: "b", text: "Copy", onclick: () => copyText(r.path)})),
    el("div", {class: "tiny", style: "margin-top:8px"}, "sha256"), el("div", {class: "en-path"}, el("div", {class: "cmd", id: "en-outsha", text: r.sha256}), el("button", {class: "b", text: "Copy", onclick: () => copyText(r.sha256)})),
    el("div", {id: "en-test", style: "margin-top:14px"}),
    el("div", {class: "row en-act", style: "margin-top:14px"}, el("button", {class: "b pri", id: "en-testit", text: "Test it", onclick: enTest}),
      el("button", {class: "b", id: "en-share", text: "Share your score", hidden: !(EN.test && EN.test.result), onclick: enShare}),
      el("button", {class: "b", text: "See it in Models", onclick: () => showTab("models")})),
    el("div", {id: "en-jobtop", style: "margin-top:16px;padding-top:12px;border-top:1px solid var(--line)"})];
  setTimeout(renderTest, 0);
  return el("div", {class: "card"}, ...kids);
}
function renderTest() {
  const t = EN.test, box = $("#en-test"); if (!box) return;
  const b = $("#en-testit"); if (b) b.disabled = !!(t && t.busy);
  const sh = $("#en-share"); if (sh) sh.hidden = !(t && t.result);
  if (!t) { box.replaceChildren(); return; }
  const kids = [];
  if (t.error) kids.push(el("div", {class: "alert bad", id: "en-testerr", role: "alert", style: "display:block"}, el("div", {text: t.error}), t.detail ? el("pre", {class: "out", text: t.detail}) : null));
  else if (t.result) {
    const by = c => (t.result.classes || []).find(x => x.class === c) || {};
    kids.push(el("div", {class: "alert ok", id: "en-testres", style: "display:block"}, el("b", {text: "It runs. "}), "Decode " + fmtNum(by("prose").decode_tps) + " t/s on prose, prefill " + fmtNum(by("long").prefill_tps) + " t/s on a long prompt."));
  } else if (t.msg) kids.push(el("div", {class: "empty", id: "en-testmsg"}, el("span", {class: "spin"}), t.msg));
  box.replaceChildren(...kids);
}
async function enTest() {
  EN.test = {busy: true, msg: "Starting a server with your new file ..."}; renderTest();
  try {
    const r = await enApi("test", {body: {id: EN.job.id}, timeout: 200000});
    if (!r.ok) { EN.test = {error: "The launcher refused to start it.", detail: r.text}; renderTest(); return; }
    const sid = r.sid; EN.test.sid = sid;
    EN.test.msg = (r.note ? r.note + " " : "") + "Loading the model (this can take a minute or two) ...";
    renderTest();
    const t0 = Date.now();
    for (;;) {
      await new Promise(res => setTimeout(res, 2000));
      const st = await api("/api/status?sid=" + encodeURIComponent(sid));
      if (st.health === "ok") break;
      if (!st.running) { EN.test = {error: "The server stopped before it was ready (exit " + st.exit_code + "). Open the Servers tab for its log.", sid}; renderTest(); return; }
      if (Date.now() - t0 > 15 * 60 * 1000) { EN.test = {error: "The server did not become ready within 15 minutes.", sid}; renderTest(); return; }
    }
    EN.test.msg = "Running the benchmark (3 repetitions) ..."; renderTest();
    await api("/api/bench", {body: {sid, reps: 3, warmup: 10}});
    for (;;) {
      await new Promise(res => setTimeout(res, 1500));
      const d = await api("/api/bench"), jb = d.job;
      if (jb.error) { EN.test = {error: "The benchmark failed: " + jb.error, sid}; renderTest(); return; }
      if (!jb.running && jb.result) { EN.test = {result: jb.result, sid, busy: false}; renderTest(); return; }
    }
  } catch (e) { EN.test = {error: e.message}; renderTest(); }
}
async function enShare() {
  const t = EN.test; if (!t || !t.result) return;
  setTarget({sid: t.sid, label: "Encode test"});
  showTab("speed");
  setTimeout(() => checkHighScore(t.result.ts), 500);
}

// ------------------------------------------------------------------ recent jobs
function renderJobs() {
  const box = $("#en-jobs"), jobs = (EN.st && EN.st.jobs) || [];
  if (!box) return;
  if (!jobs.length) { box.replaceChildren(); return; }
  box.replaceChildren(el("div", {class: "card"}, el("h3", {}, "Recent encodes"), ...jobs.map(j => el("div", {class: "en-jobrow", "data-job": j.id},
    el("b", {text: (j.name || "model") + " → " + (j.tier || "")}), el("span", {class: "chip " + (j.status === "done" ? "ok" : EN_LIVE.includes(j.status) ? "warn" : "bad"), text: EN_STATUS[j.status] || j.status}),
    el("span", {class: "tiny", text: fmtAgo(j.created)}), el("span", {class: "sp"}),
    el("button", {class: "b", text: j.status === "done" ? "Open" : ["interrupted", "failed", "cancelled"].includes(j.status) ? "Open / resume" : "Open", onclick: () => enOpenJob(j.id)})))));
}
