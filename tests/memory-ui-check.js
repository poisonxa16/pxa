// Headless UI check for chat memory (tools/pxa_chat memory: "Memory updated" row + Undo, Memory panel, switch,
// per-chat toggle, Advanced extras) and the cross-chat e2e (tell a preference in chat 1, a new chat uses it,
// forget it, it is gone). Against the mock or a real server:
//   [PXA_TOK=...] node tests/memory-ui-check.js http://127.0.0.1:17811/ /tmp/shots [--real] [--key KEY] [--shots DIR2]
// It CLEARS the memory store of the Control it runs against first. Exits 1 on any failed check or page error.
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const args = process.argv.slice(2);
const base = args[0] || "http://127.0.0.1:17811/", out = args[1] || "/tmp", real = args.includes("--real");
const opt = k => args.includes(k) ? args[args.indexOf(k) + 1] : null;
const keyArg = opt("--key"), shots2 = opt("--shots");
const sleep = ms => new Promise(r => setTimeout(r, ms));
const red = s => String(s).replace(/token=[^&#\s"]+/g, "token=<redacted>");
const fails = [], ok = (c, msg) => { console.log((c ? "ok   " : "FAIL ") + red(msg)); if (!c) fails.push(msg); };
(async () => {
  const b = await puppeteer.launch({executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new", args: ["--no-sandbox", "--disable-gpu"]});
  const p = await b.newPage(); await p.setViewport({width: 1440, height: 960});
  const errs = []; p.on("pageerror", e => errs.push(e.message)); p.on("console", m => { if (m.type() === "error" && !/favicon|401|404|422/.test(m.text())) errs.push("console: " + m.text()); });
  await p.evaluateOnNewDocument(() => { localStorage.setItem("pxa-control.agent.session", "null"); localStorage.setItem("pxa-control.agent.adv", "false"); localStorage.setItem("pxa-control.agent.view", '"assistant"'); localStorage.setItem("pxa-control.agent.preset", '"assistant"'); });
  const tok = process.env.PXA_TOK ? "?token=" + encodeURIComponent(process.env.PXA_TOK) : "";
  try {
    await p.goto(base + tok + "#chat", {waitUntil: "networkidle2"}); await sleep(2000);
    if (keyArg) await p.evaluate(k => window.pxaAgent.state.selected = k, keyArg);
    const api = (path, body) => p.evaluate(async (path, body) => { const r = await fetch(path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {}); return {status: r.status, j: await r.json()}; }, path, body);
    const shot = async (n, sel) => { const o = {path: `${out}/${n}.png`}; if (sel) { const e = await p.$(sel); if (e) { await e.screenshot(o); } else await p.screenshot(o); } else await p.screenshot(o); if (shots2) fs.copyFileSync(o.path, `${shots2}/${n}.png`); };
    const waitIdle = async ms => { const t0 = Date.now(); await sleep(400); while (Date.now() - t0 < (ms || 180000)) { if (!(await p.evaluate(() => !!window.pxaAgent.state.run))) return true; await sleep(300); } return false; };
    const ask = async (t, ms) => { await p.click("#ag-in"); await p.evaluate(t => { const i = document.querySelector("#ag-in"); i.value = t; i.dispatchEvent(new Event("input")); }, t); await p.keyboard.press("Enter"); return waitIdle(ms); };
    const lastA = () => p.evaluate(() => { const a = [...document.querySelectorAll(".ag-a")].pop(); return a ? {text: [...a.querySelectorAll(".ag-md")].map(x => x.innerText).join("\n"), mem: [...a.querySelectorAll(".ag-memrow")].map(r => ({say: r.querySelector(".ag-say").textContent, body: r.querySelector(".ag-tool-b").textContent}))} : null; });
    const facts = async () => (await api("/api/chat/memory")).j.facts;

    await api("/api/chat/memory", {op: "clear", confirm: true});
    await api("/api/chat/memory", {op: "settings", enabled: true});
    ok(await p.$("#ag-mem-open"), "sidebar has a Memory button");

    // chat 1: a preference -> the model saves it -> "Memory updated" row
    ok(await ask(real ? "Quick note about me: I'm vegetarian, I don't eat meat or fish. Please remember that for future chats." : "Please remember that I'm vegetarian"), "chat 1 finished");
    let a = await lastA();
    ok(a.mem.some(m => m.say === "Memory updated"), "a 'Memory updated' row: " + JSON.stringify(a.mem));
    let fs1 = await facts();
    ok(fs1.some(f => /vegetarian/i.test(f.text)), "the fact is in memory: " + fs1.map(f => f.text).join(" | "));
    console.log("     chat 1 reply: " + a.text.slice(0, 200).replace(/\n/g, " "));
    await p.click(".ag-a:last-of-type .ag-memrow > summary"); await sleep(250);
    ok(await p.$eval(".ag-a:last-of-type .ag-memrow", e => e.open && /Remembered|Updated|Already/.test(e.textContent)), "the row expands to what was saved");
    await p.evaluate(() => document.querySelector(".ag-a:last-of-type .ag-memrow").scrollIntoView({block: "center"}));
    await shot("memory-updated-row");
    await shot("memory-updated-row-close", ".ag-a:last-of-type");
    // Undo removes it, then put it back with the panel's add box later
    const undo = await p.$(".ag-a:last-of-type .ag-mem-undo");
    ok(!!undo, "an Undo button");
    if (undo) { await undo.click(); await sleep(800); ok(!(await facts()).some(f => /vegetarian/i.test(f.text)), "Undo removed the fact");
      ok(await p.$eval(".ag-a:last-of-type .ag-memrow .ag-say", e => e.textContent === "Memory change undone"), "the row says it was undone"); }

    // Memory panel: add, pin, edit, delete+undo
    await p.click("#ag-mem-open"); await sleep(500);
    ok(await p.$eval("#ag-memp", e => !e.hidden), "Memory panel opens");
    ok(await p.$eval("#ag-mem-on", e => e.checked), "'Remember things across chats' is on by default");
    await p.type("#ag-mem-add", "The user is vegetarian (no meat or fish)."); await p.click("#ag-mem-addb"); await sleep(600);
    await p.type("#ag-mem-add", "The user lives in Toronto."); await p.keyboard.press("Enter"); await sleep(600);
    await p.type("#ag-mem-add", "My password is hunter22"); await p.keyboard.press("Enter"); await sleep(600);
    const toastTxt = await p.$eval("#toast", e => e.textContent);
    ok(/secret/i.test(toastTxt), "a secret is refused with a friendly reason: " + toastTxt);
    let items = await p.$$eval(".ag-mem-it .ag-mem-txt", x => x.map(e => e.textContent));
    ok(items.length === 2 && !items.some(t => /hunter/.test(t)), "the panel lists 2 facts, no secret: " + items.join(" | "));
    const toronto = await p.$("xpath/.//div[contains(@class,'ag-mem-it')][.//div[contains(text(),'Toronto')]]");
    await (await toronto.$(".ag-mem-pin")).click(); await sleep(600);
    ok((await facts()).find(f => /Toronto/.test(f.text)).pinned, "pin works");
    const veg = await p.$("xpath/.//div[contains(@class,'ag-mem-it')][.//div[contains(text(),'vegetarian')]]");
    await (await veg.$(".ag-mem-edit")).click(); await sleep(200);
    await p.evaluate(() => { const i = document.querySelector(".ag-mem-editrow input"); i.value = "The user is vegetarian: no meat or fish."; });
    await p.keyboard.press("Enter"); await sleep(600);
    ok((await facts()).some(f => f.text === "The user is vegetarian: no meat or fish."), "edit works");
    ok(await p.$eval("#ag-mem-add", e => e.value === ""), "the refused secret is not left in the box");
    await p.evaluate(() => { document.querySelector("#ag-mem-add").blur(); document.querySelector("#toast").hidden = true; });
    await shot("memory-panel");
    await shot("memory-panel-close", ".ag-memp-card");
    const tor2 = await p.$("xpath/.//div[contains(@class,'ag-mem-it')][.//div[contains(text(),'Toronto')]]");
    await (await tor2.$(".ag-mem-del")).click(); await sleep(600);
    ok(!(await facts()).some(f => /Toronto/.test(f.text)), "delete works");
    await p.click(".ag-mem-toast button"); await sleep(700);
    ok((await facts()).some(f => /Toronto/.test(f.text)), "the delete toast's Undo puts it back");
    await p.keyboard.press("Escape"); await sleep(300);
    ok(await p.$eval("#ag-memp", e => e.hidden), "Esc closes the panel");

    // chat 2 (new chat): the preference is used
    await p.click("#ag-new"); await sleep(500);
    ok(await ask(real ? "Suggest one dinner I could cook tonight. Just the dish name and one line why." : "What should I make for dinner?"), "chat 2 finished");
    a = await lastA();
    console.log("     chat 2 reply: " + a.text.slice(0, 240).replace(/\n/g, " "));
    const MEAT = /\b(chicken|beef|pork|steak|salmon|tuna|fish|shrimp|bacon|lamb|turkey|ham)\b/i;
    const usesIt = real ? (!MEAT.test(a.text.replace(/no (meat|fish)|meat[- ]free|without (meat|fish)/gi, "")) || /vegetarian|vegan|meat[- ]free|plant/i.test(a.text)) : /meat-free/.test(a.text);
    ok(usesIt, "chat 2 uses the remembered preference (a vegetarian dinner)");
    const run2 = await p.evaluate(() => window.pxaAgent.state.session);

    // forget it in chat 3, then a new chat no longer gets it
    await p.click("#ag-new"); await sleep(500);
    ok(await ask(real ? "Please forget that I'm vegetarian; I eat everything now." : "Forget that I'm vegetarian"), "forget chat finished");
    a = await lastA();
    ok(a.mem.some(m => /Memory updated/.test(m.say) && /Forgot/.test(m.body)), "a 'Memory updated: Forgot ...' row: " + JSON.stringify(a.mem));
    // a real model may save "eats everything" in its place: that is fine, only the vegetarian fact must be gone
    ok(!(await facts()).some(f => /vegetarian/i.test(f.text) && !/not vegetarian|no longer|eats everything/i.test(f.text)), "the vegetarian fact is gone: " + (await facts()).map(f => f.text).join(" | "));
    await p.click("#ag-new"); await sleep(500);
    ok(await ask(real ? "Suggest one dinner I could cook tonight. Just the dish name and one line why." : "What should I make for dinner?"), "chat 4 finished");
    a = await lastA();
    console.log("     chat 4 reply: " + a.text.slice(0, 240).replace(/\n/g, " "));
    if (!real) ok(/chicken/.test(a.text), "after forget, dinner is no longer vegetarian-only");

    // Advanced: per-chat toggle + Advanced block in the panel; switch off/on
    await p.evaluate(() => { document.querySelector("#ag-mode button[data-v=advanced]").click(); });
    await sleep(300);
    ok(await p.$eval("#ag-usemem", e => e.offsetParent !== null && e.checked), "Advanced drawer shows 'Use memory in this chat' (on)");
    await p.click("#ag-mem-open"); await sleep(400);
    ok(await p.$eval("#ag-mem-adv", e => e.offsetParent !== null), "Advanced block (k, budget, import/export, raw JSON) shows in Advanced mode");
    await p.evaluate(() => document.querySelector("#ag-mem-adv").open = true); await sleep(150);
    ok(await p.$eval("#ag-mem-raw", e => e.textContent.trim().startsWith("[")), "raw facts JSON");
    ok(await p.$eval("#ag-mem-k", e => e.value === "8"), "recall k shown");
    await p.click(".ag-switch"); await sleep(600);
    ok((await api("/api/chat/memory")).j.enabled === false, "the switch turns memory off");
    await p.click(".ag-switch"); await sleep(600);
    ok((await api("/api/chat/memory")).j.enabled === true, "and on again");
    // Clear all with a friendly confirm
    await p.click("#ag-mem-clear"); await sleep(200);
    ok(/This can't be undone/.test(await p.$eval("#ag-mem-confirm", e => e.textContent)), "Clear all asks first, in plain words");
    await p.click("#ag-mem-clear-yes"); await sleep(600);
    ok((await facts()).length === 0 && await p.$(".ag-mem-empty"), "Clear all empties memory and shows the friendly empty state");
    await p.keyboard.press("Escape");
    await p.evaluate(() => { document.querySelector("#ag-mode button[data-v=simple]").click(); });
    // mobile: the panel is full-screen
    await p.setViewport({width: 390, height: 800}); await sleep(300);
    await p.evaluate(() => window.pxaAgent.openMemory(true)); await sleep(300);
    const mob = await p.evaluate(() => [Math.round(document.querySelector(".ag-memp-card").getBoundingClientRect().width), Math.round(document.querySelector("#ag-root").getBoundingClientRect().width)]);
    ok(mob[0] >= mob[1] - 2, `mobile: the panel takes the full width (${mob[0]} of ${mob[1]}px)`);
    await p.evaluate(() => window.pxaAgent.openMemory(false));
    // clean up the test chats
    const ss = (await api("/api/chat/sessions")).j.sessions || [];
    await p.evaluate(async ids => { for (const id of ids) await fetch("/api/chat/session", {method: "DELETE", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id})}); }, ss.filter(s => /vegetarian|dinner|remember|forget/i.test(s.title || "")).map(s => s.id));
    ok(!errs.length, "no page errors: " + errs.join(" | "));
  } catch (e) { ok(false, "exception: " + (e && e.stack || e)); }
  await b.close();
  console.log(fails.length ? `${fails.length} FAILED` : "all memory UI checks passed");
  process.exit(fails.length ? 1 : 0);
})();
