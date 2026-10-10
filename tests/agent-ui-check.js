// Headless UI check for the chat Assistant view (tools/pxa_chat/ui). Run by hand against a Control that has
// a chat server (the mock: tests/pxa_chat_mock.py, or a real one):
//   [PXA_TOK=...] node tests/agent-ui-check.js http://127.0.0.1:17811/ /tmp/shots [--real] [--key KEY]
// needs puppeteer-core and Chrome (CHROME=/path). Exits 1 on any failed check or page error.
const puppeteer = require("puppeteer-core");
const args = process.argv.slice(2);
const base = args[0] || "http://127.0.0.1:17811/", out = args[1] || "/tmp", real = args.includes("--real");
const keyArg = args.includes("--key") ? args[args.indexOf("--key") + 1] : null;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fails = [], ok = (c, msg) => { console.log((c ? "ok   " : "FAIL ") + msg); if (!c) fails.push(msg); };
const CANNED = /All done\. Here is what I found|Let me think about the best way to help/;
(async () => {
  const b = await puppeteer.launch({executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new", args: ["--no-sandbox", "--disable-gpu"]});
  const p = await b.newPage(); await p.setViewport({width: 1440, height: 960});
  const errs = []; p.on("pageerror", e => errs.push(e.message)); p.on("console", m => { if (m.type() === "error" && !/favicon|401|404/.test(m.text())) errs.push("console: " + m.text()); });
  await p.evaluateOnNewDocument(() => { localStorage.setItem("pxa-control.agent.session", "null"); localStorage.setItem("pxa-control.agent.adv", "false"); localStorage.setItem("pxa-control.agent.view", '"assistant"'); localStorage.setItem("pxa-control.agent.preset", '"assistant"'); });
  const tok = process.env.PXA_TOK ? "?token=" + encodeURIComponent(process.env.PXA_TOK) : "";   // a Control with a token
  await p.goto(base + tok + "#chat", {waitUntil: "networkidle2"}); await sleep(2000);
  if (keyArg) { await p.evaluate(k => window.pxaAgent.state.selected = k, keyArg); }
  const shot = async n => p.screenshot({path: `${out}/${n}.png`});
  const waitIdle = async (ms) => { const t0 = Date.now(); await sleep(400); while (Date.now() - t0 < (ms || 120000)) { if (!(await p.evaluate(() => !!window.pxaAgent.state.run))) return true; await sleep(300); } return false; };
  const ask = async (t, ms) => { await p.click("#ag-in"); await p.evaluate(t => { const i = document.querySelector("#ag-in"); i.value = t; i.dispatchEvent(new Event("input")); }, t); await p.keyboard.press("Enter"); return waitIdle(ms); };
  const lastA = () => p.evaluate(() => { const a = [...document.querySelectorAll(".ag-a")].pop(); return a ? {text: [...a.querySelectorAll(".ag-md")].map(x => x.innerText).join("\n"), tools: [...a.querySelectorAll(".ag-tool .ag-say")].map(x => x.textContent), html: a.innerHTML, meta: (a.querySelector(".ag-meta") || {}).textContent || "", notes: [...a.querySelectorAll(".ag-note")].map(n => n.textContent)} : null; });

  ok(await p.$("#ag-root .ag-side") && await p.$(".ag-comp #ag-in"), "layout: sidebar + composer");
  const fill = await p.evaluate(() => { const r = document.querySelector("#ag-root").getBoundingClientRect(); return {gap: Math.round(innerHeight - r.bottom), h: Math.round(r.height), scroll: document.documentElement.scrollHeight - innerHeight}; });
  ok(fill.gap >= 0 && fill.gap <= 48 && fill.h > 600 && fill.scroll <= 2, `panel fills the window without page scroll (gap ${fill.gap}px, height ${fill.h}px, page overflow ${fill.scroll}px)`);
  const starters = await p.$$eval(".ag-starter", x => x.length);
  ok(starters >= 3 && starters <= 4, `welcome shows ${starters} starter prompts`);
  await shot("after-01-welcome");

  // 1. calculator: compact tool row + the model's own final words
  ok(await ask("What is 18% of 2,450?"), "calc turn finished");
  let a = await lastA();
  ok(a.tools.some(t => /^Calculated /.test(t)), "tool row reads 'Calculated ...': " + a.tools.join(" | "));
  ok(/441/.test(a.text) && !CANNED.test(a.html), "final reply has 441 and no canned wrapper: " + a.text.slice(0, 120));
  ok(/tok\/s|tokens/.test(a.meta), "quiet footer shows tokens / tok/s: " + a.meta);
  const uRight = await p.$eval(".ag-u", e => getComputedStyle(e).alignSelf);
  ok(uRight === "flex-end", "user message is right-aligned");
  await p.click(".ag-a .ag-tool > summary"); await sleep(200);
  ok(await p.$eval(".ag-a .ag-tool", e => e.open && /Input/.test(e.textContent)), "tool row expands to input/output");
  await shot("after-02-calc");

  // 2. markdown
  ok(await ask(real ? "Give me a short markdown table of 3 GPUs with their VRAM, then a one-line bash code block that prints hello." : "show me a markdown table"), "markdown turn finished");
  a = await lastA();
  ok(/<table>/.test(a.html), "table rendered");
  ok(/class="ag-code"/.test(a.html) && /data-agcopy/.test(a.html), "code block with copy button + language label");
  await shot("after-03-markdown");

  // 3. approval inline card: save a file
  await p.click("#ag-in"); await p.evaluate(() => { const i = document.querySelector("#ag-in"); i.value = "Save a shopping list file for a pasta dinner as shopping-list.md"; });
  await p.keyboard.press("Enter");
  let card = null; for (let i = 0; i < 120 && !card; i++) { await sleep(500); card = await p.$(".ag-ap:not(.answered)"); }
  ok(!!card, "approval shows as an inline card in the conversation");
  ok(!(await p.$("dialog[open]")), "no modal dialog for the approval");
  if (card) { await shot("after-04-approval"); const btn = await p.$$(".ag-ap:not(.answered) .ag-ap-btns button"); ok(btn.length >= 2, "Allow once / Always / Deny buttons"); await btn[0].click(); }
  await waitIdle();
  a = await lastA();
  ok(a.tools.some(t => /^Saved /.test(t)), "row reads 'Saved ...': " + a.tools.join(" | "));
  ok(await p.$(".ag-ap.answered.ok"), "card collapses to 'Allowed once'");

  // 4. Stop mid-reply with Esc
  await p.click("#ag-in"); await p.evaluate(r => { const i = document.querySelector("#ag-in"); i.value = r ? "Write a long, detailed 1500-word essay about the history of graphics cards." : "slow please"; }, real);
  await p.keyboard.press("Enter"); await sleep(real ? 6000 : 2500);
  ok(await p.$eval("#ag-send", e => e.classList.contains("stop")), "Send turned into Stop while streaming");
  await p.keyboard.press("Escape");
  ok(await waitIdle(30000), "Esc stopped the reply");
  a = await lastA();
  ok(a.notes.some(n => /Stopped/.test(n)), "a quiet 'Stopped.' notice");

  // 5. regenerate the last reply
  const turnsBefore = await p.$$eval(".ag-a", x => x.length);
  await p.hover(".ag-a:last-of-type"); await p.click(".ag-a:last-of-type .ag-ib[title=Regenerate]");
  await sleep(real ? 4000 : 1500); await p.keyboard.press("Escape"); await waitIdle(30000);
  ok(await p.$$eval(".ag-a", x => x.length) === turnsBefore, "regenerate replaces the last reply (no extra turn)");

  // 6. edit-and-resend the first message
  await p.hover(".ag-u"); await p.click(".ag-u .ag-ib[title='Edit and resend']");
  await p.evaluate(() => { const t = document.querySelector(".ag-edit"); t.value = "What is 18% of 2,450? Answer in one short sentence."; });
  await p.click(".ag-editbox button.b.pri"); await waitIdle();
  ok(await p.$$eval(".ag-a", x => x.length) === 1, "edit-and-resend branches from that message");
  a = await lastA(); ok(/441/.test(a.text), "edited turn answered: " + a.text.slice(0, 100));
  await shot("after-05-edited");

  // 7. sidebar: grouped, rename, new chat with Ctrl+K, delete
  ok(await p.$$eval(".ag-group", g => g.map(x => x.textContent).includes("Today")), "chat list grouped (Today)");
  await p.hover(".ag-item.on"); await p.click(".ag-item.on .ag-ib[title=Rename]");
  await p.evaluate(() => { const i = document.querySelector(".ag-rename"); i.value = "Percent math"; }); await p.keyboard.press("Enter"); await sleep(800);
  ok(await p.$eval("#ag-title", e => e.textContent) === "Percent math", "rename updates the title");
  await p.keyboard.down("Control"); await p.keyboard.press("k"); await p.keyboard.up("Control"); await sleep(500);
  ok(!!(await p.$("#ag-empty")), "Ctrl+K opens a new chat (welcome)");
  await p.click(".ag-item .ag-item-t"); await sleep(1200);
  ok(await p.$$eval(".ag-a", x => x.length) >= 1, "reopening a saved chat restores it");
  await p.hover(".ag-item.on"); await p.click(".ag-item.on .ag-ib[title=Delete]"); await sleep(200);
  ok(!!(await p.$(".ag-item.confirm")), "delete asks inline first");
  await p.click(".ag-del-yes"); await sleep(1000);
  ok(!!(await p.$("#ag-empty")), "deleted chat -> welcome");

  // 8. advanced drawer, light theme, mobile, classic switch
  await p.click('#ag-mode button[data-v=advanced]'); await sleep(400);
  ok(await p.$eval("#ag-drawer", e => getComputedStyle(e).display !== "none" && e.offsetWidth > 200), "Advanced opens the settings drawer");
  ok(await p.$("#ag-sys") && await p.$("#ag-temp") && await p.$$eval(".ag-tools input", x => x.length) > 3, "drawer: system prompt, sampling, tool toggles");
  await shot("after-06-advanced");
  await p.click("#theme"); await sleep(300);
  await p.evaluate(() => window.pxaAgent.openSession && null);
  await shot("after-07-light");
  await p.click("#theme"); await p.click('#ag-mode button[data-v=simple]'); await sleep(300);
  await p.setViewport({width: 390, height: 844, isMobile: true, hasTouch: true}); await sleep(600);
  ok(await p.$eval(".ag-main", e => e.offsetWidth >= 340), "mobile: conversation takes the width");
  await shot("after-08-mobile");
  await p.setViewport({width: 1440, height: 960}); await sleep(300);
  await p.click('#ag-view button[data-v=classic]'); await sleep(300);
  ok(await p.$eval("#ag-root", e => e.hidden) && await p.$eval(".cols.chat:not(#ag-root)", e => !e.hidden), "Classic chat still there behind the switch");
  await p.click('#ag-view button[data-v=assistant]');
  ok(errs.length === 0, "no page errors" + (errs.length ? ": " + errs.join(" | ") : ""));
  await b.close();
  console.log(fails.length ? `${fails.length} FAILED` : "ALL OK");
  process.exit(fails.length ? 1 : 0);
})().catch(e => { console.error(String(e && e.message || e).replace(/token=[^&#\s]+/g, "token=<redacted>")); process.exit(1); });
