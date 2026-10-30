// Headless WebKit: open Control's chat tab and check the chat page rendered with NO server running (the smoke has no model):
// the assistant composer (#ag-in, 'Message the assistant') is visible and the welcome line ('Good morning. What can I help with?') is there.
// (The classic chat controls like #c-in stay hidden until a server runs: that is expected, not tested here.) Saves a screenshot.
// env: URL (http://127.0.0.1:PORT/), TOKEN (sent as a header, never printed), SHOT (png path)
const { webkit } = require('playwright');
(async () => {
  const b = await webkit.launch(process.env.WK_EXE ? { executablePath: process.env.WK_EXE } : {});
  const ctx = await b.newContext({ viewport: { width: 1280, height: 800 }, extraHTTPHeaders: { 'X-PXA-Token': process.env.TOKEN } });
  const p = await ctx.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push(String(e).slice(0, 300)));
  await p.goto(process.env.URL + '#chat', { waitUntil: 'load' });
  await p.waitForSelector('#p-chat:not([hidden])', { timeout: 30000 });
  await p.waitForSelector('#ag-in', { state: 'visible', timeout: 30000 });
  const txt = await p.innerText('#p-chat');
  await p.screenshot({ path: process.env.SHOT });
  await b.close();
  if (errs.length) { console.log('page errors:', errs.join(' | ')); process.exit(1); }
  if (!/What can I help with/i.test(txt)) { console.log('chat text missing:', txt.slice(0, 200)); process.exit(1); }
  console.log('chat page text ok (' + txt.length + ' chars)');
})().catch(e => { console.log('FAIL', String(e).slice(0, 400)); process.exit(1); });
